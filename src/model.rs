use crate::cache::{BlockId, BlockKey, KvTensor, PrefixCache};
use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::Path;

#[derive(Clone, Copy)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub vocab: usize,
    pub seq_len: usize,
}

#[derive(Clone)]
struct Weights {
    embedding: Range<usize>,
    rms_att: Range<usize>,
    wq: Range<usize>,
    wk: Range<usize>,
    wv: Range<usize>,
    wo: Range<usize>,
    rms_ffn: Range<usize>,
    w1: Range<usize>,
    w2: Range<usize>,
    w3: Range<usize>,
    rms_final: Range<usize>,
    classifier: Range<usize>,
}

struct Layout {
    offset: usize,
}
impl Layout {
    fn take(&mut self, count: usize) -> io::Result<Range<usize>> {
        let start = self.offset;
        self.offset = start.checked_add(count).ok_or_else(invalid_checkpoint)?;
        Ok(start..self.offset)
    }
}

fn invalid_checkpoint() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid checkpoint layout")
}

fn product(values: &[usize]) -> io::Result<usize> {
    values.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b).ok_or_else(invalid_checkpoint)
    })
}

impl Weights {
    fn new(c: Config, shared: bool, available: usize) -> io::Result<Self> {
        let mut layout = Layout { offset: 7 };
        let embedding = layout.take(product(&[c.vocab, c.dim])?)?;
        let rms_att = layout.take(product(&[c.layers, c.dim])?)?;
        let wq = layout.take(product(&[c.layers, c.dim, c.dim])?)?;
        let wk = layout.take(product(&[c.layers, c.dim, c.kv_dim()])?)?;
        let wv = layout.take(product(&[c.layers, c.dim, c.kv_dim()])?)?;
        let wo = layout.take(product(&[c.layers, c.dim, c.dim])?)?;
        let rms_ffn = layout.take(product(&[c.layers, c.dim])?)?;
        let w1 = layout.take(product(&[c.layers, c.dim, c.hidden_dim])?)?;
        let w2 = layout.take(product(&[c.layers, c.hidden_dim, c.dim])?)?;
        let w3 = layout.take(product(&[c.layers, c.dim, c.hidden_dim])?)?;
        let rms_final = layout.take(c.dim)?;
        layout.take(product(&[c.seq_len, c.dim / c.heads])?)?; // old RoPE tables
        let classifier = if shared {
            embedding.clone()
        } else {
            layout.take(product(&[c.vocab, c.dim])?)?
        };
        if layout.offset > available {
            return Err(invalid_checkpoint());
        }
        Ok(Self {
            embedding,
            rms_att,
            wq,
            wk,
            wv,
            wo,
            rms_ffn,
            w1,
            w2,
            w3,
            rms_final,
            classifier,
        })
    }
}

impl Config {
    pub fn kv_dim(self) -> usize {
        self.dim * self.kv_heads / self.heads
    }
}

pub struct State {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

impl State {
    fn new(c: Config) -> io::Result<Self> {
        let kv_size = product(&[c.layers, c.seq_len, c.kv_dim()])?;
        Ok(Self {
            x: vec![0.0; c.dim],
            xb: vec![0.0; c.dim],
            xb2: vec![0.0; c.dim],
            hb: vec![0.0; c.hidden_dim],
            hb2: vec![0.0; c.hidden_dim],
            q: vec![0.0; c.dim],
            att: vec![0.0; product(&[c.heads, c.seq_len])?],
            logits: vec![0.0; c.vocab],
            keys: vec![0.0; kv_size],
            values: vec![0.0; kv_size],
        })
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}

/// One request's input to a forward pass.
pub struct ForwardItem<'a> {
    pub state: &'a mut State,
    pub token: usize,
    pub pos: usize,
}

fn snapshot_kv(c: Config, state: &State, start_pos: usize, len: usize) -> (KvTensor, KvTensor) {
    let kv_dim = c.kv_dim();
    let mut keys = KvTensor::new(c.layers, len, kv_dim);
    let mut values = KvTensor::new(c.layers, len, kv_dim);
    for layer in 0..c.layers {
        let start = (layer * c.seq_len + start_pos) * kv_dim;
        let end = start + len * kv_dim;
        keys.layer_mut(layer)
            .copy_from_slice(&state.keys[start..end]);
        values
            .layer_mut(layer)
            .copy_from_slice(&state.values[start..end]);
    }
    (keys, values)
}

fn restore_kv(
    c: Config,
    state: &mut State,
    cache: &PrefixCache,
    prompt_tokens: &[usize],
    chunk_size: usize,
    steps: usize,
) -> (usize, Option<BlockId>) {
    if chunk_size == 0 {
        return (0, None);
    }
    // Leave one prompt token and one forward step to produce the next logits.
    let max_reusable = prompt_tokens
        .len()
        .saturating_sub(1)
        .min(steps.saturating_sub(1));
    let kv_dim = c.kv_dim();
    let mut cached_len: usize = 0;
    let mut parent_id = None;
    while let Some(end) = cached_len.checked_add(chunk_size) {
        if end > max_reusable {
            break;
        }
        let key = BlockKey::new(parent_id, &prompt_tokens[cached_len..end]);
        let Some(block) = cache.get(&key) else {
            break;
        };
        for layer in 0..c.layers {
            let start = (layer * c.seq_len + cached_len) * kv_dim;
            let end_offset = start + chunk_size * kv_dim;
            state.keys[start..end_offset].copy_from_slice(block.keys.layer(layer));
            state.values[start..end_offset].copy_from_slice(block.values.layer(layer));
        }
        cached_len = end;
        parent_id = Some(block.id);
    }
    (cached_len, parent_id)
}

pub struct Transformer {
    pub config: Config,
    mmap: Mmap,
    weights: Weights,
    pub prefix_cache: PrefixCache,
}

impl Transformer {
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < 28 || mmap.len() % 4 != 0 || !cfg!(target_endian = "little") {
            return Err(invalid_checkpoint());
        }
        let raw: Vec<i32> = mmap[..28]
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if raw.iter().enumerate().any(|(i, &v)| v <= 0 && i != 5) || raw[5] == 0 {
            return Err(invalid_checkpoint());
        }
        let c = Config {
            dim: raw[0] as usize,
            hidden_dim: raw[1] as usize,
            layers: raw[2] as usize,
            heads: raw[3] as usize,
            kv_heads: raw[4] as usize,
            vocab: raw[5].unsigned_abs() as usize,
            seq_len: raw[6] as usize,
        };
        if c.dim % c.heads != 0
            || c.heads % c.kv_heads != 0
            || c.dim % 2 != 0
            || c.kv_dim() % 2 != 0
        {
            return Err(invalid_checkpoint());
        }
        let weights = Weights::new(c, raw[5] > 0, mmap.len() / 4)?;
        product(&[c.layers, c.seq_len, c.kv_dim()])?;
        product(&[c.heads, c.seq_len])?;
        Ok(Self {
            config: c,
            mmap,
            weights,
            prefix_cache: PrefixCache::default(),
        })
    }

    pub fn new_state(&self) -> State {
        State::new(self.config).expect("model dimensions were checked when loaded")
    }

    /// Save one processed K/V block, returning its stable ID for the next block.
    pub fn cache_block(
        &mut self,
        state: &State,
        parent_id: Option<BlockId>,
        tokens: &[usize],
        start_pos: usize,
    ) -> BlockId {
        assert!(!tokens.is_empty());
        assert!(start_pos
            .checked_add(tokens.len())
            .is_some_and(|end| end <= self.config.seq_len));
        let key = BlockKey::new(parent_id, tokens);
        if let Some(entry) = self.prefix_cache.get(&key) {
            return entry.id;
        }
        let (keys, values) = snapshot_kv(self.config, state, start_pos, tokens.len());
        self.prefix_cache.insert(key, keys, values)
    }

    /// Restore the longest chain of matching full blocks from the prompt start.
    pub fn restore_cached_prefix(
        &self,
        state: &mut State,
        prompt_tokens: &[usize],
        chunk_size: usize,
        steps: usize,
    ) -> (usize, Option<BlockId>) {
        restore_kv(
            self.config,
            state,
            &self.prefix_cache,
            prompt_tokens,
            chunk_size,
            steps,
        )
    }

    pub fn forward<'a>(&self, s: &'a mut State, token: usize, pos: usize) -> &'a [f32] {
        self.forward_batch(&mut [ForwardItem {
            state: s,
            token,
            pos,
        }]);
        &s.logits
    }

    /// Advances every item through the same model layers, sharing each weight row.
    pub fn forward_batch(&self, batch: &mut [ForwardItem<'_>]) {
        let c = self.config;
        let d = c.dim;
        let kv = c.kv_dim();
        for item in batch.iter_mut() {
            assert!(item.token < c.vocab && item.pos < c.seq_len);
        }
        // The mapped checkpoint contains aligned little-endian f32 weights.
        let f = unsafe {
            std::slice::from_raw_parts(self.mmap.as_ptr().cast::<f32>(), self.mmap.len() / 4)
        };
        let w = &self.weights;
        for item in batch.iter_mut() {
            item.state.x.copy_from_slice(
                &f[w.embedding.start + item.token * d..w.embedding.start + (item.token + 1) * d],
            );
        }

        for layer in 0..c.layers {
            for item in batch.iter_mut() {
                rmsnorm(
                    &mut item.state.xb,
                    &item.state.x,
                    &f[w.rms_att.start + layer * d..][..d],
                );
            }
            matmul_batch(
                batch,
                &f[w.wq.start + layer * d * d..][..d * d],
                d,
                |item| (&item.state.xb, &mut item.state.q),
            );
            matmul_batch(
                batch,
                &f[w.wk.start + layer * d * kv..][..d * kv],
                d,
                |item| {
                    let base = (layer * c.seq_len + item.pos) * kv;
                    (&item.state.xb, &mut item.state.keys[base..base + kv])
                },
            );
            matmul_batch(
                batch,
                &f[w.wv.start + layer * d * kv..][..d * kv],
                d,
                |item| {
                    let base = (layer * c.seq_len + item.pos) * kv;
                    (&item.state.xb, &mut item.state.values[base..base + kv])
                },
            );
            for item in batch.iter_mut() {
                attention(item.state, c, layer, item.pos);
            }
            matmul_batch(
                batch,
                &f[w.wo.start + layer * d * d..][..d * d],
                d,
                |item| (&item.state.xb, &mut item.state.xb2),
            );
            for item in batch.iter_mut() {
                for i in 0..d {
                    item.state.x[i] += item.state.xb2[i];
                }
                rmsnorm(
                    &mut item.state.xb,
                    &item.state.x,
                    &f[w.rms_ffn.start + layer * d..][..d],
                );
            }
            matmul_batch(
                batch,
                &f[w.w1.start + layer * d * c.hidden_dim..][..d * c.hidden_dim],
                d,
                |item| (&item.state.xb, &mut item.state.hb),
            );
            matmul_batch(
                batch,
                &f[w.w3.start + layer * d * c.hidden_dim..][..d * c.hidden_dim],
                d,
                |item| (&item.state.xb, &mut item.state.hb2),
            );
            for item in batch.iter_mut() {
                for i in 0..c.hidden_dim {
                    let value = item.state.hb[i];
                    item.state.hb[i] = value / (1.0 + (-value).exp()) * item.state.hb2[i];
                }
            }
            matmul_batch(
                batch,
                &f[w.w2.start + layer * d * c.hidden_dim..][..d * c.hidden_dim],
                c.hidden_dim,
                |item| (&item.state.hb, &mut item.state.xb),
            );
            for item in batch.iter_mut() {
                for i in 0..d {
                    item.state.x[i] += item.state.xb[i];
                }
            }
        }
        for item in batch.iter_mut() {
            rmsnorm_inplace(&mut item.state.x, &f[w.rms_final.clone()]);
        }
        matmul_batch(batch, &f[w.classifier.clone()], d, |item| {
            (&item.state.x, &mut item.state.logits)
        });
    }
}

fn matmul(out: &mut [f32], x: &[f32], weights: &[f32], width: usize) {
    for (row, result) in weights.chunks_exact(width).zip(out.iter_mut()) {
        let mut sum = 0.0;
        for i in 0..width {
            sum += row[i] * x[i];
        }
        *result = sum;
    }
}

fn attention(s: &mut State, c: Config, layer: usize, pos: usize) {
    let d = c.dim;
    let kv = c.kv_dim();
    let head_size = d / c.heads;
    let base = (layer * c.seq_len + pos) * kv;
    for i in (0..d).step_by(2) {
        let frequency = 1.0 / 10000.0_f32.powf((i % head_size) as f32 / head_size as f32);
        let angle = pos as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        rotate(&mut s.q[i..i + 2], cos, sin);
        if i < kv {
            rotate(&mut s.keys[base + i..base + i + 2], cos, sin);
        }
    }
    for h in 0..c.heads {
        let q = &s.q[h * head_size..(h + 1) * head_size];
        let att = &mut s.att[h * c.seq_len..h * c.seq_len + pos + 1];
        let kv_head = h / (c.heads / c.kv_heads);
        for (t, score) in att.iter_mut().enumerate() {
            let offset = (layer * c.seq_len + t) * kv + kv_head * head_size;
            let key = &s.keys[offset..offset + head_size];
            let mut dot = 0.0;
            for i in 0..head_size {
                dot += q[i] * key[i];
            }
            *score = dot / (head_size as f32).sqrt();
        }
        softmax(att);
        let xb = &mut s.xb[h * head_size..(h + 1) * head_size];
        xb.fill(0.0);
        for (t, &a) in att.iter().enumerate() {
            let offset = (layer * c.seq_len + t) * kv + kv_head * head_size;
            let value = &s.values[offset..offset + head_size];
            for i in 0..head_size {
                xb[i] += a * value[i];
            }
        }
    }
}

fn matmul_batch<'state, F>(
    batch: &mut [ForwardItem<'state>],
    weights: &[f32],
    width: usize,
    mut buffers: F,
) where
    F: for<'a> FnMut(&'a mut ForwardItem<'state>) -> (&'a [f32], &'a mut [f32]),
{
    if let [item] = batch {
        let (x, out) = buffers(item);
        matmul(out, x, weights, width);
        return;
    }
    for (output, row) in weights.chunks_exact(width).enumerate() {
        for item in batch.iter_mut() {
            let (x, out) = buffers(item);
            let mut sum = 0.0;
            for i in 0..width {
                sum += row[i] * x[i];
            }
            out[output] = sum;
        }
    }
}

fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let mut ss = 0.0;
    for &v in x {
        ss += v * v;
    }
    let scale = 1.0 / (ss / x.len() as f32 + 1e-5).sqrt();
    for i in 0..x.len() {
        out[i] = weight[i] * (scale * x[i]);
    }
}

fn rmsnorm_inplace(x: &mut [f32], weight: &[f32]) {
    let mut ss = 0.0;
    for &v in x.iter() {
        ss += v * v;
    }
    let scale = 1.0 / (ss / x.len() as f32 + 1e-5).sqrt();
    for i in 0..x.len() {
        x[i] = weight[i] * (scale * x[i]);
    }
}

fn rotate(pair: &mut [f32], cos: f32, sin: f32) {
    let a = pair[0];
    let b = pair[1];
    pair[0] = a * cos - b * sin;
    pair[1] = a * sin + b * cos;
}

fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::MmapMut;

    #[test]
    fn batched_forward_matches_two_separate_requests() {
        let c = Config {
            dim: 4,
            hidden_dim: 8,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            vocab: 8,
            seq_len: 4,
        };
        let weights = Weights::new(c, false, usize::MAX).unwrap();
        let mut mmap = MmapMut::map_anon(weights.classifier.end * 4).unwrap();
        for i in 7..weights.classifier.end {
            let value = ((i * 7 % 23) as f32 - 11.0) / 30.0;
            mmap[i * 4..(i + 1) * 4].copy_from_slice(&value.to_le_bytes());
        }
        let model = Transformer {
            config: c,
            mmap: mmap.make_read_only().unwrap(),
            weights,
            prefix_cache: PrefixCache::default(),
        };
        let mut first = model.new_state();
        let mut second = model.new_state();
        model.forward(&mut first, 1, 0);
        model.forward(&mut second, 3, 0);
        model.forward(&mut second, 4, 1);
        let mut batched_first = model.new_state();
        let mut batched_second = model.new_state();
        model.forward(&mut batched_first, 1, 0);
        model.forward(&mut batched_second, 3, 0);
        model.forward(&mut batched_second, 4, 1);

        model.forward(&mut first, 2, 1);
        model.forward(&mut second, 5, 2);
        model.forward_batch(&mut [
            ForwardItem {
                state: &mut batched_first,
                token: 2,
                pos: 1,
            },
            ForwardItem {
                state: &mut batched_second,
                token: 5,
                pos: 2,
            },
        ]);
        assert_eq!(batched_first.x, first.x);
        assert_eq!(batched_second.x, second.x);
        assert_eq!(batched_first.logits, first.logits);
        assert_eq!(batched_second.logits, second.logits);
        assert_eq!(batched_first.keys, first.keys);
        assert_eq!(batched_second.keys, second.keys);
        assert_eq!(batched_first.values, first.values);
        assert_eq!(batched_second.values, second.values);
    }

    #[test]
    fn batched_classifier_matches_separate_matmuls() {
        let c = Config {
            dim: 4,
            hidden_dim: 4,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            vocab: 3,
            seq_len: 2,
        };
        let mut first = State::new(c).unwrap();
        let mut second = State::new(c).unwrap();
        first.x.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        second.x.copy_from_slice(&[4.0, 3.0, 2.0, 1.0]);
        let weights = [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let mut expected_first = [0.0; 3];
        let mut expected_second = [0.0; 3];
        matmul(&mut expected_first, &first.x, &weights, c.dim);
        matmul(&mut expected_second, &second.x, &weights, c.dim);

        matmul_batch(
            &mut [
                ForwardItem {
                    state: &mut first,
                    token: 0,
                    pos: 0,
                },
                ForwardItem {
                    state: &mut second,
                    token: 0,
                    pos: 0,
                },
            ],
            &weights,
            c.dim,
            |item| (&item.state.x, &mut item.state.logits),
        );
        assert_eq!(first.logits, expected_first);
        assert_eq!(second.logits, expected_second);
    }

    #[test]
    fn snapshot_copies_only_selected_positions_with_compact_layer_stride() {
        let c = Config {
            dim: 4,
            hidden_dim: 4,
            layers: 2,
            heads: 1,
            kv_heads: 1,
            vocab: 8,
            seq_len: 4,
        };
        let mut state = State::new(c).unwrap();
        for (i, value) in state.keys.iter_mut().enumerate() {
            *value = i as f32;
        }
        for (i, value) in state.values.iter_mut().enumerate() {
            *value = 100.0 + i as f32;
        }

        let (keys, values) = snapshot_kv(c, &state, 2, 2);
        assert_eq!(keys.layer(0), &state.keys[8..16]);
        assert_eq!(keys.layer(1), &state.keys[24..32]);
        assert_eq!(values.layer(0), &state.values[8..16]);
        assert_eq!(values.layer(1), &state.values[24..32]);
    }

    #[test]
    fn restore_stops_at_first_missing_block_and_preserves_layer_positions() {
        let c = Config {
            dim: 4,
            hidden_dim: 4,
            layers: 2,
            heads: 1,
            kv_heads: 1,
            vocab: 8,
            seq_len: 6,
        };
        let mut source = State::new(c).unwrap();
        for (i, value) in source.keys.iter_mut().enumerate() {
            *value = i as f32;
        }
        for (i, value) in source.values.iter_mut().enumerate() {
            *value = 100.0 + i as f32;
        }
        let mut cache = PrefixCache::default();
        let (keys, values) = snapshot_kv(c, &source, 0, 2);
        let first = cache.insert(BlockKey::new(None, &[1, 2]), keys, values);
        let (keys, values) = snapshot_kv(c, &source, 2, 2);
        let second = cache.insert(BlockKey::new(Some(first), &[3, 4]), keys, values);

        let mut restored = State::new(c).unwrap();
        assert_eq!(
            restore_kv(c, &mut restored, &cache, &[1, 2, 3, 4, 5], 2, 5),
            (4, Some(second))
        );
        for layer in 0..c.layers {
            let base = layer * c.seq_len * c.kv_dim();
            assert_eq!(
                &restored.keys[base..base + 16],
                &source.keys[base..base + 16]
            );
            assert_eq!(
                &restored.values[base..base + 16],
                &source.values[base..base + 16]
            );
            assert_eq!(restored.keys[base + 16], 0.0);
        }

        let mut diverged = State::new(c).unwrap();
        assert_eq!(
            restore_kv(c, &mut diverged, &cache, &[1, 2, 3, 9, 5], 2, 5),
            (2, Some(first))
        );
    }
}
