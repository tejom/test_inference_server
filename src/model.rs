use crate::cache::{CacheEntry, KvTensor, PrefixCache};
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

struct State {
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
}

fn snapshot_kv(c: Config, state: &State, len: usize) -> CacheEntry {
    let kv_dim = c.kv_dim();
    let mut keys = KvTensor::new(c.layers, len, kv_dim);
    let mut values = KvTensor::new(c.layers, len, kv_dim);
    for layer in 0..c.layers {
        let start = layer * c.seq_len * kv_dim;
        let end = start + len * kv_dim;
        keys.layer_mut(layer)
            .copy_from_slice(&state.keys[start..end]);
        values
            .layer_mut(layer)
            .copy_from_slice(&state.values[start..end]);
    }
    CacheEntry { keys, values }
}

pub struct Transformer {
    pub config: Config,
    mmap: Mmap,
    weights: Weights,
    state: State,
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
        let state = State::new(c)?;
        Ok(Self {
            config: c,
            mmap,
            weights,
            state,
            prefix_cache: PrefixCache::default(),
        })
    }

    /// Save the already processed positions in `tokens` as a full prefix snapshot.
    pub fn cache_prefix(&mut self, tokens: &[usize]) {
        let len = tokens.len();
        if len == 0 || len > self.config.seq_len || self.prefix_cache.get(tokens).is_some() {
            return;
        }
        let entry = snapshot_kv(self.config, &self.state, len);
        self.prefix_cache.insert(tokens.to_vec(), entry);
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        let c = self.config;
        assert!(token < c.vocab && pos < c.seq_len);
        // mmap starts on a page boundary; the checkpoint is little-endian f32 data
        // after its seven-i32 header. It remains read-only for this model's lifetime.
        let f = unsafe {
            std::slice::from_raw_parts(self.mmap.as_ptr().cast::<f32>(), self.mmap.len() / 4)
        };
        let w = &self.weights;
        let s = &mut self.state;
        let d = c.dim;
        let kv = c.kv_dim();
        let head_size = d / c.heads;
        s.x.copy_from_slice(&f[w.embedding.start + token * d..w.embedding.start + (token + 1) * d]);
        for layer in 0..c.layers {
            rmsnorm(&mut s.xb, &s.x, &f[w.rms_att.start + layer * d..][..d]);
            let base = (layer * c.seq_len + pos) * kv;
            matmul(
                &mut s.q,
                &s.xb,
                &f[w.wq.start + layer * d * d..][..d * d],
                d,
            );
            matmul(
                &mut s.keys[base..base + kv],
                &s.xb,
                &f[w.wk.start + layer * d * kv..][..d * kv],
                d,
            );
            matmul(
                &mut s.values[base..base + kv],
                &s.xb,
                &f[w.wv.start + layer * d * kv..][..d * kv],
                d,
            );
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
            matmul(
                &mut s.xb2,
                &s.xb,
                &f[w.wo.start + layer * d * d..][..d * d],
                d,
            );
            for i in 0..d {
                s.x[i] += s.xb2[i];
            }
            rmsnorm(&mut s.xb, &s.x, &f[w.rms_ffn.start + layer * d..][..d]);
            matmul(
                &mut s.hb,
                &s.xb,
                &f[w.w1.start + layer * d * c.hidden_dim..][..d * c.hidden_dim],
                d,
            );
            matmul(
                &mut s.hb2,
                &s.xb,
                &f[w.w3.start + layer * d * c.hidden_dim..][..d * c.hidden_dim],
                d,
            );
            for i in 0..c.hidden_dim {
                let value = s.hb[i];
                s.hb[i] = value / (1.0 + (-value).exp()) * s.hb2[i];
            }
            matmul(
                &mut s.xb,
                &s.hb,
                &f[w.w2.start + layer * d * c.hidden_dim..][..d * c.hidden_dim],
                c.hidden_dim,
            );
            for i in 0..d {
                s.x[i] += s.xb[i];
            }
        }
        rmsnorm_inplace(&mut s.x, &f[w.rms_final.clone()]);
        matmul(&mut s.logits, &s.x, &f[w.classifier.clone()], d);
        &s.logits
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

    #[test]
    fn snapshot_uses_compact_layer_stride() {
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

        let snapshot = snapshot_kv(c, &state, 2);
        assert_eq!(snapshot.keys.layer(0), &state.keys[0..8]);
        assert_eq!(snapshot.keys.layer(1), &state.keys[16..24]);
        assert_eq!(snapshot.values.layer(0), &state.values[0..8]);
        assert_eq!(snapshot.values.layer(1), &state.values[16..24]);
    }
}
