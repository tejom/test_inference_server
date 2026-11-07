use std::collections::HashMap;

/// A flat K/V allocation with a layer and position shaped interface.
/// The backing data is laid out as [layer][position][kv_dim].
#[allow(dead_code)] // Used when generation is connected to the cache.
pub struct KvTensor {
    data: Vec<f32>,
    layers: usize,
    positions: usize,
    kv_dim: usize,
}

#[allow(dead_code)]
impl KvTensor {
    pub fn new(layers: usize, positions: usize, kv_dim: usize) -> Self {
        let len = layers
            .checked_mul(positions)
            .and_then(|n| n.checked_mul(kv_dim))
            .expect("KV tensor dimensions overflow");
        Self {
            data: vec![0.0; len],
            layers,
            positions,
            kv_dim,
        }
    }

    pub fn at(&self, layer: usize, position: usize) -> &[f32] {
        let start = self.offset(layer, position);
        &self.data[start..start + self.kv_dim]
    }

    pub fn at_mut(&mut self, layer: usize, position: usize) -> &mut [f32] {
        let start = self.offset(layer, position);
        &mut self.data[start..start + self.kv_dim]
    }

    pub fn layer(&self, layer: usize) -> &[f32] {
        assert!(layer < self.layers);
        let width = self.positions * self.kv_dim;
        &self.data[layer * width..(layer + 1) * width]
    }

    pub fn layer_mut(&mut self, layer: usize) -> &mut [f32] {
        assert!(layer < self.layers);
        let width = self.positions * self.kv_dim;
        &mut self.data[layer * width..(layer + 1) * width]
    }

    fn offset(&self, layer: usize, position: usize) -> usize {
        assert!(layer < self.layers && position < self.positions);
        (layer * self.positions + position) * self.kv_dim
    }
}

/// K/V tensors for a processed token prefix.
#[allow(dead_code)]
pub struct CacheEntry {
    pub keys: KvTensor,
    pub values: KvTensor,
}

/// Exact token sequences identify reusable K/V prefixes for one model.
#[derive(Default)]
#[allow(dead_code)]
pub struct PrefixCache {
    entries: HashMap<Vec<usize>, CacheEntry>,
}

#[allow(dead_code)]
impl PrefixCache {
    pub fn get(&self, tokens: &[usize]) -> Option<&CacheEntry> {
        self.entries.get(tokens)
    }

    pub fn insert(&mut self, tokens: Vec<usize>, entry: CacheEntry) -> Option<CacheEntry> {
        self.entries.insert(tokens, entry)
    }
}
