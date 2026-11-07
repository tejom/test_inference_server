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

pub type BlockId = u64;

/// The parent block identifies all preceding tokens and their positions.
#[derive(Clone, Hash, Eq, PartialEq)]
pub struct BlockKey {
    pub parent_id: Option<BlockId>,
    pub tokens: Vec<usize>,
}

impl BlockKey {
    pub fn new(parent_id: Option<BlockId>, tokens: &[usize]) -> Self {
        Self {
            parent_id,
            tokens: tokens.to_vec(),
        }
    }
}

/// K/V tensors for one processed token block.
#[allow(dead_code)]
pub struct BlockEntry {
    pub id: BlockId,
    pub keys: KvTensor,
    pub values: KvTensor,
}

/// A block is identified by its parent and its own token IDs.
#[derive(Default)]
#[allow(dead_code)]
pub struct PrefixCache {
    entries: HashMap<BlockKey, BlockEntry>,
    next_id: BlockId,
}

#[allow(dead_code)]
impl PrefixCache {
    pub fn get(&self, key: &BlockKey) -> Option<&BlockEntry> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: BlockKey, keys: KvTensor, values: KvTensor) -> BlockId {
        if let Some(entry) = self.entries.get(&key) {
            return entry.id;
        }
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("prefix cache block IDs exhausted");
        let id = self.next_id;
        self.entries.insert(key, BlockEntry { id, keys, values });
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor() -> KvTensor {
        KvTensor::new(1, 2, 1)
    }

    #[test]
    fn block_identity_includes_parent() {
        let mut cache = PrefixCache::default();
        let first = cache.insert(BlockKey::new(None, &[1, 2]), tensor(), tensor());
        let child = cache.insert(BlockKey::new(Some(first), &[3, 4]), tensor(), tensor());
        let root = cache.insert(BlockKey::new(None, &[3, 4]), tensor(), tensor());
        let first_again = cache.insert(BlockKey::new(None, &[1, 2]), tensor(), tensor());

        assert_ne!(child, root);
        assert_eq!(first, first_again);
        assert_eq!(cache.entries.len(), 3);
    }
}
