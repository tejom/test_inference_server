use std::collections::HashMap;

// The storage API is intentionally unused until generation performs cache lookup.
#[allow(dead_code)]
/// K/V vectors for a processed token prefix, laid out as
/// [layer][cached position][kv_dim] in each vector.
pub struct CacheEntry {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
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
