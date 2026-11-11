use std::collections::HashMap;

#[derive(Default)]
struct Node {
    value: String,
    workers: Vec<usize>,
    children: HashMap<String, Node>,
}

impl Node {
    fn add_worker(&mut self, worker: usize) {
        if !self.workers.contains(&worker) {
            self.workers.push(worker);
        }
    }

    fn remove_worker(&mut self, worker: usize) {
        self.workers.retain(|&candidate| candidate != worker);
        self.children.retain(|_, child| {
            child.remove_worker(worker);
            !child.workers.is_empty()
        });
    }
}

#[derive(Default)]
pub struct PrefixTree {
    root: Node,
}

pub struct PrefixMatch<'a> {
    pub matched_words: usize,
    pub workers: &'a [usize],
}

impl PrefixTree {
    pub fn new(worker_count: usize) -> Self {
        Self {
            root: Node {
                workers: (0..worker_count).collect(),
                ..Node::default()
            },
        }
    }

    pub fn insert(&mut self, prompt: &str, worker: usize) {
        let mut node = &mut self.root;
        node.add_worker(worker);
        for word in prompt.split_whitespace() {
            node = node
                .children
                .entry(word.to_owned())
                .or_insert_with(|| Node {
                    value: word.to_owned(),
                    ..Node::default()
                });
            node.add_worker(worker);
        }
    }

    pub fn remove_worker(&mut self, worker: usize) {
        // Keep the root's worker list for fallback after a worker recovers.
        self.root.children.retain(|_, child| {
            child.remove_worker(worker);
            !child.workers.is_empty()
        });
    }

    pub fn find(&self, prompt: &str) -> PrefixMatch<'_> {
        let mut node = &self.root;
        let mut matched_words = 0;
        for word in prompt.split_whitespace() {
            let Some(child) = node.children.get(word) else {
                break;
            };
            node = child;
            matched_words += 1;
        }
        PrefixMatch {
            matched_words,
            workers: &node.workers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_prefix_returns_all_workers_at_last_matching_word() {
        let mut tree = PrefixTree::default();
        tree.insert("Once upon a time", 0);
        tree.insert("Once upon a hill", 1);

        let result = tree.find("Once upon a tiger");
        assert_eq!(result.matched_words, 3);
        assert_eq!(result.workers, &[0, 1]);

        let result = tree.find("Once upon a time again");
        assert_eq!(result.matched_words, 4);
        assert_eq!(result.workers, &[0]);
        assert_eq!(tree.root.children["Once"].value, "Once");
    }

    #[test]
    fn no_match_returns_root_workers_without_duplicates() {
        let mut tree = PrefixTree::new(3);
        tree.insert("Once upon a time", 0);
        tree.insert("Once upon a hill", 1);
        tree.insert("Once upon a time", 0);

        let result = tree.find("Another story");
        assert_eq!(result.matched_words, 0);
        assert_eq!(result.workers, &[0, 1, 2]);
    }

    #[test]
    fn removing_worker_prunes_only_its_branches() {
        let mut tree = PrefixTree::new(2);
        tree.insert("Once upon a time", 0);
        tree.insert("Once upon a hill", 1);
        tree.insert("Another story", 0);

        tree.remove_worker(0);

        assert_eq!(tree.find("Once upon a time").workers, &[1]);
        assert_eq!(tree.find("Once upon a hill").workers, &[1]);
        assert!(!tree.root.children.contains_key("Another"));
        assert_eq!(tree.find("Another story").workers, &[0, 1]);
    }
}
