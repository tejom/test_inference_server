use crate::tokenizer::Tokenizer;
use crate::Sampler;
use llama2_rs::cache::BlockId;
use llama2_rs::model::{State, Transformer};

/// Everything needed to resume one generation request at its next token.
pub struct RequestState {
    pub(crate) prompt_tokens: Vec<usize>,
    pub(crate) token: usize,
    pub(crate) pos: usize,
    steps: usize,
    pub(crate) cache_chunk_size: usize,
    pub(crate) parent_id: Option<BlockId>,
    pub(crate) output: Vec<u8>,
    pub(crate) model_state: State,
    pub(crate) sampler: Sampler,
    pub(crate) stopped: bool,
}

impl RequestState {
    pub fn new(
        model: &Transformer,
        tokenizer: &Tokenizer,
        sampler: Sampler,
        prompt: &str,
        steps: usize,
        cache_chunk_size: usize,
    ) -> Self {
        let prompt_tokens = tokenizer.encode(prompt);
        let mut model_state = model.new_state();
        let (cached_len, parent_id) =
            model.restore_cached_prefix(&mut model_state, &prompt_tokens, cache_chunk_size, steps);
        eprintln!(
            "prefix cache: {cached_len}/{} prompt tokens reused",
            prompt_tokens.len()
        );
        let mut output = Vec::new();
        for i in 1..=cached_len {
            output.extend(tokenizer.decode(prompt_tokens[i - 1], prompt_tokens[i]));
        }
        Self {
            token: prompt_tokens[cached_len],
            pos: cached_len,
            prompt_tokens,
            steps,
            cache_chunk_size,
            parent_id,
            output,
            model_state,
            sampler,
            stopped: false,
        }
    }

    pub fn is_done(&self) -> bool {
        self.stopped || self.pos >= self.steps
    }

    pub fn into_result(mut self) -> (Vec<u8>, Sampler) {
        self.output.push(b'\n');
        (self.output, self.sampler)
    }
}
