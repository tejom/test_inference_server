use crate::cache::BlockId;
use crate::model::{State, Transformer};
use crate::tokenizer::Tokenizer;
use crate::Sampler;

/// Everything needed to resume one generation request at its next token.
pub struct RequestState {
    prompt_tokens: Vec<usize>,
    token: usize,
    pos: usize,
    steps: usize,
    cache_chunk_size: usize,
    parent_id: Option<BlockId>,
    output: Vec<u8>,
    model_state: State,
    sampler: Sampler,
    stopped: bool,
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

    /// Processes one prompt or decode token and advances this request.
    pub fn step(&mut self, model: &mut Transformer, tokenizer: &Tokenizer) {
        if self.is_done() {
            return;
        }
        let pos = self.pos;
        let token = self.token;
        let logits = model.forward(&mut self.model_state, token, pos);
        let next = if pos + 1 < self.prompt_tokens.len() {
            self.prompt_tokens[pos + 1]
        } else {
            self.sampler.sample(logits)
        };
        if next == 1 {
            self.stopped = true;
            return;
        }
        if self.cache_chunk_size > 0
            && pos + 1 < self.prompt_tokens.len()
            && (pos + 1) % self.cache_chunk_size == 0
        {
            let block_start = pos + 1 - self.cache_chunk_size;
            self.parent_id = Some(model.cache_block(
                &self.model_state,
                self.parent_id,
                &self.prompt_tokens[block_start..pos + 1],
                block_start,
            ));
        }
        self.output.extend(tokenizer.decode(token, next));
        self.token = next;
        self.pos += 1;
    }

    pub fn into_result(mut self) -> (Vec<u8>, Sampler) {
        self.output.push(b'\n');
        (self.output, self.sampler)
    }
}

/// Runs the step API to completion while preserving the caller's sampler state.
pub fn generate(
    model: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: &str,
    steps: usize,
    cache_chunk_size: usize,
) -> Vec<u8> {
    let mut request = RequestState::new(
        model,
        tokenizer,
        sampler.clone(),
        prompt,
        steps,
        cache_chunk_size,
    );
    while !request.is_done() {
        request.step(model, tokenizer);
    }
    let (output, next_sampler) = request.into_result();
    *sampler = next_sampler;
    output
}
