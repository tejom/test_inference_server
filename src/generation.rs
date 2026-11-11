use crate::request::RequestState;
use crate::tokenizer::Tokenizer;
use crate::Sampler;
use llama2_rs::model::{ForwardItem, Transformer};

/// Advances one request by one prompt or decode token.
pub fn step_request(model: &mut Transformer, tokenizer: &Tokenizer, request: &mut RequestState) {
    step_batch(model, tokenizer, &mut [request]);
}

/// Advances each active request by one prompt or decode token.
pub fn step_batch(
    model: &mut Transformer,
    tokenizer: &Tokenizer,
    requests: &mut [&mut RequestState],
) {
    let mut batch: Vec<_> = requests
        .iter_mut()
        .filter(|request| !request.is_done())
        .map(|request| ForwardItem {
            state: &mut request.model_state,
            token: request.token,
            pos: request.pos,
        })
        .collect();
    if batch.is_empty() {
        return;
    }
    model.forward_batch(&mut batch);

    for request in requests.iter_mut().filter(|request| !request.is_done()) {
        let pos = request.pos;
        let token = request.token;
        if request.cache_chunk_size > 0 && (pos + 1) % request.cache_chunk_size == 0 {
            let block_start = pos + 1 - request.cache_chunk_size;
            request.parent_id = Some(model.cache_block(
                &request.model_state,
                request.parent_id,
                &request.token_history[block_start..pos + 1],
                block_start,
            ));
        }
        let next = if pos + 1 < request.prompt_tokens.len() {
            request.prompt_tokens[pos + 1]
        } else {
            request.sampler.sample(request.model_state.logits())
        };
        if next == 1 {
            request.stopped = true;
            continue;
        }
        if pos + 1 >= request.prompt_tokens.len() {
            request.generated_tokens += 1;
            request.token_history.push(next);
        }
        request.output.extend(tokenizer.decode(token, next));
        request.token = next;
        request.pos += 1;
    }
}

/// CLI generation uses the same step function as the server worker.
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
        step_request(model, tokenizer, &mut request);
    }
    let (output, next_sampler) = request.into_result();
    *sampler = next_sampler;
    output
}
