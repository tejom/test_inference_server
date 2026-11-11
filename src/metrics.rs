use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Default)]
pub struct Metrics {
    requests: AtomicU64,
    prompt_tokens: AtomicU64,
    reused_tokens: AtomicU64,
    generated_tokens: AtomicU64,
    batch_steps: AtomicU64,
    batch_items: AtomicU64,
    duration_us: AtomicU64,
}

impl Metrics {
    pub fn started(&self, prompt_tokens: usize, reused_tokens: usize) {
        self.prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.reused_tokens
            .fetch_add(reused_tokens as u64, Ordering::Relaxed);
    }

    pub fn batch_step(&self, items: usize) {
        self.batch_steps.fetch_add(1, Ordering::Relaxed);
        self.batch_items.fetch_add(items as u64, Ordering::Relaxed);
    }

    pub fn completed(&self, generated_tokens: usize, duration: Duration) {
        self.generated_tokens
            .fetch_add(generated_tokens as u64, Ordering::Relaxed);
        self.duration_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);
        self.requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn render(&self) -> Vec<u8> {
        let get = |value: &AtomicU64| value.load(Ordering::Relaxed);
        format!(
            "# HELP llama2_requests_total Completed generation requests.\n\
             # TYPE llama2_requests_total counter\n\
             llama2_requests_total {}\n\
             # HELP llama2_prompt_tokens_total Prompt tokens received.\n\
             # TYPE llama2_prompt_tokens_total counter\n\
             llama2_prompt_tokens_total {}\n\
             # HELP llama2_prefix_reused_tokens_total Prompt tokens restored from the prefix cache.\n\
             # TYPE llama2_prefix_reused_tokens_total counter\n\
             llama2_prefix_reused_tokens_total {}\n\
             # HELP llama2_generated_tokens_total Output tokens generated.\n\
             # TYPE llama2_generated_tokens_total counter\n\
             llama2_generated_tokens_total {}\n\
             # HELP llama2_batch_steps_total Batched forward calls.\n\
             # TYPE llama2_batch_steps_total counter\n\
             llama2_batch_steps_total {}\n\
             # HELP llama2_batch_items_total Requests processed across batched forward calls.\n\
             # TYPE llama2_batch_items_total counter\n\
             llama2_batch_items_total {}\n\
             # HELP llama2_request_duration_seconds Request time from receipt to model completion.\n\
             # TYPE llama2_request_duration_seconds summary\n\
             llama2_request_duration_seconds_sum {:.6}\n\
             llama2_request_duration_seconds_count {}\n",
            get(&self.requests),
            get(&self.prompt_tokens),
            get(&self.reused_tokens),
            get(&self.generated_tokens),
            get(&self.batch_steps),
            get(&self.batch_items),
            get(&self.duration_us) as f64 / 1_000_000.0,
            get(&self.requests),
        )
        .into_bytes()
    }
}
