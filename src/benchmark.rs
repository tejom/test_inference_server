use crate::model::{ForwardItem, Transformer};
use std::fmt;
use std::hint::black_box;
use std::time::{Duration, Instant};

pub struct ForwardBenchmark {
    pub batch_size: usize,
    pub steps: usize,
    pub separate: Duration,
    pub batched: Duration,
}

impl fmt::Display for ForwardBenchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tokens = (self.batch_size * self.steps) as f64;
        let separate = self.separate.as_secs_f64();
        let batched = self.batched.as_secs_f64();
        write!(
            f,
            "{} requests × {} steps\nseparate: {:.3} s, {:.1} tokens/s\nbatched:  {:.3} s, {:.1} tokens/s\nspeedup:  {:.2}x",
            self.batch_size,
            self.steps,
            separate,
            tokens / separate,
            batched,
            tokens / batched,
            separate / batched,
        )
    }
}

fn run_once(model: &Transformer, batch_size: usize, steps: usize, batched: bool) -> Duration {
    let mut states: Vec<_> = (0..batch_size).map(|_| model.new_state()).collect();
    let start = Instant::now();
    for pos in 0..steps {
        if batched {
            let mut batch: Vec<_> = states
                .iter_mut()
                .enumerate()
                .map(|(i, state)| ForwardItem {
                    state,
                    token: i % model.config.vocab,
                    pos,
                })
                .collect();
            model.forward_batch(&mut batch);
        } else {
            for (i, state) in states.iter_mut().enumerate() {
                model.forward(state, i % model.config.vocab, pos);
            }
        }
    }
    let elapsed = start.elapsed();
    for state in &states {
        black_box(state.logits());
    }
    elapsed
}

/// Compare the same number of single-request forwards with batched forwards.
pub fn benchmark_forward(
    model: &Transformer,
    batch_size: usize,
    steps: usize,
    repeats: usize,
) -> ForwardBenchmark {
    assert!(batch_size > 0 && steps > 0 && steps <= model.config.seq_len && repeats > 0);
    let mut warmup = model.new_state();
    model.forward(&mut warmup, 0, 0);

    let mut separate = Vec::with_capacity(repeats);
    let mut batched = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        for use_batch in [repeat % 2 != 0, repeat % 2 == 0] {
            let elapsed = run_once(model, batch_size, steps, use_batch);
            if use_batch {
                batched.push(elapsed);
            } else {
                separate.push(elapsed);
            }
        }
    }
    separate.sort_unstable();
    batched.sort_unstable();
    ForwardBenchmark {
        batch_size,
        steps,
        separate: separate[repeats / 2],
        batched: batched[repeats / 2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    #[ignore = "set LLAMA2_BENCH_MODEL to a checkpoint path"]
    fn bench_forward() {
        let path = std::env::var("LLAMA2_BENCH_MODEL").expect("set LLAMA2_BENCH_MODEL");
        let model = Transformer::load(Path::new(&path)).unwrap();
        println!("{}", benchmark_forward(&model, 4, 8, 3));
    }
}
