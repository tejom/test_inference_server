mod generation;
mod metrics;
mod request;
#[allow(dead_code)] // Fields reserved for the upcoming scheduler.
mod server;
mod tokenizer;

use generation::generate;
use llama2_rs::model::Transformer;
use std::env;
use std::error::Error;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizer::Tokenizer;

struct Options {
    checkpoint: PathBuf,
    tokenizer: PathBuf,
    mode: String,
    prompt: Option<String>,
    system_prompt: Option<String>,
    steps: usize,
    cache_chunk_size: usize,
    temperature: f32,
    topp: f32,
    seed: u64,
    port: u16,
    host: String,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = env::args().skip(1);
        let checkpoint = PathBuf::from(
            args.next()
                .ok_or("usage: llama2-rs <checkpoint> [options]")?,
        );
        let mut options = Self {
            checkpoint,
            tokenizer: PathBuf::from("tokenizer.bin"),
            mode: "generate".into(),
            prompt: None,
            system_prompt: None,
            steps: 256,
            cache_chunk_size: 8,
            temperature: 1.0,
            topp: 0.9,
            seed: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            port: 8080,
            host: "127.0.0.1".into(),
        };
        while let Some(flag) = args.next() {
            let value = args.next().ok_or(format!("missing value for {flag}"))?;
            match flag.as_str() {
                "-z" => options.tokenizer = value.into(),
                "-m" => options.mode = value,
                "-i" => options.prompt = Some(value),
                "-y" => options.system_prompt = Some(value),
                "-n" => options.steps = value.parse()?,
                "-C" => options.cache_chunk_size = value.parse()?,
                "-t" => options.temperature = value.parse()?,
                "-p" => options.topp = value.parse()?,
                "-s" => options.seed = value.parse()?,
                "-P" => options.port = value.parse()?,
                "-H" => options.host = value,
                _ => return Err(format!("unknown option: {flag}").into()),
            }
        }
        if !options.temperature.is_finite()
            || options.temperature < 0.0
            || !options.topp.is_finite()
            || !(0.0..=1.0).contains(&options.topp)
        {
            return Err("invalid sampling parameters".into());
        }
        if options.seed == 0 {
            options.seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        }
        Ok(options)
    }
}

#[derive(Clone)]
struct Sampler {
    temperature: f32,
    topp: f32,
    state: u64,
}
impl Sampler {
    fn sample(&mut self, logits: &[f32]) -> usize {
        if self.temperature == 0.0 {
            let mut best = 0;
            for i in 1..logits.len() {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            return best;
        }
        let mut probs: Vec<f32> = logits.iter().map(|&v| v / self.temperature).collect();
        let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for p in &mut probs {
            *p = (*p - max).exp();
            sum += *p;
        }
        for p in &mut probs {
            *p /= sum;
        }
        let coin = self.random();
        if self.topp > 0.0 && self.topp < 1.0 {
            let cutoff = (1.0 - self.topp) / (probs.len() - 1) as f32;
            let mut indices: Vec<usize> =
                (0..probs.len()).filter(|&i| probs[i] >= cutoff).collect();
            indices.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            let mut cumulative = 0.0;
            let mut end = indices.len();
            for (j, &i) in indices.iter().enumerate() {
                cumulative += probs[i];
                if cumulative > self.topp {
                    end = j + 1;
                    break;
                }
            }
            let mut r = coin * cumulative;
            for &i in &indices[..end] {
                r -= probs[i];
                if r < 0.0 {
                    return i;
                }
            }
            *indices.get(end.saturating_sub(1)).unwrap_or(&0)
        } else {
            let mut r = coin;
            for (i, &p) in probs.iter().enumerate() {
                r -= p;
                if r < 0.0 {
                    return i;
                }
            }
            probs.len() - 1
        }
    }
    fn random(&mut self) -> f32 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let value = ((self.state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32) >> 8;
        value as f32 / 16_777_216.0
    }
}

fn read_line(label: &str) -> io::Result<String> {
    print!("{label}");
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(line.trim_end_matches(['\r', '\n']).to_owned())
}

fn chat(
    model: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    options: &Options,
    steps: usize,
) -> io::Result<()> {
    let system = match &options.system_prompt {
        Some(s) => s.clone(),
        None => read_line("Enter system prompt (optional): ")?,
    };
    let mut first = true;
    let mut pos = 0;
    let mut state = model.new_state();
    while pos < steps {
        let user = if first {
            match &options.prompt {
                Some(prompt) => prompt.clone(),
                None => read_line("User: ")?,
            }
        } else {
            read_line("User: ")?
        };
        let rendered = if first && !system.is_empty() {
            format!("[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]")
        } else {
            format!("[INST] {user} [/INST]")
        };
        first = false;
        let tokens = tokenizer.encode(&rendered);
        print!("Assistant: ");
        io::stdout().flush()?;
        let mut next = 0;
        let mut prev = 0;
        let mut consumed = 0;
        for &token in &tokens {
            if pos >= steps {
                break;
            }
            next = sampler.sample(model.forward(&mut state, token, pos));
            pos += 1;
            prev = token;
            consumed += 1;
        }
        if consumed < tokens.len() {
            println!();
            break;
        }
        while next != 2 {
            let piece = tokenizer.decode(prev, next);
            io::stdout().write_all(&piece)?;
            io::stdout().flush()?;
            if pos >= steps {
                break;
            }
            prev = next;
            next = sampler.sample(model.forward(&mut state, prev, pos));
            pos += 1;
        }
        println!();
        if io::stdin().is_terminal() == false {
            break;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let mut model = Transformer::load(&options.checkpoint)?;
    let tokenizer = Tokenizer::load(&options.tokenizer, model.config.vocab)?;
    let steps = if options.steps == 0 {
        model.config.seq_len
    } else {
        options.steps.min(model.config.seq_len)
    };
    let mut sampler = Sampler {
        temperature: options.temperature,
        topp: options.topp,
        state: options.seed,
    };
    match options.mode.as_str() {
        "generate" => {
            let output = generate(
                &mut model,
                &tokenizer,
                &mut sampler,
                options.prompt.as_deref().unwrap_or(""),
                steps,
                options.cache_chunk_size,
            );
            io::stdout().write_all(&output)?;
        }
        "chat" => chat(&mut model, &tokenizer, &mut sampler, &options, steps)?,
        "server" => {
            let server = server::Server::new(
                &options.host,
                options.port,
                model,
                tokenizer,
                sampler,
                options.cache_chunk_size,
            )?;
            server.serve(steps);
            if server.shutdown().is_err() {
                return Err("server worker panicked".into());
            }
        }
        _ => return Err(format!("unknown mode: {}", options.mode).into()),
    }
    Ok(())
}
