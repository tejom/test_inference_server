mod cache;
mod model;
mod tokenizer;

use model::Transformer;
use std::env;
use std::error::Error;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Response, Server, StatusCode};
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

fn generate(
    model: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: &str,
    steps: usize,
    cache_chunk_size: usize,
) -> Vec<u8> {
    let prompt_tokens = tokenizer.encode(prompt);
    let mut output = Vec::new();
    let mut token = prompt_tokens[0];
    let mut parent_id = None;
    for pos in 0..steps {
        let logits = model.forward(token, pos);
        let next = if pos + 1 < prompt_tokens.len() {
            prompt_tokens[pos + 1]
        } else {
            sampler.sample(logits)
        };
        if next == 1 {
            break;
        }
        if cache_chunk_size > 0
            && pos + 1 < prompt_tokens.len()
            && (pos + 1) % cache_chunk_size == 0
        {
            let block_start = pos + 1 - cache_chunk_size;
            parent_id = Some(model.cache_block(
                parent_id,
                &prompt_tokens[block_start..pos + 1],
                block_start,
            ));
        }
        output.extend(tokenizer.decode(token, next));
        token = next;
    }
    output.push(b'\n');
    output
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
            next = sampler.sample(model.forward(token, pos));
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
            next = sampler.sample(model.forward(prev, pos));
            pos += 1;
        }
        println!();
        if io::stdin().is_terminal() == false {
            break;
        }
    }
    Ok(())
}

fn response(body: impl Into<Vec<u8>>, status: u16) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(body.into())
        .with_status_code(StatusCode(status))
        .with_header(Header::from_bytes("Content-Type", "text/plain; charset=utf-8").unwrap())
}

fn serve(
    model: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    steps: usize,
    cache_chunk_size: usize,
    port: u16,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let address = format!("127.0.0.1:{port}");
    let server = Server::http(&address)?;
    eprintln!("Listening on http://{address}");
    for mut request in server.incoming_requests() {
        let answer = match (request.method(), request.url()) {
            (&Method::Get, "/health") => response(b"ok\n".to_vec(), 200),
            (&Method::Post, "/generate") => {
                let mut body = Vec::new();
                let mut limited = request.as_reader().take(8193);
                match limited.read_to_end(&mut body) {
                    Ok(_) if body.len() > 8192 => response(b"prompt too large\n".to_vec(), 413),
                    Ok(_) => match String::from_utf8(body) {
                        Ok(prompt) if !prompt.contains('\0') => response(
                            generate(model, tokenizer, sampler, &prompt, steps, cache_chunk_size),
                            200,
                        ),
                        _ => response(b"prompt must be UTF-8 without NUL\n".to_vec(), 400),
                    },
                    Err(_) => response(b"failed to read prompt\n".to_vec(), 400),
                }
            }
            _ => response(b"not found\n".to_vec(), 404),
        };
        if let Err(error) = request.respond(answer) {
            eprintln!("response error: {error}");
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
        "server" => serve(
            &mut model,
            &tokenizer,
            &mut sampler,
            steps,
            options.cache_chunk_size,
            options.port,
        )
        .map_err(|e| -> Box<dyn Error> { e })?,
        _ => return Err(format!("unknown mode: {}", options.mode).into()),
    }
    Ok(())
}
