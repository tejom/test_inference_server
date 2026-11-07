# llama2-rs

A standalone Rust port of `llama2.c` inference, tokenizer, sampler, chat loop, and local HTTP generation server. It reads the original `.bin` checkpoint and tokenizer formats. Checkpoint weights are memory mapped, and one transformer instance stays alive across HTTP requests.

## Build

```sh
cargo build --release
```

The port is a separate Git repository. It does not bundle model weights or a tokenizer. From this directory, use the files in the parent `llama2.c` checkout:

```sh
cargo run --release -- ../stories15M.bin -z ../tokenizer.bin -i 'Once upon a time' -n 128
```

Options: `-m generate|chat|server`, `-i` prompt, `-y` system prompt for chat, `-n` total steps, `-t` temperature, `-p` top-p, `-s` seed, `-z` tokenizer path, `-P` HTTP port, and `-C` prefix cache chunk size. `-C` defaults to 8; `-C 0` disables cache storage. The other defaults match `run.c` where practical.

## HTTP

```sh
cargo run --release -- ../stories15M.bin -z ../tokenizer.bin -m server -P 8080 -n 128
curl --data-binary 'Once upon a time' -H 'Content-Type: text/plain' http://127.0.0.1:8080/generate
curl http://127.0.0.1:8080/health
```

`POST /generate` accepts up to 8192 bytes of UTF-8 plain text and returns the full generated text, including the prompt. Sampling options are fixed when the server starts. Requests run sequentially so one model and its KV buffers can be reused safely. Prefix cache restoration is not implemented yet; each request currently starts at position zero.

`Transformer` owns a `PrefixCache` with exact token-sequence keys. After each completed chunk of prompt tokens, generation copies all processed K/V positions into a `KvTensor` entry. The map currently grows without eviction. Lookup and restoration are not wired into generation yet, so cache entries do not speed up requests yet.

The HTTP layer uses [`tiny_http`](https://docs.rs/tiny_http/) and checkpoint mapping uses [`memmap2`](https://docs.rs/memmap2/).
