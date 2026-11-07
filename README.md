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

Options: `-m generate|chat|server`, `-i` prompt, `-y` system prompt for chat, `-n` total steps, `-t` temperature, `-p` top-p, `-s` seed, `-z` tokenizer path, `-P` HTTP port, and `-C` prefix cache chunk size. `-C` defaults to 8; `-C 0` disables prefix caching. The other defaults match `run.c` where practical.

## HTTP

```sh
cargo run --release -- ../stories15M.bin -z ../tokenizer.bin -m server -P 8080 -n 128
curl --data-binary 'Once upon a time' -H 'Content-Type: text/plain' http://127.0.0.1:8080/generate
curl http://127.0.0.1:8080/health
```

`POST /generate` accepts up to 8192 bytes of UTF-8 plain text and returns the full generated text, including the prompt. Sampling options are fixed when the server starts. The HTTP listener queues requests while one model worker processes them sequentially; a separate reply worker sends completed responses. Batching is not implemented yet.

`Transformer` owns a `PrefixCache` keyed by `BlockKey { parent_id, tokens }`. After each completed chunk of prompt tokens, generation copies only that chunk's K/V positions into a `KvTensor` entry. The returned block ID becomes the next chunk's parent ID, so the key includes the preceding context through the chain. Before generation, matching blocks are restored in order and the forward loop resumes at the first uncached token. At least one prompt token is processed normally to produce logits. The map currently grows without eviction.

The HTTP layer uses [`tiny_http`](https://docs.rs/tiny_http/) and checkpoint mapping uses [`memmap2`](https://docs.rs/memmap2/).
