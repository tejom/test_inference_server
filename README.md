# llama2-rs

A standalone Rust port of `llama2.c` inference, tokenizer, sampler, chat loop, and local HTTP generation server. It reads the original `.bin` checkpoint and tokenizer formats. Checkpoint weights are memory mapped, and one transformer instance stays alive across HTTP requests.

## Build

```sh
cargo build --release
cargo build --release --workspace
```

The repository is a Cargo workspace with the existing model package and a
`llama2-router` package.

The port is a separate Git repository. It does not bundle model weights or a tokenizer. From this directory, use the files in the parent `llama2.c` checkout:

```sh
cargo run --release -- ../stories15M.bin -z ../tokenizer.bin -i 'Once upon a time' -n 128
```

Options: `-m generate|chat|server`, `-i` prompt, `-y` system prompt for chat, `-n` total steps, `-t` temperature, `-p` top-p, `-s` seed, `-z` tokenizer path, `-P` HTTP port, `-H` server bind address, and `-C` prefix cache chunk size. `-H` defaults to `127.0.0.1`; the container uses `0.0.0.0`. `-C` defaults to 8; `-C 0` disables prefix caching. The other defaults match `run.c` where practical.

## HTTP

```sh
cargo run --release -- ../stories15M.bin -z ../tokenizer.bin -m server -P 8080 -n 128
curl --data-binary 'Once upon a time' -H 'Content-Type: text/plain' http://127.0.0.1:8080/generate
curl http://127.0.0.1:8080/health
```

`POST /generate` accepts up to 8192 bytes of UTF-8 plain text and returns the full generated text, including the prompt. Sampling options are fixed when the server starts. The model worker takes requests from a FIFO queue, keeps up to four active, and advances them together one token per request per turn. Completed results go to a separate reply worker.

`Transformer` owns a `PrefixCache` keyed by `BlockKey { parent_id, tokens }`. After each completed chunk of prompt tokens, generation copies only that chunk's K/V positions into a `KvTensor` entry. The returned block ID becomes the next chunk's parent ID, so the key includes the preceding context through the chain. Before generation, matching blocks are restored in order and the forward loop resumes at the first uncached token. At least one prompt token is processed normally to produce logits. The map currently grows without eviction.

The HTTP layer uses [`tiny_http`](https://docs.rs/tiny_http/) and checkpoint mapping uses [`memmap2`](https://docs.rs/memmap2/).

## Router

Start model workers on ports 8080 and 8082, then run:

```sh
cargo run -p llama2-router -- router/workers.yaml
```

The YAML file sets the router listen address, health check interval, and worker
URLs. The router checks each worker's `GET /health`. It routes requests to the
best known prompt prefix while pending request counts are balanced; when the
load gap exceeds both one request and a 1.5 ratio, it chooses a least busy
healthy worker. Ties use round robin. Each request runs in its own thread while
the listener accepts more requests. `GET /workers` returns each worker's health
and pending request count, `GET /live` checks the router process, and `GET /ready`
succeeds when at least one worker is healthy.
Worker URLs use plain HTTP, and generation responses are buffered before being
returned to the client.

## Minikube

The Dockerfiles use the parent `llama2.c` directory as their build context. The
worker image includes `stories15M.bin` and `tokenizer.bin` from that directory.
The local setup script builds the worker image on each Minikube node, starts
two worker pods, Prometheus, and Grafana, and forwards their HTTP ports to your host:

```sh
./scripts/minikube-local.sh
```

Use `./scripts/minikube-local.sh --skip-build` to reuse images already loaded
on the Minikube nodes.

The script writes `router/workers.minikube.yaml`, which a host router can read
in another terminal:

```sh
cargo run --release -p llama2-router -- router/workers.minikube.yaml
curl --data-binary 'Once upon a time' http://127.0.0.1:8081/generate
```

The router runs on your host and uses the two forwarded worker ports. The
worker pods use a headless Service for stable names inside the cluster. The
router Dockerfile, Deployment, and NodePort Service are available in the repo
for a later in-cluster experiment, but this script does not use them.

Each worker serves Prometheus text at `/metrics`. Prometheus scrapes both pods
every five seconds. Open `http://127.0.0.1:19090/targets` to check they are
`UP`, then use the Prometheus query page for `llama2_requests_total`,
`llama2_prefix_reused_tokens_total`, or `llama2_batch_items_total`. Sending the
same prompt twice to each worker should increase the reused-token counter.
Prometheus data uses an ephemeral volume and is cleared if its pod is replaced.

The Grafana worker dashboard is at
`http://127.0.0.1:13000/d/llama2-workers` (sign in with `admin` / `admin`
on first use). It refreshes every 5 seconds and shows request rate, total
completed requests, and one-minute averages for prefix reuse, batch size, and
request time for each worker.

## Forward benchmark

Compare four separate forwards with one forward batch of four, over eight positions. The binary prints median time, tokens per second, and speedup across three runs:

```sh
cargo run --release --bin bench_forward -- ../stories15M.bin 4 8 3
```

The same comparison is available as an ignored test:

```sh
LLAMA2_BENCH_MODEL=../stories15M.bin cargo test --release bench_forward -- --ignored --nocapture
```

This measures model forward time only. The HTTP server currently batches up to four active requests.
