# RWKV Language Model

This project provides an example implementation for:

- pretraining RWKV language models on MiniPile datasets;
- serving OpenAI-compatible inference APIs with hot-reloadable `infer` config.

## Dataset Details

- [The MiniPile Challenge for Data-Efficient Language Models](https://arxiv.org/abs/2304.08442)
- MiniPile is a 6GB subset of the [deduplicated The Pile corpus](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated).
- More details on the MiniPile curation procedure and pre-training results can be found in the [MiniPile paper](https://arxiv.org/abs/2304.08442).
- For more details on the Pile corpus, see [the Pile datasheet](https://arxiv.org/abs/2201.07311).
- You can download tokenized data from [BlinkDL/minipile-tokenized](https://huggingface.co/datasets/BlinkDL/minipile-tokenized).

## Config Layout

```text
examples/rwkv-lm/config/
  model/
    rwkv-lm-0.1b.toml
    rwkv-lm-7.2b.toml
  train/
    rwkv-lm-0.1b.toml
  infer/
    rwkv-7.2b-g1e.toml
```

- `train/*.toml` and `infer/*.toml` both reference model config via `model_cfg`.
- Use `--train-cfg <name>` / `--infer-cfg <name>` (name only, no `.toml` suffix).

# Usage

## CUDA backend (Recommended)

```bash
git clone https://github.com/rwkv-rs/rwkv-rs.git
cd rwkv-rs

# Train (use --release for speed; add f16 feature if needed)
cargo run --example rwkv-lm-train --release --features cuda -- \
  --config-dir examples/rwkv-lm/config \
  --train-cfg rwkv-lm-0.1b

# Infer server
cargo run --example rwkv-lm-infer --release --features cuda -- \
  --config-dir examples/rwkv-lm/config \
  --infer-cfg rwkv-7.2b-g1e
```

## WGPU backend

```bash
git clone https://github.com/rwkv-rs/rwkv-rs.git
cd rwkv-rs

# Train
cargo run --example rwkv-lm-train --release --features wgpu -- \
  --config-dir examples/rwkv-lm/config \
  --train-cfg rwkv-lm-0.1b

# Infer server
cargo run --example rwkv-lm-infer --release --features wgpu -- \
  --config-dir examples/rwkv-lm/config \
  --infer-cfg rwkv-7.2b-g1e
```

## Metal backend

```bash
git clone https://github.com/rwkv-rs/rwkv-rs.git
cd rwkv-rs

# Train
cargo run --example rwkv-lm-train --release --features metal -- \
  --config-dir examples/rwkv-lm/config \
  --train-cfg rwkv-lm-0.1b

# Infer server
cargo run --example rwkv-lm-infer --release --features metal -- \
  --config-dir examples/rwkv-lm/config \
  --infer-cfg rwkv-7.2b-g1e
```

## Inference API Examples

After the infer server starts (default bind address from `infer.toml` is `0.0.0.0:8080`), you can call:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `POST /admin/models/reload` (hot-reload models in `infer/*.toml`)

If `api_key` is set in infer config, add:

```http
Authorization: Bearer <your_api_key>
```

### Chat Completions request body example

```json
{
  "model": "rwkv-7.2b-g1e",
  "messages": [
    {"role": "User", "content": "Hello!"}
  ],
  "stream": false,
  "max_tokens": 4096,
  "temperature": 1.0,
  "top_k": 500,
  "top_p": 0.3,
  "presence_penalty": 0.5,
  "repetition_penalty": 0.5,
  "penalty_decay": 0.996,
  "stop": ["\nUser:"]
}
```

Call example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-7.2b-g1e",
    "messages": [{"role": "User", "content": "Hello!"}],
    "stream": false,
    "max_tokens": 4096,
    "temperature": 1.0,
    "top_k": 500,
    "top_p": 0.3,
    "presence_penalty": 0.5,
    "repetition_penalty": 0.5,
    "penalty_decay": 0.996,
    "stop": ["\nUser:"]
  }'
```

### Sampling parameter validation

The infer API rejects invalid sampling parameters with `400 invalid_request_error`:

- `temperature`: finite and in `[0.001, 1000]`
- `top_p`: finite and in `[0, 1]`
- `top_k`: `>= 0` (`0` means disabled)
- `max_tokens` / `max_output_tokens`: `>= 1`
- `presence_penalty` / `repetition_penalty` / `penalty_decay`: finite

### Admin hot-reload request body example

```json
{
  "upsert": [
    {
      "model_name": "rwkv-7.2b-g1e",
      "model_cfg": "rwkv-lm-7.2b",
      "weights_path": "../../weights/rwkv7-g1d-7.2b-20260131-ctx8192.bpk",
      "tokenizer_vocab_path": "../../assets/rwkv_vocab_v20230424.txt",
      "device_type": 0,
      "device_ids": [0],
      "max_batch_size": 4,
      "max_context_len": 4096,
      "paragraph_len": 256,
      "decode_first": true
    }
  ],
  "remove_model_names": []
}
```

## Local Bench

`examples/rwkv-lm` now provides a single inference bench:

- target: `rwkv-lm-infer-bench`
- transport: local SDK only
- request shape: equivalent to `POST /v1/completions`
- model selection: runs every model declared in `infer/*.toml`

It does not start the HTTP server. The bench builds the local client directly, then submits one burst per concurrency level.

### Run

CUDA example:

```bash
cargo bench -p rwkv-lm --bench rwkv-lm-infer-bench --features cuda -- \
  --config-dir config \
  --infer-cfg rwkv-7.2b-g1e
```

WGPU example:

```bash
cargo bench -p rwkv-lm --bench rwkv-lm-infer-bench --no-default-features --features inferring,wgpu,std,nn,cubecl -- \
  --config-dir config \
  --infer-cfg rwkv-7.2b-g1e
```

Optional output override:

```bash
cargo bench -p rwkv-lm --bench rwkv-lm-infer-bench --features cuda -- \
  --config-dir config \
  --infer-cfg rwkv-7.2b-g1e \
  --output-dir /tmp/rwkv-bench
```

Default output directory:

```text
examples/rwkv-lm/logs/bench/rwkv-lm-infer-bench/<infer_cfg>/<model_name>/
```

### Workload

For each model in the infer config:

- concurrency levels are `1, 2, 4, ...`
- when the next doubling would exceed `max_batch_size`, the bench runs one final level at `max_batch_size`
- each concurrency level runs exactly one round
- the bench prints terminal summaries only; chart artifacts are written as SVG

Fixed prompt:

```text
User: You are a very talented expert in abstract algebra.
Answer this question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
A. 0
B. 4
C. 2
D. 6

Assistant: <think>
```

Fixed decoding params:

```json
{
  "max_tokens": 4096,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "presence_penalty": 0.5,
  "repetition_penalty": 0.5,
  "penalty_decay": 0.996,
  "stop": ["\n</think>", "\n\nUser: "]
}
```

### Artifacts

For the best concurrency level of each model, the bench polls local `/health` data during execution and writes SVG charts with `kuva`.

Per model, the output directory contains:

- `concurrency_vs_request_throughput.svg`
- `concurrency_vs_output_tokens_per_sec.svg`
- `concurrency_vs_mean_latency_ms.svg`
- `best_concurrency_queue_prefill_tps.svg`
- `best_concurrency_queue_decode_tps.svg`
- `best_concurrency_queue_batch_utilization.svg`
- `best_concurrency_gpu_utilization.svg`
- `best_concurrency_gpu_memory_utilization.svg`
