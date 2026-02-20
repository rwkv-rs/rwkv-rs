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
    rwkv-lm-7.2b.toml
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
  --infer-cfg rwkv-lm-7.2b
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
  --infer-cfg rwkv-lm-7.2b
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
  --infer-cfg rwkv-lm-7.2b
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
  "model": "rwkv-lm-7.2b",
  "messages": [
    {"role": "User", "content": "Hello!"}
  ],
  "stream": false,
  "max_tokens": 4096,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "presence_penalty": 0.2,
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
    "model": "rwkv-lm-7.2b",
    "messages": [{"role": "User", "content": "Hello!"}],
    "stream": false,
    "max_tokens": 4096,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "presence_penalty": 0.2,
    "repetition_penalty": 0.5,
    "penalty_decay": 0.996,
    "stop": ["\nUser:"]
  }'
```

### Admin hot-reload request body example

```json
{
  "upsert": [
    {
      "model_name": "rwkv-lm-7.2b",
      "model_cfg": "rwkv-lm-7.2b",
      "weights_path": "../../weights/rwkv7-g1d-7.2b-20260131-ctx8192.mpk",
      "tokenizer_vocab_path": "../../assets/rwkv_vocab_v20230424.txt",
      "device_type": 0,
      "device_ids": [0],
      "max_batch_size": 4,
      "max_context_len": 4096,
      "paragraph_len": 256,
      "decode_first": true
    }
  ],
  "remove_model_names": [],
  "dry_run": false
}
```

## Benchmark & Profiling Workflow

Use layered benchmarking:

1. `rwkv-nn` kernel microbench (Divan)
2. runtime stage profiling (Tracy)
3. system GPU timeline (nsys)
4. serving pressure benchmark + report

### 1) Kernel microbench (`crates/rwkv-nn/benches`)

Run one kernel target (backend selected by features):

```bash
cargo bench --bench wkv7_pretrain_forward --features cuda
cargo bench --bench wkv7_statepass_backward --features cuda
cargo bench --bench rapid_sample_forward --features cuda
```

Available targets:
- `wkv7_pretrain_forward`
- `wkv7_pretrain_backward`
- `wkv7_statepass_forward`
- `wkv7_statepass_backward`
- `wkv7_statetune_forward`
- `wkv7_statetune_backward`
- `wkv7_infer_forward`
- `rapid_sample_forward`

### 2/3/4) Scenario benches (`examples/rwkv-lm/benches/`)

```bash
# Serving pressure benchmark
cargo bench --bench serve_bench -- \
  --model rwkv-lm-7.2b --base-url http://127.0.0.1:8080

# Sweep benchmark matrix
cargo bench --bench sweep_bench -- \
  --model rwkv-lm-7.2b --base-url http://127.0.0.1:8080

# nsys profiling wrapper
cargo bench --bench profile_nsys_bench -- \
  --output-prefix logs/bench/nsys/rwkv -- \
  cargo run --example rwkv-lm-infer --features cuda

# tracy passthrough
cargo bench --bench profile_tracy_bench -- \
  cargo run --example rwkv-lm-infer --features cuda

# Generate report from benchmark JSON
cargo bench --bench report_bench -- \zh
  --input-json logs/bench/serve.json --output-dir logs/bench/report

# Record arbitrary train benchmark command
cargo bench --bench train_command_bench -- \
  --output-json logs/bench/train_run.json -- \
  cargo run --example rwkv-lm-train --features cuda
```

Serving/sweep benches write JSON results and static Markdown/SVG reports.
