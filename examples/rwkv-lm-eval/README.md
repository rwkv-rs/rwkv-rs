# RWKV Language Model Evaluation

This project provides an example implementation for:

- running `rwkv-eval` benchmarks against OpenAI-compatible completion APIs;
- persisting evaluation tasks, completions, eval results, checker results, and scores into PostgreSQL;
- resuming interrupted evaluation tasks under the same `git_hash`.

## Config Layout

```text
examples/rwkv-lm-eval/
  config/
    example.toml
  scripts/
    init_db.sql
```

- Use `--eval-config <name>` with the config file stem only, no `.toml` suffix.
- Runtime control is now part of `eval/*.toml`, not CLI flags:
  - `run_mode`
  - `skip_checker`
  - `attempt_concurrency`
  - `judger_concurrency`
  - `checker_concurrency`
  - `db_pool_max_connections`

## Run Modes

- `new`
  - create a brand new task;
  - error if the database already contains a matching task identity.
- `resume`
  - requires database persistence to be enabled;
  - requires an existing matching task with status `Running` or `Failed`;
  - requires `git_hash` to match exactly;
  - reuses the old `task_id` and only runs missing attempts.
- `rerun`
  - always creates a new task;
  - keeps prior history intact even if the config is otherwise identical.

Task identity is matched by:

- `config_path`
- `evaluator`
- `git_hash`
- `model_id`
- `benchmark_id`
- `sampling_config`

## Database Setup

Initialize the schema before enabling `[space_db]`:

```bash
psql "postgres://username:password@host:5432/database_name?sslmode=verify-full" \
  -f examples/rwkv-lm-eval/scripts/init_db.sql
```

- `upload_to_space = false` disables PostgreSQL entirely, even if `[space_db]` is filled.
- `upload_to_space = true` requires a complete `[space_db]` config.

## Usage

### Run Evaluator

```bash
git clone https://github.com/rwkv-rs/rwkv-rs.git
cd rwkv-rs

cargo run -p rwkv-lm-eval --example rwkv-lm-eval-test --release -- \
  --config-dir examples/rwkv-lm-eval/config \
  --eval-config example
```

### Run HTTP API

The package now also exposes a read-only Axum API over the persisted evaluation data.
It loads PostgreSQL credentials from the same eval config file and serves an OpenAPI
document at `/openapi.json`.

```bash
cargo run -p rwkv-lm-eval --example rwkv-lm-eval --release -- \
  --config-dir examples/rwkv-lm-eval/config \
  --eval-config example \
  --bind 127.0.0.1:8080
```

Main endpoints:

- `GET /health`
- `GET /openapi.json`
- `GET /api/v1/meta`
- `GET /api/v1/models`
- `GET /api/v1/benchmarks`
- `GET /api/v1/tasks`
- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/tasks/{task_id}/attempts`
- `GET /api/v1/completions/{completions_id}`
- `GET /api/v1/review-queue`

Example `example.toml` controls:

- benchmark/model selection
- target model endpoints
- `llm_judger` / `llm_checker`
- database connectivity
- run mode and concurrency

Model config types:

- `[[models]]` uses `IntApiConfig`
- `[[models]]` must provide `model_arch_version`, `model_data_version`, `model_num_params`
- `[llm_judger]` and `[llm_checker]` use `ExtApiConfig`
- `ExtApiConfig` only requires `base_url`, `api_key`, `model`
- `.env` also supports `RWKV_EVAL_EXT_MODELS` plus role-specific `RWKV_EVAL_LLM_JUDGER_MODELS` / `RWKV_EVAL_LLM_CHECKER_MODELS`
- Fallback model lists are tried in order against the same `base_url` and `api_key`; the first healthy OpenAI-compatible chat model is selected at runtime

## Config Notes

Relevant fields in `example.toml`:

```toml
experiment_name = "your_experiment_name"
experiment_desc = "desc desc"
run_mode = "new"
skip_checker = false
attempt_concurrency = 8
judger_concurrency = 8
checker_concurrency = 8
db_pool_max_connections = 32
git_hash = "a8dc285c786fc425c9effee232453213b4b5ce8e"
```

- `run_mode` must be one of `new`, `resume`, `rerun`.
- `skip_checker = true` disables checker requests and skips retrying missing checker rows on `resume`.
- `attempt_concurrency` must be `> 0`.
- `judger_concurrency` must be `> 0`.
- `checker_concurrency` must be `> 0` when checker is enabled.
- `db_pool_max_connections` must be `> 0`.
- `upload_to_space = false` is the default dry-run path.
- `run_mode = resume` requires `upload_to_space = true`.

## Persistence Behavior

- Every completed attempt is written to `completions` and `eval`.
- Failed answers additionally write one `checker` row.
- `checker` only runs for attempts with `eval.is_passed = false`.
- `checker` is diagnostic only and does not change `eval.is_passed` or `scores`.
- `skip_checker = true` keeps failed attempts in `eval` but skips checker persistence entirely.
- On startup, any persisted `Running` task is marked `Failed`; related `Running` completions are also marked `Failed`.
- If an attempt fails at runtime, the task is marked `Failed` and the process stops.
- `scores` are written only after the task has a complete set of successful attempts.
- `resume` reconstructs progress by reading existing completion/eval rows, reuses finished attempts, and fills both missing attempts and missing checker rows.
- If persistence is disabled, the evaluator does not connect to PostgreSQL and does not persist checker results.

Logs are written under `examples/rwkv-lm-eval/logs/`.
