# Agent Benchmark Integration Design

This document defines a no-interface-change integration plan for three agent benchmarks in
`rwkv-eval`:

- `browsecomp`
- `browsecomp_zh`
- `mcp_universe`

The user request spelled the third benchmark as `MCP-Ubiverse`.
This document uses the official benchmark name `MCP-Universe`.

Scope:

- `crates/rwkv-eval/src/datasets/**/*.rs`
- `examples/rwkv-lm-eval/src/evaluating.rs` behavior as it exists today

This document follows the constraints in
`crates/rwkv-infer/src/config_loader/BENCHMARK_SOP.md`.

## Summary

Under the current `rwkv-eval` architecture, these three benchmarks do not have the same
integration shape.

- `BrowseComp` and `BrowseComp-ZH` can be integrated as native `rwkv-eval` benchmarks.
- `MCP-Universe` cannot be integrated as a faithful native benchmark without changing the current
  benchmark interface, task identity model, and result schema.
- Therefore `MCP-Universe` must be integrated as a delegated adapter benchmark that shells out to
  the official MCP-Universe runtime and maps its task-level success signal back into the existing
  `Record { is_passed, fail_reason }` shape.

This is not a wording preference.
It is a direct consequence of the current evaluation stack.

## Hard Constraints

The following are fixed constraints for this design:

- Do not change `BenchmarkInfo`.
- Do not change `Benchmark`.
- Do not change `Record`.
- Do not change the runner's `pass@k` aggregation logic.
- Do not change the PostgreSQL schema.
- Keep benchmark-specific logic in benchmark files or family-local `*_common.rs`.
- Do not introduce a new global benchmark abstraction layer.

Relevant current constraints:

- `Benchmark::answer_and_judge()` returns only `Record`, which carries a boolean pass/fail result.
- The runner requires non-empty `avg_ks` and `pass_ks`.
- The runner computes metrics only from `bool` attempt outcomes.
- Failed attempts always go through the generic checker when DB persistence is enabled.
- `task` identity does not encode arbitrary benchmark-local runtime configuration.

As a result:

- all three benchmarks must be expressed as `pass@1` benchmarks;
- any benchmark-local scalar score must be collapsed into `is_passed`;
- hidden test-set plaintext must never be written into `context` or `ref_answer`;
- external runtime configuration that affects reproducibility must be pinned in benchmark code, not
  left as a free-form runtime switch.

## Common Registration Rules

For all three benchmarks, the registration defaults should be:

- `n_shots = &[0]`
- `avg_ks = &[1.0]`
- `pass_ks = &[1]`

This is not because agent benchmarks are naturally `pass@1` only.
It is because the current runner only understands pass/fail attempts and `pass@k`.

For initial rollout, benchmark selection should use:

```toml
benchmark_field = []
extra_benchmark_name = ["browsecomp", "browsecomp_zh", "mcp_universe"]
```

Do not rely on `benchmark_field` for the initial rollout.
These benchmarks have stronger setup assumptions than the existing field-wide suites.

## Field Placement

### BrowseComp

- internal name: `browsecomp`
- display name: `BrowseComp`
- field: `Field::InstructionFollowing`
- file: `crates/rwkv-eval/src/datasets/instruction_following/browsecomp.rs`

Reasoning:

- it is not multiple-choice;
- it is not math-specific free answer;
- it does not expose an explicit tool-call protocol through the current `rwkv-eval` interface;
- the benchmark-specific prompt and grader should stay local to the benchmark file.

### BrowseComp-ZH

- internal name: `browsecomp_zh`
- display name: `BrowseComp-ZH`
- field: `Field::InstructionFollowing`
- file: `crates/rwkv-eval/src/datasets/instruction_following/browsecomp_zh.rs`

Reasoning is the same as `BrowseComp`.

### MCP-Universe

- internal name: `mcp_universe`
- display name: `MCP-Universe`
- field: `Field::FunctionCalling`
- file: `crates/rwkv-eval/src/datasets/function_calling/mcp_universe.rs`

Reasoning:

- the benchmark measures real tool-using agent behavior;
- the closest existing field is `FunctionCalling`;
- however, it is not in the same family as `tau_bench`, so its runtime and evaluation logic must
  remain benchmark-local instead of being pushed into `function_calling/mod.rs`.

## Shared File Layout

Recommended layout:

```text
crates/rwkv-eval/src/datasets/instruction_following/
  mod.rs
  instruction_following.rs
  browsecomp_common.rs
  browsecomp.rs
  browsecomp_zh.rs

crates/rwkv-eval/src/datasets/function_calling/
  mod.rs
  mcp_universe.rs
```

Use `browsecomp_common.rs` only for family-shared prompt, grading, redaction, and parsing helpers.
Do not put decryption logic there unless both benchmarks truly share the same crypto and file
layout.

## Sampling Policy

These three benchmarks should not reuse the current math/free-answer sampling presets.

Recommended benchmark-local sampling config for `BrowseComp` and `BrowseComp-ZH`:

```rust
SamplingConfig {
    temperature: 0.0,
    top_k: 1,
    top_p: 1.0,
    presence_penalty: 0.0,
    repetition_penalty: 0.0,
    penalty_decay: 1.0,
}
```

Reasoning:

- the benchmark is scored on one short final answer;
- evaluation should be deterministic;
- `avg_k = 1.0` means there is no benefit to stochastic exploration in the current runner.

For `MCP-Universe`, the `SamplingConfig` stored in `BenchmarkInfo` is metadata only.
The actual agent runtime is controlled by the delegated MCP-Universe benchmark config.
Therefore its `SamplingConfig` should also be fixed and deterministic, and not treated as the real
agent-generation control surface.

## BrowseComp Family

### Official Sources

- BrowseComp announcement by OpenAI, published April 10, 2025:
  `https://openai.com/index/browsecomp/`
- reference implementation:
  `https://github.com/openai/simple-evals`
- public encrypted CSV:
  `https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv`

OpenAI states that BrowseComp contains `1,266` encrypted tasks and explicitly asks users not to
reveal the benchmark in plain text online.

### BrowseComp Data Design

Local cache:

- `dataset_root/browsecomp/browse_comp_test_set.csv`

Item shape:

```rust
pub struct BrowseCompItem {
    id: usize,
    question: String,
    answer: String,
}
```

Implementation rules:

- keep the downloaded file encrypted on disk;
- decrypt only in memory during `load()`;
- never write decrypted question text into `context`;
- never write decrypted reference answer into `ref_answer`;
- never print decrypted plaintext to stdout/stderr.

This differs from most existing benchmarks and is required by the benchmark's anti-leakage design.

### BrowseComp Loader Rules

`load()`:

- read the encrypted CSV directly;
- reproduce the official `SHA256(password) + XOR + base64` decryption logic from
  `browsecomp_eval.py`;
- populate in-memory plaintext `question` and `answer`;
- return invalid if any row fails to decrypt.

`check()`:

- verify the file parses cleanly;
- verify the decrypted sample count is exactly `1,266`;
- verify no item has an empty question or answer.

`download()`:

- use `download_url_files()`;
- do not transform the CSV into another format.

### BrowseComp Prompt Contract

`BrowseComp` should use `CoTMode::NoCoT`.

Do not use the math-style `<think>` path.
Instead use a visible, structured answer format close to the official evaluation contract, for
example:

```text
You are a web research agent.
Find the answer to the question below.
Return exactly three sections:

Explanation: <brief justification>
Exact Answer: <short final answer>
Confidence: <0.00-1.00>

Question:
{question}
```

The benchmark should judge only the final answer correctness.
The explanation is helpful for grading and debugging but is not the score target.

### BrowseComp Grading

`BrowseComp` should set `with_llm_judger = true`.

Use a benchmark-local grader helper in `browsecomp_common.rs`.
Do not reuse the math judge prompt.

Grader contract:

- input:
  - original question
  - hidden gold answer
  - model raw response
- output:
  - strict JSON schema
  - at minimum:

```json
{
  "is_passed": true,
  "reason": "..."
}
```

Retry policy:

- retry malformed or transport-failed grader responses up to 3 times;
- after 3 failures, panic;
- do not silently convert grader failure into `false`.

### BrowseComp Persistence Policy

Because the test set is encrypted, persistence must be redacted.

`Record` mapping:

- `context`
  - store only a redacted summary such as benchmark id and prompt hash
  - example:
    `browsecomp:id=42 prompt_sha256=<...>`
- `answer`
  - store the model raw response
- `ref_answer`
  - store only a redacted digest
  - example:
    `browsecomp_answer_sha256=<...>`
- `fail_reason`
  - store the grader's short reason

Do not persist plaintext question or answer in:

- `context`
- `ref_answer`
- logs intended for long-term retention

Operational recommendation:

- set `upload_to_space = false` when running `browsecomp`.

### BrowseComp Checker Limitation

When DB persistence is enabled, failed attempts will still go through the generic checker.
That checker was designed for ordinary prompt/answer benchmarks and is not authoritative for
`BrowseComp`.

For `BrowseComp`, checker rows should be treated as low-value diagnostics only.
They are not benchmark score signals.

## BrowseComp-ZH

### Official Sources

- dataset card:
  `https://huggingface.co/datasets/PALIN2018/BrowseComp-ZH`
- paper:
  `https://arxiv.org/abs/2504.19314`
- project repository referenced from the paper summary:
  `https://github.com/PALIN2018/BrowseComp-ZH`

As of May 9, 2025, the public dataset card states:

- test split only;
- `289` encrypted tasks;
- `11` domains;
- all samples are encrypted to preserve evaluation value.

### BrowseComp-ZH Data Design

Local cache:

- `dataset_root/browsecomp_zh/browsecomp-zh-encrypted.xlsx`

Item shape:

```rust
pub struct BrowseCompZhItem {
    id: usize,
    topic: String,
    question: String,
    answer: String,
}
```

Implementation rules:

- download the original encrypted Excel file directly;
- decrypt in memory only;
- keep plaintext out of `context` and `ref_answer`;
- keep the implementation pure Rust;
- a small benchmark-local XLSX reader based on ZIP + worksheet/sharedStrings parsing is acceptable;
- do not shell out to Python for dataset loading.

### BrowseComp-ZH Loader Rules

`load()`:

- read the encrypted workbook directly;
- parse `Topic`, `Question`, `Answer`, `canary` from the original schema;
- decrypt question and answer in memory;
- populate topic if available.

`check()`:

- verify decrypted sample count is exactly `289`;
- verify no decrypted question or answer is empty.

`download()`:

- use `download_hf_parquet_splits()`;
- local directory name must be `browsecomp_zh`, not the HF repo name.

### BrowseComp-ZH Prompt Contract

`BrowseComp-ZH` should also use `CoTMode::NoCoT`.

The prompt should be Chinese-first and preserve the same explicit answer contract:

```text
你是一个网页研究代理。
请回答下面的问题，并严格输出三个部分：

解释：<简短说明>
最终答案：<短答案>
置信度：<0.00-1.00>

问题：
{question}
```

### BrowseComp-ZH Grading

`BrowseComp-ZH` should set `with_llm_judger = true`.

Use the same family-local grader shape as `BrowseComp`, but the grader prompt should be Chinese.

Grader JSON schema:

```json
{
  "is_passed": true,
  "reason": "..."
}
```

The benchmark score remains binary `pass@1` even though the benchmark paper also reports
calibration-style analysis.
That additional metric is out of scope under the current interface.

### BrowseComp-ZH Persistence Policy

Because the dataset is encrypted, use the same redaction policy as `BrowseComp`.

`Record` mapping:

- `context`
  - redacted summary only
- `answer`
  - model raw response
- `ref_answer`
  - redacted digest only
- `fail_reason`
  - short grader reason

Do not persist plaintext question or gold answer.

## MCP-Universe

### Official Sources

- official repository:
  `https://github.com/SalesforceAIResearch/MCP-Universe`
- paper:
  `https://arxiv.org/abs/2508.14704`

The official README describes MCP-Universe as a benchmark suite over real MCP servers and lists
the following benchmark domains:

- web search
- location navigation
- browser automation
- financial analysis
- repository management
- 3D design

It also requires external runtime dependencies such as:

- Python 3.10+
- Docker
- multiple service API keys
- domain-specific environment variables
- real external service access

### Why MCP-Universe Is Different

`MCP-Universe` is not a normal prompt-to-answer benchmark.

It evaluates:

- long-horizon multi-step tool use;
- interaction with real MCP servers;
- environment side effects;
- dynamic evaluator execution;
- task traces and artifacts outside the current `Record` schema.

Under the current `rwkv-eval` interface, a faithful native port is not possible because:

1. `Benchmark::answer_and_judge()` receives only `model_name` and an OpenAI-compatible client, not
   the full runtime configuration that the official MCP-Universe runner needs.
2. The current result schema cannot store full task traces, tool artifacts, or official report
   objects.
3. The current task identity model does not encode arbitrary benchmark-local runtime configuration.

Therefore `MCP-Universe` must be integrated as a delegated adapter benchmark.

### MCP-Universe Benchmark Scope in rwkv-rs

In `rwkv-rs`, `mcp_universe` should not mean "arbitrary upstream MCP-Universe config supplied at
runtime".

That would break reproducibility and task identity.

Instead, `mcp_universe` must mean one pinned profile owned by `rwkv-rs`.

Recommended v1 profile:

- include:
  - web search
  - location navigation
  - financial analysis
- exclude from v1:
  - repository management
  - browser automation
  - 3D design

Reasoning:

- the excluded domains are more operationally fragile;
- they can perform destructive or heavyweight external actions;
- they are not a good fit for a shared evaluation host without stronger sandboxing and artifact
  handling.

If broader coverage is needed later, add separate benchmark names instead of runtime switches:

- `mcp_universe_repo`
- `mcp_universe_browser`
- `mcp_universe_3d`

Do not overload one benchmark name with multiple free-form profiles.

### MCP-Universe Integration Shape

File:

- `crates/rwkv-eval/src/datasets/function_calling/mcp_universe.rs`

Field:

- `Field::FunctionCalling`

The benchmark should work as a thin adapter over the official MCP-Universe runtime:

1. validate local MCP-Universe installation and environment;
2. build a task list from the pinned `rwkv-rs` profile;
3. for each task, generate a temporary benchmark config owned by `rwkv-rs`;
4. shell out to the official MCP-Universe runner;
5. read the resulting task-level success signal;
6. map it back into `Record`.

### MCP-Universe Runtime Configuration

Because the current `Benchmark` trait does not expose all model endpoint details required by the
external runtime, the adapter should recover the selected model config from
`rwkv_config::validated::eval::EVAL_CFG` by `model_name`, then inject:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

into the delegated MCP-Universe child process.

This keeps the public benchmark interface unchanged while still allowing the official Python runtime
to use the active eval target model.

### MCP-Universe Loader and Check Rules

`load()`:

- inspect the pinned MCP-Universe profile owned by `rwkv-rs`;
- collect task config paths from the official repo checkout;
- keep only tasks in the selected read-only domains;
- return invalid if no usable tasks are found.

`check()`:

- validate the local MCP-Universe checkout exists;
- validate the expected benchmark config files exist;
- validate the curated task list is non-empty.

For the pinned `rwkv-rs` compatibility profile, a fixed task count is acceptable and preferred.
Current pinned count:

- `55` web search tasks
- `45` location navigation tasks
- `40` financial analysis tasks
- total `140`

`download()`:

- clone or fast-forward the official MCP-Universe repo into
  `dataset_root/mcp_universe/runtime`;
- do not auto-install Python packages or Docker;
- leave official runtime dependency installation to the user.

### MCP-Universe Prompt and Result Mapping

The official runner owns the real agent loop.
Therefore `get_expected_context()` should return only a compact task synopsis, not a synthetic full
prompt that pretends to capture the whole benchmark.

Recommended shape:

```text
MCP-Universe task
domain: {domain}
task: {task_path}
summary: {question_or_short_description}
```

`Record` mapping:

- `context`
  - task id, domain, temporary config path, report path
- `answer`
  - final assistant answer or serialized result summary from the official runner
- `ref_answer`
  - evaluator summary or output-format summary, not a fabricated gold answer
- `is_passed`
  - official task success bool from MCP-Universe
- `fail_reason`
  - evaluator failure summary or runner error summary

Do not try to squeeze the full MCP trace into `context`.
Full trace data should live in MCP-Universe's own report/log artifacts.

### MCP-Universe Checker Limitation

The generic checker is not authoritative for tool-using MCP tasks.
If DB persistence is enabled, checker rows on failed tasks are diagnostic noise only.

They must not be interpreted as benchmark semantics.

### MCP-Universe Task Identity Limitation

This is the most important remaining architectural limitation.

The current task identity does not encode benchmark-local external runtime configuration.
That means the following are not fully visible to the existing resume/rerun identity model:

- MCP-Universe repo revision
- delegated benchmark YAML contents
- external runner implementation changes
- external endpoint env changes

Because we are not changing interfaces, the mitigation must be operational and benchmark-local:

- pin one `rwkv-rs` owned MCP-Universe profile in code;
- avoid free-form runtime profile switches;
- prefer `run_mode = rerun` for MCP-Universe;
- use a dedicated eval config file for MCP-Universe runs.

## Validation Plan

### BrowseComp

- unit test CSV parsing and decryption
- unit test that plaintext is not written into `context` or `ref_answer`
- unit test prompt rendering
- unit test grader JSON parsing
- smoke test `answer_and_judge()` with mocked model and judge responses

### BrowseComp-ZH

- unit test parquet parsing and decryption
- unit test that plaintext is not written into `context` or `ref_answer`
- unit test Chinese prompt rendering
- unit test Chinese grader JSON parsing
- smoke test `answer_and_judge()` with mocked model and judge responses

### MCP-Universe

- unit test profile task discovery
- unit test environment validation errors
- unit test delegated result JSON to `Record` mapping
- smoke test on one read-only task with the official runtime installed

## Recommended Initial Rollout

Phase 1:

- implement `browsecomp`
- implement `browsecomp_zh`
- document `mcp_universe` as experimental delegated benchmark

Phase 2:

- implement `mcp_universe` read-only profile adapter

Do not start with MCP-Universe first.
Its runtime surface is much larger and its integration risk is higher.

## References

- BrowseComp announcement, OpenAI, April 10, 2025:
  `https://openai.com/index/browsecomp/`
- BrowseComp reference implementation:
  `https://github.com/openai/simple-evals`
- BrowseComp public encrypted CSV:
  `https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv`
- BrowseComp-ZH dataset card, updated May 9, 2025:
  `https://huggingface.co/datasets/PALIN2018/BrowseComp-ZH`
- BrowseComp-ZH paper, published April 27, 2025:
  `https://arxiv.org/abs/2504.19314`
- MCP-Universe repository:
  `https://github.com/SalesforceAIResearch/MCP-Universe`
- MCP-Universe paper:
  `https://arxiv.org/abs/2508.14704`
