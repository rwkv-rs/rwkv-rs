# Token 0 Early Stop Bug Analysis

## Summary

This document explains the `rwkv-infer` bug where generation could appear to emit `0`
and then stop or stall immediately.

The root cause was not in the HTTP layer.
It was in the inference core's special-token semantics:

- `0` was hardcoded as the EOS token.
- `0` was also hardcoded as the prefill padding token.
- The tokenizer output loop also dropped token `0` unconditionally.

This made `rwkv-infer` treat token id `0` as a special value even when the model
or tokenizer did not define it as EOS.

## User-visible symptom

The observed behavior could show up in two closely related ways:

- Internal logs or debugging output showed the first sampled token was `0`.
- The request then terminated immediately or appeared to "pause" after that.

Depending on the caller and logging path, the user might also see a literal `0`
around the failure because downstream code or instrumentation exposed the sampled token id
instead of cleanly reporting the stop reason.

## Why it happened

Before this fix, `rwkv-infer` had three coupled assumptions:

1. `END_TOKEN_ID` was hardcoded to `0`.
2. `PREFILL_PAD_TOKEN_ID` was also hardcoded to `0`.
3. `TokenizerLoop` silently discarded `token_id == 0`.

That created two failure paths.

### Failure path 1: false EOS

If the executor sampled token id `0`, `InferenceExecutionLoop` immediately interpreted it as EOS.
That request then finished with `FinishReason::Stop` without treating the token as normal output.

Effect:

- generation stopped too early
- `generated_tokens` stayed at `0`
- the output loop never got a normal token event for that token

### Failure path 2: pad/EOS semantic coupling

Prefill padding also used token id `0`.
If any backend implementation or mask handling was imperfect, the padding token could leak into
the model state and increase the chance of sampling `0` early.

Even if mask handling was correct, using the same id for both padding and EOS made the code
much harder to reason about and debug.

## What was changed

### 1. Hardcoded EOS was removed from the inference core

File:

- `crates/rwkv-infer/src/inference_core/special_token.rs`

Changes:

- Removed the `END_TOKEN_ID = 0` assumption.
- Introduced `SpecialTokenConfig`:
  - `eos_token_ids: Vec<i32>`
  - `prefill_pad_token_id: i32`
- Kept `prefill_pad_token_id` defaulting to `0` for compatibility.
- Defaulted `eos_token_ids` to an empty list.

Meaning:

- token id `0` is no longer special unless explicitly configured as EOS

### 2. Execution stop logic now uses configured EOS ids

File:

- `crates/rwkv-infer/src/inference_core/execution_loop.rs`

Changes:

- Replaced `sampled_token.token_id == 0` with `special_tokens.is_eos(sampled_token.token_id)`.
- Replaced hardcoded prefill padding `0` with `special_tokens.prefill_pad_token_id`.
- Replaced the decode fallback token from EOS to the configured pad token.
- Passed configured EOS ids into the constraint tokenizer metadata instead of always using `0`.

Meaning:

- stop semantics now come from configuration, not from a hidden runtime constant

### 3. TokenizerLoop no longer discards token 0 blindly

File:

- `crates/rwkv-infer/src/inference_core/tokenizer_loop.rs`

Changes:

- Removed the branch that skipped `token_id == 0`.

Meaning:

- output handling no longer makes its own EOS assumption
- EOS handling is now owned by the execution state machine

### 4. Infer config can now carry EOS and pad-token settings

Files:

- `crates/rwkv-config/src/raw/infer.rs`
- `examples/rwkv-lm/src/inferring.rs`

Changes:

- `GenerationConfig` now supports:
  - `eos_token_ids: Option<Vec<i32>>`
  - `prefill_pad_token_id: Option<i32>`
- `InferenceExecutionConfig` is built with `SpecialTokenConfig` from infer config.

Meaning:

- the runtime can be configured per model instead of assuming a universal EOS id

## Why this fixes the bug

After the change:

- token id `0` is treated as a normal token unless the model config explicitly declares it as EOS
- prefill padding and EOS semantics are no longer forced to share the same id
- the output path no longer suppresses token `0` by itself

This directly breaks the incorrect chain:

`sampled token 0 -> interpreted as EOS -> token suppressed -> generation stops`

## Unit tests added

File:

- `crates/rwkv-infer/src/inference_core/execution_loop.rs`

Added tests:

- `token_zero_is_not_eos_when_not_configured`
- `token_zero_is_eos_only_when_configured`

These tests validate the key semantic requirement:

- when EOS is empty, token `0` must behave like a normal generated token
- when EOS is `[0]`, token `0` must stop generation

Also added small config-level tests in:

- `crates/rwkv-infer/src/inference_core/special_token.rs`

## Validation status

### Completed

- `cargo fmt --all`
- `cargo test -p rwkv-config -- --nocapture`
- `cargo test -p rwkv-infer token_zero_is_ -- --nocapture`
- `cargo test -p rwkv-infer -- --nocapture`

All targeted and crate-level unit tests passed after fixing the token semantics.

### Environment issue encountered during validation

The first `rwkv-infer` test runs failed before Rust compilation finished because the local
environment was missing build tools required by `xgrammar-rs`:

- `cmake`
- `libclang.so`

These were resolved locally without touching system packages:

- downloaded a prebuilt `cmake` binary into `/tmp`
- downloaded and extracted Ubuntu `libclang` packages into `/tmp`
- reran tests with `CMAKE`, `LIBCLANG_PATH`, and `LD_LIBRARY_PATH` pointed at those local paths

That issue was an environment/build dependency problem, not a bug in the fix itself.

## Performance impact

This fix is primarily a correctness fix, not a throughput optimization.
However, it does resolve two hidden performance costs:

- false early-stop requests no longer terminate immediately and require retries
- the runtime no longer wastes work by suppressing valid token-0 outputs under a bad EOS assumption

In practice, this improves effective request completion quality and avoids wasted TTFT/throughput
from requests that aborted for the wrong reason.

The validation work also removed a test-environment bottleneck during development by setting up
local `cmake` and `libclang` artifacts, which made the `rwkv-infer` test suite runnable again
in this environment.

## Remaining risk

If the model still frequently samples token `0` first after this fix, then the next place to
inspect is not the HTTP layer but the backend execution path:

- prefill masking correctness
- model/tokenizer/vocab alignment
- executor sample output alignment
- constrained sampling interactions

This patch fixes the inference-core semantic bug.
It does not by itself prove the backend never over-samples token `0`.
