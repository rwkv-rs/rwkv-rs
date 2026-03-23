# Inference Protocol

This document describes the usable API base paths and request protocol for the current
`polymath`, `wmt24pp`, and `arena_hard_v2` benchmark configs.

## Base URL

- Write `base_url` in TOML as the service root.
- Valid examples:
  - `http://127.0.0.1:8080`
  - `https://api.ablai.top`
- Do not append endpoint paths into `base_url`.
- Do not write:
  - `/v1`
  - `/v1/completions`
  - `/v1/chat/completions`

The evaluator normalizes `base_url` to `/v1` internally in
`examples/rwkv-lm-eval/src/evaluating/client.rs`.

## Request Path

The client is built against:

- `<base_url>/v1`

Generation then uses:

- `POST /v1/completions` as the primary path
- `POST /v1/chat/completions` as a compatibility fallback

## Current Configs

The current real-run configs are:

- `examples/rwkv-lm-eval/config/polymath.toml`
- `examples/rwkv-lm-eval/config/arena_hard_v2.toml`
- `examples/rwkv-lm-eval/config/wmt24pp.toml`

These are TOML-only:

- target model comes from `[[models]]`
- judger comes from `[llm_judger]`
- checker comes from `[llm_checker]`
- database comes from `[space_db]`

## Current Model Split

- target: local RWKV infer service
- judger: external OpenAI-compatible API
- checker: external OpenAI-compatible API

`llm_judger` and `llm_checker` are intentionally configured as different models.
