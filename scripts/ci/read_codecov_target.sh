#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-codecov.yml}"

if [ ! -f "$config_path" ]; then
  echo "codecov config not found: $config_path" >&2
  exit 1
fi

coverage_target=$(
  sed -n '/^project:/,/^patch:/p' "$config_path" \
    | grep -E '^[[:space:]]*target:' \
    | head -n 1 \
    | sed -E 's/.*target:[[:space:]]*([0-9]+)%.*/\1/'
)

if [ -z "$coverage_target" ]; then
  echo "failed to parse coverage target from $config_path" >&2
  exit 1
fi

echo "$coverage_target"
