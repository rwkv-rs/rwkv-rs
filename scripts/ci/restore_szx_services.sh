#!/usr/bin/env bash
set -euo pipefail

infer_was_active="${INFER_WAS_ACTIVE:-false}"
eval_was_active="${EVAL_WAS_ACTIVE:-false}"

restored=()

if [ "$eval_was_active" = "true" ]; then
  systemctl start rwkv-eval.service
  restored+=("rwkv-eval.service")
fi

if [ "$infer_was_active" = "true" ]; then
  systemctl start rwkv-infer.service
  restored+=("rwkv-infer.service")
fi

if [ "${#restored[@]}" -eq 0 ]; then
  echo "No rwkv services were active before validation; nothing to restore."
else
  printf 'Restored services after failed validation: %s\n' "${restored[*]}"
fi
