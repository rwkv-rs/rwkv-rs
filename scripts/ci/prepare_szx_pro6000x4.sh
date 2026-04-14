#!/usr/bin/env bash
set -euo pipefail

service_is_active() {
  systemctl is-active --quiet "$1"
}

pid_belongs_to_service() {
  local pid="$1"
  local service="$2"
  [ -r "/proc/$pid/cgroup" ] || return 1
  grep -Fq "/$service" "/proc/$pid/cgroup"
}

gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' \
    | awk 'NF' \
    | sort -u
}

describe_pid() {
  local pid="$1"
  local owner="foreign"
  local command="unknown"

  if pid_belongs_to_service "$pid" "rwkv-eval.service"; then
    owner="rwkv-eval.service"
  elif pid_belongs_to_service "$pid" "rwkv-infer.service"; then
    owner="rwkv-infer.service"
  fi

  if ps -p "$pid" -o comm= >/dev/null 2>&1; then
    command=$(ps -p "$pid" -o comm= | xargs)
  fi

  printf 'pid=%s owner=%s command=%s\n' "$pid" "$owner" "$command"
}

classify_gpu_pids() {
  ours=()
  foreign=()

  while IFS= read -r pid; do
    [ -n "$pid" ] || continue
    if pid_belongs_to_service "$pid" "rwkv-eval.service" || pid_belongs_to_service "$pid" "rwkv-infer.service"; then
      ours+=("$pid")
    else
      foreign+=("$pid")
    fi
  done < <(gpu_pids)
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required on szx-pro6000x4" >&2
  exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "nvidia-smi is installed but GPU access is unavailable" >&2
  exit 1
fi

infer_was_active=false
eval_was_active=false

if service_is_active "rwkv-infer.service"; then
  infer_was_active=true
fi
if service_is_active "rwkv-eval.service"; then
  eval_was_active=true
fi

while true; do
  classify_gpu_pids
  if [ "${#foreign[@]}" -eq 0 ]; then
    break
  fi

  echo "Foreign GPU occupancy detected on szx-pro6000x4; waiting for exclusive access."
  for pid in "${foreign[@]}"; do
    describe_pid "$pid"
  done
  sleep 30
done

services_stopped=false
if [ "$infer_was_active" = "true" ] || [ "$eval_was_active" = "true" ] || [ "${#ours[@]}" -gt 0 ]; then
  echo "Stopping rwkv services before validation."
  systemctl stop rwkv-eval.service
  systemctl stop rwkv-infer.service
  services_stopped=true

  while true; do
    mapfile -t remaining_gpu_pids < <(gpu_pids)
    if [ "${#remaining_gpu_pids[@]}" -eq 0 ]; then
      break
    fi

    echo "Waiting for GPU drain after stopping rwkv services."
    for pid in "${remaining_gpu_pids[@]}"; do
      describe_pid "$pid"
    done
    sleep 10
  done
else
  echo "No rwkv services were active and no rwkv-owned GPU occupancy was detected."
fi

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "infer_was_active=$infer_was_active"
    echo "eval_was_active=$eval_was_active"
    echo "services_stopped=$services_stopped"
  } >> "$GITHUB_OUTPUT"
else
  echo "infer_was_active=$infer_was_active"
  echo "eval_was_active=$eval_was_active"
  echo "services_stopped=$services_stopped"
fi
