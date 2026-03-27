#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_DIR="${ROOT_DIR}/examples/rwkv-lm-eval/config"
EVAL_CONFIG="all_coding_gpu3"
CHECK_ONLY=0
SKIP_BUILD=0
SKIP_MSB_PROBE=0

usage() {
  cat <<'EOF'
Usage: run_native_coding_bench.sh [options]

Options:
  --config-dir <dir>       Config directory. Default: examples/rwkv-lm-eval/config
  --eval-config <name>     Eval config stem. Default: all_coding_gpu3
  --check-only             Only validate the host environment and microsandbox setup
  --skip-build             Do not rebuild the release evaluator binary
  --skip-msb-probe         Skip the minimal `msb exe` runtime probe
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    --eval-config)
      EVAL_CONFIG="$2"
      shift 2
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-msb-probe)
      SKIP_MSB_PROBE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${ROOT_DIR}"

fail() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

echo "repo root: ${ROOT_DIR}"
echo "config dir: ${CONFIG_DIR}"
echo "eval config: ${EVAL_CONFIG}"

[[ "$(uname -s)" == "Linux" ]] || fail "coding benchmarks require Linux"
[[ -e /dev/kvm ]] || fail "/dev/kvm is missing; microsandbox coding benchmarks require KVM"
[[ -r /dev/kvm && -w /dev/kvm ]] || fail "/dev/kvm exists but is not readable/writable for the current user"

need_cmd cargo
need_cmd msb
need_cmd python3

if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "warning: WSL detected. Coding benchmarks are expected to run on native Linux with working /dev/kvm."
fi

export MSB_PATH="${MSB_PATH:-$(command -v msb)}"
echo "msb path: ${MSB_PATH}"

echo "ensuring microsandbox server is running"
msb server start --dev --detach >/dev/null 2>&1 || true
msb server status >/dev/null

if [[ "${SKIP_MSB_PROBE}" -eq 0 ]]; then
  echo "running minimal microsandbox probe"
  msb exe --scope none --cpus 1 --memory 128 -e python python:3.12 -- -c 'print(123)' >/tmp/rwkv_msb_probe.out
  if ! grep -qx '123' /tmp/rwkv_msb_probe.out; then
    cat /tmp/rwkv_msb_probe.out >&2
    fail "microsandbox probe did not print the expected output"
  fi
fi

CONFIG_PATH="${CONFIG_DIR}/${EVAL_CONFIG}.toml"
[[ -f "${CONFIG_PATH}" ]] || fail "missing config file: ${CONFIG_PATH}"

MODEL_BASE_URL="$(awk -F'"' '/^base_url = "/ {print $2; exit}' "${CONFIG_PATH}")"
if [[ -n "${MODEL_BASE_URL}" ]]; then
  echo "probing model endpoint: ${MODEL_BASE_URL}"
  python3 - "${MODEL_BASE_URL}" <<'PY'
import socket
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
host = url.hostname
port = url.port or (443 if url.scheme == "https" else 80)
with socket.create_connection((host, port), timeout=3):
    pass
PY
fi

UPLOAD_TO_SPACE="$(awk -F'= ' '/^upload_to_space = /{gsub(/ /, "", $2); print $2; exit}' "${CONFIG_PATH}")"
if [[ "${UPLOAD_TO_SPACE}" == "true" ]]; then
  DB_HOST="$(awk -F'"' '/^\[space_db\]/{flag=1; next} flag && /^host = "/{print $2; exit}' "${CONFIG_PATH}")"
  DB_PORT="$(awk -F'"' '/^\[space_db\]/{flag=1; next} flag && /^port = "/{print $2; exit}' "${CONFIG_PATH}")"
  if [[ -n "${DB_HOST}" && -n "${DB_PORT}" ]]; then
    echo "probing postgres endpoint: ${DB_HOST}:${DB_PORT}"
    python3 - "${DB_HOST}" "${DB_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.create_connection((host, port), timeout=3):
    pass
PY
  fi
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "native coding benchmark environment check passed"
  exit 0
fi

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "building release evaluator"
  cargo build -p rwkv-lm-eval --example rwkv-lm-eval-test --release
fi

echo "starting coding benchmark run"
exec target/release/examples/rwkv-lm-eval-test \
  --config-dir "${CONFIG_DIR}" \
  --eval-config "${EVAL_CONFIG}"
