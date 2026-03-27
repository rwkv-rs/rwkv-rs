#!/usr/bin/env bash
set -euo pipefail

remote_host="${REMOTE_HOST:-rwkv@192.168.0.157}"
local_judger_port="${LOCAL_JUDGER_PORT:-18091}"
local_checker_port="${LOCAL_CHECKER_PORT:-18090}"
remote_judger_port="${REMOTE_JUDGER_PORT:-18091}"
remote_checker_port="${REMOTE_CHECKER_PORT:-18090}"

exec ssh -NT \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${local_judger_port}:127.0.0.1:${remote_judger_port}" \
  -L "127.0.0.1:${local_checker_port}:127.0.0.1:${remote_checker_port}" \
  "${remote_host}"
