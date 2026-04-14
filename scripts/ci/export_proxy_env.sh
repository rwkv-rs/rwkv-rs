#!/usr/bin/env bash
set -euo pipefail

http_proxy_value="${CI_HTTP_PROXY:-}"
https_proxy_value="${CI_HTTPS_PROXY:-}"
no_proxy_value="${CI_NO_PROXY:-}"

existing_http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
existing_https_proxy="${https_proxy:-${HTTPS_PROXY:-}}"
existing_no_proxy="${no_proxy:-${NO_PROXY:-}}"

write_env() {
  if [ -n "${GITHUB_ENV:-}" ]; then
    echo "$1" >> "$GITHUB_ENV"
  else
    export "$1"
  fi
}

if [ -n "$http_proxy_value" ] && [ -z "$existing_http_proxy" ]; then
  write_env "http_proxy=$http_proxy_value"
  write_env "HTTP_PROXY=$http_proxy_value"
fi

if [ -n "$https_proxy_value" ] && [ -z "$existing_https_proxy" ]; then
  write_env "https_proxy=$https_proxy_value"
  write_env "HTTPS_PROXY=$https_proxy_value"
elif [ -n "$http_proxy_value" ] && [ -z "$existing_https_proxy" ]; then
  write_env "https_proxy=$http_proxy_value"
  write_env "HTTPS_PROXY=$http_proxy_value"
fi

if [ -n "$no_proxy_value" ] && [ -z "$existing_no_proxy" ]; then
  write_env "no_proxy=$no_proxy_value"
  write_env "NO_PROXY=$no_proxy_value"
fi
