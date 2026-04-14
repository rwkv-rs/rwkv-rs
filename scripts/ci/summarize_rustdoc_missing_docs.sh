#!/usr/bin/env bash
set -euo pipefail

rustdoc_outcome="${RUSTDOC_OUTCOME:-}"
if [ -z "$rustdoc_outcome" ]; then
  echo "RUSTDOC_OUTCOME is required" >&2
  exit 1
fi

step_summary_path="${GITHUB_STEP_SUMMARY:-}"
if [ -z "$step_summary_path" ]; then
  echo "GITHUB_STEP_SUMMARY is not set" >&2
  exit 1
fi

log_path="${RUSTDOC_LOG_PATH:-target/ci/rustdoc.log}"
missing_docs_log="${MISSING_DOCS_LOG_PATH:-target/ci/missing-docs.log}"
summary_path="${SUMMARY_PATH:-target/ci/missing-docs-summary.md}"

mkdir -p "$(dirname "$log_path")"

write_outputs() {
  if [ -z "${GITHUB_OUTPUT:-}" ]; then
    return 0
  fi

  {
    echo "rustdoc_status=$1"
    echo "missing_docs_count=$2"
    echo "missing_docs_file_count=$3"
  } >> "$GITHUB_OUTPUT"
}

if [ "$rustdoc_outcome" != "success" ]; then
  write_outputs failed 0 0

  {
    echo "### Rustdoc Missing Docs Debt"
    echo
    echo "- Status: unavailable"
    echo "- Rustdoc failed before missing docs could be summarized."
    echo "- Check the \`Build rustdoc\` step output for the real failure."
  } > "$summary_path"

  cat "$summary_path" >> "$step_summary_path"
  exit 0
fi

if [ ! -f "$log_path" ]; then
  write_outputs failed 0 0

  {
    echo "### Rustdoc Missing Docs Debt"
    echo
    echo "- Status: unavailable"
    echo "- Rustdoc completed without a parsable log artifact."
  } > "$summary_path"

  cat "$summary_path" >> "$step_summary_path"
  exit 0
fi

grep -F "warning: missing documentation" "$log_path" > "$missing_docs_log" || true

missing_docs_count=$(wc -l < "$missing_docs_log" | tr -d '[:space:]')
missing_docs_file_count=$(cut -d: -f1 "$missing_docs_log" | sort -u | wc -l | tr -d '[:space:]')

rustdoc_status=clean
if [ "$missing_docs_count" -gt 0 ]; then
  rustdoc_status=warnings
fi

write_outputs "$rustdoc_status" "$missing_docs_count" "$missing_docs_file_count"

{
  echo "### Rustdoc Missing Docs Debt"
  echo
  if [ "$missing_docs_count" -gt 0 ]; then
    echo "- Status: non-blocking warning"
    echo "- Warning count: $missing_docs_count"
    echo "- Files affected: $missing_docs_file_count"
    echo "- Policy: this workflow does not fail on \`missing_docs\`, but it should not pass silently."
    echo
    echo "Top files by warning count:"
    echo
    echo '```text'
    cut -d: -f1 "$missing_docs_log" | sort | uniq -c | sort -nr | head -n 10 | sed 's/^ *//'
    echo '```'
    echo
    echo "Sample warnings:"
    echo
    echo '```text'
    head -n 20 "$missing_docs_log"
    echo '```'
  else
    echo "- No \`missing_docs\` warnings were emitted in this run."
  fi
} > "$summary_path"

cat "$summary_path" >> "$step_summary_path"
