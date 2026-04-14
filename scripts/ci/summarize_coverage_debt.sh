#!/usr/bin/env bash
set -euo pipefail

coverage_target="${COVERAGE_TARGET:-}"
if [ -z "$coverage_target" ]; then
  echo "COVERAGE_TARGET is required" >&2
  exit 1
fi

summary_path="${SUMMARY_PATH:-target/coverage/summary.txt}"
coverage_run_outcome="${COVERAGE_RUN_OUTCOME:-}"
coverage_reports_outcome="${COVERAGE_REPORTS_OUTCOME:-}"
mode="${MODE:-ci}"

step_summary_path="${GITHUB_STEP_SUMMARY:-}"
if [ -z "$step_summary_path" ]; then
  echo "GITHUB_STEP_SUMMARY is not set" >&2
  exit 1
fi

coverage_status=tests_failed
coverage_pct=

{
  echo "### Coverage Debt"
  echo
} >> "$step_summary_path"

write_outputs() {
  if [ -z "${GITHUB_OUTPUT:-}" ]; then
    return 0
  fi

  {
    echo "coverage_status=$coverage_status"
    echo "coverage_pct=$coverage_pct"
  } >> "$GITHUB_OUTPUT"
}

if [ "$coverage_run_outcome" != "success" ]; then
  echo "- Status: unavailable" >> "$step_summary_path"
  echo "- Workspace tests failed before coverage debt could be evaluated." >> "$step_summary_path"
  write_outputs
  exit 0
fi

if [ "$coverage_reports_outcome" != "success" ] || [ ! -f "$summary_path" ]; then
  coverage_status=unavailable
  echo "- Status: unavailable" >> "$step_summary_path"
  echo "- Coverage reports could not be generated after tests passed." >> "$step_summary_path"
  if [ "$mode" = "deploy" ]; then
    echo "- Deployment remains allowed because workspace tests already passed." >> "$step_summary_path"
  else
    echo "- This is non-blocking debt and should be fixed." >> "$step_summary_path"
  fi
  write_outputs
  exit 0
fi

coverage_pct=$(
  awk '/^TOTAL/ {for (i=NF; i>=1; i--) if ($i ~ /%$/) {gsub(/%/, "", $i); print $i; exit}}' "$summary_path"
)

if [ -z "$coverage_pct" ]; then
  coverage_status=unavailable
  echo "- Status: unavailable" >> "$step_summary_path"
  echo "- Coverage summary was generated but the percentage could not be parsed." >> "$step_summary_path"
  if [ "$mode" = "deploy" ]; then
    echo "- Deployment remains allowed because workspace tests already passed." >> "$step_summary_path"
  else
    echo "- This is non-blocking debt and should be fixed." >> "$step_summary_path"
  fi
  write_outputs
  exit 0
fi

below_target=$(awk -v cov="$coverage_pct" -v target="$coverage_target" 'BEGIN {print (cov + 0 < target + 0) ? "true" : "false"}')

echo "- Workspace line coverage: ${coverage_pct}%" >> "$step_summary_path"
echo "- Target: ${coverage_target}%" >> "$step_summary_path"

if [ "$below_target" = "true" ]; then
  coverage_status=below_target
  echo "- Status: non-blocking warning" >> "$step_summary_path"
  echo "- Tests passed, but coverage is below target and should be treated as debt." >> "$step_summary_path"
else
  coverage_status=target_met
  echo "- Status: target met" >> "$step_summary_path"
fi

write_outputs
