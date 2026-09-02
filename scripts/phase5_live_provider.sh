#!/usr/bin/env bash
set -euo pipefail

# Run one Phase 5 live LLM-provider proposal from the server. This wrapper keeps
# long commands out of SSH and never stores API key values in the repository.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SUMMARY_FILE="${SUMMARY_FILE:-/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json}"
RUN_ROOT="${RUN_ROOT:-/data/yuyangliu/runs/phase_5_llm_single_iteration}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S_${PHASE5_PROVIDER:-deepseek})}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_ID}"
PHASE5_PROVIDER="${PHASE5_PROVIDER:-deepseek}"
PHASE5_SECRET_FILE="${PHASE5_SECRET_FILE:-/data/yuyangliu/secrets/deepseek.env}"

cd "$PROJECT_ROOT"

if [ -f "$PHASE5_SECRET_FILE" ]; then
  # shellcheck disable=SC1090
  source "$PHASE5_SECRET_FILE"
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python3 scripts/phase5_llm_single_iteration.py \
  --provider "$PHASE5_PROVIDER" \
  --allow-api-spend \
  --summary-file "$SUMMARY_FILE" \
  --output-dir "$OUTPUT_DIR"
