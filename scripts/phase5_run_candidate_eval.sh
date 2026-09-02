#!/usr/bin/env bash
set -euo pipefail

# Run one Phase 5 candidate through fixed-budget multi-seed training/eval and
# write a public-only accept/reject decision.  This wrapper deliberately stops at
# a single candidate decision; it does not start a Phase 6 multi-round loop.

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-$DATA_ROOT/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${1:-}}"
RUN_NAME="${RUN_NAME:-formal_5seed_eval}"

if [ -z "$CANDIDATE_DIR" ]; then
  echo "CANDIDATE_DIR is required" >&2
  exit 2
fi

CANDIDATE_PHASE1="${CANDIDATE_PHASE1:-$CANDIDATE_DIR/candidate_phase1_push_recovery.yaml}"
CANDIDATE_PHASE2="${CANDIDATE_PHASE2:-$CANDIDATE_DIR/candidate_phase2_baseline.yaml}"
RUN_DIR="${RUN_DIR:-$CANDIDATE_DIR/$RUN_NAME}"
DECISION_FILE="${DECISION_FILE:-$CANDIDATE_DIR/accept_reject.json}"

for required_file in "$BASELINE_SUMMARY" "$CANDIDATE_PHASE1" "$CANDIDATE_PHASE2"; do
  if [ ! -f "$required_file" ]; then
    echo "Missing required file: $required_file" >&2
    exit 2
  fi
done

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON:-python3}"

PHASE1_CONFIG="$CANDIDATE_PHASE1" \
PHASE2_CONFIG="$CANDIDATE_PHASE2" \
RUN_DIR="$RUN_DIR" \
bash scripts/phase2_baseline_variance.sh

"$PYTHON_BIN" scripts/phase5_accept_reject.py \
  --baseline-summary "$BASELINE_SUMMARY" \
  --candidate-summary "$RUN_DIR/summary.json" \
  --proposal-summary "$CANDIDATE_DIR/run_summary.json" \
  --analysis-file "$CANDIDATE_DIR/analysis.json" \
  --output-file "$DECISION_FILE"

printf 'PHASE5_FORMAL_RUN_DIR=%s\n' "$RUN_DIR"
printf 'PHASE5_DECISION_FILE=%s\n' "$DECISION_FILE"
