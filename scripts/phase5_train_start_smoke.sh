#!/usr/bin/env bash
set -euo pipefail

# Run a short CUDA train-start smoke for a Phase 5 candidate directory.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CANDIDATE_DIR="${CANDIDATE_DIR:?set CANDIDATE_DIR to a Phase 5 output directory}"
RUN_DIR="${RUN_DIR:-$CANDIDATE_DIR/train_start_smoke}"
SUMMARY_FILE="${SUMMARY_FILE:-$CANDIDATE_DIR/train_start_smoke_summary.json}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-501}"
TASK="${TASK:-Unitree-G1-29dof-Velocity}"
PROFILE="${PROFILE:-phase2_train}"
NUM_ENVS="${NUM_ENVS:-128}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"

cd "$PROJECT_ROOT"
source scripts/activate_environment.sh
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/phase1_push_recovery_train_smoke.py \
  --headless \
  --device "$DEVICE" \
  --task "$TASK" \
  --seed "$SEED" \
  --profile "$PROFILE" \
  --num-envs "$NUM_ENVS" \
  --max-iterations "$MAX_ITERATIONS" \
  --config "$CANDIDATE_DIR/candidate_phase1_push_recovery.yaml" \
  --phase2-config "$CANDIDATE_DIR/candidate_phase2_baseline.yaml" \
  --run-dir "$RUN_DIR" \
  --summary-file "$SUMMARY_FILE"
