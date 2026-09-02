#!/usr/bin/env bash
set -euo pipefail

# Phase 1: verify the project-local standing push-recovery overlay, protected
# failure classifier, rich JSON report, random-policy failure gate, and short
# PPO training smoke.

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-$DATA_ROOT/envs/isaaclab}"
SYSTEM_LIB_DIR="${SYSTEM_LIB_DIR:-$DATA_ROOT/vendor/system-libs/root/usr/lib/x86_64-linux-gnu}"

RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/runs/phase_1_push_recovery}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_ID}"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-64}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-220}"
TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-256}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-1}"
TRAIN_TIMEOUT_SECONDS="${TRAIN_TIMEOUT_SECONDS:-300}"

mkdir -p "$RUN_DIR"

export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/cache/conda}"
export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-$DATA_ROOT/vendor/unitree_ros}"
export LD_LIBRARY_PATH="$SYSTEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck disable=SC1091
source "$DATA_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

COMMANDS_FILE="$RUN_DIR/commands.sh"
cat > "$COMMANDS_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_ROOT"
source scripts/activate_environment.sh
export PYTHONPATH="$PROJECT_ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}"

export DEVICE="$DEVICE"
export SEED="$SEED"
export EVAL_NUM_ENVS="$EVAL_NUM_ENVS"
export EVAL_MAX_STEPS="$EVAL_MAX_STEPS"
export TRAIN_NUM_ENVS="$TRAIN_NUM_ENVS"
export TRAIN_ITERATIONS="$TRAIN_ITERATIONS"
export TRAIN_TIMEOUT_SECONDS="$TRAIN_TIMEOUT_SECONDS"

bash scripts/phase1_push_recovery_smoke.sh
EOF
chmod +x "$COMMANDS_FILE"

printf 'PHASE1_RUN_DIR=%s\n' "$RUN_DIR"
printf 'PHASE1_COMMANDS=%s\n' "$COMMANDS_FILE"

run_logged() {
  local step_name="$1"
  shift
  local log_file="$RUN_DIR/${step_name}.log"
  printf '\n[%s] %s\n' "$step_name" "$log_file"
  set +e
  (
    cd "$PROJECT_ROOT"
    "$@"
  ) > "$log_file" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$RUN_DIR/${step_name}.exit"
  return "$rc"
}

OVERALL_RC=0

run_logged zero_strong_push \
  python scripts/phase1_push_recovery_eval.py \
    --headless \
    --device "$DEVICE" \
    --seed "$SEED" \
    --profile strong_push \
    --policy zero \
    --num-envs "$EVAL_NUM_ENVS" \
    --max-steps "$EVAL_MAX_STEPS" \
    --summary-file "$RUN_DIR/zero_strong_push_summary.json" || OVERALL_RC=$?

run_logged random_strong_push \
  python scripts/phase1_push_recovery_eval.py \
    --headless \
    --device "$DEVICE" \
    --seed "$SEED" \
    --profile strong_push \
    --policy random \
    --num-envs "$EVAL_NUM_ENVS" \
    --max-steps "$EVAL_MAX_STEPS" \
    --summary-file "$RUN_DIR/random_strong_push_summary.json" || OVERALL_RC=$?

run_logged zero_no_push \
  python scripts/phase1_push_recovery_eval.py \
    --headless \
    --device "$DEVICE" \
    --seed "$SEED" \
    --profile no_push \
    --policy zero \
    --num-envs "$EVAL_NUM_ENVS" \
    --max-steps "$EVAL_MAX_STEPS" \
    --summary-file "$RUN_DIR/zero_no_push_summary.json" || OVERALL_RC=$?

run_logged train_smoke \
  timeout --signal=INT --kill-after=10s "${TRAIN_TIMEOUT_SECONDS}s" \
  python scripts/phase1_push_recovery_train_smoke.py \
    --headless \
    --device "$DEVICE" \
    --seed "$SEED" \
    --profile no_push \
    --num-envs "$TRAIN_NUM_ENVS" \
    --max-iterations "$TRAIN_ITERATIONS" \
    --run-dir "$RUN_DIR/train_smoke" \
    --summary-file "$RUN_DIR/train_smoke_summary.json" || OVERALL_RC=$?

run_logged collect_summary \
  python scripts/phase1_collect_smoke_summary.py \
    --run-dir "$RUN_DIR" \
    --config configs/phase1_push_recovery.yaml || OVERALL_RC=$?

printf '\nPhase 1 smoke complete: %s/summary.json\n' "$RUN_DIR"
exit "$OVERALL_RC"
