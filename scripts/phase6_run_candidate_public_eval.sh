#!/usr/bin/env bash
set -euo pipefail

# Train/evaluate a Phase 6 candidate on a fixed paired public seed subset.
# This wrapper intentionally runs public eval only.  IID-hidden/held-out eval is
# reserved for milestone/final audit and must not feed later Agent prompts.

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-$DATA_ROOT/envs/isaaclab}"
SYSTEM_LIB_DIR="${SYSTEM_LIB_DIR:-$DATA_ROOT/vendor/system-libs/root/usr/lib/x86_64-linux-gnu}"

CANDIDATE_DIR="${CANDIDATE_DIR:-${1:-}}"
if [ -z "$CANDIDATE_DIR" ]; then
  echo "CANDIDATE_DIR is required" >&2
  exit 2
fi

CANDIDATE_PHASE1="${CANDIDATE_PHASE1:-$CANDIDATE_DIR/candidate_phase1_push_recovery.yaml}"
CANDIDATE_PHASE2="${CANDIDATE_PHASE2:-$CANDIDATE_DIR/candidate_phase2_baseline.yaml}"
SEEDS="${SEEDS:-101 102 103}"
RUN_NAME="${RUN_NAME:-public_3seed_eval}"
RUN_DIR="${RUN_DIR:-$CANDIDATE_DIR/$RUN_NAME}"
PYTHON_BIN="${PYTHON:-python3}"

for required_file in "$CANDIDATE_PHASE1" "$CANDIDATE_PHASE2"; do
  if [ ! -f "$required_file" ]; then
    echo "Missing required file: $required_file" >&2
    exit 2
  fi
done

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

"$PYTHON_BIN" - "$CANDIDATE_PHASE2" "$SEEDS" > "$RUN_DIR/phase6_public_env.sh" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
train = cfg["training"]
public = cfg["evaluation"]["public"]
seeds = [int(item) for item in sys.argv[2].replace(",", " ").split()]
print("TRAIN_SEEDS=(" + " ".join(str(x) for x in seeds) + ")")
for key, value in {
    "TRAIN_TASK": train["task"],
    "TRAIN_PROFILE": train["profile"],
    "TRAIN_NUM_ENVS": train["num_envs"],
    "TRAIN_ITERATIONS": train["max_iterations"],
    "TRAIN_DEVICE": train["device"],
    "TRAIN_TIMEOUT_SECONDS": train["timeout_seconds_per_seed"],
    "PUBLIC_PROFILE": public["profile"],
    "PUBLIC_SEED": public["seed"],
    "PUBLIC_NUM_ENVS": public["num_envs"],
    "PUBLIC_MAX_STEPS": public["max_steps"],
}.items():
    print(f'{key}="{value}"')
PY
# shellcheck disable=SC1090
source "$RUN_DIR/phase6_public_env.sh"

COMMANDS_FILE="$RUN_DIR/commands.sh"
cat > "$COMMANDS_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_ROOT"
source scripts/activate_environment.sh
export PYTHONPATH="$PROJECT_ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}"

SEEDS="$SEEDS" CANDIDATE_DIR="$CANDIDATE_DIR" CANDIDATE_PHASE1="$CANDIDATE_PHASE1" CANDIDATE_PHASE2="$CANDIDATE_PHASE2" RUN_DIR="$RUN_DIR" bash scripts/phase6_run_candidate_public_eval.sh
EOF
chmod +x "$COMMANDS_FILE"

printf 'PHASE6_PUBLIC_RUN_DIR=%s\n' "$RUN_DIR"
printf 'PHASE6_PUBLIC_COMMANDS=%s\n' "$COMMANDS_FILE"

run_logged() {
  local step_name="$1"
  local step_dir="$2"
  shift 2
  mkdir -p "$step_dir"
  local log_file="$step_dir/${step_name}.log"
  local seconds_file="$step_dir/${step_name}.seconds"
  printf '\n[%s] %s\n' "$step_name" "$log_file"
  local start_s
  local end_s
  start_s="$(date +%s)"
  set +e
  (
    cd "$PROJECT_ROOT"
    "$@"
  ) > "$log_file" 2>&1
  local rc=$?
  set -e
  end_s="$(date +%s)"
  printf '%s\n' "$rc" > "$step_dir/${step_name}.exit"
  printf '%s\n' "$((end_s - start_s))" > "$seconds_file"
  return "$rc"
}

extract_checkpoint() {
  local summary_file="$1"
  "$PYTHON_BIN" - "$summary_file" <<'PY'
import sys
from pathlib import Path

project_src = Path.cwd() / "src"
if str(project_src) not in sys.path:
    sys.path.insert(0, str(project_src))

from humanoid_agent.immutable_runner.checkpoints import find_latest_checkpoint_from_summary

try:
    checkpoint = find_latest_checkpoint_from_summary(sys.argv[1])
    print(checkpoint if checkpoint is not None else "")
except Exception:
    print("")
PY
}

OVERALL_RC=0

for seed in "${TRAIN_SEEDS[@]}"; do
  seed_dir="$RUN_DIR/seed_${seed}"
  mkdir -p "$seed_dir"
  run_logged train "$seed_dir" \
    timeout --signal=INT --kill-after=10s "${TRAIN_TIMEOUT_SECONDS}s" \
    "$PYTHON_BIN" scripts/phase1_push_recovery_train_smoke.py \
      --headless \
      --device "$TRAIN_DEVICE" \
      --task "$TRAIN_TASK" \
      --seed "$seed" \
      --profile "$TRAIN_PROFILE" \
      --num-envs "$TRAIN_NUM_ENVS" \
      --max-iterations "$TRAIN_ITERATIONS" \
      --config "$CANDIDATE_PHASE1" \
      --phase2-config "$CANDIDATE_PHASE2" \
      --run-dir "$seed_dir/train" \
      --summary-file "$seed_dir/train_summary.json" || OVERALL_RC=$?

  checkpoint="$(extract_checkpoint "$seed_dir/train_summary.json")"
  if [ -n "$checkpoint" ] && [ -f "$checkpoint" ]; then
    run_logged public_eval "$seed_dir" \
      "$PYTHON_BIN" scripts/phase1_push_recovery_eval.py \
        --headless \
        --device "$TRAIN_DEVICE" \
        --task "$TRAIN_TASK" \
        --seed "$PUBLIC_SEED" \
        --profile "$PUBLIC_PROFILE" \
        --policy checkpoint \
        --checkpoint "$checkpoint" \
        --num-envs "$PUBLIC_NUM_ENVS" \
        --max-steps "$PUBLIC_MAX_STEPS" \
        --config "$CANDIDATE_PHASE1" \
        --summary-file "$seed_dir/public_eval_summary.json" || OVERALL_RC=$?
  else
    printf 'missing checkpoint for seed %s\n' "$seed" > "$seed_dir/checkpoint_missing.txt"
    OVERALL_RC=1
  fi
done

run_logged collect "$RUN_DIR" \
  "$PYTHON_BIN" scripts/phase6_collect_public_summary.py \
    --run-dir "$RUN_DIR" \
    --phase1-config "$CANDIDATE_PHASE1" \
    --phase2-config "$CANDIDATE_PHASE2" \
    --seeds "$SEEDS" \
    --output-file "$RUN_DIR/summary.json" || OVERALL_RC=$?

printf '\nPhase 6 public-only candidate eval complete: %s/summary.json\n' "$RUN_DIR"
exit "$OVERALL_RC"
