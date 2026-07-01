#!/usr/bin/env bash
set -euo pipefail

# Phase 0: run unchanged official baselines to prove the training, checkpoint
# resume, play/evaluation, and reproduction workflow before implementing the
# custom standing push-recovery task.

DATA_ROOT="${DATA_ROOT:-/data/yuyangliu}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-$DATA_ROOT/envs/isaaclab}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$DATA_ROOT/vendor/IsaacLab}"
UNITREE_ROOT="${UNITREE_ROOT:-$DATA_ROOT/vendor/unitree_rl_lab}"
SYSTEM_LIB_DIR="${SYSTEM_LIB_DIR:-$DATA_ROOT/vendor/system-libs/root/usr/lib/x86_64-linux-gnu}"

RUN_ROOT="${RUN_ROOT:-$DATA_ROOT/runs/phase_0_official_baseline}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_ID}"

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-1}"
RESUME_ITERATIONS="${RESUME_ITERATIONS:-1}"
ANT_NUM_ENVS="${ANT_NUM_ENVS:-1024}"
UNITREE_NUM_ENVS="${UNITREE_NUM_ENVS:-512}"
PLAY_NUM_ENVS="${PLAY_NUM_ENVS:-64}"
PLAY_TIMEOUT_SECONDS="${PLAY_TIMEOUT_SECONDS:-60}"

ANT_TASK="${ANT_TASK:-Isaac-Ant-v0}"
UNITREE_TASK="${UNITREE_TASK:-Unitree-G1-29dof-Velocity}"
ANT_EXPERIMENT="${ANT_EXPERIMENT:-ant}"
UNITREE_EXPERIMENT="${UNITREE_EXPERIMENT:-unitree_g1_29dof_velocity}"

mkdir -p "$RUN_DIR"

export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_ROOT/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$DATA_ROOT/cache/conda}"
export UNITREE_ROS_DIR="${UNITREE_ROS_DIR:-$DATA_ROOT/vendor/unitree_ros}"
export LD_LIBRARY_PATH="$SYSTEM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# shellcheck disable=SC1091
source "$DATA_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

COMMANDS_FILE="$RUN_DIR/commands.sh"
SUMMARY_FILE="$RUN_DIR/summary.json"

cat > "$COMMANDS_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "$PROJECT_ROOT"
source scripts/activate_environment.sh

export DEVICE="$DEVICE"
export SEED="$SEED"
export TRAIN_ITERATIONS="$TRAIN_ITERATIONS"
export RESUME_ITERATIONS="$RESUME_ITERATIONS"
export ANT_NUM_ENVS="$ANT_NUM_ENVS"
export UNITREE_NUM_ENVS="$UNITREE_NUM_ENVS"
export PLAY_NUM_ENVS="$PLAY_NUM_ENVS"
export PLAY_TIMEOUT_SECONDS="$PLAY_TIMEOUT_SECONDS"

bash scripts/phase0_official_baseline.sh
EOF
chmod +x "$COMMANDS_FILE"

printf 'PHASE0_RUN_DIR=%s\n' "$RUN_DIR"
printf 'PHASE0_COMMANDS=%s\n' "$COMMANDS_FILE"

quote_cmd() {
  printf '%q ' "$@"
}

record_step_header() {
  local step_name="$1"
  local cwd="$2"
  shift 2
  {
    printf '\n# %s\n' "$step_name"
    printf '( cd %q && ' "$cwd"
    quote_cmd "$@"
    printf ')\n'
  } >> "$COMMANDS_FILE"
}

run_step() {
  local step_name="$1"
  local cwd="$2"
  shift 2
  local log_file="$RUN_DIR/${step_name}.log"
  record_step_header "$step_name" "$cwd" "$@"
  printf '\n[%s] running: %s\n' "$step_name" "$log_file"
  (
    cd "$cwd"
    "$@"
  ) > "$log_file" 2>&1
  local rc=$?
  printf '%s\n' "$rc" > "$RUN_DIR/${step_name}.exit"
}

run_play_step() {
  local step_name="$1"
  local cwd="$2"
  shift 2
  local log_file="$RUN_DIR/${step_name}.log"
  record_step_header "$step_name" "$cwd" timeout --signal=INT --kill-after=10s "${PLAY_TIMEOUT_SECONDS}s" "$@"
  printf '\n[%s] running with %ss timeout: %s\n' "$step_name" "$PLAY_TIMEOUT_SECONDS" "$log_file"
  set +e
  (
    cd "$cwd"
    timeout --signal=INT --kill-after=10s "${PLAY_TIMEOUT_SECONDS}s" "$@"
  ) > "$log_file" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$RUN_DIR/${step_name}.exit"
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && [ "$rc" -ne 130 ]; then
    printf 'ERROR: %s exited with %s\n' "$step_name" "$rc" >&2
    exit "$rc"
  fi
}

latest_run_with_suffix() {
  local root="$1"
  local suffix="$2"
  find "$root" -mindepth 1 -maxdepth 1 -type d -name "*_${suffix}" -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    printf 'ERROR: required file is missing: %s\n' "$path" >&2
    exit 1
  fi
}

ANT_RUN_SUFFIX="phase0_ant_train"
ANT_RESUME_SUFFIX="phase0_ant_resume"
UNITREE_RUN_SUFFIX="phase0_unitree_train"
UNITREE_RESUME_SUFFIX="phase0_unitree_resume"

run_step ant_train "$ISAACLAB_ROOT" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$ANT_TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$ANT_NUM_ENVS" \
  --max_iterations "$TRAIN_ITERATIONS" \
  --seed "$SEED" \
  --run_name "$ANT_RUN_SUFFIX"

ANT_TRAIN_DIR="$(latest_run_with_suffix "$ISAACLAB_ROOT/logs/rsl_rl/$ANT_EXPERIMENT" "$ANT_RUN_SUFFIX")"
ANT_CHECKPOINT="$ANT_TRAIN_DIR/model_0.pt"
require_file "$ANT_CHECKPOINT"

run_step ant_resume "$ISAACLAB_ROOT" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "$ANT_TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$ANT_NUM_ENVS" \
  --max_iterations "$RESUME_ITERATIONS" \
  --seed "$SEED" \
  --resume \
  --load_run "$(basename "$ANT_TRAIN_DIR")" \
  --checkpoint "$(basename "$ANT_CHECKPOINT")" \
  --run_name "$ANT_RESUME_SUFFIX"

ANT_RESUME_DIR="$(latest_run_with_suffix "$ISAACLAB_ROOT/logs/rsl_rl/$ANT_EXPERIMENT" "$ANT_RESUME_SUFFIX")"
ANT_RESUME_CHECKPOINT="$ANT_RESUME_DIR/model_0.pt"
require_file "$ANT_RESUME_CHECKPOINT"

run_play_step ant_play "$ISAACLAB_ROOT" \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task "$ANT_TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$PLAY_NUM_ENVS" \
  --checkpoint "$ANT_CHECKPOINT"

require_file "$ANT_TRAIN_DIR/exported/policy.pt"
require_file "$ANT_TRAIN_DIR/exported/policy.onnx"

run_step unitree_train "$UNITREE_ROOT" \
  python scripts/rsl_rl/train.py \
  --task "$UNITREE_TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$UNITREE_NUM_ENVS" \
  --max_iterations "$TRAIN_ITERATIONS" \
  --seed "$SEED" \
  --run_name "$UNITREE_RUN_SUFFIX"

UNITREE_TRAIN_DIR="$(latest_run_with_suffix "$UNITREE_ROOT/logs/rsl_rl/$UNITREE_EXPERIMENT" "$UNITREE_RUN_SUFFIX")"
UNITREE_CHECKPOINT="$UNITREE_TRAIN_DIR/model_0.pt"
require_file "$UNITREE_CHECKPOINT"

run_step unitree_resume "$UNITREE_ROOT" \
  python scripts/rsl_rl/train.py \
  --task "$UNITREE_TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$UNITREE_NUM_ENVS" \
  --max_iterations "$RESUME_ITERATIONS" \
  --seed "$SEED" \
  --resume \
  --load_run "$(basename "$UNITREE_TRAIN_DIR")" \
  --checkpoint "$(basename "$UNITREE_CHECKPOINT")" \
  --run_name "$UNITREE_RESUME_SUFFIX"

UNITREE_RESUME_DIR="$(latest_run_with_suffix "$UNITREE_ROOT/logs/rsl_rl/$UNITREE_EXPERIMENT" "$UNITREE_RESUME_SUFFIX")"
UNITREE_RESUME_CHECKPOINT="$UNITREE_RESUME_DIR/model_0.pt"
require_file "$UNITREE_RESUME_CHECKPOINT"

run_play_step unitree_play "$UNITREE_ROOT" \
  python scripts/rsl_rl/play.py \
  --task "$UNITREE_TASK" \
  --headless \
  --device "$DEVICE" \
  --num_envs "$PLAY_NUM_ENVS" \
  --checkpoint "$UNITREE_CHECKPOINT"

require_file "$UNITREE_TRAIN_DIR/exported/policy.pt"
require_file "$UNITREE_TRAIN_DIR/exported/policy.onnx"

export PHASE0_RUN_DIR="$RUN_DIR"
export PHASE0_SUMMARY_FILE="$SUMMARY_FILE"
export ANT_TASK UNITREE_TASK DEVICE SEED TRAIN_ITERATIONS RESUME_ITERATIONS
export ANT_TRAIN_DIR ANT_RESUME_DIR ANT_CHECKPOINT ANT_RESUME_CHECKPOINT
export UNITREE_TRAIN_DIR UNITREE_RESUME_DIR UNITREE_CHECKPOINT UNITREE_RESUME_CHECKPOINT

python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

run_dir = Path(os.environ["PHASE0_RUN_DIR"])


def read_exit(name: str) -> int | None:
    path = run_dir / f"{name}.exit"
    return int(path.read_text().strip()) if path.exists() else None


def log_contains(name: str, pattern: str) -> bool:
    path = run_dir / f"{name}.log"
    return path.exists() and pattern in path.read_text(errors="replace")


def log_has_fatal(name: str) -> bool:
    path = run_dir / f"{name}.log"
    if not path.exists():
        return True
    text = path.read_text(errors="replace")
    fatal_patterns = ("Traceback (most recent call last)", "RuntimeError:", "Exception:", "ERROR:")
    return any(pattern in text for pattern in fatal_patterns)


def artifact(path: str) -> dict[str, object]:
    p = Path(path)
    return {"path": str(p), "exists": p.is_file(), "bytes": p.stat().st_size if p.is_file() else 0}


baselines = {
    "official_isaac_ant": {
        "task": os.environ["ANT_TASK"],
        "train_dir": os.environ["ANT_TRAIN_DIR"],
        "resume_dir": os.environ["ANT_RESUME_DIR"],
        "checkpoint": artifact(os.environ["ANT_CHECKPOINT"]),
        "resume_checkpoint": artifact(os.environ["ANT_RESUME_CHECKPOINT"]),
        "exported_jit": artifact(str(Path(os.environ["ANT_TRAIN_DIR"]) / "exported/policy.pt")),
        "exported_onnx": artifact(str(Path(os.environ["ANT_TRAIN_DIR"]) / "exported/policy.onnx")),
        "train_exit": read_exit("ant_train"),
        "resume_exit": read_exit("ant_resume"),
        "play_exit": read_exit("ant_play"),
        "resume_loaded_checkpoint": log_contains("ant_resume", "Loading model checkpoint"),
        "play_loaded_checkpoint": log_contains("ant_play", "Loading model checkpoint"),
        "play_has_no_fatal_error": not log_has_fatal("ant_play"),
        "learning_iteration_seen": log_contains("ant_train", "Learning iteration 0/"),
    },
    "official_unitree_g1_velocity": {
        "task": os.environ["UNITREE_TASK"],
        "train_dir": os.environ["UNITREE_TRAIN_DIR"],
        "resume_dir": os.environ["UNITREE_RESUME_DIR"],
        "checkpoint": artifact(os.environ["UNITREE_CHECKPOINT"]),
        "resume_checkpoint": artifact(os.environ["UNITREE_RESUME_CHECKPOINT"]),
        "exported_jit": artifact(str(Path(os.environ["UNITREE_TRAIN_DIR"]) / "exported/policy.pt")),
        "exported_onnx": artifact(str(Path(os.environ["UNITREE_TRAIN_DIR"]) / "exported/policy.onnx")),
        "train_exit": read_exit("unitree_train"),
        "resume_exit": read_exit("unitree_resume"),
        "play_exit": read_exit("unitree_play"),
        "resume_loaded_checkpoint": log_contains("unitree_resume", "Loading model checkpoint"),
        "play_loaded_checkpoint": log_contains("unitree_play", "Loading model checkpoint"),
        "play_has_no_fatal_error": not log_has_fatal("unitree_play"),
        "learning_iteration_seen": log_contains("unitree_train", "Learning iteration 0/"),
    },
}

for baseline in baselines.values():
    # The official play scripts may be killed by timeout before stdout buffers
    # flush the "Loading model checkpoint" line. Exported policy artifacts are
    # stronger evidence: they are written only after resolving and loading the
    # checkpoint into the runner.
    baseline["play_checkpoint_evidence"] = bool(
        baseline["play_loaded_checkpoint"]
        or (baseline["exported_jit"]["exists"] and baseline["exported_onnx"]["exists"])
    )
    baseline["pass"] = all(
        [
            baseline["checkpoint"]["exists"],
            baseline["resume_checkpoint"]["exists"],
            baseline["exported_jit"]["exists"],
            baseline["exported_onnx"]["exists"],
            baseline["train_exit"] == 0,
            baseline["resume_exit"] == 0,
            baseline["play_exit"] in (0, 124, 130),
            baseline["resume_loaded_checkpoint"],
            baseline["play_checkpoint_evidence"],
            baseline["play_has_no_fatal_error"],
            baseline["learning_iteration_seen"],
        ]
    )

summary = {
    "phase": "phase_0_official_baseline",
    "status": "pass" if all(item["pass"] for item in baselines.values()) else "fail",
    "run_dir": str(run_dir),
    "device": os.environ["DEVICE"],
    "seed": int(os.environ["SEED"]),
    "train_iterations": int(os.environ["TRAIN_ITERATIONS"]),
    "resume_iterations": int(os.environ["RESUME_ITERATIONS"]),
    "commands_file": str(run_dir / "commands.sh"),
    "baselines": baselines,
}

Path(os.environ["PHASE0_SUMMARY_FILE"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
if summary["status"] != "pass":
    raise SystemExit(1)
PY

printf '\nPHASE0_OFFICIAL_BASELINE_OK\n'
printf 'Summary: %s\n' "$SUMMARY_FILE"
