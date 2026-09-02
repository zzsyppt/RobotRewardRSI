#!/usr/bin/env python3
"""Record and render three fixed-protocol Stair checkpoint rollouts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.binding import require_sha256_checksum  # noqa: E402
from true_rl_agent.runner.checkpoint import (  # noqa: E402
    build_checkpoint_safety_report,
    file_sha256,
    validate_checkpoint_safety_report,
)
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.runner.specs import require_execution_device  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (  # noqa: E402
    preflight_stair_execution_source_manifest,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_EVAL_TRANSITION_EXTRAS_KEY,
    STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (  # noqa: E402
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (  # noqa: E402
    build_stair_execution_source_preflight_summary,
)
from true_rl_agent.tasks.stair_climbing.video_renderer_environment import (  # noqa: E402
    ensure_renderer_environment,
    renderer_worker_command,
    renderer_worker_environment,
    require_active_renderer_environment,
)

VIDEO_MANIFEST_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.checkpoint_video_manifest.v7"
)
TRAJECTORY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.checkpoint_trajectory.v4"
)
CAPTURE_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.checkpoint_capture_result.v6"
)
ROLLOUT_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.checkpoint_video_rollout_summary.v2"
)
_CAPTURE_RESULT_FILENAME = "capture_result.json"
_LABEL_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_PUBLIC_VISUALIZATION_SPLITS = frozenset({"public_dev"})
_OPTIONAL_HAND_JOINT_PATTERN = re.compile(
    r"^(left|right)_(zero|one|two|three|four|five|six)_joint$"
)
_G1_REPLAY_JOINT_ALIASES = {
    "torso_joint": "waist_yaw_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "left_elbow_roll_joint": "left_wrist_roll_joint",
    "right_elbow_roll_joint": "right_wrist_roll_joint",
}
_ROBOT_LANE_OFFSETS_M = (-1.2, 0.0, 1.2)
_ROBOT_COLORS_RGBA = (
    (0.16, 0.43, 0.78, 1.0),
    (0.86, 0.27, 0.20, 1.0),
    (0.16, 0.58, 0.34, 1.0),
)


@dataclass(frozen=True)
class StairCheckpointVideoPlan:
    """Validated task-owned recording plan with no raw seed exposure."""

    split: str
    run_index: int
    env_indices: tuple[int, int, int]
    evaluation_cell_id: str
    bin_id: str
    step_height_m: float
    tread_depth_m: float
    progress_goal_m: float
    episode_steps: int
    control_dt_s: float
    video_fps: int
    resolution: tuple[int, int]
    camera_eye: tuple[float, float, float]
    camera_lookat: tuple[float, float, float]
    camera_fov_deg: float

    @property
    def encoded_frame_count(self) -> int:
        return self.episode_steps

    @property
    def encoded_duration_s(self) -> float:
        return self.encoded_frame_count / self.video_fps

    @property
    def source_duration_s(self) -> float:
        return self.episode_steps * self.control_dt_s

    @property
    def playback_speed(self) -> float:
        return self.source_duration_s / self.encoded_duration_s

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "run_index": self.run_index,
            "tracked_env_indices": list(self.env_indices),
            "robot_count": len(self.env_indices),
            "evaluation_cell_id": self.evaluation_cell_id,
            "bin_id": self.bin_id,
            "step_height_m": self.step_height_m,
            "tread_depth_m": self.tread_depth_m,
            "progress_goal_m": self.progress_goal_m,
            "episode_steps": self.episode_steps,
            "control_dt_s": self.control_dt_s,
            "source_duration_s": self.source_duration_s,
            "video_fps": self.video_fps,
            "encoded_frame_count": self.encoded_frame_count,
            "encoded_duration_s": self.encoded_duration_s,
            "playback_speed": self.playback_speed,
            "state_source": "task_owned_pre_reset_transition_recorder",
            "transition_recorder_schema_version": (
                STAIR_TRANSITION_RECORDER_SCHEMA_VERSION
            ),
            "frame_mapping": "one_encoded_frame_per_control_step",
            "reset_behavior": "terminal_frame_then_labeled_new_episode",
            "resolution": list(self.resolution),
            "camera": {
                "mode": "static_full_stair_overview",
                "origin_type": "per_env_terrain_origin_with_visual_lanes",
                "eye": list(self.camera_eye),
                "lookat": list(self.camera_lookat),
                "vertical_fov_deg": self.camera_fov_deg,
            },
        }


@dataclass(frozen=True)
class ReplayBox:
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]


@dataclass(frozen=True)
class TrajectoryCaptureResult:
    trajectory_path: Path
    tracked_outcomes: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
    captured_frame_count: int
    reset_counts: tuple[int, int, int]
    joint_names: tuple[str, ...]
    rollout_summary: Mapping[str, Any]
    checkpoint_report: Mapping[str, Any]
    source_preflight: Mapping[str, str]


def _build_parser(app_launcher_class: Any | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument(
        "--config",
        default="configs/tasks/stair_protocol_v2.json",
    )
    parser.add_argument("--robot-urdf", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", default="public_dev", choices=("public_dev",))
    parser.add_argument("--run-index", type=int, default=1)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cell-id")
    selection.add_argument("--env-indices", type=int, nargs=3)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--capture-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--render-worker", action="store_true", help=argparse.SUPPRESS)
    if app_launcher_class is None:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--enable_cameras", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    else:
        app_launcher_class.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, enable_cameras=False)
    return parser


def build_video_plan(
    config: StairClimbingTaskConfig,
    *,
    split: str,
    run_index: int,
    env_indices: Sequence[int] | None,
    evaluation_cell_id: str | None = None,
    video_fps: int,
    resolution: tuple[int, int],
) -> StairCheckpointVideoPlan:
    """Bind a visual rollout to the frozen public Stair domain."""

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("Stair checkpoint video requires StairClimbingTaskConfig")
    if split not in _PUBLIC_VISUALIZATION_SPLITS:
        raise ValueError("Stair checkpoint video only permits public_dev rollouts")
    if (
        isinstance(run_index, bool)
        or not isinstance(run_index, int)
        or not 1 <= run_index <= config.public_seed_count
    ):
        raise ValueError("Stair checkpoint video run_index is outside public_dev")
    if (env_indices is None) == (evaluation_cell_id is None):
        raise ValueError(
            "Stair checkpoint video requires exactly one cell-id or env-index selection"
        )
    profile = config.evaluation_profile
    assignment = profile.private_cell_assignment()
    eval_num_envs = config.evaluation_protocol().eval_num_envs
    if len(assignment) != eval_num_envs:
        raise ValueError("Stair checkpoint video cell assignment drifted")
    if evaluation_cell_id is not None:
        if not isinstance(evaluation_cell_id, str) or not evaluation_cell_id:
            raise ValueError("Stair checkpoint video cell-id must be non-empty")
        matching_indices = tuple(
            index
            for index, cell in enumerate(assignment)
            if cell.cell_id == evaluation_cell_id
        )
        if len(matching_indices) != profile.episodes_per_seed_per_cell:
            raise ValueError("Stair checkpoint video cell-id is outside the active profile")
        normalized_env_indices = (
            matching_indices[0],
            matching_indices[len(matching_indices) // 2],
            matching_indices[-1],
        )
    else:
        if (
            not isinstance(env_indices, Sequence)
            or isinstance(env_indices, (str, bytes))
            or len(env_indices) != 3
        ):
            raise ValueError("Stair checkpoint video requires exactly three env indices")
        normalized_env_indices = tuple(env_indices)
    if any(
        isinstance(env_index, bool)
        or not isinstance(env_index, int)
        or not 0 <= env_index < eval_num_envs
        for env_index in normalized_env_indices
    ):
        raise ValueError("Stair checkpoint video env index is outside eval envs")
    if len(set(normalized_env_indices)) != 3:
        raise ValueError("Stair checkpoint video env indices must be distinct")
    if (
        isinstance(video_fps, bool)
        or not isinstance(video_fps, int)
        or not 1 <= video_fps <= 60
    ):
        raise ValueError("Stair checkpoint video fps must be in [1, 60]")
    if (
        not isinstance(resolution, tuple)
        or len(resolution) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolution
        )
    ):
        raise ValueError("Stair checkpoint video resolution must be positive")

    selected_cells = tuple(assignment[index] for index in normalized_env_indices)
    if len({cell.cell_id for cell in selected_cells}) != 1:
        raise ValueError(
            "Stair checkpoint video env indices must share one active evaluation cell"
        )
    cell = selected_cells[0]
    protocol = config.protocol_v1.to_dict()
    control_dt_s = float(protocol["policy_compatibility"]["control_dt_s"])
    camera_eye, camera_lookat = _overview_camera(
        step_height_m=cell.height_m,
        step_count=int(protocol["terrain"]["step_count"]),
        tread_depth_m=cell.tread_depth_m,
    )
    return StairCheckpointVideoPlan(
        split=split,
        run_index=run_index,
        env_indices=normalized_env_indices,
        evaluation_cell_id=cell.cell_id,
        bin_id=cell.bin_id,
        step_height_m=cell.height_m,
        tread_depth_m=cell.tread_depth_m,
        progress_goal_m=cell.progress_goal_m,
        episode_steps=cell.episode_horizon_control_steps,
        control_dt_s=control_dt_s,
        video_fps=video_fps,
        resolution=resolution,
        camera_eye=camera_eye,
        camera_lookat=camera_lookat,
        camera_fov_deg=50.0,
    )


def build_visualization_rollout_summary(
    config: StairClimbingTaskConfig,
    *,
    split: str,
    outcomes: Sequence[StairEpisodeOutcome],
) -> dict[str, Any]:
    """Aggregate one canonical rollout without reclassifying any outcome."""

    profile = config.evaluation_profile
    eval_num_envs = config.evaluation_protocol().eval_num_envs
    if len(outcomes) != eval_num_envs or not all(
        isinstance(outcome, StairEpisodeOutcome) for outcome in outcomes
    ):
        raise ValueError("Stair visualization rollout outcome coverage is incomplete")
    env_indices = [outcome.env_index for outcome in outcomes]
    if set(env_indices) != set(range(eval_num_envs)):
        raise ValueError("Stair visualization rollout env identity is incomplete")
    assignment = profile.private_cell_assignment()
    cell_ids = tuple(cell.cell_id for cell in profile.cells)
    cell_counts = Counter(outcome.evaluation_cell_id for outcome in outcomes)
    if set(cell_counts) != set(cell_ids) or any(
        cell_counts[cell_id] != profile.episodes_per_seed_per_cell
        for cell_id in cell_ids
    ):
        raise ValueError("Stair visualization rollout cell coverage is incomplete")
    if any(
        outcome.evaluation_cell_id != assignment[outcome.env_index].cell_id
        for outcome in outcomes
    ):
        raise ValueError("Stair visualization rollout cell identity drifted")
    cell_success_counts = {
        cell_id: sum(
            int(outcome.success)
            for outcome in outcomes
            if outcome.evaluation_cell_id == cell_id
        )
        for cell_id in cell_ids
    }
    cell_success_rates = {
        cell_id: cell_success_counts[cell_id] / profile.episodes_per_seed_per_cell
        for cell_id in cell_ids
    }
    height_success_rates = {
        anchor.bin_id: statistics.fmean(
            cell_success_rates[cell.cell_id]
            for cell in profile.cells
            if cell.bin_id == anchor.bin_id
        )
        for anchor in profile.anchor_cells
    }
    tread_success_rates = {
        f"{tread:.2f}": statistics.fmean(
            cell_success_rates[cell.cell_id]
            for cell in profile.cells
            if abs(cell.tread_depth_m - tread) <= 1.0e-12
        )
        for tread in profile.tread_depths_m
    }
    primary_counts = Counter(outcome.primary_reason for outcome in outcomes)
    return {
        "schema_version": ROLLOUT_SUMMARY_SCHEMA_VERSION,
        "scope": "one_active_profile_public_dev_run_not_acceptance_score",
        "split": split,
        "episode_count": len(outcomes),
        "evaluation_profile_hash": profile.profile_hash,
        "episodes_per_cell": profile.episodes_per_seed_per_cell,
        "complete_height_tread_coverage": True,
        "outcome_source": "task_owned_formal_episode_state_machine",
        "reclassified_by_video_exporter": False,
        "cell_success_counts": cell_success_counts,
        "cell_success_rates": cell_success_rates,
        "height_success_rates": height_success_rates,
        "tread_success_rates": tread_success_rates,
        "macro_success_rate": statistics.fmean(cell_success_rates.values()),
        "primary_outcome_counts": dict(sorted(primary_counts.items())),
    }


def _overview_camera(
    *,
    step_height_m: float,
    step_count: int,
    tread_depth_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if step_height_m <= 0.0 or step_count <= 0 or tread_depth_m <= 0.0:
        raise ValueError("Stair overview camera requires positive stair geometry")
    top_height_m = step_height_m * step_count
    stair_run_m = tread_depth_m * step_count
    eye = (0.5, -max(8.0, stair_run_m + 4.0), max(4.2, top_height_m + 2.0))
    lookat = (
        max(2.8, 0.75 + 0.65 * stair_run_m),
        0.0,
        0.45 * top_height_m + 0.55,
    )
    return eye, lookat


def build_replay_stair_boxes(
    protocol: Mapping[str, Any],
    *,
    step_height_m: float,
    tread_depth_m: float,
) -> tuple[ReplayBox, ...]:
    """Rebuild the frozen stair mesh as boxes relative to its task origin."""

    terrain = protocol["terrain"]
    size_x, size_y = (float(value) for value in terrain["size_m"])
    floor_depth = float(terrain["floor_depth_m"])
    start_length = float(terrain["start_platform_length_m"])
    tread_depth = float(tread_depth_m)
    step_count = int(terrain["step_count"])
    if tread_depth <= 0.0:
        raise ValueError("Stair replay tread depth must be positive")
    top_length = size_x - start_length - step_count * tread_depth
    if top_length <= 0.0:
        raise ValueError("Stair replay tread depth leaves no top platform")
    origin_x = start_length / 2.0
    boxes = [
        ReplayBox(
            center=(size_x / 2.0 - origin_x, 0.0, -floor_depth / 2.0),
            half_extents=(size_x / 2.0, size_y / 2.0, floor_depth / 2.0),
        )
    ]
    for level in range(1, step_count + 1):
        segment_start = start_length + (level - 1) * tread_depth
        height = level * step_height_m
        boxes.append(
            ReplayBox(
                center=(
                    segment_start + tread_depth / 2.0 - origin_x,
                    0.0,
                    (height - floor_depth) / 2.0,
                ),
                half_extents=(
                    tread_depth / 2.0,
                    size_y / 2.0,
                    (floor_depth + height) / 2.0,
                ),
            )
        )
    top_start = start_length + step_count * tread_depth
    top_height = step_count * step_height_m
    boxes.append(
        ReplayBox(
            center=(
                top_start + top_length / 2.0 - origin_x,
                0.0,
                (top_height - floor_depth) / 2.0,
            ),
            half_extents=(
                top_length / 2.0,
                size_y / 2.0,
                (floor_depth + top_height) / 2.0,
            ),
        )
    )
    return tuple(boxes)


def classify_replay_joint_mapping(
    trajectory_joint_names: Sequence[str],
    replay_joint_names: Sequence[str],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Resolve the Isaac G1 schema onto the vendor 23-DoF replay URDF."""

    replay_names = set(replay_joint_names)
    mapped = []
    missing = []
    for trajectory_name in trajectory_joint_names:
        replay_name = _G1_REPLAY_JOINT_ALIASES.get(
            trajectory_name,
            trajectory_name,
        )
        if replay_name in replay_names:
            mapped.append((trajectory_name, replay_name))
        else:
            missing.append(trajectory_name)
    unsupported = [
        name for name in missing if _OPTIONAL_HAND_JOINT_PATTERN.fullmatch(name) is None
    ]
    if unsupported:
        raise ValueError(
            "Stair replay URDF is missing policy-controlled body joints: "
            + ", ".join(unsupported)
        )
    return tuple(mapped), tuple(missing)


def _resolve_local_path(value: str, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty local path")
    if "://" in value:
        raise ValueError(f"{field_name} must not be a URI")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _validate_label(value: str) -> str:
    if not isinstance(value, str) or _LABEL_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "Stair checkpoint video label must be a lowercase filesystem slug"
        )
    return value


class _TrajectoryCaptureVecEnv:
    """Proxy canonical eval while retaining exact task-owned pre-reset states."""

    def __init__(self, env: Any, *, plan: StairCheckpointVideoPlan) -> None:
        self.env = env
        self._env_indices = plan.env_indices
        self._plan = plan
        self._root_positions: list[Any] = []
        self._root_quaternions: list[Any] = []
        self._joint_positions: list[Any] = []
        self._terminal_flags: list[tuple[bool, bool, bool]] = []
        self._episode_start_flags: list[tuple[bool, bool, bool]] = []
        self._episode_indices: list[tuple[int, int, int]] = []
        self._episode_control_steps: list[tuple[int, int, int]] = []
        self._current_episode_indices = [0, 0, 0]
        self._next_episode_start = [True, True, True]
        self._last_observation: Any | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self.env.reset(*args, **kwargs)
        if not isinstance(result, tuple) or not result:
            raise RuntimeError("Stair trajectory env reset contract drifted")
        self._last_observation = result[0]
        self._root_positions.clear()
        self._root_quaternions.clear()
        self._joint_positions.clear()
        self._terminal_flags.clear()
        self._episode_start_flags.clear()
        self._episode_indices.clear()
        self._episode_control_steps.clear()
        self._current_episode_indices = [0, 0, 0]
        self._next_episode_start = [True, True, True]
        return result

    def step(self, actions: Any) -> tuple[Any, ...]:
        result = self.env.step(actions)
        if not isinstance(result, tuple) or len(result) < 3:
            raise RuntimeError("Stair trajectory env step contract drifted")
        self._last_observation = result[0]
        if len(self._root_positions) < self._plan.episode_steps:
            self._capture_pre_reset(result)
        return result

    @property
    def captured_frame_count(self) -> int:
        return len(self._root_positions)

    @property
    def last_observation(self) -> Any:
        if self._last_observation is None:
            raise RuntimeError("Stair trajectory env has not been reset")
        return self._last_observation

    def trajectory_arrays(
        self,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, tuple[int, int, int]]:
        import numpy as np

        captured_count = len(self._root_positions)
        if captured_count != self._plan.episode_steps:
            raise RuntimeError(
                "Stair trajectory capture did not fill the fixed continuous horizon"
            )
        timeline_shape = (self._plan.episode_steps, len(self._env_indices))
        terminal_flags = np.asarray(self._terminal_flags, dtype=np.bool_)
        episode_start_flags = np.asarray(self._episode_start_flags, dtype=np.bool_)
        episode_indices = np.asarray(self._episode_indices, dtype=np.int64)
        episode_control_steps = np.asarray(
            self._episode_control_steps,
            dtype=np.int64,
        )
        if any(
            value.shape != timeline_shape
            for value in (
                terminal_flags,
                episode_start_flags,
                episode_indices,
                episode_control_steps,
            )
        ):
            raise RuntimeError("Stair trajectory episode timeline shape drifted")
        reset_counts = tuple(
            int(value) for value in terminal_flags.sum(axis=0).tolist()
        )
        return (
            np.asarray(self._root_positions),
            np.asarray(self._root_quaternions),
            np.asarray(self._joint_positions),
            terminal_flags,
            episode_start_flags,
            episode_indices,
            episode_control_steps,
            reset_counts,
        )

    def _capture_pre_reset(self, result: tuple[Any, ...]) -> None:
        extras = result[-1]
        if not isinstance(extras, Mapping):
            raise RuntimeError("Stair trajectory step extras contract drifted")
        raw = extras.get(STAIR_EVAL_TRANSITION_EXTRAS_KEY)
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                "Stair trajectory requires task-owned pre-reset recorder evidence"
            )
        if raw.get("schema_version") != STAIR_TRANSITION_RECORDER_SCHEMA_VERSION:
            raise RuntimeError("Stair trajectory recorder schema drifted")
        required_fields = (
            "root_pos_w",
            "root_quat_w",
            "joint_pos",
            "episode_length",
            "terrain_env_origins_w",
            "done_mask",
        )
        if any(name not in raw for name in required_fields):
            raise RuntimeError("Stair trajectory recorder payload is incomplete")

        wrapper_done = _wrapper_done_mask(result)
        terminal_flags = tuple(
            _as_bool(raw["done_mask"][index]) for index in self._env_indices
        )
        observed_wrapper_done = tuple(
            _as_bool(wrapper_done[index]) for index in self._env_indices
        )
        if observed_wrapper_done != terminal_flags:
            raise RuntimeError("Stair trajectory wrapper/recorder done masks differ")

        root_positions = []
        root_quaternions = []
        joint_positions = []
        for index in self._env_indices:
            root_position = (
                raw["root_pos_w"][index] - raw["terrain_env_origins_w"][index]
            ).detach()
            root_positions.append(root_position.cpu().numpy().copy())
            root_quaternions.append(
                raw["root_quat_w"][index].detach().cpu().numpy().copy()
            )
            joint_positions.append(
                raw["joint_pos"][index].detach().cpu().numpy().copy()
            )
        self._root_positions.append(root_positions)
        self._root_quaternions.append(root_quaternions)
        self._joint_positions.append(joint_positions)
        self._terminal_flags.append(terminal_flags)
        self._episode_start_flags.append(tuple(self._next_episode_start))
        self._episode_indices.append(tuple(self._current_episode_indices))
        self._episode_control_steps.append(
            tuple(
                int(_scalar_value(raw["episode_length"][index]))
                for index in self._env_indices
            )
        )
        self._next_episode_start = [False, False, False]
        for slot, terminal in enumerate(terminal_flags):
            if terminal:
                self._current_episode_indices[slot] += 1
                self._next_episode_start[slot] = True


def _wrapper_done_mask(result: tuple[Any, ...]) -> Any:
    if len(result) == 4:
        return result[2]
    if len(result) == 5:
        return result[2] | result[3]
    raise RuntimeError("Stair trajectory env step arity is unsupported")


def _as_bool(value: Any) -> bool:
    return bool(_scalar_value(value))


def _scalar_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _capture_trajectory(
    *,
    config: StairClimbingTaskConfig,
    source_manifest_preflight: Any,
    checkpoint_path: Path,
    device: str,
    output_dir: Path,
    plan: StairCheckpointVideoPlan,
    checkpoint_report: Mapping[str, Any],
) -> TrajectoryCaptureResult:
    import gymnasium as gym
    import numpy as np
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
        G1RoughPPORunnerCfg,
    )
    from rsl_rl.runners import OnPolicyRunner

    from true_rl_agent.tasks.stair_climbing.evaluation_runtime import (
        run_stair_seed_rollout,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_eval_env_cfg,
        register_production_stair_envs,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_training import (
        apply_stair_rsl_rl_runner_contract,
        verify_stair_rsl_rl_observation_groups,
    )
    from true_rl_agent.tasks.stair_climbing.runtime_profile import (
        STAIR_EVAL_GYM_ID,
    )
    from true_rl_agent.tasks.stair_climbing.termination_capture import (
        verify_stair_termination_manager_resolution,
    )

    source_manifest_preflight.validate_for(config)
    resolver = config.seed_resolver()
    resolver.namespace_cohort_hash(STAIR_FORMAL_EVAL_NAMESPACE)
    register_production_stair_envs()
    env_cfg = build_stair_eval_env_cfg(
        config.protocol_v1.to_dict(),
        seed=resolver.base_seed,
        device=device,
        evaluation_profile=config.evaluation_profile_config,
    )
    base_env = gym.make(STAIR_EVAL_GYM_ID, cfg=env_cfg)
    env = None
    try:
        raw_env = base_env.unwrapped
        verify_stair_termination_manager_resolution(
            raw_env,
            protocol=config.protocol_v1.to_dict(),
        )
        agent_cfg = G1RoughPPORunnerCfg()
        config.training_agent_configuration().apply_and_verify(agent_cfg)
        agent_cfg.seed = config.train_seed
        agent_cfg.device = device
        execution = config.protocol_v2.to_public_dict()["training_execution"]
        apply_stair_rsl_rl_runner_contract(
            agent_cfg,
            execution_contract=execution,
        )
        rsl_env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        env = _TrajectoryCaptureVecEnv(rsl_env, plan=plan)
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        verify_stair_rsl_rl_observation_groups(
            runner,
            execution_contract=execution,
        )
        runner.load(str(checkpoint_path))
        inference_policy = runner.get_inference_policy(device=env.unwrapped.device)
        outcomes = run_stair_seed_rollout(
            env=env,
            inference_policy=inference_policy,
            protocol=config.protocol_v1,
            semantics=config.evaluation_semantics,
            seed_resolver=resolver,
            split=plan.split,
            run_index=plan.run_index,
            seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE,
        )
        if len(outcomes) != resolver.evaluation_protocol.eval_num_envs:
            raise RuntimeError("Stair checkpoint trajectory outcome coverage is incomplete")
        while env.captured_frame_count < plan.episode_steps:
            actions = inference_policy(env.last_observation)
            env.step(actions)
        (
            root_positions,
            root_quaternions,
            joint_positions,
            terminal_flags,
            episode_start_flags,
            episode_indices,
            episode_control_steps,
            reset_counts,
        ) = env.trajectory_arrays()
        captured_count = env.captured_frame_count
        joint_names = tuple(str(name) for name in raw_env.scene["robot"].joint_names)
        trajectory_path = output_dir / "trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            schema_version=np.asarray(TRAJECTORY_SCHEMA_VERSION),
            root_pos_m=root_positions,
            root_quat_wxyz=root_quaternions,
            joint_pos_rad=joint_positions,
            joint_names=np.asarray(joint_names),
            env_indices=np.asarray(plan.env_indices),
            control_steps=np.arange(1, captured_count + 1, dtype=np.int64),
            terminal_flags=terminal_flags,
            episode_start_flags=episode_start_flags,
            episode_indices=episode_indices,
            episode_control_steps=episode_control_steps,
            captured_frame_count=np.asarray(captured_count),
            reset_counts=np.asarray(reset_counts),
        )
        return TrajectoryCaptureResult(
            trajectory_path=trajectory_path,
            tracked_outcomes=tuple(
                outcomes[index].to_public_dict() for index in plan.env_indices
            ),
            captured_frame_count=captured_count,
            reset_counts=reset_counts,
            joint_names=joint_names,
            rollout_summary=build_visualization_rollout_summary(
                config,
                split=plan.split,
                outcomes=outcomes,
            ),
            checkpoint_report=dict(checkpoint_report),
            source_preflight=source_manifest_preflight.to_summary_dict(),
        )
    finally:
        (env if env is not None else base_env).close()


def _write_capture_result(
    output_dir: Path,
    trajectory: TrajectoryCaptureResult,
) -> Path:
    path = output_dir / _CAPTURE_RESULT_FILENAME
    payload = {
        "schema_version": CAPTURE_RESULT_SCHEMA_VERSION,
        "trajectory_filename": trajectory.trajectory_path.name,
        "tracked_outcomes": [dict(item) for item in trajectory.tracked_outcomes],
        "captured_frame_count": trajectory.captured_frame_count,
        "reset_counts": list(trajectory.reset_counts),
        "joint_names": list(trajectory.joint_names),
        "rollout_summary": dict(trajectory.rollout_summary),
        "checkpoint_report": dict(trajectory.checkpoint_report),
        "source_preflight": dict(trajectory.source_preflight),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _load_capture_result(output_dir: Path) -> TrajectoryCaptureResult:
    path = output_dir / _CAPTURE_RESULT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Stair capture worker result is missing or invalid") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Stair capture worker result must be an object")
    if payload.get("schema_version") != CAPTURE_RESULT_SCHEMA_VERSION:
        raise RuntimeError("Stair capture worker result schema drifted")
    trajectory_path = output_dir / str(payload.get("trajectory_filename", ""))
    if trajectory_path.parent != output_dir or not trajectory_path.is_file():
        raise RuntimeError("Stair capture worker trajectory path is invalid")
    tracked_outcomes = payload.get("tracked_outcomes")
    joint_names = payload.get("joint_names")
    captured_count = payload.get("captured_frame_count")
    reset_counts = payload.get("reset_counts")
    rollout_summary = payload.get("rollout_summary")
    checkpoint_report = payload.get("checkpoint_report")
    source_preflight = payload.get("source_preflight")
    if (
        not isinstance(tracked_outcomes, list)
        or len(tracked_outcomes) != 3
        or not all(isinstance(item, Mapping) for item in tracked_outcomes)
    ):
        raise RuntimeError("Stair capture worker outcomes are invalid")
    if not isinstance(joint_names, list) or not all(
        isinstance(name, str) and name for name in joint_names
    ):
        raise RuntimeError("Stair capture worker joint schema is invalid")
    if (
        isinstance(captured_count, bool)
        or not isinstance(captured_count, int)
        or captured_count <= 0
    ):
        raise RuntimeError("Stair capture worker captured_frame_count is invalid")
    if (
        not isinstance(reset_counts, list)
        or len(reset_counts) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in reset_counts
        )
    ):
        raise RuntimeError("Stair capture worker reset_counts are invalid")
    if (
        not isinstance(rollout_summary, Mapping)
        or rollout_summary.get("schema_version")
        != ROLLOUT_SUMMARY_SCHEMA_VERSION
        or rollout_summary.get("complete_height_tread_coverage") is not True
        or rollout_summary.get("reclassified_by_video_exporter") is not False
    ):
        raise RuntimeError("Stair capture worker rollout summary is invalid")
    try:
        validated_checkpoint_report = validate_checkpoint_safety_report(
            checkpoint_report
        )
    except ValueError as exc:
        raise RuntimeError("Stair capture worker checkpoint report is invalid") from exc
    if not isinstance(source_preflight, Mapping):
        raise RuntimeError("Stair capture worker source preflight is invalid")
    return TrajectoryCaptureResult(
        trajectory_path=trajectory_path,
        tracked_outcomes=tuple(dict(item) for item in tracked_outcomes),
        captured_frame_count=captured_count,
        reset_counts=tuple(reset_counts),
        joint_names=tuple(joint_names),
        rollout_summary=dict(rollout_summary),
        checkpoint_report=validated_checkpoint_report,
        source_preflight={str(key): str(value) for key, value in source_preflight.items()},
    )


def _render_trajectory(
    *,
    config: StairClimbingTaskConfig,
    trajectory: TrajectoryCaptureResult,
    robot_urdf: Path,
    output_dir: Path,
    label: str,
    plan: StairCheckpointVideoPlan,
) -> tuple[Path, dict[str, Any]]:
    try:
        import cv2
        import imageio.v2 as imageio
        import numpy as np
        import pybullet as bullet
    except ImportError as exc:
        raise RuntimeError(
            "Stair state-replay video requires pybullet, imageio, numpy, and cv2"
        ) from exc

    data = np.load(trajectory.trajectory_path, allow_pickle=False)
    if str(data["schema_version"].item()) != TRAJECTORY_SCHEMA_VERSION:
        raise RuntimeError("Stair replay trajectory schema drifted")
    root_positions = data["root_pos_m"]
    root_quaternions = data["root_quat_wxyz"]
    joint_positions = data["joint_pos_rad"]
    env_indices = tuple(int(value) for value in data["env_indices"].tolist())
    control_steps = data["control_steps"]
    terminal_flags = data["terminal_flags"]
    episode_start_flags = data["episode_start_flags"]
    episode_indices = data["episode_indices"]
    episode_control_steps = data["episode_control_steps"]
    joint_names = tuple(str(value) for value in data["joint_names"].tolist())
    robot_count = len(plan.env_indices)
    if env_indices != plan.env_indices:
        raise RuntimeError("Stair replay tracked env indices drifted")
    if root_positions.shape != (plan.episode_steps, robot_count, 3):
        raise RuntimeError("Stair replay root trajectory shape drifted")
    if root_quaternions.shape != (plan.episode_steps, robot_count, 4):
        raise RuntimeError("Stair replay quaternion trajectory shape drifted")
    if joint_positions.shape != (
        plan.episode_steps,
        robot_count,
        len(joint_names),
    ):
        raise RuntimeError("Stair replay joint trajectory shape drifted")
    timeline_shape = (plan.episode_steps, robot_count)
    if control_steps.shape != (plan.episode_steps,):
        raise RuntimeError("Stair replay control-step timeline shape drifted")
    if control_steps.tolist() != list(range(1, plan.episode_steps + 1)):
        raise RuntimeError("Stair replay control-step timeline drifted")
    if any(
        value.shape != timeline_shape
        for value in (
            terminal_flags,
            episode_start_flags,
            episode_indices,
            episode_control_steps,
        )
    ):
        raise RuntimeError("Stair replay episode timeline shape drifted")
    if tuple(int(value) for value in data["reset_counts"].tolist()) != (
        trajectory.reset_counts
    ):
        raise RuntimeError("Stair replay reset counts drifted")

    client = bullet.connect(bullet.DIRECT)
    if client < 0:
        raise RuntimeError("Stair replay could not start PyBullet DIRECT")
    writer = None
    try:
        bullet.setAdditionalSearchPath(str(robot_urdf.parent), physicsClientId=client)
        robot_ids = tuple(
            bullet.loadURDF(
                str(robot_urdf),
                useFixedBase=False,
                flags=bullet.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=client,
            )
            for _ in plan.env_indices
        )
        robot_id = robot_ids[0]
        replay_joint_indices: dict[str, int] = {}
        for index in range(bullet.getNumJoints(robot_id, physicsClientId=client)):
            info = bullet.getJointInfo(robot_id, index, physicsClientId=client)
            replay_joint_indices[info[1].decode("utf-8")] = index
        mapped_pairs, missing_names = classify_replay_joint_mapping(
            joint_names,
            tuple(replay_joint_indices),
        )
        trajectory_indices = {name: index for index, name in enumerate(joint_names)}
        mapping = tuple(
            (
                trajectory_indices[trajectory_name],
                replay_joint_indices[replay_name],
            )
            for trajectory_name, replay_name in mapped_pairs
        )
        for body_id, color in zip(
            robot_ids,
            _ROBOT_COLORS_RGBA,
            strict=True,
        ):
            for link_index in range(
                -1,
                bullet.getNumJoints(body_id, physicsClientId=client),
            ):
                bullet.changeVisualShape(
                    body_id,
                    link_index,
                    rgbaColor=color,
                    physicsClientId=client,
                )

        boxes = build_replay_stair_boxes(
            config.protocol_v1.to_dict(),
            step_height_m=plan.step_height_m,
            tread_depth_m=plan.tread_depth_m,
        )
        for index, box in enumerate(boxes):
            shade = 0.62 if index % 2 == 0 else 0.70
            visual = bullet.createVisualShape(
                bullet.GEOM_BOX,
                halfExtents=box.half_extents,
                rgbaColor=(shade, shade, shade + 0.03, 1.0),
                physicsClientId=client,
            )
            bullet.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=box.center,
                physicsClientId=client,
            )

        width, height = plan.resolution
        view = bullet.computeViewMatrix(
            cameraEyePosition=plan.camera_eye,
            cameraTargetPosition=plan.camera_lookat,
            cameraUpVector=(0.0, 0.0, 1.0),
        )
        projection = bullet.computeProjectionMatrixFOV(
            fov=plan.camera_fov_deg,
            aspect=width / height,
            nearVal=0.05,
            farVal=30.0,
        )
        video_path = output_dir / f"{label}.mp4"
        writer = imageio.get_writer(
            video_path,
            fps=plan.video_fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        first_outcome_markers = tuple(
            (
                int(outcome["episode_steps"]),
                (
                    "SUCCESS"
                    if outcome["success"] is True
                    else str(outcome["primary_reason"]).upper()
                ),
            )
            for outcome in trajectory.tracked_outcomes
        )
        for source_index in range(plan.episode_steps):
            for robot_slot, (body_id, lane_offset_m) in enumerate(
                zip(robot_ids, _ROBOT_LANE_OFFSETS_M, strict=True)
            ):
                root_quat = root_quaternions[source_index, robot_slot]
                root_position = root_positions[source_index, robot_slot].copy()
                root_position[1] += lane_offset_m
                bullet.resetBasePositionAndOrientation(
                    body_id,
                    root_position,
                    (root_quat[1], root_quat[2], root_quat[3], root_quat[0]),
                    physicsClientId=client,
                )
                for trajectory_index, replay_index in mapping:
                    bullet.resetJointState(
                        body_id,
                        replay_index,
                        float(
                            joint_positions[
                                source_index,
                                robot_slot,
                                trajectory_index,
                            ]
                        ),
                        physicsClientId=client,
                    )
            image = bullet.getCameraImage(
                width,
                height,
                viewMatrix=view,
                projectionMatrix=projection,
                renderer=bullet.ER_TINY_RENDERER,
                flags=bullet.ER_NO_SEGMENTATION_MASK,
                physicsClientId=client,
            )
            rgb = np.asarray(image[2], dtype=np.uint8).reshape(height, width, 4)[
                :, :, :3
            ].copy()
            caption = (
                f"{label} | {plan.evaluation_cell_id} "
                f"h={plan.step_height_m:.2f} m t={plan.tread_depth_m:.2f} m | "
                f"playback={source_index / plan.video_fps:05.2f} s | "
                f"control_step={int(control_steps[source_index]):03d}"
            )
            cv2.putText(
                rgb,
                caption,
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (24, 24, 24),
                2,
                cv2.LINE_AA,
            )
            for slot, (name, env_index) in enumerate(
                zip(("A", "B", "C"), plan.env_indices, strict=True)
            ):
                marker = ""
                if bool(terminal_flags[source_index, slot]):
                    marker = " TERMINAL (PRE-RESET)"
                elif bool(episode_start_flags[source_index, slot]):
                    marker = (
                        " RESET/START"
                        if int(episode_indices[source_index, slot]) > 0
                        else " START"
                    )
                episode_caption = (
                    f"{name} env{env_index} ep={int(episode_indices[source_index, slot]) + 1} "
                    f"ep_step={int(episode_control_steps[source_index, slot])}{marker}"
                )
                outcome_step, outcome_label = first_outcome_markers[slot]
                if int(control_steps[source_index]) >= outcome_step:
                    episode_caption += f" FIRST={outcome_label}@{outcome_step}"
                cv2.putText(
                    rgb,
                    episode_caption,
                    (24, 74 + slot * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (24, 24, 24),
                    2,
                    cv2.LINE_AA,
                )
            writer.append_data(rgb)
        writer.close()
        writer = None
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError("Stair replay did not produce a non-empty mp4")
        return video_path, {
            "backend": "pybullet_tiny_renderer",
            "source": "exact_task_owned_pre_reset_isaac_state_trajectory",
            "transition_recorder_schema_version": (
                STAIR_TRANSITION_RECORDER_SCHEMA_VERSION
            ),
            "robot_urdf": str(robot_urdf),
            "robot_urdf_checksum": file_sha256(robot_urdf),
            "mapped_joint_count": len(mapped_pairs),
            "robot_count": robot_count,
            "tracked_env_indices": list(plan.env_indices),
            "visual_lane_offsets_m": list(_ROBOT_LANE_OFFSETS_M),
            "robot_colors_rgba": [list(color) for color in _ROBOT_COLORS_RGBA],
            "frame_mapping": "one_encoded_frame_per_control_step",
            "reset_behavior": "terminal_frame_then_labeled_new_episode",
            "terminal_states_preserved": True,
            "episode_identity_overlaid": True,
            "first_episode_outcome_overlaid": True,
            "rendered_frame_count": plan.encoded_frame_count,
            "joint_aliases_used": {
                trajectory_name: replay_name
                for trajectory_name, replay_name in mapped_pairs
                if trajectory_name != replay_name
            },
            "unmapped_visual_only_hand_joints": list(missing_names),
            "physics_recomputed": False,
        }
    finally:
        if writer is not None:
            writer.close()
        bullet.disconnect(physicsClientId=client)


def _build_manifest(
    *,
    label: str,
    config: StairClimbingTaskConfig,
    source_preflight: Mapping[str, str],
    checkpoint_path: Path,
    checkpoint_report: Mapping[str, Any],
    device: str,
    plan: StairCheckpointVideoPlan,
    trajectory: TrajectoryCaptureResult,
    video_path: Path,
    render_report: Mapping[str, Any],
) -> dict[str, Any]:
    video_digest = file_sha256(video_path)
    trajectory_digest = file_sha256(trajectory.trajectory_path)
    if video_digest is None or trajectory_digest is None:
        raise RuntimeError("Stair checkpoint video hashes could not be computed")
    resolver = config.seed_resolver()
    return {
        "schema_version": VIDEO_MANIFEST_SCHEMA_VERSION,
        "status": "pass",
        "purpose": "diagnostic_state_replay_not_acceptance_evidence",
        "label": label,
        "task_id": "unitree_stair_climbing",
        "execution_device": device,
        "checkpoint": {
            "path": str(checkpoint_path),
            "checksum": checkpoint_report["checkpoint_hash"],
            "safety_status": checkpoint_report["status"],
        },
        "protocol": {
            "task_protocol_hash": config.task_protocol_hash,
            "eval_protocol_hash": config.evaluation_protocol().protocol_hash,
            "evaluation_profile_hash": config.evaluation_profile.profile_hash,
            "formal_seed_cohort_hash": resolver.namespace_cohort_hash(
                STAIR_FORMAL_EVAL_NAMESPACE
            ),
            "scope": "active_height_tread_profile_diagnostic_replay",
            "raw_seed_exposed": False,
        },
        "source_preflight": dict(source_preflight),
        "recording": plan.to_manifest_dict(),
        "trajectory": {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "path": str(trajectory.trajectory_path),
            "checksum": trajectory_digest,
            "captured_frame_count": trajectory.captured_frame_count,
            "reset_counts": list(trajectory.reset_counts),
            "joint_count": len(trajectory.joint_names),
        },
        "renderer": dict(render_report),
        "canonical_rollout_summary": dict(trajectory.rollout_summary),
        "tracked_first_episode_outcomes": [
            {
                "env_index": env_index,
                "outcome": dict(outcome),
            }
            for env_index, outcome in zip(
                plan.env_indices,
                trajectory.tracked_outcomes,
                strict=True,
            )
        ],
        "video": {
            "path": str(video_path),
            "checksum": video_digest,
            "size_bytes": video_path.stat().st_size,
        },
    }


def _canonical_worker_arguments(
    *,
    args: argparse.Namespace,
    config_path: Path,
    checkpoint_path: Path,
    expected_checkpoint_hash: str,
    robot_urdf: Path,
    output_dir: Path,
    label: str,
) -> list[str]:
    selection_arguments = (
        ["--cell-id", str(args.cell_id)]
        if args.cell_id is not None
        else [
            "--env-indices",
            *(str(index) for index in args.env_indices),
        ]
    )
    return [
        "--checkpoint",
        str(checkpoint_path),
        "--checkpoint-sha256",
        expected_checkpoint_hash,
        "--config",
        str(config_path),
        "--robot-urdf",
        str(robot_urdf),
        "--output-dir",
        str(output_dir),
        "--label",
        label,
        "--split",
        args.split,
        "--run-index",
        str(args.run_index),
        *selection_arguments,
        "--fps",
        str(args.fps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--device",
        str(args.device),
    ]


def _render_captured_trajectory(
    *,
    config: StairClimbingTaskConfig,
    checkpoint_path: Path,
    expected_checkpoint_hash: str,
    robot_urdf: Path,
    output_dir: Path,
    label: str,
    device: str,
    plan: StairCheckpointVideoPlan,
) -> tuple[Path, Path]:
    renderer_environment = require_active_renderer_environment(PROJECT_ROOT)
    trajectory = _load_capture_result(output_dir)
    source_preflight = _validate_source_preflight_summary(
        config,
        trajectory.source_preflight,
    )
    checkpoint_report = validate_checkpoint_safety_report(
        trajectory.checkpoint_report,
        expected_checkpoint_hash=expected_checkpoint_hash,
    )
    if file_sha256(checkpoint_path) != checkpoint_report["checkpoint_hash"]:
        raise RuntimeError("Stair checkpoint bytes changed after trajectory capture")
    video_path, render_report = _render_trajectory(
        config=config,
        trajectory=trajectory,
        robot_urdf=robot_urdf,
        output_dir=output_dir,
        label=label,
        plan=plan,
    )
    render_report = {
        **render_report,
        "environment": renderer_environment.to_manifest_dict(),
    }
    manifest = _build_manifest(
        label=label,
        config=config,
        source_preflight=source_preflight,
        checkpoint_path=checkpoint_path,
        checkpoint_report=checkpoint_report,
        device=device,
        plan=plan,
        trajectory=trajectory,
        video_path=video_path,
        render_report=render_report,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return video_path, manifest_path


def _validate_source_preflight_summary(
    config: StairClimbingTaskConfig,
    value: Mapping[str, str],
) -> dict[str, str]:
    expected = build_stair_execution_source_preflight_summary(
        task_protocol_hash=config.task_protocol_hash,
        source_manifest_hash=value.get("source_manifest_hash"),
    )
    if dict(value) != expected:
        raise RuntimeError("Stair capture worker source preflight drifted")
    return expected


def main(
    argv: list[str] | None = None,
    *,
    app_launcher_class: Any | None = None,
) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    render_worker_requested = "--render-worker" in raw_argv
    if app_launcher_class is None and not render_worker_requested:
        from isaaclab.app import AppLauncher

        app_launcher_class = AppLauncher
    args = _build_parser(app_launcher_class).parse_args(raw_argv)
    try:
        if args.capture_worker and args.render_worker:
            raise ValueError("Stair checkpoint workers are mutually exclusive")
        args.device = require_execution_device(
            args.device,
            "stair checkpoint video execution device",
        )
        config_path = _resolve_local_path(args.config, field_name="config")
        checkpoint_path = _resolve_local_path(
            args.checkpoint,
            field_name="checkpoint",
        )
        robot_urdf = _resolve_local_path(args.robot_urdf, field_name="robot_urdf")
        output_dir = _resolve_local_path(
            args.output_dir,
            field_name="output_dir",
        )
        label = _validate_label(args.label)
        expected_checkpoint_hash = require_sha256_checksum(
            args.checkpoint_sha256,
            "stair checkpoint video expected checkpoint hash",
        )
        config = StairClimbingTaskConfig.defaults(config_path)
        plan = build_video_plan(
            config,
            split=args.split,
            run_index=args.run_index,
            env_indices=args.env_indices,
            evaluation_cell_id=args.cell_id,
            video_fps=args.fps,
            resolution=(args.width, args.height),
        )
        checkpoint_report = None
        if not args.render_worker:
            checkpoint_report = build_checkpoint_safety_report(
                checkpoint_path,
                expected_checkpoint_hash=expected_checkpoint_hash,
            ).to_public_dict()
            if checkpoint_report["status"] != "pass":
                raise ValueError(str(checkpoint_report["reason"]))
        if not robot_urdf.is_file():
            raise ValueError("Stair replay robot URDF does not exist")
        source_manifest_preflight = (
            None
            if args.render_worker
            else preflight_stair_execution_source_manifest(config)
        )
        if args.render_worker and not output_dir.is_dir():
            raise ValueError("Stair render worker requires captured output directory")
        if not args.render_worker and output_dir.exists():
            raise ValueError("Stair checkpoint video output directory already exists")
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.capture_worker:
        assert checkpoint_report is not None
        assert source_manifest_preflight is not None
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            print(f"Stair checkpoint output creation failed: {exc}", file=sys.stderr)
            return 2
        app_launcher = app_launcher_class(args)
        simulation_app = app_launcher.app
        trajectory = None
        capture_status = 1
        try:
            trajectory = _capture_trajectory(
                config=config,
                source_manifest_preflight=source_manifest_preflight,
                checkpoint_path=checkpoint_path,
                device=args.device,
                output_dir=output_dir,
                plan=plan,
                checkpoint_report=checkpoint_report,
            )
            _write_capture_result(output_dir, trajectory)
            capture_status = 0
        except Exception as exc:  # noqa: BLE001
            print(f"Stair checkpoint trajectory capture failed: {exc}", file=sys.stderr)
        return close_isaac_app_on_success(
            simulation_app,
            exit_code=capture_status,
        )

    if args.render_worker:
        try:
            video_path, manifest_path = _render_captured_trajectory(
                config=config,
                checkpoint_path=checkpoint_path,
                expected_checkpoint_hash=expected_checkpoint_hash,
                robot_urdf=robot_urdf,
                output_dir=output_dir,
                label=label,
                device=args.device,
                plan=plan,
            )
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "video_path": str(video_path),
                        "manifest_path": str(manifest_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Stair checkpoint state replay failed: {exc}", file=sys.stderr)
            return 1

    assert checkpoint_report is not None
    worker_args = _canonical_worker_arguments(
        args=args,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_hash=expected_checkpoint_hash,
        robot_urdf=robot_urdf,
        output_dir=output_dir,
        label=label,
    )
    worker = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *worker_args,
            "--capture-worker",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if worker.returncode != 0:
        print(
            f"Stair checkpoint capture worker failed with code {worker.returncode}",
            file=sys.stderr,
        )
        return worker.returncode
    try:
        renderer_environment = ensure_renderer_environment(PROJECT_ROOT)
    except Exception as exc:  # noqa: BLE001
        print(f"Stair renderer environment failed: {exc}", file=sys.stderr)
        return 1
    renderer = subprocess.run(
        renderer_worker_command(
            renderer_environment,
            Path(__file__),
            [*worker_args, "--render-worker"],
        ),
        cwd=PROJECT_ROOT,
        env=renderer_worker_environment(renderer_environment),
        check=False,
    )
    if renderer.returncode != 0:
        print(
            f"Stair checkpoint render worker failed with code {renderer.returncode}",
            file=sys.stderr,
        )
    return renderer.returncode


if __name__ == "__main__":
    raise SystemExit(main())
