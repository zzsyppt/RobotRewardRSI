from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.isaaclab_stair_checkpoint_video import (  # noqa: E402
    ROLLOUT_SUMMARY_SCHEMA_VERSION,
    TrajectoryCaptureResult,
    _build_parser,
    _load_capture_result,
    _TrajectoryCaptureVecEnv,
    _validate_source_preflight_summary,
    _write_capture_result,
    build_replay_stair_boxes,
    build_video_plan,
    build_visualization_rollout_summary,
    classify_replay_joint_mapping,
)

from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_EVAL_TRANSITION_EXTRAS_KEY,
    STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
)


class _FakeAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--enable_cameras", action="store_true")
        parser.add_argument("--device", default="cuda:0")


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return _FakeTensor(self.value[index])

    def __sub__(self, other):
        return _FakeTensor(
            [left - right for left, right in zip(self.value, other.value, strict=True)]
        )

    def detach(self):
        return self

    def clone(self):
        if isinstance(self.value, list):
            return _FakeTensor(self.value.copy())
        return _FakeTensor(self.value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()

    def item(self):
        return self.value


class _FakeCaptureEnv:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, *args, **kwargs):
        self.reset_calls += 1
        self.step_calls = 0
        return {"policy": "observation"}, {}

    def step(self, actions):
        self.step_calls += 1
        done = [True, False, True] if self.step_calls == 1 else [False] * 3
        episode_steps = (
            [1, 1, 1]
            if self.step_calls == 1
            else [self.step_calls - 1, self.step_calls, self.step_calls - 1]
        )
        delta = 0.5 * self.step_calls
        raw = {
            "schema_version": STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
            "root_pos_w": _FakeTensor(
                [
                    [2.0 + delta, 2.0, 4.0],
                    [3.0 + delta, 3.0, 5.0],
                    [4.0 + delta, 4.0, 6.0],
                ]
            ),
            "root_quat_w": _FakeTensor([[1.0, 0.0, 0.0, 0.0]] * 3),
            "joint_pos": _FakeTensor(
                [[0.1 + delta, 0.2 + delta]] * 3
            ),
            "episode_length": _FakeTensor(episode_steps),
            "terrain_env_origins_w": _FakeTensor([[1.0, 2.0, 3.0]] * 3),
            "done_mask": _FakeTensor(done),
        }
        return (
            {"policy": f"observation-{self.step_calls}"},
            [0.0] * 3,
            done,
            {STAIR_EVAL_TRANSITION_EXTRAS_KEY: raw},
        )


class StairCheckpointVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )

    def _plan(self, env_indices=(0, 3, 7), *, cell_id=None):
        return build_video_plan(
            self.config,
            split="public_dev",
            run_index=1,
            env_indices=None if cell_id is not None else env_indices,
            evaluation_cell_id=cell_id,
            video_fps=15,
            resolution=(1280, 720),
        )

    @staticmethod
    def _capture_wrapper(plan):
        wrapped = _TrajectoryCaptureVecEnv(
            _FakeCaptureEnv(),
            plan=plan,
        )
        return wrapped

    def test_parser_defaults_to_three_stream_slow_motion_capture(self) -> None:
        args = _build_parser(_FakeAppLauncher).parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--checkpoint-sha256",
                f"sha256:{'1' * 64}",
                "--robot-urdf",
                "g1_23dof.urdf",
                "--output-dir",
                "videos/test",
                "--label",
                "round-014-stair-030",
                "--env-indices",
                "32",
                "35",
                "39",
            ]
        )

        self.assertTrue(args.headless)
        self.assertFalse(args.enable_cameras)
        self.assertEqual(args.env_indices, [32, 35, 39])
        self.assertEqual(args.fps, 15)
        self.assertEqual((args.width, args.height), (1280, 720))

    def test_plan_maps_three_envs_to_one_frozen_bin(self) -> None:
        plan = self._plan((32, 35, 39))

        self.assertEqual(plan.env_indices, (32, 35, 39))
        self.assertEqual(plan.evaluation_cell_id, "stair_030__tread_400")
        self.assertEqual(plan.bin_id, "stair_030")
        self.assertEqual(plan.step_height_m, 0.30)
        self.assertEqual(plan.tread_depth_m, 0.40)
        self.assertEqual(plan.episode_steps, 600)
        self.assertEqual(plan.encoded_frame_count, 600)
        self.assertEqual(plan.encoded_duration_s, 40.0)
        self.assertEqual(plan.source_duration_s, 12.0)
        self.assertAlmostEqual(plan.playback_speed, 0.3)
        self.assertEqual(plan.camera_eye[:2], (0.5, -8.0))
        self.assertGreater(plan.camera_eye[2], 8 * plan.step_height_m)
        self.assertEqual(plan.camera_fov_deg, 50.0)
        manifest = plan.to_manifest_dict()
        self.assertEqual(manifest["robot_count"], 3)
        self.assertEqual(manifest["tracked_env_indices"], [32, 35, 39])
        self.assertEqual(
            manifest["camera"]["mode"],
            "static_full_stair_overview",
        )

    def test_plan_covers_first_and_last_frozen_bins(self) -> None:
        first = self._plan((0, 3, 7))
        last = self._plan((272, 275, 279))

        self.assertEqual(first.bin_id, "stair_010")
        self.assertEqual(last.bin_id, "stair_040")
        self.assertEqual(last.tread_depth_m, 0.50)
        self.assertEqual(last.episode_steps, 710)
        self.assertGreater(last.camera_eye[2], first.camera_eye[2])

    def test_plan_selects_three_robots_from_an_active_profile_cell(self) -> None:
        plan = self._plan(cell_id="stair_040__tread_500")

        self.assertEqual(plan.env_indices, (272, 276, 279))
        self.assertEqual(plan.evaluation_cell_id, "stair_040__tread_500")
        self.assertEqual(plan.tread_depth_m, 0.50)
        self.assertEqual(plan.progress_goal_m, 5.15)

    def test_plan_rejects_invalid_three_env_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "only permits public_dev"):
            build_video_plan(
                self.config,
                split="sealed_validation",
                run_index=1,
                env_indices=(0, 3, 7),
                evaluation_cell_id=None,
                video_fps=15,
                resolution=(1600, 900),
            )
        with self.assertRaisesRegex(ValueError, "exactly three"):
            self._plan((0, 3))
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            self._plan((0, 0, 7))
        with self.assertRaisesRegex(ValueError, "outside eval envs"):
            self._plan((272, 275, 280))
        with self.assertRaisesRegex(ValueError, "active evaluation cell"):
            self._plan((7, 8, 9))

    def test_rollout_summary_aggregates_canonical_outcomes_without_reclassification(
        self,
    ) -> None:
        assignment = self.config.evaluation_profile.private_cell_assignment()
        outcomes = tuple(
            StairEpisodeOutcome(
                env_index=index,
                bin_id=cell.bin_id,
                episode_steps=20,
                primary_reason="success" if index % 2 == 0 else "timeout",
                success=index % 2 == 0,
                diagnostic_flags=(
                    ("first_valid_top_arrival_achieved",)
                    if index % 2 == 0
                    else ()
                ),
                max_progress_m=4.0 if index % 2 == 0 else 2.0,
                progress_fraction=1.0 if index % 2 == 0 else 0.5,
                active_terminal_terms=(),
                evaluation_cell_id=cell.cell_id,
            )
            for index, cell in enumerate(assignment)
        )

        summary = build_visualization_rollout_summary(
            self.config,
            split="public_dev",
            outcomes=outcomes,
        )

        self.assertEqual(summary["episode_count"], 280)
        self.assertEqual(summary["episodes_per_cell"], 8)
        self.assertEqual(set(summary["cell_success_rates"].values()), {0.5})
        self.assertEqual(summary["macro_success_rate"], 0.5)
        self.assertFalse(summary["reclassified_by_video_exporter"])

    def test_render_worker_consumes_captured_source_preflight(self) -> None:
        summary = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.execution_source_preflight.v1"
            ),
            "status": "pass",
            "task_protocol_hash": self.config.task_protocol_hash,
            "source_manifest_hash": "b" * 64,
        }

        self.assertEqual(
            _validate_source_preflight_summary(self.config, summary),
            summary,
        )
        with self.assertRaisesRegex(RuntimeError, "drifted"):
            _validate_source_preflight_summary(
                self.config,
                {**summary, "status": "fail"},
            )

    def test_replay_geometry_matches_frozen_stair_layout(self) -> None:
        protocol = self.config.protocol_v1.to_dict()
        boxes = build_replay_stair_boxes(
            protocol,
            step_height_m=0.30,
            tread_depth_m=0.40,
        )

        self.assertEqual(len(boxes), 10)
        self.assertEqual(boxes[0].center, (3.25, 0.0, -0.05))
        for actual, expected in zip(
            boxes[1].center,
            (0.95, 0.0, 0.10),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            boxes[-1].center,
            (5.60, 0.0, 1.15),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(boxes[-1].half_extents, (1.65, 2.0, 1.25))

        wide_tread_boxes = build_replay_stair_boxes(
            protocol,
            step_height_m=0.30,
            tread_depth_m=0.50,
        )
        for actual, expected in zip(
            wide_tread_boxes[1].center,
            (1.0, 0.0, 0.10),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            wide_tread_boxes[-1].center,
            (6.0, 0.0, 1.15),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            wide_tread_boxes[-1].half_extents,
            (1.25, 2.0, 1.25),
        )

    def test_replay_joint_mapping_only_allows_visual_hand_omissions(self) -> None:
        trajectory_names = (
            "left_hip_pitch_joint",
            "torso_joint",
            "right_elbow_joint",
            "left_elbow_pitch_joint",
            "left_zero_joint",
            "right_six_joint",
        )
        mapped, missing = classify_replay_joint_mapping(
            trajectory_names,
            (
                "left_hip_pitch_joint",
                "waist_yaw_joint",
                "right_elbow_joint",
                "left_elbow_joint",
            ),
        )

        self.assertEqual(
            mapped,
            (
                ("left_hip_pitch_joint", "left_hip_pitch_joint"),
                ("torso_joint", "waist_yaw_joint"),
                ("right_elbow_joint", "right_elbow_joint"),
                ("left_elbow_pitch_joint", "left_elbow_joint"),
            ),
        )
        self.assertEqual(missing, ("left_zero_joint", "right_six_joint"))
        with self.assertRaisesRegex(ValueError, "policy-controlled body joints"):
            classify_replay_joint_mapping(
                trajectory_names,
                ("left_hip_pitch_joint", "right_elbow_joint", "left_elbow_joint"),
            )

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") is not None,
        "numpy is unavailable",
    )
    def test_trajectory_capture_continues_after_automatic_reset(self) -> None:
        plan = replace(self._plan((0, 1, 2)), episode_steps=3)
        wrapped = self._capture_wrapper(plan)

        self.assertEqual(wrapped._root_positions, [])
        wrapped.reset(seed=123)
        self.assertEqual(wrapped._root_positions, [])
        wrapped.step([0.0])
        wrapped.step([0.0])
        wrapped.step([0.0])
        self.assertEqual(wrapped.last_observation["policy"], "observation-3")

        (
            root_positions,
            root_quaternions,
            joints,
            terminal_flags,
            episode_start_flags,
            episode_indices,
            episode_control_steps,
            reset_counts,
        ) = wrapped.trajectory_arrays()
        self.assertEqual(root_positions.shape, (3, 3, 3))
        self.assertEqual(root_quaternions.shape, (3, 3, 4))
        self.assertEqual(joints.shape, (3, 3, 2))
        self.assertEqual(root_positions.tolist()[0][0], [1.5, 0.0, 1.0])
        self.assertEqual(terminal_flags.tolist()[0], [True, False, True])
        self.assertEqual(episode_start_flags.tolist()[0], [True, True, True])
        self.assertEqual(episode_start_flags.tolist()[1], [True, False, True])
        self.assertEqual(episode_indices.tolist()[0], [0, 0, 0])
        self.assertEqual(episode_indices.tolist()[1], [1, 0, 1])
        self.assertEqual(episode_control_steps.tolist()[1], [1, 2, 1])
        self.assertEqual(reset_counts, (1, 0, 1))

        wrapped.reset(seed=456)
        self.assertEqual(wrapped.captured_frame_count, 0)
        self.assertEqual(wrapped._terminal_flags, [])

    def test_capture_result_survives_isaac_worker_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            trajectory_path = output_dir / "trajectory.npz"
            trajectory_path.write_bytes(b"trajectory")
            expected = TrajectoryCaptureResult(
                trajectory_path=trajectory_path,
                tracked_outcomes=(
                    {"status": "success"},
                    {"status": "fall"},
                    {"status": "timeout"},
                ),
                captured_frame_count=600,
                reset_counts=(1, 2, 3),
                joint_names=("left_hip_pitch_joint", "torso_joint"),
                rollout_summary={
                    "schema_version": ROLLOUT_SUMMARY_SCHEMA_VERSION,
                    "complete_height_tread_coverage": True,
                    "reclassified_by_video_exporter": False,
                },
                checkpoint_report={
                    "schema_version": "true_rl_agent.checkpoint_safety_report.v1",
                    "status": "pass",
                    "checkpoint_hash": f"sha256:{'a' * 64}",
                    "expected_checkpoint_hash": None,
                    "file_exists": True,
                    "byte_read_status": "pass",
                    "weights_only_load_status": "pass",
                    "checkpoint_hash_match_status": "not_checked",
                    "loader_error_class": None,
                    "loader_error_summary": None,
                    "reason": "checkpoint_safe_to_load",
                },
                source_preflight={
                    "schema_version": (
                        "true_rl_agent.task.stair_climbing."
                        "execution_source_preflight.v1"
                    ),
                    "status": "pass",
                    "task_protocol_hash": self.config.task_protocol_hash,
                    "source_manifest_hash": "b" * 64,
                },
            )

            _write_capture_result(output_dir, expected)
            actual = _load_capture_result(output_dir)

            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
