from __future__ import annotations

import os
import subprocess
import sys
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeTracker,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (  # noqa: E402
    build_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.termination_capture import (  # noqa: E402
    build_termination_mask_snapshot,
    is_stair_termination_capture_callable,
)
from true_rl_agent.tasks.stair_climbing.transition_decoder import (  # noqa: E402
    decode_stair_transition_batch,
)

_DYNAMIC_ENABLED = os.environ.get("TRUE_RL_AGENT_ISAAC_DYNAMIC_TEST") == "1"
_ISAAC_CHILD = os.environ.get("TRUE_RL_AGENT_ISAAC_TEST_CHILD") == "1"
_SIMULATION_APP = None
if _DYNAMIC_ENABLED and _ISAAC_CHILD:
    try:
        from isaaclab.app import AppLauncher

        _APP_LAUNCHER = AppLauncher(headless=True)
        _SIMULATION_APP = _APP_LAUNCHER.app
        import torch

        from true_rl_agent.tasks.stair_climbing import (
            isaaclab_reward_oracle as reward_oracle,
        )
        from true_rl_agent.tasks.stair_climbing import (
            isaaclab_runtime_env as isaac_runtime,
        )
    except BaseException:
        traceback.print_exc()
        if _SIMULATION_APP is not None:
            _SIMULATION_APP.close(exit_code=1)
            _SIMULATION_APP = None
        raise
else:
    torch = None
    isaac_runtime = None


def _run_in_isaac_subprocess(test_name: str) -> None:
    # Kit owns process shutdown. Keep it outside the unittest discovery process;
    # the child prints its complete result before closing with that exit status.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), test_name, "-v"],
        cwd=ROOT,
        env={**os.environ, "TRUE_RL_AGENT_ISAAC_TEST_CHILD": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or "\nOK\n" not in output:
        raise AssertionError(
            f"Isaac test child exited {result.returncode}:\n{output[-12000:]}"
        )


@unittest.skipUnless(
    _DYNAMIC_ENABLED,
    "set TRUE_RL_AGENT_ISAAC_DYNAMIC_TEST=1 to run Isaac integration",
)
class StairIsaacLabRuntimeEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        self.protocol = self.config.protocol_v1
        assert self.config.evaluation_semantics is not None
        self.semantics = self.config.evaluation_semantics

    def test_train_and_eval_builders_install_only_the_declared_runtime(self) -> None:
        if not _ISAAC_CHILD:
            _run_in_isaac_subprocess(f"{type(self).__name__}.{self._testMethodName}")
            return
        reward_program = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        ).reward_program
        train_cfg = isaac_runtime.build_stair_train_env_cfg(
            self.protocol.to_dict(),
            seed=11,
            device="cpu",
            reward_program=reward_program,
            runner_owned_training_distribution=(
                self.config.protocol_v2.runner_owned_training_distribution
            ),
        )
        eval_cfg = isaac_runtime.build_stair_eval_env_cfg(
            self.protocol.to_dict(),
            seed=11,
            device="cpu",
        )
        oracle_cfg = reward_oracle.build_stair_reward_oracle_env_cfg(
            self.protocol.to_dict(),
            seed=11,
            device="cpu",
            reward_program=reward_program,
        )

        self.assertIsNone(train_cfg.recorders)
        self.assertEqual(
            tuple(
                train_cfg.scene.terrain.terrain_generator.sub_terrains[
                    "one_way_stairs"
                ].step_height_range
            ),
            (0.05, 0.35),
        )
        runtime_distribution = (
            isaac_runtime.stair_training_distribution_runtime_evidence(
                train_cfg,
                protocol=self.protocol.to_dict(),
                contract=self.config.protocol_v2.runner_owned_training_distribution,
            )
        )
        self.assertEqual(
            runtime_distribution["terrain"]["step_height_range_m"],
            [0.05, 0.35],
        )
        train_cfg.scene.terrain.terrain_generator.sub_terrains[
            "one_way_stairs"
        ].step_height_range = (0.05, 0.20)
        with self.assertRaisesRegex(
            RuntimeError,
            "effective Stair runner-owned training distribution drifted",
        ):
            isaac_runtime.stair_training_distribution_runtime_evidence(
                train_cfg,
                protocol=self.protocol.to_dict(),
                contract=self.config.protocol_v2.runner_owned_training_distribution,
            )
        self.assertIsInstance(
            eval_cfg.recorders,
            isaac_runtime.StairEvalRecorderManagerCfg,
        )
        term_names = tuple(
            self.protocol.to_dict()["phase0_probe"]["terminal_reason_by_manager_term"]
        )
        for term_name in term_names:
            with self.subTest(term=term_name):
                self.assertIs(
                    is_stair_termination_capture_callable(
                        getattr(train_cfg.terminations, term_name).func
                    ),
                    True,
                )
                self.assertIs(
                    is_stair_termination_capture_callable(
                        getattr(eval_cfg.terminations, term_name).func
                    ),
                    True,
                )
                self.assertIs(
                    is_stair_termination_capture_callable(
                        getattr(oracle_cfg.terminations, term_name).func
                    ),
                    True,
                )

    def test_real_recorder_payload_decodes_tracks_and_aggregates(self) -> None:
        if not _ISAAC_CHILD:
            _run_in_isaac_subprocess(f"{type(self).__name__}.{self._testMethodName}")
            return
        public_dev = self.protocol.to_dict()["evaluation"]["splits"]["public_dev"]
        episodes_per_bin = int(public_dev["episodes_per_seed_per_bin"])
        assignment = tuple(
            str(item["bin_id"])
            for item in self.protocol.eval_bins
            for _ in range(episodes_per_bin)
        )
        num_envs = len(assignment)
        protocol_payload = self.protocol.to_dict()
        bins = {
            str(item["bin_id"]): index
            for index, item in enumerate(self.protocol.eval_bins)
        }
        frozen_origins = protocol_payload["terrain"]["evaluation_grid_origins_m"]
        origins = torch.tensor(
            [frozen_origins[bins[bin_id]] for bin_id in assignment],
            dtype=torch.float32,
        )
        terrain_types = torch.tensor(
            [bins[bin_id] for bin_id in assignment],
            dtype=torch.int64,
        )
        root_pos = origins.clone()
        root_pos[:, 2] += 0.8
        root_quat = torch.zeros((num_envs, 4), dtype=torch.float32)
        root_quat[:, 3] = 1.0  # Native Isaac Lab 3 buffers use XYZW.
        foot_pos = origins[:, None, :].repeat(1, 2, 1)
        foot_pos[:, 0, 1] -= 0.1
        foot_pos[:, 1, 1] += 0.1
        foot_pos[:, :, 2] += 0.05
        robot_body_pos = torch.cat((foot_pos, root_pos[:, None, :]), dim=1)
        contact_forces = torch.zeros((num_envs, 3, 3), dtype=torch.float32)
        contact_forces[:, :2, 2] = 100.0
        joint_names = tuple(
            protocol_payload["policy_compatibility"]["robot_schema"]["joint_names"]
        )

        robot_data = SimpleNamespace(
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            root_lin_vel_w=torch.zeros_like(root_pos),
            root_ang_vel_w=torch.zeros_like(root_pos),
            root_lin_vel_b=torch.zeros_like(root_pos),
            root_ang_vel_b=torch.zeros_like(root_pos),
            projected_gravity_b=torch.tensor([0.0, 0.0, -1.0]).repeat(num_envs, 1),
            joint_pos=torch.zeros((num_envs, 37), dtype=torch.float32),
            body_pos_w=robot_body_pos,
            body_lin_vel_w=torch.zeros_like(robot_body_pos),
        )
        contact_data = SimpleNamespace(net_forces_w=contact_forces)
        terrain = SimpleNamespace(
            env_origins=origins,
            terrain_types=terrain_types,
        )
        scene = _Scene(
            robot=_FakeAsset(
                data=robot_data,
                body_ids={
                    "left_ankle_roll_link": 0,
                    "right_ankle_roll_link": 1,
                    "torso_link": 2,
                },
                joint_ids={name: index for index, name in enumerate(joint_names)},
            ),
            contact_forces=_FakeAsset(
                data=contact_data,
                body_ids={
                    "left_ankle_roll_link": 0,
                    "right_ankle_roll_link": 1,
                    "torso_link": 2,
                },
            ),
            terrain=terrain,
        )
        env = SimpleNamespace(
            scene=scene,
            extras={},
            episode_length_buf=torch.ones(num_envs, dtype=torch.int64),
        )

        manager_to_reason = protocol_payload["phase0_probe"][
            "terminal_reason_by_manager_term"
        ]
        term_order = tuple(manager_to_reason)
        term_masks = {
            name: torch.full(
                (num_envs,),
                reason == "timeout",
                dtype=torch.bool,
            )
            for name, reason in manager_to_reason.items()
        }
        done = torch.ones(num_envs, dtype=torch.bool)
        snapshot = build_termination_mask_snapshot(
            term_order=term_order,
            timeout_term_names=tuple(
                name
                for name, reason in manager_to_reason.items()
                if reason == "timeout"
            ),
            term_masks=term_masks,
            done_mask=done,
        )

        recorder = isaac_runtime.StairEvalTransitionRecorder(
            isaac_runtime.StairEvalTransitionRecorderCfg(),
            env,
        )
        with patch.object(
            isaac_runtime,
            "capture_current_termination_masks",
            return_value=snapshot,
        ):
            recorder.record_post_step()

        raw = env.extras[isaac_runtime.STAIR_EVAL_TRANSITION_EXTRAS_KEY]
        self.assertTrue(torch.equal(raw["root_pos_w"], root_pos))
        self.assertTrue(torch.equal(
            raw["root_quat_w"],
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(num_envs, 1),
        ))
        self.assertEqual(tuple(raw["joint_pos"].shape), (num_envs, 37))
        self.assertTrue(torch.equal(raw["terrain_env_origins_w"], origins))
        self.assertTrue(torch.equal(raw["terrain_types"], terrain_types))
        transitions = decode_stair_transition_batch(
            raw,
            wrapper_done_mask=done,
            protocol=self.protocol,
            semantics=self.semantics,
            expected_bin_assignment=assignment,
        )
        outcomes = [
            StairEpisodeTracker(
                self.protocol,
                self.semantics,
                env_index=index,
                bin_id=bin_id,
            ).observe(transitions[index])
            for index, bin_id in enumerate(assignment)
        ]
        self.assertTrue(all(outcome is not None for outcome in outcomes))
        summary = build_stair_split_summary(
            self.protocol,
            semantics=self.semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="public_dev",
            per_seed_outcomes=[outcomes, outcomes, outcomes],
        )
        self.assertEqual(summary["total_episode_count"], 168)
        self.assertEqual(summary["primary_outcome_counts"]["timeout"], 168)
        self.assertEqual(summary["classification_warning_count"], 0)


class _Scene(dict):
    def __init__(self, *, robot, contact_forces, terrain) -> None:
        super().__init__(robot=robot, contact_forces=contact_forces)
        self.terrain = terrain


class _FakeAsset:
    def __init__(self, *, data, body_ids, joint_ids=None) -> None:
        self.data = data
        self._body_ids = dict(body_ids)
        self._joint_ids = {} if joint_ids is None else dict(joint_ids)

    def find_bodies(self, names, *, preserve_order):
        if preserve_order is not True:
            raise AssertionError("test asset requires ordered body resolution")
        return [self._body_ids[name] for name in names], list(names)

    def find_joints(self, names, *, preserve_order):
        if preserve_order is not True:
            raise AssertionError("test asset requires ordered joint resolution")
        return [self._joint_ids[name] for name in names], list(names)


if __name__ == "__main__":
    program = unittest.main(exit=False)
    exit_code = 0 if program.result.wasSuccessful() else 1
    sys.stdout.flush()
    sys.stderr.flush()
    if _SIMULATION_APP is not None:
        _SIMULATION_APP.close(exit_code=exit_code)
    raise SystemExit(exit_code)
