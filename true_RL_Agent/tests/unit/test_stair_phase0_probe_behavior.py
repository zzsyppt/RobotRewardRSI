from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from true_rl_agent.tasks.stair_climbing.isaaclab_phase0_probe import (
        run_first_episode_probe,
    )
    from true_rl_agent.tasks.stair_climbing.phase0_contract import (
        PHASE0_TRANSITION_EXTRAS_KEY,
        StairPhase0Bundle,
    )


class _FakeScene:
    def __init__(self, robot: Any, env_origins: Any) -> None:
        self.robot = robot
        self.terrain = SimpleNamespace(env_origins=env_origins)

    def __getitem__(self, name: str) -> Any:
        if name != "robot":
            raise KeyError(name)
        return self.robot


class _AutoResetFakeEnv:
    def __init__(
        self,
        raw_env: Any,
        *,
        termination_masks: Mapping[str, Any],
        wrapper_dones: Any,
        terminal_root_pos_w: Any,
    ) -> None:
        self.raw_env = raw_env
        self.termination_masks = dict(termination_masks)
        self.wrapper_dones = wrapper_dones
        self.terminal_root_pos_w = terminal_root_pos_w

    def step(self, actions: Any) -> tuple[dict[str, Any], Any, Any, dict[str, Any]]:
        _ = actions
        transition = {
            "root_pos_w": self.terminal_root_pos_w.clone(),
            "episode_length": torch.ones(1, dtype=torch.long),
            "termination_masks": {
                name: mask.clone()
                for name, mask in self.termination_masks.items()
            },
        }
        self.raw_env.scene["robot"].data.root_pos_w = torch.tensor(
            [[0.0, 0.0, 0.9]],
            dtype=torch.float32,
        )
        return (
            {"policy": torch.zeros((1, 310), dtype=torch.float32)},
            torch.zeros(1, dtype=torch.float32),
            self.wrapper_dones.clone(),
            {PHASE0_TRANSITION_EXTRAS_KEY: transition},
        )


@unittest.skipIf(torch is None, "torch is required for dynamic Phase 0 probe tests")
class StairPhase0ProbeBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        protocol = StairPhase0Bundle.load(PROJECT_ROOT).protocol.to_dict()
        protocol["terrain"]["evaluation_bins"] = [
            protocol["terrain"]["evaluation_bins"][0]
        ]
        protocol["evaluation"]["env_to_bin_mapping"]["envs_per_bin"] = 1
        protocol["evaluation"]["protocol"]["eval_num_envs"] = 1
        protocol["phase0_probe"]["control_steps"] = 1
        protocol["phase0_probe"]["in_distribution_bin_ids"] = [
            protocol["terrain"]["evaluation_bins"][0]["bin_id"]
        ]
        self.protocol = protocol

    def _run(
        self,
        *,
        termination_masks: Mapping[str, Any],
        wrapper_dones: Any,
        terminal_root_pos_w: Any | None = None,
    ) -> dict[str, Any]:
        initial_root = torch.tensor([[0.0, 0.0, 0.9]], dtype=torch.float32)
        robot = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=initial_root.clone())
        )
        raw_env = SimpleNamespace(
            device="cpu",
            num_envs=1,
            scene=_FakeScene(
                robot,
                torch.zeros((1, 3), dtype=torch.float32),
            ),
        )
        terminal_root = (
            terminal_root_pos_w
            if terminal_root_pos_w is not None
            else torch.tensor([[0.8, 0.0, 0.9]], dtype=torch.float32)
        )
        env = _AutoResetFakeEnv(
            raw_env,
            termination_masks=termination_masks,
            wrapper_dones=wrapper_dones,
            terminal_root_pos_w=terminal_root,
        )

        def policy(observation: Any) -> Any:
            _ = observation
            return torch.zeros((1, 37), dtype=torch.float32)

        return run_first_episode_probe(
            env=env,
            raw_env=raw_env,
            policy=policy,
            observation={
                "policy": torch.zeros((1, 310), dtype=torch.float32)
            },
            protocol=copy.deepcopy(self.protocol),
        )

    def test_terminal_transition_uses_pre_reset_state(self) -> None:
        rollout = self._run(
            termination_masks={
                "time_out": torch.tensor([False]),
                "base_contact": torch.tensor([True]),
            },
            wrapper_dones=torch.tensor([True]),
        )
        record = rollout["first_episodes"]["stair_010"][0]
        self.assertAlmostEqual(record["max_progress_m"], 0.8, places=6)
        self.assertTrue(record["reached_first_step"])

    def test_wrapper_done_and_manager_masks_must_match(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "wrapper done mask does not match termination term masks",
        ):
            self._run(
                termination_masks={
                    "time_out": torch.tensor([False]),
                    "base_contact": torch.tensor([False]),
                },
                wrapper_dones=torch.tensor([True]),
            )

    def test_simultaneous_timeout_and_contact_use_declared_priority(self) -> None:
        rollout = self._run(
            termination_masks={
                "time_out": torch.tensor([True]),
                "base_contact": torch.tensor([True]),
            },
            wrapper_dones=torch.tensor([True]),
        )
        record = rollout["first_episodes"]["stair_010"][0]
        self.assertEqual(record["terminal_terms"], ["time_out", "base_contact"])
        self.assertEqual(record["terminal_reason"], "base_contact")
        self.assertFalse(record["success"])


if __name__ == "__main__":
    unittest.main()
