from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.tasks.stair_climbing import evaluation_runtime  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_EVAL_TRANSITION_EXTRAS_KEY,
    STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
)


class StairEvaluationRuntimePipelineTests(unittest.TestCase):
    def test_recorder_schema_flows_through_all_splits_and_seven_bins(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
        )
        protocol = config.protocol_v1
        resolver = config.seed_resolver()
        assignment = resolver.private_bin_assignment()
        env = _RecorderSchemaEnv(_recorder_payload(protocol, assignment))

        with (
            patch.object(evaluation_runtime, "_set_runner_seed"),
            patch.object(evaluation_runtime, "_assert_finite_tree"),
        ):
            summaries = evaluation_runtime.run_all_stair_splits(
                env=env,
                inference_policy=lambda observation: observation,
                protocol=protocol,
                seed_resolver=resolver,
            )

        expected_episodes = {
            "public_dev": 168,
            "public_accept": 280,
            "sealed_validation": 392,
        }
        self.assertEqual(set(summaries), set(expected_episodes))
        for split, total in expected_episodes.items():
            with self.subTest(split=split):
                summary = summaries[split]
                self.assertTrue(summary["complete_seven_bin_coverage"])
                self.assertEqual(summary["bin_count"], 7)
                self.assertEqual(summary["total_episode_count"], total)
                self.assertEqual(
                    summary["primary_outcome_counts"]["timeout"],
                    total,
                )
                self.assertEqual(summary["classification_warning_count"], 0)
                self.assertEqual(summary["score_stats"]["mean"], 0.0)
        self.assertEqual(env.reset_count, 15)
        self.assertEqual(env.step_count, 15)


class _RecorderSchemaEnv:
    step_dt = 0.02

    def __init__(self, payload: dict[str, object]) -> None:
        self.unwrapped = self
        self.payload = payload
        self.reset_count = 0
        self.step_count = 0

    def reset(self, *, seed: int):
        self.reset_count += 1
        return {"policy": seed}, {}

    def step(self, actions: object):
        self.step_count += 1
        done = self.payload["done_mask"]
        return (
            actions,
            0.0,
            done,
            {STAIR_EVAL_TRANSITION_EXTRAS_KEY: self.payload},
        )


def _recorder_payload(protocol, assignment: tuple[str, ...]) -> dict[str, object]:
    payload = protocol.to_dict()
    bins = {
        str(item["bin_id"]): (index, float(item["height_m"]))
        for index, item in enumerate(protocol.eval_bins)
    }
    frozen_origins = payload["terrain"]["evaluation_grid_origins_m"]
    origins = [list(frozen_origins[bins[bin_id][0]]) for bin_id in assignment]
    terrain_types = [bins[bin_id][0] for bin_id in assignment]
    root_pos = [
        [origin[0], origin[1], origin[2] + 0.8]
        for origin in origins
    ]
    root_quat = [[1.0, 0.0, 0.0, 0.0] for _ in assignment]
    foot_pos = [
        [
            [origin[0], origin[1] - 0.1, origin[2] + 0.05],
            [origin[0], origin[1] + 0.1, origin[2] + 0.05],
        ]
        for origin in origins
    ]
    zeros_by_foot = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for _ in assignment]
    contact_forces = [
        [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]]
        for _ in assignment
    ]
    manager_to_reason = payload["phase0_probe"][
        "terminal_reason_by_manager_term"
    ]
    term_order = tuple(manager_to_reason)
    timeout_terms = {
        name for name, reason in manager_to_reason.items() if reason == "timeout"
    }
    if len(timeout_terms) != 1:
        raise AssertionError("test fixture requires one frozen timeout term")
    term_masks = {
        name: _Tensor([name in timeout_terms for _ in assignment], dtype="bool")
        for name in term_order
    }
    done = _Tensor([True for _ in assignment], dtype="bool")
    return {
        "schema_version": STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
        "root_pos_w": _Tensor(root_pos),
        "root_quat_w": _Tensor(root_quat),
        "foot_pos_w": _Tensor(foot_pos),
        "foot_lin_vel_w": _Tensor(zeros_by_foot),
        "foot_contact_forces_w": _Tensor(contact_forces),
        "base_contact_forces_w": _Tensor(
            [[[0.0, 0.0, 0.0]] for _ in assignment]
        ),
        "episode_length": _Tensor([1 for _ in assignment], dtype="int64"),
        "terrain_env_origins_w": _Tensor(origins),
        "terrain_types": _Tensor(terrain_types, dtype="int64"),
        "termination_term_order": term_order,
        "termination_masks": term_masks,
        "done_mask": done,
    }


class _FakeScalar:
    def __init__(self, value: bool) -> None:
        self.value = bool(value)

    def item(self) -> bool:
        return self.value


class _Tensor:
    def __init__(self, values: object, *, dtype: str = "float32") -> None:
        self.values = copy.deepcopy(values)
        self.dtype = dtype
        self.shape = _shape(self.values)

    def detach(self) -> "_Tensor":
        return self

    def clone(self) -> "_Tensor":
        return _Tensor(self.values, dtype=self.dtype)

    def cpu(self) -> "_Tensor":
        return self

    def tolist(self) -> object:
        return copy.deepcopy(self.values)

    def bool(self) -> "_Tensor":
        return _Tensor(_map(self.values, bool), dtype="bool")

    def isfinite(self) -> "_Tensor":
        return _Tensor(
            _map(self.values, lambda value: math.isfinite(float(value))),
            dtype="bool",
        )

    def all(self) -> _FakeScalar:
        return _FakeScalar(all(bool(value) for value in _flatten(self.values)))

    def __or__(self, other: "_Tensor") -> "_Tensor":
        return _Tensor(
            _map_binary(self.values, other.values, lambda left, right: left or right),
            dtype="bool",
        )

    def __eq__(self, other: object) -> "_Tensor":  # type: ignore[override]
        if not isinstance(other, _Tensor):
            return _Tensor(_map(self.values, lambda value: False), dtype="bool")
        return _Tensor(
            _map_binary(self.values, other.values, lambda left, right: left == right),
            dtype="bool",
        )


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child = _shape(value[0])
    if any(_shape(item) != child for item in value):
        raise ValueError("ragged tensor fixture")
    return (len(value), *child)


def _map(value: object, operation):
    if isinstance(value, list):
        return [_map(item, operation) for item in value]
    return operation(value)


def _map_binary(left: object, right: object, operation):
    if isinstance(left, list) and isinstance(right, list):
        return [
            _map_binary(a, b, operation)
            for a, b in zip(left, right, strict=True)
        ]
    return operation(left, right)


def _flatten(value: object):
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


if __name__ == "__main__":
    unittest.main()
