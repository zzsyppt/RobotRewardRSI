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
from true_rl_agent.tasks.stair_climbing.seed_policy import (  # noqa: E402
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.transition_decoder import (  # noqa: E402
    decode_stair_transition_batch,
)


class StairEvaluationRuntimePipelineTests(unittest.TestCase):
    def test_decoder_does_not_create_a_35cm_riser_fall_false_positive(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        assert config.evaluation_semantics is not None
        assignment = config.base_seed_resolver().private_bin_assignment()
        env_index = assignment.index("stair_035")
        payload = _recorder_payload(config.protocol_v1, assignment)
        origin = payload["terrain_env_origins_w"].values[env_index]
        lower_tread_height = 6 * 0.35
        payload["root_pos_w"].values[env_index] = [
            origin[0] + 3.20,
            origin[1],
            origin[2] + lower_tread_height + 0.74,
        ]
        payload["foot_pos_w"].values[env_index] = [
            [origin[0] + 3.20, origin[1] - 0.1, origin[2] + lower_tread_height],
            [origin[0] + 3.22, origin[1] + 0.1, origin[2] + lower_tread_height],
        ]

        transitions = decode_stair_transition_batch(
            payload,
            wrapper_done_mask=payload["done_mask"],
            protocol=config.protocol_v1,
            semantics=config.evaluation_semantics,
            expected_bin_assignment=assignment,
        )

        transition = transitions[env_index]
        self.assertEqual(transition.foot_surface_levels, (7, 7))
        for clearance in transition.foot_surface_clearances_m:
            self.assertAlmostEqual(clearance, -0.35)
        self.assertEqual(transition.foot_upward_forces_n, (100.0, 100.0))
        self.assertAlmostEqual(transition.root_clearance_m, 0.74)

    def test_decoder_keeps_horizontal_collision_separate_from_upward_support(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        assert config.evaluation_semantics is not None
        assignment = config.base_seed_resolver().private_bin_assignment()
        env_index = assignment.index("stair_035")
        payload = _recorder_payload(config.protocol_v1, assignment)
        payload["foot_contact_forces_w"].values[env_index] = [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
        ]

        transition = decode_stair_transition_batch(
            payload,
            wrapper_done_mask=payload["done_mask"],
            protocol=config.protocol_v1,
            semantics=config.evaluation_semantics,
            expected_bin_assignment=assignment,
        )[env_index]

        self.assertEqual(transition.foot_contacts, (True, True))
        self.assertEqual(transition.foot_upward_forces_n, (0.0, 0.0))

    def test_recorder_schema_flows_through_all_splits_and_seven_bins(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        protocol = config.protocol_v1
        assert config.evaluation_semantics is not None
        resolver = config.base_seed_resolver()
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
                semantics=config.evaluation_semantics,
                seed_resolver=resolver,
                seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE,
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

    def test_batched_history_keeps_each_env_first_episode_only(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        protocol = config.protocol_v1
        assert config.evaluation_semantics is not None
        resolver = config.base_seed_resolver()
        assignment = resolver.private_bin_assignment()
        first_done = [index < len(assignment) // 2 for index in range(len(assignment))]
        second_done = [not value for value in first_done]
        first = _payload_with_timeout_mask(
            _recorder_payload(protocol, assignment),
            done_values=first_done,
            episode_steps=[1 for _ in assignment],
        )
        second = _payload_with_timeout_mask(
            _recorder_payload(protocol, assignment),
            done_values=second_done,
            episode_steps=[1 if done else 2 for done in first_done],
        )
        env = _RecorderSequenceEnv([first, second])

        with (
            patch.object(evaluation_runtime, "_set_runner_seed"),
            patch.object(evaluation_runtime, "_assert_finite_tree"),
        ):
            outcomes = evaluation_runtime.run_stair_seed_rollout(
                env=env,
                inference_policy=lambda observation: observation,
                protocol=protocol,
                semantics=config.evaluation_semantics,
                seed_resolver=resolver,
                split="public_accept",
                run_index=1,
                seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE,
            )

        self.assertEqual(env.step_count, 2)
        self.assertEqual(len(outcomes), len(assignment))
        midpoint = len(outcomes) // 2
        self.assertEqual(
            [outcome.episode_steps for outcome in outcomes[:midpoint]],
            [1] * midpoint,
        )
        self.assertEqual(
            [outcome.episode_steps for outcome in outcomes[midpoint:]],
            [2] * (len(outcomes) - midpoint),
        )
        self.assertTrue(
            all(outcome.primary_reason == "timeout" for outcome in outcomes)
        )

    def test_cuda_done_polling_preserves_first_episode_with_one_host_sync(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        protocol = config.protocol_v1
        assert config.evaluation_semantics is not None
        resolver = config.base_seed_resolver()
        assignment = resolver.private_bin_assignment()
        payload = _recorder_payload(protocol, assignment)
        env = _RecorderSequenceEnv([payload] * 64)

        with (
            patch.object(evaluation_runtime, "_set_runner_seed"),
            patch.object(evaluation_runtime, "_assert_finite_tree"),
            patch.object(evaluation_runtime, "_is_cuda_tensor", return_value=True),
            patch.object(
                evaluation_runtime,
                "_all_true",
                wraps=evaluation_runtime._all_true,
            ) as all_true,
        ):
            execution = evaluation_runtime.run_stair_split_execution(
                env=env,
                inference_policy=lambda observation: observation,
                protocol=protocol,
                semantics=config.evaluation_semantics,
                seed_resolver=resolver,
                split="public_accept",
                seed_count=1,
                seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE,
            )
            outcomes = list(execution.per_seed_outcomes[0])

        self.assertEqual(env.step_count, 64)
        self.assertEqual(all_true.call_count, 1)
        self.assertEqual(len(outcomes), len(assignment))
        self.assertTrue(all(outcome.episode_steps == 1 for outcome in outcomes))
        self.assertEqual(execution.vector_env_step_calls, 64)
        self.assertEqual(
            execution.simulated_env_control_steps,
            64 * len(assignment),
        )
        self.assertEqual(
            execution.closed_first_episode_control_steps,
            len(assignment),
        )

    def test_active_profile_runtime_closes_all_280_cell_identities(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        assert config.evaluation_semantics is not None
        resolver = config.seed_resolver()
        cell_assignment = resolver.private_cell_assignment()
        bin_assignment = resolver.private_bin_assignment()
        payload = _recorder_payload(
            config.protocol_v1,
            bin_assignment,
            cell_assignment=cell_assignment,
        )
        env = _RecorderSchemaEnv(payload)

        with (
            patch.object(evaluation_runtime, "_set_runner_seed"),
            patch.object(evaluation_runtime, "_assert_finite_tree"),
        ):
            outcomes = evaluation_runtime.run_stair_seed_rollout(
                env=env,
                inference_policy=lambda observation: observation,
                protocol=config.protocol_v1,
                semantics=config.evaluation_semantics,
                seed_resolver=resolver,
                split="public_accept",
                run_index=1,
                seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE,
            )

        self.assertEqual(len(outcomes), 280)
        self.assertEqual(
            [outcome.evaluation_cell_id for outcome in outcomes],
            [cell.cell_id for cell in cell_assignment],
        )
        self.assertTrue(
            all(outcome.primary_reason == "timeout" for outcome in outcomes)
        )


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


class _RecorderSequenceEnv:
    step_dt = 0.02

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.unwrapped = self
        self.payloads = payloads
        self.step_count = 0
        self._step_index = 0

    def reset(self, *, seed: int):
        self._step_index = 0
        return {"policy": seed}, {}

    def step(self, actions: object):
        payload = self.payloads[self._step_index]
        self._step_index += 1
        self.step_count += 1
        return (
            actions,
            0.0,
            payload["done_mask"],
            {STAIR_EVAL_TRANSITION_EXTRAS_KEY: payload},
        )


def _recorder_payload(
    protocol,
    assignment: tuple[str, ...],
    *,
    cell_assignment=None,
) -> dict[str, object]:
    payload = protocol.to_dict()
    bins = {
        str(item["bin_id"]): (index, float(item["height_m"]))
        for index, item in enumerate(protocol.eval_bins)
    }
    frozen_origins = payload["terrain"]["evaluation_grid_origins_m"]
    if cell_assignment is None:
        origins = [list(frozen_origins[bins[bin_id][0]]) for bin_id in assignment]
        terrain_types = [bins[bin_id][0] for bin_id in assignment]
    else:
        origins = [list(cell.expected_origin_m) for cell in cell_assignment]
        terrain_types = [cell.terrain_type_index for cell in cell_assignment]
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
        "joint_pos": _Tensor([[0.0] * 37 for _ in assignment]),
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


def _payload_with_timeout_mask(
    payload: dict[str, object],
    *,
    done_values: list[bool],
    episode_steps: list[int],
) -> dict[str, object]:
    result = dict(payload)
    term_order = result["termination_term_order"]
    if not isinstance(term_order, tuple):
        raise AssertionError("test fixture requires tuple termination order")
    timeout_name = next(name for name in term_order if name == "time_out")
    result["termination_masks"] = {
        name: _Tensor(
            done_values if name == timeout_name else [False] * len(done_values),
            dtype="bool",
        )
        for name in term_order
    }
    result["done_mask"] = _Tensor(done_values, dtype="bool")
    result["episode_length"] = _Tensor(episode_steps, dtype="int64")
    return result


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
