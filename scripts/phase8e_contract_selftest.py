#!/usr/bin/env python3
"""Self-test for Phase 8E-2 callable component contracts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.components import DiagnosticTerm, RewardInputs, RewardTerm, ScheduleTerm, TrainingComponents  # noqa: E402
from humanoid_agent.code_sandbox.loader import load_training_components  # noqa: E402
from humanoid_agent.cold_start_lab.recipe_adapter import (  # noqa: E402
    ComponentContractError,
    MAX_DIAGNOSTIC_TERMS,
    build_reward_inputs,
    build_synthetic_reward_inputs,
    evaluate_diagnostic_terms,
    evaluate_reward_terms,
    fast_validate_components,
    precompute_schedule_table,
    validate_training_components,
)
from humanoid_agent.cold_start_lab.training_adapter import reward_inputs_contract  # noqa: E402


CALLABLE_CANDIDATE = PROJECT_ROOT / "examples" / "cold_start_candidates" / "declarative_curriculum_candidate"


class Vector:
    """Tiny tensor-like vector for local contract tests without torch."""

    def __init__(self, values: list[float]):
        self.values = [float(item) for item in values]
        self.shape = (len(self.values),)
        self.dtype = "float64-test-vector"

    def __add__(self, other: Any) -> "Vector":
        return _binary(self, other, lambda left, right: left + right)

    def __radd__(self, other: Any) -> "Vector":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Vector":
        return _binary(self, other, lambda left, right: left - right)

    def __rsub__(self, other: Any) -> "Vector":
        return _binary(self, other, lambda left, right: right - left)

    def __mul__(self, other: Any) -> "Vector":
        return _binary(self, other, lambda left, right: left * right)

    def __rmul__(self, other: Any) -> "Vector":
        return self.__mul__(other)

    def tolist(self) -> list[float]:
        return list(self.values)


class FakeShapeNestedOutput:
    """Object that lies through .shape but materializes to a nested vector."""

    shape = (2,)
    dtype = "fake-shape"

    def detach(self) -> "FakeShapeNestedOutput":
        return self

    def cpu(self) -> "FakeShapeNestedOutput":
        return self

    def tolist(self) -> list[list[float]]:
        return [[1.0], [1.0]]


def _binary(left: Vector, right: Any, op) -> Vector:
    if isinstance(right, Vector):
        return Vector([op(a, b) for a, b in zip(left.values, right.values, strict=True)])
    return Vector([op(a, float(right)) for a in left.values])


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect_contract_error(fn, message: str) -> None:
    try:
        fn()
    except ComponentContractError:
        return
    raise AssertionError(message)


def _inputs(num_envs: int = 4) -> RewardInputs:
    base = {
        "base_height": Vector([0.8] * num_envs),
        "roll": Vector([0.1] * num_envs),
        "pitch": Vector([0.2] * num_envs),
        "base_pos_xy": [[0.0, 0.0] for _ in range(num_envs)],
        "base_lin_vel": Vector([0.0] * num_envs),
        "base_ang_vel": Vector([0.0] * num_envs),
        "projected_gravity": [[0.0, 0.0, -1.0] for _ in range(num_envs)],
        "commanded_base_vel": [[0.0, 0.0, 0.0] for _ in range(num_envs)],
        "joint_pos": Vector([0.0] * num_envs),
        "joint_vel": Vector([0.0] * num_envs),
        "joint_pos_error": Vector([0.0] * num_envs),
        "last_action": Vector([0.0] * num_envs),
        "prev_action": Vector([0.0] * num_envs),
        "action_delta": Vector([0.0] * num_envs),
        "push_active": Vector([0.0] * num_envs),
        "push_vec_public": Vector([0.0] * num_envs),
        "progress_fraction": Vector([0.5] * num_envs),
    }
    return build_reward_inputs(base)


def _bad_reward(_inp: RewardInputs) -> list[float]:
    return [0.0, math.inf]


def _bad_shape(_inp: RewardInputs) -> list[list[float]]:
    return [[0.0], [1.0]]


def _faked_shape(_inp: RewardInputs) -> FakeShapeNestedOutput:
    return FakeShapeNestedOutput()


def _bad_diagnostic(_inp: RewardInputs) -> dict[str, str]:
    return {"unsafe_text": "not allowed"}


def _empty_diagnostic(_inp: RewardInputs) -> dict[str, float]:
    return {}


def _too_many_diagnostics(_inp: RewardInputs) -> dict[str, float]:
    return {f"field_{index}": float(index) for index in range(1500)}


def _diagnostic_batch(start: int):
    def _fn(_inp: RewardInputs) -> dict[str, float]:
        return {f"field_{start + index}": float(index) for index in range(128)}

    return _fn


def main() -> int:
    field_names = set(RewardInputs.field_names())
    expected_expanded_fields = {
        "base_pos_xy",
        "projected_gravity",
        "commanded_base_vel",
        "prev_action",
        "action_delta",
        "joint_pos_error",
    }
    _assert(expected_expanded_fields.issubset(field_names), f"expanded RewardInputs fields missing: {field_names}")

    manifest, components = load_training_components(
        CALLABLE_CANDIDATE,
        expected_schema_version="humanoid_agent.cold_start.candidate.v1",
        allow_llm_authored=True,
    )
    _assert(manifest.candidate_id == "phase8e_declarative_curriculum_candidate", "unexpected callable candidate id")

    public_summary = validate_training_components(components)
    public_text = json.dumps(public_summary, ensure_ascii=False, sort_keys=True)
    _assert("function" not in public_text.lower(), "public summary must not expose function reprs")
    _assert(public_summary["counts"]["reward_terms"] == 1, "callable candidate should expose one reward term")
    _assert(public_summary["counts"]["schedules"] == 1, "callable candidate should expose one schedule")
    _assert(public_summary["counts"]["diagnostics"] == 1, "callable candidate should expose one diagnostic term")

    schedule = precompute_schedule_table(
        components,
        progress_points=[0.0, 0.25, 0.5, 0.75, 1.0],
        allowed_keys=("push_magnitude_scale",),
    )
    values = [row["values"]["delayed_push_ramp"]["push_magnitude_scale"] for row in schedule["rows"]]
    _assert(values == [0.0, 0.0, 0.5, 1.0, 1.0], f"unexpected schedule table: {values}")

    reward_summaries = evaluate_reward_terms(components, _inputs(), num_envs=4)
    _assert(len(reward_summaries) == 1, "one reward summary expected")
    _assert(reward_summaries[0].shape == (4,), "reward output shape must be guarded")
    diagnostic_summaries = evaluate_diagnostic_terms(components, _inputs(), num_envs=4)
    _assert(len(diagnostic_summaries) == 1, "one diagnostic summary expected")
    _assert("height_margin" in diagnostic_summaries[0]["fields"], "diagnostic vector summary missing")
    contract = reward_inputs_contract()
    synthetic_inputs = build_synthetic_reward_inputs(contract, num_envs=3, num_joints=2, num_actions=4, prefer_torch=False)
    _assert(tuple(synthetic_inputs.base_lin_vel[0]) == (0.0, 0.0, 0.0), f"base_lin_vel synthetic shape mismatch: {synthetic_inputs.base_lin_vel}")
    _assert(len(synthetic_inputs.joint_pos) == 3 and len(synthetic_inputs.joint_pos[0]) == 2, f"joint_pos synthetic shape mismatch: {synthetic_inputs.joint_pos}")
    _assert(len(synthetic_inputs.last_action) == 3 and len(synthetic_inputs.last_action[0]) == 4, f"last_action synthetic shape mismatch: {synthetic_inputs.last_action}")
    fast_ok = fast_validate_components(
        components,
        contract=contract,
        num_envs=4,
        num_joints=2,
        num_actions=4,
    )
    _assert(fast_ok["status"] == "pass", f"callable candidate should pass fast synthetic validation: {fast_ok}")
    tensor_backend = fast_ok["synthetic_inputs"].get("tensor_backend", {})
    _assert(tensor_backend.get("backend") in {"torch", "synthetic"}, f"fast validation must report tensor backend: {fast_ok}")
    _assert(tensor_backend.get("prefer_torch") is True, f"fast validation backend must record prefer_torch: {tensor_backend}")
    fast_bad_shape = fast_validate_components(
        TrainingComponents(name="bad", reward_terms=[RewardTerm("bad_shape", fn=_bad_shape)]),
        contract=contract,
        num_envs=2,
        num_joints=2,
        num_actions=2,
        prefer_torch=False,
    )
    _assert(fast_bad_shape["status"] == "fail", f"bad reward shape must fail fast validation: {fast_bad_shape}")
    _assert(
        fast_bad_shape["synthetic_inputs"]["tensor_backend"].get("backend") == "synthetic"
        and fast_bad_shape["synthetic_inputs"]["tensor_backend"].get("prefer_torch") is False,
        f"prefer_torch=False must be visible in fast validation backend summary: {fast_bad_shape}",
    )
    _assert(
        any(issue.get("component_name") == "bad_shape" and issue.get("kind") == "reward" for issue in fast_bad_shape["issues"]),
        f"fast validation must report component-specific reward issue: {fast_bad_shape}",
    )

    _expect_contract_error(
        lambda: build_reward_inputs({"base_height": Vector([1.0])}),
        "missing RewardInputs fields must fail",
    )
    _expect_contract_error(
        lambda: validate_training_components(TrainingComponents(name="bad", reward_terms=[{"name": "legacy"}])),
        "legacy dict reward component must fail the Phase 8E-2 callable contract",
    )
    _expect_contract_error(
        lambda: validate_training_components(TrainingComponents(name="bad", reward_terms=[RewardTerm("bad", fn=42)])),
        "non-callable reward fn must fail",
    )
    _expect_contract_error(
        lambda: validate_training_components(TrainingComponents(name="bad", diagnostics=[{"name": "legacy"}])),
        "legacy dict diagnostic component must fail the Phase 8E callable contract",
    )
    max_diagnostic_terms_ok = TrainingComponents(
        name="max_diag_ok",
        diagnostics=[DiagnosticTerm(f"empty_diag_{index}", fn=_empty_diagnostic) for index in range(MAX_DIAGNOSTIC_TERMS)],
    )
    max_diagnostic_summary = validate_training_components(max_diagnostic_terms_ok)
    _assert(
        max_diagnostic_summary["counts"]["diagnostics"] == MAX_DIAGNOSTIC_TERMS,
        f"max_diagnostic_terms legal boundary must remain usable: {max_diagnostic_summary}",
    )
    too_many_diagnostic_terms = TrainingComponents(
        name="bad",
        diagnostics=[DiagnosticTerm(f"empty_diag_{index}", fn=_empty_diagnostic) for index in range(1500)],
    )
    _expect_contract_error(
        lambda: validate_training_components(too_many_diagnostic_terms),
        "too many DiagnosticTerm components must fail structure validation",
    )
    fast_too_many_diagnostic_terms = fast_validate_components(
        too_many_diagnostic_terms,
        contract=contract,
        num_envs=2,
        num_joints=2,
        num_actions=2,
        prefer_torch=False,
    )
    _assert(
        fast_too_many_diagnostic_terms["status"] == "fail"
        and any("max_diagnostic_terms" in issue.get("message", "") for issue in fast_too_many_diagnostic_terms["issues"]),
        f"fast validation must reject too many DiagnosticTerm components: {fast_too_many_diagnostic_terms}",
    )
    _expect_contract_error(
        lambda: evaluate_diagnostic_terms(
            TrainingComponents(name="bad", diagnostics=[DiagnosticTerm("bad_diag", fn=_bad_diagnostic)]),
            _inputs(2),
            num_envs=2,
        ),
        "text diagnostic payload must fail",
    )
    _expect_contract_error(
        lambda: evaluate_diagnostic_terms(
            TrainingComponents(name="bad", diagnostics=[DiagnosticTerm("too_many_diag_fields", fn=_too_many_diagnostics)]),
            _inputs(2),
            num_envs=2,
        ),
        "diagnostic payloads with too many fields must fail",
    )
    fast_too_many_diagnostics = fast_validate_components(
        TrainingComponents(name="bad", diagnostics=[DiagnosticTerm("too_many_diag_fields", fn=_too_many_diagnostics)]),
        contract=contract,
        num_envs=2,
        num_joints=2,
        num_actions=2,
        prefer_torch=False,
    )
    _assert(
        fast_too_many_diagnostics["status"] == "fail"
        and any(issue.get("code") == "diagnostic_term_invalid" for issue in fast_too_many_diagnostics["issues"]),
        f"fast validation must reject oversized diagnostic payloads: {fast_too_many_diagnostics}",
    )
    total_too_many_diagnostics = TrainingComponents(
        name="bad",
        diagnostics=[DiagnosticTerm(f"diag_batch_{index}", fn=_diagnostic_batch(index * 128)) for index in range(5)],
    )
    _expect_contract_error(
        lambda: evaluate_diagnostic_terms(total_too_many_diagnostics, _inputs(2), num_envs=2),
        "total diagnostic field count must fail",
    )
    fast_total_too_many_diagnostics = fast_validate_components(
        total_too_many_diagnostics,
        contract=contract,
        num_envs=2,
        num_joints=2,
        num_actions=2,
        prefer_torch=False,
    )
    _assert(
        fast_total_too_many_diagnostics["status"] == "fail"
        and any("max_total_diagnostic_fields" in issue.get("message", "") for issue in fast_total_too_many_diagnostics["issues"]),
        f"fast validation must reject excessive total diagnostic fields: {fast_total_too_many_diagnostics}",
    )
    _expect_contract_error(
        lambda: validate_training_components(
            TrainingComponents(name="bad", schedules=[ScheduleTerm("bad_schedule", fn=lambda _p: {}, allowed_keys=("hidden_eval_rate",))])
        ),
        "forbidden schedule key must fail",
    )
    _expect_contract_error(
        lambda: precompute_schedule_table(
            TrainingComponents(
                name="bad",
                schedules=[ScheduleTerm("bad_schedule", fn=lambda _p: {"push_magnitude_scale": "large"}, allowed_keys=("push_magnitude_scale",))],
            ),
            progress_points=[0.5],
            allowed_keys=("push_magnitude_scale",),
        ),
        "non-numeric schedule values must fail",
    )
    _expect_contract_error(
        lambda: evaluate_reward_terms(TrainingComponents(name="bad", reward_terms=[RewardTerm("bad_reward", fn=_bad_reward)]), _inputs(2), num_envs=2),
        "non-finite reward output must fail",
    )
    _expect_contract_error(
        lambda: evaluate_reward_terms(TrainingComponents(name="bad", reward_terms=[RewardTerm("bad_shape", fn=_bad_shape)]), _inputs(2), num_envs=2),
        "wrong reward output shape must fail",
    )
    _expect_contract_error(
        lambda: evaluate_reward_terms(
            TrainingComponents(name="bad", reward_terms=[RewardTerm("faked_shape", fn=_faked_shape)]),
            _inputs(2),
            num_envs=2,
        ),
        "faked .shape reward output must fail against materialized shape",
    )

    print(json.dumps({"status": "pass", "test": "phase8e_contract_selftest"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
