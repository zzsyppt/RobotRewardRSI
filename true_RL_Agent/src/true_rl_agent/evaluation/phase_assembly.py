"""Identity, receipt binding and cost assembly shared by phased task evaluators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import (
    EvaluationPhase, EvaluationPhaseReceiptBinding, EvaluationPhaseResult,
)
from true_rl_agent.runner.specs import EvalJobSpec, EvalPhaseJobSpec


def bind_phase_receipts(
    *,
    eval_plan: EvalJobSpec,
    by_phase: Mapping[EvaluationPhase, EvaluationPhaseResult],
    phase_receipt_refs: Mapping[EvaluationPhase | str, Mapping[str, Any]],
) -> tuple[EvaluationPhaseReceiptBinding, ...]:
    receipt_refs: dict[EvaluationPhase, Mapping[str, Any]] = {}
    if not isinstance(phase_receipt_refs, Mapping):
        raise TypeError("phase receipt refs must be an object")
    for raw_phase, raw_ref in phase_receipt_refs.items():
        phase = EvaluationPhase.parse(raw_phase)
        if phase in receipt_refs:
            raise ValueError("phase receipt refs contain a duplicate phase")
        if not isinstance(raw_ref, Mapping):
            raise TypeError("phase receipt ref must be an object")
        receipt_refs[phase] = raw_ref
    if set(receipt_refs) != set(EvaluationPhase):
        raise ValueError("phase assembly requires one receipt ref per phase")

    bindings: list[EvaluationPhaseReceiptBinding] = []
    for phase in EvaluationPhase:
        raw_ref = receipt_refs[phase]
        if set(raw_ref) != {
            "schema_version",
            "kind",
            "path",
            "checksum",
            "size_bytes",
        } or raw_ref.get("kind") != "eval_phase":
            raise ValueError("phase receipt ref contract mismatch")
        result = by_phase[phase]
        phase_spec = EvalPhaseJobSpec(
            job_id=result.job_id,
            evaluation_plan=eval_plan,
            phase=phase,
        )
        bindings.append(
            EvaluationPhaseReceiptBinding(
                phase=phase,
                job_id=result.job_id,
                path=raw_ref.get("path"),
                checksum=raw_ref.get("checksum"),
                size_bytes=raw_ref.get("size_bytes"),
                ref_schema_version=raw_ref.get("schema_version"),
                job_spec_hash=canonical_json_hash(phase_spec.to_public_dict()),
                phase_result_hash=canonical_json_hash(result.to_private_dict()),
                resource_binding_hash=result.resource_binding_hash,
            )
        )
    return tuple(bindings)



def require_phase_result_matches_plan(
    result: EvaluationPhaseResult,
    eval_plan: EvalJobSpec,
) -> None:
    checkpoint = eval_plan.execution_checkpoint()
    expected = {
        "evaluation_plan_hash": canonical_json_hash(eval_plan.to_public_dict()),
        "task_id": eval_plan.task_id,
        "subject": eval_plan.subject,
        "subject_manifest_hash": eval_plan.subject_manifest_hash,
        "program_source_checksum": eval_plan.program_source_checksum,
        "training_binding_hash": eval_plan.training_binding_hash,
        "capability_profile_hash": eval_plan.capability_profile_hash,
        "checkpoint_hash": checkpoint.checksum,
        "eval_protocol_hash": eval_plan.evaluation_protocol().protocol_hash,
        "evaluation_binding_hash": eval_plan.evaluation_binding.binding_hash,
        "execution_device": eval_plan.execution_device,
    }
    if any(getattr(result, name) != value for name, value in expected.items()):
        raise ValueError("phase result differs from its evaluation plan")



def merge_phase_costs(
    by_phase: Mapping[EvaluationPhase, EvaluationPhaseResult],
) -> dict[str, Any]:
    merged_splits: dict[str, Any] = {}
    common: dict[str, Any] | None = None
    for phase in EvaluationPhase:
        cost = by_phase[phase].execution_cost
        if set(cost) != {"schema_version", "clock_source", "measurement_scope", "splits"}:
            raise ValueError("phase execution cost fields mismatch")
        phase_common = {
            "schema_version": cost["schema_version"],
            "clock_source": cost["clock_source"],
            "measurement_scope": cost["measurement_scope"],
        }
        if common is None:
            common = phase_common
        elif common != phase_common:
            raise ValueError("phase execution cost contracts differ")
        split_costs = cost.get("splits")
        if not isinstance(split_costs, Mapping) or set(split_costs) != {phase.value}:
            raise ValueError("phase execution cost split mismatch")
        merged_splits[phase.value] = dict(split_costs[phase.value])
    assert common is not None
    return {**common, "splits": merged_splits}

