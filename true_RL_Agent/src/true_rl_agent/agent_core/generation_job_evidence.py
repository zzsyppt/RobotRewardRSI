"""Validate train/evaluation receipts and project their measured work."""
from __future__ import annotations
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
from true_rl_agent.agent_core.generation_cost import GenerationCostContract
from true_rl_agent.agent_core.session_genesis import REFERENCE_EVAL_ARTIFACT_FILENAME, VerifiedSessionGenesis
from true_rl_agent.agent_core.trial_receipts import TrialReceiptRef, load_trial_eval_phase_receipt, load_trial_train_receipt
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.json_contract import load_strict_json_object

def validated_job_cost_artifacts(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
) -> tuple[dict[str, int], dict[str, tuple[int, int]], str]:
    train_steps: dict[str, int] = {}
    eval_costs: dict[str, tuple[int, int]] = {}
    evidence: list[dict[str, Any]] = []
    if genesis.reference_genesis_import_ref is None:
        _append_local_reference_job_cost_artifacts(
            foundation,
            genesis=genesis,
            train_steps=train_steps,
            eval_costs=eval_costs,
            evidence=evidence,
        )
    elif genesis.reference_eval_artifact_path() is None:
        raise ValueError("imported reference Genesis has no verified private artifact")
    for root in (
        foundation.config.output_dir / "parent_public_dev_receipts",
        foundation.config.output_dir / "trial_receipts",
        foundation.config.output_dir
        / "final_confirmation"
        / "phase_receipts",
    ):
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError("trial receipt root is invalid")
        for path in sorted(root.rglob("train.json")):
            ref = trial_receipt_ref(path, kind="train")
            loaded = load_trial_train_receipt(ref)
            _insert_unique_cost(
                train_steps,
                loaded.result.job_id,
                _train_planned_steps(
                    loaded.result.summary,
                    expected_steps=(
                        foundation.cost_contract.scientific_cost_basis.train_steps_per_job
                    ),
                ),
            )
            evidence.append(
                _job_cost_evidence_projection(
                    foundation,
                    path=path,
                    kind="trial-train-receipt",
                    job_identity={"job_id": loaded.result.job_id},
                )
            )
        for path in sorted(root.rglob("eval_phases/*.json")):
            ref = trial_receipt_ref(path, kind="eval_phase")
            loaded = load_trial_eval_phase_receipt(ref)
            _insert_unique_cost(
                eval_costs,
                loaded.result.job_id,
                _execution_cost_totals(loaded.result.execution_cost),
            )
            evidence.append(
                _job_cost_evidence_projection(
                    foundation,
                    path=path,
                    kind="trial-eval-phase-receipt",
                    job_identity={"job_id": loaded.result.job_id},
                )
            )
    evidence.sort(key=lambda item: (item["relative_path"], item["kind"]))
    return train_steps, eval_costs, canonical_json_hash(evidence)



def _append_local_reference_job_cost_artifacts(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    train_steps: dict[str, int],
    eval_costs: dict[str, tuple[int, int]],
    evidence: list[dict[str, Any]],
) -> None:
    """Count only reference work executed inside this session output."""

    private_path = foundation.config.output_dir / REFERENCE_EVAL_ARTIFACT_FILENAME
    private = load_strict_json_object(
        private_path,
        context="reference Genesis private eval artifact",
    )
    train_result = private.get("train_result")
    if not isinstance(train_result, Mapping):
        raise TypeError("reference Genesis train result is invalid")
    train_job_id = str(train_result.get("job_id"))
    training_source = (genesis.artifact.runtime_snapshot or {}).get(
        "execution_placement", {}).get("reference_training_source")
    if training_source is None:
        _insert_unique_cost(
            train_steps,
            train_job_id,
            _train_planned_steps(
                train_result.get("summary"),
                expected_steps=(
                    foundation.cost_contract.scientific_cost_basis.train_steps_per_job
                ),
            ),
        )
    eval_plan = genesis.reference_eval_plan()
    if eval_plan is None:
        raise ValueError("reference Genesis has no eval plan")
    evidence.append(
        _job_cost_evidence_projection(
            foundation,
            path=private_path,
            kind="reference-eval-artifact",
            job_identity={
                "train_job_id": train_job_id,
                "eval_job_id": eval_plan.job_id,
            },
        )
    )
    reference_assembly = private.get("evaluation_assembly")
    if reference_assembly is None:
        summary_path = eval_plan.metadata.get("summary_path")
        if not isinstance(summary_path, str) or not summary_path:
            raise ValueError("reference eval plan has no summary path")
        reference_summary_path = Path(summary_path)
        reference_summary = load_strict_json_object(
            reference_summary_path,
            context="reference evaluation summary",
        )
        if reference_summary.get("job_id") != eval_plan.job_id:
            raise ValueError("reference evaluation summary job identity mismatch")
        _insert_unique_cost(
            eval_costs,
            eval_plan.job_id,
            _execution_cost_totals(
                reference_summary.get("evaluation_execution_cost")
            ),
        )
        evidence.append(
            _job_cost_evidence_projection(
                foundation,
                path=reference_summary_path,
                kind="reference-eval-summary",
                job_identity={"job_id": eval_plan.job_id},
            )
        )
        return
    if not isinstance(reference_assembly, Mapping):
        raise TypeError("reference evaluation assembly is invalid")
    raw_receipts = reference_assembly.get("phase_receipts")
    if not isinstance(raw_receipts, list) or len(raw_receipts) != len(
        EvaluationPhase
    ):
        raise ValueError("reference evaluation phase receipts are incomplete")
    expected_plan_hash = canonical_json_hash(eval_plan.to_public_dict())
    observed_phases: set[EvaluationPhase] = set()
    for raw_binding in raw_receipts:
        if not isinstance(raw_binding, Mapping):
            raise TypeError("reference evaluation phase receipt is invalid")
        phase = EvaluationPhase.parse(raw_binding.get("phase"))
        if phase in observed_phases:
            raise ValueError("reference evaluation phase receipt is duplicated")
        observed_phases.add(phase)
        ref = TrialReceiptRef(
            kind="eval_phase",
            path=str(raw_binding.get("path")),
            checksum=str(raw_binding.get("checksum")),
            size_bytes=raw_binding.get("size_bytes"),
            schema_version=str(raw_binding.get("ref_schema_version")),
        )
        loaded = load_trial_eval_phase_receipt(ref)
        result = loaded.result
        if (
            result.phase is not phase
            or result.job_id != raw_binding.get("job_id")
            or result.evaluation_plan_hash != expected_plan_hash
        ):
            raise ValueError("reference evaluation phase receipt identity mismatch")
        _insert_unique_cost(
            eval_costs,
            result.job_id,
            _execution_cost_totals(result.execution_cost),
        )
        evidence.append(
            _job_cost_evidence_projection(
                foundation,
                path=Path(ref.path),
                kind="reference-eval-phase-receipt",
                job_identity={"job_id": result.job_id},
            )
        )
    if observed_phases != set(EvaluationPhase):
        raise ValueError("reference evaluation phase receipts are incomplete")



def _job_cost_evidence_projection(
    foundation: GenerationSessionFoundation,
    *,
    path: Path,
    kind: str,
    job_identity: Mapping[str, str],
) -> dict[str, Any]:
    output_root = foundation.config.output_dir.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("job cost evidence path is not a regular file")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("job cost evidence escaped the session output") from exc
    raw = resolved.read_bytes()
    if not raw or len(raw) > 64 * 1024 * 1024:
        raise ValueError("job cost evidence size is invalid")
    return {
        "kind": kind,
        "relative_path": relative.as_posix(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "job_identity": dict(job_identity),
    }



def trial_receipt_ref(path: Path, *, kind: str) -> TrialReceiptRef:
    payload = path.read_bytes()
    if not payload:
        raise ValueError("trial receipt is empty")
    return TrialReceiptRef(
        kind=kind,
        path=str(path.resolve()),
        checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )



def _train_planned_steps(summary: Any, *, expected_steps: int) -> int:
    if (
        isinstance(expected_steps, bool)
        or not isinstance(expected_steps, int)
        or expected_steps <= 0
    ):
        raise ValueError("expected train steps must be positive")
    found: list[Mapping[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            accounting = value.get("budget_accounting")
            if isinstance(accounting, Mapping):
                found.append(accounting)
            for item in value.values():
                visit(item)
        elif isinstance(value, list | tuple):
            for item in value:
                visit(item)

    visit(summary)
    if len(found) != 1 or found[0].get("status") != "pass":
        raise ValueError("train result has no unique passing budget accounting")
    steps = found[0].get("planned_env_steps")
    if steps != expected_steps:
        raise ValueError("train result differs from the fixed budget")
    return steps



def _execution_cost_totals(value: Any) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise TypeError("evaluation execution cost is missing")
    splits = value.get("splits")
    if not isinstance(splits, Mapping) or not splits:
        raise ValueError("evaluation execution cost splits are invalid")
    seed_units = 0
    control_steps = 0
    for split in splits.values():
        if not isinstance(split, Mapping):
            raise TypeError("evaluation execution cost split is invalid")
        seeds = split.get("seed_count")
        steps = split.get("simulated_env_control_steps")
        if (
            isinstance(seeds, bool)
            or not isinstance(seeds, int)
            or seeds <= 0
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
        ):
            raise ValueError("evaluation execution cost counters are invalid")
        seed_units += seeds
        control_steps += steps
    return seed_units, control_steps



def eval_seed_units_for_job(
    logical_job_id: str,
    job_kind: str,
    *,
    cost_contract: GenerationCostContract,
) -> int:
    basis = cost_contract.scientific_cost_basis
    if job_kind in {"eval", "baseline_eval"}:
        return basis.reference_eval_seed_units
    suffixes = {
        "-public-dev": basis.seed_units_for_phase("public_dev"),
        "-public-accept": basis.seed_units_for_phase("public_accept"),
        "-sealed-validation": basis.seed_units_for_phase("sealed_validation"),
    }
    matches = [
        value for suffix, value in suffixes.items() if logical_job_id.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError("eval-phase job id has an unknown phase")
    return matches[0]



def _insert_unique_cost(target: dict[str, Any], job_id: str, value: Any) -> None:
    if not job_id or job_id == "None" or job_id in target:
        raise ValueError("job cost artifact identity is duplicated")
    target[job_id] = value
