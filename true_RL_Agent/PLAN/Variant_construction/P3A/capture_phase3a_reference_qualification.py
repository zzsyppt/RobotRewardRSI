"""Run and seal the full C1 reference train plus phased qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase3a_reference_qualification.v1"
)


def _normalized_text_hash(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _captured_head(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def _training_loop_elapsed_s(run_root: Path, session_id: str) -> int | None:
    stdout_path = (
        run_root
        / "runner"
        / session_id
        / "train"
        / "runner_process"
        / "stdout.log"
    )
    if not stdout_path.is_file():
        return None
    matches = re.findall(
        r"Time elapsed:\s*(\d{2}):(\d{2}):(\d{2})",
        stdout_path.read_text(encoding="utf-8", errors="replace"),
    )
    if not matches:
        return None
    hours, minutes, seconds = (int(value) for value in matches[-1])
    return hours * 3600 + minutes * 60 + seconds


def _load_and_verify_calibration(
    *,
    project_root: Path,
    path: Path,
    manifest: Any,
    config: Any,
) -> dict[str, Any]:
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.json_contract import load_strict_json_object

    payload = load_strict_json_object(
        path,
        context="Phase 3A calibration receipt",
        require_canonical=False,
        trailing_newline=True,
    )
    declared_hash = payload.get("receipt_hash")
    if declared_hash != canonical_json_hash(
        {key: value for key, value in payload.items() if key != "receipt_hash"}
    ):
        raise ValueError("Phase 3A calibration receipt hash mismatch")
    qualification = payload.get("terrain_qualification")
    if (
        payload.get("status") != "pass"
        or not isinstance(qualification, Mapping)
        or qualification.get("terrain_profile_status") != "pass"
        or qualification.get("calibration_protocol_hash")
        != config.irregular_calibration_protocol.protocol_hash
        or payload.get("resolved_cell_manifest_hash") != manifest.manifest_hash
        or payload.get("resolved_cell_task_protocol_hash")
        != manifest.resolved_cell_task_protocol_hash
    ):
        raise ValueError("Phase 3A calibration does not qualify the active cell")
    source_hashes = payload.get("product_source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("Phase 3A calibration source closure is missing")
    for relative_path, expected in source_hashes.items():
        if (
            not isinstance(relative_path, str)
            or not isinstance(expected, str)
            or _normalized_text_hash(project_root / relative_path) != expected
        ):
            raise ValueError("Phase 3A calibration source closure drifted")
    return payload


def _required_gate_statuses(public_result: Any) -> dict[str, str]:
    metrics = public_result.metrics
    acceptance = metrics.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("C1 qualified result has no acceptance evidence")
    required = acceptance.get("required_pass_gates")
    if not isinstance(required, list) or not required:
        raise ValueError("C1 qualified result has no required gates")
    statuses: dict[str, str] = {}
    for gate_name in required:
        gate = metrics.get(gate_name)
        if not isinstance(gate_name, str) or not isinstance(gate, Mapping):
            raise ValueError("C1 qualified result misses a required gate")
        status = gate.get("status")
        if status not in {"pass", "fail"}:
            raise ValueError("C1 qualified result gate status is invalid")
        statuses[gate_name] = str(status)
    return statuses


def _source_hashes(project_root: Path) -> dict[str, str]:
    paths = (
        "configs/robots/unitree_g1_37dof_isaaclab_v2.json",
        "scripts/isaaclab_stair_rsl_rl_train.py",
        "scripts/isaaclab_stair_rsl_rl_eval.py",
        "src/true_rl_agent/robots/defaults.py",
        "src/true_rl_agent/tasks/stair_climbing/config.py",
        "src/true_rl_agent/tasks/stair_climbing/irregular_training.py",
        "src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py",
        "src/true_rl_agent/tasks/stair_climbing/ordered_summary.py",
        "src/true_rl_agent/tasks/stair_climbing/phased_adapter.py",
        "src/true_rl_agent/tasks/stair_climbing/phased_evaluation.py",
        "src/true_rl_agent/tasks/stair_climbing/public_projection.py",
        "src/true_rl_agent/tasks/stair_climbing/reference_qualification.py",
        "src/true_rl_agent/tasks/stair_climbing/reference_program.py",
        "src/true_rl_agent/tasks/stair_climbing/reference_programs/g1_irregular_v1/training_code/components.py",
        "src/true_rl_agent/tasks/stair_climbing/reference_programs/official_g1_rough_v1/training_code/components.py",
        "src/true_rl_agent/tasks/stair_climbing/reward_program.py",
        "src/true_rl_agent/tasks/stair_climbing/reward_source_semantics.py",
        "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
        "src/true_rl_agent/tasks/stair_climbing/vnext_reward_contract.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_irregular_terrain.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
        "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
        "configs/tasks/stair_g1_37dof_irregular_evaluation_suite_v1.json",
        "configs/tasks/stair_g1_37dof_irregular_training_profile_v1.json",
        "configs/tasks/stair_g1_37dof_irregular_acceptance_protocol_v1.json",
        "configs/tasks/stair_g1_37dof_irregular_reference_gate_v1.json",
        "configs/tasks/stair_g1_37dof_irregular_calibration_protocol_v1.json",
        "PLAN/Variant_construction/P3A/capture_phase3a_reference_qualification.py",
    )
    return {path: _normalized_text_hash(project_root / path) for path in paths}


def _resolve(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    from true_rl_agent.composition.base import (
        TaskCompositionContext,
        resolve_task_composition,
    )
    from true_rl_agent.tasks.stair_climbing.composition import (
        StairClimbingCompositionProvider,
    )
    from true_rl_agent.tasks.stair_climbing.config import (
        STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
    )

    provider = StairClimbingCompositionProvider()
    values = dict(provider.cli_defaults())
    values.update(
        {
            "cell_variant_id": STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
            "execution_device": args.device,
            "start_mode": "reference",
        }
    )
    composition = resolve_task_composition(
        provider=provider,
        values=values,
        context=TaskCompositionContext(
            project_root=args.project_root,
            output_dir=args.run_root,
        ),
    )
    manifest = composition.resolved_cell_manifest
    reference = composition.reference_program_artifact
    if manifest is None or reference is None:
        raise ValueError("C1 reference composition is incomplete")
    return composition, manifest, reference


def _capture(args: argparse.Namespace) -> dict[str, Any] | None:
    from true_rl_agent.agent_core.ledger import primary_ledger_epoch_hash
    from true_rl_agent.agent_core.trial_receipts import TrialReceiptJournal
    from true_rl_agent.backends.isaac_lab_execution_environment import (
        build_current_isaac_lab_execution_environment_manifest,
    )
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.evaluation.phases import EvaluationPhase
    from true_rl_agent.json_contract import canonical_json_bytes
    from true_rl_agent.runner.checkpoint import file_sha256
    from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
    from true_rl_agent.runner.job_argv import (
        validate_eval_job_argv,
        validate_train_job_argv,
    )

    composition, manifest, reference = _resolve(args)
    config = composition.task_plugin.config
    calibration = _load_and_verify_calibration(
        project_root=args.project_root,
        path=args.calibration_receipt,
        manifest=manifest,
        config=config,
    )
    backend_environment = build_current_isaac_lab_execution_environment_manifest()
    backend = composition.backend_adapter
    evaluation_binding = backend.evaluation_binding_for(
        composition.task_plugin.task_spec
    )
    jobs = backend.materialize_reference_jobs(
        session_id=args.session_id,
        task_spec=composition.task_plugin.task_spec,
        reference=reference,
        artifact_root=args.run_root / "reference_program_artifacts",
        policy_initialization=composition.policy_initialization,
        evaluation_binding=evaluation_binding,
        backend_execution_environment_manifest_hash=(
            backend_environment.manifest_hash
        ),
    )
    validate_train_job_argv(jobs.train_spec, jobs.train_spec.metadata["cli_argv"])
    validate_eval_job_argv(jobs.eval_spec, jobs.eval_spec.metadata["cli_argv"])
    planned_identity = {
        "schema_version": "true_rl_agent.phase3a_reference_commitment.v1",
        "session_id": args.session_id,
        "resolved_cell_manifest_hash": manifest.manifest_hash,
        "calibration_receipt_hash": calibration["receipt_hash"],
        "reference_manifest_hash": reference.manifest.manifest_hash,
        "train_spec_hash": canonical_json_hash(jobs.train_spec.to_public_dict()),
        "eval_plan_hash": canonical_json_hash(jobs.eval_spec.to_public_dict()),
        "backend_execution_environment_manifest_hash": (
            backend_environment.manifest_hash
        ),
    }
    commitment_hash = canonical_json_hash(planned_identity)
    if args.mode == "preflight":
        payload = {
            "status": "pass",
            "mode": "preflight",
            "commitment_hash": commitment_hash,
            "planned_identity": planned_identity,
            "budget_steps": jobs.train_spec.budget_steps,
            "train_num_envs": config.train_num_envs,
            "train_steps_per_iteration": config.train_steps_per_iteration,
            "train_max_iterations": config.train_max_iterations,
            "eval_num_envs": config.eval_num_envs,
            "phase_seed_counts": {
                phase.value: config.ordered_evaluation_suite.seed_count(
                    phase.value
                )
                for phase in EvaluationPhase
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return None

    if args.output.exists():
        raise FileExistsError("Phase 3A reference receipt path must be fresh")
    journal = TrialReceiptJournal(
        args.run_root / "reference_receipts",
        contract={
            "schema_version": "true_rl_agent.trial_execution_contract.v1",
            "round_id": args.session_id,
            "session_genesis_hash": commitment_hash,
            "ledger_epoch_hash": primary_ledger_epoch_hash(commitment_hash),
            "purpose": "phase3a_pre_genesis_reference_qualification",
            "planned_identity": planned_identity,
        },
    )
    journal.bind_contract()
    loaded_train = journal.load_train(jobs.train_spec)
    train_wall_time_s: float | None = None
    if loaded_train is None:
        print("phase3a reference training=started", flush=True)
        started = time.perf_counter()
        train_result = composition.runner.run_train(jobs.train_spec)
        train_wall_time_s = time.perf_counter() - started
        train_ref = journal.publish_train(jobs.train_spec, train_result)
        print("phase3a reference training=completed", flush=True)
    else:
        train_result, train_ref = loaded_train
        print("phase3a reference training=reused", flush=True)
    checkpoint_path = Path(str(train_result.checkpoint.metadata["local_path"]))
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != train_result.checkpoint.checksum:
        raise ValueError("C1 reference checkpoint bytes differ from sealed result")
    eval_plan = bind_eval_spec_to_checkpoint(jobs.eval_spec, train_result.checkpoint)
    phase_results = []
    phase_refs = {}
    for phase in EvaluationPhase:
        phase_spec = backend.materialize_eval_phase(
            eval_plan=eval_plan,
            phase=phase,
        )
        loaded = journal.load_eval_phase(phase_spec)
        if loaded is None:
            print(f"phase3a reference evaluation {phase.value}=started", flush=True)
            result = composition.runner.run_eval_phase(phase_spec)
            ref = journal.publish_eval_phase(phase_spec, result)
            print(f"phase3a reference evaluation {phase.value}=completed", flush=True)
        else:
            result, ref = loaded
            print(f"phase3a reference evaluation {phase.value}=reused", flush=True)
        phase_results.append(result)
        phase_refs[phase] = ref.to_dict()
    adapter = composition.task_plugin.phased_evaluation_adapter()
    assembly = adapter.assemble_reference_full_summary(
        eval_plan=eval_plan,
        phase_results=phase_results,
        phase_receipt_refs=phase_refs,
    )
    raw_public_result = composition.runner.project_assembled_eval_summary(
        eval_plan,
        assembly,
    )
    qualification = composition.task_plugin.qualify_reference_result(
        public_result=raw_public_result,
        train_summary=train_result.summary,
    )
    required_gate_statuses = _required_gate_statuses(qualification.public_result)
    metrics = qualification.public_result.metrics
    sealed_gate = metrics.get("sealed_validation_gate")
    sealed_status = (
        sealed_gate.get("status") if isinstance(sealed_gate, Mapping) else None
    )
    formal_training = train_result.summary.get("formal_training_executed") is True
    cell_loop_ready = (
        qualification.passed
        and qualification.public_result.status == "pass"
        and formal_training
        and sealed_status == "pass"
        and all(status == "pass" for status in required_gate_statuses.values())
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "captured_repository_head": _captured_head(args.project_root),
        "head_identity_semantics": "provenance only; normalized source bytes bind execution",
        "session_id": args.session_id,
        "planned_identity": planned_identity,
        "commitment_hash": commitment_hash,
        "cell_variant_id": manifest.cell_variant_id,
        "resolved_cell_manifest": manifest.to_dict(),
        "calibration_receipt": {
            "receipt_hash": calibration["receipt_hash"],
            "calibration_protocol_hash": calibration["calibration_protocol_hash"],
            "terrain_qualification_result_hash": calibration[
                "terrain_qualification"
            ]["result_hash"],
        },
        "reference_program": reference.manifest.to_public_dict(),
        "train_receipt_ref": train_ref.to_dict(),
        "phase_receipt_refs": {
            phase.value: phase_refs[phase] for phase in EvaluationPhase
        },
        "evaluation_assembly": assembly.to_private_dict(),
        "checkpoint": {
            "path": str(checkpoint_path.resolve(strict=True)),
            "checksum": checkpoint_hash,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "execution_cost": {
            "train_wall_time_s": train_wall_time_s,
            "train_wall_time_semantics": (
                "measured_this_invocation"
                if train_wall_time_s is not None
                else "train_receipt_reused_not_remeasured"
            ),
            "train_loop_elapsed_s": _training_loop_elapsed_s(
                args.run_root,
                args.session_id,
            ),
            "train_loop_elapsed_semantics": (
                "last_rsl_rl_time_elapsed_line"
            ),
            "evaluation": assembly.summary.get("evaluation_execution_cost"),
        },
        "qualification_gate": dict(qualification.gate),
        "qualification_private_evidence": dict(qualification.private_evidence),
        "qualified_public_result": qualification.public_result.to_public_dict(),
        "required_gate_statuses": required_gate_statuses,
        "sealed_validation_status": sealed_status,
        "formal_training_executed": formal_training,
        "cell_loop_ready": cell_loop_ready,
        "product_source_hash_mode": "normalized_lf_sha256",
        "product_source_hashes": _source_hashes(args.project_root),
        "status": "pass" if cell_loop_ready else "fail",
    }
    receipt["receipt_hash"] = canonical_json_hash(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(receipt, trailing_newline=True))
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "cell_loop_ready": cell_loop_ready,
                "receipt_hash": receipt["receipt_hash"],
                "qualification_gate": dict(qualification.gate),
                "required_gate_statuses": required_gate_statuses,
                "sealed_validation_status": sealed_status,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--calibration-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve(strict=True)
    args.run_root = args.run_root.resolve(strict=False)
    args.calibration_receipt = args.calibration_receipt.resolve(strict=True)
    args.output = args.output.resolve(strict=False)
    receipt = _capture(args)
    return 0 if receipt is None or receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
