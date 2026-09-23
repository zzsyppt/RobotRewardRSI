"""Qualify or independently reload one task × robot reference Genesis."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from true_rl_agent.agent_core.session_genesis import (
    SESSION_GENESIS_FILENAME,
    prepare_session_genesis,
    session_genesis_identity_hash,
)
from true_rl_agent.backends.isaac_lab_execution_environment import IsaacLabExecutionEnvironmentManifest
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from true_rl_agent.composition.cli import (
    add_task_runtime_arguments,
    add_task_selection_argument,
    select_task_composition_provider,
)
from true_rl_agent.composition.defaults import build_application_composition_registry
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object
from true_rl_agent.reference_artifact import load_reference_program_artifact, REFERENCE_ARTIFACT_MANIFEST_FILENAME
from true_rl_agent.runner.base import Runner
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.job_argv import (
    validate_eval_job_argv,
    validate_train_job_argv,
)
from true_rl_agent.runner.process_handle import isolate_orchestrator_process_group
from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory


class _LoadOnlyRunner(Runner):
    def run_train(self, train_spec):
        raise RuntimeError("load-only reference validation attempted training")

    def run_eval(self, eval_spec):
        raise RuntimeError("load-only reference validation attempted evaluation")

    def run_eval_phase(self, eval_spec):
        raise RuntimeError("load-only reference validation attempted phased evaluation")

    def run_baseline_eval(self, eval_spec):
        raise RuntimeError("load-only reference validation attempted baseline evaluation")


def _preflight_jobs(composition, *, output, session_id, execution_environment=None):
    backend = composition.backend_adapter
    binding = backend.resolve_session_evaluation_binding(
        session_id=session_id, session_dir=output,
        task_spec=composition.task_plugin.task_spec, allow_create=True)
    jobs = backend.materialize_reference_jobs(
        session_id=session_id, task_spec=composition.task_plugin.task_spec,
        reference=composition.reference_program_artifact,
        artifact_root=output / "reference_program_artifacts",
        policy_initialization=composition.policy_initialization, evaluation_binding=binding,
        backend_execution_environment_manifest_hash=(execution_environment["manifest_hash"] if execution_environment else None))
    validate_train_job_argv(jobs.train_spec, jobs.train_spec.metadata["cli_argv"])
    validate_eval_job_argv(jobs.eval_spec, jobs.eval_spec.metadata["cli_argv"])
    if execution_environment is not None:
        _publish_json(output / "backend_execution_environment.json", execution_environment)
        _publish_json(output / "reference_train_job.json", _private_train_job(jobs.train_spec))
    return {"train_job_spec_hash": canonical_json_hash(jobs.train_spec.to_public_dict()),
            "evaluation_plan_hash": canonical_json_hash(jobs.eval_spec.to_public_dict()),
            "evaluation_phases": [phase.value for phase in EvaluationPhase]}



IMPORT_FILENAME = "completed_reference_training_import.json"


def _private_train_job(spec):
    return json.loads(canonical_json_bytes({"spec": spec.to_public_dict(), "metadata": dict(spec.metadata)}))


def _publish_json(path, value):
    if path.exists():
        if load_strict_json_object(path) != value:
            raise ValueError("existing reference evidence differs: " + str(path))
    else:
        create_once_bytes(path, canonical_json_bytes(value, trailing_newline=True))


def _file_ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def _verify_import_receipt(receipt):
    if receipt.get("schema_version") != "true_rl_agent.completed_reference_training_import.v1":
        raise ValueError("unsupported completed-training import receipt")
    required = {"execution_contract", "started", "resolved_cell", "summary", "checkpoint",
                "process", "execution_environment", "reference_manifest", "reference_components"}
    files = receipt.get("source_files", {})
    if not required <= set(files):
        raise ValueError("completed-training import source evidence is incomplete")
    for evidence in files.values():
        if _file_ref(evidence["path"]) != evidence:
            raise ValueError("completed-training import source bytes changed: " + evidence["path"])


def _load_completed_training(args, composition, expected_contract):
    """Recover the original train job without writing inside its source output."""
    source = args.source_output_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("continuation output must be disjoint from the source output")
    if (source / SESSION_GENESIS_FILENAME).exists() or (source / "reference_eval_artifact.json").exists():
        raise ValueError("source already has reference evaluation or Genesis")
    runner_dir = source / "runner" / args.session_id
    if any(p.is_file() for d in runner_dir.glob("eval*") for p in d.rglob("*")):
        raise ValueError("source already contains evaluation execution evidence")
    receipt_dir = source / "reference_program_artifacts" / "evaluation_receipts"
    if receipt_dir.exists() and any(receipt_dir.rglob("*")):
        raise ValueError("source already contains evaluation receipts")
    contract_path = source / "reference_qualification_execution.json"
    original_contract = load_strict_json_object(contract_path)
    if original_contract != expected_contract:
        raise ValueError("source qualification scientific or execution contract differs")
    started_path = source / "reference_qualification_started.json"
    started = load_strict_json_object(started_path)
    if started.get("contract_hash") != canonical_json_hash(original_contract) or started.get("execution_device") != args.execution_device:
        raise ValueError("source qualification admission differs")
    process_path = args.prior_process_receipt.resolve(strict=True)
    process = load_strict_json_object(process_path)
    argv = process.get("argv")
    if not isinstance(argv, list) or process.get("argv_hash") != canonical_json_hash(argv):
        raise ValueError("source process argv evidence differs")
    original_args = parse_args(argv[2:])
    original_output = (Path(process["cwd"]) / original_args.output_dir).resolve()
    if (original_output != source or original_args.mode != "qualify"
            or original_args.session_id != args.session_id or original_args.task_id != args.task_id
            or original_args.cell_variant_id != args.cell_variant_id):
        raise ValueError("source process belongs to another qualification")
    spent, cap = process.get("elapsed_wall_s"), process.get("wall_timeout_s")
    if (isinstance(spent, bool) or not isinstance(spent, (int, float)) or not math.isfinite(spent)
            or isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap)
            or not 0 <= spent < cap or process.get("wrapper_retries") != 0
            or not isinstance(process.get("returncode"), int) or "ended_at_unix" not in process):
        raise ValueError("source process has no completed, unexhausted wall budget")
    environment_path = args.source_execution_environment.resolve(strict=True)
    environment = IsaacLabExecutionEnvironmentManifest.from_mapping(load_strict_json_object(environment_path)).to_dict()
    original = resolve_task_composition(
        provider=original_args._provider, values={**original_args._provider.cli_defaults(), **vars(original_args)},
        context=TaskCompositionContext(project_root=Path(process["cwd"]), output_dir=source, read_only=True))
    if session_genesis_identity_hash(original.genesis_identity_snapshot()) != expected_contract["genesis_identity_hash"]:
        raise ValueError("source composition scientific identity differs")
    program = original.reference_program_artifact
    artifacts = list((source / "reference_program_artifacts" / program.manifest.reference_id).glob("*/" + REFERENCE_ARTIFACT_MANIFEST_FILENAME))
    if len(artifacts) != 1:
        raise ValueError("source must contain exactly one verified reference artifact")
    artifact = load_reference_program_artifact(artifacts[0].parent, expected_program=program)
    binding = original.backend_adapter.resolve_session_evaluation_binding(
        session_id=args.session_id, session_dir=source, task_spec=original.task_plugin.task_spec, allow_create=False)
    # This existing backend seam only constructs argv/specs; the source artifact
    # was loaded above, so no materializer creates temporary source-side files.
    jobs = original.backend_adapter._materialize_program_jobs(
        run_id=args.session_id, task_spec=original.task_plugin.task_spec,
        policy_initialization=original.policy_initialization, evaluation_binding=binding,
        candidate=None, reference=artifact, expected_initial_policy_state_hash=None,
        execution_environment_manifest_hash=environment["manifest_hash"])
    train_spec = jobs.train_spec
    persisted = source / "reference_train_job.json"
    if persisted.exists() and load_strict_json_object(persisted) != _private_train_job(train_spec):
        raise ValueError("persisted source train job differs from reconstructed identity")
    train_result = composition.runner.recover_completed_train(train_spec)
    if train_result.summary.get("formal_training_executed") is not True:
        raise ValueError("source training has not completed its formal budget")
    source_files = {
        "execution_contract": _file_ref(contract_path), "started": _file_ref(started_path),
        "resolved_cell": _file_ref(source / "resolved_cell_manifest.json"),
        "summary": _file_ref(train_spec.metadata["summary_path"]),
        "checkpoint": _file_ref(train_result.checkpoint.metadata["local_path"]),
        "process": _file_ref(process_path), "execution_environment": _file_ref(environment_path),
        "reference_manifest": _file_ref(artifacts[0]),
        "reference_components": _file_ref(artifact.path + "/training_code/components.py"),
    }
    if source_files["checkpoint"]["sha256"] != train_result.checkpoint.checksum:
        raise ValueError("source checkpoint changed during import")
    completion = train_result.summary.get("runner_training_completion")
    if completion is not None and completion["source_summary_sha256"] != source_files["summary"]["sha256"]:
        raise ValueError("source summary changed during import")
    if persisted.exists():
        source_files["train_job"] = _file_ref(persisted)
    receipt = {"schema_version": "true_rl_agent.completed_reference_training_import.v1",
        "source_output_dir": str(source), "session_id": args.session_id,
        "genesis_identity_hash": expected_contract["genesis_identity_hash"],
        "source_execution_environment_hash": environment["manifest_hash"],
        "train_job": _private_train_job(train_spec),
        "train_job_source": "persisted" if persisted.exists() else "reconstructed_from_verified_original_configuration",
        "normalized_train_result_hash": canonical_json_hash(train_result.to_public_dict()),
        "qualification_wall_cap_s": cap, "prior_qualification_wall_s": spent,
        "remaining_qualification_wall_s": cap - spent, "source_files": source_files}
    _verify_import_receipt(receipt)
    return (train_spec, train_result), receipt


def parse_args(argv=None):
    registry = build_application_composition_registry()
    provider = select_task_composition_provider(argv, registry)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_task_selection_argument(parser, registry=registry, provider=provider)
    add_task_runtime_arguments(parser, provider.cli_defaults())
    provider.add_cli_arguments(parser)
    parser.set_defaults(task_id=None, cell_variant_id=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "qualify", "resume-evaluation", "load-only"), required=True)
    parser.add_argument("--source-output-dir", type=Path)
    parser.add_argument("--source-execution-environment", type=Path)
    parser.add_argument("--prior-process-receipt", type=Path)
    args = parser.parse_args(argv)
    sources = (args.source_output_dir, args.source_execution_environment, args.prior_process_receipt)
    if args.mode == "resume-evaluation" and any(value is None for value in sources):
        parser.error("resume-evaluation requires source output, source execution environment and prior process receipt")
    if args.mode != "resume-evaluation" and any(value is not None for value in sources):
        parser.error("source arguments are only valid for resume-evaluation; load-only uses the stored import receipt")
    if args.task_id is None or args.cell_variant_id is None:
        parser.error("reference qualification requires explicit task and cell")
    args._provider = provider
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve()
    read_only = args.mode == "load-only"
    if args.mode == "resume-evaluation" and output.exists() and any(output.iterdir()):
        raise ValueError("resume-evaluation requires a new clean output directory")
    composition = resolve_task_composition(
        provider=args._provider, values={**args._provider.cli_defaults(), **vars(args)},
        context=TaskCompositionContext(project_root=PROJECT_ROOT, output_dir=output, read_only=read_only))
    if composition.reference_program_artifact is None:
        raise ValueError("selected cell has no reference program")
    phased = composition.task_plugin.phased_evaluation_adapter()
    if phased is None:
        raise ValueError("reference qualification requires phased evaluation")
    snapshot = composition.genesis_identity_snapshot()
    workload = phased.generation_workload(cell_manifest=composition.resolved_cell_manifest)
    seed_units = (workload.public_dev_seed_units_per_job
                  + workload.public_accept_seed_units_per_job
                  + workload.sealed_validation_seed_units_per_job)
    contract = {
        "schema_version": "true_rl_agent.reference_qualification_execution.v1",
        "session_id": args.session_id,
        "genesis_identity_hash": session_genesis_identity_hash(snapshot),
        "workload": workload.to_dict(),
        "train_jobs": 1, "eval_jobs": len(EvaluationPhase),
        "infrastructure_retries": 0,
        "runner_timeout_s_per_job": composition.runner.timeout_s,
        "max_reference_eval_seed_units": seed_units,
        "max_reference_eval_control_steps": seed_units * workload.max_eval_control_steps_per_seed_unit,
    }
    completed_training = None
    import_receipt = None
    import_path = output / IMPORT_FILENAME
    if args.mode == "resume-evaluation":
        completed_training, import_receipt = _load_completed_training(args, composition, contract)
    elif read_only and import_path.exists():
        import_receipt = load_strict_json_object(import_path)
        _verify_import_receipt(import_receipt)
    if import_receipt is not None:
        if (import_receipt["session_id"] != args.session_id
                or import_receipt["genesis_identity_hash"] != contract["genesis_identity_hash"]):
            raise ValueError("completed-training import identity differs")
        environment_path = output / "backend_execution_environment.json"
        if read_only:
            eval_environment = IsaacLabExecutionEnvironmentManifest.from_mapping(load_strict_json_object(environment_path)).to_dict()
        else:
            eval_environment = dict(composition.backend_adapter.resolve_session_execution_environment(
                session_id=args.session_id, session_dir=output, allow_create=True))
            _publish_json(environment_path, eval_environment)
        contract.update(schema_version="true_rl_agent.reference_qualification_execution.v2",
            train_jobs=0, reused_train_jobs=1,
            evaluation_execution_environment_manifest_hash=eval_environment["manifest_hash"],
            completed_training_import_hash=canonical_json_hash(import_receipt),
            **{key: import_receipt[key] for key in ("qualification_wall_cap_s",
                "prior_qualification_wall_s", "remaining_qualification_wall_s")})
        if not read_only:
            _publish_json(import_path, import_receipt)
    contract_path = output / "reference_qualification_execution.json"
    if contract_path.exists():
        if load_strict_json_object(contract_path) != contract:
            raise ValueError("reference execution contract differs from the existing qualification")
    elif read_only:
        raise FileNotFoundError(contract_path)
    else:
        create_once_bytes(contract_path, canonical_json_bytes(contract, trailing_newline=True))
    contract_hash = canonical_json_hash(contract)
    if args.mode == "preflight":
        jobs = _preflight_jobs(composition, output=output, session_id=args.session_id)
        print(json.dumps({"status": "prepared", "contract_hash": contract_hash,
                          "cell_variant_id": composition.resolved_cell_manifest.cell_variant_id,
                          **jobs}))
        return 0

    def prepare(runner, *, require_existing):
        return prepare_session_genesis(
            output_dir=output, session_id=args.session_id, task_plugin=composition.task_plugin,
            backend_adapter=composition.backend_adapter, runner=runner,
            policy_initialization=composition.policy_initialization, genesis_identity_snapshot=snapshot,
            reference_program_artifact=composition.reference_program_artifact,
            reference_phased_evaluation_adapter=phased,
            require_existing=require_existing,
            bind_backend_execution_environment=not require_existing,
            **({"completed_reference_training": completed_training} if completed_training is not None else {}))

    if read_only or (output / SESSION_GENESIS_FILENAME).exists():
        genesis = prepare(_LoadOnlyRunner(), require_existing=True)
    else:
        _preflight_jobs(composition, output=output, session_id=args.session_id)
        if (output / "reference_qualification_started.json").exists():
            raise ValueError("reference qualification already attempted; infrastructure retries are disabled")
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            raise ValueError("reference qualification requires CUDA_VISIBLE_DEVICES unset to match the leased host GPU")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        isolate_orchestrator_process_group()
        inventory = discover_nvidia_gpu_inventory()
        gpu = next((item for item in inventory if item.execution_device == args.execution_device), None)
        if gpu is None:
            raise ValueError("reference execution device is absent from NVIDIA inventory")
        manager = GpuLeaseManager(lease_root=args.lease_root.resolve(),
            devices={item.execution_device: item.identity for item in inventory})
        holder = PreGenesisGpuLeaseHolder(session_id=args.session_id,
            genesis_identity_hash=contract["genesis_identity_hash"], session_contract_hash=contract_hash,
            lane_id="reference", invocation_segment_id="qualification", logical_job_id="reference-qualification")
        # Direct reference Genesis must remain importable. Its jobs keep their
        # host CUDA ordinal; the CLI holds the physical lease for all four jobs.
        with manager.acquire(gpu.execution_device, holder=holder) as lease:
            environment = composition.backend_adapter.resolve_session_execution_environment(
                session_id=args.session_id, session_dir=output, allow_create=True)
            if completed_training is None:
                _preflight_jobs(composition, output=output, session_id=args.session_id, execution_environment=environment)
            else:
                _publish_json(output / "backend_execution_environment.json", dict(environment))
                _verify_import_receipt(import_receipt)
            create_once_bytes(output / "reference_qualification_started.json", canonical_json_bytes({
                "contract_hash": contract_hash, "resource_binding_hash": lease.resource_binding_hash,
                "execution_device": gpu.execution_device,
            }, trailing_newline=True))
            genesis = prepare(composition.runner, require_existing=False)
            if import_receipt is not None:
                _verify_import_receipt(import_receipt)
    print(json.dumps({"status": "pass", "mode": args.mode,
                      "cell_variant_id": composition.resolved_cell_manifest.cell_variant_id,
                      "session_genesis_hash": genesis.session_genesis_hash,
                      "reference": genesis.initial_current_best.to_public_dict(),
                      "provider_requests": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
