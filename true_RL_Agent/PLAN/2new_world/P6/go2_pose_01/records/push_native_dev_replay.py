"""Replay one frozen Push reference checkpoint for native public-dev video.

P6/run_owned_probe.py owns the GPU lease. This diagnostic never trains, runs
accept/sealed, creates Genesis, or changes the formal directory's artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import qualify_reference as qualification_cli
from scripts.qualify_reference import _LoadOnlyRunner

from true_rl_agent.agent_core.session_genesis import (
    SESSION_GENESIS_FILENAME,
    prepare_session_genesis,
    session_genesis_identity_hash,
)
from true_rl_agent.agent_core.trial_receipts import (
    TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
    TrialReceiptJournal,
    load_trial_eval_phase_receipt,
)
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.job_argv import validate_eval_job_argv
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.push_recovery.composition import (
    PushRecoveryCompositionProvider,
)

CELLS = ("push.g1_37dof.recovery_v2", "push.go2_12dof.recovery_v2")


def load_composition(args):
    formal = args.formal_dir.resolve(strict=True)
    contract = load_strict_json_object(formal / "reference_qualification_execution.json")
    provider = PushRecoveryCompositionProvider()
    composition = resolve_task_composition(provider=provider,
        values={**provider.cli_defaults(), "cell_variant_id": args.cell,
                "execution_device": args.device, "isaac_python": args.isaac_python,
                "push_protocol_path": getattr(args, "push_protocol_path", None),
                "eval_base_seed": getattr(args, "eval_base_seed", None),
                "reference_program_id": getattr(args, "reference_program_id", None),
                "runner_timeout_s": contract["runner_timeout_s_per_job"]},
        context=TaskCompositionContext(project_root=ROOT, output_dir=formal, read_only=True))
    if args.task_id != composition.task_id:
        raise ValueError("formal task differs from the requested task")
    config = composition.task_plugin.config
    if (config.train_seed, config.budget_steps,
        config.public_seed_count, config.public_accept_seed_count,
        config.sealed_validation_seed_count, config.eval_num_envs) != (42, 49152000, 3, 3, 7, 64):
        raise ValueError("native replay requires the frozen P6 training and evaluation cohort")
    adapter = composition.task_plugin.phased_evaluation_adapter()
    workload = adapter.generation_workload(cell_manifest=composition.resolved_cell_manifest)
    expected = {
        "schema_version": "true_rl_agent.reference_qualification_execution.v1",
        "session_id": contract["session_id"],
        "genesis_identity_hash": session_genesis_identity_hash(composition.genesis_identity_snapshot()),
        "workload": workload.to_dict(), "train_jobs": 1, "eval_jobs": 3,
        "infrastructure_retries": 0,
        "runner_timeout_s_per_job": composition.runner.timeout_s,
        "max_reference_eval_seed_units": 13,
        "max_reference_eval_control_steps": 13 * workload.max_eval_control_steps_per_seed_unit,
    }
    if contract.get("schema_version") == "true_rl_agent.reference_qualification_execution.v2":
        imported = load_strict_json_object(formal / qualification_cli.IMPORT_FILENAME)
        qualification_cli._verify_import_receipt(imported)
        if (imported["session_id"] != contract["session_id"]
            or imported["genesis_identity_hash"] != expected["genesis_identity_hash"]):
            raise ValueError("continuation training import differs from the current scientific identity")
        environment = load_strict_json_object(formal / "backend_execution_environment.json")
        composition.backend_adapter.validate_session_execution_environment(execution_environment=environment)
        expected.update(schema_version="true_rl_agent.reference_qualification_execution.v2",
            train_jobs=0, reused_train_jobs=1,
            completed_training_import_hash=canonical_json_hash(imported),
            evaluation_execution_environment_manifest_hash=environment["manifest_hash"],
            **{key: imported[key] for key in ("qualification_wall_cap_s",
                "prior_qualification_wall_s", "remaining_qualification_wall_s")})
    if contract != expected:
        raise ValueError("formal reference execution contract differs from the current composition")
    started = load_strict_json_object(formal / "reference_qualification_started.json")
    if started["contract_hash"] != canonical_json_hash(contract) or started["execution_device"] != args.device:
        raise ValueError("formal reference admission or device differs")
    return composition, contract


def recover_unqualified_plan(composition, contract, *, formal, output):
    """Reuse production completed-summary recovery, never its execution branch."""
    backend = composition.backend_adapter
    session_id = contract["session_id"]
    imported = None
    if contract["schema_version"] == "true_rl_agent.reference_qualification_execution.v2":
        imported = load_strict_json_object(formal / qualification_cli.IMPORT_FILENAME)
        qualification_cli._verify_import_receipt(imported)
        files = imported["source_files"]
        original_contract = load_strict_json_object(files["execution_contract"]["path"])
        if original_contract["genesis_identity_hash"] != contract["genesis_identity_hash"]:
            raise ValueError("continuation and original training scientific identities differ")
        source_args = qualification_cli.parse_args(load_strict_json_object(files["process"]["path"])["argv"][2:])
        source_args.source_output_dir = Path(imported["source_output_dir"])
        source_args.source_execution_environment = Path(files["execution_environment"]["path"])
        source_args.prior_process_receipt = Path(files["process"]["path"])
        source_args.output_dir = formal
        (_, trained), recovered_import = qualification_cli._load_completed_training(
            source_args, composition, original_contract)
        if recovered_import != imported:
            raise ValueError("retained completed training differs from its continuation import")
        summary_path = Path(files["summary"]["path"])
        environment_hash = contract["evaluation_execution_environment_manifest_hash"]
    else:
        summary_path = Path(backend.config.artifact_root) / session_id / "train/summary.json"
        summary = load_strict_json_object(summary_path)
        environment_hash = summary["backend_execution_environment_verification"]["expected_manifest_hash"]
    jobs = backend.materialize_reference_jobs(session_id=session_id,
        task_spec=composition.task_plugin.task_spec,
        reference=composition.reference_program_artifact,
        artifact_root=output / "recovered_reference_source",
        policy_initialization=composition.policy_initialization,
        evaluation_binding=backend.evaluation_binding_for(composition.task_plugin.task_spec),
        backend_execution_environment_manifest_hash=environment_hash)
    # This exact production recovery branch reads an existing summary and fails
    # if missing/invalid; it cannot launch a process or clear stale artifacts.
    if imported is None:
        trained = composition.runner.recover_completed_train(jobs.train_spec)
    plan = bind_eval_spec_to_checkpoint(jobs.eval_spec, trained.checkpoint)
    commitment = {"schema_version": "true_rl_agent.session_genesis.reference_phase_receipts.v1",
        "session_id": session_id, "genesis_identity_hash": contract["genesis_identity_hash"],
        "reference_manifest_hash": composition.reference_program_artifact.manifest.manifest_hash,
        "evaluation_plan_hash": canonical_json_hash(plan.to_public_dict())}
    journal = TrialReceiptJournal(formal / "reference_program_artifacts/evaluation_receipts",
        contract={"schema_version": TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
            "round_id": session_id, "session_genesis_hash": contract["genesis_identity_hash"],
            "ledger_epoch_hash": canonical_json_hash(commitment),
            "purpose": "session_genesis_reference_phased_evaluation", "commitment": commitment})
    results, refs = [], {}
    for phase in EvaluationPhase:
        spec = backend.materialize_eval_phase(eval_plan=plan, phase=phase)
        loaded = journal.load_eval_phase(spec)
        if loaded is None:
            raise ValueError("unqualified replay requires every completed formal phase: " + phase.value)
        result, ref = loaded
        # Also verify the on-disk journal envelope through the public reader.
        if dict(load_trial_eval_phase_receipt(ref).contract) != dict(journal.contract):
            raise ValueError("formal phase receipt commitment differs")
        results.append(result)
        refs[phase] = ref.to_dict()
    assembly = composition.task_plugin.phased_evaluation_adapter().assemble_reference_full_summary(
        eval_plan=plan, phase_results=results, phase_receipt_refs=refs)
    public = composition.runner.project_assembled_eval_summary(plan, assembly)
    qualification = composition.task_plugin.qualify_reference_result(
        public_result=public, train_summary=trained.summary)
    if qualification.passed:
        raise ValueError("passing reference evidence has no Genesis; resolve that incomplete commit before replay")
    return plan, {"source_kind": "unqualified_reference_diagnostic",
        "qualification_gate": dict(qualification.gate),
        "qualification_evidence": dict(qualification.private_evidence),
        "completed_training_import_hash": canonical_json_hash(imported) if imported is not None else None,
        "train_summary_path": str(summary_path),
        "train_summary_sha256": file_sha256(summary_path),
        "expected_public_dev": expected_public_dev(results[0], refs[EvaluationPhase.PUBLIC_DEV]),
        "formal_phase_receipts": {phase.value: ref for phase, ref in refs.items()}}


def source_plan(composition, contract, *, formal, output):
    if (formal / SESSION_GENESIS_FILENAME).exists():
        # An invalid existing Genesis is an error, never a fallback trigger.
        genesis = prepare_session_genesis(output_dir=formal, session_id=contract["session_id"],
            task_plugin=composition.task_plugin, backend_adapter=composition.backend_adapter,
            runner=_LoadOnlyRunner(), policy_initialization=composition.policy_initialization,
            genesis_identity_snapshot=composition.genesis_identity_snapshot(),
            reference_program_artifact=composition.reference_program_artifact,
            reference_phased_evaluation_adapter=composition.task_plugin.phased_evaluation_adapter(),
            require_existing=True, bind_backend_execution_environment=False)
        dev = genesis.reference_eval_phase_receipt(EvaluationPhase.PUBLIC_DEV)
        if dev is None:
            raise ValueError("qualified reference has no verified public_dev receipt")
        return genesis.reference_eval_plan(), {"source_kind": "qualified_reference_genesis",
            "session_genesis_hash": genesis.session_genesis_hash,
            "expected_public_dev": expected_public_dev(dev.result, dev.receipt_ref.to_dict()),
            "reference_eval_artifact_sha256": file_sha256(genesis.reference_eval_artifact_path())}
    return recover_unqualified_plan(composition, contract, formal=formal, output=output)


def episode_cohort(result):
    if result.phase is not EvaluationPhase.PUBLIC_DEV:
        raise ValueError("native episode comparison requires public_dev")
    seeds = result.task_evidence["seed_summaries"]
    if len(seeds) != 3 or any(len(seed.get("per_episode", ())) != 64 for seed in seeds):
        raise ValueError("formal dev comparison requires all 3 x 64 original episode records")
    episodes = [seed["per_episode"] for seed in seeds]
    if any([row["env_id"] for row in rows] != list(range(64)) for rows in episodes):
        raise ValueError("formal dev episode environment order differs")
    return episodes


def expected_public_dev(result, receipt):
    return {"phase_receipt": receipt, "phase_result_hash": canonical_json_hash(result.to_private_dict()),
            "episodes": episode_cohort(result)}


def compare_dev_replay(expected, result, recordings, *, env_index, base_seed):
    from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed

    actual = episode_cohort(result)
    mismatches = [{"run_index": seed + 1, "env_id": env}
        for seed in range(3) for env in range(64) if actual[seed][env] != expected["episodes"][seed][env]]
    media_match = len(recordings) == 3 and all(
        row.get("run_index") == index and row.get("env_index") == env_index
        and row.get("split") == "public_dev"
        and row.get("seed") == derive_runner_seed(base_seed=base_seed, split_name="public_dev", run_index=index)
        and row.get("episode_outcome") == expected["episodes"][index - 1][env_index]
        for index, row in enumerate(recordings, start=1))
    return {"status": "PASS" if not mismatches and media_match else "FAIL",
        "comparison": "exact recursive equality of every episode field, including push events and recovery",
        "compared_episodes": 192, "mismatched_episodes": mismatches,
        "selected_env_index": env_index, "selected_media_matches_formal_episodes": media_match,
        "formal_phase_receipt": expected["phase_receipt"], "formal_phase_result_hash": expected["phase_result_hash"]}


def video_job(composition, plan, *, output, env_index):
    if not 0 <= env_index < 64:
        raise ValueError("video env index must be in the original 64-environment cohort")
    checkpoint = plan.execution_checkpoint()
    path = Path(checkpoint.metadata["local_path"])
    if file_sha256(path) != checkpoint.checksum:
        raise ValueError("reference checkpoint bytes differ from their bound checksum")
    if (plan.public_seed_count, plan.public_accept_seed_count,
        plan.sealed_validation_seed_count, plan.eval_num_envs) != (3, 3, 7, 64):
        raise ValueError("reference plan differs from the frozen P6 cohort")
    plan = composition.backend_adapter.rematerialize_eval_plan(eval_plan=plan,
        round_id="p6-native-dev-replay", summary_path=(output / "eval/summary.json").as_posix())
    spec = composition.backend_adapter.materialize_eval_phase(eval_plan=plan, phase=EvaluationPhase.PUBLIC_DEV)
    argv = [*spec.metadata["cli_argv"], "--video-dir", str(output / "video"),
            "--video-env-index", str(env_index)]
    spec = replace(spec, metadata={**spec.metadata, "cli_argv": argv, "cli_command": " ".join(argv)})
    validate_eval_job_argv(spec, argv)
    return spec


def _write(output, name, value):
    (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--task-id", choices=("unitree_push_recovery",), required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-env-index", type=int, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--isaac-python", default=sys.executable)
    parser.add_argument("--push-protocol-path", type=Path, default=None)
    parser.add_argument("--eval-base-seed", type=int, default=None)
    from true_rl_agent.tasks.push_recovery.reference_program import PUSH_REFERENCE_IDS
    parser.add_argument("--reference-program-id", choices=PUSH_REFERENCE_IDS, default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    formal, output = args.formal_dir.resolve(strict=True), args.output.resolve()
    if output == formal or output.is_relative_to(formal) or formal.is_relative_to(output):
        raise ValueError("native replay output must be separate from the formal artifact tree")
    if not 0 <= args.video_env_index < 64 or args.timeout <= 0:
        raise ValueError("invalid video environment index or process timeout")
    output.mkdir(parents=True, exist_ok=False)
    report = {"purpose": "P6 same-checkpoint native public-dev diagnostic",
        "mode": args.mode, "formal_dir": str(formal), "cell": args.cell,
        "status": "RUNNING", "started_at_unix": time.time(), "phases_executed": [],
        "train_jobs": 0, "accept_jobs": 0, "sealed_jobs": 0, "provider_requests": 0,
        "genesis_created": False, "formal_evidence_replaced": False, "infrastructure_retries": 0}
    try:
        composition, contract = load_composition(args)
        plan, source = source_plan(composition, contract, formal=formal, output=output)
        expected = source.pop("expected_public_dev")
        _write(output, "expected_public_dev.json", expected)
        report.update(source, contract_hash=canonical_json_hash(contract),
            expected_public_dev_sha256=file_sha256(output / "expected_public_dev.json"),
            checkpoint_sha256=plan.execution_checkpoint().checksum,
            source_evaluation_plan_hash=canonical_json_hash(plan.to_public_dict()))
        spec = video_job(composition, plan, output=output, env_index=args.video_env_index)
        workload = composition.task_plugin.phased_evaluation_adapter().generation_workload(
            cell_manifest=composition.resolved_cell_manifest)
        report.update(budget={"dev_seeds": 3, "envs_per_seed": 64, "episodes": 192,
            "max_env_control_steps": 3 * workload.max_eval_control_steps_per_seed_unit,
            "runner_timeout_s": args.timeout}, selected_env_index=args.video_env_index)
        _write(output, "native_dev_job.json", {"spec": spec.to_public_dict(), "argv": spec.metadata["cli_argv"]})
        if args.mode == "preflight":
            report["status"] = "PREPARED"
            return 0
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            raise ValueError("native replay requires CUDA_VISIBLE_DEVICES unset to preserve the outer lease's host ordinal")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        _write(output, "result.json", report)
        runner = SubprocessRunner(allowed_executables=(args.isaac_python,), cwd=ROOT, timeout_s=args.timeout)
        result = runner.run_eval_phase(spec)
        _write(output, "public_dev_result.json", result.to_private_dict())
        recordings = load_strict_json_object(output / "video/recordings.json")
        rows = recordings["recordings"]
        if (recordings["checkpoint"]["checkpoint_hash"] != spec.execution_checkpoint().checksum
            or recordings["resolved_cell_manifest_hash"] != composition.resolved_cell_manifest.manifest_hash):
            raise ValueError("native recordings checkpoint or cell identity differs")
        if len(rows) != 3 or any(row["env_index"] != args.video_env_index or row["split"] != "public_dev" for row in rows):
            raise ValueError("native recordings differ from the selected formal dev cohort")
        comparison = compare_dev_replay(expected, result, rows, env_index=args.video_env_index,
                                        base_seed=composition.task_plugin.config.eval_base_seed)
        _write(output, "formal_dev_comparison.json", comparison)
        report["formal_dev_comparison"] = comparison
        if comparison["status"] != "PASS":
            raise ValueError("native replay episode evidence differs from the original formal dev receipt")
        import imageio_ffmpeg

        for row in rows:
            if row["recording_status"] != "recorded" or row["frame_count"] <= 0:
                raise ValueError("selected episode has no pre-reset native frames; inspect its recorded outcome")
            video_path = output / "video" / row["file"]
            if file_sha256(video_path) != "sha256:" + row["sha256"]:
                raise ValueError("native video bytes differ from their recording sidecar")
            reader = imageio_ffmpeg.read_frames(str(video_path))
            try:
                next(reader)
                decoded_frames = sum(1 for _ in reader)
            finally:
                reader.close()
            if decoded_frames != row["frame_count"]:
                raise ValueError("native video decoded frame count differs from its task interval")
        report.update(status="PASS", phases_executed=["public_dev"],
            public_dev=composition.task_plugin.phased_evaluation_adapter().project_public_phase_summary(phase_result=result),
            evaluation_execution_cost=dict(result.execution_cost),
            recordings_sha256=file_sha256(output / "video/recordings.json"),
            all_media_decoded=True,
            recorded_outcomes=[row["episode_outcome"]["failure_label"] for row in rows])
        return 0
    except Exception:  # noqa: BLE001 - Preserve diagnostic failure without promoting formal artifacts.
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        report["ended_at_unix"] = time.time()
        _write(output, "result.json", report)
        print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
