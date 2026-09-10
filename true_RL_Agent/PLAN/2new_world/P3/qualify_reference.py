"""P3 acceptance through production phase jobs, receipts and task qualification.

No candidate search or Genesis promotion. A separate study-final probe checks
the committed private geometry path without claiming a completed formal study.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.agent_core.trial_receipts import TrialReceiptJournal, TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION
from true_rl_agent.backends.isaac_lab_execution_environment import build_current_isaac_lab_execution_environment_manifest
from true_rl_agent.composition.base import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash, EvaluationBindingV1
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.gpu_lease import GpuLeaseManager
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope
from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory
from true_rl_agent.runner.specs import CheckpointRef, GpuProcessBinding
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.final_confirmation import create_stair_generation_final_confirmation_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-summary", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provider = StairClimbingCompositionProvider()
    values = dict(provider.cli_defaults())
    values.update(execution_device=args.device, cell_variant_id=args.cell)
    composition = provider.resolve(values=values, context=TaskCompositionContext(project_root=ROOT, output_dir=output))
    config = composition.task_plugin.config
    final_plan, final_path = create_stair_generation_final_confirmation_plan(
        output_dir=output, session_id=output.name, config=config,
    )
    environment = build_current_isaac_lab_execution_environment_manifest()
    inventory = discover_nvidia_gpu_inventory()
    gpu = next(item for item in inventory if item.execution_device == args.device)
    binding = GpuProcessBinding(host_execution_device=args.device, process_execution_device="cuda:0",
                               resource_binding_hash=gpu.identity_hash)
    backend = composition.backend_adapter
    jobs = backend.materialize_reference_jobs(
        session_id=output.name, task_spec=composition.task_plugin.task_spec,
        reference=composition.reference_program_artifact, artifact_root=output / "reference_program",
        policy_initialization=composition.policy_initialization,
        evaluation_binding=backend.evaluation_binding_for(composition.task_plugin.task_spec),
        backend_execution_environment_manifest_hash=environment.manifest_hash, gpu_process_binding=binding,
    )
    checkpoint = args.checkpoint.resolve()
    trained = json.loads(args.checkpoint_summary.read_text())
    if trained["checkpoint"]["checksum"] != file_sha256(checkpoint):
        raise ValueError("Training receipt and checkpoint differ")
    if trained["training_binding_hash"] != jobs.eval_spec.training_binding_hash:
        raise ValueError("P3 must retain the checkpoint's complete training binding")
    plan = replace(jobs.eval_spec, checkpoint=CheckpointRef(
        uri="artifact://p3-reference-qualification/best.pt", checksum=file_sha256(checkpoint), role="eval", sealed=True,
        metadata={"local_path": str(checkpoint), "resolved_cell_manifest_hash": trained["resolved_cell_manifest_hash"]},
    ), checkpoint_output_plan=None)
    plan = backend.rematerialize_eval_plan(eval_plan=plan, round_id="p3-qualification",
                                           summary_path=str(output / "evaluation/summary.json"))
    identity = canonical_json_hash({"purpose": "P3 reference acceptance", "job": plan.to_public_dict()})
    scope = PreGenesisExecutionScope(
        journal_root=output / "attempts", session_id=output.name, genesis_identity_hash=identity,
        session_contract_hash=identity, lane_id=args.device, invocation_segment_id=output.name,
        budget_contract_hash=identity, max_infrastructure_retries=0, attempt_budget_units=1,
        resource_binding_hash=gpu.identity_hash,
    )
    runner = SubprocessRunner(allowed_executables=(sys.executable,), cwd=ROOT, timeout_s=1800,
        gpu_lease_manager=GpuLeaseManager(lease_root=Path("/tmp/true_rl_agent_gpu_leases"),
                                        devices={item.execution_device: item.identity for item in inventory}))
    journal = TrialReceiptJournal(output / "receipts", contract={
        "schema_version": TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION, "round_id": output.name,
        "session_genesis_hash": identity, "ledger_epoch_hash": identity,
        "purpose": "p3_reference_qualification", "commitment": {"evaluation_plan_hash": canonical_json_hash(plan.to_public_dict())},
    })
    journal.bind_contract()
    results, refs = [], {}
    report = {"purpose": "P3 pre-Genesis acceptance; no search or promotion", "status": "RUNNING",
              "cell": args.cell, "checkpoint": file_sha256(checkpoint), "training_summary": str(args.checkpoint_summary),
              "training_binding_hash": plan.training_binding_hash, "environment": environment.to_dict(), "phases": {}}
    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    save()
    try:
        for phase in EvaluationPhase:
            spec = backend.materialize_eval_phase(eval_plan=plan, phase=phase, attempt_scope=scope)
            if phase == EvaluationPhase.PUBLIC_DEV:
                cells = config.ordered_evaluation_suite.private_cell_assignment("public_dev")
                rough = [i for i, cell in enumerate(cells) if cell.stratum_id == "c2_mechanism_ood"]
                index = max(rough, key=lambda i: cells[i].surface_profile["amplitude_m"])
                argv = [*spec.metadata["cli_argv"], "--video-dir", str(output / "video"), "--video-env-index", str(index)]
                spec = replace(spec, metadata={**spec.metadata, "cli_argv": argv, "cli_command": " ".join(argv)})
            result = runner.run_eval_phase(spec)
            refs[phase] = journal.publish_eval_phase(spec, result).to_dict()
            results.append(result)
            report["phases"][phase.value] = result.to_private_dict()
            save()
        adapter = composition.task_plugin.phased_evaluation_adapter()
        assembly = adapter.assemble_reference_full_summary(eval_plan=plan, phase_results=results, phase_receipt_refs=refs)
        raw = runner.project_assembled_eval_summary(plan, assembly)
        qualified = composition.task_plugin.qualify_reference_result(public_result=raw, train_summary=trained)
        report["reference_qualification"] = dict(qualified.private_evidence)
        report["reference_gate"] = dict(qualified.gate)
        (output / "assembly.json").write_text(json.dumps(assembly.to_private_dict(), indent=2, sort_keys=True) + "\n")
        save()

        final_binding = EvaluationBindingV1(
            eval_protocol_hash=plan.evaluation_binding.eval_protocol_hash,
            acceptance_protocol_hash=plan.evaluation_binding.acceptance_protocol_hash,
            task_protocol_hash=plan.evaluation_binding.task_protocol_hash, seed_cohort_hash=final_plan.seed_cohort_hash,
        )
        final_eval = backend.rematerialize_eval_plan(
            eval_plan=replace(plan, evaluation_binding=final_binding), round_id="p3-study-final-probe",
            summary_path=str(output / "final_probe/summary.json"),
        )
        argv = [*final_eval.metadata["cli_argv"], "--stair-eval-purpose", "final_confirmation", "--stair-final-plan", str(final_path)]
        final_eval = replace(final_eval, metadata={**final_eval.metadata, "cli_argv": argv, "cli_command": " ".join(argv)})
        final_spec = backend.materialize_eval_phase(eval_plan=final_eval, phase=EvaluationPhase.SEALED_VALIDATION, attempt_scope=scope)
        report["study_final_probe"] = runner.run_eval_phase(final_spec).to_private_dict()
        report["status"] = "PASS" if qualified.gate["status"] == "pass" else "NO_GO"
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
