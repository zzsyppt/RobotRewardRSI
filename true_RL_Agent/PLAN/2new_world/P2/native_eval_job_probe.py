"""P2 backend qualification through the actual production job runner.

This consumes a frozen historical checkpoint without claiming a newly trained
reference or writing any formal session ledger. Materialization, phase binding,
GPU visibility, lease and process ownership use the production implementation.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.backends.isaac_lab_execution_environment import (
    build_current_isaac_lab_execution_environment_manifest,
)
from true_rl_agent.composition.base import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.process_handle import run_owned_process
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory
from true_rl_agent.runner.specs import CheckpointRef, GpuProcessBinding
from true_rl_agent.tasks.stair_climbing.composition import (
    StairClimbingCompositionProvider,
)

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("g1", "go2"), required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--checkpoint-summary", type=Path)
parser.add_argument("--train", action="store_true")
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--timeout", type=int)
parser.add_argument("--video-env-index", type=int, default=0)
args = parser.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
provider = StairClimbingCompositionProvider()
values = dict(provider.cli_defaults())
values.update(
    execution_device=args.device,
    cell_variant_id=(
        "stair.g1_37dof.unified_v2"
        if args.robot == "g1"
        else "stair.go2_12dof.unified_v2"
    ),
)
composition = provider.resolve(
    values=values, context=TaskCompositionContext(project_root=ROOT, output_dir=output)
)
manifest = build_current_isaac_lab_execution_environment_manifest()
inventory = discover_nvidia_gpu_inventory()
gpu = next(item for item in inventory if item.execution_device == args.device)
resource_hash = gpu.identity_hash
binding = GpuProcessBinding(
    host_execution_device=args.device,
    process_execution_device="cuda:0",
    resource_binding_hash=resource_hash,
)
backend = composition.backend_adapter
jobs = backend.materialize_reference_jobs(
    session_id=output.name,
    task_spec=composition.task_plugin.task_spec,
    reference=composition.reference_program_artifact,
    artifact_root=output / "reference_program",
    policy_initialization=composition.policy_initialization,
    evaluation_binding=backend.evaluation_binding_for(
        composition.task_plugin.task_spec
    ),
    backend_execution_environment_manifest_hash=manifest.manifest_hash,
    gpu_process_binding=binding,
)

if args.train:
    if args.checkpoint is not None:
        parser.error("P2 full training starts from the bound random initialization")
    spec = jobs.train_spec
else:
    if args.checkpoint is None:
        parser.error("public replay requires --checkpoint")
    checkpoint = args.checkpoint.resolve()
    checkpoint_metadata = {"local_path": str(checkpoint)}
    if args.checkpoint_summary:
        trained = json.loads(args.checkpoint_summary.read_text())
        assert trained["checkpoint"]["checksum"] == file_sha256(checkpoint)
        checkpoint_metadata["resolved_cell_manifest_hash"] = trained[
            "resolved_cell_manifest_hash"
        ]
    plan = replace(
        jobs.eval_spec,
        checkpoint=CheckpointRef(
            uri="artifact://p2-diagnostic-replay/best.pt",
            checksum=file_sha256(checkpoint),
            role="eval",
            sealed=True,
            metadata=checkpoint_metadata,
        ),
        checkpoint_output_plan=None,
    )
    plan = backend.rematerialize_eval_plan(
        eval_plan=plan,
        round_id="p2-diagnostic",
        summary_path=str(output / "evaluation/summary.json"),
    )
    spec = backend.materialize_eval_phase(
        eval_plan=plan, phase=EvaluationPhase.PUBLIC_DEV
    )
    if not args.no_video:
        spec = replace(
            spec,
            metadata={
                **dict(spec.metadata),
                "cli_argv": [
                    *spec.metadata["cli_argv"],
                    "--video-dir",
                    str(output / "video"),
                    "--video-env-index",
                    str(args.video_env_index),
                ],
            },
        )
spec = replace(
    spec,
    metadata={
        **dict(spec.metadata),
        "cli_command": " ".join(spec.metadata["cli_argv"]),
    },
)
identity = canonical_json_hash(
    {"purpose": "P2 backend qualification", "job": spec.to_public_dict()}
)
scope = PreGenesisExecutionScope(
    journal_root=output / "attempts",
    session_id="new-world-p2-acceptance",
    genesis_identity_hash=identity,
    session_contract_hash=identity,
    lane_id=args.device,
    invocation_segment_id=output.name,
    budget_contract_hash=identity,
    max_infrastructure_retries=0,
    attempt_budget_units=1,
    resource_binding_hash=resource_hash,
)
spec = replace(spec, attempt_scope=scope)
manager = GpuLeaseManager(
    lease_root=Path("/tmp/true_rl_agent_gpu_leases"),
    devices={item.execution_device: item.identity for item in inventory},
)
report = {
    "purpose": "P2 backend qualification; no formal session or reference promotion",
    "argv": list(spec.metadata["cli_argv"]),
    "execution_environment": manifest.to_dict(),
    "job_spec": spec.to_public_dict(),
}
(output / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
runner = SubprocessRunner(
    allowed_executables=(sys.executable,),
    cwd=ROOT,
    timeout_s=args.timeout or (36000 if args.train else 900),
    gpu_lease_manager=manager,
)
try:
    if args.train or args.checkpoint_summary:
        result = runner.run_train(spec) if args.train else runner.run_eval_phase(spec)
        report.update(status="PASS", returncode=0, admission="production_runner")
    else:
        # Old anchors can have a retired cell binding. Do not rewrite their
        # metadata to pass formal admission; this branch is explicitly diagnostic.
        report["admission"] = "historical_diagnostic_cli_only"
        holder = PreGenesisGpuLeaseHolder(
            session_id=scope.session_id,
            genesis_identity_hash=identity,
            session_contract_hash=identity,
            lane_id=args.device,
            invocation_segment_id=output.name,
            logical_job_id=spec.job_id,
        )
        with manager.acquire(args.device, holder=holder):
            result = run_owned_process(
                report["argv"],
                cwd=ROOT,
                log_dir=output / "process",
                timeout_s=runner.timeout_s,
                environment_overrides={
                    "CUDA_VISIBLE_DEVICES": gpu.uuid,
                    "NVIDIA_VISIBLE_DEVICES": None,
                },
            )
        report.update(
            status="PASS" if result.returncode == 0 else "FAIL",
            returncode=result.returncode,
            child_pid=result.pid,
        )
except Exception as exc:
    report.update(status="FAIL", returncode=1, error=str(exc))
    raise
finally:
    (output / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
sys.exit(report["returncode"])
