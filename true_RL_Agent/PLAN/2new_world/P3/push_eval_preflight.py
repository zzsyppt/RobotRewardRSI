"""Exercise the current Push checkpoint-eval entrypoint; no reference promotion."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.backends.isaac_lab_execution_environment import (
    build_current_isaac_lab_execution_environment_manifest,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
from true_rl_agent.policy_baselines import (
    BaselineCompatibilitySpec, BaselineManifest, CheckpointProvenance,
)
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.process_handle import run_owned_process
from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.push_recovery.composition import (
    PUSH_RECOVERY_BACKEND_ID, PUSH_RECOVERY_ROBOT,
    PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH, PUSH_RECOVERY_ACTION_SCHEMA_HASH,
    PUSH_RECOVERY_POLICY_ARCHITECTURE,
)
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin

checkpoint, output = (Path(value).resolve() for value in sys.argv[1:3])
device = sys.argv[3]
output.mkdir(parents=True, exist_ok=False)
config = PushRecoveryTaskConfig(eval_num_envs=16, eval_base_seed=20260908)
task = PushRecoveryTaskPlugin(config).task_spec
protocol = config.evaluation_protocol()
binding = EvaluationBindingV1(
    eval_protocol_hash=protocol.protocol_hash,
    acceptance_protocol_hash=config.acceptance_protocol().protocol_hash,
    task_protocol_hash=task.resolved_task_protocol_hash(),
    seed_cohort_hash=config.seed_cohort_hash(),
)
runtime = build_current_isaac_lab_execution_environment_manifest()
baseline = BaselineManifest(
    baseline_id="p2-short-resume-diagnostic", task_id=task.task_id,
    checkpoint=CheckpointRef(
        uri="artifact://p2/audit2_push_resume/resumed.pt",
        checksum=file_sha256(checkpoint), role="warm_start", sealed=True,
    ),
    provenance=CheckpointProvenance(
        source_run_id="audit2_push_resume", source_task_id=task.task_id,
        created_by="PLAN/2new_world/P2/push_runtime_probe.py",
    ),
    compatibility=BaselineCompatibilitySpec(
        backend_id=PUSH_RECOVERY_BACKEND_ID, robot=PUSH_RECOVERY_ROBOT,
        observation_schema_hash=PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH,
        action_schema_hash=PUSH_RECOVERY_ACTION_SCHEMA_HASH,
        policy_architecture=PUSH_RECOVERY_POLICY_ARCHITECTURE,
    ),
)
(output / "baseline_manifest.json").write_text(
    json.dumps(baseline.to_public_dict(), indent=2) + "\n"
)
arguments = {
    "job-id": "p3-push-learner-eval", "round": "p3-preflight",
    "task-id": task.task_id, "subject-kind": "baseline",
    "subject-id": "p2-short-resume-diagnostic", "checkpoint": checkpoint,
    "subject-manifest-hash": baseline.manifest_hash,
    "checkpoint-sha256": file_sha256(checkpoint),
    "public-seeds": protocol.public_seed_count,
    "public-accept-seeds": protocol.public_accept_seed_count,
    "sealed-validation-seeds": protocol.sealed_validation_seed_count,
    "eval-protocol-version": protocol.eval_protocol_version,
    "task-protocol-hash": binding.task_protocol_hash,
    "acceptance-protocol-hash": binding.acceptance_protocol_hash,
    "seed-cohort-hash": binding.seed_cohort_hash,
    "evaluation-binding-hash": binding.binding_hash,
    "capability-profile-hash": task.candidate_capability_profile.profile_hash,
    "summary-path": output / "summary.json",
    "profile": config.eval_profile,
    "runtime-dynamics-hash": config.eval_runtime_dynamics()["runtime_dynamics_hash"],
    "num-envs": config.eval_num_envs, "seed": config.eval_base_seed,
    "max-steps": 100, "device": "cuda:0",
    "backend-execution-environment-manifest-hash": runtime.manifest_hash,
}
argv = [sys.executable, str(ROOT / "scripts/isaaclab_rsl_rl_eval.py")]
for name, value in arguments.items():
    argv.extend(("--" + name, str(value)))
inventory = discover_nvidia_gpu_inventory()
gpu = next(item for item in inventory if item.execution_device == device)
identity = canonical_json_hash({"purpose": "P3 preflight", "argv": argv})
manager = GpuLeaseManager(
    lease_root=Path("/tmp/true_rl_agent_gpu_leases"),
    devices={item.execution_device: item.identity for item in inventory},
)
holder = PreGenesisGpuLeaseHolder(
    session_id="p3-preflight", genesis_identity_hash=identity,
    session_contract_hash=identity, lane_id=device,
    invocation_segment_id=output.name, logical_job_id=output.name,
)
report = {
    "purpose": "Construction evidence only; short checkpoint and new diagnostic seeds, no formal acceptance or reference qualification",
    "argv": argv, "cwd": str(ROOT), "runtime_manifest_hash": runtime.manifest_hash,
}
(output / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
with manager.acquire(device, holder=holder):
    result = run_owned_process(
        argv, cwd=ROOT, log_dir=output / "process", timeout_s=300,
        environment_overrides={"CUDA_VISIBLE_DEVICES": gpu.uuid},
    )
report["returncode"] = result.returncode
(output / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(result.returncode)
