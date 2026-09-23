"""One declared strength, fixed B checkpoint, three public-dev diagnostic seeds.

Use run_owned_probe.py --timeout 900 for the sole lease and wall cap. This direct
production-task rollout does not enter Runner, Genesis, or qualification. No
training, holdout, provider call, automatic retry, or checkpoint re-admission.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider

REFERENCE_ID = "push_standing_recovery_stationary_reference_v3"
SOURCE_SHA = "sha256:43440ad6adec5d99914a41142a951402d753acde5bdf41a162ed1e29d2b74d7d"
ANCHORS = {
    "push.g1_37dof.recovery_v2": ("sha256:44fafbd14fe0e59716c2142a819b81f43d6a3bac8422e0e6d5802f451bceb837", "7fca96669153de0679c7cdd52911fecd407dc48e9f9c6e56adfb588a34061ace", "sha256:de65bfd495ce3d151f74c7f69be12e305f0248477d3dd88f3e80ff21375f0835"),
    "push.go2_12dof.recovery_v2": ("sha256:ea0b925f685c9aa9b74978f3b8914df8b382df894951eb32e97dd161f6756ca9", "a82c526bd986363c63d6a88a7474c30777ae38e4589d3e421d110196c46043a7", "sha256:6a2f3b3da7d5e0bfb8e9759233d46ef1784e79ae4c7d87a5503e7eaf5e654223"),
}


def write(output, name, value):
    (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args):
    checkpoint_sha, old_hash, summary_sha = ANCHORS[args.cell]
    if file_sha256(args.training_summary) != summary_sha:
        raise ValueError("original B training summary bytes differ from the frozen anchor")
    if file_sha256(args.push_protocol_path) != args.expected_protocol_sha256:
        raise ValueError("protocol bytes differ from the declared calibration point")
    trained = json.loads(args.training_summary.read_text(encoding="utf-8"))
    old = load_resolved_cell_manifest(args.training_manifest, expected_manifest_hash=old_hash)
    if (trained["checkpoint"]["checksum"] != checkpoint_sha or trained["resolved_cell_manifest_hash"] != old_hash
            or trained["program_source_checksum"] != SOURCE_SHA or old.cell_variant_id != args.cell):
        raise ValueError("B checkpoint, source, and original manifest lineage disagree")
    provider = PushRecoveryCompositionProvider()
    composition = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": args.cell,
        "execution_device": args.device, "isaac_python": sys.executable, "eval_base_seed": 42,
        "reference_program_id": REFERENCE_ID, "push_protocol_path": args.push_protocol_path},
        context=TaskCompositionContext(project_root=ROOT, output_dir=args.output / "composition"))
    new = composition.resolved_cell_manifest
    if (composition.task_plugin.config.eval_base_seed, composition.task_plugin.config.public_seed_count, composition.task_plugin.config.eval_num_envs) != (42, 3, 64):
        raise ValueError("calibration requires the original three public-dev seeds and 64 environments")
    original, diagnostic = old.to_dict(), new.to_dict()
    for key in ("robot_profile", "runtime_adapter"):
        if original[key] != diagnostic[key]:
            raise ValueError("diagnostic robot/runtime adapter differs from B")
    abi_keys = ("action_content_hash", "observation_content_hash", "policy_io_schema_hash",
                "agent_configuration_hash", "sensor_schema_hash")
    if any(original["contract_hashes"][key] != diagnostic["contract_hashes"][key] for key in abi_keys):
        raise ValueError("diagnostic policy/agent ABI differs from B")
    before = original["robot_task_binding"]["task_projection"]["push_protocol"]
    after = diagnostic["robot_task_binding"]["task_projection"]["push_protocol"]
    allowed = {"low_velocity_scale", "medium_velocity_scale", "high_velocity_scale"}
    if any(before[key] != after[key] for key in before.keys() - allowed):
        raise ValueError("calibration may change only the three declared push magnitudes")
    if composition.reference_program_artifact.manifest.source_checksum != SOURCE_SHA:
        raise ValueError("stationary v3 source changed")
    return composition, {"original_training_manifest_hash": old_hash, "diagnostic_evaluation_manifest_hash": new.manifest_hash,
        "checkpoint_sha256": checkpoint_sha, "original_training_binding_hash": trained["training_binding_hash"],
        "original_reference_manifest_hash": trained["subject_manifest_hash"], "original_program_source_checksum": SOURCE_SHA,
        "original_protocol": before, "diagnostic_protocol": after,
        "diagnostic_resolved_protocol": diagnostic["robot_task_binding"]["task_projection"]["push_resolved_protocol"],
        "same_robot_and_policy_abi": True, "checkpoint_role": "fixed_B_inference_anchor_not_new_protocol_training"}


def run(args, composition, report):
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from true_rl_agent.backends.rsl_rl_checkpoint import load_inference_checkpoint
    from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration
    from true_rl_agent.tasks.push_recovery.runtime import (physical_runtime_config_from_manifest,
        instantiate_physical_push_configs, make_physical_push_environment, push_policy_io_evidence)
    from true_rl_agent.tasks.push_recovery.rollout import run_physical_push_rollout, aggregate_recovery_episodes
    manifest = composition.resolved_cell_manifest
    config = physical_runtime_config_from_manifest(manifest, "strong_push")
    rollout_args = SimpleNamespace(seed=42, num_envs=64, device=args.device, max_steps=None,
        public_seeds=3, public_accept_seeds=3, sealed_validation_seeds=7,
        evaluation_subject=ExperimentSubject.baseline("p6-B-" + args.cell.split(".")[1]),
        task_id=composition.task_id, round=args.output.name)
    torch.manual_seed(42)
    env_cfg, agent_cfg = instantiate_physical_push_configs(config, rollout_args, evaluation=True)
    base = make_physical_push_environment(env_cfg)
    wrapped = None
    try:
        io = push_policy_io_evidence(base, manifest)
        agent_cfg.device = args.device
        wrapped = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
        learner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        constructed = rsl_rl_agent_configuration(config["runtime_adapter_id"]).verify_runner(learner)
        loaded = load_inference_checkpoint(learner, str(args.checkpoint))
        write(args.output, "runtime_admission.json", {"policy_io": io, "constructed_learner": constructed, "checkpoint_loading": loaded})
        policy = learner.get_inference_policy(device=base.device)
        records = []
        for index in range(1, 4):
            report["public_dev_seed_invocations"].append(index)
            write(args.output, "result.json", report)
            result = run_physical_push_rollout(wrapped, config, rollout_args, inference_policy=policy,
                                               split_name="public_dev", run_index=index)
            write(args.output, f"public_dev_seed_{index:02d}.json", result)
            if result["status"] != "pass":
                raise ValueError(f"public dev seed {index} has invalid physical evidence")
            report["public_dev_seed_completions"].append(index)
            records.extend(result["per_episode"])
        return aggregate_recovery_episodes(records)
    finally:
        (wrapped or base).close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--cell", choices=ANCHORS, required=True)
    for name in ("output", "checkpoint", "training-summary", "training-manifest", "push-protocol-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-environment-hash")
    parser.add_argument("--device", required=True)
    args = parser.parse_args(argv)
    if args.mode == "run" and not args.expected_environment_hash:
        parser.error("run requires the target preflight --expected-environment-hash")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "diagnostic_only": True, "enters_runner_or_genesis_or_qualification": False,
        "public_dev_seed_invocations": [], "public_dev_seed_completions": [],
        "train_jobs": 0, "holdout_jobs": 0, "provider_calls": 0, "automatic_retries": 0,
        "planned_public_dev": {"base_seed": 42, "seed_count": 3, "num_envs": 64}, "started_at_unix": time.time()}
    app, code = None, 1
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            raise ValueError("outer lease uses host device ordinals; CUDA_VISIBLE_DEVICES must be unset")
        composition, identity = prepare(args)
        if file_sha256(args.checkpoint) != identity["checkpoint_sha256"]:
            raise ValueError("checkpoint bytes differ from the original B anchor")
        environment = composition.backend_adapter.resolve_session_execution_environment(
            session_id="p6-calibration", session_dir=args.output, allow_create=True)
        if args.expected_environment_hash and environment["manifest_hash"] != args.expected_environment_hash:
            raise ValueError("target environment differs from the declared preflight")
        write(args.output, "backend_execution_environment.json", environment)
        write(args.output, "source_identity.json", {**identity, "script_sha256": file_sha256(__file__),
            "sources": {key: {"path": str(getattr(args, key).resolve()), "sha256": file_sha256(getattr(args, key))}
                        for key in ("checkpoint", "training_summary", "training_manifest", "push_protocol_path")}})
        if args.mode == "run":
            from isaaclab.app import AppLauncher
            from true_rl_agent.runner.isaac_app_lifecycle import bind_isaac_log_path
            launch = argparse.Namespace(headless=True, device=args.device)
            bind_isaac_log_path(launch, args.output / "result.json")
            app = AppLauncher(launch, multi_gpu=False).app
            report["aggregate"] = run(args, composition, report)
        report.update(status="PASS" if args.mode == "run" else "PREPARED", execution_environment_hash=environment["manifest_hash"])
        code = 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["ended_at_unix"] = time.time()
        write(args.output, "result.json", report)
    if app is not None:
        from true_rl_agent.runner.isaac_app_lifecycle import close_isaac_app_on_success
        code = close_isaac_app_on_success(app, exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
