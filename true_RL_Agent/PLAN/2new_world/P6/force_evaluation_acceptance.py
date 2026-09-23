"""Bounded production Push-force evaluation of an unchanged training checkpoint.

Use run_owned_probe.py for the sole GPU lease and wall cap. This driver never
trains, qualifies a reference, reads holdouts, or implements force/scoring logic.
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
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.checkpoint import file_sha256


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args):
    from true_rl_agent.composition import TaskCompositionContext
    from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
    from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
    from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig

    trained = read(args.source_train_summary)
    if trained["status"] != "pass" or trained["subject_kind"] != "reference":
        raise ValueError("acceptance anchor must be a completed reference training result")
    expected_sha = "sha256:" + args.expected_checkpoint_sha256.removeprefix("sha256:")
    checkpoint = args.checkpoint or Path(trained["checkpoint"]["metadata"]["local_path"])
    if trained["checkpoint"]["checksum"] != expected_sha or file_sha256(checkpoint) != expected_sha:
        raise ValueError("checkpoint bytes differ from the declared original training anchor")
    old = load_resolved_cell_manifest(args.source_manifest,
                                     expected_manifest_hash=trained["resolved_cell_manifest_hash"])
    source_config = PushRecoveryTaskConfig.from_manifest(old)
    provider = PushRecoveryCompositionProvider()
    values = {**provider.cli_defaults(), "cell_variant_id": args.cell_variant_id,
              "isaac_python": sys.executable, "execution_device": args.device,
              "push_protocol": source_config.push_protocol.to_dict(),
              "reference_program_id": trained["subject_id"], "eval_base_seed": args.eval_base_seed}
    composition = provider.resolve(values=values, context=TaskCompositionContext(
        project_root=ROOT, output_dir=args.output / "composition"))
    new = composition.resolved_cell_manifest
    before, after = old.to_dict(), new.to_dict()
    for field in ("robot_profile", "runtime_adapter"):
        if before[field] != after[field]:
            raise ValueError("active evaluation changed the training robot/runtime: " + field)
    for field in ("action_content_hash", "observation_content_hash", "policy_io_schema_hash",
                  "agent_configuration_hash", "sensor_schema_hash"):
        if before["contract_hashes"][field] != after["contract_hashes"][field]:
            raise ValueError("active evaluation changed the trained policy ABI: " + field)
    old_projection = before["robot_task_binding"]["task_projection"]
    projection = after["robot_task_binding"]["task_projection"]
    if old_projection["push_protocol"] != projection["push_protocol"]:
        raise ValueError("evaluation-only acceptance changed the original training protocol")
    artifact = composition.reference_program_artifact
    if (artifact.manifest.source_checksum != trained["program_source_checksum"]
            or artifact.manifest.manifest_hash != trained["subject_manifest_hash"]):
        raise ValueError("original reference program source is no longer identical")
    cfg = composition.task_plugin.config
    if (cfg.eval_base_seed, cfg.public_seed_count, cfg.eval_num_envs) != (1500000000, 3, 64):
        raise ValueError("this declared acceptance uses fresh base seed 1500000000 and 3 x 64 dev")
    if cfg.push_evaluation_protocol is None:
        raise ValueError("acceptance requires the active force evaluation protocol")
    source_environment = trained["backend_execution_environment_verification"]["expected_manifest_hash"]
    identity = {
        "checkpoint_sha256": expected_sha, "source_training_manifest_hash": old.manifest_hash,
        "source_training_binding_hash": trained["training_binding_hash"],
        "source_reference_manifest_hash": trained["subject_manifest_hash"],
        "source_reference_id": trained["subject_id"],
        "source_program_checksum": trained["program_source_checksum"],
        "source_training_environment_hash": source_environment,
        "source_train_summary_sha256": file_sha256(args.source_train_summary),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "evaluation_manifest_hash": new.manifest_hash,
        "evaluation_protocol": cfg.push_evaluation_protocol.to_dict(),
        "same_robot_and_policy_abi": True, "same_training_protocol": True,
        "checkpoint_role": "unchanged_training_anchor_for_new_evaluation_not_new_force_training",
        "case": args.case, "eval_base_seed": cfg.eval_base_seed,
        "video_seed_index": args.video_seed_index, "video_env_index": args.video_env_index,
        "driver_sha256": file_sha256(__file__),
    }
    if args.source_environment:
        source = read(args.source_environment)
        composition.backend_adapter.validate_session_execution_environment(execution_environment=source)
        if source["manifest_hash"] != source_environment:
            raise ValueError("supplied source environment differs from the original training receipt")
        identity["source_environment_sha256"] = file_sha256(args.source_environment)
    if args.case == "native":
        previous = read(args.expected_rollout_dir / "source_identity.json")
        for field in ("checkpoint_sha256", "source_training_manifest_hash", "evaluation_manifest_hash",
                      "evaluation_protocol", "eval_base_seed"):
            if identity[field] != previous[field]:
                raise ValueError("native replay source identity differs: " + field)
        if previous["case"] != "public_dev" or read(args.expected_rollout_dir / "result.json")["status"] != "PASS":
            raise ValueError("native replay requires a completed numerical public-dev acceptance")
        identity["replay_source_identity_sha256"] = file_sha256(args.expected_rollout_dir / "source_identity.json")
        identity["replay_source_seed_sha256"] = {
            str(i): file_sha256(args.expected_rollout_dir / f"seed_{i:02d}.json")
            for i in range(1, args.video_seed_index + 1)}
    return composition, checkpoint, identity


def run(args, composition, checkpoint, report):
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from true_rl_agent.backends.rsl_rl_checkpoint import load_inference_checkpoint
    from true_rl_agent.evaluation.subject import ExperimentSubject
    from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration
    from true_rl_agent.tasks.push_recovery.runtime import (physical_runtime_config_from_manifest,
        instantiate_physical_push_configs, make_physical_push_environment, push_policy_io_evidence)
    from true_rl_agent.tasks.push_recovery.rollout import run_physical_push_rollout, aggregate_recovery_episodes

    manifest = composition.resolved_cell_manifest
    profile = "no_push" if args.case == "no_push" else "strong_push"
    config = physical_runtime_config_from_manifest(manifest, profile, evaluation=True)
    rollout_args = SimpleNamespace(seed=args.eval_base_seed, num_envs=64, device=args.device,
        max_steps=None, public_seeds=3, public_accept_seeds=3, sealed_validation_seeds=7,
        policy="zero", action_scale=None, task_id=composition.task_id, round=args.output.name,
        evaluation_subject=ExperimentSubject.baseline("p6-force-evaluation-" + args.cell_variant_id))
    torch.manual_seed(args.eval_base_seed)
    env_cfg, agent_cfg = instantiate_physical_push_configs(config, rollout_args, evaluation=True)
    options = None
    if args.case == "native":
        from true_rl_agent.runner.native_video import NativeVideoOptions
        options = NativeVideoOptions(args.output / "video", env_index=args.video_env_index)
        options.configure_viewer(env_cfg)
    base = make_physical_push_environment(env_cfg, render_mode="rgb_array" if options else None)
    wrapped = None
    try:
        io = push_policy_io_evidence(base, manifest)
        recorder = None
        if options:
            from true_rl_agent.runner.native_video import NativeVideoRecorder
            recorder = NativeVideoRecorder(base, options)
        wrapped = RslRlVecEnvWrapper(recorder.env if recorder else base, clip_actions=agent_cfg.clip_actions)
        agent_cfg.device = args.device
        learner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        constructed = rsl_rl_agent_configuration(config["runtime_adapter_id"]).verify_runner(learner)
        loaded = load_inference_checkpoint(learner, str(checkpoint))
        write(args.output / "runtime_admission.json", {
            "policy_io": io, "constructed_learner": constructed, "checkpoint_loading": loaded})
        policy = None if args.case == "zero_action" else learner.get_inference_policy(device=base.device)
        count = args.video_seed_index if recorder else (3 if args.case == "public_dev" else 1)
        records, comparisons = [], []
        for index in range(1, count + 1):
            if recorder and index == args.video_seed_index:
                rollout_args._push_video_recorder = recorder
            report["seed_invocations"].append(index)
            write(args.output / "result.json", report)
            result = run_physical_push_rollout(wrapped, config, rollout_args,
                inference_policy=policy, split_name="public_dev", run_index=index)
            write(args.output / f"seed_{index:02d}.json", result)
            if result["status"] != "pass" or len(result["per_episode"]) != 64:
                raise ValueError("production rollout evidence failed for seed " + str(index))
            if recorder:
                expected = read(args.expected_rollout_dir / f"seed_{index:02d}.json")
                equal = result["per_episode"] == expected["per_episode"]
                comparisons.append({"run_index": index, "all_64_episode_records_equal": equal,
                                    "source_sha256": file_sha256(args.expected_rollout_dir / f"seed_{index:02d}.json")})
                if not equal:
                    raise ValueError("native replay differs from the recorded numerical episodes")
            report["seed_completions"].append(index)
            records.extend(result["per_episode"])
        aggregate = aggregate_recovery_episodes(records, force_sequence=True)
        if args.case == "no_push" and (aggregate["push_event_count"] != 0 or aggregate["control_episode_count"] != 64):
            raise ValueError("no-push control leaked events or formal recovery episodes")
        if recorder:
            import imageio_ffmpeg
            sidecar = recorder.records[-1]
            stream = imageio_ffmpeg.read_frames(str(options.output_dir / sidecar["file"]), pix_fmt="rgb24")
            header = next(stream)
            decoded = sum(1 for _ in stream)
            selected = records[(count - 1) * 64 + args.video_env_index]
            if decoded != selected["episode_steps"] or decoded != sidecar["frame_count"]:
                raise ValueError("native decode does not contain the full pre-reset episode")
            if tuple(header["size"]) != (1280, 720) or abs(header["fps"] - 1 / base.step_dt) > 1e-6:
                raise ValueError("native video dimensions or playback rate differ")
            write(args.output / "video_comparison.json", {"status": "PASS", "comparisons": comparisons,
                "recording": sidecar, "decoded_frames": decoded, "selected_episode": selected,
                "comparison_scope": "all_production_episode_records_including_force_events_not_independent_raw_npz",
                "replay_episodes_not_new_samples": len(records)})
        return aggregate
    finally:
        (wrapped or base).close()


def main(argv=None):
    from true_rl_agent.tasks.push_recovery.config import PUSH_ACTIVE_CELLS
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--case", choices=("public_dev", "no_push", "zero_action", "native"), default="public_dev")
    parser.add_argument("--cell-variant-id", choices=PUSH_ACTIVE_CELLS, required=True)
    for field in ("source-train-summary", "source-manifest", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("checkpoint", "source-environment", "expected-rollout-dir"):
        parser.add_argument("--" + field, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-identity-hash")
    parser.add_argument("--eval-base-seed", type=int, default=1500000000)
    parser.add_argument("--device", required=True)
    parser.add_argument("--video-seed-index", type=int, choices=(1, 2, 3))
    parser.add_argument("--video-env-index", type=int, choices=range(64))
    args = parser.parse_args(argv)
    if args.mode == "run" and not args.expected_identity_hash:
        parser.error("run requires the frozen target preflight --expected-identity-hash")
    if args.case == "native" and any(getattr(args, k) is None for k in ("video_seed_index", "video_env_index", "expected_rollout_dir")):
        parser.error("native requires seed/env indices and the original numerical rollout directory")
    if args.case != "native" and any(getattr(args, k) is not None for k in ("video_seed_index", "video_env_index", "expected_rollout_dir")):
        parser.error("video parameters belong only to the native replay case")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "case": args.case, "implementation_acceptance_only": True,
        "qualification_claimed": False, "train_jobs": 0, "holdout_jobs": 0, "provider_calls": 0,
        "automatic_retries": 0, "seed_invocations": [], "seed_completions": [],
        "planned_public_dev_episodes": 192 if args.case == "public_dev" else 0,
        "evaluated_public_dev_episode_count": 0, "started_at_unix": time.time()}
    app, code = None, 1
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            raise ValueError("outer lease uses host device ordinals; CUDA_VISIBLE_DEVICES must be unset")
        composition, checkpoint, identity = prepare(args)
        environment = composition.backend_adapter.resolve_session_execution_environment(
            session_id="p6-force-evaluation-acceptance", session_dir=args.output, allow_create=True)
        identity["evaluation_execution_environment_hash"] = environment["manifest_hash"]
        identity["identity_hash"] = canonical_json_hash(identity)
        if args.expected_identity_hash and args.expected_identity_hash != identity["identity_hash"]:
            raise ValueError("execution identity differs from frozen target preflight")
        write(args.output / "backend_execution_environment.json", environment)
        write(args.output / "source_identity.json", identity)
        report["identity_hash"] = identity["identity_hash"]
        if args.mode == "run":
            from isaaclab.app import AppLauncher
            from true_rl_agent.runner.isaac_app_lifecycle import bind_isaac_log_path
            launch = argparse.Namespace(headless=True, device=args.device, enable_cameras=args.case == "native")
            bind_isaac_log_path(launch, args.output / "result.json")
            app = AppLauncher(launch, multi_gpu=False).app
            report["aggregate"] = run(args, composition, checkpoint, report)
        report["status"], code = ("PASS" if args.mode == "run" else "PREPARED"), 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["evaluated_public_dev_episode_count"] = 64 * len(report["seed_completions"]) if args.case == "public_dev" else 0
        report["ended_at_unix"] = time.time()
        write(args.output / "result.json", report)
    if app is not None:
        from true_rl_agent.runner.isaac_app_lifecycle import close_isaac_app_on_success
        code = close_isaac_app_on_success(app, exit_code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
