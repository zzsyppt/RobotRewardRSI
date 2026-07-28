#!/usr/bin/env python3
"""Validate the frozen stair environment and official G1 rough warm-start."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher  # type: ignore[import-not-found]  # noqa: E402

from true_rl_agent.runner.checkpoint import (  # noqa: E402
    build_checkpoint_safety_report,
    validate_checkpoint_safety_report,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import (  # noqa: E402
    DEFAULT_STAIR_EXTENSION_DESCRIPTOR_REF,
    PHASE0_SOURCE_ARTIFACT_REFS,
    StairPhase0Bundle,
    canonical_json_sha256,
    verify_checkpoint_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptor-ref",
        default=str(DEFAULT_STAIR_EXTENSION_DESCRIPTOR_REF).replace("\\", "/"),
        help="Project-relative frozen stair extension descriptor ref.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


args_cli = _build_parser().parse_args()
if not args_cli.device:
    args_cli.device = "cuda:0"

bundle = StairPhase0Bundle.load(
    PROJECT_ROOT,
    descriptor_ref=args_cli.descriptor_ref,
)
checkpoint_safety = build_checkpoint_safety_report(
    args_cli.checkpoint,
    expected_checkpoint_hash=f"sha256:{bundle.baseline.checkpoint_sha256}",
)
validate_checkpoint_safety_report(
    checkpoint_safety.to_public_dict(),
    expected_checkpoint_hash=f"sha256:{bundle.baseline.checkpoint_sha256}",
)
checkpoint_verification = verify_checkpoint_file(args_cli.checkpoint, bundle.baseline)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # type: ignore[import-not-found]  # noqa: E402
import isaaclab  # type: ignore[import-not-found]  # noqa: E402
import isaaclab_tasks  # type: ignore[import-not-found]  # noqa: F401,E402
import torch  # type: ignore[import-not-found]  # noqa: E402
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,  # type: ignore[import-not-found]  # noqa: E402
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (  # type: ignore[import-not-found]  # noqa: E402
    G1RoughPPORunnerCfg,
)
from rsl_rl.runners import (
    OnPolicyRunner,  # type: ignore[import-not-found]  # noqa: E402
)

from true_rl_agent.tasks.stair_climbing.isaaclab_env_cfg import (  # noqa: E402
    PHASE0_STAIR_GYM_ID,
    build_phase0_stair_env_cfg,
    build_stair_env_cfg,
    register_phase0_stair_env,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_phase0_probe import (  # noqa: E402
    assert_runtime_environment_matches,
    build_runtime_environment_evidence,
    phase0_task_viability_passed,
    run_first_episode_probe,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_terrain import (  # noqa: E402
    build_geometry_report,
)


def main() -> int:
    protocol = bundle.protocol.to_dict()
    manifest = bundle.baseline.to_dict()
    report_path = Path(args_cli.output).resolve()
    env: RslRlVecEnvWrapper | None = None
    gym_env: Any | None = None
    base_report = {
        "schema_version": "true_rl_agent.task.stair_climbing.phase0_report.v1",
        "task_id": bundle.protocol.task_id,
        "protocol_id": bundle.protocol.protocol_id,
        "task_protocol_hash": bundle.protocol.protocol_hash,
        "baseline_id": bundle.baseline.baseline_id,
        "baseline_manifest_hash": bundle.baseline.manifest_hash,
        "checkpoint": {
            "sha256": checkpoint_verification.sha256,
            "size_bytes": checkpoint_verification.size_bytes,
        },
        "checkpoint_safety": checkpoint_safety.to_public_dict(),
        "source_artifacts": _source_artifact_hashes(),
    }
    try:
        geometry = build_geometry_report(protocol)
        _assert_geometry(geometry, protocol)
        runtime_versions = _runtime_versions()
        _assert_version_lock(runtime_versions, manifest["version_lock"])
        checkpoint_structure = _inspect_checkpoint(
            checkpoint_verification.path,
            manifest,
        )

        register_phase0_stair_env()
        production_train_cfg = build_stair_env_cfg(
            protocol,
            mode="train",
            seed=int(protocol["phase0_probe"]["seed"]),
            device=args_cli.device,
        )
        production_eval_cfg = build_stair_env_cfg(
            protocol,
            mode="eval",
            seed=int(protocol["phase0_probe"]["seed"]),
            device=args_cli.device,
        )
        for mode, production_cfg in (
            ("train", production_train_cfg),
            ("eval", production_eval_cfg),
        ):
            if getattr(
                production_cfg.recorders,
                "terminal_transition",
                None,
            ) is not None:
                raise RuntimeError(
                    f"Phase 0 transition recorder leaked into production {mode} config"
                )
        env_cfg = build_phase0_stair_env_cfg(
            protocol,
            seed=int(protocol["phase0_probe"]["seed"]),
            device=args_cli.device,
        )
        if getattr(env_cfg.recorders, "terminal_transition", None) is None:
            raise RuntimeError("Phase 0 transition recorder is not configured")
        gym_env = gym.make(PHASE0_STAIR_GYM_ID, cfg=env_cfg)
        raw_env = gym_env.unwrapped
        env = RslRlVecEnvWrapper(gym_env, clip_actions=1.0)

        agent_cfg = G1RoughPPORunnerCfg()
        agent_cfg.device = args_cli.device
        agent_cfg.seed = int(protocol["phase0_probe"]["seed"])
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=args_cli.device,
        )
        runner.load(str(checkpoint_verification.path))
        policy = runner.get_inference_policy(device=raw_env.device)

        observation = env.get_observations()
        policy_observation = observation["policy"]
        with torch.inference_mode():
            action = policy(observation)
        _assert_finite(policy_observation, "initial policy observation")
        _assert_finite(action, "initial policy action")

        environment_evidence = build_runtime_environment_evidence(
            raw_env=raw_env,
            policy_observation=policy_observation,
            action=action,
        )
        assert_runtime_environment_matches(environment_evidence, protocol)
        _assert_command(raw_env, protocol)

        rollout = run_first_episode_probe(
            env=env,
            raw_env=raw_env,
            policy=policy,
            observation=observation,
            protocol=protocol,
        )
        viability_passed = phase0_task_viability_passed(
            protocol,
            rollout["first_episodes"],
        )
        payload = {
            **base_report,
            "status": "pass" if viability_passed else "fail",
            "runtime_versions": runtime_versions,
            "checkpoint_structure": checkpoint_structure,
            "environment": environment_evidence,
            "geometry": geometry,
            "rollout": rollout,
            "admission": {
                "checkpoint_structurally_compatible": True,
                "runtime_schema_compatible": True,
                "terrain_binding_compatible": True,
                "first_episode_probe_complete": True,
                "task_viability_gate_passed": viability_passed,
                "warm_start_candidate_admitted": viability_passed,
                "accepted_current_best": False,
                "baseline_eval_only_still_required": True,
                "session_genesis_still_required": True,
                "report_is_acceptance_evidence": False,
            },
        }
        _write_report(report_path, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0 if viability_passed else 2
    except Exception as exc:
        failure = {
            **base_report,
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_report(report_path, failure)
        raise
    finally:
        if env is not None:
            env.close()
        elif gym_env is not None:
            gym_env.close()


def _inspect_checkpoint(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint top level must be a mapping")
    required_keys = manifest["checkpoint"]["required_top_level_keys"]
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise RuntimeError(f"checkpoint is missing required keys: {missing}")
    iteration = checkpoint.get("iter")
    if iteration != manifest["checkpoint"]["iteration"]:
        raise RuntimeError(
            "checkpoint iteration mismatch: "
            f"expected {manifest['checkpoint']['iteration']}, observed {iteration}"
        )
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint model_state_dict must be a mapping")
    actor_weights = [
        tensor
        for name, tensor in state.items()
        if "actor" in name and name.endswith("weight") and getattr(tensor, "ndim", 0) == 2
    ]
    critic_weights = [
        tensor
        for name, tensor in state.items()
        if "critic" in name and name.endswith("weight") and getattr(tensor, "ndim", 0) == 2
    ]
    if not actor_weights or not critic_weights:
        raise RuntimeError("checkpoint actor/critic linear layers were not found")
    observed = {
        "iteration": int(iteration),
        "top_level_keys": sorted(str(key) for key in checkpoint),
        "model_state_key_count": len(state),
        "actor_input_dim": int(actor_weights[0].shape[1]),
        "actor_hidden_dims": [int(tensor.shape[0]) for tensor in actor_weights[:-1]],
        "actor_output_dim": int(actor_weights[-1].shape[0]),
        "critic_input_dim": int(critic_weights[0].shape[1]),
        "critic_hidden_dims": [int(tensor.shape[0]) for tensor in critic_weights[:-1]],
        "critic_output_dim": int(critic_weights[-1].shape[0]),
    }
    expected = manifest["compatibility"]
    checks = {
        "actor_input_dim": expected["policy_observation_dim"],
        "actor_hidden_dims": expected["actor_hidden_dims"],
        "actor_output_dim": expected["actor_output_dim"],
        "critic_input_dim": expected["policy_observation_dim"],
        "critic_hidden_dims": expected["critic_hidden_dims"],
        "critic_output_dim": 1,
    }
    for key, expected_value in checks.items():
        if observed[key] != expected_value:
            raise RuntimeError(
                f"checkpoint architecture mismatch for {key}: "
                f"expected {expected_value}, observed {observed[key]}"
            )
    return observed


def _source_artifact_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for ref in PHASE0_SOURCE_ARTIFACT_REFS:
        digest = hashlib.sha256()
        with (PROJECT_ROOT / ref).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[ref] = digest.hexdigest()
    return result


def _runtime_versions() -> dict[str, str]:
    return {
        "isaac_lab": _isaac_lab_release_version(),
        "isaaclab_extension": _distribution_version("isaaclab"),
        "isaaclab_tasks_extension": _distribution_version("isaaclab_tasks"),
        "isaaclab_rl_extension": _distribution_version("isaaclab_rl"),
        "rsl_rl": _distribution_version("rsl-rl-lib"),
        "torch": str(torch.__version__),
    }


def _isaac_lab_release_version() -> str:
    module_path = Path(isaaclab.__file__).resolve()
    for parent in module_path.parents:
        version_path = parent / "VERSION"
        if version_path.is_file():
            return version_path.read_text(encoding="utf-8").strip()
    raise RuntimeError("Isaac Lab VERSION file could not be resolved from the module")


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required runtime distribution is missing: {name}") from exc


def _assert_version_lock(
    observed: dict[str, str],
    expected: dict[str, str],
) -> None:
    mismatches = {
        key: {"expected": expected[key], "observed": observed.get(key)}
        for key in expected
        if observed.get(key) != expected[key]
    }
    if mismatches:
        raise RuntimeError(f"runtime version lock mismatch: {mismatches}")


def _assert_geometry(
    geometry: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> None:
    bins = protocol["terrain"]["evaluation_bins"]
    if len(geometry) != len(bins):
        raise RuntimeError("geometry report does not cover every evaluation bin")
    expected_origin = [
        protocol["terrain"]["start_platform_length_m"] / 2.0,
        protocol["terrain"]["size_m"][1] / 2.0,
        0.0,
    ]
    for item, expected_bin in zip(geometry, bins, strict=True):
        if item["bin_id"] != expected_bin["bin_id"]:
            raise RuntimeError("geometry bin order mismatch")
        if item["origin_m"] != expected_origin:
            raise RuntimeError("stair origin is not on the bottom start platform")
        if not item["is_non_flat"] or item["missing_top_levels_m"]:
            raise RuntimeError(f"stair geometry degraded for {item['bin_id']}")
        expected_top = protocol["terrain"]["step_count"] * expected_bin["height_m"]
        if abs(item["max_height_m"] - expected_top) > 1e-7:
            raise RuntimeError(f"stair top height mismatch for {item['bin_id']}")


def _assert_command(raw_env: Any, protocol: Mapping[str, Any]) -> None:
    command = raw_env.command_manager.get_command("base_velocity")
    _assert_finite(command, "base velocity command")
    expected_forward = float(protocol["evaluation"]["forward_command_mps"])
    if not torch.allclose(
        command[:, 0],
        torch.full_like(command[:, 0], expected_forward),
    ):
        raise RuntimeError("runtime forward command does not match protocol")
    if not torch.allclose(command[:, 1:], torch.zeros_like(command[:, 1:])):
        raise RuntimeError("runtime lateral/yaw commands must be zero")


def _assert_finite(value: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{context} contains NaN or Inf")


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    report = dict(payload)
    report["report_hash"] = canonical_json_sha256(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
