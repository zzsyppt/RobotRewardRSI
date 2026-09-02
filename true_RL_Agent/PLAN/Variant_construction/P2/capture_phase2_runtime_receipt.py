"""Capture the Phase 2 Stair morphology projection in one real Isaac scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

RECEIPT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase2_runtime_receipt.v1"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_text_hash(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _tensor_summary(torch: Any, value: Any) -> dict[str, Any]:
    if not torch.is_tensor(value):
        raise TypeError("Phase 2 runtime receipt expected a tensor")
    summary = {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
    }
    if value.is_floating_point():
        summary["finite"] = bool(torch.isfinite(value).all().item())
    return summary


def _observation_summary(torch: Any, observation: Any) -> dict[str, Any]:
    if isinstance(observation, Mapping):
        return {
            str(name): _tensor_summary(torch, value)
            for name, value in sorted(observation.items())
        }
    return {"policy": _tensor_summary(torch, observation)}


def _captured_head(project_root: Path, explicit: str | None) -> str | None:
    if explicit is not None:
        if len(explicit) != 40 or any(
            character not in "0123456789abcdef" for character in explicit
        ):
            raise ValueError("captured repository HEAD must be a lowercase Git SHA")
        return explicit
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def _capture(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve(strict=True)
    output = args.output.resolve()
    sys.path.insert(0, str(project_root / "src"))

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device=args.device)
    _ = launcher.app

    import gymnasium as gym
    import torch

    from true_rl_agent.tasks.stair_climbing.composition import (
        StairClimbingCompositionConfig,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime import (
        StairRewardInputBuilder,
        stair_kinematic_base_state,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_eval_env_cfg,
        register_production_stair_envs,
    )
    from true_rl_agent.tasks.stair_climbing.legacy_transition_reader import (
        normalize_stair_transition_payload,
    )
    from true_rl_agent.tasks.stair_climbing.plugin import (
        StairClimbingTaskPlugin,
    )
    from true_rl_agent.tasks.stair_climbing.reference_program import (
        resolve_official_stair_reference_program,
    )
    from true_rl_agent.tasks.stair_climbing.reward_inputs import (
        StairRewardInputsV2,
    )
    from true_rl_agent.tasks.stair_climbing.reward_program import (
        evaluate_stair_reward_program,
    )
    from true_rl_agent.tasks.stair_climbing.robot_binding import (
        STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
        resolve_stair_cell_manifest,
    )
    from true_rl_agent.tasks.stair_climbing.runtime_profile import (
        STAIR_EVAL_GYM_ID,
        STAIR_EVAL_TRANSITION_EXTRAS_KEY,
        STAIR_TRANSITION_RECORDER_SCHEMA_VERSION,
    )
    from true_rl_agent.tasks.stair_climbing.runtime_projection import (
        resolve_stair_runtime_projection,
        stair_runtime_projection_from_manifest,
    )

    composition = StairClimbingCompositionConfig.defaults(project_root=project_root)
    config = composition.to_task_config(project_root=project_root)
    plugin = StairClimbingTaskPlugin(config)
    manifest = resolve_stair_cell_manifest(
        cell_variant_id=STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
        config=config,
        plugin=plugin,
    )
    projection = stair_runtime_projection_from_manifest(manifest)
    reference = resolve_official_stair_reference_program(
        config,
        project_root=project_root,
    ).reward_program

    register_production_stair_envs()
    env_cfg = build_stair_eval_env_cfg(
        config.protocol_v1.to_dict(),
        seed=args.seed,
        device=args.device,
        runtime_projection=projection,
    )
    env_cfg.scene.num_envs = 1
    env = gym.make(STAIR_EVAL_GYM_ID, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    observation, _ = env.reset(seed=args.seed)
    action = torch.zeros_like(base.action_manager.action)
    observation_after, _, _, _, _ = env.step(action)

    contact_sensor = base.scene[projection.contact_sensor_name]
    resolved = resolve_stair_runtime_projection(
        projection,
        robot=base.scene["robot"],
        contact_sensor=contact_sensor,
    )
    vnext_projection = replace(projection, legacy_reward_adapter_id=None)
    vnext_resolved = resolve_stair_runtime_projection(
        vnext_projection,
        robot=base.scene["robot"],
        contact_sensor=contact_sensor,
    )
    legacy_base_state = stair_kinematic_base_state(
        base.scene["robot"].data,
        runtime_projection=projection,
        resolved_projection=resolved,
    )
    vnext_base_state = stair_kinematic_base_state(
        base.scene["robot"].data,
        runtime_projection=vnext_projection,
        resolved_projection=vnext_resolved,
    )
    kinematic_base_parity = {
        name: float(
            torch.max(
                torch.abs(
                    getattr(legacy_base_state, name)
                    - getattr(vnext_base_state, name)
                )
            ).item()
        )
        for name in (
            "pos_w",
            "quat_w",
            "lin_vel_w",
            "ang_vel_w",
            "lin_vel_b",
            "ang_vel_b",
            "projected_gravity_b",
        )
    }
    if any(value != 0.0 for value in kinematic_base_parity.values()):
        raise RuntimeError("legacy/VNext G1 kinematic base selection drifted")
    raw_transition = base.extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY]
    if raw_transition.get("schema_version") != STAIR_TRANSITION_RECORDER_SCHEMA_VERSION:
        raise RuntimeError("legacy G1 recorder schema changed during Phase 2 probe")
    normalized = normalize_stair_transition_payload(raw_transition)

    builder = StairRewardInputBuilder(
        base,
        contract=reference.contract,
        base_protocol=config.protocol_v1.to_dict(),
        runtime_projection=projection,
    )
    reward_inputs = builder.build()
    if type(reward_inputs) is not StairRewardInputsV2:
        raise RuntimeError("legacy G1 reward adapter did not preserve V2")
    evaluation = evaluate_stair_reward_program(
        reference,
        reward_inputs,
        num_envs=1,
    )
    if not bool(torch.isfinite(evaluation.total_rate).all().item()):
        raise RuntimeError("Phase 2 reference reward output is non-finite")

    source_paths = (
        "src/true_rl_agent/robots/base.py",
        "src/true_rl_agent/composition/cell_manifest.py",
        "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
        "src/true_rl_agent/tasks/stair_climbing/legacy_reward_adapter.py",
        "src/true_rl_agent/tasks/stair_climbing/legacy_runtime_projection.py",
        "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
        "src/true_rl_agent/tasks/stair_climbing/evaluation_semantics.py",
        "src/true_rl_agent/tasks/stair_climbing/reward_inputs.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
        "src/true_rl_agent/tasks/stair_climbing/legacy_transition_reader.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
        "src/true_rl_agent/tasks/stair_climbing/transition_decoder.py",
        "src/true_rl_agent/tasks/stair_climbing/episode_evidence.py",
        "src/true_rl_agent/tasks/stair_climbing/terrain_evidence.py",
        "configs/tasks/stair_g1_37dof_clean_legacy_binding_v1.json",
        "configs/tasks/stair_evaluation_semantics_v7.json",
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "capture_kind": "real_headless_g1_scene_reward_and_recorder",
        "captured_repository_head": _captured_head(
            project_root,
            args.captured_repository_head,
        ),
        "head_identity_semantics": "capture provenance only",
        "environment": {
            "python": ".".join(str(item) for item in sys.version_info[:3]),
            "device": str(args.device),
            "torch": str(torch.__version__),
            "cuda_device_name": torch.cuda.get_device_name(args.device),
        },
        "cell": {
            "cell_variant_id": manifest.cell_variant_id,
            "manifest_hash": manifest.manifest_hash,
            "robot_profile_id": projection.robot_profile_id,
            "runtime_adapter_id": projection.runtime_adapter_id,
        },
        "runtime_projection": resolved.to_summary_dict(),
        "kinematic_base_parity": {
            "source": projection.kinematic_base_source,
            "legacy_reward_adapter_id_removed_only": True,
            "maximum_absolute_differences": kinematic_base_parity,
            "status": "exact",
        },
        "observation_before_step": _observation_summary(torch, observation),
        "observation_after_step": _observation_summary(torch, observation_after),
        "action": _tensor_summary(torch, action),
        "transition": {
            "source_schema_version": raw_transition["schema_version"],
            "normalized_schema_version": normalized["schema_version"],
            "joint_pos": _tensor_summary(torch, normalized["joint_pos"]),
            "support_pos_w": _tensor_summary(torch, normalized["support_pos_w"]),
            "support_contact_forces_w": _tensor_summary(
                torch,
                normalized["support_contact_forces_w"],
            ),
            "forbidden_contact_forces_w": _tensor_summary(
                torch,
                normalized["forbidden_contact_forces_w"],
            ),
        },
        "reward": {
            "input_type": type(reward_inputs).__name__,
            "input_schema_hash": reward_inputs.schema_hash,
            "field_count": len(reward_inputs.field_names),
            "term_names": list(reference.term_names),
            "total_rate": _tensor_summary(torch, evaluation.total_rate),
            "total_rate_mean": float(evaluation.total_rate.mean().item()),
            "runtime_input_fingerprint": builder.last_input_fingerprint,
        },
        "product_source_hash_mode": "normalized_lf_sha256",
        "product_source_hashes": {
            path: _normalized_text_hash(project_root / path) for path in source_paths
        },
        "status": "pass",
    }
    if not math.isfinite(receipt["reward"]["total_rate_mean"]):
        raise RuntimeError("Phase 2 reward receipt contains a non-finite mean")
    receipt["receipt_hash"] = _canonical_hash(receipt)
    encoded = (
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "pass", "receipt_hash": receipt["receipt_hash"]}))
    sys.stdout.flush()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=1701, type=int)
    parser.add_argument("--captured-repository-head")
    args = parser.parse_args()
    try:
        _capture(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # The frozen host lacks optional libGLU; Kit shutdown may hide a completed
    # probe result. This one-shot process has fsynced the receipt and owns no
    # persistent state, so process exit is the deterministic cleanup boundary.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
