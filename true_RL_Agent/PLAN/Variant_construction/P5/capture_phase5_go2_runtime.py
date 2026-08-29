from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

RECEIPT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase5_go2_runtime.v1"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _normalized_text_hash(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"runtime receipt contains {type(value).__name__}")


def _resolve_go2_cell(project_root: Path) -> tuple[Any, Any]:
    from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
    from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
    from true_rl_agent.tasks.stair_climbing.robot_binding import (
        STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
        resolve_stair_cell_manifest,
    )

    protocol_path = (
        project_root / "configs/tasks/stair_go2_12dof_unified_protocol_v1.json"
    )
    provisional = StairClimbingTaskConfig.defaults(protocol_path)
    if provisional.cell_variant_id != STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID:
        raise RuntimeError("Phase 5 protocol did not resolve the Go2 row")
    manifest = resolve_stair_cell_manifest(
        cell_variant_id=provisional.cell_variant_id,
        config=provisional,
        plugin=StairClimbingTaskPlugin(provisional),
    )
    config = StairClimbingTaskConfig.defaults(
        protocol_path,
        resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash,
    )
    plugin = StairClimbingTaskPlugin(config, resolved_cell_manifest=manifest)
    rebound = resolve_stair_cell_manifest(
        cell_variant_id=config.cell_variant_id,
        config=config,
        plugin=plugin,
    )
    if rebound.to_dict() != manifest.to_dict():
        raise RuntimeError("Phase 5 Go2 manifest changed after identity binding")
    if config.protocol_state != "active":
        raise RuntimeError("Phase 5 Go2 protocol is not active")
    return config, manifest


def _geometry_probe(config: Any) -> dict[str, Any]:
    from true_rl_agent.tasks.stair_climbing.isaaclab_evaluation_terrain import (
        build_ordered_eval_terrain_generator,
    )

    cells = config.ordered_evaluation_suite.runtime_cells("public_dev")
    rows = []
    for regime in ("c0_regular", "c1_irregular_flat", "c2_rough"):
        cell = next(item for item in cells if item.terrain_regime_id == regime)
        generator = build_ordered_eval_terrain_generator(cells=(cell,), seed=5100)
        terrain_cfg = generator.sub_terrains[cell.cell_id]
        terrain_cfg.size = generator.size
        meshes, origin = terrain_cfg.function(0.0, terrain_cfg)
        if not meshes or terrain_cfg.geometry_digest != cell.geometry_digest:
            raise RuntimeError("Phase 5 Go2 evaluation mesh identity drifted")
        rows.append(
            {
                "terrain_regime_id": regime,
                "cell_id": cell.cell_id,
                "geometry_digest": cell.geometry_digest,
                "mesh_count": len(meshes),
                "origin": [round(float(value), 6) for value in origin],
            }
        )
    return {"status": "pass", "representatives": rows}


def _reward_equivalence_probe(
    config: Any,
    manifest: Any,
    official: Any,
    *,
    device: str,
) -> dict[str, Any]:
    import gymnasium as gym
    import torch

    from true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime import (
        StairRewardInputBuilder,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_eval_env_cfg,
        register_production_stair_envs,
    )
    from true_rl_agent.tasks.stair_climbing.runtime_profile import STAIR_EVAL_GYM_ID
    from true_rl_agent.tasks.stair_climbing.runtime_projection import (
        stair_runtime_projection_from_manifest,
    )

    projection = stair_runtime_projection_from_manifest(manifest)
    register_production_stair_envs()
    env_cfg = build_stair_eval_env_cfg(
        None,
        seed=5101,
        device=device,
        ordered_evaluation_suite=config.ordered_evaluation_suite,
        evaluation_split="public_dev",
        runtime_projection=projection,
        unified_protocol=config.unified_protocol,
    )
    created = gym.make(STAIR_EVAL_GYM_ID, cfg=env_cfg, render_mode=None)
    env = created.unwrapped
    maximum_raw = {term.name: 0.0 for term in official.reward_program.reward_terms}
    maximum_weighted = {term.name: 0.0 for term in official.reward_program.reward_terms}
    nonzero_raw_samples = {term.name: 0 for term in official.reward_program.reward_terms}
    maximum_total_rate_difference = 0.0
    maximum_manager_step_difference = 0.0
    sample_count = 0
    try:
        observation, _ = created.reset()
        if not isinstance(observation, Mapping) or "policy" not in observation:
            raise RuntimeError("Phase 5 Go2 policy observation group is unavailable")
        policy_observation_dimension = int(observation["policy"].shape[-1])
        builder = StairRewardInputBuilder(
            env,
            contract=official.reward_program.contract,
            base_protocol=config.runtime_task_contract(),
            runtime_projection=projection,
        )
        program_terms = {
            term.name: term for term in official.reward_program.reward_terms
        }
        expected_active = list(
            config.vnext_reward_authority.runtime_authority_payload()[
                "reference_equivalence"
            ]["active_term_names"]
        )
        if list(program_terms) != expected_active:
            raise RuntimeError("Phase 5 Go2 reference term order drifted")
        reward_manager = env.reward_manager
        if not set(expected_active) <= set(reward_manager.active_terms):
            raise RuntimeError("official Go2 reward manager is missing active terms")
        if "undesired_contacts" in reward_manager.active_terms:
            raise RuntimeError("official Go2 disabled contact term became active")
        zero_weights = {
            name: float(reward_manager.get_term_cfg(name).weight)
            for name in ("flat_orientation_l2", "dof_pos_limits")
        }
        if zero_weights != {"flat_orientation_l2": 0.0, "dof_pos_limits": 0.0}:
            raise RuntimeError("official Go2 zero-weight terms drifted")

        torch.manual_seed(5102)
        for _ in range(40):
            actions = 0.6 * torch.randn(
                (env.num_envs, projection.action_count),
                dtype=torch.float32,
                device=device,
            )
            created.step(actions)
            inputs = builder.build()
            official_total = torch.zeros(env.num_envs, device=device)
            program_total = torch.zeros_like(official_total)
            for name, program_term in program_terms.items():
                term_cfg = reward_manager.get_term_cfg(name)
                if not math.isclose(
                    float(term_cfg.weight),
                    float(program_term.weight),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise RuntimeError("Phase 5 Go2 reward weight drifted: " + name)
                official_raw = term_cfg.func(env, **term_cfg.params)
                program_raw = program_term.fn(inputs)
                raw_difference = float(
                    (official_raw - program_raw).abs().max().item()
                )
                weighted_difference = float(
                    (
                        official_raw * float(term_cfg.weight)
                        - program_raw * float(program_term.weight)
                    )
                    .abs()
                    .max()
                    .item()
                )
                maximum_raw[name] = max(maximum_raw[name], raw_difference)
                maximum_weighted[name] = max(
                    maximum_weighted[name], weighted_difference
                )
                nonzero_raw_samples[name] += int(
                    (program_raw.abs() > 1.0e-12).any().item()
                )
                official_total += official_raw * float(term_cfg.weight)
                program_total += program_raw * float(program_term.weight)
            maximum_total_rate_difference = max(
                maximum_total_rate_difference,
                float((official_total - program_total).abs().max().item()),
            )
            manager_step_reward = reward_manager.compute(env.step_dt)
            maximum_manager_step_difference = max(
                maximum_manager_step_difference,
                float(
                    (
                        manager_step_reward - program_total * float(env.step_dt)
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
            sample_count += int(env.num_envs)

        tolerance = 1.0e-6
        if (
            max(maximum_raw.values()) > tolerance
            or max(maximum_weighted.values()) > tolerance
            or maximum_total_rate_difference > tolerance
            or maximum_manager_step_difference > tolerance
        ):
            raise RuntimeError("Phase 5 Go2 reward is not numerically equivalent")
        if nonzero_raw_samples["feet_air_time"] == 0:
            raise RuntimeError("Phase 5 Go2 air-time equivalence had no active sample")

        robot = env.scene["robot"]
        height_sensor = env.scene[projection.height_sensor_name]
        runtime_schema = {
            "joint_count": len(robot.data.joint_names),
            "body_count": len(robot.data.body_names),
            "support_count": projection.support_count,
            "action_count": int(env.action_manager.action.shape[-1]),
            "policy_observation_dimension": policy_observation_dimension,
            "height_scan_ray_count": int(height_sensor.data.ray_hits_w.shape[1]),
            "default_root_height_m": round(
                float(robot.data.default_root_state[0, 2].item()), 6
            ),
        }
        if runtime_schema != {
            "joint_count": 12,
            "body_count": 19,
            "support_count": 4,
            "action_count": 12,
            "policy_observation_dimension": 235,
            "height_scan_ray_count": 187,
            "default_root_height_m": 0.4,
        }:
            raise RuntimeError("Phase 5 Go2 live runtime schema drifted")
        return {
            "status": "pass",
            "sample_count": sample_count,
            "control_step_count": 40,
            "maximum_absolute_raw_difference_by_term": maximum_raw,
            "maximum_absolute_weighted_difference_by_term": maximum_weighted,
            "maximum_absolute_total_rate_difference": maximum_total_rate_difference,
            "maximum_absolute_manager_step_difference": (
                maximum_manager_step_difference
            ),
            "nonzero_control_steps_by_term": nonzero_raw_samples,
            "runtime_schema": runtime_schema,
        }
    finally:
        created.close()


def _capture(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve(strict=True)
    output = args.output.resolve()
    work_dir = args.work_dir.resolve()
    if output.exists() or work_dir.exists():
        raise FileExistsError("Phase 5 receipt and work directory must be fresh")
    sys.path.insert(0, str(project_root / "src"))

    config, manifest = _resolve_go2_cell(project_root)
    from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
        execute_stair_training_smoke,
        preflight_stair_execution_source_manifest,
    )
    from true_rl_agent.tasks.stair_climbing.reference_program import (
        resolve_official_stair_reference_program,
    )

    source_preflight = preflight_stair_execution_source_manifest(config)
    official = resolve_official_stair_reference_program(
        config,
        project_root=project_root,
        resolved_cell_manifest=manifest,
    )

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device=args.device)
    _ = launcher.app
    import torch

    started = time.perf_counter()
    geometry = _geometry_probe(config)
    equivalence = _reward_equivalence_probe(
        config,
        manifest,
        official,
        device=args.device,
    )
    work_dir.mkdir(parents=True, exist_ok=False)
    training = execute_stair_training_smoke(
        config=config,
        source_manifest_preflight=source_preflight,
        agent_configuration=config.training_agent_configuration(),
        reward_program=official.reward_program,
        device=args.device,
        run_dir=work_dir / "train",
        checkpoint_output=work_dir / "checkpoint/model_0.pt",
        run_name="phase5_go2_unified_smoke",
        resolved_cell_manifest=manifest,
    )
    training_summary = _json_value(training.to_summary_fields())
    if training.status != "pass":
        raise RuntimeError("Phase 5 Go2 training smoke failed")
    activation = training_summary["runtime_activation_summary"]
    if (
        activation["declared_term_count"] != 8
        or activation["dispatcher_calls"] <= 1
        or activation["nonzero_reward_contribution"] is not True
        or not math.isfinite(float(activation["last_total_rate_mean"]))
    ):
        raise RuntimeError("Phase 5 Go2 reward activation is degenerate")
    assignment = training_summary["training"]["initial_terrain_assignment"]
    if assignment["regime_counts"] != {
        "c0_regular": 8,
        "c1_irregular_flat": 8,
    }:
        raise RuntimeError("Phase 5 Go2 train assignment is not 50/50")

    source_paths = (
        "configs/robots/unitree_go2_12dof_isaaclab_v1.json",
        "configs/tasks/stair_go2_12dof_unified_binding_v1.json",
        "configs/tasks/stair_go2_12dof_unified_training_profile_v1.json",
        "configs/tasks/stair_go2_12dof_unified_evaluation_suite_v1.json",
        "configs/tasks/stair_go2_12dof_vnext_reward_authority_v1.json",
        "configs/tasks/stair_go2_12dof_unified_protocol_v1.json",
        "src/true_rl_agent/robots/isaaclab_locomotion.py",
        "src/true_rl_agent/tasks/stair_climbing/row_catalog.py",
        "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
        "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py",
        "src/true_rl_agent/tasks/stair_climbing/reference_programs/official_go2_rough_vnext_v1/training_code/components.py",
        "PLAN/Variant_construction/P5/capture_phase5_go2_runtime.py",
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "capture_kind": "real_headless_go2_unified_stair_runtime_and_reward_oracle",
        "environment": {
            "python": ".".join(str(item) for item in sys.version_info[:3]),
            "device": args.device,
            "torch": str(torch.__version__),
            "cuda_device_name": torch.cuda.get_device_name(args.device),
        },
        "cell": {
            "cell_variant_id": manifest.cell_variant_id,
            "manifest_hash": manifest.manifest_hash,
            "robot_profile_hash": manifest.robot_profile["profile_hash"],
            "runtime_adapter_hash": manifest.runtime_adapter["adapter_hash"],
            "binding_hash": manifest.robot_task_binding["binding_hash"],
            "base_task_protocol_hash": config.base_task_protocol_hash,
            "resolved_cell_task_protocol_hash": (
                manifest.resolved_cell_task_protocol_hash
            ),
            "training_profile_hash": config.unified_training_profile.profile_hash,
            "evaluation_suite_hash": config.ordered_evaluation_suite.suite_hash,
            "vnext_reward_authority_hash": config.vnext_reward_authority.authority_hash,
            "reference_program_source_checksum": (
                official.reward_program.program_source_checksum
            ),
        },
        "geometry_probe": geometry,
        "reference_reward_equivalence": equivalence,
        "training_smoke": training_summary,
        "source_preflight": {
            "summary": source_preflight.to_summary_dict(),
            "evidence": source_preflight.upstream_source_evidence(),
        },
        "eligibility": {
            "reference_qualification": False,
            "candidate_acceptance": False,
            "reason": "phase5_runtime_smoke_and_oracle_only",
        },
        "cost": {
            "wall_time_s": round(time.perf_counter() - started, 6),
            "smoke_budget_steps": int(training_summary["training"]["budget_steps"]),
            "full_training_budget_steps": int(config.budget_steps),
        },
        "product_source_hash_mode": "normalized_lf_sha256",
        "product_source_hashes": {
            path: _normalized_text_hash(project_root / path) for path in source_paths
        },
        "status": "pass",
    }
    receipt["receipt_hash"] = _canonical_hash(receipt)
    encoded = (
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "status": "pass",
                "receipt_hash": receipt["receipt_hash"],
                "reward_max_abs_difference": max(
                    equivalence["maximum_absolute_raw_difference_by_term"].values()
                ),
                "smoke_budget_steps": receipt["cost"]["smoke_budget_steps"],
                "wall_time_s": receipt["cost"]["wall_time_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        _capture(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
