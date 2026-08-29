"""Capture one real C1 Isaac scene and one bounded PPO update.

This is an acceptance probe, not a second train/eval implementation.  It calls
the production task-owned environment, decoder, reward and training execution
body with the resolved irregular-cell manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase3a_runtime_receipt.v1"
)


def _mark(stage: str) -> None:
    print(f"phase3a_runtime_probe stage={stage}", flush=True)


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


def _captured_head(project_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and len(value) == 40 else None


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"runtime receipt contains a non-JSON value: {type(value).__name__}")


def _resolve_irregular_cell(project_root: Path) -> tuple[Any, Any]:
    from true_rl_agent.tasks.stair_climbing.config import (
        STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
        STAIR_PROTOCOL_V2_RELATIVE_PATH,
        StairClimbingTaskConfig,
    )
    from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
    from true_rl_agent.tasks.stair_climbing.robot_binding import (
        resolve_stair_cell_manifest,
    )

    protocol_path = project_root / STAIR_PROTOCOL_V2_RELATIVE_PATH
    provisional_config = StairClimbingTaskConfig.defaults(
        protocol_path,
        cell_variant_id=STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
    )
    provisional_manifest = resolve_stair_cell_manifest(
        cell_variant_id=STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
        config=provisional_config,
        plugin=StairClimbingTaskPlugin(provisional_config),
    )
    config = StairClimbingTaskConfig.defaults(
        protocol_path,
        cell_variant_id=STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
        resolved_cell_task_protocol_hash=(
            provisional_manifest.resolved_cell_task_protocol_hash
        ),
    )
    plugin = StairClimbingTaskPlugin(
        config,
        resolved_cell_manifest=provisional_manifest,
    )
    manifest = resolve_stair_cell_manifest(
        cell_variant_id=STAIR_G1_37DOF_IRREGULAR_CELL_VARIANT_ID,
        config=config,
        plugin=plugin,
    )
    if manifest.to_dict() != provisional_manifest.to_dict():
        raise RuntimeError("C1 manifest changed after resolved task identity was applied")
    if (
        plugin.task_spec.task_protocol_hash
        != config.formal_training_binding().task_protocol_hash
        or plugin.task_spec.task_protocol_hash
        != manifest.resolved_cell_task_protocol_hash
    ):
        raise RuntimeError("C1 TaskSpec/TrainingBindingV2/manifest identity diverged")
    return config, manifest


def _mesh_oracle_receipt(config: Any) -> dict[str, Any]:
    import numpy as np

    from true_rl_agent.tasks.stair_climbing.isaaclab_irregular_terrain import (
        IrregularOneWayStairsTerrainCfg,
        irregular_one_way_stairs_terrain,
    )
    from true_rl_agent.tasks.stair_climbing.terrain_evidence import (
        IrregularFlatStairSurfaceQuery,
    )

    cell_receipts: list[dict[str, Any]] = []
    maximum_error = 0.0
    for cell in config.ordered_evaluation_suite.cells:
        terrain_cfg = IrregularOneWayStairsTerrainCfg(
            function=irregular_one_way_stairs_terrain,
            proportion=1.0,
            size=cell.size_m,
            cell_id=cell.cell_id,
            geometry_digest=cell.geometry_digest,
            floor_depth=cell.floor_depth_m,
            start_platform_length=cell.start_platform_length_m,
            riser_heights_m=cell.riser_heights_m,
            tread_depths_m=cell.tread_depths_m,
            top_platform_length=cell.top_platform_length_m,
            top_platform_goal_offset_m=cell.top_platform_goal_offset_m,
        )
        meshes, origin = irregular_one_way_stairs_terrain(0.5, terrain_cfg)
        if len(meshes) != cell.step_count + 2:
            raise RuntimeError("C1 mesh count differs from floor + steps + top")
        expected_origin = np.asarray(
            [cell.start_platform_length_m / 2.0, cell.size_m[1] / 2.0, 0.0],
            dtype=np.float64,
        )
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(origin - expected_origin))),
        )
        query = IrregularFlatStairSurfaceQuery(cell)
        segment_start = cell.start_platform_length_m
        cumulative_height = 0.0
        boundary_checks = 0
        for index, (mesh, tread, riser) in enumerate(
            zip(
                meshes[1:-1],
                cell.tread_depths_m,
                cell.riser_heights_m,
                strict=True,
            )
        ):
            cumulative_height += riser
            bounds = mesh.bounds
            expected_bounds = np.asarray(
                [
                    [segment_start, 0.0, -cell.floor_depth_m],
                    [segment_start + tread, cell.size_m[1], cumulative_height],
                ],
                dtype=np.float64,
            )
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(bounds - expected_bounds))),
            )
            center_x = segment_start + tread / 2.0
            sample = query.sample(
                progress_m=center_x - origin[0],
                lateral_m=0.0,
            )
            maximum_error = max(
                maximum_error,
                abs(sample.height_m - cumulative_height),
            )
            if sample.level != index + 1:
                raise RuntimeError("C1 analytic query selected the wrong tread")
            epsilon = 1.0e-7
            below = query.sample(
                progress_m=segment_start - epsilon - origin[0],
                lateral_m=0.0,
            )
            above = query.sample(
                progress_m=segment_start + epsilon - origin[0],
                lateral_m=0.0,
            )
            if below.level != index or above.level != index + 1:
                raise RuntimeError("C1 analytic query changed level away from an edge")
            boundary_checks += 2
            segment_start += tread

        top_bounds = meshes[-1].bounds
        expected_top_bounds = np.asarray(
            [
                [segment_start, 0.0, -cell.floor_depth_m],
                [cell.size_m[0], cell.size_m[1], cumulative_height],
            ],
            dtype=np.float64,
        )
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(top_bounds - expected_top_bounds))),
        )
        cell_receipts.append(
            {
                "cell_id": cell.cell_id,
                "cohort": cell.cohort,
                "geometry_digest": cell.geometry_digest,
                "mesh_count": len(meshes),
                "boundary_side_checks": boundary_checks,
                "total_height_m": cell.total_height_m,
                "total_run_m": cell.total_run_m,
                "top_platform_length_m": cell.top_platform_length_m,
            }
        )
    if maximum_error > 1.0e-9:
        raise RuntimeError(f"C1 mesh/oracle maximum error is too large: {maximum_error}")
    closure = {
        "suite_hash": config.ordered_evaluation_suite.suite_hash,
        "cells": cell_receipts,
    }
    return {
        "cell_count": len(cell_receipts),
        "maximum_absolute_error_m": maximum_error,
        "geometry_closure_hash": _canonical_hash(closure),
        "cells": cell_receipts,
        "status": "pass",
    }


def _eval_scene_receipt(config: Any, manifest: Any, *, device: str, seed: int) -> dict[str, Any]:
    import gymnasium as gym
    import torch

    from true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env import (
        build_stair_eval_env_cfg,
        register_production_stair_envs,
    )
    from true_rl_agent.tasks.stair_climbing.runtime_profile import (
        STAIR_EVAL_GYM_ID,
        STAIR_EVAL_TRANSITION_EXTRAS_KEY,
    )
    from true_rl_agent.tasks.stair_climbing.runtime_projection import (
        resolve_stair_runtime_projection,
        stair_runtime_projection_from_manifest,
    )
    from true_rl_agent.tasks.stair_climbing.transition_decoder import (
        decode_stair_transition_batch,
    )

    split = "public_dev"
    projection = stair_runtime_projection_from_manifest(manifest)
    suite = config.ordered_evaluation_suite
    assignment = suite.private_cell_assignment(split)
    env_cfg = build_stair_eval_env_cfg(
        config.protocol_v1.to_dict(),
        seed=seed,
        device=device,
        ordered_evaluation_suite=suite,
        evaluation_split=split,
        runtime_projection=projection,
    )
    register_production_stair_envs()
    _mark("evaluation_scene_make_started")
    env = gym.make(STAIR_EVAL_GYM_ID, cfg=env_cfg, render_mode=None)
    try:
        base = env.unwrapped
        _mark("evaluation_scene_reset_started")
        observation, _ = env.reset(seed=seed)
        _mark("evaluation_scene_reset_completed")
        action = torch.zeros_like(base.action_manager.action)
        _mark("evaluation_scene_step_started")
        _observation, reward, terminated, truncated, extras = env.step(action)
        _mark("evaluation_scene_step_completed")
        raw = extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY]
        _mark("evaluation_scene_decode_started")
        transitions = decode_stair_transition_batch(
            raw,
            wrapper_done_mask=terminated | truncated,
            protocol=config.protocol_v1,
            semantics=config.evaluation_semantics,
            expected_bin_assignment=tuple(cell.cell_id for cell in assignment),
            expected_cell_assignment=assignment,
            runtime_projection=projection,
        )
        _mark("evaluation_scene_decode_completed")
        robot = base.scene["robot"]
        contact_sensor = base.scene[projection.contact_sensor_name]
        resolved_projection = resolve_stair_runtime_projection(
            projection,
            robot=robot,
            contact_sensor=contact_sensor,
        )
        terrain = base.scene.terrain
        terrain_types = terrain.terrain_types.detach().cpu().tolist()
        origins = terrain.env_origins.detach().cpu().tolist()
        expected_types = [cell.terrain_type_index for cell in assignment]
        if terrain_types != expected_types:
            raise RuntimeError("C1 runtime terrain type assignment drifted")
        return {
            "split": split,
            "num_envs": len(transitions),
            "ordered_cell_ids": [cell.cell_id for cell in assignment],
            "terrain_types": terrain_types,
            "terrain_origins_m": [
                [round(float(value), 7) for value in origin] for origin in origins
            ],
            "runtime_projection": resolved_projection.to_summary_dict(),
            "observation_shape": {
                str(name): [int(item) for item in value.shape]
                for name, value in observation.items()
            },
            "action_shape": [int(item) for item in action.shape],
            "reward_finite": bool(torch.isfinite(reward).all().item()),
            "decoded_transition_count": len(transitions),
            "transition_cell_ids": [item.evaluation_cell_id for item in transitions],
            "status": "pass",
        }
    finally:
        _mark("evaluation_scene_close_started")
        env.close()
        _mark("evaluation_scene_close_completed")


def _capture(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve(strict=True)
    output = args.output.resolve()
    work_dir = args.work_dir.resolve()
    if output.exists() or (
        args.probe_mode == "training_smoke" and work_dir.exists()
    ):
        raise FileExistsError("Phase 3A runtime receipt and work directory must be fresh")
    sys.path.insert(0, str(project_root / "src"))

    config, manifest = _resolve_irregular_cell(project_root)
    _mark("resolved_cell_completed")
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
    _mark("reference_preflight_completed")

    from isaaclab.app import AppLauncher

    _mark("app_launcher_started")
    launcher = AppLauncher(headless=True, device=args.device)
    _ = launcher.app
    _mark("app_launcher_completed")

    import torch

    started = time.perf_counter()
    _mark("mesh_oracle_started")
    mesh_oracle = _mesh_oracle_receipt(config)
    _mark("mesh_oracle_completed")
    eval_scene: dict[str, Any] | None = None
    training_summary: dict[str, Any] | None = None
    if args.probe_mode == "evaluation_scene":
        _mark("evaluation_scene_started")
        eval_scene = _eval_scene_receipt(
            config,
            manifest,
            device=args.device,
            seed=args.seed,
        )
        _mark("evaluation_scene_completed")
    else:
        work_dir.mkdir(parents=True, exist_ok=False)
        _mark("training_smoke_started")
        training = execute_stair_training_smoke(
            config=config,
            source_manifest_preflight=source_preflight,
            agent_configuration=config.training_agent_configuration(),
            reward_program=official.reward_program,
            device=args.device,
            run_dir=work_dir / "train",
            checkpoint_output=work_dir / "checkpoint" / "model_0.pt",
            run_name="phase3a_c1_irregular_runtime_smoke",
            resolved_cell_manifest=manifest,
        )
        _mark("training_smoke_completed")
        training_summary = training.to_summary_fields()
        if training.status != "pass":
            raise RuntimeError(
                f"C1 production training smoke failed: {training_summary}"
            )

    source_paths = (
        "configs/robots/unitree_g1_37dof_isaaclab_v2.json",
        "src/true_rl_agent/robots/defaults.py",
        "src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py",
        "src/true_rl_agent/tasks/stair_climbing/irregular_training.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_irregular_terrain.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_evaluation_terrain.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
        "src/true_rl_agent/tasks/stair_climbing/transition_decoder.py",
        "src/true_rl_agent/tasks/stair_climbing/terrain_evidence.py",
        "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
        "src/true_rl_agent/tasks/stair_climbing/reward_source_semantics.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
        "src/true_rl_agent/tasks/stair_climbing/vnext_reward_contract.py",
        "src/true_rl_agent/tasks/stair_climbing/reference_programs/g1_irregular_v1/training_code/components.py",
        "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
        "configs/tasks/stair_g1_37dof_irregular_evaluation_suite_v1.json",
        "configs/tasks/stair_g1_37dof_irregular_training_profile_v1.json",
        "PLAN/Variant_construction/P3A/capture_phase3a_runtime_receipt.py",
    )
    elapsed = time.perf_counter() - started
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "capture_kind": (
            "real_headless_c1_scene_and_decoder"
            if args.probe_mode == "evaluation_scene"
            else "real_headless_c1_reward_and_ppo_smoke"
        ),
        "probe_mode": args.probe_mode,
        "captured_repository_head": _captured_head(project_root),
        "head_identity_semantics": "capture provenance only; product bytes are authoritative",
        "environment": {
            "python": ".".join(str(item) for item in sys.version_info[:3]),
            "device": args.device,
            "torch": str(torch.__version__),
            "cuda_device_name": torch.cuda.get_device_name(args.device),
        },
        "cell": {
            "cell_variant_id": manifest.cell_variant_id,
            "manifest_hash": manifest.manifest_hash,
            "resolved_cell_task_protocol_hash": (
                manifest.resolved_cell_task_protocol_hash
            ),
            "ordered_evaluation_suite_hash": (
                config.ordered_evaluation_suite.suite_hash
            ),
            "irregular_training_profile_hash": (
                config.irregular_training_profile.profile_hash
            ),
            "reference_program_source_checksum": (
                official.reward_program.program_source_checksum
            ),
            "reward_input_schema_hash_vnext": (
                manifest.contract_hashes["reward_input_vnext_schema_hash"]
            ),
        },
        "mesh_oracle": mesh_oracle,
        "evaluation_scene": eval_scene,
        "training_smoke": (
            _json_value(training_summary) if training_summary is not None else None
        ),
        "cost": {
            "wall_time_s": round(elapsed, 6),
            "training_budget_steps": (
                128 if args.probe_mode == "training_smoke" else 0
            ),
            "evaluation_scene_steps": (
                1 if args.probe_mode == "evaluation_scene" else 0
            ),
        },
        "product_source_hash_mode": "normalized_lf_sha256",
        "product_source_hashes": {
            path: _normalized_text_hash(project_root / path) for path in source_paths
        },
        "status": "pass",
    }
    if not math.isfinite(receipt["cost"]["wall_time_s"]):
        raise RuntimeError("C1 runtime wall time is not finite")
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
                "geometry_closure_hash": mesh_oracle["geometry_closure_hash"],
                "wall_time_s": receipt["cost"]["wall_time_s"],
            }
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
    parser.add_argument("--seed", default=1731, type=int)
    parser.add_argument(
        "--probe-mode",
        choices=("evaluation_scene", "training_smoke"),
        required=True,
    )
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
