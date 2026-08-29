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

RECEIPT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3b_runtime.v1"


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
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"runtime receipt contains {type(value).__name__}")


def _resolve_unified_cell(project_root: Path) -> tuple[Any, Any]:
    from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
    from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
    from true_rl_agent.tasks.stair_climbing.robot_binding import (
        resolve_stair_cell_manifest,
    )

    protocol_path = project_root / "configs/tasks/stair_unified_protocol_v1.json"
    provisional = StairClimbingTaskConfig.defaults(protocol_path)
    manifest = resolve_stair_cell_manifest(
        cell_variant_id=provisional.cell_variant_id,
        config=provisional,
        plugin=StairClimbingTaskPlugin(provisional),
    )
    config = StairClimbingTaskConfig.defaults(
        protocol_path,
        resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash,
    )
    resolved = resolve_stair_cell_manifest(
        cell_variant_id=config.cell_variant_id,
        config=config,
        plugin=StairClimbingTaskPlugin(config, resolved_cell_manifest=manifest),
    )
    if resolved.to_dict() != manifest.to_dict():
        raise RuntimeError("Phase 3B unified manifest changed after identity binding")
    if config.protocol_state not in {"draft", "active"}:
        raise RuntimeError("Phase 3B requires the unified Stair protocol")
    return config, manifest


def _geometry_probe(config: Any) -> dict[str, Any]:
    import numpy as np

    from true_rl_agent.tasks.stair_climbing.isaaclab_evaluation_terrain import (
        build_ordered_eval_terrain_generator,
    )
    from true_rl_agent.tasks.stair_climbing.terrain_evidence import (
        RoughStairSurfaceQuery,
    )

    suite = config.ordered_evaluation_suite
    rows: list[dict[str, Any]] = []
    maximum_error = 0.0
    maximum_offset = 0.0
    for cohort_index, cohort in enumerate(
        ("public_dev", "public_accept", "sealed_validation"), start=1
    ):
        cells = suite.runtime_cells(cohort)
        generator = build_ordered_eval_terrain_generator(
            cells=cells,
            seed=cohort_index,
        )
        for cell in cells:
            if cell.terrain_regime_id != "c2_rough":
                continue
            terrain_cfg = generator.sub_terrains[cell.cell_id]
            terrain_cfg.size = generator.size
            meshes, origin = terrain_cfg.function(0.0, terrain_cfg)
            query = RoughStairSurfaceQuery(cell.spec)
            nx = int(cell.surface_profile["longitudinal_vertices"])
            ny = int(cell.surface_profile["lateral_vertices"])
            cell_error = 0.0
            cell_offset = 0.0
            watertight = all(bool(mesh.is_watertight) for mesh in meshes)
            for tread_index, mesh in enumerate(meshes[1:-1]):
                nominal = cell.cumulative_riser_heights_m[tread_index + 1]
                top_vertices = np.asarray(mesh.vertices[: nx * ny], dtype=np.float64)
                for ix in range(1, nx - 1):
                    for iy in range(1, ny - 1):
                        x, y, z = top_vertices[ix * ny + iy]
                        sampled = query.sample(
                            progress_m=float(x) - cell.start_platform_length_m / 2.0,
                            lateral_m=float(y) - cell.size_m[1] / 2.0,
                        )
                        error = abs(float(z) - sampled.height_m)
                        cell_error = max(cell_error, error)
                        cell_offset = max(cell_offset, abs(float(z) - nominal))
                boundary_indexes = {
                    *(iy for iy in range(ny)),
                    *((nx - 1) * ny + iy for iy in range(ny)),
                    *(ix * ny for ix in range(nx)),
                    *(ix * ny + ny - 1 for ix in range(nx)),
                }
                if any(
                    abs(float(top_vertices[index, 2]) - nominal) > 1.0e-10
                    for index in boundary_indexes
                ):
                    raise RuntimeError("C2 tread edge changed the nominal stair topology")
            amplitude = float(cell.surface_profile["amplitude_m"])
            if (
                len(meshes) != cell.step_count + 2
                or not watertight
                or cell_error > 1.0e-5
                or cell_offset > amplitude + 1.0e-10
                or not np.allclose(
                    origin,
                    [cell.start_platform_length_m / 2.0, cell.size_m[1] / 2.0, 0.0],
                    atol=1.0e-12,
                )
            ):
                raise RuntimeError("C2 mesh/query geometry closure failed")
            maximum_error = max(maximum_error, cell_error)
            maximum_offset = max(maximum_offset, cell_offset)
            rows.append(
                {
                    "cohort": cohort,
                    "cell_id": cell.cell_id,
                    "geometry_digest": cell.geometry_digest,
                    "amplitude_m": amplitude,
                    "observed_max_abs_offset_m": cell_offset,
                    "mesh_query_max_abs_error_m": cell_error,
                    "mesh_count": len(meshes),
                    "all_meshes_watertight": watertight,
                }
            )
    if len(rows) != 9:
        raise RuntimeError("Phase 3B must contain exactly three C2 cells per cohort")
    return {
        "status": "pass",
        "cell_count": len(rows),
        "hard_max_abs_offset_m": 0.015,
        "observed_max_abs_offset_m": maximum_offset,
        "mesh_query_max_abs_error_m": maximum_error,
        "cells": rows,
    }


def _failure_feedback_calibration(config: Any) -> dict[str, Any]:
    from true_rl_agent.tasks.stair_climbing.episode_evidence import (
        StairEnvTransition,
        StairEpisodeTracker,
    )
    from true_rl_agent.tasks.stair_climbing.terrain_evidence import (
        IrregularFlatStairSurfaceQuery,
        RoughStairSurfaceQuery,
    )

    runtime = config.ordered_evaluation_suite.runtime_cells("calibration")
    flat_cell = next(cell for cell in runtime if cell.template_id == "c1_id_02")
    rough_cell = next(cell for cell in runtime if cell.template_id == "c2_rough_02")
    if (
        flat_cell.riser_heights_m != rough_cell.riser_heights_m
        or flat_cell.tread_depths_m != rough_cell.tread_depths_m
    ):
        raise RuntimeError("failure calibration cells do not share one C1 skeleton")
    flat_query = IrregularFlatStairSurfaceQuery(flat_cell.spec)
    rough_query = RoughStairSurfaceQuery(rough_cell.spec)
    level = 2
    left = flat_cell.cumulative_tread_boundaries_m[level - 1]
    progress = (
        left
        + 0.37 * flat_cell.tread_depths_m[level - 1]
        - flat_cell.start_platform_length_m / 2.0
    )
    lateral = 0.17
    surfaces = [
        query.sample(progress_m=progress, lateral_m=lateral).height_m
        for query in (flat_query, rough_query)
    ]
    outcomes = []
    for cell, surface_height in (
        (flat_cell, surfaces[0]),
        (rough_cell, surfaces[1]),
    ):
        tracker = StairEpisodeTracker(
            None,
            config.evaluation_semantics,
            env_index=0,
            bin_id=cell.cell_id,
            evaluation_cell=cell,
        )
        outcome = None
        for step in range(1, 4):
            outcome = tracker.observe(
                StairEnvTransition(
                    env_index=0,
                    bin_id=cell.cell_id,
                    episode_step=step,
                    progress_m=progress,
                    lateral_m=lateral,
                    base_height_m=surface_height + 0.75,
                    base_clearance_m=0.75,
                    abs_roll_rad=0.0,
                    abs_pitch_rad=0.0,
                    forbidden_contact=False,
                    support_contacts=(False, True),
                    support_progress_m=(progress, progress),
                    support_surface_levels=(level, level),
                    support_lateral_m=(lateral, lateral),
                    support_planar_speeds_mps=(0.0, 0.0),
                    support_surface_clearances_m=(0.02, 0.08),
                    support_upward_forces_n=(0.0, 100.0),
                    done=step == 3,
                    termination_terms=("time_out",) if step == 3 else (),
                    evaluation_cell_id=cell.cell_id,
                )
            )
        if outcome is None:
            raise RuntimeError("failure calibration did not close its episode")
        outcomes.append(outcome)
    if (
        outcomes[0].primary_reason != "insufficient_clearance"
        or outcomes[0].primary_reason != outcomes[1].primary_reason
        or outcomes[0].diagnostic_flags != outcomes[1].diagnostic_flags
    ):
        raise RuntimeError("surface-relative C1/C2 failure classification drifted")

    threshold = float(
        config.episode_outcomes()["signals"]["insufficient_clearance"]["threshold"]
    )
    amplitude = float(rough_cell.surface_profile["amplitude_m"])
    maximum_clearance_delta = 0.0
    flip_count = 0
    out_of_margin_flip_count = 0
    sample_count = 0
    for level in range(1, rough_cell.step_count + 1):
        left = rough_cell.cumulative_tread_boundaries_m[level - 1]
        depth = rough_cell.tread_depths_m[level - 1]
        for u in (0.2, 0.4, 0.6, 0.8):
            sample_progress = (
                left + u * depth - rough_cell.start_platform_length_m / 2.0
            )
            for sample_lateral in (-0.35, -0.1, 0.2, 0.4):
                flat_height = flat_query.sample(
                    progress_m=sample_progress,
                    lateral_m=sample_lateral,
                ).height_m
                rough_height = rough_query.sample(
                    progress_m=sample_progress,
                    lateral_m=sample_lateral,
                ).height_m
                for margin in (-0.012, -0.008, -0.004, 0.0, 0.004, 0.008, 0.012):
                    world_z = flat_height + threshold + margin
                    flat_clearance = world_z - flat_height
                    rough_clearance = world_z - rough_height
                    delta = abs(rough_clearance - flat_clearance)
                    maximum_clearance_delta = max(maximum_clearance_delta, delta)
                    flipped = (flat_clearance < threshold) != (
                        rough_clearance < threshold
                    )
                    flip_count += int(flipped)
                    if flipped and abs(margin) > amplitude + 1.0e-12:
                        out_of_margin_flip_count += 1
                    sample_count += 1
    if (
        maximum_clearance_delta > amplitude + 1.0e-10
        or flip_count <= 0
        or out_of_margin_flip_count != 0
    ):
        raise RuntimeError("C2 fixed-world failure-classification drift is unbounded")
    return {
        "status": "pass",
        "paired_cells": [flat_cell.cell_id, rough_cell.cell_id],
        "surface_relative_primary_reason": outcomes[0].primary_reason,
        "surface_relative_diagnostic_flags": list(outcomes[0].diagnostic_flags),
        "surface_relative_exact_parity": True,
        "clearance_threshold_m": threshold,
        "surface_amplitude_m": amplitude,
        "fixed_world_sample_count": sample_count,
        "fixed_world_flip_count": flip_count,
        "out_of_amplitude_margin_flip_count": out_of_margin_flip_count,
        "maximum_clearance_delta_m": maximum_clearance_delta,
    }


def _scanner_and_decoder_probe(
    config: Any,
    manifest: Any,
    *,
    device: str,
    seed: int,
) -> dict[str, Any]:
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
        stair_runtime_projection_from_manifest,
    )
    from true_rl_agent.tasks.stair_climbing.terrain_evidence import (
        stair_surface_query_for_ordered_cell,
    )
    from true_rl_agent.tasks.stair_climbing.transition_decoder import (
        decode_stair_transition_batch,
    )

    split = "calibration"
    projection = stair_runtime_projection_from_manifest(manifest)
    suite = config.ordered_evaluation_suite
    assignment = suite.private_cell_assignment(split)
    flat_index = next(
        index
        for index, cell in enumerate(assignment)
        if cell.template_id == "c1_id_02"
    )
    rough_index = next(
        index
        for index, cell in enumerate(assignment)
        if cell.template_id == "c2_rough_02"
    )
    flat_cell = assignment[flat_index]
    rough_cell = assignment[rough_index]
    if (
        flat_cell.riser_heights_m != rough_cell.riser_heights_m
        or flat_cell.tread_depths_m != rough_cell.tread_depths_m
    ):
        raise RuntimeError("scanner contrast cells do not share one C1 skeleton")

    env_cfg = build_stair_eval_env_cfg(
        None,
        seed=seed,
        device=device,
        ordered_evaluation_suite=suite,
        evaluation_split=split,
        runtime_projection=projection,
        unified_protocol=config.unified_protocol,
    )
    register_production_stair_envs()
    env = gym.make(STAIR_EVAL_GYM_ID, cfg=env_cfg, render_mode=None)
    try:
        observation, _ = env.reset(seed=seed)
        base = env.unwrapped
        robot = base.scene["robot"]
        terrain = base.scene.terrain
        ids = torch.tensor(
            [flat_index, rough_index], dtype=torch.long, device=base.device
        )
        level = 3
        left = flat_cell.cumulative_tread_boundaries_m[level - 1]
        progress = (
            left
            + 0.47 * flat_cell.tread_depths_m[level - 1]
            - flat_cell.start_platform_length_m / 2.0
        )
        lateral = 0.13
        surfaces = [
            stair_surface_query_for_ordered_cell(cell.spec).sample(
                progress_m=progress,
                lateral_m=lateral,
            ).height_m
            for cell in (flat_cell, rough_cell)
        ]
        pose = robot.data.root_state_w[ids, :7].detach().clone()
        origins = terrain.env_origins[ids]
        pose[:, :3] = origins
        pose[:, 0] += float(progress)
        pose[:, 1] += float(lateral)
        pose[:, 2] += torch.tensor(
            [height + 0.80 for height in surfaces],
            dtype=pose.dtype,
            device=pose.device,
        )
        robot.write_root_pose_to_sim(pose, env_ids=ids)
        robot.write_root_velocity_to_sim(
            torch.zeros((2, 6), dtype=pose.dtype, device=pose.device),
            env_ids=ids,
        )
        base.sim.forward()
        action = torch.zeros_like(base.action_manager.action)
        _observation, reward, terminated, truncated, extras = env.step(action)
        height_sensor = base.scene[projection.height_sensor_name]
        hits = height_sensor.data.ray_hits_w[ids, :, 2]
        local_hits = hits - origins[:, 2:3]
        finite = torch.isfinite(local_hits).all(dim=0)
        deltas = torch.abs(local_hits[1, finite] - local_hits[0, finite])
        visible_ray_count = int((deltas >= 0.001).sum().item())
        maximum_delta = float(deltas.max().item()) if deltas.numel() else 0.0
        if visible_ray_count < 8 or maximum_delta < 0.002:
            raise RuntimeError(
                "C2 roughness is not observably distinct to the real height scanner"
            )

        raw = extras[STAIR_EVAL_TRANSITION_EXTRAS_KEY]
        transitions = decode_stair_transition_batch(
            raw,
            wrapper_done_mask=terminated | truncated,
            protocol=None,
            semantics=config.evaluation_semantics,
            expected_bin_assignment=tuple(cell.cell_id for cell in assignment),
            expected_cell_assignment=assignment,
            runtime_projection=projection,
        )
        rough_transition = transitions[rough_index]
        clearances = rough_transition.support_surface_clearances_m
        if not all(math.isfinite(value) for value in clearances):
            raise RuntimeError("C2 decoder produced non-finite surface clearances")
        return {
            "status": "pass",
            "split": split,
            "num_envs": len(assignment),
            "flat_cell_id": flat_cell.cell_id,
            "rough_cell_id": rough_cell.cell_id,
            "paired_progress_m": progress,
            "paired_lateral_m": lateral,
            "paired_surface_heights_m": surfaces,
            "height_scanner": {
                "ray_count": int(hits.shape[1]),
                "finite_paired_ray_count": int(finite.sum().item()),
                "visible_ray_threshold_m": 0.001,
                "visible_ray_count": visible_ray_count,
                "required_visible_ray_count": 8,
                "maximum_abs_hit_height_delta_m": maximum_delta,
                "required_maximum_delta_m": 0.002,
            },
            "decoder": {
                "transition_count": len(transitions),
                "rough_support_surface_levels": list(
                    rough_transition.support_surface_levels
                ),
                "rough_support_surface_clearances_m": list(clearances),
                "reward_finite": bool(torch.isfinite(reward).all().item()),
            },
            "observation_shapes": {
                str(name): [int(item) for item in value.shape]
                for name, value in observation.items()
            },
        }
    finally:
        env.close()


def _capture(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Phase 3B output path must be fresh")
    sys.path.insert(0, str(project_root / "src"))
    config, manifest = _resolve_unified_cell(project_root)

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device=args.device)
    _ = launcher.app
    import torch

    started = time.perf_counter()
    geometry = _geometry_probe(config)
    failure_feedback = _failure_feedback_calibration(config)
    scanner = _scanner_and_decoder_probe(
        config,
        manifest,
        device=args.device,
        seed=args.seed,
    )
    suite = config.ordered_evaluation_suite
    formal_episodes = sum(
        suite.seed_count(cohort)
        * len(suite.cells_for_cohort(cohort))
        * suite.episodes_per_seed_per_cell
        for cohort in ("public_dev", "public_accept", "sealed_validation")
    )
    if formal_episodes != 1080 or formal_episodes > 1260:
        raise RuntimeError("Phase 3B formal evaluation cost exceeded P3R")

    source_paths = (
        "configs/tasks/stair_unified_protocol_v1.json",
        "configs/tasks/stair_unified_training_profile_v1.json",
        "configs/tasks/stair_unified_evaluation_suite_v1.json",
        "src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py",
        "src/true_rl_agent/tasks/stair_climbing/unified_evaluation.py",
        "src/true_rl_agent/tasks/stair_climbing/rough_surface.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_rough_terrain.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_evaluation_terrain.py",
        "src/true_rl_agent/tasks/stair_climbing/terrain_evidence.py",
        "src/true_rl_agent/tasks/stair_climbing/transition_decoder.py",
        "PLAN/Variant_construction/P3B/capture_phase3b_runtime_receipt.py",
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "capture_kind": "real_headless_c2_mesh_scanner_and_decoder",
        "environment": {
            "python": ".".join(str(item) for item in sys.version_info[:3]),
            "device": args.device,
            "torch": str(torch.__version__),
            "cuda_device_name": torch.cuda.get_device_name(args.device),
        },
        "cell": {
            "cell_variant_id": manifest.cell_variant_id,
            "protocol_state": config.protocol_state,
            "manifest_hash": manifest.manifest_hash,
            "base_task_protocol_hash": config.base_task_protocol_hash,
            "resolved_cell_task_protocol_hash": (
                manifest.resolved_cell_task_protocol_hash
            ),
            "training_profile_hash": config.unified_training_profile.profile_hash,
            "evaluation_suite_hash": suite.suite_hash,
        },
        "geometry_probe": geometry,
        "failure_feedback_calibration": failure_feedback,
        "height_scanner_and_decoder_probe": scanner,
        "cost": {
            "formal_episode_count": formal_episodes,
            "p3r_formal_episode_cap": 1260,
            "episodes_per_seed_per_cell": suite.episodes_per_seed_per_cell,
            "formal_cells_per_cohort": len(suite.cells_for_cohort("public_dev")),
            "wall_time_s": round(time.perf_counter() - started, 6),
        },
        "product_source_hash_mode": "normalized_lf_sha256",
        "product_source_hashes": {
            path: _normalized_text_hash(project_root / path) for path in source_paths
        },
        "status": "pass",
    }
    receipt["receipt_hash"] = _canonical_hash(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    with output.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "status": "pass",
                "receipt_hash": receipt["receipt_hash"],
                "visible_ray_count": scanner["height_scanner"]["visible_ray_count"],
                "maximum_ray_delta_m": scanner["height_scanner"][
                    "maximum_abs_hit_height_delta_m"
                ],
                "formal_episode_count": formal_episodes,
            }
        ),
        flush=True,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=12021, type=int)
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
