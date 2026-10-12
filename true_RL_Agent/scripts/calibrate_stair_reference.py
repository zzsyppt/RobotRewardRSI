"""Stair anchor measurement using the production evaluator and existing GPU ownership."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.runner.checkpoint import file_sha256


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_terrain_geometry(config):
    """Compare actual mesh triangles to the evidence query on the measured cells."""
    from true_rl_agent.tasks.stair_climbing.isaaclab_rough_terrain import build_rough_stair_meshes
    from true_rl_agent.tasks.stair_climbing.isaaclab_unified_terrain import (
        build_unified_train_terrain_generator, unified_irregular_training_stairs_terrain,
    )
    from true_rl_agent.tasks.stair_climbing.isaaclab_terrain import one_way_stairs_terrain
    from true_rl_agent.tasks.stair_climbing.terrain_evidence import RoughStairSurfaceQuery

    terrain = build_unified_train_terrain_generator(config.unified_training_profile, seed=config.train_seed)
    training_functions = {name: sub.function.__name__ for name, sub in terrain.sub_terrains.items()}
    assert all(sub.function in {one_way_stairs_terrain, unified_irregular_training_stairs_terrain}
               for sub in terrain.sub_terrains.values())
    meshes_by_cell = {}
    for cell in config.ordered_evaluation_suite.cells_for_cohort("calibration"):
        if cell.terrain_regime_id != "c2_rough":
            continue
        meshes, _ = build_rough_stair_meshes(
            size=cell.size_m, floor_depth=cell.floor_depth_m,
            start_platform_length=cell.start_platform_length_m,
            riser_heights_m=cell.riser_heights_m, tread_depths_m=cell.tread_depths_m,
            top_platform_length=cell.top_platform_length_m, surface_profile=cell.surface_profile,
        )
        query = RoughStairSurfaceQuery(cell)
        errors = []
        top_count = cell.surface_profile["longitudinal_vertices"] * cell.surface_profile["lateral_vertices"]
        for mesh in meshes[1:-1]:
            assert mesh.is_watertight
            for face in mesh.faces:
                if max(face) >= top_count:
                    continue
                point = mesh.vertices[face].mean(axis=0)
                actual = query.sample(progress_m=float(point[0] - cell.start_platform_length_m / 2),
                                      lateral_m=float(point[1] - cell.size_m[1] / 2)).height_m
                errors.append(abs(actual - float(point[2])))
        assert errors and max(errors) < 1e-9, (cell.cell_id, max(errors))
        meshes_by_cell[cell.cell_id] = {"geometry_digest": cell.geometry_digest,
                                       "checked_top_triangles": len(errors), "max_height_error_m": max(errors)}
    return {"status": "PASS", "training_terrain_functions": training_functions,
            "training_c2_generator_count": 0, "rough_meshes": meshes_by_cell}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if not args.child:
        from true_rl_agent.evaluation.binding import canonical_json_hash
        from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
        from true_rl_agent.runner.process_handle import run_owned_process
        from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory

        args.output.mkdir(parents=True, exist_ok=False)
        inventory = discover_nvidia_gpu_inventory()
        gpu = next(item for item in inventory if item.execution_device == args.device)
        argv = [sys.executable, str(Path(__file__).resolve()), "--child", "--cell", args.cell,
                "--checkpoint", str(args.checkpoint.resolve()), "--device", "cuda:0", "--output", str(args.output.resolve())]
        report = {"purpose": "Fixed-anchor public calibration; no reference promotion",
                  "argv": argv, "checkpoint_sha256": file_sha256(args.checkpoint),
                  "started_at_unix": time.time(), "budget_seconds": 1800}
        identity = canonical_json_hash(report)
        holder = PreGenesisGpuLeaseHolder(
            session_id="stair-public-calibration", genesis_identity_hash=identity,
            session_contract_hash=identity, lane_id=args.device, invocation_segment_id=args.output.name,
            logical_job_id=args.output.name,
        )
        manager = GpuLeaseManager(lease_root=Path("/tmp/true_rl_agent_gpu_leases"),
                                  devices={item.execution_device: item.identity for item in inventory})
        write(args.output / "launch.json", report)
        with manager.acquire(args.device, holder=holder) as lease:
            report["lease"] = dict(lease.receipt)
            try:
                result = run_owned_process(argv, cwd=ROOT, log_dir=args.output / "process", timeout_s=1800,
                                           environment_overrides={"CUDA_VISIBLE_DEVICES": gpu.uuid, "NVIDIA_VISIBLE_DEVICES": None})
                report.update(returncode=result.returncode, child_pid=result.pid)
            except Exception:
                report.update(returncode=1, error=traceback.format_exc())
            finally:
                report["ended_at_unix"] = time.time()
                write(args.output / "launch.json", report)
        return report["returncode"]

    from isaaclab.app import AppLauncher
    from true_rl_agent.backends.isaac_lab_execution_environment import (
        build_current_isaac_lab_execution_environment_manifest, verify_current_isaac_lab_execution_environment,
    )
    from true_rl_agent.runner.isaac_app_lifecycle import bind_isaac_log_path, close_isaac_app_on_success
    from true_rl_agent.tasks.stair_climbing.calibration import select_stair_calibration_levels, stair_cell_scale_report
    from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
    from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
    from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
    from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row
    from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_CALIBRATION_EVAL_NAMESPACE
    from true_rl_agent.tasks.stair_climbing.isaaclab_execution import execute_stair_evaluation, preflight_stair_execution_source_manifest

    config = StairClimbingTaskConfig.defaults(ROOT / stair_robot_row(args.cell).protocol_relative_path)
    cell = resolve_stair_cell_manifest(cell_variant_id=args.cell, config=config, plugin=StairClimbingTaskPlugin(config))
    config = replace(config, resolved_cell_task_protocol_hash=cell.resolved_cell_task_protocol_hash)
    manifest = build_current_isaac_lab_execution_environment_manifest()
    verification = verify_current_isaac_lab_execution_environment(manifest.manifest_hash)
    preflight = preflight_stair_execution_source_manifest(config)
    report = {"cell": cell.to_dict(), "protocol": config.unified_protocol.to_dict(),
              "suite": config.ordered_evaluation_suite.to_dict(), "environment": manifest.to_dict(),
              "checkpoint_sha256": file_sha256(args.checkpoint)}
    kit_args = argparse.Namespace(headless=True, device=args.device)
    bind_isaac_log_path(kit_args, args.output / "result.json")
    app = AppLauncher(kit_args, multi_gpu=False).app
    code = 1
    try:
        report["terrain_geometry_verification"] = verify_terrain_geometry(config)
        result = execute_stair_evaluation(
            config=config, source_manifest_preflight=preflight,
            backend_execution_environment_verification=verification,
            checkpoint_path=args.checkpoint, device=args.device,
            seed_namespace=STAIR_CALIBRATION_EVAL_NAMESPACE, splits=("calibration",), resolved_cell_manifest=cell,
        )
        summary = result.public_summaries()["calibration"]
        report.update(status="PASS", summary=summary, execution=result.execution_identity(),
                      checkpoint_loading=dict(result.checkpoint_loading), constructed_learner=dict(result.constructed_learner),
                      selection=select_stair_calibration_levels(suite=config.ordered_evaluation_suite, summary=summary),
                      scale_report=stair_cell_scale_report(config.ordered_evaluation_suite, "calibration"))
        code = 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        write(args.output / "result.json", report)
    return close_isaac_app_on_success(app, exit_code=code)


if __name__ == "__main__":
    sys.exit(main())
