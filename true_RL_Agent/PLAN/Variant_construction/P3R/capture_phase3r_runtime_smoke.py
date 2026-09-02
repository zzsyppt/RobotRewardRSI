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
from types import SimpleNamespace
from typing import Any, Mapping

RECEIPT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase3r_runtime_smoke.v1"


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
        resolved_cell_task_protocol_hash=(manifest.resolved_cell_task_protocol_hash),
    )
    plugin = StairClimbingTaskPlugin(config, resolved_cell_manifest=manifest)
    resolved = resolve_stair_cell_manifest(
        cell_variant_id=config.cell_variant_id,
        config=config,
        plugin=plugin,
    )
    if resolved.to_dict() != manifest.to_dict():
        raise RuntimeError("unified Stair manifest changed after identity binding")
    if config.protocol_state != "draft":
        raise RuntimeError("Phase 3R smoke is only authorized for the draft protocol")
    return config, manifest


def _curriculum_probe(config: Any, *, device: str) -> dict[str, Any]:
    import torch

    from true_rl_agent.tasks.stair_climbing.isaaclab_unified_terrain import (
        stair_goal_progress_levels,
    )
    from true_rl_agent.tasks.stair_climbing.unified_training import (
        goal_progress_m_by_terrain_type,
    )

    goals = goal_progress_m_by_terrain_type(config.unified_training_profile)
    terrain_types = torch.tensor([0, 1, 9], dtype=torch.long, device=device)
    terrain_levels = torch.tensor([2, 0, 1], dtype=torch.long, device=device)
    terrain_origins = torch.zeros((10, 10, 3), dtype=torch.float32, device=device)
    env_origins = torch.zeros((3, 3), dtype=torch.float32, device=device)

    class Terrain:
        def __init__(self) -> None:
            self.terrain_types = terrain_types
            self.terrain_levels = terrain_levels
            self.terrain_origins = terrain_origins
            self.env_origins = env_origins

        def update_env_origins(
            self, env_ids: Any, move_up: Any, move_down: Any
        ) -> None:
            levels = self.terrain_levels[env_ids]
            levels = levels + move_up.to(levels.dtype) - move_down.to(levels.dtype)
            self.terrain_levels[env_ids] = torch.clamp(levels, min=0, max=9)
            self.env_origins[env_ids] = self.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]
            ]

    class Scene(dict):
        def __init__(self, terrain: Terrain, robot: Any) -> None:
            super().__init__(robot=robot)
            self.terrain = terrain

    fractions = (0.85, 0.60, 0.20)
    root_pos = torch.zeros((3, 3), dtype=torch.float32, device=device)
    for index, (terrain_type, fraction) in enumerate(
        zip(terrain_types.tolist(), fractions, strict=True)
    ):
        root_pos[index, 0] = float(goals[terrain_type]) * fraction
    terrain = Terrain()
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos))
    env = SimpleNamespace(
        scene=Scene(terrain, robot),
        episode_length_buf=torch.ones(3, dtype=torch.long, device=device),
    )
    env_ids = torch.arange(3, dtype=torch.long, device=device)
    before_levels = terrain_levels.detach().cpu().tolist()
    before_types = terrain_types.detach().cpu().tolist()
    stair_goal_progress_levels(
        env,
        env_ids,
        goal_progress_m_by_terrain_type=goals,
        move_up_threshold=0.8,
        move_down_threshold=0.4,
        asset_cfg=SimpleNamespace(name="robot"),
    )
    after_levels = terrain_levels.detach().cpu().tolist()
    after_types = terrain_types.detach().cpu().tolist()
    if after_levels != [3, 0, 0] or after_types != before_types:
        raise RuntimeError("unified Stair curriculum up/hold/down probe failed")
    return {
        "fractions": list(fractions),
        "levels_before": before_levels,
        "levels_after": after_levels,
        "terrain_types_before": before_types,
        "terrain_types_after": after_types,
        "terrain_type_immutable": True,
        "status": "pass",
    }


def _evaluation_geometry_probe(config: Any) -> dict[str, Any]:
    from true_rl_agent.tasks.stair_climbing.isaaclab_evaluation_terrain import (
        build_ordered_eval_terrain_generator,
    )

    suite = config.ordered_evaluation_suite
    rows: list[dict[str, Any]] = []
    for seed, cohort in enumerate(
        ("public_dev", "public_accept", "sealed_validation"),
        start=1,
    ):
        cells = suite.runtime_cells(cohort)
        generator = build_ordered_eval_terrain_generator(cells=cells, seed=seed)
        target = next(
            cell for cell in cells if cell.stratum_id == "phase_composition_ood"
        )
        terrain_cfg = generator.sub_terrains[target.cell_id]
        terrain_cfg.size = generator.size
        meshes, _ = terrain_cfg.function(0.0, terrain_cfg)
        if (
            len(meshes) != target.step_count + 2
            or float(terrain_cfg.floor_depth) != target.floor_depth_m
            or terrain_cfg.geometry_digest != target.geometry_digest
        ):
            raise RuntimeError("Phase 3R evaluation geometry probe drifted")
        rows.append(
            {
                "cohort": cohort,
                "cell_id": target.cell_id,
                "geometry_digest": target.geometry_digest,
                "floor_depth_m": target.floor_depth_m,
                "mesh_count": len(meshes),
            }
        )
    return {"status": "pass", "representatives": rows}


def _capture(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve(strict=True)
    output = args.output.resolve()
    work_dir = args.work_dir.resolve()
    if output.exists() or work_dir.exists():
        raise FileExistsError("Phase 3R receipt and work directory must be fresh")
    sys.path.insert(0, str(project_root / "src"))

    config, manifest = _resolve_unified_cell(project_root)
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
    evaluation_geometry = _evaluation_geometry_probe(config)
    work_dir.mkdir(parents=True, exist_ok=False)
    training = execute_stair_training_smoke(
        config=config,
        source_manifest_preflight=source_preflight,
        agent_configuration=config.training_agent_configuration(),
        reward_program=official.reward_program,
        device=args.device,
        run_dir=work_dir / "train",
        checkpoint_output=work_dir / "checkpoint" / "model_0.pt",
        run_name="phase3r_unified_draft_smoke",
        resolved_cell_manifest=manifest,
    )
    summary = _json_value(training.to_summary_fields())
    if training.status != "pass":
        raise RuntimeError(f"unified Stair draft smoke failed: {summary}")
    train_summary = summary["training"]
    activation = summary["runtime_activation_summary"]
    assignment = train_summary["initial_terrain_assignment"]
    if train_summary["budget_steps"] != 128:
        raise RuntimeError("Phase 3R smoke budget drifted")
    if assignment["regime_counts"] != {
        "c0_regular": 8,
        "c1_irregular_flat": 8,
    }:
        raise RuntimeError("Phase 3R runtime assignment is not 50/50")
    if (
        activation["declared_term_count"] != 15
        or activation["dispatcher_calls"] <= 1
        or activation["nonzero_reward_contribution"] is not True
        or not math.isfinite(float(activation["last_total_rate_mean"]))
    ):
        raise RuntimeError("Phase 3R VNext reward activation is degenerate")
    curriculum = _curriculum_probe(config, device=args.device)

    source_paths = (
        "configs/tasks/stair_unified_protocol_v1.json",
        "configs/tasks/stair_unified_training_profile_v1.json",
        "configs/tasks/stair_unified_evaluation_suite_v1.json",
        "configs/tasks/stair_vnext_reward_authority_v1.json",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_execution.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_training.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_runtime_env.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_evaluation_terrain.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_irregular_terrain.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_unified_terrain.py",
        "src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py",
        "src/true_rl_agent/tasks/stair_climbing/config.py",
        "src/true_rl_agent/tasks/stair_climbing/unified_protocol.py",
        "src/true_rl_agent/tasks/stair_climbing/ordered_evaluation.py",
        "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
        "src/true_rl_agent/tasks/stair_climbing/robot_binding.py",
        "src/true_rl_agent/tasks/stair_climbing/unified_evaluation.py",
        "src/true_rl_agent/tasks/stair_climbing/unified_training.py",
        "src/true_rl_agent/tasks/stair_climbing/vnext_reward_authority.py",
        "src/true_rl_agent/tasks/stair_climbing/reference_programs/official_g1_rough_vnext_v1/training_code/components.py",
        "PLAN/Variant_construction/P3R/capture_phase3r_runtime_smoke.py",
    )
    elapsed = time.perf_counter() - started
    full_budget = int(config.budget_steps)
    smoke_budget = int(train_summary["budget_steps"])
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "capture_kind": "real_headless_unified_stair_vnext_ppo_smoke",
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
            "evaluation_suite_hash": config.ordered_evaluation_suite.suite_hash,
            "vnext_reward_authority_hash": config.vnext_reward_authority.authority_hash,
            "reference_program_source_checksum": (
                official.reward_program.program_source_checksum
            ),
        },
        "training_smoke": summary,
        "curriculum_probe": curriculum,
        "evaluation_geometry_probe": evaluation_geometry,
        "source_preflight": {
            "summary": source_preflight.to_summary_dict(),
            "evidence": source_preflight.upstream_source_evidence(),
        },
        "eligibility": {
            "reference_qualification": False,
            "current_best": False,
            "candidate_acceptance": False,
            "reason": "phase3r_draft_smoke_only",
        },
        "cost": {
            "wall_time_s": round(elapsed, 6),
            "smoke_budget_steps": smoke_budget,
            "full_training_budget_steps": full_budget,
            "budget_fraction": smoke_budget / full_budget,
            "maximum_authorized_fraction": 0.05,
        },
        "product_source_hash_mode": "normalized_lf_sha256",
        "product_source_hashes": {
            path: _normalized_text_hash(project_root / path) for path in source_paths
        },
        "status": "pass",
    }
    if receipt["cost"]["budget_fraction"] > 0.05:
        raise RuntimeError("Phase 3R smoke exceeded its authorized budget")
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
                "budget_steps": smoke_budget,
                "budget_fraction": receipt["cost"]["budget_fraction"],
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
