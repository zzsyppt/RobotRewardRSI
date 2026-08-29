"""Capture deterministic legacy-official to VNext reward equivalence evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _capture(project_root: Path, *, seed: int, sample_count: int) -> dict[str, Any]:
    sys.path.insert(0, str(project_root / "src"))
    import torch

    from true_rl_agent.tasks.stair_climbing.reference_programs.official_g1_rough_v1.training_code import (
        components as legacy_components,
    )
    from true_rl_agent.tasks.stair_climbing.reference_programs.official_g1_rough_vnext_v1.training_code import (
        components as vnext_components,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)

    def randn(*shape: int) -> Any:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    def rand(*shape: int) -> Any:
        return torch.rand(*shape, generator=generator, dtype=torch.float32)

    n = sample_count
    shared = {
        "terminated_non_timeout": rand(n) > 0.8,
        "commanded_base_vel": randn(n, 3),
        "base_lin_vel_yaw": randn(n, 3),
        "base_ang_vel_world": randn(n, 3),
        "contact_time": rand(n, 2),
        "air_time": rand(n, 2),
        "contact_force_history": 2.0 * randn(n, 3, 2, 3),
        "support_lin_vel": randn(n, 2, 3),
        "ankle_limits": torch.stack(
            (-0.5 - rand(n, 4), 0.5 + rand(n, 4)),
            dim=-1,
        ),
        "ankle_pos": 2.0 * randn(n, 4),
        "hip_yaw_roll_error": randn(n, 4),
        "arm_error": randn(n, 10),
        "finger_error": randn(n, 14),
        "torso_error": randn(n, 1),
        "base_ang_vel": randn(n, 3),
        "projected_gravity": randn(n, 3),
        "action_delta": randn(n, 37),
        "hip_knee_acc": randn(n, 8),
        "locomotion_torque": randn(n, 12),
    }
    shared["commanded_base_vel"][:4] = torch.tensor(
        [[0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    shared["contact_time"][:4] = torch.tensor(
        [[2.0, 2.0], [0.5, 0.0], [0.0, 0.0], [0.5, 0.0]],
        dtype=torch.float32,
    )
    shared["air_time"][:4] = torch.tensor(
        [[0.0, 0.0], [0.0, 0.3], [0.5, 0.6], [0.0, 0.3]],
        dtype=torch.float32,
    )
    legacy = SimpleNamespace(
        terminated_non_timeout=shared["terminated_non_timeout"],
        commanded_base_vel=shared["commanded_base_vel"],
        base_lin_vel_yaw=shared["base_lin_vel_yaw"],
        base_ang_vel_world=shared["base_ang_vel_world"],
        foot_contact_time=shared["contact_time"],
        foot_air_time=shared["air_time"],
        foot_contact_force_history=shared["contact_force_history"],
        foot_lin_vel=shared["support_lin_vel"],
        ankle_soft_joint_pos_limits=shared["ankle_limits"],
        ankle_joint_pos=shared["ankle_pos"],
        hip_yaw_roll_joint_pos_error=shared["hip_yaw_roll_error"],
        arm_joint_pos_error=shared["arm_error"],
        finger_joint_pos_error=shared["finger_error"],
        torso_joint_pos_error=shared["torso_error"],
        base_ang_vel=shared["base_ang_vel"],
        projected_gravity=shared["projected_gravity"],
        action_delta=shared["action_delta"],
        hip_knee_joint_acc=shared["hip_knee_acc"],
        locomotion_joint_torque=shared["locomotion_torque"],
    )
    vnext = SimpleNamespace(
        terminated_non_timeout=shared["terminated_non_timeout"],
        commanded_base_vel=shared["commanded_base_vel"],
        base_lin_vel_yaw=shared["base_lin_vel_yaw"],
        base_ang_vel_world=shared["base_ang_vel_world"],
        support_contact_time=shared["contact_time"],
        support_air_time=shared["air_time"],
        support_contact_force_history=shared["contact_force_history"],
        support_lin_vel=shared["support_lin_vel"],
        ankle_group_soft_pos_limits=shared["ankle_limits"],
        ankle_group_pos=shared["ankle_pos"],
        hip_yaw_roll_group_pos_error=shared["hip_yaw_roll_error"],
        arm_group_pos_error=shared["arm_error"],
        finger_group_pos_error=shared["finger_error"],
        torso_group_pos_error=shared["torso_error"],
        base_ang_vel=shared["base_ang_vel"],
        projected_gravity=shared["projected_gravity"],
        action_delta=shared["action_delta"],
        hip_knee_group_acc=shared["hip_knee_acc"],
        locomotion_group_torque=shared["locomotion_torque"],
    )

    legacy_program = legacy_components.build_training_components()
    vnext_program = vnext_components.build_training_components()
    legacy_identity = [(term.name, float(term.weight)) for term in legacy_program.reward_terms]
    vnext_identity = [(term.name, float(term.weight)) for term in vnext_program.reward_terms]
    if legacy_identity != vnext_identity or len(legacy_identity) != 15:
        raise RuntimeError("official reference term identity differs")

    term_evidence: dict[str, Any] = {}
    legacy_total = torch.zeros(n, dtype=torch.float32)
    vnext_total = torch.zeros(n, dtype=torch.float32)
    for legacy_term, vnext_term in zip(
        legacy_program.reward_terms,
        vnext_program.reward_terms,
        strict=True,
    ):
        legacy_raw = legacy_term.fn(legacy)
        vnext_raw = vnext_term.fn(vnext)
        max_abs = float((legacy_raw - vnext_raw).abs().max().item())
        if max_abs != 0.0:
            raise RuntimeError(f"official/VNext term differs: {legacy_term.name}")
        legacy_total += legacy_raw * float(legacy_term.weight)
        vnext_total += vnext_raw * float(vnext_term.weight)
        term_evidence[legacy_term.name] = {
            "weight": float(legacy_term.weight),
            "shape": list(legacy_raw.shape),
            "finite": bool(torch.isfinite(legacy_raw).all().item()),
            "max_abs_raw_difference": max_abs,
        }
    total_max_abs = float((legacy_total - vnext_total).abs().max().item())
    if total_max_abs != 0.0:
        raise RuntimeError("official/VNext weighted total differs")
    air_time = vnext_components.feet_air_time(vnext)
    edge_values = [float(value) for value in air_time[:4].tolist()]
    if edge_values != [0.0, 0.30000001192092896, 0.0, 0.0]:
        raise RuntimeError("single-stance/moving air-time semantics drifted")

    legacy_path = (
        project_root
        / "src/true_rl_agent/tasks/stair_climbing/reference_programs/official_g1_rough_v1/training_code/components.py"
    )
    vnext_path = (
        project_root
        / "src/true_rl_agent/tasks/stair_climbing/reference_programs/official_g1_rough_vnext_v1/training_code/components.py"
    )
    receipt = {
        "schema_version": "true_rl_agent.variant_construction.phase4.reward_equivalence.v1",
        "status": "pass",
        "seed": seed,
        "sample_count": sample_count,
        "torch_version": str(torch.__version__),
        "term_count": len(term_evidence),
        "term_order": [name for name, _ in legacy_identity],
        "term_evidence": term_evidence,
        "weighted_total_max_abs_difference": total_max_abs,
        "feet_air_time_edge_cases": {
            "double_contact_moving": edge_values[0],
            "single_stance_moving": edge_values[1],
            "flight_moving": edge_values[2],
            "single_stance_stationary": edge_values[3],
        },
        "source_hashes": {
            str(legacy_path.relative_to(project_root)).replace("\\", "/"): _sha256(
                legacy_path
            ),
            str(vnext_path.relative_to(project_root)).replace("\\", "/"): _sha256(
                vnext_path
            ),
        },
    }
    receipt["receipt_hash"] = _canonical_hash(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--sample-count", type=int, default=257)
    args = parser.parse_args()
    project_root = args.project_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Phase 4 reward equivalence output must be fresh")
    receipt = _capture(
        project_root,
        seed=args.seed,
        sample_count=args.sample_count,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": "pass", "receipt_hash": receipt["receipt_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
