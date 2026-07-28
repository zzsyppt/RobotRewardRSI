#!/usr/bin/env python3
"""Verify Stair reward faults fail normally without poisoning CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.reward_inputs import (  # noqa: E402
    StairRewardInputsV2,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (  # noqa: E402
    StairRewardProgramError,
    evaluate_stair_reward_program,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stair reward fault probe")
    device = torch.device("cuda:0")
    config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
    )
    resolved = resolve_official_stair_reference_program(
        config,
        project_root=PROJECT_ROOT,
    )
    cases = [
        _run_fault_case(
            name="finite_out_of_bounds",
            joint_acceleration=1.0e6,
            expected_marker="reward_term_weighted:dof_acc_l2",
            device=device,
            reward_program=resolved.reward_program,
        ),
        _run_fault_case(
            name="non_finite_output",
            joint_acceleration=1.0e20,
            expected_marker="reward_term_raw:dof_acc_l2",
            device=device,
            reward_program=resolved.reward_program,
        ),
    ]
    report = {
        "schema_version": (
            "true_rl_agent.task.stair_climbing.reward_cuda_fault_probe.v1"
        ),
        "status": "pass" if all(case["status"] == "pass" for case in cases) else "fail",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "protocol_v2_hash": config.protocol_v2.protocol_hash,
        "reference_program": resolved.reward_program.to_public_dict(),
        "cases": cases,
    }
    report["report_hash"] = canonical_json_hash(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "pass" else 1


def _run_fault_case(
    *,
    name: str,
    joint_acceleration: float,
    expected_marker: str,
    device: object,
    reward_program: object,
) -> dict[str, object]:
    import torch

    inputs = _synthetic_inputs(
        reward_program.contract,
        device=device,
        joint_acceleration=joint_acceleration,
    )
    caught = None
    try:
        evaluate_stair_reward_program(reward_program, inputs, num_envs=2)
    except StairRewardProgramError as exc:
        caught = exc
    if caught is None:
        raise RuntimeError(f"Stair reward fault case did not fail: {name}")
    message = str(caught)
    if expected_marker not in message:
        raise RuntimeError(
            f"Stair reward fault case failed for the wrong reason: {name}: {message}"
        )

    cuda_probe = torch.arange(32, dtype=torch.float32, device=device)
    cuda_probe_sum = float((cuda_probe * 2.0).sum().item())
    torch.cuda.synchronize(device)
    if cuda_probe_sum != 992.0:
        raise RuntimeError(f"CUDA context probe produced the wrong result: {name}")
    return {
        "name": name,
        "status": "pass",
        "error_type": type(caught).__name__,
        "error_message": message,
        "expected_marker": expected_marker,
        "cuda_context_usable_after_failure": True,
        "cuda_probe_sum": cuda_probe_sum,
    }


def _synthetic_inputs(
    contract: object,
    *,
    device: object,
    joint_acceleration: float,
) -> StairRewardInputsV2:
    import torch

    values = {}
    for field in contract.input_fields:
        dtype = torch.bool if field.dtype == "bool" else torch.float32
        values[field.name] = torch.zeros(
            (2, *field.trailing_shape),
            dtype=dtype,
            device=device,
        )
    values["projected_gravity"][:, 2] = -1.0
    values["base_quat_world"][:, 0] = 1.0
    values["remaining_goal_distance_m"].fill_(1.0)
    values["ankle_soft_joint_pos_limits"][:, :, 0] = -1.0
    values["ankle_soft_joint_pos_limits"][:, :, 1] = 1.0
    values["hip_knee_joint_acc"].fill_(joint_acceleration)
    return StairRewardInputsV2._validated(contract=contract, values=values)


if __name__ == "__main__":
    raise SystemExit(main())
