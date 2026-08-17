"""Strict compiler for Stair complete-reward performance evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairProtocolV2,
    StairProtocolV2Draft,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    QualifiedStairRewardProgram,
)

STAIR_REWARD_BENCHMARK_CASE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.reward_benchmark_case.v1"
)
STAIR_REWARD_PERFORMANCE_REPORT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.reward_performance_report.v1"
)
STAIR_REWARD_BENCHMARK_CASES = (
    "official",
    "dispatcher",
    "dispatcher_with_eval_recorder",
)


def compile_stair_reward_performance_report(
    case_reports: Sequence[Mapping[str, Any]],
    *,
    protocol: StairProtocolV2 | StairProtocolV2Draft,
    reference_program: QualifiedStairRewardProgram,
) -> dict[str, Any]:
    """Validate three isolated measurements and calculate the canonical gate."""

    if not isinstance(protocol, (StairProtocolV2, StairProtocolV2Draft)):
        raise TypeError("Stair reward benchmark protocol is invalid")
    if not isinstance(reference_program, QualifiedStairRewardProgram):
        raise TypeError("Stair reward benchmark reference program is invalid")
    if not isinstance(case_reports, Sequence) or isinstance(
        case_reports, (str, bytes)
    ):
        raise TypeError("Stair reward benchmark cases must be a sequence")
    reports: dict[str, dict[str, Any]] = {}
    performance = protocol.complete_reward_runtime.to_public_dict()[
        "performance_gate"
    ]
    equivalence = protocol.complete_reward_runtime.to_public_dict()[
        "reference_equivalence"
    ]
    expected_program = reference_program.to_public_dict()
    hardware_fingerprint: dict[str, Any] | None = None
    for raw in case_reports:
        report = _strict_case_report(raw)
        case = report["case"]
        if case in reports:
            raise ValueError(f"duplicate Stair reward benchmark case: {case}")
        if report["protocol_v2_hash"] != protocol.protocol_hash:
            raise ValueError("Stair reward benchmark protocol hash mismatch")
        for field_name, expected in (
            ("num_envs", performance["num_envs"]),
            ("warmup_control_steps", performance["warmup_control_steps"]),
            ("measured_control_steps", performance["measured_control_steps"]),
        ):
            if report[field_name] != expected:
                raise ValueError(
                    f"Stair reward benchmark {field_name} differs from V2"
                )
        observed_hardware = report["hardware_fingerprint"]
        if hardware_fingerprint is None:
            hardware_fingerprint = observed_hardware
        elif hardware_fingerprint != observed_hardware:
            raise ValueError("Stair reward benchmark hardware differs across cases")
        if case == "official":
            if report["training_recorder_enabled"] is not False:
                raise ValueError("official benchmark unexpectedly enabled a recorder")
            if report["program"] is not None:
                raise ValueError("official benchmark cannot carry an R0 program")
            expected_official_terms = set(equivalence["active_term_names"]) | set(
                equivalence["zero_weight_term_names"]
            )
            if set(report["reward_manager_active_terms"]) != expected_official_terms:
                raise ValueError("official benchmark reward term set mismatch")
            if report["runtime"]:
                raise ValueError("official benchmark cannot carry dispatcher runtime")
        else:
            if report["program"] != expected_program:
                raise ValueError("dispatcher benchmark R0 identity mismatch")
            expected_recorder = case == "dispatcher_with_eval_recorder"
            if report["training_recorder_enabled"] is not expected_recorder:
                raise ValueError("dispatcher benchmark recorder state mismatch")
            if report["reward_manager_active_terms"] != ["complete_reward"]:
                raise ValueError("dispatcher benchmark reward authority is invalid")
            runtime = report["runtime"]
            if runtime.get("status") != "pass":
                raise ValueError("dispatcher benchmark runtime did not pass")
            expected_calls = int(performance["warmup_control_steps"]) + int(
                performance["measured_control_steps"]
            )
            if runtime.get("dispatcher_calls") != expected_calls:
                raise ValueError("dispatcher benchmark call count is not closed")
            if runtime.get("term_calls") != expected_calls * len(
                reference_program.term_names
            ):
                raise ValueError("dispatcher benchmark term call count is not closed")
        reports[case] = report
    if set(reports) != set(STAIR_REWARD_BENCHMARK_CASES):
        missing = sorted(set(STAIR_REWARD_BENCHMARK_CASES) - set(reports))
        extra = sorted(set(reports) - set(STAIR_REWARD_BENCHMARK_CASES))
        raise ValueError(
            f"Stair reward benchmark case set mismatch: missing={missing} extra={extra}"
        )
    official_throughput = float(reports["official"]["env_steps_per_second"])
    dispatcher_throughput = float(
        reports["dispatcher"]["env_steps_per_second"]
    )
    recorder_throughput = float(
        reports["dispatcher_with_eval_recorder"]["env_steps_per_second"]
    )
    dispatcher_ratio = dispatcher_throughput / official_throughput
    recorder_to_dispatcher_ratio = recorder_throughput / dispatcher_throughput
    minimum_ratio = float(performance["minimum_dispatcher_throughput_ratio"])
    result = {
        "schema_version": STAIR_REWARD_PERFORMANCE_REPORT_SCHEMA_VERSION,
        "status": "pass" if dispatcher_ratio >= minimum_ratio else "fail",
        "protocol_v2_hash": protocol.protocol_hash,
        "reference_program": expected_program,
        "hardware_fingerprint": hardware_fingerprint,
        "gate": {
            "minimum_dispatcher_throughput_ratio": minimum_ratio,
            "observed_dispatcher_throughput_ratio": round(
                dispatcher_ratio,
                8,
            ),
            "passed": dispatcher_ratio >= minimum_ratio,
        },
        "diagnostic_recorder_to_dispatcher_throughput_ratio": round(
            recorder_to_dispatcher_ratio,
            8,
        ),
        "cases": [reports[name] for name in STAIR_REWARD_BENCHMARK_CASES],
    }
    result["report_hash"] = canonical_json_hash(result)
    return result


def _strict_case_report(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Stair reward benchmark case must be an object")
    report = dict(value)
    expected_fields = {
        "schema_version",
        "status",
        "case",
        "protocol_v2_hash",
        "num_envs",
        "warmup_control_steps",
        "measured_control_steps",
        "elapsed_seconds",
        "env_steps_per_second",
        "reward_manager_active_terms",
        "training_recorder_enabled",
        "program",
        "runtime",
        "hardware_fingerprint",
        "report_hash",
    }
    if set(report) != expected_fields:
        raise ValueError("Stair reward benchmark case fields mismatch")
    if report["schema_version"] != STAIR_REWARD_BENCHMARK_CASE_SCHEMA_VERSION:
        raise ValueError("Stair reward benchmark case schema mismatch")
    if report["status"] != "pass":
        raise ValueError("Stair reward benchmark case did not pass")
    if report["case"] not in STAIR_REWARD_BENCHMARK_CASES:
        raise ValueError("Stair reward benchmark case identity is invalid")
    declared_hash = report.pop("report_hash")
    if declared_hash != canonical_json_hash(report):
        raise ValueError("Stair reward benchmark case hash mismatch")
    report["report_hash"] = declared_hash
    for field_name in ("num_envs", "warmup_control_steps", "measured_control_steps"):
        value = report[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Stair reward benchmark {field_name} is invalid")
    for field_name in ("elapsed_seconds", "env_steps_per_second"):
        value = report[field_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"Stair reward benchmark {field_name} is invalid")
    calculated_throughput = (
        int(report["num_envs"])
        * int(report["measured_control_steps"])
        / float(report["elapsed_seconds"])
    )
    if not math.isclose(
        float(report["env_steps_per_second"]),
        calculated_throughput,
        rel_tol=1.0e-5,
        abs_tol=0.01,
    ):
        raise ValueError("Stair reward benchmark throughput accounting mismatch")
    if not isinstance(report["reward_manager_active_terms"], list):
        raise ValueError("Stair reward benchmark active terms are invalid")
    if not isinstance(report["training_recorder_enabled"], bool):
        raise ValueError("Stair reward benchmark recorder state is invalid")
    if not isinstance(report["runtime"], Mapping):
        raise ValueError("Stair reward benchmark runtime evidence is invalid")
    if not isinstance(report["hardware_fingerprint"], Mapping):
        raise ValueError("Stair reward benchmark hardware is invalid")
    report["runtime"] = dict(report["runtime"])
    report["hardware_fingerprint"] = dict(report["hardware_fingerprint"])
    hardware = report["hardware_fingerprint"]
    if set(hardware) != {
        "device",
        "gpu_name",
        "torch",
        "simulation_dt_s",
        "control_dt_s",
    }:
        raise ValueError("Stair reward benchmark hardware fields mismatch")
    for field_name in ("device", "gpu_name", "torch"):
        if not isinstance(hardware[field_name], str) or not hardware[field_name]:
            raise ValueError(
                f"Stair reward benchmark hardware {field_name} is invalid"
            )
    for field_name in ("simulation_dt_s", "control_dt_s"):
        observed = hardware[field_name]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) <= 0.0
        ):
            raise ValueError(
                f"Stair reward benchmark hardware {field_name} is invalid"
            )
    return report


__all__ = [
    "STAIR_REWARD_BENCHMARK_CASE_SCHEMA_VERSION",
    "STAIR_REWARD_BENCHMARK_CASES",
    "STAIR_REWARD_PERFORMANCE_REPORT_SCHEMA_VERSION",
    "compile_stair_reward_performance_report",
]
