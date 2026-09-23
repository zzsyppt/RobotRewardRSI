"""Freeze five-round Go2 Push execution costs from existing local evidence; no jobs/API."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.agent_core.generation_cost import (
    FrozenGenerationCostContract, GenerationProviderBudget, OperationCostReserve,
    generation_operation_layout,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest

HERE = Path(__file__).resolve().parent
P6 = HERE.parent / "P6"
P5 = HERE.parent / "P5"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(name, payload):
    (HERE / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    train_root = P6 / "go2_dev_D1"
    train_process_path = train_root / "process/process.json"
    train_summary_path = train_root / "dev/composition/runner/p6-go2-matched2-dev-20260912/train/summary.json"
    train_stdout_path = train_summary_path.parent / "runner_process/stdout.log"
    dev_process_path = P6 / "force_unified_01/go2_public_dev_03_process/process.json"
    dev_result_path = P6 / "force_unified_01/go2_public_dev_03/result.json"
    prior_policy_path = P5 / "provider_live_01/policy.json"
    capacity_path = P5 / "audit_capacity/result.json"
    train_process, train_summary, dev_process = map(read, (train_process_path, train_summary_path, dev_process_path))
    assert train_process["returncode"] == dev_process["returncode"] == 0
    assert train_summary["training"]["budget_accounting"]["planned_env_steps"] == 49_152_000
    assert read(dev_result_path)["aggregate"]["episodes"] == 192
    train_s = train_process["ended_at_unix"] - train_process["started_at_unix"]
    dev_s = dev_process["ended_at_unix"] - dev_process["started_at_unix"]
    elapsed = re.findall(r"Time elapsed:\s*(\d+):(\d+):(\d+)", train_stdout_path.read_text(encoding="utf-8"))[-1]
    learning_elapsed_s = sum(int(value) * unit for value, unit in zip(elapsed, (3600, 60, 1)))
    old_policy = read(prior_policy_path)
    policy = {
        "api_key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com",
        "endpoint": "/chat/completions", "model": "deepseek-flash",
        "max_output_tokens": old_policy["max_tokens"], "timeout_s": 240,
        "temperature": 0.2, "reasoning_mode": old_policy["reasoning_mode"],
        "reasoning_effort": old_policy["reasoning_effort"],
        "empty_final_reasoning_retry": old_policy["empty_final_reasoning_retry"],
        "max_attempts_per_lane": old_policy["max_attempts"],
        "input_units_per_attempt_reserve": 65536,
        "model_context": {**old_policy["model_context"],
            "specification_source": "https://api-docs.deepseek.com/quick_start/pricing/ (verified 2026-09-22; DeepSeek-V4.1-Flash)"},
    }
    config = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v3")
    plugin = PushRecoveryTaskPlugin(config)
    manifest = resolve_push_cell_manifest(cell_variant_id=config.cell_variant_id, config=config, plugin=plugin)
    plugin = PushRecoveryTaskPlugin(replace(config, resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash))
    workload = plugin.phased_evaluation_adapter().generation_workload(cell_manifest=manifest)
    assert workload.train_steps_per_job == 49_152_000
    assert workload.max_eval_control_steps_per_seed_unit == 64 * 303
    layout = generation_operation_layout(max_generations=5, imports_reference=False, final_jobs=2)
    reserves = {}
    measured_output_lower_bounds = {
        "train_and_dev": sum(path.stat().st_size for path in train_root.rglob("*") if path.is_file()),
        "force_dev": sum(path.stat().st_size for path in dev_result_path.parent.rglob("*") if path.is_file()),
    }
    for name, (kind, _) in layout.items():
        if "train" in name:
            measured_s, reserved_s, output_bytes = train_s, 1816, 1024**3
            measured_bytes = measured_output_lower_bounds["train_and_dev"]
        elif name == "reference_qualification":
            measured_s, reserved_s, output_bytes = dev_s * 13 / 3, 1378, 278 * 1024**2
            measured_bytes = math.ceil(measured_output_lower_bounds["force_dev"] * 13 / 3)
        elif "sealed" in name:
            measured_s, reserved_s, output_bytes = dev_s * 7 / 3, 716, 150 * 1024**2
            measured_bytes = math.ceil(measured_output_lower_bounds["force_dev"] * 7 / 3)
        else:
            measured_s, reserved_s, output_bytes = dev_s, 316, 64 * 1024**2
            measured_bytes = measured_output_lower_bounds["force_dev"]
        reserves[name] = OperationCostReserve(name, kind, 1,
            math.ceil(measured_s * 1_000_000_000), measured_bytes,
            reserved_s * 1_000_000_000, output_bytes,
            "predeclared_operation_timeout_plus_3x5s_cleanup_plus_1s; measured_source_is_not_timeout")
    cost = FrozenGenerationCostContract.freeze(max_generations=5, workload=workload,
        provider_budget=GenerationProviderBudget(5, 65536, 32768), operation_sources=reserves,
        imports_reference=False, final_confirmation=True, finalization_output_reserve_bytes=128 * 1024**2,
        session_output_reserve_bytes=256 * 1024**2, headroom_numerator=1, headroom_denominator=1)
    assert FrozenGenerationCostContract.from_mapping(cost.to_dict()) == cost
    audit = {
        "schema_version": "true_rl_agent.go2_push.five_round_cost_sources.v1", "date": "2026-09-22",
        "sources": [source(path) for path in (train_process_path, train_summary_path, train_stdout_path,
            dev_process_path, dev_result_path, prior_policy_path, capacity_path)],
        "cell_variant_id": manifest.cell_variant_id, "resolved_cell_manifest_hash": manifest.manifest_hash,
        "cost_contract_hash": cost.contract_hash, "provider_policy_hash": canonical_json_hash(policy),
        "observations": {"go2_D_learning_loop_elapsed_s": learning_elapsed_s,
            "go2_D_complete_train_plus_dev_outer_s": train_s, "force_dev_3_seeds_outer_s": dev_s,
            "historical_provider_max_measured_prompt_tokens": max(item["usage"]["prompt_tokens"] for item in read(capacity_path)["frames"].values()),
            "downloaded_output_bytes_lower_bounds": measured_output_lower_bounds},
        "measurement_limits": ["D outer timing includes training plus velocity-domain dev; it conservatively bounds that historical train, not all future candidates.",
            "Force public-dev timing comes from production runtime via the P6 acceptance driver; current three-phase formal Runner execution has not yet been measured.",
            "Accept uses the dev timing estimate; sealed and reference qualification use seed-count extrapolation, not claimed measurements.",
            "Downloaded output bytes exclude unsaved checkpoint/runtime files; frozen larger output allocations are operational reserves, not observed full output sizes.",
            "Historical provider usage is Stair data. 65536 input tokens is a new declared request reserve, not a measured Push tokenizer count.",
            "deepseek-flash now denotes DeepSeek-V4.1-Flash; this is not a claim to replay the retired V4 model."],
        "runtime_envelopes": {"runner_timeout_s": config.runner_timeout_s, "termination_grace_s": 5,
            "operation_process_caps_s": {"train": 1800, "public_dev": 300, "public_accept": 300,
                "sealed_validation": 700, "reference_qualification_split": [302, 302, 726]},
            "source_to_reserve_rule": "Explicit timeout/output bounds override source measurements; no hidden proportional headroom."},
        "reference": {"mode": "new_native_reference_train_and_three_phase_qualification", "imports_reference_evidence": False,
            "old_checkpoint_is_cost_evidence_only": train_summary["checkpoint"]["checksum"],
            "no_compatible_force_domain_genesis_available": True},
        "provider_env_file": "/data/yuyangliu/secrets/deepseek.env", "credential_read": False,
        "physical_cap": cost.to_dict()["physical_cap"], "provider_cap": cost.to_dict()["provider"],
    }
    write("provider_policy.json", policy)
    write("cost_contract.json", cost.to_dict())
    write("cost_sources.json", audit)
    print(json.dumps({"status": "prepared", "physical_cap": audit["physical_cap"], "provider_cap": audit["provider_cap"]}))


if __name__ == "__main__":
    main()
