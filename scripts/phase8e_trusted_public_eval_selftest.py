#!/usr/bin/env python3
"""Self-test for Phase 8E-6-1 trusted public eval metric projection."""

from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.public_eval import build_phase8e_public_result  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _phase1_summary(*, seed: int, success_rate: float, status: str = "pass", action_l2: float = 0.4) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase1.push_recovery.summary.v1",
        "status": status,
        "run": {"seed": seed},
        "aggregate": {
            "episodes": 8,
            "success_count": int(round(success_rate * 8)),
            "success_rate": success_rate,
            "failure_breakdown": {"total": 8, "counts": {"success": int(round(success_rate * 8))}},
            "condition_flag_breakdown": {"total": 8, "counts": {"height_below": 0}, "rates": {"height_below": 0.0}},
        },
        "secondary_metrics": {
            "mean_action_l2": action_l2,
            "mean_action_rate_l2": action_l2 * 0.5,
            "mean_final_xy_drift_m": 0.15,
            "mean_max_abs_pitch_rad": 0.08,
            "mean_max_abs_roll_rad": 0.05,
        },
    }


def _budget_summary() -> dict[str, Any]:
    return {
        "status": "pass",
        "budget_accounting": {
            "status": "pass",
            "config_derived_total_timesteps": 49152000,
            "planned_env_steps": 49152000,
            "num_envs": 4096,
            "num_steps_per_env": 24,
            "max_iterations": 500,
        },
    }


def _write_safe_checkpoint(path: Path) -> None:
    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {"linear.weight": torch.ones((1, 1))}}, path)


def _assert_public_payload_safe(value: dict[str, Any]) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    for forbidden in ("1001", "2001", "iid_hidden", "hidden_eval", "accept_reject"):
        _assert(forbidden not in text, f"public payload leaked forbidden token: {forbidden}")


def test_build_public_result_from_runner_owned_summaries() -> None:
    result = build_phase8e_public_result(
        public_dev_summaries=[
            _phase1_summary(seed=1001, success_rate=0.50),
            _phase1_summary(seed=2001, success_rate=0.75),
        ],
        public_accept_summaries=[_phase1_summary(seed=1001, success_rate=0.625)],
        budget_summary=_budget_summary(),
    )

    _assert(result["schema_version"] == "humanoid_agent.phase8e.public_result.v1", "wrong schema")
    _assert(result["status"] == "pass", "completed public eval should report pass even before accept decision")
    _assert(result["public_dev"]["success_rate_stats"]["count"] == 2, "dev count mismatch")
    _assert(result["public_dev"]["success_rate_stats"]["mean"] == 0.625, "dev mean mismatch")
    _assert(result["public_accept"]["success_rate_stats"]["mean"] == 0.625, "accept mean mismatch")
    _assert(result["budget_accounting"]["status"] == "pass", "budget gate should pass")
    _assert(result["motion_quality_gate"]["status"] == "pass", "motion gate should pass for sane summaries")
    _assert(result["ood_stress_gate"]["status"] == "not_run", "OOD gate should be diagnostic only in 8E-6-1")
    for item in result["public_dev"]["seed_results"]:
        _assert("seed" not in item, "public result must not expose raw eval seeds")
    _assert_public_payload_safe(result)


def test_motion_gate_fails_without_failing_eval_completion() -> None:
    result = build_phase8e_public_result(
        public_dev_summaries=[_phase1_summary(seed=1001, success_rate=0.50, action_l2=2000.0)],
        public_accept_summaries=[],
        budget_summary=_budget_summary(),
    )
    _assert(result["status"] == "pass", "eval completion should be distinct from motion gate pass/fail")
    _assert(result["motion_quality_gate"]["status"] == "fail", "absurd action magnitude must fail motion gate")
    _assert_public_payload_safe(result)


def test_hidden_or_iid_summary_is_rejected_before_projection() -> None:
    hidden = _phase1_summary(seed=2001, success_rate=0.99)
    hidden["configuration"] = {"active_profile": "iid_hidden_eval"}
    result = build_phase8e_public_result(
        public_dev_summaries=[hidden],
        public_accept_summaries=[],
        budget_summary=_budget_summary(),
    )
    text = json.dumps(result, ensure_ascii=False, sort_keys=True)
    _assert(result["status"] == "fail", "hidden/iid source summary must fail public projection")
    _assert("forbidden_source_token" in text, "hidden/iid rejection reason must be explicit")
    _assert("0.99" not in text, "hidden/iid success rate must not be projected")
    _assert("iid_hidden" not in text, "hidden/iid marker must not be echoed into public output")


def test_phase1_checkpoint_eval_gates_safe_load_before_runner_load() -> None:
    source_path = PROJECT_ROOT / "scripts" / "phase1_push_recovery_eval.py"
    source = source_path.read_text(encoding="utf-8")
    assert_index = source.find("assert_safe_checkpoint(args_cli.checkpoint")
    load_index = source.find("runner.load(args_cli.checkpoint)")
    _assert(assert_index >= 0, "phase1 checkpoint eval must call assert_safe_checkpoint on args_cli.checkpoint")
    _assert(load_index >= 0, "phase1 checkpoint eval must still call runner.load for checkpoint policy")
    _assert(assert_index < load_index, "assert_safe_checkpoint must run before runner.load")


def test_trusted_eval_probe_writes_public_result_without_candidate_code() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    with tempfile.TemporaryDirectory(prefix="phase8e_public_eval_") as tmp:
        root = Path(tmp)
        checkpoint = root / "sealed_checkpoints" / "model_1.pt"
        _write_safe_checkpoint(checkpoint)
        dev = root / "dev_summary.json"
        accept = root / "accept_summary.json"
        budget = root / "budget_summary.json"
        trusted_summary = root / "trusted_eval_summary.json"
        public_result = root / "public_result.json"
        _write_json(dev, _phase1_summary(seed=1001, success_rate=0.50))
        _write_json(accept, _phase1_summary(seed=2001, success_rate=0.75))
        _write_json(budget, _budget_summary())

        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "phase8e_trusted_eval_probe.py"),
                "--checkpoint",
                str(checkpoint),
                "--summary-file",
                str(trusted_summary),
                "--public-result-file",
                str(public_result),
                "--public-dev-summary",
                str(dev),
                "--public-accept-summary",
                str(accept),
                "--budget-summary",
                str(budget),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        _assert(proc.returncode == 0, f"trusted eval probe failed: stdout={proc.stdout} stderr={proc.stderr}")
        summary = json.loads(trusted_summary.read_text(encoding="utf-8"))
        result = json.loads(public_result.read_text(encoding="utf-8"))
        _assert(summary["status"] == "pass", "trusted eval summary should pass")
        _assert(summary["candidate_code_loaded"] is False, "trusted eval must not import candidate code")
        _assert("public_eval_placeholder" not in summary, "real metric mode must not emit placeholder")
        _assert(result["public_dev"]["success_rate_stats"]["mean"] == 0.50, "public dev mean mismatch")
        _assert(result["public_accept"]["success_rate_stats"]["mean"] == 0.75, "public accept mean mismatch")
        _assert_public_payload_safe(result)


def main() -> int:
    test_build_public_result_from_runner_owned_summaries()
    test_motion_gate_fails_without_failing_eval_completion()
    test_hidden_or_iid_summary_is_rejected_before_projection()
    test_phase1_checkpoint_eval_gates_safe_load_before_runner_load()
    test_trusted_eval_probe_writes_public_result_without_candidate_code()
    print(
        json.dumps(
            {
                "status": "pass",
                "test": "phase8e_trusted_public_eval_selftest",
                "trusted_eval_probe_cli": "skipped_no_torch" if importlib.util.find_spec("torch") is None else "pass",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
