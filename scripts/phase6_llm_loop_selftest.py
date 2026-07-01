#!/usr/bin/env python3
"""Self-test Phase 6 loop plumbing without external APIs or Isaac training."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from importlib import util as importlib_util
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase3_verifier.verifier import _load_structured_file  # noqa: PLC2701 - self-test parser fallback.
from humanoid_agent.phase5_llm_agent.prompting import assert_prompt_text_safe, build_phase5_messages, messages_to_text
from humanoid_agent.phase6_loop.decision import build_phase6_public_decision, public_feedback_from_decision, should_expand_to_five_seeds
from humanoid_agent.phase6_loop.feedback import build_phase6_feedback_summary, assert_public_only_payload
from humanoid_agent.phase6_loop.summaries import subset_public_summary, write_json

FORBIDDEN_PROMPT_TOKENS = [
    "iid_hidden",
    "iid-hidden",
    "phase2_iid_hidden",
    "iid_hidden_success_rate",
    "iid_hidden_eval",
    "accept_reject",
]


def _sample_summary(values: list[float]) -> dict:
    seed_rows = []
    for seed, value in zip([101, 102, 103, 104, 105], values):
        seed_rows.append(
            {
                "seed": seed,
                "train_status": "pass",
                "public_eval_status": "pass",
                "iid_hidden_eval_status": "pass",
                "public_success_rate": value,
                "iid_hidden_success_rate": 0.99,
                "public_failure_breakdown": {"counts": {"success": int(value * 128)}},
                "public_condition_flag_breakdown": {"total": 128, "counts": {"terminated": 0}, "rates": {"terminated": 0.0}},
            }
        )
    return {
        "schema_version": "humanoid_agent.phase2.baseline.summary.v1",
        "status": "pass",
        "run_dir": "/tmp/phase6-selftest",
        "config": {
            "evaluation": {
                "public": {"profile": "phase2_public", "seed": 1001, "num_envs": 128, "max_steps": 220},
                "iid_hidden": {"profile": "phase2_iid_hidden", "seed": 2001, "num_envs": 128, "max_steps": 220},
                "target_success_rate_range": [0.30, 0.70],
            },
            "training": {"seeds": [101, 102, 103, 104, 105]},
        },
        "seed_results": seed_rows,
        "public_eval": {
            "success_rate_stats": {"count": len(values), "mean": sum(values) / len(values), "values": values},
            "target_success_rate_range": [0.30, 0.70],
            "target_range_met": True,
            "condition_flag_breakdown": {"total": 640, "counts": {"terminated": 0}, "rates": {"terminated": 0.0}},
        },
        "iid_hidden_eval": {"success_rate_stats": {"mean": 0.99, "values": [0.99]}},
        "compute": {"gpu_hours_estimate_total": 0.0, "gpu_hours_estimate_per_seed_mean": 0.0},
        "decision": {"phase2_workflow_complete": True, "difficulty_calibration_needed": False},
    }


def main() -> int:
    phase1 = _load_structured_file(REPO_ROOT / "configs/phase1_push_recovery.yaml")
    phase2 = _load_structured_file(REPO_ROOT / "configs/phase2_baseline.yaml")
    phase5 = _load_structured_file(REPO_ROOT / "configs/phase5_llm_agent.yaml")
    checks = []

    seed_policy = {
        "initial_train_seeds": [101, 102, 103],
        "expanded_train_seeds": [101, 102, 103, 104, 105],
        "expand_if_abs_margin_below_threshold_fraction": 0.5,
        "public_min_delta": 0.05,
        "pooled_se_multiplier": 1.0,
    }

    best3 = subset_public_summary(_sample_summary([0.45, 0.50, 0.55, 0.52, 0.48]), seeds=[101, 102, 103])
    candidate3 = subset_public_summary(_sample_summary([0.62, 0.66, 0.70, 0.52, 0.48]), seeds=[101, 102, 103])
    decision = build_phase6_public_decision(
        best_summary=best3,
        candidate_summary=candidate3,
        best_summary_path=Path("best_public_3seed.json"),
        candidate_summary_path=Path("candidate_public_3seed.json"),
        operation_diff={"operations": [{"document": "phase2", "path": "reward_weights.alive", "old": 0.25, "new": 0.3}]},
        seed_policy=seed_policy,
    )
    checks.append({"name": "clear_3seed_accept_does_not_expand", "status": "pass" if decision["accepted"] and not decision["needs_more_seeds"] else "fail"})

    checks.append(
        {
            "name": "near_threshold_3seed_expands_to_5",
            "status": "pass"
            if should_expand_to_five_seeds(public_delta=0.045, threshold=0.05, pooled_se=0.02, seed_count=3, seed_policy=seed_policy)
            else "fail",
        }
    )
    checks.append(
        {
            "name": "clear_margin_3seed_stays_3",
            "status": "pass"
            if not should_expand_to_five_seeds(public_delta=0.20, threshold=0.05, pooled_se=0.02, seed_count=3, seed_policy=seed_policy)
            else "fail",
        }
    )

    feedback = build_phase6_feedback_summary(best_public_summary=best3, history=[decision])
    assert_public_only_payload(feedback)
    prompt_text = messages_to_text(
        build_phase5_messages(
            summary=feedback,
            phase1_config=phase1,
            phase2_config=phase2,
            phase5_config=phase5,
        )
    )
    checks.append(
        {
            "name": "phase6_feedback_preserved_in_prompt",
            "status": "pass" if "phase6_feedback" in prompt_text and "reward_weights.alive" in prompt_text else "fail",
        }
    )
    checks.append(
        {
            "name": "prompt_excludes_hidden_and_eval_seed_tokens",
            "status": "pass" if not any(token in prompt_text for token in FORBIDDEN_PROMPT_TOKENS) else "fail",
        }
    )
    checks.append(
        {
            "name": "prompt_excludes_structured_eval_seed_fields",
            "status": "pass"
            if '"seed":' not in prompt_text
            and '"evaluation":' not in prompt_text
            and "phase2_public" not in prompt_text
            else "fail",
        }
    )
    try:
        assert_prompt_text_safe("benign public mean=0.820012 and delta=0.210015", context="float_false_positive_regression")
        float_guard_status = "pass"
    except ValueError:
        float_guard_status = "fail"
    checks.append({"name": "float_substrings_do_not_trigger_seed_guard", "status": float_guard_status})
    public_feedback = public_feedback_from_decision(decision)
    checks.append(
        {
            "name": "feedback_anonymizes_seed_ids",
            "status": "pass" if "\"seed\"" not in json.dumps(public_feedback, sort_keys=True) else "fail",
        }
    )

    with tempfile.TemporaryDirectory(prefix="phase6-selftest-") as tmp:
        tmp_path = Path(tmp)
        baseline_file = tmp_path / "baseline_summary.json"
        write_json(baseline_file, _sample_summary([0.45, 0.50, 0.55, 0.52, 0.48]))
        cfg_file = tmp_path / "phase6_loop.json"
        cfg = {
            "schema_version": "humanoid_agent.phase6.loop.config.v1",
            "loop": {
                "min_rounds": 1,
                "max_rounds": 1,
                "start_from": "phase2_baseline",
                "provider": "dry_run",
                "run_root": str(tmp_path / "runs"),
                "baseline_summary": str(baseline_file),
                "best_reevaluate_every_accepts": 2,
            },
            "configs": {
                "baseline_phase1": "configs/phase1_push_recovery.yaml",
                "baseline_phase2": "configs/phase2_baseline.yaml",
                "phase5_agent": "configs/phase5_llm_agent.yaml",
                "verifier": "configs/phase3_verifier.yaml",
            },
            "seed_policy": seed_policy,
            "feedback": {"include_history_rounds": 3, "public_only": True},
        }
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "scripts/phase6_multi_round_loop.py", "--phase6-config", str(cfg_file), "--rounds", "1"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            dry_summary = json.loads(completed.stdout)
        except json.JSONDecodeError:
            dry_summary = {}
        checks.append({"name": "dry_run_loop_starts_from_phase2_baseline", "status": "pass" if completed.returncode == 0 and dry_summary.get("start_from") == "phase2_baseline" else "fail"})
        prompt_file = Path(dry_summary.get("rounds", [{}])[0].get("feedback_summary", ""))
        dry_prompt_text = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else ""
        dry_prompt_ok = (
            prompt_file.exists()
            and not any(token in dry_prompt_text for token in FORBIDDEN_PROMPT_TOKENS)
            and '"seed":' not in dry_prompt_text
            and '"evaluation":' not in dry_prompt_text
            and "phase2_public" not in dry_prompt_text
        )
        checks.append({"name": "dry_run_feedback_file_is_public_only", "status": "pass" if dry_prompt_ok else "fail"})

    state_machine_checks = _orchestrator_state_machine_checks(seed_policy)
    checks.extend(state_machine_checks)

    output = {
        "schema_version": "humanoid_agent.phase6.loop.selftest.v1",
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if output["status"] == "pass" else 1


def _orchestrator_state_machine_checks(seed_policy: dict) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    module = _load_phase6_loop_script()
    original_run_phase5 = module.run_phase5_iteration
    original_run_public_eval = module._run_public_eval
    try:
        with tempfile.TemporaryDirectory(prefix="phase6-orchestrator-selftest-") as tmp:
            tmp_path = Path(tmp)
            cfg_file, baseline_file = _write_loop_test_config(tmp_path, seed_policy, rounds=2)
            eval_calls: list[dict] = []

            def fake_run_phase5_iteration(**kwargs):
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                phase1_out = output_dir / "candidate_phase1_push_recovery.yaml"
                phase2_out = output_dir / "candidate_phase2_baseline.yaml"
                phase1_out.write_text(Path(kwargs["phase1_config_file"]).read_text(encoding="utf-8"), encoding="utf-8")
                phase2_out.write_text(Path(kwargs["phase2_config_file"]).read_text(encoding="utf-8"), encoding="utf-8")
                operation_diff = output_dir / "operation_diff.json"
                write_json(
                    operation_diff,
                    {
                        "schema_version": "humanoid_agent.phase5.operation_diff.v1",
                        "operations": [
                            {
                                "document": "phase2",
                                "path": "reward_weights.alive",
                                "old": 0.25,
                                "new": 0.3,
                                "reason": "orchestrator selftest",
                            }
                        ],
                    },
                )
                return {
                    "status": "pass",
                    "candidate_phase1_config": str(phase1_out),
                    "candidate_phase2_config": str(phase2_out),
                    "operation_diff": str(operation_diff),
                    "errors": [],
                }

            def fake_run_public_eval(*, candidate_dir, phase1_config, phase2_config, seeds, run_name):
                eval_calls.append({"candidate_dir": str(candidate_dir), "seeds": list(seeds), "run_name": run_name})
                out = Path(candidate_dir) / run_name / "summary.json"
                if run_name == "public_3seed_eval" and "round_01" in str(candidate_dir):
                    summary = subset_public_summary(_sample_summary([0.51, 0.55, 0.59, 0.50, 0.49]), seeds=list(seeds))
                elif run_name == "public_5seed_eval":
                    summary = subset_public_summary(_sample_summary([0.46, 0.50, 0.54, 0.50, 0.49]), seeds=list(seeds))
                else:
                    summary = subset_public_summary(_sample_summary([0.47, 0.51, 0.55, 0.52, 0.48]), seeds=list(seeds))
                write_json(out, summary)
                return out

            module.run_phase5_iteration = fake_run_phase5_iteration
            module._run_public_eval = fake_run_public_eval
            output = StringIO()
            with redirect_stdout(output):
                rc = module.main(["--phase6-config", str(cfg_file), "--rounds", "2", "--execute", "--provider", "dry_run"])
            summary = json.loads(output.getvalue())
            round2_best = summary["rounds"][1]["best_summary"]
            checks.append(
                {
                    "name": "orchestrator_rejected_5seed_keeps_3seed_best_path",
                    "status": "pass"
                    if rc == 0
                    and summary["rounds"][0]["status"] == "rejected"
                    and "best_public_5seed_summary" not in round2_best
                    and "round_00_baseline_public_3seed_summary" in round2_best
                    else "fail",
                }
            )
            checks.append(
                {
                    "name": "orchestrator_expands_only_near_threshold",
                    "status": "pass" if any(item["run_name"] == "public_5seed_eval" for item in eval_calls) else "fail",
                }
            )
    finally:
        module.run_phase5_iteration = original_run_phase5
        module._run_public_eval = original_run_public_eval

    module = _load_phase6_loop_script()
    original_run_phase5 = module.run_phase5_iteration
    original_run_public_eval = module._run_public_eval
    try:
        with tempfile.TemporaryDirectory(prefix="phase6-eval-failure-selftest-") as tmp:
            tmp_path = Path(tmp)
            cfg_file, _ = _write_loop_test_config(tmp_path, seed_policy, rounds=2)

            def fake_run_phase5_iteration(**kwargs):
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                phase1_out = output_dir / "candidate_phase1_push_recovery.yaml"
                phase2_out = output_dir / "candidate_phase2_baseline.yaml"
                phase1_out.write_text(Path(kwargs["phase1_config_file"]).read_text(encoding="utf-8"), encoding="utf-8")
                phase2_out.write_text(Path(kwargs["phase2_config_file"]).read_text(encoding="utf-8"), encoding="utf-8")
                operation_diff = output_dir / "operation_diff.json"
                write_json(operation_diff, {"schema_version": "humanoid_agent.phase5.operation_diff.v1", "operations": []})
                return {
                    "status": "pass",
                    "candidate_phase1_config": str(phase1_out),
                    "candidate_phase2_config": str(phase2_out),
                    "operation_diff": str(operation_diff),
                    "errors": [],
                }

            def failing_then_passing_eval(*, candidate_dir, phase1_config, phase2_config, seeds, run_name):
                if "round_01" in str(candidate_dir):
                    raise RuntimeError("synthetic public eval failure")
                out = Path(candidate_dir) / run_name / "summary.json"
                write_json(out, subset_public_summary(_sample_summary([0.47, 0.51, 0.55, 0.52, 0.48]), seeds=list(seeds)))
                return out

            module.run_phase5_iteration = fake_run_phase5_iteration
            module._run_public_eval = failing_then_passing_eval
            output = StringIO()
            with redirect_stdout(output):
                rc = module.main(["--phase6-config", str(cfg_file), "--rounds", "2", "--execute", "--provider", "dry_run"])
            summary = json.loads(output.getvalue())
            checks.append(
                {
                    "name": "orchestrator_eval_failure_rejects_and_continues",
                    "status": "pass"
                    if rc == 0
                    and len(summary["rounds"]) == 2
                    and summary["rounds"][0]["status"] == "eval_failed_rejected"
                    and summary["rounds"][1]["round"] == 2
                    else "fail",
                }
            )
    finally:
        module.run_phase5_iteration = original_run_phase5
        module._run_public_eval = original_run_public_eval

    return checks


def _load_phase6_loop_script():
    module_path = REPO_ROOT / "scripts/phase6_multi_round_loop.py"
    spec = importlib_util.spec_from_file_location("phase6_multi_round_loop_selftest_target", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_loop_test_config(tmp_path: Path, seed_policy: dict, *, rounds: int) -> tuple[Path, Path]:
    baseline_file = tmp_path / "baseline_summary.json"
    write_json(baseline_file, _sample_summary([0.45, 0.50, 0.55, 0.52, 0.48]))
    cfg_file = tmp_path / "phase6_loop.json"
    cfg = {
        "schema_version": "humanoid_agent.phase6.loop.config.v1",
        "loop": {
            "min_rounds": rounds,
            "max_rounds": rounds,
            "start_from": "phase2_baseline",
            "provider": "dry_run",
            "run_root": str(tmp_path / "runs"),
            "baseline_summary": str(baseline_file),
            "best_reevaluate_every_accepts": 2,
        },
        "configs": {
            "baseline_phase1": "configs/phase1_push_recovery.yaml",
            "baseline_phase2": "configs/phase2_baseline.yaml",
            "phase5_agent": "configs/phase5_llm_agent.yaml",
            "verifier": "configs/phase3_verifier.yaml",
        },
        "seed_policy": seed_policy,
        "feedback": {"include_history_rounds": 3, "public_only": True},
    }
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_file, baseline_file


if __name__ == "__main__":
    raise SystemExit(main())
