#!/usr/bin/env python3
"""Self-test the Phase 5 LLM-agent scaffold without calling external APIs."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase3_verifier.verifier import _load_structured_file  # noqa: PLC2701 - self-test uses parser fallback.
from humanoid_agent.phase5_llm_agent.prompting import build_phase5_messages, messages_to_text
from humanoid_agent.phase5_llm_agent.patching import PatchContractError, apply_llm_patch
from humanoid_agent.phase5_llm_agent.providers import ProviderError, call_provider
from humanoid_agent.phase5_llm_agent.runner import run_phase5_iteration


def _sample_summary() -> dict:
    return {
        "schema_version": "humanoid_agent.phase2.baseline.summary.v1",
        "status": "pass",
        "run_dir": "/tmp/sample_phase5",
        "public_eval": {
            "success_rate_stats": {
                "mean": 0.52,
                "values": [0.49, 0.42, 0.75, 0.41, 0.52],
            },
            "condition_flag_breakdown": {
                "rates": {
                    "height_below_threshold": 0.46,
                    "drift_over_threshold": 0.45,
                    "pitch_over_threshold": 0.47,
                    "roll_over_threshold": 0.44,
                    "terminated": 0.0,
                }
            },
        },
        "iid_hidden_eval": {
            "success_rate_stats": {
                "mean": 0.99,
                "values": [0.99],
            },
            "condition_flag_breakdown": {"rates": {"leaked_flag": 1.0}},
        },
        "config": {
            "evaluation": {
                "public": {"profile": "phase2_public", "seed": 1001, "num_envs": 128, "max_steps": 220},
                "iid_hidden": {"profile": "phase2_iid_hidden", "seed": 2001, "num_envs": 128, "max_steps": 220},
            }
        },
        "seed_results": [
            {
                "seed": 101,
                "public_success_rate": 0.49,
                "public_failure_breakdown": {"counts": {"fall_backward": 41, "success": 63}},
                "iid_hidden_success_rate": 0.99,
                "iid_hidden_failure_breakdown": {"counts": {"success": 128}},
            },
            {
                "seed": 102,
                "public_success_rate": 0.42,
                "public_failure_breakdown": {"counts": {"fall_backward": 37, "success": 54}},
                "iid_hidden_success_rate": 0.98,
                "iid_hidden_failure_breakdown": {"counts": {"success": 127}},
            },
        ],
        "compute": {"gpu_hours_estimate_total": 0.0},
    }


def main() -> int:
    phase1 = _load_structured_file(REPO_ROOT / "configs/phase1_push_recovery.yaml")
    phase2 = _load_structured_file(REPO_ROOT / "configs/phase2_baseline.yaml")
    phase5 = _load_structured_file(REPO_ROOT / "configs/phase5_llm_agent.yaml")
    reward_bounds = _load_structured_file(REPO_ROOT / "configs/phase4_rule_baseline.yaml")["reward_weight_bounds"]
    checks = []

    with tempfile.TemporaryDirectory(prefix="phase5-selftest-") as tmp:
        tmp_path = Path(tmp)
        summary_file = tmp_path / "summary.json"
        summary_file.write_text(json.dumps(_sample_summary(), ensure_ascii=False), encoding="utf-8")
        prompt_text = messages_to_text(
            build_phase5_messages(
                summary=_sample_summary(),
                phase1_config=phase1,
                phase2_config=phase2,
                phase5_config=phase5,
            )
        )
        forbidden_prompt_tokens = [
            "iid_hidden",
            "iid-hidden",
            "phase2_iid_hidden",
            "iid_hidden_success_rate",
            "iid_hidden_eval",
            "2001",
            "1001",
            "leaked_flag",
        ]
        checks.append(
            {
                "name": "prompt_redacts_iid_hidden_and_eval_seed_fields",
                "status": "pass" if not any(token in prompt_text for token in forbidden_prompt_tokens) else "fail",
            }
        )
        run_summary = run_phase5_iteration(
            repo_root=REPO_ROOT,
            summary_file=summary_file,
            phase1_config_file=Path("configs/phase1_push_recovery.yaml"),
            phase2_config_file=Path("configs/phase2_baseline.yaml"),
            phase5_config_file=Path("configs/phase5_llm_agent.yaml"),
            verifier_config_file=Path("configs/phase3_verifier.yaml"),
            output_dir=tmp_path / "out",
            provider_name="dry_run",
        )
        checks.append({"name": "dry_run_iteration_passes", "status": "pass" if run_summary["status"] == "pass" else "fail"})
        analysis = json.loads((tmp_path / "out" / "analysis.json").read_text(encoding="utf-8"))
        checks.append({"name": "analysis_records_no_secret", "status": "pass" if "api" not in json.dumps(analysis).lower() or "api keys are read only" in json.dumps(analysis).lower() else "fail"})
        operation_diff = json.loads((tmp_path / "out" / "operation_diff.json").read_text(encoding="utf-8"))
        checks.append(
            {
                "name": "operation_only_diff_written",
                "status": "pass"
                if operation_diff.get("schema_version") == "humanoid_agent.phase5.operation_diff.v1"
                and len(operation_diff.get("operations", [])) > 0
                else "fail",
            }
        )
        verifier = json.loads((tmp_path / "out" / "verifier_summary.json").read_text(encoding="utf-8"))
        checks.append({"name": "phase3_verifier_accepts_dry_run_candidate", "status": verifier["status"]})

    bad_response = {
        "analysis": {"hypothesis": "bad"},
        "patch": {
            "operations": [
                {
                    "op": "set",
                    "document": "phase2",
                    "path": "evaluation.public.seed",
                    "value": 9999,
                    "reason": "illegal",
                }
            ]
        },
    }
    try:
        apply_llm_patch(
            phase1_config=copy.deepcopy(phase1),
            phase2_config=copy.deepcopy(phase2),
            response=bad_response,
            phase5_config=phase5,
            reward_bounds=reward_bounds,
        )
        protected_status = "fail"
    except PatchContractError:
        protected_status = "pass"
    checks.append({"name": "protected_eval_path_rejected_before_verifier", "status": protected_status})

    try:
        call_provider(
            provider_name="custom_openai_compatible",
            messages=[],
            phase5_config=phase5,
            allow_api_spend=False,
        )
        spend_guard_status = "fail"
    except ProviderError:
        spend_guard_status = "pass"
    checks.append({"name": "openai_compatible_requires_explicit_spend_flag", "status": spend_guard_status})

    output = {
        "schema_version": "humanoid_agent.phase5.llm_agent.selftest.v1",
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
