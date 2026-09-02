#!/usr/bin/env python3
"""Self-test the Phase 4 rule baseline generator."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase3_verifier.verifier import (
    _load_structured_file,  # noqa: PLC2701 - self-test uses parser fallback.
    load_verifier_config,
    verify_static_documents,
)
from humanoid_agent.phase4_rule_baseline.rules import build_rule_candidate
from humanoid_agent.phase4_rule_baseline.rules import dump_yaml_like


def _sample_summary() -> dict:
    return {
        "schema_version": "humanoid_agent.phase2.baseline.summary.v1",
        "status": "pass",
        "run_dir": "/tmp/sample",
        "public_eval": {
            "success_rate_stats": {
                "mean": 0.52,
                "values": [0.49, 0.42, 0.75, 0.41, 0.52],
            },
            "secondary_metrics": {
                "oscillation_rate": 0.31,
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
        "seed_results": [
            {"public_failure_breakdown": {"counts": {"fall_backward": 41, "fall_forward": 24, "success": 63}}},
            {"public_failure_breakdown": {"counts": {"fall_backward": 37, "fall_forward": 35, "fall_side": 2, "success": 54}}},
            {"public_failure_breakdown": {"counts": {"fall_backward": 15, "fall_forward": 12, "fall_side": 4, "success": 97}}},
            {"public_failure_breakdown": {"counts": {"fall_backward": 40, "fall_forward": 31, "fall_side": 4, "success": 53}}},
            {"public_failure_breakdown": {"counts": {"fall_backward": 35, "fall_forward": 23, "fall_side": 3, "success": 67}}},
        ],
    }


def main() -> int:
    phase1 = _load_structured_file(REPO_ROOT / "configs/phase1_push_recovery.yaml")
    phase2 = _load_structured_file(REPO_ROOT / "configs/phase2_baseline.yaml")
    rule_cfg = _load_structured_file(REPO_ROOT / "configs/phase4_rule_baseline.yaml")
    verifier_cfg = load_verifier_config(REPO_ROOT / "configs/phase3_verifier.yaml")

    candidate = build_rule_candidate(
        summary=_sample_summary(),
        phase1_config=phase1,
        phase2_config=phase2,
        rule_config=rule_cfg,
    )

    result = verify_static_documents(
        base_docs={
            "configs/phase1_push_recovery.yaml": phase1,
            "configs/phase2_baseline.yaml": phase2,
            "configs/phase3_verifier.yaml": verifier_cfg,
        },
        current_docs={
            "configs/phase1_push_recovery.yaml": candidate["phase1_config"],
            "configs/phase2_baseline.yaml": candidate["phase2_config"],
            "configs/phase3_verifier.yaml": verifier_cfg,
        },
        changed_files=["configs/phase1_push_recovery.yaml", "configs/phase2_baseline.yaml"],
        settings=verifier_cfg,
        check_imports=False,
    )

    checks = []
    push_x = candidate["phase1_config"]["profiles"]["phase2_train"]["push"]["velocity_range_mps"]["x"]
    checks.append({"name": "training_push_enabled", "status": "pass" if push_x != [0.0, 0.0] else "fail"})
    checks.append(
        {
            "name": "eval_public_unchanged",
            "status": "pass"
            if candidate["phase1_config"]["profiles"]["phase2_public"] == phase1["profiles"]["phase2_public"]
            else "fail",
        }
    )
    checks.append(
        {
            "name": "reward_changed",
            "status": "pass"
            if candidate["phase2_config"]["reward_weights"] != phase2["reward_weights"]
            else "fail",
        }
    )
    intervention_labels = {item["label"] for item in candidate["analysis"]["selected_interventions"]}
    checks.append(
        {
            "name": "oscillation_rule_fired",
            "status": "pass" if "oscillation" in intervention_labels else "fail",
        }
    )
    scientific_text = dump_yaml_like({"small": -2e-5, "large": 1e20})
    checks.append(
        {
            "name": "yaml_scientific_float_safe",
            "status": "pass"
            if "small: -2.0e-05" in scientific_text and "large: 1.0e+20" in scientific_text
            else "fail",
        }
    )
    checks.append({"name": "phase3_verifier_accepts_candidate", "status": result["status"]})

    output = {
        "schema_version": "humanoid_agent.phase4.rule_baseline.selftest.v1",
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
        "analysis": candidate["analysis"],
        "verifier_issues": result["issues"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if output["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
