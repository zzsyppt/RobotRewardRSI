#!/usr/bin/env python3
"""Collect Phase 1 smoke outputs into one gate summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from humanoid_agent.phase1_push_recovery.config import load_config


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/phase1_push_recovery.yaml")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(args.config)
    gates_cfg = config["smoke_gates"]

    paths = {
        "zero_strong_push": run_dir / "zero_strong_push_summary.json",
        "random_strong_push": run_dir / "random_strong_push_summary.json",
        "zero_no_push": run_dir / "zero_no_push_summary.json",
        "train_smoke": run_dir / "train_smoke_summary.json",
    }
    summaries = {name: _load(path) if path.exists() else {"status": "missing"} for name, path in paths.items()}

    gates: dict[str, dict] = {}
    for name, summary in summaries.items():
        gates[f"{name}_status_pass"] = {
            "pass": summary.get("status") == "pass",
            "observed": summary.get("status"),
            "expected": "pass",
        }

    zero_strong_push_effect = summaries["zero_strong_push"].get("evidence", {}).get("push_effect", {})
    gates["zero_strong_push_effect_observed"] = {
        "pass": bool(zero_strong_push_effect.get("single_step_gate_pass", zero_strong_push_effect.get("gate_pass", False))),
        "observed": zero_strong_push_effect,
        "expected": "clear single-step root velocity jump near configured push time",
    }

    zero_no_push_effect = summaries["zero_no_push"].get("evidence", {}).get("push_effect", {})
    strong_single_step_delta = zero_strong_push_effect.get("mean_single_step_delta_mps")
    no_push_single_step_delta = zero_no_push_effect.get("mean_single_step_delta_mps")
    margin = float(gates_cfg["push_effect_min_margin_over_no_push_mps"])
    gates["strong_push_delta_exceeds_no_push_control"] = {
        "pass": strong_single_step_delta is not None
        and no_push_single_step_delta is not None
        and float(strong_single_step_delta) >= float(no_push_single_step_delta) + margin,
        "observed": {
            "strong_push_mean_single_step_delta_mps": strong_single_step_delta,
            "no_push_mean_single_step_delta_mps": no_push_single_step_delta,
            "margin_mps": margin,
        },
        "expected": "strong push single-step delta >= no-push control delta + margin",
    }

    random_success = summaries["random_strong_push"].get("aggregate", {}).get("success_rate")
    gates["random_policy_mostly_fails_under_strong_push"] = {
        "pass": random_success is not None
        and float(random_success) <= float(gates_cfg["random_strong_push_max_success_rate"]),
        "observed": random_success,
        "expected": f"<= {gates_cfg['random_strong_push_max_success_rate']}",
    }

    zero_no_push_steps = summaries["zero_no_push"].get("secondary_metrics", {}).get("mean_episode_steps")

    for name in ("zero_strong_push", "random_strong_push", "zero_no_push"):
        summary = summaries[name]
        gates[f"{name}_reward_finite"] = {
            "pass": bool(summary.get("evidence", {}).get("reward_finite", False)),
            "observed": summary.get("evidence", {}).get("reward_finite"),
            "expected": True,
        }
        gates[f"{name}_observation_finite"] = {
            "pass": bool(summary.get("evidence", {}).get("observation_finite", False)),
            "observed": summary.get("evidence", {}).get("observation_finite"),
            "expected": True,
        }

    overall_pass = all(item["pass"] for item in gates.values())
    output = {
        "schema_version": "humanoid_agent.phase1.push_recovery.smoke_summary.v1",
        "status": "pass" if overall_pass else "fail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "input_files": {name: str(path) for name, path in paths.items()},
        "gates": gates,
        "key_results": {
            "zero_strong_push_success_rate": summaries["zero_strong_push"].get("aggregate", {}).get("success_rate"),
            "random_strong_push_success_rate": random_success,
            "zero_no_push_success_rate": summaries["zero_no_push"].get("aggregate", {}).get("success_rate"),
            "zero_no_push_mean_episode_steps": zero_no_push_steps,
            "zero_strong_push_mean_single_step_delta_mps": strong_single_step_delta,
            "zero_no_push_mean_single_step_delta_mps": no_push_single_step_delta,
            "train_checkpoint": summaries["train_smoke"].get("training", {}).get("latest_checkpoint"),
        },
        "diagnostics": {
            "zero_no_push_failure_breakdown": summaries["zero_no_push"].get("aggregate", {}).get("failure_breakdown"),
            "random_strong_push_failure_breakdown": summaries["random_strong_push"].get("aggregate", {}).get("failure_breakdown"),
            "classification_note": (
                "Until Phase 2 produces a trained baseline, zero/random policy failures are uncontrolled-policy "
                "references, not push-recovery policy diagnostics."
            ),
        },
        "notes": [
            "Phase 1 gates validate environment mechanics and protected reporting, not final policy quality.",
            "Push effect is gated by a single-step root velocity jump and by comparison against the no-push control.",
            "Short training smoke is a runability/export/checkpoint gate; Phase 2 measures seed variance and real baseline performance.",
        ],
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_file": str(summary_path), "status": output["status"]}, ensure_ascii=False))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
