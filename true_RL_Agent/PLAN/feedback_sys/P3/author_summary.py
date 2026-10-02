"""Summarize all registered author episode outcomes and charged requests.

This is descriptive engineering accounting, never a substitute for blind
semantic review or evidence of RL benefit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from author_experiment import digest, read, write


def summarize(run_dir, cases_path, preflight_path):
    cases = {case["case_id"]: case for case in read(cases_path)["cases"]}
    results = [(path, read(path)) for path in sorted((run_dir / "results").glob("*.json"))]
    expected = {(case_id, arm, repeat) for case_id in cases for arm in ("A", "T") for repeat in (1, 2)}
    if len(results) != 48 or {(row["case_id"], row["arm"], row["repetition"]) for _, row in results} != expected:
        raise ValueError("summary requires every registered final episode, including failures")
    usage, preflight = read(run_dir / "usage.json"), read(preflight_path)
    by_round = {path.stem: row for path, row in results}
    summaries = []
    for task in ("stair", "push"):
        for arm in ("A", "T"):
            selected = [row for _, row in results if row["arm"] == arm and cases[row["case_id"]]["task"] == task]
            attempts = [row for row in usage["attempts_detail"] if row["round_id"] in by_round
                and by_round[row["round_id"]]["arm"] == arm
                and cases[by_round[row["round_id"]]["case_id"]]["task"] == task]
            prompts = [row for row in preflight["prompts"] if row["arm"] == arm and cases[row["case_id"]]["task"] == task]
            summaries.append({"task": task, "arm": arm, "author_episodes": len(selected),
                "terminal_counts": dict(Counter(row["stop_reason"] for row in selected)),
                "verified_candidates": sum(row["stop_reason"] == "candidate_verified" for row in selected),
                "episodes_with_candidate_source": sum(row["candidate_private_path"] is not None for row in selected),
                "attempts": len(attempts), "unknown_usage_attempts": sum(not row.get("usage_known", False)
                    for row in attempts if row["status"] != "interrupted_before_prompt"),
                "input_tokens_actual_or_reserved": sum(row.get("input_tokens", 0) for row in attempts),
                "output_tokens_actual_or_reserved": sum(row.get("output_tokens", 0) for row in attempts),
                "usd_peak_actual_or_reserved": sum(row["usd_peak"] for row in attempts),
                "protocol_deviations": [row for row in usage.get("protocol_deviations", ())
                    if row["round_id"] in by_round and by_round[row["round_id"]]["arm"] == arm
                    and cases[by_round[row["round_id"]]["case_id"]]["task"] == task],
                "first_prompt_bytes_min": min(row["prompt_bytes"] for row in prompts),
                "first_prompt_bytes_max": max(row["prompt_bytes"] for row in prompts)})
    if sum(row["attempts"] for row in summaries) != usage["attempts"]:
        raise ValueError("charged request could not be assigned to a registered episode")
    return {"status": "COMPLETE_AUTHOR_EPISODES", "author_episodes": 48, "policy_clusters": 2,
        "observation_packages": 2, "focused_questions": 12, "summaries": summaries,
        "usage": {key: value for key, value in usage.items() if key != "attempts_detail"},
        "protocol_compliance": usage.get("protocol_compliance", "NOT_REPORTED"),
        "case_card_sha256": digest(cases_path), "preflight_sha256": digest(preflight_path),
        "result_files": [{"path": str(path), "sha256": digest(path)} for path, _ in results],
        "human_quality_gate": "PENDING_REVIEW", "candidate_training": "not_run",
        "limits": "Static/CPU admission is not semantic correctness or RL improvement. Twelve focuses reuse two policies; repetitions and same-policy cases are correlated."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = summarize(a.run, a.cases, a.preflight)
    write(a.output, report)
    print({"author_episodes": report["author_episodes"], "usage": report["usage"], "human_quality_gate": report["human_quality_gate"]})
