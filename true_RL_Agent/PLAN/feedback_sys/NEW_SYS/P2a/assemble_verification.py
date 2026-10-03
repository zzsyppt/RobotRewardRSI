"""Index immutable local evidence and final outcomes; excludes GB evidence from Git."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/p2a"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def ref(p):
    p = Path(p); h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return {"path": p.relative_to(ROOT).as_posix(), "sha256": h.hexdigest(), "bytes": p.stat().st_size}


def test_results(path):
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^test_\S+ \((tests\.[^)]+)\)", text, re.M)); results = {}
    for i, match in enumerate(matches):
        block = text[match.end(): matches[i+1].start() if i+1 < len(matches) else len(text)]
        status = re.search(r"\.\.\. (ok|FAIL|ERROR|skipped[^\n]*)", block)
        assert status, match[1]
        results[match[1]] = status[1]
    return results


def main():
    first = test_results(OUT / "regression.log"); final = {**first, **test_results(OUT / "v1_final.log")}
    counts = dict(Counter(v.split()[0] for v in final.values()))
    assert len(final) == 649 and counts == {"ok": 641, "skipped": 8}, counts
    sources = []
    for name in ("k1", "k2", "k3"):
        replay = read(OUT / name / "replay.json"); audit = read(OUT / (name + "_audit.json"))
        full = read(OUT / name / "behavior_full.json"); cm = read(replay["card_reference"]["manifest_path"])
        sources.append({"name": name, "replay": replay, "audit": {k: v for k, v in audit.items() if k not in
            {"claim_reports", "public_sentences", "all_group_metrics", "predeclared_examples"}},
            "predeclared_sample_count": len(audit["predeclared_examples"]), "cost": cm["cost"],
            "public_claim_count": len(full["audit"]["included_claim_ids"]),
            "public_omitted_count": len(full["audit"]["omitted_claims"]),
            "RL_marker_episode_mean": audit["all_group_metrics"]["robot_0.support.RL_foot.marker"]["episode_equal"]["mean"],
            "files": [ref(p) for p in sorted((OUT / name).rglob("*")) if p.is_file()]})
    code = [ROOT / p for p in (
        "src/true_rl_agent/tasks/stair_climbing/trajectory_adapter.py",
        "src/true_rl_agent/tasks/stair_climbing/trajectory_pipeline.py",
        "src/true_rl_agent/tasks/stair_climbing/trajectory_artifacts.py",
        "src/true_rl_agent/evaluation/trajectory_translation/contracts.py",
        "src/true_rl_agent/evaluation/trajectory_translation/measurement.py",
        "src/true_rl_agent/evaluation/trajectory_translation/report.py",
        "scripts/verify_policy_behavior_card.py", "tests/unit/test_stair_trajectory_adapter.py",
        "tests/unit/test_stair_trajectory_v1.py", "tests/unit/test_trajectory_card_verification.py")]
    result = {"schema_version": "trajectory.p2a.development_verification.v1", "base_commit": "aaf67aa23bd7e4aa8aae7a9ec9539f3aa45c6c3e",
        "scope": "Go2 Stair three-source descriptive transfer; pending parent independent acceptance; full E2 unseen gate not met",
        "sources": sources, "code": [ref(p) for p in code],
        "phase_documents": [ref(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "verification.json"],
        "supporting_files": [ref(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "final.diff"],
        "tests": {"first_full_run": read(OUT / "regression_result.json"),
            "first_counts": dict(Counter(v.split()[0] for v in first.values())),
            "first_failures": [k for k, v in first.items() if v in ("FAIL", "ERROR")],
            "repair": "independent verifier missing history ratio expected float 0.0; production unchanged; initial process had already imported old verifier",
            "final_focused_command": [".venv/Scripts/python.exe", "-B", "-m", "unittest", "tests.unit.test_stair_trajectory_v1", "tests.unit.test_trajectory_card_verification", "-v"],
            "final_focused_count": 13, "unique_tests": len(final), "final_counts": counts, "final_test_ids": final},
        "frozen_gate": {k: {a: b for a, b in v.items() if a != "facts"} for k, v in read(OUT / "frozen_verification.json")["reports"].items()},
        "v2_compat": [read(OUT / (n + "_compat.json")) for n in ("k1", "k3")],
        "frozen_card_compat": read(OUT / "frozen_compat.json"), "real_guards": read(OUT / "real_guards.json"),
        "review": {"identity": "read-only agent independently verified 87 NPZ SHA and identities, K2 endpoint, P1d 634+8/82 errata; PASS",
            "semantics": "read-only Astra ultra reviewed final diff, every public sentence and template representatives, all source audit results and v2 compatibility; PASS",
            "scope": "agent review did not independently execute tests; developer NumPy oracle has no production translator/renderer imports"},
        "e2": {"three_actual_weights": True, "three_independent_random_experiments": False, "unseen_policy_qualified": False,
            "strict_parent_difference": False, "causal_improvement": False, "push_p3_p4": False,
            "external_model_calls": 0, "gpu_runs_by_p2a": 0, "training_runs": 0}}
    (HERE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"unique_tests": len(final), "counts": counts, "sources": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
