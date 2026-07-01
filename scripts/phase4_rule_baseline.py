#!/usr/bin/env python3
"""Generate a Phase 4 hand-coded rule-baseline candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.phase3_verifier.verifier import (
    _load_structured_file,  # noqa: PLC2701 - CLI needs the same YAML fallback as verifier.
    load_verifier_config,
    verify_static_documents,
)
from humanoid_agent.phase4_rule_baseline.rules import (
    build_rule_candidate,
    dump_yaml_like,
    render_unified_diff,
    stable_json,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--phase1-config", default="configs/phase1_push_recovery.yaml")
    parser.add_argument("--phase2-config", default="configs/phase2_baseline.yaml")
    parser.add_argument("--rule-config", default="configs/phase4_rule_baseline.yaml")
    parser.add_argument("--verifier-config", default="configs/phase3_verifier.yaml")
    parser.add_argument("--output-dir", default="agent_outputs/phase4_rule_baseline/latest")
    parser.add_argument("--skip-verify", action="store_true")
    return parser


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = REPO_ROOT
    summary = _load_json(Path(args.summary_file))
    phase1_path = repo_root / args.phase1_config
    phase2_path = repo_root / args.phase2_config
    rule_path = repo_root / args.rule_config
    verifier_path = repo_root / args.verifier_config

    phase1 = _load_structured_file(phase1_path)
    phase2 = _load_structured_file(phase2_path)
    rule_cfg = _load_structured_file(rule_path)
    candidate = build_rule_candidate(
        summary=summary,
        phase1_config=phase1,
        phase2_config=phase2,
        rule_config=rule_cfg,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_phase1_text = dump_yaml_like(candidate["phase1_config"])
    candidate_phase2_text = dump_yaml_like(candidate["phase2_config"])
    phase1_out = output_dir / "candidate_phase1_push_recovery.yaml"
    phase2_out = output_dir / "candidate_phase2_baseline.yaml"
    phase1_out.write_text(candidate_phase1_text, encoding="utf-8")
    phase2_out.write_text(candidate_phase2_text, encoding="utf-8")

    analysis_out = output_dir / "analysis.json"
    analysis_out.write_text(stable_json(candidate["analysis"]), encoding="utf-8")

    diff_text = (
        render_unified_diff(
            phase1_path.read_text(encoding="utf-8"),
            candidate_phase1_text,
            fromfile=args.phase1_config,
            tofile=_display_path(phase1_out, repo_root),
        )
        + render_unified_diff(
            phase2_path.read_text(encoding="utf-8"),
            candidate_phase2_text,
            fromfile=args.phase2_config,
            tofile=_display_path(phase2_out, repo_root),
        )
    )
    diff_out = output_dir / "candidate.diff"
    diff_out.write_text(diff_text, encoding="utf-8")

    verifier_summary = None
    verifier_status = "skipped"
    if not args.skip_verify:
        settings = load_verifier_config(verifier_path)
        verifier_summary = verify_static_documents(
            base_docs={
                args.phase1_config: phase1,
                args.phase2_config: phase2,
                args.verifier_config: settings,
            },
            current_docs={
                args.phase1_config: candidate["phase1_config"],
                args.phase2_config: candidate["phase2_config"],
                args.verifier_config: settings,
            },
            changed_files=[args.phase1_config, args.phase2_config],
            settings=settings,
            repo_root=repo_root,
            check_imports=False,
        )
        verifier_status = verifier_summary["status"]
        (output_dir / "verifier_summary.json").write_text(stable_json(verifier_summary), encoding="utf-8")

    run_summary = {
        "schema_version": "humanoid_agent.phase4.rule_baseline.run.v1",
        "status": "pass" if verifier_status in {"pass", "skipped"} else "fail",
        "output_dir": str(output_dir),
        "analysis_file": str(analysis_out),
        "candidate_phase1_config": str(phase1_out),
        "candidate_phase2_config": str(phase2_out),
        "candidate_diff": str(diff_out),
        "verifier_status": verifier_status,
        "verifier_summary_file": str(output_dir / "verifier_summary.json") if verifier_summary else None,
    }
    (output_dir / "run_summary.json").write_text(stable_json(run_summary), encoding="utf-8")
    print(stable_json(run_summary))
    return 0 if run_summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
