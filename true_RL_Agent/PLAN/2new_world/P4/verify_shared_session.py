"""Persistent fixed-runner session probe; never launches physics or a provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.integration.test_generation_session import (
    SharedStairRunner, SharedPushRunner, build_fixture_foundation,
    StairClimbingCompositionProvider, PushRecoveryCompositionProvider,
    _FormalFixtureProvider, generation_runtime,
    PushCandidateProvider,
    use_test_isaac_lab_execution_environment,
)
from true_rl_agent.agent_core.generation_application import run_generation_session
from true_rl_agent.runner.process_handle import isolate_orchestrator_process_group


def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stair", "push"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--crash-after-first-dev", action="store_true")
    parser.add_argument("--pause-after-generation", type=int, choices=(3, 6, 9, 12))
    args = parser.parse_args()
    isolate_orchestrator_process_group()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (root / "provider.env").write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
    provider, cell, runner_type = (
        (StairClimbingCompositionProvider(), "stair.g1_37dof.unified_v3", SharedStairRunner)
        if args.task == "stair" else
        (PushRecoveryCompositionProvider(), "push.g1_29dof.reference_v1", SharedPushRunner))
    with use_test_isaac_lab_execution_environment():
        foundation = build_fixture_foundation(root, provider=provider, cell=cell, resume=args.resume,
                                             pause=args.pause_after_generation)
        runner = runner_type(foundation=foundation)
        if args.candidate:
            runner.scenario = "accepted"
        if args.crash_after_first_dev:
            if not args.candidate or args.resume:
                raise ValueError("crash probe requires a fresh candidate session")
            barrier = threading.Barrier(2, timeout=30)
            original = runner.run_eval_phase

            def crash_after_attempt_receipt(spec):
                result = original(spec)
                if spec.subject.kind == "candidate" and spec.phase.value == "public_dev":
                    barrier.wait()
                    if "lane-a" in spec.job_id:
                        (root / "hard_crash.json").write_text(json.dumps({
                            "job_id": spec.job_id, "after_attempt_result": True}), encoding="utf-8")
                        os._exit(91)
                return result

            runner.run_eval_phase = crash_after_attempt_receipt
        foundation = replace(foundation, runner=runner)
        candidate_provider = PushCandidateProvider if args.task == "push" and args.candidate else _FormalFixtureProvider
        source = foundation.composition.reference_program_artifact.source_bytes.decode("utf-8")
        started = time.perf_counter_ns()
        with patch.object(candidate_provider, "reference_only", not args.candidate), patch.object(
                candidate_provider, "reference_source", source), patch.object(
                generation_runtime, "OpenAICompatibleCandidateProvider", candidate_provider):
            report = run_generation_session(foundation)
        run_ns = time.perf_counter_ns() - started
        if report.get("status") == "paused":
            (root / f"pause-{args.pause_after_generation}.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report), flush=True)
            return
        assert report["checks"]["status"] == "PASS"
        assert report["closed_generation_count"] == 15
        assert report["current_best"]["anchor_kind"] == ("candidate" if args.candidate else "reference")
        before = snapshot(foundation.config.output_dir)
        audit = replace(foundation, config=replace(foundation.config,
            resume_existing=True, terminal_audit_only=True))
        started = time.perf_counter_ns()
        assert run_generation_session(audit) == report
        audit_ns = time.perf_counter_ns() - started
        assert snapshot(foundation.config.output_dir) == before
    result = {"status": "PASS", "task": args.task, "cell": cell,
        "anchor_kind": "candidate" if args.candidate else "reference",
        "resumed": args.resume,
        "run_wall_time_ns": run_ns, "audit_wall_time_ns": audit_ns,
        "closed_generations": 15, "unchanged_files_on_audit": len(before),
        "fixture_scope": "fixed runner and provider; no scientific performance claim"}
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
