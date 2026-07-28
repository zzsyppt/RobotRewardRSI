from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import (
    StaticCandidateVerifier,
    seal_and_verify_candidate,
)
from true_rl_agent.composition import (
    TaskCompositionContext,
    TaskCompositionRegistry,
    resolve_task_composition,
)
from true_rl_agent.composition.cli import (
    add_task_runtime_arguments,
    add_task_selection_argument,
    select_task_composition_provider,
)
from true_rl_agent.composition.defaults import build_application_composition_registry


def _runner_execution_payload(metadata: dict) -> dict:
    payload = {
        "cli_command": metadata["cli_command"],
        "cli_argv": list(metadata["cli_argv"]),
    }
    if "summary_path" in metadata:
        payload["summary_path"] = metadata["summary_path"]
    if "sealed_checkpoint_uri" in metadata:
        payload["sealed_checkpoint_uri"] = metadata["sealed_checkpoint_uri"]
    if "checkpoint_output_path" in metadata:
        payload["checkpoint_output_path"] = metadata["checkpoint_output_path"]
    if "work_dir" in metadata:
        payload["work_dir"] = metadata["work_dir"]
    return payload


def main(
    argv: list[str] | None = None,
    *,
    registry: TaskCompositionRegistry | None = None,
) -> int:
    registry = registry or build_application_composition_registry()
    provider = select_task_composition_provider(argv, registry)
    parser = argparse.ArgumentParser(description="Materialize an IsaacLab/RSL-RL job bundle.")
    add_task_selection_argument(
        parser,
        registry=registry,
        provider=provider,
    )
    add_task_runtime_arguments(parser, provider.cli_defaults())
    parser.add_argument(
        "--job-kind",
        choices=("candidate", "baseline-eval"),
        default="candidate",
    )
    parser.add_argument("--round-id", default="manual-round-001")
    parser.add_argument("--candidate-id", default="manual-candidate")
    parser.add_argument("--component", action="append", default=None)
    parser.add_argument("--candidate-path", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    args = parser.parse_args(argv)

    composition = resolve_task_composition(
        provider=provider,
        values=vars(args),
        context=TaskCompositionContext(
            project_root=PROJECT_ROOT,
            output_dir=args.output_dir,
        ),
    )
    if args.job_kind == "baseline-eval":
        if composition.baseline_manifest is None:
            parser.error(
                f"task does not provide a baseline manifest: {composition.task_id}"
            )
        eval_spec = composition.backend_adapter.materialize_baseline_eval(
            session_id=args.round_id,
            task_spec=composition.task_plugin.task_spec,
            baseline_manifest=composition.baseline_manifest,
        )
        payload = {
            "schema_version": "true_rl_agent.scripts.materialized_baseline_eval.v1",
            "task_id": composition.task_id,
            "eval": eval_spec.to_public_dict(),
            "runner_execution": {
                "eval": _runner_execution_payload(dict(eval_spec.metadata)),
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    components = args.component or ["reward_callable"]
    if args.candidate_path is None:
        parser.error("--candidate-path is required for candidate jobs")
    candidate = CandidatePackageRef(
        candidate_id=args.candidate_id,
        path=args.candidate_path,
        change_set=CandidateChangeSet(frozenset(components)),
    )
    verification = seal_and_verify_candidate(
        candidate,
        artifact_root=args.output_dir / "candidate_artifacts",
        verifier=StaticCandidateVerifier(),
    )
    try:
        execution_candidate = verification.require_verified()
    except ValueError as exc:
        parser.error(str(exc))
    bundle = composition.backend_adapter.materialize_jobs(
        round_id=args.round_id,
        task_spec=composition.task_plugin.task_spec,
        candidate=execution_candidate,
        policy_initialization=composition.policy_initialization,
    )
    payload = {
        "schema_version": "true_rl_agent.scripts.materialized_job.v1",
        "task_id": composition.task_id,
        "candidate_changed_components": sorted(
            execution_candidate.change_set.changed_components
        ),
        "train": bundle.train_spec.to_public_dict(),
        "eval": bundle.eval_spec.to_public_dict(),
        "runner_execution": {
            "train": _runner_execution_payload(dict(bundle.train_spec.metadata)),
            "eval": _runner_execution_payload(dict(bundle.eval_spec.metadata)),
        },
        "task_invariants": dict(bundle.task_invariants.values) if bundle.task_invariants else {},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
