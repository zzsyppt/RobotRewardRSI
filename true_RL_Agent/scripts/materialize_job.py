from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.session_genesis import prepare_session_genesis
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import seal_and_verify_candidate
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
from true_rl_agent.runner.job_argv import (
    validate_eval_job_argv,
    validate_train_job_argv,
)


def _structured_cli_argv(metadata: Mapping[str, object]) -> list[str]:
    value = metadata.get("cli_argv")
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError("materialized job metadata requires structured cli_argv")
    return list(value)


def _runner_execution_payload(metadata: Mapping[str, object]) -> dict:
    payload = {
        "cli_command": metadata["cli_command"],
        "cli_argv": _structured_cli_argv(metadata),
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
    provider.add_cli_arguments(parser)
    parser.add_argument(
        "--job-kind",
        choices=("candidate", "baseline-eval"),
        default="candidate",
    )
    parser.add_argument("--round-id", default="manual-round-001")
    parser.add_argument("--session-id", default=None)
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

    expected_initial_policy_state_hash = None
    backend_execution_environment_manifest_hash = None
    requires_session_genesis = (
        composition.genesis_anchor_kind == "reference_program"
        or composition.bootstrap_initial_policy_state_hash is not None
    )
    if requires_session_genesis:
        if args.session_id is None:
            parser.error(
                "--session-id is required for initialization-anchored candidates"
            )
        try:
            genesis = prepare_session_genesis(
                output_dir=args.output_dir,
                session_id=args.session_id,
                task_plugin=composition.task_plugin,
                backend_adapter=composition.backend_adapter,
                runner=composition.runner,
                policy_initialization=composition.policy_initialization,
                genesis_identity_snapshot=composition.genesis_identity_snapshot(),
                reference_program_artifact=composition.reference_program_artifact,
                bootstrap_initial_best_score=(
                    composition.bootstrap_initial_best_score
                ),
                bootstrap_initial_policy_state_hash=(
                    composition.bootstrap_initial_policy_state_hash
                ),
                require_existing=True,
                bind_backend_execution_environment=False,
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        expected_initial_policy_state_hash = (
            genesis.expected_initial_policy_state_hash
        )
        backend_execution_environment_manifest_hash = (
            genesis.backend_execution_environment_manifest_hash
        )

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
        verifier=composition.candidate_admission_policy,
    )
    try:
        execution_candidate = verification.require_verified()
    except ValueError as exc:
        parser.error(str(exc))
    materialization_kwargs: dict[str, Any] = {}
    if backend_execution_environment_manifest_hash is not None:
        materialization_kwargs[
            "backend_execution_environment_manifest_hash"
        ] = backend_execution_environment_manifest_hash
    bundle = composition.backend_adapter.materialize_jobs(
        round_id=args.round_id,
        task_spec=composition.task_plugin.task_spec,
        candidate=execution_candidate,
        policy_initialization=composition.policy_initialization,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        **materialization_kwargs,
    )
    train_metadata = dict(bundle.train_spec.metadata)
    eval_metadata = dict(bundle.eval_spec.metadata)
    validate_train_job_argv(
        bundle.train_spec,
        _structured_cli_argv(train_metadata),
    )
    validate_eval_job_argv(
        bundle.eval_spec,
        _structured_cli_argv(eval_metadata),
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
            "train": _runner_execution_payload(train_metadata),
            "eval": _runner_execution_payload(eval_metadata),
        },
        "task_invariants": dict(bundle.task_invariants.values) if bundle.task_invariants else {},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
