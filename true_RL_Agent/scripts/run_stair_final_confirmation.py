#!/usr/bin/env python3
"""Run the precommitted Stair final confirmation once, after 15 closed rounds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.eval_run_audit import (  # noqa: E402
    build_eval_run_audit,
)
from true_rl_agent.agent_core.ledger import (  # noqa: E402
    PROVIDER_AUTHORING_MODE,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.live_invocation import (  # noqa: E402
    find_verified_completed_live_invocation,
)
from true_rl_agent.agent_core.session_epoch import (  # noqa: E402
    acquire_live_session_lease,
)
from true_rl_agent.agent_core.session_genesis import (  # noqa: E402
    ReferenceProgramGenesisAnchor,
    prepare_session_genesis,
)
from true_rl_agent.backends.isaac_lab_execution_environment import (  # noqa: E402
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.composition.base import (  # noqa: E402
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.composition.cli import add_task_runtime_arguments  # noqa: E402
from true_rl_agent.evaluation.acceptance_replay import (  # noqa: E402
    replay_acceptance_ledger,
)
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.runner.checkpoint import (  # noqa: E402
    build_checkpoint_safety_report,
    file_sha256,
)
from true_rl_agent.runner.isaac_app_lifecycle import (  # noqa: E402
    close_isaac_app_on_success,
)
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionConfig,
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (  # noqa: E402
    STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
    STAIR_FINAL_CONFIRMATION_SPLIT,
    build_stair_final_confirmation_result,
    create_stair_final_confirmation_reservation,
    load_stair_final_confirmation_plan,
    write_stair_final_confirmation_result,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (  # noqa: E402
    StairEvaluationExecutionResult,
    execute_stair_evaluation,
    preflight_stair_execution_source_manifest,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (  # noqa: E402
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.session_start import (  # noqa: E402
    STAIR_REFERENCE_START_MODE,
)


@dataclass(frozen=True)
class _CheckpointPair:
    genesis_path: Path
    genesis_hash: str
    final_path: Path
    final_hash: str
    final_anchor: Mapping[str, Any]


def _build_parser(app_launcher_class: Any) -> argparse.ArgumentParser:
    defaults = StairClimbingCompositionConfig.defaults(
        project_root=PROJECT_ROOT
    ).to_task_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--final-confirmation-plan", type=Path, required=True)
    add_task_runtime_arguments(parser, defaults)
    parser.add_argument(
        "--start-mode",
        choices=(STAIR_REFERENCE_START_MODE,),
        default=STAIR_REFERENCE_START_MODE,
    )
    app_launcher_class.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    app_launcher_class: Any | None = None,
) -> int:
    if app_launcher_class is None:
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

        app_launcher_class = AppLauncher
    args = _build_parser(app_launcher_class).parse_args(argv)
    output_dir = args.output_dir.resolve(strict=True)
    values = vars(args).copy()
    values.update(
        {
            "mode": "live",
            "rounds": STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
            "session_id": args.session_id,
        }
    )
    composition = resolve_task_composition(
        provider=StairClimbingCompositionProvider(),
        values=values,
        context=TaskCompositionContext(
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
        ),
    )
    config = composition.task_plugin.config
    plan = load_stair_final_confirmation_plan(
        args.final_confirmation_plan,
        expected_session_id=args.session_id,
        expected_config=config,
    )
    genesis = prepare_session_genesis(
        output_dir=output_dir,
        session_id=args.session_id,
        task_plugin=composition.task_plugin,
        backend_adapter=composition.backend_adapter,
        runner=composition.runner,
        policy_initialization=composition.policy_initialization,
        genesis_identity_snapshot=composition.genesis_identity_snapshot(),
        reference_program_artifact=composition.reference_program_artifact,
        require_existing=True,
        bind_backend_execution_environment=False,
    )
    ledger_epoch_hash = primary_ledger_epoch_hash(genesis.session_genesis_hash)
    with acquire_live_session_lease(
        output_dir=output_dir,
        session_id=args.session_id,
        session_genesis_hash=genesis.session_genesis_hash,
        ledger_epoch_hash=ledger_epoch_hash,
    ) as lease:
        records = load_ledger(output_dir / "ledger.jsonl")
        replay = replay_acceptance_ledger(
            records,
            initial_current_best=genesis.initial_current_best,
            expected_session_genesis_hash=genesis.session_genesis_hash,
            acceptance_protocol=config.acceptance_protocol,
            expected_initial_policy_state_hash=(
                genesis.expected_initial_policy_state_hash
            ),
            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            expected_ledger_epoch_hash=ledger_epoch_hash,
        )
        replay.raise_for_continuation()
        eval_run_audit = build_eval_run_audit(
            records,
            acceptance_protocol=config.acceptance_protocol,
            initial_current_best=genesis.initial_current_best,
            expected_session_genesis_hash=genesis.session_genesis_hash,
            expected_initial_policy_state_hash=(
                genesis.expected_initial_policy_state_hash
            ),
            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            expected_ledger_epoch_hash=ledger_epoch_hash,
        )
        _assert_completed_live_run(
            output_dir=output_dir,
            session_id=args.session_id,
            session_genesis_hash=genesis.session_genesis_hash,
            expected_ledger_epoch_hash=ledger_epoch_hash,
            records=records,
            expected_eval_run_audit=eval_run_audit,
            expected_final_best_eval_stats=replay.current_best.to_public_dict(),
            expected_final_best_score=replay.current_best.score_for_comparison(),
        )
        checkpoints = _resolve_checkpoint_pair(
            genesis=genesis,
            records=records,
            final_current_best=replay.current_best,
        )
        _require_safe_checkpoint(
            checkpoints.genesis_path,
            checkpoints.genesis_hash,
            "Genesis",
        )
        _require_safe_checkpoint(
            checkpoints.final_path,
            checkpoints.final_hash,
            "final current-best",
        )
        source_preflight = preflight_stair_execution_source_manifest(config)
        environment = genesis.backend_execution_environment
        if not isinstance(environment, Mapping):
            raise ValueError(
                "Stair final confirmation requires a bound backend environment"
            )
        backend_environment_verification = (
            verify_current_isaac_lab_execution_environment(
                environment.get("manifest_hash")
            )
        )
        _require_bound_isaac_python(
            current_python=Path(sys.executable),
            expected_python=Path(composition.backend_adapter.config.isaac_lab_python),
        )
        ledger_checksum = file_sha256(output_dir / "ledger.jsonl")
        if ledger_checksum is None:
            raise ValueError("Stair final confirmation ledger checksum failed")
        args.device = composition.backend_adapter.config.execution_device
        app_launcher = app_launcher_class(args)
        simulation_app = app_launcher.app
        try:
            reservation, _ = create_stair_final_confirmation_reservation(
                output_dir=output_dir,
                plan=plan,
                session_genesis_hash=genesis.session_genesis_hash,
                ledger_checksum=ledger_checksum,
                genesis_checkpoint_hash=checkpoints.genesis_hash,
                final_checkpoint_hash=checkpoints.final_hash,
                source_manifest_hash=source_preflight.source_manifest_hash,
                backend_execution_environment_manifest_hash=(
                    backend_environment_verification["actual_manifest_hash"]
                ),
                execution_device=args.device,
                final_anchor=checkpoints.final_anchor,
            )
            lease.assert_owned()
            genesis_execution = _run_checkpoint_confirmation(
                config=config,
                source_preflight=source_preflight,
                backend_environment_verification=backend_environment_verification,
                checkpoint_path=checkpoints.genesis_path,
                device=args.device,
            )
            final_execution = _run_checkpoint_confirmation(
                config=config,
                source_preflight=source_preflight,
                backend_environment_verification=backend_environment_verification,
                checkpoint_path=checkpoints.final_path,
                device=args.device,
            )
            result = build_stair_final_confirmation_result(
                plan=plan,
                reservation=reservation,
                config=config,
                genesis_execution=genesis_execution,
                final_execution=final_execution,
            )
            result_path = write_stair_final_confirmation_result(
                output_dir=output_dir,
                result=result,
                plan=plan,
                reservation=reservation,
                config=config,
            )
            lease.assert_owned()
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "result_path": str(result_path),
                        "result_hash": result.result_hash,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception:
            # The durable reservation intentionally remains after any failure.
            raise
        return close_isaac_app_on_success(simulation_app, exit_code=0)


def _assert_completed_live_run(
    *,
    output_dir: Path,
    session_id: str,
    session_genesis_hash: str,
    expected_ledger_epoch_hash: str,
    records: list[dict[str, Any]],
    expected_eval_run_audit: Mapping[str, Any],
    expected_final_best_eval_stats: Mapping[str, Any],
    expected_final_best_score: float | None,
) -> None:
    if len(records) != STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS:
        raise ValueError("Stair final confirmation requires exactly 15 ledger rounds")
    try:
        receipt = find_verified_completed_live_invocation(
            output_dir=output_dir,
            session_id=session_id,
            session_genesis_hash=session_genesis_hash,
            ledger_epoch_hash=expected_ledger_epoch_hash,
            rounds_requested=STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
        )
    except ValueError as exc:
        raise ValueError(
            "Stair final confirmation requires a verified completed live invocation"
        ) from exc
    invocation_entries = tuple(receipt.invocation_dir.parent.iterdir())
    if invocation_entries != (receipt.invocation_dir,):
        raise ValueError(
            "Stair final confirmation requires exactly one live invocation"
        )
    summary = receipt.summary
    if summary.get("resumed_from_round_count") != 0:
        raise ValueError(
            "Stair final confirmation requires one live invocation from an empty ledger"
        )
    if canonical_json_bytes(summary.get("eval_run_audit")) != canonical_json_bytes(
        expected_eval_run_audit
    ):
        raise ValueError(
            "Stair final confirmation live summary eval audit differs from the ledger"
        )
    if canonical_json_bytes(
        summary.get("final_best_eval_stats")
    ) != canonical_json_bytes(expected_final_best_eval_stats):
        raise ValueError(
            "Stair final confirmation live summary current-best differs from the ledger"
        )
    if summary.get("final_best_score") != expected_final_best_score:
        raise ValueError(
            "Stair final confirmation live summary best score differs from the ledger"
        )
    audit_counts = expected_eval_run_audit.get("counts")
    if not isinstance(audit_counts, Mapping):
        raise ValueError("Stair final confirmation eval audit counts are missing")
    trained = audit_counts.get("trained_candidate_count")
    if (
        isinstance(trained, bool)
        or not isinstance(trained, int)
        or trained <= 0
        or trained > STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS
        or audit_counts.get("full_eval_completed_count") != trained
        or audit_counts.get("sealed_validation_completed_count") != trained
    ):
        raise ValueError(
            "Stair final confirmation requires complete train/eval accounting"
        )


def _resolve_checkpoint_pair(
    *,
    genesis: Any,
    records: list[dict[str, Any]],
    final_current_best: Any,
) -> _CheckpointPair:
    anchor = genesis.artifact.anchor
    if not isinstance(anchor, ReferenceProgramGenesisAnchor):
        raise ValueError("Stair final confirmation requires reference Genesis")
    genesis_path = _local_checkpoint_path(anchor.checkpoint_artifact.uri, "Genesis")
    final_hash = final_current_best.checkpoint_hash
    if not isinstance(final_hash, str):
        raise ValueError("final current-best checkpoint hash is missing")
    if final_current_best.anchor_kind == "reference":
        final_path = genesis_path
    elif final_current_best.anchor_kind == "candidate":
        matches = [
            record
            for record in records
            if record.get("round_id") == final_current_best.accepted_round_id
        ]
        if len(matches) != 1:
            raise ValueError("final current-best accepted round is not unique")
        train_result = _mapping(matches[0].get("train_result"), "train_result")
        checkpoint = _mapping(train_result.get("checkpoint"), "checkpoint")
        if checkpoint.get("checksum") != final_hash:
            raise ValueError("final current-best checkpoint hash differs from ledger")
        final_path = _candidate_checkpoint_path(
            record=matches[0],
            expected_hash=final_hash,
        )
    else:
        raise ValueError("final current-best has no evaluable checkpoint")
    return _CheckpointPair(
        genesis_path=genesis_path,
        genesis_hash=anchor.checkpoint_hash,
        final_path=final_path,
        final_hash=final_hash,
        final_anchor=final_current_best.comparison_identity(),
    )


def _candidate_checkpoint_path(
    *,
    record: Mapping[str, Any],
    expected_hash: str,
) -> Path:
    artifacts = record.get("private_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("final current-best private artifacts are missing")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("kind") == "runner_checkpoint"
        and artifact.get("reason") == "runner_local_checkpoint_path"
        and artifact.get("checksum") == expected_hash
    ]
    if len(matches) != 1:
        raise ValueError(
            "final current-best requires exactly one bound local checkpoint artifact"
        )
    return _local_checkpoint_path(matches[0].get("uri"), "final current-best")


def _run_checkpoint_confirmation(
    *,
    config: Any,
    source_preflight: Any,
    backend_environment_verification: Mapping[str, str],
    checkpoint_path: Path,
    device: str,
) -> StairEvaluationExecutionResult:
    return execute_stair_evaluation(
        config=config,
        source_manifest_preflight=source_preflight,
        backend_execution_environment_verification=backend_environment_verification,
        checkpoint_path=checkpoint_path,
        device=device,
        seed_namespace=STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
        splits=(STAIR_FINAL_CONFIRMATION_SPLIT,),
    )


def _require_safe_checkpoint(path: Path, expected_hash: str, role: str) -> None:
    report = build_checkpoint_safety_report(
        path,
        expected_checkpoint_hash=expected_hash,
    )
    if report.status != "pass":
        raise ValueError(f"{role} checkpoint is not safe to load: {report.reason}")


def _require_bound_isaac_python(
    *,
    current_python: Path,
    expected_python: Path,
) -> None:
    if current_python.resolve(strict=True) != expected_python.resolve(strict=True):
        raise ValueError(
            "Stair final confirmation must run under the Genesis-bound Isaac Python"
        )


def _local_checkpoint_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"{role} checkpoint must be a local path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{role} checkpoint path must be absolute")
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{role} checkpoint must be a regular file")
    return resolved


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
