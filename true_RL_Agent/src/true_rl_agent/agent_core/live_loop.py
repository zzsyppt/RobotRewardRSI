"""Live multi-round loop with persisted prompt payloads."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.agent_core.eval_run_audit import build_eval_run_audit
from true_rl_agent.agent_core.ledger import (
    LEDGER_RECORD_SCHEMA_VERSION,
    PROVIDER_AUTHORING_MODE,
    append_ledger_record_once,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.live_invocation import (
    LIVE_LOOP_SUMMARY_FIELDS,
    LIVE_LOOP_SUMMARY_REQUIRED_CHECKS,
    LIVE_LOOP_SUMMARY_SCHEMA_VERSION,
    LiveInvocationJournal,
    write_live_summary,
)
from true_rl_agent.agent_core.provider import CandidateProvider
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.agent_core.session_epoch import (
    VerifiedLedgerEpoch,
    acquire_live_session_lease,
    prepare_ledger_epoch,
)
from true_rl_agent.agent_core.session_genesis import (
    VerifiedSessionGenesis,
    verify_session_genesis_for_runtime,
)
from true_rl_agent.agent_core.source_request_feedback import (
    project_source_request_round_feedback,
)
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.capability_readiness import (
    READINESS_REQUIRED_CAPABILITY_IDS,
    AuthoringCapabilityReadiness,
)
from true_rl_agent.candidates.verifier import CandidateVerifier
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.round_stage_contract import (
    EVALUATED_STAGE,
    POST_TRAIN_FAILURE_STAGE,
    PRE_TRAIN_REJECTION_STAGE,
    PROVIDER_FAILURE_STAGE,
    STATIC_AUTHORING_FAILURE_STAGE,
    TRAIN_FAILURE_STAGE,
    VERIFIER_FAILURE_STAGE,
    validate_round_stage,
)
from true_rl_agent.memory import (
    ExperimentRecord,
    PromptPayloadRef,
    PublicContextBuilder,
    preflight_task_prompt_capacity,
)
from true_rl_agent.memory.task_adapter import TaskMemoryAdapter
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.prompt_artifacts import (
    AuthoringEvidenceRef,
    authoring_outcome_from_evidence,
    load_ledger_authoring_evidence,
    validate_ledger_prompt_artifacts,
    verify_authoring_evidence,
)
from true_rl_agent.runner.base import Runner
from true_rl_agent.tasks.base import TaskPlugin
from true_rl_agent.tasks.prompt_adapter import TaskPromptAdapter

DEFAULT_MAX_PROVIDER_ATTEMPTS = 5


def run_live_loop(
    *,
    output_dir: Path,
    rounds: int,
    session_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    runner: Runner,
    candidate_admission_policy: CandidateVerifier,
    provider: CandidateProvider,
    policy_initialization: PolicyInitialization,
    session_genesis: VerifiedSessionGenesis,
    session_genesis_dir: Path | None = None,
    ledger_epoch_id: str | None = None,
    authoring_capability_readiness: AuthoringCapabilityReadiness | None = None,
    policy_accept_delta: float | None = None,
    task_memory_adapter: TaskMemoryAdapter | None = None,
    max_provider_attempts: int = DEFAULT_MAX_PROVIDER_ATTEMPTS,
    completion_validator: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if max_provider_attempts <= 0:
        raise ValueError("max_provider_attempts must be positive")
    if completion_validator is not None and not callable(completion_validator):
        raise TypeError("completion_validator must be callable")
    genesis_dir = (
        session_genesis_dir.resolve()
        if session_genesis_dir is not None
        else output_dir.resolve()
    )
    acceptance_protocol = task_plugin.task_spec.acceptance_protocol
    if acceptance_protocol is None:
        raise ValueError("task does not define a fixed acceptance protocol")
    verify_session_genesis_for_runtime(
        session_genesis,
        output_dir=genesis_dir,
        session_id=session_id,
        task_plugin=task_plugin,
        backend_adapter=backend_adapter,
        policy_initialization=policy_initialization,
    )
    ledger_epoch = prepare_ledger_epoch(
        output_dir=output_dir,
        genesis_dir=genesis_dir,
        epoch_id=ledger_epoch_id,
        session_id=session_id,
        genesis=session_genesis,
    )
    ledger_epoch_hash = (
        ledger_epoch.manifest_hash
        if ledger_epoch is not None
        else primary_ledger_epoch_hash(
            session_genesis.session_genesis_hash
        )
    )
    replay_epoch_hash = (
        ledger_epoch.manifest_hash
        if ledger_epoch is not None
        else None
    )
    with acquire_live_session_lease(
        output_dir=output_dir,
        session_id=session_id,
        session_genesis_hash=session_genesis.session_genesis_hash,
        ledger_epoch_hash=ledger_epoch_hash,
    ) as lease:
        initial_ledger = load_ledger(output_dir.resolve() / "ledger.jsonl")
        if len(initial_ledger) > rounds:
            raise ValueError("live session already contains more rounds than requested")
        _validate_round_sequence(initial_ledger, session_id=session_id)
        _validate_ledger_runtime_identity(
            initial_ledger,
            task_id=task_plugin.task_spec.task_id,
            backend_id=backend_adapter.backend_id,
        )
        replay_acceptance_ledger(
            initial_ledger,
            initial_current_best=session_genesis.initial_current_best,
            expected_session_genesis_hash=session_genesis.session_genesis_hash,
            acceptance_protocol=acceptance_protocol,
            expected_initial_policy_state_hash=(
                session_genesis.expected_initial_policy_state_hash
            ),
            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            expected_ledger_epoch_hash=replay_epoch_hash,
        ).raise_for_continuation()
        task_prompt_adapter = task_plugin.prompt_adapter()
        preflight_task_prompt_capacity(
            task_prompt_adapter=task_prompt_adapter,
            candidate_capability_profile=(
                task_plugin.task_spec.candidate_capability_profile
            ),
        )
        invocation = LiveInvocationJournal.start(
            output_dir=output_dir,
            session_id=session_id,
            session_genesis_hash=session_genesis.session_genesis_hash,
            ledger_epoch_hash=ledger_epoch_hash,
            rounds_requested=rounds,
            closed_ledger_count=len(initial_ledger),
        )
        try:
            result = _run_live_loop_locked(
                output_dir=output_dir,
                rounds=rounds,
                session_id=session_id,
                task_plugin=task_plugin,
                backend_adapter=backend_adapter,
                runner=runner,
                candidate_admission_policy=candidate_admission_policy,
                provider=provider,
                policy_initialization=policy_initialization,
                session_genesis=session_genesis,
                session_genesis_dir=genesis_dir,
                ledger_epoch=ledger_epoch,
                ledger_epoch_hash=ledger_epoch_hash,
                task_prompt_adapter=task_prompt_adapter,
                authoring_capability_readiness=authoring_capability_readiness,
                policy_accept_delta=policy_accept_delta,
                task_memory_adapter=task_memory_adapter,
                max_provider_attempts=max_provider_attempts,
            )
            if completion_validator is not None:
                completion_validator(result)
            lease.assert_owned()
        except KeyboardInterrupt as exc:
            _finish_invocation_after_exception(
                invocation,
                output_dir=output_dir,
                status="interrupted",
                termination_kind="keyboard_interrupt",
                original_exception=exc,
            )
            raise
        except Exception as exc:
            _finish_invocation_after_exception(
                invocation,
                output_dir=output_dir,
                status="failed",
                termination_kind="exception",
                original_exception=exc,
            )
            raise
        closed_ledger_count = len(load_ledger(output_dir.resolve() / "ledger.jsonl"))
        if not _live_summary_checks_pass(result):
            invocation.finish(
                status="failed",
                termination_kind="checks_failed",
                closed_ledger_count=closed_ledger_count,
                live_summary_path=output_dir.resolve() / "live_summary.json",
            )
            return result
        invocation.finish(
            status="completed",
            termination_kind="normal",
            closed_ledger_count=closed_ledger_count,
            live_summary_path=output_dir.resolve() / "live_summary.json",
        )
        return result


def _live_summary_checks_pass(summary: Mapping[str, Any]) -> bool:
    checks = summary.get("checks")
    return (
        isinstance(checks, Mapping)
        and bool(checks)
        and all(type(value) is bool and value for value in checks.values())
    )


def _finish_invocation_after_exception(
    invocation: LiveInvocationJournal,
    *,
    output_dir: Path,
    status: str,
    termination_kind: str,
    original_exception: BaseException,
) -> None:
    try:
        invocation.finish(
            status=status,
            termination_kind=termination_kind,
            closed_ledger_count=len(
                load_ledger(output_dir.resolve() / "ledger.jsonl")
            ),
            exception_type=type(original_exception).__name__,
        )
    except Exception as lifecycle_error:
        add_note = getattr(original_exception, "add_note", None)
        if callable(add_note):
            add_note(
                "live invocation exit evidence could not be finalized: "
                + type(lifecycle_error).__name__
            )


def _run_live_loop_locked(
    *,
    output_dir: Path,
    rounds: int,
    session_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    runner: Runner,
    candidate_admission_policy: CandidateVerifier,
    provider: CandidateProvider,
    policy_initialization: PolicyInitialization,
    session_genesis: VerifiedSessionGenesis,
    session_genesis_dir: Path,
    ledger_epoch: VerifiedLedgerEpoch | None,
    ledger_epoch_hash: str,
    task_prompt_adapter: TaskPromptAdapter,
    authoring_capability_readiness: AuthoringCapabilityReadiness | None = None,
    policy_accept_delta: float | None = None,
    task_memory_adapter: TaskMemoryAdapter | None = None,
    max_provider_attempts: int = DEFAULT_MAX_PROVIDER_ATTEMPTS,
) -> dict[str, Any]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if max_provider_attempts <= 0:
        raise ValueError("max_provider_attempts must be positive")
    acceptance_protocol = task_plugin.task_spec.acceptance_protocol
    if acceptance_protocol is None:
        raise ValueError("task does not define a fixed acceptance protocol")
    if policy_accept_delta is not None and not math.isclose(
        float(policy_accept_delta),
        acceptance_protocol.min_delta,
    ):
        raise ValueError(
            "policy_accept_delta cannot override the task acceptance protocol"
        )
    verify_session_genesis_for_runtime(
        session_genesis,
        output_dir=session_genesis_dir,
        session_id=session_id,
        task_plugin=task_plugin,
        backend_adapter=backend_adapter,
        policy_initialization=policy_initialization,
    )
    if ledger_epoch is not None:
        ledger_epoch.assert_unchanged(
            output_dir=output_dir,
            genesis=session_genesis,
        )
    replay_epoch_hash = (
        ledger_epoch.manifest_hash
        if ledger_epoch is not None
        else None
    )
    initial_current_best = session_genesis.initial_current_best
    expected_initial_policy_state_hash = (
        session_genesis.expected_initial_policy_state_hash
    )
    backend_execution_environment_manifest_hash = (
        session_genesis.backend_execution_environment_manifest_hash
    )
    session_genesis_hash = session_genesis.session_genesis_hash
    evaluation_binding = session_genesis.evaluation_binding
    capability_profile = task_plugin.task_spec.candidate_capability_profile
    smoke_adapter = task_plugin.component_smoke_adapter()
    selected_readiness_required = set(capability_profile.capability_ids) & set(
        READINESS_REQUIRED_CAPABILITY_IDS
    )
    prompt_capability_readiness = None
    if authoring_capability_readiness is None:
        if selected_readiness_required:
            raise ValueError(
                "task selects a capability without authoring readiness"
            )
    else:
        if not isinstance(
            authoring_capability_readiness,
            AuthoringCapabilityReadiness,
        ):
            raise TypeError("authoring capability readiness is invalid")
        admission = candidate_admission_policy.admission_policy_descriptor()
        if (
            backend_adapter.candidate_execution_authorization_schema
            != admission.execution_authorization_schema_version
        ):
            raise ValueError(
                "candidate execution authorization schema does not match backend"
            )
        authoring_capability_readiness.assert_matches(
            profile=capability_profile,
            admission_policy_id=admission.policy_id,
            admission_policy_hash=admission.policy_hash,
            admission_capability_ids=admission.capability_ids,
            materializer_runtime_capability_ids=(
                backend_adapter.candidate_execution_capability_ids
            ),
            task_smoke_capability_ids=smoke_adapter.candidate_capability_ids,
        )
        authoring_capability_readiness.require_selected_capabilities_ready()
        prompt_capability_readiness = (
            authoring_capability_readiness.to_prompt_mapping()
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "ledger.jsonl"
    initial_ledger = load_ledger(ledger_path)
    if len(initial_ledger) > rounds:
        raise ValueError("live session already contains more rounds than requested")
    _validate_round_sequence(initial_ledger, session_id=session_id)
    _validate_ledger_runtime_identity(
        initial_ledger,
        task_id=task_plugin.task_spec.task_id,
        backend_id=backend_adapter.backend_id,
    )
    initial_replay = replay_acceptance_ledger(
        initial_ledger,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=replay_epoch_hash,
    )
    initial_replay.raise_for_continuation()
    resumed_from_round_count = len(initial_ledger)

    context_builder = PublicContextBuilder(
        task_memory_adapter=task_memory_adapter,
        task_prompt_adapter=task_prompt_adapter,
        acceptance_protocol=acceptance_protocol,
        initial_current_best=initial_current_best,
        session_genesis_hash=session_genesis_hash,
        candidate_capability_profile=(
            capability_profile
        ),
        initial_reference_program_source=(
            session_genesis.reference_program_source()
        ),
        initial_anchor_public_evidence=(
            session_genesis.initial_anchor_public_evidence()
        ),
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        ledger_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=replay_epoch_hash,
    )
    invocation_round_ids: list[str] = []
    for index in range(resumed_from_round_count, rounds):
        round_id = f"{session_id}-round-{index + 1:03d}"
        invocation_round_ids.append(round_id)
        ledger = load_ledger(ledger_path)
        replay = replay_acceptance_ledger(
            ledger,
            initial_current_best=initial_current_best,
            expected_session_genesis_hash=session_genesis_hash,
            acceptance_protocol=acceptance_protocol,
            expected_initial_policy_state_hash=(
                expected_initial_policy_state_hash
            ),
            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            expected_ledger_epoch_hash=replay_epoch_hash,
        )
        replay.raise_for_continuation()
        current_best_eval_stats = replay.current_best
        records = [ExperimentRecord.from_ledger_record(record) for record in ledger]
        authoring_result = CandidateAuthoringLoop(
            context_builder=context_builder,
            provider=provider,
            candidate_admission_policy=candidate_admission_policy,
            round_dir=output_dir / "rounds" / round_id,
            max_attempts=max_provider_attempts,
            component_smoke_adapter=smoke_adapter,
            authoring_capability_readiness=prompt_capability_readiness,
        ).run_round(round_id=round_id, records=records)
        if (
            authoring_result.parent_program_source_checksum
            != current_best_eval_stats.program_source_checksum
        ):
            raise ValueError(
                "authoring parent program source does not match replay current-best"
            )
        if authoring_result.candidate is None:
            assert authoring_result.provider_failure is not None
            _append_provider_failure_record(
                ledger_path=ledger_path,
                round_id=round_id,
                task_plugin=task_plugin,
                backend_adapter=backend_adapter,
                prompt_payload_ref=authoring_result.prompt_payload_ref,
                authoring_evidence_ref=(
                    authoring_result.authoring_evidence_ref
                ),
                round_public_context=authoring_result.round_public_context,
                session_genesis_hash=session_genesis_hash,
                evaluation_binding=evaluation_binding,
                parent_program_source_checksum=(
                    authoring_result.parent_program_source_checksum
                ),
                ledger_epoch_hash=ledger_epoch_hash,
            )
            continue
        candidate = authoring_result.candidate
        if authoring_result.static_failure is not None:
            _append_static_authoring_failure_record(
                ledger_path=ledger_path,
                round_id=round_id,
                task_plugin=task_plugin,
                backend_adapter=backend_adapter,
                prompt_payload_ref=authoring_result.prompt_payload_ref,
                authoring_evidence_ref=(
                    authoring_result.authoring_evidence_ref
                ),
                round_public_context=authoring_result.round_public_context,
                candidate=candidate,
                session_genesis_hash=session_genesis_hash,
                evaluation_binding=evaluation_binding,
                parent_program_source_checksum=(
                    authoring_result.parent_program_source_checksum
                ),
                ledger_epoch_hash=ledger_epoch_hash,
            )
            continue
        run_candidate_round(
            RoundInputs(
                round_id=round_id,
                task_plugin=task_plugin,
                backend_adapter=backend_adapter,
                runner=runner,
                candidate_admission_policy=candidate_admission_policy,
                candidate=candidate,
                policy_initialization=policy_initialization,
                ledger_path=ledger_path,
                public_context=authoring_result.round_public_context,
                current_best_eval_stats=current_best_eval_stats,
                session_genesis_hash=session_genesis_hash,
                evaluation_binding=evaluation_binding,
                authoring_mode=PROVIDER_AUTHORING_MODE,
                prompt_payload_ref=authoring_result.prompt_payload_ref,
                authoring_evidence_ref=(
                    authoring_result.authoring_evidence_ref
                ),
                record_candidate_runtime_failures=True,
                candidate_artifact_root=output_dir / "candidate_artifacts",
                expected_initial_policy_state_hash=(
                    expected_initial_policy_state_hash
                ),
                backend_execution_environment_manifest_hash=(
                    backend_execution_environment_manifest_hash
                ),
                parent_program_source_checksum=(
                    authoring_result.parent_program_source_checksum
                ),
                ledger_epoch_hash=ledger_epoch_hash,
            )
        )
    ledger = load_ledger(ledger_path)
    replay = replay_acceptance_ledger(
        ledger,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=replay_epoch_hash,
    )
    replay.raise_for_continuation()
    current_best_eval_stats = replay.current_best
    final_best_score = current_best_eval_stats.score_for_comparison()
    outcome_counts = Counter(
        str(entry.public_decision.get("outcome")) for entry in replay.entries
    )
    train_result_count = sum(
        1 for record in ledger if _nonempty_mapping(record.get("train_result"))
    )
    public_result_count = sum(
        1 for record in ledger if _nonempty_mapping(record.get("public_result"))
    )
    provider_failure_count = sum(
        1 for record in ledger if _nonempty_mapping(record.get("provider_failure"))
    )
    static_authoring_failure_count = sum(
        1
        for record in ledger
        if _nonempty_mapping(record.get("static_authoring_failure"))
    )
    authoring_failure_round_count = provider_failure_count + static_authoring_failure_count
    trainable_round_count = sum(
        1
        for record in ledger
        if _nonempty_mapping(record.get("candidate"))
        and not _nonempty_mapping(record.get("provider_failure"))
        and not _nonempty_mapping(record.get("static_authoring_failure"))
    )
    prompt_payload_paths = _prompt_payload_paths(ledger)
    candidate_paths = _runner_candidate_paths(ledger)
    authoring_history = _authoring_history(ledger)
    invocation_round_id_set = set(invocation_round_ids)
    invocation_attempt_count = sum(
        1
        for item in authoring_history["attempt_records"]
        if item.get("round_id") in invocation_round_id_set
    )
    runner_candidate_count = len(candidate_paths)
    eval_run_audit = build_eval_run_audit(
        ledger,
        acceptance_protocol=acceptance_protocol,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=session_genesis_hash,
        expected_initial_policy_state_hash=expected_initial_policy_state_hash,
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=replay_epoch_hash,
    )
    eval_audit_counts = eval_run_audit["counts"]
    summary = {
        "schema_version": LIVE_LOOP_SUMMARY_SCHEMA_VERSION,
        "mode": "live",
        "session_id": session_id,
        "session_genesis_hash": session_genesis_hash,
        "ledger_epoch_hash": ledger_epoch_hash,
        "session_genesis_kind": session_genesis.artifact.genesis_kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "session_genesis_dir": str(session_genesis_dir),
        "ledger_epoch": (
            ledger_epoch.to_summary_dict()
            if ledger_epoch is not None
            else None
        ),
        "ledger_path": str(ledger_path),
        "rounds_requested": rounds,
        "rounds_completed": len(ledger),
        "resumed_from_round_count": resumed_from_round_count,
        "rounds_executed_this_invocation": rounds - resumed_from_round_count,
        "prompt_payload_paths": prompt_payload_paths,
        "provider_attempt_count": len(authoring_history["attempt_records"]),
        "provider_attempt_count_for_rounds_closed_this_invocation": (
            invocation_attempt_count
        ),
        "authoring_evidence_paths": authoring_history["evidence_paths"],
        "provider_attempt_history_authority": {
            "mode": "ledger_bound_immutable_round_artifacts",
            "latest_summary_is_projection": True,
            "round_evidence_count": len(authoring_history["evidence_paths"]),
            "bound_round_count": authoring_history["bound_round_count"],
            "interrupted_attempt_count": authoring_history[
                "interrupted_attempt_count"
            ],
        },
        "candidate_paths": candidate_paths,
        "decision_outcome_counts": dict(sorted(outcome_counts.items())),
        "runner_candidate_count": runner_candidate_count,
        "trainable_round_count": trainable_round_count,
        "train_result_count": train_result_count,
        "public_result_count": public_result_count,
        "authored_candidate_count": eval_audit_counts["authored_candidate_count"],
        "trainable_candidate_count": eval_audit_counts["trainable_candidate_count"],
        "trained_candidate_count": eval_audit_counts["trained_candidate_count"],
        "full_eval_completed_count": eval_audit_counts["full_eval_completed_count"],
        "sealed_validation_completed_count": eval_audit_counts[
            "sealed_validation_completed_count"
        ],
        "accepted_count": eval_audit_counts["accepted_count"],
        "eval_incomplete_count": eval_audit_counts["eval_incomplete_count"],
        "eval_failure_bucket_counts": eval_run_audit["failure_buckets"],
        "provider_failure_count": provider_failure_count,
        "static_authoring_failure_count": static_authoring_failure_count,
        "authoring_failure_round_count": authoring_failure_round_count,
        "round_accounting": {
            "rounds_completed": len(ledger),
            "runner_candidate_count": runner_candidate_count,
            "trainable_round_count": trainable_round_count,
            "provider_failure_count": provider_failure_count,
            "static_authoring_failure_count": static_authoring_failure_count,
            "candidate_paths_semantics": (
                "candidates handed to the runner; provider/static authoring failures "
                "do not create candidate_paths entries"
            ),
        },
        "final_best_score": final_best_score,
        "final_best_eval_stats": current_best_eval_stats.to_public_dict(),
        "checks": {
            "round_count_matches_request": len(ledger) == rounds,
            "prompt_payload_count_matches_rounds": len(prompt_payload_paths) == rounds,
            "authoring_provenance_classified": (
                authoring_history["bound_round_count"] == rounds
            ),
            "authoring_evidence_count_matches_bound_rounds": (
                len(authoring_history["evidence_paths"])
                == authoring_history["bound_round_count"]
            ),
            "provider_attempt_history_complete": (
                len(authoring_history["attempt_records"])
                >= authoring_history["bound_round_count"]
            ),
            "runner_candidate_count_matches_trainable_rounds": (
                runner_candidate_count == trainable_round_count
            ),
            "authoring_failures_accounted_for": (
                len(ledger) == trainable_round_count + authoring_failure_round_count
            ),
            "at_least_one_train_result": train_result_count > 0,
            "at_least_one_public_result": public_result_count > 0,
            "eval_run_audit_passed": all(eval_run_audit["checks"].values()),
        },
        "eval_run_audit": eval_run_audit,
    }
    if set(summary["checks"]) != LIVE_LOOP_SUMMARY_REQUIRED_CHECKS:
        raise RuntimeError("live summary checks drifted from the lifecycle contract")
    if set(summary) != LIVE_LOOP_SUMMARY_FIELDS:
        raise RuntimeError("live summary fields drifted from the lifecycle contract")
    summary_path = output_dir / "live_summary.json"
    write_live_summary(summary_path, summary)
    return summary


def _append_provider_failure_record(
    *,
    ledger_path: Path,
    round_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    prompt_payload_ref: PromptPayloadRef,
    authoring_evidence_ref: AuthoringEvidenceRef,
    round_public_context: Mapping[str, Any],
    session_genesis_hash: str,
    evaluation_binding: EvaluationBindingV1,
    parent_program_source_checksum: str | None,
    ledger_epoch_hash: str,
) -> None:
    outcome = _current_authoring_outcome(
        round_id=round_id,
        prompt_payload_ref=prompt_payload_ref,
        authoring_evidence_ref=authoring_evidence_ref,
        expected_stop_reason="provider_failed",
    )
    surface = {
        "classification": "provider_failure",
        "eligible_for_policy_accept": False,
        "reason": "provider_candidate_generation_failed",
    }
    record = sanitize_public_context(
        {
            "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
            "authoring_mode": PROVIDER_AUTHORING_MODE,
            "round_id": round_id,
            "parent_program_source_checksum": (
                parent_program_source_checksum
            ),
            "session_genesis_hash": session_genesis_hash,
            "ledger_epoch_hash": ledger_epoch_hash,
            "evaluation_binding": evaluation_binding.to_public_dict(),
            "prompt_payload": prompt_payload_ref.to_public_dict(),
            "public_context": sanitize_public_context(round_public_context),
            "source_request_feedback": project_source_request_round_feedback(
                outcome["source_request_feedback"]
            ),
            "task": {"task_id": task_plugin.task_spec.task_id},
            "backend": {"backend_id": backend_adapter.backend_id},
            "candidate": None,
            "verifier": None,
            "candidate_surface": surface,
            "code_change_classification": surface,
            "provider_failure": dict(outcome["provider_failure"]),
            "curriculum": None,
            "runtime_activation_summary": None,
            "ablation_summary": None,
            "mechanism_attribution": None,
            "anti_tuning_decision": None,
            "jobs": None,
            "train_result": None,
            "public_result": None,
            "runner_error": None,
            "decision": {
                "outcome": "reject",
                "accepted": False,
                "reason": "provider_candidate_generation_failed",
                "surface_classification": "provider_failure",
                "public_score": None,
            },
            "feedback_safe": {
                "public_only": True,
                "private_eval_used": False,
                "hidden_seed_data_included": False,
            },
        }
    )
    record["private_artifacts"] = [
        authoring_evidence_ref.to_private_dict()
    ]
    validate_decision_record(
        record,
        acceptance_protocol=task_plugin.task_spec.acceptance_protocol,
        expected_session_genesis_hash=session_genesis_hash,
        expected_evaluation_binding=evaluation_binding,
    ).raise_for_failure()
    _raise_for_prompt_artifact_failure(record)
    append_ledger_record_once(ledger_path, record)


def _append_static_authoring_failure_record(
    *,
    ledger_path: Path,
    round_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    prompt_payload_ref: PromptPayloadRef,
    authoring_evidence_ref: AuthoringEvidenceRef,
    round_public_context: Mapping[str, Any],
    candidate: Any,
    session_genesis_hash: str,
    evaluation_binding: EvaluationBindingV1,
    parent_program_source_checksum: str | None,
    ledger_epoch_hash: str,
) -> None:
    outcome = _current_authoring_outcome(
        round_id=round_id,
        prompt_payload_ref=prompt_payload_ref,
        authoring_evidence_ref=authoring_evidence_ref,
        expected_stop_reason="static_failed",
    )
    surface = {
        "classification": "static_authoring_failure",
        "eligible_for_policy_accept": False,
        "reason": "candidate_static_authoring_failed",
    }
    record = sanitize_public_context(
        {
            "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
            "authoring_mode": PROVIDER_AUTHORING_MODE,
            "round_id": round_id,
            "parent_program_source_checksum": (
                parent_program_source_checksum
            ),
            "session_genesis_hash": session_genesis_hash,
            "ledger_epoch_hash": ledger_epoch_hash,
            "evaluation_binding": evaluation_binding.to_public_dict(),
            "prompt_payload": prompt_payload_ref.to_public_dict(),
            "public_context": sanitize_public_context(round_public_context),
            "source_request_feedback": project_source_request_round_feedback(
                outcome["source_request_feedback"]
            ),
            "task": {"task_id": task_plugin.task_spec.task_id},
            "backend": {"backend_id": backend_adapter.backend_id},
            "candidate": candidate.to_public_dict(),
            "verifier": None,
            "candidate_surface": surface,
            "code_change_classification": surface,
            "provider_failure": None,
            "static_authoring_failure": dict(
                outcome["static_authoring_failure"]
            ),
            "curriculum": None,
            "runtime_activation_summary": None,
            "ablation_summary": None,
            "mechanism_attribution": None,
            "anti_tuning_decision": None,
            "jobs": None,
            "train_result": None,
            "public_result": None,
            "runner_error": None,
            "decision": {
                "outcome": "reject",
                "accepted": False,
                "reason": "candidate_static_authoring_failed",
                "surface_classification": "static_authoring_failure",
                "public_score": None,
            },
            "feedback_safe": {
                "public_only": True,
                "private_eval_used": False,
                "hidden_seed_data_included": False,
            },
        }
    )
    record["private_artifacts"] = [
        authoring_evidence_ref.to_private_dict()
    ]
    validate_decision_record(
        record,
        acceptance_protocol=task_plugin.task_spec.acceptance_protocol,
        expected_session_genesis_hash=session_genesis_hash,
        expected_evaluation_binding=evaluation_binding,
    ).raise_for_failure()
    _raise_for_prompt_artifact_failure(record)
    append_ledger_record_once(ledger_path, record)


def _current_authoring_outcome(
    *,
    round_id: str,
    prompt_payload_ref: PromptPayloadRef,
    authoring_evidence_ref: AuthoringEvidenceRef,
    expected_stop_reason: str,
) -> dict[str, Any]:
    evidence = verify_authoring_evidence(
        authoring_evidence_ref,
        expected_round_id=round_id,
        expected_selected_prompt=prompt_payload_ref,
    )
    outcome = authoring_outcome_from_evidence(evidence)
    if outcome["stop_reason"] != expected_stop_reason:
        raise ValueError("authoring evidence terminal outcome does not match round path")
    return outcome


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _validate_round_sequence(
    ledger: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
) -> None:
    for index, record in enumerate(ledger, start=1):
        expected = f"{session_id}-round-{index:03d}"
        if record.get("round_id") != expected:
            raise ValueError(
                "live ledger round sequence mismatch: "
                f"expected={expected!r} actual={record.get('round_id')!r}"
            )


def _validate_ledger_runtime_identity(
    ledger: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    backend_id: str,
) -> None:
    for record in ledger:
        round_id = record.get("round_id")
        task = record.get("task")
        if not isinstance(task, Mapping) or task.get("task_id") != task_id:
            raise ValueError(
                "live ledger task identity mismatch: "
                f"round={round_id!r} expected={task_id!r}"
            )
        backend = record.get("backend")
        if (
            not isinstance(backend, Mapping)
            or backend.get("backend_id") != backend_id
        ):
            raise ValueError(
                "live ledger backend identity mismatch: "
                f"round={round_id!r} expected={backend_id!r}"
            )


def _prompt_payload_paths(ledger: Sequence[Mapping[str, Any]]) -> list[str]:
    paths: list[str] = []
    for record in ledger:
        payload = record.get("prompt_payload")
        if isinstance(payload, Mapping) and isinstance(payload.get("path"), str):
            paths.append(payload["path"])
    return paths


def _authoring_history(
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    prompt_paths: list[str] = []
    evidence_paths: list[str] = []
    bound_round_count = 0
    interrupted_attempt_count = 0
    for record in ledger:
        _raise_for_prompt_artifact_failure(record)
        evidence_ref, evidence = load_ledger_authoring_evidence(record)
        bound_round_count += 1
        evidence_paths.append(evidence_ref.path)
        raw_attempts = evidence["attempts"]
        runner_status = _ledger_runner_status(record)
        selected_attempt_id = evidence["selected_attempt_id"]
        for raw_item in raw_attempts:
            if raw_item["status"] != "completed":
                interrupted_attempt_count += 1
                continue
            item = {
                "attempt_id": raw_item["attempt_id"],
                "sequence": raw_item["sequence"],
                "prompt_payload": dict(raw_item["prompt_payload"]),
                "provider_attempt": dict(raw_item["provider_attempt"]),
            }
            if (
                runner_status is not None
                and raw_item["attempt_id"] == selected_attempt_id
            ):
                item["provider_attempt"]["runner_status"] = runner_status
            attempts.append(item)
            records.append(dict(item["provider_attempt"]))
            prompt_paths.append(str(item["prompt_payload"]["path"]))
    return {
        "attempts": attempts,
        "attempt_records": records,
        "prompt_payload_paths": prompt_paths,
        "evidence_paths": evidence_paths,
        "bound_round_count": bound_round_count,
        "interrupted_attempt_count": interrupted_attempt_count,
    }


def _ledger_runner_status(record: Mapping[str, Any]) -> str | None:
    stage = validate_round_stage(record).stage
    if stage in {TRAIN_FAILURE_STAGE, POST_TRAIN_FAILURE_STAGE}:
        return "fail"
    if stage in {VERIFIER_FAILURE_STAGE, PRE_TRAIN_REJECTION_STAGE}:
        return "not_run"
    if stage == EVALUATED_STAGE:
        return "pass"
    if stage in {PROVIDER_FAILURE_STAGE, STATIC_AUTHORING_FAILURE_STAGE}:
        return None
    raise ValueError(f"unsupported ledger round stage: {stage}")


def _raise_for_prompt_artifact_failure(record: Mapping[str, Any]) -> None:
    reasons = validate_ledger_prompt_artifacts(
        record,
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
    )
    if reasons:
        raise ValueError(
            "invalid ledger prompt artifacts: " + ", ".join(reasons)
        )


def _runner_candidate_paths(ledger: Sequence[Mapping[str, Any]]) -> list[str]:
    paths: list[str] = []
    for record in ledger:
        if not isinstance(record.get("candidate"), Mapping):
            continue
        if record.get("provider_failure") is not None:
            continue
        if record.get("static_authoring_failure") is not None:
            continue
        artifacts = record.get("private_artifacts")
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if (
                isinstance(artifact, Mapping)
                and artifact.get("kind")
                in {"candidate_artifact", "candidate_package"}
                and isinstance(artifact.get("uri"), str)
            ):
                paths.append(artifact["uri"])
                break
    return paths
