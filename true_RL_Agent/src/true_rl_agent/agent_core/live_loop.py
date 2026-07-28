"""Live multi-round loop with persisted prompt payloads."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.agent_core.eval_run_audit import build_eval_run_audit
from true_rl_agent.agent_core.ledger import append_ledger_record, load_ledger
from true_rl_agent.agent_core.provider import CandidateProvider
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.agent_core.session_genesis import (
    VerifiedSessionGenesis,
    verify_session_genesis_for_runtime,
)
from true_rl_agent.agent_core.source_request_feedback import (
    project_source_request_round_feedback,
)
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.verifier import CandidateVerifier
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.round_stage_contract import (
    EVALUATED_STAGE,
    POST_TRAIN_FAILURE_STAGE,
    PRE_TRAIN_REJECTION_STAGE,
    TRAIN_FAILURE_STAGE,
    VERIFIER_FAILURE_STAGE,
)
from true_rl_agent.memory import (
    ExperimentRecord,
    PromptPayloadRef,
    PublicContextBuilder,
)
from true_rl_agent.memory.task_adapter import TaskMemoryAdapter
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import Runner
from true_rl_agent.tasks.base import TaskPlugin

DEFAULT_MAX_PROVIDER_ATTEMPTS = 5


def run_live_loop(
    *,
    output_dir: Path,
    rounds: int,
    session_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    runner: Runner,
    verifier: CandidateVerifier,
    provider: CandidateProvider,
    policy_initialization: PolicyInitialization,
    session_genesis: VerifiedSessionGenesis,
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
        output_dir=output_dir,
        session_id=session_id,
        task_plugin=task_plugin,
        backend_adapter=backend_adapter,
        policy_initialization=policy_initialization,
    )
    initial_current_best = session_genesis.initial_current_best
    session_genesis_hash = session_genesis.session_genesis_hash
    evaluation_binding = session_genesis.evaluation_binding
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
    )
    initial_replay.raise_for_continuation()
    resumed_from_round_count = len(initial_ledger)

    context_builder = PublicContextBuilder(
        task_memory_adapter=task_memory_adapter,
        task_prompt_adapter=task_plugin.prompt_adapter(),
        acceptance_protocol=acceptance_protocol,
        initial_current_best=initial_current_best,
        session_genesis_hash=session_genesis_hash,
        candidate_capability_profile=(
            task_plugin.task_spec.candidate_capability_profile
        ),
        initial_reference_program_source=(
            session_genesis.reference_program_source()
        ),
    )
    provider_attempt_prompt_payload_paths: list[str] = []
    provider_attempt_records: list[dict[str, Any]] = []
    for index in range(resumed_from_round_count, rounds):
        round_id = f"{session_id}-round-{index + 1:03d}"
        ledger = load_ledger(ledger_path)
        replay = replay_acceptance_ledger(
            ledger,
            initial_current_best=initial_current_best,
            expected_session_genesis_hash=session_genesis_hash,
            acceptance_protocol=acceptance_protocol,
        )
        replay.raise_for_continuation()
        current_best_eval_stats = replay.current_best
        records = [ExperimentRecord.from_ledger_record(record) for record in ledger]
        authoring_result = CandidateAuthoringLoop(
            context_builder=context_builder,
            provider=provider,
            verifier=verifier,
            round_dir=output_dir / "rounds" / round_id,
            max_attempts=max_provider_attempts,
            component_smoke_adapter=task_plugin.component_smoke_adapter(),
        ).run_round(round_id=round_id, records=records)
        provider_attempt_prompt_payload_paths.extend(authoring_result.attempt_prompt_payload_paths)
        provider_attempt_records.extend(authoring_result.attempt_records)
        if authoring_result.candidate is None:
            assert authoring_result.provider_failure is not None
            _append_provider_failure_record(
                ledger_path=ledger_path,
                round_id=round_id,
                task_plugin=task_plugin,
                backend_adapter=backend_adapter,
                prompt_payload_ref=authoring_result.prompt_payload_ref,
                round_public_context=authoring_result.round_public_context,
                provider_failure=authoring_result.provider_failure,
                attempt_records=authoring_result.attempt_records,
                source_request_feedback=authoring_result.source_request_feedback,
                session_genesis_hash=session_genesis_hash,
                evaluation_binding=evaluation_binding,
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
                round_public_context=authoring_result.round_public_context,
                candidate=candidate,
                static_failure=authoring_result.static_failure,
                attempt_records=authoring_result.attempt_records,
                source_request_feedback=authoring_result.source_request_feedback,
                session_genesis_hash=session_genesis_hash,
                evaluation_binding=evaluation_binding,
            )
            continue
        result = run_candidate_round(
            RoundInputs(
                round_id=round_id,
                task_plugin=task_plugin,
                backend_adapter=backend_adapter,
                runner=runner,
                verifier=verifier,
                candidate=candidate,
                policy_initialization=policy_initialization,
                ledger_path=ledger_path,
                public_context=authoring_result.round_public_context,
                source_request_feedback=authoring_result.source_request_feedback,
                current_best_eval_stats=current_best_eval_stats,
                session_genesis_hash=session_genesis_hash,
                evaluation_binding=evaluation_binding,
                prompt_payload_ref=authoring_result.prompt_payload_ref,
                record_candidate_runtime_failures=True,
                candidate_artifact_root=output_dir / "candidate_artifacts",
            )
        )
        authoring_result.attempt_records[-1]["runner_status"] = _runner_status(result)

    ledger = load_ledger(ledger_path)
    replay = replay_acceptance_ledger(
        ledger,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
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
    runner_candidate_count = len(candidate_paths)
    eval_run_audit = build_eval_run_audit(
        ledger,
        acceptance_protocol=acceptance_protocol,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=session_genesis_hash,
    )
    eval_audit_counts = eval_run_audit["counts"]
    summary = {
        "schema_version": "true_rl_agent.live_loop.summary.v2",
        "mode": "live",
        "session_id": session_id,
        "session_genesis_hash": session_genesis_hash,
        "session_genesis_kind": session_genesis.artifact.genesis_kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "ledger_path": str(ledger_path),
        "rounds_requested": rounds,
        "rounds_completed": len(ledger),
        "resumed_from_round_count": resumed_from_round_count,
        "rounds_executed_this_invocation": rounds - resumed_from_round_count,
        "prompt_payload_paths": prompt_payload_paths,
        "provider_attempt_prompt_payload_paths": provider_attempt_prompt_payload_paths,
        "provider_attempt_records": provider_attempt_records,
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
    summary_path = output_dir / "live_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _runner_status(result: Any) -> str:
    if result.round_stage in {TRAIN_FAILURE_STAGE, POST_TRAIN_FAILURE_STAGE}:
        return "fail"
    if result.round_stage in {VERIFIER_FAILURE_STAGE, PRE_TRAIN_REJECTION_STAGE}:
        return "not_run"
    if result.round_stage == EVALUATED_STAGE:
        return "pass"
    raise ValueError(f"unsupported candidate runner stage: {result.round_stage}")


def _append_provider_failure_record(
    *,
    ledger_path: Path,
    round_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    prompt_payload_ref: PromptPayloadRef,
    round_public_context: Mapping[str, Any],
    provider_failure: Mapping[str, Any],
    attempt_records: Sequence[Mapping[str, Any]],
    source_request_feedback: Mapping[str, Any] | None,
    session_genesis_hash: str,
    evaluation_binding: EvaluationBindingV1,
) -> None:
    surface = {
        "classification": "provider_failure",
        "eligible_for_policy_accept": False,
        "reason": "provider_candidate_generation_failed",
    }
    record = sanitize_public_context(
        {
            "schema_version": "true_rl_agent.ledger_record.v1",
            "round_id": round_id,
            "session_genesis_hash": session_genesis_hash,
            "evaluation_binding": evaluation_binding.to_public_dict(),
            "prompt_payload": prompt_payload_ref.to_public_dict(),
            "public_context": sanitize_public_context(round_public_context),
            "source_request_feedback": project_source_request_round_feedback(
                source_request_feedback
            ),
            "task": {"task_id": task_plugin.task_spec.task_id},
            "backend": {"backend_id": backend_adapter.backend_id},
            "candidate": None,
            "verifier": None,
            "candidate_surface": surface,
            "code_change_classification": surface,
            "provider_failure": dict(provider_failure),
            "provider_attempt_records": [dict(item) for item in attempt_records],
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
    validate_decision_record(
        record,
        acceptance_protocol=task_plugin.task_spec.acceptance_protocol,
        expected_session_genesis_hash=session_genesis_hash,
        expected_evaluation_binding=evaluation_binding,
    ).raise_for_failure()
    append_ledger_record(ledger_path, record)


def _append_static_authoring_failure_record(
    *,
    ledger_path: Path,
    round_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    prompt_payload_ref: PromptPayloadRef,
    round_public_context: Mapping[str, Any],
    candidate: Any,
    static_failure: Mapping[str, Any],
    attempt_records: Sequence[Mapping[str, Any]],
    source_request_feedback: Mapping[str, Any] | None,
    session_genesis_hash: str,
    evaluation_binding: EvaluationBindingV1,
) -> None:
    surface = {
        "classification": "static_authoring_failure",
        "eligible_for_policy_accept": False,
        "reason": "candidate_static_authoring_failed",
    }
    record = sanitize_public_context(
        {
            "schema_version": "true_rl_agent.ledger_record.v1",
            "round_id": round_id,
            "session_genesis_hash": session_genesis_hash,
            "evaluation_binding": evaluation_binding.to_public_dict(),
            "prompt_payload": prompt_payload_ref.to_public_dict(),
            "public_context": sanitize_public_context(round_public_context),
            "source_request_feedback": project_source_request_round_feedback(
                source_request_feedback
            ),
            "task": {"task_id": task_plugin.task_spec.task_id},
            "backend": {"backend_id": backend_adapter.backend_id},
            "candidate": candidate.to_public_dict(),
            "verifier": None,
            "candidate_surface": surface,
            "code_change_classification": surface,
            "provider_failure": None,
            "static_authoring_failure": dict(static_failure),
            "provider_attempt_records": [dict(item) for item in attempt_records],
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
    validate_decision_record(
        record,
        acceptance_protocol=task_plugin.task_spec.acceptance_protocol,
        expected_session_genesis_hash=session_genesis_hash,
        expected_evaluation_binding=evaluation_binding,
    ).raise_for_failure()
    append_ledger_record(ledger_path, record)


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
                and artifact.get("kind") == "candidate_package"
                and isinstance(artifact.get("uri"), str)
            ):
                paths.append(artifact["uri"])
                break
    return paths
