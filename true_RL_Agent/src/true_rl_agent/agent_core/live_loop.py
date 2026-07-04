"""Live multi-round loop with persisted prompt payloads."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
from true_rl_agent.agent_core.context import sanitize_public_context
from true_rl_agent.agent_core.ledger import append_ledger_record, load_ledger
from true_rl_agent.agent_core.provider import CandidateProvider
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.verifier import CandidateVerifier
from true_rl_agent.memory import ExperimentRecord, PromptPayloadRef, PublicContextBuilder
from true_rl_agent.memory.task_adapter import TaskMemoryAdapter
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import Runner
from true_rl_agent.tasks.base import TaskPlugin


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
    initial_best_score: float,
    policy_accept_delta: float,
    task_memory_adapter: TaskMemoryAdapter | None = None,
    max_provider_attempts: int = 2,
) -> dict[str, Any]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if max_provider_attempts <= 0:
        raise ValueError("max_provider_attempts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "ledger.jsonl"
    if ledger_path.exists():
        raise FileExistsError(f"live output already contains a ledger: {ledger_path}")

    context_builder = PublicContextBuilder(
        task_memory_adapter=task_memory_adapter,
        task_prompt_adapter=task_plugin.prompt_adapter(),
    )
    prompt_payload_paths: list[str] = []
    provider_attempt_prompt_payload_paths: list[str] = []
    provider_attempt_records: list[dict[str, Any]] = []
    candidate_paths: list[str] = []
    current_best_score = float(initial_best_score)

    for index in range(rounds):
        round_id = f"{session_id}-round-{index + 1:03d}"
        ledger = load_ledger(ledger_path)
        records = [ExperimentRecord.from_ledger_record(record) for record in ledger]
        authoring_result = CandidateAuthoringLoop(
            context_builder=context_builder,
            provider=provider,
            verifier=verifier,
            round_dir=output_dir / "rounds" / round_id,
            max_attempts=max_provider_attempts,
        ).run_round(round_id=round_id, records=records)
        provider_attempt_prompt_payload_paths.extend(authoring_result.attempt_prompt_payload_paths)
        provider_attempt_records.extend(authoring_result.attempt_records)
        prompt_payload_paths.append(str(authoring_result.prompt_payload_path))

        if authoring_result.candidate is None:
            assert authoring_result.provider_failure is not None
            _append_provider_failure_record(
                ledger_path=ledger_path,
                round_id=round_id,
                task_plugin=task_plugin,
                backend_adapter=backend_adapter,
                prompt_payload_ref=authoring_result.prompt_payload_ref,
                public_context=authoring_result.public_context,
                provider_failure=authoring_result.provider_failure,
                attempt_records=authoring_result.attempt_records,
            )
            continue
        candidate = authoring_result.candidate
        candidate_paths.append(candidate.path)
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
                public_context=authoring_result.public_context,
                current_best_score=current_best_score,
                policy_accept_delta=policy_accept_delta,
                prompt_payload_ref=authoring_result.prompt_payload_ref,
                record_candidate_runtime_failures=True,
            )
        )
        authoring_result.attempt_records[-1]["runner_status"] = _runner_status(result)
        if result.decision.accepted and result.decision.public_score is not None:
            current_best_score = max(current_best_score, result.decision.public_score)

    ledger = load_ledger(ledger_path)
    outcome_counts = Counter(str(record.get("decision", {}).get("outcome")) for record in ledger)
    train_result_count = sum(1 for record in ledger if record.get("train_result") is not None)
    public_result_count = sum(1 for record in ledger if record.get("public_result") is not None)
    provider_failure_count = sum(1 for record in ledger if record.get("provider_failure") is not None)
    summary = {
        "schema_version": "true_rl_agent.live_loop.summary.v1",
        "mode": "live",
        "session_id": session_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "ledger_path": str(ledger_path),
        "rounds_requested": rounds,
        "rounds_completed": len(ledger),
        "prompt_payload_paths": prompt_payload_paths,
        "provider_attempt_prompt_payload_paths": provider_attempt_prompt_payload_paths,
        "provider_attempt_records": provider_attempt_records,
        "candidate_paths": candidate_paths,
        "decision_outcome_counts": dict(sorted(outcome_counts.items())),
        "train_result_count": train_result_count,
        "public_result_count": public_result_count,
        "provider_failure_count": provider_failure_count,
        "final_best_score": current_best_score,
        "checks": {
            "round_count_matches_request": len(ledger) == rounds,
            "prompt_payload_count_matches_rounds": len(prompt_payload_paths) == rounds,
            "candidate_count_matches_rounds": len(candidate_paths) == rounds,
            "at_least_one_train_result": train_result_count > 0,
            "at_least_one_public_result": public_result_count > 0,
        },
    }
    summary_path = output_dir / "live_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _runner_status(result: Any) -> str:
    if result.decision.reason == "candidate_runtime_failed":
        return "fail"
    if (
        result.decision.reason == "candidate_verifier_failed"
        and result.train_result is None
        and result.public_result is None
    ):
        return "not_run"
    return "pass"


def _append_provider_failure_record(
    *,
    ledger_path: Path,
    round_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    prompt_payload_ref: PromptPayloadRef,
    public_context: Mapping[str, Any],
    provider_failure: Mapping[str, Any],
    attempt_records: Sequence[Mapping[str, Any]],
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
            "prompt_payload": prompt_payload_ref.to_public_dict(),
            "public_context": sanitize_public_context(public_context),
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
    append_ledger_record(ledger_path, record)
