"""Fifteen-round dry-run audit for prompt and curriculum boundaries."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter, IsaacLabRslRlBackendConfig
from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import CandidateVerifier, VerificationResult
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.dry_run import DryRunAuditRunner, DryRunCandidateMetadata
from true_rl_agent.tasks.push_recovery import PushRecoveryTaskConfig, PushRecoveryTaskPlugin, PushTaskMemoryAdapter


FORBIDDEN_PROMPT_TEXT = (
    "private_seed",
    "hidden_eval",
    "trusted_eval",
    "trusted_seed",
    "checkpoint_selector",
)


@dataclass(frozen=True)
class AuditCandidateSpec:
    candidate_id: str
    changed_components: tuple[str, ...]
    public_score: float
    failure_breakdown: Mapping[str, float]
    motion_quality: Mapping[str, Any]

    def to_candidate(self) -> CandidatePackageRef:
        return CandidatePackageRef(
            candidate_id=self.candidate_id,
            path=f"audit://{self.candidate_id}",
            change_set=CandidateChangeSet(frozenset(self.changed_components)),
            metadata={"dry_run_public_score": self.public_score},
        )

    def to_dry_run_metadata(self) -> DryRunCandidateMetadata:
        return DryRunCandidateMetadata(
            changed_components=self.changed_components,
            public_score=self.public_score,
            failure_breakdown=self.failure_breakdown,
            motion_quality=self.motion_quality,
        )


class AuditVerifier(CandidateVerifier):
    def verify(self, candidate: CandidatePackageRef) -> VerificationResult:
        _ = candidate
        return VerificationResult(
            status="pass",
            reason="audit_replay_candidate_preverified",
            details={"dry_run": True},
        )


class AuditReplayProvider:
    def __init__(self, candidates: Sequence[CandidatePackageRef]) -> None:
        self._remaining = list(candidates)

    def next_candidate(self, *, prompt_payload: dict[str, Any]) -> CandidatePackageRef:
        _ = prompt_payload
        if not self._remaining:
            raise StopIteration("audit replay provider has no remaining candidates")
        return self._remaining.pop(0)


def default_audit_candidates(rounds: int) -> list[AuditCandidateSpec]:
    base = [
        ("audit-001-reward-weight", ("reward_weight_scale",), 0.95, {"fall": 0.30}),
        ("audit-002-diagnostic", ("diagnostic",), 0.32, {"fall": 0.24}),
        ("audit-003-failure-classifier", ("failure_classifier",), 0.35, {"timeout": 0.18}),
        ("audit-004-reward-code", ("reward_callable",), 0.20, {"fall": 0.20}),
        ("audit-005-schedule", ("schedule",), 0.99, {"push_drift": 0.28}),
        ("audit-006-small-code", ("reward_callable",), 0.22, {"fall": 0.19}),
        ("audit-007-recovery-logic", ("recovery_logic",), 0.27, {"push_drift": 0.16}),
        ("audit-008-seed", ("seed",), 0.99, {"fall": 0.25}),
        ("audit-009-public-projection", ("public_projection",), 0.30, {"timeout": 0.14}),
        ("audit-010-control-logic", ("control_logic",), 0.31, {"push_drift": 0.13}),
        ("audit-011-reward-code", ("reward_callable",), 0.34, {"fall": 0.12}),
        ("audit-012-budget", ("budget",), 0.99, {"fall": 0.11}),
        ("audit-013-diagnostic", ("diagnostic",), 0.36, {"timeout": 0.10}),
        ("audit-014-recovery-logic", ("recovery_logic",), 0.38, {"push_drift": 0.09}),
        ("audit-015-reward-code", ("reward_callable",), 0.40, {"fall": 0.08}),
    ]
    specs = [
        AuditCandidateSpec(
            candidate_id=candidate_id,
            changed_components=changed_components,
            public_score=public_score,
            failure_breakdown=failure_breakdown,
            motion_quality={"mean_base_height": 0.78 + index * 0.001, "mean_tilt_rad": 0.05},
        )
        for index, (candidate_id, changed_components, public_score, failure_breakdown) in enumerate(base)
    ]
    if rounds <= len(specs):
        return specs[:rounds]
    extended = list(specs)
    while len(extended) < rounds:
        template = specs[len(extended) % len(specs)]
        next_index = len(extended) + 1
        extended.append(
            AuditCandidateSpec(
                candidate_id=f"audit-{next_index:03d}-{template.candidate_id.rsplit('-', 1)[-1]}",
                changed_components=template.changed_components,
                public_score=template.public_score,
                failure_breakdown=template.failure_breakdown,
                motion_quality=template.motion_quality,
            )
        )
    return extended


def run_loop_audit(*, output_dir: Path, rounds: int = 15, session_id: str | None = None) -> dict[str, Any]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    session_id = session_id or f"audit-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "ledger.jsonl"
    if ledger_path.exists():
        raise FileExistsError(f"audit output already contains a ledger: {ledger_path}")
    candidates = default_audit_candidates(rounds)
    provider = AuditReplayProvider([candidate.to_candidate() for candidate in candidates])
    runner = DryRunAuditRunner({item.candidate_id: item.to_dry_run_metadata() for item in candidates})
    task_config = PushRecoveryTaskConfig()
    task = PushRecoveryTaskPlugin(task_config)
    context_builder = PublicContextBuilder(
        task_memory_adapter=PushTaskMemoryAdapter(),
        task_prompt_adapter=task.prompt_adapter(),
    )
    backend = IsaacLabRslRlBackendAdapter(
        IsaacLabRslRlBackendConfig(
            isaac_lab_python="/opt/isaac/python.sh",
            train_entrypoint="train_push.py",
            eval_entrypoint="eval_push.py",
            default_budget_steps=task_config.budget_steps,
            train_num_envs=task_config.train_num_envs,
            train_max_iterations=task_config.train_max_iterations,
            train_steps_per_iteration=task_config.train_steps_per_iteration,
            public_seed_count=3,
            trusted_seed_count=7,
        )
    )

    current_best_score = 0.10
    prompt_payload_paths: list[str] = []
    for index in range(rounds):
        round_id = f"{session_id}-round-{index + 1:03d}"
        ledger = load_ledger(ledger_path)
        records = [ExperimentRecord.from_ledger_record(record) for record in ledger]
        prompt_payload = context_builder.build_prompt_payload(records)
        prompt_path = output_dir / "rounds" / round_id / "prompt_payload.json"
        prompt_payload_ref = context_builder.save_prompt_payload(prompt_path, prompt_payload)
        prompt_payload_paths.append(str(prompt_path))

        candidate = provider.next_candidate(prompt_payload=prompt_payload)
        result = run_candidate_round(
            RoundInputs(
                round_id=round_id,
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                verifier=AuditVerifier(),
                candidate=candidate,
                policy_initialization=PolicyInitialization.random(),
                ledger_path=ledger_path,
                public_context=prompt_payload["public_context"],
                current_best_score=current_best_score,
                policy_accept_delta=0.05,
                prompt_payload_ref=prompt_payload_ref,
            )
        )
        if result.decision.accepted and result.decision.public_score is not None:
            current_best_score = max(current_best_score, result.decision.public_score)

    ledger = load_ledger(ledger_path)
    summary = build_audit_summary(
        session_id=session_id,
        rounds_requested=rounds,
        output_dir=output_dir,
        ledger=ledger,
        prompt_payload_paths=prompt_payload_paths,
    )
    summary_path = output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_audit_summary(
    *,
    session_id: str,
    rounds_requested: int,
    output_dir: Path,
    ledger: Sequence[Mapping[str, Any]],
    prompt_payload_paths: Sequence[str],
) -> dict[str, Any]:
    prompt_checks = [_inspect_prompt_payload(Path(path)) for path in prompt_payload_paths]
    outcome_counts = Counter(str(record.get("decision", {}).get("outcome")) for record in ledger)
    surface_counts = Counter(str(record.get("candidate_surface", {}).get("classification")) for record in ledger)
    parameter_only_accepts = [
        record.get("round_id")
        for record in ledger
        if record.get("candidate_surface", {}).get("classification") == "parameter_only"
        and record.get("decision", {}).get("accepted") is True
    ]
    curriculum_violations = [
        record.get("round_id")
        for record in ledger
        if record.get("curriculum", {}).get("llm_controlled") is not False
    ]
    train_schedule_violations = [
        record.get("round_id")
        for record in ledger
        if _job_train_contains_llm_schedule(record)
    ]

    checks = {
        "round_count_matches_request": len(ledger) == rounds_requested,
        "prompt_payload_count_matches_rounds": len(prompt_payload_paths) == rounds_requested,
        "prompt_payloads_public_only": all(item["public_only"] for item in prompt_checks),
        "curriculum_not_llm_controlled": not curriculum_violations,
        "train_jobs_do_not_include_llm_schedule": not train_schedule_violations,
        "parameter_only_candidates_not_accepted": not parameter_only_accepts,
        "has_code_level_policy_accept": (
            rounds_requested < 4 or outcome_counts.get("policy_improvement_accept", 0) > 0
        ),
    }
    return {
        "schema_version": "true_rl_agent.audit.loop.v1",
        "session_id": session_id,
        "mode": "dry_run_audit",
        "output_dir": str(output_dir),
        "ledger_path": str(output_dir / "ledger.jsonl"),
        "rounds_requested": rounds_requested,
        "rounds_completed": len(ledger),
        "prompt_payload_paths": list(prompt_payload_paths),
        "decision_outcome_counts": dict(sorted(outcome_counts.items())),
        "candidate_surface_counts": dict(sorted(surface_counts.items())),
        "checks": checks,
        "violations": {
            "prompt_payloads": [item for item in prompt_checks if not item["public_only"]],
            "curriculum_llm_controlled_rounds": curriculum_violations,
            "train_llm_schedule_rounds": train_schedule_violations,
            "parameter_only_accepted_rounds": parameter_only_accepts,
        },
    }


def _inspect_prompt_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    try:
        assert_public_only_prompt_payload(payload)
    except ValueError as exc:
        errors.append(str(exc))
    text = json.dumps(payload, sort_keys=True).lower()
    hits = [token for token in FORBIDDEN_PROMPT_TEXT if token in text]
    return {
        "path": str(path),
        "public_only": not errors and not hits,
        "errors": errors,
        "forbidden_text_hits": hits,
    }


def _job_train_contains_llm_schedule(record: Mapping[str, Any]) -> bool:
    train = (record.get("jobs") or {}).get("train") or {}
    text = json.dumps(train, sort_keys=True).lower()
    return "llm_schedule" in text or "llm_controlled" in text
