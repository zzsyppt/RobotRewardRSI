"""Push-recovery-owned dry-run audit for prompt and curriculum boundaries."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.eval_run_audit import build_eval_run_audit
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.agent_core.session_genesis import prepare_session_genesis
from true_rl_agent.candidates.artifact import (
    SealedCandidateArtifactRef,
    verify_sealed_candidate_artifact,
)
from true_rl_agent.candidates.loader import (
    CANDIDATE_SCHEMA_VERSION,
    CONTROLLED_ENTRYPOINT,
)
from true_rl_agent.candidates.schedule_contract import REWARD_WEIGHT_SCALE_KEY
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import CandidateVerifier, VerificationResult
from true_rl_agent.composition.base import ResolvedTaskComposition
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.prompting import assert_public_only_prompt_payload
from true_rl_agent.runner.dry_run import DryRunAuditRunner, DryRunCandidateMetadata
from true_rl_agent.tasks.push_recovery.composition import PUSH_RECOVERY_TASK_ID
from true_rl_agent.tasks.push_recovery.dry_run_eval import (
    build_push_dry_run_public_result,
    build_push_dry_run_train_summary,
)
from true_rl_agent.training_programs import resolve_candidate_program_identity

FORBIDDEN_PROMPT_TEXT = (
    "private_seed",
    "hidden_eval",
    "trusted_eval",
    "trusted_seed",
    "checkpoint_selector",
)
AUDIT_INITIAL_BEST_SCORE = 0.10


@dataclass(frozen=True)
class AuditCandidateSpec:
    candidate_id: str
    changed_components: tuple[str, ...]
    public_score: float
    failure_breakdown: Mapping[str, float]
    task_eval_metadata: Mapping[str, Any]

    def to_candidate(self, *, package_root: Path | None = None) -> CandidatePackageRef:
        path = (
            _materialize_audit_candidate_package(package_root, self)
            if package_root is not None
            else f"audit://{self.candidate_id}"
        )
        return CandidatePackageRef(
            candidate_id=self.candidate_id,
            path=path,
            change_set=CandidateChangeSet(frozenset(self.changed_components)),
            metadata={"dry_run_public_score": self.public_score},
        )

    def to_dry_run_metadata(self) -> DryRunCandidateMetadata:
        return DryRunCandidateMetadata(
            changed_components=self.changed_components,
            public_score=self.public_score,
            failure_breakdown=self.failure_breakdown,
            task_eval_metadata=self.task_eval_metadata,
        )


class AuditVerifier(CandidateVerifier):
    def verify(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef,
    ) -> VerificationResult:
        if not isinstance(candidate, SealedCandidateArtifactRef):
            return VerificationResult(
                status="fail",
                reason="audit_candidate_not_sealed",
            )
        try:
            verify_sealed_candidate_artifact(candidate)
            identity = resolve_candidate_program_identity(
                candidate.path,
                expected_candidate_id=candidate.candidate_id,
            )
        except (OSError, TypeError, ValueError) as exc:
            return VerificationResult(
                status="fail",
                reason="audit_candidate_artifact_invalid",
                details={"issues": [f"artifact_invalid:{type(exc).__name__}"]},
            )
        return VerificationResult(
            status="pass",
            reason="audit_replay_candidate_preverified",
            details={"dry_run": True},
            program_identity=identity,
            candidate_artifact_hash=candidate.artifact_hash,
        )


class AuditReplayProvider:
    def __init__(self, candidates: Sequence[CandidatePackageRef]) -> None:
        self._remaining = list(candidates)

    def next_candidate(self, *, prompt_payload: dict[str, Any]) -> CandidatePackageRef:
        _ = prompt_payload
        if not self._remaining:
            raise StopIteration("audit replay provider has no remaining candidates")
        return self._remaining.pop(0)


def _materialize_audit_candidate_package(
    package_root: Path,
    candidate: AuditCandidateSpec,
) -> str:
    root = package_root.resolve() / candidate.candidate_id
    training_code = root / "training_code"
    training_code.mkdir(parents=True, exist_ok=False)
    (root / "candidate_manifest.yaml").write_text(
        "\n".join(
            (
                f"schema_version: {CANDIDATE_SCHEMA_VERSION}",
                f"candidate_id: {candidate.candidate_id}",
                f"entrypoint: {CONTROLLED_ENTRYPOINT}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (training_code / "components.py").write_text(
        "def build_training_components():\n"
        f"    return {{'audit_candidate': {candidate.candidate_id!r}}}\n",
        encoding="utf-8",
    )
    return str(root)


def default_audit_candidates(rounds: int) -> list[AuditCandidateSpec]:
    base = [
        ("audit-001-reward-weight", (REWARD_WEIGHT_SCALE_KEY,), 0.95, {"fall": 0.30}),
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
            task_eval_metadata={"mean_base_height": 0.78 + index * 0.001, "mean_tilt_rad": 0.05},
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
                task_eval_metadata=template.task_eval_metadata,
            )
        )
    return extended


def run_push_recovery_loop_audit(
    *,
    composition: ResolvedTaskComposition,
    output_dir: Path,
    rounds: int = 15,
    session_id: str | None = None,
) -> dict[str, Any]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if composition.task_id != PUSH_RECOVERY_TASK_ID:
        raise ValueError(
            "push recovery audit requires a unitree_push_recovery composition"
        )
    session_id = session_id or f"audit-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "ledger.jsonl"
    if ledger_path.exists():
        raise FileExistsError(f"audit output already contains a ledger: {ledger_path}")
    session_genesis = prepare_session_genesis(
        output_dir=output_dir,
        session_id=session_id,
        task_plugin=composition.task_plugin,
        backend_adapter=composition.backend_adapter,
        runner=composition.runner,
        policy_initialization=composition.policy_initialization,
        runtime_snapshot=composition.runtime_snapshot(),
        baseline_manifest=composition.baseline_manifest,
        reference_program_manifest=composition.reference_program_manifest,
        bootstrap_initial_best_score=AUDIT_INITIAL_BEST_SCORE,
    )
    candidates = default_audit_candidates(rounds)
    package_root = output_dir / "audit_candidates"
    package_root.mkdir(parents=True, exist_ok=True)
    provider = AuditReplayProvider(
        [
            candidate.to_candidate(package_root=package_root)
            for candidate in candidates
        ]
    )
    task = composition.task_plugin
    acceptance_protocol = task.task_spec.acceptance_protocol
    if acceptance_protocol is None:
        raise ValueError("push recovery audit requires an acceptance protocol")
    runner = DryRunAuditRunner(
        {item.candidate_id: item.to_dry_run_metadata() for item in candidates},
        eval_result_builder=partial(
            build_push_dry_run_public_result,
            acceptance_protocol=acceptance_protocol,
        ),
        train_summary_builder=build_push_dry_run_train_summary,
    )
    context_builder = PublicContextBuilder(
        task_memory_adapter=composition.task_memory_adapter,
        task_prompt_adapter=task.prompt_adapter(),
        acceptance_protocol=acceptance_protocol,
        initial_current_best=session_genesis.initial_current_best,
        session_genesis_hash=session_genesis.session_genesis_hash,
        candidate_capability_profile=(
            task.task_spec.candidate_capability_profile
        ),
    )
    backend = composition.backend_adapter

    prompt_payload_paths: list[str] = []
    for index in range(rounds):
        round_id = f"{session_id}-round-{index + 1:03d}"
        ledger = load_ledger(ledger_path)
        replay = replay_acceptance_ledger(
            ledger,
            initial_current_best=session_genesis.initial_current_best,
            expected_session_genesis_hash=session_genesis.session_genesis_hash,
            acceptance_protocol=acceptance_protocol,
        )
        replay.raise_for_continuation()
        current_best_eval_stats = replay.current_best
        records = [ExperimentRecord.from_ledger_record(record) for record in ledger]
        prompt_build = context_builder.build_prompt(records)
        prompt_payload = prompt_build.provider_payload
        prompt_path = output_dir / "rounds" / round_id / "prompt_payload.json"
        prompt_payload_ref = context_builder.save_prompt_payload(prompt_path, prompt_payload)
        prompt_payload_paths.append(str(prompt_path))

        candidate = provider.next_candidate(prompt_payload=prompt_payload)
        run_candidate_round(
            RoundInputs(
                round_id=round_id,
                task_plugin=task,
                backend_adapter=backend,
                runner=runner,
                verifier=AuditVerifier(),
                candidate=candidate,
                policy_initialization=composition.policy_initialization,
                ledger_path=ledger_path,
                public_context=prompt_build.round_public_context,
                current_best_eval_stats=current_best_eval_stats,
                session_genesis_hash=session_genesis.session_genesis_hash,
                evaluation_binding=session_genesis.evaluation_binding,
                prompt_payload_ref=prompt_payload_ref,
            )
        )

    ledger = load_ledger(ledger_path)
    summary = build_audit_summary(
        session_id=session_id,
        rounds_requested=rounds,
        output_dir=output_dir,
        ledger=ledger,
        prompt_payload_paths=prompt_payload_paths,
        acceptance_protocol=acceptance_protocol,
        initial_current_best=session_genesis.initial_current_best,
        session_genesis_hash=session_genesis.session_genesis_hash,
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
    acceptance_protocol: AcceptanceProtocol,
    initial_current_best: CurrentBestEvalStats,
    session_genesis_hash: str,
) -> dict[str, Any]:
    prompt_checks = [_inspect_prompt_payload(Path(path)) for path in prompt_payload_paths]
    replay = replay_acceptance_ledger(
        ledger,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=session_genesis_hash,
        acceptance_protocol=acceptance_protocol,
    )
    replay.raise_for_continuation()
    outcome_counts = Counter(
        str(entry.public_decision.get("outcome")) for entry in replay.entries
    )
    surface_counts = Counter(str(record.get("candidate_surface", {}).get("classification")) for record in ledger)
    eval_run_audit = build_eval_run_audit(
        ledger,
        acceptance_protocol=acceptance_protocol,
        initial_current_best=initial_current_best,
        expected_session_genesis_hash=session_genesis_hash,
    )
    eval_audit_counts = eval_run_audit["counts"]
    parameter_only_accepts = [
        record.get("round_id")
        for record, replay_entry in zip(ledger, replay.entries, strict=True)
        if record.get("candidate_surface", {}).get("classification") == "parameter_only"
        and replay_entry.valid_policy_accept
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
            rounds_requested < 4 or eval_audit_counts["accepted_count"] > 0
        ),
        "eval_run_audit_passed": all(eval_run_audit["checks"].values()),
        "trainable_rounds_have_full_eval": eval_run_audit["checks"][
            "trainable_rounds_have_full_eval"
        ],
        "accepted_records_have_required_eval_evidence": eval_run_audit["checks"][
            "accepted_records_have_required_eval_evidence"
        ],
        "decision_consumed_required_gates": eval_run_audit["checks"][
            "decision_consumed_required_gates"
        ],
        "decision_threshold_consumed": eval_run_audit["checks"]["decision_threshold_consumed"],
        "decision_reasons_consistent_with_eval": eval_run_audit["checks"][
            "decision_reasons_consistent_with_eval"
        ],
    }
    return {
        "schema_version": "true_rl_agent.push_recovery.loop_audit.v1",
        "session_id": session_id,
        "session_genesis_hash": session_genesis_hash,
        "mode": "dry_run_audit",
        "output_dir": str(output_dir),
        "ledger_path": str(output_dir / "ledger.jsonl"),
        "rounds_requested": rounds_requested,
        "rounds_completed": len(ledger),
        "prompt_payload_paths": list(prompt_payload_paths),
        "decision_outcome_counts": dict(sorted(outcome_counts.items())),
        "candidate_surface_counts": dict(sorted(surface_counts.items())),
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
        "eval_run_audit": eval_run_audit,
        "checks": checks,
        "violations": {
            "prompt_payloads": [item for item in prompt_checks if not item["public_only"]],
            "curriculum_llm_controlled_rounds": curriculum_violations,
            "train_llm_schedule_rounds": train_schedule_violations,
            "parameter_only_accepted_rounds": parameter_only_accepts,
            "eval_run_audit": eval_run_audit["violations"],
        },
    }


def _inspect_prompt_payload(path: Path) -> dict[str, Any]:
    payload = load_strict_json_object(path, context="prompt payload")
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
