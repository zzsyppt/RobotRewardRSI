from __future__ import annotations

import copy
from typing import Any

from true_rl_agent.evaluation.acceptance_protocol import (
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.protocol import EvaluationProtocol

TEST_ACCEPTANCE_PROTOCOL = AcceptanceProtocol(
    protocol_id="test_acceptance_v1",
    score_source="public_accept.score_stats.mean",
    min_score_count=3,
    absolute_accept_floor=0.05,
    min_delta=0.05,
    public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
)
TEST_EVALUATION_PROTOCOL = EvaluationProtocol(
    eval_protocol_version="test_eval_v1",
    public_seed_count=1,
    public_accept_seed_count=3,
    sealed_validation_seed_count=7,
    eval_num_envs=64,
)
TEST_EVALUATION_BINDING = EvaluationBindingV1(
    eval_protocol_hash=TEST_EVALUATION_PROTOCOL.protocol_hash,
    acceptance_protocol_hash=TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
    task_protocol_hash="b" * 64,
    seed_cohort_hash="c" * 64,
)
TEST_SESSION_GENESIS_HASH = "e" * 64
TEST_SUBJECT_MANIFEST_HASH = "f" * 64
TEST_PROGRAM_SOURCE_CHECKSUM = "sha256:" + "1" * 64
TEST_TRAINING_BINDING_HASH = "2" * 64
TEST_CAPABILITY_PROFILE_HASH = "3" * 64


def _candidate_subject(candidate_id: str) -> dict[str, str]:
    return {
        "schema_version": "true_rl_agent.experiment.subject.v2",
        "kind": "candidate",
        "id": candidate_id,
    }


def _bind_candidate_program_lineage(
    record: dict[str, Any],
    *,
    candidate_id: str,
) -> None:
    subject = _candidate_subject(candidate_id)
    jobs = record.setdefault("jobs", {})
    if isinstance(jobs, dict):
        train_job = jobs.setdefault("train", {})
        if isinstance(train_job, dict):
            train_job["subject"] = subject
            train_job.setdefault(
                "subject_manifest_hash", TEST_SUBJECT_MANIFEST_HASH
            )
            train_job.setdefault(
                "program_source_checksum", TEST_PROGRAM_SOURCE_CHECKSUM
            )
            training_binding = train_job.setdefault("training_binding", {})
            if isinstance(training_binding, dict):
                training_binding.setdefault(
                    "binding_hash", TEST_TRAINING_BINDING_HASH
                )
                training_binding.setdefault(
                    "capability_profile_hash", TEST_CAPABILITY_PROFILE_HASH
                )
        eval_job = jobs.setdefault("eval", {})
        if isinstance(eval_job, dict):
            eval_job["subject"] = subject
            eval_job.setdefault(
                "subject_manifest_hash", TEST_SUBJECT_MANIFEST_HASH
            )
            eval_job.setdefault(
                "program_source_checksum", TEST_PROGRAM_SOURCE_CHECKSUM
            )
            eval_job.setdefault(
                "training_binding_hash", TEST_TRAINING_BINDING_HASH
            )
            eval_job.setdefault(
                "capability_profile_hash", TEST_CAPABILITY_PROFILE_HASH
            )
    public_result = record.get("public_result")
    if not isinstance(public_result, dict) or not public_result:
        return
    metrics = public_result.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        return
    metrics["evaluation_subject"] = subject
    metrics.setdefault("subject_manifest_hash", TEST_SUBJECT_MANIFEST_HASH)
    metrics.setdefault(
        "program_source_checksum", TEST_PROGRAM_SOURCE_CHECKSUM
    )
    metrics.setdefault("training_binding_hash", TEST_TRAINING_BINDING_HASH)
    metrics.setdefault("capability_profile_hash", TEST_CAPABILITY_PROFILE_HASH)
    sealed = metrics.get("sealed_validation")
    if isinstance(sealed, dict):
        sealed["schema_version"] = "true_rl_agent.sealed_validation.evidence.v3"
        sealed["evaluation_subject"] = subject
        sealed.setdefault("subject_manifest_hash", TEST_SUBJECT_MANIFEST_HASH)
        sealed.setdefault(
            "program_source_checksum", TEST_PROGRAM_SOURCE_CHECKSUM
        )
        sealed.setdefault("training_binding_hash", TEST_TRAINING_BINDING_HASH)
        sealed.setdefault(
            "capability_profile_hash", TEST_CAPABILITY_PROFILE_HASH
        )


def make_bootstrap_current_best(
    *,
    initial_best_score: float = 0.0,
) -> CurrentBestEvalStats:
    return CurrentBestEvalStats.bootstrap(
        initial_best_score=initial_best_score,
        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        evaluation_binding=TEST_EVALUATION_BINDING,
        anchor_id="test-bootstrap-anchor",
    )


def bind_test_ledger_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach the test session identity without changing decision evidence."""

    bound = copy.deepcopy(record)
    bound.setdefault("session_genesis_hash", TEST_SESSION_GENESIS_HASH)
    bound.setdefault("evaluation_binding", TEST_EVALUATION_BINDING.to_public_dict())
    public_result = bound.get("public_result")
    if isinstance(public_result, dict) and public_result:
        metrics = public_result.setdefault("metrics", {})
        if isinstance(metrics, dict):
            metrics.setdefault(
                "evaluation_binding",
                TEST_EVALUATION_BINDING.to_public_dict(),
            )
        jobs = bound.setdefault("jobs", {})
        if isinstance(jobs, dict):
            eval_job = jobs.setdefault("eval", {})
            if isinstance(eval_job, dict):
                eval_job.setdefault(
                    "evaluation_binding",
                    TEST_EVALUATION_BINDING.to_public_dict(),
                )
        candidate = bound.get("candidate")
        if isinstance(candidate, dict) and isinstance(
            candidate.get("candidate_id"), str
        ):
            _bind_candidate_program_lineage(
                bound,
                candidate_id=candidate["candidate_id"],
            )
    return bound


def valid_accepted_ledger_record(
    *,
    round_id: str,
    candidate_id: str,
    public_score: float,
    candidate_path: str | None = None,
    initial_best_score: float = 0.0,
) -> dict[str, Any]:
    protocol_hash = TEST_EVALUATION_PROTOCOL.protocol_hash
    checkpoint_hash = "sha256:" + "d" * 64
    sealed_seed_count = 7
    score_stats = {
        "count": 3,
        "mean": public_score,
        "std": 0.0,
        "standard_error": 0.0,
        "eval_protocol_hash": protocol_hash,
    }
    required_gates = [
        "accept_evidence_gate",
        "budget_accounting",
        "sealed_validation_gate",
    ]
    candidate = {
        "candidate_id": candidate_id,
        "changed_components": ["reward_callable"],
    }
    if candidate_path is not None:
        candidate["path"] = candidate_path
    initial_current_best = make_bootstrap_current_best(
        initial_best_score=initial_best_score,
    )
    record = {
        "round_id": round_id,
        "session_genesis_hash": TEST_SESSION_GENESIS_HASH,
        "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
        "candidate": candidate,
        "verifier": {"status": "pass"},
        "candidate_surface": {
            "classification": "code_level",
            "eligible_for_policy_accept": True,
            "reason": "code_level_candidate",
        },
        "jobs": {
            "train": {
                "job_id": f"{round_id}-train",
                "subject": _candidate_subject(candidate_id),
                "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                "training_binding": {
                    "binding_hash": TEST_TRAINING_BINDING_HASH,
                    "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
                },
            },
            "eval": {
                "job_id": f"{round_id}-eval",
                "subject": {
                    **_candidate_subject(candidate_id),
                },
                "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                "training_binding_hash": TEST_TRAINING_BINDING_HASH,
                "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
                **TEST_EVALUATION_PROTOCOL.to_public_dict(),
                "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
                "checkpoint": {"checksum": checkpoint_hash},
            },
        },
        "train_result": {"job_id": f"{round_id}-train"},
        "public_result": {
            "status": "pass",
            "public_score": public_score,
            "metrics": {
                "public_dev": {
                    "score_stats": {
                        "count": 1,
                        "mean": public_score,
                        "std": 0.0,
                        "standard_error": 0.0,
                    }
                },
                "public_accept": {"score_stats": dict(score_stats)},
                "acceptance": {
                    "score_source": TEST_ACCEPTANCE_PROTOCOL.score_source,
                    "public_score_binding": TEST_ACCEPTANCE_PROTOCOL.public_score_binding,
                    "score_stats": dict(score_stats),
                    "min_score_count": TEST_ACCEPTANCE_PROTOCOL.min_score_count,
                    "absolute_accept_floor": TEST_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
                    "min_delta": TEST_ACCEPTANCE_PROTOCOL.min_delta,
                    "acceptance_protocol_hash": TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
                    "required_pass_gates": list(required_gates),
                },
                "accept_evidence_gate": {"status": "pass"},
                "budget_accounting": {"status": "pass"},
                "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
                "evaluation_subject": {
                    **_candidate_subject(candidate_id),
                },
                "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                "training_binding_hash": TEST_TRAINING_BINDING_HASH,
                "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
                "sealed_validation": {
                    "schema_version": "true_rl_agent.sealed_validation.evidence.v3",
                    "status": "pass",
                    "reason": "test_holdout_passed",
                    "expected_seed_count": sealed_seed_count,
                    "observed_seed_count": sealed_seed_count,
                    "eval_protocol_hash": protocol_hash,
                    "evaluation_binding_hash": TEST_EVALUATION_BINDING.binding_hash,
                    "evaluation_subject": {
                        **_candidate_subject(candidate_id),
                    },
                    "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                    "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                    "training_binding_hash": TEST_TRAINING_BINDING_HASH,
                    "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
                    "checkpoint_hash": checkpoint_hash,
                    "score_stats": {
                        "count": sealed_seed_count,
                        "mean": public_score,
                        "std": 0.0,
                        "standard_error": 0.0,
                        "eval_protocol_hash": protocol_hash,
                    },
                },
                "sealed_validation_gate": {
                    "status": "pass",
                    "checkpoint_safety_status": "pass",
                    "candidate_isolation_status": "pass",
                    "candidate_code_loaded_in_eval": False,
                    "candidate_package_loaded_in_eval": False,
                    "candidate_path_argument_present": False,
                    "eval_protocol_hash_match_status": "pass",
                    "evaluation_binding_hash_match_status": "pass",
                    "checkpoint_hash_match_status": "pass",
                    "holdout_rollout_status": "pass",
                    "reason": "sealed_validation_complete",
                },
            },
        },
        "decision": {
            "outcome": "policy_improvement_accept",
            "accepted": True,
            "reason": "absolute_accept_floor",
            "surface_classification": "code_level",
            "public_score": public_score,
            "comparison": {
                "mode": "bootstrap",
                "score_source": TEST_ACCEPTANCE_PROTOCOL.score_source,
                "acceptance_protocol_hash": TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
                "candidate_score": public_score,
                "current_best_score": initial_best_score,
                "delta": public_score - initial_best_score,
                "threshold": TEST_ACCEPTANCE_PROTOCOL.min_delta,
                "delta_met": (
                    public_score - initial_best_score
                    > TEST_ACCEPTANCE_PROTOCOL.min_delta
                ),
                "absolute_accept_floor": TEST_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
                "candidate_score_stats": dict(score_stats),
                "best_score_stats": {},
                "candidate_evaluation_binding_hash": (
                    TEST_EVALUATION_BINDING.binding_hash
                ),
                "current_best": initial_current_best.comparison_identity(),
                "public_score": public_score,
            },
            "consumed_gates": list(required_gates),
            "gate_statuses": {gate: "pass" for gate in required_gates},
        },
    }
    return record


def valid_runtime_failed_ledger_record(
    *,
    round_id: str,
    candidate_id: str,
    candidate_path: str | None = None,
) -> dict[str, Any]:
    candidate = {
        "candidate_id": candidate_id,
        "changed_components": ["reward_callable"],
    }
    if candidate_path is not None:
        candidate["path"] = candidate_path
    return {
        "round_id": round_id,
        "session_genesis_hash": TEST_SESSION_GENESIS_HASH,
        "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
        "candidate": candidate,
        "candidate_surface": {
            "classification": "code_level",
            "eligible_for_policy_accept": True,
            "reason": "code_level_candidate",
        },
        "verifier": {"status": "pass"},
        "jobs": {
            "train": {"job_id": f"{round_id}-train"},
            "eval": {"job_id": f"{round_id}-eval"},
        },
        "runner_error": {
            "type": "CandidateRuntimeError",
            "message": "candidate failed inside runner",
        },
        "decision": {
            "outcome": "reject",
            "accepted": False,
            "reason": "candidate_runtime_failed",
            "surface_classification": "code_level",
            "public_score": None,
            "comparison": {},
            "consumed_gates": [],
            "gate_statuses": {},
        },
    }


__all__ = [
    "TEST_ACCEPTANCE_PROTOCOL",
    "TEST_EVALUATION_BINDING",
    "TEST_EVALUATION_PROTOCOL",
    "TEST_SESSION_GENESIS_HASH",
    "TEST_SUBJECT_MANIFEST_HASH",
    "TEST_PROGRAM_SOURCE_CHECKSUM",
    "TEST_TRAINING_BINDING_HASH",
    "TEST_CAPABILITY_PROFILE_HASH",
    "bind_test_ledger_record",
    "make_bootstrap_current_best",
    "valid_accepted_ledger_record",
    "valid_runtime_failed_ledger_record",
]
