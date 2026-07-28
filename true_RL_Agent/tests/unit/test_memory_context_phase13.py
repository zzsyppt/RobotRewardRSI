from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_CAPABILITY_PROFILE_HASH,
    TEST_EVALUATION_BINDING,
    TEST_EVALUATION_PROTOCOL,
    TEST_PROGRAM_SOURCE_CHECKSUM,
    TEST_SESSION_GENESIS_HASH,
    TEST_SUBJECT_MANIFEST_HASH,
    TEST_TRAINING_BINDING_HASH,
    bind_test_ledger_record,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
    valid_runtime_failed_ledger_record,
)
from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload
from true_rl_agent.evaluation.acceptance_replay import project_public_history
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.memory import (
    ExperimentRecord,
    PrivateArtifactRef,
    TaskMemoryAdapter,
)
from true_rl_agent.memory import (
    PublicContextBuilder as _PublicContextBuilder,
)
from true_rl_agent.prompting.payload import (
    build_prompt_payload as assemble_prompt_payload,
)
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.tasks.push_recovery import PushTaskMemoryAdapter


class _BoundPublicContextBuilder(_PublicContextBuilder):
    def _prepare_records(self, records):
        bound_records = []
        for record in records:
            if isinstance(record, ExperimentRecord):
                bound_records.append(
                    replace(
                        record,
                        public_record=bind_test_ledger_record(
                            record.ledger_public_record()
                        ),
                    )
                )
            else:
                bound_records.append(bind_test_ledger_record(dict(record)))
        return super()._prepare_records(bound_records)


def PublicContextBuilder(*args, **kwargs):
    kwargs.setdefault("acceptance_protocol", TEST_ACCEPTANCE_PROTOCOL)
    kwargs.setdefault("initial_current_best", make_bootstrap_current_best())
    kwargs.setdefault("session_genesis_hash", TEST_SESSION_GENESIS_HASH)
    return _BoundPublicContextBuilder(*args, **kwargs)


class MemoryContextPhase13Tests(unittest.TestCase):
    def test_private_artifact_rejects_malformed_checksum(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            PrivateArtifactRef(
                kind="trusted_eval_report",
                uri="artifact://private/trusted-eval.json",
                checksum="sha256:not-a-digest",
            )

    def _push_record(
        self,
        *,
        round_id: str,
        score: float,
        failure_breakdown: dict[str, float],
        current_best_score: float | None = None,
        current_best_round_id: str | None = None,
    ) -> dict[str, object]:
        protocol_hash = TEST_EVALUATION_PROTOCOL.protocol_hash
        score_stats = {
            "count": TEST_ACCEPTANCE_PROTOCOL.min_score_count,
            "mean": score,
            "std": 0.0,
            "standard_error": 0.0,
            "eval_protocol_hash": protocol_hash,
        }
        required_gates = [
            "accept_evidence_gate",
            "budget_accounting",
            "sealed_validation_gate",
        ]
        if current_best_score is None:
            current_best = make_bootstrap_current_best()
        else:
            previous = self._push_record(
                round_id=str(current_best_round_id),
                score=current_best_score,
                failure_breakdown={},
            )
            current_best = CurrentBestEvalStats.from_public_result_mapping(
                round_id=str(current_best_round_id),
                candidate_id=f"candidate-{current_best_round_id}",
                public_result=previous["public_result"],
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                subject_manifest_hash=previous["jobs"]["eval"][
                    "subject_manifest_hash"
                ],
                program_source_checksum=previous["jobs"]["eval"][
                    "program_source_checksum"
                ],
                training_binding_hash=previous["jobs"]["eval"][
                    "training_binding_hash"
                ],
                capability_profile_hash=previous["jobs"]["eval"][
                    "capability_profile_hash"
                ],
                checkpoint_hash=previous["jobs"]["eval"]["checkpoint"][
                    "checksum"
                ],
                expected_evaluation_binding=TEST_EVALUATION_BINDING,
            )
        return {
            "round_id": round_id,
            "session_genesis_hash": TEST_SESSION_GENESIS_HASH,
            "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
            "task": {"task_id": "unitree_push_recovery"},
            "candidate": {
                "candidate_id": f"candidate-{round_id}",
                "changed_components": ["reward_callable"],
            },
            "verifier": {"status": "pass"},
            "jobs": {
                "train": {
                    "job_id": f"{round_id}-train",
                    "subject": {
                        "schema_version": "true_rl_agent.experiment.subject.v2",
                        "kind": "candidate",
                        "id": f"candidate-{round_id}",
                    },
                    "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                    "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                    "training_binding": {
                        "binding_hash": TEST_TRAINING_BINDING_HASH,
                        "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
                    },
                },
                "eval": {
                    "job_id": f"{round_id}-eval",
                    **TEST_EVALUATION_PROTOCOL.to_public_dict(),
                    "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
                    "subject": {
                        "schema_version": "true_rl_agent.experiment.subject.v2",
                        "kind": "candidate",
                        "id": f"candidate-{round_id}",
                    },
                    "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                    "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                    "training_binding_hash": TEST_TRAINING_BINDING_HASH,
                    "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
                    "checkpoint": {
                        "checksum": "sha256:" + "a" * 64
                    },
                },
            },
            "train_result": {"job_id": f"{round_id}-train"},
            "decision": {
                "accepted": True,
                "outcome": "policy_improvement_accept",
                "reason": (
                    "absolute_accept_floor"
                    if current_best_score is None
                    else "noise_aware_delta"
                ),
                "surface_classification": "code_level",
                "public_score": score,
                "comparison": {
                    "mode": "bootstrap" if current_best_score is None else "accepted",
                    "score_source": TEST_ACCEPTANCE_PROTOCOL.score_source,
                    "acceptance_protocol_hash": TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
                    "candidate_score": score,
                    "current_best_score": current_best_score or 0.0,
                    "delta": score - (current_best_score or 0.0),
                    "threshold": TEST_ACCEPTANCE_PROTOCOL.min_delta,
                    "delta_met": (
                        score - (current_best_score or 0.0)
                        > TEST_ACCEPTANCE_PROTOCOL.min_delta
                    ),
                    "absolute_accept_floor": TEST_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
                    "candidate_score_stats": dict(score_stats),
                    "best_score_stats": (
                        {}
                        if current_best_score is None
                        else {
                            "count": TEST_ACCEPTANCE_PROTOCOL.min_score_count,
                            "mean": current_best_score,
                            "std": 0.0,
                            "standard_error": 0.0,
                            "eval_protocol_hash": protocol_hash,
                        }
                    ),
                    "candidate_evaluation_binding_hash": (
                        TEST_EVALUATION_BINDING.binding_hash
                    ),
                    "current_best": current_best.comparison_identity(),
                    "public_score": score,
                },
                "consumed_gates": list(required_gates),
                "gate_statuses": {gate: "pass" for gate in required_gates},
            },
            "candidate_surface": {
                "classification": "code_level",
                "eligible_for_policy_accept": True,
                "reason": "code_level_candidate",
            },
            "public_result": {
                "status": "pass",
                "public_score": score,
                "metrics": {
                    "evaluation_binding": TEST_EVALUATION_BINDING.to_public_dict(),
                    "evaluation_subject": {
                        "schema_version": "true_rl_agent.experiment.subject.v2",
                        "kind": "candidate",
                        "id": f"candidate-{round_id}",
                    },
                    "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                    "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                    "training_binding_hash": TEST_TRAINING_BINDING_HASH,
                    "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
                    "success_rate": score,
                    "failure_breakdown": failure_breakdown,
                    "motion_quality": {"mean_base_height": 0.79, "mean_tilt_rad": 0.04},
                    "trusted_seed_count": 17,
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
                    "sealed_validation": {
                        "schema_version": "true_rl_agent.sealed_validation.evidence.v3",
                        "status": "pass",
                        "reason": "test_holdout_passed",
                        "expected_seed_count": 7,
                        "observed_seed_count": 7,
                        "eval_protocol_hash": protocol_hash,
                        "evaluation_binding_hash": (
                            TEST_EVALUATION_BINDING.binding_hash
                        ),
                        "evaluation_subject": {
                            "schema_version": "true_rl_agent.experiment.subject.v2",
                            "kind": "candidate",
                            "id": f"candidate-{round_id}",
                        },
                        "subject_manifest_hash": TEST_SUBJECT_MANIFEST_HASH,
                        "program_source_checksum": TEST_PROGRAM_SOURCE_CHECKSUM,
                        "training_binding_hash": TEST_TRAINING_BINDING_HASH,
                        "capability_profile_hash": TEST_CAPABILITY_PROFILE_HASH,
                        "checkpoint_hash": "sha256:" + "a" * 64,
                        "score_stats": {
                            "count": 7,
                            "mean": score,
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
            "checkpoint_selector": "best_trusted_checkpoint",
            "hidden_eval": {"success_rate": 0.99},
            "private_seed": 12345,
            "trusted_eval": {"seed": 67890},
            "note": "sk-secret-should-redact",
        }

    def test_experiment_record_projects_public_data_and_hides_private_artifacts(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            self._push_record(round_id="round-001", score=0.42, failure_breakdown={"fall": 0.2}),
            private_artifacts=(
                PrivateArtifactRef(
                    kind="trusted_eval_report",
                    uri="artifact://private/trusted-eval.json",
                    checksum="sha256:" + "a" * 64,
                    reason="private decision evidence",
                ),
            ),
        )

        public_record = record.to_public_dict()
        public_text = json.dumps(public_record, sort_keys=True)
        private_text = json.dumps(record.to_private_dict(), sort_keys=True)

        self.assertEqual(public_record["round_id"], "round-001")
        self.assertEqual(public_record["non_prompt_artifact_count"], 1)
        self.assertNotIn("private_seed", public_text)
        self.assertNotIn("hidden_eval", public_text)
        self.assertNotIn("trusted_eval", public_text)
        self.assertNotIn("checkpoint_selector", public_text)
        self.assertNotIn("artifact://private", public_text)
        self.assertIn("artifact://private/trusted-eval.json", private_text)

    def test_invalid_declared_accept_does_not_raise_best_public_score(self) -> None:
        valid = valid_accepted_ledger_record(
            round_id="round-valid",
            candidate_id="candidate-valid",
            public_score=0.25,
        )
        invalid = copy.deepcopy(valid)
        invalid["round_id"] = "round-invalid-accept"
        invalid["decision"].update(
            accepted=True,
            outcome="reject",
            reason="candidate_verifier_failed",
            public_score=0.99,
        )

        context = PublicContextBuilder().build([valid, invalid])

        self.assertEqual(context["best_public_score"], 0.25)

    def test_unbound_public_score_does_not_enter_next_round_history(self) -> None:
        valid = valid_accepted_ledger_record(
            round_id="round-valid-score",
            candidate_id="candidate-valid-score",
            public_score=0.25,
        )
        forged = copy.deepcopy(valid)
        forged["round_id"] = "round-forged-score"
        forged["public_result"]["public_score"] = 0.99
        forged["decision"]["public_score"] = 0.99
        forged["decision"]["comparison"]["public_score"] = 0.99

        builder = PublicContextBuilder()
        context = builder.build([valid, forged])
        history = builder.build_prompt_payload([valid, forged])["public_history"][
            "recent_outcomes"
        ][-1]

        self.assertEqual(context["best_public_score"], 0.25)
        self.assertIsNone(history.get("public_score"))
        self.assertNotIn("0.99", json.dumps(history, sort_keys=True))

    def test_invalid_declared_accept_is_fail_closed_in_public_history(self) -> None:
        record = valid_accepted_ledger_record(
            round_id="round-invalid-history",
            candidate_id="candidate-invalid-history",
            public_score=0.50,
        )
        record["verifier"] = {"status": "fail"}
        record["decision"].update(
            accepted=True,
            outcome="policy_improvement_accept",
            reason="candidate_verifier_failed",
        )

        public_history = PublicContextBuilder().build_prompt_payload([record])[
            "public_history"
        ]
        history = public_history["recent_outcomes"][-1]
        experiment = public_history["experiment_memory"]["experiments"][-1]

        self.assertEqual(history["outcome"], "reject")
        self.assertFalse(history["accepted"])
        self.assertEqual(history["reason"], "decision_record_invalid")
        self.assertNotIn("public_score", history)
        self.assertFalse(experiment["accepted"])
        self.assertEqual(experiment["reason"], "decision_record_invalid")
        self.assertNotIn("score", experiment)

    def test_public_context_builder_saves_public_only_prompt_payload(self) -> None:
        records = [
            ExperimentRecord.from_ledger_record(
                self._push_record(round_id="round-001", score=0.30, failure_breakdown={"fall": 0.3})
            ),
            ExperimentRecord.from_ledger_record(
                self._push_record(
                    round_id="round-002",
                    score=0.46,
                    failure_breakdown={"push_drift": 0.4},
                    current_best_score=0.30,
                    current_best_round_id="round-001",
                )
            ),
        ]
        builder = PublicContextBuilder(
            task_memory_adapter=PushTaskMemoryAdapter()
        )

        prompt_build = builder.build_prompt(records)
        round_context = prompt_build.round_public_context
        payload = prompt_build.provider_payload
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertEqual(round_context["previous_rounds"], 2)
        self.assertEqual(round_context["best_public_score"], 0.46)
        self.assertEqual(
            round_context["task_public_summary"]["latest_failure_mode"],
            "push_recovery_drift",
        )
        self.assertEqual(round_context["task_public_summary"]["success_rate"]["best"], 0.46)
        self.assertEqual(
            payload["public_feedback_summary"]["task_public_summary"],
            round_context["task_public_summary"],
        )
        self.assertNotIn("recent_outcomes", payload["public_context"])
        self.assertNotIn("task_public_summary", payload["public_context"])
        self.assertNotIn("source_catalog", round_context)
        self.assertNotIn("task_prompt_sections", round_context)
        self.assertNotIn("private_seed", payload_text)
        self.assertNotIn("hidden_eval", payload_text)
        self.assertNotIn("trusted_eval", payload_text)
        self.assertNotIn("checkpoint_selector", payload_text)
        self.assertNotIn("trusted_seed_count", payload_text)
        self.assertNotIn("sk-secret", payload_text)

        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "round-003" / "prompt_payload.json"
            ref = builder.save_prompt_payload(prompt_path, payload)
            saved = json.loads(prompt_path.read_text(encoding="utf-8"))
            saved_text = prompt_path.read_text(encoding="utf-8")

        self.assertEqual(saved, payload)
        self.assertTrue(saved_text.startswith('{\n  "task":'), saved_text[:80])
        self.assertEqual(ref.visibility, "public_prompt_payload")
        self.assertEqual(ref.to_public_dict()["path"], str(prompt_path))

    def test_push_memory_uses_validated_public_score_not_raw_metric_mirror(self) -> None:
        record = self._push_record(
            round_id="round-score-binding",
            score=0.42,
            failure_breakdown={"push_drift": 0.4},
        )
        record["public_result"]["metrics"]["success_rate"] = 0.99

        history = project_public_history(
            [record],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        summary = PushTaskMemoryAdapter().build_public_summary(history.records)

        self.assertEqual(
            summary["success_rate"],
            {"latest": 0.42, "best": 0.42, "mean": 0.42},
        )
        self.assertNotIn("0.99", json.dumps(summary, sort_keys=True))

    def test_task_memory_adapter_is_extension_point_without_agent_core_changes(self) -> None:
        class CustomTaskMemoryAdapter(TaskMemoryAdapter):
            task_id = "custom_task"

            def build_public_summary(self, public_records):
                seen_counts.append(len(public_records))
                return {"schema_version": "custom.public_memory.v1", "custom_metric": 7}

        seen_counts: list[int] = []
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "round-001",
                "task": {"task_id": "custom_task"},
                "decision": {"accepted": False, "outcome": "reject", "public_score": None},
                "candidate_surface": {"classification": "code_level"},
            }
        )

        context = PublicContextBuilder(task_memory_adapter=CustomTaskMemoryAdapter()).build([record])

        self.assertEqual(context["task_public_summary"]["custom_metric"], 7)
        self.assertEqual(seen_counts, [1])

    def test_public_context_builder_accepts_raw_mapping_records(self) -> None:
        record = valid_runtime_failed_ledger_record(
            round_id="round-raw-mapping",
            candidate_id="candidate-raw-mapping",
        )
        record["runner_error"] = {
            "type": "CandidateRuntimeError",
            "message": "failed at D:/humanoid_agent/private/raw.txt",
        }
        record["raw_seed_id"] = "public-dev-seed-001"
        payload = PublicContextBuilder().build_prompt_payload(
            [record]
        )
        payload_text = json.dumps(payload, sort_keys=True)
        entry = payload["public_history"]["recent_outcomes"][-1]

        self.assertEqual(entry["round_id"], "round-raw-mapping")
        self.assertEqual(entry["reason"], "candidate_runtime_failed")
        self.assertIn("[path]", entry["runner_error"]["message"])
        self.assertNotIn("D:/humanoid_agent", payload_text)
        self.assertNotIn("raw_seed_id", payload_text)
        self.assertNotIn("public-dev-seed-001", payload_text)

    def test_prompt_payload_guard_blocks_checkpoint_selector_keys(self) -> None:
        with self.assertRaises(ValueError):
            assert_public_only_prompt_payload({"public_context": {"checkpoint_selector": "best_trusted"}})

    def test_push_memory_adapter_tolerates_malformed_public_task_field(self) -> None:
        summary = PushTaskMemoryAdapter().build_public_summary(
            [
                {
                    "task": "malformed",
                    "public_result": {
                        "metrics": {
                            "success_rate": 0.25,
                            "failure_breakdown": {"timeout": 0.2},
                            "motion_quality": {"mean_base_height": 0.76},
                        }
                    },
                }
            ]
        )

        self.assertEqual(summary["latest_failure_mode"], "timeout_dominated")

    def test_prompt_public_history_projects_runtime_and_error_evidence(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "round-010",
                "task": {"task_id": "custom_task"},
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "eval_incomplete",
                    "surface_classification": "code_level",
                    "public_score": None,
                    "comparison": {},
                    "consumed_gates": [],
                    "gate_statuses": {},
                },
                "candidate": {
                    "candidate_id": "candidate-010",
                    "path": "D:/humanoid_agent/private/candidate-010",
                    "changed_components": ["diagnostic", "reward_callable"],
                },
                "candidate_surface": {"classification": "code_level"},
                "verifier": {"status": "pass"},
                "jobs": {
                    "train": {"job_id": "round-010-train"},
                    "eval": {"job_id": "round-010-eval"},
                },
                "runtime_activation_summary": {
                    "reward_term_calls": 3,
                    "diagnostic_calls": 1,
                    "nonzero_reward_contribution": False,
                    "reward_output_stats": {"mean": 0.0, "finite_fraction": 1.0},
                    "capability_activation_records": ["reward_callable", "diagnostic"],
                },
                "mechanism_attribution": {
                    "status": "attribution_unclear",
                    "reason": "runner_activation_missing",
                },
                "train_result": {
                    "summary": {
                        "candidate": {"enabled": True},
                        "candidate_code_loaded_in_eval": False,
                    }
                },
                "runner_error": {
                    "type": "CandidateRuntimeError",
                    "message": "failed at /data/yuyangliu/secrets/raw.txt and D:/humanoid_agent/private/raw.txt",
                    "details": {"status": "fail"},
                },
            }
        )

        payload = PublicContextBuilder().build_prompt_payload([record])
        history = payload["public_history"]
        entry = history["recent_outcomes"][-1]
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertEqual(entry["round_id"], "round-010")
        self.assertEqual(entry["reason"], "eval_incomplete")
        self.assertEqual(entry["surface_classification"], "code_level")
        self.assertNotIn("public_score", entry)
        self.assertEqual(entry["changed_components"], ["diagnostic", "reward_callable"])
        self.assertIs(entry["candidate_package_loaded"], True)
        self.assertIs(entry["candidate_code_loaded_in_eval"], False)
        self.assertEqual(entry["runtime_activation"]["reward_term_calls"], 3)
        self.assertEqual(entry["runtime_activation"]["diagnostic_calls"], 1)
        self.assertIs(entry["runtime_activation"]["nonzero_reward_contribution"], False)
        self.assertEqual(entry["mechanism_attribution"]["status"], "attribution_unclear")
        self.assertEqual(entry["runner_error"]["type"], "CandidateRuntimeError")
        self.assertIn("[path]", entry["runner_error"]["message"])
        self.assertNotIn("/data/yuyangliu", payload_text)
        self.assertNotIn("D:/humanoid_agent", payload_text)

    def test_prompt_public_history_preserves_missing_runtime_activation_as_not_available(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "round-no-runtime-summary",
                "task": {"task_id": "custom_task"},
                "decision": {
                    "accepted": True,
                    "outcome": "policy_improvement_accept",
                    "reason": "policy_improvement_accept",
                    "surface_classification": "code_level",
                    "public_score": 0.42,
                },
                "candidate": {
                    "candidate_id": "candidate-no-runtime-summary",
                    "changed_components": ["reward_callable"],
                },
                "candidate_surface": {"classification": "code_level"},
            }
        )

        payload = PublicContextBuilder().build_prompt_payload([record])
        entry = payload["public_history"]["recent_outcomes"][-1]

        self.assertEqual(entry["runtime_activation"], {"status": "not_available"})

    def test_prompt_public_history_does_not_treat_checkpoint_eval_as_training_load_failure(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "round-010b",
                "task": {"task_id": "custom_task"},
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "required_gate_failed",
                    "surface_classification": "code_level",
                    "public_score": 0.25,
                },
                "candidate": {
                    "candidate_id": "candidate-010b",
                    "changed_components": ["diagnostic", "reward_callable"],
                },
                "candidate_surface": {"classification": "code_level"},
                "runtime_activation_summary": {
                    "reward_term_calls": 12000,
                    "diagnostic_calls": 500,
                    "nonzero_reward_contribution": True,
                    "capability_activation_records": ["reward_callable", "diagnostic"],
                },
                "train_result": {
                    "summary": {
                        "candidate": {"enabled": True},
                    }
                },
                "public_result": {
                    "status": "pass",
                    "public_score": 0.25,
                    "metrics": {
                        "candidate_package_loaded": False,
                        "candidate_code_loaded_in_eval": False,
                    },
                },
            }
        )

        payload = PublicContextBuilder().build_prompt_payload([record])
        entry = payload["public_history"]["recent_outcomes"][-1]

        self.assertIs(entry["candidate_package_loaded"], True)
        self.assertIs(entry["candidate_code_loaded_in_eval"], False)

    def test_prompt_public_history_projects_provider_repair_failure(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "round-011",
                "task": {"task_id": "custom_task"},
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "provider_candidate_generation_failed",
                    "surface_classification": "provider_failure",
                    "public_score": None,
                },
                "candidate_surface": {"classification": "provider_failure"},
                "provider_failure": {
                    "category": "json_parse_error",
                    "reason": "invalid JSON from D:/humanoid_agent/private/provider_response.txt",
                },
                "public_context": {
                    "candidate_repair_feedback": {
                        "failure_category": "json_parse_error",
                        "reason": "bad JSON near /data/yuyangliu/secrets/key.env",
                    }
                },
            }
        )

        payload = PublicContextBuilder().build_prompt_payload([record])
        entry = payload["public_history"]["recent_outcomes"][-1]
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertEqual(entry["provider_failure"]["category"], "json_parse_error")
        self.assertIn("[path]", entry["provider_failure"]["reason"])
        self.assertEqual(entry["repair_feedback"]["failure_category"], "json_parse_error")
        self.assertIn("[path]", entry["repair_feedback"]["reason"])
        self.assertNotIn("/data/yuyangliu", payload_text)
        self.assertNotIn("D:/humanoid_agent", payload_text)

    def test_prompt_public_history_projects_bounded_source_request_feedback(self) -> None:
        payload = PublicContextBuilder().build_prompt_payload(
            [
                {
                    "round_id": "round-source-denied",
                    "decision": {
                        "accepted": False,
                        "outcome": "reject",
                        "reason": "provider_candidate_generation_failed",
                        "surface_classification": "provider_failure",
                        "public_score": None,
                    },
                    "candidate_surface": {"classification": "provider_failure"},
                    "source_request_feedback": {
                        "schema_version": "true_rl_agent.source_request_round_feedback.v1",
                        "status": "source_request_denied",
                        "request_count": 1,
                        "returned_logical_ids": [],
                        "denied_logical_ids": [
                            {
                                "logical_id": "current_best/training_code/components.py",
                                "reason": "source_not_requestable",
                            }
                        ],
                        "budget": {
                            "source_request_count": 1,
                            "max_source_requests_per_round": 2,
                            "max_source_ids_per_request": 2,
                            "cumulative_returned_bytes": 0,
                            "max_source_request_bytes_per_round": 49152,
                            "raw_note": "must_not_leak",
                        },
                        "omitted_returned_logical_id_count": 0,
                        "omitted_denied_logical_id_count": 0,
                        "source_request": {"rationale": "must_not_leak"},
                        "instructions": ["must_not_leak"],
                        "content": "sk-secret D:/private/source.py",
                    },
                }
            ]
        )

        feedback = payload["public_history"]["recent_outcomes"][-1][
            "source_request_feedback"
        ]
        feedback_text = json.dumps(feedback, sort_keys=True)

        self.assertEqual(feedback["status"], "source_request_denied")
        self.assertEqual(
            feedback["denied_logical_ids"],
            [
                {
                    "logical_id": "current_best/training_code/components.py",
                    "reason": "source_not_requestable",
                }
            ],
        )
        self.assertEqual(
            set(feedback["budget"]),
            {
                "source_request_count",
                "max_source_requests_per_round",
                "max_source_ids_per_request",
                "cumulative_returned_bytes",
                "max_source_request_bytes_per_round",
            },
        )
        for forbidden in ("must_not_leak", "rationale", "instructions", "content", "D:/private"):
            self.assertNotIn(forbidden, feedback_text)

    def test_prompt_public_history_preserves_omitted_detail_count_across_recompaction(self) -> None:
        raw_payload = PublicContextBuilder().build_prompt_payload(
            [
                {
                    "round_id": "round-detail-omission",
                    "decision": {
                        "accepted": False,
                        "outcome": "reject",
                        "reason": "candidate_runtime_failed",
                        "surface_classification": "code_level",
                        "public_score": 0.0,
                    },
                    "candidate_surface": {"classification": "code_level"},
                    "runner_error": {
                        "type": "CandidateRuntimeError",
                        "message": "public runtime failure",
                        "details": {
                            "status": "fail",
                            "metric_a": 1,
                            "metric_b": 2,
                            "metric_c": 3,
                            "metric_d": 4,
                            "metric_e": 5,
                        },
                    },
                }
            ]
        )
        compacted_payload = assemble_prompt_payload(
            public_context={
                "previous_rounds": 1,
                "recent_outcomes": [
                    {
                        "round_id": "round-already-compacted",
                        "outcome": "reject",
                        "runner_error": {
                            "type": "CandidateRuntimeError",
                            "message": "public runtime failure",
                            "details": {"status": "fail", "omitted_detail_count": 5},
                        },
                    }
                ],
            }
        )

        self.assertEqual(
            raw_payload["public_history"]["recent_outcomes"][-1]["runner_error"]["details"][
                "omitted_detail_count"
            ],
            5,
        )
        self.assertEqual(
            compacted_payload["public_history"]["recent_outcomes"][-1]["runner_error"][
                "details"
            ]["omitted_detail_count"],
            5,
        )

    def test_static_authoring_failure_feedback_overrides_stale_attempt_repair_feedback(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "round-012",
                "task": {"task_id": "custom_task"},
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "candidate_static_authoring_failed",
                    "surface_classification": "static_authoring_failure",
                    "public_score": None,
                },
                "candidate_surface": {"classification": "static_authoring_failure"},
                "candidate": {
                    "candidate_id": "candidate-final-static-fail",
                    "changed_components": ["reward_callable"],
                },
                "public_context": {
                    "candidate_repair_feedback": {
                        "failure_category": "json_parse_error",
                        "reason": "earlier provider attempt failed JSON",
                    }
                },
                "static_authoring_failure": {
                    "schema_version": "true_rl_agent.static_authoring_failure.v1",
                    "candidate_id": "candidate-final-static-fail",
                    "candidate_changed_components": ["reward_callable"],
                    "failure_category": "component_smoke_error",
                    "feedback": {
                        "schema_version": "true_rl_agent.provider_repair_feedback.v1",
                        "status": "repair_required",
                        "failure_category": "component_smoke_error",
                        "reason": "final candidate failed component smoke",
                    },
                },
            }
        )

        payload = PublicContextBuilder().build_prompt_payload([record])
        entry = payload["public_history"]["recent_outcomes"][-1]

        self.assertEqual(entry["repair_feedback"]["failure_category"], "component_smoke_error")
        self.assertEqual(entry["repair_feedback"]["status"], "repair_required")
        self.assertNotIn("json_parse_error", json.dumps(entry["repair_feedback"], sort_keys=True))

    def test_prompt_public_history_is_count_and_byte_bounded(self) -> None:
        records = [
            ExperimentRecord.from_ledger_record(
                {
                    "round_id": f"round-{index:03d}",
                    "decision": {
                        "accepted": False,
                        "outcome": "reject",
                        "reason": "candidate_runtime_failed",
                        "surface_classification": "code_level",
                        "public_score": 0.0,
                    },
                    "candidate_surface": {"classification": "code_level"},
                    "runner_error": {
                        "type": "CandidateRuntimeError",
                        "message": "shape mismatch " + ("x" * 2000),
                    },
                }
            )
            for index in range(12)
        ]

        payload = PublicContextBuilder().build_prompt_payload(records)
        history = payload["public_history"]
        recent = history["recent_outcomes"]

        self.assertLessEqual(len(recent), history["limits"]["max_recent_rounds"])
        self.assertLessEqual(
            canonical_prompt_json_bytes(recent),
            history["limits"]["max_serialized_bytes"],
        )
        self.assertEqual(recent[-1]["round_id"], "round-011")
        self.assertGreater(history["omitted_round_count"], 0)

    def test_prompt_experiment_memory_retains_bounded_component_lessons(self) -> None:
        records = [
            ExperimentRecord.from_ledger_record(
                {
                    "round_id": f"round-{index:03d}",
                    "candidate": {
                        "candidate_id": f"candidate-{index:03d}",
                        "changed_components": ["reward_callable"],
                    },
                    "decision": {
                        "accepted": False,
                        "outcome": "reject",
                        "reason": "candidate_runtime_failed",
                        "surface_classification": "code_level",
                        "public_score": 0.01 * index,
                    },
                    "candidate_surface": {"classification": "code_level"},
                    "runtime_activation_summary": {
                        "component_activation_records": [
                            {
                                "component_kind": "reward_term",
                                "name": f"term_{index:03d}_{term_index}",
                                "effective_weight": 0.1,
                                "last_weighted_mean": -0.02,
                                "max_abs_weighted_contribution": 0.5,
                                "raw_seed": "must_not_leak",
                            }
                            for term_index in range(4)
                        ]
                    },
                    "runner_error": {
                        "type": "CandidateRuntimeError",
                        "message": "shape mismatch " + ("x" * 2000),
                    },
                }
            )
            for index in range(15)
        ]

        payload = PublicContextBuilder().build_prompt_payload(records)
        memory = payload["public_history"]["experiment_memory"]
        experiments = memory["experiments"]
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertEqual(len(experiments), 15)
        self.assertEqual(experiments[0]["candidate_id"], "candidate-000")
        self.assertEqual(experiments[-1]["candidate_id"], "candidate-014")
        self.assertEqual(
            experiments[0]["activated_components"][0],
            {
                "kind": "reward_term",
                "name": "term_000_0",
                "weight": 0.1,
                "last_mean": -0.02,
                "max_abs": 0.5,
            },
        )
        self.assertLessEqual(len(experiments), memory["limits"]["max_rounds"])
        self.assertLessEqual(
            canonical_prompt_json_bytes(experiments),
            memory["limits"]["max_serialized_bytes"],
        )
        self.assertNotIn("raw_seed", payload_text)
        self.assertNotIn("must_not_leak", payload_text)

    def test_direct_prompt_payload_public_history_uses_fixed_allowlist(self) -> None:
        payload = assemble_prompt_payload(
            public_context={
                "previous_rounds": 1,
                "recent_outcomes": [
                    {
                        "round_id": "round-tainted-history",
                        "outcome": "reject",
                        "reason": "candidate_runtime_failed",
                        "surface_classification": "code_level",
                        "public_score": 0.0,
                        "changed_components": ["reward_callable"],
                        "push_active": "must_not_leak",
                        "push_vec_public": "must_not_leak",
                        "raw_seed": "must_not_leak",
                        "runner_error": {
                            "type": "CandidateRuntimeError",
                            "message": "shape mismatch raw_seed=1101 push_active",
                            "details": {
                                "status": "fail",
                                "fall_forward_rate": 0.5,
                                "dominant_failure_label": "fall_forward",
                                "task_specific_metric": 7,
                                "push_active": "must_not_leak",
                                "raw_seed": "must_not_leak",
                            },
                        },
                    }
                ],
            }
        )

        payload_text = json.dumps(payload, sort_keys=True)
        entry = payload["public_history"]["recent_outcomes"][-1]

        self.assertEqual(entry["round_id"], "round-tainted-history")
        self.assertEqual(entry["changed_components"], ["reward_callable"])
        self.assertNotIn("push_active", entry)
        self.assertNotIn("push_vec_public", entry)
        self.assertNotIn("raw_seed", entry)
        self.assertNotIn("push_active", payload_text)
        self.assertNotIn("push_vec_public", payload_text)
        self.assertNotIn("raw_seed", payload_text)
        self.assertNotIn("fall_forward", payload_text)
        self.assertNotIn("task_specific_metric", payload_text)
        self.assertNotIn("must_not_leak", payload_text)
        self.assertEqual(entry["runner_error"]["details"]["status"], "fail")
        self.assertGreater(entry["runner_error"]["details"]["omitted_detail_count"], 0)

    def test_prompt_public_history_keeps_failed_gates_without_task_metric_semantics(self) -> None:
        record = valid_accepted_ledger_record(
            round_id="round-behavior-gate",
            candidate_id="candidate-behavior-gate",
            public_score=0.25,
        )
        required = record["public_result"]["metrics"]["acceptance"][
            "required_pass_gates"
        ]
        required.append("behavior_quality_gate")
        record["public_result"]["metrics"]["behavior_quality_gate"] = {
            "status": "fail",
            "failures": [
                {
                    "metric": "mean_min_base_height_m",
                    "reason": "missing_or_nonfinite",
                }
            ],
        }
        record["decision"].update(
            outcome="reject",
            accepted=False,
            reason="required_gate_failed",
        )
        record["decision"]["consumed_gates"].append("behavior_quality_gate")
        record["decision"]["gate_statuses"]["behavior_quality_gate"] = "fail"

        payload = PublicContextBuilder().build_prompt_payload([record])

        entry = payload["public_history"]["recent_outcomes"][-1]
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertEqual(
            entry["failed_gates"],
            [{"name": "behavior_quality_gate", "status": "fail"}],
        )
        self.assertIn("behavior_quality_gate", payload_text)
        self.assertNotIn("mean_min_base_height_m", payload_text)
        self.assertNotIn("fall_forward", payload_text)

    def test_prompt_public_history_keeps_bounded_eval_feedback_summaries(self) -> None:
        best_record = valid_accepted_ledger_record(
            round_id="round-best",
            candidate_id="candidate-best",
            public_score=0.40,
        )
        record = valid_accepted_ledger_record(
            round_id="round-eval-feedback",
            candidate_id="candidate-eval-feedback",
            public_score=0.42,
        )
        metrics = record["public_result"]["metrics"]
        required = metrics["acceptance"]["required_pass_gates"]
        required.append("behavior_quality_gate")
        metrics["behavior_quality_gate"] = {"status": "fail"}
        metrics["prompt_feedback"] = {
            "dominant_failure": {
                "status": "available",
                "source_split": "public_dev",
                "label": "fall_forward",
                "rate": 0.58,
                "seed_id": "must_not_leak",
            }
        }
        metrics["sealed_validation_gate"]["reason"] = "sealed_validation_pass"
        metrics["sealed_validation_gate"][
            "checkpoint_report_path"
        ] = "D:/runner/eval/report.json"
        candidate_stats = dict(metrics["acceptance"]["score_stats"])
        best_stats = CurrentBestEvalStats.from_public_result_mapping(
            round_id="round-best",
            candidate_id="candidate-best",
            public_result=best_record["public_result"],
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            subject_manifest_hash=best_record["jobs"]["eval"][
                "subject_manifest_hash"
            ],
            program_source_checksum=best_record["jobs"]["eval"][
                "program_source_checksum"
            ],
            training_binding_hash=best_record["jobs"]["eval"][
                "training_binding_hash"
            ],
            capability_profile_hash=best_record["jobs"]["eval"][
                "capability_profile_hash"
            ],
            checkpoint_hash=best_record["jobs"]["eval"]["checkpoint"][
                "checksum"
            ],
            expected_evaluation_binding=TEST_EVALUATION_BINDING,
        )
        record["decision"].update(
            outcome="reject",
            accepted=False,
            reason="required_gate_failed",
            comparison={
                "mode": "accepted",
                "score_source": "public_accept.score_stats.mean",
                "acceptance_protocol_hash": TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
                "candidate_score": 0.42,
                "current_best_score": 0.40,
                "delta": 0.02,
                "threshold": 0.05,
                "delta_met": False,
                "absolute_accept_floor": TEST_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
                "candidate_score_stats": candidate_stats,
                "best_score_stats": {
                    "count": 3,
                    "mean": 0.40,
                    "std": 0.0,
                    "standard_error": 0.0,
                    "eval_protocol_hash": TEST_EVALUATION_PROTOCOL.protocol_hash,
                },
                "candidate_evaluation_binding_hash": (
                    TEST_EVALUATION_BINDING.binding_hash
                ),
                "current_best": best_stats.comparison_identity(),
                "public_score": 0.42,
            },
        )
        record["decision"]["consumed_gates"].append("behavior_quality_gate")
        record["decision"]["gate_statuses"]["behavior_quality_gate"] = "fail"

        payload = PublicContextBuilder().build_prompt_payload([best_record, record])

        entry = payload["public_history"]["recent_outcomes"][-1]
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertEqual(entry["comparison"]["candidate_score"], 0.42)
        self.assertEqual(entry["comparison"]["current_best_score"], 0.40)
        self.assertEqual(entry["comparison"]["candidate_score_stats"]["count"], 3)
        self.assertEqual(entry["comparison"]["current_best"]["accepted_round_id"], "round-best")
        self.assertEqual(entry["dominant_failure"]["label"], "fall_forward")
        self.assertEqual(entry["dominant_failure"]["source_split"], "public_dev")
        self.assertEqual(entry["sealed_validation_gate"]["checkpoint_safety_status"], "pass")
        self.assertEqual(entry["sealed_validation_gate"]["candidate_isolation_status"], "pass")
        self.assertNotIn("values", json.dumps(entry["comparison"], sort_keys=True))
        self.assertNotIn("eval_protocol_hash", json.dumps(entry["comparison"], sort_keys=True))
        self.assertNotIn("checkpoint_report_path", payload_text)
        self.assertNotIn("D:/runner", payload_text)
        self.assertNotIn("seed_id", payload_text)
        self.assertNotIn("must_not_leak", payload_text)

    def test_direct_prompt_payload_public_history_is_still_bounded(self) -> None:
        payload = assemble_prompt_payload(
            public_context={
                "previous_rounds": 12,
                "recent_outcomes": [
                    {
                        "round_id": f"round-{index:03d}",
                        "outcome": "reject",
                        "runner_error": {
                            "message": "oversized direct public_context " + ("x" * 2000)
                        },
                    }
                    for index in range(12)
                ],
                "public_history_limits": {
                    "max_recent_rounds": 999,
                    "max_serialized_bytes": 999999,
                },
            }
        )

        history = payload["public_history"]
        recent = history["recent_outcomes"]

        self.assertLessEqual(len(recent), history["limits"]["max_recent_rounds"])
        self.assertLessEqual(
            len(json.dumps(recent, sort_keys=True).encode("utf-8")),
            history["limits"]["max_serialized_bytes"],
        )
        self.assertEqual(recent[-1]["round_id"], "round-011")
        self.assertGreater(history["omitted_round_count"], 0)

    def test_stair_extension_guide_mentions_task_specific_public_summary(self) -> None:
        guide = (PROJECT_ROOT / "docs" / "task_extension_guides" / "stair_climbing.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("task-specific public summary", guide)


if __name__ == "__main__":
    unittest.main()
    bind_test_ledger_record,
