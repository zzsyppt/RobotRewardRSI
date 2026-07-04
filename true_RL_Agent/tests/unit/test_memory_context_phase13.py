from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload
from true_rl_agent.memory import ExperimentRecord, PrivateArtifactRef, PublicContextBuilder, TaskMemoryAdapter
from true_rl_agent.prompting.payload import build_prompt_payload as assemble_prompt_payload
from true_rl_agent.tasks.push_recovery import PushTaskMemoryAdapter


class MemoryContextPhase13Tests(unittest.TestCase):
    def _push_record(self, *, round_id: str, score: float, failure_breakdown: dict[str, float]) -> dict[str, object]:
        return {
            "round_id": round_id,
            "task": {"task_id": "unitree_push_recovery"},
            "decision": {
                "accepted": True,
                "outcome": "policy_improvement_accept",
                "public_score": score,
            },
            "candidate_surface": {"classification": "code_level"},
            "public_result": {
                "status": "pass",
                "public_score": score,
                "metrics": {
                    "success_rate": score,
                    "failure_breakdown": failure_breakdown,
                    "motion_quality": {"mean_base_height": 0.79, "mean_tilt_rad": 0.04},
                    "trusted_seed_count": 17,
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
                    checksum="sha256:private",
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

    def test_public_context_builder_saves_public_only_prompt_payload(self) -> None:
        records = [
            ExperimentRecord.from_ledger_record(
                self._push_record(round_id="round-001", score=0.30, failure_breakdown={"fall": 0.3})
            ),
            ExperimentRecord.from_ledger_record(
                self._push_record(round_id="round-002", score=0.46, failure_breakdown={"push_drift": 0.4})
            ),
        ]
        builder = PublicContextBuilder(task_memory_adapter=PushTaskMemoryAdapter())

        payload = builder.build_prompt_payload(records)
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["public_context"]["previous_rounds"], 2)
        self.assertEqual(payload["public_context"]["best_public_score"], 0.46)
        self.assertEqual(
            payload["public_context"]["task_public_summary"]["latest_failure_mode"],
            "push_recovery_drift",
        )
        self.assertEqual(payload["public_context"]["task_public_summary"]["success_rate"]["best"], 0.46)
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
        payload = PublicContextBuilder().build_prompt_payload(
            [
                {
                    "round_id": "round-raw-mapping",
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
                        "message": "failed at D:/humanoid_agent/private/raw.txt",
                    },
                    "raw_seed_id": "public-dev-seed-001",
                }
            ]
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
                    "reason": "candidate_runtime_failed",
                    "surface_classification": "code_level",
                    "public_score": 0.0,
                },
                "candidate": {
                    "candidate_id": "candidate-010",
                    "path": "D:/humanoid_agent/private/candidate-010",
                    "changed_components": ["diagnostic", "reward_callable"],
                },
                "candidate_surface": {"classification": "code_level"},
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
                    }
                },
                "public_result": {
                    "status": "pass",
                    "public_score": 0.0,
                    "metrics": {
                        "candidate_package_loaded": True,
                        "candidate_code_loaded_in_eval": False,
                    },
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
        self.assertEqual(entry["reason"], "candidate_runtime_failed")
        self.assertEqual(entry["surface_classification"], "code_level")
        self.assertEqual(entry["public_score"], 0.0)
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
            len(json.dumps(recent, sort_keys=True).encode("utf-8")),
            history["limits"]["max_serialized_bytes"],
        )
        self.assertEqual(recent[-1]["round_id"], "round-011")
        self.assertGreater(history["omitted_round_count"], 0)

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
