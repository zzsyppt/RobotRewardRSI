"""CPU sampling fixtures exercise receipt-to-provider coverage, not training quality."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from true_rl_agent.agent_core.experiment_memory import build_experiment_memory, recompact_experiment_memory
from true_rl_agent.agent_core.generation_contract import latest_nonaccepted_trial
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.candidates.mechanism import RuntimeActivationEvidence
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.public_projection import project_public_train_summary, project_public_training_sampling
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.memory.context_builder import PublicContextBuilder
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.runner.training_numeric_capture import TrainingNumericCapture
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.prompt_adapter import _runtime_feedback_summary
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.prompt_adapter import _latest_runtime_feedback


ROOT = Path(__file__).resolve().parents[2]


def sampling_fixture():
    # A real CPU writer and its coverage contract; deliberately stop in the
    # middle window. Deleting the temporary archive tests portable replay.
    with tempfile.TemporaryDirectory() as directory:
        capture = TrainingNumericCapture(
            Path(directory) / "numeric", identity={"source": "cpu-test-fixture"},
            semantics={"raw_term": "task_owned_guarded_value", "control_dt_s": 0.02,
                       "terms": [{"name": "fixture_term", "weight": 0.25}]},
            planned_control_steps=24, num_envs=20, steps_per_iteration=8,
            source_step_offset=1,
        )
        for step in range(1, 13):
            capture.observe(step + 1, {"raw_term_000": np.ones(20, dtype=np.float32)})
        return capture.finish(training_status="fail")


def train_result(summary):
    subject = ExperimentSubject.candidate("sampling-candidate")
    fields = {
        "subject_manifest_hash": "a" * 64,
        "program_source_checksum": "sha256:" + "b" * 64,
        "training_binding_hash": "c" * 64,
        "capability_profile_hash": "d" * 64,
        "execution_device": "cuda:0",
    }
    return TrainJobResult(
        job_id="sampling-train", subject=subject, **fields, summary=summary,
        checkpoint=CheckpointRef(
            uri="artifact://sampling/best.pt", checksum="sha256:" + "e" * 64,
            role="eval", sealed=True,
            metadata={"training_subject": subject.to_public_dict(), **fields},
        ),
    )


def public_record(result):
    return {
        "round_id": "sampling-round",
        "candidate": {"candidate_id": result.subject.to_public_dict()["id"], "changed_components": []},
        "decision": {"accepted": False, "outcome": "reject", "reason": "eval_incomplete",
                     "surface_classification": "code_level", "public_score": None,
                     "comparison": {}, "consumed_gates": [], "gate_statuses": {}},
        "candidate_surface": {"classification": "code_level"},
        "verifier": {"status": "pass"},
        "jobs": {"train": {"job_id": result.job_id}, "eval": {"job_id": "sampling-eval"}},
        "runner_error": {"type": "RunnerExecutionError", "message": "CPU fixture stopped after training"},
        "train_result": result.to_public_dict(),
        "runtime_activation_summary": RuntimeActivationEvidence.from_train_summary(result.summary).to_public_dict(),
    }


class TrainingSamplingProjectionTests(unittest.TestCase):
    def test_real_writer_coverage_survives_public_receipt_and_both_task_provider_payloads(self):
        raw = sampling_fixture()
        original = copy.deepcopy(raw)
        self.assertFalse(Path(raw["reference"]["manifest_path"]).exists())
        result = train_result({
            "status": "fail", "training": {"numeric_evidence": raw},
            "runtime_activation_summary": {"step_calls": 13, "reward_term_calls": 13,
                                           "reward_term_count": 1, "nonzero_reward_contribution": False},
        })
        record = public_record(result)
        projected = record["train_result"]["summary"]["training"]["numeric_evidence"]
        self.assertEqual(project_public_training_sampling(projected), projected)
        self.assertEqual(project_public_train_summary(record["train_result"]["summary"]),
                         record["train_result"]["summary"])
        self.assertEqual([w["status"] for w in projected["windows"]], ["sampled", "partial", "not_reached"])
        self.assertEqual([w["sampled_control_steps"] for w in projected["windows"]], [8, 4, 0])
        self.assertEqual([w["sampled_env_transitions"] for w in projected["windows"]], [128, 64, 0])
        self.assertEqual([w["planned_sampled_env_transitions"] for w in projected["windows"]], [128] * 3)
        self.assertEqual(projected["sampled_env_count"], 16)
        self.assertEqual(projected["capture_schema_version"], "true_rl_agent.training_numeric_capture.v2")
        self.assertEqual((projected["num_envs"], projected["planned_control_steps"], projected["observed_control_steps"]), (20, 24, 12))
        self.assertEqual(projected["source_step_offset_excluded"], 1)
        self.assertEqual(projected["term_semantics_hash"], canonical_json_hash(raw["coverage"]["semantics"]))
        self.assertEqual(projected["archive_manifest_sha256"], raw["reference"]["manifest_sha256"])
        self.assertEqual(projected["training_status"], "fail")
        self.assertEqual(RuntimeActivationEvidence.from_train_summary(
            record["train_result"]["summary"]).training_sampling, projected)
        for provider, cell in ((StairClimbingCompositionProvider(), "stair.go2_12dof.unified_v3"),
                               (PushRecoveryCompositionProvider(), None)):
            with self.subTest(task=provider.task_id), tempfile.TemporaryDirectory() as directory:
                values = provider.cli_defaults()
                if cell:
                    values["cell_variant_id"] = cell
                composition = provider.resolve(values=values, context=TaskCompositionContext(
                    project_root=ROOT, output_dir=Path(directory)))
                payload = PublicContextBuilder(
                    task_prompt_adapter=composition.task_plugin.prompt_adapter(),
                    task_memory_adapter=composition.task_memory_adapter,
                    candidate_capability_profile=composition.task_plugin.task_spec.candidate_capability_profile,
                ).build_prompt_payload([record])
                shown = payload["runtime_feedback_summary"]["latest_runtime_activation"]["training_sampling"]
                self.assertEqual(shown, projected)
                encoded = json.dumps(payload)
                for excluded in ("manifest_path", "sampled_env_ids", "sampled_steps", "planned_steps", "raw_term_000",
                                 "fixture_term", "task_owned_guarded_value", raw["reference"]["manifest_path"]):
                    self.assertNotIn(excluded, encoded)
        self.assertEqual(raw, original)

    def test_missing_disabled_failed_and_partial_sampling_do_not_imply_reward_execution(self):
        cases = [(None, "not_available"), ({"status": "not_enabled"}, "not_enabled"),
                 ({"status": "not_started"}, "not_started"),
                 ({"status": "incomplete", "reason": "failed at /data/private/archive"}, "incomplete"),
                 (sampling_fixture(), "complete")]
        for raw, expected in cases:
            with self.subTest(status=expected):
                summary = {} if raw is None else {"training": {"numeric_evidence": raw}}
                evidence = RuntimeActivationEvidence.from_train_summary(summary)
                public = evidence.to_public_dict()
                self.assertEqual(public["status"], "not_available")
                self.assertEqual(evidence.reward_execution_state(), "not_available")
                self.assertEqual(public["training_sampling"]["status"], expected)
                self.assertEqual(RuntimeActivationEvidence.from_mapping(public).to_public_dict(), public)
                record = public_record(train_result(summary))
                for project in (_runtime_feedback_summary, _latest_runtime_feedback):
                    self.assertEqual(project([record])["latest_runtime_activation"]["training_sampling"],
                                     public["training_sampling"])
                payload = PublicContextBuilder().build_prompt_payload([record])
                self.assertEqual(payload["public_history"]["recent_outcomes"][-1]["runtime_activation"], public)
                self.assertNotIn("/data/private", json.dumps(payload))

    def test_window_allowlist_is_bounded_and_keeps_missing_windows_unknown(self):
        raw = sampling_fixture()
        raw["coverage"]["windows"] = [raw["coverage"]["windows"][0],
            {"name": "unbounded-extra", "status": "sampled", "raw_values": [1] * 1000}]
        projected = project_public_training_sampling(raw)
        self.assertEqual(len(projected["windows"]), 3)
        self.assertEqual([w["status"] for w in projected["windows"]],
                         ["sampled", "not_available", "not_available"])
        self.assertNotIn("raw_values", json.dumps(projected))
        self.assertEqual(project_public_training_sampling({"status": "complete"}), {"status": "incomplete"})

    def test_actual_nonzero_failure_summary_reaches_next_round_author_without_train_result(self):
        from tests.integration.test_live_loop import (
            _NonzeroFailSummaryBackend, _RuntimePathRepairProvider, run_live_loop,
            FakeTaskPlugin, FakeVerifier,
        )
        raw = sampling_fixture()
        expected = project_public_training_sampling(raw)

        class SamplingFailureBackend(_NonzeroFailSummaryBackend):
            def materialize_jobs(self, **kwargs):
                jobs = super().materialize_jobs(**kwargs)
                argv = list(jobs.train_spec.metadata["cli_argv"])
                argv[2] = argv[2].replace(
                    "summary_path.write_text",
                    "summary['training'] = " + repr({"numeric_evidence": raw}) + "\nsummary_path.write_text",
                )
                return replace(jobs, train_spec=replace(jobs.train_spec,
                    metadata={**jobs.train_spec.metadata, "cli_argv": argv, "cli_command": " ".join(argv)}))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = _RuntimePathRepairProvider(root / "candidates")
            summary = run_live_loop(
                output_dir=root / "run", rounds=2, session_id="sampling-failure",
                task_plugin=FakeTaskPlugin(), backend_adapter=SamplingFailureBackend(root / "jobs"),
                runner=SubprocessRunner(allowed_executables=(sys.executable,), timeout_s=10),
                candidate_admission_policy=FakeVerifier(), provider=provider,
                policy_initialization=PolicyInitialization.random(), initial_best_score=0.10,
                policy_accept_delta=0.05, max_provider_attempts=2,
            )
            records = load_ledger(root / "run" / "ledger.jsonl")
        self.assertEqual(summary["rounds_completed"], 2)
        self.assertTrue(summary["checks"]["eval_run_audit_passed"])
        failure = records[0]
        self.assertIsNone(failure["train_result"])
        self.assertEqual(failure["decision"]["reason"], "candidate_runtime_failed")
        self.assertEqual(failure["runner_error"]["details"]["training_sampling"], expected)
        self.assertEqual(failure["runtime_activation_summary"],
                         {"status": "not_available", "training_sampling": expected})
        payload = provider.payloads[1]
        self.assertEqual(payload["public_history"]["recent_outcomes"][0]["runtime_activation"],
                         failure["runtime_activation_summary"])
        # Both task consumers may show the failed run, without reclassifying it
        # as completed training for the shared code/source-selection policy.
        self.assertIsNone(latest_nonaccepted_trial([failure]))
        for project in (_runtime_feedback_summary, _latest_runtime_feedback):
            feedback = project([failure])
            self.assertEqual(feedback["latest_runtime_activation"]["training_sampling"], expected)
            self.assertEqual(feedback["source_candidate_id"], failure["candidate"]["candidate_id"])
        for excluded in ("manifest_path", "sampled_steps", "sampled_env_ids", "fixture_term",
                         raw["reference"]["manifest_path"]):
            self.assertNotIn(excluded, json.dumps(payload))

    def test_coverage_does_not_expand_experiment_memory_or_change_sibling_execution_states(self):
        sampled = RuntimeActivationEvidence.from_train_summary({"training": {"numeric_evidence": sampling_fixture()},
            "runtime_activation_summary": {"reward_term_calls": 12, "nonzero_reward_contribution": False}}).to_public_dict()
        base_records, enriched_records = [], []
        for index in range(20):
            trial = {"lane_id": "lane-a", "candidate": {"candidate_id": f"candidate-{index}"},
                     "trial_status": "train_failed", "runtime_activation_summary": sampled}
            record = {"round_id": f"round-{index}", "generation": {"trials": [trial, {
                "lane_id": "lane-b", "candidate": {"candidate_id": f"missing-{index}"},
                "trial_status": "train_failed", "runtime_activation_summary": {
                    "status": "not_available", "training_sampling": {"status": "not_enabled"}}}]}}
            enriched_records.append(record)
            plain = copy.deepcopy(record)
            for item in plain["generation"]["trials"]:
                item["runtime_activation_summary"].pop("training_sampling")
            base_records.append(plain)
        memory = build_experiment_memory(enriched_records)
        self.assertEqual(memory, build_experiment_memory(base_records))
        self.assertLessEqual(canonical_prompt_json_bytes(memory), 14 * 1024)
        for _ in range(3):
            memory = recompact_experiment_memory(memory)
            for experiment in memory["experiments"]:
                self.assertEqual([row[-1] for row in experiment["trial_outcomes"]],
                                 ["executed_zero", "not_available"])


if __name__ == "__main__":
    unittest.main()
