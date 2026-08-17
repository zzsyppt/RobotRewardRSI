from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import (
    TEST_CAPABILITY_PROFILE,
    TEST_PROGRAM_SOURCE_CHECKSUM,
    TEST_SUBJECT_MANIFEST_HASH,
    make_evaluation_binding,
    make_test_candidate_package,
)
from tests.fakes.fake_components import (
    FAKE_ACCEPTANCE_PROTOCOL,
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
    FakeVerifier,
    fake_round_session_fields,
)
from true_rl_agent.agent_core.decision import DecisionOutcome
from true_rl_agent.agent_core.ledger import (
    NON_PROVIDER_AUTHORING_MODE,
    PROVIDER_AUTHORING_MODE,
    load_ledger,
)
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.backends.base import BackendJobBundle
from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.prompt_artifacts import RoundAuthoringJournal
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.public_projection import (
    project_stair_public_result,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    build_stair_execution_source_preflight_summary,
    stair_training_distribution_upstream_source_evidence,
)
from true_rl_agent.training_programs import resolve_candidate_program_identity


class FakeRoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = FakeTaskPlugin()
        self.backend = FakeBackendAdapter()
        self.runner = FakeRunner()
        self.verifier = FakeVerifier()

    def test_malformed_eval_score_is_recorded_as_eval_incomplete(self) -> None:
        class MalformedScoreRunner(FakeRunner):
            def run_eval(self, eval_spec):
                _ = eval_spec
                raise ValueError(
                    "public eval public_score must be a finite non-boolean number"
                )

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            candidate = make_test_candidate_package(
                "candidate-malformed-score",
                change_set=CandidateChangeSet(
                    changed_components=frozenset({"reward_callable"})
                ),
                metadata={"fake_public_score": 0.20},
            )

            result = run_candidate_round(
                RoundInputs(
                    round_id="round-malformed-score",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=MalformedScoreRunner(),
                    candidate_admission_policy=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertEqual(result.decision.reason, "eval_incomplete")
        self.assertIsNone(result.public_result)
        self.assertEqual(ledger[0]["decision"]["reason"], "eval_incomplete")
        self.assertEqual(ledger[0]["runner_error"]["type"], "ValueError")

    def test_stair_source_evidence_faults_reject_before_acceptance(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        contract = config.protocol_v2.runner_owned_training_distribution
        self.assertIsNotNone(contract)
        upstream = stair_training_distribution_upstream_source_evidence(contract)
        expected = build_stair_execution_source_preflight_summary(
            task_protocol_hash=config.task_protocol_hash,
            source_manifest_hash=str(upstream["source_manifest_hash"]),
        )

        class StairProjectionTask(FakeTaskPlugin):
            def project_public_result(self, *, public_result, train_summary):
                return project_stair_public_result(
                    config=config,
                    public_result=public_result,
                    train_summary=train_summary,
                )

        class SourceEvidenceRunner(FakeRunner):
            def __init__(self, *, train_preflight, eval_preflight) -> None:
                self.train_preflight = train_preflight
                self.eval_preflight = eval_preflight

            def run_train(self, train_spec):
                result = super().run_train(train_spec)
                summary = dict(result.summary)
                if self.train_preflight is not None:
                    summary["environment_source_preflight"] = dict(
                        self.train_preflight
                    )
                return replace(result, summary=summary)

            def run_eval(self, eval_spec):
                _ = eval_spec
                metrics = {"stair_evaluation": {}}
                if self.eval_preflight is not None:
                    metrics["environment_source_preflight"] = dict(
                        self.eval_preflight
                    )
                return PublicEvalResult(
                    status="pass",
                    public_score=0.5,
                    metrics=metrics,
                )

        task = StairProjectionTask()
        cases = {
            "train_deleted": (None, expected, "train"),
            "eval_tampered": (
                expected,
                {**expected, "task_protocol_hash": "0" * 64},
                "eval",
            ),
        }
        for case, (train_preflight, eval_preflight, failed_stage) in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                ledger_path = Path(tmp) / "ledger.jsonl"
                candidate = make_test_candidate_package(
                    f"candidate-{case}",
                    change_set=CandidateChangeSet(
                        changed_components=frozenset({"reward_callable"})
                    ),
                    metadata={"fake_public_score": 0.5},
                )
                result = run_candidate_round(
                    RoundInputs(
                        round_id=f"round-{case}",
                        task_plugin=task,
                        backend_adapter=FakeBackendAdapter(),
                        runner=SourceEvidenceRunner(
                            train_preflight=train_preflight,
                            eval_preflight=eval_preflight,
                        ),
                        candidate_admission_policy=FakeVerifier(),
                        candidate=candidate,
                        policy_initialization=PolicyInitialization.random(),
                        ledger_path=ledger_path,
                        authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                        public_context={"previous_rounds": 0, "public_only": True},
                        **fake_round_session_fields(task_plugin=task),
                    )
                )
                record = load_ledger(ledger_path)[0]

                self.assertEqual(result.decision.reason, "eval_incomplete")
                self.assertIsNone(result.public_result)
                self.assertFalse(record["decision"]["accepted"])
                self.assertEqual(record["decision"]["comparison"], {})
                self.assertIn(
                    f"Stair {failed_stage} environment source preflight",
                    record["runner_error"]["message"],
                )

    def test_public_eval_fail_status_is_recorded_without_entering_comparison(self) -> None:
        class FailedPublicEvalRunner(FakeRunner):
            def run_eval(self, eval_spec):
                _ = eval_spec
                return PublicEvalResult(
                    status="fail",
                    public_score=0.0,
                    metrics={},
                )

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            candidate = make_test_candidate_package(
                "candidate-public-eval-fail",
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.20},
            )
            result = run_candidate_round(
                RoundInputs(
                    round_id="round-public-eval-fail",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=FailedPublicEvalRunner(),
                    candidate_admission_policy=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertEqual(result.decision.reason, "public_result_status_not_pass")
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["decision"]["comparison"], {})
        self.assertEqual(ledger[0]["decision"]["consumed_gates"], [])
        self.assertTrue(
            validate_decision_record(
                ledger[0],
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
            ).valid
        )

    def _run_candidate(
        self,
        change_kind: str,
        public_score: float,
        *,
        verifier_status: str = "pass",
    ) -> tuple[DecisionOutcome, list[dict]]:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            candidate = make_test_candidate_package(
                f"candidate-{change_kind}",
                change_set=CandidateChangeSet(changed_components=frozenset({change_kind})),
                metadata={"fake_public_score": public_score, "verifier_status": verifier_status},
            )
            result = run_candidate_round(
                RoundInputs(
                    round_id="round-001",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=self.runner,
                    candidate_admission_policy=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )
            return result.decision.outcome, load_ledger(ledger_path)

    def test_code_level_candidate_can_policy_accept_and_append_ledger(self) -> None:
        outcome, ledger = self._run_candidate("reward_callable", public_score=0.20)

        self.assertIs(outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["decision"]["outcome"], "policy_improvement_accept")
        self.assertEqual(ledger[0]["candidate"]["candidate_id"], "candidate-reward_callable")
        self.assertEqual(ledger[0]["public_result"]["public_score"], 0.20)
        self.assertEqual(
            ledger[0]["jobs"]["eval"]["checkpoint"]["uri"],
            ledger[0]["train_result"]["checkpoint"]["uri"],
        )
        self.assertEqual(ledger[0]["public_context"]["previous_rounds"], 0)
        self.assertFalse(ledger[0]["feedback_safe"]["private_eval_used"])

    def test_round_ledger_records_prompt_payload_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "ledger.jsonl"
            journal = RoundAuthoringJournal(
                root / "round-001",
                round_id="round-001",
            )
            journal.bind_budget(
                max_provider_attempts=1,
                max_source_requests_per_round=0,
                max_source_ids_per_request=2,
                max_source_request_bytes_per_round=0,
            )
            handle = journal.begin_attempt(
                {
                    "schema_version": "test.prompt.v1",
                    "public_only": True,
                    "candidate_lineage": {
                        "parent_program_source_checksum": None,
                    },
                },
            )
            prompt_ref = handle.prompt_payload_ref
            candidate = make_test_candidate_package(
                "candidate-reward_callable",
                change_set=CandidateChangeSet(
                    changed_components=frozenset({"reward_callable"})
                ),
                metadata={"fake_public_score": 0.20},
            )
            identity = resolve_candidate_program_identity(
                candidate.path,
                expected_candidate_id=candidate.candidate_id,
            )
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-001",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                    "candidate_id": candidate.candidate_id,
                    "candidate_changed_components": ["reward_callable"],
                    "candidate_subject_manifest_hash": identity.manifest_hash,
                    "candidate_program_source_checksum": identity.source_checksum,
                    "candidate_program_source_size_bytes": identity.source_size_bytes,
                    "anti_degenerate_assessment": {
                        "schema_version": "true_rl_agent.anti_degenerate_assessment.v4",
                        "status": "pass",
                        "classification": "unknown",
                        "reason": "authoring_anchor_source_not_inline",
                        "compares_to": None,
                        "comparison_anchor": "none",
                        "candidate_source_sha256": identity.source_checksum.removeprefix(
                            "sha256:"
                        ),
                        "blocking": False,
                    },
                },
                continuation_state={
                    "schema_version": "test.authoring_continuation.v1",
                    "disposition": "terminal",
                },
            )
            authoring_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="candidate_verified",
            )

            run_candidate_round(
                RoundInputs(
                    round_id="round-001",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=self.runner,
                    candidate_admission_policy=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    authoring_mode=PROVIDER_AUTHORING_MODE,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                    prompt_payload_ref=prompt_ref,
                    authoring_evidence_ref=authoring_ref,
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertEqual(ledger[0]["prompt_payload"], prompt_ref.to_public_dict())
        self.assertEqual(ledger[0]["prompt_payload"]["visibility"], "public_prompt_payload")

    def test_round_rejects_authoring_candidate_source_mismatch_before_runner(
        self,
    ) -> None:
        class NoExecutionRunner(FakeRunner):
            def run_train(self, train_spec):
                raise AssertionError("runner must not execute on authoring mismatch")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_path = root / "ledger.jsonl"
            candidate = make_test_candidate_package(
                "candidate-reward_callable",
                change_set=CandidateChangeSet(
                    changed_components=frozenset({"reward_callable"})
                ),
            )
            identity = resolve_candidate_program_identity(
                candidate.path,
                expected_candidate_id=candidate.candidate_id,
            )
            journal = RoundAuthoringJournal(
                root / "round-001",
                round_id="round-001",
            )
            journal.bind_budget(
                max_provider_attempts=1,
                max_source_requests_per_round=0,
                max_source_ids_per_request=2,
                max_source_request_bytes_per_round=0,
            )
            handle = journal.begin_attempt(
                {
                    "schema_version": "test.prompt.v1",
                    "public_only": True,
                    "candidate_lineage": {
                        "parent_program_source_checksum": None,
                    },
                },
            )
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-001",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                    "candidate_id": candidate.candidate_id,
                    "candidate_changed_components": ["reward_callable"],
                    "candidate_subject_manifest_hash": identity.manifest_hash,
                    "candidate_program_source_checksum": "sha256:" + "9" * 64,
                    "candidate_program_source_size_bytes": identity.source_size_bytes,
                    "anti_degenerate_assessment": {
                        "schema_version": "true_rl_agent.anti_degenerate_assessment.v4",
                        "status": "pass",
                        "classification": "unknown",
                        "reason": "authoring_anchor_source_not_inline",
                        "compares_to": None,
                        "comparison_anchor": "none",
                        "blocking": False,
                    },
                },
                continuation_state={
                    "schema_version": "test.authoring_continuation.v1",
                    "disposition": "terminal",
                },
            )
            authoring_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="candidate_verified",
            )

            with self.assertRaisesRegex(
                ValueError,
                "authoring evidence selected candidate identity mismatch",
            ):
                run_candidate_round(
                    RoundInputs(
                        round_id="round-001",
                        task_plugin=self.task,
                        backend_adapter=self.backend,
                        runner=NoExecutionRunner(),
                        candidate_admission_policy=self.verifier,
                        candidate=candidate,
                        policy_initialization=PolicyInitialization.random(),
                        ledger_path=ledger_path,
                        authoring_mode=PROVIDER_AUTHORING_MODE,
                        public_context={
                            "previous_rounds": 0,
                            "public_only": True,
                        },
                        **fake_round_session_fields(task_plugin=self.task),
                        prompt_payload_ref=handle.prompt_payload_ref,
                        authoring_evidence_ref=authoring_ref,
                    )
                )

            self.assertFalse(ledger_path.exists())

    def test_parameter_only_candidate_rejects_even_with_high_score(self) -> None:
        outcome, ledger = self._run_candidate("reward_weight", public_score=0.95)

        self.assertIs(outcome, DecisionOutcome.REJECT)
        self.assertEqual(ledger[0]["decision"]["reason"], "parameter_only_candidate")
        self.assertFalse(ledger[0]["decision"]["accepted"])

    def test_diagnostic_only_candidate_records_without_policy_accept(self) -> None:
        outcome, ledger = self._run_candidate("diagnostic", public_score=0.95)

        self.assertIs(outcome, DecisionOutcome.DIAGNOSTIC_IMPROVEMENT_RECORD)
        self.assertFalse(ledger[0]["decision"]["accepted"])

    def test_engineering_only_candidate_records_without_policy_accept(self) -> None:
        outcome, ledger = self._run_candidate("failure_classifier", public_score=0.95)

        self.assertIs(outcome, DecisionOutcome.ENGINEERING_IMPROVEMENT_RECORD)
        self.assertFalse(ledger[0]["decision"]["accepted"])

    def test_verifier_failure_rejects_without_training(self) -> None:
        outcome, ledger = self._run_candidate("reward_callable", public_score=0.95, verifier_status="fail")

        self.assertIs(outcome, DecisionOutcome.REJECT)
        self.assertEqual(ledger[0]["decision"]["reason"], "candidate_verifier_failed")
        self.assertEqual(
            ledger[0]["candidate"]["changed_components"],
            ["reward_callable"],
        )
        self.assertEqual(ledger[0]["verifier"]["status"], "fail")
        self.assertEqual(ledger[0]["candidate_surface"]["classification"], "no_code_change")
        self.assertEqual(
            ledger[0]["private_artifacts"][0]["kind"],
            "candidate_artifact",
        )
        self.assertIsNone(ledger[0]["jobs"])
        self.assertIsNone(ledger[0]["train_result"])
        self.assertIsNone(ledger[0]["public_result"])

    def test_task_invariant_violation_rejects_before_training(self) -> None:
        class TamperedBackend(FakeBackendAdapter):
            def materialize_jobs(self, **kwargs):
                bundle = super().materialize_jobs(**kwargs)
                return BackendJobBundle(
                    train_spec=bundle.train_spec,
                    eval_spec=bundle.eval_spec,
                    task_invariants=TaskInvariantSpec({"fall_threshold": 0.10}),
                )

        class TrackingRunner(FakeRunner):
            def __init__(self) -> None:
                self.train_calls = 0

            def run_train(self, train_spec):
                self.train_calls += 1
                return super().run_train(train_spec)

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            runner = TrackingRunner()
            candidate = make_test_candidate_package(
                "candidate-reward_callable",
                change_set=CandidateChangeSet(changed_components=frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.95},
            )

            result = run_candidate_round(
                RoundInputs(
                    round_id="round-001",
                    task_plugin=self.task,
                    backend_adapter=TamperedBackend(),
                    runner=runner,
                    candidate_admission_policy=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )

            ledger = load_ledger(ledger_path)
            self.assertIs(result.decision.outcome, DecisionOutcome.REJECT)
            self.assertEqual(result.decision.reason, "task_invariant_violation")
            self.assertEqual(runner.train_calls, 0)
            self.assertIsNone(ledger[0]["train_result"])

    def test_round_decision_records_comparison_and_consumed_eval_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            candidate = make_test_candidate_package(
                "candidate-reward_callable",
                change_set=CandidateChangeSet(changed_components=frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.30},
            )

            result = run_candidate_round(
                RoundInputs(
                    round_id="round-001",
                    task_plugin=self.task,
                    backend_adapter=self.backend,
                    runner=self.runner,
                    candidate_admission_policy=self.verifier,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **fake_round_session_fields(task_plugin=self.task),
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertIs(result.decision.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
        self.assertIn("comparison", ledger[0]["decision"])
        self.assertEqual(ledger[0]["decision"]["comparison"]["mode"], "bootstrap")
        self.assertIn("sealed_validation_gate", ledger[0]["decision"]["consumed_gates"])
        self.assertEqual(ledger[0]["decision"]["gate_statuses"]["sealed_validation_gate"], "pass")

    def test_fake_runner_uses_eval_spec_public_accept_seed_count(self) -> None:
        eval_spec = EvalJobSpec(
            job_id="fake-eval",
            task_id="fake_push_recovery",
            subject=ExperimentSubject.candidate("candidate-reward_callable"),
            subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
            program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
            training_binding_hash="7" * 64,
            capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
            checkpoint=CheckpointRef(
                "artifact://fake/best.pt",
                "sha256:" + "f" * 64,
                "eval",
                sealed=True,
            ),
            public_seed_count=4,
            public_accept_seed_count=2,
            sealed_validation_seed_count=8,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=4,
                public_accept_seed_count=2,
                sealed_validation_seed_count=8,
                acceptance_protocol_hash=FAKE_ACCEPTANCE_PROTOCOL.protocol_hash,
            ),
            execution_device="cuda:0",
            metadata={"fake_public_score": 0.42, "changed_components": ["reward_callable"]},
        )

        result = self.runner.run_eval(eval_spec)

        self.assertEqual(result.metrics["public_accept"]["score_stats"]["count"], 2)
        self.assertEqual(
            result.metrics["acceptance"]["min_score_count"],
            FAKE_ACCEPTANCE_PROTOCOL.min_score_count,
        )
        self.assertEqual(result.metrics["accept_evidence_gate"]["observed_count"], 2)
        self.assertEqual(len(result.metrics["public_accept"]["seed_results"]), 2)


if __name__ == "__main__":
    unittest.main()
