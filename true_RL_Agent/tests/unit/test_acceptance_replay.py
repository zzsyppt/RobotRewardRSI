from __future__ import annotations

import copy
import json
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_EVALUATION_BINDING,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
    valid_runtime_failed_ledger_record,
)
from true_rl_agent.agent_core.decision import decide_candidate
from true_rl_agent.agent_core.eval_run_audit import build_eval_run_audit
from true_rl_agent.agent_core.ledger import (
    LEDGER_RECORD_SCHEMA_V2,
    LEDGER_RECORD_SCHEMA_V3,
    LEDGER_RECORD_SCHEMA_V5,
    LEDGER_RECORD_SCHEMA_VERSION,
    NON_PROVIDER_AUTHORING_MODE,
    PROVIDER_AUTHORING_MODE,
)
from true_rl_agent.candidates.mechanism import MECHANISM_ATTRIBUTION_SCHEMA_VERSION
from true_rl_agent.candidates.schema import CandidateSurfaceAssessment
from true_rl_agent.evaluation.acceptance_protocol import (
    PAIRED_HEADROOM_DELTA_RULE,
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.memory import PublicContextBuilder
from true_rl_agent.prompt_artifacts import RoundAuthoringJournal
from true_rl_agent.runner.initial_policy import InitialPolicyStatePairing


def _bind_neutral_current_mechanism(record: dict) -> None:
    record["runtime_activation_gate"] = {
        "status": "attribution_unclear",
        "reason": "runner_activation_missing",
    }
    record["ablation_summary"] = {"status": "not_run"}
    record["mechanism_attribution"] = {
        "schema_version": MECHANISM_ATTRIBUTION_SCHEMA_VERSION,
        "status": "not_attributed",
        "reason": "runner_activation_missing",
        "causal_claim": False,
        "runtime_activation_gate": dict(record["runtime_activation_gate"]),
        "attribution_method": "runtime_activation_gate",
        "shadow_ablation_status": "not_run",
    }


class AcceptanceReplayTests(unittest.TestCase):
    def test_paired_headroom_decision_replays_from_one_canonical_threshold(self) -> None:
        protocol = AcceptanceProtocol(
            protocol_id="replay_paired_acceptance_v1",
            score_source=TEST_ACCEPTANCE_PROTOCOL.score_source,
            min_score_count=3,
            absolute_accept_floor=0.05,
            min_delta=0.0,
            public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
            decision_rule=PAIRED_HEADROOM_DELTA_RULE,
            score_ceiling=1.0,
            headroom_reduction_fraction=0.2,
            paired_standard_error_multiplier=1.5,
        )
        binding = EvaluationBindingV1(
            eval_protocol_hash=TEST_EVALUATION_BINDING.eval_protocol_hash,
            acceptance_protocol_hash=protocol.protocol_hash,
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash=TEST_EVALUATION_BINDING.seed_cohort_hash,
        )
        initial = CurrentBestEvalStats.bootstrap(
            initial_best_score=0.0,
            acceptance_protocol=protocol,
            evaluation_binding=binding,
            anchor_id="paired-bootstrap",
        )
        first = _paired_decision_record(
            round_id="round-paired-001",
            candidate_id="candidate-paired-001",
            samples=[0.5, 0.5, 0.5],
            current_best=initial,
            protocol=protocol,
            binding=binding,
        )
        first_best = CurrentBestEvalStats.from_public_result_mapping(
            round_id=first["round_id"],
            candidate_id=first["candidate"]["candidate_id"],
            public_result=first["public_result"],
            acceptance_protocol=protocol,
            subject_manifest_hash=first["jobs"]["eval"]["subject_manifest_hash"],
            program_source_checksum=first["jobs"]["eval"]["program_source_checksum"],
            training_binding_hash=first["jobs"]["eval"]["training_binding_hash"],
            capability_profile_hash=first["jobs"]["eval"]["capability_profile_hash"],
            checkpoint_hash=first["jobs"]["eval"]["checkpoint"]["checksum"],
            expected_evaluation_binding=binding,
        )
        second = _paired_decision_record(
            round_id="round-paired-002",
            candidate_id="candidate-paired-002",
            samples=[0.7, 0.7, 0.7],
            current_best=first_best,
            protocol=protocol,
            binding=binding,
        )
        second["parent_program_source_checksum"] = (
            first_best.program_source_checksum
        )

        persisted_records = json.loads(json.dumps([first, second]))
        replay = replay_acceptance_ledger(
            persisted_records,
            initial_current_best=initial,
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=protocol,
        )

        replay.raise_for_invalid_declared_accepts()
        self.assertEqual(replay.current_best.accepted_round_id, second["round_id"])
        self.assertEqual(
            second["decision"]["reason"],
            "paired_headroom_delta",
        )
        prompt_context = PublicContextBuilder(
            acceptance_protocol=protocol,
            initial_current_best=initial,
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        ).build([first, second])
        serialized_context = json.dumps(prompt_context, sort_keys=True)
        self.assertNotIn("score_samples", serialized_context)
        self.assertNotIn("threshold_components", serialized_context)

        adjacent_runtime_record = copy.deepcopy(persisted_records[1])
        components = adjacent_runtime_record["decision"]["comparison"][
            "threshold_components"
        ]
        components["headroom_delta"] = math.nextafter(
            components["headroom_delta"],
            math.inf,
        )
        adjacent_runtime_replay = replay_acceptance_ledger(
            [persisted_records[0], adjacent_runtime_record],
            initial_current_best=initial,
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=protocol,
        )
        self.assertNotIn(
            "acceptance_replay_threshold_components_mismatch",
            adjacent_runtime_replay.entries[1].sequence_reasons,
        )

        forged = copy.deepcopy(persisted_records[1])
        forged["decision"]["comparison"]["threshold_components"][
            "headroom_delta"
        ] = 0.0
        forged_replay = replay_acceptance_ledger(
            [persisted_records[0], forged],
            initial_current_best=initial,
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=protocol,
        )
        self.assertIn(
            "acceptance_replay_threshold_components_mismatch",
            forged_replay.entries[1].sequence_reasons,
        )

    def test_continuation_verifies_prompt_bytes_and_authoring_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-001",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                    "candidate_id": "candidate-001",
                    "candidate_changed_components": ["reward_callable"],
                },
                continuation_state={
                    "schema_version": "test.authoring_continuation.v1",
                    "disposition": "terminal",
                },
            )
            evidence_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="candidate_verified",
            )
            valid = valid_accepted_ledger_record(
                round_id="round-001",
                candidate_id="candidate-001",
                public_score=0.50,
            )
            valid["schema_version"] = LEDGER_RECORD_SCHEMA_V5
            valid["authoring_mode"] = PROVIDER_AUTHORING_MODE
            valid["prompt_payload"] = prompt_ref.to_public_dict()
            valid.setdefault("private_artifacts", []).append(
                evidence_ref.to_private_dict()
            )
            replay_acceptance_ledger(
                [valid],
                initial_current_best=make_bootstrap_current_best(),
                expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            ).raise_for_continuation()

            forged_path = copy.deepcopy(valid)
            forged_path["prompt_payload"]["path"] = str(
                root / "forged" / "prompt.json"
            )
            replay = replay_acceptance_ledger(
                [forged_path],
                initial_current_best=make_bootstrap_current_best(),
                expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            )
            with self.assertRaisesRegex(
                ValueError,
                "prompt_artifact_content_invalid",
            ):
                replay.raise_for_continuation()

            legacy_path_only = copy.deepcopy(valid)
            legacy_path_only["prompt_payload"] = {
                "path": prompt_ref.path,
                "visibility": "public_prompt_payload",
            }
            replay = replay_acceptance_ledger(
                [legacy_path_only],
                initial_current_best=make_bootstrap_current_best(),
                expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            )
            with self.assertRaisesRegex(ValueError, "prompt_artifact_ref_invalid"):
                replay.raise_for_continuation()

            released_v2 = copy.deepcopy(legacy_path_only)
            released_v2["schema_version"] = LEDGER_RECORD_SCHEMA_V2
            released_v2.pop("authoring_mode")
            released_v2["private_artifacts"] = []
            replay = replay_acceptance_ledger(
                [released_v2],
                initial_current_best=make_bootstrap_current_best(),
                expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            )
            with self.assertRaisesRegex(
                ValueError,
                "authoring_evidence_legacy_unverified",
            ):
                replay.raise_for_continuation()

            deleted = copy.deepcopy(valid)
            deleted.pop("prompt_payload")
            deleted["private_artifacts"] = [
                artifact
                for artifact in deleted["private_artifacts"]
                if artifact.get("kind") != "provider_authoring_evidence"
            ]
            replay = replay_acceptance_ledger(
                [deleted],
                initial_current_best=make_bootstrap_current_best(),
                expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            )
            with self.assertRaisesRegex(ValueError, "prompt_artifact_ref_missing"):
                replay.raise_for_continuation()

            for unsupported in (LEDGER_RECORD_SCHEMA_V3, "ledger_record.v999"):
                unknown = copy.deepcopy(valid)
                unknown["schema_version"] = unsupported
                with self.subTest(unsupported=unsupported):
                    with self.assertRaisesRegex(ValueError, "unsupported"):
                        replay_acceptance_ledger(
                            [unknown],
                            initial_current_best=make_bootstrap_current_best(),
                            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                        )

            missing_schema = copy.deepcopy(valid)
            missing_schema.pop("schema_version")
            replay = replay_acceptance_ledger(
                [missing_schema],
                initial_current_best=make_bootstrap_current_best(),
                expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            )
            with self.assertRaisesRegex(
                ValueError,
                "authoring_evidence_mode_mismatch",
            ):
                replay.raise_for_continuation()

            Path(prompt_ref.path).write_bytes(
                Path(prompt_ref.path).read_bytes() + b"\n"
            )
            replay = replay_acceptance_ledger(
                [valid],
                initial_current_best=make_bootstrap_current_best(),
                expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            )
            with self.assertRaisesRegex(
                ValueError,
                "prompt_artifact_content_invalid",
            ):
                replay.raise_for_continuation()

    def test_current_schema_binds_parent_source_to_replay_head(self) -> None:
        for schema_version in (LEDGER_RECORD_SCHEMA_VERSION,):
            with self.subTest(schema_version=schema_version):
                first = valid_accepted_ledger_record(
                    round_id="round-001",
                    candidate_id="candidate-001",
                    public_score=0.50,
                )
                first["schema_version"] = schema_version
                first["authoring_mode"] = NON_PROVIDER_AUTHORING_MODE
                first["parent_program_source_checksum"] = None
                _bind_neutral_current_mechanism(first)
                second = _delta_accept_record(
                    round_id="round-002",
                    candidate_id="candidate-002",
                    candidate_score=0.60,
                    actual_best=first,
                    declared_best_score=0.50,
                    declared_delta=0.10,
                )
                second["schema_version"] = schema_version
                second["authoring_mode"] = NON_PROVIDER_AUTHORING_MODE
                second["parent_program_source_checksum"] = first["jobs"][
                    "eval"
                ]["program_source_checksum"]
                _bind_neutral_current_mechanism(second)
                replay_acceptance_ledger(
                    [first, second],
                    initial_current_best=make_bootstrap_current_best(),
                    expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ).raise_for_continuation()

                mutations = {}
                missing = copy.deepcopy(second)
                missing.pop("parent_program_source_checksum")
                mutations["missing"] = [first, missing]
                drifted = copy.deepcopy(second)
                drifted["parent_program_source_checksum"] = "sha256:" + "9" * 64
                mutations["drifted"] = [first, drifted]
                mutations["deleted_parent_round"] = [second]
                mutations["reordered"] = [second, first]

                for label, records in mutations.items():
                    with self.subTest(label=label):
                        replay = replay_acceptance_ledger(
                            records,
                            initial_current_best=make_bootstrap_current_best(),
                            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                        )
                        with self.assertRaisesRegex(
                            ValueError,
                            "invalid session binding",
                        ):
                            replay.raise_for_continuation()

    def test_v2_non_provider_history_remains_replayable_without_prompt_artifacts(
        self,
    ) -> None:
        legacy = valid_accepted_ledger_record(
            round_id="round-legacy-non-provider",
            candidate_id="candidate-legacy-non-provider",
            public_score=0.50,
        )
        legacy["schema_version"] = LEDGER_RECORD_SCHEMA_V2
        replay_acceptance_ledger(
            [legacy],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        ).raise_for_continuation()

        with_prompt = copy.deepcopy(legacy)
        with_prompt["prompt_payload"] = {
            "path": "artifact://legacy/prompt.json",
            "visibility": "public_prompt_payload",
        }
        replay = replay_acceptance_ledger(
            [with_prompt],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        with self.assertRaisesRegex(ValueError, "prompt_artifact_ref_unexpected"):
            replay.raise_for_continuation()

    def test_initial_policy_state_pairing_is_replay_owned_and_fail_closed(
        self,
    ) -> None:
        expected = "a" * 64
        accepted = _bind_initial_policy_state_pairing(
            valid_accepted_ledger_record(
                round_id="round-paired",
                candidate_id="candidate-paired",
                public_score=0.50,
            ),
            expected=expected,
        )
        replay_acceptance_ledger(
            [accepted],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            expected_initial_policy_state_hash=expected,
        ).raise_for_continuation()

        mutations = {}
        missing_job_binding = copy.deepcopy(accepted)
        missing_job_binding["jobs"]["train"].pop(
            "expected_initial_policy_state_hash"
        )
        mutations["missing_job_binding"] = missing_job_binding

        missing_result_pairing = copy.deepcopy(accepted)
        missing_result_pairing["train_result"].pop(
            "initial_policy_state_pairing"
        )
        mutations["missing_result_pairing"] = missing_result_pairing

        drifted_result = copy.deepcopy(accepted)
        drifted_result["train_result"]["initial_policy_state_pairing"] = (
            InitialPolicyStatePairing(
                expected_initial_policy_state_hash="b" * 64,
                actual_initial_policy_state_hash="b" * 64,
                match=True,
            ).to_public_dict()
        )
        mutations["drifted_result"] = drifted_result

        mismatched_result = copy.deepcopy(accepted)
        mismatched_result["train_result"]["initial_policy_state_pairing"] = (
            InitialPolicyStatePairing(
                expected_initial_policy_state_hash=expected,
                actual_initial_policy_state_hash="b" * 64,
                match=False,
            ).to_public_dict()
        )
        mutations["mismatched_result"] = mismatched_result

        missing_train_checkpoint = copy.deepcopy(accepted)
        missing_train_checkpoint["train_result"].pop("checkpoint")
        mutations["missing_train_checkpoint"] = missing_train_checkpoint

        drifted_train_checkpoint = copy.deepcopy(accepted)
        drifted_train_checkpoint["train_result"]["checkpoint"]["checksum"] = (
            "sha256:" + "9" * 64
        )
        mutations["drifted_train_checkpoint"] = drifted_train_checkpoint

        missing_eval_pairing_hash = copy.deepcopy(accepted)
        missing_eval_pairing_hash["jobs"]["eval"].pop(
            "initial_policy_state_pairing_hash"
        )
        mutations["missing_eval_pairing_hash"] = missing_eval_pairing_hash

        for label, record in mutations.items():
            with self.subTest(label=label):
                replay = replay_acceptance_ledger(
                    [record],
                    initial_current_best=make_bootstrap_current_best(),
                    expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                    expected_initial_policy_state_hash=expected,
                )
                self.assertFalse(replay.entries[0].valid_policy_accept)
                with self.assertRaisesRegex(
                    ValueError,
                    "invalid session binding",
                ):
                    replay.raise_for_continuation()

    def test_tampered_candidate_runtime_failure_refuses_continuation(self) -> None:
        valid = valid_runtime_failed_ledger_record(
            round_id="round-runtime-failure",
            candidate_id="candidate-runtime-failure",
        )
        replay_acceptance_ledger(
            [valid],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        ).raise_for_continuation()

        tampered = copy.deepcopy(valid)
        tampered["runner_error"]["details"]["candidate_failure"][
            "reason_code"
        ] = "forged_reason"
        replay = replay_acceptance_ledger(
            [tampered],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

        self.assertFalse(replay.entries[0].valid)
        with self.assertRaisesRegex(ValueError, "invalid session binding"):
            replay.raise_for_continuation()

    def test_rejected_record_mechanism_tamper_refuses_continuation(self) -> None:
        tampered = valid_runtime_failed_ledger_record(
            round_id="round-mechanism-tamper",
            candidate_id="candidate-mechanism-tamper",
        )
        tampered["mechanism_attribution"]["status"] = "causally_proven"
        replay = replay_acceptance_ledger(
            [tampered],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

        self.assertIn(
            "mechanism_attribution_contract_invalid",
            replay.entries[0].sequence_reasons,
        )
        with self.assertRaisesRegex(ValueError, "invalid session binding"):
            replay.raise_for_continuation()

    def test_continuation_rejects_genesis_or_binding_drift(self) -> None:
        cases = {}
        wrong_genesis = valid_accepted_ledger_record(
            round_id="round-genesis",
            candidate_id="candidate-genesis",
            public_score=0.50,
        )
        wrong_genesis["session_genesis_hash"] = "a" * 64
        cases["genesis"] = wrong_genesis

        wrong_binding = valid_accepted_ledger_record(
            round_id="round-binding",
            candidate_id="candidate-binding",
            public_score=0.50,
        )
        drifted_binding = EvaluationBindingV1(
            eval_protocol_hash=TEST_EVALUATION_BINDING.eval_protocol_hash,
            acceptance_protocol_hash=(
                TEST_EVALUATION_BINDING.acceptance_protocol_hash
            ),
            task_protocol_hash=TEST_EVALUATION_BINDING.task_protocol_hash,
            seed_cohort_hash="a" * 64,
        )
        wrong_binding["evaluation_binding"] = drifted_binding.to_public_dict()
        cases["binding"] = wrong_binding

        for label, record in cases.items():
            with self.subTest(label=label):
                replay = replay_acceptance_ledger(
                    [record],
                    initial_current_best=make_bootstrap_current_best(),
                    expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                )
                with self.assertRaisesRegex(ValueError, "invalid session binding"):
                    replay.raise_for_continuation()

    def test_public_context_requires_trusted_protocol_for_evaluated_history(self) -> None:
        accepted = valid_accepted_ledger_record(
            round_id="round-001",
            candidate_id="candidate-001",
            public_score=0.50,
        )

        with self.assertRaisesRegex(
            ValueError,
            "acceptance_protocol is required",
        ):
            PublicContextBuilder().build([accepted])

        self.assertEqual(PublicContextBuilder().build([])["previous_rounds"], 0)

    def test_forged_previous_best_cannot_self_certify_second_accept(self) -> None:
        first = valid_accepted_ledger_record(
            round_id="round-001",
            candidate_id="candidate-001",
            public_score=0.50,
        )
        second = _delta_accept_record(
            round_id="round-002",
            candidate_id="candidate-002",
            candidate_score=0.51,
            actual_best=first,
            declared_best_score=0.0,
            declared_delta=0.51,
        )

        intrinsic = validate_decision_record(
            second,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        replay = replay_acceptance_ledger(
            [first, second],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )
        audit = build_eval_run_audit(
            [first, second],
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        )
        restored = CurrentBestEvalStats.from_ledger_records(
            [first, second],
            initial_current_best=make_bootstrap_current_best(),
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

        self.assertTrue(intrinsic.valid_policy_accept, intrinsic.reasons)
        self.assertIn(
            "acceptance_replay_current_best_score_mismatch",
            replay.entries[1].sequence_reasons,
        )
        self.assertIn(
            "acceptance_replay_delta_mismatch",
            replay.entries[1].sequence_reasons,
        )
        self.assertFalse(replay.entries[1].valid_policy_accept)
        self.assertFalse(audit["checks"]["acceptance_replay_consistent"])
        self.assertFalse(audit["checks"]["decision_records_valid"])
        self.assertEqual(audit["counts"]["accepted_count"], 1)
        self.assertEqual(restored.accepted_round_id, "round-001")
        self.assertEqual(restored.score_for_comparison(), 0.50)
        with self.assertRaisesRegex(
            ValueError,
            "round-002.*acceptance_replay_current_best_score_mismatch",
        ):
            replay.raise_for_invalid_declared_accepts()

    def test_continuation_guard_allows_valid_accept_and_invalid_reject(self) -> None:
        accepted = valid_accepted_ledger_record(
            round_id="round-valid",
            candidate_id="candidate-valid",
            public_score=0.50,
        )
        replay_acceptance_ledger(
            [accepted],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        ).raise_for_invalid_declared_accepts()

        invalid_reject = copy.deepcopy(accepted)
        invalid_reject["round_id"] = "round-invalid-reject"
        invalid_reject["verifier"] = {"status": "fail"}
        invalid_reject["decision"].update(
            accepted=False,
            outcome="reject",
            reason="candidate_verifier_failed",
        )
        replay_acceptance_ledger(
            [invalid_reject],
            initial_current_best=make_bootstrap_current_best(),
            expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        ).raise_for_invalid_declared_accepts()

    def test_prompt_history_consumes_fail_closed_replay_projection(self) -> None:
        first = valid_accepted_ledger_record(
            round_id="round-001",
            candidate_id="candidate-001",
            public_score=0.50,
        )
        second = _delta_accept_record(
            round_id="round-002",
            candidate_id="candidate-002",
            candidate_score=0.51,
            actual_best=first,
            declared_best_score=0.0,
            declared_delta=0.51,
        )

        context = PublicContextBuilder(
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(),
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        ).build([first, second])

        self.assertEqual(context["best_public_score"], 0.50)
        self.assertFalse(context["recent_outcomes"][-1]["accepted"])


def _paired_decision_record(
    *,
    round_id: str,
    candidate_id: str,
    samples: list[float],
    current_best: CurrentBestEvalStats,
    protocol: AcceptanceProtocol,
    binding: EvaluationBindingV1,
) -> dict:
    score = statistics.fmean(samples)
    record = valid_accepted_ledger_record(
        round_id=round_id,
        candidate_id=candidate_id,
        public_score=score,
    )
    record["evaluation_binding"] = binding.to_public_dict()
    record["jobs"]["eval"]["evaluation_binding"] = binding.to_public_dict()
    metrics = record["public_result"]["metrics"]
    metrics["evaluation_binding"] = binding.to_public_dict()
    score_stats = {
        "count": len(samples),
        "mean": score,
        "std": statistics.stdev(samples),
        "standard_error": statistics.stdev(samples) / len(samples) ** 0.5,
        "eval_protocol_hash": binding.eval_protocol_hash,
    }
    metrics["public_accept"]["score_stats"] = copy.deepcopy(score_stats)
    metrics["acceptance"].update(
        {
            "score_source": protocol.score_source,
            "public_score_binding": protocol.public_score_binding,
            "score_stats": copy.deepcopy(score_stats),
            "score_samples": list(samples),
            "min_score_count": protocol.min_score_count,
            "absolute_accept_floor": protocol.absolute_accept_floor,
            "min_delta": protocol.min_delta,
            "acceptance_protocol_hash": protocol.protocol_hash,
        }
    )
    sealed = metrics["sealed_validation"]
    sealed["evaluation_binding_hash"] = binding.binding_hash
    sealed["score_stats"]["mean"] = score
    record["public_result"]["public_score"] = score
    decision = decide_candidate(
        surface=CandidateSurfaceAssessment(
            "code_level",
            True,
            "code_level_candidate",
        ),
        public_result=PublicEvalResult(**record["public_result"]),
        current_best=current_best,
        acceptance_protocol=protocol,
    )
    record["decision"] = decision.to_public_dict()
    return record


def _delta_accept_record(
    *,
    round_id: str,
    candidate_id: str,
    candidate_score: float,
    actual_best: dict,
    declared_best_score: float,
    declared_delta: float,
) -> dict:
    record = valid_accepted_ledger_record(
        round_id=round_id,
        candidate_id=candidate_id,
        public_score=candidate_score,
    )
    record["parent_program_source_checksum"] = actual_best["jobs"]["eval"][
        "program_source_checksum"
    ]
    best_stats = copy.deepcopy(
        actual_best["public_result"]["metrics"]["acceptance"]["score_stats"]
    )
    candidate_stats = copy.deepcopy(
        record["public_result"]["metrics"]["acceptance"]["score_stats"]
    )
    actual_best_stats = CurrentBestEvalStats.from_public_result_mapping(
        round_id=actual_best["round_id"],
        candidate_id=actual_best["candidate"]["candidate_id"],
        public_result=actual_best["public_result"],
        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        subject_manifest_hash=actual_best["jobs"]["eval"][
            "subject_manifest_hash"
        ],
        program_source_checksum=actual_best["jobs"]["eval"][
            "program_source_checksum"
        ],
        training_binding_hash=actual_best["jobs"]["eval"][
            "training_binding_hash"
        ],
        capability_profile_hash=actual_best["jobs"]["eval"][
            "capability_profile_hash"
        ],
        checkpoint_hash=actual_best["jobs"]["eval"]["checkpoint"]["checksum"],
        expected_evaluation_binding=TEST_EVALUATION_BINDING,
    )
    record["decision"].update(
        reason="noise_aware_delta",
        comparison={
            "mode": "accepted",
            "score_source": TEST_ACCEPTANCE_PROTOCOL.score_source,
            "acceptance_protocol_hash": TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
            "candidate_score": candidate_score,
            "current_best_score": declared_best_score,
            "delta": declared_delta,
            "threshold": TEST_ACCEPTANCE_PROTOCOL.min_delta,
            "delta_met": declared_delta > TEST_ACCEPTANCE_PROTOCOL.min_delta,
            "absolute_accept_floor": TEST_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
            "candidate_score_stats": candidate_stats,
            "best_score_stats": best_stats,
            "candidate_evaluation_binding_hash": (
                TEST_EVALUATION_BINDING.binding_hash
            ),
            "current_best": actual_best_stats.comparison_identity(),
            "public_score": candidate_score,
        },
    )
    return record


def _bind_initial_policy_state_pairing(
    record: dict,
    *,
    expected: str,
) -> dict:
    bound = copy.deepcopy(record)
    bound["jobs"]["train"]["expected_initial_policy_state_hash"] = expected
    pairing = InitialPolicyStatePairing(
        expected_initial_policy_state_hash=expected,
        actual_initial_policy_state_hash=expected,
        match=True,
    )
    checkpoint = {
        "uri": "artifact://paired/eval.pt",
        "checksum": bound["jobs"]["eval"]["checkpoint"]["checksum"],
        "role": "eval",
        "sealed": True,
    }
    bound["jobs"]["eval"]["checkpoint"] = copy.deepcopy(checkpoint)
    bound["jobs"]["eval"]["initial_policy_state_pairing_hash"] = (
        pairing.pairing_hash
    )
    bound["train_result"]["checkpoint"] = copy.deepcopy(checkpoint)
    bound["train_result"]["initial_policy_state_pairing"] = (
        pairing.to_public_dict()
    )
    return bound


if __name__ == "__main__":
    unittest.main()
