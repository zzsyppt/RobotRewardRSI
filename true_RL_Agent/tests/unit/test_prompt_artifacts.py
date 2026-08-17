from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (  # noqa: E402
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_EVALUATION_BINDING,
    TEST_SESSION_GENESIS_HASH,
    valid_accepted_ledger_record,
)
from true_rl_agent.agent_core.decision import decide_candidate  # noqa: E402
from true_rl_agent.agent_core.ledger import (  # noqa: E402
    LEDGER_RECORD_SCHEMA_V2,
    LEDGER_RECORD_SCHEMA_V3,
    LEDGER_RECORD_SCHEMA_V4,
    LEDGER_RECORD_SCHEMA_V5,
    LEDGER_RECORD_SCHEMA_V6,
    LEDGER_RECORD_SCHEMA_VERSION,
    NON_PROVIDER_AUTHORING_MODE,
    PROVIDER_AUTHORING_MODE,
)
from true_rl_agent.agent_core.live_loop import _authoring_history  # noqa: E402
from true_rl_agent.agent_core.source_request_limits import (  # noqa: E402
    BEST_OBSERVED_SOURCE_ID,
    CURRENT_BEST_SOURCE_ID,
    LATEST_FAILED_SOURCE_ID,
)
from true_rl_agent.authoring_artifact_contract import (  # noqa: E402
    MAX_AUTHORING_ATTEMPT_RESULT_BYTES,
    MAX_AUTHORING_BUDGET_ARTIFACT_BYTES,
    MAX_AUTHORING_EVIDENCE_ABSOLUTE_BYTES,
    MAX_AUTHORING_PROVIDER_ATTEMPTS,
)
from true_rl_agent.candidates.mechanism import (  # noqa: E402
    MechanismDecision,
    build_mechanism_attribution_summary,
)
from true_rl_agent.candidates.schema import CandidateSurfaceAssessment  # noqa: E402
from true_rl_agent.evaluation.acceptance_replay import (  # noqa: E402
    replay_acceptance_ledger,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats  # noqa: E402
from true_rl_agent.evaluation.public_result import PublicEvalResult  # noqa: E402
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.prompt_artifacts import (  # noqa: E402
    AUTHORING_EVIDENCE_SCHEMA_V2,
    AUTHORING_EVIDENCE_SCHEMA_V3,
    AuthoringEvidenceRef,
    PromptPayloadRef,
    RoundAuthoringAnchorRef,
    RoundAuthoringJournal,
    project_selected_authoring_mechanism_delta,
    publish_prompt_payload,
    validate_ledger_prompt_artifacts,
    verify_authoring_evidence,
    verify_prompt_payload,
    verify_round_authoring_anchor,
)
from true_rl_agent.prompt_serialization import (  # noqa: E402
    MAX_AUTHORING_PROMPT_PAYLOAD_BYTES,
    canonical_prompt_json,
)


def _bind_test_budget(journal: RoundAuthoringJournal) -> None:
    journal.bind_budget(
        max_provider_attempts=5,
        max_source_requests_per_round=2,
        max_source_ids_per_request=2,
        max_source_request_bytes_per_round=48 * 1024,
    )


def _test_continuation() -> dict[str, str]:
    return {"schema_version": "test.authoring_continuation.v1"}


def _test_prompt(payload: dict) -> dict:
    return {
        "candidate_lineage": {"parent_program_source_checksum": None},
        **payload,
    }


def _test_prompt_with_parent(
    *,
    parent_program_source_checksum: str,
    current_best_source: str,
) -> dict:
    content_sha256 = hashlib.sha256(
        current_best_source.encode("utf-8")
    ).hexdigest()
    return {
        "candidate_lineage": {
            "parent_program_source_checksum": parent_program_source_checksum,
            "current_best": {
                "current_head_program_source_checksum": (
                    parent_program_source_checksum
                ),
                "current_head_prompt_content_sha256": content_sha256,
                "source_logical_id": CURRENT_BEST_SOURCE_ID,
                "source_delivery": "inline",
                "source_payload_ref": CURRENT_BEST_SOURCE_ID,
                "source_hash_scope": "prompt_visible_sanitized_content",
            },
        },
        "source_payloads": [
            {
                "logical_id": CURRENT_BEST_SOURCE_ID,
                "content": current_best_source,
                "content_sha256": content_sha256,
            }
        ],
    }


def _publish_test_evidence(path: Path, payload: dict) -> AuthoringEvidenceRef:
    encoded = canonical_json_bytes(payload, trailing_newline=True)
    path.write_bytes(encoded)
    return AuthoringEvidenceRef(
        path=str(path),
        checksum="sha256:" + hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )


def _publish_v2_evidence_copy(
    evidence_ref: AuthoringEvidenceRef,
) -> AuthoringEvidenceRef:
    payload = json.loads(Path(evidence_ref.path).read_text(encoding="utf-8"))
    payload["schema_version"] = AUTHORING_EVIDENCE_SCHEMA_V2
    payload.pop("budget_contract")
    payload.pop("terminal_outcome")
    payload.pop("round_authoring_anchor")
    path = Path(evidence_ref.path).with_name("authoring_evidence_v2.json")
    encoded = canonical_json_bytes(payload, trailing_newline=True)
    path.write_bytes(encoded)
    return AuthoringEvidenceRef(
        path=str(path),
        checksum="sha256:" + hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )


def _publish_v3_evidence_copy(
    evidence_ref: AuthoringEvidenceRef,
) -> AuthoringEvidenceRef:
    payload = json.loads(Path(evidence_ref.path).read_text(encoding="utf-8"))
    payload["schema_version"] = AUTHORING_EVIDENCE_SCHEMA_V3
    payload.pop("round_authoring_anchor")
    path = Path(evidence_ref.path).with_name("authoring_evidence_v3.json")
    encoded = canonical_json_bytes(payload, trailing_newline=True)
    path.write_bytes(encoded)
    return AuthoringEvidenceRef(
        path=str(path),
        checksum="sha256:" + hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )


def _neutral_mechanism_fields() -> dict:
    runtime_gate = {
        "status": "attribution_unclear",
        "reason": "runner_activation_missing",
    }
    ablation = {"status": "not_run"}
    return {
        "runtime_activation_gate": runtime_gate,
        "ablation_summary": ablation,
        "mechanism_attribution": build_mechanism_attribution_summary(
            runtime_activation_gate=MechanismDecision(
                runtime_gate["status"],
                runtime_gate["reason"],
            ),
            ablation_summary=ablation,
        ),
    }


def _rebind_record_program_source(
    record: dict,
    program_source_checksum: str,
) -> None:
    record["candidate"]["program_source_checksum"] = program_source_checksum
    record["jobs"]["train"][
        "program_source_checksum"
    ] = program_source_checksum
    record["jobs"]["eval"][
        "program_source_checksum"
    ] = program_source_checksum
    metrics = record["public_result"]["metrics"]
    metrics["program_source_checksum"] = program_source_checksum
    metrics["sealed_validation"][
        "program_source_checksum"
    ] = program_source_checksum


def _accepted_current_best(record: dict) -> CurrentBestEvalStats:
    eval_job = record["jobs"]["eval"]
    return CurrentBestEvalStats.from_public_result_mapping(
        round_id=record["round_id"],
        candidate_id=record["candidate"]["candidate_id"],
        public_result=record["public_result"],
        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        subject_manifest_hash=eval_job["subject_manifest_hash"],
        program_source_checksum=eval_job["program_source_checksum"],
        training_binding_hash=eval_job["training_binding_hash"],
        capability_profile_hash=eval_job["capability_profile_hash"],
        checkpoint_hash=eval_job["checkpoint"]["checksum"],
        expected_evaluation_binding=TEST_EVALUATION_BINDING,
    )


def _bind_decision_to_current_best(
    record: dict,
    current_best: CurrentBestEvalStats,
) -> None:
    record["decision"] = decide_candidate(
        surface=CandidateSurfaceAssessment(
            "code_level",
            True,
            "code_level_candidate",
        ),
        public_result=PublicEvalResult(**record["public_result"]),
        current_best=current_best,
        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
    ).to_public_dict()


class PromptArtifactTests(unittest.TestCase):
    def test_selected_authoring_mechanism_delta_is_source_bound(self) -> None:
        source_digest = "a" * 64
        anchor_source = "def build_training_components():\n    return None\n"
        anchor_digest = hashlib.sha256(anchor_source.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            prompt_ref = publish_prompt_payload(
                Path(tmp) / "prompt.json",
                {
                    "source_payloads": [
                        {
                            "logical_id": (
                                "current_best/training_code/components.py"
                            ),
                            "content": anchor_source,
                            "content_sha256": anchor_digest,
                        }
                    ]
                },
            )
            evidence = {
                "schema_version": AUTHORING_EVIDENCE_SCHEMA_V3,
                "selected_attempt_id": "attempt_000001",
                "attempts": [
                    {
                        "attempt_id": "attempt_000001",
                        "prompt_payload": prompt_ref.to_public_dict(),
                        "provider_attempt": {
                            "candidate_program_source_checksum": (
                                f"sha256:{source_digest}"
                            ),
                            "anti_degenerate_assessment": {
                                "schema_version": (
                                    "true_rl_agent.anti_degenerate_assessment.v4"
                                ),
                                "status": "pass",
                                "blocking": False,
                                "classification": "mechanism_delta_present",
                                "comparison_anchor": "current_best",
                                "compares_to": anchor_digest,
                                "candidate_source_sha256": source_digest,
                                "changed_mechanism_count": 1,
                                "changed_mechanisms": [
                                    "RewardTerm:progress_reward"
                                ],
                            },
                        },
                    }
                ],
            }

            self.assertEqual(
                project_selected_authoring_mechanism_delta(evidence),
                {
                    "schema_version": (
                        "true_rl_agent.candidate_mechanism_delta.v1"
                    ),
                    "comparison_anchor": "current_best",
                    "compares_to": anchor_digest,
                    "candidate_source_sha256": source_digest,
                    "changed_mechanisms": ["RewardTerm:progress_reward"],
                },
            )
            source_tampered = copy.deepcopy(evidence)
            source_tampered["attempts"][0]["provider_attempt"][
                "candidate_program_source_checksum"
            ] = "sha256:" + ("c" * 64)
            with self.assertRaisesRegex(ValueError, "source identity mismatch"):
                project_selected_authoring_mechanism_delta(source_tampered)

            schema_tampered = copy.deepcopy(evidence)
            schema_tampered["attempts"][0]["provider_attempt"][
                "anti_degenerate_assessment"
            ]["schema_version"] = "true_rl_agent.anti_degenerate_assessment.v3"
            with self.assertRaisesRegex(ValueError, "assessment schema"):
                project_selected_authoring_mechanism_delta(schema_tampered)

            anchor_tampered = copy.deepcopy(evidence)
            anchor_tampered["attempts"][0]["provider_attempt"][
                "anti_degenerate_assessment"
            ]["compares_to"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "comparison identity"):
                project_selected_authoring_mechanism_delta(anchor_tampered)

            count_tampered = copy.deepcopy(evidence)
            count_tampered["attempts"][0]["provider_attempt"][
                "anti_degenerate_assessment"
            ]["changed_mechanism_count"] = True
            with self.assertRaisesRegex(ValueError, "not canonical"):
                project_selected_authoring_mechanism_delta(count_tampered)

            missing_delta = copy.deepcopy(evidence)
            missing_assessment = missing_delta["attempts"][0]["provider_attempt"][
                "anti_degenerate_assessment"
            ]
            missing_assessment["classification"] = "unknown"
            missing_assessment["changed_mechanism_count"] = 0
            missing_assessment["changed_mechanisms"] = []
            with self.assertRaisesRegex(ValueError, "no canonical mechanism delta"):
                project_selected_authoring_mechanism_delta(missing_delta)

    def test_no_anchor_mechanism_projection_still_binds_candidate_source(self) -> None:
        source_digest = "d" * 64
        with tempfile.TemporaryDirectory() as tmp:
            prompt_ref = publish_prompt_payload(Path(tmp) / "prompt.json", {})
            evidence = {
                "schema_version": AUTHORING_EVIDENCE_SCHEMA_V3,
                "selected_attempt_id": "attempt_000001",
                "attempts": [
                    {
                        "attempt_id": "attempt_000001",
                        "prompt_payload": prompt_ref.to_public_dict(),
                        "provider_attempt": {
                            "candidate_program_source_checksum": (
                                f"sha256:{source_digest}"
                            ),
                            "anti_degenerate_assessment": {
                                "schema_version": (
                                    "true_rl_agent.anti_degenerate_assessment.v4"
                                ),
                                "status": "pass",
                                "blocking": False,
                                "classification": "unknown",
                                "reason": "authoring_anchor_source_not_inline",
                                "comparison_anchor": "none",
                                "compares_to": None,
                                "candidate_source_sha256": source_digest,
                            },
                        },
                    }
                ],
            }
            self.assertIsNone(project_selected_authoring_mechanism_delta(evidence))

            evidence["attempts"][0]["provider_attempt"][
                "candidate_program_source_checksum"
            ] = "sha256:" + ("e" * 64)
            with self.assertRaisesRegex(ValueError, "source identity mismatch"):
                project_selected_authoring_mechanism_delta(evidence)

    def test_current_provider_attribution_is_exact_sealed_projection(self) -> None:
        anchor_source = "def build_training_components():\n    return None\n"
        anchor_digest = hashlib.sha256(anchor_source.encode("utf-8")).hexdigest()
        parent_program_source_checksum = "sha256:" + ("a" * 64)
        candidate_digest = "f" * 64
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-001")
            _bind_test_budget(journal)
            handle = journal.begin_attempt(
                _test_prompt_with_parent(
                    parent_program_source_checksum=(
                        parent_program_source_checksum
                    ),
                    current_best_source=anchor_source,
                )
            )
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-001",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                    "candidate_id": "candidate-001",
                    "candidate_changed_components": ["reward_callable"],
                    "candidate_program_source_checksum": (
                        f"sha256:{candidate_digest}"
                    ),
                    "anti_degenerate_assessment": {
                        "schema_version": (
                            "true_rl_agent.anti_degenerate_assessment.v4"
                        ),
                        "status": "pass",
                        "blocking": False,
                        "classification": "mechanism_delta_present",
                        "comparison_anchor": "current_best",
                        "compares_to": anchor_digest,
                        "candidate_source_sha256": candidate_digest,
                        "changed_mechanism_count": 1,
                        "changed_mechanisms": ["RewardTerm:progress_reward"],
                    },
                },
                continuation_state=_test_continuation(),
            )
            evidence_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="candidate_verified",
            )
            delta = {
                "schema_version": "true_rl_agent.candidate_mechanism_delta.v1",
                "comparison_anchor": "current_best",
                "compares_to": anchor_digest,
                "candidate_source_sha256": candidate_digest,
                "changed_mechanisms": ["RewardTerm:progress_reward"],
            }
            runtime_gate = {
                "status": "pass",
                "reason": "runner_activation_verified",
            }
            ablation = {"status": "not_run"}
            record = {
                "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
                "authoring_mode": PROVIDER_AUTHORING_MODE,
                "round_id": "round-001",
                "parent_program_source_checksum": (
                    parent_program_source_checksum
                ),
                "prompt_payload": handle.prompt_payload_ref.to_public_dict(),
                "private_artifacts": [evidence_ref.to_private_dict()],
                "candidate": {
                    "candidate_id": "candidate-001",
                    "changed_components": ["reward_callable"],
                    "program_source_checksum": f"sha256:{candidate_digest}",
                },
                "runtime_activation_gate": runtime_gate,
                "ablation_summary": ablation,
                "mechanism_attribution": build_mechanism_attribution_summary(
                    runtime_activation_gate=MechanismDecision(
                        runtime_gate["status"],
                        runtime_gate["reason"],
                    ),
                    ablation_summary=ablation,
                    authoring_mechanism_delta=delta,
                ),
            }
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    record,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                (),
            )

            v3_ref = _publish_v3_evidence_copy(evidence_ref)
            historical_v6 = copy.deepcopy(record)
            historical_v6["schema_version"] = LEDGER_RECORD_SCHEMA_V6
            historical_v6["private_artifacts"] = [v3_ref.to_private_dict()]
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    historical_v6,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                (),
            )

            for schema_version, source_record in (
                (LEDGER_RECORD_SCHEMA_V6, historical_v6),
                (LEDGER_RECORD_SCHEMA_VERSION, record),
            ):
                with self.subTest(
                    schema_version=schema_version,
                    binding="ledger_parent_drift",
                ):
                    misbound_parent = copy.deepcopy(source_record)
                    misbound_parent["parent_program_source_checksum"] = (
                        "sha256:" + ("b" * 64)
                    )
                    self.assertEqual(
                        validate_ledger_prompt_artifacts(
                            misbound_parent,
                            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                        ),
                        ("authoring_evidence_outcome_projection_mismatch",),
                    )

            forged_lineage_prompt = verify_prompt_payload(
                handle.prompt_payload_ref
            )
            forged_lineage_prompt["candidate_lineage"]["current_best"][
                "current_head_program_source_checksum"
            ] = "sha256:" + ("b" * 64)
            forged_prompt_ref = publish_prompt_payload(
                Path(tmp) / "forged-v6-lineage-prompt.json",
                forged_lineage_prompt,
            )
            forged_evidence = json.loads(
                Path(v3_ref.path).read_text(encoding="utf-8")
            )
            forged_evidence["attempts"][0][
                "prompt_payload"
            ] = forged_prompt_ref.to_public_dict()
            forged_evidence_ref = _publish_test_evidence(
                Path(tmp) / "forged-v6-lineage-evidence.json",
                forged_evidence,
            )
            forged_v6_lineage = copy.deepcopy(historical_v6)
            forged_v6_lineage["prompt_payload"] = (
                forged_prompt_ref.to_public_dict()
            )
            forged_v6_lineage["private_artifacts"] = [
                forged_evidence_ref.to_private_dict()
            ]
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    forged_v6_lineage,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                ("authoring_evidence_outcome_projection_mismatch",),
            )

            v7_with_v3 = copy.deepcopy(historical_v6)
            v7_with_v3["schema_version"] = LEDGER_RECORD_SCHEMA_VERSION
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    v7_with_v3,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                ("authoring_evidence_schema_not_current",),
            )

            for label, mutation in (
                ("status", lambda value: value.__setitem__("status", "causally_proven")),
                ("causal", lambda value: value.__setitem__("causal_claim", True)),
                (
                    "delta",
                    lambda value: value["authoring_mechanism_delta"].__setitem__(
                        "changed_mechanisms", ["RewardTerm:forged"]
                    ),
                ),
            ):
                with self.subTest(label=label):
                    tampered = copy.deepcopy(record)
                    mutation(tampered["mechanism_attribution"])
                    self.assertIn(
                        "authoring_evidence_outcome_projection_mismatch",
                        validate_ledger_prompt_artifacts(
                            tampered,
                            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                        ),
                    )

    def test_v6_true_cold_start_keeps_legacy_no_delta_compatibility(self) -> None:
        candidate_digest = "c" * 64
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-cold")
            _bind_test_budget(journal)
            handle = journal.begin_attempt(_test_prompt({}))
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-cold",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                    "candidate_id": "candidate-cold",
                    "candidate_changed_components": ["reward_callable"],
                    "candidate_program_source_checksum": (
                        f"sha256:{candidate_digest}"
                    ),
                    "anti_degenerate_assessment": {
                        "schema_version": (
                            "true_rl_agent.anti_degenerate_assessment.v4"
                        ),
                        "status": "warning",
                        "blocking": False,
                        "classification": "unknown",
                        "reason": "authoring_anchor_source_not_inline",
                        "comparison_anchor": "none",
                        "compares_to": None,
                        "candidate_source_sha256": candidate_digest,
                    },
                },
                continuation_state=_test_continuation(),
            )
            evidence_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="candidate_verified",
            )
            v3_ref = _publish_v3_evidence_copy(evidence_ref)
            record = {
                "schema_version": LEDGER_RECORD_SCHEMA_V6,
                "authoring_mode": PROVIDER_AUTHORING_MODE,
                "round_id": "round-cold",
                "parent_program_source_checksum": None,
                "prompt_payload": handle.prompt_payload_ref.to_public_dict(),
                "private_artifacts": [v3_ref.to_private_dict()],
                "candidate": {
                    "candidate_id": "candidate-cold",
                    "changed_components": ["reward_callable"],
                    "program_source_checksum": f"sha256:{candidate_digest}",
                },
                **_neutral_mechanism_fields(),
            }

            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    record,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                (),
            )

    def test_v7_non_provider_record_does_not_require_an_authoring_anchor(
        self,
    ) -> None:
        record = {
            "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
            "authoring_mode": NON_PROVIDER_AUTHORING_MODE,
            "round_id": "round-push-replay",
            **_neutral_mechanism_fields(),
        }

        self.assertEqual(
            validate_ledger_prompt_artifacts(
                record,
                expected_authoring_mode=NON_PROVIDER_AUTHORING_MODE,
            ),
            (),
        )

    def test_v6_r10_shape_with_parent_and_no_projected_delta_is_invalid(
        self,
    ) -> None:
        parent_checksum = "sha256:" + ("a" * 64)
        current_best_source = (
            "def build_training_components():\n    return 'current-best'\n"
        )
        failed_source = (
            "def build_training_components():\n    return 'name-too-long'\n"
        )
        failed_digest = hashlib.sha256(failed_source.encode("utf-8")).hexdigest()
        candidate_digest = "d" * 64
        initial_prompt = _test_prompt_with_parent(
            parent_program_source_checksum=parent_checksum,
            current_best_source=current_best_source,
        )
        repair_prompt = {
            "candidate_lineage": copy.deepcopy(
                initial_prompt["candidate_lineage"]
            ),
            "source_payloads": [
                {
                    "logical_id": LATEST_FAILED_SOURCE_ID,
                    "content": failed_source,
                    "content_sha256": failed_digest,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-r10")
            _bind_test_budget(journal)
            failed_handle = journal.begin_attempt(initial_prompt)
            journal.complete_attempt(
                failed_handle,
                {
                    "round_id": "round-r10",
                    "attempt": 1,
                    "attempt_id": failed_handle.attempt_id,
                    "candidate_id": "candidate-r10-long-name",
                    "candidate_changed_components": ["reward_callable"],
                    "candidate_program_source_checksum": "sha256:" + ("e" * 64),
                },
                continuation_state=_test_continuation(),
            )
            repaired_handle = journal.begin_attempt(repair_prompt)
            journal.complete_attempt(
                repaired_handle,
                {
                    "round_id": "round-r10",
                    "attempt": 2,
                    "attempt_id": repaired_handle.attempt_id,
                    "candidate_id": "candidate-r10-repaired-name",
                    "candidate_changed_components": ["reward_callable"],
                    "candidate_program_source_checksum": (
                        f"sha256:{candidate_digest}"
                    ),
                    "anti_degenerate_assessment": {
                        "schema_version": (
                            "true_rl_agent.anti_degenerate_assessment.v4"
                        ),
                        "status": "warning",
                        "blocking": False,
                        "classification": "unknown",
                        "reason": "authoring_anchor_source_not_inline",
                        "comparison_anchor": "none",
                        "compares_to": None,
                        "candidate_source_sha256": candidate_digest,
                    },
                },
                continuation_state=_test_continuation(),
            )
            evidence_ref = journal.publish_evidence(
                selected_attempt_id=repaired_handle.attempt_id,
                stop_reason="candidate_verified",
            )
            v3_ref = _publish_v3_evidence_copy(evidence_ref)
            record = {
                "schema_version": LEDGER_RECORD_SCHEMA_V6,
                "authoring_mode": PROVIDER_AUTHORING_MODE,
                "round_id": "round-r10",
                "parent_program_source_checksum": parent_checksum,
                "prompt_payload": repaired_handle.prompt_payload_ref.to_public_dict(),
                "private_artifacts": [v3_ref.to_private_dict()],
                "candidate": {
                    "candidate_id": "candidate-r10-repaired-name",
                    "changed_components": ["reward_callable"],
                    "program_source_checksum": f"sha256:{candidate_digest}",
                },
                **_neutral_mechanism_fields(),
            }

            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    record,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                ("authoring_evidence_outcome_projection_mismatch",),
            )

            r2 = valid_accepted_ledger_record(
                round_id="round-r2",
                candidate_id="candidate-r2",
                public_score=0.50,
            )
            _rebind_record_program_source(r2, parent_checksum)
            initial_current_best = _accepted_current_best(r2)

            r10 = valid_accepted_ledger_record(
                round_id="round-r10",
                candidate_id="candidate-r10-repaired-name",
                public_score=0.60,
            )
            _rebind_record_program_source(
                r10,
                f"sha256:{candidate_digest}",
            )
            _bind_decision_to_current_best(r10, initial_current_best)
            for field_name in (
                "schema_version",
                "authoring_mode",
                "parent_program_source_checksum",
                "prompt_payload",
                "private_artifacts",
                "runtime_activation_gate",
                "ablation_summary",
                "mechanism_attribution",
            ):
                r10[field_name] = copy.deepcopy(record[field_name])

            r11_parent_source = (
                "def build_training_components():\n    return 'r10'\n"
            )
            r11_anchor_digest = hashlib.sha256(
                r11_parent_source.encode("utf-8")
            ).hexdigest()
            r11_candidate_digest = "f" * 64
            r11_journal = RoundAuthoringJournal(
                Path(tmp) / "round-r11",
                round_id="round-r11",
            )
            _bind_test_budget(r11_journal)
            r11_handle = r11_journal.begin_attempt(
                _test_prompt_with_parent(
                    parent_program_source_checksum=(
                        f"sha256:{candidate_digest}"
                    ),
                    current_best_source=r11_parent_source,
                )
            )
            r11_journal.complete_attempt(
                r11_handle,
                {
                    "round_id": "round-r11",
                    "attempt": 1,
                    "attempt_id": r11_handle.attempt_id,
                    "candidate_id": "candidate-r11",
                    "candidate_changed_components": ["reward_callable"],
                    "candidate_program_source_checksum": (
                        f"sha256:{r11_candidate_digest}"
                    ),
                    "anti_degenerate_assessment": {
                        "schema_version": (
                            "true_rl_agent.anti_degenerate_assessment.v4"
                        ),
                        "status": "pass",
                        "blocking": False,
                        "classification": "mechanism_delta_present",
                        "comparison_anchor": "current_best",
                        "compares_to": r11_anchor_digest,
                        "candidate_source_sha256": r11_candidate_digest,
                        "changed_mechanism_count": 1,
                        "changed_mechanisms": ["RewardTerm:r11_reward"],
                    },
                },
                continuation_state=_test_continuation(),
            )
            r11_evidence_ref = r11_journal.publish_evidence(
                selected_attempt_id=r11_handle.attempt_id,
                stop_reason="candidate_verified",
            )
            r11_v3_ref = _publish_v3_evidence_copy(r11_evidence_ref)
            r11_delta = {
                "schema_version": "true_rl_agent.candidate_mechanism_delta.v1",
                "comparison_anchor": "current_best",
                "compares_to": r11_anchor_digest,
                "candidate_source_sha256": r11_candidate_digest,
                "changed_mechanisms": ["RewardTerm:r11_reward"],
            }
            r11 = valid_accepted_ledger_record(
                round_id="round-r11",
                candidate_id="candidate-r11",
                public_score=0.70,
            )
            _rebind_record_program_source(
                r11,
                f"sha256:{r11_candidate_digest}",
            )
            _bind_decision_to_current_best(r11, _accepted_current_best(r10))
            r11_runtime_gate = {
                "status": "pass",
                "reason": "runner_activation_verified",
            }
            r11_ablation = {"status": "not_run"}
            r11.update(
                {
                    "schema_version": LEDGER_RECORD_SCHEMA_V6,
                    "authoring_mode": PROVIDER_AUTHORING_MODE,
                    "parent_program_source_checksum": (
                        f"sha256:{candidate_digest}"
                    ),
                    "prompt_payload": r11_handle.prompt_payload_ref.to_public_dict(),
                    "private_artifacts": [r11_v3_ref.to_private_dict()],
                    "runtime_activation_gate": r11_runtime_gate,
                    "ablation_summary": r11_ablation,
                    "mechanism_attribution": (
                        build_mechanism_attribution_summary(
                            runtime_activation_gate=MechanismDecision(
                                r11_runtime_gate["status"],
                                r11_runtime_gate["reason"],
                            ),
                            ablation_summary=r11_ablation,
                            authoring_mechanism_delta=r11_delta,
                        )
                    ),
                }
            )

            replay = replay_acceptance_ledger(
                [r10, r11],
                initial_current_best=initial_current_best,
                expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            )

            self.assertIn(
                "authoring_evidence_outcome_projection_mismatch",
                replay.entries[0].sequence_reasons,
            )
            self.assertIn(
                "acceptance_replay_parent_program_source_checksum_mismatch",
                replay.entries[1].sequence_reasons,
            )
            self.assertEqual(
                replay.current_best.program_source_checksum,
                parent_checksum,
            )

    def test_prompt_artifact_is_exact_provider_bytes_and_write_once(self) -> None:
        payload = {
            "task": {"task_id": "generic-task"},
            "schema_version": "test.prompt.v1",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "round-001" / "prompt_payload.json"
            ref = publish_prompt_payload(path, payload)
            original = path.read_bytes()

            self.assertEqual(
                original,
                canonical_prompt_json(payload).encode("utf-8"),
            )
            self.assertEqual(ref.path, str(path.resolve()))
            self.assertEqual(ref.size_bytes, len(original))
            self.assertEqual(
                ref.checksum,
                "sha256:" + hashlib.sha256(original).hexdigest(),
            )
            self.assertEqual(verify_prompt_payload(ref), payload)

            with self.assertRaises(FileExistsError):
                publish_prompt_payload(path, payload)
            self.assertEqual(path.read_bytes(), original)

    def test_prompt_artifact_rejects_byte_or_reference_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt_payload.json"
            ref = publish_prompt_payload(
                path,
                {"schema_version": "test.prompt.v1", "public_only": True},
            )
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_prompt_payload(ref)

            mapping = ref.to_public_dict()
            mapping["path"] = str(Path(tmp) / "missing.json")
            missing = PromptPayloadRef.from_mapping(mapping)
            with self.assertRaisesRegex(ValueError, "regular file"):
                verify_prompt_payload(missing)

    def test_prompt_payload_limit_is_enforced_before_publish_or_verify(self) -> None:
        oversized_size = MAX_AUTHORING_PROMPT_PAYLOAD_BYTES + 1
        with self.assertRaisesRegex(ValueError, "authoring payload limit"):
            PromptPayloadRef(
                path="unused.json",
                checksum="sha256:" + "0" * 64,
                size_bytes=oversized_size,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            publish_path = root / "publish.json"
            oversized_text = "x" * MAX_AUTHORING_PROMPT_PAYLOAD_BYTES
            with self.assertRaisesRegex(ValueError, "authoring payload limit"):
                publish_prompt_payload(publish_path, {"text": oversized_text})
            self.assertFalse(publish_path.exists())

            artifact_path = root / "artifact.json"
            artifact_path.write_bytes(b"{" + b" " * oversized_size)
            forged = object.__new__(PromptPayloadRef)
            object.__setattr__(forged, "path", str(artifact_path))
            object.__setattr__(forged, "checksum", "sha256:" + "0" * 64)
            object.__setattr__(forged, "size_bytes", 1)
            object.__setattr__(forged, "visibility", "public_prompt_payload")
            object.__setattr__(forged, "schema_version", "true_rl_agent.prompt_payload_ref.v1")
            with self.assertRaisesRegex(ValueError, "exceeds maximum size"):
                verify_prompt_payload(forged)

    def test_journal_resume_rejects_oversized_prompt_before_json_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            attempt_dir = root / "authoring_attempts" / "attempt_000001"
            attempt_dir.mkdir(parents=True)
            (attempt_dir / "prompt_payload.json").write_bytes(
                b"{" + b" " * MAX_AUTHORING_PROMPT_PAYLOAD_BYTES
            )
            with self.assertRaisesRegex(ValueError, "exceeds maximum size"):
                journal.snapshot()

    def test_authoring_artifact_type_caps_are_enforced_at_write_and_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            with self.assertRaisesRegex(ValueError, "between 1 and"):
                journal.bind_budget(
                    max_provider_attempts=MAX_AUTHORING_PROVIDER_ATTEMPTS + 1,
                    max_source_requests_per_round=0,
                    max_source_ids_per_request=2,
                    max_source_request_bytes_per_round=0,
                )

            _bind_test_budget(journal)
            budget_path = root / "authoring_budget.json"
            budget_path.write_bytes(
                b"x" * (MAX_AUTHORING_BUDGET_ARTIFACT_BYTES + 1)
            )
            with self.assertRaisesRegex(ValueError, "exceeds maximum size"):
                journal.begin_attempt(_test_prompt({}))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            handle = journal.begin_attempt(_test_prompt({}))
            Path(handle.result_path).write_bytes(
                b"x" * (MAX_AUTHORING_ATTEMPT_RESULT_BYTES + 1)
            )
            with self.assertRaisesRegex(ValueError, "exceeds maximum size"):
                journal.snapshot()

        with self.assertRaisesRegex(ValueError, "typed artifact limit"):
            AuthoringEvidenceRef(
                path="unused.json",
                checksum="sha256:" + "0" * 64,
                size_bytes=MAX_AUTHORING_EVIDENCE_ABSOLUTE_BYTES + 1,
            )

    def test_authoring_evidence_binds_all_attempts_and_selected_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            refs = []
            records = [
                {
                    "round_id": "round-001",
                    "attempt": index,
                    "provider_status": "fail" if index == 1 else "pass",
                }
                for index in (1, 2)
            ]
            handles = []
            for index, record in enumerate(records, start=1):
                handle = journal.begin_attempt(
                    _test_prompt(
                        {"schema_version": "test.prompt.v1", "attempt": index}
                    )
                )
                record["attempt_id"] = handle.attempt_id
                journal.complete_attempt(
                    handle,
                    record,
                    continuation_state=_test_continuation(),
                )
                handles.append(handle)
                refs.append(handle.prompt_payload_ref)
            evidence_ref = journal.publish_evidence(
                selected_attempt_id=handles[-1].attempt_id,
                stop_reason="candidate_verified",
            )
            evidence = verify_authoring_evidence(
                evidence_ref,
                expected_round_id="round-001",
                expected_selected_prompt=refs[-1],
            )

            self.assertEqual(len(evidence["attempts"]), 2)
            self.assertEqual(
                evidence["attempts"][0]["prompt_payload"],
                refs[0].to_public_dict(),
            )
            anchor = verify_round_authoring_anchor(
                RoundAuthoringAnchorRef.from_mapping(
                    evidence["round_authoring_anchor"]
                ),
                expected_round_id="round-001",
            )
            self.assertEqual(anchor["prompt_payload"], refs[0].to_public_dict())
            with self.assertRaisesRegex(ValueError, "selected prompt mismatch"):
                verify_authoring_evidence(
                    evidence_ref,
                    expected_round_id="round-001",
                    expected_selected_prompt=refs[0],
                )

            missing = copy.deepcopy(evidence)
            missing.pop("round_authoring_anchor")
            missing_ref = _publish_test_evidence(
                root / "forged-anchor-missing.json",
                missing,
            )
            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                verify_authoring_evidence(
                    missing_ref,
                    expected_round_id="round-001",
                    expected_selected_prompt=refs[-1],
                )

            forged_anchor = copy.deepcopy(anchor)
            forged_anchor["prompt_payload"] = refs[1].to_public_dict()
            forged_anchor_base = {
                key: value
                for key, value in forged_anchor.items()
                if key != "contract_hash"
            }
            forged_anchor["contract_hash"] = "sha256:" + hashlib.sha256(
                canonical_json_bytes(forged_anchor_base)
            ).hexdigest()
            forged_anchor_path = root / "forged-round-anchor.json"
            forged_anchor_bytes = canonical_json_bytes(
                forged_anchor,
                trailing_newline=True,
            )
            forged_anchor_path.write_bytes(forged_anchor_bytes)
            forged = copy.deepcopy(evidence)
            forged["round_authoring_anchor"] = RoundAuthoringAnchorRef(
                path=str(forged_anchor_path),
                checksum=(
                    "sha256:"
                    + hashlib.sha256(forged_anchor_bytes).hexdigest()
                ),
                size_bytes=len(forged_anchor_bytes),
            ).to_private_dict()
            forged_ref = _publish_test_evidence(
                root / "forged-anchor-different-prompt.json",
                forged,
            )
            with self.assertRaisesRegex(ValueError, "anchor.*mismatch"):
                verify_authoring_evidence(
                    forged_ref,
                    expected_round_id="round-001",
                    expected_selected_prompt=refs[-1],
                )

    def test_authoring_evidence_requires_the_final_attempt_as_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            handles = []
            for index in (1, 2):
                handle = journal.begin_attempt(
                    _test_prompt(
                        {"schema_version": "test.prompt.v1", "attempt": index}
                    )
                )
                journal.complete_attempt(
                    handle,
                    {
                        "round_id": "round-001",
                        "attempt": index,
                        "attempt_id": handle.attempt_id,
                        "provider_status": "pass",
                    },
                    continuation_state=_test_continuation(),
                )
                handles.append(handle)

            with self.assertRaisesRegex(ValueError, "final attempt"):
                journal.publish_evidence(
                    selected_attempt_id=handles[0].attempt_id,
                    stop_reason="candidate_verified",
                )

            evidence_ref = journal.publish_evidence(
                selected_attempt_id=handles[-1].attempt_id,
                stop_reason="candidate_verified",
            )
            payload = json.loads(
                Path(evidence_ref.path).read_text(encoding="utf-8")
            )
            payload["selected_attempt_id"] = handles[0].attempt_id
            forged_path = root / "forged-earlier-selection.json"
            forged_bytes = canonical_json_bytes(payload, trailing_newline=True)
            forged_path.write_bytes(forged_bytes)
            forged_ref = AuthoringEvidenceRef(
                path=str(forged_path),
                checksum="sha256:" + hashlib.sha256(forged_bytes).hexdigest(),
                size_bytes=len(forged_bytes),
            )
            with self.assertRaisesRegex(ValueError, "final attempt"):
                verify_authoring_evidence(
                    forged_ref,
                    expected_round_id="round-001",
                    expected_selected_prompt=handles[0].prompt_payload_ref,
                )

    def test_existing_attempt_without_anchor_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            journal.begin_attempt(
                _test_prompt(
                    {"schema_version": "test.prompt.v1", "public_only": True}
                )
            )
            (root / "authoring_anchor.json").unlink()

            with self.assertRaisesRegex(ValueError, "anchor is missing"):
                journal.snapshot()
            with self.assertRaisesRegex(ValueError, "anchor is missing"):
                journal.begin_attempt(
                    _test_prompt(
                        {
                            "schema_version": "test.prompt.v1",
                            "public_only": True,
                        }
                    )
                )

    def test_interrupted_before_prompt_can_freeze_next_attempt_anchor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            interrupted_dir = (
                root / "authoring_attempts" / "attempt_000001"
            )
            interrupted_dir.mkdir(parents=True)

            snapshot = journal.snapshot()
            self.assertEqual(snapshot.reserved_attempt_count, 1)
            self.assertIsNone(snapshot.round_authoring_anchor)
            handle = journal.begin_attempt(
                _test_prompt(
                    {"schema_version": "test.prompt.v1", "public_only": True}
                )
            )
            resumed = journal.snapshot()

            self.assertEqual(handle.attempt_id, "attempt_000002")
            self.assertEqual(
                [item["status"] for item in resumed.attempts],
                ["interrupted_before_prompt", "interrupted_after_prompt"],
            )
            self.assertEqual(
                resumed.round_authoring_anchor["prompt_payload"],
                handle.prompt_payload_ref.to_public_dict(),
            )

    def test_anchor_contract_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            journal.begin_attempt(
                _test_prompt(
                    {"schema_version": "test.prompt.v1", "public_only": True}
                )
            )
            anchor_path = root / "authoring_anchor.json"
            anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
            anchor["comparison_anchor"] = "current_best"
            anchor_path.write_bytes(
                canonical_json_bytes(anchor, trailing_newline=True)
            )

            with self.assertRaisesRegex(ValueError, "contract hash mismatch"):
                journal.snapshot()

    def test_round_authoring_anchor_reads_are_bounded_before_json_parse(
        self,
    ) -> None:
        max_anchor_bytes = 16 * 1024
        oversized = b"not-json" + (b"x" * max_anchor_bytes)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            journal.begin_attempt(_test_prompt({}))
            anchor_path = root / "authoring_anchor.json"
            anchor_path.write_bytes(oversized)

            with self.assertRaisesRegex(ValueError, "exceeds maximum size"):
                journal.snapshot()

            with self.assertRaisesRegex(ValueError, "at most 16384"):
                RoundAuthoringAnchorRef(
                    path=str(anchor_path),
                    checksum="sha256:" + hashlib.sha256(oversized).hexdigest(),
                    size_bytes=len(oversized),
                )

            bounded_claim = RoundAuthoringAnchorRef(
                path=str(anchor_path),
                checksum="sha256:" + hashlib.sha256(oversized).hexdigest(),
                size_bytes=max_anchor_bytes,
            )
            with self.assertRaisesRegex(ValueError, "exceeds maximum size"):
                verify_round_authoring_anchor(
                    bounded_claim,
                    expected_round_id="round-001",
                )

    def test_parent_without_verifiable_source_anchor_fails_before_provider(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            parent_checksum = "sha256:" + ("a" * 64)

            with self.assertRaisesRegex(
                ValueError,
                "parent but no verifiable source anchor",
            ):
                journal.begin_attempt(
                    {
                        "candidate_lineage": {
                            "parent_program_source_checksum": parent_checksum,
                        },
                        "source_payloads": [],
                    }
                )

            self.assertFalse((root / "authoring_anchor.json").exists())
            with self.assertRaisesRegex(ValueError, "anchor is missing"):
                journal.snapshot()

    def test_parent_cannot_fall_back_to_best_observed_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-001")
            _bind_test_budget(journal)
            observed_source = "def build_training_components():\n    return None\n"
            observed_hash = hashlib.sha256(
                observed_source.encode("utf-8")
            ).hexdigest()

            with self.assertRaisesRegex(
                ValueError,
                "parent must use the current-best source anchor",
            ):
                journal.begin_attempt(
                    {
                        "candidate_lineage": {
                            "parent_program_source_checksum": (
                                "sha256:" + ("a" * 64)
                            ),
                            "best_observed_experiment": {
                                "source_logical_id": BEST_OBSERVED_SOURCE_ID,
                            },
                        },
                        "source_payloads": [
                            {
                                "logical_id": BEST_OBSERVED_SOURCE_ID,
                                "content": observed_source,
                                "content_sha256": observed_hash,
                            }
                        ],
                    }
                )

    def test_round_anchor_rejects_duplicate_current_best_payloads(self) -> None:
        source = "def build_training_components():\n    return 'current'\n"
        prompt = _test_prompt_with_parent(
            parent_program_source_checksum="sha256:" + ("a" * 64),
            current_best_source=source,
        )
        prompt["source_payloads"].append(dict(prompt["source_payloads"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-001")
            _bind_test_budget(journal)

            with self.assertRaisesRegex(ValueError, "must be unique"):
                journal.begin_attempt(prompt)

    def test_round_anchor_rejects_duplicate_best_observed_payloads(self) -> None:
        source = "def build_training_components():\n    return 'observed'\n"
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        payload = {
            "logical_id": BEST_OBSERVED_SOURCE_ID,
            "content": source,
            "content_sha256": digest,
        }
        prompt = {
            "candidate_lineage": {
                "parent_program_source_checksum": None,
                "best_observed_experiment": {
                    "source_logical_id": BEST_OBSERVED_SOURCE_ID,
                },
            },
            "source_payloads": [payload, dict(payload)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-cold")
            _bind_test_budget(journal)

            with self.assertRaisesRegex(ValueError, "must be unique"):
                journal.begin_attempt(prompt)

    def test_cold_start_cannot_claim_a_current_best_source(self) -> None:
        current_best_source = (
            "def build_training_components():\n    return 'forged-best'\n"
        )
        prompt = _test_prompt_with_parent(
            parent_program_source_checksum="sha256:" + ("a" * 64),
            current_best_source=current_best_source,
        )
        prompt["candidate_lineage"][
            "parent_program_source_checksum"
        ] = None
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-cold")
            _bind_test_budget(journal)

            with self.assertRaisesRegex(
                ValueError,
                "cold-start.*current-best",
            ):
                journal.begin_attempt(prompt)

    def test_cold_start_may_compare_with_a_nonaccepted_observation(self) -> None:
        observed_source = (
            "def build_training_components():\n    return 'observed'\n"
        )
        observed_hash = hashlib.sha256(
            observed_source.encode("utf-8")
        ).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-cold")
            _bind_test_budget(journal)
            handle = journal.begin_attempt(
                {
                    "candidate_lineage": {
                        "parent_program_source_checksum": None,
                        "best_observed_experiment": {
                            "source_logical_id": BEST_OBSERVED_SOURCE_ID,
                        },
                    },
                    "source_payloads": [
                        {
                            "logical_id": BEST_OBSERVED_SOURCE_ID,
                            "content": observed_source,
                            "content_sha256": observed_hash,
                        }
                    ],
                }
            )
            anchor = verify_round_authoring_anchor(
                journal.snapshot().round_authoring_anchor_ref,
                expected_round_id="round-cold",
            )

        self.assertEqual(
            anchor["prompt_payload"],
            handle.prompt_payload_ref.to_public_dict(),
        )
        self.assertIsNone(anchor["parent_program_source_checksum"])
        self.assertEqual(
            anchor["comparison_anchor"],
            "best_observed_nonaccepted",
        )

    def test_v2_evidence_is_read_only_compatible_with_v4_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(
                Path(tmp),
                round_id="round-001",
            )
            _bind_test_budget(journal)
            handle = journal.begin_attempt(
                _test_prompt(
                    {"schema_version": "test.prompt.v1", "public_only": True}
                )
            )
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-001",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                },
                continuation_state=_test_continuation(),
            )
            current_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="candidate_verified",
            )
            v2_ref = _publish_v2_evidence_copy(current_ref)

            evidence = verify_authoring_evidence(
                v2_ref,
                expected_round_id="round-001",
                expected_selected_prompt=handle.prompt_payload_ref,
            )
            self.assertEqual(
                evidence["schema_version"],
                AUTHORING_EVIDENCE_SCHEMA_V2,
            )

            historical = {
                "schema_version": LEDGER_RECORD_SCHEMA_V4,
                "authoring_mode": PROVIDER_AUTHORING_MODE,
                "round_id": "round-001",
                "prompt_payload": handle.prompt_payload_ref.to_public_dict(),
                "private_artifacts": [v2_ref.to_private_dict()],
                "candidate": None,
            }
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    historical,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                (),
            )

            current = {
                **historical,
                "schema_version": LEDGER_RECORD_SCHEMA_V5,
            }
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    current,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                ("authoring_evidence_schema_not_current",),
            )

    def test_v3_evidence_remains_verifiable_after_anchor_schema_upgrade(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-001")
            _bind_test_budget(journal)
            handle = journal.begin_attempt(
                _test_prompt(
                    {"schema_version": "test.prompt.v1", "public_only": True}
                )
            )
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-001",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                },
                continuation_state=_test_continuation(),
            )
            current_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="candidate_verified",
            )
            v3_ref = _publish_v3_evidence_copy(current_ref)

            evidence = verify_authoring_evidence(
                v3_ref,
                expected_round_id="round-001",
                expected_selected_prompt=handle.prompt_payload_ref,
            )

            self.assertEqual(
                evidence["schema_version"],
                AUTHORING_EVIDENCE_SCHEMA_V3,
            )
            self.assertNotIn("round_authoring_anchor", evidence)

    def test_v2_evidence_compatibility_does_not_resume_old_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(
                Path(tmp),
                round_id="round-001",
            )
            _bind_test_budget(journal)
            handle = journal.begin_attempt(
                _test_prompt(
                    {"schema_version": "test.prompt.v1", "public_only": True}
                )
            )
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-001",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                },
                continuation_state=_test_continuation(),
            )
            result_path = Path(handle.result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["schema_version"] = (
                "true_rl_agent.authoring_attempt_result.v1"
            )
            result_path.write_bytes(
                canonical_json_bytes(result, trailing_newline=True)
            )

            with self.assertRaisesRegex(
                ValueError,
                "authoring attempt result identity mismatch",
            ):
                journal.snapshot()

    def test_journal_and_evidence_enforce_the_sealed_attempt_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            journal.bind_budget(
                max_provider_attempts=2,
                max_source_requests_per_round=0,
                max_source_ids_per_request=2,
                max_source_request_bytes_per_round=0,
            )
            handles = []
            for index in (1, 2):
                handle = journal.begin_attempt(
                    _test_prompt(
                        {"schema_version": "test.prompt.v1", "attempt": index}
                    )
                )
                journal.complete_attempt(
                    handle,
                    {
                        "round_id": "round-001",
                        "attempt": index,
                        "attempt_id": handle.attempt_id,
                    },
                    continuation_state=_test_continuation(),
                )
                handles.append(handle)
            with self.assertRaisesRegex(ValueError, "budget is exhausted"):
                journal.begin_attempt(
                    _test_prompt(
                        {"schema_version": "test.prompt.v1", "attempt": 3}
                    )
                )

            evidence_ref = journal.publish_evidence(
                selected_attempt_id=handles[-1].attempt_id,
                stop_reason="candidate_verified",
            )
            evidence_path = Path(evidence_ref.path)
            tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
            budget = tampered["budget_contract"]
            budget["max_provider_attempts"] = 1
            base = {
                key: value
                for key, value in budget.items()
                if key != "contract_hash"
            }
            budget["contract_hash"] = "sha256:" + hashlib.sha256(
                canonical_json_bytes(base)
            ).hexdigest()
            encoded = canonical_json_bytes(tampered, trailing_newline=True)
            evidence_path.write_bytes(encoded)
            rehashed = AuthoringEvidenceRef(
                path=str(evidence_path),
                checksum="sha256:" + hashlib.sha256(encoded).hexdigest(),
                size_bytes=len(encoded),
            )

            with self.assertRaisesRegex(ValueError, "attempt budget"):
                verify_authoring_evidence(
                    rehashed,
                    expected_round_id="round-001",
                    expected_selected_prompt=handles[-1].prompt_payload_ref,
                )

    def test_evidence_binds_the_per_request_source_id_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(Path(tmp), round_id="round-001")
            _bind_test_budget(journal)
            handle = journal.begin_attempt(
                _test_prompt(
                    {"schema_version": "test.prompt.v1", "public_only": True}
                )
            )
            provider_failure = {
                "schema_version": "true_rl_agent.provider_failure.v1",
                "category": "source_request_denied",
                "reason": "final source request",
                "details": {},
            }
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-001",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                    "provider_status": "source_request",
                    "provider_failure": provider_failure,
                    "source_request_result": {
                        "status": "source_request_denied",
                        "returned_logical_ids": [],
                        "denied_logical_ids": [
                            {
                                "logical_id": "api/candidate_components.py",
                                "reason": "source_request_id_limit_exceeded",
                            }
                        ],
                        "returned_bytes": 0,
                        "budget": {
                            "source_request_count": 1,
                            "max_source_requests_per_round": 2,
                            "max_source_ids_per_request": 2,
                            "cumulative_returned_bytes": 0,
                            "max_source_request_bytes_per_round": 48 * 1024,
                        },
                    },
                },
                continuation_state=_test_continuation(),
            )
            evidence_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="provider_failed",
                provider_failure=provider_failure,
            )
            v3_ref = _publish_v3_evidence_copy(evidence_ref)
            v3_evidence = json.loads(
                Path(v3_ref.path).read_text(encoding="utf-8")
            )
            historical_failure = {
                "schema_version": LEDGER_RECORD_SCHEMA_V6,
                "authoring_mode": PROVIDER_AUTHORING_MODE,
                "round_id": "round-001",
                "parent_program_source_checksum": None,
                "prompt_payload": handle.prompt_payload_ref.to_public_dict(),
                "private_artifacts": [v3_ref.to_private_dict()],
                "candidate": None,
                "provider_failure": provider_failure,
                "source_request_feedback": v3_evidence[
                    "terminal_outcome"
                ]["source_request_feedback"],
            }
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    historical_failure,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                (),
            )
            evidence_path = Path(evidence_ref.path)
            tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
            tampered["attempts"][0]["provider_attempt"][
                "source_request_result"
            ]["budget"]["max_source_ids_per_request"] = 999
            tampered["terminal_outcome"]["source_request_feedback"][
                "budget"
            ]["max_source_ids_per_request"] = 999
            encoded = canonical_json_bytes(tampered, trailing_newline=True)
            evidence_path.write_bytes(encoded)
            rehashed = AuthoringEvidenceRef(
                path=str(evidence_path),
                checksum="sha256:" + hashlib.sha256(encoded).hexdigest(),
                size_bytes=len(encoded),
            )

            with self.assertRaisesRegex(
                ValueError,
                "source request usage differs from the sealed budget",
            ):
                verify_authoring_evidence(
                    rehashed,
                    expected_round_id="round-001",
                    expected_selected_prompt=handle.prompt_payload_ref,
                )

    def test_ledger_prompt_artifact_validation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = RoundAuthoringJournal(root, round_id="round-001")
            _bind_test_budget(journal)
            handle = journal.begin_attempt(
                _test_prompt(
                    {"schema_version": "test.prompt.v1", "public_only": True}
                ),
            )
            prompt_ref = handle.prompt_payload_ref
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-001",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                },
                continuation_state=_test_continuation(),
            )
            evidence_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="candidate_verified",
            )
            record = {
                "schema_version": LEDGER_RECORD_SCHEMA_V5,
                "authoring_mode": PROVIDER_AUTHORING_MODE,
                "round_id": "round-001",
                "prompt_payload": prompt_ref.to_public_dict(),
                "private_artifacts": [evidence_ref.to_private_dict()],
            }
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    record,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                (),
            )

            outcome_tampered = copy.deepcopy(record)
            outcome_tampered["provider_failure"] = {
                "schema_version": "true_rl_agent.provider_failure.v1",
                "category": "forged",
                "reason": "forged",
                "details": {},
            }
            self.assertIn(
                "authoring_evidence_outcome_projection_mismatch",
                validate_ledger_prompt_artifacts(
                    outcome_tampered,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
            )

            missing = copy.deepcopy(record)
            missing["prompt_payload"]["path"] = str(root / "missing.json")
            reasons = validate_ledger_prompt_artifacts(
                missing,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
            )
            self.assertIn("prompt_artifact_content_invalid", reasons)
            self.assertIn("authoring_evidence_content_invalid", reasons)

            deleted = copy.deepcopy(record)
            deleted.pop("prompt_payload")
            deleted["private_artifacts"] = []
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    deleted,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                (
                    "prompt_artifact_ref_missing",
                    "authoring_evidence_ref_missing",
                ),
            )

            legacy = copy.deepcopy(record)
            legacy["prompt_payload"] = {
                "path": prompt_ref.path,
                "visibility": "public_prompt_payload",
            }
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    legacy,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                ("prompt_artifact_ref_invalid",),
            )

            released_v2 = copy.deepcopy(legacy)
            released_v2["schema_version"] = LEDGER_RECORD_SCHEMA_V2
            released_v2.pop("authoring_mode")
            released_v2["private_artifacts"] = []
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    released_v2,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                ("authoring_evidence_legacy_unverified",),
            )
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    released_v2,
                    expected_authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                ),
                ("prompt_artifact_ref_unexpected",),
            )
            legacy_non_provider = copy.deepcopy(released_v2)
            legacy_non_provider.pop("prompt_payload")
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    legacy_non_provider,
                    expected_authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                ),
                (),
            )

            non_provider = copy.deepcopy(record)
            non_provider["authoring_mode"] = NON_PROVIDER_AUTHORING_MODE
            non_provider.pop("prompt_payload")
            non_provider["private_artifacts"] = []
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    non_provider,
                    expected_authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                ),
                (),
            )
            self.assertEqual(
                validate_ledger_prompt_artifacts(
                    non_provider,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                ),
                ("authoring_evidence_mode_mismatch",),
            )

            for unsupported in (LEDGER_RECORD_SCHEMA_V3, "ledger_record.v999"):
                unknown = copy.deepcopy(record)
                unknown["schema_version"] = unsupported
                with self.subTest(unsupported=unsupported):
                    with self.assertRaisesRegex(ValueError, "unsupported"):
                        validate_ledger_prompt_artifacts(
                            unknown,
                            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                        )

    def test_history_refuses_v2_prefix_without_fabricating_attempts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = valid_accepted_ledger_record(
                round_id="round-001",
                candidate_id="candidate-legacy",
                public_score=0.50,
            )
            legacy["schema_version"] = LEDGER_RECORD_SCHEMA_V2
            legacy.pop("authoring_mode", None)
            legacy["prompt_payload"] = {
                "path": str(root / "legacy-prompt.json"),
                "visibility": "public_prompt_payload",
            }

            journal = RoundAuthoringJournal(
                root / "round-002",
                round_id="round-002",
            )
            _bind_test_budget(journal)
            handle = journal.begin_attempt(
                _test_prompt(
                    {"schema_version": "test.prompt.v1", "public_only": True}
                ),
            )
            prompt_ref = handle.prompt_payload_ref
            journal.complete_attempt(
                handle,
                {
                    "round_id": "round-002",
                    "attempt": 1,
                    "attempt_id": handle.attempt_id,
                    "candidate_id": "candidate-bound",
                    "candidate_changed_components": ["reward_callable"],
                },
                continuation_state=_test_continuation(),
            )
            evidence_ref = journal.publish_evidence(
                selected_attempt_id=handle.attempt_id,
                stop_reason="candidate_verified",
            )
            bound = valid_accepted_ledger_record(
                round_id="round-002",
                candidate_id="candidate-bound",
                public_score=0.60,
            )
            bound["schema_version"] = LEDGER_RECORD_SCHEMA_V5
            bound["authoring_mode"] = PROVIDER_AUTHORING_MODE
            bound["prompt_payload"] = prompt_ref.to_public_dict()
            bound["private_artifacts"] = [evidence_ref.to_private_dict()]

            with self.assertRaisesRegex(
                ValueError,
                "authoring_evidence_legacy_unverified",
            ):
                _authoring_history([legacy, bound])

            history = _authoring_history([bound])

            self.assertEqual(history["bound_round_count"], 1)
            self.assertEqual(history["interrupted_attempt_count"], 0)
            self.assertEqual(len(history["evidence_paths"]), 1)
            self.assertEqual(len(history["attempt_records"]), 1)
            self.assertEqual(
                history["attempt_records"][0]["runner_status"],
                "pass",
            )


if __name__ == "__main__":
    unittest.main()
