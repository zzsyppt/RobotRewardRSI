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

from tests.fakes.decision_records import (  # noqa: E402
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_EVALUATION_BINDING,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from tests.fakes.fake_components import FakeTaskPlugin, FakeVerifier  # noqa: E402
from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop  # noqa: E402
from true_rl_agent.agent_core.authoring_response import (  # noqa: E402
    CandidateDraftResponse,
    SourceRequestResponse,
)
from true_rl_agent.agent_core.provider_errors import (
    ProviderCandidateError,  # noqa: E402
)
from true_rl_agent.candidates.schema import (  # noqa: E402
    CandidateChangeSet,
    CandidatePackageRef,
)
from true_rl_agent.memory import (
    PublicContextBuilder as _PublicContextBuilder,  # noqa: E402
)
from true_rl_agent.prompting.runner_capability_schemas import (
    CURRICULUM_DESIGN_CAPABILITY_ID,  # noqa: E402
)
from true_rl_agent.prompting.source_catalog import (  # noqa: E402
    LATEST_FAILED_SOURCE_ID,
    prompt_source_catalog_logical_ids,
)
from true_rl_agent.tasks.component_smoke import (  # noqa: E402
    ComponentSmokeFixture,
    SmokeTensor,
)
from true_rl_agent.tasks.push_recovery.reward_inputs import (
    PushRecoveryRewardInputs,  # noqa: E402
)


class PublicContextBuilder(_PublicContextBuilder):
    def __init__(self, **kwargs) -> None:
        protocol = kwargs.get("acceptance_protocol")
        if protocol is not None:
            kwargs.setdefault(
                "initial_current_best",
                make_bootstrap_current_best(),
            )
            kwargs.setdefault("session_genesis_hash", TEST_SESSION_GENESIS_HASH)
        super().__init__(**kwargs)


def _write_candidate_package(root: Path, candidate_id: str, components_py: str) -> CandidatePackageRef:
    candidate_dir = root / candidate_id
    (candidate_dir / "training_code").mkdir(parents=True)
    (candidate_dir / "candidate_manifest.yaml").write_text(
        "\n".join(
            [
                "schema_version: true_rl_agent.candidate.v1",
                f"candidate_id: {candidate_id}",
                "entrypoint: training_code.components:build_training_components",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (candidate_dir / "training_code" / "components.py").write_text(components_py, encoding="utf-8")
    return CandidatePackageRef(
        candidate_id=candidate_id,
        path=str(candidate_dir),
        change_set=CandidateChangeSet(frozenset({"reward_callable"})),
        metadata={"fake_public_score": 0.20, "verifier_status": "pass"},
    )


def _accepted_record(round_id: str, candidate: CandidatePackageRef, public_score: float) -> dict:
    return valid_accepted_ledger_record(
        round_id=round_id,
        candidate_id=candidate.candidate_id,
        public_score=public_score,
        candidate_path=candidate.path,
    )


class _SmokeAdapter:
    task_id = "fake_push_recovery"

    def build_component_smoke_fixture(self) -> ComponentSmokeFixture:
        num_envs = 3
        vector = SmokeTensor.vector([0.2, 0.3, 0.4])
        matrix2 = SmokeTensor.matrix([[0.0, 0.0], [0.1, 0.0], [0.0, -0.1]])
        matrix3 = SmokeTensor.matrix([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
        return ComponentSmokeFixture(
            task_id=self.task_id,
            reward_inputs=PushRecoveryRewardInputs(
                base_height=vector,
                roll=SmokeTensor.vector([0.01, 0.02, 0.03]),
                pitch=SmokeTensor.vector([0.02, 0.01, 0.0]),
                base_pos_xy=matrix2,
                base_lin_vel=matrix3,
                base_ang_vel=matrix3,
                projected_gravity=matrix3,
                commanded_base_vel=matrix3,
                joint_pos=matrix3,
                joint_vel=matrix3,
                joint_pos_error=matrix3,
                last_action=matrix3,
                prev_action=matrix3,
                action_delta=matrix3,
                push_active=SmokeTensor.vector([0.0, 1.0, 0.0]),
                push_vec_public=matrix2,
                progress_fraction=SmokeTensor.vector([0.1, 0.4, 0.8]),
            ),
            expected_reward_output_shape=(num_envs,),
            public_summary={"fixture_owner": "task_owned_test", "num_envs": num_envs},
        )


class _ProviderErrorThenSuccessProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            raise ProviderCandidateError(
                "json_parse_error",
                "bad JSON at D:/humanoid_agent/private/raw.txt",
                details={"raw_excerpt": "sk-live-secret D:/humanoid_agent/private/raw.txt"},
            )
        return CandidatePackageRef(
            candidate_id="candidate-002",
            path="memory://candidate-002",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30},
        )


class _ImmediateSuccessProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        return CandidatePackageRef(
            candidate_id="candidate-ready-preview",
            path="memory://candidate-ready-preview",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30, "verifier_status": "pass"},
        )


class _SourceRequestThenSuccessProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_authoring_response(self, *, prompt_payload: dict) -> SourceRequestResponse | CandidateDraftResponse:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            return SourceRequestResponse(
                logical_ids=("current_best/training_code/components.py",),
                rationale="Need the current best source before drafting.",
                source_catalog_logical_ids=tuple(sorted(prompt_source_catalog_logical_ids(prompt_payload))),
            )
        return CandidateDraftResponse(
            candidate=CandidatePackageRef(
                candidate_id="candidate-002",
                path="memory://candidate-002",
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.30},
            )
        )


class _TwoSourceRequestsThenSuccessProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_authoring_response(self, *, prompt_payload: dict) -> SourceRequestResponse | CandidateDraftResponse:
        self.payloads.append(prompt_payload)
        if len(self.payloads) <= 2:
            return SourceRequestResponse(
                logical_ids=("current_best/training_code/components.py",),
                rationale="Need source before drafting.",
                source_catalog_logical_ids=tuple(sorted(prompt_source_catalog_logical_ids(prompt_payload))),
            )
        return CandidateDraftResponse(
            candidate=CandidatePackageRef(
                candidate_id="candidate-003",
                path="memory://candidate-003",
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.33},
            )
        )


class _TwoDistinctSourceRequestsThenSuccessProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_authoring_response(self, *, prompt_payload: dict) -> SourceRequestResponse | CandidateDraftResponse:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            return SourceRequestResponse(
                logical_ids=("current_best/training_code/components.py",),
                rationale="Need current best before drafting.",
                source_catalog_logical_ids=tuple(sorted(prompt_source_catalog_logical_ids(prompt_payload))),
            )
        if len(self.payloads) == 2:
            return SourceRequestResponse(
                logical_ids=("attempt/latest_failed/training_code/components.py",),
                rationale="Need latest failed source as repair evidence.",
                source_catalog_logical_ids=tuple(sorted(prompt_source_catalog_logical_ids(prompt_payload))),
            )
        return CandidateDraftResponse(
            candidate=CandidatePackageRef(
                candidate_id="candidate-004",
                path="memory://candidate-004",
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.34},
            )
        )


class _VerifierFailThenSuccessProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        if index == 1:
            candidate_dir = self.candidate_root / "candidate-001"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def failed_static_reward(inp):\n    return inp.base_height\n",
                encoding="utf-8",
            )
            return CandidatePackageRef(
                candidate_id="candidate-001",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
            )
        return CandidatePackageRef(
            candidate_id="candidate-002",
            path="memory://candidate-002",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30, "verifier_status": "pass"},
        )


class _VerifierFailThenProviderFailureProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            candidate_dir = self.candidate_root / "candidate-001"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def failed_static_reward(inp):\n    return inp.base_height\n",
                encoding="utf-8",
            )
            return CandidatePackageRef(
                candidate_id="candidate-001",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
            )
        raise ProviderCandidateError(
            "schema_error",
            "candidate schema failed at D:/humanoid_agent/private/raw.txt",
            details={"raw_excerpt": "sk-live-secret /data/yuyangliu/secrets/key.env"},
        )


class _VerifierFailThenSourceRequestThenSuccessProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_authoring_response(self, *, prompt_payload: dict) -> SourceRequestResponse | CandidateDraftResponse:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            candidate_dir = self.candidate_root / "candidate-001"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def failed_static_reward(inp):\n    return inp.base_height\n",
                encoding="utf-8",
            )
            return CandidateDraftResponse(
                candidate=CandidatePackageRef(
                    candidate_id="candidate-001",
                    path=str(candidate_dir),
                    change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                    metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
                )
            )
        if len(self.payloads) == 2:
            return SourceRequestResponse(
                logical_ids=(LATEST_FAILED_SOURCE_ID,),
                rationale="Need the latest failed source before repairing.",
                source_catalog_logical_ids=tuple(sorted(prompt_source_catalog_logical_ids(prompt_payload))),
            )
        return CandidateDraftResponse(
            candidate=CandidatePackageRef(
                candidate_id="candidate-002",
                path="memory://candidate-002",
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.30, "verifier_status": "pass"},
            )
        )


class _AlwaysProviderFailureProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        raise ProviderCandidateError(
            "schema_error",
            "candidate schema failed at /data/yuyangliu/secrets/key.env",
            details={"raw_excerpt": "sk-live-secret /data/yuyangliu/secrets/key.env"},
        )


class _SmokeFailThenSuccessProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            return _write_candidate_package(
                self.candidate_root,
                "candidate-smoke-fail",
                """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def broken_reward(inp):
    raise RuntimeError("D:/humanoid_agent/private/raw.txt /data/yuyangliu/secrets/key.env")

def build_training_components():
    return TrainingComponents(
        name="smoke_fail",
        reward_terms=[RewardTerm(name="broken_reward", weight=1.0, fn=broken_reward)],
        rationale="This candidate should fail component smoke.",
    )
""",
            )
        return _write_candidate_package(
            self.candidate_root,
            "candidate-smoke-pass",
            """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def stable_reward(inp):
    return inp.base_height * 0.5

def build_training_components():
    return TrainingComponents(
        name="smoke_pass",
        reward_terms=[RewardTerm(name="stable_reward", weight=1.0, fn=stable_reward)],
        rationale="This candidate should pass component smoke.",
    )
""",
        )


class _EvaluatedRepeatThenRewriteProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_authoring_response(self, *, prompt_payload: dict) -> CandidateDraftResponse:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def recovery_reward(inp):
    drift = inp.base_pos_xy.norm(dim=-1)
    return inp.base_height - 0.1 * drift

def build_training_components():
    return TrainingComponents(
        name="renamed_repeat",
        reward_terms=[RewardTerm(name="renamed_reward", weight=1.0, fn=recovery_reward)],
        rationale="Same evaluated mechanism with renamed metadata.",
    )
"""
            candidate_id = "candidate-history-repeat"
        else:
            source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def recovery_reward(inp):
    drift = inp.base_pos_xy.norm(dim=-1)
    tilt = inp.roll * inp.roll + inp.pitch * inp.pitch
    return inp.base_height - 0.1 * drift - 0.2 * tilt

def build_training_components():
    return TrainingComponents(
        name="structural_rewrite",
        reward_terms=[RewardTerm(name="recovery_reward", weight=1.0, fn=recovery_reward)],
        rationale="Add a structurally distinct tilt mechanism.",
    )
"""
            candidate_id = "candidate-structural-rewrite"
        return CandidateDraftResponse(
            candidate=_write_candidate_package(
                self.candidate_root,
                candidate_id,
                source,
            )
        )


class CandidateAuthoringLoopTests(unittest.TestCase):
    def test_runner_capability_readiness_is_forwarded_to_provider_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _ImmediateSuccessProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=1,
                runner_capability_readiness={
                    CURRICULUM_DESIGN_CAPABILITY_ID: {
                        "task_enabled": True,
                        "compiler_ready": True,
                        "verifier_ready": True,
                    },
                },
            )

            result = loop.run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        visible = {
            item["capability_id"]: item
            for item in provider.payloads[0]["candidate_capabilities"]["prompt_visible_capabilities"]
        }
        self.assertIn(CURRICULUM_DESIGN_CAPABILITY_ID, visible)
        self.assertEqual(visible[CURRICULUM_DESIGN_CAPABILITY_ID]["exposure_mode"], "schema_preview_only")
        self.assertFalse(visible[CURRICULUM_DESIGN_CAPABILITY_ID]["candidate_writable"])

    def test_provider_parse_failure_repairs_with_static_frame_and_public_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _ProviderErrorThenSuccessProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-002")
        self.assertEqual(len(result.attempt_records), 2)
        self.assertEqual(result.attempt_records[0]["provider_status"], "fail")
        self.assertEqual(result.attempt_records[1]["provider_status"], "pass")
        self.assertEqual(provider.payloads[0]["authoring_context_frame"]["frame_kind"], "candidate_draft")
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "static_repair")
        repair_feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        repair_text = json.dumps(repair_feedback, sort_keys=True)
        self.assertEqual(repair_feedback["failure_category"], "json_parse_error")
        self.assertNotIn("D:/humanoid_agent", repair_text)
        self.assertNotIn("sk-live-secret", repair_text)

    def test_source_request_response_returns_source_followup_before_candidate_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_candidate = _write_candidate_package(
                root / "accepted",
                "accepted-best",
                """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def current_best_reward(inp):
    return inp.base_height

def build_training_components():
    return TrainingComponents(
        name="accepted_best",
        reward_terms=[RewardTerm(name="current_best_reward", weight=1.0, fn=current_best_reward)],
    )
""",
            )
            provider = _SourceRequestThenSuccessProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(
                round_id="round-001",
                records=[_accepted_record("round-accepted", accepted_candidate, 0.31)],
            )

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-002")
        self.assertEqual(len(result.attempt_records), 2)
        self.assertEqual(result.attempt_records[0]["provider_status"], "source_request")
        self.assertEqual(result.attempt_records[0]["provider_response_mode"], "source_request")
        self.assertNotIn("prompt_payload_path", result.attempt_records[0])
        self.assertEqual(result.attempt_records[0]["source_request_result"]["status"], "source_payload_returned")
        self.assertEqual(result.attempt_records[1]["provider_status"], "pass")
        self.assertEqual(result.attempt_records[1]["provider_response_mode"], "candidate_draft")
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "source_followup")
        self.assertEqual(result.round_public_context["previous_rounds"], 1)
        self.assertEqual(
            result.round_public_context["recent_outcomes"][-1]["round_id"],
            "round-accepted",
        )
        self.assertNotIn("candidate_repair_feedback", result.round_public_context)
        self.assertNotIn("source_payloads", result.round_public_context)
        self.assertNotIn("recent_outcomes", provider.payloads[1]["public_context"])
        repair_feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        repair_text = json.dumps(repair_feedback, sort_keys=True)
        self.assertEqual(repair_feedback["status"], "source_payload_returned")
        self.assertEqual(repair_feedback["returned_logical_ids"], ["current_best/training_code/components.py"])
        self.assertEqual(len(provider.payloads[1]["source_payloads"]), 1)
        self.assertIn("current_best_reward", provider.payloads[1]["source_payloads"][0]["content"])
        self.assertEqual(provider.payloads[1]["public_history"]["recent_outcomes"], [])
        followup_frame = provider.payloads[1]["authoring_context_frame"]
        self.assertEqual(followup_frame["legacy_alias_sections"], ["public_context"])
        self.assertLess(
            followup_frame["budget_metadata"]["actual_prompt_payload_bytes"],
            followup_frame["budget_metadata"]["hard_limits"]["max_prompt_payload_bytes"],
        )
        self.assertIn("source_request_service_enabled", repair_text)
        self.assertNotIn("D:/humanoid_agent", repair_text)
        self.assertNotIn("/data/yuyangliu", repair_text)
        self.assertEqual(result.source_request_feedback["status"], "source_payload_returned")
        self.assertEqual(
            result.source_request_feedback["returned_logical_ids"],
            ["current_best/training_code/components.py"],
        )
        self.assertGreater(
            result.source_request_feedback["budget"]["cumulative_returned_bytes"],
            0,
        )

    def test_source_request_for_unavailable_source_returns_bounded_followup_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _SourceRequestThenSuccessProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.attempt_records[0]["provider_status"], "source_request")
        self.assertEqual(result.attempt_records[0]["source_request_result"]["status"], "source_request_denied")
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "source_followup")
        feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        feedback_text = json.dumps(feedback, sort_keys=True)
        self.assertEqual(feedback["status"], "source_request_denied")
        self.assertEqual(feedback["returned_logical_ids"], [])
        self.assertEqual(provider.payloads[1]["source_payloads"], [])
        self.assertIn("source_not_requestable", feedback_text)
        self.assertNotIn("D:/humanoid_agent", feedback_text)
        self.assertNotIn("/data/yuyangliu", feedback_text)
        self.assertEqual(
            result.source_request_feedback,
            {
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
                    "max_source_request_bytes_per_round": 48 * 1024,
                },
                "omitted_returned_logical_id_count": 0,
                "omitted_denied_logical_id_count": 0,
            },
        )

    def test_final_attempt_source_request_keeps_same_bounded_round_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=_SourceRequestThenSuccessProvider(),
                verifier=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=1,
            ).run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "provider_failed")
        self.assertEqual(result.provider_failure["category"], "source_request_denied")
        self.assertEqual(result.source_request_feedback["status"], "source_request_denied")
        self.assertEqual(
            result.source_request_feedback["denied_logical_ids"],
            [
                {
                    "logical_id": "current_best/training_code/components.py",
                    "reason": "source_not_requestable",
                }
            ],
        )

    def test_repeated_source_requests_are_capped_before_candidate_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_candidate = _write_candidate_package(
                root / "accepted",
                "accepted-best",
                "def current_best_reward(inp):\n    return inp.base_height\n",
            )
            provider = _TwoSourceRequestsThenSuccessProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=3,
                max_source_requests_per_round=1,
            )

            result = loop.run_round(
                round_id="round-001",
                records=[_accepted_record("round-accepted", accepted_candidate, 0.31)],
            )

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-003")
        self.assertEqual(result.attempt_records[0]["source_request_result"]["status"], "source_payload_returned")
        self.assertEqual(result.attempt_records[1]["source_request_result"]["status"], "source_request_denied")
        feedback = provider.payloads[2]["provider_or_runtime_repair_feedback"]
        self.assertEqual(feedback["denied_logical_ids"][0]["reason"], "source_request_turn_limit_exceeded")
        self.assertEqual(result.source_request_feedback["status"], "source_request_partial")
        self.assertEqual(result.source_request_feedback["request_count"], 2)
        self.assertEqual(result.source_request_feedback["budget"]["max_source_requests_per_round"], 1)

    def test_repeated_source_requests_use_full_round_source_repository_not_followup_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_candidate = _write_candidate_package(
                root / "accepted",
                "accepted-best",
                "def current_best_reward(inp):\n    return inp.base_height\n",
            )
            runtime_failed_candidate = _write_candidate_package(
                root / "runtime_failed",
                "runtime-failed",
                "def latest_failed_reward(inp):\n    return inp.base_height * 0.1\n",
            )
            provider = _TwoDistinctSourceRequestsThenSuccessProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=3,
            )

            result = loop.run_round(
                round_id="round-001",
                records=[
                    _accepted_record("round-accepted", accepted_candidate, 0.31),
                    {
                        "round_id": "round-runtime-failed",
                        "session_genesis_hash": TEST_SESSION_GENESIS_HASH,
                        "evaluation_binding": (
                            TEST_EVALUATION_BINDING.to_public_dict()
                        ),
                        "decision": {
                            "accepted": False,
                            "outcome": "reject",
                            "reason": "candidate_runtime_failed",
                            "surface_classification": "code_level",
                            "public_score": None,
                            "comparison": {},
                            "consumed_gates": [],
                            "gate_statuses": {},
                        },
                        "candidate": {
                            "candidate_id": runtime_failed_candidate.candidate_id,
                            "path": runtime_failed_candidate.path,
                            "changed_components": sorted(runtime_failed_candidate.change_set.changed_components),
                        },
                        "candidate_surface": {"classification": "code_level"},
                        "verifier": {"status": "pass"},
                        "jobs": {
                            "train": {"job_id": "runtime-failed-train"},
                            "eval": {"job_id": "runtime-failed-eval"},
                        },
                        "runner_error": {
                            "type": "CandidateRuntimeError",
                            "message": "candidate failed inside runner",
                        },
                    },
                ],
            )

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-004")
        self.assertEqual(result.attempt_records[0]["source_request_result"]["status"], "source_payload_returned")
        self.assertEqual(result.attempt_records[0]["source_request_result"]["returned_logical_ids"], ["current_best/training_code/components.py"])
        self.assertEqual(result.attempt_records[1]["source_request_result"]["status"], "source_payload_returned")
        self.assertEqual(
            result.attempt_records[1]["source_request_result"]["returned_logical_ids"],
            ["attempt/latest_failed/training_code/components.py"],
        )
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "source_followup")
        self.assertEqual(provider.payloads[2]["authoring_context_frame"]["frame_kind"], "source_followup")
        payload_2_ids = [item["logical_id"] for item in provider.payloads[1]["source_payloads"]]
        payload_3_by_id = {item["logical_id"]: item for item in provider.payloads[2]["source_payloads"]}
        self.assertEqual(payload_2_ids, ["current_best/training_code/components.py"])
        self.assertIn("current_best/training_code/components.py", payload_3_by_id)
        self.assertIn("attempt/latest_failed/training_code/components.py", payload_3_by_id)
        self.assertIn("current_best_reward", payload_3_by_id["current_best/training_code/components.py"]["content"])
        self.assertIn("latest_failed_reward", payload_3_by_id["attempt/latest_failed/training_code/components.py"]["content"])

    def test_verifier_failure_repair_uses_latest_failed_source_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _VerifierFailThenSuccessProvider(root / "candidates")
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-002")
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "static_repair")
        feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        failed_sources = [
            item
            for item in provider.payloads[1]["source_payloads"]
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        ]
        self.assertEqual(feedback["failure_category"], "verifier_error")
        self.assertEqual(feedback["previous_candidate_id"], "candidate-001")
        self.assertEqual(len(failed_sources), 1)
        self.assertIn("failed_static_reward", failed_sources[0]["content"])

    def test_source_followup_after_verifier_failure_preserves_same_round_failed_source_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _VerifierFailThenSourceRequestThenSuccessProvider(root / "candidates")
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=3,
            )

            result = loop.run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-002")
        self.assertEqual(result.attempt_records[0]["provider_status"], "pass")
        self.assertEqual(result.attempt_records[0]["verifier_status"], "fail")
        self.assertEqual(result.attempt_records[1]["provider_status"], "source_request")
        self.assertEqual(result.attempt_records[1]["source_request_result"]["status"], "source_payload_returned")
        self.assertEqual(
            result.attempt_records[1]["source_request_result"]["returned_logical_ids"],
            [LATEST_FAILED_SOURCE_ID],
        )
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "static_repair")
        self.assertEqual(provider.payloads[2]["authoring_context_frame"]["frame_kind"], "source_followup")
        followup_sources = {
            item["logical_id"]: item
            for item in provider.payloads[2]["source_payloads"]
        }
        self.assertIn(LATEST_FAILED_SOURCE_ID, followup_sources)
        self.assertIn("failed_static_reward", followup_sources[LATEST_FAILED_SOURCE_ID]["content"])

    def test_verifier_failure_then_provider_failure_keeps_attempt_records_public_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _VerifierFailThenProviderFailureProvider(root / "candidates")
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        attempt_text = json.dumps(result.attempt_records, sort_keys=True)
        self.assertEqual(result.stop_reason, "provider_failed")
        self.assertIsNone(result.candidate)
        self.assertEqual(result.provider_failure["category"], "schema_error")
        self.assertIsNone(result.source_request_feedback)
        self.assertEqual(len(result.attempt_records), 2)
        self.assertEqual(result.attempt_records[0]["candidate_id"], "candidate-001")
        self.assertEqual(result.attempt_records[0]["verifier_status"], "fail")
        self.assertNotIn("prompt_payload_path", result.attempt_records[0])
        self.assertNotIn("candidate_path", result.attempt_records[0])
        self.assertEqual(result.attempt_records[0]["candidate_source_visibility"], "private_non_prompt_artifact")
        self.assertFalse(result.attempt_records[0]["candidate_source_included_in_prompt"])
        self.assertNotIn(str(root / "candidates"), attempt_text)
        self.assertNotIn("D:/humanoid_agent", attempt_text)
        self.assertNotIn("/data/yuyangliu", attempt_text)
        self.assertNotIn("sk-live-secret", attempt_text)

    def test_all_provider_attempts_fail_returns_public_safe_failure_without_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _AlwaysProviderFailureProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        result_text = json.dumps(result.provider_failure, sort_keys=True)
        self.assertEqual(result.stop_reason, "provider_failed")
        self.assertIsNone(result.candidate)
        self.assertEqual(result.provider_failure["category"], "schema_error")
        self.assertEqual(len(result.attempt_records), 2)
        self.assertEqual(len(result.attempt_prompt_payload_paths), 2)
        self.assertTrue(str(result.prompt_payload_path).endswith("prompt_payload_attempt_002.json"))
        self.assertNotIn("/data/yuyangliu", result_text)
        self.assertNotIn("sk-live-secret", result_text)

    def test_component_smoke_failure_repairs_with_failed_source_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _SmokeFailThenSuccessProvider(root / "candidates")
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=2,
                component_smoke_adapter=_SmokeAdapter(),
            )

            result = loop.run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-smoke-pass")
        self.assertEqual(result.attempt_records[0]["component_smoke"]["status"], "fail")
        self.assertEqual(result.attempt_records[1]["component_smoke"]["status"], "pass")
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "static_repair")
        feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        feedback_text = json.dumps(feedback, sort_keys=True)
        failed_sources = [
            item
            for item in provider.payloads[1]["source_payloads"]
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        ]
        self.assertEqual(feedback["failure_category"], "component_smoke_error")
        self.assertEqual(feedback["component_smoke_result"]["reason"], "reward_callable_failed")
        self.assertEqual(feedback["previous_candidate_id"], "candidate-smoke-fail")
        self.assertEqual(len(failed_sources), 1)
        self.assertIn("broken_reward", failed_sources[0]["content"])
        self.assertNotIn("D:/humanoid_agent", feedback_text)
        self.assertNotIn("/data/yuyangliu", feedback_text)
        self.assertNotIn("sk-live-secret", feedback_text)

    def test_evaluated_mechanism_repeat_is_rewritten_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluated_candidate = _write_candidate_package(
                root / "evaluated",
                "evaluated-drift",
                """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def recovery_reward(inp):
    drift = inp.base_pos_xy.norm(dim=-1)
    return inp.base_height - 0.1 * drift

def build_training_components():
    return TrainingComponents(
        name="evaluated_drift",
        reward_terms=[RewardTerm(name="recovery_reward", weight=1.0, fn=recovery_reward)],
        rationale="Previously evaluated drift mechanism.",
    )
""",
            )
            provider = _EvaluatedRepeatThenRewriteProvider(root / "candidates")
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                verifier=FakeVerifier(),
                round_dir=root / "round-next",
                max_attempts=2,
            )

            result = loop.run_round(
                round_id="round-next",
                records=[
                    _accepted_record(
                        "round-evaluated",
                        evaluated_candidate,
                        0.31,
                    )
                ],
            )

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-structural-rewrite")
        self.assertEqual(len(result.attempt_records), 2)
        assessment = result.attempt_records[0]["anti_degenerate_assessment"]
        self.assertTrue(assessment["blocking"])
        self.assertEqual(assessment["classification"], "historical_mechanism_repeat")
        self.assertEqual(
            assessment["equivalence_classification"],
            "rationale_only_delta",
        )
        self.assertEqual(assessment["matched_round_id"], "round-evaluated")
        self.assertEqual(assessment["matched_candidate_id"], "evaluated-drift")
        self.assertEqual(
            provider.payloads[1]["authoring_context_frame"]["frame_kind"],
            "static_repair",
        )
        feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        self.assertEqual(feedback["status"], "rewrite_required")
        self.assertEqual(feedback["failure_category"], "anti_degenerate_warning")
        self.assertEqual(
            feedback["anti_degenerate_assessment"]["matched_candidate_id"],
            "evaluated-drift",
        )
        self.assertNotIn(str(root), json.dumps(feedback, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
