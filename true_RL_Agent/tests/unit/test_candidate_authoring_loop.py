from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (  # noqa: E402
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_CANDIDATE_FAILURE_CONTRACT,
    TEST_SESSION_GENESIS_HASH,
    bind_verified_candidate_record,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from tests.fakes.fake_components import FakeTaskPlugin, FakeVerifier  # noqa: E402
from true_rl_agent.agent_core.authoring_loop import (  # noqa: E402
    AuthoringBudgetExhaustedError,
    CandidateAuthoringLoop,
    _evaluated_candidate_sources,
    _merge_source_repository_payload,
    _source_repository_from_prompt_refs,
)
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
from true_rl_agent.candidates.authoring_policy import (  # noqa: E402
    single_component_verifier_equivalence_key,
    verifier_equivalence_exclusion_set_hash,
)
from true_rl_agent.candidates.component_source_semantics import (  # noqa: E402
    project_single_component_operation_delta,
)
from true_rl_agent.candidates.verifier import (  # noqa: E402
    StaticCandidateVerifier,
    StaticCandidateVerifierPolicy,
    VerificationResult,
    seal_and_verify_candidate,
)
from true_rl_agent.memory import (  # noqa: E402
    ExperimentRecord,
    PrivateArtifactRef,
    PublicContextBuilder as _PublicContextBuilder,
)
from true_rl_agent.prompt_artifacts import (  # noqa: E402
    RoundAuthoringAnchorRef,
    project_selected_authoring_mechanism_delta,
    publish_prompt_payload,
    verify_authoring_evidence,
    verify_round_authoring_anchor,
)
from true_rl_agent.prompting.payload import serialize_prompt_payload  # noqa: E402
from true_rl_agent.prompting.runner_capability_schemas import (
    CURRICULUM_DESIGN_CAPABILITY_ID,  # noqa: E402
)
from true_rl_agent.prompting.source_catalog import (  # noqa: E402
    CURRENT_BEST_SOURCE_ID,
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
    (candidate_dir / "rationale.md").write_text("Candidate rationale.\n", encoding="utf-8")
    (candidate_dir / "operation_diff.json").write_text("{}\n", encoding="utf-8")
    (candidate_dir / "diagnostics.json").write_text("{}\n", encoding="utf-8")
    return CandidatePackageRef(
        candidate_id=candidate_id,
        path=str(candidate_dir),
        change_set=CandidateChangeSet(frozenset({"reward_callable"})),
        metadata={"fake_public_score": 0.20, "verifier_status": "pass"},
    )


def _accepted_record(round_id: str, candidate: CandidatePackageRef, public_score: float) -> dict:
    verification = seal_and_verify_candidate(
        candidate,
        artifact_root=Path(candidate.path).parent / ".accepted-artifacts",
        verifier=FakeVerifier(),
    )
    record = valid_accepted_ledger_record(
        round_id=round_id,
        candidate_id=candidate.candidate_id,
        public_score=public_score,
    )
    return bind_verified_candidate_record(record, verification=verification)


def _r10_components_source(*, component_name: str, include_support: bool) -> str:
    support_function = (
        "\n\ndef foot_contact_support_reward(inputs):\n"
        "    return inputs.base_height * 0.5\n"
        if include_support
        else ""
    )
    support_term = (
        "\n            RewardTerm(\n"
        "                name=\"foot_contact_support_reward\",\n"
        "                weight=1.0,\n"
        "                fn=foot_contact_support_reward,\n"
        "            ),"
        if include_support
        else ""
    )
    return (
        "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n"
        "\n\n"
        "def balance_reward(inputs):\n"
        "    return inputs.base_height\n"
        f"{support_function}\n\n"
        "def build_training_components():\n"
        "    return TrainingComponents(\n"
        f"        name=\"{component_name}\",\n"
        "        reward_terms=[\n"
        "            RewardTerm(\n"
        "                name=\"balance_reward\",\n"
        "                weight=1.0,\n"
        "                fn=balance_reward,\n"
        "            ),"
        f"{support_term}\n"
        "        ],\n"
        "        rationale=\"Add one support mechanism while retaining balance.\",\n"
        "    )\n"
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
                provider_usage={"prompt_tokens": 99, "prompt_cache_miss_tokens": 99},
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


class _SingleDraftProvider:
    def __init__(self, candidate: CandidatePackageRef) -> None:
        self.candidate = candidate
        self.payloads: list[dict] = []

    def next_authoring_response(self, *, prompt_payload: dict) -> CandidateDraftResponse:
        self.payloads.append(prompt_payload)
        return CandidateDraftResponse(candidate=self.candidate)


class _InterruptingProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        raise KeyboardInterrupt


class _PromptPolicyFakeVerifier(FakeVerifier):
    def __init__(self, static_verifier: StaticCandidateVerifier) -> None:
        self.static_verifier = static_verifier

    def authoring_prompt_rules(self):
        return self.static_verifier.authoring_prompt_rules()


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
                provider_usage={"prompt_tokens": 70},
            )
        return CandidateDraftResponse(
            candidate=CandidatePackageRef(
                candidate_id="candidate-002",
                path="memory://candidate-002",
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.30},
            ),
            provider_usage={"prompt_tokens": 80, "prompt_cache_hit_tokens": 60},
        )


class _SourceRequestThenInterruptProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_authoring_response(
        self,
        *,
        prompt_payload: dict,
    ) -> SourceRequestResponse | CandidateDraftResponse:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            return SourceRequestResponse(
                logical_ids=("current_best/training_code/components.py",),
                rationale="Need the current best source before drafting.",
                source_catalog_logical_ids=tuple(
                    sorted(prompt_source_catalog_logical_ids(prompt_payload))
                ),
            )
        raise KeyboardInterrupt


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
    def __init__(
        self,
        candidate_root: Path,
        *,
        failed_source: str = (
            "def failed_static_reward(inp):\n    return inp.base_height\n"
        ),
    ) -> None:
        self.candidate_root = candidate_root
        self.failed_source = failed_source
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        if index == 1:
            candidate = _write_candidate_package(
                self.candidate_root,
                "candidate-001",
                self.failed_source,
            )
            return replace(
                candidate,
                metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
            )
        return CandidatePackageRef(
            candidate_id="candidate-002",
            path="memory://candidate-002",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30, "verifier_status": "pass"},
        )


class _NameOnlyRepairProvider:
    def __init__(
        self,
        candidate_root: Path,
        *,
        failed_source: str,
        repaired_source: str,
    ) -> None:
        self.candidate_root = candidate_root
        self.failed_source = failed_source
        self.repaired_source = repaired_source
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            candidate = _write_candidate_package(
                self.candidate_root / "attempt-1",
                "stair-foot-support-v1",
                self.failed_source,
            )
            return replace(
                candidate,
                metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
            )
        return _write_candidate_package(
            self.candidate_root / "attempt-2",
            "stair-foot-support-v1",
            self.repaired_source,
        )


class _IssueVerifier(FakeVerifier):
    def __init__(self, issues: list[str]) -> None:
        self.issues = list(issues)

    def verify(self, candidate: CandidatePackageRef) -> VerificationResult:
        if candidate.metadata.get("verifier_status") == "fail":
            return VerificationResult(
                status="fail",
                reason="static_candidate_verifier_failed",
                details={"issues": list(self.issues)},
            )
        return super().verify(candidate)


class _AuthoringDeltaIssueVerifier(FakeVerifier):
    def __init__(self) -> None:
        self.parent_source: str | None = None

    def verify_authoring_delta(
        self,
        *,
        candidate: CandidatePackageRef,
        verification: VerificationResult,
        parent_source: str | None,
    ) -> VerificationResult:
        self.parent_source = parent_source
        return VerificationResult(
            status="fail",
            reason="task_authoring_delta_invalid",
            details={"issues": ["task_owned_authoring_delta_issue"]},
        )


class _VerifierFailThenInterruptProvider(_VerifierFailThenSuccessProvider):
    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        if self.payloads:
            self.payloads.append(prompt_payload)
            raise KeyboardInterrupt
        return super().next_candidate(prompt_payload=prompt_payload)


class _VerifierFailThenProviderFailureProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            candidate = _write_candidate_package(
                self.candidate_root,
                "candidate-001",
                "def failed_static_reward(inp):\n    return inp.base_height\n",
            )
            return replace(
                candidate,
                metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
            )
        raise ProviderCandidateError(
            "schema_error",
            "candidate schema failed at D:/humanoid_agent/private/raw.txt",
            details={"raw_excerpt": "sk-live-secret /data/yuyangliu/secrets/key.env"},
        )


class _VerifierFailThenSourceRequestThenSuccessProvider:
    def __init__(
        self,
        candidate_root: Path,
        *,
        requested_source_id: str = LATEST_FAILED_SOURCE_ID,
    ) -> None:
        self.candidate_root = candidate_root
        self.requested_source_id = requested_source_id
        self.payloads: list[dict] = []

    def next_authoring_response(self, *, prompt_payload: dict) -> SourceRequestResponse | CandidateDraftResponse:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            candidate = _write_candidate_package(
                self.candidate_root,
                "candidate-001",
                "def failed_static_reward(inp):\n    return inp.base_height\n",
            )
            return CandidateDraftResponse(
                candidate=replace(
                    candidate,
                    metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
                )
            )
        if len(self.payloads) == 2:
            return SourceRequestResponse(
                logical_ids=(self.requested_source_id,),
                rationale="Need a previously visible source before repairing.",
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


class _TransportFailureProvider:
    def __init__(self, *, retryable: bool | None) -> None:
        self.retryable = retryable
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        details = {
            "failure_kind": "network_error" if self.retryable else "http_error",
            "http_status": None if self.retryable else 401,
        }
        if self.retryable is not None:
            details["retryable"] = self.retryable
        raise ProviderCandidateError(
            "runtime_error",
            "provider network request failed",
            details=details,
            provider_receipt={
                "schema_version": "true_rl_agent.provider_response_receipt.v2",
                "transport_policy_id": (
                    "true_rl_agent.openai_compatible_transport.https_or_loopback_http_no_redirect.v1"
                ),
                "response_max_bytes": 524288,
            },
        )


class _AlwaysTruncatedProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        raise ProviderCandidateError(
            "response_truncated",
            "provider response reached its output limit before candidate JSON completed",
            details={"finish_reason": "length"},
            provider_usage={"completion_tokens": 8192},
            provider_receipt={
                "schema_version": "true_rl_agent.provider_response_receipt.v1",
                "requested_model": "deepseek-v4-pro",
                "requested_thinking_mode": (
                    "enabled" if len(self.payloads) == 1 else "disabled"
                ),
                "finish_reason": "length",
            },
        )


class _ThreeTruncationsThenInterruptProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        if len(self.payloads) <= 3:
            raise ProviderCandidateError(
                "response_truncated",
                "provider response reached its output limit before candidate JSON completed",
                details={"finish_reason": "length"},
            )
        raise KeyboardInterrupt


class _VerifierFailThenTruncationThenSuccessProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        if len(self.payloads) == 1:
            candidate = _write_candidate_package(
                self.candidate_root,
                "candidate-001",
                "def failed_static_reward(inp):\n    return inp.base_height\n",
            )
            return CandidatePackageRef(
                candidate_id=candidate.candidate_id,
                path=candidate.path,
                change_set=candidate.change_set,
                metadata={"fake_public_score": 0.20, "verifier_status": "fail"},
            )
        if len(self.payloads) == 2:
            raise ProviderCandidateError(
                "response_truncated",
                "provider response reached its output limit before candidate JSON completed",
                details={"finish_reason": "length"},
            )
        return CandidatePackageRef(
            candidate_id="candidate-003",
            path="memory://candidate-003",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30, "verifier_status": "pass"},
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
    def test_task_authoring_delta_failure_precedes_component_smoke(self) -> None:
        anchor_source = _r10_components_source(
            component_name="anchor_components",
            include_support=False,
        )
        candidate_source = _r10_components_source(
            component_name="candidate_components",
            include_support=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor_candidate = _write_candidate_package(
                root / "anchor",
                "candidate-anchor",
                anchor_source,
            )
            candidate = _write_candidate_package(
                root / "candidate",
                "candidate-authoring-delta-fail",
                candidate_source,
            )
            verifier = _AuthoringDeltaIssueVerifier()
            with patch(
                "true_rl_agent.agent_core.authoring_loop.run_component_smoke_check"
            ) as smoke:
                result = CandidateAuthoringLoop(
                    context_builder=PublicContextBuilder(
                        task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                    ),
                    provider=_SingleDraftProvider(candidate),
                    candidate_admission_policy=verifier,
                    round_dir=root / "round",
                    max_attempts=1,
                ).run_round(
                    round_id="round-authoring-delta-fail",
                    records=[
                        _accepted_record("round-anchor", anchor_candidate, 0.31)
                    ],
                )

        self.assertEqual(result.stop_reason, "static_failed")
        self.assertIsNotNone(verifier.parent_source)
        self.assertIn("anchor_components", str(verifier.parent_source))
        self.assertEqual(
            result.attempt_records[0]["verifier_reason"],
            "task_authoring_delta_invalid",
        )
        smoke.assert_not_called()

    def test_ledger_negative_verifier_equivalence_blocks_renamed_parameter_variant(self) -> None:
        anchor_source = _r10_components_source(
            component_name="anchor_components",
            include_support=False,
        )
        first_source = _r10_components_source(
            component_name="first_components",
            include_support=True,
        )
        renamed_source = first_source.replace(
            "foot_contact_support_reward",
            "renamed_support_reward",
        ).replace("weight=1.0,\n                fn=renamed", "weight=0.25,\n                fn=renamed")
        first_delta = project_single_component_operation_delta(
            candidate_source=first_source,
            anchor_source=anchor_source,
        )
        blocked_key = single_component_verifier_equivalence_key(first_delta)
        exclusion_set_hash = verifier_equivalence_exclusion_set_hash((blocked_key,))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor_candidate = _write_candidate_package(
                root / "anchor",
                "candidate-anchor",
                anchor_source,
            )
            records = [_accepted_record("round-anchor", anchor_candidate, 0.31)]
            candidate = _write_candidate_package(
                root / "candidates",
                "candidate-renamed-parameter-repeat",
                renamed_source,
            )
            provider = _SingleDraftProvider(candidate)
            lane_intent = {
                "schema_version": "test.generation_lane_intent.v1",
                "lane_id": "lane-a",
                "intent": "novel",
                "verifier_equivalence_exclusion_set_hash": exclusion_set_hash,
            }
            with self.assertRaisesRegex(
                ValueError,
                "exclusion set does not match authoring gate",
            ):
                CandidateAuthoringLoop(
                    context_builder=PublicContextBuilder(
                        task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                    ),
                    provider=provider,
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=root / "unbound-round",
                    max_attempts=1,
                    generation_lane_intent=lane_intent,
                )
            with patch(
                "true_rl_agent.agent_core.authoring_loop._evaluated_candidate_sources",
                return_value=(),
            ):
                result = CandidateAuthoringLoop(
                    context_builder=PublicContextBuilder(
                        task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                    ),
                    provider=provider,
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=root / "round-repeat",
                    max_attempts=1,
                    generation_lane_intent=lane_intent,
                    blocked_verifier_equivalence_keys=(blocked_key,),
                ).run_round(round_id="round-repeat", records=records)

        self.assertEqual(result.stop_reason, "static_failed")
        assessment = result.attempt_records[0][
            "historical_verifier_exclusion_assessment"
        ]
        self.assertTrue(assessment["blocking"])
        self.assertEqual(
            assessment["reason"],
            "historical_verifier_equivalent_repeat",
        )
        self.assertEqual(assessment["verifier_equivalence_key"], blocked_key)

    def test_terminal_resume_preserves_candidate_metadata_without_provider_call(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round-001"
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            with patch(
                "true_rl_agent.agent_core.authoring_loop._result",
                side_effect=RuntimeError("injected post-completion interruption"),
            ), self.assertRaisesRegex(RuntimeError, "post-completion"):
                CandidateAuthoringLoop(
                    context_builder=context_builder,
                    provider=_ImmediateSuccessProvider(),
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=round_dir,
                    max_attempts=1,
                ).run_round(round_id="round-001", records=[])

            resumed_provider = _InterruptingProvider()
            resumed = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=resumed_provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=round_dir,
                max_attempts=1,
            ).run_round(round_id="round-001", records=[])

        self.assertEqual(resumed_provider.payloads, [])
        self.assertEqual(
            resumed.candidate.metadata,
            {"fake_public_score": 0.30, "verifier_status": "pass"},
        )

    def test_terminal_provider_failure_resume_does_not_reuse_failed_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "round-001"
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            with patch(
                "true_rl_agent.agent_core.authoring_loop._result",
                side_effect=RuntimeError("injected post-failure interruption"),
            ), self.assertRaisesRegex(RuntimeError, "post-failure"):
                CandidateAuthoringLoop(
                    context_builder=context_builder,
                    provider=_VerifierFailThenProviderFailureProvider(
                        root / "candidates"
                    ),
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=round_dir,
                    max_attempts=2,
                ).run_round(round_id="round-001", records=[])

            resumed_provider = _InterruptingProvider()
            resumed = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=resumed_provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=round_dir,
                max_attempts=2,
            ).run_round(round_id="round-001", records=[])

        self.assertEqual(resumed_provider.payloads, [])
        self.assertEqual(resumed.stop_reason, "provider_failed")
        self.assertIsNone(resumed.candidate)
        self.assertIsNotNone(resumed.provider_failure)

    def test_resume_restores_verifier_repair_feedback_and_failed_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "round-001"
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            with self.assertRaises(KeyboardInterrupt):
                CandidateAuthoringLoop(
                    context_builder=context_builder,
                    provider=_VerifierFailThenInterruptProvider(
                        root / "candidates"
                    ),
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=round_dir,
                    max_attempts=3,
                ).run_round(round_id="round-001", records=[])

            resumed_provider = _ImmediateSuccessProvider()
            resumed = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=resumed_provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=round_dir,
                max_attempts=3,
            ).run_round(round_id="round-001", records=[])

        payload = resumed_provider.payloads[0]
        self.assertEqual(resumed.stop_reason, "candidate_verified")
        self.assertEqual(
            payload["authoring_context_frame"]["frame_kind"],
            "static_repair",
        )
        self.assertEqual(
            payload["provider_or_runtime_repair_feedback"][
                "previous_candidate_id"
            ],
            "candidate-001",
        )
        failed_source = next(
            item
            for item in payload["source_payloads"]
            if item["logical_id"]
            == "attempt/latest_failed/training_code/components.py"
        )
        self.assertIn("failed_static_reward", failed_source["content"])

    def test_source_repository_restore_does_not_reinterpret_legacy_lineage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_candidate = _write_candidate_package(
                root / "accepted",
                "accepted-current-best",
                "def current_best_reward(inp):\n    return inp.base_height\n",
            )
            accepted_record = _accepted_record(
                "round-accepted",
                accepted_candidate,
                0.31,
            )
            interrupted = _InterruptingProvider()
            with self.assertRaises(KeyboardInterrupt):
                CandidateAuthoringLoop(
                    context_builder=PublicContextBuilder(
                        task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                    ),
                    provider=interrupted,
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=root / "round-next",
                    max_attempts=2,
                ).run_round(
                    round_id="round-next",
                    records=[accepted_record],
                )

            current_payload = interrupted.payloads[0]
            legacy_payload = dict(current_payload)
            legacy_payload["candidate_lineage"] = {
                "schema_version": "true_rl_agent.candidate_lineage.v3",
                "legacy_public_score": 0.31,
            }
            legacy_ref = publish_prompt_payload(
                root / "legacy_prompt.json",
                legacy_payload,
            )
            restored = _source_repository_from_prompt_refs((legacy_ref,))
            merged = _merge_source_repository_payload(
                restored,
                current_payload,
            )

        self.assertEqual(
            {item["logical_id"] for item in restored["source_payloads"]},
            {CURRENT_BEST_SOURCE_ID},
        )
        self.assertEqual(
            merged["candidate_lineage"],
            current_payload["candidate_lineage"],
        )
        restored_current_best = next(
            item
            for item in restored["source_payloads"]
            if item["logical_id"] == CURRENT_BEST_SOURCE_ID
        )
        merged_current_best = next(
            item
            for item in merged["source_payloads"]
            if item["logical_id"] == CURRENT_BEST_SOURCE_ID
        )
        self.assertEqual(merged_current_best, restored_current_best)

    def test_source_repository_drops_stale_inline_slot_when_current_is_pending(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_candidate = _write_candidate_package(
                root,
                "candidate-first",
                "def first_reward(inp):\n    return inp.base_height\n",
            )
            missing_candidate = CandidatePackageRef(
                candidate_id="candidate-missing",
                path=str(root / "candidate-missing"),
                change_set=CandidateChangeSet(
                    frozenset({"reward_callable"})
                ),
            )
            builder = PublicContextBuilder()
            first_prompt = builder.build_prompt(
                [],
                latest_failed_candidate=first_candidate,
            ).provider_payload
            repository = _merge_source_repository_payload(
                None,
                first_prompt,
            )
            current_prompt = builder.build_prompt(
                [],
                latest_failed_candidate=missing_candidate,
            ).provider_payload

            merged = _merge_source_repository_payload(
                repository,
                current_prompt,
            )

        self.assertNotIn(
            LATEST_FAILED_SOURCE_ID,
            {
                item["logical_id"]
                for item in merged["source_payloads"]
            },
        )
        latest_item = next(
            item
            for item in merged["source_catalog"]["items"]
            if item["logical_id"] == LATEST_FAILED_SOURCE_ID
        )
        self.assertEqual(latest_item["status"], "pending")

    def test_source_repository_replaces_changed_inline_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_source = "def first_reward(inp):\n    return inp.base_height\n"
            second_source = "def second_reward(inp):\n    return inp.pitch\n"
            first_candidate = _write_candidate_package(
                root,
                "candidate-first",
                first_source,
            )
            second_candidate = _write_candidate_package(
                root,
                "candidate-second",
                second_source,
            )
            builder = PublicContextBuilder()
            first_prompt = builder.build_prompt(
                [],
                latest_failed_candidate=first_candidate,
            ).provider_payload
            repository = _merge_source_repository_payload(
                None,
                first_prompt,
            )
            second_prompt = builder.build_prompt(
                [],
                latest_failed_candidate=second_candidate,
            ).provider_payload

            merged = _merge_source_repository_payload(
                repository,
                second_prompt,
            )
            second_source_payload = (
                Path(second_candidate.path) / "training_code" / "components.py"
            ).read_bytes()

        latest_payload = next(
            item
            for item in merged["source_payloads"]
            if item["logical_id"] == LATEST_FAILED_SOURCE_ID
        )
        self.assertEqual(
            latest_payload["content"],
            second_source_payload.decode("utf-8"),
        )
        self.assertNotEqual(
            latest_payload["content_sha256"],
            hashlib.sha256(first_source.encode("utf-8")).hexdigest(),
        )

    def test_source_repository_retains_matching_payload_omitted_by_compact_frame(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_candidate = _write_candidate_package(
                root / "accepted",
                "accepted-current-best",
                "def current_best_reward(inp):\n    return inp.base_height\n",
            )
            failed_candidate = _write_candidate_package(
                root / "failed",
                "candidate-failed",
                "def failed_reward(inp):\n    return inp.pitch\n",
            )
            accepted_record = _accepted_record(
                "round-accepted",
                accepted_candidate,
                0.31,
            )
            builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            initial_prompt = builder.build_prompt(
                [accepted_record],
            ).provider_payload
            repository = _merge_source_repository_payload(
                None,
                initial_prompt,
            )
            compact_prompt = builder.build_prompt(
                [accepted_record],
                latest_failed_candidate=failed_candidate,
                candidate_repair_feedback={
                    "schema_version": (
                        "true_rl_agent.provider_repair_feedback.v1"
                    ),
                    "previous_candidate_id": failed_candidate.candidate_id,
                    "failure_category": "static_verifier_rejected",
                },
                frame_kind="static_repair",
            ).provider_payload

            merged = _merge_source_repository_payload(
                repository,
                compact_prompt,
            )

        compact_ids = {
            item["logical_id"] for item in compact_prompt["source_payloads"]
        }
        merged_ids = {
            item["logical_id"] for item in merged["source_payloads"]
        }
        self.assertNotIn(CURRENT_BEST_SOURCE_ID, compact_ids)
        self.assertEqual(
            merged_ids,
            {CURRENT_BEST_SOURCE_ID, LATEST_FAILED_SOURCE_ID},
        )

    def test_interrupted_prompt_consumes_the_sealed_provider_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round-001"
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            interrupted = _InterruptingProvider()
            first = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=interrupted,
                candidate_admission_policy=FakeVerifier(),
                round_dir=round_dir,
                max_attempts=1,
            )
            with self.assertRaises(KeyboardInterrupt):
                first.run_round(round_id="round-001", records=[])

            first_attempt = round_dir / "authoring_attempts" / "attempt_000001"
            self.assertTrue((first_attempt / "prompt_payload.json").is_file())
            self.assertFalse((first_attempt / "attempt_result.json").exists())

            resumed_provider = _ImmediateSuccessProvider()
            resumed_loop = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=resumed_provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=round_dir,
                max_attempts=1,
            )
            with self.assertRaisesRegex(
                AuthoringBudgetExhaustedError,
                "reserved=1 max=1",
            ):
                resumed_loop.run_round(round_id="round-001", records=[])

        self.assertEqual(resumed_provider.payloads, [])

    def test_interrupted_prompt_resumes_only_with_remaining_sealed_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round-001"
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            with self.assertRaises(KeyboardInterrupt):
                CandidateAuthoringLoop(
                    context_builder=context_builder,
                    provider=_InterruptingProvider(),
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=round_dir,
                    max_attempts=2,
                ).run_round(round_id="round-001", records=[])

            resumed_provider = _ImmediateSuccessProvider()
            resumed = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=resumed_provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=round_dir,
                max_attempts=2,
            ).run_round(round_id="round-001", records=[])
            evidence = verify_authoring_evidence(
                resumed.authoring_evidence_ref,
                expected_round_id="round-001",
                expected_selected_prompt=resumed.prompt_payload_ref,
            )
            anchor = verify_round_authoring_anchor(
                RoundAuthoringAnchorRef.from_mapping(
                    evidence["round_authoring_anchor"]
                ),
                expected_round_id="round-001",
            )

        self.assertEqual(resumed.prompt_payload_path.parent.name, "attempt_000002")
        self.assertEqual(
            [attempt["status"] for attempt in evidence["attempts"]],
            ["interrupted_after_prompt", "completed"],
        )
        self.assertEqual(evidence["selected_attempt_id"], "attempt_000002")
        self.assertEqual(
            anchor["prompt_payload"],
            evidence["attempts"][0]["prompt_payload"],
        )
        self.assertEqual(len(resumed_provider.payloads), 1)
        self.assertEqual(resumed.attempt_records[0]["attempt"], 2)

    def test_r8_shape_resume_skips_interrupted_attempt_and_keeps_truncation_feedback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round-001"
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            interrupted = _ThreeTruncationsThenInterruptProvider()
            with self.assertRaises(KeyboardInterrupt):
                CandidateAuthoringLoop(
                    context_builder=context_builder,
                    provider=interrupted,
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=round_dir,
                    max_attempts=5,
                ).run_round(round_id="round-001", records=[])

            resumed_provider = _ImmediateSuccessProvider()
            resumed = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=resumed_provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=round_dir,
                max_attempts=5,
            ).run_round(round_id="round-001", records=[])
            evidence = verify_authoring_evidence(
                resumed.authoring_evidence_ref,
                expected_round_id="round-001",
                expected_selected_prompt=resumed.prompt_payload_ref,
            )

        self.assertEqual(len(resumed_provider.payloads), 1)
        self.assertEqual(
            resumed_provider.payloads[0][
                "provider_or_runtime_repair_feedback"
            ]["failure_category"],
            "response_truncated",
        )
        self.assertEqual(
            [attempt["status"] for attempt in evidence["attempts"]],
            [
                "completed",
                "completed",
                "completed",
                "interrupted_after_prompt",
                "completed",
            ],
        )
        self.assertEqual(
            [attempt["attempt"] for attempt in resumed.attempt_records],
            [1, 2, 3, 5],
        )
        self.assertEqual(
            evidence["budget_contract"]["max_provider_attempts"],
            5,
        )
        self.assertEqual(evidence["selected_attempt_id"], "attempt_000005")

    def test_resume_restores_source_budget_feedback_and_followup_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round-001"
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            interrupted = _SourceRequestThenInterruptProvider()
            with self.assertRaises(KeyboardInterrupt):
                CandidateAuthoringLoop(
                    context_builder=context_builder,
                    provider=interrupted,
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=round_dir,
                    max_attempts=3,
                    max_source_requests_per_round=1,
                ).run_round(round_id="round-001", records=[])

            resumed_provider = _ImmediateSuccessProvider()
            resumed = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=resumed_provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=round_dir,
                max_attempts=3,
                max_source_requests_per_round=1,
            ).run_round(round_id="round-001", records=[])
            evidence = verify_authoring_evidence(
                resumed.authoring_evidence_ref,
                expected_round_id="round-001",
                expected_selected_prompt=resumed.prompt_payload_ref,
            )

        self.assertEqual(
            [attempt["status"] for attempt in evidence["attempts"]],
            ["completed", "interrupted_after_prompt", "completed"],
        )
        self.assertEqual(
            [attempt["attempt"] for attempt in resumed.attempt_records],
            [1, 3],
        )
        self.assertEqual(resumed.source_request_feedback["request_count"], 1)
        self.assertEqual(
            resumed.source_request_feedback["budget"]["source_request_count"],
            1,
        )
        self.assertEqual(
            resumed_provider.payloads[0]["authoring_context_frame"]["frame_kind"],
            "source_followup",
        )

    def test_resume_rejects_changed_authoring_budget_before_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "round-001"
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            with self.assertRaises(KeyboardInterrupt):
                CandidateAuthoringLoop(
                    context_builder=context_builder,
                    provider=_InterruptingProvider(),
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=round_dir,
                    max_attempts=2,
                ).run_round(round_id="round-001", records=[])

            resumed_provider = _ImmediateSuccessProvider()
            with self.assertRaisesRegex(ValueError, "sealed round contract"):
                CandidateAuthoringLoop(
                    context_builder=context_builder,
                    provider=resumed_provider,
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=round_dir,
                    max_attempts=3,
                ).run_round(round_id="round-001", records=[])

        self.assertEqual(resumed_provider.payloads, [])

    def test_actual_verifier_policy_is_forwarded_to_provider_prompt(self) -> None:
        policy = replace(
            StaticCandidateVerifierPolicy(),
            component_load_timeout_s=3.0,
        )
        static_verifier = StaticCandidateVerifier(policy=policy)
        provider = _ImmediateSuccessProvider()
        with tempfile.TemporaryDirectory() as tmp:
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=_PromptPolicyFakeVerifier(static_verifier),
                round_dir=Path(tmp) / "round-001",
                max_attempts=1,
            ).run_round(round_id="round-001", records=[])

        prompt_rules = provider.payloads[0]["required_candidate_package"][
            "component_contract"
        ]["static_verifier_rules"]
        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertFalse(prompt_rules["additional_files_allowed"])
        self.assertEqual(prompt_rules["component_load_timeout_s"], 3.0)
        self.assertEqual(
            prompt_rules["verification_policy_hash"],
            static_verifier.admission_policy_descriptor().policy_hash,
        )

    def test_prompt_fails_before_provider_when_verifier_blocks_visible_import(self) -> None:
        default_policy = StaticCandidateVerifierPolicy()
        allowed_imports = dict(default_policy.allowed_imports)
        component_module = "true_rl_agent.candidates.components"
        allowed_imports[component_module] = (
            allowed_imports[component_module] - {"RewardTerm"}
        )
        verifier = _PromptPolicyFakeVerifier(
            StaticCandidateVerifier(
                policy=replace(default_policy, allowed_imports=allowed_imports)
            )
        )
        provider = _ImmediateSuccessProvider()
        with tempfile.TemporaryDirectory() as tmp:
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=verifier,
                round_dir=Path(tmp) / "round-001",
                max_attempts=1,
            )
            with self.assertRaisesRegex(
                ValueError,
                "blocks prompt-visible component imports: RewardTerm",
            ):
                loop.run_round(round_id="round-001", records=[])

        self.assertEqual(provider.payloads, [])

    def test_authoring_capability_readiness_is_forwarded_to_provider_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _ImmediateSuccessProvider()
            loop = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=1,
                authoring_capability_readiness={
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
                candidate_admission_policy=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=2,
            )

            result = loop.run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-002")
        self.assertEqual(len(result.attempt_records), 2)
        self.assertEqual(result.attempt_records[0]["provider_status"], "fail")
        self.assertEqual(
            result.attempt_records[0]["provider_usage"],
            {"request_units": 99, "request_cache_miss_units": 99},
        )
        self.assertEqual(result.attempt_records[1]["provider_status"], "pass")
        self.assertEqual(provider.payloads[0]["authoring_context_frame"]["frame_kind"], "candidate_draft")
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "static_repair")
        for draft_context_section in (
            "public_feedback_summary",
            "runtime_feedback_summary",
            "public_history",
            "known_good_examples",
            "exploration_directive",
            "selection_guidance",
        ):
            self.assertIn(draft_context_section, provider.payloads[1])
        self.assertNotIn(
            LATEST_FAILED_SOURCE_ID,
            {
                item["logical_id"]
                for item in provider.payloads[1]["source_payloads"]
            },
        )
        repair_feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        repair_text = json.dumps(repair_feedback, sort_keys=True)
        self.assertEqual(repair_feedback["failure_category"], "json_parse_error")
        self.assertNotIn("D:/humanoid_agent", repair_text)
        self.assertNotIn("sk-live-secret", repair_text)
        self.assertNotIn("request_cache_miss_units", repair_text)

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
                candidate_admission_policy=FakeVerifier(),
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
        self.assertEqual(
            result.attempt_records[0]["provider_usage"],
            {"request_units": 70},
        )
        self.assertEqual(result.attempt_records[0]["provider_response_mode"], "source_request")
        self.assertNotIn("prompt_payload_path", result.attempt_records[0])
        self.assertEqual(result.attempt_records[0]["source_request_result"]["status"], "source_payload_returned")
        self.assertEqual(result.attempt_records[1]["provider_status"], "pass")
        self.assertEqual(
            result.attempt_records[1]["provider_usage"],
            {"request_units": 80, "request_cache_hit_units": 60},
        )
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
        self.assertNotIn(
            "provider_usage",
            repair_feedback["source_request"],
        )
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
                candidate_admission_policy=FakeVerifier(),
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
                candidate_admission_policy=FakeVerifier(),
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
                candidate_admission_policy=FakeVerifier(),
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
                candidate_admission_policy=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=3,
            )
            accepted_record = _accepted_record(
                "round-accepted",
                accepted_candidate,
                0.31,
            )
            runtime_failed_record = _accepted_record(
                "round-runtime-failed",
                runtime_failed_candidate,
                0.32,
            )
            runtime_failed_record["parent_program_source_checksum"] = (
                accepted_record["candidate"]["program_source_checksum"]
            )
            runtime_failed_record["decision"] = {
                "accepted": False,
                "outcome": "reject",
                "reason": "candidate_runtime_failed",
                "surface_classification": "code_level",
                "public_score": None,
                "comparison": {},
                "consumed_gates": [],
                "gate_statuses": {},
            }
            runtime_failed_record["train_result"] = None
            runtime_failed_record["public_result"] = None
            runtime_failed_record["jobs"]["train"][
                "candidate_failure_contract_hash"
            ] = TEST_CANDIDATE_FAILURE_CONTRACT.contract_hash
            runtime_failed_record["runner_error"] = {
                "type": "CandidateRuntimeError",
                "message": "candidate failed inside runner",
                "details": {
                    "status": "fail",
                    "subprocess_returncode": 1,
                    "candidate_failure_contract": (
                        TEST_CANDIDATE_FAILURE_CONTRACT.to_public_dict()
                    ),
                    "candidate_failure": (
                        TEST_CANDIDATE_FAILURE_CONTRACT.evidence(
                            stage="reward_execution",
                            reason_code="runtime_rejected",
                        ).to_public_dict()
                    ),
                },
            }

            result = loop.run_round(
                round_id="round-001",
                records=[accepted_record, runtime_failed_record],
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
                candidate_admission_policy=FakeVerifier(),
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

    def test_static_repair_payload_keeps_full_failed_source_but_not_draft_context(
        self,
    ) -> None:
        current_best_source = (
            "def current_best_reward(inp):\n    return inp.base_height\n"
            + "# current-best-real-shape-padding\n" * 170
        )
        failed_source = (
            "def failed_static_reward(inp):\n    return inp.base_height\n"
            + "# latest-failed-real-shape-padding\n" * 180
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_candidate = _write_candidate_package(
                root / "accepted",
                "accepted-current-best",
                current_best_source,
            )
            accepted_record = _accepted_record(
                "round-accepted",
                accepted_candidate,
                0.31,
            )
            provider = _VerifierFailThenInterruptProvider(
                root / "candidates",
                failed_source=failed_source,
            )
            with self.assertRaises(KeyboardInterrupt):
                CandidateAuthoringLoop(
                    context_builder=PublicContextBuilder(
                        task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                    ),
                    provider=provider,
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=root / "round-next",
                    max_attempts=3,
                ).run_round(
                    round_id="round-next",
                    records=[accepted_record],
                )
            failed_source_payload = (
                root / "candidates" / "candidate-001" / "training_code" / "components.py"
            ).read_bytes()

        initial_payload, repair_payload = provider.payloads
        repair_frame = repair_payload["authoring_context_frame"]
        repair_sources = repair_payload["source_payloads"]
        self.assertEqual(repair_frame["frame_kind"], "static_repair")
        self.assertEqual(
            [item["logical_id"] for item in repair_sources],
            [LATEST_FAILED_SOURCE_ID],
        )
        failed_payload = repair_sources[0]
        self.assertEqual(failed_payload["content"], failed_source_payload.decode("utf-8"))
        self.assertEqual(
            failed_payload["bytes"],
            len(failed_source_payload),
        )
        self.assertEqual(
            failed_payload["content_sha256"],
            hashlib.sha256(failed_source_payload).hexdigest(),
        )
        repair_text = serialize_prompt_payload(repair_payload)
        self.assertNotIn(current_best_source, repair_text)
        for omitted_section in (
            "public_feedback_summary",
            "runtime_feedback_summary",
            "public_history",
            "known_good_examples",
            "selection_guidance",
        ):
            self.assertNotIn(omitted_section, repair_payload)
        self.assertIn("exploration_directive", repair_payload)
        for stable_contract in (
            "task",
            "loop_contract",
            "required_candidate_package",
            "candidate_capabilities",
            "reward_inputs_contract",
            "public_eval_contract",
            "control_effectiveness_contract",
        ):
            self.assertEqual(
                repair_payload[stable_contract],
                initial_payload[stable_contract],
            )
        self.assertEqual(
            repair_payload["candidate_lineage"],
            initial_payload["candidate_lineage"],
        )
        repair_hashes = {
            item["logical_id"]: item["content_sha256"]
            for item in repair_frame["source_hashes"]
        }
        self.assertIn(CURRENT_BEST_SOURCE_ID, repair_hashes)
        self.assertIn(LATEST_FAILED_SOURCE_ID, repair_hashes)
        self.assertLess(
            repair_frame["budget_metadata"]["actual_prompt_payload_bytes"],
            initial_payload["authoring_context_frame"]["budget_metadata"][
                "actual_prompt_payload_bytes"
            ],
        )
        self.assertEqual(
            repair_frame["budget_metadata"]["actual_prompt_payload_bytes"],
            len(repair_text.encode("utf-8")),
        )

    def test_r10_name_only_repair_keeps_round_initial_mechanism_anchor(
        self,
    ) -> None:
        parent_source = _r10_components_source(
            component_name="accepted-balance-v1",
            include_support=False,
        )
        failed_name = "official_g1_rough_complete_reward_v1_plus_improved_slip_and_support"
        repaired_name = "g1_rough_stair_slip_support_v1"
        self.assertGreater(len(failed_name), 64)
        failed_source = _r10_components_source(
            component_name=failed_name,
            include_support=True,
        )
        repaired_source = _r10_components_source(
            component_name=repaired_name,
            include_support=True,
        )
        self.assertEqual(
            failed_source.replace(failed_name, repaired_name),
            repaired_source,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_candidate = _write_candidate_package(
                root / "accepted",
                "accepted-r2",
                parent_source,
            )
            accepted_record = _accepted_record(
                "round-r2",
                accepted_candidate,
                0.8557142857142856,
            )
            provider = _NameOnlyRepairProvider(
                root / "candidates",
                failed_source=failed_source,
                repaired_source=repaired_source,
            )
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=_IssueVerifier(
                    ["invalid_training_components_name"]
                ),
                round_dir=root / "round-r10",
                max_attempts=2,
            ).run_round(
                round_id="round-r10",
                records=[accepted_record],
            )
            evidence = verify_authoring_evidence(
                result.authoring_evidence_ref,
                expected_round_id="round-r10",
                expected_selected_prompt=result.prompt_payload_ref,
            )
            delta = project_selected_authoring_mechanism_delta(evidence)
            anchor = verify_round_authoring_anchor(
                RoundAuthoringAnchorRef.from_mapping(
                    evidence["round_authoring_anchor"]
                ),
                expected_round_id="round-r10",
            )
            raw_evidence_ref = result.authoring_evidence_ref.to_private_dict()
            evaluated_record = ExperimentRecord(
                round_id="round-r10",
                public_record={
                    "round_id": "round-r10",
                    "candidate": {"candidate_id": result.candidate.candidate_id},
                    "decision": {"accepted": False},
                    "parent_program_source_checksum": (
                        accepted_record["candidate"]["program_source_checksum"]
                    ),
                    "public_result": {"status": "pass"},
                },
                private_artifacts=(
                    PrivateArtifactRef(
                        kind=raw_evidence_ref["kind"],
                        uri=raw_evidence_ref["uri"],
                        checksum=raw_evidence_ref["checksum"],
                        size_bytes=raw_evidence_ref["size_bytes"],
                        reason=raw_evidence_ref["reason"],
                    ),
                ),
                prompt_payload_ref=result.prompt_payload_ref,
            )
            with patch(
                "true_rl_agent.agent_core.authoring_loop."
                "resolve_candidate_components_source_from_record",
                return_value={
                    "status": "inline",
                    "content": repaired_source,
                    "content_sha256": hashlib.sha256(
                        repaired_source.encode("utf-8")
                    ).hexdigest(),
                },
            ):
                evaluated_projection = _evaluated_candidate_sources(
                    [evaluated_record]
                )

        repair_source_ids = {
            item["logical_id"] for item in provider.payloads[1]["source_payloads"]
        }
        initial_parent_payload = next(
            item
            for item in provider.payloads[0]["source_payloads"]
            if item["logical_id"] == CURRENT_BEST_SOURCE_ID
        )
        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(
            provider.payloads[1]["authoring_context_frame"]["frame_kind"],
            "static_repair",
        )
        self.assertNotIn(CURRENT_BEST_SOURCE_ID, repair_source_ids)
        self.assertEqual(repair_source_ids, {LATEST_FAILED_SOURCE_ID})
        self.assertEqual(
            anchor["prompt_payload"],
            result.attempt_prompt_payload_refs[0].to_public_dict(),
        )
        self.assertEqual(
            evidence["attempts"][1]["provider_attempt"][
                "anti_degenerate_assessment"
            ]["comparison_anchor"],
            "current_best",
        )
        self.assertEqual(
            delta["changed_mechanisms"],
            ["RewardTerm:foot_contact_support_reward"],
        )
        self.assertEqual(
            delta["compares_to"],
            initial_parent_payload["content_sha256"],
        )
        self.assertEqual(
            anchor["parent_program_source_checksum"],
            accepted_record["candidate"]["program_source_checksum"],
        )
        self.assertEqual(
            evaluated_projection[0]["anchor_content"],
            initial_parent_payload["content"],
        )
        self.assertEqual(
            evaluated_projection[0]["anchor_content_sha256"],
            initial_parent_payload["content_sha256"],
        )
    def test_verifier_repair_instructions_are_specific_to_current_issue(self) -> None:
        cases = (
            (
                "name",
                ["invalid_training_components_name"],
            ),
            (
                "import",
                [
                    "forbidden_top_level:training_code/components.py:Import",
                    "forbidden_import:training_code/components.py:import",
                ],
            ),
        )
        for label, issues in cases:
            with self.subTest(issue_kind=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                provider = _VerifierFailThenSuccessProvider(root / "candidates")
                result = CandidateAuthoringLoop(
                    context_builder=PublicContextBuilder(
                        task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                    ),
                    provider=provider,
                    candidate_admission_policy=_IssueVerifier(issues),
                    round_dir=root / "round-001",
                    max_attempts=2,
                ).run_round(round_id="round-001", records=[])

                self.assertEqual(result.stop_reason, "candidate_verified")
                repair_payload = provider.payloads[1]
                feedback = repair_payload[
                    "provider_or_runtime_repair_feedback"
                ]
                instructions = "\n".join(feedback["instructions"])
                prompt_rules = repair_payload["required_candidate_package"][
                    "component_contract"
                ]["static_verifier_rules"]
                self.assertEqual(feedback["verifier_details"]["issues"], issues)
                self.assertEqual(
                    feedback["verifier_policy_ref"],
                    {
                        "verification_policy_id": prompt_rules[
                            "verification_policy_id"
                        ],
                        "verification_policy_hash": prompt_rules[
                            "verification_policy_hash"
                        ],
                    },
                )
                if label == "name":
                    name_contract = prompt_rules["component_callable_contract"][
                        "component_name_contract"
                    ]
                    self.assertIn(
                        f"{name_contract['max_chars']} characters",
                        instructions,
                    )
                    self.assertIn(
                        f"{name_contract['max_utf8_bytes']} UTF-8 bytes",
                        instructions,
                    )
                    self.assertIn("component_name_contract", instructions)
                    self.assertNotIn("torch", instructions.lower())
                    self.assertNotIn("closed-world", instructions.lower())
                else:
                    self.assertIn("Imports are closed-world", instructions)
                    self.assertIn("Do not import torch", instructions)
                    self.assertIn(
                        "from __future__ import annotations",
                        instructions,
                    )
                    self.assertIn(
                        "from true_rl_agent.candidates.components import",
                        instructions,
                    )
                    self.assertNotIn("64 characters", instructions)
                    self.assertNotIn("component_name_contract", instructions)

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
                candidate_admission_policy=FakeVerifier(),
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

    def test_static_repair_source_request_uses_all_prior_provider_visible_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_candidate = _write_candidate_package(
                root / "accepted",
                "accepted-current-best",
                "def current_best_reward(inp):\n    return inp.base_height\n",
            )
            accepted_record = _accepted_record(
                "round-accepted",
                accepted_candidate,
                0.31,
            )
            provider = _VerifierFailThenSourceRequestThenSuccessProvider(
                root / "candidates",
                requested_source_id=CURRENT_BEST_SOURCE_ID,
            )
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=root / "round-next",
                max_attempts=3,
            ).run_round(
                round_id="round-next",
                records=[accepted_record],
            )

        compact_ids = {
            item["logical_id"] for item in provider.payloads[1]["source_payloads"]
        }
        self.assertEqual(
            provider.payloads[1]["authoring_context_frame"]["frame_kind"],
            "static_repair",
        )
        self.assertNotIn(CURRENT_BEST_SOURCE_ID, compact_ids)
        self.assertEqual(
            result.attempt_records[1]["source_request_result"]["status"],
            "source_payload_returned",
        )
        self.assertEqual(
            result.attempt_records[1]["source_request_result"][
                "returned_logical_ids"
            ],
            [CURRENT_BEST_SOURCE_ID],
        )

    def test_resumed_static_repair_restores_all_prior_provider_visible_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "round-next"
            accepted_candidate = _write_candidate_package(
                root / "accepted",
                "accepted-current-best",
                "def current_best_reward(inp):\n    return inp.base_height\n",
            )
            accepted_record = _accepted_record(
                "round-accepted",
                accepted_candidate,
                0.31,
            )
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            )
            with self.assertRaises(KeyboardInterrupt):
                CandidateAuthoringLoop(
                    context_builder=context_builder,
                    provider=_VerifierFailThenInterruptProvider(
                        root / "candidates"
                    ),
                    candidate_admission_policy=FakeVerifier(),
                    round_dir=round_dir,
                    max_attempts=4,
                ).run_round(
                    round_id="round-next",
                    records=[accepted_record],
                )

            resumed_provider = _SourceRequestThenSuccessProvider()
            resumed = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=resumed_provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=round_dir,
                max_attempts=4,
            ).run_round(
                round_id="round-next",
                records=[accepted_record],
            )

        compact_ids = {
            item["logical_id"]
            for item in resumed_provider.payloads[0]["source_payloads"]
        }
        self.assertEqual(
            resumed_provider.payloads[0]["authoring_context_frame"][
                "frame_kind"
            ],
            "static_repair",
        )
        self.assertNotIn(CURRENT_BEST_SOURCE_ID, compact_ids)
        self.assertEqual(
            resumed.attempt_records[1]["source_request_result"]["status"],
            "source_payload_returned",
        )
        self.assertEqual(
            resumed.attempt_records[1]["source_request_result"][
                "returned_logical_ids"
            ],
            [CURRENT_BEST_SOURCE_ID],
        )

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
                candidate_admission_policy=FakeVerifier(),
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
                candidate_admission_policy=FakeVerifier(),
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
        self.assertEqual(result.prompt_payload_path.name, "prompt_payload.json")
        self.assertEqual(result.prompt_payload_path.parent.name, "attempt_000002")
        self.assertNotIn("/data/yuyangliu", result_text)
        self.assertNotIn("sk-live-secret", result_text)

    def test_nonretryable_transport_failure_stops_without_burning_attempt_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _TransportFailureProvider(retryable=False)
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=5,
            ).run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "provider_failed")
        self.assertEqual(len(provider.payloads), 1)
        self.assertEqual(len(result.attempt_records), 1)
        self.assertFalse(result.provider_failure["details"]["retryable"])

    def test_retryable_transport_failure_uses_existing_attempt_budget_without_code_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _TransportFailureProvider(retryable=True)
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=2,
            ).run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "provider_failed")
        self.assertEqual(len(provider.payloads), 2)
        retry_feedback = provider.payloads[1][
            "provider_or_runtime_repair_feedback"
        ]
        self.assertEqual(
            retry_feedback["schema_version"],
            "true_rl_agent.provider_transport_retry_feedback.v1",
        )
        self.assertIn("same candidate authoring request", retry_feedback["instructions"][0])
        self.assertNotIn("components_py", json.dumps(retry_feedback, sort_keys=True))

    def test_legacy_runtime_failure_without_retryability_keeps_attempt_budget_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _TransportFailureProvider(retryable=None)
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=2,
            ).run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "provider_failed")
        self.assertEqual(len(provider.payloads), 2)
        self.assertEqual(len(result.attempt_records), 2)
        self.assertNotIn("retryable", result.provider_failure["details"])
        self.assertEqual(
            provider.payloads[1]["provider_or_runtime_repair_feedback"][
                "schema_version"
            ],
            "true_rl_agent.provider_repair_feedback.v1",
        )

    def test_output_truncations_remain_bounded_by_the_round_attempt_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _AlwaysTruncatedProvider()
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=Path(tmp) / "round-001",
                max_attempts=5,
            ).run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "provider_failed")
        self.assertEqual(len(provider.payloads), 5)
        self.assertEqual(len(result.attempt_records), 5)
        self.assertEqual(result.provider_failure["category"], "response_truncated")
        self.assertEqual(
            provider.payloads[1]["provider_or_runtime_repair_feedback"][
                "failure_category"
            ],
            "response_truncated",
        )
        receipt_text = json.dumps(
            result.attempt_records[-1]["provider_response_receipt"],
            sort_keys=True,
        )
        self.assertIn("disabled", receipt_text)
        self.assertNotIn("chain of thought", receipt_text)

    def test_truncation_after_verifier_failure_preserves_latest_failed_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _VerifierFailThenTruncationThenSuccessProvider(
                root / "candidates"
            )
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                ),
                provider=provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=root / "round-001",
                max_attempts=3,
            ).run_round(round_id="round-001", records=[])

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "candidate-003")
        self.assertEqual(len(result.attempt_records), 3)
        third_sources = {
            item["logical_id"]: item
            for item in provider.payloads[2]["source_payloads"]
        }
        self.assertIn(LATEST_FAILED_SOURCE_ID, third_sources)
        self.assertIn(
            "failed_static_reward",
            third_sources[LATEST_FAILED_SOURCE_ID]["content"],
        )
        second_payload = provider.payloads[1]
        third_payload = provider.payloads[2]
        self.assertEqual(
            third_payload["authoring_context_frame"]["frame_kind"],
            "static_repair",
        )
        for compact_section in (
            "public_feedback_summary",
            "runtime_feedback_summary",
            "public_history",
            "known_good_examples",
            "selection_guidance",
        ):
            self.assertNotIn(compact_section, second_payload)
            self.assertNotIn(compact_section, third_payload)
        self.assertLessEqual(
            third_payload["authoring_context_frame"]["budget_metadata"][
                "actual_prompt_payload_bytes"
            ],
            second_payload["authoring_context_frame"]["budget_metadata"][
                "actual_prompt_payload_bytes"
            ]
            + 4096,
        )
        self.assertEqual(
            third_payload["provider_or_runtime_repair_feedback"][
                "failure_category"
            ],
            "response_truncated",
        )

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
                candidate_admission_policy=FakeVerifier(),
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
                candidate_admission_policy=FakeVerifier(),
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
            "no_component_mechanism_delta",
        )
        self.assertEqual(assessment["matched_round_id"], "round-evaluated")
        self.assertEqual(assessment["matched_candidate_id"], "evaluated-drift")
        self.assertEqual(
            provider.payloads[1]["authoring_context_frame"]["frame_kind"],
            "static_repair",
        )
        feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        self.assertEqual(feedback["status"], "rewrite_required")
        self.assertEqual(
            feedback["failure_category"],
            "authoring_policy_violation",
        )
        self.assertEqual(
            feedback["anti_degenerate_assessment"]["matched_candidate_id"],
            "evaluated-drift",
        )
        self.assertNotIn(str(root), json.dumps(feedback, sort_keys=True))

    def test_evaluated_source_projection_carries_parent_and_decision(self) -> None:
        anchor_source = "def build_training_components(): pass\n"
        anchor_sha256 = hashlib.sha256(anchor_source.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            prompt_ref = publish_prompt_payload(
                Path(tmp) / "historical_prompt.json",
                {
                    "source_payloads": [
                        {
                            "logical_id": (
                                "current_best/training_code/components.py"
                            ),
                            "content": anchor_source,
                            "content_sha256": anchor_sha256,
                        }
                    ]
                },
            )
            record = {
                "round_id": "round-rejected",
                "candidate": {"candidate_id": "rejected-contact"},
                "decision": {"accepted": False},
                "parent_program_source_checksum": "sha256:" + ("a" * 64),
                "prompt_payload": prompt_ref.to_public_dict(),
                "public_result": {"status": "pass"},
            }
            with patch(
                "true_rl_agent.agent_core.authoring_loop."
                "resolve_candidate_components_source_from_record",
                return_value={
                    "status": "inline",
                    "content": "def build_training_components(): pass\n",
                    "content_sha256": "b" * 64,
                },
            ):
                projected = _evaluated_candidate_sources([record])

        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0]["accepted"], False)
        self.assertEqual(
            projected[0]["parent_program_source_checksum"],
            "sha256:" + ("a" * 64),
        )
        self.assertEqual(projected[0]["anchor_content"], anchor_source)
        self.assertEqual(projected[0]["anchor_content_sha256"], anchor_sha256)
        self.assertEqual(projected[0]["anchor_role"], "current_best")

        with patch(
            "true_rl_agent.agent_core.authoring_loop."
            "resolve_candidate_components_source_from_record",
            return_value={
                "status": "inline",
                "content": "def build_training_components(): pass\n",
                "content_sha256": "b" * 64,
            },
        ):
            missing_prompt_projection = _evaluated_candidate_sources([record])
        self.assertEqual(len(missing_prompt_projection), 1)
        self.assertNotIn("anchor_content", missing_prompt_projection[0])
        self.assertNotIn("anchor_content_sha256", missing_prompt_projection[0])


if __name__ == "__main__":
    unittest.main()
