from __future__ import annotations

import copy
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

from tests.fakes.evaluation import make_test_candidate_package  # noqa: E402
from tests.fakes.fake_components import (  # noqa: E402
    FAKE_RSL_RL_AGENT_CONFIGURATION,
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
    FakeVerifier,
    prepare_fake_session_genesis,
)
from true_rl_agent.agent_core.authoring_response import (  # noqa: E402
    CandidateDraftResponse,
    SourceRequestResponse,
)
from true_rl_agent.agent_core.ledger import load_ledger  # noqa: E402
from true_rl_agent.agent_core.live_loop import (
    run_live_loop as _run_live_loop,  # noqa: E402
)
from true_rl_agent.agent_core.provider_errors import (
    ProviderCandidateError,  # noqa: E402
)
from true_rl_agent.candidates.schema import (  # noqa: E402
    CandidateChangeSet,
    CandidatePackageRef,
)
from true_rl_agent.evaluation.decision_record_contract import (  # noqa: E402
    validate_decision_record,
)
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.policy import PolicyInitialization  # noqa: E402
from true_rl_agent.prompt_artifacts import (  # noqa: E402
    load_ledger_authoring_evidence,
)
from true_rl_agent.prompting.source_catalog import (  # noqa: E402
    CURRENT_BEST_SOURCE_ID,
    MAX_PROMPT_SOURCE_INLINE_BYTES,
    prompt_source_catalog_logical_ids,
)
from true_rl_agent.runner.base import CandidateRuntimeError  # noqa: E402
from true_rl_agent.runner.candidate_failure import (  # noqa: E402
    CandidateFailureContract,
    CandidateFailureRule,
)
from true_rl_agent.runner.subprocess_runner import SubprocessRunner  # noqa: E402
from true_rl_agent.tasks.component_smoke import (  # noqa: E402
    ComponentSmokeFixture,
    SmokeTensor,
)
from true_rl_agent.tasks.push_recovery.reward_inputs import (
    PushRecoveryRewardInputs,  # noqa: E402
)

_LIVE_TEST_FAILURE_CONTRACT = CandidateFailureContract(
    contract_id="true_rl_agent.test.live_candidate_failure.v1",
    rules=(
        CandidateFailureRule(
            stage="reward_execution",
            reason_codes=("runtime_rejected",),
            requires_initial_policy_state=False,
        ),
    ),
)


def run_live_loop(**kwargs):
    initial_best_score = kwargs.pop("initial_best_score")
    kwargs["session_genesis"] = prepare_fake_session_genesis(
        output_dir=kwargs["output_dir"],
        session_id=kwargs["session_id"],
        initial_best_score=initial_best_score,
        task_plugin=kwargs["task_plugin"],
        backend_adapter=kwargs["backend_adapter"],
        runner=kwargs["runner"],
    )
    return _run_live_loop(**kwargs)


def _completed_provider_attempts(ledger: list[dict]) -> list[dict]:
    attempts: list[dict] = []
    for record in ledger:
        _, evidence = load_ledger_authoring_evidence(record)
        attempts.extend(
            dict(item["provider_attempt"])
            for item in evidence["attempts"]
            if item["status"] == "completed"
        )
    return attempts


class _SequenceProvider:
    def __init__(self) -> None:
        self.contexts: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        public_context = prompt_payload["public_context"]
        self.contexts.append(public_context)
        index = len(self.contexts)
        return make_test_candidate_package(
            f"candidate-{index:03d}",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.20 + 0.01 * index},
        )


class _InterruptProvider:
    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        raise KeyboardInterrupt("private interrupt detail")


class _RepairProvider:
    def __init__(self) -> None:
        self.contexts: list[dict] = []
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        public_context = prompt_payload["public_context"]
        self.contexts.append(public_context)
        index = len(self.contexts)
        verifier_status = "fail" if index == 1 else "pass"
        return make_test_candidate_package(
            f"candidate-{index:03d}",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.25, "verifier_status": verifier_status},
        )


class _PathSourceRepairProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        if index == 1:
            candidate_dir = self.candidate_root / "candidate-001"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "candidate_manifest.yaml").write_text(
                "schema_version: true_rl_agent.candidate.v1\n"
                "candidate_id: candidate-001\n"
                "entrypoint: training_code.components:build_training_components\n",
                encoding="utf-8",
            )
            (candidate_dir / "training_code" / "components.py").write_text(
                "\n".join(
                    [
                        "PATH_A = 'D:/humanoid_agent/private/raw.txt'",
                        "PATH_B = '/data/yuyangliu/secrets/key.env'",
                        "def build_training_components():",
                        "    raise RuntimeError(PATH_A + PATH_B)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_candidate_metadata(candidate_dir)
            return CandidatePackageRef(
                candidate_id="candidate-001",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.25, "verifier_status": "fail"},
            )
        return make_test_candidate_package(
            "candidate-002",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.25, "verifier_status": "pass"},
        )


class _RuntimeRepairProvider:
    def __init__(self) -> None:
        self.contexts: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        public_context = prompt_payload["public_context"]
        self.contexts.append(public_context)
        index = len(self.contexts)
        return make_test_candidate_package(
            f"candidate-{index:03d}",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30},
        )


class _RuntimePathRepairProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        if index == 1:
            candidate_dir = self.candidate_root / "candidate-001"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "candidate_manifest.yaml").write_text(
                "schema_version: true_rl_agent.candidate.v1\n"
                "candidate_id: candidate-001\n"
                "entrypoint: training_code.components:build_training_components\n",
                encoding="utf-8",
            )
            (candidate_dir / "training_code" / "components.py").write_text(
                "def runtime_failed_reward(inp):\n    return inp.base_height\n",
                encoding="utf-8",
            )
            _write_candidate_metadata(candidate_dir)
            return CandidatePackageRef(
                candidate_id="candidate-001",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.30},
            )
        candidate_dir = self.candidate_root / "candidate-002"
        (candidate_dir / "training_code").mkdir(parents=True)
        (candidate_dir / "candidate_manifest.yaml").write_text(
            "schema_version: true_rl_agent.candidate.v1\n"
            "candidate_id: candidate-002\n"
            "entrypoint: training_code.components:build_training_components\n",
            encoding="utf-8",
        )
        (candidate_dir / "training_code" / "components.py").write_text(
            "def repaired_reward(inp):\n    return inp.base_height\n",
            encoding="utf-8",
        )
        _write_candidate_metadata(candidate_dir)
        return CandidatePackageRef(
            candidate_id="candidate-002",
            path=str(candidate_dir),
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30},
        )


class _RuntimeThenLocalSuccessProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        term_name = {
            1: "runtime_failed_reward",
            2: "accepted_success_reward",
        }.get(index, "fresh_success_reward")
        reward_expression = (
            "inp.base_height" if index < 3 else "inp.base_height.relu()"
        )
        candidate = _write_candidate_package(
            self.candidate_root,
            f"candidate-{index:03d}",
            (
                "from true_rl_agent.candidates.components import "
                "RewardTerm, TrainingComponents\n\n"
                f"def {term_name}(inp):\n    return {reward_expression}\n\n"
                "def build_training_components():\n"
                "    return TrainingComponents(\n"
                "        name='runtime_retry_components',\n"
                f"        reward_terms=[RewardTerm(name={term_name!r}, "
                f"weight=1.0, fn={term_name})],\n"
                "    )\n"
            ),
        )
        return replace(
            candidate,
            metadata={
                **dict(candidate.metadata),
                "fake_public_score": 0.30 + 0.10 * index,
            },
        )


class _AlwaysVerifierFailProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        return make_test_candidate_package(
            f"candidate-{index:03d}",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30, "verifier_status": "fail"},
        )


def _write_candidate_metadata(candidate_dir: Path) -> None:
    (candidate_dir / "rationale.md").write_text("Integration candidate.\n", encoding="utf-8")
    (candidate_dir / "operation_diff.json").write_text("{}\n", encoding="utf-8")
    (candidate_dir / "diagnostics.json").write_text("{}\n", encoding="utf-8")


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
    _write_candidate_metadata(candidate_dir)
    return CandidatePackageRef(
        candidate_id=candidate_id,
        path=str(candidate_dir),
        change_set=CandidateChangeSet(frozenset({"reward_callable"})),
        metadata={"fake_public_score": 0.30, "verifier_status": "pass"},
    )


class _AlwaysSmokeFailProvider:
    def __init__(self, candidate_root: Path) -> None:
        self.candidate_root = candidate_root
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        return _write_candidate_package(
            self.candidate_root,
            f"candidate-smoke-fail-{index:03d}",
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


class _SmokeTaskPlugin(FakeTaskPlugin):
    def component_smoke_adapter(self):
        return _SmokeAdapter()


class _TrainCountingRunner(FakeRunner):
    def __init__(self) -> None:
        self.train_calls = 0

    def run_train(self, train_spec):
        self.train_calls += 1
        return super().run_train(train_spec)


class _PromptPayloadOnlyProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        return make_test_candidate_package(
            f"payload-candidate-{index:03d}",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.20 + 0.01 * index},
        )


class _DeniedSourceThenSuccessProvider:
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
                logical_ids=(CURRENT_BEST_SOURCE_ID,),
                rationale="Need unavailable current-best source.",
                source_catalog_logical_ids=tuple(
                    sorted(prompt_source_catalog_logical_ids(prompt_payload))
                ),
            )
        index = len(self.payloads)
        return CandidateDraftResponse(
            candidate=make_test_candidate_package(
                f"source-feedback-candidate-{index:03d}",
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.20 + 0.01 * index},
            )
        )


class _DeniedSourceThenStaticFailureProvider(_DeniedSourceThenSuccessProvider):
    def next_authoring_response(
        self,
        *,
        prompt_payload: dict,
    ) -> SourceRequestResponse | CandidateDraftResponse:
        response = super().next_authoring_response(prompt_payload=prompt_payload)
        if isinstance(response, CandidateDraftResponse) and len(self.payloads) == 2:
            return CandidateDraftResponse(
                candidate=replace(
                    response.candidate,
                    metadata={**dict(response.candidate.metadata), "verifier_status": "fail"},
                )
            )
        return response


class _ProviderFailureThenSuccessProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads)
        if index == 1:
            raise ProviderCandidateError(
                "json_parse_error",
                "candidate proposal was not valid JSON",
                details={"raw_excerpt": "sk-live-secret D:\\humanoid_agent\\private\\raw.txt"},
            )
        return make_test_candidate_package(
            f"provider-repair-{index:03d}",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={"fake_public_score": 0.30},
        )


class _AlwaysProviderFailureProvider:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        raise ProviderCandidateError(
            "schema_error",
            "components_py failed at D:/humanoid_agent/private/raw.txt",
            details={"raw_excerpt": "sk-live-secret D:/humanoid_agent/private/raw.txt"},
        )


class _FailIfCalledProvider:
    def __init__(self) -> None:
        self.calls = 0

    def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
        _ = prompt_payload
        self.calls += 1
        raise AssertionError("provider must not run after replay rejects the ledger")


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
                metadata={
                    **dict(candidate.metadata),
                    "fake_public_score": 0.25,
                    "verifier_status": "fail",
                },
            )
        raise ProviderCandidateError(
            "schema_error",
            "components_py failed at D:/humanoid_agent/private/raw.txt",
            details={"raw_excerpt": "sk-live-secret /data/yuyangliu/secrets/key.env"},
        )


class _RuntimeFailOnceRunner(FakeRunner):
    def run_train(self, train_spec):
        if train_spec.subject.subject_id == "candidate-001":
            raise CandidateRuntimeError(
                "candidate program failed during reward_execution (runtime_rejected)",
                public_details={
                    "status": "fail",
                    "subprocess_returncode": 1,
                    "candidate_failure_contract": (
                        _LIVE_TEST_FAILURE_CONTRACT.to_public_dict()
                    ),
                    "candidate_failure": (
                        _LIVE_TEST_FAILURE_CONTRACT.evidence(
                            stage="reward_execution",
                            reason_code="runtime_rejected",
                        ).to_public_dict()
                    ),
                },
            )
        return super().run_train(train_spec)


class _CandidateFailureContractBackend(FakeBackendAdapter):
    def materialize_jobs(self, **kwargs):
        jobs = super().materialize_jobs(**kwargs)
        return replace(
            jobs,
            train_spec=replace(
                jobs.train_spec,
                candidate_failure_contract=_LIVE_TEST_FAILURE_CONTRACT,
            ),
        )


class _NonzeroFailSummaryBackend(FakeBackendAdapter):
    """Run a real local subprocess that reports an identity-bound train failure."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def materialize_jobs(self, **kwargs):
        jobs = super().materialize_jobs(**kwargs)
        candidate = kwargs["candidate"]
        train_spec = replace(
            jobs.train_spec,
            subject_manifest_hash=candidate.subject_manifest_hash,
            program_source_checksum=candidate.program_source_checksum,
            candidate_artifact=candidate,
            candidate_failure_contract=_LIVE_TEST_FAILURE_CONTRACT,
        )
        round_dir = self.output_root / kwargs["round_id"]
        summary_path = round_dir / "train-summary.json"
        checkpoint_output_path = round_dir / "checkpoint.pt"
        script = "\n".join(
            (
                "import json",
                "import sys",
                "from pathlib import Path",
                "summary_path = Path(sys.argv[1])",
                "summary_path.parent.mkdir(parents=True, exist_ok=True)",
                "summary = {",
                "    'job_id': sys.argv[2],",
                "    'task_id': sys.argv[3],",
                "    'subject_kind': sys.argv[4],",
                "    'subject_id': sys.argv[5],",
                "    'subject_manifest_hash': sys.argv[6],",
                "    'training_binding_hash': sys.argv[7],",
                "    'capability_profile_hash': sys.argv[8],",
                "    'program_source_checksum': sys.argv[9],",
                "    'policy_architecture_hash': sys.argv[10],",
                "    'optimizer_training_profile_hash': sys.argv[11],",
                "    'agent_configuration_hash': sys.argv[12],",
                "    'initialization_seed_schedule_hash': sys.argv[13],",
                "    'execution_device': sys.argv[14],",
                "    'candidate_failure_contract_hash': sys.argv[15],",
                "    'status': 'fail',",
                "    'candidate_failure': json.loads(sys.argv[16]),",
                "    'errors': [],",
                "}",
                "summary_path.write_text(json.dumps(summary), encoding='utf-8')",
                "raise SystemExit(1)",
            )
        )
        metadata = {
            **dict(train_spec.metadata),
            "cli_argv": [
                sys.executable,
                "-c",
                script,
                str(summary_path),
                train_spec.job_id,
                train_spec.task_id,
                train_spec.subject.kind,
                train_spec.subject.subject_id,
                train_spec.subject_manifest_hash,
                train_spec.training_binding.binding_hash,
                train_spec.training_binding.capability_profile_hash,
                train_spec.program_source_checksum,
                train_spec.training_binding.policy_architecture_hash,
                train_spec.training_binding.optimizer_training_profile_hash,
                train_spec.training_binding.agent_configuration_hash,
                train_spec.training_binding.initialization_seed_schedule_hash,
                train_spec.execution_device,
                _LIVE_TEST_FAILURE_CONTRACT.contract_hash,
                json.dumps(
                    _LIVE_TEST_FAILURE_CONTRACT.evidence(
                        stage="reward_execution",
                        reason_code="runtime_rejected",
                    ).to_public_dict()
                ),
                "--job-id",
                train_spec.job_id,
                "--round",
                kwargs["round_id"],
                "--task-id",
                train_spec.task_id,
                "--subject-kind",
                train_spec.subject.kind,
                "--subject-id",
                train_spec.subject.subject_id,
                "--subject-manifest-hash",
                train_spec.subject_manifest_hash,
                "--program-source-checksum",
                train_spec.program_source_checksum,
                "--candidate",
                candidate.path,
                "--candidate-artifact-hash",
                candidate.artifact_hash,
                "--budget-steps",
                str(train_spec.budget_steps),
                "--init",
                train_spec.policy_initialization.kind,
                "--summary-path",
                str(summary_path),
                "--checkpoint-output",
                str(checkpoint_output_path),
                "--training-binding-hash",
                str(train_spec.training_binding.binding_hash),
                "--capability-profile-hash",
                train_spec.training_binding.capability_profile_hash,
                "--policy-architecture-hash",
                train_spec.training_binding.policy_architecture_hash,
                "--optimizer-training-profile-hash",
                train_spec.training_binding.optimizer_training_profile_hash,
                "--agent-configuration-hash",
                train_spec.training_binding.agent_configuration_hash,
                "--agent-configuration-json",
                FAKE_RSL_RL_AGENT_CONFIGURATION.to_canonical_json(),
                "--initialization-seed-schedule-hash",
                train_spec.training_binding.initialization_seed_schedule_hash,
                "--seed",
                "42",
                "--device",
                train_spec.execution_device,
                "--candidate-failure-contract-hash",
                _LIVE_TEST_FAILURE_CONTRACT.contract_hash,
            ],
            "summary_path": str(summary_path),
            "checkpoint_output_path": str(checkpoint_output_path),
        }
        metadata["cli_command"] = " ".join(metadata["cli_argv"])
        return replace(
            jobs,
            train_spec=replace(train_spec, metadata=metadata),
        )


class LiveLoopTests(unittest.TestCase):
    def test_prompt_capacity_preflight_fails_before_invocation_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch(
                "true_rl_agent.agent_core.live_loop.preflight_task_prompt_capacity",
                side_effect=ValueError("minimum core does not fit"),
            ):
                with self.assertRaisesRegex(ValueError, "minimum core does not fit"):
                    run_live_loop(
                        output_dir=output_dir,
                        rounds=1,
                        session_id="live-prompt-preflight",
                        task_plugin=FakeTaskPlugin(),
                        backend_adapter=FakeBackendAdapter(),
                        runner=FakeRunner(),
                        candidate_admission_policy=FakeVerifier(),
                        provider=_SequenceProvider(),
                        policy_initialization=PolicyInitialization.random(),
                        initial_best_score=0.10,
                    )

            self.assertFalse((output_dir / "live_invocations").exists())
            self.assertEqual(load_ledger(output_dir / "ledger.jsonl"), [])

    def test_denied_source_then_same_round_success_reaches_next_round_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            provider = _DeniedSourceThenSuccessProvider()
            run_live_loop(
                output_dir=output_dir,
                rounds=2,
                session_id="live-source-feedback",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(output_dir / "ledger.jsonl")

        feedback = ledger[0]["source_request_feedback"]
        next_round_feedback = provider.payloads[2]["public_history"]["recent_outcomes"][-1][
            "source_request_feedback"
        ]

        self.assertEqual(feedback["status"], "source_request_denied")
        self.assertEqual(
            feedback["denied_logical_ids"],
            [{"logical_id": CURRENT_BEST_SOURCE_ID, "reason": "source_not_requestable"}],
        )
        self.assertEqual(feedback["budget"]["max_source_requests_per_round"], 2)
        self.assertEqual(feedback["budget"]["max_source_ids_per_request"], 2)
        self.assertEqual(feedback["budget"]["max_source_request_bytes_per_round"], 48 * 1024)
        self.assertEqual(next_round_feedback, feedback)
        self.assertNotIn("provider_attempt_records", ledger[0])
        self.assertNotIn("source_request_feedback", ledger[0]["public_context"])
        feedback_text = json.dumps(feedback, sort_keys=True)
        for forbidden in ("rationale", "instructions", "content", "memory://"):
            self.assertNotIn(forbidden, feedback_text)

    def test_final_attempt_source_denial_uses_same_next_round_feedback_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            provider = _DeniedSourceThenSuccessProvider()
            run_live_loop(
                output_dir=output_dir,
                rounds=2,
                session_id="live-final-source-denial",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=1,
            )
            ledger = load_ledger(output_dir / "ledger.jsonl")

        feedback = ledger[0]["source_request_feedback"]
        next_round_feedback = provider.payloads[1]["public_history"]["recent_outcomes"][-1][
            "source_request_feedback"
        ]

        self.assertEqual(ledger[0]["provider_failure"]["category"], "source_request_denied")
        self.assertNotIn("provider_attempt_records", ledger[0])
        self.assertEqual(next_round_feedback, feedback)
        self.assertEqual(feedback["status"], "source_request_denied")
        self.assertEqual(
            feedback["denied_logical_ids"][0]["reason"],
            "source_not_requestable",
        )

    def test_source_feedback_is_persisted_on_static_authoring_failure_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            provider = _DeniedSourceThenStaticFailureProvider()
            run_live_loop(
                output_dir=output_dir,
                rounds=2,
                session_id="live-static-source-feedback",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(output_dir / "ledger.jsonl")

        feedback = ledger[0]["source_request_feedback"]

        self.assertIsNotNone(ledger[0]["static_authoring_failure"])
        self.assertNotIn("provider_attempt_records", ledger[0])
        self.assertEqual(feedback["status"], "source_request_denied")
        self.assertEqual(
            provider.payloads[2]["public_history"]["recent_outcomes"][-1][
                "source_request_feedback"
            ],
            feedback,
        )

    def test_live_loop_saves_prompt_payloads_and_ledger_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _SequenceProvider()
            summary = run_live_loop(
                output_dir=Path(tmp),
                rounds=2,
                session_id="live-test",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
            )
            ledger = load_ledger(Path(tmp) / "ledger.jsonl")
            payloads = [
                json.loads(Path(path).read_text(encoding="utf-8"))
                for path in summary["prompt_payload_paths"]
            ]
            invocation_dirs = list((Path(tmp) / "live_invocations").iterdir())
            invocation_start = json.loads(
                (invocation_dirs[0] / "start.json").read_text(encoding="utf-8")
            )
            invocation_exit = json.loads(
                (invocation_dirs[0] / "exit.json").read_text(encoding="utf-8")
            )
            live_summary_bytes = (Path(tmp) / "live_summary.json").read_bytes()

        self.assertEqual(summary["mode"], "live")
        self.assertEqual(summary["rounds_completed"], 2)
        self.assertEqual(len(summary["prompt_payload_paths"]), 2)
        self.assertEqual(len(provider.contexts), 2)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(len(invocation_dirs), 1)
        self.assertEqual(invocation_start["closed_ledger_count_at_start"], 0)
        self.assertEqual(invocation_exit["status"], "completed")
        self.assertEqual(invocation_exit["termination_kind"], "normal")
        self.assertEqual(invocation_exit["closed_ledger_count_at_exit"], 2)
        self.assertFalse(invocation_exit["ledger_authority"])
        self.assertEqual(
            invocation_exit["live_summary"]["checksum"],
            "sha256:" + hashlib.sha256(live_summary_bytes).hexdigest(),
        )
        self.assertTrue(all(Path(path).name == "prompt_payload.json" for path in summary["prompt_payload_paths"]))
        for payload in payloads:
            self.assertEqual(payload["schema_version"], "true_rl_agent.prompt_payload.v4")
            self.assertNotIn("trusted_seed", json.dumps(payload, sort_keys=True))
        self.assertEqual(summary["train_result_count"], 2)
        self.assertEqual(summary["public_result_count"], 2)
        self.assertEqual(summary["runner_candidate_count"], 2)
        self.assertEqual(summary["trainable_round_count"], 2)
        self.assertEqual(summary["authored_candidate_count"], 2)
        self.assertEqual(summary["trained_candidate_count"], 2)
        self.assertEqual(summary["full_eval_completed_count"], 2)
        self.assertEqual(summary["sealed_validation_completed_count"], 2)
        self.assertEqual(summary["accepted_count"], 1)
        self.assertEqual(summary["eval_incomplete_count"], 0)
        self.assertTrue(summary["checks"]["eval_run_audit_passed"])
        self.assertEqual(summary["eval_run_audit"]["failure_buckets"]["accepted"], 1)
        self.assertEqual(summary["eval_run_audit"]["failure_buckets"]["decision_threshold"], 1)
        self.assertEqual(summary["authoring_failure_round_count"], 0)
        self.assertNotIn("candidate_count_matches_rounds", summary["checks"])
        self.assertTrue(summary["checks"]["runner_candidate_count_matches_trainable_rounds"])
        self.assertTrue(summary["checks"]["authoring_failures_accounted_for"])
        self.assertTrue(summary["checks"]["at_least_one_train_result"])
        self.assertTrue(summary["checks"]["at_least_one_public_result"])
        self.assertEqual(summary["final_best_eval_stats"]["mode"], "accepted")
        self.assertEqual(ledger[1]["decision"]["comparison"]["current_best"]["accepted_round_id"], "live-test-round-001")
        self.assertAlmostEqual(ledger[1]["decision"]["comparison"]["best_score_stats"]["mean"], 0.21)
        self.assertIn("sealed_validation_gate", ledger[1]["decision"]["consumed_gates"])

    def test_live_loop_records_keyboard_interrupt_without_ledgering_partial_round(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(KeyboardInterrupt):
                run_live_loop(
                    output_dir=root,
                    rounds=1,
                    session_id="live-interrupt-test",
                    task_plugin=FakeTaskPlugin(),
                    backend_adapter=FakeBackendAdapter(),
                    runner=FakeRunner(),
                    candidate_admission_policy=FakeVerifier(),
                    provider=_InterruptProvider(),
                    policy_initialization=PolicyInitialization.random(),
                    initial_best_score=0.10,
                    policy_accept_delta=0.05,
                )
            invocation_dirs = list((root / "live_invocations").iterdir())
            start = json.loads(
                (invocation_dirs[0] / "start.json").read_text(encoding="utf-8")
            )
            exit_record = json.loads(
                (invocation_dirs[0] / "exit.json").read_text(encoding="utf-8")
            )

            self.assertEqual(load_ledger(root / "ledger.jsonl"), [])
            self.assertFalse((root / "live_summary.json").exists())
            self.assertEqual(len(invocation_dirs), 1)
            self.assertEqual(start["closed_ledger_count_at_start"], 0)
            self.assertEqual(exit_record["status"], "interrupted")
            self.assertEqual(
                exit_record["termination_kind"],
                "keyboard_interrupt",
            )
            self.assertEqual(exit_record["exception_type"], "KeyboardInterrupt")
            self.assertNotIn("private interrupt detail", json.dumps(exit_record))
            self.assertEqual(exit_record["closed_ledger_count_at_exit"], 0)

    def test_completion_validator_failure_is_not_recorded_as_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def reject_summary(summary: dict) -> None:
                self.assertEqual(summary["rounds_completed"], 1)
                raise RuntimeError("private validator detail")

            with self.assertRaisesRegex(RuntimeError, "private validator detail"):
                run_live_loop(
                    output_dir=root,
                    rounds=1,
                    session_id="live-validator-failure",
                    task_plugin=FakeTaskPlugin(),
                    backend_adapter=FakeBackendAdapter(),
                    runner=FakeRunner(),
                    candidate_admission_policy=FakeVerifier(),
                    provider=_SequenceProvider(),
                    policy_initialization=PolicyInitialization.random(),
                    initial_best_score=0.10,
                    policy_accept_delta=0.05,
                    completion_validator=reject_summary,
                )
            invocation_dir = next((root / "live_invocations").iterdir())
            exit_record = json.loads(
                (invocation_dir / "exit.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(load_ledger(root / "ledger.jsonl")), 1)
            self.assertEqual(exit_record["status"], "failed")
            self.assertEqual(exit_record["termination_kind"], "exception")
            self.assertEqual(exit_record["exception_type"], "RuntimeError")
            self.assertNotIn("private validator detail", json.dumps(exit_record))
            self.assertNotIn("live_summary", exit_record)

    def test_lifecycle_write_failure_does_not_mask_original_exception(self) -> None:
        class _RuntimeErrorProvider:
            def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
                _ = prompt_payload
                raise RuntimeError("private provider detail")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "true_rl_agent.agent_core.live_invocation.LiveInvocationJournal.finish",
                side_effect=OSError("private lifecycle detail"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "private provider detail",
                ) as raised:
                    run_live_loop(
                        output_dir=root,
                        rounds=1,
                        session_id="live-lifecycle-write-failure",
                        task_plugin=FakeTaskPlugin(),
                        backend_adapter=FakeBackendAdapter(),
                        runner=FakeRunner(),
                        candidate_admission_policy=FakeVerifier(),
                        provider=_RuntimeErrorProvider(),
                        policy_initialization=PolicyInitialization.random(),
                        initial_best_score=0.10,
                        policy_accept_delta=0.05,
                    )

            invocation_dir = next((root / "live_invocations").iterdir())
            start_record = json.loads(
                (invocation_dir / "start.json").read_text(encoding="utf-8")
            )

            self.assertEqual(load_ledger(root / "ledger.jsonl"), [])
            self.assertFalse((invocation_dir / "exit.json").exists())
            self.assertNotIn("private provider detail", json.dumps(start_record))
            add_note = getattr(raised.exception, "add_note", None)
            notes = getattr(raised.exception, "__notes__", ())
            if callable(add_note):
                self.assertTrue(any("OSError" in note for note in notes))
            else:
                self.assertEqual(notes, ())

    def test_lifecycle_write_failure_preserves_exception_without_add_note(self) -> None:
        class _Python310StyleRuntimeError(RuntimeError):
            add_note = None

        class _RuntimeErrorProvider:
            def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
                _ = prompt_payload
                raise _Python310StyleRuntimeError("private provider detail")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "true_rl_agent.agent_core.live_invocation.LiveInvocationJournal.finish",
                side_effect=OSError("private lifecycle detail"),
            ):
                with self.assertRaisesRegex(
                    _Python310StyleRuntimeError,
                    "private provider detail",
                ):
                    run_live_loop(
                        output_dir=root,
                        rounds=1,
                        session_id="live-python310-lifecycle-write-failure",
                        task_plugin=FakeTaskPlugin(),
                        backend_adapter=FakeBackendAdapter(),
                        runner=FakeRunner(),
                        candidate_admission_policy=FakeVerifier(),
                        provider=_RuntimeErrorProvider(),
                        policy_initialization=PolicyInitialization.random(),
                        initial_best_score=0.10,
                        policy_accept_delta=0.05,
                    )

            invocation_dir = next((root / "live_invocations").iterdir())
            self.assertEqual(load_ledger(root / "ledger.jsonl"), [])
            self.assertFalse((invocation_dir / "exit.json").exists())

    def test_live_loop_passes_saved_prompt_payload_to_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _PromptPayloadOnlyProvider()
            summary = run_live_loop(
                output_dir=Path(tmp),
                rounds=1,
                session_id="live-payload",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
            )
            saved_payload = json.loads(Path(summary["prompt_payload_paths"][0]).read_text(encoding="utf-8"))
            ledger = load_ledger(Path(tmp) / "ledger.jsonl")

        self.assertEqual(len(provider.payloads), 1)
        self.assertEqual(provider.payloads[0], saved_payload)
        self.assertIn("schema_version", provider.payloads[0])
        self.assertNotIn("candidate_component_contract", provider.payloads[0])
        self.assertIn("component_contract", provider.payloads[0]["required_candidate_package"])
        self.assertEqual(saved_payload["task"]["task_id"], "fake_push_recovery")
        self.assertEqual(saved_payload["task"]["prompt_adapter"], "neutral_fallback")
        self.assertEqual(saved_payload["reward_inputs_contract"]["status"], "not_available")
        self.assertNotIn("task_prompt_sections", saved_payload["public_context"])
        self.assertEqual(list(saved_payload)[0], "schema_version")
        self.assertLess(
            list(saved_payload).index("reward_inputs_contract"),
            list(saved_payload).index("public_history"),
        )
        self.assertEqual(ledger[0]["prompt_payload"]["path"], summary["prompt_payload_paths"][0])
        self.assertEqual(ledger[0]["prompt_payload"]["visibility"], "public_prompt_payload")

    def test_live_loop_repairs_provider_candidate_before_consuming_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _RepairProvider()
            summary = run_live_loop(
                output_dir=Path(tmp),
                rounds=1,
                session_id="live-repair",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(Path(tmp) / "ledger.jsonl")

        self.assertEqual(len(provider.contexts), 2)
        repair_feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        self.assertEqual(repair_feedback["failure_category"], "verifier_error")
        self.assertEqual(repair_feedback["previous_candidate_id"], "candidate-001")
        instructions = " ".join(repair_feedback["instructions"])
        self.assertIn("Fix only the current verifier failure", instructions)
        self.assertNotIn("Do not import torch", instructions)
        self.assertNotIn("candidate_repair_feedback", provider.contexts[1])
        self.assertEqual(
            provider.contexts[1]["candidate_repair_feedback_ref"]["canonical_section"],
            "provider_or_runtime_repair_feedback",
        )
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["candidate"]["candidate_id"], "candidate-002")
        self.assertEqual(ledger[0]["candidate_surface"]["classification"], "code_level")
        self.assertEqual(summary["rounds_completed"], 1)
        self.assertEqual(summary["provider_attempt_count"], 2)
        selected_prompt = Path(summary["prompt_payload_paths"][0])
        self.assertEqual(selected_prompt.name, "prompt_payload.json")
        self.assertEqual(selected_prompt.parent.name, "attempt_000002")
        self.assertEqual(ledger[0]["prompt_payload"]["path"], summary["prompt_payload_paths"][0])
        self.assertEqual(ledger[0]["prompt_payload"]["visibility"], "public_prompt_payload")
        self.assertTrue(summary["checks"]["at_least_one_train_result"])

    def test_live_loop_scrubs_paths_from_previous_components_in_repair_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _PathSourceRepairProvider(root / "candidates")
            run_live_loop(
                output_dir=root / "run",
                rounds=1,
                session_id="live-path-source-repair",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )

        self.assertEqual(len(provider.payloads), 2)
        repair_feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        source_payloads = provider.payloads[1]["source_payloads"]
        failed_sources = [
            item
            for item in source_payloads
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        ]
        repair_text = json.dumps(repair_feedback, sort_keys=True)
        payload_text = json.dumps(provider.payloads[1], sort_keys=True)
        self.assertNotIn("previous_components_py", repair_feedback)
        self.assertEqual(
            repair_feedback["previous_candidate_source_logical_id"],
            "attempt/latest_failed/training_code/components.py",
        )
        self.assertEqual(len(failed_sources), 1)
        self.assertIn("[path]", failed_sources[0]["content"])
        self.assertNotIn("D:/humanoid_agent", repair_text)
        self.assertNotIn("D:/humanoid_agent", payload_text)
        self.assertNotIn("/data/yuyangliu", repair_text)
        self.assertNotIn("/data/yuyangliu", payload_text)
        self.assertNotIn("private/raw.txt", repair_text)
        self.assertNotIn("private/raw.txt", payload_text)
        self.assertNotIn("/secrets/", repair_text)
        self.assertNotIn("/secrets/", payload_text)

    def test_live_loop_rejects_oversized_failed_source_before_repair_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / "candidates" / "candidate-001"
            oversized = "def oversized_reward(inp):\n    return inp.base_height\n" + (
                "# filler\n" * (MAX_PROMPT_SOURCE_INLINE_BYTES // 4)
            )

            class _OversizedRepairProvider(_PathSourceRepairProvider):
                def next_candidate(self, *, prompt_payload: dict) -> CandidatePackageRef:
                    self.payloads.append(prompt_payload)
                    index = len(self.payloads)
                    if index == 1:
                        candidate = _write_candidate_package(
                            candidate_dir.parent,
                            "candidate-001",
                            oversized,
                        )
                        return replace(
                            candidate,
                            metadata={
                                **dict(candidate.metadata),
                                "fake_public_score": 0.25,
                                "verifier_status": "fail",
                            },
                        )
                    return make_test_candidate_package(
                        "candidate-002",
                        change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                        metadata={"fake_public_score": 0.25, "verifier_status": "pass"},
                    )

            provider = _OversizedRepairProvider(root / "candidates")
            run_live_loop(
                output_dir=root / "run",
                rounds=1,
                session_id="live-oversized-source-repair",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )

        payload = provider.payloads[1]
        failed_item = next(
            item
            for item in payload["source_catalog"]["items"]
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        )
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertEqual(failed_item["status"], "pending")
        self.assertEqual(
            failed_item["pending_reason"],
            "candidate_components_source_exceeds_package_limit",
        )
        self.assertNotIn("content_sha256", failed_item)
        self.assertNotIn("bytes", failed_item)
        self.assertNotIn("# filler", payload_text)
        self.assertEqual(payload["source_payloads"], [])

    def test_live_loop_repairs_provider_parse_failure_before_consuming_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _ProviderFailureThenSuccessProvider()
            summary = run_live_loop(
                output_dir=Path(tmp),
                rounds=1,
                session_id="live-provider-repair",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(Path(tmp) / "ledger.jsonl")
            attempts = _completed_provider_attempts(ledger)

        self.assertEqual(len(provider.payloads), 2)
        repair_feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        repair_text = json.dumps(repair_feedback, sort_keys=True)
        self.assertEqual(repair_feedback["failure_category"], "json_parse_error")
        self.assertEqual(repair_feedback["provider_status"], "fail")
        self.assertIn("Return one valid JSON object", " ".join(repair_feedback["instructions"]))
        self.assertNotIn("sk-live-secret", repair_text)
        self.assertNotIn("D:\\humanoid_agent", repair_text)
        self.assertEqual(attempts[0]["provider_status"], "fail")
        self.assertEqual(
            attempts[0]["provider_failure"]["category"], "json_parse_error"
        )
        self.assertEqual(attempts[1]["provider_status"], "pass")
        self.assertEqual(ledger[0]["candidate"]["candidate_id"], "provider-repair-002")
        self.assertEqual(summary["rounds_completed"], 1)

    def test_live_loop_records_provider_failure_when_all_attempts_fail_before_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _AlwaysProviderFailureProvider()
            summary = run_live_loop(
                output_dir=Path(tmp),
                rounds=1,
                session_id="live-provider-fail",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(Path(tmp) / "ledger.jsonl")

        self.assertEqual(len(provider.payloads), 2)
        repair_feedback = provider.payloads[1]["provider_or_runtime_repair_feedback"]
        self.assertEqual(repair_feedback["failure_category"], "schema_error")
        ledger_text = json.dumps(ledger, sort_keys=True)
        self.assertNotIn("D:/humanoid_agent", ledger_text)
        self.assertNotIn("private/raw.txt", ledger_text)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(
            ledger[0]["ledger_epoch_hash"],
            summary["ledger_epoch_hash"],
        )
        self.assertEqual(ledger[0]["decision"]["reason"], "provider_candidate_generation_failed")
        self.assertEqual(ledger[0]["provider_failure"]["category"], "schema_error")
        self.assertIsNone(ledger[0]["candidate"])
        self.assertEqual(summary["rounds_completed"], 1)
        self.assertEqual(summary["provider_failure_count"], 1)
        self.assertEqual(summary["candidate_paths"], [])
        self.assertEqual(summary["runner_candidate_count"], 0)
        self.assertEqual(summary["trainable_round_count"], 0)
        self.assertEqual(summary["authoring_failure_round_count"], 1)
        self.assertNotIn("candidate_count_matches_rounds", summary["checks"])
        self.assertTrue(summary["checks"]["runner_candidate_count_matches_trainable_rounds"])
        self.assertTrue(summary["checks"]["authoring_failures_accounted_for"])

    def test_resume_rejects_all_unsealed_authoring_ledger_projections(self) -> None:
        def tamper_provider_failure(record: dict) -> None:
            record["provider_failure"]["category"] = "timeout"

        def add_attempt_copy(record: dict) -> None:
            record["provider_attempt_records"] = [{"fabricated": True}]

        def remove_source_feedback(record: dict) -> None:
            record["source_request_feedback"] = None

        def tamper_static_failure(record: dict) -> None:
            record["static_authoring_failure"]["feedback"][
                "failure_category"
            ] = "fabricated_feedback_category"

        cases = (
            (
                "provider_failure",
                _AlwaysProviderFailureProvider,
                1,
                tamper_provider_failure,
            ),
            (
                "attempt_copy",
                _AlwaysProviderFailureProvider,
                1,
                add_attempt_copy,
            ),
            (
                "source_feedback",
                _DeniedSourceThenStaticFailureProvider,
                2,
                remove_source_feedback,
            ),
            (
                "static_failure",
                _DeniedSourceThenStaticFailureProvider,
                2,
                tamper_static_failure,
            ),
        )
        for case, provider_factory, max_attempts, tamper in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)
                session_id = f"live-authoring-projection-{case}"
                task = FakeTaskPlugin()
                backend = FakeBackendAdapter()
                runner = FakeRunner()
                run_live_loop(
                    output_dir=output_dir,
                    rounds=1,
                    session_id=session_id,
                    task_plugin=task,
                    backend_adapter=backend,
                    runner=runner,
                    candidate_admission_policy=FakeVerifier(),
                    provider=provider_factory(),
                    policy_initialization=PolicyInitialization.random(),
                    initial_best_score=0.10,
                    policy_accept_delta=0.05,
                    max_provider_attempts=max_attempts,
                )
                ledger_path = output_dir / "ledger.jsonl"
                record = load_ledger(ledger_path)[0]
                tamper(record)
                ledger_path.write_bytes(
                    canonical_json_bytes(record, trailing_newline=True)
                )
                resumed_provider = _FailIfCalledProvider()

                with self.assertRaisesRegex(
                    ValueError,
                    "authoring_evidence_outcome_projection_mismatch",
                ):
                    run_live_loop(
                        output_dir=output_dir,
                        rounds=2,
                        session_id=session_id,
                        task_plugin=task,
                        backend_adapter=backend,
                        runner=runner,
                        candidate_admission_policy=FakeVerifier(),
                        provider=resumed_provider,
                        policy_initialization=PolicyInitialization.random(),
                        initial_best_score=0.10,
                        policy_accept_delta=0.05,
                        max_provider_attempts=max_attempts,
                    )

                self.assertEqual(resumed_provider.calls, 0)

    def test_live_loop_mixed_verifier_then_provider_failure_projects_sealed_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _VerifierFailThenProviderFailureProvider(root / "candidates")
            summary = run_live_loop(
                output_dir=root / "run",
                rounds=1,
                session_id="live-mixed-provider-fail",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=FakeRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(root / "run" / "ledger.jsonl")
            attempts = _completed_provider_attempts(ledger)

        self.assertEqual(len(provider.payloads), 2)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["decision"]["reason"], "provider_candidate_generation_failed")
        self.assertEqual(ledger[0]["provider_failure"]["category"], "schema_error")
        self.assertEqual(summary["provider_failure_count"], 1)
        self.assertEqual(summary["candidate_paths"], [])
        self.assertNotIn("provider_attempt_records", ledger[0])
        attempt_text = json.dumps(attempts, sort_keys=True)
        self.assertEqual(summary["provider_attempt_count"], 2)
        self.assertEqual(len(attempts), 2)
        self.assertNotIn("candidate_path", attempts[0])
        self.assertEqual(attempts[0]["candidate_id"], "candidate-001")
        self.assertEqual(attempts[0]["verifier_status"], "fail")
        self.assertNotIn("candidate_path", attempts[0])
        self.assertEqual(attempts[0]["candidate_source_visibility"], "private_non_prompt_artifact")
        self.assertFalse(attempts[0]["candidate_source_included_in_prompt"])
        self.assertNotIn(str(root / "candidates"), attempt_text)
        self.assertNotIn("training_code", attempt_text)
        self.assertNotIn("components.py", attempt_text)
        self.assertNotIn("D:/humanoid_agent", attempt_text)
        self.assertNotIn("/data/yuyangliu", attempt_text)
        self.assertNotIn("sk-live-secret", attempt_text)

    def test_live_loop_records_max_attempt_verifier_failure_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _AlwaysVerifierFailProvider()
            runner = _TrainCountingRunner()
            summary = run_live_loop(
                output_dir=Path(tmp),
                rounds=1,
                session_id="live-static-fail",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(Path(tmp) / "ledger.jsonl")
            attempts = _completed_provider_attempts(ledger)

        self.assertEqual(len(provider.payloads), 2)
        self.assertEqual(provider.payloads[0]["authoring_context_frame"]["frame_kind"], "candidate_draft")
        self.assertEqual(provider.payloads[1]["authoring_context_frame"]["frame_kind"], "static_repair")
        self.assertEqual(runner.train_calls, 0)
        self.assertEqual(
            ledger[0]["ledger_epoch_hash"],
            summary["ledger_epoch_hash"],
        )
        self.assertEqual(ledger[0]["decision"]["reason"], "candidate_static_authoring_failed")
        self.assertEqual(ledger[0]["candidate_surface"]["classification"], "static_authoring_failure")
        self.assertEqual(ledger[0]["static_authoring_failure"]["failure_category"], "verifier_error")
        self.assertEqual(ledger[0]["candidate"]["candidate_id"], "candidate-002")
        self.assertNotIn("provider_attempt_records", ledger[0])
        self.assertIsNone(ledger[0]["train_result"])
        self.assertIsNone(ledger[0]["public_result"])
        self.assertEqual(summary["provider_failure_count"], 0)
        self.assertEqual(summary["static_authoring_failure_count"], 1)
        self.assertEqual(summary["authoring_failure_round_count"], 1)
        self.assertEqual(summary["train_result_count"], 0)
        self.assertEqual(summary["public_result_count"], 0)
        self.assertEqual(summary["candidate_paths"], [])
        self.assertEqual(summary["runner_candidate_count"], 0)
        self.assertEqual(summary["trainable_round_count"], 0)
        self.assertNotIn("candidate_count_matches_rounds", summary["checks"])
        self.assertTrue(summary["checks"]["runner_candidate_count_matches_trainable_rounds"])
        self.assertTrue(summary["checks"]["authoring_failures_accounted_for"])
        self.assertEqual(attempts[0]["verifier_status"], "fail")
        self.assertEqual(attempts[1]["verifier_status"], "fail")

    def test_live_loop_records_max_attempt_smoke_failure_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _AlwaysSmokeFailProvider(root / "candidates")
            runner = _TrainCountingRunner()
            summary = run_live_loop(
                output_dir=root / "run",
                rounds=1,
                session_id="live-smoke-static-fail",
                task_plugin=_SmokeTaskPlugin(),
                backend_adapter=FakeBackendAdapter(),
                runner=runner,
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=1,
            )
            ledger = load_ledger(root / "run" / "ledger.jsonl")
            attempts = _completed_provider_attempts(ledger)

        ledger_text = json.dumps(ledger, sort_keys=True)
        self.assertEqual(len(provider.payloads), 1)
        self.assertEqual(runner.train_calls, 0)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["decision"]["reason"], "candidate_static_authoring_failed")
        self.assertEqual(ledger[0]["static_authoring_failure"]["failure_category"], "component_smoke_error")
        self.assertNotIn("provider_attempt_records", ledger[0])
        self.assertEqual(
            attempts[0]["component_smoke"]["status"],
            "fail",
        )
        self.assertIsNone(ledger[0]["train_result"])
        self.assertIsNone(ledger[0]["public_result"])
        self.assertEqual(summary["static_authoring_failure_count"], 1)
        self.assertEqual(summary["authoring_failure_round_count"], 1)
        self.assertEqual(summary["train_result_count"], 0)
        self.assertEqual(summary["public_result_count"], 0)
        self.assertEqual(summary["candidate_paths"], [])
        self.assertEqual(summary["runner_candidate_count"], 0)
        self.assertEqual(summary["trainable_round_count"], 0)
        self.assertNotIn("candidate_count_matches_rounds", summary["checks"])
        self.assertTrue(summary["checks"]["runner_candidate_count_matches_trainable_rounds"])
        self.assertTrue(summary["checks"]["authoring_failures_accounted_for"])
        self.assertNotIn(str(root / "candidates"), ledger_text)
        self.assertNotIn("D:/humanoid_agent", ledger_text)
        self.assertNotIn("/data/yuyangliu", ledger_text)

    def test_live_loop_records_runtime_failure_without_same_round_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = _RuntimeRepairProvider()
            summary = run_live_loop(
                output_dir=Path(tmp),
                rounds=1,
                session_id="live-runtime-repair",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=_CandidateFailureContractBackend(),
                runner=_RuntimeFailOnceRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(Path(tmp) / "ledger.jsonl")

        self.assertEqual(len(provider.contexts), 1)
        self.assertNotIn("candidate_repair_feedback", provider.contexts[0])
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["candidate"]["candidate_id"], "candidate-001")
        self.assertEqual(ledger[0]["decision"]["reason"], "candidate_runtime_failed")
        self.assertEqual(ledger[0]["decision"]["outcome"], "reject")
        self.assertIsNotNone(ledger[0]["runner_error"])
        self.assertEqual(summary["provider_attempt_count"], 1)
        self.assertEqual(summary["rounds_completed"], 1)
        self.assertEqual(summary["train_result_count"], 0)
        self.assertEqual(summary["public_result_count"], 0)
        self.assertEqual(summary["runner_candidate_count"], 1)
        self.assertEqual(summary["trainable_round_count"], 1)
        self.assertEqual(summary["authoring_failure_round_count"], 0)
        self.assertNotIn("candidate_count_matches_rounds", summary["checks"])
        self.assertTrue(summary["checks"]["runner_candidate_count_matches_trainable_rounds"])
        self.assertTrue(summary["checks"]["authoring_failures_accounted_for"])
        self.assertTrue(summary["checks"]["eval_run_audit_passed"])
        self.assertEqual(
            summary["eval_run_audit"]["failure_buckets"],
            {"candidate_runtime": 1},
        )

    def test_next_round_prompt_uses_latest_runtime_failed_source_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _RuntimePathRepairProvider(root / "candidates")
            summary = run_live_loop(
                output_dir=root / "run",
                rounds=2,
                session_id="live-runtime-source-repair",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=_CandidateFailureContractBackend(),
                runner=_RuntimeFailOnceRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(root / "run" / "ledger.jsonl")

        self.assertEqual(len(provider.payloads), 2)
        self.assertEqual(summary["provider_attempt_count"], 2)
        self.assertEqual(ledger[0]["candidate"]["candidate_id"], "candidate-001")
        self.assertEqual(ledger[0]["decision"]["reason"], "candidate_runtime_failed")
        self.assertEqual(ledger[1]["candidate"]["candidate_id"], "candidate-002")

        next_payload = provider.payloads[1]
        public_history_text = json.dumps(next_payload["public_history"], sort_keys=True)
        payload_text = json.dumps(next_payload, sort_keys=True)
        failed_sources = [
            item
            for item in next_payload["source_payloads"]
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        ]

        self.assertNotIn("provider_or_runtime_repair_feedback", next_payload)
        self.assertIn("candidate_runtime_failed", public_history_text)
        self.assertIn("CandidateRuntimeError", public_history_text)
        self.assertEqual(len(failed_sources), 1)
        self.assertIn("runtime_failed_reward", failed_sources[0]["content"])
        self.assertNotIn(str(root / "candidates"), payload_text)

    def test_nonzero_train_failure_summary_is_ledgered_and_next_round_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _RuntimePathRepairProvider(root / "candidates")
            summary = run_live_loop(
                output_dir=root / "run",
                rounds=2,
                session_id="live-subprocess-train-failure",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=_NonzeroFailSummaryBackend(root / "jobs"),
                runner=SubprocessRunner(
                    allowed_executables=(sys.executable,),
                    timeout_s=10,
                ),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(root / "run" / "ledger.jsonl")

        first_error = ledger[0]["runner_error"]
        next_history_text = json.dumps(
            provider.payloads[1]["public_history"],
            sort_keys=True,
        )

        self.assertEqual(summary["rounds_completed"], 2)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(len(provider.payloads), 2)
        self.assertEqual(ledger[0]["decision"]["reason"], "candidate_runtime_failed")
        self.assertEqual(ledger[1]["decision"]["reason"], "candidate_runtime_failed")
        self.assertEqual(ledger[1]["candidate"]["candidate_id"], "candidate-002")
        self.assertIsNone(ledger[0]["train_result"])
        self.assertIsNone(ledger[0]["public_result"])
        self.assertEqual(first_error["type"], "CandidateRuntimeError")
        self.assertEqual(first_error["details"]["status"], "fail")
        self.assertEqual(first_error["details"]["subprocess_returncode"], 1)
        self.assertEqual(
            summary["eval_run_audit"]["rounds"][0]["round_stage"],
            "train_failure",
        )
        self.assertTrue(
            summary["eval_run_audit"]["rounds"][0]["round_stage_consistent"]
        )
        self.assertTrue(summary["checks"]["eval_run_audit_passed"])
        self.assertEqual(
            summary["eval_run_audit"]["failure_buckets"],
            {"candidate_runtime": 2},
        )
        self.assertEqual(summary["provider_attempt_count"], 2)
        self.assertIn("candidate_runtime_failed", next_history_text)
        self.assertIn("CandidateRuntimeError", next_history_text)

        acceptance_protocol = FakeTaskPlugin().task_spec.acceptance_protocol
        self.assertTrue(
            validate_decision_record(
                ledger[0],
                acceptance_protocol=acceptance_protocol,
            ).valid
        )
        forged_owner = copy.deepcopy(ledger[0])
        forged_owner["runner_error"]["details"]["candidate_failure"][
            "failure_owner"
        ] = "infrastructure"
        self.assertIn(
            "candidate_runtime_error_details_invalid",
            validate_decision_record(
                forged_owner,
                acceptance_protocol=acceptance_protocol,
            ).reasons,
        )
        forged_contract = copy.deepcopy(ledger[0])
        forged_contract["jobs"]["train"][
            "candidate_failure_contract_hash"
        ] = "f" * 64
        self.assertIn(
            "candidate_failure_contract_hash_mismatch",
            validate_decision_record(
                forged_contract,
                acceptance_protocol=acceptance_protocol,
            ).reasons,
        )
        forged_reason = copy.deepcopy(ledger[0])
        forged_reason["runner_error"]["details"]["candidate_failure"][
            "reason_code"
        ] = "forged_reason"
        self.assertIn(
            "candidate_runtime_error_details_invalid",
            validate_decision_record(
                forged_reason,
                acceptance_protocol=acceptance_protocol,
            ).reasons,
        )

    def test_runtime_failed_source_does_not_persist_after_success_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _RuntimeThenLocalSuccessProvider(root / "candidates")
            summary = run_live_loop(
                output_dir=root / "run",
                rounds=3,
                session_id="live-runtime-source-drop",
                task_plugin=FakeTaskPlugin(),
                backend_adapter=_CandidateFailureContractBackend(),
                runner=_RuntimeFailOnceRunner(),
                candidate_admission_policy=FakeVerifier(),
                provider=provider,
                policy_initialization=PolicyInitialization.random(),
                initial_best_score=0.10,
                policy_accept_delta=0.05,
                max_provider_attempts=2,
            )
            ledger = load_ledger(root / "run" / "ledger.jsonl")

        self.assertEqual(len(provider.payloads), 3)
        self.assertEqual(summary["provider_attempt_count"], 3)
        self.assertEqual(ledger[0]["decision"]["reason"], "candidate_runtime_failed")
        self.assertTrue(ledger[1]["decision"]["accepted"])
        self.assertTrue(summary["checks"]["eval_run_audit_passed"])
        self.assertEqual(
            summary["eval_run_audit"]["failure_buckets"],
            {"accepted": 2, "candidate_runtime": 1},
        )

        round_2_payload = provider.payloads[1]
        round_3_payload = provider.payloads[2]
        round_2_failed_sources = [
            item
            for item in round_2_payload["source_payloads"]
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        ]
        round_3_failed_sources = [
            item
            for item in round_3_payload["source_payloads"]
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        ]
        round_3_failed_item = next(
            item
            for item in round_3_payload["source_catalog"]["items"]
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        )
        round_3_text = json.dumps(round_3_payload, sort_keys=True)

        self.assertEqual(len(round_2_failed_sources), 1)
        self.assertIn("runtime_failed_reward", round_2_failed_sources[0]["content"])
        self.assertEqual(round_3_failed_item["status"], "pending")
        self.assertEqual(round_3_failed_item["pending_reason"], "no_latest_failed_candidate")
        self.assertEqual(round_3_failed_sources, [])
        self.assertIn("accepted_success_reward", round_3_text)
        self.assertNotIn("runtime_failed_reward", round_3_text)

    def test_live_loop_refuses_ledger_without_genesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ledger.jsonl").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "durable state but no Genesis"):
                run_live_loop(
                    output_dir=Path(tmp),
                    rounds=1,
                    session_id="live-test",
                    task_plugin=FakeTaskPlugin(),
                    backend_adapter=FakeBackendAdapter(),
                    runner=FakeRunner(),
                    candidate_admission_policy=FakeVerifier(),
                    provider=_SequenceProvider(),
                    policy_initialization=PolicyInitialization.random(),
                    initial_best_score=0.10,
                    policy_accept_delta=0.05,
                )


if __name__ == "__main__":
    unittest.main()
