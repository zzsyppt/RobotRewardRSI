from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.fakes.fake_components import (
    FAKE_ACCEPTANCE_PROTOCOL,
    FAKE_SESSION_GENESIS_HASH,
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
    FakeVerifier,
    fake_bootstrap_current_best,
    fake_evaluation_binding,
    fake_training_binding,
)
from true_rl_agent.agent_core.authoring_loop import (
    CandidateAuthoringLoop,
    CandidateAuthoringResult,
)
from true_rl_agent.agent_core.generation import (
    FixedWidthGenerationInputs,
    GenerationLaneRuntime,
    _verified_authoring_anchor_prompt,
    execute_fixed_width_generation,
)
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_MANIFEST_SCHEMA_VERSION,
    GENERATION_MANIFEST_V2_SCHEMA_VERSION,
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationLaneAssignment,
    GenerationManifest,
    GenerationSessionContract,
    generation_resource_plan_hash,
    generation_training_initialization_hash,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    GenerationContinuationPolicy,
    GenerationLifecycleIdentity,
    GenerationSegmentJournal,
    audit_generation_lifecycle,
    build_generation_terminal_binding,
)
from true_rl_agent.agent_core.generation_record import (
    _load_and_validate_formal,
    _load_bound_phase_projection,
    _load_bound_train_projection,
    commit_generation_record,
    load_bound_generation_manifest,
    load_bound_generation_trial_phase_receipt,
    load_bound_generation_trial_train_receipt,
    validate_generation_ledger_record,
)
from true_rl_agent.agent_core.ledger import (
    NON_PROVIDER_AUTHORING_MODE,
    PROVIDER_AUTHORING_MODE,
    append_ledger_record,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.round import RoundInputs, execute_candidate_trial
from true_rl_agent.agent_core.trial_receipts import (
    TrialReceiptRef,
    _train_result_from_private_mapping,
    _train_result_to_private_mapping,
)
from true_rl_agent.backends.base import BackendJobBundle
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.evaluation.acceptance_replay import (
    project_public_history,
    replay_acceptance_ledger,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.prompt_artifacts import RoundAuthoringJournal
from true_rl_agent.runner.base import (
    RunnerExecutionError,
    RunnerFailureKind,
    TrainJobResult,
)
from true_rl_agent.runner.job_attempts import JobAttemptJournal, JobAttemptScope
from true_rl_agent.runner.specs import EvalPhaseJobSpec, GpuProcessBinding
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.generation_attempt_scopes import (
    StairGenerationLaneOperationScopes,
    recover_stair_generation_lane_operation_scopes,
)


def _write_candidate(
    root: Path,
    *,
    candidate_id: str,
    reward_expression: str,
    public_score: float,
) -> CandidatePackageRef:
    package = root / candidate_id
    source_dir = package / "training_code"
    source_dir.mkdir(parents=True)
    (package / "candidate_manifest.yaml").write_text(
        "schema_version: true_rl_agent.candidate.v1\n"
        f"candidate_id: {candidate_id}\n"
        "entrypoint: training_code.components:build_training_components\n",
        encoding="utf-8",
    )
    (source_dir / "components.py").write_text(
        "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
        "def test_reward(inputs):\n"
        f"    return {reward_expression}\n\n"
        "def build_training_components():\n"
        "    return TrainingComponents(\n"
        f"        name={candidate_id!r},\n"
        "        reward_terms=[\n"
        "            RewardTerm(name='test_reward', weight=1.0, fn=test_reward),\n"
        "        ],\n"
        "        rationale='Change one reward mechanism.',\n"
        "    )\n",
        encoding="utf-8",
    )
    (package / "rationale.md").write_text(
        "Change one reward mechanism while preserving all runner contracts.\n",
        encoding="utf-8",
    )
    (package / "operation_diff.json").write_text("{}\n", encoding="utf-8")
    (package / "diagnostics.json").write_text("{}\n", encoding="utf-8")
    return CandidatePackageRef(
        candidate_id=candidate_id,
        path=str(package),
        change_set=CandidateChangeSet(frozenset({"reward_callable"})),
        metadata={"fake_public_score": public_score},
    )


class _OneCandidateProvider:
    def __init__(self, candidate: CandidatePackageRef) -> None:
        self.candidate = candidate
        self.payloads: list[dict[str, Any]] = []

    def next_candidate(
        self,
        *,
        prompt_payload: dict[str, Any],
    ) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        return self.candidate


class _CandidateSequenceProvider:
    def __init__(self, *candidates: CandidatePackageRef) -> None:
        if not candidates:
            raise ValueError("candidate sequence must be non-empty")
        self.candidates = tuple(candidates)
        self.payloads: list[dict[str, Any]] = []

    def next_candidate(self, *, prompt_payload: dict[str, Any]) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        index = len(self.payloads) - 1
        if index >= len(self.candidates):
            raise AssertionError("candidate provider was called too many times")
        return self.candidates[index]


class _CpuPhasedBackend(FakeBackendAdapter):
    def materialize_jobs(self, **kwargs: Any) -> BackendJobBundle:
        gpu_process_binding = kwargs.pop("gpu_process_binding", None)
        bundle = super().materialize_jobs(**kwargs)
        execution_device = (
            gpu_process_binding.process_execution_device
            if gpu_process_binding is not None
            else "cpu"
        )
        return BackendJobBundle(
            train_spec=replace(
                bundle.train_spec,
                execution_device=execution_device,
                gpu_process_binding=gpu_process_binding,
            ),
            eval_spec=replace(
                bundle.eval_spec,
                execution_device=execution_device,
                gpu_process_binding=gpu_process_binding,
            ),
            task_invariants=bundle.task_invariants,
        )

    def materialize_eval_phase(
        self,
        *,
        eval_plan,
        phase: EvaluationPhase,
        attempt_scope=None,
    ) -> EvalPhaseJobSpec:
        return EvalPhaseJobSpec(
            job_id=f"{eval_plan.job_id}-{phase.value.replace('_', '-')}",
            evaluation_plan=eval_plan,
            phase=phase,
            metadata={"phase": phase.value},
            attempt_scope=attempt_scope,
        )


@dataclass(frozen=True)
class _Screen:
    trial_id: str
    phase_result_hash: str
    evaluation_plan_hash: str
    checkpoint_hash: str
    execution_device: str
    resource_binding_hash: str | None
    score: float

    @property
    def evidence_hash(self) -> str:
        return canonical_json_hash(self.to_selector_dict())

    def to_selector_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "phase_result_hash": self.phase_result_hash,
            "evaluation_plan_hash": self.evaluation_plan_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "execution_device": self.execution_device,
            "resource_binding_hash": self.resource_binding_hash,
            "score": self.score,
            "acceptance_authority": False,
        }


@dataclass(frozen=True)
class _Selection:
    selected_trial_id: str | None
    screens: tuple[_Screen, _Screen]
    guard: _Screen

    @property
    def reason(self) -> str:
        return (
            "public_dev_objective_rank"
            if self.selected_trial_id is not None
            else "no_public_dev_selection"
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "test.selection.v1",
            "selected_trial_id": self.selected_trial_id,
            "screen_evidence_hashes": [
                canonical_json_hash(item.to_selector_dict()) for item in self.screens
            ],
            "guard_reference_evidence_hash": canonical_json_hash(
                self.guard.to_selector_dict()
            ),
            "reason": self.reason,
            "acceptance_authority": False,
        }


@dataclass(frozen=True)
class _Assembly:
    selected_trial_id: str
    phase_receipt_refs: Mapping[EvaluationPhase, Mapping[str, Any]]


class _PhasedRunner(FakeRunner):
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self._lock = threading.Lock()
        self._train_barrier = threading.Barrier(2, timeout=5.0)
        self._phase_barriers = {
            EvaluationPhase.PUBLIC_DEV: threading.Barrier(2, timeout=5.0),
            EvaluationPhase.PUBLIC_ACCEPT: threading.Barrier(1, timeout=5.0),
        }
        self.phase_calls: list[tuple[str, EvaluationPhase]] = []
        self.phase_scopes: list[tuple[EvaluationPhase, JobAttemptScope | None]] = []

    def _event(self, value: str) -> None:
        with self._lock:
            self.events.append(value)

    def run_train(self, train_spec):
        self._event(f"train-start:{train_spec.subject.subject_id}")
        self._train_barrier.wait()
        result = super().run_train(train_spec)
        if train_spec.gpu_process_binding is not None:
            checkpoint = replace(
                result.checkpoint,
                metadata={
                    **dict(result.checkpoint.metadata),
                    "gpu_process_binding": (train_spec.gpu_process_binding.to_dict()),
                },
            )
            result = replace(
                result,
                checkpoint=checkpoint,
                gpu_process_binding=train_spec.gpu_process_binding,
            )
        self._event(f"train-end:{train_spec.subject.subject_id}")
        return result

    def run_eval_phase(self, eval_spec: EvalPhaseJobSpec) -> EvaluationPhaseResult:
        subject_id = eval_spec.subject.subject_id
        self._event(f"phase-start:{eval_spec.phase.value}:{subject_id}")
        if eval_spec.phase in self._phase_barriers:
            self._phase_barriers[eval_spec.phase].wait()
        if eval_spec.phase is EvaluationPhase.PUBLIC_ACCEPT:
            self.assert_selection_precedes_public_accept()
        self.phase_calls.append((subject_id, eval_spec.phase))
        self.phase_scopes.append((eval_spec.phase, eval_spec.attempt_scope))
        plan = eval_spec.evaluation_plan
        score = float(plan.metadata["fake_public_score"])
        result = EvaluationPhaseResult(
            job_id=eval_spec.job_id,
            task_id=eval_spec.task_id,
            phase=eval_spec.phase,
            evaluation_plan_hash=canonical_json_hash(plan.to_public_dict()),
            subject=eval_spec.subject,
            subject_manifest_hash=eval_spec.subject_manifest_hash,
            program_source_checksum=eval_spec.program_source_checksum,
            training_binding_hash=eval_spec.training_binding_hash,
            capability_profile_hash=eval_spec.capability_profile_hash,
            checkpoint_hash=eval_spec.execution_checkpoint().checksum,
            eval_protocol_hash=eval_spec.evaluation_binding.eval_protocol_hash,
            evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
            execution_device=eval_spec.execution_device,
            resource_binding_hash=(
                eval_spec.attempt_scope.resource_binding_hash
                if eval_spec.attempt_scope is not None
                else None
            ),
            task_evidence={"score": score, "phase": eval_spec.phase.value},
            execution_cost={"seed_units": 1},
            environment_source_preflight={"status": "pass"},
            backend_execution_environment_verification={"status": "pass"},
            checkpoint_safety_report={"status": "pass"},
            gpu_process_binding=(
                eval_spec.gpu_process_binding.to_dict()
                if eval_spec.gpu_process_binding is not None
                else None
            ),
        )
        self._event(f"phase-end:{eval_spec.phase.value}:{subject_id}")
        return result

    def assert_selection_precedes_public_accept(self) -> None:
        with self._lock:
            if "selection-sealed" not in self.events:
                raise AssertionError(
                    "public_accept started before selection was sealed"
                )
    def project_assembled_eval_summary(self, eval_spec, assembly):
        if not isinstance(assembly, _Assembly):
            raise TypeError("test phased runner requires a bound assembly")
        if set(assembly.phase_receipt_refs) != set(EvaluationPhase):
            raise ValueError("test assembly is missing phase receipts")
        return super().run_eval(eval_spec)


class _PublicDevFailureRunner(_PhasedRunner):
    def __init__(
        self,
        events: list[str],
        *,
        failing_subject_id: str,
        failure_kind: RunnerFailureKind,
    ) -> None:
        super().__init__(events)
        self.failing_subject_id = failing_subject_id
        self.failure_kind = failure_kind

    def run_eval_phase(self, eval_spec: EvalPhaseJobSpec) -> EvaluationPhaseResult:
        result = super().run_eval_phase(eval_spec)
        if (
            eval_spec.phase is EvaluationPhase.PUBLIC_DEV
            and eval_spec.subject.subject_id == self.failing_subject_id
        ):
            raise RunnerExecutionError(
                "injected public_dev runner failure",
                failure_kind=self.failure_kind,
            )
        return result


class _TrainFailureRunner(_PhasedRunner):
    def __init__(
        self,
        events: list[str],
        *,
        failing_subject_id: str,
        failure_kind: RunnerFailureKind,
    ) -> None:
        super().__init__(events)
        self._phase_barriers[EvaluationPhase.PUBLIC_DEV] = threading.Barrier(
            1,
            timeout=5.0,
        )
        self.failing_subject_id = failing_subject_id
        self.failure_kind = failure_kind

    def run_train(self, train_spec):
        result = super().run_train(train_spec)
        if train_spec.subject.subject_id == self.failing_subject_id:
            raise RunnerExecutionError(
                "injected train runner failure",
                failure_kind=self.failure_kind,
            )
        return result


class _CrashAfterAttemptResult(RuntimeError):
    pass


class _AttemptJournalPhasedRunner(_PhasedRunner):
    """Use the production attempt wrapper around deterministic fake jobs."""

    def __init__(
        self,
        events: list[str],
        *,
        runner_output_root: Path,
        crash_after_public_dev_subject: str | None = None,
    ) -> None:
        super().__init__(events)
        self._attempt_executor = SubprocessRunner(("test-only",), timeout_s=10)
        self._runner_output_root = runner_output_root
        self._runner_output_root.mkdir(parents=True, exist_ok=True)
        self._crash_after_public_dev_subject = crash_after_public_dev_subject
        self.crashed_job_id: str | None = None
        self.actual_job_calls: dict[tuple[str, str], int] = {}
        self.recovered_job_calls: dict[tuple[str, str], int] = {}

    def run_train(self, train_spec):
        def execute(_log_dir, execution_context, _timeout_s):
            if execution_context is not None:
                raise AssertionError("CPU test job acquired a GPU execution context")
            result = _PhasedRunner.run_train(self, train_spec)
            self._record_actual("train", train_spec.job_id)
            self._publish_runner_output(
                kind="train",
                job_id=train_spec.job_id,
                payload=_train_result_to_private_mapping(result),
            )
            return result

        def recover(execution_context):
            if execution_context is not None:
                raise AssertionError(
                    "CPU test recovery acquired a GPU execution context"
                )
            self._record_recovery("train", train_spec.job_id)
            return _train_result_from_private_mapping(
                self._load_runner_output(kind="train", job_id=train_spec.job_id)
            )

        return self._attempt_executor._execute_with_attempts(
            train_spec,
            job_kind="train",
            operation=execute,
            recover_completed=recover,
        )

    def run_eval_phase(self, eval_spec: EvalPhaseJobSpec) -> EvaluationPhaseResult:
        def execute(_log_dir, execution_context, _timeout_s):
            if execution_context is not None:
                raise AssertionError("CPU test job acquired a GPU execution context")
            result = _PhasedRunner.run_eval_phase(self, eval_spec)
            self._record_actual("eval_phase", eval_spec.job_id)
            self._publish_runner_output(
                kind="eval_phase",
                job_id=eval_spec.job_id,
                payload=result.to_private_dict(),
            )
            return result

        def recover(execution_context):
            if execution_context is not None:
                raise AssertionError(
                    "CPU test recovery acquired a GPU execution context"
                )
            self._record_recovery("eval_phase", eval_spec.job_id)
            return EvaluationPhaseResult.from_private_mapping(
                self._load_runner_output(
                    kind="eval_phase",
                    job_id=eval_spec.job_id,
                )
            )

        result = self._attempt_executor._execute_with_attempts(
            eval_spec,
            job_kind="eval_phase",
            operation=execute,
            recover_completed=recover,
        )
        if (
            self._crash_after_public_dev_subject is not None
            and eval_spec.phase is EvaluationPhase.PUBLIC_DEV
            and eval_spec.subject.subject_id == self._crash_after_public_dev_subject
        ):
            self._crash_after_public_dev_subject = None
            self.crashed_job_id = eval_spec.job_id
            raise _CrashAfterAttemptResult(
                "fault injection after attempt result and before phase receipt"
            )
        return result

    def _runner_output_path(self, *, kind: str, job_id: str) -> Path:
        key = canonical_json_hash({"kind": kind, "job_id": job_id})
        return self._runner_output_root / f"{key}.json"

    def _publish_runner_output(
        self,
        *,
        kind: str,
        job_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        create_once_bytes(
            self._runner_output_path(kind=kind, job_id=job_id),
            canonical_json_bytes(payload, trailing_newline=True),
        )

    def _load_runner_output(self, *, kind: str, job_id: str) -> dict[str, Any]:
        raw = read_bounded_regular_file(
            self._runner_output_path(kind=kind, job_id=job_id),
            max_bytes=8 * 1024 * 1024,
            label="test runner output",
        )
        payload = strict_json_object_loads(raw, context="test runner output")
        if raw != canonical_json_bytes(payload, trailing_newline=True):
            raise ValueError("test runner output is not canonical JSON")
        return payload

    def _record_actual(self, kind: str, job_id: str) -> None:
        key = (kind, job_id)
        self.actual_job_calls[key] = self.actual_job_calls.get(key, 0) + 1

    def _record_recovery(self, kind: str, job_id: str) -> None:
        key = (kind, job_id)
        self.recovered_job_calls[key] = self.recovered_job_calls.get(key, 0) + 1


class _PhasedAdapter:
    def __init__(
        self,
        events: list[str],
        *,
        force_no_selection: bool = False,
    ) -> None:
        self.events = events
        self.force_no_selection = force_no_selection
        self.screen_inputs: list[dict[str, Any]] = []

    def selection_contract(self) -> Mapping[str, Any]:
        return {"schema_version": "test.selection_contract.v1", "width": 2}

    def cost_contract(self) -> Mapping[str, Any]:
        return {"schema_version": "test.cost_contract.v1", "max_seed_units": 23}

    def project_public_dev_screen(
        self,
        *,
        trial_id: str,
        phase_result: EvaluationPhaseResult,
        screen_role: str,
        train_result: TrainJobResult,
        train_receipt_checksum: str,
        mechanism_operation_delta: Mapping[str, Any] | None,
    ) -> _Screen:
        self.screen_inputs.append(
            {
                "trial_id": trial_id,
                "screen_role": screen_role,
                "train_result": train_result,
                "train_receipt_checksum": train_receipt_checksum,
                "mechanism_operation_delta": mechanism_operation_delta,
            }
        )
        return _Screen(
            trial_id=trial_id,
            phase_result_hash=canonical_json_hash(phase_result.to_private_dict()),
            evaluation_plan_hash=phase_result.evaluation_plan_hash,
            checkpoint_hash=phase_result.checkpoint_hash,
            execution_device=phase_result.execution_device,
            resource_binding_hash=phase_result.resource_binding_hash,
            score=float(phase_result.task_evidence["score"]),
        )

    def project_public_phase_summary(
        self,
        *,
        phase_result: EvaluationPhaseResult,
    ) -> Mapping[str, Any]:
        if phase_result.phase is EvaluationPhase.SEALED_VALIDATION:
            raise ValueError("sealed evidence is private")
        return {"score": float(phase_result.task_evidence["score"])}

    def select(
        self,
        screens: Sequence[_Screen],
        *,
        guard_reference: _Screen,
    ) -> _Selection:
        ordered = tuple(screens)
        winner = (
            None
            if self.force_no_selection
            else max(ordered, key=lambda item: item.score)
        )
        self.events.append("selection-sealed")
        return _Selection(
            selected_trial_id=(winner.trial_id if winner is not None else None),
            screens=(ordered[0], ordered[1]),
            guard=guard_reference,
        )

    def assemble_full_summary(self, **kwargs: Any) -> _Assembly:
        return _Assembly(
            selected_trial_id=kwargs["trial_id"],
            phase_receipt_refs=kwargs["phase_receipt_refs"],
        )


@dataclass(frozen=True)
class _GuardReference:
    evidence: _Screen
    phase_receipt_ref: TrialReceiptRef
    parent_hash: str

    def assert_matches_parent(self, parent) -> None:
        if canonical_json_hash(parent.to_public_dict()) != self.parent_hash:
            raise ValueError("guard reference parent mismatch")


class FixedWidthGenerationTests(unittest.TestCase):
    def test_generation_anchor_loader_uses_first_prompt_after_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = RoundAuthoringJournal(
                Path(tmp),
                round_id="generation-anchor-repair",
            )
            journal.bind_budget(
                max_provider_attempts=2,
                max_source_requests_per_round=0,
                max_source_ids_per_request=2,
                max_source_request_bytes_per_round=48 * 1024,
            )
            first = journal.begin_attempt(
                {"candidate_lineage": {}, "anchor_marker": "frozen-first"}
            )
            journal.complete_attempt(
                first,
                {
                    "round_id": "generation-anchor-repair",
                    "attempt": 1,
                    "attempt_id": first.attempt_id,
                    "provider_status": "fail",
                },
                continuation_state={"schema_version": "test.continuation.v1"},
            )
            selected = journal.begin_attempt(
                {"candidate_lineage": {}, "anchor_marker": "compact-repair"}
            )
            journal.complete_attempt(
                selected,
                {
                    "round_id": "generation-anchor-repair",
                    "attempt": 2,
                    "attempt_id": selected.attempt_id,
                    "provider_status": "pass",
                },
                continuation_state={"schema_version": "test.continuation.v1"},
            )
            evidence_ref = journal.publish_evidence(
                selected_attempt_id=selected.attempt_id,
                stop_reason="candidate_verified",
            )
            result = CandidateAuthoringResult(
                stop_reason="candidate_verified",
                prompt_payload_ref=selected.prompt_payload_ref,
                round_public_context={},
                attempt_records=[],
                attempt_prompt_payload_refs=(
                    first.prompt_payload_ref,
                    selected.prompt_payload_ref,
                ),
                authoring_evidence_ref=evidence_ref,
                parent_program_source_checksum=None,
            )

            anchor_prompt = _verified_authoring_anchor_prompt(
                result,
                expected_round_id="generation-anchor-repair",
            )

        self.assertEqual(anchor_prompt["anchor_marker"], "frozen-first")

    def test_full_generation_is_serial_author_parallel_execute_and_single_commit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = FakeTaskPlugin()
            backend = _CpuPhasedBackend()
            binding = fake_evaluation_binding(task)
            parent_candidate = _write_candidate(
                root / "parent-provider",
                candidate_id="parent-mechanism",
                reward_expression="inputs.base_height",
                public_score=0.30,
            )
            initial = fake_bootstrap_current_best(
                initial_best_score=0.10,
                task_plugin=task,
            )
            parent_execution = execute_candidate_trial(
                RoundInputs(
                    round_id="parent-round",
                    task_plugin=task,
                    backend_adapter=FakeBackendAdapter(),
                    runner=FakeRunner(),
                    candidate_admission_policy=FakeVerifier(),
                    candidate=parent_candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=root / "parent-ledger.jsonl",
                    trial_receipt_root=root / "parent-trial",
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    public_context={"previous_rounds": 0, "public_only": True},
                    current_best_eval_stats=initial,
                    session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                    ledger_epoch_hash=primary_ledger_epoch_hash(
                        FAKE_SESSION_GENESIS_HASH
                    ),
                    evaluation_binding=binding,
                )
            )
            candidate_parent = replay_acceptance_ledger(
                [parent_execution.record],
                initial_current_best=initial,
                expected_session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                expected_ledger_epoch_hash=primary_ledger_epoch_hash(
                    FAKE_SESSION_GENESIS_HASH
                ),
            ).current_best
            parent_mapping = candidate_parent.to_public_dict()
            parent_mapping.update(
                {
                    "mode": "reference",
                    "anchor_kind": "reference",
                    "anchor_id": "reference-parent",
                    "accepted_round_id": None,
                    "accepted_candidate_id": None,
                }
            )
            sealed = dict(parent_mapping["sealed_validation_evidence"])
            sealed["evaluation_subject"] = {
                "schema_version": "true_rl_agent.experiment.subject.v2",
                "kind": "reference",
                "id": "reference-parent",
            }
            parent_mapping["sealed_validation_evidence"] = sealed
            parent = CurrentBestEvalStats.from_mapping(parent_mapping)
            self.assertEqual(parent.anchor_id, "reference-parent")

            source_bytes = (
                Path(parent_candidate.path) / "training_code" / "components.py"
            ).read_bytes()
            source = source_bytes.decode("utf-8")
            self.assertEqual(
                "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
                parent.program_source_checksum,
            )
            context_builder = PublicContextBuilder(
                task_prompt_adapter=task.prompt_adapter(),
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                initial_current_best=parent,
                session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                candidate_capability_profile=task.task_spec.candidate_capability_profile,
                initial_reference_program_source=source,
                ledger_authoring_mode=PROVIDER_AUTHORING_MODE,
                expected_ledger_epoch_hash=primary_ledger_epoch_hash(
                    FAKE_SESSION_GENESIS_HASH
                ),
            )
            base_context = context_builder.build([])
            candidate_a = _write_candidate(
                root / "generation-provider",
                candidate_id="candidate-generation-a",
                reward_expression="inputs.base_height + inputs.roll",
                public_score=0.50,
            )
            candidate_b = _write_candidate(
                root / "generation-provider",
                candidate_id="candidate-generation-b",
                reward_expression="inputs.base_height + inputs.pitch",
                public_score=0.60,
            )
            lane_intents = {
                lane_id: {
                    "schema_version": "test.generation_lane_intent.v1",
                    "lane_id": lane_id,
                    "intent": "novel",
                }
                for lane_id in ("lane-a", "lane-b")
            }
            lane_assignments = (
                GenerationLaneAssignment(
                    "lane-a",
                    "cpu",
                    None,
                    canonical_json_hash(lane_intents["lane-a"]),
                ),
                GenerationLaneAssignment(
                    "lane-b",
                    "cpu",
                    None,
                    canonical_json_hash(lane_intents["lane-b"]),
                ),
            )
            guard_phase_ref = TrialReceiptRef(
                kind="eval_phase",
                path=str(root / "guard" / "public_dev.json"),
                checksum="sha256:" + "9" * 64,
                size_bytes=1,
            )
            guard_evidence = _Screen(
                trial_id="reference-parent",
                phase_result_hash="6" * 64,
                evaluation_plan_hash="7" * 64,
                checkpoint_hash=parent.checkpoint_hash,
                execution_device="cpu",
                resource_binding_hash=None,
                score=0.30,
            )
            guard = _GuardReference(
                evidence=guard_evidence,
                phase_receipt_ref=guard_phase_ref,
                parent_hash=canonical_json_hash(parent.to_public_dict()),
            )
            training_binding = fake_training_binding(PolicyInitialization.random())

            def _build_inputs(
                *,
                case_id: str,
                provider_a: _OneCandidateProvider,
                provider_b: _OneCandidateProvider,
                events: list[str],
                runner: _PhasedRunner,
                adapter: _PhasedAdapter,
                authoring_max_attempts: Mapping[str, int] | None = None,
                max_generations: int = 15,
                continuation_policy_hash: str = "2" * 64,
                gpu_bindings: tuple[GpuProcessBinding, ...] | None = None,
                invocation_segment_id: str | None = None,
                stair_job_identity: bool = False,
            ) -> FixedWidthGenerationInputs:
                case_lane_assignments = (
                    lane_assignments
                    if gpu_bindings is None
                    else tuple(
                        replace(
                            assignment,
                            execution_device=binding.process_execution_device,
                            resource_binding_hash=binding.resource_binding_hash,
                            gpu_process_binding=binding,
                        )
                        for assignment, binding in zip(
                            lane_assignments,
                            gpu_bindings,
                            strict=True,
                        )
                    )
                )
                session_cost_hash = canonical_json_hash(
                    {"case_id": case_id, "kind": "test-session-cost"}
                )
                bound_contract_hashes = {
                    "authoring_policy": "1" * 64,
                    "continuation_policy": continuation_policy_hash,
                    "evaluation_cost": canonical_json_hash(
                        dict(adapter.cost_contract())
                    ),
                    "evaluation_protocol": binding.eval_protocol_hash,
                    "exploration_projector": "e" * 64,
                    "final_confirmation_plan": "3" * 64,
                    "provider_policy": "4" * 64,
                    "resource_plan": generation_resource_plan_hash(
                        case_lane_assignments
                    ),
                    "retry_policy": "5" * 64,
                    "selector": canonical_json_hash(dict(adapter.selection_contract())),
                }
                contract_kwargs: dict[str, Any] = {}
                if gpu_bindings is not None:
                    bound_contract_hashes["session_cost"] = session_cost_hash
                    contract_kwargs = {
                        "schema_version": (
                            GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION
                        ),
                        "session_mode": "development_generation_pilot",
                    }
                contract = GenerationSessionContract(
                    session_id="generation-session",
                    task_id=task.task_spec.task_id,
                    backend_id=backend.backend_id,
                    max_generations=max_generations,
                    max_trials=max_generations * 2,
                    max_train_jobs=max_generations * 2,
                    bound_contract_hashes=bound_contract_hashes,
                    **contract_kwargs,
                )
                assignment_by_lane = {
                    assignment.lane_id: assignment
                    for assignment in case_lane_assignments
                }

                def _trial_id(lane_id: str) -> str:
                    prefix = f"generation-{case_id}" if stair_job_identity else case_id
                    return f"{prefix}-{lane_id}"

                def _author(
                    *,
                    lane_id: str,
                    provider: _OneCandidateProvider,
                    exclusion,
                ) -> CandidateAuthoringResult:
                    events.append(f"author-start:{lane_id}")
                    result = CandidateAuthoringLoop(
                        context_builder=context_builder,
                        provider=provider,
                        candidate_admission_policy=FakeVerifier(),
                        round_dir=root / "authoring" / case_id / lane_id,
                        max_attempts=(authoring_max_attempts or {}).get(
                            lane_id,
                            1,
                        ),
                        sibling_semantic_exclusion=exclusion,
                        generation_lane_intent=lane_intents[lane_id],
                    ).run_round(round_id=_trial_id(lane_id), records=[])
                    events.append(f"author-end:{lane_id}")
                    return result

                def _round_inputs(
                    lane_id: str,
                    authored: CandidateAuthoringResult,
                ) -> RoundInputs:
                    return RoundInputs(
                        round_id=_trial_id(lane_id),
                        task_plugin=task,
                        backend_adapter=backend,
                        runner=runner,
                        candidate_admission_policy=FakeVerifier(),
                        candidate=authored.candidate,
                        policy_initialization=PolicyInitialization.random(),
                        ledger_path=(
                            root / "unused-round-ledgers" / case_id / f"{lane_id}.jsonl"
                        ),
                        trial_receipt_root=(
                            root / "trial-receipts" / case_id / lane_id
                        ),
                        authoring_mode=PROVIDER_AUTHORING_MODE,
                        public_context=authored.round_public_context,
                        current_best_eval_stats=parent,
                        session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                        ledger_epoch_hash=primary_ledger_epoch_hash(
                            FAKE_SESSION_GENESIS_HASH
                        ),
                        evaluation_binding=binding,
                        parent_program_source_checksum=(parent.program_source_checksum),
                        prompt_payload_ref=authored.prompt_payload_ref,
                        authoring_evidence_ref=authored.authoring_evidence_ref,
                        train_attempt_scope=lane_scopes[lane_id],
                        eval_attempt_scope=lane_scopes[lane_id],
                        gpu_process_binding=(
                            assignment_by_lane[lane_id].gpu_process_binding
                        ),
                    )

                manifest = GenerationManifest(
                    schema_version=(
                        GENERATION_MANIFEST_V2_SCHEMA_VERSION
                        if gpu_bindings is not None
                        else GENERATION_MANIFEST_SCHEMA_VERSION
                    ),
                    generation_id=f"generation-{case_id}",
                    generation_index=1,
                    session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                    generation_session_contract_hash=contract.contract_hash,
                    ledger_epoch_hash=primary_ledger_epoch_hash(
                        FAKE_SESSION_GENESIS_HASH
                    ),
                    parent_program_source_checksum=(parent.program_source_checksum),
                    parent_checkpoint_hash=parent.checkpoint_hash,
                    current_best_evidence_hash=canonical_json_hash(
                        parent.to_public_dict()
                    ),
                    base_public_context_hash=canonical_json_hash(base_context),
                    evaluation_binding_hash=binding.binding_hash,
                    training_initialization_hash=(
                        generation_training_initialization_hash(
                            policy_initialization_kind="random",
                            training_binding_hash=training_binding.binding_hash,
                            expected_initial_policy_state_hash=None,
                        )
                    ),
                    guard_reference_evidence_hash=guard_evidence.evidence_hash,
                    guard_reference_phase_receipt_hash=canonical_json_hash(
                        guard_phase_ref.to_dict()
                    ),
                    lanes=case_lane_assignments,
                )
                lane_scopes = {
                    lane_id: JobAttemptScope(
                        journal_root=root / "job-attempts" / case_id,
                        session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                        session_contract_hash=contract.contract_hash,
                        generation_id=manifest.generation_id,
                        lane_id=lane_id,
                        invocation_segment_id=(
                            invocation_segment_id or f"{case_id}-segment"
                        ),
                        budget_contract_hash=(
                            canonical_json_hash(
                                {"case_id": case_id, "kind": "test-budget"}
                            )
                            if gpu_bindings is None
                            else session_cost_hash
                        ),
                        attempt_budget_units=1,
                        max_infrastructure_retries=0,
                        resource_binding_hash=(
                            assignment_by_lane[lane_id].resource_binding_hash
                        ),
                    )
                    for lane_id in ("lane-a", "lane-b")
                }
                sealed_scopes = {
                    lane_id: replace(
                        scope,
                        attempt_budget_units=2,
                        max_infrastructure_retries=1,
                    )
                    for lane_id, scope in lane_scopes.items()
                }
                lanes = tuple(
                    GenerationLaneRuntime(
                        lane_id=lane_id,
                        trial_id=_trial_id(lane_id),
                        lane_intent=lane_intents[lane_id],
                        lane_intent_hash=canonical_json_hash(lane_intents[lane_id]),
                        author=(
                            lambda exclusion, lane_id=lane_id, provider=provider: (
                                _author(
                                    lane_id=lane_id,
                                    provider=provider,
                                    exclusion=exclusion,
                                )
                            )
                        ),
                        build_round_inputs=(
                            lambda authored, lane_id=lane_id: _round_inputs(
                                lane_id, authored
                            )
                        ),
                        attempt_scope=lane_scopes[lane_id],
                        sealed_attempt_scope=sealed_scopes[lane_id],
                    )
                    for lane_id, provider in (
                        ("lane-a", provider_a),
                        ("lane-b", provider_b),
                    )
                )
                return FixedWidthGenerationInputs(
                    manifest=manifest,
                    session_contract=contract,
                    lanes=lanes,
                    guard_reference=guard,
                    phased_adapter=adapter,
                    receipt_root=root / "generation-receipts" / case_id,
                    ledger_path=root / "generation-ledgers" / f"{case_id}.jsonl",
                    parent_current_best=parent,
                    initial_current_best=parent,
                    acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                )

            failed_candidate_a = _write_candidate(
                root / "generation-provider",
                candidate_id="candidate-generation-a-static-failure",
                reward_expression=(
                    "inputs.base_height + inputs.roll + inputs.pitch * 0.25"
                ),
                public_score=0.40,
            )
            failed_candidate_a = replace(
                failed_candidate_a,
                metadata={
                    **failed_candidate_a.metadata,
                    "verifier_status": "fail",
                },
            )
            failed_candidate_b = _write_candidate(
                root / "generation-provider",
                candidate_id="candidate-generation-b-static-failure",
                reward_expression=(
                    "inputs.base_height + inputs.pitch + inputs.roll * 0.25"
                ),
                public_score=0.40,
            )
            failed_candidate_b = replace(
                failed_candidate_b,
                metadata={
                    **failed_candidate_b.metadata,
                    "verifier_status": "fail",
                },
            )
            provider_a = _OneCandidateProvider(candidate_a)
            provider_b = _CandidateSequenceProvider(
                failed_candidate_b,
                candidate_b,
            )
            events: list[str] = []
            runner = _PhasedRunner(events)
            adapter = _PhasedAdapter(events)
            generation_inputs = _build_inputs(
                case_id="accepted",
                provider_a=provider_a,
                provider_b=provider_b,
                events=events,
                runner=runner,
                adapter=adapter,
                authoring_max_attempts={"lane-b": 2},
            )

            result = execute_fixed_width_generation(generation_inputs)
            counts_before_reuse = (len(events), len(runner.phase_calls))
            reused = execute_fixed_width_generation(generation_inputs)

            ledger = load_ledger(generation_inputs.ledger_path)
            with self.subTest(scenario="accepted_generation_and_exact_reuse"):
                self.assertTrue(result.committed)
                self.assertFalse(reused.committed)
                self.assertTrue(reused.reused_execution)
                self.assertEqual(
                    (len(events), len(runner.phase_calls)), counts_before_reuse
                )
                self.assertEqual(len(ledger), 1)
                self.assertEqual(ledger[0]["record_kind"], "generation")
                self.assertEqual(ledger[0]["generation_outcome"], "accepted")
                self.assertEqual(
                    [item["trial_status"] for item in ledger[0]["trials"]],
                    ["qualified_not_selected", "accepted"],
                )
                self.assertEqual(len(adapter.screen_inputs), 2)
                trials_by_id = {
                    item["trial_id"]: item for item in ledger[0]["trials"]
                }
                for screen_input in adapter.screen_inputs:
                    trial = trials_by_id[screen_input["trial_id"]]
                    self.assertEqual(screen_input["screen_role"], "candidate")
                    self.assertEqual(
                        screen_input["train_receipt_checksum"],
                        trial["train"]["train_receipt_checksum"],
                    )
                    self.assertEqual(
                        canonical_json_hash(
                            screen_input["train_result"].to_public_dict()
                        ),
                        trial["train"]["train_result_hash"],
                    )
                    self.assertEqual(
                        screen_input["mechanism_operation_delta"],
                        trial["mechanism_operation_delta"],
                    )
                self.assertEqual(
                    result.resulting_current_best.anchor_id,
                    "candidate-generation-b",
                )
                self.assertLess(
                    events.index("author-end:lane-a"),
                    events.index("author-start:lane-b"),
                )
                self.assertEqual(
                    len(
                        [
                            item
                            for item in runner.phase_calls
                            if item[1] is EvaluationPhase.PUBLIC_DEV
                        ]
                    ),
                    2,
                )
                self.assertEqual(
                    len(
                        [
                            item
                            for item in runner.phase_calls
                            if item[1] is EvaluationPhase.PUBLIC_ACCEPT
                        ]
                    ),
                    1,
                )
                self.assertEqual(
                    len(
                        [
                            item
                            for item in runner.phase_calls
                            if item[1] is EvaluationPhase.SEALED_VALIDATION
                        ]
                    ),
                    1,
                )
                self.assertEqual(
                    [
                        item[0]
                        for item in runner.phase_calls
                        if item[1] is EvaluationPhase.SEALED_VALIDATION
                    ],
                    ["candidate-generation-b"],
                )
                self.assertTrue(
                    all(
                        scope is not None and scope.max_infrastructure_retries == 0
                        for phase, scope in runner.phase_scopes
                        if phase is not EvaluationPhase.SEALED_VALIDATION
                    )
                )
                self.assertEqual(
                    [
                        scope.max_infrastructure_retries
                        for phase, scope in runner.phase_scopes
                        if phase is EvaluationPhase.SEALED_VALIDATION
                        and scope is not None
                    ],
                    [1],
                )
                next_guard_phase = load_bound_generation_trial_phase_receipt(
                    ledger[0],
                    trial_id="accepted-lane-b",
                    phase=EvaluationPhase.PUBLIC_DEV,
                )
                self.assertIs(
                    next_guard_phase.result.phase,
                    EvaluationPhase.PUBLIC_DEV,
                )
                self.assertEqual(
                    canonical_json_hash(next_guard_phase.result.to_private_dict()),
                    ledger[0]["trials"][1]["public_dev"]["phase_result_hash"],
                )
                selected_train = load_bound_generation_trial_train_receipt(
                    ledger[0],
                    trial_id="accepted-lane-b",
                )
                self.assertEqual(
                    selected_train.result.checkpoint.checksum,
                    ledger[0]["trials"][1]["train"]["checkpoint_hash"],
                )
                self.assertTrue(provider_b.payloads)
                provider_b_text = repr(provider_b.payloads[0])
                self.assertNotIn("candidate-generation-a", provider_b_text)
                self.assertNotIn("inputs.base_height + inputs.roll", provider_b_text)
                self.assertEqual(len(provider_b.payloads), 2)
                self.assertIn(
                    "current_best/training_code/components.py",
                    {
                        item["logical_id"]
                        for item in provider_b.payloads[0]["source_payloads"]
                    },
                )
                self.assertEqual(
                    provider_b.payloads[1]["authoring_context_frame"]["frame_kind"],
                    "static_repair",
                )
                self.assertEqual(
                    {
                        item["logical_id"]
                        for item in provider_b.payloads[1]["source_payloads"]
                    },
                    {"attempt/latest_failed/training_code/components.py"},
                )
                self.assertTrue(
                    all(
                        trial["mechanism_operation_delta"] is not None
                        for trial in ledger[0]["trials"]
                    )
                )

            with self.subTest(
                scenario="lane_a_static_repair_still_builds_sibling_exclusion"
            ):
                repaired_a_provider = _CandidateSequenceProvider(
                    failed_candidate_a,
                    candidate_a,
                )
                repaired_a_b_provider = _OneCandidateProvider(candidate_b)
                repaired_a_events: list[str] = []
                repaired_a_inputs = _build_inputs(
                    case_id="lane-a-static-repair",
                    provider_a=repaired_a_provider,
                    provider_b=repaired_a_b_provider,
                    events=repaired_a_events,
                    runner=_PhasedRunner(repaired_a_events),
                    adapter=_PhasedAdapter(repaired_a_events),
                    authoring_max_attempts={"lane-a": 2},
                )
                repaired_a_result = execute_fixed_width_generation(
                    repaired_a_inputs
                )
                repaired_a_ledger = load_ledger(repaired_a_inputs.ledger_path)

                self.assertTrue(repaired_a_result.committed)
                self.assertEqual(
                    repaired_a_ledger[0]["generation_outcome"],
                    "accepted",
                )
                self.assertEqual(len(repaired_a_provider.payloads), 2)
                self.assertEqual(len(repaired_a_b_provider.payloads), 1)
                self.assertEqual(
                    repaired_a_provider.payloads[1]["authoring_context_frame"][
                        "frame_kind"
                    ],
                    "static_repair",
                )
                self.assertEqual(
                    {
                        item["logical_id"]
                        for item in repaired_a_provider.payloads[1][
                            "source_payloads"
                        ]
                    },
                    {"attempt/latest_failed/training_code/components.py"},
                )
                self.assertTrue(
                    all(
                        trial["mechanism_operation_delta"] is not None
                        for trial in repaired_a_ledger[0]["trials"]
                    )
                )

            with self.subTest(
                scenario=(
                    "completed_attempt_is_reloaded_after_crash_before_trial_receipt"
                )
            ):
                case_id = "attempt-result-gap"
                runner_outputs = root / "runner-outputs" / case_id
                continuation_policy = GenerationContinuationPolicy(
                    max_segments=3,
                    allowed_continuation_reasons=(
                        "operator_pause",
                        "typed_infrastructure_failure",
                        "exception",
                    ),
                )
                first_events: list[str] = []
                first_runner = _AttemptJournalPhasedRunner(
                    first_events,
                    runner_output_root=runner_outputs,
                    crash_after_public_dev_subject="candidate-generation-a",
                )
                first_provider_a = _OneCandidateProvider(candidate_a)
                first_provider_b = _OneCandidateProvider(candidate_b)
                provisional_inputs = _build_inputs(
                    case_id=case_id,
                    provider_a=first_provider_a,
                    provider_b=first_provider_b,
                    events=first_events,
                    runner=first_runner,
                    adapter=_PhasedAdapter(first_events),
                    continuation_policy_hash=continuation_policy.policy_hash,
                    stair_job_identity=True,
                )
                lifecycle_root = (root / "generation-lifecycle" / case_id).resolve()
                lifecycle_identity = GenerationLifecycleIdentity(
                    session_id=provisional_inputs.session_contract.session_id,
                    session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                    generation_session_contract_hash=(
                        provisional_inputs.session_contract.contract_hash
                    ),
                    final_confirmation_plan_hash=(
                        provisional_inputs.session_contract.bound_contract_hashes[
                            "final_confirmation_plan"
                        ]
                    ),
                    ledger_epoch_hash=primary_ledger_epoch_hash(
                        FAKE_SESSION_GENESIS_HASH
                    ),
                    output_root=str(lifecycle_root),
                )
                attempt_root = root / "job-attempts" / case_id
                first_segment = GenerationSegmentJournal.start(
                    output_dir=lifecycle_root,
                    identity=lifecycle_identity,
                    policy=continuation_policy,
                    session_contract=provisional_inputs.session_contract,
                    closed_generation_count=0,
                    cost_chain_head_hash="a" * 64,
                    attempt_journal_root=attempt_root,
                )
                first_inputs = _build_inputs(
                    case_id=case_id,
                    provider_a=first_provider_a,
                    provider_b=first_provider_b,
                    events=first_events,
                    runner=first_runner,
                    adapter=_PhasedAdapter(first_events),
                    continuation_policy_hash=continuation_policy.policy_hash,
                    invocation_segment_id=first_segment.segment_id,
                    stair_job_identity=True,
                )

                with self.assertRaisesRegex(
                    _CrashAfterAttemptResult,
                    "after attempt result and before phase receipt",
                ):
                    execute_fixed_width_generation(first_inputs)

                crashed_job_id = first_runner.crashed_job_id
                self.assertIsNotNone(crashed_job_id)
                assert crashed_job_id is not None
                target_results = []
                for result_path in attempt_root.rglob("attempt_00.result.json"):
                    payload = strict_json_object_loads(
                        result_path.read_bytes(),
                        context="fault-injected attempt result",
                    )
                    if payload.get("logical_job_id") == crashed_job_id:
                        target_results.append((result_path, payload))
                self.assertEqual(len(target_results), 1)
                target_result_path, target_attempt_result = target_results[0]
                target_result_bytes_before_resume = target_result_path.read_bytes()
                self.assertEqual(target_attempt_result["outcome"], "completed")
                lane_a_phase_root = root / "trial-receipts" / case_id / "lane-a"
                self.assertEqual(
                    list(lane_a_phase_root.rglob("eval_phases/public_dev.json")),
                    [],
                )

                first_segment.release_unclean_for_test()
                second_segment = GenerationSegmentJournal.start(
                    output_dir=lifecycle_root,
                    identity=lifecycle_identity,
                    policy=continuation_policy,
                    session_contract=first_inputs.session_contract,
                    closed_generation_count=0,
                    cost_chain_head_hash="a" * 64,
                    attempt_journal_root=attempt_root,
                    recovery_process_incarnation=lambda _pid: None,
                    process_group_alive=lambda _group: False,
                )
                self.assertNotEqual(first_segment.segment_id, second_segment.segment_id)

                resumed_events: list[str] = []
                resumed_runner = _AttemptJournalPhasedRunner(
                    resumed_events,
                    runner_output_root=runner_outputs,
                )
                resumed_inputs = _build_inputs(
                    case_id=case_id,
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(candidate_b),
                    events=resumed_events,
                    runner=resumed_runner,
                    adapter=_PhasedAdapter(resumed_events),
                    continuation_policy_hash=continuation_policy.policy_hash,
                    invocation_segment_id=second_segment.segment_id,
                    stair_job_identity=True,
                )
                resumed_lanes = []
                for lane in resumed_inputs.lanes:
                    original_round_inputs = lane.build_round_inputs
                    self.assertIsInstance(lane.attempt_scope, JobAttemptScope)
                    self.assertIsInstance(lane.sealed_attempt_scope, JobAttemptScope)
                    assert isinstance(lane.attempt_scope, JobAttemptScope)
                    assert isinstance(lane.sealed_attempt_scope, JobAttemptScope)
                    operation_scopes = recover_stair_generation_lane_operation_scopes(
                        attempt_journal_root=attempt_root,
                        generation_id=resumed_inputs.manifest.generation_id,
                        lane_id=lane.lane_id,
                        current_invocation_segment_id=second_segment.segment_id,
                        current_scopes=StairGenerationLaneOperationScopes(
                            train=lane.attempt_scope,
                            public_dev=lane.attempt_scope,
                            public_accept=lane.attempt_scope,
                            sealed_validation=lane.sealed_attempt_scope,
                        ),
                    )

                    def _build_resumed_round_inputs(
                        authored,
                        *,
                        original_round_inputs=original_round_inputs,
                        operation_scopes=operation_scopes,
                    ):
                        return replace(
                            original_round_inputs(authored),
                            train_attempt_scope=operation_scopes.train,
                            eval_attempt_scope=operation_scopes.public_dev,
                        )

                    resumed_lanes.append(
                        replace(
                            lane,
                            build_round_inputs=_build_resumed_round_inputs,
                            **operation_scopes.to_generation_lane_runtime_kwargs(),
                        )
                    )
                resumed_inputs = replace(
                    resumed_inputs,
                    lanes=tuple(resumed_lanes),
                )

                resumed = execute_fixed_width_generation(resumed_inputs)

                self.assertTrue(resumed.committed)
                self.assertFalse(resumed.reused_execution)
                self.assertEqual(
                    first_runner.actual_job_calls[("eval_phase", crashed_job_id)],
                    1,
                )
                self.assertNotIn(
                    ("eval_phase", crashed_job_id),
                    resumed_runner.actual_job_calls,
                )
                self.assertEqual(
                    resumed_runner.recovered_job_calls,
                    {("eval_phase", crashed_job_id): 1},
                )
                self.assertEqual(
                    target_result_path.read_bytes(),
                    target_result_bytes_before_resume,
                )
                self.assertEqual(
                    list(attempt_root.rglob("attempt_01.reservation.json")),
                    [],
                )

                reservations = sorted(attempt_root.rglob("attempt_00.reservation.json"))
                results = sorted(attempt_root.rglob("attempt_00.result.json"))
                # Two train + two public_dev + one selected public_accept + one sealed.
                self.assertEqual(len(reservations), 6)
                self.assertEqual(len(results), 6)
                logical_jobs: set[tuple[str, str, str]] = set()
                segment_by_job: dict[tuple[str, str], str] = {}
                for reservation_path in reservations:
                    raw = reservation_path.read_bytes()
                    payload = strict_json_object_loads(
                        raw,
                        context="generation attempt reservation",
                    )
                    self.assertEqual(
                        raw,
                        canonical_json_bytes(payload, trailing_newline=True),
                    )
                    journal, _reservation = JobAttemptJournal.from_reservation_path(
                        reservation_path
                    )
                    result_ref = journal.load_result(0)
                    self.assertIn(
                        result_ref.payload["outcome"],
                        {"completed", "scientific_failure"},
                    )
                    logical_jobs.add(
                        (
                            journal.logical_job_id,
                            journal.job_kind,
                            journal.job_input_hash,
                        )
                    )
                    segment_by_job[(journal.job_kind, journal.logical_job_id)] = (
                        journal.scope.invocation_segment_id
                    )
                self.assertEqual(len(logical_jobs), 6)
                self.assertTrue(
                    all(
                        segment_id == first_segment.segment_id
                        for (kind, job_id), segment_id in segment_by_job.items()
                        if kind == "train" or job_id.endswith("-public-dev")
                    )
                )
                self.assertTrue(
                    all(
                        segment_id == second_segment.segment_id
                        for (kind, job_id), segment_id in segment_by_job.items()
                        if kind == "eval_phase" and not job_id.endswith("-public-dev")
                    )
                )

                lane_a_phase_receipts = list(
                    lane_a_phase_root.rglob("eval_phases/public_dev.json")
                )
                self.assertEqual(len(lane_a_phase_receipts), 1)
                phase_receipt = strict_json_object_loads(
                    lane_a_phase_receipts[0].read_bytes(),
                    context="resumed public-dev phase receipt",
                )
                self.assertEqual(
                    phase_receipt["phase_result_hash"],
                    target_attempt_result["output_hash"],
                )
                second_segment.finish(
                    status="paused",
                    reason="operator_pause",
                    closed_generation_count=1,
                    cost_chain_head_hash=second_segment.cost_chain_head_hash,
                )

            with self.subTest(
                scenario="bound_v2_finalizes_once_and_replays_from_private_manifest"
            ):
                formal_gpu_bindings = (
                    GpuProcessBinding(
                        host_execution_device="cuda:1",
                        process_execution_device="cuda:0",
                        resource_binding_hash="8" * 64,
                    ),
                    GpuProcessBinding(
                        host_execution_device="cuda:2",
                        process_execution_device="cuda:0",
                        resource_binding_hash="9" * 64,
                    ),
                )
                bound_events: list[str] = []
                bound_runner = _PhasedRunner(bound_events)
                bound_adapter = _PhasedAdapter(bound_events)
                bound_inputs = _build_inputs(
                    case_id="accepted-bound-v2",
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(candidate_b),
                    events=bound_events,
                    runner=bound_runner,
                    adapter=bound_adapter,
                    max_generations=1,
                    gpu_bindings=formal_gpu_bindings,
                )

                bound_result = execute_fixed_width_generation(bound_inputs)
                bound_ledger = load_ledger(bound_inputs.ledger_path)

                self.assertTrue(bound_result.committed)
                self.assertEqual(len(bound_ledger), 1)
                bound_record = bound_ledger[0]
                self.assertEqual(bound_record["generation_outcome"], "accepted")
                self.assertIsNotNone(bound_record["formal"])
                self.assertTrue(bound_record["formal"]["accepted"])
                self.assertEqual(
                    load_bound_generation_manifest(bound_record),
                    bound_inputs.manifest,
                )
                self.assertEqual(
                    bound_record["manifest"],
                    bound_inputs.manifest.to_public_dict(),
                )
                tampered_public_manifest = deepcopy(bound_record)
                tampered_public_manifest["manifest"]["lanes"][0]["gpu_process_binding"][
                    "process_execution_device"
                ] = "cuda:7"
                tampered_public_manifest["manifest"]["manifest_hash"] = "0" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "manifest public projection mismatch",
                ):
                    validate_generation_ledger_record(
                        tampered_public_manifest,
                        acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                        expected_parent=parent,
                    )
                self.assertNotIn(
                    "host_execution_device",
                    repr(bound_record["manifest"]),
                )
                self.assertIn(
                    "host_execution_device",
                    repr(bound_inputs.manifest.to_dict()),
                )
                assert_public_projection_safe(
                    {
                        key: value
                        for key, value in bound_record.items()
                        if key != "private_artifacts"
                    }
                )
                bound_replay = replay_acceptance_ledger(
                    bound_ledger,
                    initial_current_best=parent,
                    expected_session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                    acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                    expected_ledger_epoch_hash=primary_ledger_epoch_hash(
                        FAKE_SESSION_GENESIS_HASH
                    ),
                )
                self.assertEqual(
                    bound_replay.current_best,
                    bound_result.resulting_current_best,
                )
                last_train_end = max(
                    index
                    for index, event in enumerate(bound_events)
                    if event.startswith("train-end:")
                )
                first_public_dev = min(
                    index
                    for index, event in enumerate(bound_events)
                    if event.startswith("phase-start:public_dev:")
                )
                first_public_accept = min(
                    index
                    for index, event in enumerate(bound_events)
                    if event.startswith("phase-start:public_accept:")
                )
                self.assertLess(last_train_end, first_public_dev)
                self.assertLess(
                    bound_events.index("selection-sealed"),
                    first_public_accept,
                )
                self.assertEqual(
                    [phase for _subject, phase in bound_runner.phase_calls],
                    [
                        EvaluationPhase.PUBLIC_DEV,
                        EvaluationPhase.PUBLIC_DEV,
                        EvaluationPhase.PUBLIC_ACCEPT,
                        EvaluationPhase.SEALED_VALIDATION,
                    ],
                )
                bound_experiment = ExperimentRecord.from_ledger_record(bound_record)
                self.assertNotIn(
                    "host_execution_device",
                    repr(bound_experiment.ledger_public_record()),
                )

            with self.subTest(
                scenario="bound_manifest_rejects_legacy_unbound_receipts"
            ):
                gpu_bindings = (
                    GpuProcessBinding(
                        host_execution_device="cuda:1",
                        process_execution_device="cuda:0",
                        resource_binding_hash="8" * 64,
                    ),
                    GpuProcessBinding(
                        host_execution_device="cuda:2",
                        process_execution_device="cuda:0",
                        resource_binding_hash="9" * 64,
                    ),
                )
                bound_manifest = replace(
                    generation_inputs.manifest,
                    schema_version=GENERATION_MANIFEST_V2_SCHEMA_VERSION,
                    lanes=tuple(
                        replace(
                            assignment,
                            execution_device=binding.process_execution_device,
                            resource_binding_hash=binding.resource_binding_hash,
                            gpu_process_binding=binding,
                        )
                        for assignment, binding in zip(
                            generation_inputs.manifest.lanes,
                            gpu_bindings,
                            strict=True,
                        )
                    ),
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "versioned session cost contract",
                ):
                    execute_fixed_width_generation(
                        replace(generation_inputs, manifest=bound_manifest)
                    )
                legacy_trial = ledger[0]["trials"][0]
                legacy_assignment = generation_inputs.manifest.lanes[0]
                _load_bound_train_projection(
                    legacy_trial,
                    assignment=legacy_assignment,
                    private_artifacts=ledger[0]["private_artifacts"],
                )
                _load_bound_phase_projection(
                    legacy_trial,
                    assignment=legacy_assignment,
                    phase=EvaluationPhase.PUBLIC_DEV,
                    private_artifacts=ledger[0]["private_artifacts"],
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "train projection/receipt binding mismatch",
                ):
                    _load_bound_train_projection(
                        legacy_trial,
                        assignment=bound_manifest.lanes[0],
                        private_artifacts=ledger[0]["private_artifacts"],
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "phase projection/receipt binding mismatch",
                ):
                    _load_bound_phase_projection(
                        legacy_trial,
                        assignment=bound_manifest.lanes[0],
                        phase=EvaluationPhase.PUBLIC_DEV,
                        private_artifacts=ledger[0]["private_artifacts"],
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "formal round receipt binding mismatch",
                ):
                    _load_and_validate_formal(
                        ledger[0]["formal"],
                        manifest=bound_manifest,
                        private_artifacts=ledger[0]["private_artifacts"],
                        selected_trial_id=ledger[0]["selection"]["selected_trial_id"],
                        trials=ledger[0]["trials"],
                        session_genesis_hash=ledger[0]["session_genesis_hash"],
                        ledger_epoch_hash=ledger[0]["ledger_epoch_hash"],
                    )

            replay = replay_acceptance_ledger(
                ledger,
                initial_current_best=parent,
                expected_session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                expected_ledger_epoch_hash=primary_ledger_epoch_hash(
                    FAKE_SESSION_GENESIS_HASH
                ),
            )
            with self.subTest(scenario="replay_advances_head_once"):
                self.assertEqual(
                    replay.current_best.anchor_id,
                    "candidate-generation-b",
                )

            with self.subTest(
                scenario="one_generation_lifecycle_closes_terminal_binding"
            ):
                continuation_policy = GenerationContinuationPolicy(
                    max_segments=2,
                    allowed_continuation_reasons=(
                        "operator_pause",
                        "typed_infrastructure_failure",
                        "exception",
                    ),
                )
                terminal_events: list[str] = []
                terminal_inputs = _build_inputs(
                    case_id="terminal-one",
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(candidate_b),
                    events=terminal_events,
                    runner=_PhasedRunner(terminal_events),
                    adapter=_PhasedAdapter(terminal_events),
                    max_generations=1,
                    continuation_policy_hash=continuation_policy.policy_hash,
                )
                lifecycle_root = (
                    root / "generation-lifecycle" / "terminal-one"
                ).resolve()
                attempt_root = root / "job-attempts" / "terminal-one"
                lifecycle_identity = GenerationLifecycleIdentity(
                    session_id=terminal_inputs.session_contract.session_id,
                    session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                    generation_session_contract_hash=(
                        terminal_inputs.session_contract.contract_hash
                    ),
                    final_confirmation_plan_hash=(
                        terminal_inputs.session_contract.bound_contract_hashes[
                            "final_confirmation_plan"
                        ]
                    ),
                    ledger_epoch_hash=primary_ledger_epoch_hash(
                        FAKE_SESSION_GENESIS_HASH
                    ),
                    output_root=str(lifecycle_root),
                )
                segment = GenerationSegmentJournal.start(
                    output_dir=lifecycle_root,
                    identity=lifecycle_identity,
                    policy=continuation_policy,
                    session_contract=terminal_inputs.session_contract,
                    closed_generation_count=0,
                    cost_chain_head_hash="a" * 64,
                    attempt_journal_root=attempt_root,
                )
                terminal_result = execute_fixed_width_generation(terminal_inputs)
                self.assertEqual(
                    terminal_result.record["generation_outcome"], "accepted"
                )
                segment.finish(
                    status="completed",
                    reason="normal",
                    closed_generation_count=1,
                    cost_chain_head_hash="a" * 64,
                )
                terminal_binding = build_generation_terminal_binding(
                    ledger_path=terminal_inputs.ledger_path,
                    output_dir=lifecycle_root,
                    attempt_journal_root=attempt_root,
                    identity=lifecycle_identity,
                    policy=continuation_policy,
                    session_contract=terminal_inputs.session_contract,
                    initial_current_best=parent,
                    acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                )
                self.assertEqual(terminal_binding.closed_generation_count, 1)
                self.assertEqual(
                    terminal_binding.current_head_hash,
                    canonical_json_hash(
                        terminal_result.resulting_current_best.to_public_dict()
                    ),
                )

            with self.subTest(scenario="session_contract_drift_is_rejected"):
                drifted_contract_record = deepcopy(ledger[0])
                drifted_contract_record["generation_id"] = "generation-contract-drift"
                drifted_contract_record["generation_index"] = 2
                drifted_contract_record["generation_session_contract_hash"] = "f" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "session contract changed within the ledger",
                ):
                    replay_acceptance_ledger(
                        [ledger[0], drifted_contract_record],
                        initial_current_best=parent,
                        expected_session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                        acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                        expected_ledger_epoch_hash=primary_ledger_epoch_hash(
                            FAKE_SESSION_GENESIS_HASH
                        ),
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "session contract changed within the ledger",
                ):
                    commit_generation_record(
                        ledger_path=generation_inputs.ledger_path,
                        record=drifted_contract_record,
                        initial_current_best=parent,
                        acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                    )
                self.assertEqual(load_ledger(generation_inputs.ledger_path), ledger)
            with self.subTest(scenario="trial_delta_must_match_train_contract"):
                tampered_delta = deepcopy(ledger[0])
                selected_trial = next(
                    item
                    for item in tampered_delta["trials"]
                    if item["trial_id"] == tampered_delta["selection"][
                        "selected_trial_id"
                    ]
                )
                selected_trial["mechanism_operation_delta"][
                    "candidate_source_checksum"
                ] = "sha256:" + "9" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "train projection/receipt binding mismatch",
                ):
                    validate_generation_ledger_record(
                        tampered_delta,
                        acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                        expected_parent=parent,
                    )
            projected = project_public_history(
                ledger,
                initial_current_best=parent,
                expected_session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                expected_authoring_mode=PROVIDER_AUTHORING_MODE,
                expected_ledger_epoch_hash=primary_ledger_epoch_hash(
                    FAKE_SESSION_GENESIS_HASH
                ),
            )
            with self.subTest(scenario="public_projection_is_safe_and_replay_equal"):
                self.assertEqual(projected.current_best, replay.current_best)
                generation_experiment = ExperimentRecord.from_ledger_record(ledger[0])
                public_record = generation_experiment.ledger_public_record()
                self.assertNotIn("private_artifacts", public_record)
                self.assertNotIn(str(root), repr(public_record))
                self.assertEqual(len(public_record["trials"]), 2)
                next_prompt = context_builder.build_prompt([generation_experiment])
                self.assertEqual(
                    next_prompt.parent_program_source_checksum,
                    result.resulting_current_best.program_source_checksum,
                )
                self.assertEqual(
                    next_prompt.provider_payload["candidate_lineage"]["status"],
                    "accepted_current_best",
                )
                self.assertEqual(
                    next_prompt.provider_payload["candidate_lineage"]["current_best"][
                        "source_delivery"
                    ],
                    "inline",
                )
            with self.subTest(scenario="authoring_receipt_tamper_is_rejected"):
                tampered = deepcopy(ledger[0])
                authoring_ref = next(
                    item
                    for item in tampered["private_artifacts"]
                    if item["kind"] == "lane-a_authoring_evidence"
                )
                authoring_ref["checksum"] = "sha256:" + "0" * 64
                with self.assertRaisesRegex(ValueError, "not uniquely bound"):
                    validate_generation_ledger_record(
                        tampered,
                        acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                        expected_parent=parent,
                    )

            with self.subTest(scenario="phase_projection_tamper_is_rejected"):
                tampered_phase = deepcopy(ledger[0])
                tampered_phase["trials"][1]["public_accept"]["task_evidence_hash"] = (
                    "0" * 64
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "phase projection/receipt binding mismatch",
                ):
                    validate_generation_ledger_record(
                        tampered_phase,
                        acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                        expected_parent=parent,
                    )

            with self.subTest(scenario="sibling_exclusion_tamper_is_rejected"):
                tampered_exclusion = deepcopy(ledger[0])
                tampered_exclusion["trials"][1]["authoring"][
                    "sibling_exclusion_hash"
                ] = "0" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "lane B authoring prompt exclusion mismatch",
                ):
                    validate_generation_ledger_record(
                        tampered_exclusion,
                        acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                        expected_parent=parent,
                    )

            with self.subTest(scenario="round_and_generation_records_cannot_mix"):
                with self.assertRaisesRegex(ValueError, "record kinds cannot be mixed"):
                    append_ledger_record(
                        generation_inputs.ledger_path,
                        dict(parent_execution.record),
                    )

            with self.subTest(scenario="no_selection_never_queries_sealed"):
                tie_events: list[str] = []
                tie_runner = _PhasedRunner(tie_events)
                tie_adapter = _PhasedAdapter(
                    tie_events,
                    force_no_selection=True,
                )
                tie_inputs = _build_inputs(
                    case_id="no-selection",
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(candidate_b),
                    events=tie_events,
                    runner=tie_runner,
                    adapter=tie_adapter,
                )
                tie_result = execute_fixed_width_generation(tie_inputs)
                self.assertEqual(
                    tie_result.record["generation_outcome"], "no_selection"
                )
                self.assertIsNone(tie_result.record["selection"]["selected_trial_id"])
                self.assertIsNone(tie_result.record["formal"])
                self.assertEqual(
                    [item["trial_status"] for item in tie_result.record["trials"]],
                    ["qualified_not_selected", "qualified_not_selected"],
                )
                self.assertEqual(
                    [item["status_reason"] for item in tie_result.record["trials"]],
                    ["no_public_dev_selection", "no_public_dev_selection"],
                )
                self.assertEqual(tie_result.resulting_current_best, parent)
                self.assertEqual(
                    sum(
                        phase is EvaluationPhase.PUBLIC_ACCEPT
                        for _, phase in tie_runner.phase_calls
                    ),
                    0,
                )
                self.assertEqual(
                    sum(
                        phase is EvaluationPhase.SEALED_VALIDATION
                        for _, phase in tie_runner.phase_calls
                    ),
                    0,
                )
                tie_experiment = ExperimentRecord.from_ledger_record(
                    tie_result.record
                )
                self.assertTrue(
                    tie_result.record["trials"][1]["candidate"]["artifact_hash"]
                )
                self.assertIn(
                    "lane-b_candidate_artifact",
                    {artifact.kind for artifact in tie_experiment.private_artifacts},
                )
                tie_prompt = context_builder.build_prompt([tie_experiment])
                tie_memory = tie_prompt.round_public_context["experiment_memory"]
                self.assertEqual(len(tie_memory["experiments"]), 1)
                self.assertEqual(tie_memory["experiments"][0]["reason"], "no_selection")
                self.assertNotIn("candidate_id", tie_memory["experiments"][0])
                self.assertNotIn("candidate_components", tie_memory["experiments"][0])
                latest_failed_payloads = [
                    item
                    for item in tie_prompt.provider_payload["source_payloads"]
                    if item["logical_id"]
                    == "attempt/latest_failed/training_code/components.py"
                ]
                self.assertEqual(
                    len(latest_failed_payloads),
                    1,
                    tie_prompt.provider_payload["source_catalog"],
                )
                latest_failed_payload = latest_failed_payloads[0]
                self.assertIn(
                    "candidate-generation-b",
                    latest_failed_payload["content"],
                )

            with self.subTest(
                scenario="formal_reject_best_observed_source_matches_selected_lane"
            ):
                rejected_a = _write_candidate(
                    root / "formal-reject-provider",
                    candidate_id="candidate-formal-reject-a",
                    reward_expression="inputs.base_height + inputs.roll",
                    public_score=0.24,
                )
                rejected_b = _write_candidate(
                    root / "formal-reject-provider",
                    candidate_id="candidate-formal-reject-b",
                    reward_expression="inputs.base_height + inputs.pitch",
                    public_score=0.20,
                )
                reject_events: list[str] = []
                reject_result = execute_fixed_width_generation(
                    _build_inputs(
                        case_id="formal-reject-source-binding",
                        provider_a=_OneCandidateProvider(rejected_a),
                        provider_b=_OneCandidateProvider(rejected_b),
                        events=reject_events,
                        runner=_PhasedRunner(reject_events),
                        adapter=_PhasedAdapter(reject_events),
                    )
                )
                self.assertEqual(
                    reject_result.record["generation_outcome"],
                    "formal_rejected",
                )
                reject_prompt = context_builder.build_prompt(
                    [ExperimentRecord.from_ledger_record(reject_result.record)]
                ).provider_payload
                observed = reject_prompt["candidate_lineage"][
                    "best_observed_experiment"
                ]
                payloads = {
                    item["logical_id"]: item
                    for item in reject_prompt["source_payloads"]
                }
                self.assertEqual(
                    observed["candidate_id"],
                    "candidate-formal-reject-a",
                )
                self.assertEqual(
                    observed["source_logical_id"],
                    "memory/best_observed/training_code/components.py",
                )
                self.assertIn(
                    "candidate-formal-reject-a",
                    payloads[
                        "memory/best_observed/training_code/components.py"
                    ]["content"],
                )
                self.assertIn(
                    "candidate-formal-reject-b",
                    payloads[
                        "attempt/latest_failed/training_code/components.py"
                    ]["content"],
                )
                self.assertEqual(
                    observed["source_sha256"],
                    payloads[
                        "memory/best_observed/training_code/components.py"
                    ]["content_sha256"],
                )
                self.assertNotEqual(
                    payloads[
                        "memory/best_observed/training_code/components.py"
                    ]["content_sha256"],
                    payloads[
                        "attempt/latest_failed/training_code/components.py"
                    ]["content_sha256"],
                )

            with self.subTest(scenario="lane_b_repeat_closes_without_training"):
                duplicate_b = _write_candidate(
                    root / "generation-provider",
                    candidate_id="candidate-generation-duplicate",
                    reward_expression="inputs.base_height + inputs.roll",
                    public_score=0.55,
                )
                incomplete_events: list[str] = []
                incomplete_runner = _PhasedRunner(incomplete_events)
                incomplete_inputs = _build_inputs(
                    case_id="b-incomplete",
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(duplicate_b),
                    events=incomplete_events,
                    runner=incomplete_runner,
                    adapter=_PhasedAdapter(incomplete_events),
                )
                incomplete = execute_fixed_width_generation(incomplete_inputs)
                self.assertEqual(
                    incomplete.record["generation_outcome"],
                    "paired_authoring_incomplete",
                )
                self.assertEqual(
                    [item["trial_status"] for item in incomplete.record["trials"]],
                    ["verified_not_run", "candidate_failed"],
                )
                self.assertEqual(incomplete_runner.phase_calls, [])
                self.assertFalse(
                    any(item.startswith("train-start:") for item in incomplete_events)
                )
                incomplete_prompt = context_builder.build_prompt(
                    [ExperimentRecord.from_ledger_record(incomplete.record)]
                )
                incomplete_memory = incomplete_prompt.round_public_context[
                    "experiment_memory"
                ]["experiments"]
                self.assertEqual(len(incomplete_memory), 1)
                self.assertEqual(
                    incomplete_memory[0]["reason"],
                    "paired_authoring_incomplete",
                )
                self.assertNotIn("candidate_components", incomplete_memory[0])

            with self.subTest(scenario="lane_a_failure_never_starts_lane_b"):
                duplicate_a = _write_candidate(
                    root / "generation-provider",
                    candidate_id="candidate-generation-parent-repeat",
                    reward_expression="inputs.base_height",
                    public_score=0.40,
                )
                a_failure_events: list[str] = []
                a_failure_runner = _PhasedRunner(a_failure_events)
                unused_b_provider = _OneCandidateProvider(candidate_b)
                a_failure_inputs = _build_inputs(
                    case_id="a-incomplete",
                    provider_a=_OneCandidateProvider(duplicate_a),
                    provider_b=unused_b_provider,
                    events=a_failure_events,
                    runner=a_failure_runner,
                    adapter=_PhasedAdapter(a_failure_events),
                )
                a_failure = execute_fixed_width_generation(a_failure_inputs)
                self.assertEqual(
                    [item["trial_status"] for item in a_failure.record["trials"]],
                    ["candidate_failed", "not_started"],
                )
                self.assertEqual(unused_b_provider.payloads, [])
                self.assertEqual(a_failure_runner.phase_calls, [])
                self.assertFalse(
                    any(item.startswith("train-start:") for item in a_failure_events)
                )

            with self.subTest(scenario="lane_a_receipt_is_reused_after_b_interrupt"):
                resume_policy = GenerationContinuationPolicy(
                    max_segments=3,
                    allowed_continuation_reasons=(
                        "operator_pause",
                        "typed_infrastructure_failure",
                        "exception",
                    ),
                )
                resume_events: list[str] = []
                resume_runner = _PhasedRunner(resume_events)
                resume_provider_a = _OneCandidateProvider(candidate_a)
                resume_provider_b = _OneCandidateProvider(candidate_b)
                resume_inputs = _build_inputs(
                    case_id="author-resume",
                    provider_a=resume_provider_a,
                    provider_b=resume_provider_b,
                    events=resume_events,
                    runner=resume_runner,
                    adapter=_PhasedAdapter(resume_events),
                    continuation_policy_hash=resume_policy.policy_hash,
                )
                resume_lifecycle_root = (
                    root / "generation-lifecycle" / "author-resume"
                ).resolve()
                resume_attempt_root = root / "job-attempts" / "author-resume"
                resume_identity = GenerationLifecycleIdentity(
                    session_id=resume_inputs.session_contract.session_id,
                    session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                    generation_session_contract_hash=(
                        resume_inputs.session_contract.contract_hash
                    ),
                    final_confirmation_plan_hash=(
                        resume_inputs.session_contract.bound_contract_hashes[
                            "final_confirmation_plan"
                        ]
                    ),
                    ledger_epoch_hash=primary_ledger_epoch_hash(
                        FAKE_SESSION_GENESIS_HASH
                    ),
                    output_root=str(resume_lifecycle_root),
                )
                first_segment = GenerationSegmentJournal.start(
                    output_dir=resume_lifecycle_root,
                    identity=resume_identity,
                    policy=resume_policy,
                    session_contract=resume_inputs.session_contract,
                    closed_generation_count=0,
                    cost_chain_head_hash="a" * 64,
                    attempt_journal_root=resume_attempt_root,
                )
                original_b = resume_inputs.lanes[1]
                b_calls = 0

                def _interrupt_b_once(exclusion):
                    nonlocal b_calls
                    b_calls += 1
                    if b_calls == 1:
                        raise RuntimeError(
                            "injected interruption before lane B authoring"
                        )
                    return original_b.author(exclusion)

                resume_inputs = replace(
                    resume_inputs,
                    lanes=(
                        resume_inputs.lanes[0],
                        replace(original_b, author=_interrupt_b_once),
                    ),
                )
                with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                    execute_fixed_width_generation(resume_inputs)
                first_segment.release_unclean_for_test()
                self.assertEqual(len(resume_provider_a.payloads), 1)
                self.assertEqual(resume_provider_b.payloads, [])
                second_segment = GenerationSegmentJournal.start(
                    output_dir=resume_lifecycle_root,
                    identity=resume_identity,
                    policy=resume_policy,
                    session_contract=resume_inputs.session_contract,
                    closed_generation_count=0,
                    cost_chain_head_hash="a" * 64,
                    attempt_journal_root=resume_attempt_root,
                    recovery_process_incarnation=lambda _pid: None,
                    process_group_alive=lambda _group: False,
                )
                resumed = execute_fixed_width_generation(resume_inputs)
                self.assertEqual(len(resume_provider_a.payloads), 1)
                self.assertEqual(len(resume_provider_b.payloads), 1)
                self.assertEqual(resumed.record["generation_outcome"], "accepted")
                self.assertEqual(len(load_ledger(resume_inputs.ledger_path)), 1)
                second_segment.finish(
                    status="paused",
                    reason="operator_pause",
                    closed_generation_count=1,
                    cost_chain_head_hash=second_segment.cost_chain_head_hash,
                )
                resume_audit = audit_generation_lifecycle(
                    output_dir=resume_lifecycle_root,
                    identity=resume_identity,
                    policy=resume_policy,
                    require_closed=True,
                )
                self.assertEqual(resume_audit.segment_count, 2)
                self.assertEqual(resume_audit.closed_generation_count, 1)
                self.assertTrue(resume_audit.continuation_eligible)

            with self.subTest(
                scenario="train_timeout_closes_candidate_execution_incomplete"
            ):
                train_timeout_events: list[str] = []
                train_timeout_runner = _TrainFailureRunner(
                    train_timeout_events,
                    failing_subject_id=candidate_a.candidate_id,
                    failure_kind=RunnerFailureKind.TIMEOUT,
                )
                train_timeout_inputs = _build_inputs(
                    case_id="train-timeout",
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(candidate_b),
                    events=train_timeout_events,
                    runner=train_timeout_runner,
                    adapter=_PhasedAdapter(train_timeout_events),
                )
                train_timeout_result = execute_fixed_width_generation(
                    train_timeout_inputs
                )
                self.assertEqual(
                    train_timeout_result.record["generation_outcome"],
                    "candidate_execution_incomplete",
                )
                trials = train_timeout_result.record["trials"]
                self.assertEqual(
                    [trial["trial_status"] for trial in trials],
                    ["candidate_failed", "screened_only"],
                )
                self.assertEqual(trials[0]["status_reason"], "train_incomplete")
                self.assertIsNone(trials[0]["train"])
                self.assertIsNone(trials[0]["public_dev"])
                self.assertIsNotNone(trials[1]["train"])
                self.assertIsNotNone(trials[1]["public_dev"])
                self.assertIsNone(train_timeout_result.record["selection"])
                self.assertIsNone(train_timeout_result.record["formal"])
                self.assertEqual(
                    train_timeout_runner.phase_calls,
                    [(candidate_b.candidate_id, EvaluationPhase.PUBLIC_DEV)],
                )

            with self.subTest(
                scenario="train_contract_failure_remains_fatal"
            ):
                train_contract_events: list[str] = []
                train_contract_runner = _TrainFailureRunner(
                    train_contract_events,
                    failing_subject_id=candidate_a.candidate_id,
                    failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                )
                train_contract_inputs = _build_inputs(
                    case_id="train-contract-failure",
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(candidate_b),
                    events=train_contract_events,
                    runner=train_contract_runner,
                    adapter=_PhasedAdapter(train_contract_events),
                )
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    "injected train runner failure",
                ):
                    execute_fixed_width_generation(train_contract_inputs)

            with self.subTest(
                scenario="public_dev_timeout_closes_candidate_execution_incomplete"
            ):
                timeout_events: list[str] = []
                timeout_runner = _PublicDevFailureRunner(
                    timeout_events,
                    failing_subject_id=candidate_a.candidate_id,
                    failure_kind=RunnerFailureKind.TIMEOUT,
                )
                timeout_inputs = _build_inputs(
                    case_id="public-dev-timeout",
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(candidate_b),
                    events=timeout_events,
                    runner=timeout_runner,
                    adapter=_PhasedAdapter(timeout_events),
                )
                timeout_result = execute_fixed_width_generation(timeout_inputs)
                self.assertEqual(
                    timeout_result.record["generation_outcome"],
                    "candidate_execution_incomplete",
                )
                trials = timeout_result.record["trials"]
                self.assertEqual(
                    [trial["trial_status"] for trial in trials],
                    ["candidate_failed", "screened_only"],
                )
                self.assertEqual(
                    trials[0]["status_reason"],
                    "eval_incomplete_public_dev",
                )
                self.assertIsNotNone(trials[0]["train"])
                self.assertIsNone(trials[0]["public_dev"])
                self.assertIsNotNone(trials[1]["public_dev"])
                self.assertIsNone(timeout_result.record["selection"])
                self.assertIsNone(timeout_result.record["formal"])
                self.assertFalse(
                    any(
                        phase is not EvaluationPhase.PUBLIC_DEV
                        for _, phase in timeout_runner.phase_calls
                    )
                )

            with self.subTest(
                scenario="public_dev_contract_failure_remains_fatal"
            ):
                contract_events: list[str] = []
                contract_runner = _PublicDevFailureRunner(
                    contract_events,
                    failing_subject_id=candidate_a.candidate_id,
                    failure_kind=RunnerFailureKind.CONTRACT_INVALID,
                )
                contract_inputs = _build_inputs(
                    case_id="public-dev-contract-failure",
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(candidate_b),
                    events=contract_events,
                    runner=contract_runner,
                    adapter=_PhasedAdapter(contract_events),
                )
                with self.assertRaisesRegex(
                    RunnerExecutionError,
                    "injected public_dev runner failure",
                ):
                    execute_fixed_width_generation(contract_inputs)

            with self.subTest(scenario="lane_b_prompt_must_bind_exclusion"):
                unbound_events: list[str] = []
                unbound_runner = _PhasedRunner(unbound_events)
                unbound_inputs = _build_inputs(
                    case_id="unbound-b-prompt",
                    provider_a=_OneCandidateProvider(candidate_a),
                    provider_b=_OneCandidateProvider(candidate_b),
                    events=unbound_events,
                    runner=unbound_runner,
                    adapter=_PhasedAdapter(unbound_events),
                )
                unbound_b_provider = _OneCandidateProvider(candidate_b)

                def _author_b_without_generation_directives(_exclusion):
                    return CandidateAuthoringLoop(
                        context_builder=context_builder,
                        provider=unbound_b_provider,
                        candidate_admission_policy=FakeVerifier(),
                        round_dir=(root / "authoring" / "unbound-b-prompt" / "lane-b"),
                        max_attempts=1,
                        generation_lane_intent=lane_intents["lane-b"],
                    ).run_round(
                        round_id="unbound-b-prompt-lane-b",
                        records=[],
                    )

                unbound_inputs = replace(
                    unbound_inputs,
                    lanes=(
                        unbound_inputs.lanes[0],
                        replace(
                            unbound_inputs.lanes[1],
                            author=_author_b_without_generation_directives,
                        ),
                    ),
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "lane B prompt exclusion mismatch",
                ):
                    execute_fixed_width_generation(unbound_inputs)
                self.assertFalse(
                    any(item.startswith("train-start:") for item in unbound_events)
                )


if __name__ == "__main__":
    unittest.main()
