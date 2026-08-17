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
    execute_fixed_width_generation,
)
from true_rl_agent.agent_core.generation_record import (
    commit_generation_record,
    load_bound_generation_trial_phase_receipt,
    validate_generation_ledger_record,
)
from true_rl_agent.agent_core.generation_contract import (
    GenerationLaneAssignment,
    GenerationManifest,
    GenerationSessionContract,
    generation_resource_plan_hash,
    generation_training_initialization_hash,
)
from true_rl_agent.agent_core.ledger import (
    NON_PROVIDER_AUTHORING_MODE,
    PROVIDER_AUTHORING_MODE,
    append_ledger_record,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.round import RoundInputs, execute_candidate_trial
from true_rl_agent.agent_core.trial_receipts import TrialReceiptRef
from true_rl_agent.backends.base import BackendJobBundle
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.evaluation.acceptance_replay import (
    project_public_history,
    replay_acceptance_ledger,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.specs import EvalPhaseJobSpec


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

    def next_candidate(self, *, prompt_payload: dict[str, Any]) -> CandidatePackageRef:
        self.payloads.append(prompt_payload)
        return self.candidate


class _CpuPhasedBackend(FakeBackendAdapter):
    def materialize_jobs(self, **kwargs: Any) -> BackendJobBundle:
        bundle = super().materialize_jobs(**kwargs)
        return BackendJobBundle(
            train_spec=replace(bundle.train_spec, execution_device="cpu"),
            eval_spec=replace(bundle.eval_spec, execution_device="cpu"),
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
            job_id=f"{eval_plan.job_id}-{phase.value}",
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
            "reason": "public_dev_objective_rank",
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
            EvaluationPhase.PUBLIC_ACCEPT: threading.Barrier(2, timeout=5.0),
        }
        self.phase_calls: list[tuple[str, EvaluationPhase]] = []

    def _event(self, value: str) -> None:
        with self._lock:
            self.events.append(value)

    def run_train(self, train_spec):
        self._event(f"train-start:{train_spec.subject.subject_id}")
        self._train_barrier.wait()
        result = super().run_train(train_spec)
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
        )
        self._event(f"phase-end:{eval_spec.phase.value}:{subject_id}")
        return result

    def assert_selection_precedes_public_accept(self) -> None:
        with self._lock:
            if "selection-sealed" not in self.events:
                raise AssertionError("public_accept started before selection was sealed")

    def project_assembled_eval_summary(self, eval_spec, assembly):
        if not isinstance(assembly, _Assembly):
            raise TypeError("test phased runner requires a bound assembly")
        if set(assembly.phase_receipt_refs) != set(EvaluationPhase):
            raise ValueError("test assembly is missing phase receipts")
        return super().run_eval(eval_spec)


class _PhasedAdapter:
    def __init__(
        self,
        events: list[str],
        *,
        force_no_selection: bool = False,
    ) -> None:
        self.events = events
        self.force_no_selection = force_no_selection

    def selection_contract(self) -> Mapping[str, Any]:
        return {"schema_version": "test.selection_contract.v1", "width": 2}

    def cost_contract(self) -> Mapping[str, Any]:
        return {"schema_version": "test.cost_contract.v1", "max_seed_units": 23}

    def project_public_dev_screen(
        self,
        *,
        trial_id: str,
        phase_result: EvaluationPhaseResult,
    ) -> _Screen:
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
    def test_full_generation_is_serial_author_parallel_execute_and_single_commit(self) -> None:
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
            training_binding = fake_training_binding(
                PolicyInitialization.random()
            )

            def _build_inputs(
                *,
                case_id: str,
                provider_a: _OneCandidateProvider,
                provider_b: _OneCandidateProvider,
                events: list[str],
                runner: _PhasedRunner,
                adapter: _PhasedAdapter,
            ) -> FixedWidthGenerationInputs:
                contract = GenerationSessionContract(
                    session_id="generation-session",
                    task_id=task.task_spec.task_id,
                    backend_id=backend.backend_id,
                    max_generations=15,
                    max_trials=30,
                    max_train_jobs=30,
                    bound_contract_hashes={
                        "authoring_policy": "1" * 64,
                        "continuation_policy": "2" * 64,
                        "evaluation_cost": canonical_json_hash(
                            dict(adapter.cost_contract())
                        ),
                        "evaluation_protocol": binding.eval_protocol_hash,
                        "final_confirmation_plan": "3" * 64,
                        "provider_policy": "4" * 64,
                        "resource_plan": generation_resource_plan_hash(
                            lane_assignments
                        ),
                        "retry_policy": "5" * 64,
                        "selector": canonical_json_hash(
                            dict(adapter.selection_contract())
                        ),
                    },
                )

                def _trial_id(lane_id: str) -> str:
                    return f"{case_id}-{lane_id}"

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
                        max_attempts=1,
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
                            root
                            / "unused-round-ledgers"
                            / case_id
                            / f"{lane_id}.jsonl"
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
                        parent_program_source_checksum=(
                            parent.program_source_checksum
                        ),
                        prompt_payload_ref=authored.prompt_payload_ref,
                        authoring_evidence_ref=authored.authoring_evidence_ref,
                    )

                manifest = GenerationManifest(
                    generation_id=f"generation-{case_id}",
                    generation_index=1,
                    session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
                    generation_session_contract_hash=contract.contract_hash,
                    ledger_epoch_hash=primary_ledger_epoch_hash(
                        FAKE_SESSION_GENESIS_HASH
                    ),
                    parent_program_source_checksum=(
                        parent.program_source_checksum
                    ),
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
                    lanes=lane_assignments,
                )
                lanes = tuple(
                    GenerationLaneRuntime(
                        lane_id=lane_id,
                        trial_id=_trial_id(lane_id),
                        lane_intent=lane_intents[lane_id],
                        lane_intent_hash=canonical_json_hash(
                            lane_intents[lane_id]
                        ),
                        author=(
                            lambda exclusion, lane_id=lane_id, provider=provider:
                            _author(
                                lane_id=lane_id,
                                provider=provider,
                                exclusion=exclusion,
                            )
                        ),
                        build_round_inputs=(
                            lambda authored, lane_id=lane_id: _round_inputs(
                                lane_id, authored
                            )
                        ),
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

            provider_a = _OneCandidateProvider(candidate_a)
            provider_b = _OneCandidateProvider(candidate_b)
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
                    2,
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
                self.assertTrue(provider_b.payloads)
                provider_b_text = repr(provider_b.payloads[0])
                self.assertNotIn("candidate-generation-a", provider_b_text)
                self.assertNotIn(
                    "inputs.base_height + inputs.roll", provider_b_text
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

            with self.subTest(scenario="session_contract_drift_is_rejected"):
                drifted_contract_record = deepcopy(ledger[0])
                drifted_contract_record["generation_id"] = (
                    "generation-contract-drift"
                )
                drifted_contract_record["generation_index"] = 2
                drifted_contract_record[
                    "generation_session_contract_hash"
                ] = "f" * 64
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
                self.assertEqual(
                    load_ledger(generation_inputs.ledger_path), ledger
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
                generation_experiment = ExperimentRecord.from_ledger_record(
                    ledger[0]
                )
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
                    next_prompt.provider_payload["candidate_lineage"][
                        "current_best"
                    ]["source_delivery"],
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
                tampered_phase["trials"][1]["public_accept"][
                    "task_evidence_hash"
                ] = "0" * 64
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
                with self.assertRaisesRegex(
                    ValueError, "record kinds cannot be mixed"
                ):
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
                self.assertIsNone(
                    tie_result.record["selection"]["selected_trial_id"]
                )
                self.assertIsNone(tie_result.record["formal"])
                self.assertEqual(
                    [
                        item["trial_status"]
                        for item in tie_result.record["trials"]
                    ],
                    ["qualified_not_selected", "qualified_not_selected"],
                )
                self.assertEqual(tie_result.resulting_current_best, parent)
                self.assertEqual(
                    sum(
                        phase is EvaluationPhase.PUBLIC_ACCEPT
                        for _, phase in tie_runner.phase_calls
                    ),
                    2,
                )
                self.assertEqual(
                    sum(
                        phase is EvaluationPhase.SEALED_VALIDATION
                        for _, phase in tie_runner.phase_calls
                    ),
                    0,
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
                    [
                        item["trial_status"]
                        for item in incomplete.record["trials"]
                    ],
                    ["verified_not_run", "candidate_failed"],
                )
                self.assertEqual(incomplete_runner.phase_calls, [])
                self.assertFalse(
                    any(
                        item.startswith("train-start:")
                        for item in incomplete_events
                    )
                )

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
                    [
                        item["trial_status"]
                        for item in a_failure.record["trials"]
                    ],
                    ["candidate_failed", "not_started"],
                )
                self.assertEqual(unused_b_provider.payloads, [])
                self.assertEqual(a_failure_runner.phase_calls, [])
                self.assertFalse(
                    any(
                        item.startswith("train-start:")
                        for item in a_failure_events
                    )
                )

            with self.subTest(scenario="lane_a_receipt_is_reused_after_b_interrupt"):
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
                self.assertEqual(len(resume_provider_a.payloads), 1)
                self.assertEqual(resume_provider_b.payloads, [])
                resumed = execute_fixed_width_generation(resume_inputs)
                self.assertEqual(len(resume_provider_a.payloads), 1)
                self.assertEqual(len(resume_provider_b.payloads), 1)
                self.assertEqual(
                    resumed.record["generation_outcome"], "accepted"
                )
                self.assertEqual(
                    len(load_ledger(resume_inputs.ledger_path)), 1
                )

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
                        round_dir=(
                            root
                            / "authoring"
                            / "unbound-b-prompt"
                            / "lane-b"
                        ),
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
                    any(
                        item.startswith("train-start:")
                        for item in unbound_events
                    )
                )


if __name__ == "__main__":
    unittest.main()
