from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import threading
import unittest
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.fakes.backend_environment import (
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
    use_test_isaac_lab_execution_environment,
)
from true_rl_agent.agent_core import authoring_loop as candidate_authoring_loop
from true_rl_agent.agent_core.authoring_response import CandidateDraftResponse
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError
from true_rl_agent.agent_core.session_genesis import (
    VerifiedSessionGenesis,
    prepare_session_genesis,
    session_genesis_identity_hash,
)
from true_rl_agent.agent_core.trial_receipts import (
    _train_result_from_private_mapping,
    _train_result_to_private_mapping,
)
from true_rl_agent.candidates.capability_readiness import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
)
from true_rl_agent.candidates.loader import (
    load_training_components,
    load_training_components_source_bytes,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object
from true_rl_agent.runner.base import Runner, TrainJobResult
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.checkpoint import seal_checkpoint
from true_rl_agent.runner.initial_policy import InitialPolicyStatePairing
from true_rl_agent.runner.job_argv import validate_eval_job_argv
from true_rl_agent.runner.job_attempts import JobAttemptJournal
from true_rl_agent.runner.process_handle import OWNED_PROCESS_IDENTITY_SCHEMA_VERSION
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, EvalPhaseJobSpec
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing import (
    candidate_admission as stair_candidate_admission,
)
from true_rl_agent.tasks.stair_climbing import generation_session
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    build_stair_evaluation_summary,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    STAIR_FINAL_CONFIRMATION_DIRNAME,
)
from true_rl_agent.tasks.stair_climbing.gpu_process_binding import (
    STAIR_GPU_COMPUTE_CONTEXT_OBSERVATION_SCHEMA_VERSION,
    STAIR_GPU_RUNTIME_EVIDENCE_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    StairEvaluationExecutionResult,
    preflight_stair_execution_source_manifest,
)
from true_rl_agent.tasks.stair_climbing.launch_readiness import (
    STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
    STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
    _inspect_v3_launch_subject_values,
)
from true_rl_agent.tasks.stair_climbing.phased_evaluation import (
    build_stair_eval_phase_summary,
    build_stair_full_eval_summary,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    STAIR_REWARD_ENVIRONMENT_AXIS_EVIDENCE_SCHEMA_VERSION,
    STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION,
    qualify_stair_candidate_reward_program,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_unified_stair_summary,
)
from true_rl_agent.training_programs import CANONICAL_TRAINING_COMPONENTS_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_ID = "p6c-production-shaped-fixture"
FORMAL_SESSION_ID = "p6d-production-shaped-fixture"
SOURCE_COMMIT = "a" * 40
GPU_UUIDS = (
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
)
HARD_CRASH_EXIT_CODE = 86


def _torch_runtime_importable() -> bool:
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


_CANDIDATE_TERMS = {
    (1, "lane-a"): (
        "fixture_base_elevation",
        "return inputs.base_height",
        0.5,
    ),
    (1, "lane-b"): (
        "fixture_goal_progress",
        "return inputs.goal_progress_fraction",
        0.2,
    ),
    (2, "lane-a"): (
        "fixture_progress_height",
        "return inputs.base_height * inputs.goal_progress_fraction",
        0.4,
    ),
    (2, "lane-b"): (
        "fixture_support_height",
        (
            "return inputs.support_pos_relative_to_terrain_origin[:, :, 2]"
            ".max(dim=1).values"
        ),
        0.1,
    ),
}


def _config(root: Path, *, pause_after_generation: int | None) -> Any:
    return generation_session.StairGenerationPilotConfig(
        project_root=PROJECT_ROOT,
        output_dir=(root / "output").resolve(),
        lease_root=(root / "leases").resolve(),
        session_id=SESSION_ID,
        expected_source_commit=SOURCE_COMMIT,
        machine_id="fixture-machine",
        gpu0_uuid=GPU_UUIDS[0],
        gpu1_uuid=GPU_UUIDS[1],
        provider_env_file=root / "provider.env",
        max_allocated_active_gpu_hours=12.5,
        max_simulated_control_steps=1_577_542_128,
        max_output_bytes=3_758_096_384,
        nvidia_smi_executable=str(root / _nvidia_smi_name()),
        resume_existing=(root / "output").exists(),
        session_mode=generation_session.DEVELOPMENT_GENERATION_VALIDATION_MODE,
        pause_after_generation=pause_after_generation,
    )


def _formal_config(root: Path, *, pause_after_generation: int | None) -> Any:
    return generation_session.StairGenerationPilotConfig(
        project_root=PROJECT_ROOT,
        output_dir=(root / "formal-output").resolve(),
        lease_root=(root / "formal-leases").resolve(),
        session_id=FORMAL_SESSION_ID,
        expected_source_commit=SOURCE_COMMIT,
        machine_id="fixture-machine",
        gpu0_uuid=GPU_UUIDS[0],
        gpu1_uuid=GPU_UUIDS[1],
        provider_env_file=root / "provider.env",
        max_allocated_active_gpu_hours=50.5,
        max_simulated_control_steps=6_311_042_448,
        max_output_bytes=14_495_514_624,
        nvidia_smi_executable=str(root / _nvidia_smi_name()),
        resume_existing=(root / "formal-output").exists(),
        session_mode=generation_session.FORMAL_GENERATION_SESSION_MODE,
        pause_after_generation=pause_after_generation,
    )


def _nvidia_smi_name() -> str:
    return "fake-nvidia-smi.cmd" if os.name == "nt" else "fake-nvidia-smi"


def _foundation(root: Path, *, pause_after_generation: int | None, crash: bool):
    config = _config(root, pause_after_generation=pause_after_generation)
    with use_test_isaac_lab_execution_environment():
        foundation = generation_session._create_foundation(
            config,
            source_commit=SOURCE_COMMIT,
            inventory=GPU_UUIDS,
        )
    if foundation.runner.attempt_admission_hook is None:
        raise AssertionError(
            "P6-C fixture requires the task-owned attempt admission hook"
        )
    runner = _AttemptBackedStairRunner(
        foundation=foundation,
        crash_after_gen2_public_dev=crash,
    )
    return replace(foundation, runner=runner)


def _formal_foundation(
    root: Path,
    *,
    pause_after_generation: int | None,
    crash_after_final_genesis: bool = False,
):
    config = _formal_config(
        root,
        pause_after_generation=pause_after_generation,
    )
    generation_session._publish_or_verify_source_checkout_identity(
        config.output_dir,
        source_commit=SOURCE_COMMIT,
    )
    with use_test_isaac_lab_execution_environment():
        foundation = generation_session._create_foundation(
            config,
            source_commit=SOURCE_COMMIT,
            inventory=GPU_UUIDS,
        )
    if foundation.runner.attempt_admission_hook is None:
        raise AssertionError(
            "P6-D fixture requires the task-owned attempt admission hook"
        )
    return replace(
        foundation,
        runner=_AttemptBackedStairRunner(
            foundation=foundation,
            crash_after_gen2_public_dev=False,
            crash_after_final_genesis=crash_after_final_genesis,
        ),
    )


def _prepare_or_load_genesis(
    foundation,
    *,
    require_existing: bool,
) -> VerifiedSessionGenesis:
    composition = foundation.composition
    snapshot = composition.genesis_identity_snapshot()
    identity_hash = session_genesis_identity_hash(snapshot)
    train_scope = generation_session._session_genesis_attempt_scope(
        journal_root=foundation.config.output_dir / "job_attempts",
        session_id=foundation.config.session_id,
        genesis_identity_hash=identity_hash,
        session_contract_hash=foundation.session_contract.contract_hash,
        lane_id="lane-a",
        invocation_segment_id="session-genesis",
        budget_contract_hash=foundation.cost_contract.contract_hash,
        resource_binding_hash=foundation.gpu_bindings[0].resource_binding_hash,
        max_infrastructure_retries=(
            generation_session.STAIR_GENERATION_MAX_INFRASTRUCTURE_RETRIES_PER_ELIGIBLE_JOB
        ),
    )
    reference_evaluation_kwargs = (
        generation_session._session_genesis_reference_evaluation_kwargs(
            foundation,
            genesis_identity_hash=identity_hash,
        )
    )
    genesis = prepare_session_genesis(
        output_dir=foundation.config.output_dir,
        session_id=foundation.config.session_id,
        task_plugin=composition.task_plugin,
        backend_adapter=composition.backend_adapter,
        runner=foundation.runner,
        policy_initialization=composition.policy_initialization,
        genesis_identity_snapshot=snapshot,
        baseline_manifest=composition.baseline_manifest,
        reference_program_artifact=composition.reference_program_artifact,
        bootstrap_initial_best_score=composition.bootstrap_initial_best_score,
        bootstrap_initial_policy_state_hash=(
            composition.bootstrap_initial_policy_state_hash
        ),
        bind_backend_execution_environment=True,
        generation_session_contract_ref=foundation.session_contract_ref,
        gpu_process_binding=foundation.gpu_bindings[0],
        reference_train_attempt_scope=train_scope,
        **reference_evaluation_kwargs,
        require_existing=require_existing,
    )
    generation_session._bind_unified_objective_family_mask(
        foundation,
        genesis=genesis,
    )
    return genesis


class _FixtureProvider:
    reference_source: str = ""

    def __init__(self, config, **_kwargs: Any) -> None:
        self.output_dir = Path(config.output_dir)
        parts = self.output_dir.parts
        self.lane_id = parts[-1]
        match = re.search(r"generation-(\d{3})", parts[-2])
        if match is None:
            raise ValueError("fixture provider path has no generation identity")
        self.generation_index = int(match.group(1))

    def next_authoring_response(
        self,
        *,
        prompt_payload: dict[str, Any],
    ) -> CandidateDraftResponse:
        if self.generation_index == 3:
            raise ProviderCandidateError(
                "json_parse_error",
                "fixture intentionally leaves generation 3 authoring incomplete",
                provider_usage=_provider_usage(),
            )
        if not self.reference_source:
            raise RuntimeError("fixture provider reference source is unavailable")
        key = (self.generation_index, self.lane_id)
        term_name, expression, weight = _CANDIDATE_TERMS[key]
        candidate_id = f"fixture-g{self.generation_index}-{self.lane_id}"
        candidate_dir = self.output_dir / candidate_id
        if candidate_dir.exists():
            raise AssertionError("provider candidate operation was repeated")
        _write_candidate(
            candidate_dir,
            candidate_id=candidate_id,
            source=_candidate_source(
                self.reference_source,
                candidate_id=candidate_id,
                term_name=term_name,
                expression=expression,
                weight=weight,
            ),
        )
        if not isinstance(prompt_payload, dict) or not prompt_payload:
            raise AssertionError("fixture provider did not receive the real prompt")
        return CandidateDraftResponse(
            candidate=CandidatePackageRef(
                candidate_id=candidate_id,
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={},
            ),
            provider_usage=_provider_usage(),
        )


class _FormalFixtureProvider(_FixtureProvider):
    """Accept one real pair, then close the remaining formal generations cheaply."""

    def next_authoring_response(
        self,
        *,
        prompt_payload: dict[str, Any],
    ) -> CandidateDraftResponse:
        if self.generation_index == 1:
            return super().next_authoring_response(prompt_payload=prompt_payload)
        if not isinstance(prompt_payload, dict) or not prompt_payload:
            raise AssertionError(
                "formal fixture provider did not receive the real prompt"
            )
        raise ProviderCandidateError(
            "runtime_error",
            "fixture intentionally exercises a zero-train formal generation",
            details={"retryable": False},
            provider_usage=_provider_usage(),
        )


def _formal_wrapper_component_load_check(candidate, smoke_adapter, **_kwargs: Any):
    """Keep finalization coverage portable when the task's Torch runtime is absent."""

    components = load_training_components(Path(candidate.path))
    return {
        "schema_version": "true_rl_agent.component_smoke_result.v1",
        "status": "skip",
        "reason": "formal_wrapper_fixture_torch_runtime_absent",
        "task_id": smoke_adapter.task_id,
        "checked_reward_terms": len(components.reward_terms),
        "checked_diagnostics": len(components.diagnostics),
    }


def _formal_wrapper_qualification_without_torch(
    foundation,
    *,
    artifact,
    static_verifier_policy,
    **_kwargs: Any,
):
    """Qualify sealed source in-process without isolated Torch execution."""

    adapter = foundation.composition.task_plugin.component_smoke_adapter()
    program = qualify_stair_candidate_reward_program(
        artifact=artifact,
        contract=adapter.contract,
        static_verifier_policy=static_verifier_policy,
        trusted_reference_artifact=(foundation.composition.reference_program_artifact),
    )
    return {
        "schema_version": (
            "true_rl_agent.task.stair_climbing.candidate_qualification.v1"
        ),
        "status": "pass",
        "reason": "stair_candidate_reward_program_qualified",
        "qualification": program.to_public_dict(),
        "determinism": {"status": "formal_wrapper_fixture_torch_runtime_absent"},
        "environment_axis": {
            "schema_version": (STAIR_REWARD_ENVIRONMENT_AXIS_EVIDENCE_SCHEMA_VERSION),
            "status": "pass",
            "reverse_permutation_checks": len(program.term_names),
            "single_row_locality_checks": len(program.term_names),
            "term_names": list(program.term_names),
        },
        "smoke_fixture_hash": adapter.fixture_hash,
    }


def _provider_usage() -> dict[str, int]:
    return {
        "request_units": 100,
        "response_units": 50,
        "reasoning_units": 10,
        "total_units": 150,
    }


def _candidate_source(
    reference_source: str,
    *,
    candidate_id: str,
    term_name: str,
    expression: str,
    weight: float,
) -> str:
    function_marker = "\ndef build_training_components():"
    term_marker = "        ],\n        rationale=("
    if function_marker not in reference_source or term_marker not in reference_source:
        raise ValueError("fixture reference source shape changed")
    source = reference_source.replace(
        function_marker,
        f"\n\ndef {term_name}(inputs):\n    {expression}\n{function_marker}",
        1,
    )
    term = (
        "            RewardTerm(\n"
        f'                name="{term_name}",\n'
        f"                weight={weight!r},\n"
        f"                fn={term_name},\n"
        "            ),\n"
    )
    source = source.replace(term_marker, term + term_marker, 1)
    return re.sub(
        r'(return TrainingComponents\(\n\s+name=)["\'][^"\']+["\']',
        rf'\1"{candidate_id}-complete-reward"',
        source,
        count=1,
    )


def _write_candidate(candidate_dir: Path, *, candidate_id: str, source: str) -> None:
    training_dir = candidate_dir / "training_code"
    training_dir.mkdir(parents=True, exist_ok=False)
    (candidate_dir / "candidate_manifest.yaml").write_text(
        "\n".join(
            (
                "schema_version: true_rl_agent.candidate.v1",
                f"candidate_id: {candidate_id}",
                "entrypoint: training_code.components:build_training_components",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (training_dir / "components.py").write_text(source, encoding="utf-8")
    (candidate_dir / "rationale.md").write_text(
        "Add one bounded Stair reward mechanism for the integration fixture.\n",
        encoding="utf-8",
    )
    _write_json(
        candidate_dir / "operation_diff.json",
        {
            "schema_version": "true_rl_agent.provider.operation_diff.v1",
            "candidate_id": candidate_id,
            "declared_changed_components": ["reward_callable"],
            "derived_from": ["declared_changed_components"],
            "materialized_by": "openai_compatible_candidate_materializer",
            "provider_backend": "openai_compatible_chat",
            "verifier_derived_change_set_is_authoritative": True,
        },
    )
    _write_json(
        candidate_dir / "diagnostics.json",
        {
            "schema_version": "true_rl_agent.provider.diagnostics.v1",
            "candidate_id": candidate_id,
            "diagnostic_items": [],
            "materialized_by": "openai_compatible_candidate_materializer",
            "notes": [
                "Provider JSON v2 does not accept LLM-authored diagnostics.",
                "Task runtime diagnostics use prompt-visible capabilities only.",
            ],
            "provider_backend": "openai_compatible_chat",
            "status": "provider_materialized",
        },
    )


class _AttemptBackedStairRunner(Runner):
    def __init__(
        self,
        *,
        foundation,
        crash_after_gen2_public_dev: bool,
        crash_after_final_genesis: bool = False,
    ) -> None:
        self.foundation = foundation
        self.output_root = foundation.config.output_dir / "fixture_runner_outputs"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.delegate = SubprocessRunner(
            ("fixture-only",),
            timeout_s=30,
            gpu_lease_manager=foundation.gpu_lease_manager,
            attempt_admission_hook=foundation.runner.attempt_admission_hook,
        )
        self.crash_after_gen2_public_dev = crash_after_gen2_public_dev
        self.crash_after_final_genesis = crash_after_final_genesis
        self.public_dev_barrier = (
            threading.Barrier(2, timeout=20) if crash_after_gen2_public_dev else None
        )

    def run_train(self, train_spec):
        result = self.delegate._execute_with_attempts(
            train_spec,
            job_kind="train",
            operation=lambda _logs, context, _timeout_s: self._run_train_once(
                train_spec,
                context=context,
            ),
            recover_completed=lambda _context: _train_result_from_private_mapping(
                self._load_output("train", train_spec.job_id)
            ),
        )
        return result

    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        return self.delegate._execute_with_attempts(
            eval_spec,
            job_kind="eval",
            operation=lambda _logs, context, _timeout_s: self._run_full_eval_once(
                eval_spec,
                context=context,
            ),
            recover_completed=lambda _context: _public_result_from_mapping(
                self._load_output("eval", eval_spec.job_id)
            ),
        )

    def run_eval_phase(self, eval_spec: EvalPhaseJobSpec) -> EvaluationPhaseResult:
        raw_argv = eval_spec.metadata.get("cli_argv")
        if not isinstance(raw_argv, list):
            raise AssertionError("fixture eval phase has no structured argv")
        validate_eval_job_argv(eval_spec, raw_argv)
        result = self.delegate._execute_with_attempts(
            eval_spec,
            job_kind="eval_phase",
            operation=lambda _logs, context, _timeout_s: self._run_phase_once(
                eval_spec,
                context=context,
            ),
            recover_completed=lambda _context: (
                EvaluationPhaseResult.from_private_mapping(
                    self._load_output("eval_phase", eval_spec.job_id)
                )
            ),
        )
        if (
            self.crash_after_gen2_public_dev
            and eval_spec.phase is EvaluationPhase.PUBLIC_DEV
            and f"{SESSION_ID}-generation-002-" in eval_spec.job_id
        ):
            assert self.public_dev_barrier is not None
            self.public_dev_barrier.wait()
            if "lane-a" in eval_spec.job_id:
                _write_json(
                    self.foundation.config.output_dir / "fixture_hard_crash.json",
                    {"job_id": eval_spec.job_id, "after_attempt_result": True},
                )
                os._exit(HARD_CRASH_EXIT_CODE)
        if (
            self.crash_after_final_genesis
            and eval_spec.job_id
            == f"{FORMAL_SESSION_ID}-final-confirmation-genesis-eval-sealed-validation"
        ):
            _write_json(
                self.foundation.config.output_dir
                / "fixture_finalization_hard_crash.json",
                {"job_id": eval_spec.job_id, "after_attempt_result": True},
            )
            os._exit(HARD_CRASH_EXIT_CODE)
        return result

    def project_assembled_eval_summary(self, eval_spec, assembly):
        return self.delegate.project_assembled_eval_summary(eval_spec, assembly)

    def _run_train_once(self, train_spec, *, context) -> TrainJobResult:
        binding, physical_uuid = self._binding(context)
        self._claim_actual("train", train_spec.job_id)
        checkpoint_path = Path(train_spec.metadata["checkpoint_output_path"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_bytes = (train_spec.job_id + "\n").encode("utf-8")
        create_once_bytes(checkpoint_path, checkpoint_bytes)
        checksum = "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest()
        pairing = (
            InitialPolicyStatePairing(
                expected_initial_policy_state_hash=(
                    train_spec.expected_initial_policy_state_hash
                ),
                actual_initial_policy_state_hash=(
                    train_spec.expected_initial_policy_state_hash
                ),
                match=True,
            )
            if train_spec.expected_initial_policy_state_hash is not None
            else None
        )
        summary = self._train_summary(
            train_spec,
            binding=binding,
            physical_uuid=physical_uuid,
        )
        self._publish_summary(train_spec.metadata, summary)
        checkpoint = seal_checkpoint(
            CheckpointRef(
                uri=str(checkpoint_path),
                checksum=checksum,
                role="train_output",
                metadata={
                    "local_path": str(checkpoint_path.resolve()),
                    "training_subject": train_spec.subject.to_public_dict(),
                    "subject_manifest_hash": train_spec.subject_manifest_hash,
                    "program_source_checksum": train_spec.program_source_checksum,
                    "training_binding_hash": train_spec.training_binding.binding_hash,
                    "capability_profile_hash": (
                        train_spec.training_binding.capability_profile_hash
                    ),
                    "execution_device": binding.process_execution_device,
                    "gpu_process_binding": binding.to_dict(),
                    **(
                        {"initial_policy_state_pairing": pairing.to_public_dict()}
                        if pairing is not None
                        else {}
                    ),
                },
            ),
            sealed_uri=str(train_spec.metadata["sealed_checkpoint_uri"]),
        )
        result = TrainJobResult(
            job_id=train_spec.job_id,
            subject=train_spec.subject,
            subject_manifest_hash=train_spec.subject_manifest_hash,
            program_source_checksum=train_spec.program_source_checksum,
            training_binding_hash=train_spec.training_binding.binding_hash,
            capability_profile_hash=train_spec.training_binding.capability_profile_hash,
            execution_device=binding.process_execution_device,
            checkpoint=checkpoint,
            gpu_process_binding=binding,
            initial_policy_state_pairing=pairing,
            summary=summary,
        )
        self._publish_output(
            "train",
            train_spec.job_id,
            _train_result_to_private_mapping(result),
        )
        return result

    def _run_full_eval_once(
        self, eval_spec: EvalJobSpec, *, context
    ) -> PublicEvalResult:
        binding, physical_uuid = self._binding(context)
        self._claim_actual("eval", eval_spec.job_id)
        execution = self._execution(
            eval_spec=eval_spec,
            split_successes={phase: 2 for phase in EvaluationPhase},
        )
        summary = build_stair_full_eval_summary(
            config=self.foundation.composition.task_plugin.config,
            identity=eval_spec.job_identity(),
            evaluation_binding=eval_spec.evaluation_binding,
            execution=execution,
            checkpoint_safety_report=_checkpoint_report(
                eval_spec.execution_checkpoint().checksum
            ),
            subject_manifest_hash=eval_spec.subject_manifest_hash,
            program_source_checksum=eval_spec.program_source_checksum,
            training_binding_hash=eval_spec.training_binding_hash,
            capability_profile_hash=eval_spec.capability_profile_hash,
        )
        summary.update(_gpu_summary_fields(eval_spec.job_id, binding, physical_uuid))
        self._publish_summary(eval_spec.metadata, summary)
        result = self.delegate._public_result_from_summary(
            summary,
            job_id=eval_spec.job_id,
            expected_checkpoint_hash=eval_spec.execution_checkpoint().checksum,
            expected_eval_protocol_hash=eval_spec.evaluation_protocol().protocol_hash,
            expected_evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
            expected_sealed_seed_count=eval_spec.sealed_validation_seed_count,
            evaluation_binding=eval_spec.evaluation_binding,
            evaluation_subject=eval_spec.subject.to_public_dict(),
            subject_manifest_hash=eval_spec.subject_manifest_hash,
            program_source_checksum=eval_spec.program_source_checksum,
            training_binding_hash=eval_spec.training_binding_hash,
            capability_profile_hash=eval_spec.capability_profile_hash,
            expected_execution_device=binding.process_execution_device,
        )
        self._publish_output("eval", eval_spec.job_id, result.to_public_dict())
        return result

    def _run_phase_once(
        self,
        eval_spec: EvalPhaseJobSpec,
        *,
        context,
    ) -> EvaluationPhaseResult:
        binding, physical_uuid = self._binding(context)
        self._claim_actual("eval_phase", eval_spec.job_id)
        success_count = _success_count(eval_spec)
        execution = self._execution(
            eval_spec=eval_spec.evaluation_plan,
            split_successes={eval_spec.phase: success_count},
        )
        summary = build_stair_eval_phase_summary(
            config=self.foundation.composition.task_plugin.config,
            identity=eval_spec.job_identity(),
            evaluation_plan_hash=eval_spec.evaluation_plan_hash,
            phase=eval_spec.phase,
            execution=execution,
            evaluation_binding=eval_spec.evaluation_binding,
            checkpoint_safety_report=_checkpoint_report(
                eval_spec.execution_checkpoint().checksum
            ),
            expected_seed_namespace=execution.seed_namespace,
        )
        summary.update(_gpu_summary_fields(eval_spec.job_id, binding, physical_uuid))
        self._publish_summary(eval_spec.metadata, summary)
        result = self.delegate._phase_result_from_summary(
            summary,
            eval_spec=eval_spec,
            expected_checkpoint_hash=eval_spec.execution_checkpoint().checksum,
            actual_gpu_process_binding=binding,
        )
        self._publish_output(
            "eval_phase",
            eval_spec.job_id,
            result.to_private_dict(),
        )
        return result

    def _train_summary(self, train_spec, *, binding, physical_uuid: str) -> dict:
        config = self.foundation.composition.task_plugin.config
        initial_policy_state_hash = (
            train_spec.expected_initial_policy_state_hash
            or canonical_json_hash(
                {
                    "fixture": "reference-initial-policy-state",
                    "session_id": self.foundation.config.session_id,
                }
            )
        )
        if train_spec.candidate_artifact is not None:
            components = load_training_components(
                Path(train_spec.candidate_artifact.path)
            )
            term_names = [term.name for term in components.reward_terms]
        elif self.foundation.config.session_id == FORMAL_SESSION_ID:
            source_path = (
                Path(train_spec.reference_artifact.path)
                / CANONICAL_TRAINING_COMPONENTS_PATH
            )
            term_names = re.findall(
                r'RewardTerm\(\s*name=["\']([^"\']+)["\']',
                source_path.read_text(encoding="utf-8"),
            )
            if not term_names:
                raise AssertionError("formal fixture reference reward terms are absent")
        else:
            source_path = (
                Path(train_spec.reference_artifact.path)
                / CANONICAL_TRAINING_COMPONENTS_PATH
            )
            components = load_training_components_source_bytes(
                source_path.read_bytes(),
                module_name="true_rl_fixture_reference_program",
                source_name=str(source_path),
            )
            term_names = [term.name for term in components.reward_terms]
        calls = config.train_steps_per_iteration * config.train_max_iterations + 1
        qualification_hash = canonical_json_hash(
            {
                "schema_version": STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION,
                "program_source_checksum": train_spec.program_source_checksum,
            }
        )
        source_preflight = _source_preflight(config)
        reward_contract = (
            self.foundation.composition.task_plugin.component_smoke_adapter().contract
        )
        summary = {
            "schema_version": "true_rl_agent.fixture.train_summary.v1",
            **train_spec.job_identity().to_public_dict(),
            "status": "pass",
            "errors": [],
            "execution_device": binding.process_execution_device,
            "environment_source_preflight": source_preflight,
            "program_source_checksum": train_spec.program_source_checksum,
            "subject_manifest_hash": train_spec.subject_manifest_hash,
            "subject_kind": train_spec.subject.kind,
            "subject_id": train_spec.subject.subject_id,
            "reward_program_qualification_hash": qualification_hash,
            "formal_training_executed": True,
            "training": {
                "initial_policy_state_hash": initial_policy_state_hash,
                "budget_accounting": build_budget_accounting(
                    budget_steps=config.budget_steps,
                    num_envs=config.train_num_envs,
                    num_steps_per_env=config.train_steps_per_iteration,
                    max_iterations=config.train_max_iterations,
                ),
            },
            "runtime_activation_summary": {
                "runtime_status": config.train_runtime_profile().to_public_dict()[
                    "complete_reward_runtime"
                ],
                "status": "pass",
                "dispatcher_calls": calls,
                "term_calls": calls * len(term_names),
                "declared_term_count": len(term_names),
                "activation_preflight": {
                    "schema_version": STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION,
                    "status": "pass",
                    "real_transition_dispatcher_calls": 1,
                    "transition_common_step": 1,
                    "repeat_count": (
                        reward_contract.dispatcher_contract[
                            "determinism_preflight_repeats"
                        ]
                    ),
                    "term_names": term_names,
                },
                "input_guard": "pass",
                "rng_guard": "pass",
                "source_alias_guard": "pass",
                "reward_manager_active_terms": ["complete_reward"],
                "reward_manager_term_weight": 1.0,
                "training_recorder_enabled": False,
                "reward_term_calls": calls * len(term_names),
                "diagnostic_calls": 0,
                "nonzero_reward_contribution": True,
                "component_activation_records": [
                    {
                        "component_kind": "reward_term",
                        "name": name,
                        "max_abs_weighted_contribution": 1.0,
                    }
                    for name in term_names
                ],
                "capability_activation_records": [
                    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID
                ],
                "program": {
                    "subject": train_spec.subject.to_public_dict(),
                    "program_source_checksum": train_spec.program_source_checksum,
                    "subject_manifest_hash": train_spec.subject_manifest_hash,
                    "qualification_hash": qualification_hash,
                    "input_schema_hash": reward_contract.input_schema_hash,
                    "term_names": term_names,
                },
            },
            **_gpu_summary_fields(train_spec.job_id, binding, physical_uuid),
        }
        return summary

    def _execution(
        self, *, eval_spec, split_successes
    ) -> StairEvaluationExecutionResult:
        config = self.foundation.composition.task_plugin.config
        final_cohort = config.seed_resolver().namespace_cohort_hash(
            STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
        )
        seed_namespace = (
            STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
            if eval_spec.evaluation_binding.seed_cohort_hash == final_cohort
            else STAIR_FORMAL_EVAL_NAMESPACE
        )
        summaries = {}
        outcomes = {}
        steps = {}
        walls = {}
        for phase, success_count in split_successes.items():
            per_seed = tuple(
                tuple(
                    _outcomes(
                        config,
                        phase=phase,
                        successes_per_cell=success_count,
                    )
                )
                for _ in range(_seed_count(config, phase))
            )
            summaries[phase.value] = (
                build_unified_stair_summary(
                    suite=config.ordered_evaluation_suite,
                    semantics=config.evaluation_semantics,
                    split=phase.value,
                    per_seed_outcomes=per_seed,
                )
                if config.is_unified_protocol
                else build_stair_evaluation_summary(
                    config.protocol_v1,
                    semantics=config.evaluation_semantics,
                    evaluation_protocol=config.evaluation_protocol(),
                    split=phase.value,
                    per_seed_outcomes=per_seed,
                    evaluation_profile=config.evaluation_profile,
                )
            )
            outcomes[phase.value] = per_seed
            steps[phase.value] = 100
            walls[phase.value] = 0.01
        return StairEvaluationExecutionResult(
            summaries=summaries,
            per_seed_outcomes=outcomes,
            split_vector_env_step_calls=steps,
            split_wall_times_s=walls,
            checkpoint_hash=eval_spec.execution_checkpoint().checksum,
            seed_namespace=seed_namespace,
            seed_cohort_hash=config.seed_resolver().namespace_cohort_hash(
                seed_namespace
            ),
            source_manifest_preflight=_source_preflight(config),
            backend_execution_environment_verification=(
                TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
            ),
            execution_device=eval_spec.execution_device,
        )

    def _binding(self, context):
        if context is None:
            raise AssertionError("bound fixture job has no GPU execution context")
        return context.gpu_process_binding, context.physical_id

    def _claim_actual(self, kind: str, job_id: str) -> None:
        path = self._path("actual", kind, job_id)
        if path.exists():
            raise AssertionError(f"runner operation repeated: {kind}:{job_id}")
        _write_json(path, {"kind": kind, "job_id": job_id})

    def _publish_summary(self, metadata: Mapping[str, Any], summary: Mapping) -> None:
        path = Path(str(metadata["summary_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        create_once_bytes(path, canonical_json_bytes(summary, trailing_newline=True))

    def _publish_output(self, kind: str, job_id: str, payload: Mapping) -> None:
        _write_json(self._path("result", kind, job_id), payload)

    def _load_output(self, kind: str, job_id: str) -> dict[str, Any]:
        return load_strict_json_object(
            self._path("result", kind, job_id),
            context="fixture runner typed output",
        )

    def _path(self, prefix: str, kind: str, job_id: str) -> Path:
        digest = canonical_json_hash({"kind": kind, "job_id": job_id})
        return self.output_root / f"{prefix}-{digest}.json"


def _source_preflight(config) -> dict[str, str]:
    return preflight_stair_execution_source_manifest(config).to_summary_dict()


def _gpu_summary_fields(job_id: str, binding, physical_uuid: str) -> dict[str, Any]:
    process_id = 10_000 + int(canonical_json_hash({"job_id": job_id})[:6], 16)
    observation = {
        "schema_version": STAIR_GPU_COMPUTE_CONTEXT_OBSERVATION_SCHEMA_VERSION,
        "source": "nvidia-smi-query-compute-apps",
        "monotonic_ns": process_id,
        "process_id": process_id,
        "rows": [{"gpu_uuid": physical_uuid, "used_gpu_memory_mib": 1}],
    }
    return {
        "gpu_process_binding": binding.to_dict(),
        "owned_process_identity": {
            "schema_version": OWNED_PROCESS_IDENTITY_SCHEMA_VERSION,
            "process_id": process_id,
            "process_group_id": process_id,
        },
        "stair_gpu_runtime_evidence": {
            "schema_version": STAIR_GPU_RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "cuda_visibility_binding": "single_physical_gpu_uuid",
            "physical_gpu_uuid": physical_uuid,
            "renderer_active_gpu_ordinal": int(
                binding.host_execution_device.removeprefix("cuda:")
            ),
            "renderer_multi_gpu": False,
            "renderer_multi_gpu_auto_enable": False,
            "renderer_multi_gpu_max_gpu_count": 1,
            "nvidia_visible_devices_unset": True,
            "external_compute_context_observation": observation,
        },
    }


def _outcomes(
    config,
    *,
    phase: EvaluationPhase,
    successes_per_cell: int,
) -> list[StairEpisodeOutcome]:
    outcomes = []
    if config.is_unified_protocol:
        suite = config.ordered_evaluation_suite
        episodes_per_cell = suite.episodes_per_seed_per_cell
        cells = suite.private_cell_assignment(phase.value)
    else:
        episodes_per_cell = config.evaluation_profile.episodes_per_seed_per_cell
        cells = config.evaluation_profile.private_cell_assignment()
    for env_index, cell in enumerate(cells):
        success = env_index % episodes_per_cell < successes_per_cell
        outcomes.append(
            StairEpisodeOutcome(
                env_index=env_index,
                bin_id=cell.bin_id,
                episode_steps=cell.episode_horizon_control_steps,
                primary_reason="success" if success else "timeout",
                success=success,
                diagnostic_flags=(
                    ("first_valid_top_arrival_achieved",)
                    if success
                    else ("goal_not_reached_detected",)
                ),
                max_progress_m=cell.progress_goal_m if success else 0.5,
                progress_fraction=1.0 if success else 0.1,
                active_terminal_terms=() if success else ("time_out",),
                evaluation_cell_id=cell.cell_id,
            )
        )
    return outcomes


def _seed_count(config, phase: EvaluationPhase) -> int:
    return {
        EvaluationPhase.PUBLIC_DEV: config.public_seed_count,
        EvaluationPhase.PUBLIC_ACCEPT: config.public_accept_seed_count,
        EvaluationPhase.SEALED_VALIDATION: config.sealed_validation_seed_count,
    }[phase]


def _success_count(eval_spec: EvalPhaseJobSpec) -> int:
    # The unified suite has three episodes per cell.  Keep generation 2 lane A
    # ahead on public_dev, then below the accepted parent on formal evaluation.
    job_id = eval_spec.job_id
    if f"{FORMAL_SESSION_ID}-generation-001-lane-a" in job_id:
        return 3
    if f"{FORMAL_SESSION_ID}-generation-001-lane-b" in job_id:
        return 2
    if f"{FORMAL_SESSION_ID}-final-confirmation-final-eval" in job_id:
        return 3
    if f"{SESSION_ID}-generation-001-lane-a" in job_id:
        return 3
    if f"{SESSION_ID}-generation-001-lane-b" in job_id:
        return 2
    if f"{SESSION_ID}-generation-002-lane-a" in job_id:
        return 3 if eval_spec.phase is EvaluationPhase.PUBLIC_DEV else 2
    if f"{SESSION_ID}-generation-002-lane-b" in job_id:
        return 2
    return 2


def _checkpoint_report(checksum: str) -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.checkpoint_safety_report.v1",
        "status": "pass",
        "checkpoint_hash": checksum,
        "expected_checkpoint_hash": checksum,
        "file_exists": True,
        "byte_read_status": "pass",
        "weights_only_load_status": "pass",
        "checkpoint_hash_match_status": "pass",
        "loader_error_class": None,
        "loader_error_summary": None,
        "reason": "checkpoint_safe_to_load",
    }


def _public_result_from_mapping(value: Mapping[str, Any]) -> PublicEvalResult:
    return PublicEvalResult(
        status=value["status"],
        public_score=value["public_score"],
        metrics=value["metrics"],
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    create_once_bytes(path, canonical_json_bytes(payload, trailing_newline=True))


def _startup_observation() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.stair.generation_compute_idle.v1",
        "stage": "startup",
        "source": "nvidia-smi-query-compute-apps",
        "target_gpu_uuids": list(GPU_UUIDS),
        "rows": [],
    }


def _run_worker(root: Path, action: str) -> None:
    pause = {"pause1": 1, "crash2": 2, "pause2": 2}.get(action)
    foundation = _foundation(
        root, pause_after_generation=pause, crash=action == "crash2"
    )
    _FixtureProvider.reference_source = (
        foundation.composition.reference_program_artifact.source_bytes.decode("utf-8")
    )
    if action == "setup":
        journal = generation_session.StairGenerationActualCostJournal(
            foundation.config.output_dir,
            contract=foundation.cost_contract,
        )
        journal.publish_reservation()
        genesis = _prepare_or_load_genesis(foundation, require_existing=False)
        usage, charges, _ = generation_session._collect_actual_cost_usage(
            foundation,
            genesis=genesis,
            ledger=[],
        )
        journal.publish_snapshot(
            event_id="post-genesis",
            previous_cost_chain_head_hash=generation_session._latest_cost_head(journal),
            usage=usage,
            conservative_charges=charges,
            finalization_output_reserve_bytes=0,
        )
        _write_json(root / "worker-setup.json", {"status": "complete"})
        return

    genesis = _prepare_or_load_genesis(foundation, require_existing=True)
    journal = generation_session.StairGenerationActualCostJournal(
        foundation.config.output_dir,
        contract=foundation.cost_contract,
    )
    if action == "finalize":
        result = generation_session._finalize_missing_terminal_validation_report(
            foundation,
            genesis=genesis,
            cost_journal=journal,
        )
        _write_json(root / "worker-finalize.json", result)
        return
    with patch.object(
        generation_session,
        "OpenAICompatibleCandidateProvider",
        _FixtureProvider,
    ):
        result = generation_session._run_validation_session(
            foundation,
            genesis=genesis,
            cost_journal=journal,
            startup_resource_observation=_startup_observation(),
        )
    _write_json(root / f"worker-{action}.json", result)


def _run_formal_worker(root: Path, action: str) -> None:
    pause = {
        "pause3": 3,
        "pause6": 6,
        "pause9": 9,
        "pause12": 12,
    }.get(action)
    foundation = _formal_foundation(
        root,
        pause_after_generation=pause,
        crash_after_final_genesis=action == "final-crash",
    )
    _FormalFixtureProvider.reference_source = (
        foundation.composition.reference_program_artifact.source_bytes.decode("utf-8")
    )
    if action == "setup":
        journal = generation_session.StairGenerationActualCostJournal(
            foundation.config.output_dir,
            contract=foundation.cost_contract,
        )
        journal.publish_reservation()
        genesis = _prepare_or_load_genesis(foundation, require_existing=False)
        usage, charges, _ = generation_session._collect_actual_cost_usage(
            foundation,
            genesis=genesis,
            ledger=[],
        )
        journal.publish_snapshot(
            event_id="post-genesis",
            previous_cost_chain_head_hash=generation_session._latest_cost_head(journal),
            usage=usage,
            conservative_charges=charges,
            finalization_output_reserve_bytes=0,
        )
        _write_json(root / "formal-worker-setup.json", {"status": "complete"})
        return

    genesis = _prepare_or_load_genesis(foundation, require_existing=True)
    journal = generation_session.StairGenerationActualCostJournal(
        foundation.config.output_dir,
        contract=foundation.cost_contract,
    )
    if action == "audit":
        result = generation_session._load_and_validate_existing_formal_report(
            foundation,
            genesis=genesis,
            cost_journal=journal,
        )
    else:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    generation_session,
                    "OpenAICompatibleCandidateProvider",
                    _FormalFixtureProvider,
                )
            )
            if not _torch_runtime_importable():
                # This test owns formal lifecycle/finalization coverage. Stair's
                # real callable/determinism checks remain covered in Torch-gated
                # tests; sealed source qualification still runs here.
                stack.enter_context(
                    patch.object(
                        candidate_authoring_loop,
                        "run_component_smoke_check",
                        _formal_wrapper_component_load_check,
                    )
                )
                stack.enter_context(
                    patch.object(
                        stair_candidate_admission,
                        "_run_qualification_worker",
                        side_effect=lambda **kwargs: (
                            _formal_wrapper_qualification_without_torch(
                                foundation,
                                **kwargs,
                            )
                        ),
                    )
                )
            result = generation_session._run_formal_session(
                foundation,
                genesis=genesis,
                cost_journal=journal,
                startup_resource_observation=_startup_observation(),
            )
        if action == "finish":
            report_path = (
                foundation.config.output_dir
                / generation_session.STAIR_GENERATION_REPORT_FILENAME
            )
            create_once_bytes(
                report_path,
                canonical_json_bytes(result, trailing_newline=True),
            )
    _write_json(root / f"formal-worker-{action}.json", result)


class StairGenerationSessionProductionShapeTests(unittest.TestCase):
    @unittest.skipUnless(
        _torch_runtime_importable(),
        "production Stair candidate qualification requires Torch",
    )
    def test_three_generations_pause_resume_and_hard_crash_recovery(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "provider.env").write_text(
                "DEEPSEEK_API_KEY=fixture-only\n",
                encoding="utf-8",
            )
            _write_fake_nvidia_smi(root / _nvidia_smi_name())

            self._worker(root, "setup")
            self._worker(root, "pause1")
            crashed = self._worker(root, "crash2", expected=HARD_CRASH_EXIT_CODE)
            self.assertEqual(crashed.returncode, HARD_CRASH_EXIT_CODE)
            crash_evidence = load_strict_json_object(
                root / "output" / "fixture_hard_crash.json",
                context="fixture hard crash evidence",
            )
            target_result_path = _attempt_result_for_job(
                root / "output" / "job_attempts",
                crash_evidence["job_id"],
            )
            target_result_bytes = target_result_path.read_bytes()

            self._worker(root, "pause2")
            self.assertEqual(target_result_path.read_bytes(), target_result_bytes)
            self._worker(root, "finish")
            report_path = (
                root / "output" / generation_session.STAIR_GENERATION_REPORT_FILENAME
            )
            self.assertFalse(report_path.exists())
            self._worker(root, "finalize")
            self.assertTrue(report_path.is_file())

            output = root / "output"
            ledger = generation_session.load_ledger(output / "ledger.jsonl")
            self.assertEqual(len(ledger), 3)
            self.assertEqual(
                [record["generation_id"] for record in ledger],
                [
                    f"{SESSION_ID}-generation-001",
                    f"{SESSION_ID}-generation-002",
                    f"{SESSION_ID}-generation-003",
                ],
            )
            self.assertEqual(ledger[0]["generation_outcome"], "accepted")
            self.assertTrue(ledger[0]["formal"]["accepted"])
            self.assertEqual(ledger[1]["generation_outcome"], "formal_rejected")
            self.assertFalse(ledger[1]["formal"]["accepted"])
            self.assertIsNone(ledger[2]["formal"])
            self.assertIsNone(ledger[2]["selection"])
            self.assertEqual(
                ledger[2]["generation_outcome"],
                "paired_authoring_incomplete",
            )
            self.assertEqual(
                [trial["trial_status"] for trial in ledger[2]["trials"]],
                ["candidate_failed", "not_started"],
            )
            self.assertTrue(
                all(trial["train"] is None for trial in ledger[2]["trials"])
            )
            self.assertEqual(
                ledger[2]["parent_current_best"]["program_source_checksum"],
                ledger[1]["resulting_current_best"]["program_source_checksum"],
            )
            self.assertEqual(
                ledger[2]["resulting_current_best_hash"],
                ledger[2]["parent_current_best_hash"],
            )

            starts = [
                load_strict_json_object(path, context="fixture segment start")
                for path in sorted(
                    (output / "generation_segments").glob("segment-*/start.json")
                )
            ]
            self.assertEqual(len(starts), 4)
            exits = sorted((output / "generation_segments").glob("segment-*/exit.json"))
            recoveries = list(
                (output / "generation_recoveries").glob("recovery-*.json")
            )
            self.assertEqual(len(exits), 3)
            self.assertEqual(len(recoveries), 1)
            exit_payloads = [
                load_strict_json_object(path, context="fixture segment exit")
                for path in exits
            ]
            self.assertEqual(
                [(item["status"], item["reason"]) for item in exit_payloads],
                [
                    ("paused", "operator_pause"),
                    ("paused", "operator_pause"),
                    ("completed", "normal"),
                ],
            )

            scopes_by_job = _attempt_scopes(output / "job_attempts")
            gen2 = f"{SESSION_ID}-generation-002"
            crash_segment = starts[1]["segment_id"]
            resumed_segment = starts[2]["segment_id"]
            self.assertTrue(
                all(
                    scope.invocation_segment_id == crash_segment
                    for job_id, scope in scopes_by_job.items()
                    if job_id.startswith(gen2)
                    and job_id.endswith(("-train", "-public-dev"))
                )
            )
            self.assertTrue(
                all(
                    scope.invocation_segment_id == resumed_segment
                    for job_id, scope in scopes_by_job.items()
                    if job_id.startswith(gen2)
                    and job_id.endswith(("-public-accept", "-sealed-validation"))
                )
            )
            self.assertFalse(
                any(
                    job_id.startswith(f"{SESSION_ID}-generation-003")
                    for job_id in scopes_by_job
                )
            )
            self.assertEqual(
                list((output / "job_attempts").rglob("attempt_01.reservation.json")),
                [],
            )

            markers = [
                load_strict_json_object(path, context="fixture runner marker")
                for path in (output / "fixture_runner_outputs").glob("actual-*.json")
            ]
            self.assertEqual(len(markers), len({item["job_id"] for item in markers}))
            self.assertEqual(
                sum(
                    item["job_id"]
                    == f"{SESSION_ID}-parent-eval-public-dev"
                    for item in markers
                ),
                0,
            )
            self.assertEqual(
                generation_session._generation_provider_attempt_count(
                    output,
                    generation_id=f"{SESSION_ID}-generation-001",
                ),
                2,
            )
            self.assertEqual(
                generation_session._generation_provider_attempt_count(
                    output,
                    generation_id=f"{SESSION_ID}-generation-002",
                ),
                2,
            )
            self.assertEqual(
                generation_session._generation_provider_attempt_count(
                    output,
                    generation_id=f"{SESSION_ID}-generation-003",
                ),
                5,
            )
            self.assertEqual(
                len(list((output / "generation_closures").glob("*.json"))),
                3,
            )
            prompt_payloads = list(
                (output / "authoring").glob(
                    "**/authoring_attempts/*/prompt_payload.json"
                )
            )
            self.assertGreaterEqual(len(prompt_payloads), 4)
            for prompt_payload in prompt_payloads:
                self.assertNotIn(
                    b"sealed_validation_evidence",
                    prompt_payload.read_bytes(),
                )
            report = load_strict_json_object(
                output / generation_session.STAIR_GENERATION_REPORT_FILENAME,
                context="fixture validation report",
            )
            self.assertEqual(len(report["generations"]), 3)
            self.assertEqual(
                [item["generation_index"] for item in report["generations"]],
                [1, 2, 3],
            )
            self.assertFalse(report["final_confirmation"]["enabled"])
            cost_journal = generation_session.StairGenerationActualCostJournal(
                output,
                contract=(
                    generation_session.load_stair_generation_session_cost_contract(
                        output
                        / generation_session.STAIR_GENERATION_SESSION_COST_FILENAME
                    )
                ),
            )
            costs_by_event = {
                item.payload["event_id"]: item.payload["actual"]
                for item in cost_journal.load_chain()
            }
            post_genesis = costs_by_event["post-genesis"]
            post_parent = costs_by_event["post-parent-bootstrap"]
            generation_one = costs_by_event["generation-001-settlement"]
            self.assertEqual(
                report["session_metrics"]["parent_bootstrap_workload_delta"][
                    "eval_seed_units"
                ],
                post_parent["eval_seed_units"] - post_genesis["eval_seed_units"],
            )
            self.assertEqual(
                report["generations"][0]["actual_workload_delta"]["eval_seed_units"],
                generation_one["eval_seed_units"] - post_parent["eval_seed_units"],
            )
            self.assertEqual(
                report["session_metrics"]["parent_bootstrap_workload_delta"][
                    "eval_seed_units"
                ],
                0,
            )

    @unittest.skip(
        "Phase 3R keeps the unified Stair protocol draft; Phase 3B re-enables formal generation"
    )
    def test_formal_fifteen_generation_boundaries_and_finalization(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "provider.env").write_text(
                "DEEPSEEK_API_KEY=fixture-only\n",
                encoding="utf-8",
            )
            _write_fake_nvidia_smi(root / _nvidia_smi_name())

            for action in (
                "setup",
                "pause3",
                "pause6",
                "pause9",
                "pause12",
            ):
                self._formal_worker(root, action)
                if action.startswith("pause"):
                    subject = _inspect_v3_launch_subject_values(
                        formal_output_dir=root / "formal-output",
                        source_root=PROJECT_ROOT,
                        session_id=FORMAL_SESSION_ID,
                        expected_source_commit=SOURCE_COMMIT,
                        subject_mode=(STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME),
                    )
                    self.assertEqual(
                        subject["subject_mode"],
                        STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
                    )

            crashed = self._formal_worker(
                root,
                "final-crash",
                expected=HARD_CRASH_EXIT_CODE,
            )
            self.assertEqual(crashed.returncode, HARD_CRASH_EXIT_CODE)
            crash_evidence = load_strict_json_object(
                root / "formal-output" / "fixture_finalization_hard_crash.json",
                context="formal fixture finalization crash evidence",
            )
            final_genesis_attempt_result = _attempt_result_for_job(
                root / "formal-output" / "job_attempts",
                crash_evidence["job_id"],
            )
            final_genesis_attempt_bytes = final_genesis_attempt_result.read_bytes()
            finalization_subject = _inspect_v3_launch_subject_values(
                formal_output_dir=root / "formal-output",
                source_root=PROJECT_ROOT,
                session_id=FORMAL_SESSION_ID,
                expected_source_commit=SOURCE_COMMIT,
                subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
            )
            self.assertTrue(finalization_subject["lifecycle"]["terminal"])
            self.assertFalse(finalization_subject["formal_report"]["exists"])
            self._formal_worker(root, "finish")
            self.assertEqual(
                final_genesis_attempt_result.read_bytes(),
                final_genesis_attempt_bytes,
            )

            output = root / "formal-output"
            ledger_path = output / "ledger.jsonl"
            ledger_before_audit = ledger_path.read_bytes()
            report_path = output / generation_session.STAIR_GENERATION_REPORT_FILENAME
            report_before_audit = report_path.read_bytes()
            audit_subject = _inspect_v3_launch_subject_values(
                formal_output_dir=output,
                source_root=PROJECT_ROOT,
                session_id=FORMAL_SESSION_ID,
                expected_source_commit=SOURCE_COMMIT,
                subject_mode=STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
            )
            self.assertTrue(audit_subject["formal_report"]["exists"])
            self.assertTrue(
                audit_subject["final_confirmation"]["result"]["file"]["exists"]
            )
            self._formal_worker(root, "audit")
            self.assertEqual(ledger_path.read_bytes(), ledger_before_audit)
            self.assertEqual(report_path.read_bytes(), report_before_audit)

            ledger = generation_session.load_ledger(ledger_path)
            self.assertEqual(len(ledger), 15)
            audit_foundation = _formal_foundation(
                root,
                pause_after_generation=None,
            )
            audit_genesis = _prepare_or_load_genesis(
                audit_foundation,
                require_existing=True,
            )
            audit_plan = audit_foundation.final_confirmation_plan
            self.assertIsNotNone(audit_plan)
            stored_reservation = (
                generation_session.load_stair_generation_final_confirmation_reservation(
                    generation_session.stair_final_confirmation_reservation_path(
                        output
                    ),
                    expected_plan=audit_plan,
                )
            )
            tampered_anchor = dict(stored_reservation.final_anchor)
            tampered_anchor["anchor_id"] = "hash-valid-tampered-anchor"
            tampered_reservation = replace(
                stored_reservation,
                final_anchor=tampered_anchor,
                reservation_hash=None,
            )
            self.assertEqual(
                generation_session.StairFinalConfirmationReservationV2.from_mapping(
                    tampered_reservation.to_dict(),
                    expected_plan=audit_plan,
                ),
                tampered_reservation,
            )
            with self.assertRaisesRegex(
                ValueError,
                "authoritative terminal replay",
            ):
                generation_session._validate_formal_final_reservation_against_terminal_replay(
                    audit_foundation,
                    genesis=audit_genesis,
                    ledger=ledger,
                    reservation=tampered_reservation,
                )

            selected_trial_id = ledger[0]["formal"]["selected_trial_id"]
            selected_train = (
                generation_session.load_bound_generation_trial_train_receipt(
                    ledger[0],
                    trial_id=selected_trial_id,
                )
            )
            checkpoint_path = Path(
                selected_train.result.checkpoint.metadata["local_path"]
            )
            checkpoint_bytes = checkpoint_path.read_bytes()
            try:
                checkpoint_path.write_bytes(checkpoint_bytes + b"checkpoint-drift")
                with self.assertRaisesRegex(
                    ValueError,
                    "final eval checkpoint checksum mismatch",
                ):
                    generation_session._validate_formal_final_reservation_against_terminal_replay(
                        audit_foundation,
                        genesis=audit_genesis,
                        ledger=ledger,
                        reservation=stored_reservation,
                    )
            finally:
                checkpoint_path.write_bytes(checkpoint_bytes)

            self.assertEqual(
                [record["generation_id"] for record in ledger],
                [
                    f"{FORMAL_SESSION_ID}-generation-{index:03d}"
                    for index in range(1, 16)
                ],
            )
            self.assertEqual(ledger[0]["generation_outcome"], "accepted")
            self.assertTrue(ledger[0]["formal"]["accepted"])
            self.assertNotEqual(
                ledger[0]["resulting_current_best_hash"],
                ledger[0]["parent_current_best_hash"],
            )
            self.assertEqual(
                ledger[0]["resulting_current_best"]["anchor_kind"],
                "candidate",
            )
            self.assertTrue(
                all(
                    record["generation_outcome"] == "paired_authoring_incomplete"
                    and record["selection"] is None
                    and record["formal"] is None
                    for record in ledger[1:]
                )
            )
            self.assertTrue(
                all(
                    record["resulting_current_best_hash"]
                    == record["parent_current_best_hash"]
                    for record in ledger[1:]
                )
            )
            self.assertEqual(
                ledger[-1]["resulting_current_best"],
                ledger[0]["resulting_current_best"],
            )

            starts = [
                load_strict_json_object(path, context="formal fixture segment start")
                for path in sorted(
                    (output / "generation_segments").glob("segment-*/start.json")
                )
            ]
            exits = [
                load_strict_json_object(path, context="formal fixture segment exit")
                for path in sorted(
                    (output / "generation_segments").glob("segment-*/exit.json")
                )
            ]
            self.assertEqual(len(starts), 5)
            self.assertEqual(len(exits), 5)
            self.assertEqual(
                [item["generations_at_end"] for item in exits],
                [3, 6, 9, 12, 15],
            )
            self.assertEqual(
                [(item["status"], item["reason"]) for item in exits],
                [
                    ("paused", "operator_pause"),
                    ("paused", "operator_pause"),
                    ("paused", "operator_pause"),
                    ("paused", "operator_pause"),
                    ("completed", "normal"),
                ],
            )
            self.assertEqual(
                list((output / "generation_recoveries").glob("recovery-*.json")),
                [],
            )

            report = load_strict_json_object(
                report_path,
                context="formal fixture report",
            )
            self.assertEqual(
                report["schema_version"],
                generation_session.STAIR_GENERATION_FORMAL_REPORT_SCHEMA_VERSION,
            )
            self.assertEqual(len(report["generations"]), 15)
            self.assertEqual(report["engineering_status"], "go")
            self.assertFalse(report["acceptance_authority"])
            self.assertEqual(
                report["scientific_output"],
                {
                    "generation_outcome_counts": {
                        "accepted": 1,
                        "paired_authoring_incomplete": 14,
                    },
                    "completed_scientific_generation_count": 1,
                    "accepted_generation_count": 1,
                    "head_transition_count": 1,
                    "initial_head_hash": ledger[0]["parent_current_best_hash"],
                    "final_head_hash": ledger[-1]["resulting_current_best_hash"],
                    "head_moved_from_session_start": True,
                },
            )
            final = report["final_confirmation"]
            self.assertTrue(final["diagnostic_only"])
            self.assertEqual(
                final["reservation"]["final_anchor"],
                ledger[-1]["resulting_current_best"],
            )
            self.assertEqual(
                final["reservation"]["checkpoints"]["final"]["checkpoint_hash"],
                ledger[-1]["resulting_current_best"]["checkpoint_hash"],
            )
            self.assertEqual(
                final["reservation"]["checkpoints"]["final"]["source_manifest_hash"],
                ledger[-1]["resulting_current_best"]["subject_manifest_hash"],
            )
            self.assertNotEqual(
                final["reservation"]["checkpoints"]["final"]["checkpoint_hash"],
                final["reservation"]["checkpoints"]["genesis"]["checkpoint_hash"],
            )
            self.assertEqual(
                final["result"]["confirmation_namespace"],
                STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
            )
            self.assertGreater(final["result"]["paired"]["mean"], 0.0)
            self.assertNotIn("accepted", final["result"])
            self.assertNotIn("passed", final["result"])

            final_phase_receipts = list(
                (
                    output
                    / STAIR_FINAL_CONFIRMATION_DIRNAME
                    / "phase_receipts"
                    / "final"
                ).glob("*/eval_phases/sealed_validation.json")
            )
            self.assertEqual(len(final_phase_receipts), 1)
            final_phase = load_strict_json_object(
                final_phase_receipts[0],
                context="formal fixture final private phase receipt",
            )["phase_result"]
            self.assertEqual(
                final_phase["subject"],
                {
                    "schema_version": "true_rl_agent.experiment.subject.v2",
                    "kind": "candidate",
                    "id": ledger[-1]["resulting_current_best"]["accepted_candidate_id"],
                },
            )
            self.assertEqual(
                final_phase["subject_manifest_hash"],
                ledger[-1]["resulting_current_best"]["subject_manifest_hash"],
            )
            self.assertEqual(
                final_phase["checkpoint_hash"],
                ledger[-1]["resulting_current_best"]["checkpoint_hash"],
            )

            markers = [
                load_strict_json_object(path, context="formal fixture runner marker")
                for path in (output / "fixture_runner_outputs").glob("actual-*.json")
            ]
            self.assertEqual(len(markers), len({item["job_id"] for item in markers}))
            final_jobs = [
                item["job_id"]
                for item in markers
                if "-final-confirmation-" in item["job_id"]
            ]
            generation_one_train_jobs = [
                item["job_id"]
                for item in markers
                if f"{FORMAL_SESSION_ID}-generation-001-" in item["job_id"]
                and item["kind"] == "train"
            ]
            self.assertEqual(len(generation_one_train_jobs), 2)
            self.assertTrue(
                any("-lane-b-" in job_id for job_id in generation_one_train_jobs)
            )
            self.assertEqual(
                sorted(final_jobs),
                sorted(
                    [
                        f"{FORMAL_SESSION_ID}-final-confirmation-genesis-eval-sealed-validation",
                        f"{FORMAL_SESSION_ID}-final-confirmation-final-eval-sealed-validation",
                    ]
                ),
            )
            self.assertEqual(
                list((output / "job_attempts").rglob("attempt_01.reservation.json")),
                [],
            )

    def _worker(
        self,
        root: Path,
        action: str,
        *,
        expected: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        # Unified final confirmation runs phased evaluations and naturally sits
        # just below the legacy 240-second ceiling on the Linux production host.
        timeout_s = 360 if action == "finish" else 240
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.integration.test_stair_generation_session",
                "--fixture-worker",
                str(root),
                action,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            start_new_session=(os.name != "nt"),
            check=False,
        )
        if completed.returncode != expected:
            self.fail(
                f"fixture worker {action} exited {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed

    def _formal_worker(
        self,
        root: Path,
        action: str,
        *,
        expected: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.integration.test_stair_generation_session",
                "--formal-fixture-worker",
                str(root),
                action,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=240,
            start_new_session=(os.name != "nt"),
            check=False,
        )
        if completed.returncode != expected:
            self.fail(
                f"formal fixture worker {action} exited {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed


def _attempt_scopes(root: Path) -> dict[str, Any]:
    scopes = {}
    for path in sorted(root.glob("*/attempt_00.reservation.json")):
        journal, _ = JobAttemptJournal.from_reservation_path(path)
        if journal.logical_job_id in scopes:
            raise AssertionError("fixture has duplicate logical job journals")
        scopes[journal.logical_job_id] = journal.scope
    return scopes


def _attempt_result_for_job(root: Path, job_id: str) -> Path:
    matches = []
    for path in root.glob("*/attempt_00.reservation.json"):
        journal, _ = JobAttemptJournal.from_reservation_path(path)
        if journal.logical_job_id == job_id:
            matches.append(journal.root / "attempt_00.result.json")
    if len(matches) != 1 or not matches[0].is_file():
        raise AssertionError("fixture hard-crash attempt result is missing")
    return matches[0]


def _write_fake_nvidia_smi(path: Path) -> None:
    if os.name == "nt":
        path.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
    else:
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--fixture-worker":
    _run_worker(Path(sys.argv[2]).resolve(), sys.argv[3])
elif (
    __name__ == "__main__"
    and len(sys.argv) == 4
    and sys.argv[1] == "--formal-fixture-worker"
):
    _run_formal_worker(Path(sys.argv[2]).resolve(), sys.argv[3])
elif __name__ == "__main__":
    unittest.main()
