"""Fixed provider and attempt-backed runner for shared-host integration tests."""


from __future__ import annotations
import hashlib
import importlib.util
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from tests.fakes.backend_environment import TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
from true_rl_agent.agent_core.authoring_response import CandidateDraftResponse
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError
from true_rl_agent.agent_core.trial_receipts import _train_result_from_private_mapping, _train_result_to_private_mapping
from true_rl_agent.candidates.capability_readiness import COMPLETE_REWARD_PROGRAM_CAPABILITY_ID
from true_rl_agent.candidates.loader import load_training_components, load_training_components_source_bytes
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
from true_rl_agent.runner.process_handle import OWNED_PROCESS_IDENTITY_SCHEMA_VERSION
from true_rl_agent.runner.specs import CheckpointRef, EvalPhaseJobSpec
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.evaluation_contract import build_stair_evaluation_summary
from true_rl_agent.runner.gpu_process_binding import GPU_COMPUTE_CONTEXT_OBSERVATION_SCHEMA_VERSION, GPU_RUNTIME_EVIDENCE_SCHEMA_VERSION
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import StairEvaluationExecutionResult, preflight_stair_execution_source_manifest
from true_rl_agent.tasks.stair_climbing.phased_evaluation import build_stair_eval_phase_summary
from true_rl_agent.tasks.stair_climbing.reward_program import STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION
from true_rl_agent.tasks.stair_climbing.runtime_profile import STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE, STAIR_FORMAL_EVAL_NAMESPACE
from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_stair_summary
from true_rl_agent.training_programs import CANONICAL_TRAINING_COMPONENTS_PATH


SESSION_ID = "p6c-production-shaped-fixture"


FORMAL_SESSION_ID = "p6d-production-shaped-fixture"


GPU_UUIDS = (
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
)


HARD_CRASH_EXIT_CODE = 86


def _isaac_assets_source_available() -> bool:
    """Detect the installed source package without requiring a live Kit app."""

    return importlib.util.find_spec("isaaclab_assets") is not None


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


def _nvidia_smi_name() -> str:
    return "fake-nvidia-smi.cmd" if os.name == "nt" else "fake-nvidia-smi"


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
    reference_only = False
    """Accept one real pair, then close the remaining formal generations cheaply."""

    def next_authoring_response(
        self,
        *,
        prompt_payload: dict[str, Any],
    ) -> CandidateDraftResponse:
        if self.generation_index == 1 and not self.reference_only:
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
    def __init__(self, *, foundation):
        self.foundation = foundation
        self.output_root = foundation.config.output_dir / "fixture_runner_outputs"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.delegate = SubprocessRunner(("fixture-only",), timeout_s=30,
            gpu_lease_manager=foundation.gpu_lease_manager,
            attempt_admission_hook=foundation.runner.attempt_admission_hook)

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

    def run_eval(self, eval_spec):
        raise AssertionError("shared session fixture must use phased evaluation")

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


    def _run_phase_once(
        self,
        eval_spec: EvalPhaseJobSpec,
        *,
        context,
    ) -> EvaluationPhaseResult:
        binding, physical_uuid = self._binding(context)
        self._claim_actual("eval_phase", eval_spec.job_id)
        success_thirds = _success_thirds(eval_spec)
        config = self._evaluation_config(eval_spec.evaluation_plan)
        execution = self._execution(
            config=config,
            eval_spec=eval_spec.evaluation_plan,
            split_success_thirds={eval_spec.phase: success_thirds},
        )
        summary = build_stair_eval_phase_summary(
            config=config,
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
        if train_spec.runtime_diagnostic_context_hash is not None:
            comparison = load_strict_json_object(
                Path(train_spec.metadata["summary_path"]).parent / "runtime_comparison.json",
                context="fixture runtime comparison",
            )
            if canonical_json_hash(comparison) != train_spec.runtime_diagnostic_context_hash:
                raise AssertionError("fixture runtime comparison differs from TrainJobSpec")
            # This runner models a positive task result; actual differential
            # execution is covered by the dispatcher tests and Isaac probes.
            summary["runtime_diagnostic_context_hash"] = train_spec.runtime_diagnostic_context_hash
            summary["runtime_activation_summary"]["operation_effect"] = {
                "schema_version": "true_rl_agent.stair_reward_operation_effect.v1",
                "context_hash": train_spec.runtime_diagnostic_context_hash,
                "mechanism_operation_delta": comparison["mechanism_operation_delta"],
                "comparison": "same_transition_weighted_rate_difference",
                "coverage": "all_dispatched_training_transitions",
                "sampled_control_steps": calls,
                "sampled_env_transitions": calls * config.train_num_envs,
                "nonzero_env_transitions": calls * config.train_num_envs,
                "max_abs_weighted_delta": 1.0,
            }
        return summary

    def _evaluation_config(self, eval_spec):
        return self.foundation.composition.task_plugin.config

    def _execution(
        self, *, config, eval_spec, split_success_thirds
    ) -> StairEvaluationExecutionResult:
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
        for phase, success_thirds in split_success_thirds.items():
            per_seed = tuple(
                tuple(
                    _outcomes(
                        config,
                        phase=phase,
                        success_thirds=success_thirds,
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
        "schema_version": GPU_COMPUTE_CONTEXT_OBSERVATION_SCHEMA_VERSION,
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
        "gpu_runtime_evidence": {
            "schema_version": GPU_RUNTIME_EVIDENCE_SCHEMA_VERSION,
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
    success_thirds: int,
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
        # The scenario uses thirds (reference 2/3, improved candidate 3/3),
        # independently of the suite's episode count. Capability controls stay
        # successful while the OOD objectives provide the selection signal.
        successes = round(episodes_per_cell * success_thirds / 3)
        success = env_index % episodes_per_cell < successes
        if config.is_unified_protocol and suite.is_calibrated:
            objective_strata = {stratum for members in suite.score_contract["families"].values()
                                for stratum in members}
            if cell.stratum_id not in objective_strata:
                success = True
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


def _success_thirds(eval_spec: EvalPhaseJobSpec) -> int:
    # Success is expressed in thirds. Keep generation 2 lane A
    # ahead on public_dev, then below the accepted parent on formal evaluation.
    job_id = eval_spec.job_id
    if f"{FORMAL_SESSION_ID}-generation-001-lane-a" in job_id:
        return 3
    if f"{FORMAL_SESSION_ID}-generation-001-lane-b" in job_id:
        return 2
    if f"{FORMAL_SESSION_ID}-final-confirmation-final-eval" in job_id:
        # The final slot may still evaluate the reference checkpoint. Synthetic
        # performance follows the evaluated subject, not the name of the slot.
        return 3 if eval_spec.subject.kind == "candidate" else 2
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


def _write_fake_nvidia_smi(path: Path) -> None:
    if os.name == "nt":
        path.write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
    else:
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
