"""Complete-reward program capability and guarded dispatcher for Stair."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from true_rl_agent.candidates.artifact import (
    SealedCandidateArtifactRef,
    load_sealed_candidate_artifact,
)
from true_rl_agent.candidates.authoring_policy import require_component_name
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents
from true_rl_agent.candidates.loader import load_training_components_source_bytes
from true_rl_agent.candidates.verifier import (
    StaticCandidateVerifierPolicy,
    audit_candidate_python_source,
    static_verifier_source_policy_hash,
    training_component_purity_issues,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.prompting.capabilities import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
)
from true_rl_agent.runner.candidate_failure import (
    is_infrastructure_runtime_exception,
)
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairCompleteRewardRuntimeContract,
)
from true_rl_agent.tasks.stair_climbing.reward_inputs import (
    StairRewardInputsV2,
    stair_reward_input_values,
)
from true_rl_agent.tasks.stair_climbing.reward_source_semantics import (
    audit_stair_reward_source_semantics,
)
from true_rl_agent.training_programs import (
    CANONICAL_TRAINING_COMPONENTS_PATH,
    VerifiedReferenceProgramArtifact,
)

STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.qualified_reward_program.v1"
)
STAIR_REWARD_RUNTIME_EVIDENCE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.complete_reward_runtime_evidence.v1"
)


class StairRewardProgramError(ValueError):
    """Raised before a malformed or impure complete reward reaches PPO."""


class StairRewardOutputContractError(StairRewardProgramError):
    """Raised when a candidate reward output violates the frozen runtime ABI."""


class StairRewardArtifactIntegrityError(RuntimeError):
    """Raised when sealed source identity changes outside candidate execution."""


@dataclass(frozen=True, init=False)
class QualifiedStairRewardProgram:
    """Task-owned execution capability layered over exact verified source bytes."""

    reward_terms: tuple[RewardTerm, ...]
    contract: StairCompleteRewardRuntimeContract
    subject: ExperimentSubject
    subject_manifest_hash: str
    program_source_checksum: str
    program_source_size_bytes: int
    static_verifier_policy_hash: str
    qualification_hash: str

    _TOKEN = object()

    def __init__(
        self,
        *,
        reward_terms: tuple[RewardTerm, ...],
        contract: StairCompleteRewardRuntimeContract,
        subject: ExperimentSubject,
        subject_manifest_hash: str,
        program_source_checksum: str,
        program_source_size_bytes: int,
        static_verifier_policy_hash: str,
        qualification_hash: str,
        _token: object | None = None,
    ) -> None:
        if _token is not self._TOKEN:
            raise TypeError(
                "qualified Stair reward programs must come from the task verifier"
            )
        if (
            not isinstance(reward_terms, tuple)
            or not reward_terms
            or any(not isinstance(term, RewardTerm) for term in reward_terms)
        ):
            raise TypeError("qualified Stair reward terms are invalid")
        if not isinstance(contract, StairCompleteRewardRuntimeContract):
            raise TypeError("qualified Stair reward contract is invalid")
        if not isinstance(subject, ExperimentSubject) or subject.kind not in {
            "candidate",
            "reference",
        }:
            raise TypeError("qualified Stair reward subject is invalid")
        require_sha256_digest(
            subject_manifest_hash,
            "qualified Stair reward subject manifest hash",
        )
        require_sha256_checksum(
            program_source_checksum,
            "qualified Stair reward source checksum",
        )
        if (
            isinstance(program_source_size_bytes, bool)
            or not isinstance(program_source_size_bytes, int)
            or program_source_size_bytes <= 0
        ):
            raise ValueError("qualified Stair reward source size must be positive")
        require_sha256_digest(
            static_verifier_policy_hash,
            "qualified Stair static verifier policy hash",
        )
        require_sha256_digest(
            qualification_hash,
            "qualified Stair reward program hash",
        )
        object.__setattr__(self, "reward_terms", reward_terms)
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "subject_manifest_hash", subject_manifest_hash)
        object.__setattr__(
            self,
            "program_source_checksum",
            program_source_checksum,
        )
        object.__setattr__(
            self,
            "program_source_size_bytes",
            program_source_size_bytes,
        )
        object.__setattr__(
            self,
            "static_verifier_policy_hash",
            static_verifier_policy_hash,
        )
        object.__setattr__(self, "qualification_hash", qualification_hash)

    @property
    def term_names(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.reward_terms)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION,
            "subject": self.subject.to_public_dict(),
            "subject_manifest_hash": self.subject_manifest_hash,
            "program_source_checksum": self.program_source_checksum,
            "program_source_size_bytes": self.program_source_size_bytes,
            "input_schema_hash": self.contract.input_schema_hash,
            "static_verifier_policy_hash": self.static_verifier_policy_hash,
            "term_names": list(self.term_names),
            "qualification_hash": self.qualification_hash,
        }

    def __deepcopy__(self, memo: dict[int, Any]) -> "QualifiedStairRewardProgram":
        memo[id(self)] = self
        return self


def qualify_stair_reference_reward_program(
    *,
    artifact: VerifiedReferenceProgramArtifact,
    contract: StairCompleteRewardRuntimeContract,
) -> QualifiedStairRewardProgram:
    """Qualify exact bytes carried by a verified reference capability."""

    if not isinstance(artifact, VerifiedReferenceProgramArtifact):
        raise TypeError("Stair reference qualification requires a verified artifact")
    artifact.manifest.verify_source_bytes(artifact.source_bytes)
    if artifact.manifest.manifest_hash is None:
        raise StairRewardProgramError("Stair reference manifest is not finalized")
    return _qualify_exact_stair_reward_source(
        contract=contract,
        subject=ExperimentSubject.reference(artifact.manifest.reference_id),
        subject_manifest_hash=artifact.manifest.manifest_hash,
        source_bytes=artifact.source_bytes,
        expected_source_checksum=artifact.manifest.source_checksum,
        source_name=artifact.manifest.source_uri,
        require_reference_term_set=True,
        static_verifier_policy=StaticCandidateVerifierPolicy(),
        trusted_reference_source_text=artifact.source_text,
    )


def qualify_stair_candidate_reward_program(
    *,
    artifact: SealedCandidateArtifactRef,
    contract: StairCompleteRewardRuntimeContract,
    static_verifier_policy: StaticCandidateVerifierPolicy,
    trusted_reference_artifact: VerifiedReferenceProgramArtifact,
) -> QualifiedStairRewardProgram:
    """Revalidate one sealed snapshot and qualify only its canonical bytes."""

    if not isinstance(artifact, SealedCandidateArtifactRef):
        raise TypeError("Stair candidate qualification requires a sealed artifact")
    if not isinstance(static_verifier_policy, StaticCandidateVerifierPolicy):
        raise TypeError("Stair candidate qualification requires a static verifier policy")
    if not isinstance(trusted_reference_artifact, VerifiedReferenceProgramArtifact):
        raise TypeError(
            "Stair candidate qualification requires a trusted reference artifact"
        )
    verified_before = load_sealed_candidate_artifact(
        artifact.path,
        change_set=artifact.change_set,
        expected_candidate_id=artifact.candidate_id,
        expected_artifact_hash=artifact.artifact_hash,
        expected_subject_manifest_hash=artifact.subject_manifest_hash,
        expected_program_source_checksum=artifact.program_source_checksum,
    )
    source_path = Path(verified_before.path) / CANONICAL_TRAINING_COMPONENTS_PATH
    if source_path.is_symlink() or not source_path.is_file():
        raise StairRewardArtifactIntegrityError(
            "Stair candidate source must be a regular artifact file"
        )
    source_bytes = source_path.read_bytes()
    verified_after = load_sealed_candidate_artifact(
        artifact.path,
        change_set=artifact.change_set,
        expected_candidate_id=artifact.candidate_id,
        expected_artifact_hash=artifact.artifact_hash,
        expected_subject_manifest_hash=artifact.subject_manifest_hash,
        expected_program_source_checksum=artifact.program_source_checksum,
    )
    if verified_before != verified_after:
        raise StairRewardArtifactIntegrityError(
            "Stair candidate artifact changed during qualification"
        )
    return _qualify_exact_stair_reward_source(
        contract=contract,
        subject=ExperimentSubject.candidate(artifact.candidate_id),
        subject_manifest_hash=artifact.subject_manifest_hash,
        source_bytes=source_bytes,
        expected_source_checksum=artifact.program_source_checksum,
        source_name=str(source_path),
        require_reference_term_set=False,
        static_verifier_policy=static_verifier_policy,
        trusted_reference_source_text=trusted_reference_artifact.source_text,
    )


def _qualify_exact_stair_reward_source(
    *,
    contract: StairCompleteRewardRuntimeContract,
    subject: ExperimentSubject,
    subject_manifest_hash: str,
    source_bytes: bytes,
    expected_source_checksum: str,
    source_name: str,
    require_reference_term_set: bool,
    static_verifier_policy: StaticCandidateVerifierPolicy,
    trusted_reference_source_text: str,
) -> QualifiedStairRewardProgram:
    """Mint the capability after a role-specific artifact factory binds bytes."""

    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise StairRewardProgramError("Stair reward source bytes must be non-empty")
    expected_source_checksum = require_sha256_checksum(
        expected_source_checksum,
        "Stair reward expected source checksum",
    )
    import hashlib

    actual_checksum = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
    if actual_checksum != expected_source_checksum:
        raise StairRewardArtifactIntegrityError(
            "Stair reward source checksum mismatch"
        )
    try:
        source_text = source_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise StairRewardProgramError(
            "Stair reward source must be valid UTF-8"
        ) from exc
    if not isinstance(static_verifier_policy, StaticCandidateVerifierPolicy):
        raise TypeError("Stair reward qualification static verifier policy is invalid")
    issues = audit_candidate_python_source(
        source_text,
        policy=static_verifier_policy,
    )
    if issues:
        raise StairRewardProgramError(
            "Stair reward source violates the pure-function policy: "
            + ", ".join(issues)
        )
    task_semantics_issues = audit_stair_reward_source_semantics(
        source_text,
        contract=contract,
        trusted_reference_source_text=trusted_reference_source_text,
    )
    if task_semantics_issues:
        raise StairRewardProgramError(
            "Stair reward source violates the task input semantics: "
            + ", ".join(task_semantics_issues)
        )
    module_name = (
        "true_rl_agent_stair_reward_"
        + actual_checksum.removeprefix("sha256:")[:16]
    )
    try:
        components = load_training_components_source_bytes(
            source_bytes,
            module_name=module_name,
            source_name=source_name,
        )
    except Exception as exc:
        if is_infrastructure_runtime_exception(exc):
            raise
        raise StairRewardProgramError(
            "Stair reward source could not produce valid training components"
        ) from exc
    component_issues = training_component_purity_issues(
        components,
        expected_module_name=module_name,
        expected_source_name=source_name,
    )
    if component_issues:
        raise StairRewardProgramError(
            "Stair reward component purity failed: " + ", ".join(component_issues)
        )
    terms = _validate_complete_training_components(components)
    runtime = contract.to_public_dict()
    expected_reference_terms = tuple(
        runtime["reference_equivalence"]["active_term_names"]
    )
    if require_reference_term_set and tuple(term.name for term in terms) != (
        expected_reference_terms
    ):
        raise StairRewardProgramError(
            "Stair reference reward terms differ from the canonical term order"
        )
    subject_manifest_hash = require_sha256_digest(
        subject_manifest_hash,
        "Stair reward subject manifest hash",
    )
    if not isinstance(subject, ExperimentSubject) or subject.kind not in {
        "candidate",
        "reference",
    }:
        raise TypeError("Stair reward subject must be candidate or reference")
    # Reward qualification binds source-purity semantics, not mutable worker
    # timing used by the later authoring/admission infrastructure.
    policy_hash = static_verifier_source_policy_hash(
        policy=static_verifier_policy,
    )
    term_contract = [
        {
            "name": term.name,
            "weight": float(term.weight),
            "clip_min": term.clip_min,
            "clip_max": term.clip_max,
        }
        for term in terms
    ]
    qualification_payload = {
        "schema_version": STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION,
        "subject": subject.to_public_dict(),
        "subject_manifest_hash": subject_manifest_hash,
        "program_source_checksum": actual_checksum,
        "program_source_size_bytes": len(source_bytes),
        "input_schema_hash": contract.input_schema_hash,
        "complete_reward_runtime_hash": canonical_json_hash(runtime),
        "static_verifier_policy_hash": policy_hash,
        "term_contract": term_contract,
    }
    return QualifiedStairRewardProgram(
        reward_terms=terms,
        contract=contract,
        subject=subject,
        subject_manifest_hash=subject_manifest_hash,
        program_source_checksum=actual_checksum,
        program_source_size_bytes=len(source_bytes),
        static_verifier_policy_hash=policy_hash,
        qualification_hash=canonical_json_hash(qualification_payload),
        _token=QualifiedStairRewardProgram._TOKEN,
    )


@dataclass(frozen=True)
class StairRewardEvaluation:
    total_rate: Any
    raw_terms: Mapping[str, Any]
    weighted_terms: Mapping[str, Any]


class StairCompleteRewardDispatcher:
    """Run one qualified complete program exactly once per environment step."""

    def __init__(
        self,
        program: QualifiedStairRewardProgram,
        *,
        num_envs: int,
    ) -> None:
        if not isinstance(program, QualifiedStairRewardProgram):
            raise TypeError("Stair dispatcher requires a qualified reward program")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("Stair dispatcher num_envs must be positive")
        self.program = program
        self.num_envs = num_envs
        self.step_calls = 0
        self.term_calls = 0
        self.last_common_step: int | None = None
        self.last_evaluation: StairRewardEvaluation | None = None
        self._sampled_term_stats: dict[str, dict[str, Any]] = {}
        self._last_total_mean: Any = None
        self._max_abs_total: Any = None
        self._evidence_interval_steps = int(
            program.contract.dispatcher_contract[
                "evidence_sample_interval_steps"
            ]
        )

    def compute(
        self,
        inputs: StairRewardInputsV2,
        *,
        common_step: int,
    ) -> Any:
        if isinstance(common_step, bool) or not isinstance(common_step, int):
            raise RuntimeError("Stair common step must be an integer")
        if self.last_common_step is not None and common_step != self.last_common_step + 1:
            raise RuntimeError(
                "Stair reward dispatcher was not called exactly once per common step"
            )
        evaluation = evaluate_stair_reward_program(
            self.program,
            inputs,
            num_envs=self.num_envs,
        )
        self.step_calls += 1
        self.term_calls += len(self.program.reward_terms)
        self.last_common_step = common_step
        self.last_evaluation = evaluation
        self._record_evidence(evaluation)
        return evaluation.total_rate

    def runtime_summary(self) -> dict[str, Any]:
        term_records = []
        for name in self.program.term_names:
            stats = self._sampled_term_stats.get(name)
            term_records.append(
                {
                    "name": name,
                    "call_count": self.step_calls,
                    "sample_count": int(stats["sample_count"]) if stats else 0,
                    "last_raw_mean": _tensor_scalar(stats["last_raw_mean"])
                    if stats
                    else None,
                    "last_weighted_mean": _tensor_scalar(
                        stats["last_weighted_mean"]
                    )
                    if stats
                    else None,
                    "max_abs_weighted_rate": _tensor_scalar(
                        stats["max_abs_weighted_rate"]
                    )
                    if stats
                    else None,
                }
            )
        last_total_mean = _tensor_scalar(self._last_total_mean)
        max_abs_total = _tensor_scalar(self._max_abs_total)
        component_records = [
            {
                "name": record["name"],
                "component_kind": "reward_term",
                "call_count": record["call_count"],
                "observation_count": record["sample_count"],
                "last_weighted_mean": record["last_weighted_mean"],
                "max_abs_weighted_contribution": record[
                    "max_abs_weighted_rate"
                ],
            }
            for record in term_records
        ]
        return {
            "schema_version": STAIR_REWARD_RUNTIME_EVIDENCE_SCHEMA_VERSION,
            "status": "pass" if self.step_calls > 0 else "not_executed",
            "program": self.program.to_public_dict(),
            "dispatcher_calls": self.step_calls,
            "term_calls": self.term_calls,
            "declared_term_count": len(self.program.term_names),
            "last_common_step": self.last_common_step,
            "input_guard": "pass" if self.step_calls > 0 else "not_executed",
            "rng_guard": "pass" if self.step_calls > 0 else "not_executed",
            "source_alias_guard": "pass" if self.step_calls > 0 else "not_executed",
            "last_total_rate_mean": last_total_mean,
            "max_abs_total_rate": max_abs_total,
            "term_records": term_records,
            "reward_term_calls": self.term_calls,
            "diagnostic_calls": 0,
            "nonzero_reward_contribution": (
                max_abs_total is not None and max_abs_total > 1.0e-12
            ),
            "reward_output_stats": {
                "complete_reward": {
                    "last_mean": last_total_mean,
                    "max_abs": max_abs_total,
                }
            },
            "diagnostic_outputs": {},
            "component_activation_records": component_records,
            "feature_activation_records": [],
            "capability_activation_records": [
                COMPLETE_REWARD_PROGRAM_CAPABILITY_ID
            ],
        }

    def _record_evidence(self, evaluation: StairRewardEvaluation) -> None:
        import torch

        if (
            self.step_calls != 1
            and self.step_calls % self._evidence_interval_steps != 0
        ):
            return
        for name in self.program.term_names:
            raw = evaluation.raw_terms[name]
            weighted = evaluation.weighted_terms[name]
            raw_mean = raw.detach().mean()
            weighted_mean = weighted.detach().mean()
            weighted_max = weighted.detach().abs().max()
            current = self._sampled_term_stats.get(name)
            self._sampled_term_stats[name] = {
                "sample_count": 1 + (int(current["sample_count"]) if current else 0),
                "last_raw_mean": raw_mean,
                "last_weighted_mean": weighted_mean,
                "max_abs_weighted_rate": torch.maximum(
                    current["max_abs_weighted_rate"],
                    weighted_max,
                )
                if current
                else weighted_max,
            }
        total = evaluation.total_rate.detach()
        self._last_total_mean = total.mean()
        total_max = total.abs().max()
        self._max_abs_total = (
            torch.maximum(self._max_abs_total, total_max)
            if self._max_abs_total is not None
            else total_max
        )


def evaluate_stair_reward_program(
    program: QualifiedStairRewardProgram,
    inputs: StairRewardInputsV2,
    *,
    num_envs: int,
) -> StairRewardEvaluation:
    """Evaluate one complete program with input and RNG isolation guards."""

    if not isinstance(program, QualifiedStairRewardProgram):
        raise TypeError("Stair reward evaluation requires a qualified program")
    if type(inputs) is not StairRewardInputsV2:
        raise TypeError("Stair reward evaluation received invalid inputs")
    if inputs.schema_hash != program.contract.input_schema_hash:
        raise RuntimeError("Stair reward input schema hash mismatch")
    import torch

    values = stair_reward_input_values(inputs)
    reference_tensor, input_float_tensors = _validate_reward_inputs(
        values,
        contract=program.contract,
        num_envs=num_envs,
        torch=torch,
    )
    versions_before = _tensor_versions(values, torch=torch)
    rng_before = _capture_rng_state(torch=torch)
    raw_terms: dict[str, Any] = {}
    weighted_terms: dict[str, Any] = {}
    total = None
    caught: Exception | None = None
    try:
        output = program.contract.output_contract
        term_bound = float(output["maximum_absolute_term_rate"])
        for term in program.reward_terms:
            value = term.fn(inputs)
            value = _guard_reward_vector(
                value,
                name=term.name,
                num_envs=num_envs,
                reference_tensor=reference_tensor,
                torch=torch,
            )
            if term.clip_min is not None or term.clip_max is not None:
                value = value.clamp(min=term.clip_min, max=term.clip_max)
            weighted = value * float(term.weight)
            weighted = _guard_reward_vector(
                weighted,
                name=term.name,
                num_envs=num_envs,
                reference_tensor=reference_tensor,
                torch=torch,
            )
            raw_terms[term.name] = value
            weighted_terms[term.name] = weighted
            total = weighted if total is None else total + weighted
        if total is None:
            raise StairRewardOutputContractError(
                "Stair complete reward has no active terms"
            )
        total = _guard_reward_vector(
            total,
            name="complete_reward",
            num_envs=num_envs,
            reference_tensor=reference_tensor,
            torch=torch,
        )
        _require_reward_tensor_batches(
            input_float_tensors=input_float_tensors,
            raw_terms=raw_terms,
            weighted_terms=weighted_terms,
            total=total,
            maximum_absolute_term_rate=term_bound,
            maximum_absolute_total_rate=float(
                output["maximum_absolute_total_rate"]
            ),
            torch=torch,
        )
    except Exception as exc:  # preserve guard cleanup before projection
        if is_infrastructure_runtime_exception(exc):
            raise
        caught = exc
    versions_after = _tensor_versions(values, torch=torch)
    rng_after = _capture_rng_state(torch=torch)
    input_changed = versions_after != versions_before
    rng_changed = not _rng_states_equal(rng_before, rng_after, torch=torch)
    if rng_changed:
        _restore_rng_state(rng_before, torch=torch)
    if input_changed:
        raise StairRewardProgramError("Stair reward program mutated its input tensors")
    if rng_changed:
        raise StairRewardProgramError("Stair reward program changed process RNG state")
    if caught is not None:
        if isinstance(caught, StairRewardProgramError):
            raise caught
        raise StairRewardProgramError(
            f"Stair reward program raised {type(caught).__name__}: {caught}"
        ) from caught
    assert total is not None
    return StairRewardEvaluation(
        total_rate=total,
        raw_terms=raw_terms,
        weighted_terms=weighted_terms,
    )


def assert_stair_reward_program_deterministic(
    program: QualifiedStairRewardProgram,
    inputs_factory: Callable[[], StairRewardInputsV2],
) -> dict[str, Any]:
    """Run the frozen activation fixture without duplicating real env-step calls."""

    import torch

    repeats = int(
        program.contract.dispatcher_contract[
            "determinism_preflight_repeats"
        ]
    )
    evaluations = []
    for _ in range(repeats):
        inputs = inputs_factory()
        evaluations.append(
            evaluate_stair_reward_program(
                program,
                inputs,
                num_envs=_input_num_envs(inputs),
            )
        )
    baseline = evaluations[0]
    for candidate in evaluations[1:]:
        if not torch.equal(baseline.total_rate, candidate.total_rate):
            raise StairRewardProgramError(
                "Stair reward program is not deterministic for identical inputs"
            )
        for name in program.term_names:
            if not torch.equal(baseline.raw_terms[name], candidate.raw_terms[name]):
                raise StairRewardProgramError(
                    f"Stair reward term is not deterministic: {name}"
                )
    return {
        "schema_version": "true_rl_agent.task.stair_climbing.reward_determinism.v1",
        "status": "pass",
        "repeat_count": repeats,
        "term_names": list(program.term_names),
    }


def _input_num_envs(inputs: StairRewardInputsV2) -> int:
    values = stair_reward_input_values(inputs)
    for value in values:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    raise StairRewardProgramError("Stair reward fixture has no batch dimension")


def _validate_complete_training_components(
    components: TrainingComponents,
) -> tuple[RewardTerm, ...]:
    if type(components) is not TrainingComponents:
        raise TypeError("Stair reward entrypoint must return TrainingComponents")
    if components.diagnostics or components.schedules:
        raise StairRewardProgramError(
            "Stair complete reward source cannot carry diagnostics or schedules"
        )
    terms = tuple(components.reward_terms)
    if not terms:
        raise StairRewardProgramError("Stair complete reward requires reward terms")
    names = []
    for term in terms:
        if type(term) is not RewardTerm:
            raise StairRewardProgramError("Stair complete reward term type is invalid")
        try:
            require_component_name(term.name, "Stair reward term name")
        except ValueError as exc:
            raise StairRewardProgramError("Stair reward term name is invalid") from exc
        if not term.name.isidentifier() or term.name.startswith("_"):
            raise StairRewardProgramError("Stair reward term name is invalid")
        if not callable(term.fn):
            raise StairRewardProgramError(
                f"Stair reward term is not callable: {term.name}"
            )
        weight = term.weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
        ):
            raise StairRewardProgramError(
                f"Stair reward term weight is invalid: {term.name}"
            )
        for field_name, value in (
            ("clip_min", term.clip_min),
            ("clip_max", term.clip_max),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise StairRewardProgramError(
                    f"Stair reward term {field_name} is invalid: {term.name}"
                )
        if (
            term.clip_min is not None
            and term.clip_max is not None
            and float(term.clip_min) >= float(term.clip_max)
        ):
            raise StairRewardProgramError(
                f"Stair reward term clip interval is invalid: {term.name}"
            )
        names.append(term.name)
    if len(names) != len(set(names)):
        raise StairRewardProgramError("Stair reward term names must be unique")
    return terms


def _guard_reward_vector(
    value: Any,
    *,
    name: str,
    num_envs: int,
    reference_tensor: Any,
    torch: Any,
) -> Any:
    if not torch.is_tensor(value):
        raise StairRewardOutputContractError(
            f"Stair reward term is not a tensor: {name}"
        )
    if tuple(value.shape) != (num_envs,):
        raise StairRewardOutputContractError(
            f"Stair reward term shape mismatch: {name}: {tuple(value.shape)!r}"
        )
    if value.device != reference_tensor.device:
        raise StairRewardOutputContractError(
            f"Stair reward term device mismatch: {name}"
        )
    if not value.dtype.is_floating_point:
        raise StairRewardOutputContractError(
            f"Stair reward term dtype mismatch: {name}"
        )
    return value


def _validate_reward_inputs(
    values: tuple[Any, ...],
    *,
    contract: StairCompleteRewardRuntimeContract,
    num_envs: int,
    torch: Any,
) -> tuple[Any, list[tuple[str, Any]]]:
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise RuntimeError("Stair reward num_envs must be positive")
    if len(values) != len(contract.input_fields):
        raise RuntimeError("Stair reward input value count mismatch")
    reference_tensor = None
    float_tensors: list[tuple[str, Any]] = []
    for field, value in zip(contract.input_fields, values, strict=True):
        if not torch.is_tensor(value):
            raise RuntimeError(
                f"Stair reward input is not a tensor: {field.name}"
            )
        if value.is_inference():
            raise RuntimeError(
                f"Stair reward input lacks version tracking: {field.name}"
            )
        expected_shape = (num_envs, *field.trailing_shape)
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"Stair reward input shape mismatch: {field.name}: "
                f"expected={expected_shape!r} actual={tuple(value.shape)!r}"
            )
        if field.dtype == "bool":
            if value.dtype != torch.bool:
                raise RuntimeError(
                    f"Stair reward input dtype mismatch: {field.name}"
                )
            continue
        if not value.dtype.is_floating_point:
            raise RuntimeError(
                f"Stair reward input dtype mismatch: {field.name}"
            )
        if reference_tensor is None:
            reference_tensor = value
        elif value.device != reference_tensor.device:
            raise RuntimeError(
                f"Stair reward input device mismatch: {field.name}"
            )
        float_tensors.append((field.name, value))
    if reference_tensor is None:
        raise RuntimeError("Stair reward inputs contain no floating tensors")
    for field, value in zip(contract.input_fields, values, strict=True):
        if value.device != reference_tensor.device:
            raise RuntimeError(
                f"Stair reward input device mismatch: {field.name}"
            )
        if value.dtype.is_floating_point and not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"Stair reward input is non-finite: {field.name}")
    return reference_tensor, float_tensors


def _require_reward_tensor_batches(
    *,
    input_float_tensors: list[tuple[str, Any]],
    raw_terms: Mapping[str, Any],
    weighted_terms: Mapping[str, Any],
    total: Any,
    maximum_absolute_term_rate: float,
    maximum_absolute_total_rate: float,
    torch: Any,
) -> None:
    if not input_float_tensors or not raw_terms or not weighted_terms:
        raise StairRewardOutputContractError("Stair reward runtime guard is empty")
    input_batch = torch.cat(
        tuple(value.reshape(value.shape[0], -1) for _, value in input_float_tensors),
        dim=1,
    )
    raw_batch = torch.stack(tuple(raw_terms.values()), dim=0)
    weighted_batch = torch.stack(tuple(weighted_terms.values()), dim=0)
    combined = (
        torch.isfinite(input_batch).all()
        & torch.isfinite(raw_batch).all()
        & torch.isfinite(weighted_batch).all()
        & (weighted_batch.detach().abs().max() <= maximum_absolute_term_rate)
        & torch.isfinite(total).all()
        & (total.detach().abs().max() <= maximum_absolute_total_rate)
    )
    if bool(combined.item()):
        return
    failed: list[str] = []
    for name, value in input_float_tensors:
        if not bool(torch.isfinite(value).all().item()):
            failed.append(f"reward_input:{name}")
    for name, value in raw_terms.items():
        if not bool(torch.isfinite(value).all().item()):
            failed.append(f"reward_term_raw:{name}")
    for name, value in weighted_terms.items():
        valid = torch.isfinite(value).all() & (
            value.detach().abs().max() <= maximum_absolute_term_rate
        )
        if not bool(valid.item()):
            failed.append(f"reward_term_weighted:{name}")
    total_valid = torch.isfinite(total).all() & (
        total.detach().abs().max() <= maximum_absolute_total_rate
    )
    if not bool(total_valid.item()):
        failed.append("reward_total:complete_reward")
    raise StairRewardOutputContractError(
        "Stair reward input/output is non-finite or out of bounds: "
        + ", ".join(failed)
    )


def _tensor_versions(values: tuple[Any, ...], *, torch: Any) -> tuple[int, ...]:
    return tuple(
        int(value._version)  # noqa: SLF001 - trusted mutation guard
        for value in values
        if torch.is_tensor(value)
    )


@dataclass(frozen=True)
class _RngState:
    python: object
    numpy: object | None
    torch_cpu: Any
    torch_cuda: tuple[Any, ...]


def _capture_rng_state(*, torch: Any) -> _RngState:
    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except ImportError:
        pass
    cuda_states: tuple[Any, ...] = ()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return _RngState(
        python=random.getstate(),
        numpy=numpy_state,
        torch_cpu=torch.random.get_rng_state().clone(),
        torch_cuda=cuda_states,
    )


def _rng_states_equal(left: _RngState, right: _RngState, *, torch: Any) -> bool:
    if left.python != right.python or not torch.equal(left.torch_cpu, right.torch_cpu):
        return False
    if len(left.torch_cuda) != len(right.torch_cuda) or any(
        not torch.equal(a, b) for a, b in zip(left.torch_cuda, right.torch_cuda)
    ):
        return False
    if left.numpy is None or right.numpy is None:
        return left.numpy is right.numpy
    import numpy as np

    return (
        left.numpy[0] == right.numpy[0]
        and np.array_equal(left.numpy[1], right.numpy[1])
        and left.numpy[2:] == right.numpy[2:]
    )


def _restore_rng_state(state: _RngState, *, torch: Any) -> None:
    random.setstate(state.python)
    if state.numpy is not None:
        import numpy as np

        np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda:
        torch.cuda.set_rng_state_all(list(state.torch_cuda))


def _tensor_scalar(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value.detach().cpu().item()), 8)


__all__ = [
    "QualifiedStairRewardProgram",
    "STAIR_REWARD_PROGRAM_QUALIFICATION_SCHEMA_VERSION",
    "STAIR_REWARD_RUNTIME_EVIDENCE_SCHEMA_VERSION",
    "StairCompleteRewardDispatcher",
    "StairRewardEvaluation",
    "StairRewardArtifactIntegrityError",
    "StairRewardOutputContractError",
    "StairRewardProgramError",
    "assert_stair_reward_program_deterministic",
    "evaluate_stair_reward_program",
    "qualify_stair_candidate_reward_program",
    "qualify_stair_reference_reward_program",
]
