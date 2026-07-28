"""Create-once session trust root for acceptance replay."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.binding import (
    EvaluationBindingV1,
    canonical_json_hash,
    is_sha256_checksum,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import (
    ExperimentSubject,
    validate_experiment_subject_id,
)
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    load_strict_json_object,
    strict_json_loads,
)
from true_rl_agent.memory.records import PrivateArtifactRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import BaselineManifest
from true_rl_agent.runner.base import Runner
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.specs import BaselineEvalJobSpec, CheckpointRef, EvalJobSpec
from true_rl_agent.safety.eval_seal import TrustedEvalSeal, build_trusted_eval_seal
from true_rl_agent.tasks.base import TaskPlugin, TaskSpec
from true_rl_agent.training_binding import TrainingBinding
from true_rl_agent.training_programs import ReferenceProgramManifest

SESSION_GENESIS_SCHEMA_VERSION = "true_rl_agent.session_genesis.v2"
BASELINE_EVAL_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.session_genesis.baseline_eval_artifact.v1"
)
REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION = (
    "true_rl_agent.session_genesis.reference_eval_artifact.v1"
)
REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION = (
    "true_rl_agent.reference_qualification_gate.v1"
)
SESSION_GENESIS_FILENAME = "session_genesis.json"
BASELINE_EVAL_ARTIFACT_FILENAME = "baseline_eval_artifact.json"
REFERENCE_EVAL_ARTIFACT_FILENAME = "reference_eval_artifact.json"


@dataclass(frozen=True)
class BootstrapGenesisAnchor:
    anchor_id: str
    initial_best_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_id, str) or not self.anchor_id:
            raise ValueError("bootstrap anchor_id must be non-empty")
        object.__setattr__(
            self,
            "initial_best_score",
            _finite_number(
                self.initial_best_score,
                "bootstrap initial_best_score",
            ),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "BootstrapGenesisAnchor":
        mapping = _exact_mapping(
            value,
            {"kind", "anchor_id", "initial_best_score"},
            "bootstrap genesis anchor",
        )
        if mapping.get("kind") != "bootstrap":
            raise ValueError("bootstrap genesis anchor kind mismatch")
        return cls(
            anchor_id=_nonempty_text(mapping.get("anchor_id"), "bootstrap anchor_id"),
            initial_best_score=_finite_number(
                mapping.get("initial_best_score"),
                "bootstrap initial_best_score",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "bootstrap",
            "anchor_id": self.anchor_id,
            "initial_best_score": self.initial_best_score,
        }


@dataclass(frozen=True)
class BaselineGenesisAnchor:
    baseline_id: str
    baseline_manifest_hash: str
    checkpoint_hash: str
    public_result: Mapping[str, Any]
    public_result_hash: str
    trusted_eval_seal: Mapping[str, Any]
    private_eval_artifact: PrivateArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_id, str) or not self.baseline_id:
            raise ValueError("baseline genesis baseline_id must be non-empty")
        require_sha256_digest(
            self.baseline_manifest_hash,
            "baseline genesis manifest hash",
        )
        require_sha256_checksum(
            self.checkpoint_hash,
            "baseline genesis checkpoint hash",
        )
        if canonical_json_hash(self.public_result) != self.public_result_hash:
            raise ValueError("baseline genesis public result hash mismatch")
        assert_public_projection_safe(self.public_result)
        _validate_private_artifact_ref(
            self.private_eval_artifact,
            expected_kind="baseline_eval_artifact",
            expected_reason="session_genesis_baseline_evidence",
            field_prefix="baseline Genesis",
        )
        seal_hash = self.trusted_eval_seal.get("seal_hash")
        require_sha256_digest(seal_hash, "baseline genesis trusted eval seal hash")
        object.__setattr__(
            self,
            "public_result",
            _freeze_json_value(_json_object(self.public_result, "public_result")),
        )
        object.__setattr__(
            self,
            "trusted_eval_seal",
            _freeze_json_value(
                _json_object(self.trusted_eval_seal, "trusted_eval_seal")
            ),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "BaselineGenesisAnchor":
        mapping = _exact_mapping(
            value,
            {
                "kind",
                "baseline_id",
                "baseline_manifest_hash",
                "checkpoint_hash",
                "public_result",
                "public_result_hash",
                "trusted_eval_seal",
                "private_eval_artifact",
            },
            "baseline genesis anchor",
        )
        if mapping.get("kind") != "policy_baseline":
            raise ValueError("baseline genesis anchor kind mismatch")
        artifact = _exact_mapping(
            mapping.get("private_eval_artifact"),
            {"kind", "uri", "checksum", "reason"},
            "baseline private eval artifact",
        )
        return cls(
            baseline_id=_nonempty_text(
                mapping.get("baseline_id"),
                "baseline genesis baseline_id",
            ),
            baseline_manifest_hash=require_sha256_digest(
                mapping.get("baseline_manifest_hash"),
                "baseline genesis manifest hash",
            ),
            checkpoint_hash=require_sha256_checksum(
                mapping.get("checkpoint_hash"),
                "baseline genesis checkpoint hash",
            ),
            public_result=_json_object(
                mapping.get("public_result"),
                "baseline genesis public_result",
            ),
            public_result_hash=require_sha256_digest(
                mapping.get("public_result_hash"),
                "baseline genesis public result hash",
            ),
            trusted_eval_seal=_json_object(
                mapping.get("trusted_eval_seal"),
                "baseline genesis trusted_eval_seal",
            ),
            private_eval_artifact=PrivateArtifactRef(
                kind=_nonempty_text(
                    artifact.get("kind"),
                    "baseline private artifact kind",
                ),
                uri=_nonempty_text(
                    artifact.get("uri"),
                    "baseline private artifact uri",
                ),
                checksum=require_sha256_checksum(
                    artifact.get("checksum"),
                    "baseline private artifact checksum",
                ),
                reason=_nonempty_text(
                    artifact.get("reason"),
                    "baseline private artifact reason",
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "policy_baseline",
            "baseline_id": self.baseline_id,
            "baseline_manifest_hash": self.baseline_manifest_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "public_result": _json_copy(self.public_result),
            "public_result_hash": self.public_result_hash,
            "trusted_eval_seal": _json_copy(self.trusted_eval_seal),
            "private_eval_artifact": self.private_eval_artifact.to_private_dict(),
        }


@dataclass(frozen=True)
class ReferenceProgramGenesisAnchor:
    reference_manifest: ReferenceProgramManifest
    training_binding: TrainingBinding
    checkpoint_hash: str
    public_result: Mapping[str, Any]
    public_result_hash: str
    trusted_eval_seal: Mapping[str, Any]
    qualification_gate: Mapping[str, Any]
    private_eval_artifact: PrivateArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.reference_manifest, ReferenceProgramManifest):
            raise TypeError("reference Genesis manifest is invalid")
        if not isinstance(self.training_binding, TrainingBinding):
            raise TypeError("reference Genesis training binding is invalid")
        if self.training_binding.initialization_mode != "random":
            raise ValueError("reference Genesis training must use random initialization")
        if (
            self.training_binding.capability_profile_hash
            != self.reference_manifest.capability_profile_hash
        ):
            raise ValueError("reference Genesis capability profile mismatch")
        require_sha256_checksum(
            self.checkpoint_hash,
            "reference Genesis checkpoint hash",
        )
        if canonical_json_hash(self.public_result) != self.public_result_hash:
            raise ValueError("reference Genesis public result hash mismatch")
        assert_public_projection_safe(self.public_result)
        _validate_reference_qualification_gate(self.qualification_gate)
        _validate_reference_seal(
            self.trusted_eval_seal,
            reference_manifest=self.reference_manifest,
            training_binding=self.training_binding,
            checkpoint_hash=self.checkpoint_hash,
        )
        _validate_private_artifact_ref(
            self.private_eval_artifact,
            expected_kind="reference_eval_artifact",
            expected_reason="session_genesis_reference_evidence",
            field_prefix="reference Genesis",
        )
        object.__setattr__(
            self,
            "public_result",
            _freeze_json_value(_json_object(self.public_result, "public_result")),
        )
        object.__setattr__(
            self,
            "trusted_eval_seal",
            _freeze_json_value(
                _json_object(self.trusted_eval_seal, "trusted_eval_seal")
            ),
        )
        object.__setattr__(
            self,
            "qualification_gate",
            _freeze_json_value(
                _json_object(self.qualification_gate, "qualification_gate")
            ),
        )

    @classmethod
    def from_mapping(cls, value: Any) -> "ReferenceProgramGenesisAnchor":
        mapping = _exact_mapping(
            value,
            {
                "kind",
                "reference_manifest",
                "training_binding",
                "checkpoint_hash",
                "public_result",
                "public_result_hash",
                "trusted_eval_seal",
                "qualification_gate",
                "private_eval_artifact",
            },
            "reference genesis anchor",
        )
        if mapping.get("kind") != "reference_program":
            raise ValueError("reference genesis anchor kind mismatch")
        artifact = _exact_mapping(
            mapping.get("private_eval_artifact"),
            {"kind", "uri", "checksum", "reason"},
            "reference private eval artifact",
        )
        return cls(
            reference_manifest=ReferenceProgramManifest.from_mapping(
                mapping.get("reference_manifest")
            ),
            training_binding=TrainingBinding.from_mapping(
                mapping.get("training_binding")
            ),
            checkpoint_hash=require_sha256_checksum(
                mapping.get("checkpoint_hash"),
                "reference Genesis checkpoint hash",
            ),
            public_result=_json_object(
                mapping.get("public_result"),
                "reference Genesis public_result",
            ),
            public_result_hash=require_sha256_digest(
                mapping.get("public_result_hash"),
                "reference Genesis public result hash",
            ),
            trusted_eval_seal=_json_object(
                mapping.get("trusted_eval_seal"),
                "reference Genesis trusted_eval_seal",
            ),
            qualification_gate=_json_object(
                mapping.get("qualification_gate"),
                "reference Genesis qualification_gate",
            ),
            private_eval_artifact=PrivateArtifactRef(
                kind=_nonempty_text(
                    artifact.get("kind"),
                    "reference private artifact kind",
                ),
                uri=_nonempty_text(
                    artifact.get("uri"),
                    "reference private artifact uri",
                ),
                checksum=require_sha256_checksum(
                    artifact.get("checksum"),
                    "reference private artifact checksum",
                ),
                reason=_nonempty_text(
                    artifact.get("reason"),
                    "reference private artifact reason",
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "reference_program",
            "reference_manifest": self.reference_manifest.to_public_dict(),
            "training_binding": self.training_binding.to_private_dict(),
            "checkpoint_hash": self.checkpoint_hash,
            "public_result": _json_copy(self.public_result),
            "public_result_hash": self.public_result_hash,
            "trusted_eval_seal": _json_copy(self.trusted_eval_seal),
            "qualification_gate": _json_copy(self.qualification_gate),
            "private_eval_artifact": self.private_eval_artifact.to_private_dict(),
        }


GenesisAnchor = (
    BootstrapGenesisAnchor
    | BaselineGenesisAnchor
    | ReferenceProgramGenesisAnchor
)


@dataclass(frozen=True)
class SessionGenesisArtifact:
    session_id: str
    task_id: str
    backend_id: str
    policy_initialization_kind: str
    runtime_snapshot_hash: str
    evaluation_binding: EvaluationBindingV1
    anchor: GenesisAnchor
    session_genesis_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "session_id",
            "task_id",
            "backend_id",
            "policy_initialization_kind",
        ):
            if not isinstance(getattr(self, field_name), str) or not getattr(
                self,
                field_name,
            ):
                raise ValueError(f"session genesis {field_name} must be non-empty")
        validate_experiment_subject_id(self.session_id)
        if self.policy_initialization_kind not in {"random", "warm_start"}:
            raise ValueError("session genesis policy initialization kind is invalid")
        require_sha256_digest(
            self.runtime_snapshot_hash,
            "session genesis runtime_snapshot_hash",
        )
        if not isinstance(self.evaluation_binding, EvaluationBindingV1):
            raise TypeError("session genesis evaluation_binding must be EvaluationBindingV1")
        if not isinstance(
            self.anchor,
            BootstrapGenesisAnchor
            | BaselineGenesisAnchor
            | ReferenceProgramGenesisAnchor,
        ):
            raise TypeError("session genesis anchor type is invalid")
        canonical_hash = canonical_json_hash(self._hash_payload())
        if self.session_genesis_hash is not None:
            require_sha256_digest(
                self.session_genesis_hash,
                "session_genesis_hash",
            )
            if self.session_genesis_hash != canonical_hash:
                raise ValueError("session genesis hash mismatch")
        object.__setattr__(self, "session_genesis_hash", canonical_hash)

    @property
    def genesis_kind(self) -> str:
        if isinstance(self.anchor, BaselineGenesisAnchor):
            return "policy_baseline"
        if isinstance(self.anchor, ReferenceProgramGenesisAnchor):
            return "reference_program"
        return "bootstrap"

    @classmethod
    def from_mapping(cls, value: Any) -> "SessionGenesisArtifact":
        mapping = _exact_mapping(
            value,
            {
                "schema_version",
                "session_id",
                "task_id",
                "backend_id",
                "policy_initialization_kind",
                "runtime_snapshot_hash",
                "evaluation_binding",
                "anchor",
                "session_genesis_hash",
            },
            "session genesis",
        )
        if mapping.get("schema_version") != SESSION_GENESIS_SCHEMA_VERSION:
            raise ValueError("session genesis schema_version mismatch")
        raw_anchor = _json_object(mapping.get("anchor"), "session genesis anchor")
        anchor_kind = raw_anchor.get("kind")
        if anchor_kind == "policy_baseline":
            anchor: GenesisAnchor = BaselineGenesisAnchor.from_mapping(raw_anchor)
        elif anchor_kind == "reference_program":
            anchor = ReferenceProgramGenesisAnchor.from_mapping(raw_anchor)
        elif anchor_kind == "bootstrap":
            anchor = BootstrapGenesisAnchor.from_mapping(raw_anchor)
        else:
            raise ValueError("session genesis anchor kind is invalid")
        return cls(
            session_id=_nonempty_text(mapping.get("session_id"), "session_id"),
            task_id=_nonempty_text(mapping.get("task_id"), "task_id"),
            backend_id=_nonempty_text(mapping.get("backend_id"), "backend_id"),
            policy_initialization_kind=_nonempty_text(
                mapping.get("policy_initialization_kind"),
                "policy_initialization_kind",
            ),
            runtime_snapshot_hash=require_sha256_digest(
                mapping.get("runtime_snapshot_hash"),
                "runtime_snapshot_hash",
            ),
            evaluation_binding=EvaluationBindingV1.from_mapping(
                mapping.get("evaluation_binding")
            ),
            anchor=anchor,
            session_genesis_hash=require_sha256_digest(
                mapping.get("session_genesis_hash"),
                "session_genesis_hash",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        assert self.session_genesis_hash is not None
        return {
            **self._hash_payload(),
            "session_genesis_hash": self.session_genesis_hash,
        }

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SESSION_GENESIS_SCHEMA_VERSION,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "backend_id": self.backend_id,
            "policy_initialization_kind": self.policy_initialization_kind,
            "runtime_snapshot_hash": self.runtime_snapshot_hash,
            "evaluation_binding": self.evaluation_binding.to_public_dict(),
            "anchor": self.anchor.to_dict(),
        }


class VerifiedSessionGenesis:
    """Opaque result of contextual Genesis verification."""

    __slots__ = ("_artifact", "_artifact_path", "_initial_current_best")
    _TOKEN = object()

    def __init__(
        self,
        *,
        artifact: SessionGenesisArtifact,
        artifact_path: Path,
        initial_current_best: CurrentBestEvalStats,
        _token: object,
    ) -> None:
        if _token is not self._TOKEN:
            raise TypeError("VerifiedSessionGenesis must come from the Genesis validator")
        object.__setattr__(self, "_artifact", artifact)
        object.__setattr__(self, "_artifact_path", artifact_path)
        object.__setattr__(self, "_initial_current_best", initial_current_best)

    def __setattr__(self, name: str, value: object) -> None:
        _ = name, value
        raise AttributeError("VerifiedSessionGenesis is immutable")

    @classmethod
    def _validated(
        cls,
        *,
        artifact: SessionGenesisArtifact,
        artifact_path: Path,
        initial_current_best: CurrentBestEvalStats,
    ) -> "VerifiedSessionGenesis":
        return cls(
            artifact=artifact,
            artifact_path=artifact_path,
            initial_current_best=initial_current_best,
            _token=cls._TOKEN,
        )

    @property
    def artifact(self) -> SessionGenesisArtifact:
        return self._artifact

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    @property
    def session_genesis_hash(self) -> str:
        assert self._artifact.session_genesis_hash is not None
        return self._artifact.session_genesis_hash

    @property
    def evaluation_binding(self) -> EvaluationBindingV1:
        return self._artifact.evaluation_binding

    @property
    def initial_current_best(self) -> CurrentBestEvalStats:
        return self._initial_current_best

    def assert_file_unchanged(self, output_dir: Path) -> None:
        expected = output_dir.resolve() / SESSION_GENESIS_FILENAME
        if self._artifact_path.resolve() != expected:
            raise ValueError("verified session genesis belongs to a different output directory")
        try:
            current_bytes = expected.read_bytes()
        except OSError as exc:
            raise ValueError("session genesis could not be read after verification") from exc
        expected_bytes = canonical_json_bytes(
            self._artifact.to_dict(),
            trailing_newline=True,
        )
        if current_bytes != expected_bytes:
            raise ValueError("session genesis bytes changed after verification")
        current = SessionGenesisArtifact.from_mapping(_read_json(expected))
        if current.to_dict() != self._artifact.to_dict():
            raise ValueError("session genesis changed after verification")

    def baseline_anchor_public(self) -> dict[str, Any] | None:
        if not isinstance(self._artifact.anchor, BaselineGenesisAnchor):
            return None
        return {
            "anchor_kind": "baseline",
            "baseline_id": self._artifact.anchor.baseline_id,
            "public_score": self._initial_current_best.score_for_comparison(),
            "task_protocol_hash": self.evaluation_binding.task_protocol_hash,
            "eval_protocol_hash": self.evaluation_binding.eval_protocol_hash,
            "acceptance_protocol_hash": (
                self.evaluation_binding.acceptance_protocol_hash
            ),
            "evaluation_binding_hash": self.evaluation_binding.binding_hash,
        }

    def reference_anchor_public(self) -> dict[str, Any] | None:
        anchor = self._artifact.anchor
        if not isinstance(anchor, ReferenceProgramGenesisAnchor):
            return None
        manifest = anchor.reference_manifest
        return {
            "anchor_kind": "reference_program",
            "reference_id": manifest.reference_id,
            "reference_manifest_hash": manifest.manifest_hash,
            "program_source_checksum": manifest.source_checksum,
            "training_binding_hash": anchor.training_binding.binding_hash,
            "capability_profile_hash": manifest.capability_profile_hash,
            "public_score": self._initial_current_best.score_for_comparison(),
            "task_protocol_hash": self.evaluation_binding.task_protocol_hash,
            "eval_protocol_hash": self.evaluation_binding.eval_protocol_hash,
            "acceptance_protocol_hash": (
                self.evaluation_binding.acceptance_protocol_hash
            ),
            "evaluation_binding_hash": self.evaluation_binding.binding_hash,
        }

    def reference_program_source(self) -> str | None:
        anchor = self._artifact.anchor
        if not isinstance(anchor, ReferenceProgramGenesisAnchor):
            return None
        payload = _validate_private_reference_artifact(
            anchor=anchor,
            expected_path=(
                self._artifact_path.parent / REFERENCE_EVAL_ARTIFACT_FILENAME
            ),
        )
        source = payload.get("program_source")
        if not isinstance(source, str) or not source:
            raise ValueError("reference Genesis private source is missing")
        return source


def prepare_session_genesis(
    *,
    output_dir: Path,
    session_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    runner: Runner,
    policy_initialization: PolicyInitialization,
    runtime_snapshot: Mapping[str, Any],
    baseline_manifest: BaselineManifest | None = None,
    reference_program_manifest: ReferenceProgramManifest | None = None,
    bootstrap_initial_best_score: float | None = None,
) -> VerifiedSessionGenesis:
    """Create a Genesis once, or validate the existing one without rerunning eval."""

    if not isinstance(runtime_snapshot, Mapping):
        raise TypeError("runtime_snapshot must be an object")
    session_id = validate_experiment_subject_id(session_id)
    if (
        baseline_manifest is not None
        and reference_program_manifest is not None
    ):
        raise ValueError("Genesis cannot carry baseline and reference manifests")
    if baseline_manifest is not None and bootstrap_initial_best_score is not None:
        raise ValueError(
            "baseline Genesis cannot also declare a bootstrap initial score"
        )
    if (
        reference_program_manifest is not None
        and bootstrap_initial_best_score is not None
    ):
        raise ValueError(
            "reference Genesis cannot also declare a bootstrap initial score"
        )
    normalized_bootstrap_score = (
        _finite_number(
            bootstrap_initial_best_score,
            "bootstrap initial_best_score",
        )
        if bootstrap_initial_best_score is not None
        else None
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    genesis_path = output_dir / SESSION_GENESIS_FILENAME
    if not genesis_path.exists():
        for partial_path in (
            output_dir / "ledger.jsonl",
            output_dir / BASELINE_EVAL_ARTIFACT_FILENAME,
            output_dir / REFERENCE_EVAL_ARTIFACT_FILENAME,
        ):
            if partial_path.exists():
                raise ValueError(
                    "session has durable state but no Genesis; refusing implicit rebuild: "
                    + str(partial_path)
                )
    evaluation_binding = backend_adapter.resolve_session_evaluation_binding(
        session_id=session_id,
        session_dir=output_dir,
        task_spec=task_plugin.task_spec,
        allow_create=not genesis_path.exists(),
    )
    if not isinstance(evaluation_binding, EvaluationBindingV1):
        raise TypeError(
            "backend session evaluation binding must be EvaluationBindingV1"
        )
    runtime_snapshot_hash = canonical_json_hash(_json_copy(runtime_snapshot))
    context = _GenesisContext(
        session_id=session_id,
        task_plugin=task_plugin,
        backend_adapter=backend_adapter,
        policy_initialization=policy_initialization,
        runtime_snapshot_hash=runtime_snapshot_hash,
        baseline_manifest=baseline_manifest,
        reference_program_manifest=reference_program_manifest,
        bootstrap_initial_best_score=normalized_bootstrap_score,
        evaluation_binding=evaluation_binding,
    )
    if genesis_path.exists():
        return _load_verified_session_genesis(genesis_path, context=context)

    if context.genesis_anchor_kind == "reference_program":
        raise ValueError(
            "reference program Genesis materialization is unavailable; "
            "a verified reference Genesis must be created by the reference runner"
        )
    if context.genesis_anchor_kind == "policy_baseline":
        artifact = _create_baseline_genesis(
            output_dir=output_dir,
            context=context,
            runner=runner,
        )
    else:
        artifact = _create_bootstrap_genesis(
            context=context,
        )
    _atomic_create_json(genesis_path, artifact.to_dict())
    return _load_verified_session_genesis(genesis_path, context=context)


def verify_session_genesis_for_runtime(
    genesis: VerifiedSessionGenesis,
    *,
    output_dir: Path,
    session_id: str,
    task_plugin: TaskPlugin,
    backend_adapter: BackendAdapter,
    policy_initialization: PolicyInitialization,
) -> None:
    """Cheap live-loop guard after contextual preparation has succeeded."""

    if not isinstance(genesis, VerifiedSessionGenesis):
        raise TypeError("run requires a VerifiedSessionGenesis")
    artifact = genesis.artifact
    if artifact.session_id != session_id:
        raise ValueError("session Genesis session_id mismatch")
    if artifact.task_id != task_plugin.task_spec.task_id:
        raise ValueError("session Genesis task_id mismatch")
    if artifact.backend_id != backend_adapter.backend_id:
        raise ValueError("session Genesis backend_id mismatch")
    if artifact.policy_initialization_kind != policy_initialization.kind:
        raise ValueError("session Genesis policy initialization mismatch")
    expected_anchor_kind = task_plugin.task_spec.genesis_anchor_kind_for(
        policy_initialization.kind
    )
    if artifact.genesis_kind != expected_anchor_kind:
        raise ValueError("session Genesis anchor policy mismatch")
    binding = backend_adapter.resolve_session_evaluation_binding(
        session_id=session_id,
        session_dir=output_dir.resolve(),
        task_spec=task_plugin.task_spec,
        allow_create=False,
    )
    if not isinstance(binding, EvaluationBindingV1):
        raise TypeError(
            "backend session evaluation binding must be EvaluationBindingV1"
        )
    backend_adapter.validate_session_evaluation_binding(
        task_spec=task_plugin.task_spec,
        evaluation_binding=binding,
    )
    if artifact.evaluation_binding != binding:
        raise ValueError("session Genesis evaluation binding mismatch")
    _assert_policy_initialization_matches_anchor(policy_initialization, artifact.anchor)
    genesis.assert_file_unchanged(output_dir)


@dataclass(frozen=True)
class _GenesisContext:
    session_id: str
    task_plugin: TaskPlugin
    backend_adapter: BackendAdapter
    policy_initialization: PolicyInitialization
    runtime_snapshot_hash: str
    baseline_manifest: BaselineManifest | None
    reference_program_manifest: ReferenceProgramManifest | None
    bootstrap_initial_best_score: float | None
    evaluation_binding: EvaluationBindingV1

    def __post_init__(self) -> None:
        binding = self.evaluation_binding
        if binding.task_protocol_hash != self.task_spec.resolved_task_protocol_hash():
            raise ValueError("Genesis binding task protocol mismatch")
        if binding.acceptance_protocol_hash != self.acceptance_protocol.protocol_hash:
            raise ValueError("Genesis binding acceptance protocol mismatch")
        self.backend_adapter.validate_session_evaluation_binding(
            task_spec=self.task_spec,
            evaluation_binding=binding,
        )
        if self.genesis_anchor_kind == "bootstrap":
            if self.baseline_manifest is not None or self.reference_program_manifest is not None:
                raise ValueError("bootstrap Genesis cannot carry a manifest")
        elif self.genesis_anchor_kind == "policy_baseline":
            if self.baseline_manifest is None or self.reference_program_manifest is not None:
                raise ValueError("policy-baseline Genesis requires only a baseline manifest")
        elif self.genesis_anchor_kind == "reference_program":
            if self.reference_program_manifest is None or self.baseline_manifest is not None:
                raise ValueError("reference Genesis requires only a reference manifest")
            manifest = self.reference_program_manifest
            if manifest.task_id != self.task_spec.task_id:
                raise ValueError("reference Genesis manifest task_id mismatch")
            if (
                manifest.capability_profile_hash
                != self.task_spec.candidate_capability_profile.profile_hash
            ):
                raise ValueError("reference Genesis manifest capability profile mismatch")
        else:
            raise ValueError("Genesis anchor policy is invalid")

    @property
    def task_spec(self) -> TaskSpec:
        return self.task_plugin.task_spec

    @property
    def acceptance_protocol(self) -> AcceptanceProtocol:
        protocol = self.task_spec.acceptance_protocol
        if protocol is None:
            raise ValueError("session Genesis requires a fixed acceptance protocol")
        return protocol

    @property
    def genesis_anchor_kind(self) -> str:
        return self.task_spec.genesis_anchor_kind_for(
            self.policy_initialization.kind
        )

def _create_bootstrap_genesis(
    *,
    context: _GenesisContext,
) -> SessionGenesisArtifact:
    if context.genesis_anchor_kind != "bootstrap":
        raise ValueError("task policy does not allow a bootstrap Genesis")
    if context.policy_initialization.kind != "random":
        raise ValueError("bootstrap Genesis requires random policy initialization")
    if context.bootstrap_initial_best_score is None:
        raise ValueError("bootstrap Genesis requires initial_best_score")
    return SessionGenesisArtifact(
        session_id=context.session_id,
        task_id=context.task_spec.task_id,
        backend_id=context.backend_adapter.backend_id,
        policy_initialization_kind=context.policy_initialization.kind,
        runtime_snapshot_hash=context.runtime_snapshot_hash,
        evaluation_binding=context.evaluation_binding,
        anchor=BootstrapGenesisAnchor(
            anchor_id=context.session_id,
            initial_best_score=context.bootstrap_initial_best_score,
        ),
    )


def _create_baseline_genesis(
    *,
    output_dir: Path,
    context: _GenesisContext,
    runner: Runner,
) -> SessionGenesisArtifact:
    manifest = context.baseline_manifest
    if manifest is None:
        raise ValueError("baseline Genesis requires a baseline manifest")
    eval_spec = _materialize_baseline_eval(context, manifest)
    public_result = runner.run_baseline_eval(eval_spec)
    seal = _build_baseline_seal(
        eval_spec=eval_spec,
        task_spec=context.task_spec,
        public_result=public_result,
    )
    current_best = CurrentBestEvalStats.from_baseline_public_result_mapping(
        baseline_id=manifest.baseline_id,
        baseline_manifest_hash=str(manifest.manifest_hash),
        checkpoint_hash=manifest.checkpoint.checksum,
        public_result=public_result.to_public_dict(),
        acceptance_protocol=context.acceptance_protocol,
        expected_evaluation_binding=context.evaluation_binding,
        capability_profile_hash=(
            context.task_spec.candidate_capability_profile.profile_hash
        ),
    )
    if not current_best.has_full_eval_stats():
        raise ValueError("baseline evaluation cannot establish a full current-best")

    private_payload = _baseline_eval_artifact_payload(
        session_id=context.session_id,
        backend_id=context.backend_adapter.backend_id,
        eval_spec=eval_spec,
        public_result=public_result.to_public_dict(),
        trusted_eval_seal=seal.to_public_dict(),
    )
    private_path = output_dir / BASELINE_EVAL_ARTIFACT_FILENAME
    _atomic_create_json(private_path, private_payload)
    private_checksum = file_sha256(private_path)
    if not is_sha256_checksum(private_checksum):
        raise ValueError("baseline eval artifact checksum could not be computed")
    anchor = BaselineGenesisAnchor(
        baseline_id=manifest.baseline_id,
        baseline_manifest_hash=str(manifest.manifest_hash),
        checkpoint_hash=manifest.checkpoint.checksum,
        public_result=public_result.to_public_dict(),
        public_result_hash=canonical_json_hash(public_result.to_public_dict()),
        trusted_eval_seal=seal.to_public_dict(),
        private_eval_artifact=PrivateArtifactRef(
            kind="baseline_eval_artifact",
            uri=str(private_path),
            checksum=private_checksum,
            reason="session_genesis_baseline_evidence",
        ),
    )
    return SessionGenesisArtifact(
        session_id=context.session_id,
        task_id=context.task_spec.task_id,
        backend_id=context.backend_adapter.backend_id,
        policy_initialization_kind=context.policy_initialization.kind,
        runtime_snapshot_hash=context.runtime_snapshot_hash,
        evaluation_binding=context.evaluation_binding,
        anchor=anchor,
    )


def _load_verified_session_genesis(
    path: Path,
    *,
    context: _GenesisContext,
) -> VerifiedSessionGenesis:
    artifact = SessionGenesisArtifact.from_mapping(_read_json(path))
    if artifact.session_id != context.session_id:
        raise ValueError("session Genesis session_id drift")
    if artifact.task_id != context.task_spec.task_id:
        raise ValueError("session Genesis task_id drift")
    if artifact.backend_id != context.backend_adapter.backend_id:
        raise ValueError("session Genesis backend_id drift")
    if artifact.policy_initialization_kind != context.policy_initialization.kind:
        raise ValueError("session Genesis policy initialization drift")
    if artifact.runtime_snapshot_hash != context.runtime_snapshot_hash:
        raise ValueError("session Genesis runtime snapshot drift")
    if artifact.evaluation_binding != context.evaluation_binding:
        raise ValueError("session Genesis evaluation binding drift")
    if artifact.genesis_kind != context.genesis_anchor_kind:
        raise ValueError("session Genesis anchor policy drift")
    _assert_policy_initialization_matches_anchor(
        context.policy_initialization,
        artifact.anchor,
    )

    if isinstance(artifact.anchor, BootstrapGenesisAnchor):
        if context.baseline_manifest is not None:
            raise ValueError("bootstrap Genesis cannot replace a baseline manifest")
        if context.genesis_anchor_kind != "bootstrap":
            raise ValueError("task policy does not allow a bootstrap Genesis")
        if context.bootstrap_initial_best_score is None:
            raise ValueError("bootstrap Genesis requires initial_best_score")
        if (
            artifact.anchor.initial_best_score
            != context.bootstrap_initial_best_score
        ):
            raise ValueError("session Genesis bootstrap initial score drift")
        initial_current_best = CurrentBestEvalStats.bootstrap(
            initial_best_score=artifact.anchor.initial_best_score,
            acceptance_protocol=context.acceptance_protocol,
            evaluation_binding=artifact.evaluation_binding,
            anchor_id=artifact.anchor.anchor_id,
        )
    elif isinstance(artifact.anchor, BaselineGenesisAnchor):
        manifest = context.baseline_manifest
        if manifest is None:
            raise ValueError("baseline Genesis requires the expected baseline manifest")
        _assert_manifest_matches_anchor(manifest, artifact.anchor)
        eval_spec = _materialize_baseline_eval(context, manifest)
        public_result = PublicEvalResult(
            status=artifact.anchor.public_result.get("status"),
            public_score=artifact.anchor.public_result.get("public_score"),
            metrics=_json_object(
                artifact.anchor.public_result.get("metrics"),
                "baseline Genesis public result metrics",
            ),
        )
        expected_seal = _build_baseline_seal(
            eval_spec=eval_spec,
            task_spec=context.task_spec,
            public_result=public_result,
        )
        if artifact.anchor.trusted_eval_seal != expected_seal.to_public_dict():
            raise ValueError("baseline Genesis trusted eval seal mismatch")
        _validate_private_baseline_artifact(
            anchor=artifact.anchor,
            expected_path=path.parent / BASELINE_EVAL_ARTIFACT_FILENAME,
            expected_payload=_baseline_eval_artifact_payload(
                session_id=context.session_id,
                backend_id=context.backend_adapter.backend_id,
                eval_spec=eval_spec,
                public_result=public_result.to_public_dict(),
                trusted_eval_seal=expected_seal.to_public_dict(),
            ),
        )
        initial_current_best = CurrentBestEvalStats.from_baseline_public_result_mapping(
            baseline_id=manifest.baseline_id,
            baseline_manifest_hash=str(manifest.manifest_hash),
            checkpoint_hash=manifest.checkpoint.checksum,
            public_result=public_result.to_public_dict(),
            acceptance_protocol=context.acceptance_protocol,
            expected_evaluation_binding=context.evaluation_binding,
            capability_profile_hash=(
                context.task_spec.candidate_capability_profile.profile_hash
            ),
        )
        if not initial_current_best.has_full_eval_stats():
            raise ValueError("baseline Genesis current-best is incomplete")
    else:
        manifest = context.reference_program_manifest
        if manifest is None:
            raise ValueError(
                "reference Genesis requires the expected reference manifest"
            )
        anchor = artifact.anchor
        if anchor.reference_manifest != manifest:
            raise ValueError("session Genesis reference manifest drift")
        if (
            anchor.training_binding.capability_profile_hash
            != context.task_spec.candidate_capability_profile.profile_hash
        ):
            raise ValueError("session Genesis reference capability profile drift")
        public_result = PublicEvalResult(
            status=anchor.public_result.get("status"),
            public_score=anchor.public_result.get("public_score"),
            metrics=_json_object(
                anchor.public_result.get("metrics"),
                "reference Genesis public result metrics",
            ),
        )
        _validate_reference_public_result(
            anchor=anchor,
            public_result=public_result,
            evaluation_binding=context.evaluation_binding,
        )
        private_payload = _validate_private_reference_artifact(
            anchor=anchor,
            expected_path=path.parent / REFERENCE_EVAL_ARTIFACT_FILENAME,
        )
        eval_spec = _reference_eval_spec_from_private_artifact(
            private_payload,
            anchor=anchor,
            evaluation_binding=context.evaluation_binding,
        )
        expected_seal = _build_reference_seal(
            eval_spec=eval_spec,
            task_spec=context.task_spec,
            public_result=public_result,
        )
        if anchor.trusted_eval_seal != expected_seal.to_public_dict():
            raise ValueError("reference Genesis trusted eval seal mismatch")
        initial_current_best = CurrentBestEvalStats.from_reference_public_result_mapping(
            reference_id=manifest.reference_id,
            reference_manifest_hash=str(manifest.manifest_hash),
            program_source_checksum=manifest.source_checksum,
            training_binding_hash=str(anchor.training_binding.binding_hash),
            capability_profile_hash=manifest.capability_profile_hash,
            checkpoint_hash=anchor.checkpoint_hash,
            public_result=public_result.to_public_dict(),
            acceptance_protocol=context.acceptance_protocol,
            expected_evaluation_binding=context.evaluation_binding,
        )
        if not initial_current_best.has_full_eval_stats():
            raise ValueError("reference Genesis current-best is incomplete")
    return VerifiedSessionGenesis._validated(
        artifact=artifact,
        artifact_path=path.resolve(),
        initial_current_best=initial_current_best,
    )


def _materialize_baseline_eval(
    context: _GenesisContext,
    manifest: BaselineManifest,
) -> BaselineEvalJobSpec:
    spec = context.backend_adapter.materialize_baseline_eval(
        session_id=context.session_id,
        task_spec=context.task_spec,
        baseline_manifest=manifest,
        evaluation_binding=context.evaluation_binding,
    )
    if spec.evaluation_binding != context.evaluation_binding:
        raise ValueError("baseline eval spec evaluation binding drift")
    if spec.checkpoint != manifest.checkpoint:
        raise ValueError("baseline eval spec checkpoint drift")
    return spec


def _build_baseline_seal(
    *,
    eval_spec: BaselineEvalJobSpec,
    task_spec: TaskSpec,
    public_result: PublicEvalResult,
) -> TrustedEvalSeal:
    metrics = public_result.metrics
    sealed_gate = _json_object(
        metrics.get("sealed_validation_gate"),
        "baseline sealed_validation_gate",
    )
    return build_trusted_eval_seal(
        eval_spec=eval_spec,
        task_invariants=task_spec.invariants,
        checkpoint_safety_report=metrics.get("checkpoint_safety_report"),
        candidate_isolation_status=_nonempty_text(
            sealed_gate.get("candidate_isolation_status"),
            "baseline candidate isolation status",
        ),
    )


def _baseline_eval_artifact_payload(
    *,
    session_id: str,
    backend_id: str,
    eval_spec: BaselineEvalJobSpec,
    public_result: Mapping[str, Any],
    trusted_eval_seal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": BASELINE_EVAL_ARTIFACT_SCHEMA_VERSION,
        "session_id": session_id,
        "task_id": eval_spec.task_id,
        "backend_id": backend_id,
        "eval_job": eval_spec.to_public_dict(),
        "public_result": _json_copy(public_result),
        "trusted_eval_seal": _json_copy(trusted_eval_seal),
    }


def _validate_private_baseline_artifact(
    *,
    anchor: BaselineGenesisAnchor,
    expected_path: Path,
    expected_payload: Mapping[str, Any],
) -> None:
    ref = anchor.private_eval_artifact
    path = Path(ref.uri)
    if path.resolve() != expected_path.resolve():
        raise ValueError("baseline Genesis private artifact path drift")
    actual_checksum = file_sha256(path)
    if actual_checksum != ref.checksum:
        raise ValueError("baseline Genesis private artifact checksum mismatch")
    if _read_json(path) != expected_payload:
        raise ValueError("baseline Genesis private artifact content mismatch")


def _validate_reference_qualification_gate(value: Mapping[str, Any]) -> None:
    gate = _exact_mapping(
        value,
        {"schema_version", "status", "protocol_hash", "evidence_hash"},
        "reference qualification gate",
    )
    if gate.get("schema_version") != REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION:
        raise ValueError("reference qualification gate schema_version mismatch")
    if gate.get("status") != "pass":
        raise ValueError("reference qualification gate must pass")
    require_sha256_digest(
        gate.get("protocol_hash"),
        "reference qualification protocol hash",
    )
    require_sha256_digest(
        gate.get("evidence_hash"),
        "reference qualification evidence hash",
    )


def _validate_reference_seal(
    value: Mapping[str, Any],
    *,
    reference_manifest: ReferenceProgramManifest,
    training_binding: TrainingBinding,
    checkpoint_hash: str,
) -> None:
    seal = _json_object(value, "reference trusted eval seal")
    expected = {
        "subject_kind": "reference",
        "subject_id": reference_manifest.reference_id,
        "subject_manifest_hash": reference_manifest.manifest_hash,
        "program_source_checksum": reference_manifest.source_checksum,
        "training_binding_hash": training_binding.binding_hash,
        "capability_profile_hash": reference_manifest.capability_profile_hash,
        "checkpoint_hash": checkpoint_hash,
        "safe_load_status": "pass",
        "checkpoint_safety_status": "pass",
        "candidate_isolation_status": "pass",
        "candidate_code_loaded": False,
    }
    for field_name, expected_value in expected.items():
        if seal.get(field_name) != expected_value:
            raise ValueError(
                f"reference trusted eval seal {field_name} mismatch"
            )
    for field_name in (
        "eval_config_hash",
        "evaluation_binding_hash",
        "task_invariant_hash",
        "checkpoint_safety_report_hash",
        "seal_hash",
    ):
        require_sha256_digest(
            seal.get(field_name),
            f"reference trusted eval seal {field_name}",
        )


def _validate_reference_public_result(
    *,
    anchor: ReferenceProgramGenesisAnchor,
    public_result: PublicEvalResult,
    evaluation_binding: EvaluationBindingV1,
) -> None:
    metrics = public_result.metrics
    if EvaluationBindingV1.from_mapping(
        metrics.get("evaluation_binding")
    ) != evaluation_binding:
        raise ValueError("reference Genesis public evaluation binding mismatch")
    expected_subject = ExperimentSubject.reference(
        anchor.reference_manifest.reference_id
    )
    if ExperimentSubject.from_mapping(
        metrics.get("evaluation_subject")
    ) != expected_subject:
        raise ValueError("reference Genesis public subject mismatch")
    expected_lineage = {
        "subject_manifest_hash": anchor.reference_manifest.manifest_hash,
        "program_source_checksum": anchor.reference_manifest.source_checksum,
        "training_binding_hash": anchor.training_binding.binding_hash,
        "capability_profile_hash": (
            anchor.reference_manifest.capability_profile_hash
        ),
    }
    for field_name, expected_value in expected_lineage.items():
        if metrics.get(field_name) != expected_value:
            raise ValueError(
                f"reference Genesis public {field_name} mismatch"
            )
    gate = _json_object(
        metrics.get("reference_capability_gate"),
        "reference public capability gate",
    )
    if gate != _json_copy(anchor.qualification_gate):
        raise ValueError("reference Genesis qualification gate projection mismatch")
    if gate.get("protocol_hash") != evaluation_binding.task_protocol_hash:
        raise ValueError("reference Genesis qualification protocol mismatch")
    acceptance = _json_object(
        metrics.get("acceptance"),
        "reference public acceptance",
    )
    required_gates = acceptance.get("required_pass_gates")
    if (
        not isinstance(required_gates, list)
        or "reference_capability_gate" not in required_gates
    ):
        raise ValueError(
            "reference Genesis acceptance must require capability gate"
        )


def _validate_private_reference_artifact(
    *,
    anchor: ReferenceProgramGenesisAnchor,
    expected_path: Path,
) -> dict[str, Any]:
    ref = anchor.private_eval_artifact
    path = Path(ref.uri)
    if path.resolve() != expected_path.resolve():
        raise ValueError("reference Genesis private artifact path drift")
    actual_checksum = file_sha256(path)
    if actual_checksum != ref.checksum:
        raise ValueError("reference Genesis private artifact checksum mismatch")
    payload = _exact_mapping(
        _read_json(path),
        {
            "schema_version",
            "reference_manifest",
            "training_binding",
            "eval_job",
            "checkpoint_hash",
            "public_result",
            "trusted_eval_seal",
            "qualification_gate",
            "program_source",
        },
        "reference Genesis private artifact",
    )
    if payload.get("schema_version") != REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("reference private artifact schema_version mismatch")
    expected_values = {
        "reference_manifest": anchor.reference_manifest.to_public_dict(),
        "training_binding": anchor.training_binding.to_private_dict(),
        "checkpoint_hash": anchor.checkpoint_hash,
        "public_result": _json_copy(anchor.public_result),
        "trusted_eval_seal": _json_copy(anchor.trusted_eval_seal),
        "qualification_gate": _json_copy(anchor.qualification_gate),
    }
    for field_name, expected_value in expected_values.items():
        if payload.get(field_name) != expected_value:
            raise ValueError(
                f"reference private artifact {field_name} mismatch"
            )
    source = payload.get("program_source")
    if not isinstance(source, str) or not source:
        raise ValueError("reference private artifact program_source is invalid")
    source_bytes = source.encode("utf-8")
    if len(source_bytes) != anchor.reference_manifest.source_size_bytes:
        raise ValueError("reference private artifact source size mismatch")
    import hashlib

    checksum = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
    if checksum != anchor.reference_manifest.source_checksum:
        raise ValueError("reference private artifact source checksum mismatch")
    return dict(payload)


def _reference_eval_spec_from_private_artifact(
    payload: Mapping[str, Any],
    *,
    anchor: ReferenceProgramGenesisAnchor,
    evaluation_binding: EvaluationBindingV1,
) -> EvalJobSpec:
    raw = _exact_mapping(
        payload.get("eval_job"),
        {
            "job_id",
            "task_id",
            "subject",
            "subject_manifest_hash",
            "program_source_checksum",
            "training_binding_hash",
            "capability_profile_hash",
            "schema_version",
            "eval_protocol_version",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "eval_num_envs",
            "eval_protocol_hash",
            "evaluation_binding",
            "checkpoint",
        },
        "reference private eval job",
    )
    protocol = EvaluationProtocol.from_mapping(
        {
            key: raw[key]
            for key in (
                "schema_version",
                "eval_protocol_version",
                "public_seed_count",
                "public_accept_seed_count",
                "sealed_validation_seed_count",
                "eval_num_envs",
                "eval_protocol_hash",
            )
        }
    )
    if EvaluationBindingV1.from_mapping(
        raw.get("evaluation_binding")
    ) != evaluation_binding:
        raise ValueError("reference private eval job binding mismatch")
    checkpoint = _exact_mapping(
        raw.get("checkpoint"),
        {"uri", "checksum", "role", "sealed"},
        "reference private eval checkpoint",
    )
    spec = EvalJobSpec(
        job_id=_nonempty_text(raw.get("job_id"), "reference eval job_id"),
        task_id=_nonempty_text(raw.get("task_id"), "reference eval task_id"),
        subject=ExperimentSubject.from_mapping(raw.get("subject")),
        subject_manifest_hash=require_sha256_digest(
            raw.get("subject_manifest_hash"),
            "reference eval subject_manifest_hash",
        ),
        program_source_checksum=require_sha256_checksum(
            raw.get("program_source_checksum"),
            "reference eval program_source_checksum",
        ),
        training_binding_hash=require_sha256_digest(
            raw.get("training_binding_hash"),
            "reference eval training_binding_hash",
        ),
        capability_profile_hash=require_sha256_digest(
            raw.get("capability_profile_hash"),
            "reference eval capability_profile_hash",
        ),
        checkpoint=CheckpointRef(
            uri=_nonempty_text(
                checkpoint.get("uri"),
                "reference eval checkpoint uri",
            ),
            checksum=require_sha256_checksum(
                checkpoint.get("checksum"),
                "reference eval checkpoint checksum",
            ),
            role=_nonempty_text(
                checkpoint.get("role"),
                "reference eval checkpoint role",
            ),
            sealed=checkpoint.get("sealed"),
        ),
        public_seed_count=protocol.public_seed_count,
        public_accept_seed_count=protocol.public_accept_seed_count,
        sealed_validation_seed_count=protocol.sealed_validation_seed_count,
        eval_num_envs=protocol.eval_num_envs,
        eval_protocol_version=protocol.eval_protocol_version,
        evaluation_binding=evaluation_binding,
    )
    expected = {
        "subject": ExperimentSubject.reference(
            anchor.reference_manifest.reference_id
        ),
        "subject_manifest_hash": anchor.reference_manifest.manifest_hash,
        "program_source_checksum": anchor.reference_manifest.source_checksum,
        "training_binding_hash": anchor.training_binding.binding_hash,
        "capability_profile_hash": (
            anchor.reference_manifest.capability_profile_hash
        ),
        "checkpoint_hash": anchor.checkpoint_hash,
    }
    actual = {
        "subject": spec.subject,
        "subject_manifest_hash": spec.subject_manifest_hash,
        "program_source_checksum": spec.program_source_checksum,
        "training_binding_hash": spec.training_binding_hash,
        "capability_profile_hash": spec.capability_profile_hash,
        "checkpoint_hash": spec.execution_checkpoint().checksum,
    }
    if actual != expected:
        raise ValueError("reference private eval job lineage mismatch")
    if spec.to_public_dict() != dict(raw):
        raise ValueError("reference private eval job canonical projection mismatch")
    return spec


def _build_reference_seal(
    *,
    eval_spec: EvalJobSpec,
    task_spec: TaskSpec,
    public_result: PublicEvalResult,
) -> TrustedEvalSeal:
    metrics = public_result.metrics
    sealed_gate = _json_object(
        metrics.get("sealed_validation_gate"),
        "reference sealed_validation_gate",
    )
    return build_trusted_eval_seal(
        eval_spec=eval_spec,
        task_invariants=task_spec.invariants,
        checkpoint_safety_report=metrics.get("checkpoint_safety_report"),
        candidate_isolation_status=_nonempty_text(
            sealed_gate.get("candidate_isolation_status"),
            "reference candidate isolation status",
        ),
    )


def _assert_manifest_matches_anchor(
    manifest: BaselineManifest,
    anchor: BaselineGenesisAnchor,
) -> None:
    if manifest.baseline_id != anchor.baseline_id:
        raise ValueError("session Genesis baseline_id drift")
    if manifest.manifest_hash != anchor.baseline_manifest_hash:
        raise ValueError("session Genesis baseline manifest hash drift")
    if manifest.checkpoint.checksum != anchor.checkpoint_hash:
        raise ValueError("session Genesis checkpoint hash drift")


def _assert_policy_initialization_matches_anchor(
    policy_initialization: PolicyInitialization,
    anchor: GenesisAnchor,
) -> None:
    if policy_initialization.kind == "random":
        return
    baseline = policy_initialization.baseline
    if baseline is None or not isinstance(anchor, BaselineGenesisAnchor):
        raise ValueError("warm-start policy initialization requires baseline Genesis")
    if (
        baseline.baseline_id != anchor.baseline_id
        or baseline.baseline_manifest_hash != anchor.baseline_manifest_hash
        or baseline.checkpoint.checksum != anchor.checkpoint_hash
    ):
        raise ValueError("warm-start policy initialization differs from Genesis")


def _validate_private_artifact_ref(
    ref: PrivateArtifactRef,
    *,
    expected_kind: str,
    expected_reason: str,
    field_prefix: str,
) -> None:
    if ref.kind != expected_kind:
        raise ValueError(f"{field_prefix} private artifact kind mismatch")
    if not isinstance(ref.uri, str) or not ref.uri:
        raise ValueError(f"{field_prefix} private artifact uri must be non-empty")
    if not Path(ref.uri).is_absolute():
        raise ValueError(f"{field_prefix} private artifact uri must be absolute")
    require_sha256_checksum(
        ref.checksum,
        f"{field_prefix} private artifact checksum",
    )
    if ref.reason != expected_reason:
        raise ValueError(f"{field_prefix} private artifact reason mismatch")


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(value, trailing_newline=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return load_strict_json_object(
            path,
            context=f"session Genesis artifact {path.name}",
            require_canonical=True,
            trailing_newline=True,
        )
    except ValueError as exc:
        raise ValueError(f"failed to read session Genesis artifact: {path.name}") from exc


def _exact_mapping(
    value: Any,
    expected_keys: set[str],
    field_name: str,
) -> Mapping[str, Any]:
    mapping = _json_object(value, field_name)
    if set(mapping) != expected_keys:
        raise ValueError(f"{field_name} fields mismatch")
    return mapping


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return _json_copy(value)


def _json_copy(value: Any) -> Any:
    normalized = _normalize_json_value(value)
    try:
        return strict_json_loads(
            canonical_json_bytes(normalized),
            context="canonical JSON value",
        )
    except ValueError as exc:
        raise ValueError("value must be canonical JSON data") from exc


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_normalize_json_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError("value must contain only canonical JSON types")


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite non-boolean number")
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} must be finite")
    return parsed


__all__ = [
    "BASELINE_EVAL_ARTIFACT_FILENAME",
    "REFERENCE_EVAL_ARTIFACT_FILENAME",
    "REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION",
    "REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION",
    "SESSION_GENESIS_FILENAME",
    "BaselineGenesisAnchor",
    "BootstrapGenesisAnchor",
    "ReferenceProgramGenesisAnchor",
    "SessionGenesisArtifact",
    "VerifiedSessionGenesis",
    "prepare_session_genesis",
    "verify_session_genesis_for_runtime",
]
