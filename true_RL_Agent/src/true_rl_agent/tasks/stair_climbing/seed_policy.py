"""Private Stair evaluation seed and seven-bin assignment policy."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Sequence

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.json_contract import canonical_json_text
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationCell,
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.ordered_evaluation import (
    OrderedStairEvaluationCell,
    StairOrderedEvaluationSuite,
)

STAIR_PRIVATE_SEED_RESOLVER_VERSION = (
    "stair_v2_private_seed_and_bin_resolver_v1"
)
STAIR_EVALUATION_SEED_RESOLVER_VERSION = (
    "stair_evaluation_profile_seed_resolver_v1"
)
STAIR_ORDERED_EVALUATION_SEED_RESOLVER_VERSION = (
    "stair_ordered_cell_seed_resolver_v1"
)
STAIR_FORMAL_EVAL_NAMESPACE = "formal_reference_candidate_eval"
STAIR_PILOT_EVAL_NAMESPACE = "pilot_reference_eval"
STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE = (
    "final_never_fed_paired_confirmation_v1"
)
STAIR_CALIBRATION_EVAL_NAMESPACE = "c1_irregular_terrain_calibration_v1"
STAIR_PUBLIC_DEV_DIAGNOSTIC_EVAL_NAMESPACE = "public_dev_diagnostic_v1"
_DIAGNOSTIC_NAMESPACE_PREFIX = "public_dev_diagnostic_"
STAIR_FORMAL_TRAIN_NAMESPACE = "formal_reference_candidate_train"
STAIR_PILOT_TRAIN_NAMESPACE = "pilot_reference_train"
STAIR_PRIVATE_TRAIN_SEED_RESOLVER_VERSION = (
    "stair_v2_private_training_seed_resolver_v2"
)
STAIR_TRAIN_SEED_BASE = 42
_VALID_SPLITS = ("public_dev", "public_accept", "sealed_validation")
_VALID_EVAL_NAMESPACES = (
    STAIR_FORMAL_EVAL_NAMESPACE,
    STAIR_PILOT_EVAL_NAMESPACE,
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
)
_ORDERED_VALID_SPLITS = (*_VALID_SPLITS, "calibration")
_ORDERED_VALID_EVAL_NAMESPACES = (
    *_VALID_EVAL_NAMESPACES,
    STAIR_CALIBRATION_EVAL_NAMESPACE,
    STAIR_PUBLIC_DEV_DIAGNOSTIC_EVAL_NAMESPACE,
)
_VALID_TRAIN_NAMESPACES = (
    STAIR_FORMAL_TRAIN_NAMESPACE,
    STAIR_PILOT_TRAIN_NAMESPACE,
)


def diagnostic_seed_namespace(cohort: str = "v1") -> str:
    """Name a public-dev-only reset cohort without changing the task domain."""
    if not isinstance(cohort, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cohort) is None:
        raise ValueError("Stair diagnostic cohort must be a nonempty simple identifier")
    return _DIAGNOSTIC_NAMESPACE_PREFIX + cohort


def diagnostic_cohort_from_namespace(namespace: str) -> str:
    if not isinstance(namespace, str) or not namespace.startswith(_DIAGNOSTIC_NAMESPACE_PREFIX):
        raise ValueError("Stair diagnostic seed namespace is invalid")
    cohort = namespace[len(_DIAGNOSTIC_NAMESPACE_PREFIX):]
    diagnostic_seed_namespace(cohort)
    return cohort


def is_public_dev_diagnostic_namespace(namespace: str) -> bool:
    try:
        diagnostic_cohort_from_namespace(namespace)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class StairPrivateSeedResolver:
    """Runner-private deterministic seeds with a public cohort commitment."""

    base_seed: int
    task_protocol_hash: str
    evaluation_protocol: EvaluationProtocol
    bin_ids: tuple[str, ...]
    episodes_per_seed_per_bin: int
    resolver_version: str = STAIR_PRIVATE_SEED_RESOLVER_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
            or self.base_seed < 0
        ):
            raise ValueError("stair eval base_seed must be a non-negative integer")
        require_sha256_digest(
            self.task_protocol_hash,
            "stair seed resolver task protocol hash",
        )
        if not isinstance(self.evaluation_protocol, EvaluationProtocol):
            raise TypeError("stair seed resolver evaluation_protocol is invalid")
        if self.resolver_version != STAIR_PRIVATE_SEED_RESOLVER_VERSION:
            raise ValueError("stair private seed resolver version mismatch")
        if (
            not isinstance(self.bin_ids, tuple)
            or not self.bin_ids
            or len(self.bin_ids) != len(set(self.bin_ids))
            or any(not isinstance(item, str) or not item for item in self.bin_ids)
        ):
            raise ValueError("stair bin_ids must be a non-empty unique tuple")
        if (
            isinstance(self.episodes_per_seed_per_bin, bool)
            or not isinstance(self.episodes_per_seed_per_bin, int)
            or self.episodes_per_seed_per_bin <= 0
        ):
            raise ValueError(
                "stair episodes_per_seed_per_bin must be a positive integer"
            )
        if (
            len(self.bin_ids) * self.episodes_per_seed_per_bin
            != self.evaluation_protocol.eval_num_envs
        ):
            raise ValueError("stair seven-bin env count does not match eval protocol")

    @property
    def _base_cohort_hash(self) -> str:
        return canonical_json_hash(self._cohort_payload())

    def namespace_cohort_hash(self, namespace: str) -> str:
        """Bind the private cohort commitment to one explicit execution domain."""

        _require_namespace(
            namespace,
            allowed=_VALID_EVAL_NAMESPACES,
            field_name="stair evaluation seed namespace",
        )
        return canonical_json_hash(
            {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing.seed_namespace_cohort.v1"
                ),
                "base_cohort_hash": self._base_cohort_hash,
                "namespace": namespace,
            }
        )

    def derive_seed(
        self,
        *,
        split: str,
        run_index: int,
        namespace: str,
    ) -> int:
        self._validate_run(split=split, run_index=run_index)
        _require_namespace(
            namespace,
            allowed=_VALID_EVAL_NAMESPACES,
            field_name="stair evaluation seed namespace",
        )
        payload = {
            **self._cohort_payload(),
            "namespace": namespace,
            "split": split,
            "run_index": run_index,
            "private_base_seed": self.base_seed,
        }
        digest = hashlib.sha256(canonical_json_text(payload).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % (2**31 - 1)

    def private_bin_assignment(self) -> tuple[str, ...]:
        return tuple(
            bin_id
            for bin_id in self.bin_ids
            for _ in range(self.episodes_per_seed_per_bin)
        )

    def public_seed_record(
        self,
        *,
        split: str,
        run_index: int,
        namespace: str,
    ) -> dict[str, Any]:
        self._validate_run(split=split, run_index=run_index)
        _require_namespace(
            namespace,
            allowed=_VALID_EVAL_NAMESPACES,
            field_name="stair evaluation seed namespace",
        )
        return {
            "resolver_version": self.resolver_version,
            "split": split,
            "run_index": run_index,
            "split_seed_count": self._split_seed_count(split),
            "bin_count": len(self.bin_ids),
            "episodes_per_seed_per_bin": self.episodes_per_seed_per_bin,
            "complete_bin_coverage_required": True,
            "seed_cohort_hash": self.namespace_cohort_hash(namespace),
        }

    def _cohort_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.task.stair_climbing.seed_cohort.v1",
            "resolver_version": self.resolver_version,
            "task_protocol_hash": self.task_protocol_hash,
            "eval_protocol_hash": self.evaluation_protocol.protocol_hash,
            "bin_ids": list(self.bin_ids),
            "episodes_per_seed_per_bin": self.episodes_per_seed_per_bin,
            "public_seed_count": self.evaluation_protocol.public_seed_count,
            "public_accept_seed_count": (
                self.evaluation_protocol.public_accept_seed_count
            ),
            "sealed_validation_seed_count": (
                self.evaluation_protocol.sealed_validation_seed_count
            ),
            "private_base_seed_commitment": canonical_json_hash(
                {
                    "schema_version": "true_rl_agent.private_seed_commitment.v1",
                    "base_seed": self.base_seed,
                }
            ),
        }

    def _validate_run(self, *, split: str, run_index: int) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"unknown stair evaluation split: {split}")
        if (
            isinstance(run_index, bool)
            or not isinstance(run_index, int)
            or run_index <= 0
            or run_index > self._split_seed_count(split)
        ):
            raise ValueError("stair evaluation run_index is out of range")

    def _split_seed_count(self, split: str) -> int:
        return {
            "public_dev": self.evaluation_protocol.public_seed_count,
            "public_accept": self.evaluation_protocol.public_accept_seed_count,
            "sealed_validation": (
                self.evaluation_protocol.sealed_validation_seed_count
            ),
        }[split]


@dataclass(frozen=True)
class StairEvaluationSeedResolver:
    """Expanded course assignment with historical runner seed derivation."""

    legacy_resolver: StairPrivateSeedResolver
    evaluation_profile: StairEvaluationProfile
    resolver_version: str = STAIR_EVALUATION_SEED_RESOLVER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.legacy_resolver, StairPrivateSeedResolver):
            raise TypeError("Stair evaluation resolver requires a legacy resolver")
        if not isinstance(self.evaluation_profile, StairEvaluationProfile):
            raise TypeError("Stair evaluation resolver requires an evaluation profile")
        if self.resolver_version != STAIR_EVALUATION_SEED_RESOLVER_VERSION:
            raise ValueError("Stair evaluation resolver version mismatch")

    @property
    def base_seed(self) -> int:
        return self.legacy_resolver.base_seed

    @property
    def evaluation_protocol(self) -> EvaluationProtocol:
        return self.evaluation_profile.evaluation_protocol

    def namespace_cohort_hash(self, namespace: str) -> str:
        return canonical_json_hash(
            {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing."
                    "evaluation_profile_seed_cohort.v1"
                ),
                "resolver_version": self.resolver_version,
                "evaluation_profile_hash": self.evaluation_profile.profile_hash,
                "eval_protocol_hash": self.evaluation_protocol.protocol_hash,
                "legacy_seed_namespace_cohort_hash": (
                    self.legacy_resolver.namespace_cohort_hash(namespace)
                ),
            }
        )

    def derive_seed(
        self,
        *,
        split: str,
        run_index: int,
        namespace: str,
    ) -> int:
        return self.legacy_resolver.derive_seed(
            split=split,
            run_index=run_index,
            namespace=namespace,
        )

    def private_cell_assignment(self) -> tuple[StairEvaluationCell, ...]:
        return self.evaluation_profile.private_cell_assignment()

    def private_bin_assignment(self) -> tuple[str, ...]:
        return tuple(cell.bin_id for cell in self.private_cell_assignment())

    def public_seed_record(
        self,
        *,
        split: str,
        run_index: int,
        namespace: str,
    ) -> dict[str, Any]:
        legacy = self.legacy_resolver.public_seed_record(
            split=split,
            run_index=run_index,
            namespace=namespace,
        )
        return {
            "resolver_version": self.resolver_version,
            "split": split,
            "run_index": run_index,
            "split_seed_count": legacy["split_seed_count"],
            "cell_count": len(self.evaluation_profile.cells),
            "episodes_per_seed_per_cell": (
                self.evaluation_profile.episodes_per_seed_per_cell
            ),
            "complete_cell_coverage_required": True,
            "seed_cohort_hash": self.namespace_cohort_hash(namespace),
        }

    def _split_seed_count(self, split: str) -> int:
        return self.legacy_resolver._split_seed_count(split)


@dataclass(frozen=True)
class StairOrderedEvaluationSeedResolver:
    """Private seeds bound to the exact split-local ordered C1 cohort."""

    base_seed: int
    task_protocol_hash: str
    evaluation_suite: StairOrderedEvaluationSuite
    resolver_version: str = STAIR_ORDERED_EVALUATION_SEED_RESOLVER_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
            or self.base_seed < 0
        ):
            raise ValueError("Stair ordered eval base_seed must be non-negative")
        require_sha256_digest(
            self.task_protocol_hash,
            "Stair ordered eval task protocol hash",
        )
        if not isinstance(self.evaluation_suite, StairOrderedEvaluationSuite):
            raise TypeError("Stair ordered resolver requires an ordered suite")
        if self.resolver_version != STAIR_ORDERED_EVALUATION_SEED_RESOLVER_VERSION:
            raise ValueError("Stair ordered seed resolver version mismatch")

    @property
    def evaluation_protocol(self) -> EvaluationProtocol:
        return self.evaluation_suite.evaluation_protocol

    def derive_seed(
        self,
        *,
        split: str,
        run_index: int,
        namespace: str,
    ) -> int:
        self._validate_run(split=split, run_index=run_index)
        self._validate_namespace_for_split(split=split, namespace=namespace)
        digest = hashlib.sha256(
            canonical_json_text(
                {
                    **self._public_commitment_payload(split=split),
                    "namespace": namespace,
                    "run_index": run_index,
                    "private_base_seed": self.base_seed,
                }
            ).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") % (2**31 - 1)

    def namespace_cohort_hash(self, namespace: str) -> str:
        if not is_public_dev_diagnostic_namespace(namespace):
            _require_namespace(
                namespace,
                allowed=_ORDERED_VALID_EVAL_NAMESPACES,
                field_name="Stair ordered evaluation seed namespace",
            )
        splits = (
            ("calibration",)
            if namespace == STAIR_CALIBRATION_EVAL_NAMESPACE
            else _VALID_SPLITS
        )
        if is_public_dev_diagnostic_namespace(namespace):
            splits = ("public_dev",)
        return canonical_json_hash(
            {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing."
                    "ordered_seed_namespace_cohort.v1"
                ),
                "resolver_version": self.resolver_version,
                "suite_hash": self.evaluation_suite.source_suite_hash,
                "task_protocol_hash": self.task_protocol_hash,
                "namespace": namespace,
                "split_cohort_hashes": {
                    split: self.evaluation_suite.cohort_hash(split)
                    for split in splits
                },
            }
        )

    def private_cell_assignment(
        self,
        split: str,
    ) -> tuple[OrderedStairEvaluationCell, ...]:
        self._validate_split(split)
        return self.evaluation_suite.private_cell_assignment(split)

    def private_bin_assignment(self, split: str) -> tuple[str, ...]:
        return tuple(cell.cell_id for cell in self.private_cell_assignment(split))

    def public_seed_record(
        self,
        *,
        split: str,
        run_index: int,
        namespace: str,
    ) -> dict[str, Any]:
        self._validate_run(split=split, run_index=run_index)
        self._validate_namespace_for_split(split=split, namespace=namespace)
        cells = self.evaluation_suite.cells_for_cohort(split)
        return {
            "resolver_version": self.resolver_version,
            "split": split,
            "run_index": run_index,
            "split_seed_count": self._split_seed_count(split),
            "cell_count": len(cells),
            "episodes_per_seed_per_cell": (
                self.evaluation_suite.episodes_per_seed_per_cell
            ),
            "complete_cell_coverage_required": True,
            "ordered_suite_hash": self.evaluation_suite.suite_hash,
            **({"source_suite_hash": self.evaluation_suite.source_suite_hash}
               if self.evaluation_suite.source_suite_hash != self.evaluation_suite.suite_hash else {}),
            "split_cohort_hash": self.evaluation_suite.cohort_hash(split),
            "seed_cohort_hash": self.namespace_cohort_hash(namespace),
        }

    def _public_commitment_payload(self, *, split: str) -> dict[str, Any]:
        self._validate_split(split)
        return {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.ordered_seed_domain.v1"
            ),
            "resolver_version": self.resolver_version,
            "task_protocol_hash": self.task_protocol_hash,
            "ordered_suite_hash": self.evaluation_suite.source_suite_hash,
            "split": split,
            "split_cohort_hash": self.evaluation_suite.cohort_hash(split),
        }

    def _validate_run(self, *, split: str, run_index: int) -> None:
        self._validate_split(split)
        if (
            isinstance(run_index, bool)
            or not isinstance(run_index, int)
            or run_index <= 0
            or run_index > self._split_seed_count(split)
        ):
            raise ValueError("Stair ordered evaluation run_index is out of range")

    @staticmethod
    def _validate_split(split: str) -> None:
        if split not in _ORDERED_VALID_SPLITS:
            raise ValueError(f"unknown Stair ordered evaluation split: {split}")

    @staticmethod
    def _validate_namespace_for_split(*, split: str, namespace: str) -> None:
        if split == "public_dev" and is_public_dev_diagnostic_namespace(namespace):
            return
        allowed = (
            (STAIR_CALIBRATION_EVAL_NAMESPACE,)
            if split == "calibration"
            else _VALID_EVAL_NAMESPACES
        )
        _require_namespace(
            namespace,
            allowed=allowed,
            field_name="Stair ordered evaluation seed namespace",
        )

    def _split_seed_count(self, split: str) -> int:
        self._validate_split(split)
        return self.evaluation_suite.seed_count(split)


@dataclass(frozen=True)
class StairPrivateTrainingSeedResolver:
    """Task-owned deterministic separation of pilot and formal training seeds."""

    base_seed: int
    task_protocol_hash: str
    resolver_version: str = STAIR_PRIVATE_TRAIN_SEED_RESOLVER_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.base_seed, bool)
            or not isinstance(self.base_seed, int)
            or self.base_seed < 0
        ):
            raise ValueError(
                "stair training seed base_seed must be a non-negative integer"
            )
        require_sha256_digest(
            self.task_protocol_hash,
            "stair training seed resolver task protocol hash",
        )
        if self.resolver_version != STAIR_PRIVATE_TRAIN_SEED_RESOLVER_VERSION:
            raise ValueError("stair private training seed resolver version mismatch")

    def derive_seed(self, *, namespace: str, training_seed_index: int = 0) -> int:
        namespace = _require_namespace(
            namespace,
            allowed=_VALID_TRAIN_NAMESPACES,
            field_name="stair training seed namespace",
        )
        training_seed_index = _training_seed_index(training_seed_index)
        digest = hashlib.sha256(
            canonical_json_text(
                {
                    **self._public_commitment_payload(),
                    "namespace": namespace,
                    "training_seed_index": training_seed_index,
                    "private_base_seed": self.base_seed,
                }
            ).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") % (2**31 - 1)

    def namespace_binding_hash(
        self,
        *,
        namespace: str,
        training_seed_index: int = 0,
    ) -> str:
        namespace = _require_namespace(
            namespace,
            allowed=_VALID_TRAIN_NAMESPACES,
            field_name="stair training seed namespace",
        )
        training_seed_index = _training_seed_index(training_seed_index)
        return canonical_json_hash(
            {
                **self._public_commitment_payload(),
                "namespace": namespace,
                "training_seed_index": training_seed_index,
                "derived_seed_commitment": canonical_json_hash(
                    {
                        "schema_version": (
                            "true_rl_agent.private_training_seed_commitment.v1"
                        ),
                        "seed": self.derive_seed(
                            namespace=namespace,
                            training_seed_index=training_seed_index,
                        ),
                    }
                ),
            }
        )

    def namespace_cohort_hash(self, *, namespace: str, seed_count: int) -> str:
        seed_count = _positive_int(seed_count, "stair training seed cohort size")
        return canonical_json_hash(
            {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing.training_seed_cohort.v1"
                ),
                "namespace": namespace,
                "binding_hashes": [
                    self.namespace_binding_hash(
                        namespace=namespace,
                        training_seed_index=index,
                    )
                    for index in range(seed_count)
                ],
            }
        )

    def assert_namespaces_disjoint(self, *, pilot_seed_count: int = 1) -> None:
        pilot_seed_count = _positive_int(
            pilot_seed_count,
            "stair pilot training seed count",
        )
        pilot = {
            self.derive_seed(
                namespace=STAIR_PILOT_TRAIN_NAMESPACE,
                training_seed_index=index,
            )
            for index in range(pilot_seed_count)
        }
        formal = self.derive_seed(namespace=STAIR_FORMAL_TRAIN_NAMESPACE)
        if len(pilot) != pilot_seed_count or formal in pilot:
            raise ValueError("stair pilot and formal training seeds overlap")
        pilot_bindings = {
            self.namespace_binding_hash(
                namespace=STAIR_PILOT_TRAIN_NAMESPACE,
                training_seed_index=index,
            )
            for index in range(pilot_seed_count)
        }
        formal_binding = self.namespace_binding_hash(
            namespace=STAIR_FORMAL_TRAIN_NAMESPACE
        )
        if len(pilot_bindings) != pilot_seed_count or formal_binding in pilot_bindings:
            raise ValueError("stair pilot and formal training bindings overlap")

    def _public_commitment_payload(self) -> dict[str, Any]:
        return {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.training_seed_domain.v1"
            ),
            "resolver_version": self.resolver_version,
            "task_protocol_hash": self.task_protocol_hash,
            "private_base_seed_commitment": canonical_json_hash(
                {
                    "schema_version": "true_rl_agent.private_seed_commitment.v1",
                    "base_seed": self.base_seed,
                }
            ),
        }


def assert_seed_namespaces_disjoint(
    resolver: (
        StairPrivateSeedResolver
        | StairEvaluationSeedResolver
        | StairOrderedEvaluationSeedResolver
    ),
    *,
    splits: Sequence[str] = _VALID_SPLITS,
) -> None:
    split_names = tuple(splits)
    if (
        not split_names
        or len(split_names) != len(set(split_names))
        or any(split not in _VALID_SPLITS for split in split_names)
    ):
        raise ValueError("stair seed namespace split set is invalid")
    values_by_namespace: dict[str, list[int]] = {}
    for namespace in _VALID_EVAL_NAMESPACES:
        values = [
            resolver.derive_seed(
                split=split,
                run_index=index,
                namespace=namespace,
            )
            for split in split_names
            for index in range(1, resolver._split_seed_count(split) + 1)
        ]
        if len(values) != len(set(values)):
            raise ValueError(
                f"stair {namespace} evaluation seed namespace contains duplicates"
            )
        values_by_namespace[namespace] = values
    for index, left_name in enumerate(_VALID_EVAL_NAMESPACES):
        left = set(values_by_namespace[left_name])
        for right_name in _VALID_EVAL_NAMESPACES[index + 1 :]:
            if left & set(values_by_namespace[right_name]):
                raise ValueError(
                    "stair evaluation seed namespaces overlap: "
                    f"{left_name}, {right_name}"
                )
    if isinstance(resolver, StairOrderedEvaluationSeedResolver):
        calibration = [
            resolver.derive_seed(
                split="calibration",
                run_index=index,
                namespace=STAIR_CALIBRATION_EVAL_NAMESPACE,
            )
            for index in range(1, resolver._split_seed_count("calibration") + 1)
        ]
        if len(calibration) != len(set(calibration)):
            raise ValueError("stair calibration seed namespace contains duplicates")
        all_formal = {
            value for values in values_by_namespace.values() for value in values
        }
        if all_formal & set(calibration):
            raise ValueError("stair calibration and formal evaluation seeds overlap")


def _require_namespace(
    value: Any,
    *,
    allowed: Sequence[str],
    field_name: str,
) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field_name} is invalid")
    return value


def _training_seed_index(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("stair training_seed_index must be a non-negative integer")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


__all__ = [
    "STAIR_CALIBRATION_EVAL_NAMESPACE",
    "STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE",
    "STAIR_FORMAL_EVAL_NAMESPACE",
    "STAIR_FORMAL_TRAIN_NAMESPACE",
    "STAIR_PILOT_EVAL_NAMESPACE",
    "STAIR_PUBLIC_DEV_DIAGNOSTIC_EVAL_NAMESPACE",
    "STAIR_PILOT_TRAIN_NAMESPACE",
    "STAIR_PRIVATE_SEED_RESOLVER_VERSION",
    "STAIR_EVALUATION_SEED_RESOLVER_VERSION",
    "STAIR_ORDERED_EVALUATION_SEED_RESOLVER_VERSION",
    "STAIR_PRIVATE_TRAIN_SEED_RESOLVER_VERSION",
    "STAIR_TRAIN_SEED_BASE",
    "StairPrivateSeedResolver",
    "StairEvaluationSeedResolver",
    "StairOrderedEvaluationSeedResolver",
    "StairPrivateTrainingSeedResolver",
    "assert_seed_namespaces_disjoint",
    "diagnostic_seed_namespace",
    "diagnostic_cohort_from_namespace",
    "is_public_dev_diagnostic_namespace",
]
