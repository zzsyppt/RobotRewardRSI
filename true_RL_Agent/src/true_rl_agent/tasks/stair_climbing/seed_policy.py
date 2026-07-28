"""Private Stair evaluation seed and seven-bin assignment policy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.json_contract import canonical_json_text

STAIR_PRIVATE_SEED_RESOLVER_VERSION = (
    "stair_v2_private_seed_and_bin_resolver_v1"
)
STAIR_FORMAL_EVAL_NAMESPACE = "formal_reference_candidate_eval"
STAIR_PILOT_EVAL_NAMESPACE = "pilot_reference_eval"
_VALID_SPLITS = ("public_dev", "public_accept", "sealed_validation")


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
    def cohort_hash(self) -> str:
        return canonical_json_hash(self._cohort_payload())

    def derive_seed(
        self,
        *,
        split: str,
        run_index: int,
        namespace: str = STAIR_FORMAL_EVAL_NAMESPACE,
    ) -> int:
        self._validate_run(split=split, run_index=run_index)
        if namespace not in {
            STAIR_FORMAL_EVAL_NAMESPACE,
            STAIR_PILOT_EVAL_NAMESPACE,
        }:
            raise ValueError("stair evaluation seed namespace is invalid")
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
    ) -> dict[str, Any]:
        self._validate_run(split=split, run_index=run_index)
        return {
            "resolver_version": self.resolver_version,
            "split": split,
            "run_index": run_index,
            "split_seed_count": self._split_seed_count(split),
            "bin_count": len(self.bin_ids),
            "episodes_per_seed_per_bin": self.episodes_per_seed_per_bin,
            "complete_bin_coverage_required": True,
            "seed_cohort_hash": self.cohort_hash,
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


def assert_seed_namespaces_disjoint(
    resolver: StairPrivateSeedResolver,
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
    formal_values: list[int] = []
    pilot_values: list[int] = []
    for split in split_names:
        for index in range(1, resolver._split_seed_count(split) + 1):
            formal_values.append(
                resolver.derive_seed(
                    split=split,
                    run_index=index,
                    namespace=STAIR_FORMAL_EVAL_NAMESPACE,
                )
            )
            pilot_values.append(
                resolver.derive_seed(
                    split=split,
                    run_index=index,
                    namespace=STAIR_PILOT_EVAL_NAMESPACE,
                )
            )
    formal = set(formal_values)
    pilot = set(pilot_values)
    if len(formal) != len(formal_values):
        raise ValueError("stair formal evaluation seed namespace contains duplicates")
    if len(pilot) != len(pilot_values):
        raise ValueError("stair pilot evaluation seed namespace contains duplicates")
    if formal & pilot:
        raise ValueError("stair pilot and formal seed namespaces overlap")


__all__ = [
    "STAIR_FORMAL_EVAL_NAMESPACE",
    "STAIR_PILOT_EVAL_NAMESPACE",
    "STAIR_PRIVATE_SEED_RESOLVER_VERSION",
    "StairPrivateSeedResolver",
    "assert_seed_namespaces_disjoint",
]
