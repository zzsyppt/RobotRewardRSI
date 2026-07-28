"""Runner-owned seed derivation for push recovery eval splits."""

from __future__ import annotations

from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.protocol import EvaluationProtocol

MAX_TORCH_SEED = 2**31 - 1
SEED_POLICY_NAME = "runner_owned_split_derived_counts_only_no_raw_seed_ids"
SPLIT_OFFSETS = {
    "public_eval": 0,
    "public_dev": 10_000,
    "public_accept": 20_000,
    "sealed_validation": 30_000,
}


def derive_runner_seed(*, base_seed: int, split_name: str, run_index: int) -> int:
    base_seed = _require_non_negative_integer(base_seed, "base_seed")
    run_index = _require_positive_integer(run_index, "run_index")
    if split_name not in SPLIT_OFFSETS:
        raise ValueError(f"unsupported eval seed split: {split_name!r}")
    offset = SPLIT_OFFSETS[split_name]
    return (base_seed + offset + run_index - 1) % MAX_TORCH_SEED


def public_seed_record(*, split_name: str, run_index: int, split_seed_count: int) -> dict[str, Any]:
    run_index = _require_positive_integer(run_index, "run_index")
    split_seed_count = _require_positive_integer(
        split_seed_count,
        "split_seed_count",
    )
    if split_name not in SPLIT_OFFSETS:
        raise ValueError(f"unsupported eval seed split: {split_name!r}")
    return {
        "seed_policy": SEED_POLICY_NAME,
        "split": split_name,
        "run_index": run_index,
        "split_seed_count": split_seed_count,
    }


def build_seed_cohort_hash(
    *,
    base_seed: int,
    evaluation_protocol: EvaluationProtocol,
) -> str:
    """Hash the private split seeds without publishing the seed values."""

    split_counts = {
        "public_dev": evaluation_protocol.public_seed_count,
        "public_accept": evaluation_protocol.public_accept_seed_count,
        "sealed_validation": evaluation_protocol.sealed_validation_seed_count,
    }
    private_manifest = {
        split_name: [
            derive_runner_seed(
                base_seed=base_seed,
                split_name=split_name,
                run_index=index + 1,
            )
            for index in range(seed_count)
        ]
        for split_name, seed_count in split_counts.items()
    }
    return canonical_json_hash(
        {
            "schema_version": "true_rl_agent.push_recovery.seed_cohort.v1",
            "seed_policy": SEED_POLICY_NAME,
            "splits": private_manifest,
        }
    )


def _require_non_negative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value
