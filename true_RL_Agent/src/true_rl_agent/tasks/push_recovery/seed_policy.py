"""Runner-owned seed derivation for push recovery eval splits."""

from __future__ import annotations

from typing import Any


MAX_TORCH_SEED = 2**31 - 1
SEED_POLICY_NAME = "runner_owned_split_derived_counts_only_no_raw_seed_ids"
SPLIT_OFFSETS = {
    "public_eval": 0,
    "public_dev": 10_000,
    "public_accept": 20_000,
    "trusted": 30_000,
}


def derive_runner_seed(*, base_seed: int, split_name: str, run_index: int) -> int:
    if run_index <= 0:
        raise ValueError("run_index must be positive")
    offset = SPLIT_OFFSETS.get(split_name, 40_000)
    return (int(base_seed) + offset + int(run_index) - 1) % MAX_TORCH_SEED


def public_seed_record(*, split_name: str, run_index: int, split_seed_count: int) -> dict[str, Any]:
    if split_seed_count <= 0:
        raise ValueError("split_seed_count must be positive")
    return {
        "seed_policy": SEED_POLICY_NAME,
        "split": split_name,
        "run_index": int(run_index),
        "split_seed_count": int(split_seed_count),
    }
