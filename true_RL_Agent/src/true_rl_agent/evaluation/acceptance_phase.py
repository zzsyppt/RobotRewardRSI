"""Shared state transition into acceptance comparison."""

from __future__ import annotations


def acceptance_comparison_entered(
    *,
    policy_eligible: bool,
    public_result_status: object,
) -> bool:
    """Return whether a round must consume gates and compare eval evidence."""

    return policy_eligible and public_result_status == "pass"


__all__ = ["acceptance_comparison_entered"]
