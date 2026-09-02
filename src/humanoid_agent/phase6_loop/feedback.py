"""Build public-only feedback summaries for Phase 6 prompts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from humanoid_agent.phase5_llm_agent.prompting import build_public_prompt_summary
from humanoid_agent.phase6_loop.decision import public_feedback_from_decision


def build_phase6_feedback_summary(
    *,
    best_public_summary: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    include_history_rounds: int = 3,
) -> dict[str, Any]:
    """Return a prompt-safe summary for the next Phase 6 proposal.

    `history` entries are Phase 6 public decisions, not raw decision files.
    """

    prompt_summary = build_public_prompt_summary(best_public_summary)
    prompt_summary["schema_version"] = "humanoid_agent.phase6.prompt_feedback_summary.v1"
    prompt_summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    prompt_summary["phase6_feedback"] = {
        "best_source": "current_public_best",
        "start_policy": "phase2_baseline_first",
        "history": [
            public_feedback_from_decision(item)
            for item in (history or [])[-include_history_rounds:]
        ],
        "feedback_safety": {
            "public_only": True,
            "operation_diff_only": True,
            "raw_decision_file_not_included": True,
            "raw_eval_summary_not_included": True,
            "heldout_results_not_included": True,
        },
    }
    return prompt_summary


def assert_public_only_payload(value: Any) -> None:
    text = str(value).lower()
    forbidden = [
        "iid_hidden",
        "iid-hidden",
        "phase2_iid_hidden",
        "iid_hidden_eval",
        "iid_hidden_success_rate",
        "heldout_mean",
        "accept_reject",
    ]
    hits = [token for token in forbidden if token in text]
    if hits:
        raise ValueError(f"Phase 6 feedback payload contains forbidden tokens: {hits}")
