"""Public context construction for next-round prompts."""

from __future__ import annotations

from typing import Any, Mapping


FORBIDDEN_PUBLIC_KEY_PARTS = ("private", "hidden", "secret", "api_key", "token")
FORBIDDEN_PUBLIC_VALUE_PARTS = ("sk-",)


def sanitize_public_context(value: Any) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in FORBIDDEN_PUBLIC_KEY_PARTS):
                continue
            clean[str(key)] = sanitize_public_context(item)
        return clean
    if isinstance(value, list):
        return [sanitize_public_context(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_context(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if any(part in lowered for part in FORBIDDEN_PUBLIC_VALUE_PARTS):
            return "[redacted]"
    return value


def build_public_context(ledger_records: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_scores = [
        record.get("decision", {}).get("public_score")
        for record in ledger_records
        if record.get("decision", {}).get("accepted") is True
    ]
    numeric_accepted_scores = [float(score) for score in accepted_scores if isinstance(score, int | float)]
    last_record = ledger_records[-1] if ledger_records else {}
    context = {
        "public_only": True,
        "previous_rounds": len(ledger_records),
        "best_public_score": max(numeric_accepted_scores) if numeric_accepted_scores else None,
        "last_outcome": last_record.get("decision", {}).get("outcome"),
        "pure_knob_streak": _pure_knob_streak(ledger_records),
        "recent_outcomes": [
            {
                "round_id": record.get("round_id"),
                "outcome": record.get("decision", {}).get("outcome"),
                "surface": record.get("candidate_surface", {}).get("classification"),
            }
            for record in ledger_records[-5:]
        ],
    }
    return sanitize_public_context(context)


def _pure_knob_streak(ledger_records: list[dict[str, Any]]) -> int:
    streak = 0
    for record in reversed(ledger_records):
        if record.get("candidate_surface", {}).get("classification") != "parameter_only":
            break
        streak += 1
    return streak
