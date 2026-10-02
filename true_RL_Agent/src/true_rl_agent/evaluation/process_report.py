"""Shape and capacity of task-authored process statements.

Tasks own templates, operands, physical meanings and evidence selection. This
module never infers a predicate from prose or edits an observation's wording.
The full task artifact keeps slots and evidence references; the public view
keeps readable text and the same stable statement ids.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any


SENTENCE_SECTIONS = (
    "results", "process", "counterexample", "visual", "gaps", "parent_diff",
)
REPORT_SECTIONS = (*SENTENCE_SECTIONS, "actionability")
SENTENCE_KINDS = frozenset({"S0", "S1", "S2", "S3", "S4", "S5"})
UNSUPPORTED_NORMALITY_PHRASES = (
    "没有异常", "行为正常", "未发现异常", "no anomaly", "no anomalies",
    "behavior is normal", "behaviour is normal",
)


def report_bytes(value: Mapping[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _scope(value: Any, field: str) -> None:
    if not isinstance(value, Mapping) or not _text(value.get("label")):
        raise ValueError(f"{field} requires a nonempty label")
    has_n, has_d = "numerator" in value, "denominator" in value
    if has_n != has_d:
        raise ValueError(f"{field} needs numerator and denominator together")
    if has_n:
        n, d = value["numerator"], value["denominator"]
        if not _count(n) or not _count(d) or n > d:
            raise ValueError(f"{field} has invalid counts")


def validate_sentence(row: Mapping[str, Any]) -> None:
    """Validate inspectable slots, not truth or causal meaning of task prose."""
    if not isinstance(row, Mapping):
        raise ValueError("sentence must be an object")
    for field in ("id", "text", "subject", "predicate", "rule_version"):
        if not _text(row.get(field)):
            raise ValueError(f"sentence requires {field}")
    kind = row.get("kind")
    if kind not in SENTENCE_KINDS or row.get("section") not in SENTENCE_SECTIONS:
        raise ValueError("unsupported sentence kind or section")
    if any(p in row["text"].casefold() for p in UNSUPPORTED_NORMALITY_PHRASES):
        raise ValueError("unqualified normality statement is not evidence")
    _scope(row.get("scope"), "scope")
    refs = row.get("evidence_refs")
    if not isinstance(refs, (list, tuple)) or not refs or not all(_text(r) for r in refs):
        raise ValueError("sentence requires resolvable evidence references")
    window = row.get("window")
    if kind in {"S0", "S1", "S2", "S3"}:
        if not isinstance(window, Mapping) or not window:
            raise ValueError("S0-S3 require an explicit time scope")
        if not ("start" in window and "end" in window) and window.get("kind") not in {
            "whole_episode", "group_episodes", "whole_window", "cohort_episodes",
        }:
            raise ValueError("time scope must name its range or whole-record scope")
    if isinstance(window, Mapping) and ("start" in window or "end" in window):
        start, end = window.get("start"), window.get("end")
        if not _count(start) or not _count(end) or end < start:
            raise ValueError("invalid step interval")
        confirmed = window.get("confirmed")
        if confirmed is not None and (not _count(confirmed) or not start <= confirmed <= end):
            raise ValueError("confirmation step must be inside its interval")
        terminal = window.get("terminal")
        if terminal is not None and (not _count(terminal) or end > terminal):
            raise ValueError("interval exceeds episode terminal")
        if window.get("before_terminal") is True:
            event_step = confirmed if confirmed is not None else start
            if terminal is None or not event_step < terminal:
                raise ValueError("terminal-same-step is not before terminal")
    if kind in {"S2", "S4"}:
        comparison = row.get("comparison")
        if not isinstance(comparison, Mapping) or not comparison:
            raise ValueError("comparison sentence requires comparison scope")
        if kind == "S2" and row.get("descriptive", False):
            if comparison.get("discriminating") is not False:
                raise ValueError("descriptive terms require a non-discriminating comparison")
            for side in ("left", "right"):
                _scope(comparison.get(side), f"comparison.{side}")
                if "denominator" not in comparison[side]:
                    raise ValueError("descriptive comparison requires both denominators")
                expected = f"{comparison[side]['numerator']}/{comparison[side]['denominator']}"
                if expected not in row["text"]:
                    raise ValueError("descriptive comparison must print both scoped counts")
            if "not discriminating" not in row["text"].casefold() and "不具区分力" not in row["text"]:
                raise ValueError("descriptive comparison must state its limitation")
        if kind == "S2" and "denominator" not in row["scope"]:
            raise ValueError("S2 requires its group denominator")
        if kind == "S4":
            for side in ("left", "right"):
                _scope(comparison.get(side), f"comparison.{side}")
                if "denominator" not in comparison[side]:
                    raise ValueError("S4 requires both denominators")
                if comparison[side].get("rule_version") != row["rule_version"]:
                    raise ValueError("S4 requires identical rule versions")
            if not all(_count(comparison.get(k)) for k in ("paired", "missing")):
                raise ValueError("S4 requires paired and missing counts")
    if kind == "S3":
        for field in ("media_identity", "alignment_status", "evidence_category", "visibility"):
            if not row.get(field):
                raise ValueError(f"S3 requires {field}")
        if row["section"] != "visual":
            raise ValueError("visual observation belongs in the visual section")
    if kind == "S5":
        if not _text(row.get("gap_type")):
            raise ValueError("gap statement requires gap_type")
        if "denominator" not in row["scope"]:
            raise ValueError("S5 requires a denominator, including explicit zero")
    if "required" in row and not isinstance(row["required"], bool):
        raise ValueError("required must be boolean")
    if "priority" in row and (not isinstance(row["priority"], int)
                              or isinstance(row["priority"], bool)):
        raise ValueError("priority must be an integer")


def validate_public_report(value: Mapping[str, Any], *, max_bytes: int) -> None:
    """Structural replay check; full-slot/truth checks run before projection."""
    report = value.get("report")
    if not isinstance(report, Mapping) or set(report) != set(REPORT_SECTIONS):
        raise ValueError("process report requires all seven sections")
    seen: set[str] = set()
    for section in SENTENCE_SECTIONS:
        rows = report[section]
        if not isinstance(rows, list) or not rows:
            raise ValueError("each narrative section needs content or an explicit gap")
        for row in rows:
            if (not isinstance(row, Mapping) or set(row) != {"id", "text"}
                    or not _text(row.get("id")) or not _text(row.get("text"))):
                raise ValueError("public sentence must contain id and readable text")
            if row["id"] in seen:
                raise ValueError("duplicate public statement id")
            seen.add(row["id"])
            if any(p in row["text"].casefold() for p in UNSUPPORTED_NORMALITY_PHRASES):
                raise ValueError("unqualified normality statement is not evidence")
    if not isinstance(report["actionability"], Mapping) or not report["actionability"]:
        raise ValueError("actionability requires task-owned content")
    if report_bytes(value) > max_bytes:
        raise ValueError("process report exceeds its byte budget")


def render_process_report(
    *, envelope: Mapping[str, Any], sentences: Sequence[Mapping[str, Any]],
    actionability: Mapping[str, Any], max_bytes: int,
    update_projection: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Project validated task sentences without changing or clipping their text.

    Lower priority optional statements are removed first. Required statements
    and the last statement of every section remain atomic. Capacity failure is
    explicit; tasks must revise a template rather than silently lose evidence.
    """
    if not _count(max_bytes) or max_bytes == 0:
        raise ValueError("max_bytes must be positive")
    if "report" in envelope or "report_projection" in envelope:
        raise ValueError("report envelope must not include a stale projection")
    rows = [copy.deepcopy(dict(s)) for s in sentences]
    ids: set[str] = set()
    for row in rows:
        validate_sentence(row)
        if row["id"] in ids:
            raise ValueError("duplicate statement id")
        ids.add(row["id"])
    if not actionability:
        raise ValueError("actionability cannot be empty")
    if any(not any(r["section"] == section for r in rows) for section in SENTENCE_SECTIONS):
        raise ValueError("task must supply missing-section S5 statements")
    if "report.capacity_omissions" in ids:
        raise ValueError("reserved capacity statement id")
    candidates = sorted((r for r in rows if not r.get("required", True)),
                        key=lambda r: (r.get("priority", 0), r["id"]))
    value = copy.deepcopy(dict(envelope))
    value["report"] = {
        section: [{"id": r["id"], "text": r["text"]} for r in rows if r["section"] == section]
        for section in SENTENCE_SECTIONS
    }
    value["report"]["actionability"] = copy.deepcopy(dict(actionability))
    value["report_projection"] = {"statement_count": len(rows), "omitted_statement_ids": []}
    if candidates:
        # Retain the same task-authored policy when a later parent S4 expands.
        value["report_projection"]["optional_statement_ids"] = [r["id"] for r in candidates]
    return fit_process_report_capacity(value, max_bytes=max_bytes, update_projection=update_projection)


def fit_process_report_capacity(
    report: Mapping[str, Any], *, max_bytes: int,
    update_projection: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Reuse declared whole-statement omissions after a trusted view update.

    Old views without an optional-id receipt remain strict. Required statements
    and the final statement of each section cannot be removed to make room.
    """
    value = copy.deepcopy(dict(report))
    projection = value["report_projection"]
    omitted = projection["omitted_statement_ids"]

    def refresh() -> None:
        gaps = value["report"]["gaps"]
        gaps[:] = [row for row in gaps if row["id"] != "report.capacity_omissions"]
        if omitted:
            gaps.append({"id": "report.capacity_omissions",
                "text": (f"Report capacity omitted {len(omitted)} optional statements from "
                         f"{projection['statement_count']} statements. Their complete evidence remains in the full derived artifact.")})
        if update_projection is not None:
            update_projection(value)

    refresh()
    for identity in projection.get("optional_statement_ids", []):
        if report_bytes(value) <= max_bytes:
            break
        for section in SENTENCE_SECTIONS:
            rows = value["report"][section]
            candidate = next((row for row in rows if row["id"] == identity), None)
            if candidate is not None and len(rows) > 1:
                rows.remove(candidate)
                omitted.append(identity)
                refresh()
                break
    validate_public_report(value, max_bytes=max_bytes)
    return value
