"""Append-only public ledger helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from true_rl_agent.json_contract import (
    canonical_json_text,
    strict_json_object_loads,
)


def append_ledger_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_text(record) + "\n")


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = strict_json_object_loads(
                line,
                context=f"ledger line {line_number}",
            )
        except ValueError as exc:
            raise ValueError(f"invalid ledger JSON at line {line_number}: {path}") from exc
        records.append(value)
    return records
