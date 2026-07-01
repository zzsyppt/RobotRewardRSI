"""Append-only public ledger helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_ledger_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid ledger JSON at line {line_number}: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"ledger line {line_number} is not an object: {path}")
        records.append(value)
    return records
