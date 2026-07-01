"""Session directory management for AgentCore runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentSession:
    session_id: str
    root_dir: Path

    @property
    def session_dir(self) -> Path:
        return self.root_dir / self.session_id

    @property
    def ledger_path(self) -> Path:
        return self.session_dir / "ledger.jsonl"

    def round_dir(self, round_id: str) -> Path:
        return self.session_dir / "rounds" / round_id

    @classmethod
    def create(cls, *, root_dir: Path, session_id: str) -> "AgentSession":
        session = cls(session_id=session_id, root_dir=root_dir)
        (session.session_dir / "rounds").mkdir(parents=True, exist_ok=True)
        return session
