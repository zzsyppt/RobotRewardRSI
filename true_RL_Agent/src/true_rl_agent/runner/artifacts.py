"""Runner-owned artifact path layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunnerArtifactLayout:
    root: Path
    session_id: str

    @property
    def session_dir(self) -> Path:
        return self.root / self.session_id

    def round_dir(self, round_id: str) -> Path:
        return self.session_dir / round_id

    def train_dir(self, round_id: str) -> Path:
        return self.round_dir(round_id) / "train"

    def eval_dir(self, round_id: str) -> Path:
        return self.round_dir(round_id) / "eval"

    def checkpoints_dir(self, round_id: str) -> Path:
        return self.round_dir(round_id) / "checkpoints"

