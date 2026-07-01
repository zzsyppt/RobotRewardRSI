"""Dry-run isolation policy declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IsolationPolicy:
    kind: str = "dry_run"
    rootfs: str | None = None
    subprocess_boundary: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"dry_run", "subprocess", "rootfs"}:
            raise ValueError(f"unsupported isolation kind: {self.kind}")
        if self.kind == "rootfs" and not self.rootfs:
            raise ValueError("rootfs isolation requires rootfs")

