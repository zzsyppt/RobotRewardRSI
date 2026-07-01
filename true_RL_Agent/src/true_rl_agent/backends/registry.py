"""Backend adapter registry."""

from __future__ import annotations

from true_rl_agent.backends.base import BackendAdapter


class BackendRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BackendAdapter] = {}

    def register(self, adapter: BackendAdapter) -> None:
        backend_id = adapter.backend_id
        if backend_id in self._adapters:
            raise ValueError(f"backend already registered: {backend_id}")
        self._adapters[backend_id] = adapter

    def get(self, backend_id: str) -> BackendAdapter:
        try:
            return self._adapters[backend_id]
        except KeyError as exc:
            raise KeyError(f"unknown backend: {backend_id}") from exc

