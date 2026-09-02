"""Task-neutral lazy composition provider lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any

from true_rl_agent.composition.base import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    TaskCompositionError,
    TaskCompositionProvider,
)


class LazyTaskCompositionProvider(TaskCompositionProvider):
    """Load and cache one concrete provider only when its task is selected."""

    def __init__(
        self,
        *,
        task_id: str,
        provider_factory: Callable[[], TaskCompositionProvider],
    ) -> None:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("lazy task composition task_id must be non-empty")
        if not callable(provider_factory):
            raise TypeError("lazy task composition provider_factory must be callable")
        self._task_id = task_id
        self._provider_factory = provider_factory
        self._provider: TaskCompositionProvider | None = None
        self._provider_lock = Lock()

    @property
    def task_id(self) -> str:
        return self._task_id

    def cli_defaults(self) -> Mapping[str, Any]:
        return self._load().cli_defaults()

    def add_cli_arguments(self, parser: Any) -> None:
        self._load().add_cli_arguments(parser)

    def resolve(
        self,
        *,
        values: Mapping[str, Any],
        context: TaskCompositionContext,
    ) -> ResolvedTaskComposition:
        return self._load().resolve(values=values, context=context)

    def run_audit(
        self,
        *,
        composition: ResolvedTaskComposition,
        output_dir: Path,
        rounds: int,
        session_id: str | None,
    ) -> dict[str, Any]:
        return self._load().run_audit(
            composition=composition,
            output_dir=output_dir,
            rounds=rounds,
            session_id=session_id,
        )

    def _load(self) -> TaskCompositionProvider:
        provider = self._provider
        if provider is not None:
            return provider
        with self._provider_lock:
            provider = self._provider
            if provider is not None:
                return provider
            provider = self._provider_factory()
            if not isinstance(provider, TaskCompositionProvider) or provider is self:
                raise TaskCompositionError(
                    "lazy task composition factory must return TaskCompositionProvider"
                )
            if provider.task_id != self.task_id:
                raise TaskCompositionError(
                    "lazy task composition factory returned a different task_id"
                )
            self._provider = provider
            return provider


__all__ = ["LazyTaskCompositionProvider"]
