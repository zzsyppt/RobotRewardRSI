"""Registry for task-owned application composition providers."""

from __future__ import annotations

from true_rl_agent.composition.base import TaskCompositionProvider


class TaskCompositionRegistry:
    def __init__(self, *, default_task_id: str) -> None:
        if not isinstance(default_task_id, str) or not default_task_id:
            raise ValueError("default_task_id must be non-empty")
        self._default_task_id = default_task_id
        self._providers: dict[str, TaskCompositionProvider] = {}

    @property
    def default_task_id(self) -> str:
        return self._default_task_id

    def register(self, provider: TaskCompositionProvider) -> None:
        if not isinstance(provider, TaskCompositionProvider):
            raise TypeError(
                "task composition provider must implement TaskCompositionProvider"
            )
        task_id = provider.task_id
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task composition provider task_id must be non-empty")
        if task_id in self._providers:
            raise ValueError(f"task composition already registered: {task_id}")
        self._providers[task_id] = provider

    def get(self, task_id: str) -> TaskCompositionProvider:
        try:
            return self._providers[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task composition: {task_id}") from exc

    def default_provider(self) -> TaskCompositionProvider:
        try:
            return self.get(self.default_task_id)
        except KeyError as exc:
            raise ValueError(
                f"default task composition is not registered: {self.default_task_id}"
            ) from exc

    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))
