"""Task plugin registry."""

from __future__ import annotations

from true_rl_agent.tasks.base import TaskPlugin


class TaskRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, TaskPlugin] = {}

    def register(self, plugin: TaskPlugin) -> None:
        task_id = plugin.task_spec.task_id
        if task_id in self._plugins:
            raise ValueError(f"task already registered: {task_id}")
        self._plugins[task_id] = plugin

    def get(self, task_id: str) -> TaskPlugin:
        try:
            return self._plugins[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task: {task_id}") from exc
