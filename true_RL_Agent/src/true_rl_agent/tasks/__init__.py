"""Task plugin interfaces and task-specific runtime packages."""

from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.tasks.prompt_adapter import (
    NeutralTaskPromptAdapter,
    TaskPromptAdapter,
)

__all__ = [
    "NeutralTaskPromptAdapter",
    "TaskInvariantSpec",
    "TaskPlugin",
    "TaskPromptAdapter",
    "TaskSpec",
]
