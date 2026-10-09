"""Curriculum capability registry for first-version gated curriculum support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class CurriculumCapability:
    capability_id: str
    kind: str
    owner: str
    supported: bool
    effective_now: bool
    prompt_visible: bool = False
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.capability_id,
            "kind": self.kind,
            "owner": self.owner,
            "supported": self.supported,
            "effective_now": self.effective_now,
            "prompt_visible": self.prompt_visible,
            "description": self.description,
            "metadata": dict(self.metadata),
        }


class CapabilityRegistry:
    """Closed registry for curriculum capability ids and effectiveness labels."""

    def __init__(self, capabilities: Iterable[CurriculumCapability] = ()) -> None:
        self._capabilities: dict[str, CurriculumCapability] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: CurriculumCapability) -> None:
        if capability.capability_id in self._capabilities:
            raise ValueError(f"curriculum capability already registered: {capability.capability_id}")
        self._capabilities[capability.capability_id] = capability

    def get(self, capability_id: str) -> CurriculumCapability:
        try:
            return self._capabilities[capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown curriculum capability: {capability_id}") from exc

    def validate_ids(self, capability_ids: Iterable[str]) -> None:
        for capability_id in capability_ids:
            self.get(capability_id)

    def to_public_list(self, *, include_ineffective: bool = False) -> list[dict[str, Any]]:
        capabilities = [
            capability
            for capability in self._capabilities.values()
            if include_ineffective or capability.effective_now
        ]
        return [capability.to_public_dict() for capability in sorted(capabilities, key=lambda item: item.capability_id)]

    def prompt_effective_capabilities(self) -> list[dict[str, Any]]:
        return [
            capability.to_public_dict()
            for capability in sorted(self._capabilities.values(), key=lambda item: item.capability_id)
            if capability.prompt_visible and capability.effective_now
        ]


def build_default_curriculum_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry(
        (
            CurriculumCapability(
                capability_id="task_default_curriculum",
                kind="runner_mode",
                owner="runner",
                supported=True,
                effective_now=True,
                prompt_visible=False,
                description="Use the task-owned default training distribution without LLM schedule control.",
            ),
            CurriculumCapability(
                capability_id="reward_weight_scale",
                kind="schedule_key",
                owner="runner",
                supported=True,
                effective_now=False,
                prompt_visible=False,
                description="Reserved numeric reward scaling key; disabled as curriculum tuning in the first version.",
            ),
            CurriculumCapability(
                capability_id="external_push_curriculum",
                kind="train_distribution",
                owner="runner",
                supported=True,
                effective_now=False,
                prompt_visible=False,
                description="Future runner-owned push progression capability.",
            ),
            CurriculumCapability(
                capability_id="train_only_dr_scale",
                kind="train_distribution",
                owner="runner",
                supported=True,
                effective_now=False,
                prompt_visible=False,
                description="Future runner-owned train-only robustness variation.",
            ),
        )
    )
