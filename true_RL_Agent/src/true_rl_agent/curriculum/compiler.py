"""Validate and explain first-version curriculum plans without tuning schedules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from true_rl_agent.curriculum.capabilities import CapabilityRegistry
from true_rl_agent.curriculum.schema import CurriculumPlan


@dataclass(frozen=True)
class CompiledCurriculum:
    effective_plan: CurriculumPlan
    effective_training_config: Mapping[str, Any]
    proposal_status: str
    lineage: Mapping[str, Any]
    explanations: tuple[str, ...] = field(default_factory=tuple)
    disabled_proposal_capabilities: tuple[str, ...] = field(default_factory=tuple)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "mode": self.effective_plan.mode,
            "llm_controlled": self.effective_plan.llm_controlled,
            "proposal_status": self.proposal_status,
            "lineage": dict(self.lineage),
            "explanations": list(self.explanations),
            "disabled_proposal_capabilities": list(self.disabled_proposal_capabilities),
        }


class CurriculumCompiler:
    """First-version compiler: validate ids, freeze config, and record lineage."""

    def compile(
        self,
        *,
        plan: CurriculumPlan,
        task_default_training_config: Mapping[str, Any],
        registry: CapabilityRegistry,
        llm_proposal: Mapping[str, Any] | None = None,
    ) -> CompiledCurriculum:
        proposal_capabilities = self._extract_proposal_capabilities(llm_proposal)
        registry.validate_ids(proposal_capabilities)

        effective_training_config = dict(task_default_training_config)
        proposal_status = "disabled_by_phase" if llm_proposal is not None else "not_provided"
        default_hash = self._stable_hash(task_default_training_config)
        effective_hash = self._stable_hash(effective_training_config)
        explanations = [
            self._mode_explanation(plan.mode),
            "curriculum compiler validates registry ids and lineage only; it does not optimize or rewrite schedules",
        ]
        if llm_proposal is not None:
            explanations.append(
                "disabled_by_phase: first-version LLM curriculum proposals are recorded but do not enter TrainJobSpec"
            )

        lineage = {
            "mode": plan.mode,
            "source": plan.mode,
            "llm_controlled": plan.llm_controlled,
            "task_default_config_sha256": default_hash,
            "effective_config_sha256": effective_hash,
            "proposal_status": proposal_status,
        }
        return CompiledCurriculum(
            effective_plan=plan,
            effective_training_config=effective_training_config,
            proposal_status=proposal_status,
            lineage=lineage,
            explanations=tuple(explanations),
            disabled_proposal_capabilities=tuple(proposal_capabilities),
        )

    def _extract_proposal_capabilities(self, proposal: Mapping[str, Any] | None) -> tuple[str, ...]:
        if proposal is None:
            return ()
        capability_ids: list[str] = []
        for design in self._proposal_designs(proposal):
            self._extend_capabilities_from_mapping(capability_ids, design)
        return tuple(dict.fromkeys(capability_ids))

    def _proposal_designs(self, proposal: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        designs: list[Mapping[str, Any]] = [proposal]
        curriculum_design = proposal.get("curriculum_design")
        if isinstance(curriculum_design, Mapping):
            designs.append(curriculum_design)
        operation_diff = proposal.get("operation_diff")
        if isinstance(operation_diff, Mapping):
            nested_design = operation_diff.get("curriculum_design")
            if isinstance(nested_design, Mapping):
                designs.append(nested_design)
        return tuple(designs)

    def _extend_capabilities_from_mapping(self, capability_ids: list[str], design: Mapping[str, Any]) -> None:
        top_level = design.get("capabilities")
        if top_level is not None:
            self._extend_capability_list(capability_ids, top_level, "capabilities")
        stage_list = design.get("stage_list", design.get("stages"))
        if stage_list is None:
            return
        if not isinstance(stage_list, list):
            raise ValueError("curriculum proposal stage_list must be a list")
        for index, stage in enumerate(stage_list):
            if not isinstance(stage, Mapping):
                raise ValueError(f"curriculum proposal stage_list[{index}] must be an object")
            stage_capabilities = stage.get("capabilities")
            if stage_capabilities is not None:
                self._extend_capability_list(capability_ids, stage_capabilities, f"stage_list[{index}].capabilities")

    def _extend_capability_list(self, capability_ids: list[str], value: Any, path: str) -> None:
        if not isinstance(value, list):
            raise ValueError(f"curriculum proposal {path} must be list[str]")
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"curriculum proposal {path}[{index}] must be a string")
            capability_ids.append(item)

    def _mode_explanation(self, mode: str) -> str:
        if mode == "frozen":
            return "frozen: preserve the supplied task default training config exactly"
        if mode == "runner_owned":
            return "runner_owned: runner may own fixed curriculum behavior, but LLM schedule control is disabled"
        if mode == "task_default":
            return "task_default: use the task plugin default train distribution and eval protocol"
        return f"{mode}: unsupported mode"

    def _stable_hash(self, value: Mapping[str, Any]) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
