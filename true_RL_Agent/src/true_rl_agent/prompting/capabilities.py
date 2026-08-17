"""Candidate authoring capability registry for prompt payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.agent_core.provider_contract import (
    PROMPT_ADVERTISED_DECLARED_COMPONENTS,
)
from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.candidates.capability_readiness import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
    READINESS_REQUIRED_CAPABILITY_IDS,
    require_readiness_entry,
)
from true_rl_agent.candidates.schedule_contract import (
    REWARD_WEIGHT_CHANGE_COMPONENT,
    REWARD_WEIGHT_SCALE_KEY,
    allowed_schedule_keys,
    effective_schedule_keys,
)
from true_rl_agent.candidates.verifier import static_verifier_prompt_rules
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload
from true_rl_agent.prompting.runner_capability_schemas import (
    CURRICULUM_DESIGN_CAPABILITY_ID,
    RUNNER_CAPABILITY_READINESS_REQUIREMENTS,
    TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
    runner_capability_readiness_gate,
    runner_capability_schema_by_id,
    runner_capability_schema_ids,
)

CANDIDATE_AUTHORING_CAPABILITIES_SCHEMA_VERSION = "true_rl_agent.candidate_authoring_capabilities.v1"
TRAINING_COMPONENTS_SIGNATURE = (
    "TrainingComponents(name: str, reward_terms: list[RewardTerm] = [], "
    "diagnostics: list[DiagnosticTerm] = [], schedules: list[ScheduleTerm] = [], "
    "metadata: dict = {}, rationale: str = '')"
)
COMPONENT_ALLOWED_IMPORT = (
    "from true_rl_agent.candidates.components import "
    "DiagnosticTerm, RewardInputs, RewardTerm, ScheduleTerm, TrainingComponents"
)
COMPONENT_CONTRACT_FORBIDDEN_SURFACES = (
    "runner state",
    "environment objects",
    "optimizer or PPO hyperparameters",
    "eval protocol",
    "success metric",
    "seeds",
    "budgets",
    "checkpoints",
    "filesystem",
)
MINIMAL_COMPONENTS_PY = (
    "from __future__ import annotations\n\n"
    "from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents\n\n\n"
    "def stance_margin(inp: RewardInputs):\n"
    "    return inp.base_height - 0.5 * (inp.roll * inp.roll + inp.pitch * inp.pitch)\n\n\n"
    "def build_training_components() -> TrainingComponents:\n"
    "    return TrainingComponents(\n"
    "        name=\"stance_margin_candidate\",\n"
    "        reward_terms=[RewardTerm(name=\"stance_margin\", weight=1.0, fn=stance_margin)],\n"
    "        diagnostics=[],\n"
    "        schedules=[],\n"
    "        rationale=\"Add a public reward callable over RewardInputs.\",\n"
    "    )\n"
)


@dataclass(frozen=True)
class CandidateAuthoringCapability:
    capability_id: str
    kind: str
    owner: str
    prompt_visible: bool
    candidate_writable: bool
    effective_now: bool
    materializer_owner: str
    verifier_owner: str
    runtime_effect: str
    public_feedback_projection: str
    component_class: str | None = None
    component_signature: str | None = None
    declared_changed_component: str | None = None
    derived_change_components: tuple[str, ...] = ()
    candidate_allowed_actions: tuple[str, ...] = ()
    source_items: tuple[str, ...] = ()
    effectiveness: str = "not_effective"
    allowed_schedule_keys: tuple[str, ...] = ()
    effective_schedule_keys: tuple[str, ...] = ()
    disabled_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "capability_id": self.capability_id,
            "kind": self.kind,
            "owner": self.owner,
            "prompt_visible": self.prompt_visible,
            "candidate_writable": self.candidate_writable,
            "effective_now": self.effective_now,
            "materializer_owner": self.materializer_owner,
            "verifier_owner": self.verifier_owner,
            "runtime_effect": self.runtime_effect,
            "public_feedback_projection": self.public_feedback_projection,
        }
        optional_fields = {
            "component_class": self.component_class,
            "component_signature": self.component_signature,
            "declared_changed_component": self.declared_changed_component,
            "derived_change_components": list(self.derived_change_components),
            "candidate_allowed_actions": list(self.candidate_allowed_actions),
            "source_items": list(self.source_items),
            "effectiveness": self.effectiveness,
            "allowed_schedule_keys": list(self.allowed_schedule_keys),
            "effective_schedule_keys": list(self.effective_schedule_keys),
            "disabled_reason": self.disabled_reason,
            "metadata": dict(self.metadata),
        }
        for key, item in optional_fields.items():
            if item in (None, [], {}):
                continue
            value[key] = item
        assert_public_only_prompt_payload(value)
        return value


class CandidateAuthoringCapabilityRegistry:
    def __init__(self, capabilities: Sequence[CandidateAuthoringCapability]) -> None:
        self._capabilities = tuple(capabilities)
        self._validate_unique_ids()

    @property
    def capabilities(self) -> tuple[CandidateAuthoringCapability, ...]:
        return self._capabilities

    def capability_ids(self) -> tuple[str, ...]:
        return tuple(capability.capability_id for capability in self._capabilities)

    def prompt_visible_capabilities(self) -> tuple[CandidateAuthoringCapability, ...]:
        return tuple(
            capability for capability in self._capabilities if capability.prompt_visible
        )

    def to_prompt_section(
        self,
        *,
        capability_profile: CandidateCapabilityProfile | None = None,
        task_specific_metadata: Mapping[str, Any] | None = None,
        authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        selected_ids = self._selected_ids(capability_profile)
        projected_capabilities = [
            self._project_capability(
                capability,
                capability_profile=capability_profile,
                authoring_capability_readiness=authoring_capability_readiness,
            )
            for capability in self._capabilities
            if capability.capability_id in selected_ids
        ]
        visible_capabilities = [
            capability
            for capability in projected_capabilities
            if capability.get("prompt_visible") is True
        ]
        section = {
            "schema_version": CANDIDATE_AUTHORING_CAPABILITIES_SCHEMA_VERSION,
            "registry_owner": "agent_core",
            "registry_model": "closed_registry_with_prompt_visibility",
            "prompt_visible_capabilities": visible_capabilities,
            "disabled_or_reserved_capabilities": [],
            "provider_declared_components_prompt_advertised": (
                _declared_components(visible_capabilities)
            ),
            "policies": {
                "non_prompt_visible_capabilities_are_not_candidate_writable": True,
                "verifier_derived_change_set_is_authoritative": True,
                "task_specific_metadata_owner": "task_prompt_adapter",
                "runner_owned_capabilities_require_materializer_and_verifier_before_prompt_visibility": True,
                "runner_owned_schema_preview_is_not_provider_output": True,
            },
        }
        if capability_profile is not None:
            section["task_capability_profile_binding"] = {
                "profile_id": capability_profile.profile_id,
                "profile_hash": capability_profile.profile_hash,
            }
        if task_specific_metadata:
            section["task_specific_metadata"] = _task_specific_metadata(task_specific_metadata)
        assert_public_only_prompt_payload(section)
        return section

    def _selected_ids(
        self,
        capability_profile: CandidateCapabilityProfile | None,
    ) -> frozenset[str]:
        known = frozenset(self.capability_ids())
        if capability_profile is None:
            return known
        if not isinstance(capability_profile, CandidateCapabilityProfile):
            raise TypeError(
                "capability_profile must be a CandidateCapabilityProfile"
            )
        unknown = sorted(set(capability_profile.capability_ids) - known)
        if unknown:
            raise ValueError(
                "capability profile selects unknown capability ID(s): "
                + ", ".join(unknown)
            )
        return frozenset(capability_profile.capability_ids)

    def _project_capability(
        self,
        capability: CandidateAuthoringCapability,
        *,
        capability_profile: CandidateCapabilityProfile | None,
        authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        value = capability.to_public_dict()
        if capability.capability_id in READINESS_REQUIRED_CAPABILITY_IDS:
            if capability_profile is None:
                return value
            assert capability_profile.profile_hash is not None
            readiness_value = None
            if isinstance(authoring_capability_readiness, Mapping):
                readiness_value = authoring_capability_readiness.get(
                    capability.capability_id
                )
            if readiness_value is None:
                return value
            readiness = require_readiness_entry(
                capability.capability_id,
                readiness_value,
                expected_profile_hash=capability_profile.profile_hash,
            )
            value.update(
                {
                    "prompt_visible": True,
                    "candidate_writable": True,
                    "effective_now": True,
                    "effectiveness": "effective_now",
                    "activation_mode": "complete_replacement",
                    "readiness": readiness,
                }
            )
            value.pop("disabled_reason", None)
            assert_public_only_prompt_payload(value)
            return value
        if capability.capability_id not in runner_capability_schema_ids():
            return value

        readiness_values: Mapping[str, Any] | None = None
        if isinstance(authoring_capability_readiness, Mapping):
            candidate_values = authoring_capability_readiness.get(
                capability.capability_id
            )
            if isinstance(candidate_values, Mapping):
                readiness_values = candidate_values

        gate = runner_capability_readiness_gate(capability.capability_id, readiness_values)
        value["readiness_gate"] = gate
        value["readiness_requirements"] = list(RUNNER_CAPABILITY_READINESS_REQUIREMENTS)
        if gate["status"] == "ready":
            value["prompt_visible"] = True
            value["candidate_writable"] = False
            value["effective_now"] = False
            value["exposure_mode"] = "schema_preview_only"
            value["schema_status"] = "schema_preview_ready"
            value["output_schema"] = runner_capability_schema_by_id(capability.capability_id)
        else:
            value["prompt_visible"] = False
            value["candidate_writable"] = False
            value["effective_now"] = False
            value["exposure_mode"] = "reserved_until_ready"
            value["schema_status"] = "schema_draft_hidden_until_ready"
            value["schema_ref"] = (
                f"true_rl_agent.runner_capability_schema.{capability.capability_id}.v1"
            )
        assert_public_only_prompt_payload(value)
        return value

    def _validate_unique_ids(self) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []
        for capability in self._capabilities:
            if capability.capability_id in seen:
                duplicates.append(capability.capability_id)
            seen.add(capability.capability_id)
        if duplicates:
            raise ValueError(
                "duplicate candidate authoring capability id(s): "
                + ", ".join(sorted(set(duplicates)))
            )


def default_candidate_authoring_registry() -> CandidateAuthoringCapabilityRegistry:
    return CandidateAuthoringCapabilityRegistry(
        (
            CandidateAuthoringCapability(
                capability_id=COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
                kind="complete_reward_program",
                owner="candidate_components",
                prompt_visible=False,
                candidate_writable=False,
                effective_now=False,
                materializer_owner="provider_candidate_package",
                verifier_owner="composition_candidate_admission_policy",
                runtime_effect=(
                    "all weighted RewardTerm outputs replace the complete training reward"
                ),
                public_feedback_projection=(
                    "task-owned complete reward activation and bounded term summaries"
                ),
                component_class="RewardTerm",
                component_signature=(
                    "RewardTerm(name: str, weight: float, "
                    "fn: Callable[[task RewardInputs], Any], "
                    "clip_min: float | None = None, "
                    "clip_max: float | None = None, description: str = '')"
                ),
                declared_changed_component="reward_callable",
                derived_change_components=("reward_callable",),
                candidate_allowed_actions=(
                    "replace the complete reward with pure RewardTerm callables",
                    "use only fields declared by the task reward input contract",
                    "set finite weights and optional bounded output clips",
                ),
                source_items=(
                    "current_best/training_code/components.py",
                    "api/candidate_components.py",
                ),
                disabled_reason="complete_reward_runtime_not_ready",
            ),
            CandidateAuthoringCapability(
                capability_id="reward_term",
                kind="reward_callable",
                owner="candidate_components",
                prompt_visible=True,
                candidate_writable=True,
                effective_now=True,
                component_class="RewardTerm",
                component_signature=(
                    "RewardTerm(name: str, weight: float, fn: Callable[[RewardInputs], Any], "
                    "clip_min: float | None = None, clip_max: float | None = None, "
                    "description: str = '')"
                ),
                declared_changed_component="reward_callable",
                derived_change_components=("reward_callable",),
                candidate_allowed_actions=(
                    "define pure callables over prompt-visible RewardInputs",
                    "combine public tensors with bounded arithmetic",
                    "set reward weight and optional output clips inside the candidate package",
                ),
                source_items=("api/candidate_components.py",),
                effectiveness="effective_now",
                materializer_owner="provider_candidate_package",
                verifier_owner="static_candidate_verifier",
                runtime_effect="guarded vector output can be added to training reward",
                public_feedback_projection="runtime activation and public reward contribution summaries",
            ),
            CandidateAuthoringCapability(
                capability_id="diagnostic_term",
                kind="diagnostic_callable",
                owner="candidate_components",
                prompt_visible=True,
                candidate_writable=True,
                effective_now=True,
                component_class="DiagnosticTerm",
                component_signature=(
                    "DiagnosticTerm(name: str, fn: Callable[[RewardInputs], Any], "
                    "description: str = '')"
                ),
                declared_changed_component="diagnostic",
                derived_change_components=("diagnostic",),
                candidate_allowed_actions=(
                    "define pure diagnostic callables over prompt-visible RewardInputs",
                    "expose bounded public measurements for later feedback",
                ),
                source_items=("api/candidate_components.py",),
                effectiveness="public_feedback_only",
                materializer_owner="provider_candidate_package",
                verifier_owner="static_candidate_verifier",
                runtime_effect="summarized for future prompt feedback; not a decision rule",
                public_feedback_projection="public diagnostic output summaries",
            ),
            CandidateAuthoringCapability(
                capability_id=f"{REWARD_WEIGHT_SCALE_KEY}_schedule",
                kind="schedule_term",
                owner="candidate_components",
                prompt_visible=True,
                candidate_writable=True,
                effective_now=True,
                component_class="ScheduleTerm",
                component_signature="ScheduleTerm(name: str, key: str, value: float)",
                declared_changed_component=REWARD_WEIGHT_CHANGE_COMPONENT,
                derived_change_components=(REWARD_WEIGHT_CHANGE_COMPONENT,),
                candidate_allowed_actions=(
                    "use ScheduleTerm only with allowed effective keys",
                    f"set {REWARD_WEIGHT_SCALE_KEY} through the allowed ScheduleTerm key",
                ),
                source_items=("api/candidate_components.py",),
                effectiveness="key_specific",
                allowed_schedule_keys=allowed_schedule_keys(),
                effective_schedule_keys=effective_schedule_keys(),
                materializer_owner="provider_candidate_package",
                verifier_owner="static_candidate_verifier",
                runtime_effect="only currently effective schedule keys alter training",
                public_feedback_projection="public candidate surface and mechanism attribution summaries",
            ),
            CandidateAuthoringCapability(
                capability_id=CURRICULUM_DESIGN_CAPABILITY_ID,
                kind="train_protocol_design",
                owner="runner",
                prompt_visible=False,
                candidate_writable=False,
                effective_now=False,
                materializer_owner="runner",
                verifier_owner="runner",
                runtime_effect="reserved until runner curriculum compiler and verifier are enabled",
                public_feedback_projection="none",
                disabled_reason="not_prompt_writable_in_this_prompt_upgrade",
            ),
            CandidateAuthoringCapability(
                capability_id=TRAIN_ENVIRONMENT_DESIGN_CAPABILITY_ID,
                kind="train_distribution_design",
                owner="runner",
                prompt_visible=False,
                candidate_writable=False,
                effective_now=False,
                materializer_owner="runner",
                verifier_owner="runner",
                runtime_effect="reserved until runner environment compiler and verifier are enabled",
                public_feedback_projection="none",
                disabled_reason="not_prompt_writable_in_this_prompt_upgrade",
            ),
            CandidateAuthoringCapability(
                capability_id="observation_projection",
                kind="observation_interface",
                owner="runner",
                prompt_visible=False,
                candidate_writable=False,
                effective_now=False,
                materializer_owner="runner",
                verifier_owner="runner",
                runtime_effect="not prompt writable",
                public_feedback_projection="none",
                disabled_reason="runner_owned_extension",
            ),
            CandidateAuthoringCapability(
                capability_id="feature_projection",
                kind="feature_interface",
                owner="runner",
                prompt_visible=False,
                candidate_writable=False,
                effective_now=False,
                materializer_owner="runner",
                verifier_owner="runner",
                runtime_effect="not prompt writable",
                public_feedback_projection="none",
                disabled_reason="runner_owned_extension",
            ),
        )
    )


def candidate_authoring_capabilities(
    *,
    capability_profile: CandidateCapabilityProfile | None = None,
    task_specific_metadata: Mapping[str, Any] | None = None,
    authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return default_candidate_authoring_registry().to_prompt_section(
        capability_profile=capability_profile,
        task_specific_metadata=task_specific_metadata,
        authoring_capability_readiness=authoring_capability_readiness,
    )


def candidate_authoring_capability_ids() -> tuple[str, ...]:
    return default_candidate_authoring_registry().capability_ids()


def build_candidate_capability_profile(
    *,
    profile_id: str,
    capability_ids: Sequence[str],
) -> CandidateCapabilityProfile:
    return CandidateCapabilityProfile.build(
        profile_id=profile_id,
        capability_ids=capability_ids,
        known_capability_ids=candidate_authoring_capability_ids(),
    )


def prompt_visible_component_classes(capabilities: Mapping[str, Any] | None = None) -> list[str]:
    section = _capability_section(capabilities)
    return [
        str(capability["component_class"])
        for capability in _prompt_visible_capabilities(section)
        if isinstance(capability.get("component_class"), str)
    ]


def candidate_component_contract_from_capabilities(
    capabilities: Mapping[str, Any] | None = None,
    *,
    verifier_prompt_rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    section = _capability_section(capabilities)
    visible = _prompt_visible_capabilities(section)
    if not visible:
        contract = {
            "schema_version": "true_rl_agent.candidate_component_contract.v1",
            "capability_schema_ref": section["schema_version"],
            "status": "unavailable",
            "reason": "no_prompt_visible_effective_capability",
            "classes": {},
            "allowed_schedule_keys": [],
            "forbidden_surfaces": list(COMPONENT_CONTRACT_FORBIDDEN_SURFACES),
        }
        assert_public_only_prompt_payload(contract)
        return contract

    classes = {
        str(capability["component_class"]): str(capability["component_signature"])
        for capability in visible
        if isinstance(capability.get("component_class"), str)
        and isinstance(capability.get("component_signature"), str)
    }
    classes["TrainingComponents"] = TRAINING_COMPONENTS_SIGNATURE
    allowed_names = sorted({"TrainingComponents", *classes})
    if any(name in classes for name in ("RewardTerm", "DiagnosticTerm")):
        allowed_names.append("RewardInputs")
        allowed_names.sort()
    if verifier_prompt_rules is None:
        verifier_rules = static_verifier_prompt_rules()
    elif not isinstance(verifier_prompt_rules, Mapping):
        raise TypeError("candidate verifier prompt rules must be an object")
    else:
        verifier_rules = dict(verifier_prompt_rules)
    verifier_allowed_imports = verifier_rules.get("allowed_imports")
    if not isinstance(verifier_allowed_imports, Mapping):
        raise ValueError("candidate verifier prompt rules omit allowed imports")
    verifier_component_names = verifier_allowed_imports.get(
        "true_rl_agent.candidates.components"
    )
    if not isinstance(verifier_component_names, list | tuple):
        raise ValueError(
            "candidate verifier prompt rules omit the component import allowlist"
        )
    missing_imports = set(allowed_names) - set(verifier_component_names)
    if missing_imports:
        raise ValueError(
            "candidate verifier blocks prompt-visible component imports: "
            + ", ".join(sorted(missing_imports))
        )
    verifier_rules["allowed_imports"] = {
        "true_rl_agent.candidates.components": allowed_names,
    }
    contract = {
        "schema_version": "true_rl_agent.candidate_component_contract.v1",
        "capability_schema_ref": section["schema_version"],
        "status": "available",
        "module": "true_rl_agent.candidates.components",
        "entrypoint": "training_code.components:build_training_components",
        "builder_signature": "def build_training_components() -> TrainingComponents:",
        "allowed_import": (
            "from true_rl_agent.candidates.components import "
            + ", ".join(allowed_names)
        ),
        "classes": classes,
        "allowed_schedule_keys": _allowed_schedule_keys(section),
        "forbidden_surfaces": list(COMPONENT_CONTRACT_FORBIDDEN_SURFACES),
        "static_verifier_rules": verifier_rules,
    }
    complete_replacement = any(
        capability.get("capability_id")
        == COMPLETE_REWARD_PROGRAM_CAPABILITY_ID
        for capability in visible
    )
    if complete_replacement:
        contract["complete_reward_runtime_semantics"] = {
            "composition_order": [
                "raw_term = fn(task_reward_inputs)",
                "guarded_term = clip(raw_term, clip_min, clip_max)",
                "weighted_term = weight * guarded_term",
                "training_reward = sum(all weighted terms)",
            ],
            "activation_mode": "complete_replacement",
            "base_reward_included": False,
            "reference_reward_called": False,
            "optimization_direction": (
                "Training maximizes the sum of candidate-defined weighted terms."
            ),
            "source_requirement": (
                "Return one complete training_code/components.py program."
            ),
        }
    elif "RewardTerm" in classes:
        contract["reward_term_runtime_semantics"] = {
            "composition_order": [
                "raw_output = fn(reward_inputs)",
                "guarded_output = clip(raw_output, clip_min, clip_max)",
                "weighted_contribution = weight * guarded_output",
                "training_reward = training_reward + weighted_contribution",
            ],
            "optimization_direction": (
                "Training maximizes total reward: positive weighted contributions reinforce behavior "
                "and negative weighted contributions penalize behavior."
            ),
            "penalty_sign_rule": (
                "With a positive weight, a penalty callable must return non-positive values; "
                "returning a positive cost rewards the behavior instead of penalizing it."
            ),
        }
        contract["minimal_components_py"] = MINIMAL_COMPONENTS_PY
    assert_public_only_prompt_payload(contract)
    return contract


def control_effectiveness_contract_from_capabilities(
    capabilities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    section = _capability_section(capabilities)
    component_controls = {}
    for capability in _prompt_visible_capabilities(section):
        component_class = capability.get("component_class")
        if not isinstance(component_class, str):
            continue
        component_controls[component_class] = {
            "allowed": bool(capability.get("candidate_writable", False)),
            "effectiveness": str(capability.get("effectiveness", "not_effective")),
            "training_effect": str(capability.get("runtime_effect", "not wired")),
        }
    contract = {
        "schema_version": "true_rl_agent.control_effectiveness_contract.v1",
        "capability_schema_ref": section["schema_version"],
        "component_controls": component_controls,
        "schedule_controls": {
            "allowed_keys": _allowed_schedule_keys(section),
            "effective_keys": _effective_schedule_keys(section),
        },
    }
    assert_public_only_prompt_payload(contract)
    return contract


def _capability_section(capabilities: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return capabilities if capabilities is not None else candidate_authoring_capabilities()


def _declared_components(
    capabilities: Sequence[Mapping[str, Any]],
) -> list[str]:
    selected = {
        capability.get("declared_changed_component")
        for capability in capabilities
        if capability.get("candidate_writable") is True
        and capability.get("effective_now") is True
    }
    return [
        component
        for component in PROMPT_ADVERTISED_DECLARED_COMPONENTS
        if component in selected
    ]


def _task_specific_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    clean = sanitize_prompt_public_context(dict(value))
    assert_public_only_prompt_payload(clean)
    return clean


def _prompt_visible_capabilities(section: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = section.get("prompt_visible_capabilities")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _allowed_schedule_keys(section: Mapping[str, Any]) -> list[str]:
    return _unique_keys(section, "allowed_schedule_keys")


def _effective_schedule_keys(section: Mapping[str, Any]) -> list[str]:
    return _unique_keys(section, "effective_schedule_keys")


def _unique_keys(section: Mapping[str, Any], field_name: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for capability in _prompt_visible_capabilities(section):
        value = capability.get(field_name)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, str) or item in seen:
                continue
            keys.append(item)
            seen.add(item)
    return keys
