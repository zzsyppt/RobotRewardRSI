"""Prompt-facing candidate capability projection."""

from __future__ import annotations

from typing import Any


REWARD_INPUT_FIELDS = (
    "base_height",
    "roll",
    "pitch",
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "commanded_base_vel",
    "joint_pos",
    "joint_vel",
    "last_action",
    "push_active",
    "push_vec_public",
    "progress_fraction",
)


def reward_inputs_projection() -> dict[str, Any]:
    return {
        "schema_version": "true_rl_agent.reward_inputs.public.v1",
        "fields": list(REWARD_INPUT_FIELDS),
    }


def candidate_allowed_controls() -> list[str]:
    return ["RewardTerm", "DiagnosticTerm", "ScheduleTerm"]


def build_prompt_payload(*, public_context: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "true_rl_agent.prompt_payload.v1",
        "public_context": public_context,
        "reward_inputs": reward_inputs_projection(),
        "candidate_allowed_controls": candidate_allowed_controls(),
        "anti_tuning_notice": (
            "Changing only reward_weight, terrain_range, threshold, seed, or budget "
            "is parameter_only and cannot be a policy improvement accept."
        ),
        "runner_owned_extensions": {
            "observation_projection": "not_prompt_writable",
            "feature_projection": "not_prompt_writable",
        },
    }
    assert_public_only_prompt_payload(payload)
    return payload


def assert_public_only_prompt_payload(payload: Any) -> None:
    forbidden_key_parts = ("private", "hidden", "secret", "api_key", "token")
    forbidden_value_parts = ("sk-",)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if any(part in lowered for part in forbidden_key_parts):
                    raise ValueError(f"prompt payload contains forbidden key: {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, str):
            lowered = value.lower()
            if any(part in lowered for part in forbidden_value_parts):
                raise ValueError(f"prompt payload contains forbidden value at: {path}")

    visit(payload, "$")
