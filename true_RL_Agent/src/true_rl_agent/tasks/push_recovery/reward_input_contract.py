"""Push-recovery task-owned RewardInputs contract helpers."""

from __future__ import annotations

from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryRewardInputs

PUBLIC_PUSH_SIGNAL_REWARD_INPUT_FIELDS = ("push_active", "push_vec_public")
RESERVED_ZERO_FALLBACK_REWARD_INPUT_FIELDS: tuple[str, ...] = ()


def push_recovery_reward_input_fields() -> tuple[str, ...]:
    fields = [name for name in PushRecoveryRewardInputs.field_names() if name not in PUBLIC_PUSH_SIGNAL_REWARD_INPUT_FIELDS]
    insert_index = fields.index("progress_fraction") if "progress_fraction" in fields else len(fields)
    fields[insert_index:insert_index] = list(PUBLIC_PUSH_SIGNAL_REWARD_INPUT_FIELDS)
    return tuple(fields)


def public_push_signal_reward_input_fields() -> tuple[str, ...]:
    fields = set(push_recovery_reward_input_fields())
    missing = [name for name in PUBLIC_PUSH_SIGNAL_REWARD_INPUT_FIELDS if name not in fields]
    if missing:
        raise RuntimeError(f"public push reward input fields are not in PushRecoveryRewardInputs: {missing}")
    return PUBLIC_PUSH_SIGNAL_REWARD_INPUT_FIELDS


def reserved_zero_fallback_reward_input_fields() -> tuple[str, ...]:
    fields = set(push_recovery_reward_input_fields())
    missing = [name for name in RESERVED_ZERO_FALLBACK_REWARD_INPUT_FIELDS if name not in fields]
    if missing:
        raise RuntimeError(f"reserved push reward input fields are not in PushRecoveryRewardInputs: {missing}")
    return RESERVED_ZERO_FALLBACK_REWARD_INPUT_FIELDS


def prompt_effective_reward_input_fields() -> tuple[str, ...]:
    reserved = set(reserved_zero_fallback_reward_input_fields())
    return tuple(name for name in push_recovery_reward_input_fields() if name not in reserved)
