"""Shared candidate ScheduleTerm contract."""

from __future__ import annotations

from dataclasses import dataclass

REWARD_WEIGHT_SCALE_KEY = "reward_weight_scale"
REWARD_WEIGHT_CHANGE_COMPONENT = "reward_weight"
UNKNOWN_SCHEDULE_CHANGE_COMPONENT = "schedule"
REWARD_WEIGHT_SCALE_MIN = 0.0
REWARD_WEIGHT_SCALE_MAX = 10.0


@dataclass(frozen=True)
class ScheduleKeyContract:
    key: str
    changed_component: str
    effective_now: bool
    min_value: float
    max_value: float
    runtime_activation_record: str

    def bounds(self) -> tuple[float, float]:
        return (self.min_value, self.max_value)


SCHEDULE_KEY_CONTRACTS: dict[str, ScheduleKeyContract] = {
    REWARD_WEIGHT_SCALE_KEY: ScheduleKeyContract(
        key=REWARD_WEIGHT_SCALE_KEY,
        changed_component=REWARD_WEIGHT_CHANGE_COMPONENT,
        effective_now=True,
        min_value=REWARD_WEIGHT_SCALE_MIN,
        max_value=REWARD_WEIGHT_SCALE_MAX,
        runtime_activation_record=REWARD_WEIGHT_SCALE_KEY,
    )
}


def allowed_schedule_keys() -> tuple[str, ...]:
    return tuple(SCHEDULE_KEY_CONTRACTS)


def effective_schedule_keys() -> tuple[str, ...]:
    return tuple(
        key for key, contract in SCHEDULE_KEY_CONTRACTS.items() if contract.effective_now
    )


def schedule_key_bounds() -> dict[str, list[float]]:
    return {
        key: [contract.min_value, contract.max_value]
        for key, contract in SCHEDULE_KEY_CONTRACTS.items()
        if contract.effective_now
    }


def schedule_bounds_for_key(key: str) -> tuple[float, float] | None:
    contract = SCHEDULE_KEY_CONTRACTS.get(key)
    if contract is None:
        return None
    return contract.bounds()


def schedule_change_component(key: str) -> str:
    contract = SCHEDULE_KEY_CONTRACTS.get(key)
    if contract is None:
        return UNKNOWN_SCHEDULE_CHANGE_COMPONENT
    return contract.changed_component


def schedule_runtime_activation_record(key: str) -> str | None:
    contract = SCHEDULE_KEY_CONTRACTS.get(key)
    if contract is None or not contract.effective_now:
        return None
    return contract.runtime_activation_record
