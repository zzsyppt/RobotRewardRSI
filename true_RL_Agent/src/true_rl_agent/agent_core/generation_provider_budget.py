"""Provider reservations and actual usage from the frozen session budget."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from true_rl_agent.agent_core.generation_contract import GENERATION_LANE_IDS
from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider
from true_rl_agent.prompt_artifacts import (
    AuthoringEvidenceRef, PromptPayloadRef, RoundAuthoringJournal,
    verify_authoring_evidence, verify_prompt_payload,
)

@dataclass(frozen=True)
class CostGuardedProvider:
    """Apply the frozen session provider cap immediately before transport."""

    delegate: OpenAICompatibleCandidateProvider
    output_dir: Path
    generation_id: str
    lane_id: str
    provider_caps: Mapping[str, Any]
    max_generations: int = 1
    allow_conservative_interrupted_recovery: bool = False

    def next_authoring_response(
        self,
        *,
        prompt_payload: dict[str, Any],
    ) -> Any:
        _assert_provider_request_budget(
            output_dir=self.output_dir,
            generation_id=self.generation_id,
            current_lane_id=self.lane_id,
            current_prompt_payload=prompt_payload,
            provider_caps=self.provider_caps,
            max_generations=self.max_generations,
            allow_conservative_interrupted_recovery=(
                self.allow_conservative_interrupted_recovery
            ),
        )
        return self.delegate.next_authoring_response(prompt_payload=prompt_payload)

    def next_candidate(
        self,
        *,
        prompt_payload: dict[str, Any],
    ) -> Any:
        _assert_provider_request_budget(
            output_dir=self.output_dir,
            generation_id=self.generation_id,
            current_lane_id=self.lane_id,
            current_prompt_payload=prompt_payload,
            provider_caps=self.provider_caps,
            max_generations=self.max_generations,
            allow_conservative_interrupted_recovery=(
                self.allow_conservative_interrupted_recovery
            ),
        )
        return self.delegate.next_candidate(prompt_payload=prompt_payload)



def _assert_provider_request_budget(
    *,
    output_dir: Path,
    generation_id: str,
    current_lane_id: str,
    current_prompt_payload: Mapping[str, Any],
    provider_caps: Mapping[str, Any],
    max_generations: int = 1,
    allow_conservative_interrupted_recovery: bool = False,
) -> None:
    if not isinstance(allow_conservative_interrupted_recovery, bool):
        raise TypeError("provider interrupted recovery policy must be boolean")
    if current_lane_id not in GENERATION_LANE_IDS:
        raise ValueError("provider budget gate lane is invalid")
    if (
        isinstance(max_generations, bool)
        or not isinstance(max_generations, int)
        or max_generations <= 0
    ):
        raise ValueError("provider budget generation count is invalid")
    generation_prefix, marker, generation_suffix = generation_id.rpartition(
        "-generation-"
    )
    if (
        marker != "-generation-"
        or not generation_prefix
        or not generation_suffix.isdigit()
        or int(generation_suffix) not in range(1, max_generations + 1)
    ):
        raise ValueError("provider budget generation identity is invalid")
    _assert_expected_authoring_directories(
        output_dir,
        generation_prefix=generation_prefix,
        max_generations=max_generations,
    )
    attempt_count = 0
    request_units = 0
    response_units = 0
    total_units = 0
    current_attempt_seen = False
    attempt_counts_by_generation: dict[str, int] = {}
    for generation_index in range(1, max_generations + 1):
        scanned_generation_id = f"{generation_prefix}-generation-{generation_index:03d}"
        attempt_counts_by_generation[scanned_generation_id] = 0
        for lane_id in GENERATION_LANE_IDS:
            round_id = f"{scanned_generation_id}-{lane_id}"
            round_dir = output_dir / "authoring" / scanned_generation_id / lane_id
            if not round_dir.exists():
                if (
                    scanned_generation_id == generation_id
                    and lane_id == current_lane_id
                ):
                    raise ValueError("current provider journal does not exist")
                continue
            if round_dir.is_symlink() or not round_dir.is_dir():
                raise ValueError("provider budget journal directory is invalid")
            attempts = (
                RoundAuthoringJournal(
                    round_dir,
                    round_id=round_id,
                )
                .snapshot()
                .attempts
            )
            attempt_counts_by_generation[scanned_generation_id] += len(attempts)
            for index, attempt in enumerate(attempts):
                attempt_count += 1
                is_current = (
                    scanned_generation_id == generation_id
                    and lane_id == current_lane_id
                    and index == len(attempts) - 1
                )
                status = attempt.get("status")
                if is_current:
                    if current_attempt_seen or status != "interrupted_after_prompt":
                        raise ValueError(
                            "current provider attempt is not its unique prompt reservation"
                        )
                    prompt_ref = PromptPayloadRef.from_mapping(
                        attempt.get("prompt_payload")
                    )
                    if verify_prompt_payload(prompt_ref) != dict(
                        current_prompt_payload
                    ):
                        raise ValueError("current provider prompt reservation mismatch")
                    current_attempt_seen = True
                    continue
                if status == "interrupted_before_prompt":
                    continue
                if status == "interrupted_after_prompt":
                    if not allow_conservative_interrupted_recovery:
                        raise RuntimeError(
                            "prior provider request has no authoritative usage receipt"
                        )
                    request_units += provider_caps["input_units_per_attempt_reserve"]
                    response_units += (
                        provider_caps["output_units_per_attempt_cap"]
                    )
                    total_units += (
                        provider_caps["input_units_per_attempt_reserve"]
                        + provider_caps["output_units_per_attempt_cap"]
                    )
                    continue
                provider_attempt = attempt.get("provider_attempt")
                if status != "completed" or not isinstance(provider_attempt, Mapping):
                    raise ValueError("prior provider attempt state is invalid")
                usage = provider_attempt.get("provider_usage")
                if not isinstance(usage, Mapping) or not all(
                    field_name in usage
                    for field_name in (
                        "request_units",
                        "response_units",
                        "total_units",
                    )
                ):
                    raise RuntimeError(
                        "prior provider request has no authoritative usage receipt"
                    )
                values: list[int] = []
                for field_name in (
                    "request_units",
                    "response_units",
                    "total_units",
                ):
                    value = usage[field_name]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError("prior provider usage counter is invalid")
                    values.append(value)
                if (
                    values[0] > provider_caps["input_units_per_attempt_reserve"]
                    or values[1] > provider_caps["output_units_per_attempt_cap"]
                    or values[2]
                    > provider_caps["input_units_per_attempt_reserve"]
                    + provider_caps["output_units_per_attempt_cap"]
                ):
                    raise RuntimeError("prior provider request exceeded its frozen cap")
                request_units += values[0]
                response_units += values[1]
                total_units += values[2]
    if not current_attempt_seen:
        raise ValueError("current provider prompt reservation was not found")
    provider_attempt_cap = (
        max_generations
        * len(GENERATION_LANE_IDS)
        * provider_caps["attempts_per_lane_per_generation"]
    )
    if attempt_count > provider_attempt_cap:
        raise RuntimeError("provider attempt cap is exhausted")
    per_generation_attempt_cap = (
        len(GENERATION_LANE_IDS) * provider_caps["attempts_per_lane_per_generation"]
    )
    if any(
        count > per_generation_attempt_cap
        for count in attempt_counts_by_generation.values()
    ):
        raise RuntimeError("per-generation provider attempt cap is exhausted")
    current_generation_index = int(generation_suffix)
    if any(
        attempt_counts_by_generation[f"{generation_prefix}-generation-{index:03d}"]
        for index in range(current_generation_index + 1, max_generations + 1)
    ):
        raise ValueError("provider attempts exist ahead of the current generation")
    request_units += provider_caps["input_units_per_attempt_reserve"]
    response_units += provider_caps["output_units_per_attempt_cap"]
    total_units += (
        provider_caps["input_units_per_attempt_reserve"]
        + provider_caps["output_units_per_attempt_cap"]
    )
    current_attempt_count = attempt_counts_by_generation[generation_id]
    remaining_generation_attempts = per_generation_attempt_cap - current_attempt_count
    request_units += (
        remaining_generation_attempts
        * provider_caps["input_units_per_attempt_reserve"]
    )
    response_units += (
        remaining_generation_attempts
        * provider_caps["output_units_per_attempt_cap"]
    )
    total_units += remaining_generation_attempts * (
        provider_caps["input_units_per_attempt_reserve"]
        + provider_caps["output_units_per_attempt_cap"]
    )
    if (
        request_units
        > provider_attempt_cap * provider_caps["input_units_per_attempt_reserve"]
        or response_units
        > provider_attempt_cap * provider_caps["output_units_per_attempt_cap"]
        or total_units
        > provider_attempt_cap
        * (
            provider_caps["input_units_per_attempt_reserve"]
            + provider_caps["output_units_per_attempt_cap"]
        )
    ):
        raise RuntimeError("provider unit cap cannot reserve this request")



def collect_provider_actuals(
    foundation: GenerationSessionFoundation,
    *,
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    provider_caps = foundation.cost_contract.to_dict()["provider"]
    totals = {
        "attempts": 0,
        "request_units": 0,
        "response_units": 0,
        "total_units": 0,
        "reasoning_units": 0,
        "conservative_attempts": 0,
    }
    charges: list[str] = []
    snapshots: dict[tuple[str, str], Any] = {}
    session_contract = getattr(foundation, "session_contract", None)
    max_generations = getattr(session_contract, "max_generations", 1)
    _assert_expected_authoring_directories(
        foundation.config.output_dir,
        generation_prefix=foundation.config.session_id,
        max_generations=max_generations,
    )
    for generation_index in range(1, max_generations + 1):
        generation_id = (
            f"{foundation.config.session_id}-generation-{generation_index:03d}"
        )
        for lane_id in GENERATION_LANE_IDS:
            round_id = f"{generation_id}-{lane_id}"
            round_dir = (
                foundation.config.output_dir / "authoring" / generation_id / lane_id
            )
            if not round_dir.exists():
                continue
            if round_dir.is_symlink() or not round_dir.is_dir():
                raise ValueError("authoring round directory is invalid")
            snapshot = RoundAuthoringJournal(
                round_dir,
                round_id=round_id,
            ).snapshot()
            snapshots[(generation_id, lane_id)] = snapshot
            for attempt in snapshot.attempts:
                totals["attempts"] += 1
                status = attempt["status"]
                if status == "interrupted_before_prompt":
                    continue
                provider_attempt = attempt.get("provider_attempt")
                if status == "interrupted_after_prompt":
                    _charge_full_provider_reserve(
                        totals,
                        charges,
                        label=f"{round_id}:{attempt['sequence']}",
                        provider_caps=provider_caps,
                    )
                    continue
                if status != "completed" or not isinstance(provider_attempt, Mapping):
                    raise ValueError("authoring attempt status is invalid")
                _accumulate_provider_attempt(
                    totals,
                    charges,
                    provider_attempt=provider_attempt,
                    label=f"{round_id}:{attempt['sequence']}",
                    provider_caps=provider_caps,
                )
    ledger_lanes: set[tuple[str, str]] = set()
    for record in ledger:
        generation_id = record.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id:
            raise TypeError("generation ledger identity is invalid")
        trials = record.get("trials")
        artifacts = record.get("private_artifacts")
        if not isinstance(trials, list) or not isinstance(artifacts, list):
            raise TypeError("generation ledger authoring evidence is invalid")
        for trial in trials:
            if not isinstance(trial, Mapping):
                raise TypeError("generation trial is invalid")
            authoring = trial.get("authoring")
            if authoring is None:
                continue
            if not isinstance(authoring, Mapping):
                raise TypeError("generation trial authoring is invalid")
            lane_id = trial.get("lane_id")
            if not isinstance(lane_id, str):
                raise TypeError("generation trial lane id is invalid")
            ledger_lanes.add((generation_id, lane_id))
            prompt = _unique_private_artifact(
                artifacts,
                kind=f"{lane_id}_prompt_payload",
                checksum=authoring.get("prompt_payload_checksum"),
            )
            evidence = _unique_private_artifact(
                artifacts,
                kind=f"{lane_id}_authoring_evidence",
                checksum=authoring.get("authoring_evidence_checksum"),
            )
            prompt_ref = PromptPayloadRef(
                path=prompt["uri"],
                checksum=prompt["checksum"],
                size_bytes=prompt["size_bytes"],
            )
            evidence_payload = verify_authoring_evidence(
                AuthoringEvidenceRef(
                    path=evidence["uri"],
                    checksum=evidence["checksum"],
                    size_bytes=evidence["size_bytes"],
                ),
                expected_round_id=str(trial.get("trial_id")),
                expected_selected_prompt=prompt_ref,
            )
            snapshot = snapshots.get((generation_id, lane_id))
            if (
                snapshot is None
                or list(snapshot.attempts) != evidence_payload["attempts"]
            ):
                raise ValueError(
                    "ledger authoring evidence differs from its live journal"
                )
    if not ledger_lanes.issubset(snapshots):
        raise ValueError("ledger authoring journal is missing")
    return {**totals, "conservative_charges": charges}



def _assert_expected_authoring_directories(
    output_dir: Path,
    *,
    generation_prefix: str,
    max_generations: int,
) -> None:
    authoring_root = Path(output_dir) / "authoring"
    if not authoring_root.exists() and not authoring_root.is_symlink():
        return
    if authoring_root.is_symlink() or not authoring_root.is_dir():
        raise ValueError("authoring root directory is invalid")
    expected_generations = {
        f"{generation_prefix}-generation-{index:03d}"
        for index in range(1, max_generations + 1)
    }
    for generation_path in authoring_root.iterdir():
        if (
            generation_path.name not in expected_generations
            or generation_path.is_symlink()
            or not generation_path.is_dir()
        ):
            raise ValueError("authoring root contains an unknown generation")
        for lane_path in generation_path.iterdir():
            if (
                lane_path.name not in GENERATION_LANE_IDS
                or lane_path.is_symlink()
                or not lane_path.is_dir()
            ):
                raise ValueError("authoring generation contains an unknown lane")



def _accumulate_provider_attempt(
    totals: dict[str, int],
    charges: list[str],
    *,
    provider_attempt: Mapping[str, Any],
    label: str,
    provider_caps: Mapping[str, Any],
) -> None:
    usage = provider_attempt.get("provider_usage")
    if not isinstance(usage, Mapping) or not all(
        field_name in usage
        for field_name in ("request_units", "response_units", "total_units")
    ):
        _charge_full_provider_reserve(totals, charges, label=label, provider_caps=provider_caps)
        return
    for source, target in (
        ("request_units", "request_units"),
        ("response_units", "response_units"),
        ("total_units", "total_units"),
    ):
        value = usage[source]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("provider usage counter is invalid")
        totals[target] += value
    reasoning = usage.get("reasoning_units", 0)
    if isinstance(reasoning, bool) or not isinstance(reasoning, int) or reasoning < 0:
        raise ValueError("provider reasoning usage is invalid")
    totals["reasoning_units"] += reasoning



def _charge_full_provider_reserve(
    totals: dict[str, int],
    charges: list[str],
    *,
    label: str,
    provider_caps: Mapping[str, Any],
) -> None:
    totals["request_units"] += provider_caps["input_units_per_attempt_reserve"]
    totals["response_units"] += provider_caps["output_units_per_attempt_cap"]
    totals["total_units"] += (
        provider_caps["input_units_per_attempt_reserve"]
        + provider_caps["output_units_per_attempt_cap"]
    )
    totals["conservative_attempts"] += 1
    charges.append("provider-full-reserve:" + label)



def _unique_private_artifact(
    artifacts: list[Any],
    *,
    kind: str,
    checksum: Any,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == kind
        and item.get("checksum") == checksum
    ]
    if len(matches) != 1:
        raise ValueError(f"private artifact is not unique: {kind}")
    item = matches[0]
    if not isinstance(item.get("uri"), str) or not isinstance(
        item.get("size_bytes"), int
    ):
        raise TypeError("private artifact reference is invalid")
    return item

