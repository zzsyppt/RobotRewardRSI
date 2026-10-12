"""Append-only generation cost receipts and lifecycle cost replay.

Current contracts declare operation reserves. The legacy usage shape is retained
only to read and reproduce previously frozen session-level fallback accounting.
"""
from __future__ import annotations
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from true_rl_agent.agent_core.generation_cost import GenerationCostContract
from true_rl_agent.agent_core.generation_lifecycle import GenerationLifecycleAudit
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object

_EMPTY_COST_EVIDENCE_HASH = canonical_json_hash([])

@dataclass(frozen=True)
class LegacyGenerationActualCostUsage:
    """Cumulative execution measurements used by one cost-chain edge."""

    provider_attempts: int = 0
    provider_request_units: int = 0
    provider_response_units: int = 0
    provider_total_units: int = 0
    provider_reasoning_units: int = 0
    provider_conservative_attempts: int = 0
    job_attempts: int = 0
    train_jobs: int = 0
    train_steps: int = 0
    eval_seed_units: int = 0
    simulated_control_steps: int = 0
    resource_lease_wall_duration_ns: int = 0
    unmeasured_bound_attempts: int = 0
    observed_output_bytes: int = 0
    observed_output_files: int = 0
    job_cost_evidence_hash: str = _EMPTY_COST_EVIDENCE_HASH

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name == "job_cost_evidence_hash":
                continue
            require_nonnegative_cost_integer(getattr(self, field_name), field_name)
        if re.fullmatch(r"[0-9a-f]{64}", self.job_cost_evidence_hash) is None:
            raise ValueError("job_cost_evidence_hash must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }



@dataclass(frozen=True)
class GenerationActualCostUsage(LegacyGenerationActualCostUsage):
    """Usage with operation-specific fallback for unmeasured GPU attempts."""

    conservatively_charged_gpu_capacity_ns: int = 0



@dataclass(frozen=True)
class GenerationActualCostReceiptRef:
    path: Path
    payload_hash: str
    payload: Mapping[str, Any]



def replay_generation_combined_cost_chain(
    *,
    contract: GenerationCostContract,
    actual_cost_receipts: Sequence[GenerationActualCostReceiptRef],
    lifecycle: GenerationLifecycleAudit,
) -> str:
    """Replay bounded-session actual-cost and lifecycle-recovery edges as one chain."""

    if not contract.uses_operation_reserves:
        raise TypeError("bounded generation combined cost replay requires the v2 cost contract")
    outgoing: dict[str, tuple[str, str]] = {}

    def add_edge(before: str, after: str, label: str) -> None:
        if before == after:
            if label != "recovery":
                raise ValueError("bounded generation actual-cost chain contains a self edge")
            return
        if before in outgoing:
            raise ValueError("bounded generation combined cost chain forks")
        outgoing[before] = (after, label)

    for receipt in actual_cost_receipts:
        add_edge(
            receipt.payload["previous_cost_chain_head_hash"],
            receipt.payload_hash,
            "actual-cost",
        )
    for _segment_start_head, before, after in lifecycle.recovery_cost_edges:
        add_edge(before, after, "recovery")

    head = contract.contract_hash
    positions = {head: 0}
    consumed_edges = 0
    while head in outgoing:
        next_head, _label = outgoing[head]
        if next_head in positions:
            raise ValueError("bounded generation combined cost chain contains a cycle")
        consumed_edges += 1
        positions[next_head] = consumed_edges
        head = next_head
    if consumed_edges != len(outgoing):
        raise ValueError("bounded generation combined cost chain has an orphan edge")

    for segment_start_head, before, after in lifecycle.recovery_cost_edges:
        if (
            segment_start_head not in positions
            or before not in positions
            or after not in positions
            or positions[segment_start_head] > positions[before]
            or positions[before] > positions[after]
        ):
            raise ValueError("bounded generation recovery cost edge is out of lifecycle order")

    if lifecycle.segment_count:
        lifecycle_head = lifecycle.cost_chain_head_hash
        if lifecycle.active_segment_id is None:
            if lifecycle_head != head:
                raise ValueError("bounded generation closed lifecycle cost head diverged")
        elif lifecycle_head not in positions:
            raise ValueError("bounded generation active lifecycle cost head is not in the chain")
    return head



def replay_generation_final_cost_chain(
    *,
    contract: GenerationCostContract,
    actual_cost_receipts: Sequence[GenerationActualCostReceiptRef],
    lifecycle: GenerationLifecycleAudit,
    require_finalization_settlement: bool,
) -> str:
    """Replay the terminal generation head plus the sole finalization edge."""

    if contract.final_confirmation_seed_units <= 0:
        raise TypeError("formal cost replay requires the v3 cost contract")
    if not lifecycle.terminal:
        raise ValueError("formal cost replay requires a terminal lifecycle")
    generation_head = lifecycle.cost_chain_head_hash
    matching = [
        index
        for index, receipt in enumerate(actual_cost_receipts)
        if receipt.payload_hash == generation_head
    ]
    if len(matching) != 1:
        raise ValueError("formal cost chain lacks its generation head")
    split = matching[0] + 1
    generation_chain = actual_cost_receipts[:split]
    finalization_chain = actual_cost_receipts[split:]
    replay_generation_combined_cost_chain(
        contract=contract,
        actual_cost_receipts=generation_chain,
        lifecycle=lifecycle,
    )
    if len(finalization_chain) > 1:
        raise ValueError("formal cost chain has excess finalization edges")
    head = generation_head
    if finalization_chain:
        final = finalization_chain[0]
        if (
            final.payload.get("event_id") != "final-confirmation-settlement"
            or final.payload.get("previous_cost_chain_head_hash") != generation_head
        ):
            raise ValueError("formal finalization cost edge is invalid")
        head = final.payload_hash
    if require_finalization_settlement and not finalization_chain:
        raise ValueError("formal cost chain lacks finalization settlement")
    return head



class GenerationActualCostJournal:
    """Small append-only actual-cost chain for the session driver."""

    def __init__(
        self,
        output_dir: Path,
        *,
        contract: GenerationCostContract,
    ) -> None:
        if not isinstance(output_dir, Path):
            raise TypeError("actual-cost output_dir must be a Path")
        self.output_dir = output_dir.resolve(strict=False)
        self.root = self.output_dir / "generation_actual_cost"
        self.contract = contract

    def publish_reservation(self) -> GenerationActualCostReceiptRef:
        existing = self.receipt_for_event("session-reservation")
        if existing is not None:
            return existing
        empty_usage: LegacyGenerationActualCostUsage = (
            GenerationActualCostUsage()
            if self.contract.uses_operation_reserves
            else LegacyGenerationActualCostUsage()
        )
        return self.publish_snapshot(
            event_id="session-reservation",
            previous_cost_chain_head_hash=self.contract.contract_hash,
            usage=empty_usage,
            conservative_charges=(),
            finalization_output_reserve_bytes=0,
        )

    def publish_snapshot(
        self,
        *,
        event_id: str,
        previous_cost_chain_head_hash: str,
        usage: LegacyGenerationActualCostUsage,
        conservative_charges: tuple[str, ...] = (),
        finalization_output_reserve_bytes: int = (
            1024 * 1024
        ),
    ) -> GenerationActualCostReceiptRef:
        event_id = require_cost_event_id(event_id)
        if (
            self.contract.uses_operation_reserves
            and type(usage) is LegacyGenerationActualCostUsage
        ):
            usage = GenerationActualCostUsage(**usage.to_dict())
        expected_usage_type = (
            GenerationActualCostUsage
            if self.contract.uses_operation_reserves
            else LegacyGenerationActualCostUsage
        )
        if type(usage) is not expected_usage_type:
            raise TypeError("actual-cost usage is invalid")
        if not isinstance(previous_cost_chain_head_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", previous_cost_chain_head_hash
        ):
            raise ValueError("actual-cost previous chain head is invalid")
        require_nonnegative_cost_integer(
            finalization_output_reserve_bytes,
            "finalization_output_reserve_bytes",
        )
        if not isinstance(conservative_charges, tuple) or any(
            not isinstance(item, str) or not item or len(item) > 128
            for item in conservative_charges
        ):
            raise ValueError("actual-cost conservative charges are invalid")

        loaded = self.load_chain()
        matching_indexes = [
            index
            for index, item in enumerate(loaded)
            if item.payload["event_id"] == event_id
        ]
        if len(matching_indexes) > 1:
            raise ValueError("actual-cost event is duplicated")
        if matching_indexes and matching_indexes[0] != len(loaded) - 1:
            raise ValueError("actual-cost event was already superseded")
        previous_task_hash = (
            loaded[-2].payload_hash
            if matching_indexes and len(loaded) > 1
            else (
                None
                if matching_indexes
                else (loaded[-1].payload_hash if loaded else None)
            )
        )
        reservation, actual, checks = project_actual_generation_cost(
            self.contract,
            usage=usage,
            finalization_output_reserve_bytes=finalization_output_reserve_bytes,
        )
        payload = {
            "schema_version": self.contract.actual_cost_schema_version,
            "session_cost_contract_hash": self.contract.contract_hash,
            "event_id": event_id,
            "previous_cost_chain_head_hash": previous_cost_chain_head_hash,
            "previous_task_cost_receipt_hash": previous_task_hash,
            "reservation": reservation,
            "actual": actual,
            "conservative_charges": list(conservative_charges),
            "cap_checks": checks,
            "acceptance_authority": False,
        }
        if matching_indexes:
            existing = loaded[-1]
            if dict(existing.payload) != payload:
                raise ValueError(
                    "existing actual-cost event differs from recomputation"
                )
            return existing
        sequence = len(loaded)
        path = self.root / f"{sequence:04d}-{event_id}.json"
        encoded = canonical_json_bytes(payload, trailing_newline=True)
        self.root.mkdir(parents=True, exist_ok=True)
        create_once_bytes(path, encoded)
        return GenerationActualCostReceiptRef(
            path=path,
            payload_hash=canonical_json_hash(payload),
            payload=payload,
        )

    def receipt_for_event(
        self,
        event_id: str,
    ) -> GenerationActualCostReceiptRef | None:
        event_id = require_cost_event_id(event_id)
        return self.load_snapshot().get(event_id)

    def load_snapshot(self) -> dict[str, GenerationActualCostReceiptRef]:
        """Freshly validate the chain, then expose its events for one consumer.

        No cache lives on the journal. A later write or independent replay
        calls this method again and revalidates the on-disk evidence.
        """
        return {item.payload["event_id"]: item for item in self.load_chain()}

    def load_chain(self) -> list[GenerationActualCostReceiptRef]:
        if not self.root.exists():
            return []
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("actual-cost receipt root is invalid")
        result: list[GenerationActualCostReceiptRef] = []
        seen_events: set[str] = set()
        previous_task_hash: str | None = None
        for expected_index, path in enumerate(sorted(self.root.glob("*.json"))):
            match = re.fullmatch(r"([0-9]{4})-([a-z0-9-]+)\.json", path.name)
            if match is None or int(match.group(1)) != expected_index:
                raise ValueError("actual-cost receipt sequence is invalid")
            payload = load_strict_json_object(
                path,
                context="actual-cost receipt",
            )
            _validate_actual_cost_receipt(
                payload,
                contract=self.contract,
                expected_event_id=match.group(2),
                previous_task_hash=previous_task_hash,
            )
            if payload["event_id"] in seen_events:
                raise ValueError("actual-cost event is duplicated")
            seen_events.add(payload["event_id"])
            if expected_index == 0:
                if (
                    payload["event_id"] != "session-reservation"
                    or payload["previous_cost_chain_head_hash"]
                    != self.contract.contract_hash
                    or payload["actual"]
                    != project_actual_generation_cost(
                        self.contract,
                        usage=(
                            GenerationActualCostUsage()
                            if self.contract.uses_operation_reserves
                            else LegacyGenerationActualCostUsage()
                        ),
                        finalization_output_reserve_bytes=0,
                    )[1]
                ):
                    raise ValueError("actual-cost reservation receipt mismatch")
            elif payload["event_id"] == "session-reservation":
                raise ValueError("actual-cost reservation receipt is duplicated")
            encoded = canonical_json_bytes(payload, trailing_newline=True)
            if path.read_bytes() != encoded:
                raise ValueError("actual-cost receipt is not canonical JSON")
            ref = GenerationActualCostReceiptRef(
                path=path,
                payload_hash=canonical_json_hash(payload),
                payload=payload,
            )
            result.append(ref)
            previous_task_hash = ref.payload_hash
        return result



def _validate_actual_cost_receipt(
    value: Mapping[str, Any],
    *,
    contract: GenerationCostContract,
    expected_event_id: str,
    previous_task_hash: str | None,
) -> None:
    expected_fields = {
        "schema_version",
        "session_cost_contract_hash",
        "event_id",
        "previous_cost_chain_head_hash",
        "previous_task_cost_receipt_hash",
        "reservation",
        "actual",
        "conservative_charges",
        "cap_checks",
        "acceptance_authority",
    }
    if set(value) != expected_fields:
        raise ValueError("actual-cost receipt fields mismatch")
    if (
        value.get("schema_version") != contract.actual_cost_schema_version
        or value.get("session_cost_contract_hash") != contract.contract_hash
        or value.get("event_id") != expected_event_id
        or value.get("previous_task_cost_receipt_hash") != previous_task_hash
        or value.get("acceptance_authority") is not False
    ):
        raise ValueError("actual-cost receipt identity mismatch")
    previous = value.get("previous_cost_chain_head_hash")
    if not isinstance(previous, str) or re.fullmatch(r"[0-9a-f]{64}", previous) is None:
        raise ValueError("actual-cost prior chain head is invalid")
    reservation = value.get("reservation")
    actual = value.get("actual")
    checks = value.get("cap_checks")
    conservative = value.get("conservative_charges")
    if (
        not isinstance(reservation, Mapping)
        or not isinstance(actual, Mapping)
        or not isinstance(checks, Mapping)
        or not isinstance(conservative, list)
        or any(not isinstance(item, str) or not item for item in conservative)
    ):
        raise ValueError("actual-cost receipt sections are invalid")
    # Re-project the payload through the publishing logic's immutable fields.
    usage_type = (
        GenerationActualCostUsage
        if contract.uses_operation_reserves
        else LegacyGenerationActualCostUsage
    )
    usage_fields = set(usage_type.__dataclass_fields__)
    extra_actual = {
        "allocated_gpu_capacity_hours_charged",
        "finalization_output_reserve_bytes",
        "accounted_output_bytes",
    }
    if set(actual) != usage_fields | extra_actual:
        raise ValueError("actual-cost usage fields mismatch")
    usage = usage_type(
        **{field_name: actual[field_name] for field_name in usage_fields}
    )
    reserve = actual.get("finalization_output_reserve_bytes")
    require_nonnegative_cost_integer(reserve, "finalization_output_reserve_bytes")
    expected_reservation, expected_actual, expected_checks = project_actual_generation_cost(
        contract,
        usage=usage,
        finalization_output_reserve_bytes=reserve,
    )
    if (
        dict(reservation) != expected_reservation
        or dict(actual) != expected_actual
        or dict(checks) != expected_checks
    ):
        raise ValueError("actual-cost projection mismatch")



def project_actual_generation_cost(
    contract: GenerationCostContract,
    *,
    usage: LegacyGenerationActualCostUsage,
    finalization_output_reserve_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    """Replay the persisted v1/v2 usage arithmetic selected by the frozen contract."""
    provider_caps = contract.to_dict()["provider"]
    if contract.uses_operation_reserves:
        if type(usage) is not GenerationActualCostUsage:
            raise TypeError("bounded generation actual-cost usage requires the v2 projection")
        charged_gpu_hours = (
            usage.resource_lease_wall_duration_ns
            + usage.conservatively_charged_gpu_capacity_ns
        ) / 3_600_000_000_000
        resource_accounting_complete = (
            usage.unmeasured_bound_attempts == 0
            and usage.conservatively_charged_gpu_capacity_ns == 0
        ) or (
            usage.unmeasured_bound_attempts > 0
            and usage.conservatively_charged_gpu_capacity_ns > 0
        )
    else:
        if type(usage) is not LegacyGenerationActualCostUsage:
            raise TypeError("P6-B actual-cost usage requires the v1 projection")
        charged_gpu_hours = (
            usage.resource_lease_wall_duration_ns / 3_600_000_000_000
            + usage.unmeasured_bound_attempts
            * float(contract.max_allocated_active_gpu_hours)
        )
        resource_accounting_complete = usage.unmeasured_bound_attempts == 0
    accounted_output_bytes = (
        usage.observed_output_bytes + finalization_output_reserve_bytes
    )
    reservation = {
        "provider_attempt_cap": provider_caps["attempt_cap"],
        "provider_input_unit_cap": provider_caps["input_unit_cap"],
        "provider_output_unit_cap": provider_caps["output_unit_cap"],
        "provider_total_unit_cap": provider_caps["total_unit_cap"],
        "train_job_cap": contract.physical_train_job_cap,
        "train_step_cap": contract.physical_train_step_cap,
        "eval_seed_unit_cap": contract.physical_eval_seed_unit_cap,
        "allocated_gpu_capacity_hours_cap": float(
            contract.max_allocated_active_gpu_hours
        ),
        "simulated_control_step_cap": contract.max_simulated_control_steps,
        "output_byte_cap": contract.max_output_bytes,
    }
    actual = {
        **usage.to_dict(),
        "allocated_gpu_capacity_hours_charged": charged_gpu_hours,
        "finalization_output_reserve_bytes": (finalization_output_reserve_bytes),
        "accounted_output_bytes": accounted_output_bytes,
    }
    checks = {
        "provider_attempts_within_cap": (
            usage.provider_attempts <= provider_caps["attempt_cap"]
        ),
        "provider_request_units_within_cap": (
            usage.provider_request_units <= provider_caps["input_unit_cap"]
        ),
        "provider_response_units_within_cap": (
            usage.provider_response_units <= provider_caps["output_unit_cap"]
        ),
        "provider_total_units_within_cap": (
            usage.provider_total_units <= provider_caps["total_unit_cap"]
        ),
        "train_jobs_within_cap": (usage.train_jobs <= contract.physical_train_job_cap),
        "train_steps_within_cap": (
            usage.train_steps <= contract.physical_train_step_cap
        ),
        "eval_seed_units_within_cap": (
            usage.eval_seed_units <= contract.physical_eval_seed_unit_cap
        ),
        "allocated_gpu_capacity_hours_within_cap": (
            charged_gpu_hours <= float(contract.max_allocated_active_gpu_hours)
        ),
        "simulated_control_steps_within_cap": (
            usage.simulated_control_steps <= contract.max_simulated_control_steps
        ),
        "output_bytes_within_cap": (
            accounted_output_bytes <= contract.max_output_bytes
        ),
    }
    if contract.uses_operation_reserves:
        checks["resource_lease_accounting_complete"] = resource_accounting_complete
    else:
        checks["resource_lease_measurement_complete"] = resource_accounting_complete
    checks["all_passed"] = all(checks.values())
    return reservation, actual, checks



def require_cost_event_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 96
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is None
    ):
        raise ValueError("actual-cost event_id is invalid")
    return value



def require_nonnegative_cost_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def cost_receipt_by_hash(
    journal: GenerationActualCostJournal,
    payload_hash: str,
) -> GenerationActualCostReceiptRef:
    matches = [item for item in journal.load_chain() if item.payload_hash == payload_hash]
    if len(matches) != 1:
        raise ValueError("P6-D cost-chain receipt is not uniquely identified")
    return matches[0]



def generation_cost_receipt_projection(
    receipt: GenerationActualCostReceiptRef,
) -> dict[str, Any]:
    return {
        "path": str(receipt.path.resolve()),
        "payload_hash": receipt.payload_hash,
        "event_id": receipt.payload["event_id"],
    }



def assert_actual_cost_receipt_matches_usage(
    receipt: GenerationActualCostReceiptRef,
    *,
    usage: LegacyGenerationActualCostUsage,
    conservative_charges: tuple[str, ...],
    context: str,
) -> None:
    actual = receipt.payload["actual"]
    for field_name, current in usage.to_dict().items():
        if field_name in {"observed_output_bytes", "observed_output_files"}:
            continue
        if actual.get(field_name) != current:
            raise ValueError(f"{context} actual cost drifted: {field_name}")
    if usage.observed_output_bytes > actual["accounted_output_bytes"]:
        raise ValueError(f"{context} exceeded its settled output reserve")
    if tuple(receipt.payload["conservative_charges"]) != conservative_charges:
        raise ValueError(f"{context} conservative charges drifted")

