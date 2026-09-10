"""Read-only source/lifecycle inspection retained for frozen Stair reports.

No readiness collector, resource sampling or launch authority remains here.
"""

from __future__ import annotations
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_SESSION_CONTRACT_FILENAME,
    GenerationSessionContract,
    load_generation_session_contract,
)
from true_rl_agent.agent_core.generation_lifecycle import (
    GENERATION_CONTINUATION_POLICY_FILENAME,
    GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION,
    GenerationLifecycleIdentity,
    audit_generation_lifecycle,
    load_generation_continuation_policy,
)
from true_rl_agent.agent_core.ledger import load_ledger, primary_ledger_epoch_hash
from true_rl_agent.agent_core.session_genesis import (
    SESSION_GENESIS_FILENAME,
    SessionGenesisArtifact,
)
from true_rl_agent.composition.cell_manifest import (
    RESOLVED_CELL_MANIFEST_FILENAME,
    load_resolved_cell_manifest,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionConfig
from true_rl_agent.tasks.stair_climbing.config import STAIR_TASK_ID, StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    STAIR_FINAL_CONFIRMATION_DIRNAME,
    STAIR_FINAL_CONFIRMATION_PLAN_FILENAME,
    STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
    STAIR_FINAL_CONFIRMATION_RESERVATION_FILENAME,
    STAIR_FINAL_CONFIRMATION_RESULT_FILENAME,
    STAIR_GENERATION_REPORT_FILENAME,
    bind_stair_generation_final_config,
    load_stair_generation_final_confirmation_plan,
    load_stair_generation_final_confirmation_reservation,
    load_stair_generation_final_confirmation_result,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    FORMAL_GENERATION_SESSION_MODE,
    STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES,
    STAIR_GENERATION_SESSION_COST_FILENAME,
    STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES,
    StairGenerationActualCostJournal,
    StairGenerationSessionCostContractV3,
    load_stair_generation_session_cost_contract,
    replay_stair_generation_combined_cost_chain,
    replay_stair_generation_formal_cost_chain,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    objective_family_mask_from_public_result,
)

FORMAL_MAX_OUTPUT_BYTES = (
    STAIR_GENERATION_FORMAL_SUCCESSFUL_OUTPUT_BYTES
    + STAIR_GENERATION_FORMAL_RETRY_OUTPUT_BYTES
)

MAX_READINESS_RECEIPT_BYTES = 8 * 1024 * 1024

STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT = "formal_fresh_absent"

STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME = "formal_nonterminal_resume"

STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION = "formal_terminal_finalization"

STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY = "formal_terminal_audit_only"

STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME = "generation_source_checkout.json"

STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION = (
    "true_rl_agent.stair.generation_source_checkout.v1"
)

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

def _file_evidence(path: Path) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    if not target.is_file():
        raise ValueError(
            "Stair launch readiness evidence source must be a regular file"
        )
    digest = hashlib.sha256()
    size_bytes = 0
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size_bytes += len(block)
    return {
        "path": str(target),
        "sha256": "sha256:" + digest.hexdigest(),
        "size_bytes": size_bytes,
    }

def _resolved_formal_task_config(
    *,
    output_dir: Path,
    project_root: Path,
) -> StairClimbingTaskConfig:
    manifest = load_resolved_cell_manifest(
        (output_dir / RESOLVED_CELL_MANIFEST_FILENAME).resolve(),
        expected_task_id=STAIR_TASK_ID,
    )
    defaults = StairClimbingCompositionConfig.defaults(project_root=project_root)
    values = defaults.to_task_defaults()
    values["cell_variant_id"] = manifest.cell_variant_id
    composition = StairClimbingCompositionConfig.from_values(
        values
    ).bind_selected_cell_contract(project_root=project_root)
    return composition.to_task_config(
        project_root=project_root,
        resolved_cell_task_protocol_hash=(
            manifest.resolved_cell_task_protocol_hash
        ),
    )

def inspect_historical_formal_launch_subject(
    *,
    formal_output_dir: Path,
    source_root: Path,
    session_id: str,
    expected_source_commit: str,
    subject_mode: str,
) -> dict[str, Any]:
    output_dir = Path(formal_output_dir).resolve(strict=False)
    project_root = Path(source_root).resolve(strict=True)
    if subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT:
        _assert_path_absent(output_dir, "formal output")
        return {
            "subject_mode": STAIR_FORMAL_LAUNCH_SUBJECT_FRESH_ABSENT,
            "formal_output_absent": True,
            "current_accounted_output_bytes": 0,
        }
    if subject_mode not in (
        STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME,
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION,
        STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY,
    ):
        raise ValueError("Stair launch readiness v3 subject_mode is invalid")
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(
            "formal resume readiness requires a real formal output directory"
        )

    source_path = output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME
    source_evidence, source_payload = _canonical_object_file(
        source_path,
        label="Stair generation source checkout identity",
        max_bytes=4096,
    )
    source_mapping = _exact_mapping(
        source_payload,
        {"schema_version", "source_commit", "source_checkout_hash"},
        "Stair generation source checkout identity",
    )
    source_body = {
        "schema_version": STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION,
        "source_commit": source_mapping["source_commit"],
    }
    if (
        source_mapping["schema_version"]
        != STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION
        or _COMMIT_PATTERN.fullmatch(str(source_mapping["source_commit"])) is None
        or (
            subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME
            and source_mapping["source_commit"] != expected_source_commit
        )
        or source_mapping["source_checkout_hash"]
        != canonical_json_hash(source_body)
    ):
        raise ValueError("formal resume source checkout identity mismatch")

    contract_path = output_dir / GENERATION_SESSION_CONTRACT_FILENAME
    contract_evidence, contract_payload = _canonical_object_file(
        contract_path,
        label="generation session contract",
        max_bytes=MAX_READINESS_RECEIPT_BYTES,
    )
    contract = GenerationSessionContract.from_mapping(contract_payload)
    if (
        contract.session_id != session_id
        or contract.session_mode != FORMAL_GENERATION_SESSION_MODE
        or contract.max_generations != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        or contract.task_id != STAIR_TASK_ID
        or contract.backend_id != "isaac_lab_rsl_rl"
    ):
        raise ValueError("formal resume generation contract identity mismatch")

    policy = load_generation_continuation_policy(output_dir)
    if (
        policy.schema_version != GENERATION_CONTINUATION_POLICY_V2_SCHEMA_VERSION
        or policy.max_segments != 7
        or policy.operator_pause_generation_boundaries
        != STAIR_FORMAL_GENERATION_PAUSE_BOUNDARIES
        or policy.policy_hash
        != contract.bound_contract_hashes["continuation_policy"]
    ):
        raise ValueError("formal resume continuation policy mismatch")
    policy_path = output_dir / GENERATION_CONTINUATION_POLICY_FILENAME
    policy_evidence = _authoritative_file_evidence(policy_path)

    task_config = _resolved_formal_task_config(
        output_dir=output_dir,
        project_root=project_root,
    )
    plan_path = (
        output_dir
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / STAIR_FINAL_CONFIRMATION_PLAN_FILENAME
    )
    plan = load_stair_generation_final_confirmation_plan(
        plan_path,
        expected_session_id=session_id,
        expected_config=task_config,
    )
    if (
        plan.plan_hash is None
        or plan.plan_hash
        != contract.bound_contract_hashes["final_confirmation_plan"]
    ):
        raise ValueError("formal resume final confirmation plan binding mismatch")
    plan_evidence = _authoritative_file_evidence(plan_path)

    cost_contract_path = output_dir / STAIR_GENERATION_SESSION_COST_FILENAME
    cost_contract = load_stair_generation_session_cost_contract(cost_contract_path)
    if (
        not isinstance(cost_contract, StairGenerationSessionCostContractV3)
        or cost_contract.contract_hash
        != contract.bound_contract_hashes["session_cost"]
        or cost_contract.max_output_bytes != FORMAL_MAX_OUTPUT_BYTES
    ):
        raise ValueError("formal resume cost contract binding mismatch")
    cost_contract_evidence = _authoritative_file_evidence(cost_contract_path)
    cost_chain = StairGenerationActualCostJournal(
        output_dir,
        contract=cost_contract,
    ).load_chain()
    cost_receipt_files = [
        {
            **_authoritative_file_evidence(item.path),
            "payload_hash": item.payload_hash,
        }
        for item in cost_chain
    ]
    current_accounted_output_bytes = 0
    if cost_chain:
        actual = cost_chain[-1].payload.get("actual")
        if not isinstance(actual, Mapping):
            raise ValueError("formal resume cost head has no actual usage")
        current_accounted_output_bytes = _nonnegative_int(
            actual.get("accounted_output_bytes"),
            "formal resume current accounted output bytes",
        )
        checks = cost_chain[-1].payload.get("cap_checks")
        if not isinstance(checks, Mapping) or checks.get("all_passed") is not True:
            raise ValueError("formal resume cost head exhausted a frozen cap")
    if current_accounted_output_bytes > FORMAL_MAX_OUTPUT_BYTES:
        raise ValueError("formal resume accounted output exceeds its frozen cap")

    genesis_path = output_dir / SESSION_GENESIS_FILENAME
    genesis_file = _optional_file_evidence(genesis_path)
    genesis: SessionGenesisArtifact | None = None
    if genesis_file["exists"] is True:
        _, genesis_payload = _canonical_object_file(
            genesis_path,
            label="session Genesis",
            max_bytes=MAX_READINESS_RECEIPT_BYTES,
        )
        genesis = SessionGenesisArtifact.from_mapping(genesis_payload)
        if genesis.session_id != session_id or genesis.session_genesis_hash is None:
            raise ValueError("formal resume session Genesis identity mismatch")
        contract_ref = genesis.generation_session_contract_ref
        if contract_ref is None:
            raise ValueError("formal resume session Genesis has no generation contract")
        if load_generation_session_contract(output_dir, contract_ref) != contract:
            raise ValueError("formal resume Genesis generation contract differs")

    ledger_path = output_dir / "ledger.jsonl"
    ledger_records = load_ledger(ledger_path)
    ledger_evidence = _optional_file_evidence(ledger_path)
    ledger_height = len(ledger_records)
    if ledger_height and ledger_evidence["exists"] is not True:
        raise ValueError("formal resume non-empty ledger file is absent")

    if genesis is None:
        if subject_mode != STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME:
            raise ValueError("formal terminal readiness requires Session Genesis")
        if ledger_evidence["exists"] is True or ledger_height:
            raise ValueError("pre-Genesis formal resume cannot have a ledger")
        lifecycle_files = _lifecycle_receipt_file_evidence(output_dir)
        if lifecycle_files:
            raise ValueError("pre-Genesis formal resume cannot have lifecycle receipts")
        lifecycle_mapping = {
            "initialized": False,
            "head_hash": None,
            "active": False,
            "active_segment_id": None,
            "segment_count": 0,
            "closed_generation_count": 0,
            "terminal": False,
            "continuation_eligible": True,
            "cost_chain_head_hash": None,
            "receipt_files": [],
        }
    else:
        identity = GenerationLifecycleIdentity(
            session_id=session_id,
            session_genesis_hash=genesis.session_genesis_hash,
            generation_session_contract_hash=contract.contract_hash,
            final_confirmation_plan_hash=plan.plan_hash,
            ledger_epoch_hash=primary_ledger_epoch_hash(
                genesis.session_genesis_hash
            ),
            output_root=str(output_dir),
        )
        lifecycle = audit_generation_lifecycle(
            output_dir=output_dir,
            identity=identity,
            policy=policy,
            require_closed=False,
            require_recovery_attempt_results=False,
        )
        if not cost_chain:
            raise ValueError("post-Genesis formal resume requires an actual-cost chain")
        if lifecycle.terminal:
            replay_stair_generation_formal_cost_chain(
                contract=cost_contract,
                actual_cost_receipts=cost_chain,
                lifecycle=lifecycle,
                require_finalization_settlement=(
                    subject_mode
                    == STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_AUDIT_ONLY
                ),
            )
        else:
            replay_stair_generation_combined_cost_chain(
                contract=cost_contract,
                actual_cost_receipts=cost_chain,
                lifecycle=lifecycle,
            )
        lifecycle_start_height = lifecycle.closed_generation_count
        if lifecycle.active_segment_id is None:
            if ledger_height != lifecycle_start_height:
                raise ValueError(
                    "formal resume ledger height and lifecycle height differ"
                )
        elif not (
            lifecycle_start_height
            <= ledger_height
            <= STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        ):
            raise ValueError(
                "active formal resume ledger height is outside its lifecycle budget"
            )
        lifecycle_files = _lifecycle_receipt_file_evidence(output_dir)
        if [item["payload_hash"] for item in lifecycle_files] != sorted(
            lifecycle.receipt_hashes
        ):
            raise ValueError("formal resume lifecycle receipt file set mismatch")
        lifecycle_mapping = {
            "initialized": True,
            "head_hash": lifecycle.lifecycle_head_hash,
            "active": lifecycle.active_segment_id is not None,
            "active_segment_id": lifecycle.active_segment_id,
            "segment_count": lifecycle.segment_count,
            "closed_generation_count": lifecycle.closed_generation_count,
            "terminal": lifecycle.terminal,
            "continuation_eligible": lifecycle.continuation_eligible,
            "cost_chain_head_hash": lifecycle.cost_chain_head_hash,
            "receipt_files": lifecycle_files,
        }

    reservation_path = (
        output_dir
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / STAIR_FINAL_CONFIRMATION_RESERVATION_FILENAME
    )
    reservation_file = _optional_file_evidence(reservation_path)
    reservation = None
    if reservation_file["exists"] is True:
        reservation = load_stair_generation_final_confirmation_reservation(
            reservation_path,
            expected_plan=plan,
        )
    result_path = (
        output_dir
        / STAIR_FINAL_CONFIRMATION_DIRNAME
        / STAIR_FINAL_CONFIRMATION_RESULT_FILENAME
    )
    result_file = _optional_file_evidence(result_path)
    result = None
    if result_file["exists"] is True:
        if reservation is None:
            raise ValueError("formal final result exists without its reservation")
        objective_family_mask = None
        if task_config.is_unified_protocol:
            if genesis is None:
                raise ValueError(
                    "unified formal final result exists without Session Genesis"
                )
            objective_family_mask = objective_family_mask_from_public_result(
                suite=task_config.ordered_evaluation_suite,
                public_result=getattr(genesis.anchor, "public_result", None),
            )
        result = load_stair_generation_final_confirmation_result(
            result_path,
            plan=plan,
            reservation=reservation,
            config=bind_stair_generation_final_config(
                task_config, plan=plan, plan_path=plan_path,
            ),
            objective_family_mask=objective_family_mask,
        )
    report_path = output_dir / STAIR_GENERATION_REPORT_FILENAME
    report_file = _optional_file_evidence(report_path)

    terminal = bool(lifecycle_mapping["terminal"])
    continuation_eligible = bool(lifecycle_mapping["continuation_eligible"])
    active = bool(lifecycle_mapping["active"])
    if subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_NONTERMINAL_RESUME:
        if terminal or (not active and not continuation_eligible):
            raise ValueError("formal nonterminal launch subject is not continuable")
        if (
            reservation is not None
            or result is not None
            or report_file["exists"] is True
        ):
            raise ValueError("formal nonterminal launch has terminal artifacts")
    else:
        if (
            not lifecycle_mapping["initialized"]
            or not terminal
            or active
            or continuation_eligible
            or lifecycle_mapping["closed_generation_count"]
            != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
            or ledger_height != STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS
        ):
            raise ValueError("formal terminal launch subject is not terminal")
        if subject_mode == STAIR_FORMAL_LAUNCH_SUBJECT_TERMINAL_FINALIZATION:
            if report_file["exists"] is True:
                raise ValueError(
                    "terminal finalization readiness requires the report to be absent"
                )
        elif (
            report_file["exists"] is not True
            or reservation is None
            or result is None
        ):
            raise ValueError(
                "terminal audit-only readiness requires completed final artifacts"
            )

    attempt_state = _formal_attempt_state_evidence(output_dir)
    body = {
        "subject_mode": subject_mode,
        "source_identity": {
            "file": source_evidence,
            # The closed generations retain their original source provenance.
            # The readiness binding separately pins the executable finalizer.
            "source_commit": source_mapping["source_commit"],
            "source_checkout_hash": source_mapping["source_checkout_hash"],
        },
        "session_genesis": {
            "file": genesis_file,
            "session_genesis_hash": (
                genesis.session_genesis_hash if genesis is not None else None
            ),
        },
        "generation_session_contract": {
            "file": contract_evidence,
            "contract_hash": contract.contract_hash,
        },
        "continuation_policy": {
            "file": policy_evidence,
            "policy_hash": policy.policy_hash,
        },
        "ledger": {
            "file": ledger_evidence,
            "height": ledger_height,
            "snapshot_hash": canonical_json_hash(ledger_records),
        },
        "lifecycle": lifecycle_mapping,
        "cost": {
            "contract": {
                "file": cost_contract_evidence,
                "contract_hash": cost_contract.contract_hash,
            },
            "receipt_count": len(cost_chain),
            "head": cost_receipt_files[-1] if cost_receipt_files else None,
            "receipt_files": cost_receipt_files,
        },
        "attempt_state": attempt_state,
        "final_confirmation": {
            "plan": {"file": plan_evidence, "plan_hash": plan.plan_hash},
            "reservation": {
                "file": reservation_file,
                "reservation_hash": (
                    reservation.reservation_hash if reservation is not None else None
                ),
            },
            "result": {
                "file": result_file,
                "result_hash": result.result_hash if result is not None else None,
            },
        },
        "formal_report": report_file,
        "current_accounted_output_bytes": current_accounted_output_bytes,
    }
    return {**body, "snapshot_hash": canonical_json_hash(body)}

def _formal_attempt_state_evidence(output_dir: Path) -> dict[str, Any]:
    roots = tuple(
        output_dir / name
        for name in ("job_attempts", "live_invocations", "authoring")
    )
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError("formal attempt-state root is invalid")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("formal attempt-state path must not be a symlink")
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                files.append(_file_evidence(path))
    files.sort(key=lambda item: item["path"])
    body = {
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    return {**body, "snapshot_hash": canonical_json_hash(body)}

def _canonical_object_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _authoritative_file_evidence(path)
    payload = read_bounded_regular_file(path, max_bytes=max_bytes, label=label)
    value = strict_json_object_loads(payload, context=label)
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError(f"{label} is not canonical JSON")
    return evidence, value

def _authoritative_file_evidence(path: Path) -> dict[str, Any]:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("formal-state evidence file must not be a symlink")
    return _file_evidence(unresolved)

def _optional_file_evidence(path: Path) -> dict[str, Any]:
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError("formal-state optional evidence must not be a symlink")
    absolute = unresolved.resolve(strict=False)
    if not unresolved.exists():
        return {
            "path": str(absolute),
            "exists": False,
            "sha256": None,
            "size_bytes": 0,
        }
    evidence = _authoritative_file_evidence(unresolved)
    return {
        "path": evidence["path"],
        "exists": True,
        **{key: evidence[key] for key in ("sha256", "size_bytes")},
    }

def _lifecycle_receipt_file_evidence(output_dir: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    paths: list[Path] = []
    for dirname in ("generation_segments", "generation_recoveries"):
        root = output_dir / dirname
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError("generation lifecycle receipt root is invalid")
        paths.extend(root.rglob("*.json"))
    for path in sorted(paths, key=lambda item: str(item.resolve(strict=False))):
        evidence, payload = _canonical_object_file(
            path,
            label="generation lifecycle receipt",
            max_bytes=MAX_READINESS_RECEIPT_BYTES,
        )
        result.append({**evidence, "payload_hash": canonical_json_hash(payload)})
    return sorted(result, key=lambda item: item["payload_hash"])

def _assert_path_absent(path: Path, label: str) -> None:
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"{label} could not be inspected") from exc
    raise FileExistsError(f"{label} must be absent")

def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(value) != keys:
        raise ValueError(f"{label} fields mismatch")
    return dict(value)

def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value
