from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

P6_DIR = Path(__file__).resolve().parent
ROOT = P6_DIR.parents[2]
SRC = ROOT / "src"
P5_DIR = P6_DIR.parent / "P5"
EVIDENCE_INDEX = P6_DIR / "PHASE6_EVIDENCE_INDEX.json"
GO2_PROTOCOL = ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v2.json"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.agent_core.generation_contract import (  # noqa: E402
    generation_training_initialization_hash,
    load_generation_session_contract,
)
from true_rl_agent.agent_core.generation_lifecycle import (  # noqa: E402
    GenerationLifecycleIdentity,
    GenerationTerminalBinding,
    audit_generation_lifecycle,
    load_generation_continuation_policy,
)
from true_rl_agent.agent_core.generation_receipts import (  # noqa: E402
    GenerationReceiptJournal,
)
from true_rl_agent.agent_core.generation_record import (  # noqa: E402
    load_bound_generation_manifest,
)
from true_rl_agent.agent_core.ledger import (  # noqa: E402
    PROVIDER_AUTHORING_MODE,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.session_genesis import (  # noqa: E402
    ReferenceProgramGenesisAnchor,
    SessionGenesisArtifact,
)
from true_rl_agent.agent_core.trial_receipts import (  # noqa: E402
    TRIAL_EVAL_PHASE_RECEIPT_SCHEMA_VERSION,
    TrialReceiptRef,
    load_trial_eval_phase_receipt,
    load_trial_execution_receipt,
    load_trial_train_receipt,
)
from true_rl_agent.composition.cell_manifest import (  # noqa: E402
    load_resolved_cell_manifest,
)
from true_rl_agent.evaluation.acceptance_replay import (  # noqa: E402
    replay_acceptance_ledger,
)
from true_rl_agent.evaluation.binding import (  # noqa: E402
    EvaluationBindingV1,
    attach_evaluation_binding,
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.current_best import (  # noqa: E402
    CurrentBestEvalStats,
    attach_eval_protocol_hash,
)
from true_rl_agent.evaluation.phases import (  # noqa: E402
    EvaluationPhase,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.subject import ExperimentSubject  # noqa: E402
from true_rl_agent.json_contract import load_strict_json_object  # noqa: E402
from true_rl_agent.prompt_artifacts import (  # noqa: E402
    RoundAuthoringJournal,
    verify_prompt_payload,
)
from true_rl_agent.runner.specs import EvalJobSpec, GpuProcessBinding  # noqa: E402
from true_rl_agent.runner.subprocess_runner import SubprocessRunner  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.generation_bootstrap import (  # noqa: E402
    StairGenerationGuardReference,
    build_stair_generation_guard_reference,
    load_stair_next_generation_guard_reference,
)
from true_rl_agent.tasks.stair_climbing.phased_adapter import (  # noqa: E402
    StairPhasedEvaluationAdapter,
)
from true_rl_agent.tasks.stair_climbing.plugin import (  # noqa: E402
    StairClimbingTaskPlugin,
)
from true_rl_agent.tasks.stair_climbing.selection import (  # noqa: E402
    select_stair_public_dev_winner,
)
from true_rl_agent.tasks.stair_climbing.generation_session import (  # noqa: E402
    STAIR_GENERATION_REPORT_SCHEMA_VERSION,
    collect_actual_cost_usage,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (  # noqa: E402
    DEVELOPMENT_GENERATION_PILOT_MODE,
    DEVELOPMENT_GENERATION_VALIDATION_MODE,
    StairGenerationActualCostJournal,
    StairGenerationSessionCostContract,
    StairGenerationSessionCostContractV2,
    StairGenerationSessionCostContractV4,
    load_stair_generation_session_cost_contract,
    replay_stair_generation_combined_cost_chain,
)

VERIFY_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase6_verify.v1"
INDEX_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase6_evidence_index.v1"
GO2_CELL_VARIANT_ID = "stair.go2_12dof.unified_v2"
_PILOT_REPORT_FILENAME = "generation_session_report.json"
_PILOT_REQUIRED_FILES = (
    "generation_session_contract.json",
    "generation_session_cost.json",
    "generation_continuation_policy.json",
    "resolved_cell_manifest.json",
    "session_genesis.json",
    "ledger.jsonl",
    _PILOT_REPORT_FILENAME,
)
_SESSION_REQUIRED_KEYS = {
    "cell_variant_id",
    "cost_contract_hash",
    "expected_generation_count",
    "expected_generation_outcomes",
    "generation_session_contract_hash",
    "initial_policy_state_hash",
    "reference_checkpoint_hash",
    "reference_manifest_hash",
    "resolved_cell_manifest_hash",
    "resolved_cell_task_protocol_hash",
    "session_genesis_hash",
    "session_id",
    "session_mode",
    "source_commit",
    "source_provenance",
    "source_reference_genesis_checksum",
    "source_reference_genesis_hash",
    "source_reference_session_id",
}
_PILOT_EXTRA_KEYS = {"generation_execution_receipt_checksum"}
_FROZEN_SCIENTIFIC_IDENTITY_KEYS = (
    "cell_variant_id",
    "resolved_cell_manifest_hash",
    "resolved_cell_task_protocol_hash",
    "reference_manifest_hash",
    "reference_checkpoint_hash",
    "initial_policy_state_hash",
    "source_reference_session_id",
    "source_reference_genesis_hash",
    "source_reference_genesis_checksum",
)
_P5_SESSION_GENESIS = P5_DIR / "PHASE5_SESSION_GENESIS.json"
_P5_REFERENCE_EVAL_ARTIFACT = P5_DIR / "PHASE5_REFERENCE_EVAL_ARTIFACT.json"
_P5_REFERENCE_PUBLIC_DEV = P5_DIR / "PHASE5_REFERENCE_PUBLIC_DEV.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_json(item) for item in value]
    return value


def _require_commit(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a full lowercase git commit",
    )
    return value


def _load_index(path: Path = EVIDENCE_INDEX) -> dict[str, Any]:
    payload = load_strict_json_object(path, context="Phase 6 evidence index")
    _require(
        set(payload)
        == {"schema_version", "index_hash", "pilot", "validation", "excluded_diagnostics"},
        "Phase 6 evidence index fields mismatch",
    )
    _require(payload["schema_version"] == INDEX_SCHEMA_VERSION, "Phase 6 evidence index schema mismatch")
    expected_hash = canonical_json_hash(
        {key: value for key, value in payload.items() if key != "index_hash"}
    )
    _require(payload["index_hash"] == expected_hash, "Phase 6 evidence index hash mismatch")
    pilot = payload.get("pilot")
    _require(isinstance(pilot, Mapping), "Phase 6 pilot evidence is absent")
    _validate_session_index(pilot, expected_mode=DEVELOPMENT_GENERATION_PILOT_MODE)
    _require(
        set(pilot) == _SESSION_REQUIRED_KEYS | _PILOT_EXTRA_KEYS,
        "Phase 6 pilot evidence fields mismatch",
    )
    validation = payload.get("validation")
    if validation is not None:
        _require(isinstance(validation, Mapping), "Phase 6 validation evidence is invalid")
        _require(
            set(validation) == _SESSION_REQUIRED_KEYS,
            "Phase 6 validation evidence fields mismatch",
        )
        _validate_session_index(
            validation,
            expected_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
        )
        _require(
            all(validation[key] == pilot[key] for key in _FROZEN_SCIENTIFIC_IDENTITY_KEYS),
            "Phase 6 validation scientific identity differs from the pilot",
        )
    excluded = payload.get("excluded_diagnostics")
    _require(isinstance(excluded, list) and excluded, "Phase 6 excluded diagnostics are absent")
    excluded_ids: set[str] = set()
    for item in excluded:
        _require(
            isinstance(item, Mapping)
            and set(item)
            == {"status", "reason", "relative_output_dir", "session_id", "source_commit"},
            "Phase 6 excluded diagnostic fields mismatch",
        )
        _require(item.get("status") == "excluded", "Phase 6 diagnostic is not excluded")
        _require(
            isinstance(item.get("relative_output_dir"), str)
            and not Path(str(item["relative_output_dir"])).is_absolute(),
            "Phase 6 excluded diagnostic path must be relative",
        )
        _require_commit(item.get("source_commit"), "Phase 6 excluded diagnostic source commit")
        excluded_ids.add(str(item.get("session_id")))
    _require(str(pilot["session_id"]) not in excluded_ids, "Phase 6 pilot is marked excluded")
    if validation is not None:
        _require(str(validation["session_id"]) not in excluded_ids, "Phase 6 validation is marked excluded")
        _require(validation["session_id"] != pilot["session_id"], "Phase 6 pilot/validation session collision")
    return dict(payload)


def _validate_session_index(entry: Mapping[str, Any], *, expected_mode: str) -> None:
    missing = _SESSION_REQUIRED_KEYS - set(entry)
    _require(not missing, f"Phase 6 session evidence fields are missing: {sorted(missing)}")
    _require(entry.get("cell_variant_id") == GO2_CELL_VARIANT_ID, "Phase 6 evidence selected the wrong robot row")
    _require(entry.get("session_mode") == expected_mode, "Phase 6 evidence session mode mismatch")
    expected_provenance = (
        "terminal_report_attested_no_source_checkout_artifact"
        if expected_mode == DEVELOPMENT_GENERATION_PILOT_MODE
        else "typed_source_checkout_receipt"
    )
    _require(
        entry.get("source_provenance") == expected_provenance,
        "Phase 6 evidence source provenance mismatch",
    )
    expected_count = 1 if expected_mode == DEVELOPMENT_GENERATION_PILOT_MODE else 3
    _require(entry.get("expected_generation_count") == expected_count, "Phase 6 evidence generation count mismatch")
    outcomes = entry.get("expected_generation_outcomes")
    _require(
        isinstance(outcomes, list)
        and len(outcomes) == expected_count
        and all(isinstance(value, str) and value for value in outcomes),
        "Phase 6 evidence generation outcomes are invalid",
    )
    _require_commit(entry.get("source_commit"), "Phase 6 source commit")
    for key in (
        "cost_contract_hash",
        "generation_session_contract_hash",
        "initial_policy_state_hash",
        "reference_manifest_hash",
        "resolved_cell_manifest_hash",
        "resolved_cell_task_protocol_hash",
        "session_genesis_hash",
        "source_reference_genesis_hash",
    ):
        require_sha256_digest(entry.get(key), f"Phase 6 evidence {key}")
    for key in ("reference_checkpoint_hash", "source_reference_genesis_checksum"):
        require_sha256_checksum(entry.get(key), f"Phase 6 evidence {key}")


def _absolute_output_dir(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a Path")
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"{label} must be an existing regular directory")
    return path.resolve(strict=True)


def _load_phase5_verifier() -> Any:
    path = P5_DIR / "verify_phase5.py"
    spec = importlib.util.spec_from_file_location("phase6_phase5_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Phase 5 verifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_phase5(entry: Mapping[str, Any]) -> dict[str, Any]:
    result = _load_phase5_verifier().verify(require_go=True)
    _require(result.get("status") == "pass", "Phase 5 verifier did not pass")
    qualification = result.get("qualification")
    cell = result.get("cell")
    _require(
        isinstance(qualification, Mapping)
        and qualification.get("go") is True
        and isinstance(cell, Mapping),
        "Phase 5 Go2 qualification is not GO",
    )
    expected = {
        "cell_variant_id": entry["cell_variant_id"],
        "manifest_hash": entry["resolved_cell_manifest_hash"],
        "resolved_cell_task_protocol_hash": entry["resolved_cell_task_protocol_hash"],
        "reference_manifest_hash": entry["reference_manifest_hash"],
    }
    for key, value in expected.items():
        _require(cell.get(key) == value, f"Phase 5/P6 identity drift: {key}")
    _require(
        qualification.get("checkpoint_hash") == entry["reference_checkpoint_hash"],
        "Phase 5/P6 reference checkpoint drift",
    )
    return {
        "status": "go",
        "qualification_receipt_hash": qualification.get("receipt_hash"),
        "cell_variant_id": cell["cell_variant_id"],
        "resolved_cell_manifest_hash": cell["manifest_hash"],
        "resolved_cell_task_protocol_hash": cell["resolved_cell_task_protocol_hash"],
        "reference_checkpoint_hash": qualification["checkpoint_hash"],
    }


def _git_is_ancestor(ancestor: str, descendant: str = "HEAD") -> bool:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError("Phase 6 source ancestry could not be checked")
    return completed.returncode == 0


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verify_checkpoint_files(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    checksums: list[str] = []
    for record in records:
        private_artifacts = record.get("private_artifacts")
        _require(isinstance(private_artifacts, list), "Phase 6 generation private artifacts are absent")
        for item in private_artifacts:
            if not isinstance(item, Mapping) or not str(item.get("kind", "")).endswith("_train_receipt"):
                continue
            ref = TrialReceiptRef(
                kind="train",
                path=str(item.get("uri")),
                checksum=str(item.get("checksum")),
                size_bytes=int(item.get("size_bytes")),
            )
            loaded = load_trial_train_receipt(ref)
            checkpoint_ref = loaded.result.checkpoint
            _require(
                checkpoint_ref.uri.startswith("artifact://"),
                "Phase 6 candidate checkpoint lacks a public artifact URI",
            )
            local_path = checkpoint_ref.metadata.get("local_path")
            _require(
                isinstance(local_path, str) and local_path,
                "Phase 6 candidate checkpoint lacks its runner-local path",
            )
            checkpoint = Path(local_path)
            _require(checkpoint.is_absolute(), "Phase 6 checkpoint path is not absolute")
            resolved = checkpoint.resolve(strict=True)
            _require(not checkpoint.is_symlink() and resolved.is_file(), "Phase 6 checkpoint is not a regular file")
            try:
                resolved.relative_to(output_dir)
            except ValueError as exc:
                raise AssertionError("Phase 6 candidate checkpoint escaped its session output") from exc
            observed = _file_checksum(resolved)
            _require(observed == checkpoint_ref.checksum, "Phase 6 checkpoint checksum mismatch")
            checksums.append(observed)
    _require(checksums, "Phase 6 session has no candidate checkpoint evidence")
    return tuple(checksums)


def _verify_candidate_private_receipts(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checksums: dict[str, list[str]] = {
        "train": [],
        "eval_phase": [],
        "trial_execution": [],
    }
    public_dev_count = 0
    public_accept_count = 0
    sealed_validation_count = 0
    for record in records:
        private_artifacts = record.get("private_artifacts")
        _require(
            isinstance(private_artifacts, list),
            "Phase 6 generation private artifacts are absent",
        )
        for item in private_artifacts:
            if not isinstance(item, Mapping):
                continue
            artifact_kind = str(item.get("kind", ""))
            if not artifact_kind.startswith("lane-"):
                continue
            loader: Any
            receipt_kind: str
            bucket: str
            if artifact_kind.endswith("_train_receipt"):
                loader = load_trial_train_receipt
                receipt_kind = "train"
                bucket = "train"
            elif artifact_kind.endswith("_phase_receipt"):
                loader = load_trial_eval_phase_receipt
                receipt_kind = "eval_phase"
                bucket = "eval_phase"
                public_accept_count += int("_public_accept_" in artifact_kind)
                sealed_validation_count += int("_sealed_validation_" in artifact_kind)
            elif artifact_kind.endswith("_trial_execution_receipt"):
                loader = load_trial_execution_receipt
                receipt_kind = "execution"
                bucket = "trial_execution"
            else:
                continue
            checksum = str(item.get("checksum"))
            receipt_path = Path(str(item.get("uri")))
            _require(receipt_path.is_absolute(), "Phase 6 candidate receipt path is not absolute")
            resolved_receipt_path = receipt_path.resolve(strict=True)
            _require(
                not receipt_path.is_symlink() and resolved_receipt_path.is_file(),
                "Phase 6 candidate receipt is not a regular file",
            )
            try:
                resolved_receipt_path.relative_to(output_dir)
            except ValueError as exc:
                raise AssertionError("Phase 6 candidate receipt escaped its session output") from exc
            ref = TrialReceiptRef(
                kind=receipt_kind,
                path=str(resolved_receipt_path),
                checksum=checksum,
                size_bytes=int(item.get("size_bytes")),
            )
            loaded = loader(ref)
            loaded_ref = loaded[1].receipt_ref if isinstance(loaded, tuple) else loaded.receipt_ref
            _require(
                loaded_ref.checksum == checksum,
                "Phase 6 candidate private receipt checksum mismatch",
            )
            checksums[bucket].append(checksum)
            if bucket == "eval_phase":
                public_dev_count += int("_public_dev_" in artifact_kind)
    _require(checksums["train"], "Phase 6 candidate train receipts are absent")
    _require(checksums["eval_phase"], "Phase 6 candidate eval-phase receipts are absent")
    _require(
        public_dev_count > 0
        and public_accept_count > 0
        and sealed_validation_count > 0,
        "Phase 6 evidence lacks public_dev, public_accept, or sealed receipt closure",
    )
    _require(
        checksums["trial_execution"],
        "Phase 6 formal trial-execution receipt is absent",
    )
    return {
        "counts": {key: len(value) for key, value in checksums.items()},
        "checksums": checksums,
        "public_dev_receipt_count": public_dev_count,
        "public_accept_receipt_count": public_accept_count,
        "sealed_validation_receipt_count": sealed_validation_count,
    }


def _reference_eval_phase_results(
    genesis: SessionGenesisArtifact,
) -> dict[EvaluationPhase, EvaluationPhaseResult]:
    anchor = genesis.anchor
    _require(
        isinstance(anchor, ReferenceProgramGenesisAnchor),
        "Phase 6 prompt privacy replay lacks a reference Genesis anchor",
    )
    artifact_path = Path(anchor.private_eval_artifact.uri)
    _require(
        artifact_path.resolve(strict=True).is_file()
        and _file_checksum(artifact_path) == anchor.private_eval_artifact.checksum,
        "Phase 6 reference private eval artifact checksum mismatch",
    )
    artifact = load_strict_json_object(
        artifact_path,
        context="Phase 6 reference private eval artifact",
        require_canonical=True,
        trailing_newline=True,
    )
    assembly = artifact.get("evaluation_assembly")
    bindings = assembly.get("phase_receipts") if isinstance(assembly, Mapping) else None
    _require(
        isinstance(bindings, list),
        "Phase 6 reference private eval artifact lacks phase receipts",
    )
    results: dict[EvaluationPhase, EvaluationPhaseResult] = {}
    for phase in EvaluationPhase:
        matches = [
            item
            for item in bindings
            if isinstance(item, Mapping) and item.get("phase") == phase.value
        ]
        _require(
            len(matches) == 1,
            f"Phase 6 reference {phase.value} receipt is not unique",
        )
        item = matches[0]
        loaded = load_trial_eval_phase_receipt(
            TrialReceiptRef(
                kind="eval_phase",
                path=str(item.get("path")),
                checksum=str(item.get("checksum")),
                size_bytes=int(item.get("size_bytes")),
                schema_version=str(item.get("ref_schema_version")),
            )
        )
        _require(
            loaded.result.phase is phase,
            f"Phase 6 reference {phase.value} receipt phase mismatch",
        )
        results[phase] = loaded.result
    return results


def _optional_generation_phase_result(
    *,
    output_dir: Path,
    record: Mapping[str, Any],
    lane_id: str,
    phase: EvaluationPhase,
) -> EvaluationPhaseResult | None:
    kind = f"{lane_id}_{phase.value}_phase_receipt"
    artifacts = record.get("private_artifacts")
    _require(
        isinstance(artifacts, list),
        "Phase 6 prompt privacy replay lacks private artifacts",
    )
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("kind") == kind
    ]
    _require(len(matches) <= 1, f"Phase 6 {kind} is duplicated")
    if not matches:
        return None
    ref = _private_trial_receipt_ref(
        output_dir=output_dir,
        record=record,
        artifact_kind=kind,
        receipt_kind="eval_phase",
        require_session_root=True,
    )
    loaded = load_trial_eval_phase_receipt(ref)
    _require(
        loaded.result.phase is phase,
        f"Phase 6 {kind} phase mismatch",
    )
    return loaded.result


def _optional_generation_execution_public_evidence(
    *,
    output_dir: Path,
    record: Mapping[str, Any],
    lane_id: str,
) -> tuple[Mapping[str, Any], ...]:
    kind = f"{lane_id}_trial_execution_receipt"
    artifacts = record.get("private_artifacts")
    _require(isinstance(artifacts, list), "Phase 6 private artifacts are absent")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("kind") == kind
    ]
    _require(len(matches) <= 1, f"Phase 6 {kind} is duplicated")
    if not matches:
        return ()
    ref = _private_trial_receipt_ref(
        output_dir=output_dir,
        record=record,
        artifact_kind=kind,
        receipt_kind="execution",
        require_session_root=True,
    )
    _contract, loaded = load_trial_execution_receipt(ref)
    public_result = loaded.public_result
    _require(
        public_result is not None,
        "Phase 6 formal execution lacks its public result",
    )
    return (
        public_result.to_public_dict(),
        loaded.decision.to_public_dict(),
    )


def _json_string_fragments(value: Any) -> set[str]:
    fragments: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            fragments.add(str(key))
            fragments.update(_json_string_fragments(item))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            fragments.update(_json_string_fragments(item))
    elif isinstance(value, str):
        fragments.add(value)
    return fragments


def _high_entropy_float_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            tokens.update(_high_entropy_float_tokens(item))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            tokens.update(_high_entropy_float_tokens(item))
    elif isinstance(value, float) and math.isfinite(value):
        token = json.dumps(value, allow_nan=False)
        if sum(char.isdigit() for char in token) >= 8:
            tokens.add(token)
    return tokens


def _phase_feedback_summary(result: EvaluationPhaseResult) -> Mapping[str, Any]:
    evidence = result.task_evidence
    summary = evidence.get("summary") if isinstance(evidence, Mapping) else None
    _require(
        isinstance(summary, Mapping),
        "Phase 6 eval phase lacks its task-owned feedback summary",
    )
    return summary


def _assert_prompt_payload_private_feedback_absent(
    *,
    payload: Mapping[str, Any],
    private_summaries: Sequence[Mapping[str, Any]],
    public_authorities: Sequence[Mapping[str, Any]],
) -> None:
    private_strings = set().union(
        *(_json_string_fragments(value) for value in private_summaries)
    )
    private_cell_ids = {
        item
        for item in private_strings
        if item.startswith("public_accept.")
        or item.startswith("sealed_validation.")
    }
    private_numbers = set().union(
        *(_high_entropy_float_tokens(value) for value in private_summaries)
    )
    public_numbers = set().union(
        *(_high_entropy_float_tokens(value) for value in public_authorities)
    )
    private_only_numbers = private_numbers - public_numbers
    prompt_strings = _json_string_fragments(payload)
    prompt_numbers = _high_entropy_float_tokens(payload)
    cell_hits = sorted(
        token
        for token in private_cell_ids
        if any(token in fragment for fragment in prompt_strings)
    )
    number_hits = sorted(private_only_numbers.intersection(prompt_numbers))
    number_hits.extend(
        token
        for token in sorted(private_only_numbers - set(number_hits))
        if any(token in fragment for fragment in prompt_strings)
    )
    _require(
        not cell_hits and not number_hits,
        "Phase 6 provider prompt leaked private eval feedback: "
        + ",".join((*cell_hits[:3], *number_hits[:3])),
    )


def _verify_all_authoring_prompt_privacy(
    *,
    output_dir: Path,
    genesis: SessionGenesisArtifact,
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify every provider attempt, not only the prompt selected into ledger."""

    reference_phases = _reference_eval_phase_results(genesis)
    private_summaries: list[Mapping[str, Any]] = [
        _phase_feedback_summary(reference_phases[EvaluationPhase.PUBLIC_ACCEPT]),
        _phase_feedback_summary(reference_phases[EvaluationPhase.SEALED_VALIDATION]),
    ]
    public_authorities: list[Mapping[str, Any]] = [
        _phase_feedback_summary(reference_phases[EvaluationPhase.PUBLIC_DEV]),
    ]
    anchor = genesis.anchor
    assert isinstance(anchor, ReferenceProgramGenesisAnchor)
    public_authorities.append(anchor.public_result)
    prompt_payloads: list[Mapping[str, Any]] = []
    prompt_ref_identities: set[tuple[str, str, int]] = set()
    for record in ledger:
        for field_name in ("selection", "parent_current_best", "resulting_current_best"):
            value = record.get(field_name)
            if isinstance(value, Mapping):
                public_authorities.append(value)
        generation_id = record.get("generation_id")
        trials = record.get("trials")
        _require(
            isinstance(generation_id, str) and isinstance(trials, list),
            "Phase 6 prompt privacy replay lacks generation trials",
        )
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            lane_id = trial.get("lane_id")
            trial_id = trial.get("trial_id")
            _require(
                isinstance(lane_id, str)
                and lane_id
                and isinstance(trial_id, str)
                and trial_id,
                "Phase 6 prompt privacy replay has an invalid trial identity",
            )
            for phase in EvaluationPhase:
                result = _optional_generation_phase_result(
                    output_dir=output_dir,
                    record=record,
                    lane_id=lane_id,
                    phase=phase,
                )
                if result is None:
                    continue
                summary = _phase_feedback_summary(result)
                if phase is EvaluationPhase.PUBLIC_DEV:
                    public_authorities.append(summary)
                else:
                    private_summaries.append(summary)
            public_authorities.extend(
                _optional_generation_execution_public_evidence(
                    output_dir=output_dir,
                    record=record,
                    lane_id=lane_id,
                )
            )
            round_dir = output_dir / "authoring" / generation_id / lane_id
            snapshot = RoundAuthoringJournal(round_dir, round_id=trial_id).snapshot()
            for prompt_ref in snapshot.prompt_payload_refs:
                path = Path(prompt_ref.path)
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(output_dir)
                except ValueError as exc:
                    raise AssertionError(
                        "Phase 6 authoring prompt escaped its session output"
                    ) from exc
                _require(
                    not path.is_symlink() and resolved.is_file(),
                    "Phase 6 authoring prompt is not a regular file",
                )
                identity = (
                    str(resolved),
                    prompt_ref.checksum,
                    prompt_ref.size_bytes,
                )
                _require(
                    identity not in prompt_ref_identities,
                    "Phase 6 authoring prompt ref is duplicated across attempts",
                )
                prompt_ref_identities.add(identity)
                prompt_payloads.append(verify_prompt_payload(prompt_ref))
    _require(prompt_payloads, "Phase 6 session has no provider prompt attempts")
    for payload in prompt_payloads:
        _assert_prompt_payload_private_feedback_absent(
            payload=payload,
            private_summaries=private_summaries,
            public_authorities=public_authorities,
        )
    private_strings = set().union(
        *(_json_string_fragments(value) for value in private_summaries)
    )
    private_numbers = set().union(
        *(_high_entropy_float_tokens(value) for value in private_summaries)
    )
    public_numbers = set().union(
        *(_high_entropy_float_tokens(value) for value in public_authorities)
    )
    return {
        "prompt_payload_ref_count": len(prompt_ref_identities),
        "private_phase_summary_count": len(private_summaries),
        "public_authority_source_count": len(public_authorities),
        "private_cell_id_token_count": sum(
            item.startswith("public_accept.")
            or item.startswith("sealed_validation.")
            for item in private_strings
        ),
        "private_only_numeric_token_count": len(private_numbers - public_numbers),
        "private_feedback_leak_count": 0,
    }


def _replay_session_cost_chain(
    *,
    contract: Any,
    actual_cost_receipts: Sequence[Any],
    lifecycle: Any,
) -> str:
    """Dispatch legacy-pilot and bounded-validation cost topology by typed contract."""

    if isinstance(contract, StairGenerationSessionCostContractV2):
        return replay_stair_generation_combined_cost_chain(
            contract=contract,
            actual_cost_receipts=actual_cost_receipts,
            lifecycle=lifecycle,
        )
    if type(contract) is not StairGenerationSessionCostContract:
        raise TypeError("Phase 6 cost contract type is unsupported")
    _require(
        not lifecycle.recovery_cost_edges,
        "Phase 6 legacy pilot cost chain cannot contain recovery edges",
    )
    head = contract.contract_hash
    for receipt in actual_cost_receipts:
        _require(
            receipt.payload.get("previous_cost_chain_head_hash") == head,
            "Phase 6 legacy pilot actual-cost chain is not linear",
        )
        head = receipt.payload_hash
    if lifecycle.segment_count:
        _require(
            lifecycle.cost_chain_head_hash == head,
            "Phase 6 legacy pilot lifecycle cost head diverged",
        )
    return head


def _verify_recomputed_actual_cost(
    *,
    output_dir: Path,
    session_id: str,
    genesis: SessionGenesisArtifact,
    session_contract: Any,
    cost_contract: Any,
    ledger: Sequence[Mapping[str, Any]],
    final_cost_receipt: Any,
) -> dict[str, Any]:
    """Bind the final receipt to the product's raw-artifact cost recomputation."""

    foundation = SimpleNamespace(
        config=SimpleNamespace(output_dir=output_dir, session_id=session_id),
        cost_contract=cost_contract,
        session_contract=session_contract,
    )
    cost_genesis = SimpleNamespace(
        reference_genesis_import_ref=genesis.reference_genesis_import_ref,
        reference_eval_artifact_path=lambda: Path(
            genesis.anchor.private_eval_artifact.uri
        ),
    )
    usage, conservative_charges, open_reservations = collect_actual_cost_usage(
        foundation,
        genesis=cost_genesis,
        ledger=[dict(record) for record in ledger],
    )
    _require(
        not open_reservations,
        "Phase 6 terminal cost closure has open job-attempt reservations",
    )
    actual = final_cost_receipt.payload.get("actual")
    _require(isinstance(actual, Mapping), "Phase 6 terminal actual-cost payload is absent")
    recomputed = usage.to_dict()
    for field_name, current in recomputed.items():
        if field_name in {"observed_output_bytes", "observed_output_files"}:
            continue
        _require(
            actual.get(field_name) == current,
            f"Phase 6 terminal actual cost drifted: {field_name}",
        )
    _require(
        recomputed["observed_output_bytes"] <= actual.get("accounted_output_bytes", -1),
        "Phase 6 terminal output exceeded its settled reserve",
    )
    _require(
        tuple(final_cost_receipt.payload.get("conservative_charges", ()))
        == conservative_charges,
        "Phase 6 terminal conservative cost charges drifted",
    )
    return {
        "actual": recomputed,
        "open_job_attempt_reservation_count": 0,
        "conservative_charges": list(conservative_charges),
    }


def _initial_current_best(
    genesis: SessionGenesisArtifact,
    first_record: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    acceptance_protocol: Any,
) -> CurrentBestEvalStats:
    anchor = genesis.anchor
    _require(isinstance(anchor, ReferenceProgramGenesisAnchor), "Phase 6 Genesis is not reference anchored")
    initial = CurrentBestEvalStats.from_reference_public_result_mapping(
        reference_id=anchor.reference_manifest.reference_id,
        reference_manifest_hash=str(anchor.reference_manifest.manifest_hash),
        program_source_checksum=anchor.reference_manifest.source_checksum,
        training_binding_hash=str(anchor.training_binding.binding_hash),
        capability_profile_hash=anchor.reference_manifest.capability_profile_hash,
        checkpoint_hash=anchor.checkpoint_hash,
        public_result=anchor.public_result,
        acceptance_protocol=acceptance_protocol,
        expected_evaluation_binding=genesis.evaluation_binding,
    )
    ledger_parent = CurrentBestEvalStats.from_mapping(
        first_record.get("parent_current_best")
    )
    _require(
        ledger_parent == initial,
        "Phase 6 initial ledger parent differs from reconstructed Genesis evidence",
    )
    _require(
        initial.anchor_id == anchor.reference_manifest.reference_id
        and initial.subject_manifest_hash == anchor.reference_manifest.manifest_hash
        and initial.program_source_checksum == anchor.reference_manifest.source_checksum
        and initial.training_binding_hash == anchor.training_binding.binding_hash
        and initial.capability_profile_hash == anchor.reference_manifest.capability_profile_hash
        and initial.checkpoint_hash == anchor.checkpoint_hash
        and initial.evaluation_binding() == genesis.evaluation_binding,
        "Phase 6 initial current-best differs from Session Genesis",
    )
    _require(anchor.checkpoint_hash == entry["reference_checkpoint_hash"], "Phase 6 reference checkpoint mismatch")
    _require(anchor.reference_manifest.manifest_hash == entry["reference_manifest_hash"], "Phase 6 reference manifest mismatch")
    _require(anchor.initial_policy_state_hash == entry["initial_policy_state_hash"], "Phase 6 initial policy state mismatch")
    import_ref = genesis.reference_genesis_import_ref
    _require(import_ref is not None, "Phase 6 Genesis is not an imported reference")
    _require(
        import_ref.source_session_id == entry["source_reference_session_id"]
        and import_ref.source_session_genesis_hash == entry["source_reference_genesis_hash"]
        and import_ref.source_session_genesis_checksum == entry["source_reference_genesis_checksum"],
        "Phase 6 imported reference lineage mismatch",
    )
    return initial


def _verify_imported_reference_genesis(
    genesis: SessionGenesisArtifact,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    import_ref = genesis.reference_genesis_import_ref
    _require(import_ref is not None, "Phase 6 Genesis is not an imported reference")
    source_path = Path(import_ref.source_session_genesis_uri)
    _require(source_path.is_absolute(), "Phase 6 imported Genesis URI is not absolute")
    _require(
        not source_path.is_symlink() and source_path.is_file(),
        "Phase 6 imported Genesis URI is not a regular file",
    )
    source_bytes = source_path.read_bytes()
    p5_bytes = _P5_SESSION_GENESIS.read_bytes()
    source_checksum = "sha256:" + hashlib.sha256(source_bytes).hexdigest()
    _require(
        source_checksum
        == import_ref.source_session_genesis_checksum
        == entry["source_reference_genesis_checksum"],
        "Phase 6 imported Genesis bytes do not match their checksum",
    )
    _require(
        source_bytes == p5_bytes,
        "Phase 6 imported Genesis bytes differ from Phase 5 verified Genesis",
    )
    source_payload = load_strict_json_object(
        source_path,
        context="Phase 6 imported reference Genesis",
        require_canonical=True,
        trailing_newline=True,
    )
    p5_payload = load_strict_json_object(
        _P5_SESSION_GENESIS,
        context="Phase 5 verified reference Genesis",
        require_canonical=True,
        trailing_newline=True,
    )
    source_genesis = SessionGenesisArtifact.from_mapping(source_payload)
    p5_genesis = SessionGenesisArtifact.from_mapping(p5_payload)
    _require(
        source_genesis == p5_genesis
        and source_genesis.session_id == entry["source_reference_session_id"]
        and source_genesis.session_genesis_hash
        == import_ref.source_session_genesis_hash
        == entry["source_reference_genesis_hash"],
        "Phase 6 imported Genesis parsed identity differs from Phase 5",
    )
    _require(
        source_genesis.anchor == genesis.anchor
        and source_genesis.runtime_snapshot_hash == genesis.runtime_snapshot_hash
        and source_genesis.evaluation_binding == genesis.evaluation_binding,
        "Phase 6 imported Genesis anchor/runtime/evaluation identity drifted",
    )
    return {
        "source_session_id": source_genesis.session_id,
        "source_session_genesis_hash": source_genesis.session_genesis_hash,
        "source_session_genesis_checksum": source_checksum,
        "raw_bytes_match_phase5": True,
    }


def _verify_generation_manifest_receipt_path(
    output_dir: Path,
    record: Mapping[str, Any],
) -> Path:
    artifacts = record.get("private_artifacts")
    checksum = record.get("manifest_receipt_checksum")
    _require(isinstance(artifacts, list), "Phase 6 generation private artifacts are absent")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == "generation_manifest_receipt"
        and item.get("checksum") == checksum
    ]
    _require(len(matches) == 1, "Phase 6 generation manifest receipt is not unique")
    path = Path(str(matches[0].get("uri")))
    _require(path.is_absolute(), "Phase 6 generation manifest receipt path is not absolute")
    resolved = path.resolve(strict=True)
    _require(
        not path.is_symlink() and resolved.is_file(),
        "Phase 6 generation manifest receipt is not a regular file",
    )
    try:
        resolved.relative_to(output_dir)
    except ValueError as exc:
        raise AssertionError(
            "Phase 6 generation manifest receipt escaped its session output"
        ) from exc
    return resolved


def _p5_public_dev_guard_source() -> tuple[Any, TrialReceiptRef]:
    """Load the P5-verified public-dev bytes and their production receipt ref."""

    local_path = _P5_REFERENCE_PUBLIC_DEV.resolve(strict=True)
    local_payload = load_strict_json_object(
        local_path,
        context="Phase 6 P5 public_dev receipt copy",
        require_canonical=True,
        trailing_newline=True,
    )
    _require(
        set(local_payload)
        == {
            "schema_version",
            "contract_hash",
            "job_spec_hash",
            "phase_result",
            "phase_result_hash",
        },
        "Phase 6 P5 public_dev receipt fields drifted",
    )
    _require(
        local_payload.get("schema_version")
        == TRIAL_EVAL_PHASE_RECEIPT_SCHEMA_VERSION,
        "Phase 6 P5 public_dev receipt schema drifted",
    )
    phase_payload = local_payload.get("phase_result")
    _require(
        canonical_json_hash(phase_payload) == local_payload.get("phase_result_hash"),
        "Phase 6 P5 public_dev phase-result hash drifted",
    )
    phase_result = EvaluationPhaseResult.from_private_mapping(phase_payload)
    _require(
        phase_result.phase is EvaluationPhase.PUBLIC_DEV,
        "Phase 6 P5 guard source is not public_dev",
    )
    local_checksum = _file_checksum(local_path)
    local_size = local_path.stat().st_size
    artifact = load_strict_json_object(
        _P5_REFERENCE_EVAL_ARTIFACT,
        context="Phase 6 P5 reference evaluation artifact",
        require_canonical=True,
        trailing_newline=True,
    )
    assembly = artifact.get("evaluation_assembly")
    bindings = assembly.get("phase_receipts") if isinstance(assembly, Mapping) else None
    matches = [
        item
        for item in bindings or ()
        if isinstance(item, Mapping)
        and item.get("phase") == EvaluationPhase.PUBLIC_DEV.value
    ]
    _require(
        len(matches) == 1,
        "Phase 6 P5 public_dev production receipt is not unique",
    )
    binding = matches[0]
    production_ref = TrialReceiptRef(
        kind="eval_phase",
        path=str(binding.get("path")),
        checksum=str(binding.get("checksum")),
        size_bytes=int(binding.get("size_bytes")),
        schema_version=str(binding.get("ref_schema_version")),
    )
    _require(
        production_ref.checksum == local_checksum
        and production_ref.size_bytes == local_size
        and binding.get("phase_result_hash")
        == canonical_json_hash(phase_result.to_private_dict())
        and binding.get("job_spec_hash") == local_payload.get("job_spec_hash"),
        "Phase 6 P5 public_dev copy differs from its verified production receipt",
    )
    return phase_result, production_ref


def _initial_stair_guard_reference(
    *,
    genesis: SessionGenesisArtifact,
    parent: CurrentBestEvalStats,
    config: StairClimbingTaskConfig,
) -> tuple[StairPhasedEvaluationAdapter, StairGenerationGuardReference]:
    anchor = genesis.anchor
    _require(
        isinstance(anchor, ReferenceProgramGenesisAnchor),
        "Phase 6 guard replay lacks a reference Genesis anchor",
    )
    metrics = anchor.public_result.get("metrics")
    mask = metrics.get("objective_family_mask") if isinstance(metrics, Mapping) else None
    _require(
        isinstance(mask, Mapping),
        "Phase 6 guard replay lacks the frozen objective family mask",
    )
    frozen_mask = _plain_json(mask)
    adapter = StairPhasedEvaluationAdapter(
        config=config,
        objective_family_mask_provider=lambda: frozen_mask,
    )
    phase_result, phase_receipt_ref = _p5_public_dev_guard_source()
    guard = build_stair_generation_guard_reference(
        adapter=adapter,
        parent=parent,
        trial_id="genesis-parent",
        phase_result=phase_result,
        phase_receipt_ref=phase_receipt_ref,
    )
    return adapter, guard


def _private_trial_receipt_ref(
    *,
    output_dir: Path,
    record: Mapping[str, Any],
    artifact_kind: str,
    receipt_kind: str,
    require_session_root: bool,
) -> TrialReceiptRef:
    artifacts = record.get("private_artifacts")
    _require(
        isinstance(artifacts, list),
        "Phase 6 generation private artifacts are absent",
    )
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("kind") == artifact_kind
    ]
    _require(
        len(matches) == 1,
        f"Phase 6 {artifact_kind} is not uniquely bound",
    )
    item = matches[0]
    raw_path = str(item.get("uri"))
    path = Path(raw_path)
    _require(
        path.is_absolute()
        or PurePosixPath(raw_path).is_absolute()
        or PureWindowsPath(raw_path).is_absolute(),
        f"Phase 6 {artifact_kind} path is not absolute",
    )
    if require_session_root:
        resolved = path.resolve(strict=True)
        _require(
            not path.is_symlink() and resolved.is_file(),
            f"Phase 6 {artifact_kind} is not a regular file",
        )
        try:
            resolved.relative_to(output_dir)
        except ValueError as exc:
            raise AssertionError(
                f"Phase 6 {artifact_kind} escaped its session output"
            ) from exc
        path = resolved
        raw_path = str(resolved)
    return TrialReceiptRef(
        kind=receipt_kind,
        path=raw_path,
        checksum=str(item.get("checksum")),
        size_bytes=int(item.get("size_bytes")),
    )


def _assert_generation_guard_reference(
    *,
    output_dir: Path,
    record: Mapping[str, Any],
    manifest: Any,
    guard: StairGenerationGuardReference,
) -> TrialReceiptRef:
    observed_ref = _private_trial_receipt_ref(
        output_dir=output_dir,
        record=record,
        artifact_kind="generation_guard_reference_phase_receipt",
        receipt_kind="eval_phase",
        require_session_root=False,
    )
    _require(
        manifest.guard_reference_evidence_hash == guard.evidence.evidence_hash
        and manifest.guard_reference_phase_receipt_hash
        == canonical_json_hash(guard.phase_receipt_ref.to_dict())
        and observed_ref == guard.phase_receipt_ref,
        "Phase 6 generation guard reference differs from replayed current-best",
    )
    return observed_ref


def _rebuild_candidate_selector_screen(
    *,
    output_dir: Path,
    record: Mapping[str, Any],
    trial: Mapping[str, Any],
    adapter: StairPhasedEvaluationAdapter,
) -> Any:
    lane_id = trial.get("lane_id")
    trial_id = trial.get("trial_id")
    train_projection = trial.get("train")
    public_dev_projection = trial.get("public_dev")
    _require(
        isinstance(lane_id, str)
        and lane_id
        and isinstance(trial_id, str)
        and trial_id
        and isinstance(train_projection, Mapping)
        and isinstance(public_dev_projection, Mapping),
        "Phase 6 selector trial lacks train/public_dev evidence",
    )
    train_ref = _private_trial_receipt_ref(
        output_dir=output_dir,
        record=record,
        artifact_kind=f"{lane_id}_train_receipt",
        receipt_kind="train",
        require_session_root=True,
    )
    phase_ref = _private_trial_receipt_ref(
        output_dir=output_dir,
        record=record,
        artifact_kind=f"{lane_id}_public_dev_phase_receipt",
        receipt_kind="eval_phase",
        require_session_root=True,
    )
    train_loaded = load_trial_train_receipt(train_ref)
    phase_loaded = load_trial_eval_phase_receipt(phase_ref)
    _require(
        phase_loaded.result.phase is EvaluationPhase.PUBLIC_DEV
        and train_ref.checksum == train_projection.get("train_receipt_checksum")
        and phase_ref.checksum == public_dev_projection.get("phase_receipt_checksum")
        and phase_loaded.job_spec_hash == public_dev_projection.get("job_spec_hash")
        and canonical_json_hash(phase_loaded.result.to_private_dict())
        == public_dev_projection.get("phase_result_hash"),
        "Phase 6 selector source receipt differs from its generation projection",
    )
    mechanism_delta = trial.get("mechanism_operation_delta")
    _require(
        isinstance(mechanism_delta, Mapping),
        "Phase 6 selector trial lacks its verifier-owned mechanism delta",
    )
    return adapter.project_public_dev_screen(
        trial_id=trial_id,
        phase_result=phase_loaded.result,
        screen_role="candidate",
        train_result=train_loaded.result,
        train_receipt_checksum=train_ref.checksum,
        mechanism_operation_delta=mechanism_delta,
    )


def _assert_selection_receipt_binding(
    *,
    output_dir: Path,
    record: Mapping[str, Any],
    loaded: Any,
) -> None:
    artifacts = record.get("private_artifacts")
    checksum = record.get("selection_receipt_checksum")
    _require(
        isinstance(artifacts, list) and isinstance(checksum, str),
        "Phase 6 selection receipt binding is absent",
    )
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == "generation_selection_receipt"
        and item.get("checksum") == checksum
    ]
    _require(len(matches) == 1, "Phase 6 selection receipt is not uniquely bound")
    item = matches[0]
    path = Path(str(item.get("uri")))
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(output_dir)
    except ValueError as exc:
        raise AssertionError("Phase 6 selection receipt escaped its session output") from exc
    _require(
        not path.is_symlink()
        and resolved.is_file()
        and loaded.receipt_ref.path == str(resolved)
        and loaded.receipt_ref.checksum == checksum
        and loaded.receipt_ref.size_bytes == int(item.get("size_bytes")),
        "Phase 6 selection receipt ref differs from its private artifact",
    )


def _assert_task_owned_selector_replay(
    *,
    loaded_selection: Any,
    stored_selection: Mapping[str, Any],
    screens: Sequence[Any],
    rebuilt_selection: Any,
    guard: StairGenerationGuardReference,
    guard_receipt_ref: TrialReceiptRef,
) -> None:
    rebuilt_screen_payloads = tuple(item.to_selector_dict() for item in screens)
    rebuilt_selection_payload = rebuilt_selection.to_public_dict()
    _require(
        tuple(loaded_selection.screens) == rebuilt_screen_payloads
        and loaded_selection.selection == rebuilt_selection_payload
        and dict(stored_selection) == rebuilt_selection_payload
        and loaded_selection.guard_reference == guard.evidence.to_selector_dict()
        and loaded_selection.guard_reference_phase_receipt_checksum
        == guard_receipt_ref.checksum,
        "Phase 6 selector receipt differs from task-owned replay",
    )


def _rebuild_formal_eval_plan(
    *,
    stored_plan: Mapping[str, Any],
    train_result: Any,
    phase_results: Sequence[EvaluationPhaseResult],
) -> EvalJobSpec:
    """Rebuild the private runner plan without trusting the stored formal result."""

    _require(
        len(phase_results) == len(EvaluationPhase),
        "Phase 6 formal replay requires exactly three phase results",
    )
    private_gpu_bindings = [result.gpu_process_binding for result in phase_results]
    _require(
        all(isinstance(item, Mapping) for item in private_gpu_bindings)
        and all(item == private_gpu_bindings[0] for item in private_gpu_bindings[1:]),
        "Phase 6 formal phases disagree on their private GPU binding",
    )
    backend_verifications = [
        result.backend_execution_environment_verification for result in phase_results
    ]
    _require(
        all(isinstance(item, Mapping) for item in backend_verifications)
        and all(item == backend_verifications[0] for item in backend_verifications[1:]),
        "Phase 6 formal phases disagree on their backend environment",
    )
    backend_verification = backend_verifications[0]
    assert isinstance(backend_verification, Mapping)
    checkpoint = train_result.checkpoint
    _require(
        isinstance(stored_plan.get("checkpoint"), Mapping)
        and checkpoint.to_public_dict(include_metadata=False)
        == dict(stored_plan["checkpoint"]),
        "Phase 6 formal eval plan checkpoint differs from its train receipt",
    )
    eval_plan = EvalJobSpec(
        job_id=str(stored_plan.get("job_id")),
        task_id=str(stored_plan.get("task_id")),
        subject=ExperimentSubject.from_mapping(stored_plan.get("subject")),
        subject_manifest_hash=str(stored_plan.get("subject_manifest_hash")),
        program_source_checksum=str(stored_plan.get("program_source_checksum")),
        training_binding_hash=str(stored_plan.get("training_binding_hash")),
        capability_profile_hash=str(stored_plan.get("capability_profile_hash")),
        checkpoint=checkpoint,
        public_seed_count=int(stored_plan.get("public_seed_count")),
        public_accept_seed_count=int(stored_plan.get("public_accept_seed_count")),
        sealed_validation_seed_count=int(
            stored_plan.get("sealed_validation_seed_count")
        ),
        eval_num_envs=int(stored_plan.get("eval_num_envs")),
        eval_protocol_version=str(stored_plan.get("eval_protocol_version")),
        evaluation_binding=EvaluationBindingV1.from_mapping(
            stored_plan.get("evaluation_binding")
        ),
        execution_device=str(stored_plan.get("execution_device")),
        initial_policy_state_pairing_hash=(
            str(stored_plan.get("initial_policy_state_pairing_hash"))
            if stored_plan.get("initial_policy_state_pairing_hash") is not None
            else None
        ),
        gpu_process_binding=GpuProcessBinding.from_mapping(private_gpu_bindings[0]),
        metadata={
            "backend_id": str(backend_verification.get("backend_id")),
            "backend_execution_environment_manifest_hash": str(
                backend_verification.get("expected_manifest_hash")
            ),
        },
    )
    _require(
        eval_plan.to_public_dict() == dict(stored_plan),
        "Phase 6 rebuilt formal eval plan differs from the receipt-bound plan",
    )
    return eval_plan


def _assert_reassembled_formal_chain(
    *,
    record: Mapping[str, Any],
    trial: Mapping[str, Any],
    formal: Mapping[str, Any],
    loaded_execution: Any,
    reconstructed_public_result: Any,
    replay_entry: Any,
    execution_ref: TrialReceiptRef,
    sealed_ref: TrialReceiptRef,
) -> None:
    """Bind the reconstructed result to the decision and replayed head transition."""

    stored_public_result = loaded_execution.public_result
    _require(
        stored_public_result is not None
        and reconstructed_public_result.to_public_dict()
        == stored_public_result.to_public_dict()
        == loaded_execution.record.get("public_result"),
        "Phase 6 reconstructed formal public result differs from the trial receipt",
    )
    stored_decision = loaded_execution.decision.to_public_dict()
    replayed_decision = dict(replay_entry.public_decision)
    semantic_decision_fields = {
        "accepted",
        "comparison",
        "gate_statuses",
        "outcome",
        "public_score",
        "reason",
        "surface_classification",
    }
    _require(
        stored_decision == loaded_execution.record.get("decision")
        and all(
            stored_decision.get(field_name) == replayed_decision.get(field_name)
            for field_name in semantic_decision_fields
        )
        and sorted(stored_decision.get("consumed_gates", ()))
        == sorted(replayed_decision.get("consumed_gates", ()))
        and replayed_decision.get("generation_id") == record.get("generation_id")
        and replayed_decision.get("validation_status") == "valid",
        "Phase 6 formal decision differs from acceptance replay",
    )
    round_record_hash = canonical_json_hash(loaded_execution.record)
    _require(
        formal.get("selected_trial_id") == trial.get("trial_id")
        and formal.get("selected_lane_id") == trial.get("lane_id")
        and formal.get("trial_execution_receipt_checksum")
        == execution_ref.checksum
        == trial.get("execution_receipt_checksum")
        and formal.get("sealed_phase_receipt_checksum") == sealed_ref.checksum
        and formal.get("round_record_hash")
        == round_record_hash
        == trial.get("formal_record_hash")
        and formal.get("accepted") is (stored_decision.get("accepted") is True)
        and formal.get("decision_outcome") == stored_decision.get("outcome"),
        "Phase 6 formal projection differs from the reconstructed trial",
    )
    _require(
        record.get("parent_current_best")
        == replay_entry.current_best_before.to_public_dict()
        and record.get("resulting_current_best")
        == replay_entry.current_best_after.to_public_dict(),
        "Phase 6 formal decision is not bound to the replayed current-best transition",
    )


def _reassemble_formal_generation(
    *,
    output_dir: Path,
    record: Mapping[str, Any],
    resolved_cell_manifest: Any,
    config: StairClimbingTaskConfig,
    adapter: StairPhasedEvaluationAdapter,
    guard: StairGenerationGuardReference,
    screens: Sequence[Any],
    selection: Any,
    replay_entry: Any,
) -> dict[str, Any]:
    """Reassemble one formal generation from train plus three phase receipts."""

    formal = record.get("formal")
    _require(isinstance(formal, Mapping), "Phase 6 formal projection is absent")
    selected_trial_id = selection.selected_trial_id
    _require(
        isinstance(selected_trial_id, str) and selected_trial_id,
        "Phase 6 formal generation has no selected trial",
    )
    trials = record.get("trials")
    _require(isinstance(trials, list), "Phase 6 formal generation trials are absent")
    matches = [
        trial
        for trial in trials
        if isinstance(trial, Mapping) and trial.get("trial_id") == selected_trial_id
    ]
    _require(len(matches) == 1, "Phase 6 formal selected trial is not unique")
    trial = matches[0]
    lane_id = str(trial.get("lane_id"))
    train_ref = _private_trial_receipt_ref(
        output_dir=output_dir,
        record=record,
        artifact_kind=f"{lane_id}_train_receipt",
        receipt_kind="train",
        require_session_root=True,
    )
    phase_refs = {
        phase: _private_trial_receipt_ref(
            output_dir=output_dir,
            record=record,
            artifact_kind=f"{lane_id}_{phase.value}_phase_receipt",
            receipt_kind="eval_phase",
            require_session_root=True,
        )
        for phase in EvaluationPhase
    }
    execution_ref = _private_trial_receipt_ref(
        output_dir=output_dir,
        record=record,
        artifact_kind=f"{lane_id}_trial_execution_receipt",
        receipt_kind="execution",
        require_session_root=True,
    )
    train_loaded = load_trial_train_receipt(train_ref)
    phase_loaded = {
        phase: load_trial_eval_phase_receipt(ref)
        for phase, ref in phase_refs.items()
    }
    _contract, loaded_execution = load_trial_execution_receipt(execution_ref)
    _require(
        train_loaded.result == loaded_execution.train_result
        and loaded_execution.mechanism_operation_delta
        == trial.get("mechanism_operation_delta"),
        "Phase 6 formal execution differs from its train or mechanism evidence",
    )
    for phase, projection_key in (
        (EvaluationPhase.PUBLIC_DEV, "public_dev"),
        (EvaluationPhase.PUBLIC_ACCEPT, "public_accept"),
    ):
        projection = trial.get(projection_key)
        _require(
            isinstance(projection, Mapping)
            and phase_refs[phase].checksum == projection.get("phase_receipt_checksum")
            and phase_loaded[phase].job_spec_hash == projection.get("job_spec_hash")
            and canonical_json_hash(phase_loaded[phase].result.to_private_dict())
            == projection.get("phase_result_hash"),
            f"Phase 6 formal {phase.value} projection differs from its receipt",
        )
    _require(
        phase_refs[EvaluationPhase.SEALED_VALIDATION].checksum
        == formal.get("sealed_phase_receipt_checksum"),
        "Phase 6 formal sealed receipt differs from its projection",
    )
    jobs = loaded_execution.record.get("jobs")
    stored_plan = jobs.get("eval") if isinstance(jobs, Mapping) else None
    _require(isinstance(stored_plan, Mapping), "Phase 6 formal eval plan is absent")
    phase_results = tuple(phase_loaded[phase].result for phase in EvaluationPhase)
    eval_plan = _rebuild_formal_eval_plan(
        stored_plan=stored_plan,
        train_result=train_loaded.result,
        phase_results=phase_results,
    )
    assembly = adapter.assemble_full_summary(
        eval_plan=eval_plan,
        trial_id=selected_trial_id,
        selection_result=selection,
        selection_screens=screens,
        guard_reference=guard.evidence,
        phase_results=phase_results,
        phase_receipt_refs={
            phase: phase_refs[phase].to_dict() for phase in EvaluationPhase
        },
    )
    raw_result = SubprocessRunner(
        allowed_executables=("phase6-verifier-unused",)
    ).project_assembled_eval_summary(eval_plan, assembly)
    task_plugin = StairClimbingTaskPlugin(
        config,
        resolved_cell_manifest=resolved_cell_manifest,
    )
    mask_provider = adapter.objective_family_mask_provider
    mask = mask_provider() if mask_provider is not None else None
    _require(isinstance(mask, Mapping), "Phase 6 formal replay lacks its frozen mask")
    task_plugin.bind_objective_family_mask(mask)
    projected = task_plugin.project_candidate_public_result(
        public_result=raw_result,
        train_summary=train_loaded.result.summary,
        mechanism_operation_delta=trial.get("mechanism_operation_delta"),
    )
    projected = attach_eval_protocol_hash(
        projected,
        eval_protocol_hash=eval_plan.evaluation_protocol().protocol_hash,
    )
    reconstructed = attach_evaluation_binding(
        projected,
        binding=eval_plan.evaluation_binding,
        subject=eval_plan.evaluation_subject(),
        subject_manifest_hash=eval_plan.subject_manifest_hash,
        program_source_checksum=eval_plan.program_source_checksum,
        training_binding_hash=eval_plan.training_binding_hash,
        capability_profile_hash=eval_plan.capability_profile_hash,
    )
    _assert_reassembled_formal_chain(
        record=record,
        trial=trial,
        formal=formal,
        loaded_execution=loaded_execution,
        reconstructed_public_result=reconstructed,
        replay_entry=replay_entry,
        execution_ref=execution_ref,
        sealed_ref=phase_refs[EvaluationPhase.SEALED_VALIDATION],
    )
    return {
        "generation_id": record.get("generation_id"),
        "selected_trial_id": selected_trial_id,
        "public_result_hash": canonical_json_hash(reconstructed.to_public_dict()),
        "decision_hash": canonical_json_hash(loaded_execution.decision.to_public_dict()),
        "current_best_after_hash": canonical_json_hash(
            replay_entry.current_best_after.to_public_dict()
        ),
        "phase_receipt_checksums": {
            phase.value: phase_refs[phase].checksum for phase in EvaluationPhase
        },
    }


def _replay_generation_selector_chain(
    *,
    output_dir: Path,
    genesis: SessionGenesisArtifact,
    config: StairClimbingTaskConfig,
    initial_parent: CurrentBestEvalStats,
    ledger: Sequence[Mapping[str, Any]],
    replay_entries: Sequence[Any],
    generation_manifests: Sequence[Any],
    resolved_cell_manifest: Any,
) -> dict[str, Any]:
    """Rebuild every Stair selector input and move the guard only on acceptance."""

    _require(
        len(ledger) == len(replay_entries) == len(generation_manifests),
        "Phase 6 selector replay evidence length mismatch",
    )
    adapter, current_guard = _initial_stair_guard_reference(
        genesis=genesis,
        parent=initial_parent,
        config=config,
    )
    current_parent = initial_parent
    selection_count = 0
    screen_count = 0
    guard_hashes: list[str] = []
    selection_hashes: list[str] = []
    formal_reassemblies: list[dict[str, Any]] = []
    for record, replay_entry, manifest in zip(
        ledger,
        replay_entries,
        generation_manifests,
        strict=True,
    ):
        _require(
            replay_entry.current_best_before == current_parent,
            "Phase 6 selector replay parent differs from acceptance replay",
        )
        current_guard.assert_matches_parent(current_parent)
        observed_guard_ref = _assert_generation_guard_reference(
            output_dir=output_dir,
            record=record,
            manifest=manifest,
            guard=current_guard,
        )
        guard_hashes.append(current_guard.evidence.evidence_hash)
        generation_id = str(record.get("generation_id"))
        journal = GenerationReceiptJournal(
            output_dir / "generation_receipts" / generation_id,
            manifest=manifest,
        )
        loaded_selection = journal.load_selection()
        stored_selection = record.get("selection")
        screens: Sequence[Any] = ()
        rebuilt_selection: Any | None = None
        if stored_selection is None:
            _require(
                loaded_selection is None
                and record.get("selection_receipt_checksum") is None,
                "Phase 6 selection receipt exists without a selector decision",
            )
        else:
            _require(
                isinstance(stored_selection, Mapping) and loaded_selection is not None,
                "Phase 6 selector decision lacks its receipt",
            )
            _assert_selection_receipt_binding(
                output_dir=output_dir,
                record=record,
                loaded=loaded_selection,
            )
            trials = record.get("trials")
            _require(isinstance(trials, list), "Phase 6 selector trials are absent")
            eligible_trials = [
                trial
                for trial in trials
                if isinstance(trial, Mapping)
                and trial.get("train") is not None
                and trial.get("public_dev") is not None
            ]
            _require(
                len(eligible_trials) == 2,
                "Phase 6 selector receipt does not have two executable candidates",
            )
            screens = sorted(
                (
                    _rebuild_candidate_selector_screen(
                        output_dir=output_dir,
                        record=record,
                        trial=trial,
                        adapter=adapter,
                    )
                    for trial in eligible_trials
                ),
                key=lambda item: item.trial_id,
            )
            rebuilt_selection = select_stair_public_dev_winner(
                config=config,
                screens=screens,
                guard_reference=current_guard.evidence,
            )
            _assert_task_owned_selector_replay(
                loaded_selection=loaded_selection,
                stored_selection=stored_selection,
                screens=screens,
                rebuilt_selection=rebuilt_selection,
                guard=current_guard,
                guard_receipt_ref=observed_guard_ref,
            )
            selection_count += 1
            screen_count += len(screens)
            selection_hashes.append(rebuilt_selection.selection_hash)

        formal = record.get("formal")
        if formal is not None:
            _require(
                rebuilt_selection is not None and bool(screens),
                "Phase 6 formal generation lacks its replayed selector evidence",
            )
            formal_reassemblies.append(
                _reassemble_formal_generation(
                    output_dir=output_dir,
                    record=record,
                    resolved_cell_manifest=resolved_cell_manifest,
                    config=config,
                    adapter=adapter,
                    guard=current_guard,
                    screens=screens,
                    selection=rebuilt_selection,
                    replay_entry=replay_entry,
                )
            )

        next_parent = replay_entry.current_best_after
        if record.get("generation_outcome") == "accepted":
            _require(
                loaded_selection is not None,
                "Phase 6 accepted generation lacks a selector decision",
            )
            current_guard = load_stair_next_generation_guard_reference(
                generation_record=record,
                previous_parent=current_parent,
                acceptance_protocol=config.acceptance_protocol,
                adapter=adapter,
            )
            current_guard.assert_matches_parent(next_parent)
        else:
            _require(
                next_parent == current_parent,
                "Phase 6 non-accepted generation moved the selector guard parent",
            )
        current_parent = next_parent
    current_guard.assert_matches_parent(current_parent)
    return {
        "generation_count": len(ledger),
        "selection_count": selection_count,
        "replayed_candidate_screen_count": screen_count,
        "guard_reference_evidence_hashes": guard_hashes,
        "selection_hashes": selection_hashes,
        "formal_reassembly_count": len(formal_reassemblies),
        "formal_reassemblies": formal_reassemblies,
        "final_guard_parent_hash": canonical_json_hash(current_parent.to_public_dict()),
    }


def _verify_generation_training_initialization(
    *,
    genesis: SessionGenesisArtifact,
    ledger: Sequence[Mapping[str, Any]],
    replay_entries: Sequence[Any],
    generation_manifests: Sequence[Any],
) -> dict[str, Any]:
    """Replay the production first-input hash and bind every available train result."""

    _require(
        len(ledger) == len(replay_entries) == len(generation_manifests),
        "Phase 6 training-initialization evidence length mismatch",
    )
    anchor = genesis.anchor
    _require(
        isinstance(anchor, ReferenceProgramGenesisAnchor),
        "Phase 6 training initialization lacks a reference anchor",
    )
    initial_state_hash = anchor.initial_policy_state_hash
    verified_train_receipts = 0
    manifest_hashes: list[str] = []
    for record, replay_entry, manifest in zip(
        ledger,
        replay_entries,
        generation_manifests,
        strict=True,
    ):
        parent = replay_entry.current_best_before
        training_binding_hash = parent.training_binding_hash
        expected_hash = generation_training_initialization_hash(
            policy_initialization_kind=genesis.policy_initialization_kind,
            training_binding_hash=training_binding_hash,
            expected_initial_policy_state_hash=initial_state_hash,
        )
        _require(
            manifest.training_initialization_hash == expected_hash,
            "Phase 6 generation training initialization differs from replayed parent",
        )
        manifest_hashes.append(expected_hash)
        trials = record.get("trials")
        artifacts = record.get("private_artifacts")
        _require(
            isinstance(trials, list) and isinstance(artifacts, list),
            "Phase 6 generation training evidence is absent",
        )
        for trial in trials:
            if not isinstance(trial, Mapping) or trial.get("train") is None:
                continue
            train_projection = trial.get("train")
            lane_id = trial.get("lane_id")
            _require(
                isinstance(train_projection, Mapping)
                and isinstance(lane_id, str)
                and lane_id,
                "Phase 6 generation train projection is invalid",
            )
            matches = [
                item
                for item in artifacts
                if isinstance(item, Mapping)
                and item.get("kind") == f"{lane_id}_train_receipt"
                and item.get("checksum")
                == train_projection.get("train_receipt_checksum")
            ]
            _require(
                len(matches) == 1,
                "Phase 6 generation train receipt is not uniquely bound",
            )
            item = matches[0]
            loaded = load_trial_train_receipt(
                TrialReceiptRef(
                    kind="train",
                    path=str(item.get("uri")),
                    checksum=str(item.get("checksum")),
                    size_bytes=int(item.get("size_bytes")),
                )
            )
            result = loaded.result
            _require(
                loaded.job_spec_hash == train_projection.get("job_spec_hash")
                and canonical_json_hash(result.to_public_dict())
                == train_projection.get("train_result_hash"),
                "Phase 6 generation train receipt differs from its projection",
            )
            _require(
                result.training_binding_hash == training_binding_hash
                and result.checkpoint.metadata.get("training_binding_hash")
                == training_binding_hash,
                "Phase 6 generation train result used a different training binding",
            )
            pairing = result.initial_policy_state_pairing
            _require(
                pairing is not None
                and pairing.expected_initial_policy_state_hash
                == initial_state_hash
                and pairing.actual_initial_policy_state_hash
                == initial_state_hash
                and pairing.match is True,
                "Phase 6 generation train result initialization pairing differs",
            )
            verified_train_receipts += 1
    _require(
        verified_train_receipts > 0,
        "Phase 6 session has no train receipt initialization evidence",
    )
    return {
        "generation_count": len(generation_manifests),
        "manifest_initialization_hashes": manifest_hashes,
        "verified_train_receipt_count": verified_train_receipts,
    }


def _replay_scientific_session(
    *,
    output_dir: Path,
    entry: Mapping[str, Any],
    manifest: Any,
    genesis: SessionGenesisArtifact,
    session_contract: Any,
    cost_contract: Any,
    continuation_policy: Any,
    ledger: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    execution_receipts: Sequence[Any] | None = None,
    lifecycle: Any | None = None,
    actual_cost_receipts: Sequence[Any] | None = None,
) -> dict[str, Any]:
    expected_count = int(entry["expected_generation_count"])
    _require(len(ledger) == expected_count, "Phase 6 ledger generation count mismatch")
    _require(manifest.cell_variant_id == entry["cell_variant_id"], "Phase 6 robot row mismatch")
    _require(manifest.manifest_hash == entry["resolved_cell_manifest_hash"], "Phase 6 cell manifest mismatch")
    _require(
        manifest.resolved_cell_task_protocol_hash == entry["resolved_cell_task_protocol_hash"],
        "Phase 6 resolved task protocol mismatch",
    )
    _require(
        genesis.session_id == entry["session_id"]
        and genesis.task_id == "unitree_stair_climbing"
        and genesis.policy_initialization_kind == "random"
        and genesis.session_genesis_hash == entry["session_genesis_hash"],
        "Phase 6 Genesis identity mismatch",
    )
    _require(session_contract.contract_hash == entry["generation_session_contract_hash"], "Phase 6 generation contract mismatch")
    _require(
        session_contract.session_id == entry["session_id"]
        and session_contract.session_mode == entry["session_mode"]
        and session_contract.task_id == "unitree_stair_climbing"
        and session_contract.max_generations == expected_count,
        "Phase 6 generation contract identity mismatch",
    )
    _require(cost_contract.contract_hash == entry["cost_contract_hash"], "Phase 6 cost contract mismatch")
    _require(
        cost_contract.session_mode == entry["session_mode"]
        and cost_contract.max_generations == expected_count,
        "Phase 6 cost contract identity mismatch",
    )
    _require(
        session_contract.bound_contract_hashes.get("session_cost") == cost_contract.contract_hash
        and session_contract.bound_contract_hashes.get("continuation_policy") == continuation_policy.policy_hash,
        "Phase 6 bound session contract drift",
    )

    config = StairClimbingTaskConfig.defaults(
        GO2_PROTOCOL,
        resolved_cell_task_protocol_hash=(
            manifest.resolved_cell_task_protocol_hash
        ),
    )
    _require(
        config.task_protocol_hash
        == genesis.evaluation_binding.task_protocol_hash
        == manifest.resolved_cell_task_protocol_hash,
        "Phase 6 task-owned replay config differs from the resolved cell",
    )
    initial = _initial_current_best(
        genesis,
        ledger[0],
        entry,
        acceptance_protocol=config.acceptance_protocol,
    )
    imported_genesis = _verify_imported_reference_genesis(genesis, entry)
    ledger_epoch_hash = primary_ledger_epoch_hash(str(genesis.session_genesis_hash))
    replay = replay_acceptance_ledger(
        ledger,
        initial_current_best=initial,
        expected_session_genesis_hash=str(genesis.session_genesis_hash),
        acceptance_protocol=config.acceptance_protocol,
        expected_initial_policy_state_hash=str(genesis.anchor.initial_policy_state_hash),
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=ledger_epoch_hash,
    )
    replay.raise_for_continuation()
    _require(all(item.valid for item in replay.entries), "Phase 6 acceptance replay is invalid")
    _require(
        [record.get("generation_outcome") for record in ledger]
        == list(entry["expected_generation_outcomes"]),
        "Phase 6 generation outcomes differ from the evidence index",
    )

    refs: list[Any] = []
    generation_manifests: list[Any] = []
    for index, record in enumerate(ledger, start=1):
        generation_id = f"{entry['session_id']}-generation-{index:03d}"
        _require(record.get("generation_id") == generation_id, "Phase 6 generation ID sequence mismatch")
        _verify_generation_manifest_receipt_path(output_dir, record)
        generation_manifest = load_bound_generation_manifest(record)
        generation_manifests.append(generation_manifest)
        if execution_receipts is None:
            loaded = GenerationReceiptJournal(
                output_dir / "generation_receipts" / generation_id,
                manifest=generation_manifest,
            ).load_execution()
            _require(loaded is not None and loaded[0] == record, "Phase 6 execution receipt mismatch")
            refs.append(loaded[1])
    if execution_receipts is None:
        execution_receipts = tuple(refs)
    _require(len(execution_receipts) == expected_count, "Phase 6 execution receipt count mismatch")
    execution_receipt_checksums = [ref.checksum for ref in execution_receipts]

    identity = GenerationLifecycleIdentity(
        session_id=str(entry["session_id"]),
        session_genesis_hash=str(genesis.session_genesis_hash),
        generation_session_contract_hash=session_contract.contract_hash,
        final_confirmation_plan_hash=session_contract.bound_contract_hashes["final_confirmation_plan"],
        ledger_epoch_hash=ledger_epoch_hash,
        output_root=str(output_dir),
    )
    if lifecycle is None:
        lifecycle = audit_generation_lifecycle(
            output_dir=output_dir,
            identity=identity,
            policy=continuation_policy,
            require_closed=True,
        )
    _require(
        lifecycle.terminal
        and lifecycle.active_segment_id is None
        and lifecycle.closed_generation_count == expected_count,
        "Phase 6 lifecycle is not terminal",
    )
    if actual_cost_receipts is None:
        actual_cost_receipts = StairGenerationActualCostJournal(
            output_dir,
            contract=cost_contract,
        ).load_chain()
    _require(bool(actual_cost_receipts), "Phase 6 actual-cost chain is absent")
    cost_head = _replay_session_cost_chain(
        contract=cost_contract,
        actual_cost_receipts=actual_cost_receipts,
        lifecycle=lifecycle,
    )
    _require(cost_head == lifecycle.cost_chain_head_hash, "Phase 6 actual-cost head mismatch")
    final_cost = actual_cost_receipts[-1].payload
    _require(
        isinstance(final_cost, Mapping)
        and isinstance(final_cost.get("cap_checks"), Mapping)
        and final_cost["cap_checks"].get("all_passed") is True,
        "Phase 6 cost caps did not all pass",
    )
    recomputed_cost = _verify_recomputed_actual_cost(
        output_dir=output_dir,
        session_id=str(entry["session_id"]),
        genesis=genesis,
        session_contract=session_contract,
        cost_contract=cost_contract,
        ledger=ledger,
        final_cost_receipt=actual_cost_receipts[-1],
    )

    terminal_raw = report.get("terminal_binding")
    _require(isinstance(terminal_raw, Mapping), "Phase 6 terminal binding is absent")
    terminal = GenerationTerminalBinding(
        identity_hash=str(terminal_raw.get("identity_hash")),
        required_generation_count=int(terminal_raw.get("required_generation_count")),
        closed_generation_count=int(terminal_raw.get("closed_generation_count")),
        ledger_snapshot_hash=str(terminal_raw.get("ledger_snapshot_hash")),
        current_head_hash=str(terminal_raw.get("current_head_hash")),
        lifecycle_head_hash=str(terminal_raw.get("lifecycle_head_hash")),
        cost_chain_head_hash=str(terminal_raw.get("cost_chain_head_hash")),
        segment_count=int(terminal_raw.get("segment_count")),
    )
    _require(
        terminal.identity_hash == identity.identity_hash
        and terminal.required_generation_count == expected_count
        and terminal.closed_generation_count == expected_count
        and terminal.ledger_snapshot_hash == canonical_json_hash(list(ledger))
        and terminal.current_head_hash == canonical_json_hash(replay.current_best.to_public_dict())
        and terminal.lifecycle_head_hash == lifecycle.lifecycle_head_hash
        and terminal.cost_chain_head_hash == cost_head
        and terminal.segment_count == lifecycle.segment_count,
        "Phase 6 terminal binding does not match replay",
    )
    _require(report.get("generation_outcome", ledger[-1].get("generation_outcome")) == ledger[-1].get("generation_outcome"), "Phase 6 terminal report outcome mismatch")

    candidate_count = 0
    evaluated_count = 0
    for record in ledger:
        for trial in record.get("trials", []):
            candidate = trial.get("candidate") if isinstance(trial, Mapping) else None
            if not isinstance(candidate, Mapping):
                continue
            candidate_count += 1
            _require(candidate.get("changed_components") == ["reward_callable"], "Phase 6 candidate exceeded reward-only authority")
            _require(
                isinstance(trial.get("candidate_surface"), Mapping)
                and trial["candidate_surface"].get("classification") == "code_level",
                "Phase 6 candidate did not use the code-level reward surface",
            )
            if trial.get("train") is not None and trial.get("public_dev") is not None:
                evaluated_count += 1
    checkpoint_hashes = _verify_checkpoint_files(output_dir, ledger)
    private_receipts = _verify_candidate_private_receipts(output_dir, ledger)
    selector_replay = _replay_generation_selector_chain(
        output_dir=output_dir,
        genesis=genesis,
        config=config,
        initial_parent=initial,
        ledger=ledger,
        replay_entries=replay.entries,
        generation_manifests=generation_manifests,
        resolved_cell_manifest=manifest,
    )
    training_initialization = _verify_generation_training_initialization(
        genesis=genesis,
        ledger=ledger,
        replay_entries=replay.entries,
        generation_manifests=generation_manifests,
    )
    prompt_privacy = _verify_all_authoring_prompt_privacy(
        output_dir=output_dir,
        genesis=genesis,
        ledger=ledger,
    )
    _require(evaluated_count > 0, "Phase 6 session never evaluated a candidate")
    improvement_observed = replay.current_best != initial
    return {
        "status": "pass",
        "loop_verified": True,
        "improvement_observed": improvement_observed,
        "generation_count": expected_count,
        "generation_outcomes": [record["generation_outcome"] for record in ledger],
        "candidate_count": candidate_count,
        "evaluated_candidate_count": evaluated_count,
        "checkpoint_hashes": list(checkpoint_hashes),
        "generation_execution_receipt_checksums": execution_receipt_checksums,
        "candidate_private_receipts": private_receipts,
        "selector_replay": selector_replay,
        "training_initialization": training_initialization,
        "authoring_prompt_privacy": prompt_privacy,
        "session_genesis_hash": genesis.session_genesis_hash,
        "generation_session_contract_hash": session_contract.contract_hash,
        "cost_contract_hash": cost_contract.contract_hash,
        "lifecycle_head_hash": lifecycle.lifecycle_head_hash,
        "cost_chain_head_hash": cost_head,
        "recomputed_actual_cost": recomputed_cost,
        "initial_current_best_hash": canonical_json_hash(initial.to_public_dict()),
        "final_current_best_hash": canonical_json_hash(replay.current_best.to_public_dict()),
        "imported_reference_genesis": imported_genesis,
        "historical_gpu_idle_and_external_provider_billing": "attested_not_recomputed",
    }


def _verify_pilot(output_dir: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = _absolute_output_dir(output_dir, label="Phase 6 pilot output_dir")
    missing = [
        name
        for name in _PILOT_REQUIRED_FILES
        if (output_dir / name).is_symlink() or not (output_dir / name).is_file()
    ]
    _require(not missing, f"Phase 6 pilot artifacts are missing: {missing}")
    source_checkout_path = output_dir / "generation_source_checkout.json"
    _require(
        not source_checkout_path.exists() and not source_checkout_path.is_symlink(),
        "Phase 6 legacy pilot unexpectedly contains a typed source-checkout receipt",
    )
    report = load_strict_json_object(
        output_dir / _PILOT_REPORT_FILENAME,
        context="Phase 6 pilot report",
        require_canonical=True,
        trailing_newline=True,
    )
    _require(
        report.get("schema_version") == STAIR_GENERATION_REPORT_SCHEMA_VERSION
        and report.get("session_mode") == entry["session_mode"]
        and report.get("session_id") == entry["session_id"]
        and report.get("source_commit") == entry["source_commit"]
        and report.get("output_dir") == str(output_dir)
        and report.get("acceptance_authority") is False,
        "Phase 6 pilot report identity mismatch",
    )
    _require(_git_is_ancestor(str(entry["source_commit"])), "Phase 6 pilot source commit is not an ancestor of the verifier checkout")
    manifest = load_resolved_cell_manifest(output_dir / "resolved_cell_manifest.json", expected_task_id="unitree_stair_climbing")
    genesis = SessionGenesisArtifact.from_mapping(
        load_strict_json_object(output_dir / "session_genesis.json", context="Phase 6 pilot Genesis")
    )
    contract_ref = genesis.generation_session_contract_ref
    _require(contract_ref is not None, "Phase 6 pilot Genesis lacks generation contract")
    session_contract = load_generation_session_contract(output_dir, contract_ref)
    cost_contract = load_stair_generation_session_cost_contract(output_dir / "generation_session_cost.json")
    continuation_policy = load_generation_continuation_policy(output_dir)
    ledger = load_ledger(output_dir / "ledger.jsonl")
    result = _replay_scientific_session(
        output_dir=output_dir,
        entry=entry,
        manifest=manifest,
        genesis=genesis,
        session_contract=session_contract,
        cost_contract=cost_contract,
        continuation_policy=continuation_policy,
        ledger=ledger,
        report=report,
    )
    _verify_pilot_execution_checksum(report=report, replay=result, entry=entry)
    result["source_provenance"] = entry["source_provenance"]
    result["checkout_identity_verified"] = False
    return result


def _verify_pilot_execution_checksum(
    *,
    report: Mapping[str, Any],
    replay: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    receipt = report.get("generation_execution_receipt")
    _require(
        isinstance(receipt, Mapping)
        and receipt.get("checksum") == entry["generation_execution_receipt_checksum"]
        and replay["generation_execution_receipt_checksums"]
        == [entry["generation_execution_receipt_checksum"]],
        "Phase 6 pilot terminal execution receipt mismatch",
    )


def _verify_validation(output_dir: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    from true_rl_agent.tasks.stair_climbing.generation_session import (
        load_completed_stair_generation_session,
    )

    output_dir = _absolute_output_dir(output_dir, label="Phase 6 validation output_dir")
    completed = load_completed_stair_generation_session(
        output_dir=output_dir,
        expected_session_id=str(entry["session_id"]),
        expected_cell_variant_id=str(entry["cell_variant_id"]),
        expected_source_commit=str(entry["source_commit"]),
    )
    _require(
        type(completed.cost_contract) is StairGenerationSessionCostContractV4,
        "Phase 6 imported-reference validation must use exact cost contract v4",
    )
    _require(
        completed.cost_contract.candidate_train_process_timeout_s == 7_200,
        "Phase 6 candidate-train timeout differs from the frozen protocol",
    )
    _require(
        completed.cost_contract.termination_cleanup_margin_s == 16,
        "Phase 6 candidate-train cleanup margin differs from the frozen protocol",
    )
    result = _replay_scientific_session(
        output_dir=completed.output_dir,
        entry=entry,
        manifest=completed.resolved_cell_manifest,
        genesis=completed.session_genesis,
        session_contract=completed.session_contract,
        cost_contract=completed.cost_contract,
        continuation_policy=completed.continuation_policy,
        ledger=completed.ledger,
        report=completed.report,
        execution_receipts=completed.generation_execution_receipts,
        lifecycle=completed.lifecycle,
        actual_cost_receipts=completed.actual_cost_receipts,
    )
    result["source_checkout_hash"] = completed.source_checkout["source_checkout_hash"]
    result["source_provenance"] = entry["source_provenance"]
    result["checkout_identity_verified"] = True
    return result


def verify(
    *,
    pilot_output_dir: Path,
    validation_output_dir: Path | None = None,
    evidence_index_path: Path = EVIDENCE_INDEX,
    require_go: bool = False,
) -> dict[str, Any]:
    index = _load_index(evidence_index_path)
    pilot_entry = index["pilot"]
    phase5 = _verify_phase5(pilot_entry)
    pilot = _verify_pilot(pilot_output_dir, pilot_entry)
    validation_entry = index.get("validation")
    if validation_entry is None:
        _require(validation_output_dir is None, "unindexed validation output cannot become Phase 6 evidence")
        validation = {"status": "not_available", "reason": "fresh_validation_not_indexed"}
    elif validation_output_dir is None:
        validation = {"status": "not_checked", "reason": "validation_output_dir_not_supplied"}
    else:
        _require(
            validation_output_dir.resolve(strict=False) != pilot_output_dir.resolve(strict=False),
            "Phase 6 pilot and validation output roots must differ",
        )
        _require(
            _git_is_ancestor(str(pilot_entry["source_commit"]), str(validation_entry["source_commit"])),
            "Phase 6 pilot source is not an ancestor of validation source",
        )
        _require(
            _git_is_ancestor(str(validation_entry["source_commit"])),
            "Phase 6 validation source commit is not an ancestor of the verifier checkout",
        )
        validation = _verify_validation(validation_output_dir, validation_entry)
        _require(
            validation["session_genesis_hash"] != pilot["session_genesis_hash"],
            "Phase 6 validation reused pilot Genesis",
        )
    phase6_go = bool(
        phase5.get("status") == "go"
        and pilot.get("loop_verified") is True
        and validation.get("loop_verified") is True
        and validation.get("generation_count") == 3
    )
    if require_go:
        _require(
            phase6_go,
            "Phase 6 GO requires indexed fresh validation_3 closure; pilot-only evidence is insufficient",
        )
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "status": "pass",
        "phase6_go": phase6_go,
        "loop_verified": validation.get("loop_verified") is True,
        "improvement_observed": (
            validation.get("improvement_observed")
            if validation.get("loop_verified") is True
            else None
        ),
        "evidence_index_hash": index["index_hash"],
        "phase5": phase5,
        "pilot": pilot,
        "validation": validation,
        "excluded_diagnostics": index["excluded_diagnostics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay Phase 6 Go2 unified Stair LOOP evidence")
    parser.add_argument("--pilot-output-dir", type=Path, required=True)
    parser.add_argument("--validation-output-dir", type=Path)
    parser.add_argument("--evidence-index", type=Path, default=EVIDENCE_INDEX)
    parser.add_argument("--require-go", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                pilot_output_dir=args.pilot_output_dir,
                validation_output_dir=args.validation_output_dir,
                evidence_index_path=args.evidence_index,
                require_go=args.require_go,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
