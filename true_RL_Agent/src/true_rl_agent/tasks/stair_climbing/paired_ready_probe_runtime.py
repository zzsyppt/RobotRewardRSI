"""Task-owned runtime for the Stair P6-A2 author-only paired-ready probe.

This module deliberately reuses the production authoring loop, provider, Stair
admission policy, component smoke check, and verifier-owned sibling exclusion.
It owns only the frozen historical-input adapter and the private probe receipts.
It grants no runner, GPU, evaluation, sealed, ledger, or head-transition authority.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.authoring_loop import (
    CandidateAuthoringLoop,
    CandidateAuthoringResult,
    _restore_authoring_continuation,
)
from true_rl_agent.agent_core.openai_compatible_provider import (
    JsonTransport,
    OpenAICompatibleCandidateProvider,
    OpenAICompatibleProviderConfig,
    UrllibJsonTransport,
)
from true_rl_agent.candidates.loader import load_candidate_manifest
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.sibling_exclusion import (
    SingleComponentDeltaComparisonUnavailable,
    SingleComponentSemanticExclusion,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import (
    canonical_json_bytes,
    strict_json_object_loads,
)
from true_rl_agent.memory import ExperimentRecord, PublicPromptBuild
from true_rl_agent.prompt_artifacts import (
    AuthoringEvidenceRef,
    RoundAuthoringJournal,
    authoring_outcome_from_evidence,
    verify_authoring_evidence,
    verify_authoring_candidate_binding,
    verify_prompt_payload,
)
from true_rl_agent.prompting import (
    assert_public_only_prompt_payload,
    build_prompt_payload,
)
from true_rl_agent.prompting.task_sections import (
    PROVIDER_TASK_PROMPT_SECTION_KEYS,
)
from true_rl_agent.tasks.stair_climbing.candidate_admission import (
    StairCandidateAdmissionPolicy,
)
from true_rl_agent.tasks.stair_climbing.component_smoke import (
    StairClimbingComponentSmokeAdapter,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.paired_ready_probe import (
    STAIR_PAIRED_READY_PROBE_CASE_COUNT,
    load_stair_paired_ready_probe_plan,
    validate_stair_paired_ready_probe_plan,
)
from true_rl_agent.tasks.stair_climbing.prompt_adapter import (
    _compact_public_feedback_to_fit_task_sections,
)
from true_rl_agent.training_programs import resolve_candidate_program_identity


STAIR_PAIRED_READY_PROBE_CASE_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.paired_ready_probe_case_result.v2"
)
STAIR_PAIRED_READY_PROBE_RESULT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.paired_ready_probe_result.v2"
)
STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.paired_ready_probe_runtime_correction.v1"
)
STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_KIND = (
    "authoring_evidence_private_ref_projection_v1"
)
STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_CHANGED_PATHS = (
    "scripts/run_stair_paired_ready_probe.py",
    "src/true_rl_agent/tasks/stair_climbing/paired_ready_probe_runtime.py",
    "tests/unit/test_stair_paired_ready_probe.py",
)
STAIR_PAIRED_READY_PROBE_GO_MINIMUM = 10

_MAX_SOURCE_LEDGER_BYTES = 64 * 1024 * 1024
_MAX_SOURCE_PROMPT_BYTES = 512 * 1024
_MAX_SOURCE_ANCHOR_BYTES = 64 * 1024
_MAX_RECEIPT_BYTES = 32 * 1024 * 1024
_MAX_PROVIDER_RESPONSE_BYTES = 512 * 1024
_PROVIDER_URL = "https://api.deepseek.com/chat/completions"
_PROVIDER_TIMEOUT_SECONDS = 300
_RUNTIME_CORRECTION_FILENAME = "runtime_correction.json"
_RUNTIME_CORRECTION_TRIGGER_CASE_ID = "round-001"
_SOURCE_PROMPT_KEYS = (
    "source_catalog",
    "candidate_lineage",
    "source_payloads",
    "exploration_directive",
)
_RESULT_CATEGORIES = (
    "valid_structurally_distinct",
    "equivalent_blocked",
    "equivalent_leak",
    "authoring_invalid",
    "provider_failed",
)
_CASE_RESULT_FIELDS = {
    "schema_version",
    "plan_hash",
    "case_id",
    "round_index",
    "source_round_id",
    "production_exclusion_hash",
    "provider_request_count",
    "provider_request_hash",
    "provider_response_checksum",
    "prompt_payload_ref",
    "authoring_evidence_ref",
    "stop_reason",
    "classification",
    "candidate",
    "sibling_semantic_exclusion_assessment",
    "independent_equivalence",
    "independent_comparison_error",
    "provider_usage",
    "provider_response_receipt",
    "authority",
    "case_result_hash",
}
_ZERO_AUTHORITY = {
    "train_jobs": 0,
    "eval_jobs": 0,
    "sealed_queries": 0,
    "primary_ledger_writes": 0,
    "current_best_transitions": 0,
    "gpu_jobs": 0,
}


@dataclass(frozen=True)
class PreparedStairPairedReadyCase:
    """One frozen source case plus the production objects needed to run lane B."""

    round_index: int
    source_round_id: str
    source_record: Mapping[str, Any]
    prefix_records: tuple[Mapping[str, Any], ...]
    source_prompt_path: Path
    source_prompt_payload: Mapping[str, Any]
    a_candidate: CandidatePackageRef
    exclusion: SingleComponentSemanticExclusion
    prompt_builder: "FrozenStairPromptBuilder"
    provider_request: Mapping[str, Any]
    plan_input: Mapping[str, Any]


@dataclass(frozen=True)
class _VerifiedRuntimeCorrection:
    """Correction receipt already matched to live files and the frozen plan."""

    payload: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


class FrozenStairPromptBuilder:
    """Rebuild one saved Stair base prompt through today's production assembler.

    The historical public evidence remains frozen.  The current candidate
    capability profile and verifier rules are re-applied by ``build_prompt_payload``.
    Only candidate-draft frames are accepted; repair/source-followup behavior is
    intentionally outside this one-call probe.
    """

    def __init__(
        self,
        *,
        source_prompt_payload: Mapping[str, Any],
        task_config: StairClimbingTaskConfig,
        candidate_capability_profile: Any,
    ) -> None:
        source = _json_copy(source_prompt_payload)
        assert_public_only_prompt_payload(source)
        frame = _mapping(source.get("authoring_context_frame"), "source authoring frame")
        if frame.get("frame_kind") != "candidate_draft":
            raise ValueError("paired-ready source prompt must be a candidate draft")
        self._source = source
        self._task_config = task_config
        self._candidate_capability_profile = candidate_capability_profile
        self._public_context = _frozen_public_context(
            source,
            task_config=task_config,
        )

    def build_prompt(
        self,
        records: Sequence[ExperimentRecord | Mapping[str, Any]],
        *,
        latest_failed_candidate: Any = None,
        candidate_repair_feedback: Mapping[str, Any] | None = None,
        source_payloads: Sequence[Mapping[str, Any]] | None = None,
        frame_kind: str | None = None,
        authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None,
        candidate_verifier_prompt_rules: Mapping[str, Any] | None = None,
    ) -> PublicPromptBuild:
        _ = records
        if latest_failed_candidate is not None or candidate_repair_feedback is not None:
            raise ValueError("paired-ready probe forbids repair prompt mutation")
        if source_payloads is not None:
            raise ValueError("paired-ready probe forbids source-followup mutation")
        if frame_kind not in {None, "candidate_draft"}:
            raise ValueError("paired-ready probe only permits candidate-draft prompts")
        if not isinstance(candidate_verifier_prompt_rules, Mapping):
            raise ValueError("paired-ready probe requires current verifier rules")

        rules = _json_copy(candidate_verifier_prompt_rules)
        exclusion = rules.get("single_component_semantic_exclusion")
        if not isinstance(exclusion, Mapping):
            raise ValueError("paired-ready probe requires a sibling exclusion")
        base_rules = dict(rules)
        base_rules.pop("single_component_semantic_exclusion", None)

        base_payload = self._assemble(
            rules=base_rules,
            authoring_capability_readiness=authoring_capability_readiness,
        )
        provider_payload = self._assemble(
            rules=rules,
            authoring_capability_readiness=authoring_capability_readiness,
        )
        changed = {
            key
            for key in set(base_payload) | set(provider_payload)
            if base_payload.get(key) != provider_payload.get(key)
        }
        if changed != {"required_candidate_package", "authoring_context_frame"}:
            raise ValueError(
                "paired-ready dynamic A-to-B flow changed unexpected prompt sections"
            )
        if _count_equal_mappings(provider_payload, exclusion) != 1:
            raise ValueError("paired-ready exclusion must appear exactly once")
        if "sibling_candidate_exclusion" in provider_payload:
            raise ValueError("legacy top-level sibling exclusion is forbidden")

        lineage = _mapping(
            provider_payload.get("candidate_lineage"),
            "paired-ready candidate lineage",
        )
        parent_checksum = lineage.get("parent_program_source_checksum")
        if not isinstance(parent_checksum, str) or not parent_checksum:
            raise ValueError("paired-ready prompt has no parent source checksum")
        return PublicPromptBuild(
            round_public_context=_json_copy(self._public_context),
            provider_payload=provider_payload,
            parent_program_source_checksum=parent_checksum,
        )

    def _assemble(
        self,
        *,
        rules: Mapping[str, Any],
        authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        payload = build_prompt_payload(
            public_context=_json_copy(self._public_context),
            frame_kind="candidate_draft",
            authoring_capability_readiness=authoring_capability_readiness,
            candidate_capability_profile=self._candidate_capability_profile,
            candidate_verifier_prompt_rules=rules,
        )
        assert_public_only_prompt_payload(payload)
        return payload


@dataclass
class RecordingProbeTransport:
    """Bind the exact wire request and create-once raw response for one case."""

    delegate: JsonTransport
    request_path: Path
    response_path: Path
    expected_request: Mapping[str, Any]
    call_count: int = 0

    def __post_init__(self) -> None:
        if self.response_path.exists() and not self.request_path.exists():
            raise ValueError("paired-ready response exists without its request")
        if not self.request_path.exists():
            return
        payload = read_bounded_regular_file(
            self.request_path,
            max_bytes=2 * 1024 * 1024,
            label="paired-ready existing provider request",
        )
        existing = strict_json_object_loads(
            payload,
            context="paired-ready existing provider request",
        )
        if payload != canonical_json_bytes(existing, trailing_newline=True):
            raise ValueError("paired-ready existing provider request is not canonical")
        if existing != dict(self.expected_request):
            raise ValueError("paired-ready existing provider request differs from plan")
        self.call_count = 1

    @property
    def transport_policy_id(self) -> str:
        value = getattr(self.delegate, "transport_policy_id", None)
        return value if isinstance(value, str) else "unknown"

    @property
    def max_response_bytes(self) -> int:
        value = getattr(self.delegate, "max_response_bytes", None)
        return value if isinstance(value, int) else _MAX_PROVIDER_RESPONSE_BYTES

    def post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        api_key: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        if self.call_count != 0:
            raise RuntimeError("paired-ready case attempted a second provider request")
        if url != _PROVIDER_URL:
            raise ValueError("paired-ready provider URL differs from its frozen plan")
        if timeout_s != _PROVIDER_TIMEOUT_SECONDS:
            raise ValueError("paired-ready provider timeout differs from its frozen plan")
        if dict(payload) != dict(self.expected_request):
            raise ValueError("paired-ready wire request differs from its frozen plan")
        self.request_path.parent.mkdir(parents=True, exist_ok=True)
        create_once_bytes(
            self.request_path,
            canonical_json_bytes(payload, trailing_newline=True),
        )
        self.call_count = 1
        response = self.delegate.post_json(
            url=url,
            payload=payload,
            api_key=api_key,
            timeout_s=timeout_s,
        )
        create_once_bytes(
            self.response_path,
            canonical_json_bytes(response, trailing_newline=True),
        )
        return response


def fixed_stair_paired_ready_provider_config(
    *,
    output_dir: Path,
    env_file: Path | None,
) -> OpenAICompatibleProviderConfig:
    """Return the exact provider configuration frozen by the P6-A2 plan."""

    return OpenAICompatibleProviderConfig(
        model="deepseek-v4-flash",
        output_dir=Path(output_dir),
        base_url="https://api.deepseek.com",
        endpoint="/chat/completions",
        api_key_env="DEEPSEEK_API_KEY",
        env_file=env_file,
        max_tokens=32_768,
        timeout_s=_PROVIDER_TIMEOUT_SECONDS,
        max_response_bytes=_MAX_PROVIDER_RESPONSE_BYTES,
        reasoning_mode="enabled",
        reasoning_effort="max",
    )


def prepare_stair_paired_ready_source_cases(
    *,
    source_root: Path,
    task_config: StairClimbingTaskConfig,
    admission_policy: StairCandidateAdmissionPolicy,
    candidate_capability_profile: Any,
    authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[PreparedStairPairedReadyCase, ...]:
    """Resolve the exact fifteen historical prompts and current exclusions."""

    root = Path(source_root).resolve(strict=True)
    records = _load_source_ledger(root / "ledger.jsonl")
    if len(records) != STAIR_PAIRED_READY_PROBE_CASE_COUNT:
        raise ValueError("paired-ready source ledger must contain exactly 15 rounds")
    candidate_inventory = _candidate_inventory(root / "candidates")
    request_config = fixed_stair_paired_ready_provider_config(
        output_dir=root / ".p6a2-request-preflight-not-created",
        env_file=None,
    )
    request_provider = OpenAICompatibleCandidateProvider(request_config)

    prepared: list[PreparedStairPairedReadyCase] = []
    for round_index, record in enumerate(records, start=1):
        round_id = _required_text(record.get("round_id"), "source round_id")
        if not round_id.endswith(f"-round-{round_index:03d}"):
            raise ValueError("paired-ready source ledger round order mismatch")
        prompt_path = (
            root
            / "rounds"
            / round_id
            / "authoring_attempts"
            / "attempt_000001"
            / "prompt_payload.json"
        )
        prompt_bytes = read_bounded_regular_file(
            prompt_path,
            max_bytes=_MAX_SOURCE_PROMPT_BYTES,
            label="paired-ready source prompt",
        )
        source_prompt = strict_json_object_loads(
            prompt_bytes,
            context="paired-ready source prompt",
        )
        assert_public_only_prompt_payload(source_prompt)
        prompt_checksum = _sha256_checksum(prompt_bytes)
        anchor_path = root / "rounds" / round_id / "authoring_anchor.json"
        anchor = strict_json_object_loads(
            read_bounded_regular_file(
                anchor_path,
                max_bytes=_MAX_SOURCE_ANCHOR_BYTES,
                label="paired-ready source authoring anchor",
            ),
            context="paired-ready source authoring anchor",
        )
        anchor_prompt = _mapping(
            anchor.get("prompt_payload"),
            "paired-ready source anchor prompt",
        )
        if anchor_prompt.get("checksum") != prompt_checksum:
            raise ValueError("source first prompt is not the frozen authoring anchor")
        parent_checksum = _required_text(
            record.get("parent_program_source_checksum"),
            "source parent checksum",
        )
        if anchor.get("parent_program_source_checksum") != parent_checksum:
            raise ValueError("source authoring anchor parent checksum mismatch")
        source_lineage = _mapping(
            source_prompt.get("candidate_lineage"),
            "source candidate lineage",
        )
        if source_lineage.get("parent_program_source_checksum") != parent_checksum:
            raise ValueError("source prompt parent checksum mismatch")

        candidate_record = _mapping(record.get("candidate"), "source candidate")
        candidate_id = _required_text(
            candidate_record.get("candidate_id"),
            "source candidate_id",
        )
        candidate_source_checksum = _required_text(
            candidate_record.get("program_source_checksum"),
            "source candidate program checksum",
        )
        candidate_key = (candidate_id, candidate_source_checksum)
        candidate_paths = candidate_inventory.get(candidate_key, ())
        if len(candidate_paths) != 1:
            raise ValueError("source candidate package must resolve uniquely")
        changed = candidate_record.get("changed_components")
        if not isinstance(changed, list) or not all(
            isinstance(item, str) and item for item in changed
        ):
            raise ValueError("source candidate changed_components are invalid")
        a_candidate = CandidatePackageRef(
            candidate_id=candidate_id,
            path=str(candidate_paths[0]),
            change_set=CandidateChangeSet(frozenset(changed)),
        )
        verification = admission_policy.verify(a_candidate)
        if verification.status != "pass":
            raise ValueError("source A candidate no longer passes current admission")
        exclusion = SingleComponentSemanticExclusion.from_verified_candidate(
            candidate=a_candidate,
            prompt_payload=source_prompt,
        )
        builder = FrozenStairPromptBuilder(
            source_prompt_payload=source_prompt,
            task_config=task_config,
            candidate_capability_profile=candidate_capability_profile,
        )
        rules = dict(admission_policy.authoring_prompt_rules())
        rules["single_component_semantic_exclusion"] = dict(
            exclusion.public_projection
        )
        prompt_build = builder.build_prompt(
            records=(),
            authoring_capability_readiness=authoring_capability_readiness,
            candidate_verifier_prompt_rules=rules,
        )
        provider_request = request_provider._chat_completion_request(
            prompt_build.provider_payload
        )
        prefix_records = tuple(records[: round_index - 1])
        artifact_hash = _required_text(
            candidate_record.get("artifact_hash"),
            "source candidate artifact hash",
        )
        plan_input = {
            "round_index": round_index,
            "source_round_id": round_id,
            "prefix_ledger_hash": canonical_json_hash(list(prefix_records)),
            "prefix_ledger_count": len(prefix_records),
            "source_prompt_checksum": prompt_checksum,
            "a_candidate_artifact_checksum": _as_sha256_checksum(artifact_hash),
            "a_candidate_source_checksum": candidate_source_checksum,
            "a_candidate_anchor_checksum": parent_checksum,
            "production_exclusion_projection": dict(exclusion.public_projection),
            "provider_request": provider_request,
        }
        prepared.append(
            PreparedStairPairedReadyCase(
                round_index=round_index,
                source_round_id=round_id,
                source_record=record,
                prefix_records=prefix_records,
                source_prompt_path=prompt_path,
                source_prompt_payload=source_prompt,
                a_candidate=a_candidate,
                exclusion=exclusion,
                prompt_builder=builder,
                provider_request=provider_request,
                plan_input=plan_input,
            )
        )
    return tuple(prepared)


def _create_or_load_runtime_correction(
    *,
    output_dir: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    runtime_correction_identity: Mapping[str, Any] | None,
) -> _VerifiedRuntimeCorrection | None:
    correction_path = output_dir / _RUNTIME_CORRECTION_FILENAME
    if runtime_correction_identity is None:
        if correction_path.exists():
            raise ValueError(
                "paired-ready runtime correction exists without a correction identity"
            )
        return None

    identity = _validate_runtime_correction_identity(
        runtime_correction_identity,
        plan=plan,
    )
    expected_body = _runtime_correction_body(
        output_dir=output_dir,
        plan_path=plan_path,
        plan=plan,
        identity=identity,
    )
    expected = {
        **expected_body,
        "correction_receipt_hash": canonical_json_hash(expected_body),
    }
    if not correction_path.exists():
        _require_initial_runtime_correction_state(output_dir, plan=plan)
        _create_canonical_json(correction_path, expected)
    actual = _load_runtime_correction(correction_path)
    if actual != expected:
        raise ValueError("paired-ready runtime correction differs from observed state")
    _require_contiguous_provider_request_inventory(output_dir, plan=plan)
    return _VerifiedRuntimeCorrection(_json_copy(actual))


def _validate_runtime_correction_identity(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "plan_source_commit",
        "runtime_source_commit",
        "runtime_parent_commit",
        "changed_paths",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("paired-ready runtime correction identity fields mismatch")
    identity = _json_copy(value)
    if identity["plan_source_commit"] != plan.get("source_commit"):
        raise ValueError("paired-ready correction plan source commit mismatch")
    if identity["runtime_parent_commit"] != identity["plan_source_commit"]:
        raise ValueError("paired-ready correction must be a direct child of the plan")
    for field in (
        "plan_source_commit",
        "runtime_source_commit",
        "runtime_parent_commit",
    ):
        _require_git_commit(identity[field], field)
    changed_paths = identity["changed_paths"]
    if changed_paths != list(
        STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_CHANGED_PATHS
    ):
        raise ValueError("paired-ready correction changed-path scope mismatch")
    return identity


def _runtime_correction_body(
    *,
    output_dir: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    trigger_case = next(
        (
            item
            for item in plan["cases"]
            if item.get("case_id") == _RUNTIME_CORRECTION_TRIGGER_CASE_ID
        ),
        None,
    )
    if trigger_case is None:
        raise ValueError("paired-ready correction trigger case is absent from plan")
    case_dir = output_dir / "cases" / _RUNTIME_CORRECTION_TRIGGER_CASE_ID
    terminal = _verify_runtime_correction_trigger_case(
        case_dir=case_dir,
        planned=trigger_case,
    )
    evidence_path = Path(terminal["authoring_evidence_ref"]["uri"])
    artifact_paths = {
        "plan": plan_path,
        "original_run_log": output_dir / "run.log",
        "original_exitcode": output_dir / "run.exitcode",
        "provider_request": case_dir / "provider_request.json",
        "provider_response": case_dir / "provider_response.json",
        "authoring_evidence": evidence_path,
    }
    artifacts = {
        name: _private_artifact_ref(output_dir, path)
        for name, path in artifact_paths.items()
    }
    return {
        "schema_version": STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_SCHEMA_VERSION,
        "correction_kind": STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_KIND,
        "plan_hash": plan["plan_hash"],
        "plan_source_commit": identity["plan_source_commit"],
        "runtime_source_commit": identity["runtime_source_commit"],
        "runtime_parent_commit": identity["runtime_parent_commit"],
        "changed_paths": list(identity["changed_paths"]),
        "trigger_case_id": _RUNTIME_CORRECTION_TRIGGER_CASE_ID,
        "initial_run_exit_code": 1,
        "preserved_artifacts": artifacts,
        "terminal_outcome": terminal,
        "authority": dict(_ZERO_AUTHORITY),
    }


def _require_initial_runtime_correction_state(
    output_dir: Path,
    *,
    plan: Mapping[str, Any],
) -> None:
    if (output_dir / "probe_result.json").exists():
        raise ValueError("paired-ready correction cannot replace a closed result")
    case_results = sorted((output_dir / "cases").glob("*/case_result.json"))
    if case_results:
        raise ValueError("paired-ready correction requires zero existing case results")
    inventory = _require_contiguous_provider_request_inventory(
        output_dir,
        plan=plan,
    )
    if inventory != [_RUNTIME_CORRECTION_TRIGGER_CASE_ID]:
        raise ValueError("paired-ready correction requires only the R01 request")
    exit_payload = read_bounded_regular_file(
        output_dir / "run.exitcode",
        max_bytes=32,
        label="paired-ready original exit code",
    )
    if exit_payload.strip() != b"1":
        raise ValueError("paired-ready correction requires original exit code 1")
    log_payload = read_bounded_regular_file(
        output_dir / "run.log",
        max_bytes=1024 * 1024,
        label="paired-ready original run log",
    )
    if not all(
        marker in log_payload
        for marker in (b"AuthoringEvidenceRef", b"to_public_dict")
    ):
        raise ValueError("paired-ready correction does not match the original failure")


def _verify_runtime_correction_trigger_case(
    *,
    case_dir: Path,
    planned: Mapping[str, Any],
) -> dict[str, Any]:
    request_payload = read_bounded_regular_file(
        case_dir / "provider_request.json",
        max_bytes=2 * 1024 * 1024,
        label="paired-ready preserved provider request",
    )
    request = strict_json_object_loads(
        request_payload,
        context="paired-ready preserved provider request",
    )
    if request_payload != canonical_json_bytes(request, trailing_newline=True):
        raise ValueError("paired-ready preserved provider request is not canonical")
    if (
        request != planned["provider_request"]
        or canonical_json_hash(request) != planned["expected_provider_request_hash"]
    ):
        raise ValueError("paired-ready preserved provider request differs from plan")
    response_payload = read_bounded_regular_file(
        case_dir / "provider_response.json",
        max_bytes=_MAX_PROVIDER_RESPONSE_BYTES,
        label="paired-ready preserved provider response",
    )
    response = strict_json_object_loads(
        response_payload,
        context="paired-ready preserved provider response",
    )
    if response_payload != canonical_json_bytes(response, trailing_newline=True):
        raise ValueError("paired-ready preserved provider response is not canonical")
    evidence_paths = sorted(
        (case_dir / "authoring" / "authoring_evidence").glob("*.json")
    )
    if len(evidence_paths) != 1:
        raise ValueError("paired-ready preserved R01 evidence must be unique")
    evidence_payload = read_bounded_regular_file(
        evidence_paths[0],
        max_bytes=_MAX_RECEIPT_BYTES,
        label="paired-ready preserved authoring evidence",
    )
    evidence_ref = AuthoringEvidenceRef(
        path=str(evidence_paths[0].resolve(strict=True)),
        checksum=_sha256_checksum(evidence_payload),
        size_bytes=len(evidence_payload),
    )
    projected = _project_existing_terminal_authoring_result(
        case_dir=case_dir,
        planned=planned,
        authoring_evidence_ref=evidence_ref.to_private_dict(),
    )
    if (
        projected.stop_reason != "provider_failed"
        or not isinstance(projected.provider_failure, Mapping)
        or projected.provider_failure.get("category") != "json_parse_error"
    ):
        raise ValueError("paired-ready preserved R01 evidence outcome mismatch")
    return {
        "stop_reason": "provider_failed",
        "provider_failure_category": "json_parse_error",
        "authoring_evidence_ref": evidence_ref.to_private_dict(),
        "provider_request_hash": canonical_json_hash(request),
        "provider_response_checksum": _sha256_checksum(response_payload),
    }


def _load_runtime_correction(path: Path) -> dict[str, Any]:
    payload = read_bounded_regular_file(
        path,
        max_bytes=_MAX_RECEIPT_BYTES,
        label="paired-ready runtime correction",
    )
    value = strict_json_object_loads(payload, context="paired-ready runtime correction")
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError("paired-ready runtime correction is not canonical JSON")
    return value


def _project_runtime_correction_trigger_result(
    *,
    case_dir: Path,
    planned: Mapping[str, Any],
    correction: _VerifiedRuntimeCorrection,
) -> CandidateAuthoringResult:
    """Project the preserved terminal R01 without publishing new evidence."""

    if (
        planned.get("case_id") != _RUNTIME_CORRECTION_TRIGGER_CASE_ID
        or correction.get("trigger_case_id")
        != _RUNTIME_CORRECTION_TRIGGER_CASE_ID
    ):
        raise ValueError("paired-ready correction trigger projection mismatch")
    terminal = correction.get("terminal_outcome")
    if not isinstance(terminal, Mapping):
        raise ValueError("paired-ready correction terminal outcome is invalid")
    result = _project_existing_terminal_authoring_result(
        case_dir=case_dir,
        planned=planned,
        authoring_evidence_ref=terminal.get("authoring_evidence_ref"),
    )
    if (
        result.stop_reason != "provider_failed"
        or not isinstance(result.provider_failure, Mapping)
        or result.provider_failure.get("category") != "json_parse_error"
    ):
        raise ValueError("paired-ready correction terminal outcome drifted")
    return result


def _project_existing_terminal_authoring_result(
    *,
    case_dir: Path,
    planned: Mapping[str, Any],
    authoring_evidence_ref: Any,
) -> CandidateAuthoringResult:
    """Rebuild one terminal result from durable state without provider or writes."""

    round_id = f"p6a2-{planned['case_id']}"
    snapshot = RoundAuthoringJournal(
        case_dir / "authoring",
        round_id=round_id,
    ).snapshot()
    continuation = _restore_authoring_continuation(
        snapshot.latest_continuation_state,
        expected_round_id=round_id,
    )
    if continuation is None or continuation.disposition != "terminal":
        raise ValueError("paired-ready existing journal is not terminal")
    if (
        snapshot.reserved_attempt_count != 1
        or len(snapshot.completed_attempt_records) != 1
        or len(snapshot.prompt_payload_refs) != 1
    ):
        raise ValueError("paired-ready existing journal must contain one attempt")
    prompt_ref = continuation.selected_prompt_payload_ref
    prompt_payload = verify_prompt_payload(prompt_ref)
    if canonical_json_hash(prompt_payload) != planned.get(
        "expected_provider_prompt_hash"
    ):
        raise ValueError("paired-ready existing prompt differs from plan")
    evidence_ref = AuthoringEvidenceRef.from_private_mapping(
        authoring_evidence_ref
    )
    evidence = verify_authoring_evidence(
        evidence_ref,
        expected_round_id=round_id,
        expected_selected_prompt=prompt_ref,
    )
    outcome = authoring_outcome_from_evidence(evidence)
    if outcome.get("stop_reason") != continuation.stop_reason:
        raise ValueError("paired-ready existing terminal outcome drifted")
    selected_candidate = continuation.selected_candidate
    if continuation.stop_reason == "provider_failed" and selected_candidate is not None:
        raise ValueError("paired-ready provider failure unexpectedly selected a candidate")
    expected_candidate: Mapping[str, Any] | None = None
    if selected_candidate is not None:
        identity = resolve_candidate_program_identity(
            selected_candidate.path,
            expected_candidate_id=selected_candidate.candidate_id,
        )
        expected_candidate = {
            **selected_candidate.to_public_dict(),
            "subject_manifest_hash": identity.manifest_hash,
            "program_source_checksum": identity.source_checksum,
            "program_source_size_bytes": identity.source_size_bytes,
        }
    verify_authoring_candidate_binding(
        evidence,
        expected_candidate=expected_candidate,
    )
    return CandidateAuthoringResult(
        stop_reason=str(outcome["stop_reason"]),
        prompt_payload_ref=prompt_ref,
        round_public_context=dict(continuation.selected_round_public_context),
        attempt_records=[dict(item) for item in snapshot.completed_attempt_records],
        attempt_prompt_payload_refs=tuple(snapshot.prompt_payload_refs),
        authoring_evidence_ref=evidence_ref,
        parent_program_source_checksum=(
            continuation.selected_parent_program_source_checksum
        ),
        source_request_feedback=outcome["source_request_feedback"],
        candidate=(
            None
            if continuation.stop_reason == "provider_failed"
            else selected_candidate
        ),
        provider_failure=outcome["provider_failure"],
        static_failure=outcome["static_authoring_failure"],
    )


def _verify_existing_case_result(
    *,
    result_path: Path,
    plan_hash: str,
    planned: Mapping[str, Any],
    prepared: PreparedStairPairedReadyCase,
    case_dir: Path,
) -> dict[str, Any]:
    existing = _load_case_result(result_path, plan_hash=plan_hash)
    projected = _project_existing_terminal_authoring_result(
        case_dir=case_dir,
        planned=planned,
        authoring_evidence_ref=existing["authoring_evidence_ref"],
    )
    recomputed = _case_result(
        plan_hash=plan_hash,
        planned=planned,
        prepared=prepared,
        authoring_result=projected,
        case_dir=case_dir,
    )
    if existing != recomputed:
        raise ValueError("paired-ready existing case result differs from journal")
    return existing


def _require_contiguous_provider_request_inventory(
    output_dir: Path,
    *,
    plan: Mapping[str, Any],
) -> list[str]:
    planned_ids = [str(item["case_id"]) for item in plan["cases"]]
    actual_paths = sorted((output_dir / "cases").glob("*/provider_request.json"))
    actual_ids = [path.parent.name for path in actual_paths]
    if any(case_id not in planned_ids for case_id in actual_ids):
        raise ValueError("paired-ready provider request inventory has an unknown case")
    expected_prefix = planned_ids[: len(actual_ids)]
    if actual_ids != expected_prefix:
        raise ValueError("paired-ready provider request inventory is not contiguous")
    return actual_ids


def _private_artifact_ref(output_dir: Path, path: Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("paired-ready correction artifacts cannot be symlinks")
    resolved = source.resolve(strict=True)
    root = output_dir.resolve(strict=True)
    try:
        relative_path = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("paired-ready correction artifact escapes output_dir") from exc
    payload = read_bounded_regular_file(
        resolved,
        max_bytes=_MAX_RECEIPT_BYTES,
        label="paired-ready correction artifact",
    )
    return {
        "relative_path": relative_path.as_posix(),
        "checksum": _sha256_checksum(payload),
        "size_bytes": len(payload),
    }


def run_stair_paired_ready_probe_cases(
    *,
    output_dir: Path,
    plan_path: Path,
    prepared_cases: Sequence[PreparedStairPairedReadyCase],
    admission_policy: StairCandidateAdmissionPolicy,
    component_smoke_adapter: StairClimbingComponentSmokeAdapter,
    authoring_capability_readiness: Mapping[str, Mapping[str, Any]] | None,
    env_file: Path,
    runtime_source_commit: str,
    runtime_correction_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute all planned cases once and publish the fixed-denominator result."""

    target = Path(output_dir).resolve(strict=True)
    plan = load_stair_paired_ready_probe_plan(plan_path)
    if len(prepared_cases) != STAIR_PAIRED_READY_PROBE_CASE_COUNT:
        raise ValueError("paired-ready runtime requires exactly 15 prepared cases")
    plan_cases = plan["cases"]
    if [dict(item.plan_input) for item in prepared_cases] != [
        _plan_case_source_view(item) for item in plan_cases
    ]:
        raise ValueError("paired-ready prepared inputs differ from the frozen plan")
    _require_git_commit(runtime_source_commit, "runtime_source_commit")
    correction = _create_or_load_runtime_correction(
        output_dir=target,
        plan_path=plan_path,
        plan=plan,
        runtime_correction_identity=runtime_correction_identity,
    )
    if correction is None:
        if runtime_source_commit != plan["source_commit"]:
            raise ValueError("paired-ready runtime commit differs without correction")
    elif correction["runtime_source_commit"] != runtime_source_commit:
        raise ValueError("paired-ready correction runtime commit mismatch")
    credential_preflight = OpenAICompatibleCandidateProvider(
        fixed_stair_paired_ready_provider_config(
            output_dir=target / ".credential-preflight-not-created",
            env_file=Path(env_file),
        )
    )
    credential_preflight._api_key()  # exact provider credential resolution, no network

    receipts: list[dict[str, Any]] = []
    for prepared, planned in zip(prepared_cases, plan_cases, strict=True):
        case_dir = target / "cases" / str(planned["case_id"])
        result_path = case_dir / "case_result.json"
        if result_path.exists():
            receipts.append(
                _verify_existing_case_result(
                    result_path=result_path,
                    plan_hash=plan["plan_hash"],
                    planned=planned,
                    prepared=prepared,
                    case_dir=case_dir,
                )
            )
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        if (
            correction is not None
            and planned["case_id"] == _RUNTIME_CORRECTION_TRIGGER_CASE_ID
        ):
            result = _project_runtime_correction_trigger_result(
                case_dir=case_dir,
                planned=planned,
                correction=correction,
            )
        else:
            transport = RecordingProbeTransport(
                delegate=UrllibJsonTransport(
                    max_response_bytes=_MAX_PROVIDER_RESPONSE_BYTES
                ),
                request_path=case_dir / "provider_request.json",
                response_path=case_dir / "provider_response.json",
                expected_request=prepared.provider_request,
            )
            provider = OpenAICompatibleCandidateProvider(
                fixed_stair_paired_ready_provider_config(
                    output_dir=case_dir / "materialized_candidates",
                    env_file=Path(env_file),
                ),
                transport=transport,
            )
            result = CandidateAuthoringLoop(
                context_builder=prepared.prompt_builder,  # structural PublicPromptBuilder
                provider=provider,
                candidate_admission_policy=admission_policy,
                round_dir=case_dir / "authoring",
                max_attempts=1,
                component_smoke_adapter=component_smoke_adapter,
                max_source_requests_per_round=0,
                authoring_capability_readiness=authoring_capability_readiness,
                sibling_semantic_exclusion=prepared.exclusion,
            ).run_round(
                round_id=f"p6a2-{planned['case_id']}",
                records=prepared.prefix_records,
            )
            if transport.call_count != 1:
                raise RuntimeError(
                    "paired-ready case did not consume exactly one request"
                )
            result = _project_existing_terminal_authoring_result(
                case_dir=case_dir,
                planned=planned,
                authoring_evidence_ref=(
                    result.authoring_evidence_ref.to_private_dict()
                ),
            )
        receipt = _case_result(
            plan_hash=plan["plan_hash"],
            planned=planned,
            prepared=prepared,
            authoring_result=result,
            case_dir=case_dir,
        )
        _create_canonical_json(result_path, receipt)
        receipts.append(
            _load_case_result(
                result_path,
                plan_hash=plan["plan_hash"],
            )
        )

    aggregate = build_stair_paired_ready_probe_result(
        plan=plan,
        case_results=receipts,
        runtime_correction=correction,
    )
    result_path = target / "probe_result.json"
    if result_path.exists():
        existing = strict_json_object_loads(
            read_bounded_regular_file(
                result_path,
                max_bytes=_MAX_RECEIPT_BYTES,
                label="paired-ready aggregate result",
            ),
            context="paired-ready aggregate result",
        )
        if existing != aggregate:
            raise FileExistsError("paired-ready aggregate result already differs")
    else:
        _create_canonical_json(result_path, aggregate)
    return aggregate


def build_stair_paired_ready_probe_result(
    *,
    plan: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
    runtime_correction: _VerifiedRuntimeCorrection | None = None,
) -> dict[str, Any]:
    """Build the fixed 10/15 and zero-leak engineering verdict."""

    if len(case_results) != STAIR_PAIRED_READY_PROBE_CASE_COUNT:
        raise ValueError("paired-ready result requires all 15 closed cases")
    validated_plan = validate_stair_paired_ready_probe_plan(plan)
    plan_hash = str(validated_plan["plan_hash"])
    if runtime_correction is not None and not isinstance(
        runtime_correction,
        _VerifiedRuntimeCorrection,
    ):
        raise TypeError("paired-ready aggregate requires a verified correction")
    correction = runtime_correction
    if correction is None:
        runtime_source_commit = str(validated_plan["source_commit"])
        correction_hash = None
    else:
        if (
            correction["plan_hash"] != plan_hash
            or correction["plan_source_commit"]
            != validated_plan["source_commit"]
            or correction["terminal_outcome"]["provider_request_hash"]
            != validated_plan["cases"][0]["expected_provider_request_hash"]
        ):
            raise ValueError("paired-ready correction differs from aggregate plan")
        runtime_source_commit = str(correction["runtime_source_commit"])
        correction_hash = str(correction["correction_receipt_hash"])
    normalized = [
        _validate_case_result(item, plan_hash=plan_hash)
        for item in case_results
    ]
    if [item["round_index"] for item in normalized] != list(
        range(1, STAIR_PAIRED_READY_PROBE_CASE_COUNT + 1)
    ):
        raise ValueError("paired-ready result cases are not ordered rounds 1..15")
    for result, planned in zip(
        normalized,
        validated_plan["cases"],
        strict=True,
    ):
        for result_field, plan_field in (
            ("case_id", "case_id"),
            ("round_index", "round_index"),
            ("source_round_id", "source_round_id"),
            ("production_exclusion_hash", "production_exclusion_hash"),
            ("provider_request_hash", "expected_provider_request_hash"),
        ):
            if result.get(result_field) != planned.get(plan_field):
                raise ValueError(
                    "paired-ready case result differs from its frozen plan: "
                    f"{result_field}"
                )
    if correction is not None:
        trigger = normalized[0]
        correction_terminal = correction["terminal_outcome"]
        if (
            trigger["case_id"] != correction["trigger_case_id"]
            or trigger["classification"] != "provider_failed"
            or trigger["provider_request_hash"]
            != correction_terminal["provider_request_hash"]
            or trigger["provider_response_checksum"]
            != correction_terminal["provider_response_checksum"]
            or trigger["authoring_evidence_ref"]
            != correction_terminal["authoring_evidence_ref"]
        ):
            raise ValueError("paired-ready corrected R01 receipt semantics mismatch")
    counts = {
        category: sum(item["classification"] == category for item in normalized)
        for category in _RESULT_CATEGORIES
    }
    calls = sum(int(item["provider_request_count"]) for item in normalized)
    valid = counts["valid_structurally_distinct"]
    leaks = counts["equivalent_leak"]
    go = (
        valid >= STAIR_PAIRED_READY_PROBE_GO_MINIMUM
        and leaks == 0
        and calls == STAIR_PAIRED_READY_PROBE_CASE_COUNT
    )
    provider_execution_commit_counts = (
        {str(validated_plan["source_commit"]): len(normalized)}
        if correction is None
        else {
            str(validated_plan["source_commit"]): 1,
            runtime_source_commit: len(normalized) - 1,
        }
    )
    if sum(provider_execution_commit_counts.values()) != len(normalized):
        raise ValueError("paired-ready provider execution commit counts mismatch")
    body = {
        "schema_version": STAIR_PAIRED_READY_PROBE_RESULT_SCHEMA_VERSION,
        "plan_hash": plan_hash,
        "plan_source_commit": validated_plan["source_commit"],
        "receipt_projection_commit": runtime_source_commit,
        "runtime_correction_receipt_hash": correction_hash,
        "provider_execution_commit_counts": provider_execution_commit_counts,
        "runtime_correction_trigger_replay_case_count": (
            1 if correction is not None else 0
        ),
        "provider_requests_during_runtime_correction_trigger_replay": 0,
        "case_count": len(normalized),
        "provider_request_count": calls,
        "classification_counts": counts,
        "valid_structurally_distinct_count": valid,
        "equivalent_leak_count": leaks,
        "go_minimum_valid_count": STAIR_PAIRED_READY_PROBE_GO_MINIMUM,
        "go": go,
        "authority": dict(_ZERO_AUTHORITY),
        "case_result_hashes": [item["case_result_hash"] for item in normalized],
    }
    return {**body, "result_hash": canonical_json_hash(body)}


def _case_result(
    *,
    plan_hash: str,
    planned: Mapping[str, Any],
    prepared: PreparedStairPairedReadyCase,
    authoring_result: CandidateAuthoringResult,
    case_dir: Path,
) -> dict[str, Any]:
    prompt_payload = verify_prompt_payload(authoring_result.prompt_payload_ref)
    actual_prompt_hash = canonical_json_hash(prompt_payload)
    if actual_prompt_hash != planned["expected_provider_prompt_hash"]:
        raise ValueError("paired-ready actual prompt differs from plan")
    request = strict_json_object_loads(
        read_bounded_regular_file(
            case_dir / "provider_request.json",
            max_bytes=2 * 1024 * 1024,
            label="paired-ready provider request",
        ),
        context="paired-ready provider request",
    )
    if canonical_json_hash(request) != planned["expected_provider_request_hash"]:
        raise ValueError("paired-ready actual request differs from plan")
    attempt = authoring_result.attempt_records[-1]
    sibling = attempt.get("sibling_semantic_exclusion_assessment")
    independent_equivalence: str | None = None
    comparison_error: str | None = None
    if authoring_result.candidate is not None:
        try:
            independent_equivalence = prepared.exclusion.compare_candidate(
                candidate=authoring_result.candidate,
                candidate_prompt_payload=prompt_payload,
            )
        except SingleComponentDeltaComparisonUnavailable as exc:
            comparison_error = type(exc).__name__

    candidate_ref = None
    if authoring_result.candidate is not None:
        identity = resolve_candidate_program_identity(
            authoring_result.candidate.path,
            expected_candidate_id=authoring_result.candidate.candidate_id,
        )
        candidate_ref = {
            "candidate_id": authoring_result.candidate.candidate_id,
            "relative_path": str(
                Path(authoring_result.candidate.path).resolve(strict=True).relative_to(
                    case_dir.resolve(strict=True)
                )
            ),
            "program_source_checksum": identity.source_checksum,
            "program_source_size_bytes": identity.source_size_bytes,
        }
    response_path = case_dir / "provider_response.json"
    response_checksum = (
        _sha256_checksum(response_path.read_bytes()) if response_path.is_file() else None
    )
    body: dict[str, Any] = {
        "schema_version": STAIR_PAIRED_READY_PROBE_CASE_RESULT_SCHEMA_VERSION,
        "plan_hash": plan_hash,
        "case_id": planned["case_id"],
        "round_index": planned["round_index"],
        "source_round_id": planned["source_round_id"],
        "production_exclusion_hash": planned["production_exclusion_hash"],
        "provider_request_count": 1,
        "provider_request_hash": canonical_json_hash(request),
        "provider_response_checksum": response_checksum,
        "prompt_payload_ref": authoring_result.prompt_payload_ref.to_public_dict(),
        "authoring_evidence_ref": authoring_result.authoring_evidence_ref.to_private_dict(),
        "stop_reason": authoring_result.stop_reason,
        "classification": "authoring_invalid",
        "candidate": candidate_ref,
        "sibling_semantic_exclusion_assessment": (
            _json_copy(sibling) if isinstance(sibling, Mapping) else None
        ),
        "independent_equivalence": independent_equivalence,
        "independent_comparison_error": comparison_error,
        "provider_usage": (
            _json_copy(attempt.get("provider_usage"))
            if isinstance(attempt.get("provider_usage"), Mapping)
            else None
        ),
        "provider_response_receipt": (
            _json_copy(attempt.get("provider_response_receipt"))
            if isinstance(attempt.get("provider_response_receipt"), Mapping)
            else None
        ),
        "authority": dict(_ZERO_AUTHORITY),
    }
    body["classification"] = _derive_case_classification(body)
    return {**body, "case_result_hash": canonical_json_hash(body)}


def _validate_case_result(
    value: Any,
    *,
    plan_hash: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("paired-ready case result must be an object")
    result = _json_copy(value)
    if set(result) != _CASE_RESULT_FIELDS:
        raise ValueError("paired-ready case result fields mismatch")
    if result.get("schema_version") != STAIR_PAIRED_READY_PROBE_CASE_RESULT_SCHEMA_VERSION:
        raise ValueError("paired-ready case result schema mismatch")
    if result.get("plan_hash") != plan_hash:
        raise ValueError("paired-ready case result plan mismatch")
    if result.get("authority") != _ZERO_AUTHORITY:
        raise ValueError("paired-ready case result has runtime authority")
    if result.get("classification") not in _RESULT_CATEGORIES:
        raise ValueError("paired-ready case result classification is invalid")
    if result.get("provider_request_count") != 1:
        raise ValueError("paired-ready case must contain exactly one request")
    try:
        AuthoringEvidenceRef.from_private_mapping(
            result.get("authoring_evidence_ref")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("paired-ready case authoring evidence ref is invalid") from exc
    derived_classification = _derive_case_classification(result)
    if result.get("classification") != derived_classification:
        raise ValueError(
            "paired-ready case result classification disagrees with its evidence"
        )
    declared = result.pop("case_result_hash", None)
    if declared != canonical_json_hash(result):
        raise ValueError("paired-ready case result hash mismatch")
    return {**result, "case_result_hash": declared}


def _load_case_result(
    path: Path,
    *,
    plan_hash: str,
) -> dict[str, Any]:
    payload = read_bounded_regular_file(
        path,
        max_bytes=_MAX_RECEIPT_BYTES,
        label="paired-ready case result",
    )
    value = strict_json_object_loads(payload, context="paired-ready case result")
    if payload != canonical_json_bytes(value, trailing_newline=True):
        raise ValueError("paired-ready case result is not canonical JSON")
    return _validate_case_result(value, plan_hash=plan_hash)


def _derive_case_classification(value: Mapping[str, Any]) -> str:
    stop_reason = value.get("stop_reason")
    candidate = value.get("candidate")
    sibling = value.get("sibling_semantic_exclusion_assessment")
    independent_equivalence = value.get("independent_equivalence")
    comparison_error = value.get("independent_comparison_error")
    if stop_reason == "candidate_verified":
        if not _candidate_result_ref_is_valid(candidate):
            return "authoring_invalid"
        if comparison_error is not None:
            return "authoring_invalid"
        if independent_equivalence is not None:
            return "equivalent_leak"
        if (
            not isinstance(sibling, Mapping)
            or sibling.get("status") != "pass"
            or sibling.get("blocking") is not False
            or sibling.get("equivalence") is not None
            or sibling.get("exclusion_hash")
            != value.get("production_exclusion_hash")
        ):
            return "authoring_invalid"
        if not isinstance(value.get("prompt_payload_ref"), Mapping) or not value.get(
            "prompt_payload_ref"
        ):
            return "authoring_invalid"
        if not isinstance(value.get("authoring_evidence_ref"), Mapping) or not value.get(
            "authoring_evidence_ref"
        ):
            return "authoring_invalid"
        if not isinstance(value.get("provider_response_receipt"), Mapping) or not value.get(
            "provider_response_receipt"
        ):
            return "authoring_invalid"
        response_checksum = value.get("provider_response_checksum")
        if (
            not isinstance(response_checksum, str)
            or not response_checksum.startswith("sha256:")
            or len(response_checksum) != 71
        ):
            return "authoring_invalid"
        return "valid_structurally_distinct"
    if stop_reason == "provider_failed":
        return "provider_failed"
    if (
        isinstance(candidate, Mapping)
        and isinstance(sibling, Mapping)
        and sibling.get("blocking") is True
        and sibling.get("status") == "fail"
        and sibling.get("equivalence")
        in {"identical_or_format_only", "parameter_only_delta"}
    ):
        return "equivalent_blocked"
    return "authoring_invalid"


def _candidate_result_ref_is_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if set(value) != {
        "candidate_id",
        "relative_path",
        "program_source_checksum",
        "program_source_size_bytes",
    }:
        return False
    if not isinstance(value.get("candidate_id"), str) or not value["candidate_id"]:
        return False
    if not isinstance(value.get("relative_path"), str) or not value["relative_path"]:
        return False
    checksum = value.get("program_source_checksum")
    if (
        not isinstance(checksum, str)
        or not checksum.startswith("sha256:")
        or len(checksum) != 71
    ):
        return False
    size = value.get("program_source_size_bytes")
    return isinstance(size, int) and not isinstance(size, bool) and size > 0


def _frozen_public_context(
    source: Mapping[str, Any],
    *,
    task_config: StairClimbingTaskConfig,
) -> dict[str, Any]:
    task_sections = {
        key: copy.deepcopy(source[key])
        for key in PROVIDER_TASK_PROMPT_SECTION_KEYS
        if key in source
    }
    task_sections = _compact_public_feedback_to_fit_task_sections(
        task_sections,
        evaluation_profile=task_config.evaluation_profile_config,
    )
    history = _mapping(source.get("public_history"), "source public history")
    context: dict[str, Any] = {
        "task_prompt_sections": task_sections,
        "previous_rounds": history.get("previous_rounds", 0),
        "best_public_score": history.get("best_public_score"),
        "public_score_source": history.get("public_score_source"),
        "last_outcome": history.get("last_outcome"),
        "pure_knob_streak": history.get("pure_knob_streak", 0),
        "recent_outcomes": copy.deepcopy(history.get("recent_outcomes", [])),
        "public_history_omitted_round_count": history.get("omitted_round_count", 0),
        "experiment_memory": copy.deepcopy(history.get("experiment_memory")),
    }
    for key in _SOURCE_PROMPT_KEYS:
        if key not in source:
            raise ValueError(f"source prompt missing frozen section: {key}")
        context[key] = copy.deepcopy(source[key])
    assert_public_only_prompt_payload(context)
    return context


def _candidate_inventory(
    candidate_root: Path,
) -> dict[tuple[str, str], tuple[Path, ...]]:
    inventory: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(Path(candidate_root).glob("*")):
        if not path.is_dir():
            continue
        manifest = load_candidate_manifest(path)
        ref = CandidatePackageRef(
            candidate_id=manifest.candidate_id,
            path=str(path),
            change_set=CandidateChangeSet(frozenset()),
        )
        identity = resolve_candidate_program_identity(
            ref.path,
            expected_candidate_id=ref.candidate_id,
        )
        inventory.setdefault((manifest.candidate_id, identity.source_checksum), []).append(
            path.resolve(strict=True)
        )
    return {key: tuple(paths) for key, paths in inventory.items()}


def _load_source_ledger(path: Path) -> list[dict[str, Any]]:
    payload = read_bounded_regular_file(
        path,
        max_bytes=_MAX_SOURCE_LEDGER_BYTES,
        label="paired-ready source ledger",
    )
    records: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        records.append(
            strict_json_object_loads(
                line,
                context=f"paired-ready source ledger line {index}",
            )
        )
    return records


def _plan_case_source_view(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _json_copy(case[key])
        for key in (
            "round_index",
            "source_round_id",
            "prefix_ledger_hash",
            "prefix_ledger_count",
            "source_prompt_checksum",
            "a_candidate_artifact_checksum",
            "a_candidate_source_checksum",
            "a_candidate_anchor_checksum",
            "production_exclusion_projection",
            "provider_request",
        )
    }


def _create_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    create_once_bytes(path, canonical_json_bytes(dict(value), trailing_newline=True))


def _sha256_checksum(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _as_sha256_checksum(value: str) -> str:
    return value if value.startswith("sha256:") else f"sha256:{value}"


def _require_git_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"paired-ready {label} is not a Git commit")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _json_copy(value: Any) -> Any:
    return strict_json_object_loads(
        canonical_json_bytes(dict(value)) if isinstance(value, Mapping) else b"{}",
        context="paired-ready JSON object",
    ) if isinstance(value, Mapping) else copy.deepcopy(value)


def _count_equal_mappings(value: Any, expected: Mapping[str, Any]) -> int:
    count = 0
    if isinstance(value, Mapping):
        if dict(value) == dict(expected):
            count += 1
        for item in value.values():
            count += _count_equal_mappings(item, expected)
    elif isinstance(value, list):
        for item in value:
            count += _count_equal_mappings(item, expected)
    return count


__all__ = [
    "FrozenStairPromptBuilder",
    "PreparedStairPairedReadyCase",
    "RecordingProbeTransport",
    "STAIR_PAIRED_READY_PROBE_GO_MINIMUM",
    "STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_KIND",
    "STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_CHANGED_PATHS",
    "STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_SCHEMA_VERSION",
    "fixed_stair_paired_ready_provider_config",
    "prepare_stair_paired_ready_source_cases",
    "run_stair_paired_ready_probe_cases",
]
