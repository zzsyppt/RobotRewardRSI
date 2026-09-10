"""Match GPU process observations to owned generation attempts."""
from __future__ import annotations
import hashlib
import re
from true_rl_agent.runner.gpu_process_binding import is_full_gpu_uuid
import subprocess
from collections.abc import Mapping
from typing import Any
from true_rl_agent.agent_core.generation_foundation import GenerationSessionFoundation
from true_rl_agent.agent_core.generation_contract import GENERATION_LANE_IDS
from true_rl_agent.agent_core.generation_operations import read_operation_slot
from true_rl_agent.agent_core.session_genesis import VerifiedSessionGenesis
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.runner.gpu_lease import GpuLeaseHolder
from true_rl_agent.runner.job_attempts import JobAttemptJournal, PreGenesisExecutionScope
from true_rl_agent.runner.gpu_process_binding import validate_gpu_runtime_summary_evidence

def _resource_summary_evidence(
    foundation: GenerationSessionFoundation,
    *,
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    if not isinstance(include_final_confirmation, bool):
        raise TypeError("resource summary final-confirmation scope must be boolean")
    completed_jobs = _completed_bound_job_attempts(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    prefix_known_jobs = _known_bound_job_attempts(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    all_known_jobs = (
        prefix_known_jobs
        if generation_index is None
        else _known_bound_job_attempts(
            foundation,
            generation_index=foundation.session_contract.max_generations,
            include_final_confirmation=True,
        )
    )
    binding_by_hash = {
        binding.resource_binding_hash: (binding, identity.physical_id)
        for binding, identity in zip(
            foundation.gpu_bindings,
            foundation.gpu_identities,
            strict=True,
        )
    }
    artifact_root = foundation.config.output_dir / "runner"
    if completed_jobs and not artifact_root.is_dir():
        raise ValueError("completed bound jobs have no runner artifact root")
    projections: list[dict[str, Any]] = []
    seen_jobs: set[str] = set()
    for path in (
        sorted(artifact_root.rglob("summary.json")) if artifact_root.exists() else []
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError("runner summary path is not a regular file")
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(
                foundation.config.output_dir.resolve(strict=True)
            )
        except ValueError as exc:
            raise ValueError("runner summary escaped the session output") from exc
        raw = resolved.read_bytes()
        if not raw or len(raw) > 64 * 1024 * 1024:
            raise ValueError("runner summary size is invalid")
        summary = load_strict_json_object(
            resolved,
            context="bound runner summary",
        )
        if summary.get("gpu_process_binding") is None:
            continue
        job_id = summary.get("job_id")
        if summary.get("status") != "pass":
            if job_id in completed_jobs:
                raise ValueError("completed bound job has a non-passing summary")
            continue
        if not isinstance(job_id, str):
            raise TypeError("passing bound summary job_id must be a string")
        if job_id not in completed_jobs:
            if job_id in all_known_jobs and job_id not in prefix_known_jobs:
                future_expected = binding_by_hash.get(all_known_jobs[job_id])
                if future_expected is None:
                    raise ValueError("future bound job used an unknown physical GPU")
                future_binding, future_physical_uuid = future_expected
                validate_gpu_runtime_summary_evidence(
                    summary,
                    expected_binding=future_binding,
                    expected_physical_gpu_uuid=future_physical_uuid,
                    require_compute_context_observation=True,
                )
                continue
            raise ValueError("passing bound summary has no completed job attempt")
        if job_id in seen_jobs:
            raise ValueError("bound job has duplicate passing summaries")
        resource_hash = completed_jobs[job_id]
        expected = binding_by_hash.get(resource_hash)
        if expected is None:
            raise ValueError("bound job used an unknown physical GPU")
        binding, physical_uuid = expected
        validated = validate_gpu_runtime_summary_evidence(
            summary,
            expected_binding=binding,
            expected_physical_gpu_uuid=physical_uuid,
            require_compute_context_observation=True,
        )
        process = validated["owned_process_identity"]
        runtime = validated["gpu_runtime_evidence"]
        projections.append(
            {
                "relative_path": relative.as_posix(),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "job_id": job_id,
                "process_id": process["process_id"],
                "process_group_id": process["process_group_id"],
                "gpu_process_binding": validated["gpu_process_binding"],
                "external_compute_context_observation": runtime[
                    "external_compute_context_observation"
                ],
            }
        )
        seen_jobs.add(job_id)
    if seen_jobs != set(completed_jobs):
        raise ValueError("completed bound jobs lack unique resource summaries")
    return projections, len(completed_jobs)



def _completed_bound_job_attempts(
    foundation: GenerationSessionFoundation,
    *,
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> dict[str, str]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    root = foundation.config.output_dir / "job_attempts"
    completed: dict[str, str] = {}
    if not root.exists():
        return completed
    for path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        if journal.resource_binding_hash is None:
            continue
        if (
            journal.scope.budget_contract_hash != foundation.cost_contract.contract_hash
            or journal.scope.session_contract_hash
            != foundation.session_contract.contract_hash
        ):
            raise ValueError("resource attempt belongs to another session")
        if not _attempt_is_in_generation_prefix(
            foundation,
            journal=journal,
            reservation=reservation.payload,
            generation_index=generation_index,
            include_final_confirmation=include_final_confirmation,
        ):
            continue
        result_path = path.with_name(path.name.replace(".reservation.", ".result."))
        if not result_path.exists():
            continue
        result = journal.load_result(reservation.payload["attempt_index"])
        if result.payload.get("outcome") != "completed":
            continue
        if journal.logical_job_id in completed:
            raise ValueError("bound logical job completed more than once")
        completed[journal.logical_job_id] = journal.resource_binding_hash
    return completed



def _known_bound_job_attempts(
    foundation: GenerationSessionFoundation,
    *,
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> dict[str, str]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    root = foundation.config.output_dir / "job_attempts"
    known: dict[str, str] = {}
    if not root.exists():
        return known
    for path in sorted(root.glob("*/attempt_*.reservation.json")):
        journal, reservation = JobAttemptJournal.from_reservation_path(path)
        if journal.resource_binding_hash is None:
            continue
        if (
            journal.scope.budget_contract_hash != foundation.cost_contract.contract_hash
            or journal.scope.session_contract_hash
            != foundation.session_contract.contract_hash
        ):
            raise ValueError("resource attempt belongs to another session")
        if not _attempt_is_in_generation_prefix(
            foundation,
            journal=journal,
            reservation=reservation.payload,
            generation_index=generation_index,
            include_final_confirmation=include_final_confirmation,
        ):
            continue
        previous = known.get(journal.logical_job_id)
        if previous is not None and previous != journal.resource_binding_hash:
            raise ValueError("bound logical job changed its physical GPU")
        known[journal.logical_job_id] = journal.resource_binding_hash
    return known



def _validate_resource_closure_generation_index(
    foundation: GenerationSessionFoundation,
    generation_index: int | None,
) -> None:
    if generation_index is None:
        return
    if (
        isinstance(generation_index, bool)
        or not isinstance(generation_index, int)
        or generation_index <= 0
        or generation_index > foundation.session_contract.max_generations
    ):
        raise ValueError("strict resource closure generation index is invalid")



def _attempt_is_in_generation_prefix(
    foundation: GenerationSessionFoundation,
    *,
    journal: JobAttemptJournal,
    reservation: Mapping[str, Any],
    generation_index: int | None,
    include_final_confirmation: bool,
) -> bool:
    if not isinstance(include_final_confirmation, bool):
        raise TypeError("attempt final-confirmation scope must be boolean")
    attempt_index = reservation.get("attempt_index")
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
        raise TypeError("bounded generation attempt index must be an integer")
    read_operation_slot(
        session_id=foundation.config.session_id,
        scope=journal.scope,
        logical_job_id=journal.logical_job_id,
        job_kind=journal.job_kind,
        attempt_index=attempt_index,
        cost_contract=foundation.cost_contract,
    )
    if isinstance(journal.scope, PreGenesisExecutionScope):
        return True
    if journal.scope.generation_id == "generation-final-confirmation":
        return include_final_confirmation
    if generation_index is None:
        return True
    if journal.scope.generation_id == "generation-parent-public-dev-bootstrap":
        return True
    match = re.fullmatch(
        re.escape(foundation.config.session_id) + r"-generation-([0-9]{3})",
        journal.scope.generation_id,
    )
    if match is None:
        raise ValueError("strict attempt generation prefix is invalid")
    attempt_generation_index = int(match.group(1))
    if not 1 <= attempt_generation_index <= foundation.session_contract.max_generations:
        raise ValueError("attempt generation exceeds its strict session")
    return attempt_generation_index <= generation_index



def _reacquire_generation_gpu_leases(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    invocation_segment_id: str,
) -> dict[str, Any]:
    leases: list[Any] = []
    try:
        for lane_id, identity in zip(
            GENERATION_LANE_IDS,
            foundation.gpu_identities,
            strict=True,
        ):
            lease = foundation.gpu_lease_manager.acquire(
                identity.execution_device,
                holder=GpuLeaseHolder(
                    session_genesis_hash=genesis.session_genesis_hash,
                    generation_id=(f"{foundation.config.session_id}-resource-closure"),
                    lane_id=lane_id,
                    invocation_segment_id=invocation_segment_id,
                    logical_job_id=f"resource-closure-{lane_id}",
                ),
            )
            lease.assert_owned()
            leases.append(lease)
        return {
            "passed": True,
            "simultaneous": True,
            "resource_binding_hashes": [
                lease.resource_binding_hash for lease in leases
            ],
        }
    finally:
        for lease in reversed(leases):
            lease.release()




def build_resource_closure_evidence(
    foundation: GenerationSessionFoundation,
    *,
    genesis: VerifiedSessionGenesis,
    invocation_segment_id: str,
    startup_observation: Mapping[str, Any],
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> dict[str, Any]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    if not isinstance(include_final_confirmation, bool):
        raise TypeError("resource closure final-confirmation scope must be boolean")
    expected_startup = {
        "schema_version": f"{foundation.cost_contract.operational_evidence_namespace}.generation_compute_idle.v1",
        "stage": "startup",
        "source": "nvidia-smi-query-compute-apps",
        "target_gpu_uuids": [
            foundation.config.gpu0_uuid,
            foundation.config.gpu1_uuid,
        ],
        "rows": [],
    }
    if dict(startup_observation) != expected_startup:
        raise ValueError("generation startup GPU-idle evidence is invalid")
    final_idle = assert_target_gpus_compute_idle(
        foundation.config.nvidia_smi_executable,
        expected=(
            foundation.config.gpu0_uuid,
            foundation.config.gpu1_uuid,
        ),
        stage="final",
        schema_version=expected_startup["schema_version"],
    )
    summaries, completed_bound_job_count = _resource_summary_evidence(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    attempt_chain = _attempt_chain_closure_evidence(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    if attempt_chain["open_reservation_count"] != 0:
        raise ValueError("generation resource closure has open job attempts")
    reacquired = _reacquire_generation_gpu_leases(
        foundation,
        genesis=genesis,
        invocation_segment_id=invocation_segment_id,
    )
    payload = {
        "schema_version": (
            f"{foundation.cost_contract.operational_evidence_namespace}.generation_resource_closure.v1"
            if generation_index is None
            else f"{foundation.cost_contract.operational_evidence_namespace}.generation_resource_closure.v2"
        ),
        "startup_compute_idle": expected_startup,
        "final_compute_idle": final_idle,
        "completed_bound_job_count": completed_bound_job_count,
        "summaries": summaries,
        "attempt_chain": attempt_chain,
        "lease_reacquire": reacquired,
        "acceptance_authority": False,
    }
    if generation_index is not None:
        payload["generation_index"] = generation_index
    return payload



def validate_stored_resource_closure_evidence(
    foundation: GenerationSessionFoundation,
    *,
    resource_evidence: Mapping[str, Any],
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> None:
    _validate_resource_closure_generation_index(foundation, generation_index)
    if not isinstance(include_final_confirmation, bool):
        raise TypeError("resource closure final-confirmation scope must be boolean")
    expected_gpu_uuids = [
        foundation.config.gpu0_uuid,
        foundation.config.gpu1_uuid,
    ]
    expected_startup = {
        "schema_version": f"{foundation.cost_contract.operational_evidence_namespace}.generation_compute_idle.v1",
        "stage": "startup",
        "source": "nvidia-smi-query-compute-apps",
        "target_gpu_uuids": expected_gpu_uuids,
        "rows": [],
    }
    expected_final = {
        **expected_startup,
        "stage": "final",
    }
    expected_schema = (
        f"{foundation.cost_contract.operational_evidence_namespace}.generation_resource_closure.v1"
        if generation_index is None
        else f"{foundation.cost_contract.operational_evidence_namespace}.generation_resource_closure.v2"
    )
    if resource_evidence.get("schema_version") != expected_schema:
        raise ValueError("stored bounded generation resource closure schema mismatch")
    if generation_index is not None:
        if set(resource_evidence) != {
            "schema_version",
            "generation_index",
            "startup_compute_idle",
            "final_compute_idle",
            "completed_bound_job_count",
            "summaries",
            "attempt_chain",
            "lease_reacquire",
            "acceptance_authority",
        }:
            raise ValueError("stored bounded generation resource closure fields mismatch")
        if resource_evidence.get("generation_index") != generation_index:
            raise ValueError("stored bounded generation resource closure prefix mismatch")
    if resource_evidence.get("startup_compute_idle") != expected_startup:
        raise ValueError("stored bounded generation startup GPU evidence mismatch")
    if resource_evidence.get("final_compute_idle") != expected_final:
        raise ValueError("stored bounded generation final GPU evidence mismatch")
    summaries, completed_count = _resource_summary_evidence(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    )
    if (
        resource_evidence.get("summaries") != summaries
        or resource_evidence.get("completed_bound_job_count") != completed_count
    ):
        raise ValueError("stored bounded generation resource summaries drifted")
    if resource_evidence.get("attempt_chain") != _attempt_chain_closure_evidence(
        foundation,
        generation_index=generation_index,
        include_final_confirmation=include_final_confirmation,
    ):
        raise ValueError("stored bounded generation attempt-chain closure drifted")
    lease_reacquire = resource_evidence.get("lease_reacquire")
    if not isinstance(lease_reacquire, Mapping) or lease_reacquire != {
        "passed": True,
        "simultaneous": True,
        "resource_binding_hashes": [
            binding.resource_binding_hash for binding in foundation.gpu_bindings
        ],
    }:
        raise ValueError("stored bounded generation lease-reacquire evidence mismatch")
    if resource_evidence.get("acceptance_authority") is not False:
        raise ValueError("stored bounded generation resource evidence gained authority")



def _attempt_chain_closure_evidence(
    foundation: GenerationSessionFoundation,
    *,
    generation_index: int | None = None,
    include_final_confirmation: bool = False,
) -> dict[str, Any]:
    _validate_resource_closure_generation_index(foundation, generation_index)
    root = foundation.config.output_dir / "job_attempts"
    reservation_count = 0
    closed_result_count = 0
    open_reservation_count = 0
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("job-attempt root is invalid")
        for path in sorted(root.glob("*/attempt_*.reservation.json")):
            journal, reservation = JobAttemptJournal.from_reservation_path(path)
            if (
                journal.scope.budget_contract_hash
                != foundation.cost_contract.contract_hash
                or journal.scope.session_contract_hash
                != foundation.session_contract.contract_hash
            ):
                raise ValueError("job attempt belongs to another session")
            if not _attempt_is_in_generation_prefix(
                foundation,
                journal=journal,
                reservation=reservation.payload,
                generation_index=generation_index,
                include_final_confirmation=include_final_confirmation,
            ):
                continue
            reservation_count += 1
            result_path = path.with_name(path.name.replace(".reservation.", ".result."))
            if not result_path.exists():
                open_reservation_count += 1
                continue
            journal.load_result(reservation.payload["attempt_index"])
            closed_result_count += 1
    payload = {
        "schema_version": (
            f"{foundation.cost_contract.operational_evidence_namespace}.generation_attempt_chain_closure.v1"
            if generation_index is None
            else f"{foundation.cost_contract.operational_evidence_namespace}.generation_attempt_chain_closure.v2"
        ),
        "source": "validated-job-attempt-journal",
        "reservation_count": reservation_count,
        "closed_result_count": closed_result_count,
        "open_reservation_count": open_reservation_count,
    }
    if generation_index is not None:
        payload["generation_index"] = generation_index
    return payload



def assert_target_gpus_compute_idle(
    executable: str,
    *,
    expected: tuple[str, str],
    stage: str,
    schema_version: str = "true_rl_agent.generation_compute_idle.v1",
) -> dict[str, Any]:
    if stage not in {"preflight-only", "startup", "final"}:
        raise ValueError("GPU compute-idle stage is invalid")
    completed = subprocess.run(
        [
            executable,
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 2 or not is_full_gpu_uuid(parts[0]):
            raise ValueError("nvidia-smi returned an invalid compute-app row")
        try:
            process_id = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                "nvidia-smi returned a non-numeric compute-app PID"
            ) from exc
        if process_id <= 0:
            raise ValueError("nvidia-smi returned an invalid compute-app PID")
        rows.append({"gpu_uuid": parts[0], "process_id": process_id})
    target_rows = [item for item in rows if item["gpu_uuid"] in expected]
    if target_rows:
        raise ValueError(f"target GPUs are not compute-idle at {stage}")
    return {
        "schema_version": schema_version,
        "stage": stage,
        "source": "nvidia-smi-query-compute-apps",
        "target_gpu_uuids": list(expected),
        "rows": target_rows,
    }
