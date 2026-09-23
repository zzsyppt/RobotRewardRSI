"""Bind a sealed parent source for same-transition reward diagnostics, not PPO."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from true_rl_agent.candidates.authoring_policy import require_mechanism_operation_delta
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object


def bind_stair_training_diagnostics(train_spec, *, parent_source, parent_evidence,
                                    mechanism_operation_delta):
    if parent_source is None or parent_evidence.get("subject_manifest_hash") is None:
        return train_spec
    delta = require_mechanism_operation_delta(mechanism_operation_delta)
    checksum = "sha256:" + hashlib.sha256(parent_source.encode("utf-8")).hexdigest()
    if checksum != delta["anchor_source_checksum"] or checksum != parent_evidence["program_source_checksum"]:
        raise ValueError("Stair diagnostic parent differs from the authoring parent")
    candidate_id = parent_evidence.get("accepted_candidate_id")
    subject = (ExperimentSubject.candidate(candidate_id) if candidate_id
               else ExperimentSubject.reference(parent_evidence["anchor_id"]))
    payload = {"schema_version": "true_rl_agent.stair_runtime_comparison.v1",
        "parent_subject": subject.to_public_dict(),
        "parent_subject_manifest_hash": parent_evidence["subject_manifest_hash"],
        "parent_source": parent_source, "mechanism_operation_delta": delta}
    context_hash = canonical_json_hash(payload)
    path = Path(train_spec.metadata["summary_path"]).parent / "runtime_comparison.json"
    content = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Stair runtime comparison changed on resume")
    else:
        path.write_bytes(content)
    argv = [*train_spec.metadata["cli_argv"], "--runtime-comparison-json", str(path),
            "--runtime-comparison-hash", context_hash]
    return replace(train_spec, runtime_diagnostic_context_hash=context_hash,
        metadata={**train_spec.metadata, "cli_argv": argv, "cli_command": " ".join(argv)})


def load_stair_training_comparison(path, expected_hash, *, context, candidate_program):
    if path is None and expected_hash is None:
        return None
    payload = load_strict_json_object(Path(path), context="Stair runtime comparison")
    if payload.get("schema_version") != "true_rl_agent.stair_runtime_comparison.v1" or canonical_json_hash(payload) != expected_hash:
        raise ValueError("Stair runtime comparison identity mismatch")
    delta = require_mechanism_operation_delta(payload["mechanism_operation_delta"])
    if delta["candidate_source_checksum"] != candidate_program.program_source_checksum:
        raise ValueError("Stair runtime comparison candidate mismatch")
    from true_rl_agent.tasks.stair_climbing.reference_program import resolve_official_stair_reference_artifact
    from true_rl_agent.tasks.stair_climbing.reward_program import qualify_stair_diagnostic_parent
    reference, _ = resolve_official_stair_reference_artifact(context.config,
        resolved_cell_manifest=context.resolved_cell_manifest)
    subject = payload["parent_subject"]
    parent = qualify_stair_diagnostic_parent(
        contract=candidate_program.contract, subject=ExperimentSubject(subject["kind"], subject["id"]),
        subject_manifest_hash=payload["parent_subject_manifest_hash"],
        source_bytes=payload["parent_source"].encode("utf-8"),
        expected_source_checksum=delta["anchor_source_checksum"], source_name=str(path),
        static_verifier_policy=context.candidate_execution_authorization.static_verifier_policy,
        trusted_reference_source_text=reference.source_text)
    return {"parent_program": parent, "operation_delta": delta, "context_hash": expected_hash}
