"""Bind a planned eval job to exact runner-sealed checkpoint bytes."""

from __future__ import annotations

from dataclasses import replace

from true_rl_agent.runner.argv_contract import option_occurrences
from true_rl_agent.runner.initial_policy import (
    verified_initial_policy_state_pairing_hash,
)
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, GpuProcessBinding


def bind_eval_spec_to_checkpoint(
    eval_spec: EvalJobSpec,
    checkpoint: CheckpointRef,
) -> EvalJobSpec:
    if checkpoint.role != "eval" or not checkpoint.sealed:
        raise ValueError(
            "eval binding requires a sealed checkpoint with role='eval'"
        )
    if eval_spec.checkpoint is not None:
        raise ValueError("eval job is already bound to checkpoint bytes")
    checkpoint_output_plan = eval_spec.checkpoint_output_plan
    if checkpoint_output_plan is None:
        raise ValueError("eval job has no checkpoint output plan")
    lineage = checkpoint.metadata
    expected_lineage = {
        "training_subject": eval_spec.subject.to_public_dict(),
        "subject_manifest_hash": eval_spec.subject_manifest_hash,
        "program_source_checksum": eval_spec.program_source_checksum,
        "training_binding_hash": eval_spec.training_binding_hash,
        "capability_profile_hash": eval_spec.capability_profile_hash,
        "execution_device": eval_spec.execution_device,
    }
    for field_name, expected in expected_lineage.items():
        if lineage.get(field_name) != expected:
            raise ValueError(
                f"eval checkpoint lineage mismatch for {field_name}"
            )
    expected_gpu_process_binding = eval_spec.gpu_process_binding
    declared_gpu_process_binding = lineage.get("gpu_process_binding")
    if expected_gpu_process_binding is None:
        if declared_gpu_process_binding is not None:
            raise ValueError("eval checkpoint GPU process binding mismatch")
    else:
        checkpoint_gpu_process_binding = GpuProcessBinding.from_mapping(
            declared_gpu_process_binding
        )
        if checkpoint_gpu_process_binding != expected_gpu_process_binding:
            raise ValueError("eval checkpoint GPU process binding mismatch")
    old_uri = checkpoint_output_plan.uri
    cli_checkpoint = str(checkpoint.metadata.get("local_path") or checkpoint.uri)
    metadata = dict(eval_spec.metadata)
    cli_argv = metadata.get("cli_argv")
    if cli_argv is None:
        if "cli_command" in metadata:
            raise ValueError(
                "planned eval metadata cannot provide cli_command without cli_argv"
            )
    elif not isinstance(cli_argv, list) or not all(
        isinstance(item, str) for item in cli_argv
    ):
        raise ValueError("planned eval metadata requires structured cli_argv")
    else:
        metadata["cli_argv"] = _bind_eval_cli_argv(
            cli_argv,
            old_checkpoint_uri=old_uri,
            new_checkpoint_uri=cli_checkpoint,
            new_checkpoint_hash=checkpoint.checksum,
        )
        metadata["cli_command"] = " ".join(metadata["cli_argv"])
    return replace(
        eval_spec,
        checkpoint=checkpoint,
        checkpoint_output_plan=None,
        initial_policy_state_pairing_hash=(
            verified_initial_policy_state_pairing_hash(
                checkpoint.metadata.get("initial_policy_state_pairing")
            )
        ),
        metadata=metadata,
    )


def _bind_eval_cli_argv(
    cli_argv: list[str],
    *,
    old_checkpoint_uri: str,
    new_checkpoint_uri: str,
    new_checkpoint_hash: str,
) -> list[str]:
    checkpoint_positions = option_occurrences(cli_argv, "--checkpoint")
    if len(checkpoint_positions) != 1:
        raise ValueError("planned eval argv requires exactly one --checkpoint")
    checkpoint_index = checkpoint_positions[0]
    if cli_argv[checkpoint_index] != "--checkpoint":
        raise ValueError("planned eval --checkpoint must use a separate value token")
    if checkpoint_index + 1 >= len(cli_argv):
        raise ValueError("planned eval argv --checkpoint is missing its value")
    if cli_argv[checkpoint_index + 1] != old_checkpoint_uri:
        raise ValueError("planned eval argv checkpoint does not match output plan")
    if option_occurrences(cli_argv, "--checkpoint-sha256"):
        raise ValueError("planned eval argv must not declare a checkpoint digest")
    updated = list(cli_argv)
    updated[checkpoint_index + 1] = new_checkpoint_uri
    updated[checkpoint_index + 2 : checkpoint_index + 2] = [
        "--checkpoint-sha256",
        new_checkpoint_hash,
    ]
    return updated


__all__ = ["bind_eval_spec_to_checkpoint"]
