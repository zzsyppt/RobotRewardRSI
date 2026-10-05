"""Recover an original launch as historical training, without executing it."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object
from true_rl_agent.reference_artifact import load_reference_program_artifact, REFERENCE_ARTIFACT_MANIFEST_FILENAME
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.specs import GpuProcessBinding, TrainJobSpec
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.agent_core.trial_receipts import _train_result_to_private_mapping


SOURCE_SCHEMA = "true_rl_agent.completed_reference_training_source.v1"
RECEIPT_FILENAME = "completed_reference_training_import.json"


def _private_spec(spec: TrainJobSpec) -> dict[str, Any]:
    return {"spec": spec.to_public_dict(), "metadata": dict(spec.metadata),
            "gpu_process_binding": spec.gpu_process_binding.to_dict(),
            "reference_artifact_path": spec.reference_artifact.path}


def _file_ref(path) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def validate_training_reuse(original: TrainJobSpec, requested: TrainJobSpec) -> None:
    """Compare the complete requested job using the verified original placement.

    Only placement is imported. Binding, budget, program, initialization and all
    other public fields must match. The original spec itself is never rewritten.
    """
    comparable = replace(requested, job_id=original.job_id,
        execution_device=original.execution_device, gpu_process_binding=original.gpu_process_binding)
    if comparable.to_public_dict() != original.to_public_dict():
        raise ValueError("completed reference training differs from the requested scientific job")


@dataclass(frozen=True)
class VerifiedCompletedReferenceTraining:
    train_spec: TrainJobSpec
    train_result: TrainJobResult
    receipt_path: Path
    receipt_checksum: str

    def source_ref(self) -> dict[str, Any]:
        self.validate()
        return {"kind": "imported_historical_training", "train_job_id": self.train_spec.job_id,
                "receipt_uri": str(self.receipt_path), "receipt_checksum": self.receipt_checksum}

    def validate(self) -> None:
        if file_sha256(self.receipt_path) != self.receipt_checksum:
            raise ValueError("completed training import receipt changed")
        receipt = load_strict_json_object(self.receipt_path)
        for source in receipt["source_files"].values():
            if _file_ref(source["path"]) != source:
                raise ValueError("completed training source bytes changed: " + source["path"])
        private_spec = _private_spec(self.train_spec)
        if (canonical_json_hash(private_spec) != receipt["train_job_hash"]
                or canonical_json_hash(_train_result_to_private_mapping(self.train_result)) != receipt["normalized_train_result_hash"]):
            raise ValueError("completed training tuple differs from its verified source")


def recover_completed_reference_training(
    descriptor_path: Path, *, requested: TrainJobSpec, runner,
    output_dir: Path, require_existing: bool = False,
) -> VerifiedCompletedReferenceTraining:
    descriptor_path = descriptor_path.resolve(strict=True)
    descriptor = load_strict_json_object(descriptor_path)
    if descriptor.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("unsupported completed training source descriptor")
    source_files = {name: _file_ref(descriptor[name]["path"])
                    for name in ("launch", "summary", "checkpoint")}
    if any(source_files[name] != descriptor[name] for name in source_files):
        raise ValueError("completed training source differs from its descriptor")
    source_cwd = Path(descriptor["source_cwd"]).resolve(strict=True)
    launch = load_strict_json_object(source_files["launch"]["path"])
    if launch.get("status") != "PASS" or launch.get("returncode") != 0:
        raise ValueError("original training launch did not complete successfully")
    argv = launch["argv"]

    def option(name):
        if argv.count(name) != 1:
            raise ValueError("original training argv must bind " + name)
        return argv[argv.index(name) + 1]

    def source_path(name):
        return (source_cwd / option(name)).resolve(strict=True)

    for name, flag in (("summary", "--summary-path"), ("checkpoint", "--checkpoint-output")):
        if source_files[name] != _file_ref(source_path(flag)):
            raise ValueError("original training argv differs from descriptor " + name)
    raw = launch["job_spec"]
    artifact = load_reference_program_artifact(source_path("--reference-artifact"),
        expected_artifact_hash=raw["reference_artifact"]["artifact_hash"],
        expected_reference_id=raw["subject"]["id"], expected_task_id=raw["task_id"],
        expected_subject_manifest_hash=raw["subject_manifest_hash"],
        expected_program_source_checksum=raw["program_source_checksum"])
    gpu = GpuProcessBinding(host_execution_device=option("--gpu-host-execution-device"),
        process_execution_device=option("--gpu-process-execution-device"),
        resource_binding_hash=option("--gpu-resource-binding-hash"))
    environment = launch["execution_environment"]
    if environment["manifest_hash"] != option("--backend-execution-environment-manifest-hash"):
        raise ValueError("original training execution environment differs from argv")
    metadata = {"cli_argv": argv, "cli_command": " ".join(argv), "work_dir": str(source_cwd),
        "summary_path": option("--summary-path"),
        "checkpoint_output_path": option("--checkpoint-output"),
        "backend_id": environment["backend_id"],
        "backend_execution_environment_manifest_hash": environment["manifest_hash"]}
    if "--capture-training-numeric" in argv:
        metadata["capture_training_numeric"] = True
    # Start from current science and restore only the recorded original placement
    # and artifact location. Full equality below proves every launch identity field.
    spec = replace(requested, job_id=raw["job_id"], execution_device=raw["execution_device"],
        gpu_process_binding=gpu, reference_artifact=artifact, metadata=metadata, attempt_scope=None)
    if spec.to_public_dict() != raw:
        raise ValueError("original training launch differs from the requested reference training")
    result = runner.recover_completed_train(spec)
    if result.summary.get("formal_training_executed") is not True:
        raise ValueError("original training did not complete its formal budget")
    if (result.job_id != spec.job_id or result.checkpoint.checksum != source_files["checkpoint"]["sha256"]
            or _file_ref(result.checkpoint.metadata["local_path"]) != source_files["checkpoint"]):
        raise ValueError("recovered training differs from the original checkpoint")
    source_files.update(descriptor=_file_ref(descriptor_path),
        resolved_cell=_file_ref(source_path("--resolved-cell-manifest-path")),
        reference_manifest=_file_ref(Path(artifact.path) / REFERENCE_ARTIFACT_MANIFEST_FILENAME),
        reference_components=_file_ref(Path(artifact.path) / "training_code/components.py"))
    private_spec = _private_spec(spec)
    receipt = {"schema_version": "true_rl_agent.completed_reference_training_import.v2",
        "classification": "imported_historical_training", "source_cwd": str(source_cwd),
        "source_files": source_files, "train_job": private_spec,
        "train_job_hash": canonical_json_hash(private_spec),
        "normalized_train_result_hash": canonical_json_hash(_train_result_to_private_mapping(result)),
        "source_execution_environment_hash": environment["manifest_hash"]}
    path = output_dir / RECEIPT_FILENAME
    if path.exists():
        if load_strict_json_object(path) != receipt:
            raise ValueError("existing completed training source binding differs")
    elif require_existing:
        raise FileNotFoundError("resumed session lacks its completed training source binding")
    else:
        create_once_bytes(path, canonical_json_bytes(receipt, trailing_newline=True))
    verified = VerifiedCompletedReferenceTraining(spec, result, path, file_sha256(path))
    verified.validate()
    return verified
