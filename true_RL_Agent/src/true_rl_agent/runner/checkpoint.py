"""Runner-owned checkpoint seal and safe-load checks."""

from __future__ import annotations

from dataclasses import dataclass

from true_rl_agent.runner.specs import CheckpointRef


@dataclass(frozen=True)
class SafeLoadResult:
    status: str
    checkpoint_hash: str | None
    reason: str

    def to_public_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "checkpoint_hash": self.checkpoint_hash,
            "reason": self.reason,
        }


def seal_checkpoint(checkpoint: CheckpointRef, *, sealed_uri: str) -> CheckpointRef:
    if checkpoint.role not in {"train_output", "eval"}:
        raise ValueError(f"cannot seal checkpoint with role: {checkpoint.role}")
    return CheckpointRef(
        uri=sealed_uri,
        checksum=checkpoint.checksum,
        role="eval",
        sealed=True,
        metadata={**dict(checkpoint.metadata), "sealed_by": "runner"},
    )


def safe_load_checkpoint(checkpoint: CheckpointRef) -> SafeLoadResult:
    if checkpoint.role != "eval":
        return SafeLoadResult("fail", checkpoint.checksum, "checkpoint_role_not_eval")
    if not checkpoint.sealed:
        return SafeLoadResult("fail", checkpoint.checksum, "checkpoint_not_runner_sealed")
    if not checkpoint.checksum.startswith("sha256:"):
        return SafeLoadResult("fail", checkpoint.checksum, "checkpoint_hash_missing")
    return SafeLoadResult("pass", checkpoint.checksum, "checkpoint_safe_to_load")

