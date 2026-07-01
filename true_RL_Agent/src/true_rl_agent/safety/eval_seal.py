"""Trusted eval seal summaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from true_rl_agent.runner.checkpoint import safe_load_checkpoint
from true_rl_agent.runner.specs import EvalJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec


@dataclass(frozen=True)
class TrustedEvalSeal:
    eval_config_hash: str
    checkpoint_hash: str
    task_invariant_hash: str
    safe_load_status: str
    candidate_code_loaded: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "eval_config_hash": self.eval_config_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "task_invariant_hash": self.task_invariant_hash,
            "safe_load_status": self.safe_load_status,
            "candidate_code_loaded": self.candidate_code_loaded,
        }


def build_trusted_eval_seal(*, eval_spec: EvalJobSpec, task_invariants: TaskInvariantSpec) -> TrustedEvalSeal:
    safe_load = safe_load_checkpoint(eval_spec.checkpoint)
    if safe_load.status != "pass":
        raise ValueError(f"trusted eval checkpoint failed safe-load: {safe_load.reason}")
    eval_config = {
        "job_id": eval_spec.job_id,
        "task_id": eval_spec.task_id,
        "candidate_id": eval_spec.candidate_id,
        "checkpoint": eval_spec.checkpoint.to_public_dict(),
        "public_seed_count": eval_spec.public_seed_count,
        "trusted_seed_count": eval_spec.trusted_seed_count,
    }
    return TrustedEvalSeal(
        eval_config_hash=_stable_hash(eval_config),
        checkpoint_hash=eval_spec.checkpoint.checksum,
        task_invariant_hash=_stable_hash(dict(task_invariants.values)),
        safe_load_status=safe_load.status,
        candidate_code_loaded=False,
    )


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

