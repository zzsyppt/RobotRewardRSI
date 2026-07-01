#!/usr/bin/env python3
"""Regression test for the eval-corruption checkpoint guard.

A candidate trains inside an OS-isolated process and emits a checkpoint that the
trusted evaluator later loads.  RSL-RL's ``OnPolicyRunner.load`` uses
``torch.load(weights_only=False)``, which executes pickle reduce payloads, so a
tampered checkpoint could run code in the trusted eval process.  This test proves
``assert_safe_checkpoint`` accepts a normal checkpoint and rejects a checkpoint
that carries an executable payload, without ever running that payload.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from humanoid_agent.immutable_runner.safe_checkpoint import (  # noqa: E402
    CheckpointSafetyError,
    assert_safe_checkpoint,
    checkpoint_safety_report,
)
from humanoid_agent.cold_start_lab.safety_model import (  # noqa: E402
    _isolated_training_path_to_host,
    _parent_train_start_checkpoint_safety,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _Evil:
    """Pickle reduce payload; never executed because the guard rejects first."""

    def __reduce__(self):  # noqa: D401
        # A real attack would pass os.system / exec here.  The guard must reject
        # the file before any reduce runs, so we use a harmless placeholder.
        return (tuple, ((),))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="phase8e_eval_load_"))

    # 1. A normal RSL-RL-shaped checkpoint must pass the guard.
    benign = tmp / "model_0.pt"
    torch.save(
        {
            "model_state_dict": {"actor.0.weight": torch.zeros((4, 3)), "actor.0.bias": torch.zeros(4)},
            "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 0.001}]},
            "iter": 1,
            "infos": None,
        },
        benign,
    )
    loaded = assert_safe_checkpoint(benign)
    _assert(isinstance(loaded, dict) and "model_state_dict" in loaded, "benign checkpoint must load under the guard")

    # 2. A checkpoint carrying a pickle reduce payload must be rejected.
    malicious = tmp / "model_evil.pt"
    torch.save({"model_state_dict": {"w": torch.zeros(2)}, "infos": _Evil()}, malicious)
    try:
        assert_safe_checkpoint(malicious)
    except CheckpointSafetyError:
        pass
    else:
        raise AssertionError("malicious checkpoint with reduce payload was NOT rejected")

    # 3. A missing checkpoint must be rejected, not silently treated as safe.
    try:
        assert_safe_checkpoint(tmp / "does_not_exist.pt")
    except CheckpointSafetyError:
        pass
    else:
        raise AssertionError("missing checkpoint must be rejected")

    # 4. The in-process advisory claim is forgeable, but a trusted re-verification
    #    (fresh reference, never monkeypatched by candidate code) must catch it.
    #    This is why the authoritative gate lives in the trusted eval process, not
    #    in the candidate's training process.
    import types  # noqa: PLC0415

    fake_train = types.ModuleType("fake_train_module")
    fake_train.assert_safe_checkpoint = assert_safe_checkpoint  # imported as a module global

    def _child_advisory(path) -> bool:
        try:
            fake_train.assert_safe_checkpoint(path)
            return True
        except Exception:  # noqa: BLE001
            return False

    # Candidate code running in the same process rebinds the global (monkeypatch).
    fake_train.assert_safe_checkpoint = lambda *args, **kwargs: {"forged": True}
    child_claim = _child_advisory(malicious)
    _assert(child_claim is True, "advisory in-process claim should be forgeable (demonstrates the threat)")
    trusted_report = checkpoint_safety_report(malicious)
    _assert(
        trusted_report["safe_load"] is False,
        "trusted re-verification must reject the malicious checkpoint regardless of the forged child claim",
    )

    # 5. The minimal-rootfs parent must not trust the train-start child summary.
    #    It maps the isolated /runner_artifacts path back to the host run tree
    #    and re-runs the same trusted gate before a checkpoint can move toward eval.
    minimal_run_dir = tmp / "minimal_rootfs"
    host_runner_ckpt_dir = minimal_run_dir / "runner_artifacts" / "train_start_smoke" / "sealed_checkpoints"
    host_runner_ckpt_dir.mkdir(parents=True, exist_ok=True)
    benign_host = host_runner_ckpt_dir / "model_0.pt"
    malicious_host = host_runner_ckpt_dir / "model_evil.pt"
    benign_host.write_bytes(benign.read_bytes())
    malicious_host.write_bytes(malicious.read_bytes())
    benign_child_summary = {
        "probes": {
            "train_start": {
                "latest_checkpoint": "/runner_artifacts/train_start_smoke/sealed_checkpoints/model_0.pt",
            }
        }
    }
    malicious_child_summary = {
        "probes": {
            "train_start": {
                "latest_checkpoint": "/runner_artifacts/train_start_smoke/sealed_checkpoints/model_evil.pt",
            }
        }
    }
    benign_parent_report = _parent_train_start_checkpoint_safety(benign_child_summary, minimal_run_dir)
    malicious_parent_report = _parent_train_start_checkpoint_safety(malicious_child_summary, minimal_run_dir)
    _assert((benign_parent_report or {}).get("safe_load") is True, "parent train-start gate must accept benign checkpoint")
    _assert(
        (malicious_parent_report or {}).get("safe_load") is False,
        "parent train-start gate must reject malicious checkpoint even if child summary points to an existing file",
    )

    # 6. The parent gate must not let a forged child path choose an arbitrary
    #    readable file.  Only the runner-fixed sealed checkpoint path is trusted;
    #    /candidate_output, traversal through /runner_artifacts/.., and host
    #    absolute paths are all rejected before any file read.
    candidate_output_dir = minimal_run_dir / "candidate_output" / "train_start_smoke_public"
    candidate_output_dir.mkdir(parents=True, exist_ok=True)
    candidate_safe_checkpoint = candidate_output_dir / "model_0.pt"
    candidate_safe_checkpoint.write_bytes(benign.read_bytes())
    for forged_path in (
        "/candidate_output/train_start_smoke_public/model_0.pt",
        "/runner_artifacts/../candidate_output/train_start_smoke_public/model_0.pt",
        str(candidate_safe_checkpoint),
    ):
        _assert(
            _isolated_training_path_to_host(forged_path, minimal_run_dir) is None,
            f"forged train-start checkpoint path must be rejected: {forged_path}",
        )
        forged_report = _parent_train_start_checkpoint_safety(
            {"probes": {"train_start": {"latest_checkpoint": forged_path}}},
            minimal_run_dir,
        )
        _assert(
            (forged_report or {}).get("safe_load") is False,
            f"parent train-start gate must fail closed for forged path: {forged_path}",
        )

    print(json.dumps({"status": "pass", "test": "phase8e_eval_load_safety_selftest"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
