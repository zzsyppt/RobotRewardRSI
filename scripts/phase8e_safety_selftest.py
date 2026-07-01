#!/usr/bin/env python3
"""Self-test for the Phase 8E-1 safety model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.isolated_loader import load_mapping  # noqa: E402
from humanoid_agent.code_sandbox.isolation import default_landlock_read_paths, probe_isolation_capabilities  # noqa: E402
import humanoid_agent.cold_start_lab.safety_model as safety_model  # noqa: E402
from humanoid_agent.cold_start_lab.safety_model import (  # noqa: E402
    _child_security_checks_passed,
    run_safety_model_probe,
    run_trusted_eval_probe,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "cold_start" / "phase8e_safety_model.yaml"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401, PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def _write_benign_checkpoint(path: Path) -> None:
    """Write a real RSL-RL-shaped checkpoint that the trusted gate must accept."""

    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {"actor.0.weight": torch.zeros((4, 3)), "actor.0.bias": torch.zeros(4)},
            "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 0.001}]},
            "iter": 1,
            "infos": None,
        },
        path,
    )


def _assert_malicious_checkpoint_rejected(tmp_root: Path) -> None:
    """The trusted eval boundary must reject a checkpoint carrying a pickle payload.

    This is the authoritative gate: even if the (forgeable) training-side advisory
    claimed the checkpoint was safe, the trusted eval process re-verifies and
    refuses anything that does not load under weights_only=True.
    """

    import torch  # noqa: PLC0415

    class _Evil:
        def __reduce__(self):  # noqa: D401 - payload never runs; the gate rejects first.
            return (tuple, ((),))

    malicious = tmp_root / "malicious_ckpt" / "model_0.pt"
    malicious.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {"w": torch.zeros(2)}, "infos": _Evil()}, malicious)
    config = load_mapping(CONFIG_PATH)
    result = run_trusted_eval_probe(
        config=config,
        checkpoint_path=malicious,
        run_dir=tmp_root / "trusted_eval_malicious",
    )
    _assert(result["status"] == "reject", "trusted eval must reject a checkpoint that fails the safe-load gate")
    child = result.get("child_summary") or {}
    _assert(
        child.get("reason") == "checkpoint_failed_safe_load_gate",
        f"trusted eval rejected for the wrong reason: {child.get('reason')!r}",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional directory for self-test artifacts. When omitted, a temporary directory is used and cleaned up.",
    )
    return parser


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _hard_supported() -> bool:
    caps = probe_isolation_capabilities()
    return bool(caps.get("network_namespace_supported") and caps.get("pid_namespace_supported") and caps.get("landlock_abi_version"))


def _make_attack_candidate(root: Path, *, hidden_file: Path) -> Path:
    candidate = root / "phase8e_attack_candidate"
    _write_json(
        candidate / "candidate_manifest.yaml",
        {
            "schema_version": "humanoid_agent.cold_start.candidate.v1",
            "candidate_id": "phase8e_attack_candidate",
            "description": "Safety-model attack candidate; should be contained by OS isolation.",
            "author": "phase8e-selftest",
            "llm_authored": True,
            "entrypoint": "training_code.components:build_training_components",
            "phase": "8E-1-safety-selftest",
            "expected_behavior": "attempt forbidden process actions during component build",
        },
    )
    repo_config = PROJECT_ROOT / "configs" / "phase2_baseline.yaml"
    repo_write = PROJECT_ROOT / "phase8e_attack_repo_write_probe.tmp"
    code = f"""
    from __future__ import annotations

    import os
    import socket
    from pathlib import Path

    from humanoid_agent.code_sandbox.components import TrainingComponents


    def _attempt(fn):
        try:
            value = fn()
        except Exception as exc:
            return {{"status": "blocked", "error_type": type(exc).__name__}}
        return {{"status": "unexpected_success", "value": str(value)[:80]}}


    def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
        _ = (task_spec, base_env_cfg, base_agent_cfg)
        metadata = {{
            "secret_env_present": [
                key for key in sorted(os.environ)
                if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))
            ],
            "hidden_read": _attempt(lambda: Path({str(hidden_file)!r}).read_text(encoding="utf-8")),
            "repo_hidden_config_read": _attempt(lambda: Path({str(repo_config)!r}).read_text(encoding="utf-8")),
            "repo_write": _attempt(lambda: Path({str(repo_write)!r}).write_text("bad", encoding="utf-8")),
            "network_connect": _attempt(lambda: socket.create_connection(("203.0.113.1", 9), timeout=1.0)),
            "candidate_output_write": _attempt(lambda: Path.cwd().joinpath("candidate_probe.txt").write_text("ok", encoding="utf-8")),
        }}
        return TrainingComponents(name="phase8e_attack_candidate", metadata=metadata)
    """
    _write_text(candidate / "training_code" / "components.py", code)
    return candidate


def _make_atexit_tamper_candidate(root: Path, *, summary_file: Path) -> Path:
    candidate = root / "phase8e_atexit_tamper_candidate"
    _write_json(
        candidate / "candidate_manifest.yaml",
        {
            "schema_version": "humanoid_agent.cold_start.candidate.v1",
            "candidate_id": "phase8e_atexit_tamper_candidate",
            "description": "Attempts to rewrite the training child summary from an atexit hook.",
            "author": "phase8e-selftest",
            "llm_authored": True,
            "entrypoint": "training_code.components:build_training_components",
            "phase": "8E-1-safety-selftest",
            "expected_behavior": "summary tamper attempt must not turn reject into pass",
        },
    )
    code = f"""
    from __future__ import annotations

    import atexit
    import json
    from pathlib import Path

    from humanoid_agent.code_sandbox.components import TrainingComponents


    def _tamper_summary():
        Path({str(summary_file)!r}).write_text(
            json.dumps({{"status": "pass", "tampered_by_candidate": True}}, ensure_ascii=False),
            encoding="utf-8",
        )


    def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
        _ = (task_spec, base_env_cfg, base_agent_cfg)
        atexit.register(_tamper_summary)
        return TrainingComponents(name="phase8e_atexit_tamper_candidate")
    """
    _write_text(candidate / "training_code" / "components.py", code)
    return candidate


def _assert_security_downgrade_regressions() -> None:
    base_checks = {
        "secret_env_absent": True,
        "hidden_file_read_blocked": True,
        "repo_hidden_config_read_blocked": True,
        "repo_write_blocked": True,
        "candidate_output_write_allowed": True,
        "network_connect_blocked": True,
        "runner_inputs_no_protected_tokens": True,
        "oldroot_detached": True,
        "cuda_training_smoke_passed": False,
    }
    _assert(
        _child_security_checks_passed({"checks": base_checks}) is True,
        "CUDA-only failure should be eligible for blocked downgrade",
    )
    leaked_inputs = dict(base_checks, runner_inputs_no_protected_tokens=False)
    _assert(
        _child_security_checks_passed({"checks": leaked_inputs}) is False,
        "runner_inputs leak must not be downgraded to blocked",
    )
    leaked_oldroot = dict(base_checks, oldroot_detached=False)
    _assert(
        _child_security_checks_passed({"checks": leaked_oldroot}) is False,
        "oldroot leak must not be downgraded to blocked",
    )


def _assert_checkpoint_unsafe_not_downgraded_to_blocked(tmp_root: Path) -> None:
    """Checkpoint unsafe is a security failure, even if CUDA smoke also failed."""

    child_summary = {
        "status": "pass",
        "checks": {
            "secret_env_absent": True,
            "hidden_file_read_blocked": True,
            "repo_hidden_config_read_blocked": True,
            "repo_write_blocked": True,
            "candidate_output_write_allowed": True,
            "network_connect_blocked": True,
            "runner_inputs_no_protected_tokens": True,
            "oldroot_detached": True,
            "cuda_training_smoke_passed": False,
            "train_start_smoke_passed": True,
            "train_start_checkpoint_present": True,
            # Candidate-controlled child summary can forge this advisory check.
            # The trusted parent report below is the authoritative failure.
            "train_start_checkpoint_safe_load": True,
        },
        "probes": {
            "cuda_training_smoke": {"status": "fail"},
            "train_start": {
                "status": "reject",
                "reject_reason": "train_start_checkpoint_failed_safe_load_gate",
                "latest_checkpoint": "/runner_artifacts/train_start_smoke/sealed_checkpoints/model_0.pt",
            },
        },
    }

    class _Completed:
        returncode = 0

    original_run = safety_model.subprocess.run
    original_read_json = safety_model.read_json
    original_parent_safety = safety_model._parent_train_start_checkpoint_safety
    try:
        (tmp_root / "runner_artifacts").mkdir(parents=True, exist_ok=True)
        safety_model.subprocess.run = lambda *args, **kwargs: _Completed()  # type: ignore[assignment]
        safety_model.read_json = lambda _path: child_summary  # type: ignore[assignment]
        safety_model._parent_train_start_checkpoint_safety = (  # type: ignore[assignment]
            lambda _summary, _minimal_run_dir: {"safe_load": False, "error": "simulated unsafe checkpoint"}
        )
        result = safety_model.run_minimal_rootfs_training_boundary_probe(
            config={"candidate_schema_version": "humanoid_agent.cold_start.candidate.v1"},
            candidate_dir=tmp_root / "candidate",
            run_dir=tmp_root / "boundary",
            hidden_probe_file=None,
            runner_artifacts_dir=tmp_root / "runner_artifacts",
            timeout_seconds=1,
            capabilities={"network_namespace_supported": True, "pid_namespace_supported": True},
            training_cfg={"cuda_device": "cuda:0"},
        )
    finally:
        safety_model.subprocess.run = original_run  # type: ignore[assignment]
        safety_model.read_json = original_read_json  # type: ignore[assignment]
        safety_model._parent_train_start_checkpoint_safety = original_parent_safety  # type: ignore[assignment]

    _assert(
        result["status"] == "fail",
        f"checkpoint unsafe must remain fail, not be downgraded to {result['status']!r}",
    )


def _assert_component_load_entropy_device_available() -> None:
    """Component-load isolation must permit basic Python/torch entropy reads."""

    entropy = Path("/dev/urandom")
    if not entropy.exists():
        return
    paths = default_landlock_read_paths(repo_root=PROJECT_ROOT, candidate_dir=PROJECT_ROOT / "examples" / "cold_start_candidates" / "noop_candidate")
    resolved = entropy.resolve()
    _assert(
        resolved in paths,
        "8C component-load Landlock read paths must include /dev/urandom for torch/Python initialization",
    )


def _assert_atexit_summary_tamper_blocked(tmp_root: Path) -> None:
    run_dir = tmp_root / "atexit_tamper_child"
    candidate_output = run_dir / "candidate_output"
    fake_repo = run_dir / "fake_repo"
    candidate_output.mkdir(parents=True, exist_ok=True)
    fake_repo.mkdir(parents=True, exist_ok=True)
    summary_file = candidate_output / "public_train_boundary_summary.json"
    candidate = _make_atexit_tamper_candidate(tmp_root, summary_file=summary_file)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "phase8e_training_safety_child.py"),
        "--candidate-dir",
        str(candidate),
        "--summary-file",
        str(summary_file),
        "--repo-root",
        str(fake_repo),
        "--run-dir",
        str(candidate_output),
        "--expected-schema-version",
        "humanoid_agent.cold_start.candidate.v1",
    ]
    completed = subprocess.run(command, cwd=candidate_output, text=True, capture_output=True, timeout=30, check=False)
    _assert(completed.returncode != 0, "tamper candidate should not produce a passing safety child")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    _assert(summary.get("tampered_by_candidate") is not True, "candidate atexit hook rewrote authoritative summary")
    _assert(summary.get("status") != "pass", "candidate atexit hook forged pass status")


def main() -> int:
    args = _build_parser().parse_args()
    hard_supported = _hard_supported()
    if args.run_dir:
        tmp_root = Path(args.run_dir).resolve()
        tmp_root.mkdir(parents=True, exist_ok=True)
        temp_context = nullcontext(str(tmp_root))
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="phase8e_safety_selftest_")

    with temp_context as tmp:
        tmp_root = Path(tmp)
        _assert_component_load_entropy_device_available()
        _assert_security_downgrade_regressions()
        _assert_checkpoint_unsafe_not_downgraded_to_blocked(tmp_root)
        _assert_atexit_summary_tamper_blocked(tmp_root)
        hidden_file = tmp_root / "hidden" / "private_eval_profile.json"
        _write_text(hidden_file, "private evaluation profile")
        candidate = _make_attack_candidate(tmp_root, hidden_file=hidden_file)
        torch_available = _torch_available()
        checkpoint = tmp_root / "runner_artifacts" / "checkpoints" / "model_0.pt"
        if torch_available:
            _write_benign_checkpoint(checkpoint)
        else:
            # Without torch the trusted eval gate fails closed (cannot prove the
            # checkpoint carries no payload), which the assertions below expect.
            _write_text(checkpoint, "dummy checkpoint weights")

        previous_secret = os.environ.get("PHASE8E_SELFTEST_SECRET_KEY")
        os.environ["PHASE8E_SELFTEST_SECRET_KEY"] = "must_not_enter_candidate"
        try:
            exit_code, summary = run_safety_model_probe(
                config_path=CONFIG_PATH,
                candidate_dir=candidate,
                checkpoint_path=checkpoint,
                run_dir=tmp_root / "safety_probe",
                hidden_probe_file=hidden_file,
                require_server_enforcement=hard_supported,
            )
        finally:
            if previous_secret is None:
                os.environ.pop("PHASE8E_SELFTEST_SECRET_KEY", None)
            else:
                os.environ["PHASE8E_SELFTEST_SECRET_KEY"] = previous_secret

        if not torch_available:
            # Trusted eval fails closed without torch, so the overall model cannot pass.
            _assert(
                summary["status"] in ("fail", "blocked"),
                "without torch the eval checkpoint gate must fail closed (no pass)",
            )
        elif hard_supported:
            _assert(
                (exit_code == 0 and summary["status"] == "pass")
                or (exit_code == 2 and summary["status"] == "blocked"),
                "server safety model probe must pass or explicitly report a GPU-training blocker",
            )
        else:
            _assert(exit_code == 0 and summary["status"] == "pass", "local safety model probe must pass")
        training = summary["training_boundary"]
        if hard_supported:
            _assert(training["hard_isolation_enforced"] is True, "server must enforce hard training isolation")
            checks = training.get("checks") or {}
            _assert(checks.get("secret_env_absent") is True, "secret env leaked into training process")
            _assert(checks.get("hidden_file_read_blocked") is True, "training process read hidden file")
            _assert(checks.get("repo_hidden_config_read_blocked") is True, "training process read protected repo config")
            _assert(checks.get("repo_write_blocked") is True, "training process wrote repo")
            _assert(checks.get("candidate_output_write_allowed") is True, "candidate output write should be allowed")
            _assert(checks.get("network_connect_blocked") is True, "training process accessed network")
            if summary["status"] == "pass":
                _assert(checks.get("cuda_training_smoke_passed") is True, "CUDA training smoke must pass when safety model passes")
            else:
                _assert(summary["checks"].get("cuda_training_boundary_usable") is False, "blocked run must identify CUDA boundary")
                _assert(training.get("blocker") == "cuda_training_smoke_failed", "blocked run must expose CUDA blocker")
        else:
            _assert(training["skipped_hard_checks"] is True, "local non-Linux run should mark hard checks skipped")

        eval_boundary = summary["eval_boundary"]
        if torch_available:
            _assert(eval_boundary["status"] == "pass", "trusted eval checkpoint-only probe must pass")
            _assert(
                (eval_boundary.get("checkpoint_safety") or {}).get("safe_load") is True,
                "trusted eval must authoritatively verify checkpoint safe-load",
            )
            _assert_malicious_checkpoint_rejected(tmp_root)
        else:
            _assert(eval_boundary["status"] == "reject", "without torch the trusted eval must fail closed on checkpoint safety")
        _assert(eval_boundary["candidate_code_loaded"] is False, "trusted eval loaded candidate code")
        negative = summary["eval_negative_boundary"]
        _assert(negative["status"] == "reject", "trusted eval must reject candidate_dir input")

    print(
        json.dumps(
            {
                "status": "pass",
                "test": "phase8e_safety_selftest",
                "hard_isolation": hard_supported,
                "safety_model_status": summary["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
