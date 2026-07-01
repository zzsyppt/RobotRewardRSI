#!/usr/bin/env python3
"""Self-test for Phase 8C OS/process isolation.

Local non-Linux environments may skip the hard isolation checks.  Server
validation should run with ``--require-server-enforcement`` so unsupported
security primitives fail closed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.isolated_loader import load_mapping, run_isolated_component_load, write_summary  # noqa: E402
from humanoid_agent.code_sandbox.isolation import probe_isolation_capabilities  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "cold_start" / "phase8c_isolation.yaml"
NOOP_CANDIDATE = PROJECT_ROOT / "examples" / "cold_start_candidates" / "noop_candidate"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--require-server-enforcement", action="store_true")
    return parser


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_attack_candidate(
    candidate_dir: Path,
    *,
    hidden_file: Path,
    repo_hidden_config: Path,
    repo_write_target: Path,
    runner_artifact_target: Path,
    marker: str,
) -> None:
    _write_json(
        candidate_dir / "candidate_manifest.yaml",
        {
            "schema_version": "humanoid_agent.cold_start.candidate.v1",
            "candidate_id": "phase8c_attack_candidate",
            "description": "Attack candidate used only by Phase 8C isolation selftest.",
            "author": "human-test",
            "llm_authored": False,
            "entrypoint": "training_code.components:build_training_components",
            "phase": "8C-isolation-selftest",
            "expected_behavior": "attempts forbidden env/fs/network/process actions",
        },
    )
    code = f'''
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

from humanoid_agent.code_sandbox.components import TrainingComponents

HIDDEN_FILE = Path({str(hidden_file)!r})
REPO_HIDDEN_CONFIG = Path({str(repo_hidden_config)!r})
REPO_WRITE_TARGET = Path({str(repo_write_target)!r})
RUNNER_ARTIFACT_TARGET = Path({str(runner_artifact_target)!r})
MARKER = {marker!r}


def _attempt(fn):
    try:
        value = fn()
        return {{"status": "unexpected_success", "value": repr(value)[:200]}}
    except Exception as exc:
        return {{"status": "blocked", "error_type": type(exc).__name__, "message": str(exc)[:200]}}


def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents:
    _ = (task_spec, base_env_cfg, base_agent_cfg)
    metadata = {{
        "secret_present": os.environ.get("DEEPSEEK_API_KEY") is not None,
        "hidden_read": _attempt(lambda: HIDDEN_FILE.read_text(encoding="utf-8")),
        "repo_hidden_config_read": _attempt(lambda: REPO_HIDDEN_CONFIG.read_text(encoding="utf-8")),
        "repo_write": _attempt(lambda: REPO_WRITE_TARGET.write_text("bad", encoding="utf-8")),
        "runner_artifact_write": _attempt(lambda: RUNNER_ARTIFACT_TARGET.write_text("bad", encoding="utf-8")),
        "candidate_output_write": _attempt(lambda: Path.cwd().joinpath("candidate_output_probe.txt").write_text("ok", encoding="utf-8")),
        "network_connect": _attempt(lambda: socket.create_connection(("1.1.1.1", 53), timeout=0.5).close()),
        "spawn_residual": _attempt(
            lambda: subprocess.Popen(
                ["sh", "-c", "sleep 30", MARKER],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).pid
        ),
    }}
    return TrainingComponents(
        name="phase8c_attack_candidate",
        metadata=metadata,
        rationale="Selftest candidate; blocked operations should appear in metadata.",
    )
'''
    _write_text(candidate_dir / "training_code" / "components.py", textwrap.dedent(code).strip() + "\n")


def _find_residual_processes(marker: str) -> list[str]:
    if os.name != "posix" or shutil.which("ps") is None:
        return []
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    rows = []
    for line in completed.stdout.splitlines():
        if marker in line and "phase8c_isolation_selftest.py" not in line:
            rows.append(line.strip())
    return rows


def _capabilities_satisfy_requirements(capabilities: dict[str, Any]) -> bool:
    return bool(
        capabilities.get("network_namespace_supported")
        and capabilities.get("pid_namespace_supported")
        and capabilities.get("landlock_abi_version")
    )


def main() -> int:
    args = _build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_mapping(config_path)
    capabilities = probe_isolation_capabilities()
    hard_supported = _capabilities_satisfy_requirements(capabilities)

    if args.require_server_enforcement and not hard_supported:
        summary = {
            "schema_version": "humanoid_agent.phase8c.isolation_selftest.summary.v1",
            "status": "fail",
            "capabilities": capabilities,
            "errors": [{"type": "UnsupportedIsolation", "message": "server enforcement was required but unavailable"}],
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 1

    if not hard_supported:
        summary = {
            "schema_version": "humanoid_agent.phase8c.isolation_selftest.summary.v1",
            "status": "pass",
            "skipped_hard_checks": True,
            "capabilities": capabilities,
            "note": "Hard isolation checks require Linux unshare + Landlock; run on server with --require-server-enforcement.",
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    run_root = Path(args.run_dir).resolve() if args.run_dir else Path(tempfile.mkdtemp(prefix="phase8c_isolation_selftest_"))
    run_root.mkdir(parents=True, exist_ok=True)
    hidden_dir = run_root / "hidden_audit_not_allowed"
    hidden_file = hidden_dir / "secret.txt"
    _write_text(hidden_file, "phase8c_hidden_secret")
    repo_hidden_config = PROJECT_ROOT / "configs" / "phase2_baseline.yaml"
    repo_write_target = PROJECT_ROOT / "phase8c_repo_write_probe.tmp"
    runner_artifact_target = run_root / "attack" / "runner_artifacts" / "runner_authoritative_probe.tmp"
    if repo_write_target.exists():
        repo_write_target.unlink()
    if runner_artifact_target.exists():
        runner_artifact_target.unlink()
    marker = f"phase8c_residual_marker_{uuid.uuid4().hex}"

    old_secret = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = "phase8c_should_not_leak"
    try:
        noop_exit, noop_summary = run_isolated_component_load(
            candidate_dir=NOOP_CANDIDATE,
            run_dir=run_root / "noop",
            config=config,
            allow_llm_authored=False,
            require_noop_components=True,
        )
        _assert(noop_exit == 0, "isolated noop candidate should pass")
        _assert(noop_summary["isolation"]["parent_imported_candidate_code"] is False, "parent must not import candidate code")

        attack_candidate = run_root / "attack_candidate"
        _make_attack_candidate(
            attack_candidate,
            hidden_file=hidden_file,
            repo_hidden_config=repo_hidden_config,
            repo_write_target=repo_write_target,
            runner_artifact_target=runner_artifact_target,
            marker=marker,
        )
        attack_exit, attack_summary = run_isolated_component_load(
            candidate_dir=attack_candidate,
            run_dir=run_root / "attack",
            config=config,
            allow_llm_authored=False,
            require_noop_components=False,
        )
        _assert(attack_exit == 0, "attack candidate should load so metadata can be inspected")
        metadata = attack_summary["child_summary"]["components"]["metadata"]
        _assert(metadata["secret_present"] is False, "secret env var leaked into candidate process")
        _assert(metadata["hidden_read"]["status"] == "blocked", "candidate read hidden/audit file")
        _assert(metadata["repo_hidden_config_read"]["status"] == "blocked", "candidate read protected repo config")
        _assert(metadata["repo_write"]["status"] == "blocked", "candidate wrote to main repo")
        _assert(metadata["runner_artifact_write"]["status"] == "blocked", "candidate wrote to runner artifact directory")
        _assert(metadata["candidate_output_write"]["status"] == "unexpected_success", "candidate output dir should be writable")
        _assert(metadata["network_connect"]["status"] == "blocked", "candidate accessed network")
        _assert(not repo_write_target.exists(), "repo write target was created")
        _assert(not runner_artifact_target.exists(), "runner artifact target was created")
        time.sleep(0.5)
        residual = _find_residual_processes(marker)
        _assert(not residual, f"candidate residual child processes still exist: {residual}")
    finally:
        if old_secret is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old_secret
        if repo_write_target.exists():
            repo_write_target.unlink()
        if runner_artifact_target.exists():
            runner_artifact_target.unlink()

    summary = {
        "schema_version": "humanoid_agent.phase8c.isolation_selftest.summary.v1",
        "status": "pass",
        "capabilities": capabilities,
        "run_root": str(run_root),
        "checks": {
            "noop_candidate_passed": True,
            "parent_did_not_import_candidate_code": True,
            "secret_env_absent": True,
            "hidden_file_read_blocked": True,
            "repo_hidden_config_read_blocked": True,
            "repo_write_blocked": True,
            "runner_artifact_write_blocked": True,
            "candidate_output_write_allowed": True,
            "network_connect_blocked": True,
            "residual_child_absent": True,
        },
    }
    write_summary(run_root / "phase8c_isolation_selftest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
