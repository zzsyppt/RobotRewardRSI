#!/usr/bin/env python3
"""Self-test for the Phase 8D code-level verifier."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.isolated_loader import load_mapping, run_isolated_component_load  # noqa: E402
from humanoid_agent.code_sandbox.isolation import probe_isolation_capabilities  # noqa: E402
from humanoid_agent.code_sandbox.verifier import verify_candidate_package, write_summary  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "cold_start" / "phase8d_verifier.yaml"
NOOP_CANDIDATE = PROJECT_ROOT / "examples" / "cold_start_candidates" / "noop_candidate"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _base_manifest(candidate_id: str) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.cold_start.candidate.v1",
        "candidate_id": candidate_id,
        "description": "Phase 8D verifier selftest candidate.",
        "author": "human-test",
        "llm_authored": False,
        "entrypoint": "training_code.components:build_training_components",
        "phase": "8D-verifier-selftest",
        "expected_behavior": "selftest only",
    }


def _write_candidate(root: Path, *, candidate_id: str, code: str, manifest_extra: dict[str, Any] | None = None) -> Path:
    candidate = root / candidate_id
    manifest = _base_manifest(candidate_id)
    if manifest_extra:
        manifest.update(manifest_extra)
    _write_json(candidate / "candidate_manifest.yaml", manifest)
    _write_text(candidate / "training_code" / "components.py", code)
    return candidate


def _good_code() -> str:
    return """
    from __future__ import annotations

    from humanoid_agent.code_sandbox.components import TrainingComponents


    def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents:
        _ = (task_spec, base_env_cfg, base_agent_cfg)
        return TrainingComponents(name="phase8d_good_candidate", rationale="safe no-op")
    """


def _verify(candidate_dir: Path, config_path: Path, run_dir: Path, *, require_server: bool = False) -> dict[str, Any]:
    summary = verify_candidate_package(
        candidate_dir=candidate_dir,
        config_path=config_path,
        run_dir=run_dir,
        require_server_enforcement=require_server,
    )
    write_summary(run_dir / "phase8d_verifier_summary.json", summary)
    return summary


def _has_issue(summary: dict[str, Any], code: str) -> bool:
    return any(issue.get("code") == code for issue in summary.get("issues", []))


def _issues_for_path(summary: dict[str, Any], relative_path: str) -> list[dict[str, Any]]:
    return [issue for issue in summary.get("issues", []) if issue.get("path") == relative_path]


def _local_config(tmp_root: Path, *, require_dynamic: bool) -> Path:
    config = copy.deepcopy(load_mapping(CONFIG_PATH))
    config["dynamic"]["require_phase8c_isolation"] = require_dynamic
    path = tmp_root / ("phase8d_verifier_required.json" if require_dynamic else "phase8d_verifier_local.json")
    _write_json(path, config)
    return path


def _soft_child_config() -> dict[str, Any]:
    return {
        "isolation": {
            "require_network_namespace": False,
            "require_pid_namespace": False,
            "require_landlock": False,
            "timeout_seconds": 30,
            "env_allowlist": [],
        },
        "candidate_policy": {
            "allowed_schema_version": "humanoid_agent.cold_start.candidate.v1",
        },
    }


def _hard_supported() -> bool:
    caps = probe_isolation_capabilities()
    return bool(caps.get("network_namespace_supported") and caps.get("pid_namespace_supported") and caps.get("landlock_abi_version"))


def main() -> int:
    hard_supported = _hard_supported()
    with tempfile.TemporaryDirectory(prefix="phase8d_verifier_selftest_") as tmp:
        tmp_root = Path(tmp)
        config_path = _local_config(tmp_root, require_dynamic=hard_supported)

        noop = _verify(NOOP_CANDIDATE, config_path, tmp_root / "noop", require_server=hard_supported)
        _assert(noop["status"] == "pass", "noop candidate should pass Phase8D verifier")

        forbidden_import = _write_candidate(
            tmp_root,
            candidate_id="forbidden_import",
            code="""
            import subprocess

            from humanoid_agent.code_sandbox.components import TrainingComponents

            def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
                return TrainingComponents(name="bad")
            """,
        )
        summary = _verify(forbidden_import, config_path, tmp_root / "forbidden_import")
        _assert(summary["status"] == "fail" and _has_issue(summary, "forbidden_import"), "subprocess import must fail")

        hidden_literal = _write_candidate(
            tmp_root,
            candidate_id="hidden_literal",
            code="""
            from humanoid_agent.code_sandbox.components import TrainingComponents

            def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
                return TrainingComponents(name="bad", rationale="iid_hidden_eval")
            """,
        )
        summary = _verify(hidden_literal, config_path, tmp_root / "hidden_literal")
        _assert(summary["status"] == "fail" and _has_issue(summary, "forbidden_literal"), "hidden token literal must fail")

        protected_seed = _write_candidate(
            tmp_root,
            candidate_id="protected_seed",
            code="""
            from humanoid_agent.code_sandbox.components import TrainingComponents

            def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
                seed = 2001
                return TrainingComponents(name=f"bad-{seed}")
            """,
        )
        summary = _verify(protected_seed, config_path, tmp_root / "protected_seed")
        _assert(summary["status"] == "fail" and _has_issue(summary, "forbidden_numeric_token"), "protected seed must fail")

        forbidden_capability = _write_candidate(
            tmp_root,
            candidate_id="forbidden_capability",
            code=_good_code(),
            manifest_extra={"capabilities": ["reward_terms", "eval_wrapper"]},
        )
        summary = _verify(forbidden_capability, config_path, tmp_root / "forbidden_capability")
        _assert(
            summary["status"] == "fail" and _has_issue(summary, "forbidden_manifest_capability"),
            "forbidden manifest capability must fail",
        )

        rationale_leak = _write_candidate(tmp_root, candidate_id="rationale_leak", code=_good_code())
        _write_text(rationale_leak / "rationale.md", "candidate notes mention iid_hidden seed 2001")
        summary = _verify(rationale_leak, config_path, tmp_root / "rationale_leak")
        _assert(
            summary["status"] == "fail"
            and _has_issue(summary, "forbidden_content_token")
            and _has_issue(summary, "forbidden_numeric_token"),
            "rationale.md forbidden content must fail",
        )

        operation_diff_leak = _write_candidate(tmp_root, candidate_id="operation_diff_leak", code=_good_code())
        _write_json(operation_diff_leak / "operation_diff.json", {"note": "hidden_audit_seed", "seed": 2001})
        summary = _verify(operation_diff_leak, config_path, tmp_root / "operation_diff_leak")
        _assert(
            summary["status"] == "fail"
            and _has_issue(summary, "forbidden_content_token")
            and _has_issue(summary, "forbidden_numeric_token"),
            "operation_diff.json forbidden content must fail",
        )

        getattr_call = _write_candidate(
            tmp_root,
            candidate_id="getattr_call",
            code="""
            from humanoid_agent.code_sandbox.components import TrainingComponents

            def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
                name = getattr(task_spec, "name", "bad")
                return TrainingComponents(name=name)
            """,
        )
        summary = _verify(getattr_call, config_path, tmp_root / "getattr_call")
        _assert(summary["status"] == "fail" and _has_issue(summary, "forbidden_call"), "getattr call must fail")

        symlink_case = _write_candidate(tmp_root, candidate_id="symlink_case", code=_good_code())
        outside_target = tmp_root / "outside_link_target.py"
        _write_text(outside_target, "import subprocess\nLEAK = 'iid_hidden_eval'")
        linked = symlink_case / "training_code" / "linked.py"
        try:
            os.symlink(outside_target, linked)
        except (OSError, NotImplementedError):
            linked = None
        if linked is not None:
            summary = _verify(symlink_case, config_path, tmp_root / "symlink_case")
            linked_issues = _issues_for_path(summary, "training_code/linked.py")
            linked_codes = {issue.get("code") for issue in linked_issues}
            _assert(summary["status"] == "fail" and "symlink_not_allowed" in linked_codes, "candidate symlink must fail")
            _assert(
                "forbidden_import" not in linked_codes and "forbidden_literal" not in linked_codes,
                "verifier must reject symlink without following and parsing the target",
            )

        fast_bad_shape = _write_candidate(
            tmp_root,
            candidate_id="fast_bad_shape",
            code="""
            from humanoid_agent.code_sandbox.components import RewardTerm, TrainingComponents

            def bad_shape(inp):
                return [[0.0, 0.0] for _ in range(len(inp.base_height))]

            def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
                _ = (task_spec, base_env_cfg, base_agent_cfg)
                return TrainingComponents(
                    name="fast_bad_shape",
                    reward_terms=[RewardTerm("bad_shape", fn=bad_shape)],
                )
            """,
        )
        exit_code, fast_summary = run_isolated_component_load(
            candidate_dir=fast_bad_shape,
            run_dir=tmp_root / "fast_bad_shape_child",
            config=_soft_child_config(),
            fast_validation={
                "enabled": True,
                "num_envs": 4,
                "num_joints": 2,
                "num_actions": 2,
                "prefer_torch": False,
            },
            python_executable=sys.executable,
        )
        _assert(exit_code != 0 and fast_summary["status"] == "reject", f"bad shape must fail child fast validation: {fast_summary}")
        _assert(
            fast_summary["isolation"]["parent_imported_candidate_code"] is False,
            "parent must not import candidate code during fast validation",
        )
        _assert(
            fast_summary["isolation"].get("python_executable") == sys.executable
            and fast_summary["isolation"].get("python_executable_source") == "argument",
            f"isolated loader must record explicit child python: {fast_summary['isolation']}",
        )
        _assert(
            fast_summary.get("child_summary", {}).get("runtime", {}).get("python_executable") == sys.executable,
            f"child summary must record runtime python executable: {fast_summary}",
        )
        child_fast = fast_summary.get("child_summary", {}).get("fast_validation", {})
        _assert(child_fast.get("status") == "fail", f"child fast validation summary missing failure: {fast_summary}")
        _assert(
            child_fast.get("synthetic_inputs", {}).get("tensor_backend", {}).get("backend") == "synthetic",
            f"child fast validation must report synthetic backend when prefer_torch=False: {child_fast}",
        )
        _assert(
            any(issue.get("component_name") == "bad_shape" and issue.get("kind") == "reward" for issue in child_fast.get("issues", [])),
            f"fast validation must return component-specific reward issue: {child_fast}",
        )

        extra_file = _write_candidate(tmp_root, candidate_id="extra_file", code=_good_code())
        _write_text(extra_file / "outside.py", "x = 1")
        summary = _verify(extra_file, config_path, tmp_root / "extra_file")
        _assert(summary["status"] == "fail" and _has_issue(summary, "top_level_entry_not_allowed"), "unexpected top-level file must fail")

        dynamic_leak = _write_candidate(
            tmp_root,
            candidate_id="dynamic_leak",
            code="""
            from humanoid_agent.code_sandbox.components import TrainingComponents

            def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
                leak = "_".join(["iid", "hidden"])
                return TrainingComponents(name="dynamic_leak", metadata={"leak": leak})
            """,
        )
        summary = _verify(dynamic_leak, config_path, tmp_root / "dynamic_leak", require_server=hard_supported)
        if hard_supported:
            _assert(
                summary["status"] == "fail" and _has_issue(summary, "forbidden_public_output"),
                "dynamic public summary leak must fail when isolation is available",
            )
        else:
            _assert(summary["status"] == "pass", "dynamic leak case is only enforceable when 8C isolation is available")

    print(json.dumps({"status": "pass", "test": "phase8d_verifier_selftest", "hard_isolation": hard_supported}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
