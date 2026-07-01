#!/usr/bin/env python3
"""Local self-test for Phase 8A no-op candidate loader and runner."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.loader import CandidateLoadError, load_candidate_manifest, load_training_components  # noqa: E402
from scripts.phase8a_min_runner import build_train_smoke_command, run_phase8a, subprocess_environment  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "cold_start" / "phase8a_min.yaml"
NOOP_CANDIDATE = PROJECT_ROOT / "examples" / "cold_start_candidates" / "noop_candidate"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    manifest, components = load_training_components(
        NOOP_CANDIDATE,
        expected_schema_version="humanoid_agent.cold_start.candidate.v1",
        allow_llm_authored=False,
        task_spec={"task": "Unitree-G1-29dof-Velocity"},
    )
    _assert(manifest.candidate_id == "phase8a_noop_candidate", "unexpected candidate id")
    _assert(components.is_noop(), "noop candidate must return no-op components")

    config = _load_json(CONFIG_PATH)
    manifest_for_provenance = load_candidate_manifest(
        NOOP_CANDIDATE,
        expected_schema_version=config["candidate_policy"]["allowed_schema_version"],
    )
    trusted = config["candidate_policy"]["trusted_human_candidates"][0]
    _assert(trusted["candidate_id"] == manifest_for_provenance.candidate_id, "trusted candidate id mismatch")
    _assert(
        (PROJECT_ROOT / trusted["path"]).resolve() == manifest_for_provenance.path,
        "trusted candidate path must resolve to the human-authored noop candidate",
    )

    command, _summary = build_train_smoke_command(config, run_dir=Path("/tmp/phase8a_selftest"))
    _assert("--num-envs" in command, "train command must include --num-envs")
    _assert(command[command.index("--num-envs") + 1] == "4096", "Phase8A num_envs must be 4096")
    _assert(command[command.index("--max-iterations") + 1] == "1", "Phase8A smoke max_iterations must be 1")
    _assert(str(SRC_ROOT) in subprocess_environment().get("PYTHONPATH", ""), "runner subprocess must set PYTHONPATH")

    with tempfile.TemporaryDirectory(prefix="phase8a_selftest_") as tmp:
        tmp_root = Path(tmp)
        dry_run_dir = tmp_root / "dry_run"
        exit_code, summary = run_phase8a(
            candidate_dir=NOOP_CANDIDATE,
            config_path=CONFIG_PATH,
            run_dir=dry_run_dir,
            execute_smoke=False,
        )
        _assert(exit_code == 0, "noop dry-run should pass")
        _assert(summary["status"] == "pass", "noop dry-run status should be pass")
        _assert(summary["smoke"]["executed"] is False, "dry-run must not execute smoke")
        _assert((dry_run_dir / "commands.sh").is_file(), "runner should write replay commands")

        bad_candidate = tmp_root / "bad_candidate"
        shutil.copytree(NOOP_CANDIDATE, bad_candidate)
        manifest_path = bad_candidate / "candidate_manifest.yaml"
        manifest_data = _load_json(manifest_path)
        manifest_data["llm_authored"] = True
        manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            load_training_components(
                bad_candidate,
                expected_schema_version="humanoid_agent.cold_start.candidate.v1",
                allow_llm_authored=False,
            )
        except CandidateLoadError:
            pass
        else:
            raise AssertionError("LLM-authored candidate must be rejected before Phase8C")

        reject_dir = tmp_root / "reject"
        reject_exit_code, reject_summary = run_phase8a(
            candidate_dir=bad_candidate,
            config_path=CONFIG_PATH,
            run_dir=reject_dir,
            execute_smoke=False,
        )
        _assert(reject_exit_code != 0, "bad candidate should return non-zero")
        _assert(reject_summary["status"] == "reject", "bad candidate should be rejected")

        spoofed_candidate = tmp_root / "spoofed_candidate"
        shutil.copytree(NOOP_CANDIDATE, spoofed_candidate)
        spoof_exit_code, spoof_summary = run_phase8a(
            candidate_dir=spoofed_candidate,
            config_path=CONFIG_PATH,
            run_dir=tmp_root / "spoof_reject",
            execute_smoke=False,
        )
        _assert(spoof_exit_code != 0, "copied candidate outside trusted path should return non-zero")
        _assert(spoof_summary["status"] == "reject", "copied candidate outside trusted path should be rejected")

        tree_tamper = tmp_root / "tree_tamper_candidate"
        shutil.copytree(NOOP_CANDIDATE, tree_tamper)
        trusted_path = config["candidate_policy"]["trusted_human_candidates"][0]["path"]
        tamper_config = dict(config)
        tamper_policy = dict(config["candidate_policy"])
        tamper_item = dict(config["candidate_policy"]["trusted_human_candidates"][0])
        tamper_item["path"] = str(tree_tamper.relative_to(PROJECT_ROOT)) if tree_tamper.is_relative_to(PROJECT_ROOT) else str(tree_tamper)
        tamper_policy["trusted_human_candidates"] = [tamper_item]
        tamper_config["candidate_policy"] = tamper_policy
        tamper_config_path = tmp_root / "phase8a_tamper_config.json"
        tamper_config_path.write_text(json.dumps(tamper_config, ensure_ascii=False, indent=2), encoding="utf-8")
        (tree_tamper / "training_code" / "unused_helper.py").write_text("SIDE_EFFECT = True\n", encoding="utf-8")
        tamper_exit_code, tamper_summary = run_phase8a(
            candidate_dir=tree_tamper,
            config_path=tamper_config_path,
            run_dir=tmp_root / "tree_tamper_reject",
            execute_smoke=False,
        )
        _assert(tamper_exit_code != 0, "extra training_code file should return non-zero")
        _assert(tamper_summary["status"] == "reject", "extra training_code file should be rejected")
        _assert(trusted_path == "examples/cold_start_candidates/noop_candidate", "test should not mutate base config")

    print(json.dumps({"status": "pass", "test": "phase8a_selftest"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
