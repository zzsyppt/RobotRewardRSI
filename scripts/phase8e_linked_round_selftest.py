#!/usr/bin/env python3
"""Self-test for Phase 8E-6-2 linked training/eval/decision orchestration."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.isolated_loader import load_mapping  # noqa: E402
from humanoid_agent.code_sandbox.isolation import probe_isolation_capabilities  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e import load_ledger  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e_projection import project_ledger_record  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e_linked import (  # noqa: E402
    Phase8ELinkedError,
    build_runner_owned_budget_summary,
    resolve_train_start_checkpoint_for_eval,
    run_phase8e_linked_round,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml"
VERIFIER_CONFIG_PATH = PROJECT_ROOT / "configs" / "cold_start" / "phase8d_verifier.yaml"
DECLARATIVE_CANDIDATE = PROJECT_ROOT / "examples" / "cold_start_candidates" / "declarative_curriculum_candidate"
ISOLATED_CHECKPOINT = "/runner_artifacts/train_start_smoke/sealed_checkpoints/model_7.pt"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _hard_supported() -> bool:
    caps = probe_isolation_capabilities()
    return bool(caps.get("network_namespace_supported") and caps.get("pid_namespace_supported") and caps.get("landlock_abi_version"))


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401, PLC0415
    except Exception:
        return False
    return True


def _write_safe_checkpoint(path: Path) -> None:
    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {"dummy_weight": torch.zeros(1)}}, path)


def _local_config(tmp_root: Path, *, require_dynamic: bool) -> Path:
    verifier = copy.deepcopy(load_mapping(VERIFIER_CONFIG_PATH))
    verifier["dynamic"]["require_phase8c_isolation"] = require_dynamic
    verifier_path = tmp_root / "phase8d_verifier_for_linked_selftest.json"
    _write_json(verifier_path, verifier)

    config = copy.deepcopy(load_mapping(CONFIG_PATH))
    config["verifier"]["config_path"] = str(verifier_path)
    config["paths"]["ledger_filename"] = "phase8e_ledger.jsonl"
    config_path = tmp_root / "phase8e_loop_for_linked_selftest.json"
    _write_json(config_path, config)
    return config_path


def _training_summary(latest_checkpoint: str = ISOLATED_CHECKPOINT) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.train_start.synthetic_summary.v1",
        "status": "pass",
        "child_summary": {
            "status": "pass",
            "probes": {
                "train_start": {
                    "status": "pass",
                    "latest_checkpoint": latest_checkpoint,
                    "config_derived_total_timesteps": 49152000,
                    "num_envs": 4096,
                    "num_steps_per_env": 24,
                    "max_iterations": 500,
                    "policy_update_iterations": 500,
                }
            },
        },
        "budget_accounting": {
            "status": "pass",
            "source": "runner_fixed_budget",
            "config_derived_total_timesteps": 49152000,
            "num_envs": 4096,
            "num_steps_per_env": 24,
            "max_iterations": 500,
        },
    }


def _rollout_summary(success_rate: float) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase1.public_rollout.summary.v1",
        "status": "pass",
        "aggregate": {
            "success_rate": success_rate,
            "episodes": 64,
            "failure_breakdown": {"fall_forward": 0.1},
            "condition_flag_breakdown": {"height_below": 0.1},
        },
        "secondary_metrics": {
            "mean_episode_steps": 180.0,
            "mean_return": 0.0,
            "mean_action_l2": 0.15,
            "mean_action_rate_l2": 0.02,
            "mean_min_base_height_m": 0.56,
            "mean_max_abs_roll_rad": 0.08,
            "mean_max_abs_pitch_rad": 0.11,
            "mean_final_xy_drift_m": 0.31,
        },
    }


def _curriculum_lineage() -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.curriculum_lineage.v1",
        "public_summary": {
            "schema_version": "humanoid_agent.phase8e.curriculum_lineage.public.v1",
            "status": "pass",
            "compiler_version": "v1-test",
            "stage_count": 2,
            "execution_mode": "env_step_progress",
            "capabilities": ["external_push_curriculum", "train_only_dr_scale"],
            "attribution": {
                "reward_code_changed": False,
                "curriculum_changed": True,
                "diagnostics_changed": True,
                "capability_changed": True,
                "gain_attributable_to": "curriculum",
                "pure_knob_candidate": False,
            },
        },
        "raw_curriculum_design": {
            "stage_list": [
                {"name": "balance_warmup", "progress_range": [0.0, 0.35]},
                {"name": "target_consolidation", "progress_range": [0.35, 1.0]},
            ]
        },
        "effective_curriculum": {
            "compiler_version": "v1-test",
            "stages": [
                {"name": "balance_warmup", "progress_range": [0.0, 0.35]},
                {"name": "target_consolidation", "progress_range": [0.35, 1.0]},
            ],
        },
    }


def _prepare_candidate(tmp_root: Path) -> Path:
    candidate = tmp_root / "candidate"
    shutil.copytree(DECLARATIVE_CANDIDATE, candidate)
    return candidate


def test_resolver_rejects_candidate_controlled_checkpoint_paths(tmp_root: Path) -> None:
    runner_artifacts = tmp_root / "runner_artifacts"
    checkpoint = runner_artifacts / "train_start_smoke" / "sealed_checkpoints" / "model_7.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not-a-real-checkpoint")

    resolved = resolve_train_start_checkpoint_for_eval(
        _training_summary(),
        runner_artifacts_root=runner_artifacts,
        validate_safe_load=False,
    )
    _assert(Path(resolved["host_path"]) == checkpoint.resolve(), "sealed checkpoint should resolve under runner_artifacts")
    _assert(resolved["isolated_path"] == ISOLATED_CHECKPOINT, "isolated checkpoint path should be recorded")

    forbidden_paths = [
        "/candidate_output/train_start_smoke/sealed_checkpoints/model_7.pt",
        str(checkpoint.resolve()),
        "/runner_artifacts/train_start_smoke/sealed_checkpoints/../model_7.pt",
        "/runner_artifacts/train_start_smoke/sealed_checkpoints/not_model.pt",
    ]
    for forbidden in forbidden_paths:
        try:
            resolve_train_start_checkpoint_for_eval(
                _training_summary(forbidden),
                runner_artifacts_root=runner_artifacts,
                validate_safe_load=False,
            )
        except Phase8ELinkedError:
            pass
        else:
            raise AssertionError(f"forbidden checkpoint path should be rejected: {forbidden}")


def test_runner_owned_budget_rejects_forged_child_accounting(tmp_root: Path) -> None:
    config_path = _local_config(tmp_root, require_dynamic=False)
    clean_budget = build_runner_owned_budget_summary(config_path=config_path, training_summary=_training_summary())
    _assert(clean_budget["budget_accounting"]["status"] == "pass", "matching runner budget evidence should pass")

    forged = _training_summary()
    train_start = forged["child_summary"]["probes"]["train_start"]
    train_start["num_envs"] = 1
    train_start["max_iterations"] = 999999
    forged["training_accounting"] = {
        "num_envs": 4096,
        "max_iterations": 500,
        "num_steps_per_env": 24,
        "config_derived_total_timesteps": 49152000,
        "policy_update_iterations": 500,
    }
    forged["budget_accounting"] = {
        "status": "pass",
        "num_envs": 1,
        "max_iterations": 999999,
        "num_steps_per_env": 24,
        "config_derived_total_timesteps": 49152000,
    }

    budget = build_runner_owned_budget_summary(config_path=config_path, training_summary=forged)
    accounting = budget["budget_accounting"]
    _assert(accounting["status"] == "fail", "runner-owned budget gate must reject forged child-reported budget fields")
    violation_keys = {item.get("key") for item in accounting["violations"]}
    _assert("num_envs" in violation_keys, "forged num_envs should be recorded as a budget violation")
    _assert("max_iterations" in violation_keys, "forged max_iterations should be recorded as a budget violation")

    # Real train-start summaries only carry num_envs + max_iterations in
    # probes.train_start; the runner-owned config-derived values (total timesteps,
    # steps_per_env, policy_update_iterations) are NOT echoed by the child. Omission
    # of runner-owned values must NOT fail the gate (the runner fills them).
    minimal = {
        "status": "pass",
        "child_summary": {
            "status": "pass",
            "probes": {"train_start": {"status": "pass", "latest_checkpoint": ISOLATED_CHECKPOINT, "num_envs": 4096, "max_iterations": 500}},
        },
    }
    minimal_budget = build_runner_owned_budget_summary(config_path=config_path, training_summary=minimal)
    ma = minimal_budget["budget_accounting"]
    _assert(ma["status"] == "pass", f"minimal real-like summary must pass the runner-owned budget gate, got violations: {ma.get('violations')}")
    _assert(ma.get("config_derived_total_timesteps") == 49152000, "runner must fill config-derived total timesteps")
    _assert(ma.get("num_steps_per_env") == 24, "runner must fill steps_per_iteration")


def test_linked_round_cli_forwards_curriculum_lineage_path(tmp_root: Path) -> None:
    module_path = PROJECT_ROOT / "scripts" / "phase8e_linked_round.py"
    spec = importlib.util.spec_from_file_location("phase8e_linked_round_cli_selftest", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import linked CLI module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured: dict[str, Any] = {}

    def fake_linked_runner(**kwargs: Any) -> tuple[int, dict[str, Any]]:
        captured.update(kwargs)
        return 0, {"status": "accept"}

    module.run_phase8e_linked_round = fake_linked_runner
    curriculum_lineage_path = tmp_root / "curriculum_lineage.json"
    _write_json(curriculum_lineage_path, _curriculum_lineage())
    old_argv = sys.argv
    sys.argv = [
        str(module_path),
        "--config",
        str(CONFIG_PATH),
        "--run-dir",
        str(tmp_root / "cli_run"),
        "--lab-dir",
        str(tmp_root / "lab"),
        "--candidate-dir",
        str(tmp_root / "candidate"),
        "--train-summary",
        str(tmp_root / "train_summary.json"),
        "--runner-artifacts-root",
        str(tmp_root / "runner_artifacts"),
        "--public-dev-summary",
        str(tmp_root / "public_dev.json"),
        "--curriculum-lineage-path",
        str(curriculum_lineage_path),
        "--append-ledger",
        "--inherit-if-accepted",
    ]
    try:
        exit_code = module.main()
    finally:
        sys.argv = old_argv
    _assert(exit_code == 0, "linked CLI should accept curriculum lineage argument")
    _assert(
        captured.get("curriculum_lineage_path") == curriculum_lineage_path.resolve(),
        f"linked CLI must forward curriculum_lineage_path: {captured}",
    )


def test_linked_round_uses_runner_resolved_safe_checkpoint(tmp_root: Path) -> str:
    if not _torch_available():
        return "skipped_no_torch"

    runner_artifacts = tmp_root / "minimal_rootfs_run" / "runner_artifacts"
    checkpoint = runner_artifacts / "train_start_smoke" / "sealed_checkpoints" / "model_7.pt"
    _write_safe_checkpoint(checkpoint)

    train_summary = tmp_root / "train_start_summary.json"
    _write_json(train_summary, _training_summary())
    dev_summary = tmp_root / "public_dev_rollout.json"
    accept_summary = tmp_root / "public_accept_rollout.json"
    _write_json(dev_summary, _rollout_summary(0.09))
    _write_json(accept_summary, _rollout_summary(0.1))
    curriculum_lineage_path = tmp_root / "curriculum_lineage.json"
    _write_json(curriculum_lineage_path, _curriculum_lineage())

    lab = tmp_root / "lab"
    run_dir = tmp_root / "linked_round"
    config_path = _local_config(tmp_root, require_dynamic=_hard_supported())
    exit_code, linked = run_phase8e_linked_round(
        config_path=config_path,
        run_dir=run_dir,
        lab_dir=lab,
        candidate_dir=_prepare_candidate(tmp_root),
        train_summary_path=train_summary,
        runner_artifacts_root=runner_artifacts,
        public_dev_summary_paths=[dev_summary],
        public_accept_summary_paths=[accept_summary],
        curriculum_lineage_path=curriculum_lineage_path,
        append_ledger=True,
        inherit_if_accepted=True,
        require_server_enforcement=_hard_supported(),
    )

    _assert(exit_code == 0, "linked round should return success for accepted synthetic public result")
    _assert(linked["status"] == "accept", "linked round should accept above-floor public_accept")
    _assert(linked["linked_eval_executed"] is True, "linked eval should execute")
    _assert(linked["checkpoint_for_eval"]["host_path"] == str(checkpoint.resolve()), "eval must consume runner-resolved checkpoint")
    _assert(linked["checkpoint_for_eval"]["checkpoint_safety"]["safe_load"] is True, "checkpoint must pass trusted safe-load gate")
    _assert(linked["trusted_eval"]["status"] == "pass", "trusted eval probe should pass")
    _assert(linked["trusted_eval"]["candidate_code_loaded"] is False, "trusted eval must not load candidate code")
    _assert(Path(linked["public_result_path"]).is_file(), "trusted public_result must be written")
    ledger = load_ledger(lab / "phase8e_ledger.jsonl")
    _assert(len(ledger) == 1, "linked accept should append one ledger record")
    record = ledger[0]
    _assert(
        record.get("curriculum_lineage", {}).get("attribution", {}).get("pure_knob_candidate") is False,
        f"ledger must record curriculum attribution: {record}",
    )
    projected = project_ledger_record(record)
    _assert(
        projected.get("curriculum_lineage", {}).get("attribution", {}).get("gain_attributable_to") == "curriculum",
        f"prompt projection must preserve public curriculum attribution: {projected}",
    )
    snapshot_dir = Path(record["inherited_recipe"]["dir"])
    _assert(
        (snapshot_dir / "runner_curriculum_snapshot" / "curriculum_lineage.json").is_file(),
        f"accepted snapshot must preserve curriculum lineage bundle: {snapshot_dir}",
    )
    return "pass"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase8e_linked_round_selftest_") as tmp:
        tmp_root = Path(tmp)
        test_resolver_rejects_candidate_controlled_checkpoint_paths(tmp_root)
        test_runner_owned_budget_rejects_forged_child_accounting(tmp_root)
        test_linked_round_cli_forwards_curriculum_lineage_path(tmp_root)
        linked_status = test_linked_round_uses_runner_resolved_safe_checkpoint(tmp_root)

    print(json.dumps({"status": "pass", "test": "phase8e_linked_round_selftest", "linked_round": linked_status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
