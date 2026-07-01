#!/usr/bin/env python3
"""Self-tests for the Phase 8E full-round orchestrator.

These tests fake Isaac/unshare subprocesses; they verify orchestration,
contract checks, and summary hygiene without starting training.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.cold_start_lab.phase8e_full_round import (  # noqa: E402
    CommandResult,
    run_phase8e_full_round,
)
from humanoid_agent.cold_start_lab.phase8e_linked import Phase8ELinkedError  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e import load_ledger, project_ledger_record  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _arg_after(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _curriculum_operation_diff() -> dict[str, Any]:
    return {
        "operations": [],
        "curriculum_design": {
            "stage_list": [
                {
                    "name": "stand_balance_warmup",
                    "progress_range": [0.0, 0.30],
                    "capabilities": ["reward_weight_scale"],
                    "rationale": "stabilize standing before perturbation.",
                },
                {
                    "name": "mild_push_recovery",
                    "progress_range": [0.30, 0.80],
                    "capabilities": ["external_push_curriculum"],
                    "rationale": "introduce recoverable external pushes.",
                },
                {
                    "name": "target_consolidation",
                    "progress_range": [0.80, 1.0],
                    "capabilities": ["external_push_curriculum", "train_only_dr_scale"],
                    "rationale": "finish on target push with train-only robustness variation.",
                },
            ],
            "final_consolidation": {
                "fraction": 0.2,
                "distribution": "runner_owned_target_distribution",
                "includes_target_push": True,
                "includes_target_robustness_variation": True,
            },
            "anti_tuning_rationale": "The design changes stage distributions instead of only changing scalar weights.",
            "improvement_attribution_intent": "Improvement should come from staged push recovery and final target consolidation.",
        },
    }


def _curriculum_diagnostics() -> dict[str, Any]:
    return {
        "items": [],
        "curriculum_hypothesis": {
            "primary_failure_mode": "public push recovery is weaker than no-push standing.",
            "evidence": ["public push probe under target distribution"],
            "anti_tuning_rationale": "Reward terms alone do not expose recovery perturbations during training.",
            "improvement_attribution_intent": "Curriculum stage distribution is expected to contribute to improvement.",
        },
    }


def _make_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    (candidate / "training_code").mkdir(parents=True)
    (candidate / "candidate_manifest.yaml").write_text(
        json.dumps(
            {
                "schema_version": "humanoid_agent.cold_start.candidate.v1",
                "candidate_id": "phase8e_full_round_selftest",
                "description": "selftest no-op candidate",
                "author": "llm",
                "llm_authored": True,
                "entrypoint": "training_code.components:build_training_components",
                "phase": "8E-full-round-selftest",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (candidate / "training_code" / "components.py").write_text(
        "from humanoid_agent.code_sandbox.components import TrainingComponents\n\n"
        "def build_training_components(task_spec, base_env_cfg, base_agent_cfg):\n"
        "    return TrainingComponents(name='phase8e_full_round_selftest')\n",
        encoding="utf-8",
    )
    (candidate / "rationale.md").write_text("selftest\n", encoding="utf-8")
    _write_json(candidate / "operation_diff.json", _curriculum_operation_diff())
    _write_json(candidate / "diagnostics.json", _curriculum_diagnostics())
    return candidate


def test_staged_unverified_provider_is_not_training_ready() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-full-round-unverified-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)

        def provider_runner(**_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "staged_unverified",
                "training_ready": False,
                "candidate_dir": str(candidate),
                "verifier_status": "skipped",
            }

        def command_runner(*_args: Any, **_kwargs: Any) -> CommandResult:
            raise AssertionError("training must not run for staged_unverified candidates")

        exit_code, summary = run_phase8e_full_round(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_dir=root / "round",
            provider_runner=provider_runner,
            command_runner=command_runner,
            checkpoint_resolver=_fake_checkpoint_resolver,
            linked_runner=_fake_linked_runner,
        )
        _assert(exit_code == 1, "unverified provider candidate should fail the full round")
        _assert(summary["status"] == "reject", "unverified provider candidate should produce a reject summary")
        _assert("training" not in summary["stages"], "training stage must not be present after provider gate failure")


def test_full_round_driver_sequences_training_rollout_and_linked_decision() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-full-round-pass-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)
        commands: list[list[str]] = []
        linked_calls: list[dict[str, Any]] = []
        provider_calls: list[dict[str, Any]] = []
        prompt_payload_file = root / "provider_prompt_payload.json"
        _write_json(
            prompt_payload_file,
            {
                "public_curriculum_evidence_bundle": {
                    "schema_version": "humanoid_agent.phase8e.public_curriculum_evidence.v1",
                    "capability_registry": {
                        "capabilities": [
                            {"id": "external_push_curriculum", "status": "available_now"},
                            {"id": "train_only_dr_scale", "status": "available_now"},
                        ]
                    },
                    "public_dev_probes": [{"id": "push_probe", "public": True}],
                },
                "required_candidate_package": {
                    "component_contract": {
                        "reward_inputs_contract": {"fields": ["push_active", "push_vec_public"]},
                        "control_effectiveness_contract": {"effective_schedule_keys": ["reward_weight_scale"]},
                    }
                },
                "candidate_allowed_controls": ["RewardTerm", "ScheduleTerm", "DiagnosticTerm"],
                "curriculum_design_contract": {"schema_version": "humanoid_agent.phase8e.curriculum_design.v1"},
            },
        )

        def provider_runner(**kwargs: Any) -> dict[str, Any]:
            provider_calls.append(kwargs)
            return {
                "status": "pass",
                "training_ready": True,
                "candidate_dir": str(candidate),
                "verifier_status": "pass",
                "summary_file": str(root / "provider_summary.json"),
                "prompt": {"payload_file": str(prompt_payload_file), "public_only": True},
            }

        def command_runner(command: list[str], _cwd: Path, _env: dict[str, str], log_file: Path, _timeout: int) -> CommandResult:
            commands.append(command)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("fake command\n", encoding="utf-8")
            joined = " ".join(command)
            if "phase8e_minimal_rootfs_probe.py" in joined:
                _fake_training_summary(command)
            elif "phase1_push_recovery_eval.py" in joined:
                _fake_rollout_summary(command)
            else:
                raise AssertionError(f"unexpected command: {command}")
            return CommandResult(exit_code=0, timed_out=False, elapsed_seconds=0.01, log_file=str(log_file))

        def linked_runner(**kwargs: Any) -> tuple[int, dict[str, Any]]:
            linked_calls.append(kwargs)
            linked_dir = Path(kwargs["run_dir"])
            _write_json(linked_dir / "phase8e_linked_round_summary.json", {"status": "accept"})
            return 0, {"status": "accept"}

        exit_code, summary = run_phase8e_full_round(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_dir=root / "round",
            lab_dir=root / "lab",
            append_ledger=True,
            inherit_if_accepted=True,
            provider_runner=provider_runner,
            command_runner=command_runner,
            checkpoint_resolver=_fake_checkpoint_resolver,
            linked_runner=linked_runner,
        )
        _assert(exit_code == 0, "fake full round should exit cleanly")
        _assert(summary["status"] == "accept", "linked decision status should become full-round status")
        _assert(summary["formal_training_executed"] is True, "top-level summary must record formal training execution")
        _assert(
            summary["formal_training"]["source"] == "full_round_fixed_budget_training_stage",
            "formal training provenance should point to the full-round training stage",
        )
        _assert(len(commands) == 7, "expected one training command plus six rollout commands")
        train_command = commands[0]
        _assert(train_command[:2] == ["unshare", "-Urmnfp"], "training command must enter unshare boundary")
        _assert("--train-start-smoke" in train_command, "training command must run train-start smoke")
        _assert(summary["stages"]["curriculum_compile"]["status"] == "pass", "curriculum must compile before training")
        _assert(
            Path(_arg_after(train_command, "--phase1-config")).name == "compiled_phase1_push_recovery.yaml",
            f"training must use runner-compiled phase1 config: {train_command}",
        )
        _assert(
            Path(_arg_after(train_command, "--phase2-config")).name == "compiled_phase2_baseline.yaml",
            f"training must use runner-compiled phase2 config: {train_command}",
        )
        _assert(_arg_after(train_command, "--num-envs") == "4096", "training must use Phase 8B num_envs")
        _assert(_arg_after(train_command, "--max-iterations") == "500", "training must use Phase 8B max_iterations")
        _assert(provider_calls, "provider must be called to produce a candidate")
        _assert(
            str(provider_calls[0].get("lab_dir")) == str((root / "lab").resolve()),
            "provider must receive the shared lab_dir so its prompt inherits cross-round feedback",
        )
        _assert(linked_calls, "linked round must be called after rollouts")
        _assert(len(linked_calls[0]["public_dev_summary_paths"]) == 3, "linked round must receive three public-dev rollouts")
        _assert(len(linked_calls[0]["public_accept_summary_paths"]) == 3, "linked round must receive three public-accept rollouts")
        lineage_path = linked_calls[0].get("curriculum_lineage_path")
        _assert(lineage_path is not None and Path(lineage_path).is_file(), "linked round must receive curriculum lineage artifact")
        lineage = json.loads(Path(lineage_path).read_text(encoding="utf-8"))
        immutable_context = lineage.get("immutable_context")
        _assert(isinstance(immutable_context, dict), "curriculum lineage must include immutable_context")
        _assert(
            immutable_context.get("schema_version") == "humanoid_agent.phase8e.curriculum_immutable_context.v1",
            f"unexpected immutable_context schema: {immutable_context}",
        )
        for key in [
            "capability_registry",
            "curriculum_compiler",
            "public_evidence_bundle",
            "prompt_facing_contract",
            "training_config",
            "eval_config",
        ]:
            section = immutable_context.get(key)
            _assert(isinstance(section, dict), f"immutable_context.{key} must be a mapping")
            _assert(
                isinstance(section.get("sha256"), str) and len(section["sha256"]) == 64,
                f"immutable_context.{key} must include a sha256: {section}",
            )
        _assert(
            immutable_context.get("sha256") == lineage["public_summary"].get("immutable_context_hash"),
            "public curriculum lineage summary must expose only the immutable context hash",
        )
        public_lineage_text = json.dumps(lineage["public_summary"], ensure_ascii=False, sort_keys=True)
        _assert("1101" not in public_lineage_text and "2101" not in public_lineage_text, "public lineage must not echo raw seeds")
        _assert(
            str(prompt_payload_file) not in public_lineage_text,
            "public lineage must not expose prompt payload artifact paths",
        )
        summary_text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        _assert("1101" not in summary_text and "2101" not in summary_text, "full-round summary must not echo raw rollout seeds")


def test_training_boundary_blocked_maps_to_blocked_status() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-full-round-blocked-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)

        def provider_runner(**_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "pass",
                "training_ready": True,
                "candidate_dir": str(candidate),
                "verifier_status": "pass",
            }

        def command_runner(command: list[str], _cwd: Path, _env: dict[str, str], log_file: Path, _timeout: int) -> CommandResult:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("fake blocked command\n", encoding="utf-8")
            _fake_training_summary(command, status="blocked", train_start_status="blocked", reject_reason="cuda_training_smoke_failed")
            return CommandResult(exit_code=2, timed_out=False, elapsed_seconds=0.01, log_file=str(log_file))

        exit_code, summary = run_phase8e_full_round(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_dir=root / "round",
            provider_runner=provider_runner,
            command_runner=command_runner,
            checkpoint_resolver=_fake_checkpoint_resolver,
            linked_runner=_fake_linked_runner,
        )
        _assert(exit_code == 2, "blocked training should use retryable/blocker exit code")
        _assert(summary["status"] == "blocked", "blocked training must not be reported as generic fail")
        _assert(summary["formal_training_executed"] is False, "failed training stage must not claim formal training executed")


def test_checkpoint_safe_load_failure_maps_to_failed_safety_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-full-round-safe-load-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)

        def provider_runner(**_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "pass",
                "training_ready": True,
                "candidate_dir": str(candidate),
                "verifier_status": "pass",
            }

        def command_runner(command: list[str], _cwd: Path, _env: dict[str, str], log_file: Path, _timeout: int) -> CommandResult:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("fake command\n", encoding="utf-8")
            _fake_training_summary(command)
            return CommandResult(exit_code=0, timed_out=False, elapsed_seconds=0.01, log_file=str(log_file))

        def checkpoint_resolver(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise Phase8ELinkedError("resolved train-start checkpoint failed safe-load gate: unsafe pickle")

        exit_code, summary = run_phase8e_full_round(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_dir=root / "round",
            provider_runner=provider_runner,
            command_runner=command_runner,
            checkpoint_resolver=checkpoint_resolver,
            linked_runner=_fake_linked_runner,
        )
        _assert(exit_code == 1, "failed safety gate should stop the round")
        _assert(summary["status"] == "failed_safety_gate", "checkpoint safe-load failure must be classified as failed_safety_gate")
        _assert(summary["formal_training_executed"] is True, "training did execute before checkpoint safety failed")


def test_trusted_eval_public_gate_failure_maps_to_reject_not_safety_gate() -> None:
    for trusted_eval_reason in ("public_result_status_not_pass", "budget_accounting_status_not_pass"):
        with tempfile.TemporaryDirectory(prefix="phase8e-full-round-trusted-eval-reject-") as tmp:
            root = Path(tmp)
            candidate = _make_candidate(root)

            def provider_runner(**_kwargs: Any) -> dict[str, Any]:
                return {
                    "status": "pass",
                    "training_ready": True,
                    "candidate_dir": str(candidate),
                    "verifier_status": "pass",
                }

            def command_runner(command: list[str], _cwd: Path, _env: dict[str, str], log_file: Path, _timeout: int) -> CommandResult:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_file.write_text("fake command\n", encoding="utf-8")
                joined = " ".join(command)
                if "phase8e_minimal_rootfs_probe.py" in joined:
                    _fake_training_summary(command)
                elif "phase1_push_recovery_eval.py" in joined:
                    _fake_rollout_summary(command)
                else:
                    raise AssertionError(f"unexpected command: {command}")
                return CommandResult(exit_code=0, timed_out=False, elapsed_seconds=0.01, log_file=str(log_file))

            def linked_runner(**_kwargs: Any) -> tuple[int, dict[str, Any]]:
                raise Phase8ELinkedError(f"trusted eval probe failed: {trusted_eval_reason}")

            exit_code, summary = run_phase8e_full_round(
                config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
                run_dir=root / "round",
                provider_runner=provider_runner,
                command_runner=command_runner,
                checkpoint_resolver=_fake_checkpoint_resolver,
                linked_runner=linked_runner,
            )
            _assert(exit_code == 1, "trusted eval public gate reject should stop the round")
            _assert(
                summary["status"] == "reject",
                f"trusted eval public gate failure must be reject, not failed_safety_gate: {summary}",
            )
            _assert(
                summary["failure_classification"] == {"status": "reject", "reason": "trusted_eval_gate_failed"},
                f"trusted eval reason must remain audit-visible without safety-gate misclassification: {summary}",
            )
            _assert(summary["formal_training_executed"] is True, "training did execute before trusted eval rejection")


def test_provider_verifier_failure_appends_public_feedback_without_training() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-full-round-provider-reject-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)
        lab_dir = root / "lab"

        def provider_runner(**_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "fail",
                "training_ready": False,
                "candidate_dir": str(candidate),
                "candidate": {
                    "candidate_dir": str(candidate),
                    "manifest": {
                        "candidate_id": "phase8e_full_round_selftest",
                        "schema_version": "humanoid_agent.cold_start.candidate.v1",
                    },
                    "operation_diff": {"operations": []},
                    "diagnostics": {"items": []},
                },
                "verifier_status": "fail",
                "verifier": {
                    "status": "fail",
                    "checks": {"static": {"status": "pass"}, "dynamic": {"status": "fail"}},
                    "issues": [{"code": "isolated_component_load_failed", "severity": "error"}],
                },
                "verifier_public_projection": {
                    "status": "fail",
                    "static_status": "pass",
                    "dynamic_status": "fail",
                    "issue_codes": {"isolated_component_load_failed": 1},
                },
            }

        def command_runner(*_args: Any, **_kwargs: Any) -> CommandResult:
            raise AssertionError("training must not run when provider/8D gate fails")

        exit_code, summary = run_phase8e_full_round(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_dir=root / "round",
            lab_dir=lab_dir,
            append_ledger=True,
            provider_runner=provider_runner,
            command_runner=command_runner,
            checkpoint_resolver=_fake_checkpoint_resolver,
            linked_runner=_fake_linked_runner,
        )
        _assert(exit_code == 1, "provider/8D failure should reject before training")
        _assert(summary["status"] == "reject", "provider/8D failure should be classified as reject")
        _assert(summary["stages"]["candidate"]["verifier_status"] == "fail", "candidate stage should be retained for audit")
        _assert("training" not in summary["stages"], "training must not run after provider/8D failure")
        records = load_ledger(lab_dir / "phase8e_ledger.jsonl")
        _assert(len(records) == 1, "pre-training failure should append one public feedback ledger record")
        decision = records[0]["decision"]
        _assert(decision["reason"] == "candidate_not_training_ready_after_verifier", "ledger must preserve public failure reason")
        _assert(records[0]["verifier"]["issue_codes"]["isolated_component_load_failed"] == 1, "ledger must expose public verifier issue codes")


def test_provider_inner_loop_reject_reason_survives_missing_candidate_dir() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-full-round-inner-loop-reject-") as tmp:
        root = Path(tmp)
        missing_candidate = root / "missing_candidate"

        def provider_runner(**_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "reject",
                "training_ready": False,
                "candidate_dir": str(missing_candidate),
                "candidate": None,
                "verifier_status": "skipped",
                "inner_loop": {
                    "enabled": True,
                    "status": "rejected",
                    "reason": "candidate_failed_pretraining_inner_loop",
                    "selected_attempt": None,
                    "parent_imported_candidate_code": False,
                },
            }

        def command_runner(*_args: Any, **_kwargs: Any) -> CommandResult:
            raise AssertionError("training must not run after inner-loop pretraining rejection")

        exit_code, summary = run_phase8e_full_round(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_dir=root / "round",
            lab_dir=root / "lab",
            append_ledger=True,
            provider_runner=provider_runner,
            command_runner=command_runner,
            checkpoint_resolver=_fake_checkpoint_resolver,
            linked_runner=_fake_linked_runner,
        )
        _assert(exit_code == 1, "pretraining inner-loop rejection should be a hard reject")
        _assert(summary["status"] == "reject", f"summary should reject: {summary}")
        _assert(
            summary["failure_classification"]["reason"] == "candidate_failed_pretraining_inner_loop",
            f"full-round reason should preserve provider inner-loop reason: {summary['failure_classification']}",
        )
        _assert(summary["stages"]["candidate"]["status"] == "reject", "candidate stage should retain provider reject status")
        _assert(summary["stages"]["pretraining_feedback"]["reason"] == "candidate_dir_missing", "ledger feedback should skip without candidate files")
        _assert("training" not in summary["stages"], "training must not run after inner-loop rejection")


def test_training_failure_appends_public_feedback_without_raw_reason() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-full-round-train-fail-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)
        lab_dir = root / "lab"

        def provider_runner(**_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "pass",
                "training_ready": True,
                "candidate_dir": str(candidate),
                "verifier_status": "pass",
            }

        def command_runner(command: list[str], _cwd: Path, _env: dict[str, str], log_file: Path, _timeout: int) -> CommandResult:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("fake blocked training\n", encoding="utf-8")
            _fake_training_summary(command, status="blocked", train_start_status="blocked", reject_reason="cuda_oom_in_candidate_reward")
            return CommandResult(exit_code=2, timed_out=False, elapsed_seconds=0.01, log_file=str(log_file))

        exit_code, summary = run_phase8e_full_round(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_dir=root / "round",
            lab_dir=lab_dir,
            append_ledger=True,
            provider_runner=provider_runner,
            command_runner=command_runner,
            checkpoint_resolver=_fake_checkpoint_resolver,
            linked_runner=_fake_linked_runner,
        )
        _assert(exit_code == 2, "blocked training should use the blocker exit code")
        _assert(summary["status"] == "blocked", "training failure must be classified, not generic fail")
        _assert(summary["stages"]["training_feedback"]["status"] == "appended", "training failure must append public feedback")
        records = load_ledger(lab_dir / "phase8e_ledger.jsonl")
        _assert(len(records) == 1, "training failure should append exactly one public feedback record")
        decision = records[0]["decision"]
        _assert(decision["detail"]["stage"] == "fixed_budget_training", "feedback must mark the training stage")
        _assert(decision["detail"]["training_status"] == "blocked", "feedback must expose the runner training class")
        _assert(decision["reason"] == "fixed_budget_training_blocked", "feedback reason must be runner-owned, not raw")
        _assert(
            "cuda_oom_in_candidate_reward" not in json.dumps(records[0], ensure_ascii=False),
            "raw candidate-influenced training reason must not leak into the next prompt",
        )


def test_training_no_checkpoint_maps_to_reject_not_safety_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-full-round-nockpt-") as tmp:
        root = Path(tmp)
        candidate = _make_candidate(root)

        def provider_runner(**_kwargs: Any) -> dict[str, Any]:
            return {"status": "pass", "training_ready": True, "candidate_dir": str(candidate), "verifier_status": "pass"}

        def command_runner(command: list[str], _cwd: Path, _env: dict[str, str], log_file: Path, _timeout: int) -> CommandResult:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("fake\n", encoding="utf-8")
            # Candidate reward crashed at runtime: training produced no checkpoint.
            _fake_training_summary(
                command,
                status="fail",
                train_start_status="fail",
                checkpoint_present=False,
                training_error={
                    "component_name": "xy_velocity_command_tracking",
                    "type": "RuntimeError",
                    "message": "tensor shape mismatch. lhs has shape [N, 3], rhs has shape [N, 2]",
                    "candidate_stack_frame": {
                        "path": "training_code/components.py",
                        "line": 42,
                        "function": "xy_velocity_command_tracking",
                    },
                    "shape_mismatch": {
                        "lhs_shape": ["N", 3],
                        "rhs_shape": ["N", 2],
                    },
                    "raw_private_debug": "must_not_reach_prompt",
                },
            )
            return CommandResult(exit_code=1, timed_out=False, elapsed_seconds=0.01, log_file=str(log_file))

        exit_code, summary = run_phase8e_full_round(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_dir=root / "round", lab_dir=root / "lab", append_ledger=True,
            provider_runner=provider_runner, command_runner=command_runner,
            checkpoint_resolver=_fake_checkpoint_resolver, linked_runner=_fake_linked_runner,
        )
        _assert(summary["status"] == "reject", f"no-checkpoint training crash must be reject, got {summary['status']}")
        _assert(exit_code == 1, "reject should exit 1, not the blocked code 2")
        # Structured public failure feedback must reach the next-round prompt
        # without leaking raw candidate-influenced debug fields.
        records = load_ledger((root / "lab") / "phase8e_ledger.jsonl")
        _assert(len(records) == 1, "training crash should append one feedback record")
        detail = records[0]["decision"]["detail"]
        _assert(detail.get("stage") == "fixed_budget_training", "feedback marks the training stage")
        feedback = detail.get("public_failure_feedback") or {}
        _assert(feedback.get("stage") == "fixed_budget_training", f"feedback must preserve failed stage: {feedback}")
        _assert(feedback.get("component_name") == "xy_velocity_command_tracking", f"feedback must preserve component name: {feedback}")
        _assert(feedback.get("exception", {}).get("type") == "RuntimeError", f"feedback must expose exception type: {feedback}")
        _assert("shape mismatch" in feedback.get("exception", {}).get("message", ""), f"feedback must expose public-safe message: {feedback}")
        _assert(
            feedback.get("candidate_stack_frame", {}).get("path") == "training_code/components.py",
            f"feedback must expose public-safe candidate frame: {feedback}",
        )
        _assert(
            feedback.get("shape_mismatch", {}).get("lhs_shape") == ["N", 3]
            and feedback.get("shape_mismatch", {}).get("rhs_shape") == ["N", 2],
            f"feedback must preserve public shape mismatch detail: {feedback}",
        )
        _assert(
            feedback.get("checkpoint_missing_due_to_training_failure") is True,
            f"feedback must say checkpoint missing was caused by training failure: {feedback}",
        )
        projected = project_ledger_record(records[0])
        _assert(projected.get("public_failure_feedback") == feedback, "prompt ledger projection must include public failure feedback")
        _assert("raw_private_debug" not in json.dumps(records[0], ensure_ascii=False), "raw private debug must not reach ledger")


def _fake_training_summary(
    command: list[str],
    *,
    status: str = "pass",
    train_start_status: str = "pass",
    reject_reason: str | None = None,
    checkpoint_present: bool = True,
    training_error: dict[str, Any] | None = None,
) -> None:
    run_dir = Path(_arg_after(command, "--run-dir"))
    summary_file = run_dir / "candidate_output" / _arg_after(command, "--summary-file")
    checkpoint = run_dir / "runner_artifacts" / "train_start_smoke" / "sealed_checkpoints" / "model_500.pt"
    if checkpoint_present:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"fake checkpoint; safe-load is injected in selftest")
    train_start = {
        "status": train_start_status,
        "reject_reason": reject_reason,
        "latest_checkpoint": "/runner_artifacts/train_start_smoke/sealed_checkpoints/model_500.pt",
        "trusted_latest_checkpoint": "/runner_artifacts/train_start_smoke/sealed_checkpoints/model_500.pt",
        "num_envs": 4096,
        "max_iterations": 500,
        "num_steps_per_env": 24,
        "config_derived_total_timesteps": 49152000,
        "policy_update_iterations": 500,
        "checkpoint_present": checkpoint_present,
        "checkpoint_safe_load": checkpoint_present,
        "training_error": training_error,
    }
    _write_json(
        summary_file,
        {
            "schema_version": "humanoid_agent.phase8e.minimal_rootfs_probe.v1",
            "status": status,
            "checks": {"train_start_smoke_passed": train_start_status == "pass", "oldroot_detached": True},
            "probes": {"train_start": train_start},
        },
    )


def _fake_rollout_summary(command: list[str]) -> None:
    summary_file = Path(_arg_after(command, "--summary-file"))
    _write_json(
        summary_file,
        {
            "schema_version": "humanoid_agent.phase1.push_recovery.summary.v1",
            "status": "pass",
            "aggregate": {
                "episodes": 64,
                "success_count": 48,
                "success_rate": 0.75,
                "failure_breakdown": {},
                "condition_flag_breakdown": {},
            },
            "secondary_metrics": {
                "mean_episode_steps": 200.0,
                "mean_return": 1.0,
                "mean_action_l2": 0.1,
                "mean_action_rate_l2": 0.1,
                "mean_min_base_height_m": 0.7,
                "mean_max_abs_pitch_rad": 0.1,
                "mean_max_abs_roll_rad": 0.1,
                "mean_final_xy_drift_m": 0.1,
            },
        },
    )


def _fake_checkpoint_resolver(
    training_summary: dict[str, Any],
    *,
    runner_artifacts_root: Path,
    validate_safe_load: bool,
) -> dict[str, Any]:
    _assert(validate_safe_load is True, "checkpoint resolver must safe-load before rollout")
    latest = training_summary["probes"]["train_start"]["latest_checkpoint"]
    _assert(latest == "/runner_artifacts/train_start_smoke/sealed_checkpoints/model_500.pt", "unexpected sealed checkpoint path")
    host = runner_artifacts_root / "train_start_smoke" / "sealed_checkpoints" / "model_500.pt"
    return {
        "isolated_path": latest,
        "host_path": str(host),
        "runner_artifacts_root": str(runner_artifacts_root),
        "checkpoint_safety": {"safe_load": True, "sha256": "fake"},
    }


def _fake_linked_runner(**_kwargs: Any) -> tuple[int, dict[str, Any]]:
    return 0, {"status": "accept"}


def main() -> int:
    test_staged_unverified_provider_is_not_training_ready()
    test_full_round_driver_sequences_training_rollout_and_linked_decision()
    test_training_boundary_blocked_maps_to_blocked_status()
    test_checkpoint_safe_load_failure_maps_to_failed_safety_gate()
    test_trusted_eval_public_gate_failure_maps_to_reject_not_safety_gate()
    test_provider_verifier_failure_appends_public_feedback_without_training()
    test_provider_inner_loop_reject_reason_survives_missing_candidate_dir()
    test_training_failure_appends_public_feedback_without_raw_reason()
    test_training_no_checkpoint_maps_to_reject_not_safety_gate()
    print(json.dumps({"status": "pass", "test": "phase8e_full_round_selftest"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
