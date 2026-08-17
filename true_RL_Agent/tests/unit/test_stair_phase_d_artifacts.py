from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.runner.budget import build_budget_accounting  # noqa: E402
from true_rl_agent.runner.checkpoint import (  # noqa: E402
    CheckpointSafetyReport,
    file_sha256,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (  # noqa: E402
    StairTrainingExecutionResult,
)
from true_rl_agent.tasks.stair_climbing.phase_d_run import (  # noqa: E402
    build_stair_phase_d_process_event,
    validate_stair_phase_d_process_event,
)
from true_rl_agent.tasks.stair_climbing.pilot_artifact_store import (  # noqa: E402
    materialize_stair_phase_d_pilot_artifact,
    validate_stair_phase_d_pilot_artifact,
)
from true_rl_agent.tasks.stair_climbing.pilot_artifacts import (  # noqa: E402
    STAIR_PILOT_CURVE_SUMMARY_SCHEMA_VERSION,
    STAIR_PILOT_REQUIRED_SCALARS,
    build_stair_phase_d_report,
    build_stair_pilot_eval_summary,
    build_stair_pilot_seed_evidence,
    build_stair_pilot_train_summary,
    serialize_stair_pilot_raw_outcomes,
    validate_stair_phase_d_report,
    validate_stair_pilot_eval_summary,
    validate_stair_pilot_seed_evidence,
    validate_stair_pilot_train_summary,
    write_canonical_pilot_json,
)
from true_rl_agent.tasks.stair_climbing.reference_pilot import (  # noqa: E402
    build_stair_pilot_config,
    build_stair_reference_capability_evidence,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.termination_capture import (  # noqa: E402
    stair_termination_manager_resolution_contract,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (  # noqa: E402
    resolved_stair_training_distribution,
    stair_training_distribution_upstream_source_evidence,
)

PROTOCOL_PATH = ROOT / "configs" / "tasks" / "stair_protocol_v2_draft_r2.json"


class StairPhaseDArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_stair_pilot_config(PROTOCOL_PATH)
        self.official = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        )

    def test_fixed_budget_is_consumed_by_every_pilot_artifact(self) -> None:
        self.assertEqual(self.config.train_max_iterations, 2000)
        self.assertEqual(self.config.budget_steps, 196_608_000)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = _seed_evidence(
                self.config,
                self.official,
                root=root,
                capability_passes=True,
                normalized_tail_change=9.0,
            )
            self.assertEqual(seed["max_iterations"], 2000)
            self.assertEqual(seed["budget_steps"], 196_608_000)
            self.assertEqual(seed["status"], "pass")
            self.assertEqual(seed["gate_statuses"], {"capability": "pass"})

    def test_curve_trend_is_diagnostic_and_cannot_skip_capability_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed = _seed_evidence(
                self.config,
                self.official,
                root=Path(tmp),
                capability_passes=True,
                normalized_tail_change=1_000.0,
            )
        curve = seed["training"]["curve_summary"]
        self.assertFalse(curve["decision_authority"])
        self.assertNotIn("convergence_gate", curve)
        self.assertEqual(seed["evaluation"]["status"], "completed")

    def test_eval_summary_recomputes_capability_from_raw_outcomes(self) -> None:
        checksum = "sha256:" + "b" * 64
        outcomes = _outcomes(self.config, passes=True)
        summary = _eval_summary(
            self.config,
            checkpoint_hash=checksum,
            outcomes=outcomes,
            attempt_id="raw-outcome-binding",
        )
        forged = copy.deepcopy(summary)
        forged["checkpoint_evaluation"]["per_seed_outcomes"][0][0][
            "max_progress_m"
        ] = 0.0
        forged["summary_hash"] = _rehash(forged, "summary_hash")
        with self.assertRaisesRegex(ValueError, "differs from raw outcomes"):
            validate_stair_pilot_eval_summary(
                self.config,
                summary=forged,
                expected_checkpoint_hash=checksum,
                expected_attempt_id="raw-outcome-binding",
            )

    def test_capability_failure_fails_report_without_changing_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = _seed_evidence(
                self.config,
                self.official,
                root=root,
                capability_passes=False,
            )
            report = build_stair_phase_d_report(
                self.config,
                official=self.official,
                pilot_evidence=seed,
                event_log_checksum="sha256:" + "e" * 64,
                environment=_environment_evidence(self.config),
            )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["pilot_evidence"]["failed_gates"], ["capability"])
        self.assertEqual(report["budget_policy"]["max_iterations"], 2000)
        self.assertFalse(report["budget_policy"]["budget_optimality_claimed"])
        self.assertFalse(report["budget_policy"]["curve_convergence_claimed"])

    def test_report_rejects_budget_policy_decision_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seed = _seed_evidence(
                self.config,
                self.official,
                root=Path(tmp),
                capability_passes=True,
            )
            report = build_stair_phase_d_report(
                self.config,
                official=self.official,
                pilot_evidence=seed,
                event_log_checksum="sha256:" + "e" * 64,
                environment=_environment_evidence(self.config),
            )
        forged = copy.deepcopy(report)
        forged["budget_policy"]["curve_convergence_claimed"] = True
        forged["report_hash"] = _rehash(forged, "report_hash")
        with self.assertRaisesRegex(ValueError, "fixed budget policy"):
            validate_stair_phase_d_report(
                self.config,
                official=self.official,
                report=forged,
            )

    def test_checkpoint_bytes_are_revalidated_from_train_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint-v1")
            summary = _train_summary(
                self.config,
                self.official,
                checkpoint=checkpoint,
                attempt_id="attempt-a",
            )
            validate_stair_pilot_train_summary(
                self.config,
                official=self.official,
                summary=summary,
                checkpoint_path=checkpoint,
            )
            checkpoint.write_bytes(b"checkpoint-v2")
            with self.assertRaisesRegex(ValueError, "bytes drifted"):
                validate_stair_pilot_train_summary(
                    self.config,
                    official=self.official,
                    summary=summary,
                    checkpoint_path=checkpoint,
                )

    def test_content_addressed_artifact_rejects_payload_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "run"
            run_root.mkdir()
            write_canonical_pilot_json(
                run_root / "phase_d_pilot_report.json",
                {
                    "draft_protocol_hash": self.config.protocol_v2.protocol_hash,
                    "report_hash": "a" * 64,
                },
            )
            evidence = run_root / "pilot-2000" / "checkpoint.pt"
            evidence.parent.mkdir()
            evidence.write_bytes(b"immutable-evidence")
            store_root = root / "store"
            manifest = materialize_stair_phase_d_pilot_artifact(
                run_root=run_root,
                artifact_store_root=store_root,
            )
            validated = validate_stair_phase_d_pilot_artifact(
                manifest=manifest,
                artifact_store_root=store_root,
            )
            stored = validated["payload_root"] / "pilot-2000" / "checkpoint.pt"
            stored.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "payload differs"):
                validate_stair_phase_d_pilot_artifact(
                    manifest=manifest,
                    artifact_store_root=store_root,
                )

    def test_process_event_id_is_derived_from_lifecycle_identity(self) -> None:
        event = build_stair_phase_d_process_event(
            attempt_id="attempt-a",
            budget_steps=self.config.budget_steps,
            phase="eval",
            status="completed",
            has_exited=True,
            exit_code=0,
            failure_message=None,
        )
        validate_stair_phase_d_process_event(event)
        forged = dict(event)
        forged["event_id"] = "forged"
        with self.assertRaisesRegex(ValueError, "event_id binding"):
            validate_stair_phase_d_process_event(forged)


def _seed_evidence(
    config,
    official,
    *,
    root: Path,
    capability_passes: bool,
    normalized_tail_change: float = 0.0,
) -> dict:
    attempt_id = "pilot-attempt"
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed-budget-checkpoint")
    train_summary = _train_summary(
        config,
        official,
        checkpoint=checkpoint,
        attempt_id=attempt_id,
        normalized_tail_change=normalized_tail_change,
    )
    checkpoint_hash = file_sha256(checkpoint)
    assert checkpoint_hash is not None
    eval_summary = _eval_summary(
        config,
        checkpoint_hash=checkpoint_hash,
        outcomes=_outcomes(config, passes=capability_passes),
        attempt_id=attempt_id,
    )
    seed = build_stair_pilot_seed_evidence(
        config,
        train_summary=train_summary,
        eval_summary=eval_summary,
        train_process=_completed_process("train", attempt_id),
        eval_process=_completed_process("eval", attempt_id),
    )
    return validate_stair_pilot_seed_evidence(config.protocol_v2_path, seed)


def _eval_summary(config, *, checkpoint_hash: str, outcomes, attempt_id: str) -> dict:
    capability = build_stair_reference_capability_evidence(
        config.protocol_v1,
        protocol_v2=config.protocol_v2,
        evaluation_protocol=config.evaluation_protocol(),
        per_seed_outcomes=outcomes,
    )
    return build_stair_pilot_eval_summary(
        config,
        attempt_id=attempt_id,
        checkpoint_evaluation={
            "iteration": config.train_max_iterations - 1,
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_safety_report": _checkpoint_report(checkpoint_hash),
            "per_seed_outcomes": serialize_stair_pilot_raw_outcomes(outcomes),
            "capability_evidence": capability,
        },
    )


def _train_summary(
    config,
    official,
    *,
    checkpoint: Path,
    attempt_id: str,
    normalized_tail_change: float = 0.0,
) -> dict:
    return build_stair_pilot_train_summary(
        config,
        official=official,
        attempt_id=attempt_id,
        execution=_training_execution(config, official, checkpoint),
        curve_summary=_curve_summary(
            config,
            normalized_tail_change=normalized_tail_change,
        ),
    )


def _training_execution(config, official, checkpoint: Path):
    checksum = file_sha256(checkpoint)
    assert checksum is not None
    reward_program = official.reward_program.to_public_dict()
    term_names = reward_program["term_names"]
    dispatcher_calls = config.train_steps_per_iteration * config.train_max_iterations + 1
    activation_preflight = {
        "schema_version": (
            "true_rl_agent.task.stair_climbing.reward_activation_preflight.v2"
        ),
        "status": "pass",
        "real_transition_dispatcher_calls": 1,
        "transition_common_step": 1,
        "repeat_count": config.protocol_v2.complete_reward_runtime.dispatcher_contract[
            "determinism_preflight_repeats"
        ],
        "term_names": term_names,
    }
    return StairTrainingExecutionResult(
        status="pass",
        training_executed=True,
        checkpoint={
            "uri": str(checkpoint),
            "checksum": checksum,
            "role": "train_output",
            "metadata": {"local_path": str(checkpoint)},
        },
        runtime_activation_summary={
            "schema_version": (
                "true_rl_agent.task.stair_climbing."
                "complete_reward_runtime_evidence.v1"
            ),
            "status": "pass",
            "runtime_status": "active_pre_reset_complete_reward_v1",
            "dispatcher_calls": dispatcher_calls,
            "term_calls": dispatcher_calls * len(term_names),
            "declared_term_count": len(term_names),
            "reward_term_calls": dispatcher_calls * len(term_names),
            "diagnostic_calls": 0,
            "nonzero_reward_contribution": True,
            "program": reward_program,
            "input_guard": "pass",
            "rng_guard": "pass",
            "source_alias_guard": "pass",
            "capability_activation_records": ["complete_reward_program"],
            "component_activation_records": [
                {"name": name} for name in term_names
            ],
            "input_fingerprint": {
                "input_schema_hash": reward_program["input_schema_hash"]
            },
            "training_recorder_enabled": False,
            "reward_manager_active_terms": ["complete_reward"],
            "reward_manager_term_weight": 1.0,
            "termination_manager_resolution": (
                stair_termination_manager_resolution_contract(
                    config.protocol_v1.to_dict()
                )
            ),
            "activation_preflight": activation_preflight,
        },
        activation_preflight=activation_preflight,
        training={
            "budget_accounting": build_budget_accounting(
                budget_steps=config.budget_steps,
                num_envs=config.train_num_envs,
                num_steps_per_env=config.train_steps_per_iteration,
                max_iterations=config.train_max_iterations,
            ),
            "workload_kind": "production",
            "num_envs": config.train_num_envs,
            "num_steps_per_env": config.train_steps_per_iteration,
            "max_iterations": config.train_max_iterations,
            "budget_steps": config.budget_steps,
            "device": "cuda:1",
            "log_dir": str(checkpoint.parent / "logs"),
            "latest_checkpoint_iteration": config.train_max_iterations - 1,
            "initial_policy_state_hash": "1" * 64,
            "expected_initial_policy_state_hash": None,
            "training_distribution": _training_distribution_evidence(config),
            "execution_contract": {},
            "effective_agent_execution": _effective_agent_execution(config),
        },
        errors=(),
    )


def _curve_summary(config, *, normalized_tail_change: float) -> dict:
    contract = dict(config.protocol_v2.pilot_training_contract["curve_diagnostics"])
    contract["required_scalar_tags"] = list(contract["required_scalar_tags"])
    scalars = {}
    for tag in STAIR_PILOT_REQUIRED_SCALARS:
        observed = (
            normalized_tail_change
            if tag in contract["required_scalar_tags"]
            else 0.0
        )
        scalars[tag] = {
            "count": config.train_max_iterations,
            "first_step": 0,
            "last_step": config.train_max_iterations - 1,
            "first": 1.0,
            "last": 0.5,
            "minimum": 0.5,
            "maximum": 1.0,
            "nonfinite_count": 0,
            "tail": {
                "count": contract["tail_window_iterations"],
                "first_step": (
                    config.train_max_iterations - contract["tail_window_iterations"]
                ),
                "last_step": config.train_max_iterations - 1,
                "mean": 0.5,
                "population_stddev": 0.1,
                "slope_per_iteration": 0.0,
                "normalized_abs_tail_change": observed,
            },
        }
    return {
        "schema_version": STAIR_PILOT_CURVE_SUMMARY_SCHEMA_VERSION,
        "event_file_count": 1,
        "required_scalar_tags": list(STAIR_PILOT_REQUIRED_SCALARS),
        "available_scalar_tags": list(STAIR_PILOT_REQUIRED_SCALARS),
        "expected_last_iteration": config.train_max_iterations - 1,
        "diagnostics_contract": contract,
        "scalars": scalars,
        "tail_metrics": {
            tag: {"normalized_abs_tail_change": normalized_tail_change}
            for tag in contract["required_scalar_tags"]
        },
        "decision_authority": False,
        "status": "pass",
    }


def _outcomes(config, *, passes: bool):
    assignment = config.seed_resolver().private_bin_assignment()
    low_bins = {"stair_010", "stair_015", "stair_020"}
    return [
        tuple(
            StairEpisodeOutcome(
                env_index=env_index,
                bin_id=bin_id,
                episode_steps=600,
                primary_reason=(
                    "success" if passes and bin_id in low_bins else "timeout"
                ),
                success=passes and bin_id in low_bins,
                diagnostic_flags=(),
                max_progress_m=(
                    0.6 if passes and bin_id in low_bins else 0.1
                ),
                progress_fraction=(
                    0.6 if passes and bin_id in low_bins else 0.1
                ),
                active_terminal_terms=("time_out",),
            )
            for env_index, bin_id in enumerate(assignment)
        )
        for _ in range(config.public_accept_seed_count)
    ]


def _checkpoint_report(checksum: str) -> dict:
    return CheckpointSafetyReport(
        status="pass",
        checkpoint_hash=checksum,
        expected_checkpoint_hash=checksum,
        file_exists=True,
        byte_read_status="pass",
        weights_only_load_status="pass",
        checkpoint_hash_match_status="pass",
        loader_error_class=None,
        loader_error_summary=None,
        reason="checkpoint_safe_to_load",
    ).to_public_dict()


def _completed_process(phase: str, attempt_id: str) -> dict:
    return {
        "event_id": f"{attempt_id}:{phase}:completed",
        "phase": phase,
        "status": "completed",
        "has_exited": True,
        "exit_code": 0,
        "failure_message": None,
    }


def _environment_evidence(config) -> dict:
    hardware = {
        "platform": "test-platform",
        "machine": "test-machine",
        "device": "cuda:1",
        "cuda_available": True,
        "cuda_version": "test-cuda",
        "gpu_name": "test-gpu",
    }
    return {
        "schema_version": "true_rl_agent.task.stair_climbing.phase_d_environment.v4",
        "git_commit": "0" * 40,
        "git_worktree_scope": "true_RL_Agent",
        "git_worktree_clean": True,
        "python_version": "test-python",
        "python_executable_name": "python",
        "package_versions": {
            "torch": "test",
            "isaaclab": "test",
            "rsl-rl-lib": "test",
            "tensorboard": "test",
        },
        "upstream_training_distribution_sources": (
            stair_training_distribution_upstream_source_evidence(
                config.protocol_v2.runner_owned_training_distribution
            )
        ),
        "hardware": hardware,
        "hardware_fingerprint": canonical_json_hash(hardware),
    }


def _training_distribution_evidence(config) -> dict:
    return dict(
        resolved_stair_training_distribution(
            config.protocol_v2.runner_owned_training_distribution,
            base_protocol=config.protocol_v1.to_dict(),
        )
    )


def _effective_agent_execution(config) -> dict:
    execution = config.protocol_v2.to_public_dict()["training_execution"]
    return {
        "runner_contract_mode": "explicit",
        "observation_groups": execution["observation_groups"],
        "resolved_observation_groups": execution["observation_groups"],
        "clip_actions": execution["clip_actions"],
        "empirical_normalization": execution["empirical_normalization"],
        "logger": execution["logger"],
        "resume": execution["resume"],
    }


def _rehash(value: dict, hash_field: str) -> str:
    return canonical_json_hash(
        {key: item for key, item in value.items() if key != hash_field}
    )


if __name__ == "__main__":
    unittest.main()
