from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from scripts import run_stair_generation_session as stair_generation_script

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_MANIFEST_V2_SCHEMA_VERSION,
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
)
from true_rl_agent.agent_core.generation_lifecycle import GenerationSegmentJournal
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider,
)
from true_rl_agent.agent_core.provider_limits import (
    PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND,
)
from true_rl_agent.agent_core.session_genesis import (
    session_genesis_identity_hash,
)
from true_rl_agent.candidates.prompt_projection import build_prompt_payload
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.prompt_artifacts import PromptPayloadRef
from true_rl_agent.runner.job_attempts import (
    JobAttemptJournal,
    JobAttemptScope,
    PreGenesisExecutionScope,
)
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.job_argv import validate_eval_job_argv
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.stair_climbing import generation_session
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.generation_session import (
    STAIR_GENERATION_PREFLIGHT_SCHEMA_VERSION,
    STAIR_GENERATION_PROVIDER_THINKING_DISABLED_TEMPERATURE,
    STAIR_GENERATION_REPORT_FILENAME,
    STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME,
    STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION,
    STAIR_GENERATION_VALIDATION_REPORT_SCHEMA_VERSION,
    CompletedStairGenerationSession,
    StairGenerationPilotConfig,
    _assert_target_gpus_compute_idle,
    _attempt_is_in_generation_prefix,
    _build_resource_closure_evidence,
    _completed_scientific_generation_count,
    _create_foundation,
    _formal_normal_closed_generation_boundaries_are_valid,
    _generation_provider_readiness_required,
    _load_and_validate_existing_validation_report,
    _preflight_provider_credential,
    _provider_actuals,
    _provider_policy,
    _publish_or_load_parent_bootstrap_cost_boundary,
    _publish_or_validate_generation_closure,
    _publish_or_verify_source_checkout_identity,
    _reference_qualification_phase_reserves,
    _resource_summary_evidence,
    _run_one_generation,
    _run_validation_session,
    _session_genesis_attempt_scope,
    _stair_generation_provider_config,
    _StairCostGuardedProvider,
    _StairOperationAllocationState,
    _strict_operation_slot,
    _strict_retry_is_still_possible,
    _unified_validation_selector_reversal_metrics,
    _validated_job_cost_artifacts,
    _validation_authoring_metrics,
    _validation_workload_delta,
    _verify_source_checkout,
    discover_two_gpu_inventory,
    load_completed_stair_generation_session,
    preflight_stair_generation_pilot,
    preflight_stair_generation_provider_readiness,
    run_stair_generation_pilot,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_PILOT_MODE,
    DEVELOPMENT_GENERATION_VALIDATION_MODE,
    FORMAL_GENERATION_SESSION_MODE,
    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_OUTPUT_BYTES,
    STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES,
    StairGenerationSessionCostContractV2,
    StairGenerationSessionCostContractV4,
    StairGenerationSessionCostContractV5,
    build_stair_generation_scientific_cost_basis,
    stair_generation_operation_cost_reserves,
)
from true_rl_agent.tasks.stair_climbing.exploration import StairExplorationProjector
from true_rl_agent.tasks.stair_climbing.launch_context import (
    STAIR_GENERATION_LAUNCH_START_PATH_ENV,
    STAIR_GENERATION_LAUNCH_START_SHA256_ENV,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import (
    STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
    STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
)
from true_rl_agent.tasks.stair_climbing.phased_cost import (
    stair_phased_evaluation_cost_contract,
)

GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
SHA = "a" * 40
DIGEST = "b" * 64


class StairGenerationSessionTests(unittest.TestCase):
    def test_completed_session_loader_rejects_incomplete_root_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = (Path(tmp) / "completed").resolve()
            output_dir.mkdir()
            before = tuple(output_dir.iterdir())

            with self.assertRaisesRegex(FileNotFoundError, "missing required artifacts"):
                load_completed_stair_generation_session(
                    output_dir=output_dir,
                    expected_session_id="p6c-completed",
                    expected_cell_variant_id=STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
                    expected_source_commit=SHA,
                )

            self.assertEqual(tuple(output_dir.iterdir()), before)
            self.assertIn(
                "CompletedStairGenerationSession",
                generation_session.__all__,
            )
            self.assertIsNotNone(CompletedStairGenerationSession)

    def test_actual_cost_usage_collector_is_public_and_read_only(self) -> None:
        provider_actuals = {
            "attempts": 2,
            "request_units": 3,
            "response_units": 5,
            "total_units": 8,
            "reasoning_units": 1,
            "conservative_attempts": 1,
            "conservative_charges": ["provider-reserve"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            foundation = SimpleNamespace(
                config=SimpleNamespace(output_dir=output_dir),
                cost_contract=SimpleNamespace(
                    scientific_cost_basis=SimpleNamespace(),
                ),
            )
            before = tuple(output_dir.iterdir())
            with patch(
                "true_rl_agent.tasks.stair_climbing.generation_session."
                "_provider_actuals",
                return_value=provider_actuals,
            ):
                usage, charges, open_reservations = (
                    generation_session.collect_actual_cost_usage(
                        foundation,
                        genesis=None,
                        ledger=[],
                    )
                )

            self.assertIn("collect_actual_cost_usage", generation_session.__all__)
            self.assertEqual(usage.provider_attempts, 2)
            self.assertEqual(usage.provider_total_units, 8)
            self.assertEqual(usage.job_attempts, 0)
            self.assertEqual(charges, ("provider-reserve",))
            self.assertEqual(open_reservations, ())
            self.assertEqual(tuple(output_dir.iterdir()), before)

    def test_unified_terminal_selector_routes_public_accept_to_winner_only(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "tasks"
            / "stair_unified_protocol_v1.json"
        )
        foundation = SimpleNamespace(
            composition=SimpleNamespace(
                task_plugin=SimpleNamespace(
                    config=config,
                    objective_family_mask=lambda: {"frozen": True},
                )
            )
        )
        ledger = [
            {
                "generation_index": 1,
                "selection": {"selected_trial_id": "winner-1"},
                "trials": [
                    {
                        "trial_id": "winner-1",
                        "public_accept": {"receipt": "private"},
                    },
                    {
                        "trial_id": "loser-1",
                        "public_accept": None,
                    },
                ],
            },
            {
                "generation_index": 2,
                "selection": {"selected_trial_id": None},
                "trials": [
                    {"trial_id": "lane-a-2", "public_accept": None},
                    {"trial_id": "lane-b-2", "public_accept": None},
                ],
            },
        ]
        metrics = _unified_validation_selector_reversal_metrics(
            foundation,
            ledger=ledger,
        )

        self.assertEqual(
            metrics["comparison_source"],
            "selected_lane_only_public_accept_routing",
        )
        self.assertEqual(metrics["selected_confirmation_generation_count"], 1)
        self.assertEqual(metrics["no_selection_generation_count"], 1)
        self.assertEqual(metrics["routing_violation_count"], 0)
        self.assertEqual(metrics["status"], "pass")

    def test_unified_terminal_selector_rejects_no_selection_public_accept(
        self,
    ) -> None:
        config = StairClimbingTaskConfig.defaults(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "tasks"
            / "stair_unified_protocol_v1.json"
        )
        foundation = SimpleNamespace(
            composition=SimpleNamespace(
                task_plugin=SimpleNamespace(
                    config=config,
                    objective_family_mask=lambda: {"frozen": True},
                )
            )
        )
        metrics = _unified_validation_selector_reversal_metrics(
            foundation,
            ledger=[
                {
                    "generation_index": 1,
                    "selection": {"selected_trial_id": None},
                    "trials": [
                        {"trial_id": "lane-a", "public_accept": {}},
                        {"trial_id": "lane-b", "public_accept": None},
                    ],
                }
            ],
        )
        self.assertEqual(metrics["status"], "fail")
        self.assertEqual(metrics["routing_violation_count"], 1)

    def test_completed_scientific_generation_count_includes_no_selection(self) -> None:
        self.assertEqual(
            _completed_scientific_generation_count(
                {
                    "accepted": 4,
                    "formal_rejected": 7,
                    "no_selection": 4,
                    "paired_authoring_incomplete": 2,
                }
            ),
            15,
        )

    def test_formal_normal_boundaries_allow_continuous_or_optional_pauses(self) -> None:
        self.assertTrue(
            _formal_normal_closed_generation_boundaries_are_valid([15])
        )
        self.assertTrue(
            _formal_normal_closed_generation_boundaries_are_valid(
                [3, 6, 9, 12, 15]
            )
        )
        self.assertTrue(
            _formal_normal_closed_generation_boundaries_are_valid([6, 12, 15])
        )
        for invalid in ([], [3, 6, 9, 12], [4, 15], [3, 3, 15], [6, 3, 15]):
            with self.subTest(invalid=invalid):
                self.assertFalse(
                    _formal_normal_closed_generation_boundaries_are_valid(invalid)
                )

    def test_validation_workload_delta_includes_conservative_gpu_charge(self) -> None:
        fields = {
            "provider_attempts": 0,
            "provider_request_units": 0,
            "provider_response_units": 0,
            "provider_total_units": 0,
            "job_attempts": 0,
            "train_jobs": 0,
            "train_steps": 0,
            "eval_seed_units": 0,
            "simulated_control_steps": 0,
            "resource_lease_wall_duration_ns": 0,
            "conservatively_charged_gpu_capacity_ns": 0,
            "observed_output_bytes": 0,
        }
        current = {
            **fields,
            "resource_lease_wall_duration_ns": 1_800_000_000_000,
            "conservatively_charged_gpu_capacity_ns": 900_000_000_000,
        }

        delta = _validation_workload_delta(fields, current)

        self.assertEqual(
            delta["conservatively_charged_gpu_capacity_ns"], 900_000_000_000
        )
        self.assertEqual(delta["allocated_active_gpu_hours"], 0.75)

    def test_imported_reference_cost_excludes_source_reference_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            foundation = SimpleNamespace(
                config=SimpleNamespace(output_dir=output_dir),
            )
            source_private = output_dir.parent / "source" / "reference_eval_artifact.json"
            genesis = SimpleNamespace(
                reference_genesis_import_ref=object(),
                reference_eval_artifact_path=Mock(return_value=source_private),
            )

            train_steps, eval_costs, evidence_hash = (
                _validated_job_cost_artifacts(foundation, genesis=genesis)
            )

            self.assertEqual(train_steps, {})
            self.assertEqual(eval_costs, {})
            self.assertEqual(evidence_hash, canonical_json_hash([]))

    def _config(self, root: Path) -> StairGenerationPilotConfig:
        env_file = root / "provider.env"
        env_file.write_text(
            "DEEPSEEK_API_KEY=not-read-by-this-test\n", encoding="utf-8"
        )
        return StairGenerationPilotConfig(
            project_root=Path(__file__).resolve().parents[2],
            output_dir=(root / "session").resolve(),
            lease_root=(root / "leases").resolve(),
            session_id="p6b-generation-pilot",
            expected_source_commit=SHA,
            machine_id="test-machine",
            gpu0_uuid=GPU_A,
            gpu1_uuid=GPU_B,
            provider_env_file=env_file,
            max_allocated_active_gpu_hours=12.0,
            max_simulated_control_steps=1_000_000_000,
            max_output_bytes=40 * 1024**3,
        )

    def _cli_args(self, root: Path, mode: str) -> list[str]:
        return [
            "--output-dir",
            str(root / "formal-output"),
            "--session-id",
            "p6b-generation-pilot",
            "--expected-source-commit",
            SHA,
            "--machine-id",
            "test-machine",
            "--gpu0-uuid",
            GPU_A,
            "--gpu1-uuid",
            GPU_B,
            "--provider-env-file",
            str(root / "provider.env"),
            "--lease-root",
            str(root / "leases"),
            "--max-allocated-active-gpu-hours",
            "12",
            "--max-simulated-control-steps",
            "1000000000",
            "--max-output-bytes",
            str(40 * 1024**3),
            mode,
        ]

    def _formal_cli_args(self, root: Path) -> list[str]:
        argv = self._cli_args(root, "--formal-session")
        replacements = {
            "--max-allocated-active-gpu-hours": "50.5",
            "--max-simulated-control-steps": "6311042448",
            "--max-output-bytes": "14495514624",
        }
        for option, value in replacements.items():
            argv[argv.index(option) + 1] = value
        return argv

    def test_imported_reference_config_requires_one_absolute_source_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = self._config(root)
            source = (root / "qualified-reference").resolve()
            imported = replace(
                config,
                source_reference_genesis_dir=source,
                source_reference_session_id="qualified-reference-session",
            )
            self.assertTrue(imported.imports_reference_genesis)

            with self.assertRaisesRegex(ValueError, "provided together"):
                replace(config, source_reference_genesis_dir=source)
            with self.assertRaisesRegex(ValueError, "must be absolute"):
                replace(
                    config,
                    source_reference_genesis_dir=Path("relative-source"),
                    source_reference_session_id="qualified-reference-session",
                )
            with self.assertRaisesRegex(ValueError, "must differ"):
                replace(
                    config,
                    source_reference_genesis_dir=config.output_dir,
                    source_reference_session_id="qualified-reference-session",
                )

    def test_cli_passes_imported_reference_identity_without_changing_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = (root / "qualified-reference").resolve()
            argv = [
                *self._cli_args(root, "--pilot-session"),
                "--source-reference-genesis-dir",
                str(source),
                "--source-reference-session-id",
                "qualified-reference-session",
                "--cell-variant-id",
                STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
            ]
            with (
                patch.object(
                    stair_generation_script,
                    "run_stair_generation_pilot",
                    return_value={"checks": {"imported": True}},
                ) as run,
                patch("builtins.print"),
            ):
                self.assertEqual(stair_generation_script.main(argv), 0)

            config = run.call_args.args[0]
            self.assertEqual(config.source_reference_genesis_dir, source)
            self.assertEqual(
                config.source_reference_session_id,
                "qualified-reference-session",
            )
            self.assertEqual(
                config.cell_variant_id,
                STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
            )

            incomplete = [
                *self._cli_args(root, "--pilot-session"),
                "--source-reference-genesis-dir",
                str(source),
            ]
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                stair_generation_script.parse_stair_generation_session_args(incomplete)

    def _readiness_report(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.stair.provider_readiness.v1",
            "checked_at_utc": "2026-08-22T00:00:00Z",
            "endpoint": "https://api.deepseek.com/user/balance",
            "request_projection": {
                "method": "GET",
                "provider_balance_request_sent": True,
                "generation_request_sent": False,
            },
            "checks": {
                "authenticated_balance_response_valid": True,
                "provider_balance_available": True,
            },
            "account_projection": {
                "is_available": True,
                "currencies": ["CNY"],
            },
            "privacy": {
                "credential_persisted": False,
                "balance_amounts_persisted": False,
            },
            "side_effects": {
                "provider_attempt_charged_to_scientific_budget": False,
            },
        }

    def test_provider_policy_binds_both_adaptive_request_shapes(self) -> None:
        policy = _provider_policy()
        config = _stair_generation_provider_config(
            output_dir=Path("provider-output"),
            env_file=Path("provider.env"),
        )
        provider = OpenAICompatibleCandidateProvider(config, transport=Mock())
        initial_prompt = build_prompt_payload(public_context={"public_only": True})
        repair_prompt = {
            **initial_prompt,
            "provider_or_runtime_repair_feedback": {
                "failure_category": "json_parse_error",
                "failure": {
                    "category": "json_parse_error",
                    "details": {
                        "failure_kind": (PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND)
                    },
                },
            },
        }
        initial_request = provider._chat_completion_request(initial_prompt)
        repair_request = provider._chat_completion_request(repair_prompt)

        self.assertEqual(
            policy["schema_version"],
            "true_rl_agent.stair.generation_provider_policy.v2",
        )
        self.assertEqual(
            policy["thinking_enabled_request"],
            {
                "thinking_type": "enabled",
                "reasoning_effort": "max",
                "temperature_present": False,
            },
        )
        self.assertEqual(config.base_url, policy["base_url"])
        self.assertEqual(config.endpoint, policy["endpoint"])
        self.assertEqual(initial_request["model"], policy["model"])
        self.assertEqual(
            initial_request["response_format"]["type"],
            policy["response_format"],
        )
        self.assertEqual(initial_request["stream"], policy["stream"])
        self.assertEqual(initial_request["max_tokens"], policy["max_output_tokens"])
        self.assertEqual(config.timeout_s, policy["timeout_s"])
        self.assertEqual(
            {
                "thinking_type": initial_request["thinking"]["type"],
                "reasoning_effort": initial_request["reasoning_effort"],
                "temperature_present": "temperature" in initial_request,
            },
            policy["thinking_enabled_request"],
        )
        self.assertEqual(
            {
                "thinking_type": repair_request["thinking"]["type"],
                "reasoning_effort_present": ("reasoning_effort" in repair_request),
                "temperature": repair_request["temperature"],
            },
            policy["thinking_disabled_repair_request"],
        )
        self.assertEqual(
            policy["thinking_disabled_repair_request"],
            {
                "thinking_type": "disabled",
                "reasoning_effort_present": False,
                "temperature": (
                    STAIR_GENERATION_PROVIDER_THINKING_DISABLED_TEMPERATURE
                ),
            },
        )

    def test_preflight_only_round_trips_scratch_without_formal_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            forbidden = AssertionError("preflight crossed an execution boundary")
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_reject_parent_gpu_remap"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_verify_source_checkout",
                    return_value=SHA,
                ) as source,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "discover_two_gpu_inventory",
                    return_value=(GPU_A, GPU_B),
                ) as inventory,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "subprocess.run",
                    return_value=SimpleNamespace(stdout=""),
                ) as nvidia_smi,
                patch(
                    "true_rl_agent.runner.gpu_lease.GpuLeaseManager.acquire",
                    side_effect=forbidden,
                ),
                patch(
                    "true_rl_agent.runner.subprocess_runner.SubprocessRunner.run_train",
                    side_effect=forbidden,
                ),
                patch(
                    "true_rl_agent.runner.subprocess_runner.SubprocessRunner.run_eval",
                    side_effect=forbidden,
                ),
                patch(
                    "true_rl_agent.agent_core.openai_compatible_provider."
                    "OpenAICompatibleCandidateProvider.next_authoring_response",
                    side_effect=forbidden,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_require_stair_generation_provider_readiness",
                    side_effect=forbidden,
                ),
            ):
                report = preflight_stair_generation_pilot(config)

            self.assertEqual(
                report["schema_version"],
                STAIR_GENERATION_PREFLIGHT_SCHEMA_VERSION,
            )
            self.assertTrue(all(report["checks"].values()))
            self.assertFalse(any(report["side_effects"].values()))
            self.assertFalse(config.output_dir.exists())
            self.assertFalse(config.output_dir.is_symlink())
            self.assertFalse(config.lease_root.exists())
            source.assert_called_once_with(
                config.project_root,
                expected_commit=SHA,
            )
            inventory.assert_called_once_with(
                config.nvidia_smi_executable,
                expected=(GPU_A, GPU_B),
            )
            nvidia_smi.assert_called_once_with(
                [
                    config.nvidia_smi_executable,
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

    def test_preflight_only_cli_dispatches_without_session_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            report = {"checks": {"preflight_only": True}}
            argv = self._cli_args(root, "--preflight-only")
            with (
                patch.object(
                    stair_generation_script,
                    "preflight_stair_generation_pilot",
                    return_value=report,
                ) as preflight,
                patch.object(
                    stair_generation_script,
                    "run_stair_generation_pilot",
                ) as run,
                patch("builtins.print"),
            ):
                returncode = stair_generation_script.main(argv)

            self.assertEqual(returncode, 0)
            preflight.assert_called_once()
            run.assert_not_called()
            config = preflight.call_args.args[0]
            self.assertEqual(config.session_mode, DEVELOPMENT_GENERATION_PILOT_MODE)
            self.assertIsNone(config.pause_after_generation)
            self.assertFalse(config.terminal_audit_only)

    def test_phase4_pilot_and_validation_cli_map_fixed_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for flag, expected_mode, expected_generations in (
                ("--pilot-session", DEVELOPMENT_GENERATION_PILOT_MODE, 1),
                (
                    "--validation-session",
                    DEVELOPMENT_GENERATION_VALIDATION_MODE,
                    3,
                ),
            ):
                argv = self._cli_args(root, flag)
                if flag == "--validation-session":
                    replacements = {
                        "--max-allocated-active-gpu-hours": "12.5",
                        "--max-simulated-control-steps": "1577542128",
                        "--max-output-bytes": "3758096384",
                    }
                    for option, value in replacements.items():
                        argv[argv.index(option) + 1] = value
                with (
                    self.subTest(flag=flag),
                    patch.object(
                        stair_generation_script,
                        "run_stair_generation_pilot",
                        return_value={"checks": {"mode": flag}},
                    ) as run,
                    patch.object(
                        stair_generation_script,
                        "consume_stair_generation_launch_context",
                    ) as launch_context,
                    patch("builtins.print"),
                ):
                    self.assertEqual(stair_generation_script.main(argv), 0)
                    config = run.call_args.args[0]
                    self.assertEqual(config.session_mode, expected_mode)
                    self.assertEqual(config.max_generations, expected_generations)
                    self.assertEqual(
                        config.cell_variant_id,
                        STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
                    )
                    launch_context.assert_not_called()

    def test_generation_cli_selects_go2_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            argv = self._cli_args(root, "--pilot-session") + [
                "--cell-variant-id",
                STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
            ]
            with (
                patch.object(
                    stair_generation_script,
                    "run_stair_generation_pilot",
                    return_value={"checks": {"row": "go2"}},
                ) as run,
                patch("builtins.print"),
            ):
                self.assertEqual(stair_generation_script.main(argv), 0)

            self.assertEqual(
                run.call_args.args[0].cell_variant_id,
                STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
            )

    def test_formal_cli_maps_fixed_mode_and_precommitted_pause(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            report = {"checks": {"formal_session": True}}
            argv = self._formal_cli_args(root) + [
                "--pause-after-generation",
                "6",
            ]
            with (
                patch.object(
                    stair_generation_script,
                    "consume_stair_generation_launch_context",
                ) as launch_context,
                patch.object(
                    stair_generation_script,
                    "run_stair_generation_pilot",
                    return_value=report,
                ) as run,
                patch("builtins.print"),
            ):
                returncode = stair_generation_script.main(argv)

            self.assertEqual(returncode, 0)
            config = run.call_args.args[0]
            self.assertEqual(config.session_mode, FORMAL_GENERATION_SESSION_MODE)
            self.assertEqual(config.max_generations, 15)
            self.assertEqual(config.pause_after_generation, 6)
            self.assertEqual(config.max_allocated_active_gpu_hours, 50.5)
            self.assertEqual(config.max_simulated_control_steps, 6_311_042_448)
            self.assertEqual(config.max_output_bytes, 14_495_514_624)
            self.assertTrue(launch_context.call_args.kwargs["formal_session"])

            invalid = self._formal_cli_args(root) + [
                "--pause-after-generation",
                "2",
            ]
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                stair_generation_script.parse_stair_generation_session_args(invalid)

    def test_formal_terminal_audit_consumes_v3_launch_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            report = {"terminal_status": "completed"}
            argv = self._formal_cli_args(root) + [
                "--resume-existing",
                "--terminal-audit-only",
            ]
            with (
                patch.object(
                    stair_generation_script,
                    "consume_stair_generation_launch_context",
                ) as launch_context,
                patch.object(
                    stair_generation_script,
                    "run_stair_generation_pilot",
                    return_value=report,
                ),
                patch("builtins.print"),
            ):
                self.assertEqual(stair_generation_script.main(argv), 0)

            kwargs = launch_context.call_args.kwargs
            self.assertFalse(kwargs["helper_mode"])
            self.assertTrue(kwargs["formal_session"])
            self.assertTrue(kwargs["resume_existing"])
            self.assertTrue(kwargs["terminal_audit_only"])

    def test_cli_rejects_helper_control_conflicts_and_abbreviations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            base = self._cli_args(root, "--preflight-only")[:-1]
            invalid_suffixes = (
                (),
                ("--pause-after-generation", "1"),
                (
                    "--validation-session",
                    "--preflight-only",
                    "--pause-after-generation",
                    "1",
                ),
                ("--provider-readiness-only", "--terminal-audit-only"),
                ("--terminal-audit-only",),
                (
                    "--pilot-session",
                    "--resume-existing",
                    "--pause-after-generation",
                    "2",
                ),
                ("--validation-sess",),
            )
            with patch("sys.stderr"):
                for suffix in invalid_suffixes:
                    with self.subTest(suffix=suffix), self.assertRaises(SystemExit):
                        stair_generation_script.parse_stair_generation_session_args(
                            [*base, *suffix]
                        )

    def test_direct_fresh_formal_cli_requires_tracked_launcher_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            argv = [
                *self._cli_args(root, "--preflight-only")[:-1],
                "--formal-session",
            ]
            environment = dict(os.environ)
            environment.pop(STAIR_GENERATION_LAUNCH_START_PATH_ENV, None)
            environment.pop(STAIR_GENERATION_LAUNCH_START_SHA256_ENV, None)

            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(
                    stair_generation_script,
                    "run_stair_generation_pilot",
                ) as run,
                self.assertRaisesRegex(ValueError, "tracked launcher"),
            ):
                stair_generation_script.main(argv)

            run.assert_not_called()
            self.assertFalse((root / "formal-output").exists())

    def test_provider_readiness_only_has_no_formal_or_gpu_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            readiness = self._readiness_report()
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_reject_parent_gpu_remap"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_verify_source_checkout",
                    return_value=SHA,
                ) as source,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_require_stair_generation_provider_readiness",
                    return_value=readiness,
                ) as provider,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "discover_two_gpu_inventory"
                ) as inventory,
                patch(
                    "true_rl_agent.runner.gpu_lease.GpuLeaseManager.acquire"
                ) as lease,
            ):
                report = preflight_stair_generation_provider_readiness(config)

            self.assertTrue(all(report["checks"].values()))
            self.assertFalse(any(report["side_effects"].values()))
            self.assertFalse(config.output_dir.exists())
            self.assertEqual(report["session_id"], config.session_id)
            self.assertEqual(report["source_commit"], SHA)
            source.assert_called_once()
            provider.assert_called_once_with(config)
            inventory.assert_not_called()
            lease.assert_not_called()

    def test_provider_readiness_only_cli_dispatches_without_other_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            report = {
                **self._readiness_report(),
                "mode": "provider-readiness-only",
                "session_id": "p6b-generation-pilot",
                "source_commit": SHA,
            }
            argv = self._cli_args(root, "--provider-readiness-only")
            with (
                patch.object(
                    stair_generation_script,
                    "preflight_stair_generation_provider_readiness",
                    return_value=report,
                ) as readiness,
                patch.object(
                    stair_generation_script,
                    "preflight_stair_generation_pilot",
                ) as preflight,
                patch.object(
                    stair_generation_script,
                    "run_stair_generation_pilot",
                ) as run,
                patch("builtins.print") as emit,
            ):
                returncode = stair_generation_script.main(argv)

            self.assertEqual(returncode, 0)
            readiness.assert_called_once()
            preflight.assert_not_called()
            run.assert_not_called()
            emitted = json.loads(emit.call_args.args[0])
            self.assertEqual(emitted, report)

    def test_fresh_development_balance_failure_precedes_output_and_gpu_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_reject_parent_gpu_remap"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_verify_source_checkout",
                    return_value=SHA,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_require_stair_generation_provider_readiness",
                    side_effect=RuntimeError("provider unavailable"),
                ) as provider,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_publish_or_verify_source_checkout_identity"
                ) as source_identity,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "discover_two_gpu_inventory"
                ) as inventory,
                self.assertRaisesRegex(RuntimeError, "provider unavailable"),
            ):
                run_stair_generation_pilot(config)

            provider.assert_called_once_with(config)
            source_identity.assert_not_called()
            inventory.assert_not_called()
            self.assertFalse(config.output_dir.exists())

    def test_fresh_formal_session_does_not_call_network_provider_readiness(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                self._config(Path(tmp)),
                session_mode=FORMAL_GENERATION_SESSION_MODE,
            )
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_reject_parent_gpu_remap"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_verify_source_checkout",
                    return_value=SHA,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_require_stair_generation_provider_readiness",
                    side_effect=AssertionError(
                        "formal session attempted network provider readiness"
                    ),
                ) as provider,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_publish_or_verify_source_checkout_identity",
                    side_effect=RuntimeError("stop after formal provider gate"),
                ) as source_identity,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "discover_two_gpu_inventory"
                ) as inventory,
                self.assertRaisesRegex(
                    RuntimeError,
                    "stop after formal provider gate",
                ),
            ):
                run_stair_generation_pilot(config)

            provider.assert_not_called()
            source_identity.assert_called_once_with(
                config.output_dir,
                source_commit=SHA,
            )
            inventory.assert_not_called()
            self.assertFalse(config.output_dir.exists())

    def test_provider_readiness_decision_preserves_closed_resume_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            self.assertTrue(_generation_provider_readiness_required(config))

            formal = replace(
                config,
                session_mode=FORMAL_GENERATION_SESSION_MODE,
            )
            self.assertFalse(_generation_provider_readiness_required(formal))

            config.output_dir.mkdir(parents=True)
            resumed = replace(config, resume_existing=True)
            self.assertTrue(_generation_provider_readiness_required(resumed))

            ledger_path = config.output_dir / "ledger.jsonl"
            ledger_path.write_text("placeholder\n", encoding="utf-8")
            with patch(
                "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                return_value=[{"generation_outcome": "closed"}],
            ) as load:
                self.assertFalse(_generation_provider_readiness_required(resumed))
            load.assert_called_once_with(ledger_path)

            validation = replace(
                config,
                session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                resume_existing=True,
                pause_after_generation=1,
            )
            with patch(
                "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                return_value=[{"generation_outcome": "closed"}],
            ) as load:
                self.assertFalse(_generation_provider_readiness_required(validation))
            load.assert_called_once_with(ledger_path)

            (config.output_dir / "generation_session_report.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            with patch(
                "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger"
            ) as load:
                self.assertFalse(_generation_provider_readiness_required(resumed))
            load.assert_not_called()

    def test_foundation_freezes_two_host_bindings_and_fixed_pilot_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            _publish_or_verify_source_checkout_identity(
                config.output_dir,
                source_commit=SHA,
            )
            foundation = _create_foundation(
                config,
                source_commit=SHA,
                inventory=(GPU_A, GPU_B),
            )

            self.assertEqual(
                foundation.session_contract.schema_version,
                GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
            )
            self.assertEqual(
                foundation.session_contract.session_mode,
                DEVELOPMENT_GENERATION_PILOT_MODE,
            )
            self.assertEqual(foundation.session_contract.max_generations, 1)
            self.assertEqual(
                [item.host_execution_device for item in foundation.gpu_bindings],
                ["cuda:0", "cuda:1"],
            )
            self.assertEqual(
                [item.process_execution_device for item in foundation.gpu_bindings],
                ["cuda:0", "cuda:0"],
            )
            self.assertNotEqual(
                foundation.gpu_bindings[0].resource_binding_hash,
                foundation.gpu_bindings[1].resource_binding_hash,
            )
            self.assertEqual(foundation.cost_contract.successful_train_jobs, 3)
            self.assertEqual(foundation.cost_contract.successful_eval_seed_units, 38)
            self.assertEqual(foundation.cost_contract.physical_train_job_cap, 4)
            self.assertEqual(foundation.cost_contract.physical_eval_seed_unit_cap, 45)
            self.assertEqual(
                foundation.cost_contract.scientific_cost_basis.phased_evaluation_cost_contract_hash,
                foundation.phased_adapter.cost_contract()["contract_hash"],
            )
            self.assertFalse(foundation.final_confirmation_policy["enabled"])
            source_identity = json.loads(
                (
                    config.output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                source_identity["schema_version"],
                STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION,
            )
            self.assertEqual(source_identity["source_commit"], SHA)
            self.assertEqual(
                source_identity["source_checkout_hash"],
                canonical_json_hash(
                    {
                        "schema_version": (
                            STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION
                        ),
                        "source_commit": SHA,
                    }
                ),
            )

    def test_foundation_resolves_selected_go2_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                self._config(Path(tmp)),
                cell_variant_id=STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
            )
            _publish_or_verify_source_checkout_identity(
                config.output_dir,
                source_commit=SHA,
            )
            foundation = _create_foundation(
                config,
                source_commit=SHA,
                inventory=(GPU_A, GPU_B),
            )

            manifest = foundation.composition.resolved_cell_manifest
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertEqual(
                manifest.cell_variant_id,
                STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
            )
            projector = StairExplorationProjector(
                foundation.composition.task_plugin.config
            ).contract()
            self.assertEqual(
                foundation.session_contract.bound_contract_hashes[
                    "exploration_projector"
                ],
                canonical_json_hash(projector),
            )
            self.assertEqual(
                projector["task_protocol_hash"],
                manifest.resolved_cell_task_protocol_hash,
            )

    # Frozen P6 cost fixtures use their original 7200 s train deadline.
    # P2's current default is covered by composition/placement tests.
    @patch("true_rl_agent.tasks.stair_climbing.composition.DEFAULT_RUNNER_TIMEOUT_S", 7200)
    def test_validation_foundation_freezes_imported_reference_v4_cost(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                self._config(Path(tmp)),
                session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                max_allocated_active_gpu_hours=13.8,
                max_simulated_control_steps=1_183_555_008,
                max_output_bytes=3_758_096_384,
                source_reference_genesis_dir=(Path(tmp) / "source").resolve(),
                source_reference_session_id="qualified-reference",
            )
            _publish_or_verify_source_checkout_identity(
                config.output_dir,
                source_commit=SHA,
            )
            foundation = _create_foundation(
                config,
                source_commit=SHA,
                inventory=(GPU_A, GPU_B),
            )

            self.assertEqual(
                foundation.session_contract.session_mode,
                DEVELOPMENT_GENERATION_VALIDATION_MODE,
            )
            self.assertEqual(foundation.session_contract.max_generations, 3)
            self.assertEqual(foundation.session_contract.max_trials, 6)
            self.assertEqual(foundation.session_contract.max_train_jobs, 6)
            self.assertEqual(foundation.continuation_policy.max_segments, 4)
            self.assertIs(type(foundation.cost_contract), StairGenerationSessionCostContractV4)
            self.assertEqual(foundation.cost_contract.provider_attempt_cap, 30)
            self.assertEqual(foundation.cost_contract.physical_train_job_cap, 6)
            self.assertEqual(
                foundation.cost_contract.physical_eval_seed_unit_cap,
                76,
            )
            self.assertEqual(
                foundation.cost_contract.max_simulated_control_steps,
                1_183_555_008,
            )
            self.assertEqual(
                foundation.cost_contract.max_output_bytes,
                3_758_096_384,
            )
            operation_classes = {
                entry["operation_class"]
                for entry in foundation.cost_contract.to_dict()[
                    "operation_reserves"
                ]["entries"]
            }
            self.assertTrue(
                {
                    "reference_train",
                    "reference_qualification",
                    "parent_public_dev",
                    "reference_train_retry",
                }.isdisjoint(operation_classes)
            )
            self.assertEqual(
                foundation.cost_contract.candidate_train_process_timeout_s,
                foundation.runner.timeout_s,
            )

    @patch("true_rl_agent.tasks.stair_climbing.composition.DEFAULT_RUNNER_TIMEOUT_S", 7200)
    def test_formal_go2_foundation_freezes_imported_reference_v5_cost(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                self._config(Path(tmp)),
                session_mode=FORMAL_GENERATION_SESSION_MODE,
                cell_variant_id=STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
                max_allocated_active_gpu_hours=(
                    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
                    + STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
                ),
                max_simulated_control_steps=5_917_055_328,
                max_output_bytes=(
                    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_OUTPUT_BYTES
                    + STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_OUTPUT_BYTES
                ),
                source_reference_genesis_dir=(Path(tmp) / "source").resolve(),
                source_reference_session_id="qualified-go2-reference",
            )
            _publish_or_verify_source_checkout_identity(
                config.output_dir,
                source_commit=SHA,
            )
            foundation = _create_foundation(
                config,
                source_commit=SHA,
                inventory=(GPU_A, GPU_B),
            )

            self.assertIs(
                type(foundation.cost_contract),
                StairGenerationSessionCostContractV5,
            )
            self.assertTrue(foundation.cost_contract.imports_reference_evidence)
            self.assertEqual(foundation.cost_contract.physical_train_job_cap, 30)
            self.assertEqual(
                foundation.cost_contract.physical_eval_seed_unit_cap,
                366,
            )
            self.assertEqual(
                foundation.cost_contract.max_simulated_control_steps,
                5_917_055_328,
            )
            self.assertEqual(
                foundation.composition.resolved_cell_manifest.cell_variant_id,
                STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
            )
            self.assertNotIn(
                "final_confirmation_precommit",
                foundation.composition.task_defaults,
            )
            self.assertEqual(
                foundation.session_contract.bound_contract_hashes[
                    "final_confirmation_plan"
                ],
                foundation.final_confirmation_plan.plan_hash,
            )
            operation_classes = {
                entry["operation_class"]
                for entry in foundation.cost_contract.to_dict()[
                    "operation_reserves"
                ]["entries"]
            }
            self.assertTrue(
                {
                    "reference_train",
                    "reference_qualification",
                    "parent_public_dev",
                    "reference_train_retry",
                }.isdisjoint(operation_classes)
            )

    def test_formal_final_eval_rematerializes_imported_reference_gpu_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(
                self._config(root),
                cell_variant_id=STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
            )
            foundation = _create_foundation(
                config, source_commit=SHA, inventory=(GPU_A, GPU_B)
            )
            foundation = replace(
                foundation,
                final_confirmation_plan=SimpleNamespace(seed_cohort_hash="9" * 64),
            )
            composition = foundation.composition
            backend = composition.backend_adapter
            checkpoint_path = root / "best.pt"
            checkpoint_path.write_bytes(b"sealed-checkpoint-fixture")
            checkpoint_hash = "sha256:" + hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
            for previous_gpu in (None, foundation.gpu_bindings[1]):
                with self.subTest(previous_gpu=previous_gpu):
                    jobs = backend.materialize_reference_jobs(
                        session_id="source-reference",
                        task_spec=composition.task_plugin.task_spec,
                        reference=composition.reference_program_artifact,
                        artifact_root=root / "reference-artifacts",
                        policy_initialization=composition.policy_initialization,
                        backend_execution_environment_manifest_hash=DIGEST,
                        gpu_process_binding=previous_gpu,
                    )
                    spec = jobs.eval_spec
                    checkpoint = CheckpointRef(
                        uri=spec.checkpoint_output_plan.uri,
                        checksum=checkpoint_hash,
                        role="eval",
                        sealed=True,
                        metadata={
                            "local_path": str(checkpoint_path),
                            "training_subject": spec.subject.to_public_dict(),
                            "subject_manifest_hash": spec.subject_manifest_hash,
                            "program_source_checksum": spec.program_source_checksum,
                            "training_binding_hash": spec.training_binding_hash,
                            "capability_profile_hash": spec.capability_profile_hash,
                            "execution_device": spec.execution_device,
                            "gpu_process_binding": (
                                previous_gpu.to_dict() if previous_gpu else None
                            ),
                        },
                    )
                    template = bind_eval_spec_to_checkpoint(spec, checkpoint)
                    original_argv = list(template.metadata["cli_argv"])
                    self.assertEqual(
                        "--gpu-host-execution-device" in original_argv,
                        previous_gpu is not None,
                    )
                    target = replace(
                        template,
                        subject=ExperimentSubject.candidate("accepted-candidate"),
                        subject_manifest_hash="c" * 64,
                        program_source_checksum="sha256:" + "d" * 64,
                        training_binding_hash="e" * 64,
                    )
                    bound = generation_session._bind_formal_final_eval_plan(
                        foundation,
                        template=template,
                        target=target,
                        role="final",
                        base_job_id="formal-final-confirmation-final-eval",
                    )
                    phase = backend.materialize_eval_phase(
                        eval_plan=bound, phase=EvaluationPhase.SEALED_VALIDATION
                    )
                    argv = phase.metadata["cli_argv"]
                    validate_eval_job_argv(phase, argv)
                    expected_options = {
                        "--subject-kind": "candidate",
                        "--subject-id": "accepted-candidate",
                        "--subject-manifest-hash": target.subject_manifest_hash,
                        "--program-source-checksum": target.program_source_checksum,
                        "--training-binding-hash": target.training_binding_hash,
                        "--checkpoint-sha256": checkpoint_hash,
                        "--seed-cohort-hash": "9" * 64,
                        "--gpu-host-execution-device": "cuda:0",
                        "--gpu-process-execution-device": "cuda:0",
                        "--gpu-resource-binding-hash": (
                            foundation.gpu_bindings[0].resource_binding_hash
                        ),
                        "--stair-eval-purpose": "final_confirmation",
                        "--backend-execution-environment-manifest-hash": DIGEST,
                    }
                    for option, value in expected_options.items():
                        self.assertEqual(argv.count(option), 1, option)
                        self.assertEqual(argv[argv.index(option) + 1], value, option)
                    self.assertEqual(
                        bound.metadata["evaluation_subject"],
                        target.subject.to_public_dict(),
                    )
                    for name in (
                        "subject_manifest_hash", "program_source_checksum",
                        "training_binding_hash", "capability_profile_hash",
                    ):
                        self.assertEqual(bound.metadata[name], getattr(target, name))
                    self.assertEqual(template.metadata["cli_argv"], original_argv)
                    self.assertEqual(bound.checkpoint, target.checkpoint)
                    self.assertIsNone(bound.attempt_scope)

    def _final_source_fixture(self, root: Path, executable_commit: str):
        config = replace(self._config(root), expected_source_commit=executable_commit)
        source = _publish_or_verify_source_checkout_identity(
            config.output_dir, source_commit=SHA
        )
        final_dir = config.output_dir / generation_session.STAIR_FINAL_CONFIRMATION_DIRNAME
        final_dir.mkdir()
        return (
            SimpleNamespace(config=config, source_commit=SHA),
            source,
            final_dir / generation_session._FINAL_SOURCE_CHECKOUT_FILENAME,
        )

    def test_final_execution_source_preserves_old_receipt_without_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            foundation, source, sidecar = self._final_source_fixture(Path(tmp), SHA)
            original = (foundation.config.output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME).read_bytes()
            for publish in (False, True):
                self.assertEqual(
                    generation_session._formal_final_execution_source(
                        foundation, publish=publish
                    ),
                    source,
                )
            self.assertFalse(sidecar.exists())
            self.assertEqual(
                (foundation.config.output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME).read_bytes(),
                original,
            )

    def test_final_execution_source_repair_hash_binds_unchanged_generation_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            foundation, source, sidecar = self._final_source_fixture(Path(tmp), "c" * 40)
            original_path = foundation.config.output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME
            original = original_path.read_bytes()
            recorded = generation_session._formal_final_execution_source(
                foundation, publish=True
            )
            body = dict(recorded)
            checksum = body.pop("source_checkout_hash")
            self.assertEqual(checksum, canonical_json_hash(body))
            self.assertEqual(body["source_commit"], "c" * 40)
            self.assertEqual(body["generation_source_checkout_hash"], source["source_checkout_hash"])
            self.assertEqual(sidecar.read_bytes(), canonical_json_bytes(recorded, trailing_newline=True))
            self.assertEqual(original_path.read_bytes(), original)
            self.assertEqual(
                generation_session._formal_final_execution_source(foundation, publish=True),
                recorded,
            )

    def test_final_execution_source_cannot_change_after_legacy_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            foundation, _, sidecar = self._final_source_fixture(Path(tmp), "c" * 40)
            reservation_path = generation_session.stair_final_confirmation_reservation_path(
                foundation.config.output_dir
            )
            reservation_path.write_bytes(b"{}\n")
            with self.assertRaisesRegex(ValueError, "cannot change after reservation"):
                generation_session._formal_final_execution_source(foundation, publish=True)
            self.assertFalse(sidecar.exists())

    def test_final_execution_source_bound_repair_is_immutable_but_audit_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            foundation, _, sidecar = self._final_source_fixture(Path(tmp), "c" * 40)
            recorded = generation_session._formal_final_execution_source(foundation, publish=True)
            recorded_bytes = sidecar.read_bytes()
            later = SimpleNamespace(
                config=replace(foundation.config, expected_source_commit="d" * 40),
                source_commit=SHA,
            )
            with self.assertRaisesRegex(ValueError, "finalizer source checkout mismatch"):
                generation_session._formal_final_execution_source(later, publish=True)
            self.assertEqual(
                generation_session._formal_final_execution_source(later), recorded
            )
            self.assertEqual(sidecar.read_bytes(), recorded_bytes)

    def test_final_execution_source_rejects_rehashed_wrong_generation_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            foundation, _, sidecar = self._final_source_fixture(Path(tmp), "c" * 40)
            recorded = dict(generation_session._formal_final_execution_source(foundation, publish=True))
            recorded.pop("source_checkout_hash")
            recorded["generation_source_checkout_hash"] = "e" * 64
            recorded["source_checkout_hash"] = canonical_json_hash(recorded)
            sidecar.write_bytes(canonical_json_bytes(recorded, trailing_newline=True))
            with self.assertRaisesRegex(ValueError, "finalizer source checkout mismatch"):
                generation_session._formal_final_execution_source(foundation)

    @patch("true_rl_agent.tasks.stair_climbing.composition.DEFAULT_RUNNER_TIMEOUT_S", 7200)
    def test_formal_source_change_rejects_real_nonterminal_foundation_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = replace(
                self._config(root),
                session_mode=FORMAL_GENERATION_SESSION_MODE,
                cell_variant_id=STAIR_GO2_12DOF_UNIFIED_CELL_VARIANT_ID,
                max_allocated_active_gpu_hours=(
                    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
                    + STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
                ),
                max_simulated_control_steps=5_917_055_328,
                max_output_bytes=(
                    STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_SUCCESSFUL_OUTPUT_BYTES
                    + STAIR_GENERATION_IMPORTED_REFERENCE_FORMAL_RETRY_OUTPUT_BYTES
                ),
                source_reference_genesis_dir=(root / "source").resolve(),
                source_reference_session_id="qualified-go2-reference",
            )
            _publish_or_verify_source_checkout_identity(
                config.output_dir, source_commit=SHA
            )
            _create_foundation(config, source_commit=SHA, inventory=(GPU_A, GPU_B))
            before = {
                str(path.relative_to(config.output_dir)): (
                    path.read_bytes() if path.is_file() else None
                )
                for path in config.output_dir.rglob("*")
            }
            self.assertNotIn("session_genesis.json", before)
            self.assertIn("final_confirmation/plan.json", {
                key.replace("\\", "/") for key in before
            })
            repaired_config = replace(
                config, resume_existing=True, expected_source_commit="c" * 40
            )
            with self.assertRaisesRegex(ValueError, "terminal readiness requires Session Genesis"):
                generation_session._formal_resume_generation_source(
                    repaired_config, "c" * 40
                )
            after = {
                str(path.relative_to(config.output_dir)): (
                    path.read_bytes() if path.is_file() else None
                )
                for path in config.output_dir.rglob("*")
            }
            self.assertEqual(after, before)

    def test_terminal_validation_audit_is_offline_before_gpu_and_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                self._config(Path(tmp)),
                session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                max_allocated_active_gpu_hours=12.5,
                max_simulated_control_steps=1_577_542_128,
                max_output_bytes=3_758_096_384,
                resume_existing=True,
                terminal_audit_only=True,
            )
            config.output_dir.mkdir(parents=True)
            for filename in (
                "generation_source_checkout.json",
                "generation_session_cost.json",
                "generation_session_contract.json",
                "generation_continuation_policy.json",
                "generation_session_report.json",
            ):
                (config.output_dir / filename).write_text("{}\n", encoding="utf-8")
            foundation = SimpleNamespace(
                config=config,
                cost_contract=Mock(),
            )
            genesis = Mock()
            expected = {"checks": {"offline_terminal_audit": True}}
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_reject_parent_gpu_remap"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_verify_source_checkout",
                    return_value=SHA,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_publish_or_verify_source_checkout_identity"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_create_foundation",
                    return_value=foundation,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_load_existing_session_genesis_for_terminal_audit",
                    return_value=genesis,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "StairGenerationActualCostJournal",
                    return_value=Mock(),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_load_and_validate_existing_validation_report",
                    return_value=expected,
                ) as validate,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_require_stair_generation_provider_readiness"
                ) as provider,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "discover_two_gpu_inventory"
                ) as inventory,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_assert_target_gpus_compute_idle"
                ) as idle,
            ):
                actual = run_stair_generation_pilot(config)

            self.assertEqual(actual, expected)
            provider.assert_not_called()
            inventory.assert_not_called()
            idle.assert_not_called()
            self.assertIsNone(validate.call_args.kwargs["startup_resource_observation"])

    def test_terminal_validation_audit_offline_finalizes_a_missing_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                self._config(Path(tmp)),
                session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                max_allocated_active_gpu_hours=12.5,
                max_simulated_control_steps=1_577_542_128,
                max_output_bytes=3_758_096_384,
                resume_existing=True,
                terminal_audit_only=True,
            )
            config.output_dir.mkdir(parents=True)
            for filename in (
                "generation_source_checkout.json",
                "generation_session_cost.json",
                "generation_session_contract.json",
                "generation_continuation_policy.json",
            ):
                (config.output_dir / filename).write_text("{}\n", encoding="utf-8")
            foundation = SimpleNamespace(
                config=config,
                cost_contract=Mock(),
            )
            genesis = Mock()
            cost_journal = Mock()
            expected = {"checks": {"offline_terminal_finalize": True}}
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_reject_parent_gpu_remap"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_verify_source_checkout",
                    return_value=SHA,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_publish_or_verify_source_checkout_identity"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_create_foundation",
                    return_value=foundation,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_load_existing_session_genesis_for_terminal_audit",
                    return_value=genesis,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "StairGenerationActualCostJournal",
                    return_value=cost_journal,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_finalize_missing_terminal_validation_report",
                    return_value=expected,
                ) as finalize,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_load_and_validate_existing_validation_report"
                ) as validate,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_require_stair_generation_provider_readiness"
                ) as provider,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "discover_two_gpu_inventory"
                ) as inventory,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_assert_target_gpus_compute_idle"
                ) as idle,
            ):
                actual = run_stair_generation_pilot(config)

            self.assertEqual(actual, expected)
            finalize.assert_called_once_with(
                foundation,
                genesis=genesis,
                cost_journal=cost_journal,
            )
            validate.assert_not_called()
            provider.assert_not_called()
            inventory.assert_not_called()
            idle.assert_not_called()

    def test_source_checkout_identity_is_create_once_and_rejects_commit_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "session"
            first = _publish_or_verify_source_checkout_identity(
                output_dir,
                source_commit=SHA,
            )
            second = _publish_or_verify_source_checkout_identity(
                output_dir,
                source_commit=SHA,
            )
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                _publish_or_verify_source_checkout_identity(
                    output_dir,
                    source_commit="b" * 40,
                )

    def test_terminal_replay_rejects_missing_closure_without_writing_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            record = {
                "generation_id": "p6c-session-generation-001",
                "generation_index": 1,
                "resulting_current_best_hash": "a" * 64,
            }
            foundation = SimpleNamespace(
                config=SimpleNamespace(
                    output_dir=output_dir,
                    session_id="p6c-session",
                )
            )
            actual_cost = SimpleNamespace(
                path=output_dir / "cost.json",
                payload_hash="b" * 64,
                payload={"event_id": "generation-001-settlement"},
            )

            with self.assertRaisesRegex(FileNotFoundError, "closure is missing"):
                _publish_or_validate_generation_closure(
                    foundation,
                    ledger=[record],
                    generation_record=record,
                    generation_execution_receipt={"kind": "execution"},
                    actual_cost_receipt=actual_cost,
                    resource_evidence={},
                    segment_id="offline-audit",
                    require_existing=True,
                )

            self.assertFalse((output_dir / "generation_closures").exists())

    def test_offline_validation_report_uses_strict_lifecycle_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            report = {
                "schema_version": STAIR_GENERATION_VALIDATION_REPORT_SCHEMA_VERSION,
                "created_at_utc": "2026-08-22T00:00:00+00:00",
                "resource_evidence": {"stored": "resource"},
            }
            (output_dir / STAIR_GENERATION_REPORT_FILENAME).write_bytes(
                canonical_json_bytes(report, trailing_newline=True)
            )
            acceptance = Mock()
            foundation = SimpleNamespace(
                config=SimpleNamespace(output_dir=output_dir),
                session_contract=SimpleNamespace(max_generations=3),
                continuation_policy=Mock(),
                composition=SimpleNamespace(
                    task_plugin=SimpleNamespace(
                        task_spec=SimpleNamespace(acceptance_protocol=acceptance),
                        config=SimpleNamespace(is_unified_protocol=False),
                    )
                ),
            )
            genesis = SimpleNamespace(
                initial_current_best=Mock(),
                expected_initial_policy_state_hash=None,
            )
            ledger = [
                {"generation_id": f"p6c-session-generation-{index:03d}"}
                for index in range(1, 4)
            ]
            lifecycle = SimpleNamespace(terminal=True)
            terminal = Mock(to_dict=Mock(return_value={"terminal": True}))
            cost_ref = Mock()
            cost_journal = Mock()
            cost_journal.load_chain.return_value = [cost_ref]

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                    return_value=ledger,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_validation_lifecycle_identity",
                    return_value=Mock(),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "audit_generation_lifecycle",
                    return_value=lifecycle,
                ) as audit_lifecycle,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "build_generation_terminal_binding",
                    return_value=terminal,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_assert_validation_cost_receipt_is_current"
                ) as assert_cost,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_validate_stored_resource_closure_evidence"
                ) as validate_resource,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_build_validation_report",
                    return_value=report,
                ),
            ):
                actual = _load_and_validate_existing_validation_report(
                    foundation,
                    genesis=genesis,
                    cost_journal=cost_journal,
                    startup_resource_observation=None,
                )

            self.assertEqual(actual, report)
            audit_lifecycle.assert_called_once_with(
                output_dir=output_dir,
                identity=unittest.mock.ANY,
                policy=foundation.continuation_policy,
                require_closed=True,
            )
            self.assertIs(assert_cost.call_args.kwargs["lifecycle"], lifecycle)
            validate_resource.assert_called_once_with(
                foundation,
                resource_evidence=report["resource_evidence"],
                generation_index=3,
            )

    def test_pause_one_closure_replays_its_exact_resource_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            session_id = "p6c-session"
            binding_hash = "1" * 64
            session_contract_hash = "2" * 64
            cost_contract_hash = "3" * 64
            binding = Mock(resource_binding_hash=binding_hash)
            binding.to_dict.return_value = {
                "host_execution_device": "cuda:0",
                "process_execution_device": "cuda:0",
                "resource_binding_hash": binding_hash,
            }
            foundation = SimpleNamespace(
                config=SimpleNamespace(
                    output_dir=output_dir,
                    session_id=session_id,
                    gpu0_uuid=GPU_A,
                    gpu1_uuid=GPU_B,
                    nvidia_smi_executable="unused-nvidia-smi",
                ),
                composition=SimpleNamespace(
                    task_plugin=SimpleNamespace(
                        config=SimpleNamespace(is_unified_protocol=False)
                    )
                ),
                session_contract=SimpleNamespace(
                    session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                    max_generations=3,
                    contract_hash=session_contract_hash,
                ),
                cost_contract=SimpleNamespace(
                    contract_hash=cost_contract_hash,
                    max_generations=3,
                    imports_reference_evidence=False,
                    imported_reference_timeout_envelope=None,
                    scientific_cost_basis=SimpleNamespace(
                        reuses_reference_public_dev_receipt=False
                    ),
                ),
                gpu_bindings=(binding, Mock(resource_binding_hash="4" * 64)),
                gpu_identities=(
                    SimpleNamespace(physical_id=GPU_A),
                    SimpleNamespace(physical_id=GPU_B),
                ),
            )
            foundation.gpu_bindings[1].to_dict.return_value = {
                "host_execution_device": "cuda:1",
                "process_execution_device": "cuda:0",
                "resource_binding_hash": "4" * 64,
            }
            summary_paths: dict[str, Path] = {}

            def add_completed_attempt(
                *,
                logical_job_id: str,
                job_kind: str,
                scope: JobAttemptScope | PreGenesisExecutionScope,
            ) -> None:
                journal = JobAttemptJournal(
                    scope=scope,
                    logical_job_id=logical_job_id,
                    job_kind=job_kind,
                    job_input_hash="5" * 64,
                    environment_hash="6" * 64,
                    execution_device="cuda:0",
                    resource_binding_hash=binding_hash,
                )
                reservation = journal.reserve(0, previous_result=None)
                journal.finish(
                    reservation,
                    outcome="completed",
                    failure_kind=None,
                    retryable=False,
                    output_hash="7" * 64,
                    returncode=0,
                    resource_lease_wall_duration_ns=1,
                )
                summary_path = output_dir / "runner" / logical_job_id / "summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "job_id": logical_job_id,
                            "gpu_process_binding": binding.to_dict(),
                        }
                    ),
                    encoding="utf-8",
                )
                summary_paths[logical_job_id] = summary_path

            pregenesis_common = {
                "journal_root": output_dir / "job_attempts",
                "session_id": session_id,
                "genesis_identity_hash": "8" * 64,
                "session_contract_hash": session_contract_hash,
                "lane_id": "lane-a",
                "invocation_segment_id": "session-genesis",
                "budget_contract_hash": cost_contract_hash,
                "resource_binding_hash": binding_hash,
            }
            add_completed_attempt(
                logical_job_id=f"{session_id}-train",
                job_kind="train",
                scope=PreGenesisExecutionScope(**pregenesis_common),
            )
            add_completed_attempt(
                logical_job_id=f"{session_id}-eval",
                job_kind="eval",
                scope=PreGenesisExecutionScope(
                    **pregenesis_common,
                    attempt_budget_units=1,
                    max_infrastructure_retries=0,
                ),
            )
            add_completed_attempt(
                logical_job_id=f"{session_id}-eval-public-dev",
                job_kind="eval_phase",
                scope=JobAttemptScope(
                    journal_root=output_dir / "job_attempts",
                    session_genesis_hash="9" * 64,
                    session_contract_hash=session_contract_hash,
                    generation_id="generation-parent-public-dev-bootstrap",
                    lane_id="lane-a",
                    invocation_segment_id="segment-001",
                    budget_contract_hash=cost_contract_hash,
                    attempt_budget_units=1,
                    max_infrastructure_retries=0,
                    resource_binding_hash=binding_hash,
                ),
            )

            def generation_train_scope(index: int) -> JobAttemptScope:
                return JobAttemptScope(
                    journal_root=output_dir / "job_attempts",
                    session_genesis_hash="9" * 64,
                    session_contract_hash=session_contract_hash,
                    generation_id=f"{session_id}-generation-{index:03d}",
                    lane_id="lane-a",
                    invocation_segment_id=f"segment-{index:03d}",
                    budget_contract_hash=cost_contract_hash,
                    attempt_budget_units=1,
                    max_infrastructure_retries=0,
                    resource_binding_hash=binding_hash,
                )

            generation_one_job = f"{session_id}-generation-001-lane-a-train"
            add_completed_attempt(
                logical_job_id=generation_one_job,
                job_kind="train",
                scope=generation_train_scope(1),
            )
            startup = {
                "schema_version": "true_rl_agent.stair.generation_compute_idle.v1",
                "stage": "startup",
                "source": "nvidia-smi-query-compute-apps",
                "target_gpu_uuids": [GPU_A, GPU_B],
                "rows": [],
            }
            final_idle = {**startup, "stage": "final"}
            lease_reacquire = {
                "passed": True,
                "simultaneous": True,
                "resource_binding_hashes": [binding_hash, "4" * 64],
            }

            def validate_summary(summary: dict[str, Any], **_: Any) -> dict[str, Any]:
                return {
                    "gpu_process_binding": summary["gpu_process_binding"],
                    "owned_process_identity": {
                        "process_id": 101,
                        "process_group_id": 101,
                    },
                    "stair_gpu_runtime_evidence": {
                        "external_compute_context_observation": {
                            "process_id": 101,
                            "rows": [{"gpu_uuid": GPU_A}],
                        }
                    },
                }

            record = {
                "generation_id": f"{session_id}-generation-001",
                "generation_index": 1,
                "resulting_current_best_hash": "a" * 64,
            }
            cost_receipt = SimpleNamespace(
                path=output_dir / "cost.json",
                payload_hash="b" * 64,
                payload={"event_id": "generation-001-settlement"},
            )
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_assert_target_gpus_compute_idle",
                    return_value=final_idle,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_reacquire_generation_gpu_leases",
                    return_value=lease_reacquire,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "validate_stair_gpu_runtime_summary_evidence",
                    side_effect=validate_summary,
                ),
            ):
                resource_evidence = _build_resource_closure_evidence(
                    foundation,
                    genesis=SimpleNamespace(session_genesis_hash="9" * 64),
                    invocation_segment_id="segment-001",
                    startup_observation=startup,
                    generation_index=1,
                )
                self.assertEqual(resource_evidence["generation_index"], 1)
                self.assertEqual(resource_evidence["completed_bound_job_count"], 4)
                self.assertEqual(
                    resource_evidence["attempt_chain"]["reservation_count"], 4
                )
                closure = _publish_or_validate_generation_closure(
                    foundation,
                    ledger=[record],
                    generation_record=record,
                    generation_execution_receipt={"kind": "execution"},
                    actual_cost_receipt=cost_receipt,
                    resource_evidence=resource_evidence,
                    segment_id="segment-001",
                )

                generation_one_summary = summary_paths[generation_one_job]
                generation_one_summary_bytes = generation_one_summary.read_bytes()
                generation_one_summary.unlink()
                with self.assertRaisesRegex(
                    ValueError,
                    "completed bound jobs lack unique resource summaries",
                ):
                    _publish_or_validate_generation_closure(
                        foundation,
                        ledger=[record],
                        generation_record=record,
                        generation_execution_receipt={"kind": "execution"},
                        actual_cost_receipt=cost_receipt,
                        resource_evidence={},
                        segment_id="resume-before-generation-002",
                        require_existing=True,
                    )
                generation_one_summary.write_bytes(generation_one_summary_bytes)

                generation_two_job = f"{session_id}-generation-002-lane-a-train"
                generation_two_journal = JobAttemptJournal(
                    scope=generation_train_scope(2),
                    logical_job_id=generation_two_job,
                    job_kind="train",
                    job_input_hash="5" * 64,
                    environment_hash="6" * 64,
                    execution_device="cuda:0",
                    resource_binding_hash=binding_hash,
                )
                generation_two_reservation = generation_two_journal.reserve(
                    0,
                    previous_result=None,
                )
                generation_two_summary = (
                    output_dir / "runner" / generation_two_job / "summary.json"
                )
                generation_two_summary.parent.mkdir(parents=True, exist_ok=True)
                generation_two_summary.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "job_id": generation_two_job,
                            "gpu_process_binding": binding.to_dict(),
                        }
                    ),
                    encoding="utf-8",
                )
                self.assertEqual(
                    _publish_or_validate_generation_closure(
                        foundation,
                        ledger=[record],
                        generation_record=record,
                        generation_execution_receipt={"kind": "execution"},
                        actual_cost_receipt=cost_receipt,
                        resource_evidence={},
                        segment_id="later-segment",
                        require_existing=True,
                    ),
                    closure,
                )
                generation_two_journal.finish(
                    generation_two_reservation,
                    outcome="infrastructure_failure",
                    failure_kind="abandoned_unclean",
                    retryable=False,
                    output_hash=None,
                    returncode=None,
                    recovery_receipt_hash="c" * 64,
                )
                self.assertEqual(
                    _publish_or_validate_generation_closure(
                        foundation,
                        ledger=[record],
                        generation_record=record,
                        generation_execution_receipt={"kind": "execution"},
                        actual_cost_receipt=cost_receipt,
                        resource_evidence={},
                        segment_id="later-recovery-segment",
                        require_existing=True,
                    ),
                    closure,
                )

                unknown_summary = output_dir / "runner" / "unknown" / "summary.json"
                unknown_summary.parent.mkdir(parents=True)
                unknown_summary.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "job_id": "unknown-bound-job",
                            "gpu_process_binding": binding.to_dict(),
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "passing bound summary has no completed job attempt",
                ):
                    _publish_or_validate_generation_closure(
                        foundation,
                        ledger=[record],
                        generation_record=record,
                        generation_execution_receipt={"kind": "execution"},
                        actual_cost_receipt=cost_receipt,
                        resource_evidence={},
                        segment_id="unknown-summary-segment",
                        require_existing=True,
                    )

    def test_resume_replays_closed_prefix_before_new_generation_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            generation_id = "p6c-session-generation-001"
            closure_path = output_dir / "generation_closures" / f"{generation_id}.json"
            closure_path.parent.mkdir(parents=True)
            closure_path.write_text("{}\n", encoding="utf-8")
            parent = Mock()
            guard = Mock()
            record = {"generation_id": generation_id, "generation_index": 1}
            foundation = SimpleNamespace(
                config=SimpleNamespace(
                    output_dir=output_dir,
                    session_id="p6c-session",
                    pause_after_generation=None,
                ),
                composition=SimpleNamespace(
                    task_plugin=SimpleNamespace(
                        task_spec=SimpleNamespace(acceptance_protocol=Mock()),
                        config=SimpleNamespace(is_unified_protocol=False),
                    ),
                    backend_adapter=Mock(),
                ),
                session_contract=SimpleNamespace(
                    session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                    max_generations=3,
                    contract_hash="1" * 64,
                    bound_contract_hashes={"final_confirmation_plan": "2" * 64},
                ),
                continuation_policy=Mock(),
                cost_contract=Mock(),
                phased_adapter=Mock(),
                runner=Mock(),
                gpu_bindings=(Mock(), Mock()),
            )
            genesis = SimpleNamespace(
                session_genesis_hash="3" * 64,
                initial_current_best=parent,
                expected_initial_policy_state_hash=None,
            )
            lifecycle = SimpleNamespace(terminal=False, active_segment_id=None)
            segment = Mock(segment_id="resume-segment", cost_chain_head_hash="4" * 64)
            boundary_cost = SimpleNamespace(payload_hash="5" * 64)
            bootstrap_cost = SimpleNamespace(
                payload_hash="6" * 64,
                payload={"cap_checks": {"all_passed": True}},
            )
            failed_cost = SimpleNamespace(payload_hash="7" * 64)
            cost_journal = Mock()
            cost_journal.receipt_for_event.side_effect = lambda event_id: (
                bootstrap_cost
                if event_id == "post-parent-bootstrap"
                else boundary_cost
                if event_id == "generation-001-settlement"
                else None
            )
            cost_journal.publish_snapshot.return_value = failed_cost
            execution_ref = Mock(to_dict=Mock(return_value={"execution": 1}))

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_validation_lifecycle_identity",
                    return_value=Mock(),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                    return_value=[record],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "audit_generation_lifecycle",
                    return_value=lifecycle,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_replay_validation_combined_cost_chain",
                    return_value="4" * 64,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "GenerationSegmentJournal.start",
                    return_value=segment,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_job_attempt_scope",
                    return_value=Mock(),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "recover_stair_parent_bootstrap_attempt_scope",
                    side_effect=lambda **kwargs: kwargs["current_scope"],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "run_genesis_parent_public_dev",
                    return_value=guard,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_publish_or_load_parent_bootstrap_cost_boundary",
                    return_value=bootstrap_cost,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "build_generation_continuation_cursor",
                    return_value=SimpleNamespace(
                        current_best=parent,
                        next_generation_index=2,
                    ),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "replay_stair_generation_guard_chain",
                    return_value=(parent, guard),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_load_existing_generation_execution",
                    return_value=(record, execution_ref),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_publish_or_validate_generation_closure",
                    side_effect=ValueError(
                        "completed bound jobs lack unique resource summaries"
                    ),
                ) as replay_closure,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "collect_actual_cost_usage",
                    return_value=(Mock(), (), ()),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_assert_generation_admission"
                ) as admission,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_build_generation_inputs"
                ) as build_inputs,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "execute_fixed_width_generation"
                ) as execute,
                self.assertRaisesRegex(
                    ValueError,
                    "completed bound jobs lack unique resource summaries",
                ),
            ):
                _run_validation_session(
                    foundation,
                    genesis=genesis,
                    cost_journal=cost_journal,
                    startup_resource_observation={},
                )

            replay_closure.assert_called_once()
            admission.assert_not_called()
            build_inputs.assert_not_called()
            execute.assert_not_called()
            self.assertEqual(foundation.runner.mock_calls, [])

    def test_parent_bootstrap_cost_boundary_is_published_once_and_reused(self) -> None:
        foundation = Mock()
        genesis = Mock()
        usage = Mock()
        receipt = SimpleNamespace(
            payload={"cap_checks": {"all_passed": True}},
            payload_hash="c" * 64,
        )
        journal = Mock()
        journal.receipt_for_event.side_effect = [None, receipt]
        journal.publish_snapshot.return_value = receipt

        with (
            patch(
                "true_rl_agent.tasks.stair_climbing.generation_session."
                "collect_actual_cost_usage",
                return_value=(usage, ("conservative",), ()),
            ) as collect,
            patch(
                "true_rl_agent.tasks.stair_climbing.generation_session."
                "_latest_cost_head",
                return_value="d" * 64,
            ),
        ):
            first = _publish_or_load_parent_bootstrap_cost_boundary(
                foundation,
                genesis=genesis,
                ledger=[],
                cost_journal=journal,
                previous_cost_chain_head_hash="d" * 64,
            )
            second = _publish_or_load_parent_bootstrap_cost_boundary(
                foundation,
                genesis=genesis,
                ledger=[{"generation_index": 1}],
                cost_journal=journal,
                previous_cost_chain_head_hash="d" * 64,
            )

        self.assertIs(first, receipt)
        self.assertIs(second, receipt)
        collect.assert_called_once_with(foundation, genesis=genesis, ledger=[])
        journal.publish_snapshot.assert_called_once_with(
            event_id="post-parent-bootstrap",
            previous_cost_chain_head_hash="d" * 64,
            usage=usage,
            conservative_charges=("conservative",),
            finalization_output_reserve_bytes=0,
        )

    def test_resume_rejects_source_commit_drift_before_session_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(
                self._config(Path(tmp)),
                session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                max_allocated_active_gpu_hours=12.5,
                max_simulated_control_steps=1_577_542_128,
                max_output_bytes=3_758_096_384,
            )
            _publish_or_verify_source_checkout_identity(
                config.output_dir,
                source_commit=SHA,
            )
            _create_foundation(
                config,
                source_commit=SHA,
                inventory=(GPU_A, GPU_B),
            )
            resumed = replace(
                config,
                expected_source_commit="b" * 40,
                resume_existing=True,
            )
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_verify_source_checkout",
                    return_value="b" * 40,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "discover_two_gpu_inventory",
                    return_value=(GPU_A, GPU_B),
                ) as inventory,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_assert_target_gpus_compute_idle"
                ) as idle,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "prepare_session_genesis"
                ) as prepare,
                self.assertRaisesRegex(ValueError, "identity differs"),
            ):
                run_stair_generation_pilot(resumed)
            inventory.assert_not_called()
            idle.assert_not_called()
            prepare.assert_not_called()

    def test_foundation_rejects_self_consistent_phased_cost_science_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            task_config = StairClimbingTaskConfig.defaults(
                config.project_root
                / "configs"
                / "tasks"
                / "stair_unified_protocol_v1.json"
            )
            drifted = json.loads(
                json.dumps(stair_phased_evaluation_cost_contract(task_config))
            )
            drifted["per_generation_phases"]["sealed_validation"][
                "seed_count_per_job"
            ] = 8
            drifted["max_seed_units_per_generation"] = 24
            drifted["max_episode_executions_per_generation"] = (
                24 * drifted["episodes_per_seed_unit"]
            )
            payload = {
                key: value for key, value in drifted.items() if key != "contract_hash"
            }
            drifted["contract_hash"] = canonical_json_hash(payload)

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session_cost."
                    "stair_phased_evaluation_cost_contract",
                    return_value=drifted,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.phased_adapter."
                    "stair_phased_evaluation_cost_contract",
                    return_value=drifted,
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "differs from task config",
                ),
            ):
                _create_foundation(
                    config,
                    source_commit=SHA,
                    inventory=(GPU_A, GPU_B),
                )

    def test_pre_genesis_scope_uses_frozen_identity_not_prospective_genesis(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scope = _session_genesis_attempt_scope(
                journal_root=Path(tmp),
                session_id="p6b-generation-pilot",
                genesis_identity_hash=DIGEST,
                session_contract_hash="c" * 64,
                lane_id="lane-a",
                invocation_segment_id="session-genesis",
                budget_contract_hash="d" * 64,
                resource_binding_hash="e" * 64,
                max_infrastructure_retries=1,
            )
            self.assertIsInstance(scope, PreGenesisExecutionScope)
            self.assertEqual(scope.genesis_identity_hash, DIGEST)
            self.assertNotIn("session_genesis_hash", scope.binding_payload())
            self.assertEqual(scope.attempt_budget_units, 2)

    def test_phased_reference_eval_splits_one_frozen_reserve_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_id = "unified-reference"
            unified_config = StairClimbingTaskConfig.defaults(
                Path(__file__).resolve().parents[2]
                / "configs"
                / "tasks"
                / "stair_unified_protocol_v1.json"
            )
            cost_contract = StairGenerationSessionCostContractV2(
                session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                max_generations=3,
                scientific_cost_basis=build_stair_generation_scientific_cost_basis(
                    session_id=session_id,
                    config=unified_config,
                ),
                retry_train_jobs=1,
                retry_eval_seed_units=7,
                successful_allocated_active_gpu_hours=(
                    STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
                ),
                retry_allocated_active_gpu_hours=(
                    STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
                ),
                successful_output_bytes=(
                    STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES
                ),
                retry_output_bytes=STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
            )
            split = _reference_qualification_phase_reserves(cost_contract)
            frozen = next(
                reserve
                for reserve in stair_generation_operation_cost_reserves()
                if reserve.operation_class == "reference_qualification"
            )
            self.assertEqual(set(split), set(EvaluationPhase))
            self.assertEqual(
                sum(item.source_allocated_gpu_capacity_ns for item in split.values()),
                frozen.source_allocated_gpu_capacity_ns,
            )
            self.assertEqual(
                sum(item.source_output_bytes for item in split.values()),
                frozen.source_output_bytes,
            )

            common = {
                "journal_root": Path(tmp),
                "session_id": session_id,
                "genesis_identity_hash": "8" * 64,
                "session_contract_hash": "9" * 64,
                "lane_id": "lane-a",
                "budget_contract_hash": "a" * 64,
                "attempt_budget_units": 1,
                "max_infrastructure_retries": 0,
                "resource_binding_hash": "b" * 64,
            }
            slots = set()
            for phase in EvaluationPhase:
                phase_name = phase.value.replace("_", "-")
                slot, reserve = _strict_operation_slot(
                    session_id=session_id,
                    scope=PreGenesisExecutionScope(
                        **common,
                        invocation_segment_id=f"session-genesis-{phase_name}",
                    ),
                    logical_job_id=f"{session_id}-eval-{phase_name}",
                    job_kind="eval_phase",
                    attempt_index=0,
                    cost_contract=cost_contract,
                )
                slots.add(slot)
                self.assertEqual(reserve, split[phase])
            self.assertEqual(
                slots,
                {
                    f"reference_qualification:{phase.value}"
                    for phase in EvaluationPhase
                },
            )

            with self.assertRaisesRegex(ValueError, "no frozen operation allocation"):
                _strict_operation_slot(
                    session_id=session_id,
                    scope=JobAttemptScope(
                        journal_root=Path(tmp),
                        session_genesis_hash="c" * 64,
                        session_contract_hash="9" * 64,
                        generation_id="generation-parent-public-dev-bootstrap",
                        lane_id="lane-a",
                        invocation_segment_id="segment-parent",
                        budget_contract_hash="a" * 64,
                        attempt_budget_units=1,
                        max_infrastructure_retries=0,
                        resource_binding_hash="b" * 64,
                    ),
                    logical_job_id=f"{session_id}-parent-eval-public-dev",
                    job_kind="eval_phase",
                    attempt_index=0,
                    cost_contract=cost_contract,
                )

            legacy_cost_contract = SimpleNamespace(
                max_generations=3,
                imports_reference_evidence=False,
                imported_reference_timeout_envelope=None,
                scientific_cost_basis=SimpleNamespace(
                    reuses_reference_public_dev_receipt=False
                ),
            )
            legacy_slot, legacy_reserve = _strict_operation_slot(
                session_id=session_id,
                scope=PreGenesisExecutionScope(
                    **common,
                    invocation_segment_id="session-genesis",
                ),
                logical_job_id=f"{session_id}-eval",
                job_kind="eval",
                attempt_index=0,
                cost_contract=legacy_cost_contract,
            )
            self.assertEqual(legacy_slot, "reference_qualification")
            self.assertEqual(legacy_reserve, frozen)

            empty_state = _StairOperationAllocationState(
                reserved_success_slots=frozenset(),
                reserved_retry_classes=frozenset(),
                attempt_results={},
                measured_gpu_capacity_ns=0,
                conservatively_charged_gpu_capacity_ns=0,
                unmeasured_bound_attempts=0,
                open_attempt_output_reserve_bytes=0,
            )
            self.assertTrue(
                _strict_retry_is_still_possible(
                    operation_class="sealed_validation_retry",
                    closed_generation_count=0,
                    state=empty_state,
                    cost_contract=cost_contract,
                )
            )

    def test_real_composition_snapshot_reaches_pre_genesis_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            startup = {
                "schema_version": "test.compute-idle.v1",
                "stage": "startup",
                "rows": [],
            }
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_verify_source_checkout",
                    return_value=SHA,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "discover_two_gpu_inventory",
                    return_value=(GPU_A, GPU_B),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_assert_target_gpus_compute_idle",
                    return_value=startup,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_require_stair_generation_provider_readiness"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "prepare_session_genesis",
                    side_effect=RuntimeError("stop-after-pre-genesis-scope"),
                ) as prepare,
                self.assertRaisesRegex(
                    RuntimeError,
                    "stop-after-pre-genesis-scope",
                ),
            ):
                run_stair_generation_pilot(config)

            kwargs = prepare.call_args.kwargs
            expected_hash = session_genesis_identity_hash(
                kwargs["genesis_identity_snapshot"]
            )
            self.assertEqual(
                kwargs["reference_train_attempt_scope"].genesis_identity_hash,
                expected_hash,
            )
            self.assertEqual(
                {
                    phase.genesis_identity_hash
                    for phase in kwargs[
                        "reference_eval_phase_attempt_scopes"
                    ].values()
                },
                {expected_hash},
            )
            self.assertEqual(
                kwargs["reference_phased_evaluation_adapter"].config,
                kwargs["task_plugin"].config,
            )

    def test_inventory_requires_exact_ordinals_and_uuids(self) -> None:
        completed = SimpleNamespace(stdout=f"0, {GPU_A}\n1, {GPU_B}\n")
        with patch(
            "true_rl_agent.tasks.stair_climbing.generation_session.subprocess.run",
            return_value=completed,
        ) as run:
            actual = discover_two_gpu_inventory(
                "nvidia-smi",
                expected=(GPU_A, GPU_B),
            )
        self.assertEqual(actual, (GPU_A, GPU_B))
        run.assert_called_once()

        extra = SimpleNamespace(stdout=f"0, {GPU_A}\n1, {GPU_B}\n2, {GPU_A}\n")
        with (
            patch(
                "true_rl_agent.tasks.stair_climbing.generation_session.subprocess.run",
                return_value=extra,
            ),
            self.assertRaisesRegex(ValueError, "frozen two-GPU plan"),
        ):
            discover_two_gpu_inventory(
                "nvidia-smi",
                expected=(GPU_A, GPU_B),
            )

    def test_source_checkout_rejects_tracked_dirty_status(self) -> None:
        root = Path(__file__).resolve().parents[2]
        with (
            patch(
                "true_rl_agent.tasks.stair_climbing.generation_session.require_stair_source_checkout",
                side_effect=ValueError("source checkout has tracked modifications"),
            ) as require_checkout,
            self.assertRaisesRegex(
                ValueError,
                "tracked modifications",
            ),
        ):
            _verify_source_checkout(root, expected_commit=SHA)

        require_checkout.assert_called_once_with(
            root,
            expected_commit=SHA,
        )

    def test_compute_idle_gate_rejects_target_gpu_processes(self) -> None:
        with patch(
            "true_rl_agent.tasks.stair_climbing.generation_session.subprocess.run",
            return_value=SimpleNamespace(stdout=""),
        ):
            observed = _assert_target_gpus_compute_idle(
                "nvidia-smi",
                expected=(GPU_A, GPU_B),
                stage="startup",
            )
        self.assertEqual(observed["rows"], [])

        with (
            patch(
                "true_rl_agent.tasks.stair_climbing.generation_session.subprocess.run",
                return_value=SimpleNamespace(stdout=f"{GPU_A}, 1234\n"),
            ),
            self.assertRaisesRegex(ValueError, "not compute-idle"),
        ):
            _assert_target_gpus_compute_idle(
                "nvidia-smi",
                expected=(GPU_A, GPU_B),
                stage="final",
            )

    def test_provider_credential_is_preflighted_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "provider.env"
            env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
            _preflight_provider_credential(env_file)
            env_file.write_text(
                "DEEPSEEK_API_KEY=not-latin-测试\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "HTTP-header safe"):
                _preflight_provider_credential(env_file)

    def test_provider_cost_reads_live_authoring_journal_before_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            round_dir = output / "authoring" / "pilot-generation-001" / "lane-a"
            round_dir.mkdir(parents=True)
            snapshot = SimpleNamespace(
                attempts=(
                    {
                        "attempt_id": "attempt_000001",
                        "sequence": 1,
                        "status": "completed",
                        "prompt_payload": {},
                        "provider_attempt": {
                            "provider_usage": {
                                "request_units": 10,
                                "response_units": 20,
                                "total_units": 30,
                                "reasoning_units": 5,
                            }
                        },
                    },
                    {
                        "attempt_id": "attempt_000002",
                        "sequence": 2,
                        "status": "interrupted_after_prompt",
                        "prompt_payload": {},
                        "provider_attempt": None,
                    },
                )
            )
            foundation = SimpleNamespace(
                config=SimpleNamespace(output_dir=output, session_id="pilot")
            )
            with patch(
                "true_rl_agent.tasks.stair_climbing.generation_session.RoundAuthoringJournal.snapshot",
                return_value=snapshot,
            ):
                actual = _provider_actuals(foundation, ledger=[])

            self.assertEqual(actual["attempts"], 2)
            self.assertEqual(actual["request_units"], 300_010)
            self.assertEqual(actual["response_units"], 32_788)
            self.assertEqual(actual["total_units"], 332_798)
            self.assertEqual(actual["conservative_attempts"], 1)

            (output / "authoring" / "pilot-generation-004" / "lane-a").mkdir(
                parents=True
            )
            with self.assertRaisesRegex(ValueError, "unknown generation"):
                _provider_actuals(foundation, ledger=[])

    def test_provider_gate_blocks_prior_missing_usage_before_delegate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            generation_id = "pilot-generation-001"
            for lane_id in ("lane-a", "lane-b"):
                (output / "authoring" / generation_id / lane_id).mkdir(parents=True)
            prompt_payload = {"schema_version": "test-prompt"}
            prompt_ref = PromptPayloadRef(
                path=str(output / "current-prompt.json"),
                checksum="sha256:" + "c" * 64,
                size_bytes=1,
            )
            prior = SimpleNamespace(
                attempts=(
                    {
                        "status": "completed",
                        "prompt_payload": prompt_ref.to_public_dict(),
                        "provider_attempt": {},
                    },
                )
            )
            current = SimpleNamespace(
                attempts=(
                    {
                        "status": "interrupted_after_prompt",
                        "prompt_payload": prompt_ref.to_public_dict(),
                        "provider_attempt": None,
                    },
                )
            )
            delegate = Mock()
            provider = _StairCostGuardedProvider(
                delegate=delegate,
                output_dir=output,
                generation_id=generation_id,
                lane_id="lane-b",
            )
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.RoundAuthoringJournal.snapshot",
                    side_effect=[prior, current],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.verify_prompt_payload",
                    return_value=prompt_payload,
                ),
                self.assertRaisesRegex(RuntimeError, "authoritative usage"),
            ):
                provider.next_authoring_response(prompt_payload=prompt_payload)
            delegate.next_authoring_response.assert_not_called()

    def test_provider_gate_delegates_once_with_valid_prior_balance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            generation_id = "pilot-generation-001"
            for lane_id in ("lane-a", "lane-b"):
                (output / "authoring" / generation_id / lane_id).mkdir(parents=True)
            prompt_payload = {"schema_version": "test-prompt"}
            prompt_ref = PromptPayloadRef(
                path=str(output / "current-prompt.json"),
                checksum="sha256:" + "d" * 64,
                size_bytes=1,
            )
            prior = SimpleNamespace(
                attempts=(
                    {
                        "status": "completed",
                        "prompt_payload": prompt_ref.to_public_dict(),
                        "provider_attempt": {
                            "provider_usage": {
                                "request_units": 100,
                                "response_units": 20,
                                "total_units": 120,
                            }
                        },
                    },
                )
            )
            current = SimpleNamespace(
                attempts=(
                    {
                        "status": "interrupted_after_prompt",
                        "prompt_payload": prompt_ref.to_public_dict(),
                        "provider_attempt": None,
                    },
                )
            )
            expected = object()
            delegate = Mock()
            delegate.next_authoring_response.return_value = expected
            provider = _StairCostGuardedProvider(
                delegate=delegate,
                output_dir=output,
                generation_id=generation_id,
                lane_id="lane-b",
            )
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.RoundAuthoringJournal.snapshot",
                    side_effect=[prior, current],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.verify_prompt_payload",
                    return_value=prompt_payload,
                ),
            ):
                actual = provider.next_authoring_response(prompt_payload=prompt_payload)
            self.assertIs(actual, expected)
            delegate.next_authoring_response.assert_called_once_with(
                prompt_payload=prompt_payload
            )

    def test_resource_report_binds_raw_summary_pid_and_uuid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            foundation = _create_foundation(
                config,
                source_commit=SHA,
                inventory=(GPU_A, GPU_B),
            )
            summary_path = config.output_dir / "runner" / "job" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "job_id": "job-a",
                        "gpu_process_binding": (foundation.gpu_bindings[0].to_dict()),
                    }
                ),
                encoding="utf-8",
            )
            validated = {
                "gpu_process_binding": foundation.gpu_bindings[0].to_dict(),
                "owned_process_identity": {
                    "process_id": 123,
                    "process_group_id": 123,
                },
                "stair_gpu_runtime_evidence": {
                    "external_compute_context_observation": {
                        "process_id": 123,
                        "rows": [{"gpu_uuid": GPU_A}],
                    }
                },
            }
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._completed_bound_job_attempts",
                    return_value={
                        "job-a": foundation.gpu_bindings[0].resource_binding_hash
                    },
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.validate_stair_gpu_runtime_summary_evidence",
                    return_value=validated,
                ) as validate,
            ):
                summaries, count = _resource_summary_evidence(foundation)

            self.assertEqual(count, 1)
            self.assertEqual(summaries[0]["process_id"], 123)
            self.assertEqual(summaries[0]["job_id"], "job-a")
            self.assertEqual(len(summaries[0]["raw_sha256"]), 64)
            validate.assert_called_once_with(
                unittest.mock.ANY,
                expected_binding=foundation.gpu_bindings[0],
                expected_physical_gpu_uuid=GPU_A,
                require_compute_context_observation=True,
            )

    def test_generation_closure_excludes_later_final_confirmation_attempts(
        self,
    ) -> None:
        foundation = SimpleNamespace(
            config=SimpleNamespace(session_id="p6d-prefix"),
            composition=SimpleNamespace(
                task_plugin=SimpleNamespace(
                    config=SimpleNamespace(is_unified_protocol=False)
                )
            ),
            session_contract=SimpleNamespace(max_generations=15),
            cost_contract=Mock(),
        )
        journal = SimpleNamespace(
            scope=SimpleNamespace(generation_id="generation-final-confirmation"),
            logical_job_id="p6d-prefix-final-confirmation-genesis-eval",
            job_kind="eval_phase",
        )
        with patch(
            "true_rl_agent.tasks.stair_climbing.generation_session."
            "_strict_operation_slot"
        ):
            self.assertFalse(
                _attempt_is_in_generation_prefix(
                    foundation,
                    journal=journal,
                    reservation={"attempt_index": 0},
                    generation_index=15,
                    include_final_confirmation=False,
                )
            )
            self.assertTrue(
                _attempt_is_in_generation_prefix(
                    foundation,
                    journal=journal,
                    reservation={"attempt_index": 0},
                    generation_index=15,
                    include_final_confirmation=True,
                )
            )

    def test_driver_calls_existing_generation_coordinator_once_and_closes_one_row(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = SimpleNamespace(output_dir=root, session_id="pilot")
            acceptance = Mock()
            composition = SimpleNamespace(
                task_plugin=SimpleNamespace(
                    task_spec=SimpleNamespace(acceptance_protocol=acceptance),
                    config=SimpleNamespace(is_unified_protocol=False),
                ),
                backend_adapter=Mock(),
            )
            contract = SimpleNamespace(
                contract_hash="c" * 64,
                bound_contract_hashes={"final_confirmation_plan": "d" * 64},
            )
            cost = SimpleNamespace(contract_hash="e" * 64)
            continuation = Mock()
            foundation = SimpleNamespace(
                config=config,
                composition=composition,
                session_contract=contract,
                continuation_policy=continuation,
                cost_contract=cost,
                phased_adapter=Mock(),
                runner=Mock(),
                gpu_bindings=(Mock(), Mock()),
            )
            genesis = SimpleNamespace(
                session_genesis_hash="f" * 64,
                initial_current_best=Mock(),
                expected_initial_policy_state_hash=None,
            )
            segment = Mock(segment_id="continuation-id")
            segment.cost_chain_head_hash = "e" * 64
            generation_result = SimpleNamespace(
                committed=True,
                reused_execution=False,
                record={"generation_id": "pilot-generation-001"},
                execution_receipt_ref=Mock(),
            )
            ledger_row = {"schema_version": "ignored-by-mocked-public-projection"}
            terminal = Mock()
            terminal.to_dict.return_value = {"closed_generation_count": 1}
            post_history = Mock()
            post_exploration = Mock()
            actual_cost = SimpleNamespace(
                payload_hash="a" * 64,
                payload={"cap_checks": {"all_passed": True}},
            )
            cost_journal = Mock()
            composition.task_memory_adapter = Mock()
            composition.task_memory_adapter.build_generation_exploration_plan.return_value = post_exploration

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.GenerationSegmentJournal.start",
                    return_value=segment,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.audit_generation_lifecycle",
                    return_value=SimpleNamespace(terminal=False),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._latest_cost_head",
                    return_value="e" * 64,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._first_generation_segment_id",
                    return_value="original-segment-id",
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.run_genesis_parent_public_dev",
                    return_value=Mock(),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._build_generation_inputs",
                    return_value=(Mock(), Mock()),
                ) as build_inputs,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.execute_fixed_width_generation",
                    return_value=generation_result,
                ) as execute,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                    side_effect=[[], [ledger_row]],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.build_generation_terminal_binding",
                    return_value=terminal,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.project_public_history",
                    return_value=post_history,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._build_report",
                    return_value={"checks": {"single_commit": True}},
                ) as report,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._job_attempt_scope",
                    return_value=Mock(),
                ) as attempt_scope,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.collect_actual_cost_usage",
                    return_value=(Mock(), (), ()),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._build_resource_closure_evidence",
                    return_value={},
                ),
            ):
                cost_journal.publish_snapshot.return_value = actual_cost
                result = _run_one_generation(
                    foundation,
                    genesis=genesis,
                    cost_journal=cost_journal,
                    startup_resource_observation={},
                )

            self.assertEqual(result, {"checks": {"single_commit": True}})
            execute.assert_called_once()
            self.assertEqual(
                attempt_scope.call_args.kwargs["invocation_segment_id"],
                "original-segment-id",
            )
            self.assertEqual(
                build_inputs.call_args.kwargs["segment_id"],
                "original-segment-id",
            )
            segment.finish.assert_called_once_with(
                status="completed",
                reason="normal",
                closed_generation_count=1,
                cost_chain_head_hash="a" * 64,
            )
            report.assert_called_once()
            self.assertFalse((root / "non_authoritative_lane_ledgers").exists())

    def test_generation_exception_exit_receives_the_original_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            failure = RuntimeError("preflight-b3-wiring")
            segment = Mock(
                segment_id="segment-id",
                cost_chain_head_hash="e" * 64,
            )
            foundation = SimpleNamespace(
                config=SimpleNamespace(output_dir=root, session_id="pilot"),
                composition=SimpleNamespace(
                    task_plugin=SimpleNamespace(
                        task_spec=SimpleNamespace(acceptance_protocol=Mock()),
                        config=SimpleNamespace(is_unified_protocol=False),
                    ),
                    backend_adapter=Mock(),
                ),
                session_contract=SimpleNamespace(
                    contract_hash="c" * 64,
                    bound_contract_hashes={"final_confirmation_plan": "d" * 64},
                ),
                continuation_policy=Mock(),
                cost_contract=SimpleNamespace(contract_hash="e" * 64),
                gpu_bindings=(Mock(), Mock()),
            )
            genesis = SimpleNamespace(
                session_genesis_hash="f" * 64,
                initial_current_best=Mock(),
            )
            failed_cost = SimpleNamespace(payload_hash="a" * 64)
            cost_journal = Mock()
            cost_journal.publish_snapshot.return_value = failed_cost

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                    side_effect=[[], []],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "audit_generation_lifecycle",
                    return_value=SimpleNamespace(terminal=False),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "GenerationSegmentJournal.start",
                    return_value=segment,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_latest_cost_head",
                    return_value="e" * 64,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_first_generation_segment_id",
                    return_value="segment-id",
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_job_attempt_scope",
                    side_effect=failure,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "collect_actual_cost_usage",
                    return_value=(Mock(), (), ()),
                ),
                self.assertRaisesRegex(RuntimeError, "preflight-b3-wiring") as raised,
            ):
                _run_one_generation(
                    foundation,
                    genesis=genesis,
                    cost_journal=cost_journal,
                    startup_resource_observation={},
                )

            self.assertIs(raised.exception, failure)
            segment.finish.assert_called_once_with(
                status="failed",
                reason="exception",
                closed_generation_count=0,
                cost_chain_head_hash="a" * 64,
                unclosed_attempt_reservations=(),
                exception=failure,
            )

    def test_foundation_resume_is_exact_and_cost_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            first = _create_foundation(
                config,
                source_commit=SHA,
                inventory=(GPU_A, GPU_B),
            )
            second = _create_foundation(
                config,
                source_commit=SHA,
                inventory=(GPU_A, GPU_B),
            )
            self.assertEqual(
                first.session_contract.contract_hash,
                second.session_contract.contract_hash,
            )
            drifted = StairGenerationPilotConfig(
                **{
                    **config.__dict__,
                    "max_output_bytes": config.max_output_bytes + 1,
                }
            )
            with self.assertRaisesRegex(
                ValueError,
                "existing Stair session cost contract differs",
            ):
                _create_foundation(
                    drifted,
                    source_commit=SHA,
                    inventory=(GPU_A, GPU_B),
                )

    def test_resume_rejects_an_existing_directory_without_foundation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            config.output_dir.mkdir(parents=True)
            resume = StairGenerationPilotConfig(
                **{**config.__dict__, "resume_existing": True}
            )
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._reject_parent_gpu_remap"
                ),
                self.assertRaisesRegex(ValueError, "create-once foundation"),
            ):
                run_stair_generation_pilot(resume)

    def test_closed_ledger_resume_only_settles_without_provider_or_gpu_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            acceptance = Mock()
            composition = SimpleNamespace(
                task_plugin=SimpleNamespace(
                    task_spec=SimpleNamespace(acceptance_protocol=acceptance),
                    config=SimpleNamespace(is_unified_protocol=False),
                ),
                task_memory_adapter=Mock(),
                backend_adapter=Mock(),
            )
            foundation = SimpleNamespace(
                config=SimpleNamespace(output_dir=root, session_id="pilot"),
                composition=composition,
                session_contract=SimpleNamespace(
                    contract_hash="c" * 64,
                    bound_contract_hashes={"final_confirmation_plan": "d" * 64},
                ),
                continuation_policy=Mock(),
                cost_contract=SimpleNamespace(contract_hash="e" * 64),
                phased_adapter=Mock(),
                runner=Mock(),
                gpu_bindings=(Mock(), Mock()),
            )
            genesis = SimpleNamespace(
                session_genesis_hash="f" * 64,
                initial_current_best=Mock(),
                expected_initial_policy_state_hash=None,
            )
            record = {"generation_id": "pilot-generation-001"}
            execution_ref = Mock()
            execution_ref.to_dict.return_value = {"kind": "execution"}
            segment = Mock(
                segment_id="continuation-id",
                cost_chain_head_hash="e" * 64,
            )
            actual_cost = SimpleNamespace(
                payload_hash="a" * 64,
                payload={"cap_checks": {"all_passed": True}},
            )
            cost_journal = Mock()
            cost_journal.publish_snapshot.return_value = actual_cost
            composition.task_memory_adapter.build_generation_exploration_plan.return_value = Mock()

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                    return_value=[record],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.audit_generation_lifecycle",
                    return_value=SimpleNamespace(terminal=False),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.GenerationSegmentJournal.start",
                    return_value=segment,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._latest_cost_head",
                    return_value="e" * 64,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._first_generation_segment_id",
                    return_value="original-segment-id",
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._load_existing_generation_execution",
                    return_value=(record, execution_ref),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.collect_actual_cost_usage",
                    return_value=(Mock(), (), ()),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._generation_exploration_for_ledger",
                    return_value=Mock(),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.build_generation_terminal_binding",
                    return_value=Mock(to_dict=Mock(return_value={})),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.project_public_history",
                    return_value=Mock(),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._build_report",
                    return_value={"settled": True},
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session._build_resource_closure_evidence",
                    return_value={},
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.run_genesis_parent_public_dev"
                ) as bootstrap,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.execute_fixed_width_generation"
                ) as execute,
            ):
                report = _run_one_generation(
                    foundation,
                    genesis=genesis,
                    cost_journal=cost_journal,
                    startup_resource_observation={},
                )

            self.assertEqual(report, {"settled": True})
            bootstrap.assert_not_called()
            execute.assert_not_called()
            segment.finish.assert_called_once_with(
                status="completed",
                reason="normal",
                closed_generation_count=1,
                cost_chain_head_hash="a" * 64,
            )

    def test_validation_report_precheck_failure_keeps_terminal_recoverable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            report_path = output_dir / STAIR_GENERATION_REPORT_FILENAME
            records = [
                {
                    "generation_id": f"p6c-session-generation-{index:03d}",
                    "generation_index": index,
                }
                for index in range(1, 4)
            ]
            closure_root = output_dir / "generation_closures"
            closure_root.mkdir()
            for record in records:
                (closure_root / f"{record['generation_id']}.json").write_text(
                    "{}\n",
                    encoding="utf-8",
                )

            acceptance = Mock()
            policy = SimpleNamespace(
                max_segments=4,
                allowed_continuation_reasons=("exception",),
                policy_hash="1" * 64,
            )
            contract = SimpleNamespace(
                session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                max_generations=3,
                contract_hash="2" * 64,
                bound_contract_hashes={"final_confirmation_plan": "3" * 64},
            )
            foundation = SimpleNamespace(
                config=SimpleNamespace(
                    output_dir=output_dir,
                    session_id="p6c-session",
                    pause_after_generation=None,
                ),
                composition=SimpleNamespace(
                    task_plugin=SimpleNamespace(
                        task_spec=SimpleNamespace(
                            acceptance_protocol=acceptance,
                        ),
                        config=SimpleNamespace(is_unified_protocol=False),
                    ),
                    backend_adapter=Mock(),
                ),
                session_contract=contract,
                continuation_policy=policy,
                cost_contract=Mock(),
                phased_adapter=Mock(),
                runner=Mock(),
                gpu_bindings=(Mock(), Mock()),
            )
            parent = Mock()
            guard = Mock()
            genesis = SimpleNamespace(
                session_genesis_hash="4" * 64,
                initial_current_best=parent,
                expected_initial_policy_state_hash=None,
            )
            cursor = SimpleNamespace(
                current_best=parent,
                next_generation_index=4,
                ledger_snapshot_hash="5" * 64,
                current_head_hash="6" * 64,
            )
            cost_receipts = {
                f"generation-{index:03d}-settlement": SimpleNamespace(
                    payload_hash=str(index) * 64,
                    payload={"cap_checks": {"all_passed": True}},
                )
                for index in range(1, 4)
            }
            cost_receipts["post-parent-bootstrap"] = SimpleNamespace(
                payload_hash="0" * 64,
                payload={"cap_checks": {"all_passed": True}},
            )
            latest_cost = cost_receipts["generation-003-settlement"]
            latest_cost.payload_hash = "3" * 64
            failed_cost = SimpleNamespace(payload_hash="7" * 64)
            cost_journal = Mock()
            cost_journal.receipt_for_event.side_effect = cost_receipts.get
            cost_journal.load_chain.return_value = [latest_cost]
            cost_journal.publish_snapshot.return_value = failed_cost

            segment_dir = output_dir / "generation_segments" / "test-segment"
            segment_dir.mkdir(parents=True)
            lease = SimpleNamespace(assert_owned=Mock(), release=Mock())
            segment = GenerationSegmentJournal(
                output_dir=output_dir,
                identity=SimpleNamespace(
                    to_dict=Mock(return_value={"session_id": "p6c-session"})
                ),
                policy=policy,
                segment_id="8" * 32,
                continuation_index=2,
                segment_dir=segment_dir,
                start_record={"generations_at_start": 3},
                start_hash="9" * 64,
                cost_chain_head_hash="3" * 64,
                _lease=lease,
            )
            execution_refs = {
                record["generation_id"]: Mock(
                    to_dict=Mock(
                        return_value={"generation_id": record["generation_id"]}
                    )
                )
                for record in records
            }
            failure = RuntimeError("synthetic terminal report precheck failure")

            def load_execution(_foundation, *, record):
                return dict(record), execution_refs[record["generation_id"]]

            def fail_prevalidation(*_args, **_kwargs):
                self.assertFalse((segment_dir / "exit.json").exists())
                self.assertFalse(report_path.exists())
                raise failure

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                    side_effect=[records, records],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "audit_generation_lifecycle",
                    return_value=SimpleNamespace(terminal=False),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "GenerationSegmentJournal.start",
                    return_value=segment,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_replay_validation_combined_cost_chain",
                    return_value="3" * 64,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_first_generation_segment_id",
                    return_value="first-segment",
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_job_attempt_scope",
                    return_value=Mock(),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "recover_stair_parent_bootstrap_attempt_scope",
                    side_effect=lambda **kwargs: kwargs["current_scope"],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "run_genesis_parent_public_dev",
                    return_value=guard,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "build_generation_continuation_cursor",
                    return_value=cursor,
                ) as build_cursor,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "replay_stair_generation_guard_chain",
                    return_value=(parent, guard),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_load_existing_generation_execution",
                    side_effect=load_execution,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_publish_or_validate_generation_closure",
                    return_value={"acceptance_authority": False},
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_build_resource_closure_evidence",
                    return_value={"observer_only": True},
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "collect_actual_cost_usage",
                    return_value=(Mock(), (), ()),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_current_validation_boundary_cost_receipt",
                    return_value=latest_cost,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_build_validation_report",
                    side_effect=fail_prevalidation,
                ) as prevalidate,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "build_generation_terminal_binding"
                ) as terminal,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "execute_fixed_width_generation"
                ) as execute,
                self.assertRaises(RuntimeError) as raised,
            ):
                _run_validation_session(
                    foundation,
                    genesis=genesis,
                    cost_journal=cost_journal,
                    startup_resource_observation={},
                )

            self.assertIs(raised.exception, failure)
            self.assertFalse(getattr(raised.exception, "__notes__", ()))
            exit_record = json.loads(
                (segment_dir / "exit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(exit_record["status"], "failed")
            self.assertEqual(exit_record["reason"], "exception")
            self.assertEqual(exit_record["generations_at_end"], 3)
            self.assertTrue(exit_record["continuation_eligible"])
            self.assertEqual(exit_record["exception_type"], "RuntimeError")
            self.assertFalse(report_path.exists())
            self.assertEqual(cursor.next_generation_index, 4)
            build_cursor.assert_called_once()
            prevalidate.assert_called_once()
            self.assertIs(
                prevalidate.call_args.kwargs["actual_cost_receipt"],
                latest_cost,
            )
            terminal.assert_not_called()
            execute.assert_not_called()
            lease.release.assert_called_once()

    def test_authoring_timing_projection_does_not_claim_an_exact_crash_total(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            foundation = SimpleNamespace(
                config=SimpleNamespace(
                    output_dir=output_dir,
                    session_id="p6c-session",
                ),
                session_contract=SimpleNamespace(max_generations=3),
            )
            summaries = [
                {
                    "lane_measured_closed_attempt_wall_duration_ns": {
                        "lane-a": 100,
                        "lane-b": 0,
                    },
                    "measured_closed_attempt_wall_duration_ns": 100,
                    "total_wall_duration_ns": None,
                    "timing_complete": False,
                    "stopped_attempts": [{}],
                    "incomplete_attempts": [{}],
                },
                {
                    "lane_measured_closed_attempt_wall_duration_ns": {
                        "lane-a": 120,
                        "lane-b": 80,
                    },
                    "measured_closed_attempt_wall_duration_ns": 200,
                    "total_wall_duration_ns": 200,
                    "timing_complete": True,
                    "stopped_attempts": [{}, {}],
                    "incomplete_attempts": [],
                },
                {
                    "lane_measured_closed_attempt_wall_duration_ns": {
                        "lane-a": 150,
                        "lane-b": 150,
                    },
                    "measured_closed_attempt_wall_duration_ns": 300,
                    "total_wall_duration_ns": 300,
                    "timing_complete": True,
                    "stopped_attempts": [{}, {}],
                    "incomplete_attempts": [],
                },
            ]
            with patch(
                "true_rl_agent.tasks.stair_climbing.generation_session."
                "replay_stair_generation_authoring_timing",
                side_effect=summaries,
            ):
                metrics = _validation_authoring_metrics(foundation)

            timing = metrics["timing"]
            self.assertFalse(timing["timing_complete"])
            self.assertEqual(timing["measured_closed_attempt_wall_duration_ns"], 600)
            self.assertIsNone(timing["total_wall_duration_ns"])
            self.assertIsNone(timing["total_wall_duration_seconds"])
            self.assertEqual(timing["incomplete_attempt_count"], 1)
            self.assertIsNone(timing["generations"][0]["total_wall_duration_ns"])
            self.assertFalse(timing["generations"][0]["timing_complete"])

    def test_recovery_only_cost_head_is_reconciled_before_closed_boundary_pause(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            acceptance = Mock()
            contract = SimpleNamespace(
                session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                max_generations=3,
                contract_hash="1" * 64,
                bound_contract_hashes={"final_confirmation_plan": "2" * 64},
            )
            foundation = SimpleNamespace(
                config=SimpleNamespace(
                    output_dir=output_dir,
                    session_id="p6c-session",
                    pause_after_generation=1,
                ),
                composition=SimpleNamespace(
                    task_plugin=SimpleNamespace(
                        task_spec=SimpleNamespace(acceptance_protocol=acceptance),
                        config=SimpleNamespace(is_unified_protocol=False),
                    ),
                    backend_adapter=Mock(),
                ),
                session_contract=contract,
                continuation_policy=Mock(),
                cost_contract=Mock(),
                phased_adapter=Mock(),
                runner=Mock(),
                gpu_bindings=(Mock(), Mock()),
            )
            parent = Mock()
            guard = Mock()
            genesis = SimpleNamespace(
                session_genesis_hash="3" * 64,
                initial_current_best=parent,
                expected_initial_policy_state_hash=None,
            )
            record = {
                "generation_id": "p6c-session-generation-001",
                "generation_index": 1,
            }
            execution_ref = Mock(to_dict=Mock(return_value={"execution": 1}))
            cursor = SimpleNamespace(current_best=parent, next_generation_index=2)
            boundary_cost = SimpleNamespace(payload_hash="4" * 64)
            current_boundary_cost = SimpleNamespace(payload_hash="6" * 64)
            post_parent_bootstrap = SimpleNamespace(
                payload={"cap_checks": {"all_passed": True}}
            )
            closure = {"generation_id": record["generation_id"]}
            segment = Mock(
                segment_id="recovery-segment",
                cost_chain_head_hash="5" * 64,
            )
            cost_journal = Mock()
            settlement_reads = iter((None, boundary_cost))

            def receipt_for_event(event_id: str):
                if event_id == "post-parent-bootstrap":
                    return post_parent_bootstrap
                if event_id == "generation-001-settlement":
                    return next(settlement_reads)
                return None

            cost_journal.receipt_for_event.side_effect = receipt_for_event
            pause_result = {"status": "paused-after-recovery"}

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                    return_value=[record],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "audit_generation_lifecycle",
                    return_value=SimpleNamespace(
                        terminal=False,
                        active_segment_id="crashed-segment",
                    ),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "GenerationSegmentJournal.start",
                    return_value=segment,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_replay_validation_combined_cost_chain",
                    return_value="5" * 64,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_first_generation_segment_id",
                    return_value="first-segment",
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_job_attempt_scope",
                    return_value=Mock(),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "recover_stair_parent_bootstrap_attempt_scope",
                    side_effect=lambda **kwargs: kwargs["current_scope"],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "run_genesis_parent_public_dev",
                    return_value=guard,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "build_generation_continuation_cursor",
                    return_value=cursor,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "replay_stair_generation_guard_chain",
                    return_value=(parent, guard),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_load_existing_generation_execution",
                    return_value=(record, execution_ref),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_build_resource_closure_evidence",
                    return_value={"closed": True},
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_settle_generation_cost",
                    return_value=(boundary_cost, ()),
                ) as settle,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_current_validation_boundary_cost_receipt",
                    return_value=current_boundary_cost,
                ) as current_boundary,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_publish_or_validate_generation_closure",
                    return_value=closure,
                ) as publish_closure,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_build_validation_pause_result",
                    return_value=pause_result,
                ) as build_pause,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_assert_generation_admission"
                ) as admission,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "execute_fixed_width_generation"
                ) as execute,
            ):
                actual = _run_validation_session(
                    foundation,
                    genesis=genesis,
                    cost_journal=cost_journal,
                    startup_resource_observation={},
                )

            self.assertEqual(actual, pause_result)
            settle.assert_called_once()
            current_boundary.assert_called_once()
            publish_closure.assert_called_once()
            self.assertIs(
                publish_closure.call_args.kwargs["actual_cost_receipt"],
                boundary_cost,
            )
            segment.finish.assert_called_once_with(
                status="paused",
                reason="operator_pause",
                closed_generation_count=1,
                cost_chain_head_hash=current_boundary_cost.payload_hash,
            )
            self.assertIs(build_pause.call_args.kwargs["closure"], closure)
            self.assertIs(
                build_pause.call_args.kwargs["actual_cost_receipt"],
                current_boundary_cost,
            )
            admission.assert_not_called()
            execute.assert_not_called()
            self.assertEqual(foundation.runner.mock_calls, [])

    def test_clean_closed_boundary_rejects_duplicate_pause_without_starting_segment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve()
            generation_id = "p6c-session-generation-001"
            closure_path = output_dir / "generation_closures" / f"{generation_id}.json"
            closure_path.parent.mkdir()
            closure_path.write_text("{}\n", encoding="utf-8")
            foundation = SimpleNamespace(
                config=SimpleNamespace(
                    output_dir=output_dir,
                    session_id="p6c-session",
                    pause_after_generation=1,
                ),
                composition=SimpleNamespace(
                    task_plugin=SimpleNamespace(
                        task_spec=SimpleNamespace(acceptance_protocol=Mock()),
                        config=SimpleNamespace(is_unified_protocol=False),
                    )
                ),
                session_contract=SimpleNamespace(
                    session_mode=DEVELOPMENT_GENERATION_VALIDATION_MODE,
                    max_generations=3,
                    contract_hash="1" * 64,
                    bound_contract_hashes={"final_confirmation_plan": "2" * 64},
                ),
                continuation_policy=Mock(),
            )
            genesis = SimpleNamespace(session_genesis_hash="3" * 64)
            cost_journal = Mock()
            cost_journal.receipt_for_event.return_value = Mock()

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session.load_ledger",
                    return_value=[{"generation_id": generation_id}],
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "audit_generation_lifecycle",
                    return_value=SimpleNamespace(
                        terminal=False,
                        active_segment_id=None,
                    ),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_replay_validation_combined_cost_chain",
                    return_value="4" * 64,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_validation_boundary_was_already_paused",
                    return_value=True,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "GenerationSegmentJournal.start"
                ) as start,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "execute_fixed_width_generation"
                ) as execute,
                self.assertRaisesRegex(ValueError, "boundary is already closed"),
            ):
                _run_validation_session(
                    foundation,
                    genesis=genesis,
                    cost_journal=cost_journal,
                    startup_resource_observation={},
                )

            start.assert_not_called()
            execute.assert_not_called()

    def test_manifest_version_constant_is_the_bound_generation_contract(self) -> None:
        self.assertEqual(
            GENERATION_MANIFEST_V2_SCHEMA_VERSION,
            "true_rl_agent.generation_manifest.v2",
        )


if __name__ == "__main__":
    unittest.main()
