from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (
    StairTrainingExecutionResult,
    VerifiedStairExecutionSourceManifest,
)
from true_rl_agent.tasks.stair_climbing.resource_shadow import (
    STAIR_RESOURCE_SHADOW_PRESET_ID,
    STAIR_RESOURCE_SHADOW_WORKLOAD,
    StairResourceShadowWorkload,
    execute_stair_resource_shadow_training,
    load_stair_resource_shadow_active_marker,
    load_stair_resource_shadow_summary,
    resolve_stair_resource_shadow_fixture,
    stair_resource_shadow_kit_args,
    stair_resource_shadow_workload_contract,
    stair_resource_shadow_workload_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INITIAL_POLICY_STATE_HASH = "9" * 64


class StairResourceShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        cls.script = _load_entrypoint()

    def test_workload_is_the_single_frozen_non_authoritative_preset(self) -> None:
        contract = stair_resource_shadow_workload_contract()

        self.assertEqual(
            contract,
            {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing."
                    "resource_shadow_workload.v2"
                ),
                "preset_id": STAIR_RESOURCE_SHADOW_PRESET_ID,
                "num_envs": 4096,
                "num_steps_per_env": 24,
                "max_iterations": 64,
                "budget_steps": 6_291_456,
                "checkpoint_save_interval": 50,
                "process_execution_device": "cuda:0",
                "cuda_visibility_binding": "single_physical_gpu_uuid",
                "renderer_multi_gpu": False,
                "renderer_multi_gpu_auto_enable": False,
                "renderer_multi_gpu_max_gpu_count": 1,
                "acceptance_authority": False,
                "ledger_authority": False,
            },
        )
        self.assertEqual(
            stair_resource_shadow_workload_hash(),
            canonical_json_hash(contract),
        )
        self.assertEqual(
            STAIR_RESOURCE_SHADOW_WORKLOAD.workload_hash,
            stair_resource_shadow_workload_hash(),
        )
        self.assertEqual(
            stair_resource_shadow_kit_args(renderer_active_gpu_ordinal=1),
            "--/renderer/multiGpu/autoEnable=false "
            "--/renderer/multiGpu/maxGpuCount=1 "
            "--/renderer/activeGpu=1",
        )
        with self.assertRaisesRegex(ValueError, "workload is immutable"):
            StairResourceShadowWorkload(max_iterations=65, budget_steps=6_389_760)

    def test_fixture_is_the_current_qualified_official_reference(self) -> None:
        fixture = resolve_stair_resource_shadow_fixture(
            self.config,
            expected_initial_policy_state_hash=INITIAL_POLICY_STATE_HASH,
            project_root=PROJECT_ROOT,
        )
        contract = fixture.to_contract_dict()

        self.assertEqual(contract["fixture_id"], "official_reference_complete_reward_v1")
        self.assertEqual(contract["subject"]["kind"], "reference")
        self.assertEqual(
            contract["reward_program_qualification_hash"],
            fixture.reward_program.qualification_hash,
        )
        self.assertEqual(fixture.fixture_hash, canonical_json_hash(contract))
        self.assertEqual(
            contract["expected_initial_policy_state_hash"],
            INITIAL_POLICY_STATE_HASH,
        )
        self.assertFalse(contract["acceptance_authority"])
        self.assertFalse(contract["ledger_authority"])
        other = resolve_stair_resource_shadow_fixture(
            self.config,
            expected_initial_policy_state_hash="8" * 64,
            project_root=PROJECT_ROOT,
        )
        self.assertNotEqual(other.fixture_hash, fixture.fixture_hash)

    def test_shadow_wrapper_delegates_to_the_only_training_body(self) -> None:
        sentinel = StairTrainingExecutionResult(
            status="pass",
            training_executed=True,
            checkpoint={"checksum": "sha256:" + "a" * 64},
            runtime_activation_summary={},
            activation_preflight={},
            training={"workload_kind": "production"},
            candidate_failure=None,
            errors=(),
        )
        preflight = object.__new__(VerifiedStairExecutionSourceManifest)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            VerifiedStairExecutionSourceManifest,
            "validate_for",
        ), patch(
            "true_rl_agent.tasks.stair_climbing.resource_shadow."
            "_execute_stair_training",
            return_value=sentinel,
        ) as execute:
            result = execute_stair_resource_shadow_training(
                config=self.config,
                source_manifest_preflight=preflight,
                agent_configuration=object(),  # type: ignore[arg-type]
                reward_program=object(),  # type: ignore[arg-type]
                device="cuda:0",
                run_dir=Path(tmp) / "run",
                checkpoint_output=Path(tmp) / "checkpoint.pt",
                run_name="p6a-lane-a",
                expected_initial_policy_state_hash="b" * 64,
            )

        self.assertEqual(result.status, sentinel.status)
        self.assertEqual(
            result.training["workload_kind"],
            STAIR_RESOURCE_SHADOW_PRESET_ID,
        )
        self.assertEqual(result.training["execution_body_workload_kind"], "production")
        workload = execute.call_args.kwargs["workload"]
        # The private execution branch stays "production" so the frozen
        # isaaclab_execution.py hot-path bytes do not change; task evidence above
        # projects the actual non-authoritative preset id.
        self.assertEqual(workload.kind, "production")
        self.assertEqual(workload.num_envs, 4096)
        self.assertEqual(workload.num_steps_per_env, 24)
        self.assertEqual(workload.max_iterations, 64)
        self.assertEqual(workload.budget_steps, 6_291_456)
        self.assertEqual(workload.checkpoint_save_interval, 50)
        self.assertEqual(
            execute.call_args.kwargs["expected_initial_policy_state_hash"],
            "b" * 64,
        )

    def test_child_cli_requires_every_frozen_binding_and_device(self) -> None:
        parser = self.script._build_parser()
        required = [
            "--plan",
            str(PROJECT_ROOT / "plan.json"),
            "--plan-hash",
            "a" * 64,
            "--session-genesis-hash",
            "b" * 64,
            "--execution-environment-manifest-hash",
            "c" * 64,
            "--fixture-hash",
            "d" * 64,
            "--workload-hash",
            "e" * 64,
            "--resource-plan-hash",
            "f" * 64,
            "--expected-initial-policy-state-hash",
            "1" * 64,
            "--job-id",
            "single-gpu0-r1-lane-a",
            "--case-id",
            "single-gpu0-r1",
            "--lane-id",
            "lane-a",
            "--physical-execution-device",
            "cuda:0",
            "--output-dir",
            str(PROJECT_ROOT / "out"),
            "--active-marker",
            str(PROJECT_ROOT / "out" / "active.json"),
            "--checkpoint-path",
            str(PROJECT_ROOT / "out" / "checkpoint.pt"),
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(required)
        args = parser.parse_args([*required, "--device", "cuda:0"])
        self.assertEqual(args.device, "cuda:0")
        self.assertTrue(args.headless)

    def test_child_writes_create_once_active_and_closed_non_authoritative_summary(
        self,
    ) -> None:
        initial_hash = "2" * 64
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp).resolve() / "lane"
            output_dir.mkdir(parents=True)
            checkpoint_path = output_dir / "checkpoint.pt"
            checkpoint_path.write_bytes(b"fixed-shadow-checkpoint")
            checkpoint_hash = file_sha256(checkpoint_path)
            self.assertIsNotNone(checkpoint_hash)
            fixture_contract = {
                "fixture_id": "official",
                "expected_initial_policy_state_hash": initial_hash,
            }
            fixture_hash = canonical_json_hash(fixture_contract)
            result = StairTrainingExecutionResult(
                status="pass",
                training_executed=True,
                checkpoint={
                    "uri": str(checkpoint_path),
                    "checksum": checkpoint_hash,
                    "role": "train_output",
                },
                runtime_activation_summary={},
                activation_preflight={},
                training={
                    "workload_kind": STAIR_RESOURCE_SHADOW_PRESET_ID,
                    "device": "cuda:0",
                    "initial_policy_state_hash": initial_hash,
                    "expected_initial_policy_state_hash": initial_hash,
                    "initial_policy_state_match": True,
                },
                candidate_failure=None,
                errors=(),
            )
            context = self.script._ShadowChildContext(
                plan=object(),
                plan_path=Path(tmp) / "plan.json",
                output_dir=output_dir,
                active_marker=output_dir / "active.json",
                checkpoint_path=checkpoint_path,
                summary_path=output_dir / "summary.json",
                config=SimpleNamespace(
                    training_agent_configuration=lambda: object()
                ),
                fixture=SimpleNamespace(
                    reward_program=object(),
                    to_contract_dict=lambda: dict(fixture_contract),
                ),
                source_manifest=SimpleNamespace(
                    to_summary_dict=lambda: {"status": "pass"}
                ),
                environment_verification={
                    "status": "pass",
                    "expected_manifest_hash": "6" * 64,
                    "actual_manifest_hash": "6" * 64,
                },
                expected_initial_policy_state_hash=initial_hash,
                physical_execution_device="cuda:0",
                process_execution_device="cuda:0",
                physical_gpu_uuid="GPU-aaaaaaaa",
                renderer_active_gpu_ordinal=0,
            )
            args = SimpleNamespace(
                plan_hash="4" * 64,
                session_genesis_hash="5" * 64,
                execution_environment_manifest_hash="6" * 64,
                fixture_hash=fixture_hash,
                workload_hash=stair_resource_shadow_workload_hash(),
                resource_plan_hash="8" * 64,
                job_id="paired-r1-lane-a",
                case_id="paired-r1",
                lane_id="lane-a",
                device="cuda:0",
                physical_execution_device="cuda:0",
            )
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                self.script,
                "execute_stair_resource_shadow_training",
                return_value=result,
            ) as execute, patch.object(
                self.script.time,
                "monotonic_ns",
                side_effect=(100, 250),
            ), patch.object(self.script.os, "getpid", return_value=123), patch.object(
                self.script.os,
                "getpgid",
                return_value=456,
                create=True,
            ):
                exit_code = self.script._run_shadow_lane(
                    args=args,
                    context=context,
                )

            self.assertEqual(exit_code, 0)
            marker = json.loads(context.active_marker.read_text(encoding="utf-8"))
            summary = json.loads(context.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(marker["active_start_monotonic_ns"], 100)
            self.assertEqual(marker["process_group_id"], 456)
            self.assertEqual(summary["active_end_monotonic_ns"], 250)
            self.assertEqual(summary["active_duration_ns"], 150)
            self.assertTrue(summary["initial_policy_state_match"])
            self.assertFalse(summary["acceptance_authority"])
            self.assertFalse(summary["ledger_authority"])
            self.assertEqual(
                execute.call_args.kwargs["expected_initial_policy_state_hash"],
                initial_hash,
            )
            with self.assertRaises(FileExistsError):
                self.script._create_once_json(context.active_marker, marker)
            loaded_marker = load_stair_resource_shadow_active_marker(
                context.active_marker,
                output_dir=output_dir,
                expected_plan_hash=args.plan_hash,
                expected_job_id=args.job_id,
                expected_case_id=args.case_id,
                expected_lane_id=args.lane_id,
                expected_physical_execution_device=(
                    context.physical_execution_device
                ),
                expected_process_execution_device=args.device,
                expected_physical_gpu_uuid=context.physical_gpu_uuid,
                expected_renderer_active_gpu_ordinal=(
                    context.renderer_active_gpu_ordinal
                ),
            )
            loaded_summary = load_stair_resource_shadow_summary(
                context.summary_path,
                output_dir=output_dir,
                expected_checkpoint_path=checkpoint_path,
                expected_plan_hash=args.plan_hash,
                expected_session_genesis_hash=args.session_genesis_hash,
                expected_execution_environment_manifest_hash=(
                    args.execution_environment_manifest_hash
                ),
                expected_fixture_hash=args.fixture_hash,
                expected_workload_hash=args.workload_hash,
                expected_resource_plan_hash=args.resource_plan_hash,
                expected_initial_policy_state_hash=initial_hash,
                expected_job_id=args.job_id,
                expected_case_id=args.case_id,
                expected_lane_id=args.lane_id,
                expected_physical_execution_device=(
                    context.physical_execution_device
                ),
                expected_process_execution_device=args.device,
                expected_physical_gpu_uuid=context.physical_gpu_uuid,
                expected_renderer_active_gpu_ordinal=(
                    context.renderer_active_gpu_ordinal
                ),
                expected_status="pass",
            )
            self.assertEqual(loaded_marker, marker)
            self.assertEqual(loaded_summary, summary)

            invalid_marker = dict(marker)
            invalid_marker["renderer_active_gpu_ordinal"] = True
            invalid_marker_path = output_dir / "invalid-active.json"
            invalid_marker_path.write_bytes(
                canonical_json_bytes(invalid_marker, trailing_newline=True)
            )
            with self.assertRaisesRegex(ValueError, "renderer GPU ordinal"):
                load_stair_resource_shadow_active_marker(
                    invalid_marker_path,
                    output_dir=output_dir,
                    expected_plan_hash=args.plan_hash,
                    expected_job_id=args.job_id,
                    expected_case_id=args.case_id,
                    expected_lane_id=args.lane_id,
                    expected_physical_execution_device=(
                        context.physical_execution_device
                    ),
                    expected_process_execution_device=args.device,
                    expected_physical_gpu_uuid=context.physical_gpu_uuid,
                    expected_renderer_active_gpu_ordinal=1,
                )

            invalid_summary = dict(summary)
            invalid_summary["training"] = dict(summary["training"])
            invalid_summary["training"]["device"] = "cuda:1"
            invalid_summary_path = output_dir / "invalid-summary.json"
            invalid_summary_path.write_bytes(
                canonical_json_bytes(invalid_summary, trailing_newline=True)
            )
            with self.assertRaisesRegex(ValueError, "training binding"):
                load_stair_resource_shadow_summary(
                    invalid_summary_path,
                    output_dir=output_dir,
                    expected_checkpoint_path=checkpoint_path,
                    expected_plan_hash=args.plan_hash,
                    expected_session_genesis_hash=args.session_genesis_hash,
                    expected_execution_environment_manifest_hash=(
                        args.execution_environment_manifest_hash
                    ),
                    expected_fixture_hash=args.fixture_hash,
                    expected_workload_hash=args.workload_hash,
                    expected_resource_plan_hash=args.resource_plan_hash,
                    expected_initial_policy_state_hash=initial_hash,
                    expected_job_id=args.job_id,
                    expected_case_id=args.case_id,
                    expected_lane_id=args.lane_id,
                    expected_physical_execution_device=(
                        context.physical_execution_device
                    ),
                    expected_process_execution_device=args.device,
                    expected_physical_gpu_uuid=context.physical_gpu_uuid,
                    expected_renderer_active_gpu_ordinal=(
                        context.renderer_active_gpu_ordinal
                    ),
                    expected_status="pass",
                )


def _load_entrypoint():
    path = PROJECT_ROOT / "scripts" / "isaaclab_stair_resource_shadow_train.py"
    module_name = "test_isaaclab_stair_resource_shadow_train"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Stair resource-shadow entrypoint")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
