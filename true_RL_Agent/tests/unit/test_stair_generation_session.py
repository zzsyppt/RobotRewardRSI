from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_MANIFEST_V2_SCHEMA_VERSION,
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.prompt_artifacts import PromptPayloadRef
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.generation_session import (
    StairGenerationPilotConfig,
    _assert_target_gpus_compute_idle,
    _create_foundation,
    _preflight_provider_credential,
    _provider_actuals,
    _resource_summary_evidence,
    _run_one_generation,
    _session_genesis_attempt_scope,
    _StairCostGuardedProvider,
    _verify_source_checkout,
    discover_two_gpu_inventory,
    run_stair_generation_pilot,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_PILOT_MODE,
)
from true_rl_agent.tasks.stair_climbing.phased_cost import (
    stair_phased_evaluation_cost_contract,
)

GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
SHA = "a" * 40
DIGEST = "b" * 64


class StairGenerationSessionTests(unittest.TestCase):
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

    def test_foundation_freezes_two_host_bindings_and_fixed_pilot_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
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
            self.assertEqual(foundation.cost_contract.successful_eval_seed_units, 41)
            self.assertEqual(foundation.cost_contract.physical_train_job_cap, 4)
            self.assertEqual(foundation.cost_contract.physical_eval_seed_unit_cap, 48)
            self.assertEqual(
                foundation.cost_contract.scientific_cost_basis.phased_evaluation_cost_contract_hash,
                foundation.phased_adapter.cost_contract()["contract_hash"],
            )
            self.assertFalse(foundation.final_confirmation_policy["enabled"])

    def test_foundation_rejects_self_consistent_phased_cost_science_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            task_config = StairClimbingTaskConfig.defaults(
                config.project_root / "configs" / "tasks" / "stair_protocol_v2.json"
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
                "true_rl_agent.tasks.stair_climbing.generation_session.subprocess.run",
                side_effect=(
                    SimpleNamespace(stdout=SHA + "\n"),
                    SimpleNamespace(stdout=" M tracked.py\n"),
                ),
            ) as run,
            self.assertRaisesRegex(
                ValueError,
                "tracked modifications",
            ),
        ):
            _verify_source_checkout(root, expected_commit=SHA)

        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["git", "status", "--porcelain", "--untracked-files=no"],
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

    def test_driver_calls_existing_generation_coordinator_once_and_closes_one_row(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = SimpleNamespace(output_dir=root, session_id="pilot")
            acceptance = Mock()
            composition = SimpleNamespace(
                task_plugin=SimpleNamespace(
                    task_spec=SimpleNamespace(acceptance_protocol=acceptance)
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
                    "true_rl_agent.tasks.stair_climbing.generation_session._collect_actual_cost_usage",
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
                    task_spec=SimpleNamespace(acceptance_protocol=acceptance)
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
                    "true_rl_agent.tasks.stair_climbing.generation_session._collect_actual_cost_usage",
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

    def test_manifest_version_constant_is_the_bound_generation_contract(self) -> None:
        self.assertEqual(
            GENERATION_MANIFEST_V2_SCHEMA_VERSION,
            "true_rl_agent.generation_manifest.v2",
        )


if __name__ == "__main__":
    unittest.main()
