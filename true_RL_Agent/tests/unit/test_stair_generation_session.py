from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch


from true_rl_agent.agent_core.generation_contract import GENERATION_MANIFEST_V2_SCHEMA_VERSION
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider,
)
from true_rl_agent.agent_core.provider_limits import (
    PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND,
)
from true_rl_agent.candidates.prompt_projection import build_prompt_payload
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.prompt_artifacts import PromptPayloadRef
from true_rl_agent.runner.job_attempts import (
    JobAttemptJournal,
    JobAttemptScope,
    PreGenesisExecutionScope,
)
from true_rl_agent.runner.specs import GpuProcessBinding
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity
from true_rl_agent.agent_core.generation_runtime import generation_provider
from true_rl_agent.agent_core.generation_accounting import collect_actual_cost_usage
from true_rl_agent.tasks.stair_climbing import generation_history
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.generation_history import (
    STAIR_GENERATION_PROVIDER_THINKING_DISABLED_TEMPERATURE,
    STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME,
    STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION,
    CompletedStairGenerationSession,
    HistoricalStairGenerationConfig,
    _completed_scientific_generation_count,
    _formal_normal_closed_generation_boundaries_are_valid,
    _provider_policy,
    _session_genesis_attempt_scope,
    _unified_validation_selector_reversal_metrics,
    _validation_authoring_metrics,
    _validation_workload_delta,
    _verify_source_checkout,
    load_completed_stair_generation_session,
)
from true_rl_agent.agent_core.generation_resources import (
    assert_target_gpus_compute_idle,
    _attempt_is_in_generation_prefix,
    build_resource_closure_evidence,
    _resource_summary_evidence,
)
from true_rl_agent.agent_core.generation_provider_budget import (
    collect_provider_actuals,
    CostGuardedProvider,
)
from true_rl_agent.agent_core.generation_session import (
    _publish_or_load_parent_bootstrap_cost_boundary,
)
from true_rl_agent.agent_core.generation_closure import publish_or_validate_generation_closure
from true_rl_agent.agent_core.generation_operations import (
    _reference_qualification_phase_reserves,
    OperationAllocationState,
    read_operation_slot,
    _strict_retry_is_still_possible,
)
from true_rl_agent.agent_core.generation_job_evidence import (
    validated_job_cost_artifacts,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import DEVELOPMENT_GENERATION_VALIDATION_MODE, STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS, STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES, STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS, STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES, StairGenerationSessionCostContractV2, build_stair_generation_scientific_cost_basis, stair_generation_operation_cost_reserves
from true_rl_agent.tasks.stair_climbing.row_catalog import STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID

GPU_A = "GPU-00000000-0000-0000-0000-000000000001"
GPU_B = "GPU-00000000-0000-0000-0000-000000000002"
SHA = "a" * 40
DIGEST = "b" * 64


def _write_final_source_fixture(foundation, sidecar):
    source = generation_history._load_formal_source_checkout_identity(foundation)
    body = {"schema_version": generation_history._FINAL_SOURCE_CHECKOUT_SCHEMA,
            "source_commit": foundation.config.expected_source_commit,
            "generation_source_checkout_hash": source["source_checkout_hash"]}
    payload = {**body, "source_checkout_hash": canonical_json_hash(body)}
    sidecar.write_bytes(canonical_json_bytes(payload, trailing_newline=True))
    return payload


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
                    expected_cell_variant_id=STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID,
                    expected_source_commit=SHA,
                )

            self.assertEqual(tuple(output_dir.iterdir()), before)
            self.assertIn(
                "CompletedStairGenerationSession",
                generation_history.__all__,
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
                    uses_operation_reserves=False,
                ),
            )
            before = tuple(output_dir.iterdir())
            with patch(
                "true_rl_agent.agent_core.generation_accounting."
                "collect_provider_actuals",
                return_value=provider_actuals,
            ):
                usage, charges, open_reservations = (
                    collect_actual_cost_usage(
                        foundation,
                        genesis=None,
                        ledger=[],
                    )
                )

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
                validated_job_cost_artifacts(foundation, genesis=genesis)
            )

            self.assertEqual(train_steps, {})
            self.assertEqual(eval_costs, {})
            self.assertEqual(evidence_hash, canonical_json_hash([]))

    def _config(self, root: Path) -> HistoricalStairGenerationConfig:
        return HistoricalStairGenerationConfig(
            cell_variant_id=STAIR_G1_37DOF_UNIFIED_V2_CELL_VARIANT_ID,
            project_root=Path(__file__).resolve().parents[2],
            output_dir=(root / "session").resolve(),
            lease_root=(root / "leases").resolve(),
            session_id="p6b-generation-pilot",
            expected_source_commit=SHA,
            machine_id="test-machine",
            gpu0_uuid=GPU_A,
            gpu1_uuid=GPU_B,
            max_allocated_active_gpu_hours=12.0,
            max_simulated_control_steps=1_000_000_000,
            max_output_bytes=40 * 1024**3,
        )


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


    def test_provider_policy_binds_both_adaptive_request_shapes(self) -> None:
        policy = {**_provider_policy(), "api_key_env": "FIXTURE_KEY", "temperature": 0.2}
        config = generation_provider(SimpleNamespace(
            provider_policy=policy,
            config=SimpleNamespace(output_dir=Path("provider-output"), provider_env_file=Path("provider.env")),
            cost_contract=SimpleNamespace(to_dict=lambda: {"provider": {}}),
            session_contract=SimpleNamespace(max_generations=15),
        ), generation_id="generation-001", lane_id="lane-a").delegate.config
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


    # Frozen P6 cost fixtures use their original 7200 s train deadline.
    # P2's current default is covered by composition/placement tests.


    def _final_source_fixture(self, root: Path, executable_commit: str):
        config = replace(self._config(root), expected_source_commit=executable_commit)
        source = {"schema_version": STAIR_GENERATION_SOURCE_CHECKOUT_SCHEMA_VERSION, "source_commit": SHA}
        source["source_checkout_hash"] = canonical_json_hash(source)
        config.output_dir.mkdir()
        (config.output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME).write_bytes(
            canonical_json_bytes(source, trailing_newline=True))
        final_dir = config.output_dir / generation_history.STAIR_FINAL_CONFIRMATION_DIRNAME
        final_dir.mkdir()
        return (
            SimpleNamespace(config=config, source_commit=SHA),
            source,
            final_dir / generation_history._FINAL_SOURCE_CHECKOUT_FILENAME,
        )

    def test_final_execution_source_preserves_old_receipt_without_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            foundation, source, sidecar = self._final_source_fixture(Path(tmp), SHA)
            original = (foundation.config.output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME).read_bytes()
            self.assertEqual(generation_history._formal_final_execution_source(foundation), source)
            self.assertFalse(sidecar.exists())
            self.assertEqual(
                (foundation.config.output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME).read_bytes(),
                original,
            )


    def test_final_execution_source_bound_repair_is_immutable_but_audit_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            foundation, _, sidecar = self._final_source_fixture(Path(tmp), "c" * 40)
            recorded = _write_final_source_fixture(foundation, sidecar)
            recorded_bytes = sidecar.read_bytes()
            later = SimpleNamespace(
                config=replace(foundation.config, expected_source_commit="d" * 40),
                source_commit=SHA,
            )
            self.assertEqual(
                generation_history._formal_final_execution_source(later), recorded
            )
            self.assertEqual(sidecar.read_bytes(), recorded_bytes)

    def test_final_execution_source_rejects_rehashed_wrong_generation_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            foundation, _, sidecar = self._final_source_fixture(Path(tmp), "c" * 40)
            recorded = _write_final_source_fixture(foundation, sidecar)
            recorded.pop("source_checkout_hash")
            recorded["generation_source_checkout_hash"] = "e" * 64
            recorded["source_checkout_hash"] = canonical_json_hash(recorded)
            sidecar.write_bytes(canonical_json_bytes(recorded, trailing_newline=True))
            with self.assertRaisesRegex(ValueError, "finalizer source checkout mismatch"):
                generation_history._formal_final_execution_source(foundation)


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
                publish_or_validate_generation_closure(
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
                    operational_evidence_namespace="true_rl_agent.stair",
                    final_confirmation_seed_units=0,
                    to_dict=lambda: {"operation_reserves": {"entries": [
                        item.to_dict() for item in stair_generation_operation_cost_reserves()
                    ]}},
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
                    "gpu_runtime_evidence": {
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
                    "true_rl_agent.agent_core.generation_resources."
                    "assert_target_gpus_compute_idle",
                    return_value=final_idle,
                ),
                patch(
                    "true_rl_agent.agent_core.generation_resources."
                    "_reacquire_generation_gpu_leases",
                    return_value=lease_reacquire,
                ),
                patch(
                    "true_rl_agent.agent_core.generation_resources."
                    "validate_gpu_runtime_summary_evidence",
                    side_effect=validate_summary,
                ),
            ):
                resource_evidence = build_resource_closure_evidence(
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
                closure = publish_or_validate_generation_closure(
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
                    publish_or_validate_generation_closure(
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
                    publish_or_validate_generation_closure(
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
                    publish_or_validate_generation_closure(
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
                    publish_or_validate_generation_closure(
                        foundation,
                        ledger=[record],
                        generation_record=record,
                        generation_execution_receipt={"kind": "execution"},
                        actual_cost_receipt=cost_receipt,
                        resource_evidence={},
                        segment_id="unknown-summary-segment",
                        require_existing=True,
                    )


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
                'true_rl_agent.agent_core.generation_session.collect_actual_cost_usage',
                return_value=(usage, ("conservative",), ()),
            ) as collect,
            patch(
                'true_rl_agent.agent_core.generation_session._latest_cost_head',
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
                slot, reserve = read_operation_slot(
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
                read_operation_slot(
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

            legacy_cost_contract = replace(
                cost_contract,
                scientific_cost_basis=replace(cost_contract.scientific_cost_basis,
                    parent_bootstrap_seed_units=unified_config.public_seed_count),
            )
            legacy_slot, legacy_reserve = read_operation_slot(
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
            self.assertEqual(legacy_reserve.to_dict(), frozen.to_dict())

            empty_state = OperationAllocationState(
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


    def test_source_checkout_rejects_tracked_dirty_status(self) -> None:
        root = Path(__file__).resolve().parents[2]
        with (
            patch(
                "true_rl_agent.tasks.stair_climbing.generation_history.require_stair_source_checkout",
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
            "true_rl_agent.agent_core.generation_resources.subprocess.run",
            return_value=SimpleNamespace(stdout=""),
        ):
            observed = assert_target_gpus_compute_idle(
                "nvidia-smi",
                expected=(GPU_A, GPU_B),
                stage="startup",
            )
        self.assertEqual(observed["rows"], [])

        with (
            patch(
                "true_rl_agent.agent_core.generation_resources.subprocess.run",
                return_value=SimpleNamespace(stdout=f"{GPU_A}, 1234\n"),
            ),
            self.assertRaisesRegex(ValueError, "not compute-idle"),
        ):
            assert_target_gpus_compute_idle(
                "nvidia-smi",
                expected=(GPU_A, GPU_B),
                stage="final",
            )


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
                config=SimpleNamespace(output_dir=output, session_id="pilot"),
                cost_contract=SimpleNamespace(to_dict=lambda: {"provider": {"input_units_per_attempt_reserve": 300000, "output_units_per_attempt_cap": 32768}}),
            )
            with patch(
                "true_rl_agent.agent_core.generation_provider_budget.RoundAuthoringJournal.snapshot",
                return_value=snapshot,
            ):
                actual = collect_provider_actuals(foundation, ledger=[])

            self.assertEqual(actual["attempts"], 2)
            self.assertEqual(actual["request_units"], 300_010)
            self.assertEqual(actual["response_units"], 32_788)
            self.assertEqual(actual["total_units"], 332_798)
            self.assertEqual(actual["conservative_attempts"], 1)

            (output / "authoring" / "pilot-generation-004" / "lane-a").mkdir(
                parents=True
            )
            with self.assertRaisesRegex(ValueError, "unknown generation"):
                collect_provider_actuals(foundation, ledger=[])

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
            provider = CostGuardedProvider(
                delegate=delegate,
                provider_caps={"attempts_per_lane_per_generation": 5, "input_units_per_attempt_reserve": 300000, "output_units_per_attempt_cap": 32768},
                output_dir=output,
                generation_id=generation_id,
                lane_id="lane-b",
            )
            with (
                patch(
                    "true_rl_agent.agent_core.generation_provider_budget.RoundAuthoringJournal.snapshot",
                    side_effect=[prior, current],
                ),
                patch(
                    "true_rl_agent.agent_core.generation_provider_budget.verify_prompt_payload",
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
            provider = CostGuardedProvider(
                delegate=delegate,
                provider_caps={"attempts_per_lane_per_generation": 5, "input_units_per_attempt_reserve": 300000, "output_units_per_attempt_cap": 32768},
                output_dir=output,
                generation_id=generation_id,
                lane_id="lane-b",
            )
            # A supplier's output-limit deviation is charged, not rounded away.
            # It may use spare aggregate budget, but cannot expand that budget.
            for input_units, output_units, generations, error in (
                (100, 20, 1, None),
                (100, 65_000, 2, None),
                (300_001, 20, 2, "input exceeded"),
                (100, 32_769, 1, "unit cap cannot reserve"),
                (100, 400_000, 2, "unit cap cannot reserve"),
            ):
                with self.subTest(input_units=input_units, output_units=output_units,
                                  generations=generations):
                    prior.attempts[0]["provider_attempt"]["provider_usage"] = {
                        "request_units": input_units,
                        "response_units": output_units,
                        "total_units": input_units + output_units,
                    }
                    delegate.reset_mock()
                    provider = CostGuardedProvider(
                        delegate=delegate,
                        provider_caps=provider.provider_caps,
                        output_dir=output,
                        generation_id=generation_id,
                        lane_id="lane-b",
                        max_generations=generations,
                    )
                    with (
                        patch(
                            "true_rl_agent.agent_core.generation_provider_budget.RoundAuthoringJournal.snapshot",
                            side_effect=[prior, current],
                        ),
                        patch(
                            "true_rl_agent.agent_core.generation_provider_budget.verify_prompt_payload",
                            return_value=prompt_payload,
                        ),
                    ):
                        if error is not None:
                            with self.assertRaisesRegex(RuntimeError, error):
                                provider.next_authoring_response(prompt_payload=prompt_payload)
                            delegate.next_authoring_response.assert_not_called()
                        else:
                            actual = provider.next_authoring_response(prompt_payload=prompt_payload)
                            self.assertIs(actual, expected)
                            delegate.next_authoring_response.assert_called_once_with(
                                prompt_payload=prompt_payload
                            )

    def test_resource_report_binds_raw_summary_pid_and_uuid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            foundation = SimpleNamespace(config=config,
                gpu_identities=(GpuDeviceIdentity("cuda:0", config.machine_id, GPU_A),
                                GpuDeviceIdentity("cuda:1", config.machine_id, GPU_B)),
                gpu_bindings=(GpuProcessBinding("cuda:0", "cuda:0", "b" * 64),
                              GpuProcessBinding("cuda:1", "cuda:0", "c" * 64)))
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
                "gpu_runtime_evidence": {
                    "external_compute_context_observation": {
                        "process_id": 123,
                        "rows": [{"gpu_uuid": GPU_A}],
                    }
                },
            }
            with (
                patch(
                    "true_rl_agent.agent_core.generation_resources._completed_bound_job_attempts",
                    return_value={
                        "job-a": foundation.gpu_bindings[0].resource_binding_hash
                    },
                ),
                patch(
                    "true_rl_agent.agent_core.generation_resources.validate_gpu_runtime_summary_evidence",
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
            "true_rl_agent.agent_core.generation_resources."
            "read_operation_slot"
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
                "true_rl_agent.tasks.stair_climbing.generation_history."
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


    def test_manifest_version_constant_is_the_bound_generation_contract(self) -> None:
        self.assertEqual(
            GENERATION_MANIFEST_V2_SCHEMA_VERSION,
            "true_rl_agent.generation_manifest.v2",
        )


if __name__ == "__main__":
    unittest.main()
