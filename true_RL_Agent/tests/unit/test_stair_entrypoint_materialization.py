from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.isaaclab_stair_rsl_rl_eval import (  # noqa: E402
    _build_parser as build_eval_parser,
)
from scripts.isaaclab_stair_rsl_rl_eval import main as stair_eval_main  # noqa: E402
from scripts.isaaclab_stair_rsl_rl_train import (  # noqa: E402
    _build_parser as build_train_parser,
)
from scripts.isaaclab_stair_rsl_rl_train import (  # noqa: E402
    _load_and_qualify_program,
    _status_summary,
)
from scripts.isaaclab_stair_rsl_rl_train import main as stair_train_main  # noqa: E402

from tests.fakes.backend_environment import (  # noqa: E402
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
    make_test_isaac_lab_execution_environment_manifest,
)
from tests.fakes.evaluation import make_test_candidate_package  # noqa: E402
from tests.fakes.reference_programs import make_verified_stair_reference  # noqa: E402
from true_rl_agent.backends.isaac_lab_rsl_rl import (  # noqa: E402
    IsaacLabRslRlBackendAdapter,
)
from true_rl_agent.candidates.verifier import seal_and_verify_candidate  # noqa: E402
from true_rl_agent.composition.base import (  # noqa: E402
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.evaluation.binding import (  # noqa: E402
    EvaluationBindingV1,
    canonical_json_hash,
)
from true_rl_agent.evaluation.phases import EvaluationPhase  # noqa: E402
from true_rl_agent.json_contract import canonical_json_text  # noqa: E402
from true_rl_agent.runner.base import RunnerExecutionError  # noqa: E402
from true_rl_agent.runner.checkpoint import file_sha256  # noqa: E402
from true_rl_agent.runner.job_argv import (  # noqa: E402
    validate_eval_job_argv,
    validate_train_job_argv,
)
from true_rl_agent.runner.process_handle import (  # noqa: E402
    OWNED_PROCESS_IDENTITY_SCHEMA_VERSION,
    run_owned_process,
)
from true_rl_agent.runner.specs import CheckpointRef, GpuProcessBinding  # noqa: E402
from true_rl_agent.runner.subprocess_runner import SubprocessRunner  # noqa: E402
from true_rl_agent.tasks.stair_climbing.candidate_failure import (  # noqa: E402
    STAIR_CANDIDATE_FAILURE_CONTRACT,
    StairCandidateProgramFailure,
)
from true_rl_agent.tasks.stair_climbing.component_smoke import (  # noqa: E402
    STAIR_SMOKE_NUM_ENVS,
    StairClimbingComponentSmokeAdapter,
)
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (  # noqa: E402
    STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
    validate_stair_eval_invocation,
    validate_stair_train_invocation,
)
from true_rl_agent.tasks.stair_climbing.gpu_process_binding import (  # noqa: E402
    attach_stair_gpu_runtime_evidence,
    capture_stair_gpu_compute_context,
    validate_stair_gpu_process_binding_invocation,
    validate_stair_gpu_runtime_summary_evidence,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (  # noqa: E402
    STAIR_REWARD_ENVIRONMENT_AXIS_EVIDENCE_SCHEMA_VERSION,
    StairRewardProgramError,
    qualify_stair_candidate_reward_program,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (  # noqa: E402
    STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
    STAIR_FORMAL_EVAL_NAMESPACE,
)

TEST_INITIAL_POLICY_STATE_HASH = "1" * 64
TEST_GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _run_process_group_leader_identity_child(root: Path) -> dict[str, object]:
    child_source = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from true_rl_agent.runner.process_handle import "
        "current_owned_process_identity\n"
        "print(json.dumps({"
        "'identity': current_owned_process_identity(), "
        "'session_id': os.getsid(0) if hasattr(os, 'getsid') else None"
        "}, sort_keys=True))\n"
    )
    result = run_owned_process(
        [sys.executable, "-c", child_source],
        cwd=ROOT,
        timeout_s=10,
        log_dir=root / "owned-identity-child",
    )
    if result.returncode != 0:
        raise AssertionError(
            "owned identity child failed: " + result.stderr_tail
        )
    payload = json.loads(result.stdout_tail.strip())
    identity = payload["identity"]
    if identity["process_group_id"] != identity["process_id"]:
        raise AssertionError("owned identity child is not its process-group leader")
    if os.name != "nt":
        if identity["process_id"] != result.pid:
            raise AssertionError("owned identity child PID mismatch")
        if payload["session_id"] != result.pid:
            raise AssertionError("owned identity child did not start a new session")
    return identity


class StairEntrypointMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        self.reference = make_verified_stair_reference(
            Path(self.temporary.name) / "reference",
            task_id=STAIR_TASK_ID,
            capability_profile_hash=(
                self.config.protocol_v2.capability_profile.profile_hash
            ),
        )
        for target in (
            "scripts.isaaclab_stair_rsl_rl_train."
            "verify_current_isaac_lab_execution_environment",
            "scripts.isaaclab_stair_rsl_rl_eval."
            "verify_current_isaac_lab_execution_environment",
        ):
            patcher = patch(
                target,
                return_value=dict(
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
                ),
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_component_smoke_fixture_is_spawn_serializable(self) -> None:
        fixture = StairClimbingComponentSmokeAdapter(
            self.config
        ).build_component_smoke_fixture()

        restored = pickle.loads(pickle.dumps(fixture))
        first_field = self.config.protocol_v2.complete_reward_runtime.input_fields[0]
        value = getattr(restored.reward_inputs, first_field.name)

        self.assertEqual(restored.task_id, STAIR_TASK_ID)
        self.assertEqual(
            tuple(value.shape),
            (STAIR_SMOKE_NUM_ENVS, *first_field.trailing_shape),
        )
        with self.assertRaises(AttributeError):
            getattr(restored.reward_inputs, "not_a_reward_input")

    def test_materialized_jobs_use_stair_entrypoints_and_pass_pure_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            composition, jobs = self._materialize(
                Path(tmp),
                "stair-materialized",
                execution_device="cuda:1",
            )
            train_argv = jobs.train_spec.metadata["cli_argv"]
            eval_argv = jobs.eval_spec.metadata["cli_argv"]

            self.assertEqual(train_argv[1], "scripts/isaaclab_stair_rsl_rl_train.py")
            self.assertEqual(eval_argv[1], "scripts/isaaclab_stair_rsl_rl_eval.py")
            self.assertEqual(jobs.train_spec.budget_steps, 196_608_000)
            self.assertEqual(
                train_argv[train_argv.index("--max-iterations") + 1],
                "2000",
            )
            self.assertEqual(eval_argv.count("--program-source-checksum"), 1)
            self.assertEqual(
                jobs.train_spec.expected_initial_policy_state_hash,
                TEST_INITIAL_POLICY_STATE_HASH,
            )
            self.assertEqual(
                train_argv.count("--expected-initial-policy-state-hash"),
                1,
            )
            self.assertEqual(
                train_argv.count("--candidate-failure-contract-hash"),
                1,
            )
            self.assertEqual(
                jobs.train_spec.candidate_failure_contract,
                STAIR_CANDIDATE_FAILURE_CONTRACT,
            )
            self.assertEqual(train_argv.count("--device"), 1)
            self.assertEqual(eval_argv.count("--device"), 1)
            self.assertNotIn("--gpu-host-execution-device", train_argv)
            self.assertNotIn("--gpu-host-execution-device", eval_argv)
            self.assertEqual(jobs.train_spec.execution_device, "cuda:1")
            self.assertEqual(jobs.eval_spec.execution_device, "cuda:1")
            self.assertEqual(
                train_argv[train_argv.index("--device") + 1],
                "cuda:1",
            )
            self.assertEqual(
                eval_argv[eval_argv.index("--device") + 1],
                "cuda:1",
            )
            self.assertNotIn("strong_push", " ".join(train_argv + eval_argv))
            train_args = build_train_parser().parse_args(train_argv[2:])
            train_context = validate_stair_train_invocation(
                train_args,
                project_root=ROOT,
            )
            self.assertEqual(
                train_context.training_binding.binding_hash,
                jobs.train_spec.training_binding.binding_hash,
            )
            self.assertEqual(
                train_context.candidate_artifact.artifact_hash,
                jobs.train_spec.candidate_artifact.artifact_hash,
            )
            self.assertEqual(
                train_context.expected_initial_policy_state_hash,
                TEST_INITIAL_POLICY_STATE_HASH,
            )
            self.assertEqual(
                train_context.candidate_failure_contract_hash,
                STAIR_CANDIDATE_FAILURE_CONTRACT.contract_hash,
            )
            self.assertEqual(
                composition.task_plugin.task_spec.task_id,
                STAIR_TASK_ID,
            )

            checkpoint = Path(tmp) / "runner-checkpoint.pt"
            checkpoint.write_bytes(b"phase-b-checkpoint-fixture")
            checkpoint_hash = file_sha256(checkpoint)
            self.assertIsNotNone(checkpoint_hash)
            bound_eval_argv = list(eval_argv[2:])
            checkpoint_index = bound_eval_argv.index("--checkpoint")
            bound_eval_argv[checkpoint_index + 1] = str(checkpoint)
            bound_eval_argv[checkpoint_index + 2 : checkpoint_index + 2] = [
                "--checkpoint-sha256",
                checkpoint_hash,
            ]
            eval_args = build_eval_parser(_FakeAppLauncher).parse_args(
                bound_eval_argv
            )
            eval_context = validate_stair_eval_invocation(
                eval_args,
                project_root=ROOT,
            )
            self.assertEqual(
                eval_context.evaluation_binding,
                jobs.eval_spec.evaluation_binding,
            )
            self.assertEqual(eval_context.checkpoint_hash, checkpoint_hash)
            self.assertEqual(
                eval_context.seed_namespace,
                STAIR_FORMAL_EVAL_NAMESPACE,
            )

    def test_final_confirmation_eval_purpose_binds_the_disjoint_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(
                Path(tmp),
                "stair-final-confirmation",
                execution_device="cuda:0",
            )
            checkpoint = Path(tmp) / "final-confirmation-checkpoint.pt"
            checkpoint.write_bytes(b"final-confirmation-checkpoint-fixture")
            checkpoint_hash = file_sha256(checkpoint)
            self.assertIsNotNone(checkpoint_hash)
            argv = list(jobs.eval_spec.metadata["cli_argv"][2:])
            checkpoint_index = argv.index("--checkpoint")
            argv[checkpoint_index + 1] = str(checkpoint)
            argv[checkpoint_index + 2 : checkpoint_index + 2] = [
                "--checkpoint-sha256",
                checkpoint_hash,
            ]
            final_binding = EvaluationBindingV1(
                eval_protocol_hash=self.config.evaluation_protocol().protocol_hash,
                acceptance_protocol_hash=(
                    self.config.acceptance_protocol.protocol_hash
                ),
                task_protocol_hash=self.config.task_protocol_hash,
                seed_cohort_hash=(
                    self.config.seed_resolver().namespace_cohort_hash(
                        STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
                    )
                ),
            )
            argv[argv.index("--seed-cohort-hash") + 1] = (
                final_binding.seed_cohort_hash
            )
            argv[argv.index("--evaluation-binding-hash") + 1] = (
                final_binding.binding_hash
            )
            argv.extend(
                [
                    "--stair-eval-purpose",
                    STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION,
                ]
            )

            context = validate_stair_eval_invocation(
                build_eval_parser(_FakeAppLauncher).parse_args(argv),
                project_root=ROOT,
            )

            self.assertEqual(context.evaluation_binding, final_binding)
            self.assertEqual(
                context.seed_namespace,
                STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
            )

    def test_bound_materialization_uses_child_cuda_zero_and_phase_inherits(self) -> None:
        binding = GpuProcessBinding(
            host_execution_device="cuda:1",
            process_execution_device="cuda:0",
            resource_binding_hash="a" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            composition, jobs = self._materialize(
                Path(tmp),
                "stair-bound",
                execution_device="cuda:1",
                gpu_process_binding=binding,
            )

            for spec in (jobs.train_spec, jobs.eval_spec):
                self.assertEqual(spec.gpu_process_binding, binding)
                self.assertEqual(spec.execution_device, "cuda:0")
                argv = spec.metadata["cli_argv"]
                self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
                self.assertEqual(
                    argv[argv.index("--gpu-host-execution-device") + 1],
                    "cuda:1",
                )
                self.assertEqual(
                    argv[argv.index("--gpu-process-execution-device") + 1],
                    "cuda:0",
                )
                self.assertEqual(
                    argv[argv.index("--gpu-resource-binding-hash") + 1],
                    binding.resource_binding_hash,
                )
            validate_train_job_argv(
                jobs.train_spec,
                jobs.train_spec.metadata["cli_argv"],
            )
            validate_eval_job_argv(
                jobs.eval_spec,
                jobs.eval_spec.metadata["cli_argv"],
            )

            train_args = build_train_parser().parse_args(
                jobs.train_spec.metadata["cli_argv"][2:]
            )
            with patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": TEST_GPU_UUID},
                clear=True,
            ):
                runtime = validate_stair_gpu_process_binding_invocation(
                    train_args,
                    environment=os.environ,
                )
            self.assertIsNotNone(runtime)
            assert runtime is not None
            self.assertEqual(runtime.renderer_active_gpu_ordinal, 1)
            self.assertEqual(
                train_args.kit_args,
                "--/renderer/multiGpu/autoEnable=false "
                "--/renderer/multiGpu/maxGpuCount=1 "
                "--/renderer/activeGpu=1",
            )
            owned_child_identity = _run_process_group_leader_identity_child(
                Path(tmp)
            )
            self.assertEqual(
                owned_child_identity["schema_version"],
                OWNED_PROCESS_IDENTITY_SCHEMA_VERSION,
            )
            pid = owned_child_identity["process_id"]
            with patch(
                "true_rl_agent.tasks.stair_climbing.gpu_process_binding."
                "current_owned_process_identity",
                return_value=owned_child_identity,
            ):
                summary = attach_stair_gpu_runtime_evidence(
                    {"status": "pass"},
                    runtime,
                )
            self.assertEqual(summary["gpu_process_binding"], binding.to_dict())
            self.assertEqual(
                summary["stair_gpu_runtime_evidence"]["physical_gpu_uuid"],
                TEST_GPU_UUID,
            )
            completed = SimpleNamespace(
                stdout=f"{TEST_GPU_UUID}, {pid}, 512\n",
            )
            with patch(
                "true_rl_agent.tasks.stair_climbing.gpu_process_binding."
                "current_owned_process_identity",
                return_value=owned_child_identity,
            ), patch(
                "true_rl_agent.tasks.stair_climbing.gpu_process_binding."
                "subprocess.run",
                return_value=completed,
            ):
                observation = capture_stair_gpu_compute_context(runtime)
                observed_summary = attach_stair_gpu_runtime_evidence(
                    {"status": "pass"},
                    runtime,
                    compute_context_observation=observation,
                )
            self.assertEqual(
                observed_summary["stair_gpu_runtime_evidence"]
                ["external_compute_context_observation"]["rows"],
                [{"gpu_uuid": TEST_GPU_UUID, "used_gpu_memory_mib": 512}],
            )
            self.assertEqual(
                validate_stair_gpu_runtime_summary_evidence(
                    observed_summary,
                    expected_binding=binding,
                    expected_physical_gpu_uuid=TEST_GPU_UUID,
                    require_compute_context_observation=True,
                ),
                {
                    key: observed_summary[key]
                    for key in (
                        "gpu_process_binding",
                        "owned_process_identity",
                        "stair_gpu_runtime_evidence",
                    )
                },
            )
            with self.assertRaisesRegex(ValueError, "PID-to-UUID"):
                validate_stair_gpu_runtime_summary_evidence(
                    summary,
                    expected_binding=binding,
                    expected_physical_gpu_uuid=TEST_GPU_UUID,
                    require_compute_context_observation=True,
                )
            with patch(
                "true_rl_agent.tasks.stair_climbing.gpu_process_binding."
                "subprocess.run",
                return_value=SimpleNamespace(
                    stdout=(
                        f"{TEST_GPU_UUID}, {pid}, 512\n"
                        f"GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb, {pid}, 64\n"
                    )
                ),
            ):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    capture_stair_gpu_compute_context(runtime)
            nvidia_visible_args = build_train_parser().parse_args(
                jobs.train_spec.metadata["cli_argv"][2:]
            )
            with self.assertRaisesRegex(ValueError, "NVIDIA_VISIBLE_DEVICES"):
                validate_stair_gpu_process_binding_invocation(
                    nvidia_visible_args,
                    environment={
                        "CUDA_VISIBLE_DEVICES": TEST_GPU_UUID,
                        "NVIDIA_VISIBLE_DEVICES": "",
                    },
                )

            checkpoint = Path(tmp) / "bound.pt"
            checkpoint.write_bytes(b"bound-checkpoint")
            checkpoint_hash = file_sha256(checkpoint)
            self.assertIsNotNone(checkpoint_hash)
            bound_plan = replace(
                jobs.eval_spec,
                checkpoint=CheckpointRef(
                    uri="artifact://stair-bound/bound.pt",
                    checksum=checkpoint_hash,
                    role="eval",
                    sealed=True,
                    metadata={"local_path": str(checkpoint)},
                ),
                checkpoint_output_plan=None,
            )
            phase_spec = composition.backend_adapter.materialize_eval_phase(
                eval_plan=bound_plan,
                phase=EvaluationPhase.PUBLIC_DEV,
            )
            self.assertEqual(
                phase_spec.evaluation_plan.gpu_process_binding,
                binding,
            )
            phase_argv = phase_spec.metadata["cli_argv"]
            validate_eval_job_argv(phase_spec, phase_argv)
            self.assertEqual(
                phase_argv[phase_argv.index("--gpu-resource-binding-hash") + 1],
                binding.resource_binding_hash,
            )

    def test_bound_entrypoints_reject_visibility_drift_before_app_launch(self) -> None:
        binding = GpuProcessBinding(
            host_execution_device="cuda:1",
            process_execution_device="cuda:0",
            resource_binding_hash="b" * 64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(
                Path(tmp),
                "stair-binding-drift",
                execution_device="cuda:1",
                gpu_process_binding=binding,
            )
            fake_isaaclab = ModuleType("isaaclab")
            fake_app = ModuleType("isaaclab.app")
            fake_app.AppLauncher = _CountingFakeAppLauncher
            _CountingFakeAppLauncher.launch_count = 0
            checkpoint = Path(tmp) / "binding-drift.pt"
            checkpoint.write_bytes(b"binding-drift")
            checkpoint_hash = file_sha256(checkpoint)
            self.assertIsNotNone(checkpoint_hash)
            eval_argv = list(jobs.eval_spec.metadata["cli_argv"][2:])
            checkpoint_index = eval_argv.index("--checkpoint")
            eval_argv[checkpoint_index + 1] = str(checkpoint)
            eval_argv[checkpoint_index + 2 : checkpoint_index + 2] = [
                "--checkpoint-sha256",
                checkpoint_hash,
            ]
            with (
                patch.dict(
                    sys.modules,
                    {"isaaclab": fake_isaaclab, "isaaclab.app": fake_app},
                ),
                patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": "1"},
                    clear=True,
                ),
            ):
                self.assertEqual(
                    stair_train_main(jobs.train_spec.metadata["cli_argv"][2:]),
                    2,
                )
                self.assertEqual(
                    stair_eval_main(
                        eval_argv,
                        app_launcher_class=_CountingFakeAppLauncher,
                    ),
                    2,
                )
            self.assertEqual(_CountingFakeAppLauncher.launch_count, 0)

    def test_phase_c_train_preflight_qualifies_exact_artifact_without_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(Path(tmp), "stair-qualified")
            argv = jobs.train_spec.metadata["cli_argv"][2:]
            args = build_train_parser().parse_args(argv)
            context = validate_stair_train_invocation(args, project_root=ROOT)
            program = _load_and_qualify_program(context)

            self.assertEqual(
                program.program_source_checksum,
                jobs.train_spec.program_source_checksum,
            )
            self.assertEqual(
                program.subject_manifest_hash,
                jobs.train_spec.subject_manifest_hash,
            )
            self.assertGreater(len(program.term_names), 0)
            summary = _status_summary(
                "pass",
                args=args,
                context=context,
                source_manifest_preflight=SimpleNamespace(
                    to_summary_dict=lambda: {"status": "pass"}
                ),
                execution_environment_verification=(
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
                ),
                reward_program=program,
                attempt_id="test-attempt",
                formal_training_executed=True,
                execution_device="cuda:0",
            )
            SubprocessRunner._validate_train_binding_summary(
                summary,
                jobs.train_spec,
            )
            missing_authorization = dict(summary)
            missing_authorization.pop("candidate_execution_authorization_hash")
            with self.assertRaisesRegex(
                RunnerExecutionError,
                "candidate_execution_authorization_hash",
            ):
                SubprocessRunner._validate_train_binding_summary(
                    missing_authorization,
                    jobs.train_spec,
                )
            self.assertFalse(Path(jobs.train_spec.metadata["summary_path"]).exists())
            self.assertFalse(
                Path(jobs.train_spec.metadata["checkpoint_output_path"]).exists()
            )

    def test_qualification_failure_owner_depends_on_subject_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            _, candidate_jobs = self._materialize(
                output_dir,
                "stair-candidate-failure-owner",
            )
            candidate_context = validate_stair_train_invocation(
                build_train_parser().parse_args(
                    candidate_jobs.train_spec.metadata["cli_argv"][2:]
                ),
                project_root=ROOT,
            )
            with patch(
                "scripts.isaaclab_stair_rsl_rl_train."
                "qualify_stair_candidate_reward_program",
                side_effect=StairRewardProgramError("candidate source rejected"),
            ):
                with self.assertRaises(StairCandidateProgramFailure):
                    _load_and_qualify_program(candidate_context)

            official = resolve_official_stair_reference_program(
                self.config,
                project_root=ROOT,
            )
            provider = StairClimbingCompositionProvider(
                reference_program_artifact=official.artifact
            )
            composition = resolve_task_composition(
                provider=provider,
                values=provider.cli_defaults(),
                context=TaskCompositionContext(
                    project_root=ROOT,
                    output_dir=output_dir / "reference",
                ),
            )
            composition = self._with_test_execution_environment(composition)
            reference_jobs = composition.backend_adapter.materialize_reference_jobs(
                session_id="stair-reference-failure-owner",
                task_spec=composition.task_plugin.task_spec,
                reference=official.artifact,
                artifact_root=output_dir / "reference-artifacts",
                policy_initialization=composition.policy_initialization,
                evaluation_binding=(
                    composition.backend_adapter.evaluation_binding_for(
                        composition.task_plugin.task_spec
                    )
                ),
                backend_execution_environment_manifest_hash=(
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
                ),
            )
            reference_context = validate_stair_train_invocation(
                build_train_parser().parse_args(
                    reference_jobs.train_spec.metadata["cli_argv"][2:]
                ),
                project_root=ROOT,
            )
            with patch(
                "true_rl_agent.tasks.stair_climbing.reward_program."
                "qualify_stair_reference_reward_program",
                side_effect=StairRewardProgramError("reference source rejected"),
            ):
                with self.assertRaises(StairRewardProgramError):
                    _load_and_qualify_program(reference_context)

    def test_candidate_execution_authorization_fails_closed_on_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(Path(tmp), "stair-authorization-drift")
            original = list(jobs.train_spec.metadata["cli_argv"][2:])
            json_index = original.index(
                "--candidate-execution-authorization-json"
            ) + 1
            hash_index = original.index(
                "--candidate-execution-authorization-hash"
            ) + 1

            missing = list(original)
            option_index = missing.index("--candidate-execution-authorization-json")
            del missing[option_index : option_index + 2]
            with self.assertRaises(ValueError):
                validate_train_job_argv(jobs.train_spec, ["python", "train.py", *missing])

            stale_hash = list(original)
            envelope = json.loads(stale_hash[json_index])
            envelope["payload"]["qualification_hash"] = "f" * 64
            stale_hash[json_index] = canonical_json_text(envelope)
            with self.assertRaisesRegex(ValueError, "authorization hash mismatch"):
                validate_stair_train_invocation(
                    build_train_parser().parse_args(stale_hash),
                    project_root=ROOT,
                )

            policy_drift = list(original)
            envelope = json.loads(policy_drift[json_index])
            snapshot = envelope["payload"]["static_verifier_policy_snapshot"]
            snapshot["policy_facts"]["component_load_timeout_s"] += 1.0
            snapshot["executable_policy_hash"] = canonical_json_hash(
                snapshot["policy_facts"]
            )
            policy_drift[json_index] = canonical_json_text(envelope)
            policy_drift[hash_index] = canonical_json_hash(envelope)
            with self.assertRaisesRegex(ValueError, "admission identity mismatch"):
                validate_stair_train_invocation(
                    build_train_parser().parse_args(policy_drift),
                    project_root=ROOT,
                )

            qualification_drift = list(original)
            envelope = json.loads(qualification_drift[json_index])
            envelope["payload"]["qualification_hash"] = "f" * 64
            qualification_drift[json_index] = canonical_json_text(envelope)
            qualification_drift[hash_index] = canonical_json_hash(envelope)
            context = validate_stair_train_invocation(
                build_train_parser().parse_args(qualification_drift),
                project_root=ROOT,
            )
            with self.assertRaisesRegex(
                ValueError,
                "does not match execution authorization",
            ):
                _load_and_qualify_program(context)

    def test_initial_policy_state_binding_is_exact_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(Path(tmp), "stair-initial-state")
            original = list(jobs.train_spec.metadata["cli_argv"])
            option = "--expected-initial-policy-state-hash"
            index = original.index(option)

            missing = list(original)
            del missing[index : index + 2]
            with self.assertRaisesRegex(ValueError, "must appear exactly once"):
                validate_train_job_argv(jobs.train_spec, missing)

            duplicate = [*original, option, TEST_INITIAL_POLICY_STATE_HASH]
            with self.assertRaisesRegex(ValueError, "must appear exactly once"):
                validate_train_job_argv(jobs.train_spec, duplicate)

            replaced = list(original)
            replaced[index + 1] = "2" * 64
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_train_job_argv(jobs.train_spec, replaced)

            failure_option = "--candidate-failure-contract-hash"
            failure_index = original.index(failure_option)
            missing_failure_contract = list(original)
            del missing_failure_contract[failure_index : failure_index + 2]
            with self.assertRaisesRegex(ValueError, "must appear exactly once"):
                validate_train_job_argv(
                    jobs.train_spec,
                    missing_failure_contract,
                )

            duplicate_failure_contract = [
                *original,
                failure_option,
                STAIR_CANDIDATE_FAILURE_CONTRACT.contract_hash,
            ]
            with self.assertRaisesRegex(ValueError, "must appear exactly once"):
                validate_train_job_argv(
                    jobs.train_spec,
                    duplicate_failure_contract,
                )

    def test_candidate_qualification_failure_is_closed_before_app_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(Path(tmp), "stair-qualification-failure")
            argv = jobs.train_spec.metadata["cli_argv"][2:]
            summary_path = Path(jobs.train_spec.metadata["summary_path"])
            checkpoint_path = Path(
                jobs.train_spec.metadata["checkpoint_output_path"]
            )
            fake_isaaclab = ModuleType("isaaclab")
            fake_app = ModuleType("isaaclab.app")
            fake_app.AppLauncher = _CountingFakeAppLauncher
            _CountingFakeAppLauncher.launch_count = 0
            with (
                patch.dict(
                    sys.modules,
                    {"isaaclab": fake_isaaclab, "isaaclab.app": fake_app},
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_train."
                    "preflight_stair_execution_source_manifest",
                    return_value=SimpleNamespace(
                        to_summary_dict=lambda: {
                            "status": "pass",
                            "source_manifest_hash": "a" * 64,
                        }
                    ),
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_train."
                    "_load_and_qualify_program",
                    side_effect=StairCandidateProgramFailure.qualification(),
                ),
            ):
                self.assertEqual(stair_train_main(argv), 1)

            self.assertEqual(_CountingFakeAppLauncher.launch_count, 0)
            self.assertFalse(checkpoint_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["errors"], [])
            self.assertEqual(
                summary["candidate_failure"],
                STAIR_CANDIDATE_FAILURE_CONTRACT.evidence(
                    stage="qualification",
                    reason_code="reward_program_qualification_rejected",
                ).to_public_dict(),
            )
            self.assertEqual(
                summary["training"]["expected_initial_policy_state_hash"],
                TEST_INITIAL_POLICY_STATE_HASH,
            )
            self.assertNotIn(str(Path(tmp)), json.dumps(summary, sort_keys=True))

    def test_eval_summary_is_written_before_simulation_app_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "eval-summary.json"
            argv = [
                "--job-id",
                "stair-eval-close-order",
                "--round",
                "phase-b",
                "--task-id",
                STAIR_TASK_ID,
                "--subject-kind",
                "candidate",
                "--subject-id",
                "candidate-close-order",
                "--subject-manifest-hash",
                "1" * 64,
                "--program-source-checksum",
                "sha256:" + "2" * 64,
                "--training-binding-hash",
                "3" * 64,
                "--checkpoint",
                str(Path(tmp) / "checkpoint.pt"),
                "--checkpoint-sha256",
                "sha256:" + "4" * 64,
                "--eval-protocol-version",
                "test",
                "--task-protocol-hash",
                "5" * 64,
                "--acceptance-protocol-hash",
                "6" * 64,
                "--seed-cohort-hash",
                "7" * 64,
                "--evaluation-binding-hash",
                "8" * 64,
                "--capability-profile-hash",
                "9" * 64,
                "--summary-path",
                str(summary_path),
                "--profile",
                "test",
                "--runtime-dynamics-hash",
                "sha256:" + "a" * 64,
                "--backend-execution-environment-manifest-hash",
                TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
                "--config",
                "test",
                "--public-seeds",
                "1",
                "--public-accept-seeds",
                "1",
                "--sealed-validation-seeds",
                "1",
                "--num-envs",
                "1",
                "--seed",
                "1",
                "--headless",
            ]
            context = SimpleNamespace(
                config=object(),
                checkpoint_path=Path(tmp) / "checkpoint.pt",
                checkpoint_hash="sha256:" + "4" * 64,
            )
            checkpoint_report = SimpleNamespace(
                to_public_dict=lambda: {
                    "status": "pass",
                    "reason": "checkpoint_safe_to_load",
                }
            )
            with (
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval.validate_stair_eval_invocation",
                    return_value=context,
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval."
                    "preflight_stair_execution_source_manifest",
                    return_value=object(),
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval.build_checkpoint_safety_report",
                    return_value=checkpoint_report,
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval._run_eval",
                    return_value={"status": "pass", "public_score": 0.0},
                ),
            ):
                with self.assertRaisesRegex(SystemExit, "simulation closed"):
                    stair_eval_main(
                        argv,
                        app_launcher_class=_TerminatingFakeAppLauncher,
                    )
                with patch(
                    "scripts.isaaclab_stair_rsl_rl_eval.os._exit",
                    side_effect=SystemExit("subprocess exited"),
                ) as hard_exit:
                    with self.assertRaisesRegex(SystemExit, "subprocess exited"):
                        stair_eval_main(
                            argv,
                            app_launcher_class=_TerminatingFakeAppLauncher,
                            terminate_process=True,
                        )
                hard_exit.assert_called_once_with(0)

                with (
                    patch(
                        "scripts.isaaclab_stair_rsl_rl_eval.Path.write_text",
                        side_effect=OSError("summary publication failed"),
                    ),
                    patch(
                        "scripts.isaaclab_stair_rsl_rl_eval.os._exit",
                        side_effect=SystemExit("failed subprocess exited"),
                    ) as failed_hard_exit,
                ):
                    with self.assertRaisesRegex(
                        SystemExit,
                        "failed subprocess exited",
                    ):
                        stair_eval_main(
                            argv,
                            app_launcher_class=_TerminatingFakeAppLauncher,
                            terminate_process=True,
                        )
                failed_hard_exit.assert_called_once_with(1)

            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8"))["status"],
                "pass",
            )

    def test_source_drift_stops_formal_entrypoints_before_simulator_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(Path(tmp), "stair-source-drift")
            train_argv = jobs.train_spec.metadata["cli_argv"][2:]
            with (
                patch(
                    "scripts.isaaclab_stair_rsl_rl_train."
                    "preflight_stair_execution_source_manifest",
                    side_effect=ValueError("installed source bytes drifted"),
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_train."
                    "_load_and_qualify_program"
                ) as qualify,
            ):
                self.assertEqual(stair_train_main(train_argv), 2)
            qualify.assert_not_called()

            checkpoint = Path(tmp) / "source-drift-checkpoint.pt"
            checkpoint.write_bytes(b"source-drift-checkpoint")
            checkpoint_hash = file_sha256(checkpoint)
            self.assertIsNotNone(checkpoint_hash)
            eval_argv = list(jobs.eval_spec.metadata["cli_argv"][2:])
            checkpoint_index = eval_argv.index("--checkpoint")
            eval_argv[checkpoint_index + 1] = str(checkpoint)
            eval_argv[checkpoint_index + 2 : checkpoint_index + 2] = [
                "--checkpoint-sha256",
                checkpoint_hash,
            ]
            _CountingFakeAppLauncher.launch_count = 0
            with (
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval."
                    "preflight_stair_execution_source_manifest",
                    side_effect=ValueError("installed source bytes drifted"),
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval."
                    "build_checkpoint_safety_report"
                ) as checkpoint_preflight,
            ):
                self.assertEqual(
                    stair_eval_main(
                        eval_argv,
                        app_launcher_class=_CountingFakeAppLauncher,
                    ),
                    2,
                )
            self.assertEqual(_CountingFakeAppLauncher.launch_count, 0)
            checkpoint_preflight.assert_not_called()

    def test_execution_environment_drift_stops_before_simulator_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(Path(tmp), "stair-environment-drift")
            train_argv = jobs.train_spec.metadata["cli_argv"][2:]
            fake_isaaclab = ModuleType("isaaclab")
            fake_app = ModuleType("isaaclab.app")
            fake_app.AppLauncher = _CountingFakeAppLauncher
            _CountingFakeAppLauncher.launch_count = 0
            source_preflight = SimpleNamespace(
                to_summary_dict=lambda: {
                    "status": "pass",
                    "source_manifest_hash": "a" * 64,
                }
            )
            with (
                patch.dict(
                    sys.modules,
                    {"isaaclab": fake_isaaclab, "isaaclab.app": fake_app},
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_train."
                    "preflight_stair_execution_source_manifest",
                    return_value=source_preflight,
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_train."
                    "verify_current_isaac_lab_execution_environment",
                    side_effect=ValueError("backend execution environment drift"),
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_train."
                    "_load_and_qualify_program"
                ) as qualify,
            ):
                self.assertEqual(stair_train_main(train_argv), 2)
            self.assertEqual(_CountingFakeAppLauncher.launch_count, 0)
            qualify.assert_not_called()

            checkpoint = Path(tmp) / "environment-drift-checkpoint.pt"
            checkpoint.write_bytes(b"environment-drift-checkpoint")
            checkpoint_hash = file_sha256(checkpoint)
            self.assertIsNotNone(checkpoint_hash)
            eval_argv = list(jobs.eval_spec.metadata["cli_argv"][2:])
            checkpoint_index = eval_argv.index("--checkpoint")
            eval_argv[checkpoint_index + 1] = str(checkpoint)
            eval_argv[checkpoint_index + 2 : checkpoint_index + 2] = [
                "--checkpoint-sha256",
                checkpoint_hash,
            ]
            _CountingFakeAppLauncher.launch_count = 0
            with (
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval."
                    "preflight_stair_execution_source_manifest",
                    return_value=source_preflight,
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval."
                    "verify_current_isaac_lab_execution_environment",
                    side_effect=ValueError("backend execution environment drift"),
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval."
                    "build_checkpoint_safety_report"
                ) as checkpoint_preflight,
            ):
                self.assertEqual(
                    stair_eval_main(
                        eval_argv,
                        app_launcher_class=_CountingFakeAppLauncher,
                    ),
                    2,
                )
            self.assertEqual(_CountingFakeAppLauncher.launch_count, 0)
            checkpoint_preflight.assert_not_called()

    def test_reference_materialization_uses_exact_artifact_and_same_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            official = resolve_official_stair_reference_program(
                self.config,
                project_root=ROOT,
            )
            provider = StairClimbingCompositionProvider(
                reference_program_artifact=official.artifact
            )
            composition = resolve_task_composition(
                provider=provider,
                values=provider.cli_defaults(),
                context=TaskCompositionContext(
                    project_root=ROOT,
                    output_dir=output_dir,
                ),
            )
            composition = self._with_test_execution_environment(composition)
            binding = composition.backend_adapter.evaluation_binding_for(
                composition.task_plugin.task_spec
            )
            jobs = composition.backend_adapter.materialize_reference_jobs(
                session_id="stair-reference-materialized",
                task_spec=composition.task_plugin.task_spec,
                reference=official.artifact,
                artifact_root=output_dir / "reference_artifacts",
                policy_initialization=composition.policy_initialization,
                evaluation_binding=binding,
                backend_execution_environment_manifest_hash=(
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
                ),
            )

            self.assertEqual(jobs.train_spec.subject.kind, "reference")
            self.assertIsNone(jobs.train_spec.candidate_artifact)
            self.assertIsNotNone(jobs.train_spec.reference_artifact)
            train_argv = jobs.train_spec.metadata["cli_argv"]
            validate_train_job_argv(jobs.train_spec, train_argv)
            self.assertIn("--reference-artifact", train_argv)
            self.assertNotIn("--candidate", train_argv)
            self.assertNotIn("--expected-initial-policy-state-hash", train_argv)
            self.assertNotIn("--candidate-failure-contract-hash", train_argv)
            self.assertIsNone(jobs.train_spec.candidate_failure_contract)
            eval_argv = jobs.eval_spec.metadata["cli_argv"]
            self.assertNotIn("--reference-artifact", eval_argv)
            self.assertNotIn(jobs.train_spec.reference_artifact.path, eval_argv)

            args = build_train_parser().parse_args(train_argv[2:])
            context = validate_stair_train_invocation(args, project_root=ROOT)
            program = _load_and_qualify_program(context)
            self.assertEqual(program.subject.kind, "reference")
            self.assertEqual(
                program.program_source_checksum,
                official.artifact.manifest.source_checksum,
            )

            duplicate_argv = [
                *train_argv,
                "--reference-artifact",
                jobs.train_spec.reference_artifact.path,
            ]
            with self.assertRaises(ValueError):
                validate_train_job_argv(jobs.train_spec, duplicate_argv)

            cross_kind_argv = [
                *train_argv,
                "--candidate",
                jobs.train_spec.reference_artifact.path,
                "--candidate-artifact-hash",
                jobs.train_spec.reference_artifact.artifact_hash,
            ]
            with self.assertRaises(ValueError):
                validate_train_job_argv(jobs.train_spec, cross_kind_argv)

            gpu_binding = GpuProcessBinding(
                host_execution_device="cuda:1",
                process_execution_device="cuda:0",
                resource_binding_hash="c" * 64,
            )
            bound_reference_jobs = (
                composition.backend_adapter.materialize_reference_jobs(
                    session_id="stair-reference-bound",
                    task_spec=composition.task_plugin.task_spec,
                    reference=official.artifact,
                    artifact_root=output_dir / "bound_reference_artifacts",
                    policy_initialization=composition.policy_initialization,
                    evaluation_binding=binding,
                    backend_execution_environment_manifest_hash=(
                        TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
                    ),
                    gpu_process_binding=gpu_binding,
                )
            )
            for spec in (
                bound_reference_jobs.train_spec,
                bound_reference_jobs.eval_spec,
            ):
                self.assertEqual(spec.gpu_process_binding, gpu_binding)
                self.assertEqual(spec.execution_device, "cuda:0")
                self.assertEqual(
                    spec.metadata["cli_argv"][
                        spec.metadata["cli_argv"].index(
                            "--gpu-resource-binding-hash"
                        )
                        + 1
                    ],
                    gpu_binding.resource_binding_hash,
                )

            source_path = (
                Path(jobs.train_spec.reference_artifact.path)
                / "training_code"
                / "components.py"
            )
            source_path.write_bytes(source_path.read_bytes() + b"\n# drift\n")
            drifted_args = build_train_parser().parse_args(train_argv[2:])
            with self.assertRaisesRegex(ValueError, "file evidence mismatch"):
                validate_stair_train_invocation(drifted_args, project_root=ROOT)

    def test_preflight_rejects_binding_and_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(Path(tmp), "stair-drift")
            train_argv = list(jobs.train_spec.metadata["cli_argv"][2:])
            index = train_argv.index("--training-binding-hash")
            train_argv[index + 1] = "0" * 64
            train_args = build_train_parser().parse_args(train_argv)
            with self.assertRaisesRegex(ValueError, "training binding hash mismatch"):
                validate_stair_train_invocation(train_args, project_root=ROOT)

            train_argv = list(jobs.train_spec.metadata["cli_argv"][2:])
            index = train_argv.index("--config")
            train_argv[index + 1] = "configs/tasks/stair_protocol_v1.json"
            train_args = build_train_parser().parse_args(train_argv)
            with self.assertRaisesRegex(ValueError, "config path is not canonical"):
                validate_stair_train_invocation(train_args, project_root=ROOT)

    def _materialize(
        self,
        output_dir: Path,
        candidate_id: str,
        *,
        execution_device: str = "cuda:0",
        gpu_process_binding: GpuProcessBinding | None = None,
    ):
        provider = StairClimbingCompositionProvider(
            reference_program_artifact=self.reference
        )
        values = provider.cli_defaults()
        values["execution_device"] = execution_device
        composition = resolve_task_composition(
            provider=provider,
            values=values,
            context=TaskCompositionContext(
                project_root=ROOT,
                output_dir=output_dir,
            ),
        )
        composition = self._with_test_execution_environment(composition)
        binding = composition.backend_adapter.evaluation_binding_for(
            composition.task_plugin.task_spec
        )
        candidate = make_test_candidate_package(candidate_id)
        adapter = StairClimbingComponentSmokeAdapter(composition.task_plugin.config)

        def qualification_worker(*, artifact, static_verifier_policy, **_kwargs):
            program = qualify_stair_candidate_reward_program(
                artifact=artifact,
                contract=adapter.contract,
                static_verifier_policy=static_verifier_policy,
                trusted_reference_artifact=self.reference,
            )
            return {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing.candidate_qualification.v1"
                ),
                "status": "pass",
                "reason": "stair_candidate_reward_program_qualified",
                "qualification": program.to_public_dict(),
                "determinism": {"status": "test_fixture"},
                "environment_axis": {
                    "schema_version": (
                        STAIR_REWARD_ENVIRONMENT_AXIS_EVIDENCE_SCHEMA_VERSION
                    ),
                    "status": "pass",
                    "reverse_permutation_checks": len(program.term_names),
                    "single_row_locality_checks": len(program.term_names),
                    "term_names": list(program.term_names),
                },
                "smoke_fixture_hash": adapter.fixture_hash,
            }

        with patch(
            "true_rl_agent.tasks.stair_climbing.candidate_admission."
            "_run_qualification_worker",
            side_effect=qualification_worker,
        ):
            artifact = seal_and_verify_candidate(
                candidate,
                artifact_root=output_dir / "candidate-artifacts",
                verifier=composition.candidate_admission_policy,
            ).require_verified()
        jobs = composition.backend_adapter.materialize_jobs(
            round_id=candidate_id,
            task_spec=composition.task_plugin.task_spec,
            candidate=artifact,
            policy_initialization=composition.policy_initialization,
            evaluation_binding=binding,
            expected_initial_policy_state_hash=TEST_INITIAL_POLICY_STATE_HASH,
            backend_execution_environment_manifest_hash=(
                TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
            ),
            gpu_process_binding=gpu_process_binding,
        )
        return composition, jobs

    @staticmethod
    def _with_test_execution_environment(composition):
        adapter = composition.backend_adapter
        if not isinstance(adapter, IsaacLabRslRlBackendAdapter):
            raise TypeError("test composition requires the Isaac Lab backend")
        return replace(
            composition,
            backend_adapter=IsaacLabRslRlBackendAdapter(
                adapter.config,
                execution_environment_resolver=(
                    make_test_isaac_lab_execution_environment_manifest
                ),
            ),
        )


class _FakeAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser) -> None:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cpu")


class _TerminatingFakeAppLauncher(_FakeAppLauncher):
    def __init__(self, args, *, multi_gpu=None) -> None:
        _ = args, multi_gpu
        self.app = self

    def close(self) -> None:
        raise SystemExit("simulation closed")


class _CountingFakeAppLauncher(_FakeAppLauncher):
    launch_count = 0

    def __init__(self, args, *, multi_gpu=None) -> None:
        _ = args, multi_gpu
        type(self).launch_count += 1
        self.app = self

if __name__ == "__main__":
    unittest.main()
