from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.backend_environment import (
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
    make_test_isaac_lab_execution_environment_manifest,
)
from tests.fakes.evaluation import TEST_CAPABILITY_PROFILE, make_evaluation_binding
from true_rl_agent.backends.isaac_lab_rsl_rl import (
    IsaacLabRslRlBackendAdapter,
    IsaacLabRslRlBackendConfig,
)
from true_rl_agent.backends.rsl_rl_training_config import (
    RslRlAgentConfiguration,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import (
    StaticCandidateVerifier,
    seal_and_verify_candidate,
)
from true_rl_agent.evaluation.acceptance_protocol import (
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.policy import BaselinePolicyRef, PolicyInitialization
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.runner.job_argv import validate_eval_job_argv
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskSpec

TEST_ACCEPTANCE_PROTOCOL = AcceptanceProtocol(
    protocol_id="isaac_backend_test_acceptance_v1",
    score_source="public_accept.score_stats.mean",
    min_score_count=3,
    absolute_accept_floor=0.05,
    min_delta=0.05,
    public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
)


class IsaacLabBackendTests(unittest.TestCase):
    def test_materialize_eval_phase_reuses_the_complete_bound_plan(self) -> None:
        jobs = self.adapter.materialize_jobs(
            round_id="round-phased-eval",
            task_spec=self.task,
            candidate=self.candidate,
            policy_initialization=PolicyInitialization.random(),
        )
        bound_plan = replace(
            jobs.eval_spec,
            checkpoint=CheckpointRef(
                uri="artifact://round-phased-eval/best.pt",
                checksum="sha256:" + "8" * 64,
                role="eval",
                sealed=True,
            ),
            checkpoint_output_plan=None,
        )

        phase_spec = self.adapter.materialize_eval_phase(
            eval_plan=bound_plan,
            phase=EvaluationPhase.PUBLIC_DEV,
        )
        argv = phase_spec.metadata["cli_argv"]

        validate_eval_job_argv(phase_spec, argv)
        self.assertIs(phase_spec.evaluation_plan, bound_plan)
        self.assertEqual(phase_spec.phase, EvaluationPhase.PUBLIC_DEV)
        self.assertEqual(
            argv[argv.index("--evaluation-phase") + 1],
            "public_dev",
        )
        self.assertEqual(
            argv[argv.index("--evaluation-plan-hash") + 1],
            phase_spec.evaluation_plan_hash,
        )
        self.assertNotIn("--evaluation-phase", bound_plan.metadata["cli_argv"])

    def test_materialized_eval_argv_uses_validated_session_binding(self) -> None:
        fixed = self.adapter.evaluation_binding_for(self.task)
        session_binding = EvaluationBindingV1(
            eval_protocol_hash=fixed.eval_protocol_hash,
            acceptance_protocol_hash=fixed.acceptance_protocol_hash,
            task_protocol_hash=fixed.task_protocol_hash,
            seed_cohort_hash="9" * 64,
        )

        class SessionBoundAdapter(IsaacLabRslRlBackendAdapter):
            def validate_session_evaluation_binding(
                self,
                *,
                task_spec,
                evaluation_binding,
            ) -> None:
                _ = task_spec
                if evaluation_binding != session_binding:
                    raise ValueError("session binding mismatch")

        adapter = SessionBoundAdapter(
            self.adapter.config,
            execution_environment_resolver=(
                make_test_isaac_lab_execution_environment_manifest
            ),
        )
        jobs = adapter.materialize_jobs(
            round_id="round-session-binding",
            task_spec=self.task,
            candidate=self.candidate,
            policy_initialization=PolicyInitialization.random(),
            evaluation_binding=session_binding,
        )
        argv = jobs.eval_spec.metadata["cli_argv"]

        self.assertEqual(jobs.eval_spec.evaluation_binding, session_binding)
        self.assertEqual(
            argv[argv.index("--seed-cohort-hash") + 1],
            session_binding.seed_cohort_hash,
        )
        self.assertEqual(
            argv[argv.index("--evaluation-binding-hash") + 1],
            session_binding.binding_hash,
        )

        with self.assertRaisesRegex(ValueError, "session evaluation binding mismatch"):
            self.adapter.materialize_jobs(
                round_id="round-session-binding-rejected",
                task_spec=self.task,
                candidate=self.candidate,
                policy_initialization=PolicyInitialization.random(),
                evaluation_binding=session_binding,
            )

    def _protocol_invariants(
        self,
        *,
        public_seed_count: int = 4,
        public_accept_seed_count: int = 3,
        sealed_validation_seed_count: int = 8,
        eval_num_envs: int = 64,
    ) -> dict:
        return EvaluationProtocol(
            eval_protocol_version="fixed_public_accept_sealed_validation_v1",
            public_seed_count=public_seed_count,
            public_accept_seed_count=public_accept_seed_count,
            sealed_validation_seed_count=sealed_validation_seed_count,
            eval_num_envs=eval_num_envs,
        ).to_public_dict()

    def setUp(self) -> None:
        self._candidate_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._candidate_tmp.cleanup)
        candidate_root = Path(self._candidate_tmp.name) / "candidate-001"
        (candidate_root / "training_code").mkdir(parents=True)
        (candidate_root / "candidate_manifest.yaml").write_text(
            "schema_version: true_rl_agent.candidate.v1\n"
            "candidate_id: candidate-001\n"
            "entrypoint: training_code.components:build_training_components\n",
            encoding="utf-8",
        )
        (candidate_root / "training_code" / "components.py").write_text(
            "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
            "def reward(inputs):\n"
            "    return inputs.base_height\n\n"
            "def build_training_components():\n"
            "    return TrainingComponents(\n"
            "        name='candidate-001',\n"
            "        reward_terms=[RewardTerm(name='height', weight=1.0, fn=reward)],\n"
            "    )\n",
            encoding="utf-8",
        )
        (candidate_root / "rationale.md").write_text(
            "Backend candidate.\n", encoding="utf-8"
        )
        (candidate_root / "operation_diff.json").write_text("{}\n", encoding="utf-8")
        (candidate_root / "diagnostics.json").write_text("{}\n", encoding="utf-8")
        self.train_runtime_hash = f"sha256:{'a' * 64}"
        self.eval_runtime_hash = f"sha256:{'b' * 64}"
        self.agent_configuration = RslRlAgentConfiguration(
            policy={"class_name": "ActorCritic", "hidden_dims": [64, 64]},
            algorithm={"class_name": "PPO", "learning_rate": 0.001},
        )
        self.task = self._task()
        self.adapter = IsaacLabRslRlBackendAdapter(
            IsaacLabRslRlBackendConfig(**self._config_kwargs(self.task)),
            execution_environment_resolver=(
                make_test_isaac_lab_execution_environment_manifest
            ),
        )
        provider_candidate = CandidatePackageRef(
            candidate_id="candidate-001",
            path=str(candidate_root),
            change_set=CandidateChangeSet(
                changed_components=frozenset({"reward_callable"})
            ),
        )
        self.candidate = seal_and_verify_candidate(
            provider_candidate,
            artifact_root=Path(self._candidate_tmp.name) / "artifacts",
            verifier=StaticCandidateVerifier(),
        ).require_verified()

    def _task(
        self,
        *,
        extra_invariants: dict | None = None,
    ) -> TaskSpec:
        return TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            genesis_anchor_policy={
                "random": "bootstrap",
                "warm_start": "policy_baseline",
            },
            candidate_capability_profile=TEST_CAPABILITY_PROFILE,
            invariants=TaskInvariantSpec(
                {
                    "fall_threshold": 0.30,
                    "seed_cohort_hash": "c" * 64,
                    **self._protocol_invariants(),
                    **dict(extra_invariants or {}),
                }
            ),
            fixed_eval_protocol={
                "eval_protocol": "fixed_public_accept_and_sealed_validation",
                "seed_cohort_hash": "c" * 64,
                **self._protocol_invariants(),
            },
            fixed_train_protocol={
                "policy_architecture_hash": (
                    self.agent_configuration.policy_architecture_hash
                ),
                "optimizer_training_profile_hash": (
                    self.agent_configuration.optimizer_training_profile_hash
                ),
                "agent_configuration_hash": (
                    self.agent_configuration.agent_configuration_hash
                ),
            },
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
        )

    def _config_kwargs(
        self,
        task: TaskSpec,
        **overrides,
    ) -> dict:
        admission = StaticCandidateVerifier().admission_policy_descriptor()
        values = {
            "isaac_lab_python": "python",
            "train_entrypoint": "scripts/isaaclab_rsl_rl_train.py",
            "eval_entrypoint": "scripts/isaaclab_rsl_rl_eval.py",
            "supported_tasks": frozenset({"unitree_push_recovery"}),
            "default_budget_steps": 2048,
            "public_seed_count": 4,
            "public_accept_seed_count": 3,
            "sealed_validation_seed_count": 8,
            "eval_num_envs": 64,
            "eval_protocol_version": "fixed_public_accept_sealed_validation_v1",
            "evaluation_binding": make_evaluation_binding(
                eval_protocol_version="fixed_public_accept_sealed_validation_v1",
                public_seed_count=4,
                public_accept_seed_count=3,
                sealed_validation_seed_count=8,
                eval_num_envs=64,
                acceptance_protocol_hash=TEST_ACCEPTANCE_PROTOCOL.protocol_hash,
                task_protocol_hash=task.resolved_task_protocol_hash(),
            ),
            "robot": "unitree_g1",
            "observation_schema_hash": "obs-test-v1",
            "action_schema_hash": "act-test-v1",
            "policy_architecture": "rsl_rl_ppo_mlp",
            "train_profile": "test_train",
            "eval_profile": "test_eval",
            "train_num_envs": 64,
            "training_agent_configuration": self.agent_configuration,
            "eval_base_seed": 42,
            "execution_device": "cuda:0",
            "candidate_admission_policy_id": admission.policy_id,
            "candidate_admission_policy_hash": admission.policy_hash,
            "candidate_execution_capability_ids": frozenset(),
            "runtime_config_path": "configs/tasks/push_recovery.yaml",
            "train_runtime_dynamics_hash": self.train_runtime_hash,
            "eval_runtime_dynamics_hash": self.eval_runtime_hash,
            "curriculum_capabilities": {
                "external_push_curriculum": {"effective_now": False}
            },
        }
        values.update(overrides)
        return values

    def test_backend_declares_external_dependencies_and_future_capabilities(
        self,
    ) -> None:
        dependencies = self.adapter.external_dependencies
        capability = self.adapter.future_curriculum_capabilities[
            "external_push_curriculum"
        ]

        self.assertIn("Isaac Lab", dependencies)
        self.assertIn("RSL-RL", dependencies)
        self.assertEqual(self.adapter.backend_spec.backend_id, "isaac_lab_rsl_rl")
        self.assertFalse(capability["effective_now"])

    def test_backend_config_accepts_trusted_seed_count_only_as_legacy_constructor_alias(
        self,
    ) -> None:
        kwargs = self._config_kwargs(self.task)
        del kwargs["sealed_validation_seed_count"]
        config = IsaacLabRslRlBackendConfig(**kwargs, trusted_seed_count=8)

        self.assertEqual(config.sealed_validation_seed_count, 8)
        self.assertEqual(config.evaluation_protocol().sealed_validation_seed_count, 8)

        with self.assertRaisesRegex(ValueError, "legacy alias does not match"):
            IsaacLabRslRlBackendConfig(
                **self._config_kwargs(self.task),
                trusted_seed_count=9,
            )

        for invalid in ("8", 8.0, True):
            with self.subTest(invalid=invalid):
                kwargs = self._config_kwargs(self.task)
                del kwargs["sealed_validation_seed_count"]
                with self.assertRaisesRegex(
                    ValueError,
                    "trusted_seed_count must be a positive integer",
                ):
                    IsaacLabRslRlBackendConfig(
                        **kwargs,
                        trusted_seed_count=invalid,
                    )

    def test_backend_config_rejects_invalid_generic_runtime_fields(self) -> None:
        for field_name, value in (
            ("default_budget_steps", 0),
            ("public_seed_count", 0),
            ("sealed_validation_seed_count", 0),
            ("train_num_envs", True),
            ("eval_num_envs", -1),
            ("train_profile", ""),
            ("eval_entrypoint", " "),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    IsaacLabRslRlBackendConfig(
                        **self._config_kwargs(
                            self.task,
                            **{field_name: value},
                        )
                    )

    def test_backend_config_rejects_complete_inconsistent_budget_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "planned env steps mismatch"):
            IsaacLabRslRlBackendConfig(
                **self._config_kwargs(
                    self.task,
                    default_budget_steps=65,
                    train_num_envs=8,
                    train_max_iterations=1,
                    train_steps_per_iteration=8,
                )
            )

    def test_production_manifest_is_sampled_only_by_configured_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            other_python = Path(tmp) / "python"
            other_python.write_bytes(b"not the controller interpreter")
            adapter = IsaacLabRslRlBackendAdapter(
                IsaacLabRslRlBackendConfig(
                    **self._config_kwargs(
                        self.task,
                        isaac_lab_python=str(other_python),
                    )
                )
            )

            with self.assertRaisesRegex(
                ValueError,
                "must be sampled by config.isaac_lab_python",
            ):
                adapter.resolve_session_execution_environment(
                    session_id="python-mismatch",
                    session_dir=Path(tmp),
                    allow_create=True,
                )

    def test_production_manifest_accepts_the_controller_interpreter(self) -> None:
        with patch(
            "true_rl_agent.backends.isaac_lab_rsl_rl."
            "build_current_isaac_lab_execution_environment_manifest",
            return_value=make_test_isaac_lab_execution_environment_manifest(),
        ) as resolver:
            adapter = IsaacLabRslRlBackendAdapter(
                IsaacLabRslRlBackendConfig(
                    **self._config_kwargs(
                        self.task,
                        isaac_lab_python=sys.executable,
                    )
                )
            )
            manifest = adapter.resolve_session_execution_environment(
                session_id="same-python",
                session_dir=PROJECT_ROOT,
                allow_create=True,
            )

        self.assertEqual(
            manifest["manifest_hash"],
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
        )
        resolver.assert_called_once_with()

    def test_materialization_without_genesis_binding_does_not_sample_environment(
        self,
    ) -> None:
        with patch(
            "true_rl_agent.backends.isaac_lab_rsl_rl."
            "build_current_isaac_lab_execution_environment_manifest",
            side_effect=AssertionError("materialization must not sample Isaac"),
        ) as resolver:
            adapter = IsaacLabRslRlBackendAdapter(self.adapter.config)
            bundle = adapter.materialize_jobs(
                round_id="round-unbound-environment",
                task_spec=self.task,
                candidate=self.candidate,
                policy_initialization=PolicyInitialization.random(),
            )

        resolver.assert_not_called()
        for spec in (bundle.train_spec, bundle.eval_spec):
            self.assertNotIn(
                "backend_execution_environment_manifest_hash",
                spec.metadata,
            )
            self.assertNotIn(
                "--backend-execution-environment-manifest-hash",
                spec.metadata["cli_argv"],
            )

    def test_formal_runtime_requires_genesis_environment_binding(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "formal Isaac Lab execution requires a Genesis-bound",
        ):
            self.adapter.validate_runtime_session_execution_environment(
                execution_environment=None
            )

        self.adapter.validate_runtime_session_execution_environment(
            execution_environment=(
                make_test_isaac_lab_execution_environment_manifest().to_dict()
            )
        )

        drifted = make_test_isaac_lab_execution_environment_manifest().to_dict()
        drifted["manifest_hash"] = "0" * 64
        with self.assertRaises(ValueError):
            self.adapter.validate_runtime_session_execution_environment(
                execution_environment=drifted
            )

    def test_materializes_random_train_and_checkpoint_eval_cli_specs(self) -> None:
        bundle = self.adapter.materialize_jobs(
            round_id="round-001",
            task_spec=self.task,
            candidate=self.candidate,
            policy_initialization=PolicyInitialization.random(),
            backend_execution_environment_manifest_hash=(
                TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
            ),
        )

        self.assertEqual(bundle.train_spec.policy_initialization.kind, "random")
        self.assertIsNone(bundle.train_spec.training_start_checkpoint())
        self.assertEqual(
            bundle.train_spec.metadata["cli_argv"][:2],
            ["python", "scripts/isaaclab_rsl_rl_train.py"],
        )
        self.assertIn("--init random", bundle.train_spec.metadata["cli_command"])
        self.assertIn(
            f"--candidate {self.candidate.path}",
            bundle.train_spec.metadata["cli_command"],
        )
        train_argv = bundle.train_spec.metadata["cli_argv"]
        self.assertEqual(
            train_argv[train_argv.index("--config") + 1],
            "configs/tasks/push_recovery.yaml",
        )
        self.assertEqual(
            train_argv[train_argv.index("--runtime-dynamics-hash") + 1],
            self.train_runtime_hash,
        )
        self.assertEqual(
            train_argv[
                train_argv.index("--backend-execution-environment-manifest-hash") + 1
            ],
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
        )
        self.assertEqual(
            bundle.train_spec.metadata["backend_execution_environment_manifest_hash"],
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
        )
        for flag, expected in (
            ("--job-id", bundle.train_spec.job_id),
            ("--task-id", bundle.train_spec.task_id),
            ("--subject-kind", bundle.train_spec.subject.kind),
            ("--subject-id", bundle.train_spec.subject.subject_id),
            (
                "--subject-manifest-hash",
                bundle.train_spec.subject_manifest_hash,
            ),
            (
                "--program-source-checksum",
                bundle.train_spec.program_source_checksum,
            ),
            (
                "--training-binding-hash",
                bundle.train_spec.training_binding.binding_hash,
            ),
            (
                "--policy-architecture-hash",
                bundle.train_spec.training_binding.policy_architecture_hash,
            ),
            (
                "--optimizer-training-profile-hash",
                bundle.train_spec.training_binding.optimizer_training_profile_hash,
            ),
            (
                "--agent-configuration-hash",
                bundle.train_spec.training_binding.agent_configuration_hash,
            ),
            (
                "--initialization-seed-schedule-hash",
                bundle.train_spec.training_binding.initialization_seed_schedule_hash,
            ),
            (
                "--agent-configuration-json",
                self.agent_configuration.to_canonical_json(),
            ),
        ):
            self.assertIn(flag, train_argv)
            self.assertEqual(train_argv[train_argv.index(flag) + 1], expected)
        self.assertTrue(
            bundle.train_spec.metadata["summary_path"].endswith(
                "round-001/train/summary.json"
            )
        )
        self.assertIn(
            bundle.train_spec.metadata["summary_path"],
            bundle.train_spec.metadata["cli_argv"],
        )
        self.assertEqual(
            bundle.train_spec.metadata["sealed_checkpoint_uri"],
            "artifact://round-001/checkpoints/best.pt",
        )
        self.assertTrue(
            bundle.train_spec.metadata["checkpoint_output_path"].endswith(
                "round-001/checkpoints/best.pt"
            )
        )
        self.assertIn(
            bundle.train_spec.metadata["checkpoint_output_path"],
            bundle.train_spec.metadata["cli_argv"],
        )
        self.assertIn("--checkpoint", bundle.eval_spec.metadata["cli_command"])
        self.assertEqual(
            bundle.eval_spec.metadata["cli_argv"][:2],
            ["python", "scripts/isaaclab_rsl_rl_eval.py"],
        )
        eval_argv = bundle.eval_spec.metadata["cli_argv"]
        self.assertEqual(
            eval_argv[eval_argv.index("--config") + 1],
            "configs/tasks/push_recovery.yaml",
        )
        self.assertEqual(
            eval_argv[eval_argv.index("--runtime-dynamics-hash") + 1],
            self.eval_runtime_hash,
        )
        self.assertEqual(
            eval_argv[
                eval_argv.index("--backend-execution-environment-manifest-hash") + 1
            ],
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
        )
        self.assertEqual(
            bundle.eval_spec.metadata["backend_execution_environment_manifest_hash"],
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
        )
        self.assertEqual(
            eval_argv[eval_argv.index("--subject-kind") + 1],
            "candidate",
        )
        self.assertEqual(
            eval_argv[eval_argv.index("--subject-id") + 1],
            "candidate-001",
        )
        self.assertNotIn("--candidate-id", eval_argv)
        self.assertNotIn("--candidate", eval_argv)
        self.assertNotIn("artifact://candidate-001", eval_argv)
        self.assertNotIn(
            "--candidate artifact://candidate-001",
            bundle.eval_spec.metadata["cli_command"],
        )
        self.assertEqual(
            bundle.eval_spec.metadata["evaluation_subject"],
            {
                "schema_version": "true_rl_agent.experiment.subject.v2",
                "kind": "candidate",
                "id": "candidate-001",
            },
        )
        self.assertIn("--public-accept-seeds", bundle.eval_spec.metadata["cli_argv"])
        self.assertEqual(
            bundle.eval_spec.metadata["cli_argv"][
                bundle.eval_spec.metadata["cli_argv"].index("--public-accept-seeds") + 1
            ],
            "3",
        )
        self.assertNotIn("--checkpoint-sha256", eval_argv)
        self.assertEqual(eval_argv.count("--program-source-checksum"), 1)
        self.assertEqual(
            eval_argv[eval_argv.index("--program-source-checksum") + 1],
            self.candidate.program_source_checksum,
        )
        self.assertEqual(
            bundle.eval_spec.metadata["cli_command"],
            " ".join(eval_argv),
        )
        self.assertIsNone(bundle.eval_spec.checkpoint)
        self.assertEqual(
            bundle.eval_spec.checkpoint_output_plan.uri,
            "artifact://round-001/checkpoints/best.pt",
        )
        self.assertEqual(bundle.eval_spec.public_accept_seed_count, 3)
        self.assertEqual(bundle.eval_spec.eval_num_envs, 64)
        self.assertTrue(
            bundle.eval_spec.metadata["summary_path"].endswith(
                "round-001/eval/summary.json"
            )
        )
        self.assertIn(
            bundle.eval_spec.metadata["summary_path"],
            bundle.eval_spec.metadata["cli_argv"],
        )
        self.assertEqual(bundle.task_invariants, self.task.invariants)

    def test_real_backend_rejects_unverified_provider_candidate(self) -> None:
        provider_candidate = CandidatePackageRef(
            candidate_id=self.candidate.candidate_id,
            path=str(Path(self._candidate_tmp.name) / "candidate-001"),
            change_set=self.candidate.change_set,
        )

        with self.assertRaisesRegex(TypeError, "verified artifact"):
            self.adapter.materialize_jobs(
                round_id="round-unsealed",
                task_spec=self.task,
                candidate=provider_candidate,
                policy_initialization=PolicyInitialization.random(),
            )

    def test_materialization_revalidates_verified_candidate_bytes(self) -> None:
        manifest_path = Path(self.candidate.path) / "candidate_manifest.yaml"
        manifest_path.write_text(
            "schema_version: true_rl_agent.candidate.v999\n"
            "candidate_id: candidate-001\n"
            "entrypoint: training_code.components:build_training_components\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "file evidence mismatch"):
            self.adapter.materialize_jobs(
                round_id="round-invalid-schema",
                task_spec=self.task,
                candidate=self.candidate,
                policy_initialization=PolicyInitialization.random(),
            )

    def test_task_and_backend_agent_configuration_cannot_drift(self) -> None:
        drifted = RslRlAgentConfiguration(
            policy={"class_name": "ActorCritic", "hidden_dims": [64, 64]},
            algorithm={"class_name": "PPO", "learning_rate": 0.5},
        )
        adapter = IsaacLabRslRlBackendAdapter(
            IsaacLabRslRlBackendConfig(
                **self._config_kwargs(
                    self.task,
                    training_agent_configuration=drifted,
                )
            ),
            execution_environment_resolver=(
                make_test_isaac_lab_execution_environment_manifest
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "task_training_configuration_mismatch",
        ):
            adapter.materialize_jobs(
                round_id="round-agent-config-drift",
                task_spec=self.task,
                candidate=self.candidate,
                policy_initialization=PolicyInitialization.random(),
            )

    def test_materialization_requires_runtime_dynamics_binding(self) -> None:
        adapter = IsaacLabRslRlBackendAdapter(
            replace(
                self.adapter.config,
                train_runtime_dynamics_hash=None,
                eval_runtime_dynamics_hash=None,
            ),
            execution_environment_resolver=(
                make_test_isaac_lab_execution_environment_manifest
            ),
        )

        with self.assertRaisesRegex(
            ValueError, "backend_runtime_dynamics_hash_missing:train"
        ):
            adapter.materialize_jobs(
                round_id="round-unbound",
                task_spec=self.task,
                candidate=self.candidate,
                policy_initialization=PolicyInitialization.random(),
            )

    def test_materializes_warm_start_train_with_explicit_init_checkpoint(self) -> None:
        baseline_checksum = "sha256:" + "1" * 64
        checkpoint = CheckpointRef(
            uri="baseline://push/policy.pt",
            checksum=baseline_checksum,
            role="warm_start",
            sealed=True,
        )
        init = PolicyInitialization.warm_start(
            BaselinePolicyRef(
                baseline_id="push-baseline",
                task_id="unitree_push_recovery",
                checkpoint=checkpoint,
            )
        )

        bundle = self.adapter.materialize_jobs(
            round_id="round-002",
            task_spec=self.task,
            candidate=self.candidate,
            policy_initialization=init,
        )

        self.assertEqual(
            bundle.train_spec.training_start_checkpoint().uri,
            "baseline://push/policy.pt",
        )
        self.assertIn("--init warm_start", bundle.train_spec.metadata["cli_command"])
        self.assertIn(
            "--init-checkpoint baseline://push/policy.pt",
            bundle.train_spec.metadata["cli_command"],
        )
        self.assertIn(
            f"--init-checkpoint-sha256 {baseline_checksum}",
            bundle.train_spec.metadata["cli_command"],
        )

    def test_structured_compatibility_failure_for_unsupported_task_or_initialization(
        self,
    ) -> None:
        unsupported_task = TaskSpec(
            task_id="stair_climbing",
            name="Stair Climbing",
            supported_policy_initializations=frozenset({"random"}),
            genesis_anchor_policy={"random": "bootstrap"},
            candidate_capability_profile=TEST_CAPABILITY_PROFILE,
            invariants=TaskInvariantSpec({}),
        )

        task_report = self.adapter.check_compatibility(
            task_spec=unsupported_task,
            policy_initialization="random",
        )
        init_report = self.adapter.check_compatibility(
            task_spec=self.task,
            policy_initialization="unsupported_init",
        )

        self.assertFalse(task_report.compatible)
        self.assertIn("unsupported_task:stair_climbing", task_report.reasons)
        self.assertFalse(init_report.compatible)
        self.assertIn(
            "unsupported_policy_initialization:unsupported_init", init_report.reasons
        )

    def test_materializes_phase8b_full_budget_cli_with_explicit_max_iterations(
        self,
    ) -> None:
        task = self._task(
            extra_invariants={
                "budget_steps": 49_152_000,
                "train_num_envs": 4096,
                "train_max_iterations": 500,
                "train_steps_per_iteration": 24,
            }
        )
        adapter = IsaacLabRslRlBackendAdapter(
            IsaacLabRslRlBackendConfig(
                **self._config_kwargs(
                    task,
                    default_budget_steps=49_152_000,
                    train_num_envs=4096,
                    train_max_iterations=500,
                    train_steps_per_iteration=24,
                )
            ),
            execution_environment_resolver=(
                make_test_isaac_lab_execution_environment_manifest
            ),
        )

        bundle = adapter.materialize_jobs(
            round_id="round-full-budget",
            task_spec=task,
            candidate=self.candidate,
            policy_initialization=PolicyInitialization.random(),
        )

        argv = bundle.train_spec.metadata["cli_argv"]
        self.assertEqual(bundle.train_spec.budget_steps, 49_152_000)
        self.assertIn("--budget-steps", argv)
        self.assertEqual(argv[argv.index("--budget-steps") + 1], "49152000")
        self.assertIn("--num-envs", argv)
        self.assertEqual(argv[argv.index("--num-envs") + 1], "4096")
        self.assertIn("--max-iterations", argv)
        self.assertEqual(argv[argv.index("--max-iterations") + 1], "500")
        self.assertIn("--num-steps-per-env", argv)
        self.assertEqual(argv[argv.index("--num-steps-per-env") + 1], "24")
        self.assertIn("--max-iterations 500", bundle.train_spec.metadata["cli_command"])
        self.assertIn(
            "--num-steps-per-env 24", bundle.train_spec.metadata["cli_command"]
        )

    def test_rejects_backend_budget_mismatch_against_task_invariant(self) -> None:
        task = self._task(
            extra_invariants={"budget_steps": 49_152_000},
        )

        with self.assertRaisesRegex(ValueError, "task_budget_mismatch"):
            self.adapter.materialize_jobs(
                round_id="round-budget-mismatch",
                task_spec=task,
                candidate=self.candidate,
                policy_initialization=PolicyInitialization.random(),
            )

    def test_rejects_task_protocol_integer_type_coercion(self) -> None:
        task = self._task(
            extra_invariants={
                "budget_steps": "2048",
                "public_seed_count": "4",
            }
        )

        with self.assertRaisesRegex(ValueError, "must be a positive integer"):
            self.adapter.materialize_jobs(
                round_id="round-protocol-type-coercion",
                task_spec=task,
                candidate=self.candidate,
                policy_initialization=PolicyInitialization.random(),
            )


if __name__ == "__main__":
    unittest.main()
