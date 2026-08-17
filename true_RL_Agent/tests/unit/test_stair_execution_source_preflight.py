from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.isaaclab_stair_rsl_rl_eval import (  # noqa: E402
    _failure_summary,
    _run_eval,
)
from scripts.isaaclab_stair_rsl_rl_train import _status_summary  # noqa: E402

from tests.fakes.backend_environment import (  # noqa: E402
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (  # noqa: E402
    StairEvaluationExecutionResult,
    StairEvaluationRuntimeLease,
    VerifiedStairExecutionSourceManifest,
    execute_stair_evaluation,
    preflight_stair_execution_source_manifest,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (  # noqa: E402
    STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
    stair_training_distribution_upstream_source_evidence,
    verify_installed_stair_training_distribution_upstream_sources,
)


def _split_execution_cost() -> dict[str, int | float]:
    return {
        "wall_time_s": 0.125,
        "seed_count": 1,
        "evaluation_env_count_per_seed": 1,
        "vector_env_step_calls": 2,
        "simulated_env_control_steps": 2,
        "closed_first_episode_control_steps": 2,
    }


def _execution_identity_fields(
    config: StairClimbingTaskConfig,
    *,
    namespace: str = "formal_reference_candidate_eval",
) -> dict[str, object]:
    source = stair_training_distribution_upstream_source_evidence(
        config.protocol_v2.runner_owned_training_distribution
    )
    backend_hash = "b" * 64
    return {
        "checkpoint_hash": "sha256:" + "a" * 64,
        "seed_namespace": namespace,
        "seed_cohort_hash": config.seed_resolver().namespace_cohort_hash(namespace),
        "source_manifest_preflight": {
            "schema_version": STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
            "status": "pass",
            "task_protocol_hash": config.task_protocol_hash,
            "source_manifest_hash": source["source_manifest_hash"],
        },
        "backend_execution_environment_verification": {
            "schema_version": (
                "true_rl_agent.backend_execution_environment_verification.v1"
            ),
            "backend_id": "isaac_lab_rsl_rl",
            "status": "pass",
            "expected_manifest_hash": backend_hash,
            "actual_manifest_hash": backend_hash,
        },
    }


class StairExecutionSourcePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )

    def test_same_symbol_names_cannot_hide_installed_source_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contract, roots = self._fixture_contract(Path(tmp))
            find_spec = self._find_spec(roots)
            with patch(
                "true_rl_agent.tasks.stair_climbing.training_distribution."
                "importlib.util.find_spec",
                side_effect=find_spec,
            ):
                evidence = (
                    verify_installed_stair_training_distribution_upstream_sources(
                        contract
                    )
                )
                curriculum_path = (
                    roots["isaaclab_tasks"]
                    / "manager_based/locomotion/velocity/mdp/curriculums.py"
                )
                original = curriculum_path.read_bytes()
                changed = original.replace(b"return 1", b"return 2")
                self.assertEqual(len(changed), len(original))
                curriculum_path.write_bytes(changed)

                with self.assertRaisesRegex(ValueError, "source bytes drifted"):
                    verify_installed_stair_training_distribution_upstream_sources(
                        contract
                    )

            self.assertEqual(evidence["status"], "pass")
            self.assertIn(b"terrain_levels_vel", changed)

    def test_preflight_returns_protocol_bound_verifier_issued_capability(self) -> None:
        expected = stair_training_distribution_upstream_source_evidence(
            self.config.protocol_v2.runner_owned_training_distribution
        )
        with patch(
            "true_rl_agent.tasks.stair_climbing.isaaclab_execution."
            "verify_installed_stair_training_distribution_upstream_sources",
            return_value=expected,
        ) as verify:
            source_preflight = preflight_stair_execution_source_manifest(self.config)

        verify.assert_called_once_with(
            self.config.protocol_v2.runner_owned_training_distribution
        )
        source_preflight.validate_for(self.config)
        self.assertEqual(source_preflight.upstream_source_evidence(), expected)
        self.assertEqual(
            source_preflight.to_summary_dict(),
            {
                "schema_version": STAIR_EXECUTION_SOURCE_PREFLIGHT_SCHEMA_VERSION,
                "status": "pass",
                "task_protocol_hash": self.config.task_protocol_hash,
                "source_manifest_hash": expected["source_manifest_hash"],
            },
        )
        with self.assertRaisesRegex(TypeError, "must come from the source preflight"):
            VerifiedStairExecutionSourceManifest(
                task_protocol_hash=self.config.task_protocol_hash,
                upstream_source_evidence=expected,
            )

    def test_public_execution_boundary_rejects_description_only_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.write_bytes(b"not-loaded")
            description = stair_training_distribution_upstream_source_evidence(
                self.config.protocol_v2.runner_owned_training_distribution
            )
            with self.assertRaisesRegex(
                TypeError,
                "requires a verified source manifest preflight",
            ):
                execute_stair_evaluation(
                    config=self.config,
                    source_manifest_preflight=description,  # type: ignore[arg-type]
                    backend_execution_environment_verification=(
                        _execution_identity_fields(self.config)[
                            "backend_execution_environment_verification"
                        ]
                    ),
                    checkpoint_path=checkpoint,
                    device="cpu",
                    seed_namespace="not-reached",
                    splits=("public_dev",),
                )

    def test_multi_split_execution_requires_a_subprocess_runtime_lease(self) -> None:
        expected = stair_training_distribution_upstream_source_evidence(
            self.config.protocol_v2.runner_owned_training_distribution
        )
        with patch(
            "true_rl_agent.tasks.stair_climbing.isaaclab_execution."
            "verify_installed_stair_training_distribution_upstream_sources",
            return_value=expected,
        ):
            source_preflight = preflight_stair_execution_source_manifest(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.write_bytes(b"not-loaded")
            with self.assertRaisesRegex(
                TypeError,
                "requires a subprocess runtime lease",
            ):
                execute_stair_evaluation(
                    config=self.config,
                    source_manifest_preflight=source_preflight,
                    backend_execution_environment_verification=(
                        _execution_identity_fields(self.config)[
                            "backend_execution_environment_verification"
                        ]
                    ),
                    checkpoint_path=checkpoint,
                    device="cpu",
                    seed_namespace="not-reached",
                    splits=("public_dev", "public_accept"),
                )

        lease = StairEvaluationRuntimeLease()
        self.assertFalse(lease.is_bound)
        lease.bind(object())
        self.assertTrue(lease.is_bound)
        with self.assertRaisesRegex(RuntimeError, "already bound"):
            lease.bind(object())

    def test_evaluation_result_exposes_detached_json_summaries(self) -> None:
        source = {
            "public_dev": {
                "score_stats": {"mean": 0.25},
                "seed_macro_scores": [0.0, 0.5],
            }
        }
        result = StairEvaluationExecutionResult(
            summaries=source,
            per_seed_outcomes={
                "public_dev": ((SimpleNamespace(episode_steps=2),),)
            },
            split_vector_env_step_calls={"public_dev": 2},
            split_wall_times_s={"public_dev": 0.125},
            **_execution_identity_fields(self.config),
            execution_device="cuda:1",
        )

        source["public_dev"]["score_stats"]["mean"] = 1.0
        projected = result.public_summaries()
        self.assertEqual(projected["public_dev"]["score_stats"]["mean"], 0.25)
        json.dumps(projected, allow_nan=False)

        projected["public_dev"]["score_stats"]["mean"] = 0.75
        self.assertEqual(
            result.public_summaries()["public_dev"]["score_stats"]["mean"],
            0.25,
        )
        with self.assertRaises(TypeError):
            result.summaries["public_dev"]["score_stats"]["mean"] = 0.5
        for invalid in ({1: {}}, {"public_dev": {"score": float("nan")}}):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    StairEvaluationExecutionResult(
                        summaries=invalid,
                        per_seed_outcomes={
                            next(iter(invalid)): (
                                (SimpleNamespace(episode_steps=2),),
                            )
                        },
                        split_vector_env_step_calls={next(iter(invalid)): 2},
                        split_wall_times_s={next(iter(invalid)): 0.125},
                        **_execution_identity_fields(self.config),
                        execution_device="cuda:1",
                    )

    def test_formal_eval_projects_execution_result_before_serialization(self) -> None:
        split_summary = {
            "score_stats": {"mean": 0.25},
            "seed_macro_scores": [0.25],
        }
        execution = StairEvaluationExecutionResult(
            summaries={
                split: split_summary
                for split in ("public_dev", "public_accept", "sealed_validation")
            },
            per_seed_outcomes={
                split: ((SimpleNamespace(episode_steps=2),),)
                for split in ("public_dev", "public_accept", "sealed_validation")
            },
            split_vector_env_step_calls={
                split: 2
                for split in ("public_dev", "public_accept", "sealed_validation")
            },
            split_wall_times_s={
                split: 0.125
                for split in ("public_dev", "public_accept", "sealed_validation")
            },
            **_execution_identity_fields(self.config),
            execution_device="cuda:1",
        )
        evaluation_protocol = SimpleNamespace(
            protocol_hash="a" * 64,
            to_public_dict=lambda: {"protocol_hash": "a" * 64},
        )
        context = SimpleNamespace(
            config=SimpleNamespace(
                protocol_v1=object(),
                evaluation_semantics=object(),
                evaluation_protocol=lambda: evaluation_protocol,
            ),
            identity=SimpleNamespace(
                to_public_dict=lambda: {
                    "job_id": "job",
                    "task_id": "unitree_stair_climbing",
                    "subject_kind": "reference",
                    "subject_id": "reference",
                    "subject_manifest_hash": "b" * 64,
                }
            ),
            evaluation_binding=SimpleNamespace(
                binding_hash="c" * 64,
                to_public_dict=lambda: {"binding_hash": "c" * 64},
            ),
            evaluation_subject=object(),
            checkpoint_path=Path("checkpoint.pt"),
            checkpoint_hash="sha256:" + "d" * 64,
            subject_manifest_hash="b" * 64,
            program_source_checksum="sha256:" + "e" * 64,
            training_binding_hash="f" * 64,
            capability_profile_hash="1" * 64,
        )
        source_preflight = SimpleNamespace(
            to_summary_dict=lambda: {"status": "pass"}
        )
        checkpoint_report = {
            "status": "pass",
            "checkpoint_hash_match_status": "pass",
        }
        expected_summary = {
            "status": "pass",
            "metrics": {
                "stair_evaluation": {
                    "public_accept": split_summary,
                }
            },
            "sealed_validation_summary": split_summary,
            "evaluation_execution_cost": execution.execution_cost_artifact(),
        }
        with (
            patch(
                "scripts.isaaclab_stair_rsl_rl_eval.execute_stair_evaluation",
                return_value=execution,
            ),
            patch(
                "scripts.isaaclab_stair_rsl_rl_eval."
                "build_stair_full_eval_summary",
                return_value=expected_summary,
            ) as summary_builder,
        ):
            summary = _run_eval(
                SimpleNamespace(device="cuda:1"),
                context=context,
                source_manifest_preflight=source_preflight,
                execution_environment_verification=(
                    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
                ),
                checkpoint_report=checkpoint_report,
                runtime_lease=StairEvaluationRuntimeLease(),
            )

        json.dumps(summary, allow_nan=False)
        self.assertIs(summary_builder.call_args.kwargs["execution"], execution)
        self.assertIs(summary_builder.call_args.kwargs["config"], context.config)
        self.assertEqual(
            summary["metrics"]["stair_evaluation"]["public_accept"][
                "score_stats"
            ]["mean"],
            0.25,
        )
        self.assertEqual(summary["sealed_validation_summary"], split_summary)
        self.assertEqual(
            summary["evaluation_execution_cost"]["splits"]["public_accept"],
            _split_execution_cost(),
        )
        self.assertNotIn("evaluation_execution_cost", summary["metrics"])

    def test_formal_summaries_record_the_verified_manifest_hash(self) -> None:
        expected = stair_training_distribution_upstream_source_evidence(
            self.config.protocol_v2.runner_owned_training_distribution
        )
        with patch(
            "true_rl_agent.tasks.stair_climbing.isaaclab_execution."
            "verify_installed_stair_training_distribution_upstream_sources",
            return_value=expected,
        ):
            source_preflight = preflight_stair_execution_source_manifest(self.config)

        identity = SimpleNamespace(
            to_public_dict=lambda: {
                "job_id": "job",
                "task_id": "unitree_stair_climbing",
                "subject_kind": "reference",
                "subject_id": "reference",
                "subject_manifest_hash": "1" * 64,
            }
        )
        binding = SimpleNamespace(
            binding_hash="2" * 64,
            policy_architecture_hash="3" * 64,
            optimizer_training_profile_hash="4" * 64,
            agent_configuration_hash="5" * 64,
            initialization_seed_schedule_hash="6" * 64,
            capability_profile_hash="7" * 64,
        )
        train_summary = _status_summary(
            "pass",
            args=SimpleNamespace(round="round"),
            context=SimpleNamespace(
                training_binding=binding,
                identity=identity,
                program_source_checksum="sha256:" + "8" * 64,
                candidate_verification_hash=None,
                candidate_execution_authorization=None,
                candidate_failure_contract_hash=None,
            ),
            source_manifest_preflight=source_preflight,
            execution_environment_verification=(
                TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
            ),
            reward_program=SimpleNamespace(
                program_source_checksum="sha256:" + "8" * 64,
                qualification_hash="9" * 64,
            ),
            attempt_id="attempt",
            formal_training_executed=True,
            execution_device="cuda:0",
        )
        evaluation_protocol = SimpleNamespace(
            to_public_dict=lambda: {"protocol_hash": "a" * 64}
        )
        eval_summary = _failure_summary(
            SimpleNamespace(
                identity=identity,
                config=SimpleNamespace(
                    evaluation_protocol=lambda: evaluation_protocol
                ),
                evaluation_binding=SimpleNamespace(
                    to_public_dict=lambda: {"binding_hash": "b" * 64}
                ),
            ),
            source_manifest_preflight=source_preflight,
            execution_environment_verification=(
                TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
            ),
            checkpoint_report={"status": "pass"},
            execution_device="cuda:0",
            error=RuntimeError("test failure"),
        )

        self.assertEqual(
            train_summary["environment_source_preflight"],
            source_preflight.to_summary_dict(),
        )
        self.assertEqual(
            eval_summary["metrics"]["environment_source_preflight"],
            source_preflight.to_summary_dict(),
        )
        self.assertNotIn("environment_source_preflight", eval_summary)

    def _fixture_contract(
        self,
        root: Path,
    ) -> tuple[dict, dict[str, Path]]:
        contract = json.loads(
            json.dumps(
                self.config.protocol_v2.to_public_dict()[
                    "runner_owned_training_distribution"
                ]
            )
        )
        roots = {
            "isaaclab": root / "isaaclab",
            "isaaclab_tasks": root / "isaaclab_tasks",
        }
        source_by_symbol = {
            "TerrainImporter.update_env_origins": (
                b"class TerrainImporter:\n"
                b"    def update_env_origins(self):\n"
                b"        return 1\n"
            ),
            "G1RoughPPORunnerCfg": b"class G1RoughPPORunnerCfg:\n    pass\n",
            "G1RoughEnvCfg": b"class G1RoughEnvCfg:\n    pass\n",
            "terrain_levels_vel": b"def terrain_levels_vel():\n    return 1\n",
            "CurriculumCfg.terrain_levels": (
                b"class CurriculumCfg:\n    terrain_levels = None\n"
            ),
        }
        for item in contract["upstream_reference"]["source_files"]:
            path = roots[item["import_package"]] / item["package_relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            source = source_by_symbol[item["covered_symbols"][0]]
            path.write_bytes(source)
            item["checksum"] = "sha256:" + hashlib.sha256(source).hexdigest()
            item["size_bytes"] = len(source)
        return contract, roots

    @staticmethod
    def _find_spec(roots: dict[str, Path]):
        def find_spec(package: str):
            return SimpleNamespace(
                submodule_search_locations=[str(roots[package])]
            )

        return find_spec


if __name__ == "__main__":
    unittest.main()
