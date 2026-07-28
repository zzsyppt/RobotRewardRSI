from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.artifact import (
    CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
    VerifiedCandidateArtifactRef,
    seal_candidate_package,
    verify_sealed_candidate_artifact,
)
from true_rl_agent.candidates.components import (
    DiagnosticTerm,
    RewardInputs,
    RewardTerm,
    ScheduleTerm,
    TrainingComponents,
)
from true_rl_agent.candidates.loader import (
    load_candidate_manifest,
    load_training_components,
)
from true_rl_agent.candidates.package_contract import (
    PROVIDER_AUTHORED_CANDIDATE_FILES,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import (
    STATIC_CANDIDATE_VERIFIER_POLICY_ID,
    StaticCandidateVerifier,
    audit_candidate_python_source,
    seal_and_verify_candidate,
    static_verifier_policy_facts,
    static_verifier_policy_hash,
    static_verifier_prompt_rules,
)


class CandidatePackageTests(unittest.TestCase):
    def _write_candidate(self, root: Path, *, components_source: str | None = None) -> Path:
        candidate = root / "candidate"
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            textwrap.dedent(
                """
                schema_version: true_rl_agent.candidate.v1
                candidate_id: candidate_reward
                entrypoint: training_code.components:build_training_components
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (candidate / "training_code" / "components.py").write_text(
            components_source
            or textwrap.dedent(
                """
                from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


                def upright_bonus(inputs):
                    return inputs.base_height


                def build_training_components():
                    return TrainingComponents(
                        name="candidate_reward",
                        reward_terms=[RewardTerm(name="upright_bonus", weight=1.0, fn=upright_bonus)],
                        rationale="code-level reward callable",
                    )
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return candidate

    def test_loader_reads_manifest_and_controlled_training_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(Path(tmp))

            manifest = load_candidate_manifest(candidate)
            components = load_training_components(candidate)

            self.assertEqual(manifest.candidate_id, "candidate_reward")
            self.assertIsInstance(components, TrainingComponents)
            self.assertIsInstance(components.reward_terms[0], RewardTerm)
            self.assertTrue(callable(components.reward_terms[0].fn))
            self.assertEqual(components.to_change_set().changed_components, frozenset({"reward_callable"}))

    def test_loader_tolerates_utf8_bom_manifest_from_windows_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(Path(tmp))
            manifest_path = candidate / "candidate_manifest.yaml"
            manifest_path.write_text("\ufeff" + manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

            manifest = load_candidate_manifest(candidate)

            self.assertEqual(manifest.schema_version, "true_rl_agent.candidate.v1")
            self.assertEqual(manifest.candidate_id, "candidate_reward")

    def test_training_components_support_only_currently_wired_hooks(self) -> None:
        components = TrainingComponents(
            name="mixed",
            reward_terms=[RewardTerm("reward", 1.0, lambda inputs: 1.0)],
            diagnostics=[DiagnosticTerm("diag", lambda inputs: {"ok": True})],
            schedules=[ScheduleTerm("scale", "reward_weight_scale", 1.0)],
        )

        summary = components.to_public_summary()

        self.assertEqual(summary["effective_hook_counts"]["reward_terms"], 1)
        self.assertEqual(summary["effective_hook_counts"]["diagnostics"], 1)
        self.assertEqual(summary["effective_hook_counts"]["schedules"], 1)
        self.assertNotIn("loss_terms", summary["effective_hook_counts"])
        self.assertNotIn("observation_transforms", summary["effective_hook_counts"])

    def test_static_verifier_rejects_forbidden_files_and_forbidden_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source="import os\n\ndef build_training_components():\n    return None\n",
            )
            (candidate / "repo.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertIn("forbidden_file:repo.patch", result.details["issues"])
            self.assertIn("forbidden_token:training_code/components.py:import os", result.details["issues"])

    def test_static_verifier_accepts_candidate_package_ref_with_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))
            candidate = CandidatePackageRef(
                candidate_id="candidate_reward",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            result = StaticCandidateVerifier().verify(candidate)

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.reason, "static_candidate_verifier_passed")

    def test_sealed_artifact_binds_the_bytes_verified_by_static_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            provider_candidate = CandidatePackageRef(
                candidate_id="candidate_reward",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            artifact = seal_candidate_package(
                provider_candidate,
                artifact_root=root / "artifacts",
            )

            result = StaticCandidateVerifier().verify(artifact)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.candidate_artifact_hash, artifact.artifact_hash)
            self.assertEqual(
                result.program_identity.source_checksum,
                artifact.program_source_checksum,
            )
            self.assertEqual(
                result.program_identity.manifest_hash,
                artifact.subject_manifest_hash,
            )

    def test_unified_pipeline_is_the_only_public_verified_capability_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            verification = seal_and_verify_candidate(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=CandidateChangeSet(
                        frozenset({"reward_callable"})
                    ),
                ),
                artifact_root=root / "artifacts",
                verifier=StaticCandidateVerifier(),
            )

            artifact = verification.require_verified()

        self.assertTrue(verification.passed)
        self.assertIsInstance(artifact, VerifiedCandidateArtifactRef)
        self.assertEqual(
            artifact.verification_policy,
            STATIC_CANDIDATE_VERIFIER_POLICY_ID,
        )
        self.assertEqual(
            verification.result.details["verification_policy_hash"],
            static_verifier_policy_hash(),
        )
        with self.assertRaisesRegex(TypeError, "verification pipeline"):
            VerifiedCandidateArtifactRef(
                sealed_artifact=artifact.sealed_artifact,
                verification_policy=artifact.verification_policy,
                verification_hash=artifact.verification_hash,
            )

    def test_unified_pipeline_does_not_mint_capability_for_rejected_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(
                root,
                components_source=(
                    "import os\n\n"
                    "def build_training_components():\n"
                    "    return None\n"
                ),
            )
            verification = seal_and_verify_candidate(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=CandidateChangeSet(
                        frozenset({"reward_callable"})
                    ),
                ),
                artifact_root=root / "artifacts",
                verifier=StaticCandidateVerifier(),
            )

        self.assertFalse(verification.passed)
        self.assertIsNone(verification.verified_artifact)
        self.assertEqual(
            verification.result.reason,
            "static_candidate_verifier_failed",
        )

    def test_provider_file_contract_excludes_runner_owned_artifact_manifest(self) -> None:
        prompt_rules = static_verifier_prompt_rules()

        self.assertEqual(
            prompt_rules["allowed_files"],
            sorted(PROVIDER_AUTHORED_CANDIDATE_FILES),
        )
        self.assertNotIn(
            CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
            prompt_rules["allowed_files"],
        )

    def test_provider_created_runner_manifest_is_rejected_before_sealing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            (candidate_dir / CANDIDATE_ARTIFACT_MANIFEST_FILENAME).write_text(
                "{}\n",
                encoding="utf-8",
            )
            candidate = CandidatePackageRef(
                candidate_id="candidate_reward",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(
                    frozenset({"reward_callable"})
                ),
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertIn(
                "forbidden_file:.true_rl_agent_candidate_artifact.json",
                result.details["issues"],
            )
            with self.assertRaisesRegex(ValueError, "reserved artifact manifest"):
                seal_candidate_package(
                    candidate,
                    artifact_root=root / "artifacts",
                )

    def test_sealed_artifact_rejects_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            artifact = seal_candidate_package(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=CandidateChangeSet(
                        frozenset({"reward_callable"})
                    ),
                ),
                artifact_root=root / "artifacts",
            )
            source = Path(artifact.path) / "training_code" / "components.py"
            source.write_text("def build_training_components():\n    return None\n")

            with self.assertRaisesRegex(ValueError, "file evidence mismatch"):
                verify_sealed_candidate_artifact(artifact)

    def test_static_verifier_binds_package_ref_id_to_loaded_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))
            candidate = CandidatePackageRef(
                candidate_id="different_candidate",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            result = StaticCandidateVerifier().verify(candidate)

        self.assertEqual(result.status, "fail")
        self.assertIn("manifest_candidate_id_mismatch", result.details["issues"])

    def test_static_verifier_accepts_reward_inputs_type_import_exposed_in_prompt_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents


                    def typed_reward(inputs: RewardInputs):
                        return inputs.base_height


                    def build_training_components():
                        return TrainingComponents(
                            name="typed_reward_candidate",
                            reward_terms=[RewardTerm(name="typed_reward", weight=1.0, fn=typed_reward)],
                        )
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier().verify(candidate_dir)

        self.assertEqual(RewardInputs.field_names()[0], "base_height")
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.details["derived_change_set"], ["reward_callable"])

    def test_common_reward_inputs_are_task_neutral_and_push_task_extends_them(self) -> None:
        from true_rl_agent.tasks.push_recovery.reward_input_contract import (
            push_recovery_reward_input_fields,
        )
        from true_rl_agent.tasks.push_recovery.reward_inputs import (
            PushRecoveryRewardInputs,
        )

        common_fields = RewardInputs.field_names()
        push_fields = PushRecoveryRewardInputs.field_names()
        task_contract_fields = push_recovery_reward_input_fields()

        self.assertNotIn("push_active", common_fields)
        self.assertNotIn("push_vec_public", common_fields)
        self.assertIn("push_active", push_fields)
        self.assertIn("push_vec_public", push_fields)
        for name in common_fields:
            self.assertIn(name, push_fields)
            self.assertIn(name, task_contract_fields)
        self.assertLess(task_contract_fields.index("action_delta"), task_contract_fields.index("push_active"))
        self.assertLess(task_contract_fields.index("push_vec_public"), task_contract_fields.index("progress_fraction"))

    def test_static_verifier_tolerates_bom_components_and_ignores_pycache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))
            component_path = candidate_dir / "training_code" / "components.py"
            component_path.write_text("\ufeff" + component_path.read_text(encoding="utf-8"), encoding="utf-8")
            pycache = candidate_dir / "training_code" / "__pycache__"
            pycache.mkdir()
            (pycache / "components.cpython-312.pyc").write_bytes(b"cache")

            result = StaticCandidateVerifier().verify(candidate_dir)

            self.assertEqual(result.status, "pass")

    def test_static_verifier_rejects_entrypoint_escape_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(Path(tmp))
            (candidate / "candidate_manifest.yaml").write_text(
                textwrap.dedent(
                    """
                    schema_version: true_rl_agent.candidate.v1
                    candidate_id: candidate_reward
                    entrypoint: ../secret:build_training_components
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertIn("manifest_load_failed:ValueError", result.details["issues"])

    def test_static_verifier_rejects_dynamic_import_and_top_level_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import TrainingComponents

                    leaked = __import__("os")


                    def build_training_components():
                        return TrainingComponents(name="candidate_reward")
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertIn("forbidden_top_level:training_code/components.py:Assign", result.details["issues"])
            self.assertIn("forbidden_call:training_code/components.py:__import__", result.details["issues"])

    def test_static_verifier_rejects_non_callable_reward_term(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


                    def build_training_components():
                        return TrainingComponents(
                            name="candidate_reward",
                            reward_terms=[RewardTerm(name="bad_reward", weight=1.0, fn=1)],
                        )
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertEqual(result.reason, "training_components_invalid")
            self.assertIn("non_callable_reward_term:bad_reward", result.details["issues"])

    def test_static_verifier_times_out_hanging_component_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import TrainingComponents


                    def build_training_components():
                        while True:
                            pass
                        return TrainingComponents(name="unreachable")
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier(component_load_timeout_s=0.25).verify(candidate)

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.reason, "training_components_load_timeout")
        self.assertIn("component_load_timeout", result.details["issues"])

    def test_static_verifier_rejects_unallowlisted_schedule_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import ScheduleTerm, TrainingComponents


                    def build_training_components():
                        return TrainingComponents(
                            name="candidate_schedule",
                            schedules=[ScheduleTerm(name="obs_schedule", key="obs_transform", value=1.0)],
                        )
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertEqual(result.reason, "training_components_invalid")
            self.assertIn("schedule_key_not_allowlisted:obs_schedule:obs_transform", result.details["issues"])

    def test_static_verifier_rejects_mutation_and_process_state_surfaces(self) -> None:
        cases = {
            "tensor_in_place": "inputs.base_height.zero_()",
            "subscript_write": "inputs.base_height[0] = 0.0",
            "attribute_write": "inputs.base_height.data = inputs.base_height",
            "dynamic_builtins": (
                "def reward(inputs):\n"
                "    return __builtins__['__import__']('os')"
            ),
            "dynamic_reflection": (
                "def reward(inputs):\n"
                "    return getattr(inputs, 'base_height')"
            ),
            "class_mutation": "setattr(TrainingComponents, 'leaked', True)",
            "console_side_effect": "print('candidate side effect')",
            "global_state": "global leaked",
            "nonlocal_state": (
                "def outer():\n"
                "    leaked = 0\n"
                "    def reward(inputs):\n"
                "        nonlocal leaked\n"
                "        return inputs.base_height\n"
                "    return reward\n"
            ),
            "mutable_default": "def helper(values=[]):\n    return values",
        }
        for label, injected in cases.items():
            with self.subTest(label=label):
                source = (
                    "from true_rl_agent.candidates.components import "
                    "TrainingComponents\n\n"
                    f"{injected}\n\n"
                    "def build_training_components():\n"
                    "    return TrainingComponents(name='candidate_reward')\n"
                )
                issues = audit_candidate_python_source(source)
                self.assertTrue(issues, label)

    def test_static_verifier_requires_exact_source_defined_stateless_callables(self) -> None:
        cases = {
            "builtin": (
                "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
                "def build_training_components():\n"
                "    return TrainingComponents(\n"
                "        name='candidate_reward',\n"
                "        reward_terms=[RewardTerm(name='bad', weight=1.0, fn=abs)],\n"
                "    )\n"
            ),
            "closure": (
                "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
                "def build_training_components():\n"
                "    state = iter((0.0, 1.0))\n"
                "    reward = lambda inputs: inputs.base_height + next(state)\n"
                "    return TrainingComponents(\n"
                "        name='candidate_reward',\n"
                "        reward_terms=[RewardTerm(name='bad', weight=1.0, fn=reward)],\n"
                "    )\n"
            ),
        }
        for label, source in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                candidate = self._write_candidate(
                    Path(tmp),
                    components_source=source,
                )
                result = StaticCandidateVerifier().verify(candidate)

                self.assertEqual(result.status, "fail")
                self.assertEqual(result.reason, "training_components_invalid")
                self.assertTrue(
                    any(
                        issue.startswith(
                            (
                                "non_source_defined_reward_term:",
                                "stateful_reward_term_closure:",
                            )
                        )
                        for issue in result.details["issues"]
                    ),
                    result.details,
                )

    def test_static_verifier_rejects_random_api_variants(self) -> None:
        sources = (
            "def reward(inputs):\n    return random.random()\n",
            "def reward(inputs):\n    return np.random.normal()\n",
            "def reward(inputs):\n    return torch.randn(1)\n",
            "def reward(inputs):\n    return inputs.base_height.bernoulli()\n",
        )
        for source in sources:
            with self.subTest(source=source):
                issues = audit_candidate_python_source(source)
                self.assertTrue(
                    any(issue.startswith("forbidden_random_call:") for issue in issues),
                    issues,
                )

    def test_static_verifier_policy_is_exposed_from_one_source(self) -> None:
        prompt_rules = static_verifier_prompt_rules()

        self.assertEqual(
            prompt_rules["verification_policy_id"],
            STATIC_CANDIDATE_VERIFIER_POLICY_ID,
        )
        self.assertEqual(
            prompt_rules["verification_policy_hash"],
            static_verifier_policy_hash(),
        )
        facts = static_verifier_policy_facts()
        self.assertEqual(
            prompt_rules["blocked_ast_nodes"],
            facts["forbidden_ast_nodes"],
        )
        self.assertEqual(
            prompt_rules["blocked_assignment_target_nodes"],
            facts["forbidden_assignment_target_nodes"],
        )
        self.assertEqual(
            prompt_rules["blocked_name_prefixes"],
            facts["forbidden_name_prefixes"],
        )
        self.assertEqual(
            prompt_rules["component_callable_contract"],
            facts["component_callable_contract"],
        )
        self.assertFalse(prompt_rules["function_decorators_allowed"])
        self.assertFalse(prompt_rules["nested_functions_allowed"])
        self.assertFalse(prompt_rules["mutable_or_dynamic_defaults_allowed"])
        with patch.object(
            StaticCandidateVerifier,
            "nested_functions_allowed",
            True,
        ):
            self.assertNotEqual(
                static_verifier_policy_hash(),
                prompt_rules["verification_policy_hash"],
            )
        self.assertIn("bernoulli", prompt_rules["blocked_random_calls"])
        self.assertIn("zero_", audit_candidate_python_source("def f(x):\n    return x.zero_()\n")[0])


if __name__ == "__main__":
    unittest.main()
