from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.components import DiagnosticTerm, RewardInputs, RewardTerm, ScheduleTerm, TrainingComponents
from true_rl_agent.candidates.loader import load_candidate_manifest, load_training_components
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import StaticCandidateVerifier


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


if __name__ == "__main__":
    unittest.main()
