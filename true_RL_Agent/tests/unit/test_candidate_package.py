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

from true_rl_agent.candidates.components import DiagnosticTerm, RewardTerm, ScheduleTerm, TrainingComponents
from true_rl_agent.candidates.loader import load_candidate_manifest, load_training_components
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
