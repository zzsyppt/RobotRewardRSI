from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class Phase14ExtensionDocsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guide = PROJECT_ROOT / "docs" / "extension_and_boundary_guide.md"
        self.text = self.guide.read_text(encoding="utf-8")

    def test_extension_guide_documents_required_paths_and_boundaries(self) -> None:
        for required in (
            "Add A TaskPlugin",
            "Add A BackendAdapter",
            "Baseline Checkpoint Checklist",
            "Random Vs Warm_Start",
            "Prompt-Facing Capability Admission",
            "Trusted Eval And Runner-Owned Boundary",
            "Redirection And Reach Task Path",
            "TaskMemoryAdapter",
            "disabled_by_phase",
            "checkpoint selector",
            "hidden eval",
            "private seeds",
            "trusted seeds",
        ):
            self.assertIn(required, self.text)

    def test_extension_guide_blocks_authority_leaks(self) -> None:
        for forbidden_statement in (
            "Task plugins must not create training commands",
            "Backend adapters materialize jobs. They do not run training",
            "Do not add task-specific branches to AgentCore",
            "First-version curriculum schedule proposals are disabled_by_phase and must not enter `TrainJobSpec`",
            "The checkpoint selector is runner-owned and must not enter LLM prompt payloads",
            "Trusted eval is runner-owned evidence",
        ):
            self.assertIn(forbidden_statement, self.text)

    def test_testing_matrix_references_current_required_coverage(self) -> None:
        required_test_files = (
            "tests/unit/test_project_boundaries.py",
            "tests/unit/test_plugin_registries.py",
            "tests/unit/test_core_schema.py",
            "tests/unit/test_baseline_management.py",
            "tests/unit/test_runner_jobs.py",
            "tests/unit/test_candidate_package.py",
            "tests/unit/test_prompt_projection_and_mechanism.py",
            "tests/unit/test_curriculum_phase9.py",
            "tests/integration/test_round_public_boundaries.py",
            "tests/integration/test_live_loop.py",
            "tests/integration/test_session_genesis_live_loop.py",
            "tests/integration/test_fake_round.py",
            "tests/unit/test_push_recovery_task.py",
            "tests/unit/test_memory_context_phase13.py",
            "tests/unit/test_stair_extension_guide.py",
        )

        for relative in required_test_files:
            self.assertTrue((PROJECT_ROOT / relative).is_file(), relative)
            self.assertIn(relative, self.text)

    def test_existing_tests_cover_phase14_acceptance_categories(self) -> None:
        coverage_sources = {
            "registration": PROJECT_ROOT / "tests" / "unit" / "test_composition.py",
            "job materialization": PROJECT_ROOT / "tests" / "unit" / "test_isaac_lab_backend.py",
            "policy initialization": PROJECT_ROOT / "tests" / "unit" / "test_baseline_management.py",
            "public projection": PROJECT_ROOT / "tests" / "unit" / "test_memory_context_phase13.py",
            "trusted eval": PROJECT_ROOT / "tests" / "unit" / "test_runner_jobs.py",
            "prompt boundary": PROJECT_ROOT / "tests" / "unit" / "test_prompt_projection_and_mechanism.py",
            "safe boundary": PROJECT_ROOT / "tests" / "unit" / "test_candidate_package.py",
        }

        expected_terms = {
            "registration": ("TaskCompositionRegistry", "register"),
            "job materialization": ("materializes_random_train", "materializes_warm_start_train"),
            "policy initialization": ("warm_start", "random"),
            "public projection": ("PublicContextBuilder", "private_seed", "checkpoint_selector"),
            "trusted eval": ("trusted_eval_seal", "candidate_code"),
            "prompt boundary": ("assert_public_only_prompt_payload", "private"),
            "safe boundary": ("forbidden", "entrypoint"),
        }

        for category, path in coverage_sources.items():
            text = path.read_text(encoding="utf-8")
            for term in expected_terms[category]:
                self.assertIn(term, text, category)


if __name__ == "__main__":
    unittest.main()
