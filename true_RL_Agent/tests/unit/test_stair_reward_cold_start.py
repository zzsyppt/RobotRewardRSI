from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.materialize_job import main as materialize_job  # noqa: E402

from tests.fakes.backend_environment import (  # noqa: E402
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST,
    make_test_isaac_lab_execution_environment_manifest,
)
from tests.fakes.evaluation import make_test_candidate_package  # noqa: E402
from tests.fakes.reference_programs import (  # noqa: E402
    make_verified_stair_reference,
)
from true_rl_agent.agent_core.session_genesis import (  # noqa: E402
    prepare_session_genesis,
)
from true_rl_agent.backends.isaac_lab_rsl_rl import (  # noqa: E402
    IsaacLabRslRlBackendAdapter,
)
from true_rl_agent.composition.base import (  # noqa: E402
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.composition.defaults import (  # noqa: E402
    build_application_composition_registry,
)
from true_rl_agent.memory.context_builder import PublicContextBuilder  # noqa: E402
from true_rl_agent.tasks.stair_climbing.component_smoke import (  # noqa: E402
    StairClimbingComponentSmokeAdapter,
)
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.config import STAIR_TASK_ID  # noqa: E402
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_artifact,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (  # noqa: E402
    STAIR_REWARD_ENVIRONMENT_AXIS_EVIDENCE_SCHEMA_VERSION,
    qualify_stair_candidate_reward_program,
)
from true_rl_agent.tasks.stair_climbing.session_start import (  # noqa: E402
    STAIR_FORMAL_INITIAL_POLICY_STATE_HASH,
    STAIR_REWARD_COLD_START_MODE,
)


class StairRewardColdStartTests(unittest.TestCase):
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

    def _resolve_with_test_execution_environment(self, **kwargs):
        return self._with_test_execution_environment(
            resolve_task_composition(**kwargs)
        )

    def _cold_composition(self, output_dir: Path):
        provider = build_application_composition_registry().get(STAIR_TASK_ID)
        values = dict(provider.cli_defaults())
        values["start_mode"] = STAIR_REWARD_COLD_START_MODE
        values["initial_best_score"] = 0.0
        return self._resolve_with_test_execution_environment(
            provider=provider,
            values=values,
            context=TaskCompositionContext(
                project_root=ROOT,
                output_dir=output_dir,
            ),
        )

    def test_cold_genesis_and_first_prompt_have_no_reference_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            composition = self._cold_composition(output_dir)
            genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id="stair-reward-cold-start-test",
                task_plugin=composition.task_plugin,
                backend_adapter=composition.backend_adapter,
                runner=composition.runner,
                policy_initialization=composition.policy_initialization,
                genesis_identity_snapshot=composition.genesis_identity_snapshot(),
                bootstrap_initial_best_score=0.0,
                bootstrap_initial_policy_state_hash=(
                    composition.bootstrap_initial_policy_state_hash
                ),
                bind_backend_execution_environment=True,
            )

            self.assertEqual(genesis.artifact.genesis_kind, "bootstrap")
            self.assertEqual(
                genesis.artifact.to_dict()[
                    "backend_execution_environment"
                ],
                TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST,
            )
            self.assertEqual(genesis.initial_current_best.mode, "bootstrap")
            self.assertEqual(
                genesis.expected_initial_policy_state_hash,
                STAIR_FORMAL_INITIAL_POLICY_STATE_HASH,
            )
            self.assertIsNone(genesis.reference_program_source())
            self.assertIsNone(genesis.initial_anchor_public_evidence())

            context_builder = PublicContextBuilder(
                task_memory_adapter=composition.task_memory_adapter,
                task_prompt_adapter=composition.task_plugin.prompt_adapter(),
                acceptance_protocol=(
                    composition.task_plugin.task_spec.acceptance_protocol
                ),
                initial_current_best=genesis.initial_current_best,
                session_genesis_hash=genesis.session_genesis_hash,
                candidate_capability_profile=(
                    composition.task_plugin.task_spec.candidate_capability_profile
                ),
                initial_reference_program_source=(
                    genesis.reference_program_source()
                ),
                initial_anchor_public_evidence=(
                    genesis.initial_anchor_public_evidence()
                ),
                expected_initial_policy_state_hash=(
                    genesis.expected_initial_policy_state_hash
                ),
            )
            prompt = context_builder.build_prompt(
                [],
                authoring_capability_readiness=(
                    composition.authoring_capability_readiness.to_prompt_mapping()
                ),
                candidate_verifier_prompt_rules=(
                    composition.candidate_admission_policy.authoring_prompt_rules()
                ),
            )

            self.assertIsNone(prompt.parent_program_source_checksum)
            self.assertEqual(prompt.provider_payload["candidate_lineage"]["status"], "none")
            self.assertIsNone(
                prompt.provider_payload["candidate_lineage"]["current_best"]
            )
            self.assertEqual(prompt.provider_payload["source_payloads"], [])
            capability_ids = {
                item["capability_id"]
                for item in prompt.provider_payload["candidate_capabilities"][
                    "prompt_visible_capabilities"
                ]
            }
            self.assertEqual(capability_ids, {"complete_reward_program"})
            prompt_text = json.dumps(
                prompt.provider_payload,
                sort_keys=True,
                ensure_ascii=False,
            )
            self.assertNotIn("official-g1-rough-reward-v1", prompt_text)
            self.assertNotIn(
                "sha256:a1255b585ffc6752df710be648cf0439017d5220a41853b9dfdebc778c3c03a3",
                prompt_text,
            )
            self.assertNotIn("reference_anchor", prompt_text)
            guidance = prompt.provider_payload["selection_guidance"]
            self.assertTrue(
                any("without a reference reward program" in item for item in guidance)
            )
            self.assertTrue(any("hidden base reward" in item for item in guidance))
            self.assertTrue(
                any("complete replacement reward program" in item for item in guidance)
            )
            common_guidance = prompt.provider_payload["loop_contract"][
                "evidence_discipline"
            ]
            self.assertTrue(
                any(
                    "without an anchor, author the complete initial component set"
                    in item
                    for item in common_guidance
                )
            )

    def test_materializer_requires_cold_genesis_and_binds_its_initial_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            session_id = "stair-cold-materialize-session"
            candidate = make_test_candidate_package("stair-cold-materialized-r1")
            common_argv = [
                "--task-id",
                STAIR_TASK_ID,
                "--start-mode",
                STAIR_REWARD_COLD_START_MODE,
                "--session-id",
                session_id,
                "--round-id",
                "round-001",
                "--candidate-id",
                candidate.candidate_id,
                "--candidate-path",
                candidate.path,
                "--output-dir",
                str(output_dir),
            ]

            with (
                patch(
                    "scripts.materialize_job.resolve_task_composition",
                    side_effect=self._resolve_with_test_execution_environment,
                ),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit) as exc,
            ):
                materialize_job(common_argv)
            self.assertEqual(exc.exception.code, 2)

            composition = self._cold_composition(output_dir)
            genesis = prepare_session_genesis(
                output_dir=output_dir,
                session_id=session_id,
                task_plugin=composition.task_plugin,
                backend_adapter=composition.backend_adapter,
                runner=composition.runner,
                policy_initialization=composition.policy_initialization,
                genesis_identity_snapshot=composition.genesis_identity_snapshot(),
                bootstrap_initial_best_score=(
                    composition.bootstrap_initial_best_score
                ),
                bootstrap_initial_policy_state_hash=(
                    composition.bootstrap_initial_policy_state_hash
                ),
            )
            adapter = StairClimbingComponentSmokeAdapter(
                composition.task_plugin.config
            )
            trusted_reference, _ = resolve_official_stair_reference_artifact(
                composition.task_plugin.config,
                project_root=ROOT,
            )

            def qualification_worker(*, artifact, static_verifier_policy, **_kwargs):
                program = qualify_stair_candidate_reward_program(
                    artifact=artifact,
                    contract=adapter.contract,
                    static_verifier_policy=static_verifier_policy,
                    trusted_reference_artifact=trusted_reference,
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

            output = StringIO()
            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.candidate_admission."
                    "_run_qualification_worker",
                    side_effect=qualification_worker,
                ),
                patch(
                    "scripts.materialize_job.resolve_task_composition",
                    side_effect=self._resolve_with_test_execution_environment,
                ),
                redirect_stdout(output),
            ):
                self.assertEqual(materialize_job(common_argv), 0)

            payload = json.loads(output.getvalue())
            train = payload["train"]
            train_argv = payload["runner_execution"]["train"]["cli_argv"]
            expected_hash = genesis.expected_initial_policy_state_hash
            self.assertEqual(
                train["expected_initial_policy_state_hash"],
                expected_hash,
            )
            self.assertEqual(
                train_argv.count("--expected-initial-policy-state-hash"),
                1,
            )
            option_index = train_argv.index("--expected-initial-policy-state-hash")
            self.assertEqual(train_argv[option_index + 1], expected_hash)
            self.assertNotIn("--reference-artifact", train_argv)
            self.assertNotIn("--reference-artifact-hash", train_argv)

    def test_initial_state_anchor_matches_formal_phase_e_evidence(self) -> None:
        manifest = json.loads(
            (ROOT / "PLAN" / "stair" / "phase_e_completion_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(
            manifest["lineage"]["initial_policy_state_hash"],
            STAIR_FORMAL_INITIAL_POLICY_STATE_HASH,
        )

    def test_reference_mode_cannot_resume_a_cold_genesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            session_id = "stair-mode-isolation"
            cold = self._cold_composition(output_dir)
            prepare_session_genesis(
                output_dir=output_dir,
                session_id=session_id,
                task_plugin=cold.task_plugin,
                backend_adapter=cold.backend_adapter,
                runner=cold.runner,
                policy_initialization=cold.policy_initialization,
                genesis_identity_snapshot=cold.genesis_identity_snapshot(),
                bootstrap_initial_best_score=cold.bootstrap_initial_best_score,
                bootstrap_initial_policy_state_hash=(
                    cold.bootstrap_initial_policy_state_hash
                ),
            )

            reference = make_verified_stair_reference(
                output_dir / "reference",
                task_id=STAIR_TASK_ID,
                capability_profile_hash=(
                    cold.task_plugin.task_spec.candidate_capability_profile.profile_hash
                ),
            )
            provider = StairClimbingCompositionProvider(
                reference_program_artifact=reference
            )
            reference_composition = self._resolve_with_test_execution_environment(
                provider=provider,
                values=provider.cli_defaults(),
                context=TaskCompositionContext(
                    project_root=ROOT,
                    output_dir=output_dir,
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "Genesis identity drift|genesis kind mismatch",
            ):
                prepare_session_genesis(
                    output_dir=output_dir,
                    session_id=session_id,
                    task_plugin=reference_composition.task_plugin,
                    backend_adapter=reference_composition.backend_adapter,
                    runner=reference_composition.runner,
                    policy_initialization=(
                        reference_composition.policy_initialization
                    ),
                    genesis_identity_snapshot=(
                        reference_composition.genesis_identity_snapshot()
                    ),
                    reference_program_artifact=reference,
                    require_existing=True,
                )


if __name__ == "__main__":
    unittest.main()
