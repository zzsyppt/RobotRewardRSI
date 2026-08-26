from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from true_rl_agent.tasks.stair_climbing.generation_session import (
    STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME,
    StairGenerationPilotConfig,
    run_stair_generation_pilot,
)


SHA = "a" * 40
GPU_A = "GPU-11111111-1111-1111-1111-111111111111"
GPU_B = "GPU-22222222-2222-2222-2222-222222222222"


class StairP6BLegacyResumeTests(unittest.TestCase):
    def test_existing_p6b_report_replays_without_new_source_identity_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output_dir = root / "p6b-output"
            output_dir.mkdir()
            for filename in (
                "generation_session_cost.json",
                "generation_session_contract.json",
                "generation_continuation_policy.json",
                "session_genesis.json",
                "generation_session_report.json",
            ):
                (output_dir / filename).write_text("{}\n", encoding="utf-8")
            source_identity_path = output_dir / STAIR_GENERATION_SOURCE_CHECKOUT_FILENAME
            self.assertFalse(source_identity_path.exists())

            config = StairGenerationPilotConfig(
                project_root=Path(__file__).resolve().parents[2],
                output_dir=output_dir,
                lease_root=root / "leases",
                session_id="historical-p6b",
                expected_source_commit=SHA,
                machine_id="test-machine",
                gpu0_uuid=GPU_A,
                gpu1_uuid=GPU_B,
                provider_env_file=root / "unused-provider.env",
                max_allocated_active_gpu_hours=12.0,
                max_simulated_control_steps=1_000_000_000,
                max_output_bytes=40 * 1024**3,
                resume_existing=True,
            )
            composition = Mock()
            composition.genesis_identity_snapshot.return_value = {}
            foundation = SimpleNamespace(
                config=config,
                composition=composition,
                runner=Mock(),
                session_contract=SimpleNamespace(contract_hash="b" * 64),
                session_contract_ref=Mock(),
                cost_contract=SimpleNamespace(contract_hash="c" * 64),
                gpu_bindings=(
                    SimpleNamespace(resource_binding_hash="d" * 64),
                    SimpleNamespace(resource_binding_hash="e" * 64),
                ),
            )
            genesis = Mock()
            cost_journal = Mock()
            expected = {"schema_version": "historical.p6b.report.v1"}

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_reject_parent_gpu_remap"
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_verify_source_checkout",
                    return_value=SHA,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_publish_or_verify_source_checkout_identity"
                ) as publish_source_identity,
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_generation_provider_readiness_required",
                    return_value=False,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "discover_two_gpu_inventory",
                    return_value=(GPU_A, GPU_B),
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_create_foundation",
                    return_value=foundation,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_assert_target_gpus_compute_idle",
                    return_value={},
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "StairGenerationActualCostJournal",
                    return_value=cost_journal,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "prepare_session_genesis",
                    return_value=genesis,
                ),
                patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "_load_and_validate_existing_report",
                    return_value=expected,
                ) as load_report,
            ):
                actual = run_stair_generation_pilot(config)

            self.assertEqual(actual, expected)
            load_report.assert_called_once()
            publish_source_identity.assert_not_called()
            self.assertFalse(source_identity_path.exists())


if __name__ == "__main__":
    unittest.main()
