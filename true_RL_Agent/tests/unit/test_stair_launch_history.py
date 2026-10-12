"""Historical subject reconstruction retains the selected robot's frozen cell."""
import tempfile
import unittest
from pathlib import Path
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.row_catalog import STAIR_GO2_12DOF_UNIFIED_V2_CELL_VARIANT_ID
from true_rl_agent.tasks.stair_climbing.launch_history import _resolved_formal_task_config

_ROOT = Path(__file__).resolve().parents[2]


class StairLaunchHistoryTests(unittest.TestCase):
    def test_formal_subject_rebuilds_selected_go2_task_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            output_dir = (Path(raw_root) / "formal-output").resolve()
            provider = StairClimbingCompositionProvider()
            values = dict(provider.cli_defaults())
            values["cell_variant_id"] = STAIR_GO2_12DOF_UNIFIED_V2_CELL_VARIANT_ID
            resolved = resolve_task_composition(
                provider=provider,
                values=values,
                context=TaskCompositionContext(
                    project_root=_ROOT,
                    output_dir=output_dir,
                ),
            )

            config = _resolved_formal_task_config(
                output_dir=output_dir,
                project_root=_ROOT,
            )
            self.assertEqual(
                config.cell_variant_id,
                STAIR_GO2_12DOF_UNIFIED_V2_CELL_VARIANT_ID,
            )
            self.assertEqual(
                config.task_protocol_hash,
                resolved.resolved_cell_manifest.resolved_cell_task_protocol_hash,
            )
            self.assertEqual(
                config.ordered_evaluation_suite.suite_hash,
                resolved.task_plugin.config.ordered_evaluation_suite.suite_hash,
            )

