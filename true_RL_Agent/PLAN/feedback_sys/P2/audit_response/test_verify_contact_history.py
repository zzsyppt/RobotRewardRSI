"""CPU fixtures only: no simulator or real experiment claims."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("verify_contact_history", Path(__file__).with_name("verify_contact_history.py"))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

from tests.unit.test_stair_numeric_capture import _TwoStepEnv
from true_rl_agent.runner.numeric_evidence import NumericEvidenceWriter, iter_numeric_evidence, load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.numeric_capture import StairNumericCapture, STAIR_NUMERIC_PROFILE_V1
from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FORMAL_EVAL_NAMESPACE
from true_rl_agent.tasks.stair_climbing.evaluation_runtime import run_stair_split_execution
from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_stair_summary


class ContactHistoryVerifierTests(unittest.TestCase):
    def test_recorded_exact_predicate_mismatch_and_missing_history(self):
        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json")
        manifest = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id, config=config, plugin=StairClimbingTaskPlugin(config))
        config = replace(config, resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash)
        projection = stair_runtime_projection_from_manifest(manifest)
        cells = config.ordered_evaluation_suite.private_cell_assignment("public_dev")
        with tempfile.TemporaryDirectory() as temporary:
            capture = StairNumericCapture(Path(temporary) / "on", identity={"source": "CPU fixture"}, config=config, resolved_cell_manifest=manifest)
            execution = run_stair_split_execution(env=_TwoStepEnv(cells, projection, config), inference_policy=lambda obs: obs,
                protocol=None, semantics=config.evaluation_semantics, seed_resolver=config.seed_resolver(), split="public_dev",
                seed_count=config.public_seed_count, seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE, runtime_projection=projection,
                numeric_capture=capture)
            summary = build_unified_stair_summary(suite=config.ordered_evaluation_suite, semantics=config.evaluation_semantics,
                split="public_dev", per_seed_outcomes=execution.per_seed_outcomes)
            ref = capture.complete(summary)
            report = verifier.verify_contact_history(ref)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["counts"]["predicate_mismatches"], 0)
            self.assertEqual(report["counts"]["manager_contact_terminal_frames"], 864)
            self.assertEqual(report["counts"]["instantaneous_zero_history_nonzero_terminal_frames"], 864)
            self.assertEqual(report["counts"]["valid_control_frames"], 1296)
            original = load_numeric_evidence_manifest(ref)
            mismatch = NumericEvidenceWriter(Path(temporary) / "mismatch", original["identity"])
            for metadata, arrays in iter_numeric_evidence(ref):
                arrays["forbidden_contact_forces_w_history"][:] = 0
                mismatch.write_chunk(arrays, metadata)
            mismatch_report = verifier.verify_contact_history(mismatch.complete(original["metadata"]))
            self.assertEqual(mismatch_report["status"], "FAIL")
            self.assertEqual(mismatch_report["counts"]["predicate_mismatches"], 864)
            legacy = NumericEvidenceWriter(Path(temporary) / "legacy", original["identity"])
            for metadata, arrays in iter_numeric_evidence(ref):
                header = dict(metadata["transition_header"])
                header.pop("contact_force_history")
                legacy.write_chunk({k: v for k, v in arrays.items() if not k.endswith("forces_w_history")}, {**metadata, "transition_header": header})
            old_context = {**original["metadata"], "profile": STAIR_NUMERIC_PROFILE_V1}
            old_context.pop("contact_force_history")
            self.assertEqual(verifier.verify_contact_history(legacy.complete(old_context))["status"], "NOT_RECORDED")


if __name__ == "__main__":
    unittest.main()
