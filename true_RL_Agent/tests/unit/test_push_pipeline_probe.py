"""P6 diagnostic jobs retain production lineage and a public-dev-only phase."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location("push_pipeline_probe", ROOT / "PLAN/2new_world/P6/push_pipeline_probe.py")
PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PROBE)

from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.runner.subprocess_runner import SubprocessRunner


class PushPipelineProbeTests(unittest.TestCase):
    def test_both_cells_preflight_real_diagnostic_jobs_without_spending_budget(self):
        for cell in PROBE.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "probe"
                with patch.object(SubprocessRunner, "run_train", side_effect=AssertionError("trained")), \
                     patch.object(SubprocessRunner, "run_eval_phase", side_effect=AssertionError("evaluated")), \
                     patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment", side_effect=AssertionError("runtime queried")), \
                     redirect_stdout(io.StringIO()):
                    code = PROBE.main(["--cell", cell, "--output", str(output), "--device", "cuda:1", "--mode", "preflight"])
                self.assertEqual(code, 0)
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["budget"]["train_control_steps"], 3072)
                self.assertEqual(result["budget"]["train_seed"], 731)
                self.assertEqual(result["budget"]["eval_base_seed"],
                                 1500000000)
                self.assertEqual(result["budget"]["dev_seed_count"], 1)
                self.assertEqual(result["budget"]["eval_num_envs"], 64)
                self.assertEqual(result["phases_executed"], [])
                self.assertFalse(result["reference_qualification_performed"])
                self.assertFalse((output / "session_genesis.json").exists())
                self.assertFalse((output / "public_dev_job.json").exists())
                self.assertTrue((output / "train_job.json").is_file())

    def test_bound_checkpoint_materializes_only_dev_and_rejects_lineage_drift(self):
        cases = [(cell, None) for cell in PROBE.CELLS] + [("push.g1_37dof.recovery_v3", 42)]
        for cell, explicit_seed in cases:
            with self.subTest(cell=cell, explicit_seed=explicit_seed), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                composition = PROBE.build_composition(cell=cell, output=output,
                    device="cuda:1", isaac_python=sys.executable, timeout_s=1800)
                if explicit_seed is not None:
                    default_manifest_hash = composition.resolved_cell_manifest.manifest_hash
                    composition = PROBE.PushRecoveryCompositionProvider().resolve(
                        values={**composition.task_defaults, "eval_base_seed": explicit_seed},
                        context=PROBE.TaskCompositionContext(project_root=ROOT, output_dir=output / "explicit"))
                    self.assertNotEqual(composition.resolved_cell_manifest.manifest_hash, default_manifest_hash)
                expected_seed = explicit_seed if explicit_seed is not None else 1500000000
                self.assertEqual(composition.task_plugin.config.eval_base_seed, expected_seed)
                self.assertEqual(composition.backend_adapter.config.eval_base_seed, expected_seed)
                jobs = PROBE.materialize_jobs(composition, output=output)
                plan = jobs.eval_spec
                self.assertEqual(jobs.train_spec.budget_steps, 3072)
                self.assertEqual(jobs.train_spec.training_binding.task_protocol_hash,
                                 composition.resolved_cell_manifest.resolved_cell_task_protocol_hash)
                self.assertFalse(composition.resolved_cell_manifest.uses_legacy_training_binding_v1)
                # These explicit fixture bytes exercise materialization only;
                # they are never supplied to an execution runner or published as training evidence.
                fixture = output / "unit-test-checkpoint.bin"
                fixture.write_bytes(b"unit-test lineage fixture")
                lineage = {"training_subject": plan.subject.to_public_dict(),
                    "subject_manifest_hash": plan.subject_manifest_hash,
                    "program_source_checksum": plan.program_source_checksum,
                    "training_binding_hash": plan.training_binding_hash,
                    "capability_profile_hash": plan.capability_profile_hash,
                    "execution_device": plan.execution_device, "local_path": str(fixture),
                    "resolved_cell_manifest_hash": composition.resolved_cell_manifest.manifest_hash}
                checkpoint = CheckpointRef(uri=plan.checkpoint_output_plan.uri,
                    checksum=PROBE.file_sha256(fixture), role="eval", sealed=True, metadata=lineage)
                phase = PROBE.materialize_public_dev(composition, jobs, checkpoint)
                self.assertEqual(phase.phase.value, "public_dev")
                argv = phase.metadata["cli_argv"]
                self.assertEqual(argv[argv.index("--evaluation-phase") + 1], "public_dev")
                self.assertEqual(argv[argv.index("--public-seeds") + 1], "1")
                self.assertEqual(argv[argv.index("--seed") + 1], str(expected_seed))
                self.assertEqual(argv[argv.index("--checkpoint-sha256") + 1], checkpoint.checksum)
                self.assertIsNone(phase.attempt_scope)
                self.assertIsNone(phase.gpu_process_binding)
                wrong = replace(checkpoint, metadata={**lineage, "training_binding_hash": "f" * 64})
                with self.assertRaisesRegex(ValueError, "lineage mismatch"):
                    PROBE.materialize_public_dev(composition, jobs, wrong)


if __name__ == "__main__":
    unittest.main()
