"""Matrix evaluation identity and morphology scaling, independent of Isaac."""
import math
import unittest
from dataclasses import replace

from true_rl_agent.tasks.push_recovery.config import (
    PUSH_ACTIVE_CELLS, PUSH_FORCE_EVAL_BASE_SEED, PushRecoveryTaskConfig,
)
from true_rl_agent.tasks.push_recovery.force_protocol import ForceEvaluationProtocol
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest


class PushForceProtocolTests(unittest.TestCase):
    def test_common_task_body_scaled_pulse_fixed_gap_and_no_spatial_gate(self):
        p = ForceEvaluationProtocol()
        self.assertEqual(p, ForceEvaluationProtocol.from_mapping(p.to_dict()))
        small, large = p.resolve(0.4), p.resolve(0.74)
        for resolved, pulse_steps in ((small, 11), (large, 14)):
            self.assertEqual(math.ceil(resolved["pulse_duration_s"] / 0.02), pulse_steps)
            self.assertEqual(resolved["repeated_gap_s"], 0.22)
            self.assertEqual(resolved["repeated_high_fraction"], 0.8)
            self.assertIsNone(resolved["max_displacement_m"])
            self.assertAlmostEqual(resolved["impulse_velocity_mps"]["high"], 2 * math.sqrt(9.81 * resolved["robot_scale_m"]))
        self.assertAlmostEqual(large["recovery_height_m"] / small["recovery_height_m"], 0.74 / 0.4)
        self.assertNotEqual(small["resolved_protocol_hash"], large["resolved_protocol_hash"])

    def test_evaluation_changes_identity_without_changing_training(self):
        for cell in PUSH_ACTIVE_CELLS:
            with self.subTest(cell=cell):
                config = PushRecoveryTaskConfig.for_cell(cell)
                old = PushRecoveryTaskConfig.for_cell(cell.replace("v3", "v2"),
                    push_protocol=config.push_protocol, reference_program_id=config.reference_program_id)
                self.assertEqual(config.train_runtime_dynamics(), old.train_runtime_dynamics())
                self.assertNotEqual(config.eval_runtime_dynamics(), old.eval_runtime_dynamics())
                self.assertEqual(config.eval_base_seed, PUSH_FORCE_EVAL_BASE_SEED)
                changed = replace(config, push_evaluation_protocol=replace(config.push_evaluation_protocol, repeated_gap_s=0.24))
                self.assertEqual(changed.train_runtime_dynamics(), config.train_runtime_dynamics())
                self.assertNotEqual(changed.eval_runtime_dynamics(), config.eval_runtime_dynamics())
                manifests = []
                for c in (config, changed):
                    manifests.append(resolve_push_cell_manifest(cell_variant_id=cell, config=c, plugin=PushRecoveryTaskPlugin(c)))
                self.assertNotEqual(manifests[0].manifest_hash, manifests[1].manifest_hash)
                restored = PushRecoveryTaskConfig.from_manifest(manifests[0])
                self.assertEqual(restored.push_evaluation_protocol, config.push_evaluation_protocol)
                self.assertEqual(restored.train_runtime_dynamics(), config.train_runtime_dynamics())

    def test_old_rows_cannot_be_silently_relabelled(self):
        with self.assertRaises(ValueError):
            PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2", push_evaluation_protocol=ForceEvaluationProtocol())


if __name__ == "__main__":
    unittest.main()
