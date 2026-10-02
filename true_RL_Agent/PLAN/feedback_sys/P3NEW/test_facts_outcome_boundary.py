"""Facts terminal projection preserves native records without process diagnoses."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "facts_outcome_raw_export", Path(__file__).with_name("export_confirmation_raw.py")
)
raw_export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(raw_export)
project = raw_export.facts_episode_projection


class FactsOutcomeBoundaryTests(unittest.TestCase):
    def test_stair_keeps_native_outcome_and_measurements_but_not_diagnostics(self):
        terminal = {"episode_steps": 606, "primary_reason": "forbidden_contact", "success": False,
                    "diagnostic_flags": ["base_orientation_instability_detected", "slip_detected"],
                    "max_progress_m": 3.45792627, "progress_fraction": 0.74839761,
                    "future_process_labels": {"orientation": True}}
        row = {"task": "stair", "episode_id": "stair-example", "original_success": False,
               "original_outcome": "forbidden_contact", "original_terminal_record": terminal,
               "direct_measurements": {"raw_diagnostic_sensor_value": [0.0, 1.25]},
               "missing_fields": []}
        before = deepcopy(row)
        result = project("stair", row)
        self.assertEqual(result["original_terminal_record"], {
            "episode_steps": 606, "primary_reason": "forbidden_contact", "success": False,
            "max_progress_m": 3.45792627, "progress_fraction": 0.74839761,
        })
        self.assertEqual(result["direct_measurements"], row["direct_measurements"])
        self.assertEqual(result["original_outcome"], "forbidden_contact")
        result["direct_measurements"]["raw_diagnostic_sensor_value"][0] = 99
        result["original_terminal_record"]["episode_steps"] = 1
        self.assertEqual(row, before)

    def test_push_preserves_injection_measurements_and_native_latch_state(self):
        event = {"event_index": 0, "control_step": 69, "time_s": 1.38,
                 "requested_delta_v_world_mps": [0.006, 0.606],
                 "observed_delta_v_world_mps": [0.006, 0.606],
                 "velocity_before_world_6d": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                 "velocity_after_write_world_6d": [0.016, 0.626, 0.03, 0.04, 0.05, 0.06],
                 "velocity_after_next_physics_step_world_6d": None,
                 "injection_verified": True, "recovered": False, "recovery_control_step": None,
                 "recovery_latency_s": 2.0, "recovery_censored": True}
        terminal = {"episode_steps": 412, "stratum": "repeated", "control_only": False,
                    "planned_push_count": 2, "push_events": [event, {**event, "event_index": 1}],
                    "min_base_height_m": 0.67, "max_pitch_rad": 0.09, "min_pitch_rad": -0.03,
                    "max_abs_roll_rad": 0.12, "max_abs_pitch_rad": 0.09, "max_xy_drift_m": 0.86,
                    "final_planar_speed_mps": 0.24, "final_recovery_condition": False,
                    "success": False, "control_success": False, "control_failure_label": None,
                    "push_event_count": 2, "recovered_event_count": 0, "invalid_injection_count": 0,
                    "failure_label": "not_recovered", "mean_action_l2": 5.62, "mean_action_rate_l2": 1.0}
        expected = deepcopy(terminal)
        terminal["diagnostic_flags"] = ["production_process_label"]
        terminal["push_events"][0]["process_diagnosis"] = {"regained_stability": False}
        terminal["push_events"][1]["diagnostic_flags"] = ["another_label"]
        row = {"task": "push", "original_terminal_record": terminal}
        before = deepcopy(row)
        result = project("push", row)
        self.assertEqual(result["original_terminal_record"], expected)
        result["original_terminal_record"]["push_events"][0]["observed_delta_v_world_mps"][0] = 99
        self.assertEqual(row, before)

    def test_missing_fields_are_not_filled_and_header_without_terminal_is_copied(self):
        row = {"episode_id": "push-partial", "original_terminal_record": {"episode_steps": 2}}
        self.assertEqual(project("push", row), row)
        header = {"task": "stair", "missing_fields": ["joint_pos"]}
        result = project("stair", header)
        self.assertEqual(result, header)
        result["missing_fields"].clear()
        self.assertEqual(header["missing_fields"], ["joint_pos"])

    def test_unknown_task_and_malformed_terminal_cannot_silently_bypass_projection(self):
        with self.assertRaises(ValueError):
            project("new_task", {})
        with self.assertRaises(ValueError):
            project("push", {"task": "stair"})
        with self.assertRaises(TypeError):
            project("stair", {"original_terminal_record": None})
        with self.assertRaises(TypeError):
            project("push", {"original_terminal_record": {"push_events": [{}, "not-an-event"]}})


if __name__ == "__main__":
    unittest.main()
