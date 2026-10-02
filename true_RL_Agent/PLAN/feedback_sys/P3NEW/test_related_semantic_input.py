"""Facts views preserve operands without leaking diagnostic or audit answers."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_confirmation_materials as packing


def info(arrays):
    return {name: {"dtype": value.dtype.str, "shape": list(value.shape),
                   "c_order_bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            for name, value in arrays.items()}


class RelatedSemanticInputTests(unittest.TestCase):
    def navigation_fixture(self):
        definitions = {"bodies": {"base": ["base"], "support_effector": ["FL", "FR", "RL", "RR"],
                                  "forbidden_contact": ["base"]},
            "arithmetic_queries": {"x_window_s": .5, "x_window_gain_m": .02,
                "unloaded_speed_mps": .05, "signed_vz_mps": .05},
            "protocol": {"episode_outcomes": {"signals": {
                "base_orientation_instability": {"threshold": .65},
                "slip": {"contact_force_threshold": 1., "speed_threshold": .5},
                "insufficient_clearance": {"threshold": .018},
                "forbidden_contact": {"force_threshold_n": 1.}}}}}
        return {"definitions": {"measurement_definitions": definitions},
            "predicate_order": [["unloaded_speed3d_le_query_mps", 4, 25]],
            "scope_row_columns": ["episode_id", "run_group", "cell", "steps", "dt_s", "original_success", "original_outcome", "missing_fields"],
            "scope_rows": [["e1", "r1", "c1", 30, .02, False, "forbidden_contact", []],
                           ["e2", "r1", "c1", 30, .02, True, "success", []],
                           ["e3", "r1", "c1", 30, .02, False, "forbidden_contact", ["query"]]],
            "operands_in_same_row_order": [
                [[[25, 1, 25, 5, 29, 1, True], [25, 1, 25, 6, 30, 1, True],
                  [10, 1, 10, 1, None, 0, False], None]],
                [[[0, 0, 0, None, None, 0, False] for _ in range(4)]],
                [[None for _ in range(4)]]]}

    def test_navigation_terminal_missing_and_zero_partitions(self):
        original = self.navigation_fixture()
        before = deepcopy(original)
        navigation = packing.stair_focus_arithmetic_navigation(original)
        self.assertEqual(original, before)
        components = navigation["predicate_components"]
        foot_indices = [i for i, v in enumerate(components) if v[0] == "unloaded_speed3d_le_query_mps"]
        counts = navigation["groups"][0][-1]
        self.assertEqual([counts[i] for i in foot_indices],
                         [[2, 1, 1, 0, 1], [2, 1, 0, 1, 1], [2, 1, 0, 0, 2], [1, 2, 0, 0, 1]])
        self.assertEqual(len(components), 37)
        self.assertEqual(counts[0], [0, 3, 0, 0, 0])  # absent root summary is not zero
        for group in navigation["groups"]:
            for supplied, missing, before_T, at_T, no_confirmation in group[-1]:
                self.assertEqual(supplied + missing, group[-2])
                self.assertEqual(before_T + at_T + no_confirmation, supplied)

    def test_navigation_original_outcome_denominators_and_body_identity(self):
        navigation = packing.stair_focus_arithmetic_navigation(self.navigation_fixture())
        groups = {tuple(v[:5]): v[-2] for v in navigation["groups"]}
        self.assertEqual(groups[("all_scope", None, None, None, None)], 3)
        self.assertEqual(groups[("all_scope_by_original_outcome", None, None, False, "forbidden_contact")], 2)
        self.assertEqual(groups[("run_cell_by_original_outcome", "r1", "c1", True, "success")], 1)
        feet = [v for v in navigation["predicate_components"] if v[0] == "unloaded_speed3d_le_query_mps"]
        self.assertEqual([v[3] for v in feet], [["FL"], ["FR"], ["RL"], ["RR"]])
        self.assertEqual([v[4] for v in feet], [25]*4)

    def test_navigation_never_substitutes_any_body_or_confirmation(self):
        original = self.navigation_fixture()
        original["predicate_order"] = [["loaded_planar_speed_gt_protocol_mps", 4, 5]]
        for operand in original["operands_in_same_row_order"]:
            operand[0] = [[4, 1, 4, 1, None, 0, True] for _ in range(4)]
            # Recorded OR streak confirms after alternating short body streaks.
            operand.append([[30, 1, 30, 1, 5, 1, True], [0, 0, 0, None, None, 0, False]])
        navigation = packing.stair_focus_arithmetic_navigation(original)
        for i, component in enumerate(navigation["predicate_components"]):
            if component[0] == "loaded_planar_speed_gt_protocol_mps":
                self.assertEqual(navigation["groups"][0][-1][i], [3, 0, 0, 0, 3])

    def build(self, task, raw, curves, measurements, terminal, static=None):
        header = {"episode": {"task": task, "steps": 3, "dt_s": .02,
                    "original_terminal_record": terminal, "direct_measurements": measurements},
                  "definitions": {"unchanged": "task source definitions"}, "static_values": static or {},
                  "raw_fields": info(raw), "direct_curve_fields": info(curves), "parts": [],
                  "original_values_npz": {"file": "original_values.npz"}}
        original = deepcopy(header)
        parts = [{"direct_curves": {k: v.tolist() for k, v in curves.items()}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original_values.npz"
            np.savez(path, **{"raw__"+k: v for k, v in raw.items()})
            answer = packing.related_episode(header, parts, path)
        self.assertEqual(header, original)
        return answer

    def test_stair_full_operands_and_two_direct_curves_with_explicit_reconstruction(self):
        raw = {"episode_length": np.arange(1, 4, dtype=np.int32),
               "base_pos_w": np.array([[1, 0, 0], [2, 0, 0], [1, 0, 0]], np.float32),
               "base_quat_w": np.array([[1, 0, 0, 0]]*3, np.float32),
               "support_pos_w": np.array([[[0, 0, 1]]]*3, np.float32),
               "support_lin_vel_w": np.array([[[3, 4, -2]]]*3, np.float32),
               "joint_pos": np.zeros((3, 1), np.float32), "valid": np.ones(3, bool)}
        curves = {"support_planar_speed_mps": np.full((3, 1), 5.),
                  "support_reference_clearance_m": np.full((3, 1), .75),
                  "support_surface_height_terrain_m": np.full((3, 1), .25),
                  "root_xyz_terrain_m": raw["base_pos_w"].astype(float),
                  "support_xyz_terrain_m": raw["support_pos_w"].astype(float),
                  "support_signed_vz_mps": np.full((3, 1), -2.),
                  "support_speed3d_mps": np.full((3, 1), np.sqrt(29)),
                  "root_x_drawdown_m": np.array([0., 0., 1.]),
                  "root_x_window_max_minus_start_m": np.full(3, np.nan)}
        measurements = {"threshold_intervals_by_component": {"test_query": [[[1, 1], [3, 3]]], "empty_query": [[]]},
                        "root_x_max_drawdown_witness": {"start_step": 2, "end_step": 3},
                        "diagnostic_label": "must not enter facts"}
        answer = self.build("stair", raw, curves, measurements,
            {"success": False, "episode_steps": 3, "diagnostic_flags": ["slip_detected"]},
            {"terrain_env_origins_w": [0, 0, 0]})
        self.assertEqual(answer["profile"], "related_v2")
        self.assertEqual(answer["direct_columns"]["support_planar_speed_mps"], packing.decimal_columns(curves["support_planar_speed_mps"]))
        self.assertEqual(answer["direct_columns"]["support_reference_clearance_m"], packing.decimal_columns(curves["support_reference_clearance_m"]))
        self.assertEqual(answer["raw_columns"]["support_lin_vel_w"], packing.decimal_columns(raw["support_lin_vel_w"]))
        self.assertEqual(len(answer["reconstructible_direct_fields"]), 6)
        self.assertEqual(answer["not_supplied_direct_fields"], [])
        self.assertEqual(answer["not_supplied_raw_fields"], ["joint_pos"])
        self.assertEqual(answer["direct_arithmetic"]["threshold_intervals_by_component"], measurements["threshold_intervals_by_component"])
        self.assertNotIn("diagnostic_label", answer["direct_arithmetic"])
        self.assertNotIn("diagnostic_flags", answer["episode"]["original_terminal_record"])
        self.assertIn("history-slot axis order remains in definitions/contact_history", answer["scope_limit"])
        self.assertIn("per-slot force values are not supplied", answer["scope_limit"])
        self.assertIn("observed per-slot timestamps and validity are not recorded", answer["scope_limit"])

    def test_unavailable_operand_is_not_declared_reconstructible(self):
        raw = {"episode_length": np.arange(1, 4, dtype=np.int32), "valid": np.ones(3, bool)}
        curves = {"root_xyz_terrain_m": np.zeros((3, 3)), "support_signed_vz_mps": np.zeros((3, 1))}
        answer = self.build("stair", raw, curves, {}, {"episode_steps": 3})
        self.assertEqual(answer["reconstructible_direct_fields"], {})
        self.assertEqual(answer["not_supplied_direct_fields"], sorted(curves))
        self.assertIn("support_lin_vel_w", answer["requested_but_not_recorded_fields"])

    def test_push_native_latch_and_conditions_survive_without_audit_verdict(self):
        raw = {"control_step": np.arange(1, 4, dtype=np.int32), "pose": np.zeros((3, 6), np.float32),
               "planar_motion": np.zeros((3, 4), np.float32),
               "conditions": np.array([[True]*5, [False]*5, [True]*5])}
        event = {"event_index": 0, "anchor_step": 1, "window_observed_steps": [2, 3],
                 "stable_intervals": [[3, 3]], "first_dwell_end_step_by_arithmetic": None}
        terminal = {"episode_steps": 3, "final_recovery_condition": True,
                    "push_events": [{"event_index": 0, "recovered": False, "recovery_control_step": None,
                                     "recovery_censored": True, "recovery_latency_s": 2.}]}
        answer = self.build("push", raw, {k: raw[k].astype(float) for k in ("pose", "planar_motion")},
            {"events": [event], "arithmetic_latch_equals_original_record": [True],
             "focus_raw_operands": {"duplicate_samples": True}}, terminal)
        self.assertEqual(answer["episode"]["original_terminal_record"], terminal)
        self.assertEqual(answer["direct_arithmetic"], {"events": [event]})
        self.assertEqual(answer["raw_discrete"]["conditions"], packing.discrete_runs(raw["conditions"], raw["control_step"]))
        self.assertEqual(set(answer["reconstructible_direct_fields"]), {"pose", "planar_motion"})
        self.assertEqual(answer["not_supplied_direct_fields"], [])

    def test_original_and_focus_paths_apply_same_fact_boundary_without_mutation(self):
        row = {"task": "stair", "original_terminal_record": {"success": False, "diagnostic_flags": ["tilt"]},
               "direct_measurements": {"curve_extrema": {}, "unreviewed_diagnosis": "tilt"}}
        original = {"episode": row, "all_scope_rows": [row], "raw_fields": {"diagnostic_named_raw_measurement": {"dtype": "bool"}}}
        before = deepcopy(original)
        projected = packing.facts_original_material("stair", original)
        self.assertEqual(original, before)
        self.assertEqual(projected["episode"], projected["all_scope_rows"][0])
        self.assertEqual(projected["episode"]["original_terminal_record"], {"success": False})
        self.assertEqual(projected["episode"]["direct_measurements"], {"curve_extrema": {}})
        self.assertEqual(projected["raw_fields"], original["raw_fields"])
        push = {"task": "push", "direct_measurements": {"events": [], "arithmetic_latch_equals_original_record": [True]}}
        self.assertEqual(packing.facts_episode_projection("push", push)["direct_measurements"], {"events": []})

    def test_focus_missing_conditions_keeps_raw_event_without_diagnosis(self):
        event = {"event_index": 0, "injection_verified": False, "recovery_control_step": None,
                 "recovery_censored": True, "diagnostic_flags": ["unreviewed"]}
        row = {"episode_id": "e1", "run_group": "r1", "cell": "cell", "steps": 3, "dt_s": .02,
               "original_success": False, "original_outcome": "not_recovered", "missing_fields": ["conditions"],
               "original_terminal_record": {"push_events": [event]},
               "direct_measurements": {"focus_raw_operands": {"conditions_rle": None}}}
        focus = {"task": "push", "focus": "cell", "questions": [], "task_definitions": {},
                 "complete_task_pool_members": [], "scope_denominator": 1, "all_scope_rows": [row]}
        before = deepcopy(focus)
        answer = packing.related_focus(focus)
        self.assertEqual(answer["operands_in_same_row_order"][0],
                         {"not_recorded": "conditions", "original_events": [{k: v for k, v in event.items() if k != "diagnostic_flags"}]})
        self.assertEqual(focus, before)


if __name__ == "__main__":
    unittest.main()
