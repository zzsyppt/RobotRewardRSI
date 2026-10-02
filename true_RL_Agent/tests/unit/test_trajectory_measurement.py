"""Independent numerical gates for the enabled trajectory descriptor catalog."""
from __future__ import annotations

import json
import math
import unittest

import numpy as np

from true_rl_agent.evaluation.trajectory_translation.contracts import NumericEpisode
from true_rl_agent.evaluation.trajectory_translation.measurement import (
    aggregate_fractions,
    measure_episode,
)


FEET = tuple(f"robot_0.support.{name}" for name in ("FL", "FR", "RL", "RR"))
ROOT = "robot_0.root"
BASE = "robot_0.body.base"
JOINTS = tuple(f"{foot}_{part}_joint" for part in ("hip", "thigh", "calf")
               for foot in ("FL", "FR", "RL", "RR"))


def sample_episode(n=4, *, steps=None, dt=0.02, support=FEET, joints=JOINTS):
    steps = np.arange(1, n + 1) if steps is None else np.asarray(steps)
    n = len(steps)
    count = len(support)
    arrays = {
        "base_pos_w": np.zeros((n, 3)),
        "base_quat_w": np.tile([1.0, 0, 0, 0], (n, 1)),
        "support_pos_w": np.zeros((n, count, 3)),
        "support_lin_vel_w": np.zeros((n, count, 3)),
        "support_contact_forces_w": np.zeros((n, count, 3)),
        "support_contact_forces_w_history": np.zeros((n, 3, count, 3)),
        "forbidden_contact_forces_w": np.zeros((n, 1, 3)),
        "forbidden_contact_forces_w_history": np.zeros((n, 3, 1, 3)),
        "joint_pos": np.zeros((n, len(joints))),
        "done_mask": np.zeros(n, dtype=bool),
        "wrapper_done_mask": np.zeros(n, dtype=bool),
        "termination_0": np.zeros(n, dtype=bool),
        "termination_1": np.zeros(n, dtype=bool),
    }
    arrays["support_contact_forces_w"][:, :, 2] = 2.0
    return NumericEpisode(
        source={"acceptance_split": "public_dev", "manifest_sha256": "a" * 64,
                "trial_id": "synthetic-measurement", "checkpoint_hash": "b" * 64,
                "contact_force_history": {"history_length": 3}},
        run_index=1, env_index=0, steps=steps, dt_s=dt, arrays=arrays,
        entities={"root": (ROOT,), "support": tuple(support), "forbidden": (BASE,), "joints": tuple(joints)},
        source_chunks=({"file": "fixture.npz", "sha256": "c" * 64},),
        chunk_indices=np.zeros(n, dtype=np.int64), chunk_rows=steps - 1,
        task={"contact_threshold_n": 1.0, "origin_world": [0, 0, 0],
              "progress_direction_world": [1, 0, 0], "goal_progress_m": 4.0,
              "deadline_s": 20.0, "termination_term_order": ["time_out", "base_contact"],
              "outcome": {"success": True, "primary_reason": "success", "episode_steps": int(steps[-1])}},
    )


def measurements(records):
    return {record["key"]: record["value"] for record in records if record["record_type"] == "measurement"}


def active_intervals(value):
    return [row for row in value["intervals"] if row["state"] == "true"]


class TrajectoryMeasurementTests(unittest.TestCase):
    def test_fz_component_keeps_its_own_valid_denominator(self):
        sample = sample_episode(2)
        sample.arrays["support_contact_forces_w"][0, 0, 0] = np.nan
        data = measurements(measure_episode(sample))
        self.assertEqual(data[f"{FEET[0]}.marker"]["unknown_count"], 1)
        force = data["force_summary"]
        self.assertEqual(force["joint_valid_count"], 1)
        self.assertEqual(force["signed_fz_joint_valid_count"], 2)
        self.assertEqual(force["sample_sum_signed_fz_n"], [4, 4, 4, 4])
        self.assertEqual(force["per_entity"][0]["signed_fz_n"]["valid_count"], 2)

    def test_task_projection_and_paths_are_distinct_from_net_displacement(self):
        sample = sample_episode()
        sample.task.update(origin_world=[10, 20, 0], progress_direction_world=[0, 1, 0])
        sample.arrays["base_pos_w"][:] = [[10, 20, 0], [11, 21, 0], [10, 22, 0], [10, 23, 0]]
        root = measurements(measure_episode(sample))["root_progress"]
        self.assertEqual(root["net_world_xyz_m"], [0, 3, 0])
        self.assertEqual(root["net_progress_m"], 3)
        self.assertEqual(root["progress"]["first"], 0)
        self.assertEqual(root["progress"]["last"], 3)
        self.assertAlmostEqual(root["sampled_path_xy_m"], 2 * math.sqrt(2) + 1)
        self.assertAlmostEqual(root["sampled_path_xyz_m"], 2 * math.sqrt(2) + 1)
        self.assertEqual(root["positive_progress_variation_m"], 3)
        self.assertEqual(root["negative_progress_variation_m"], 0)
        self.assertTrue(root["path_complete"])

    def test_backtracking_and_return_do_not_reassign_success(self):
        sample = sample_episode()
        sample.arrays["base_pos_w"][:, 0] = [0, 2, 1, 2]
        data = measurements(measure_episode(sample))
        root = data["root_progress"]
        self.assertEqual(root["sampled_path_xy_m"], 4)
        self.assertEqual(root["net_progress_m"], 2)
        self.assertEqual(root["positive_progress_variation_m"], 3)
        self.assertEqual(root["negative_progress_variation_m"], 1)
        self.assertEqual(root["drawdown"]["max_drawdown_m"], 1)
        self.assertEqual((root["drawdown"]["peak_step"], root["drawdown"]["trough_step"], root["drawdown"]["return_step"]), (2, 3, 4))
        self.assertFalse(root["drawdown"]["return_observation_right_censored"])
        self.assertEqual(data["task_endpoint"]["outcome"], sample.task["outcome"])
        self.assertEqual(data["task_endpoint"]["authority"], "archived_task_evaluator")

    def test_paths_differences_and_drawdown_search_do_not_cross_missing_step(self):
        sample = sample_episode(steps=[1, 2, 4, 5])
        sample.arrays["base_pos_w"][:, 0] = [0, 1, 100, 99]
        root = measurements(measure_episode(sample))["root_progress"]
        self.assertEqual(root["sampled_path_xy_m"], 2)
        self.assertEqual(root["valid_edge_count"], 2)
        self.assertFalse(root["path_complete"])
        self.assertEqual(root["positive_progress_variation_m"], 1)
        self.assertEqual(root["negative_progress_variation_m"], 1)
        self.assertEqual(root["drawdown"]["peak_step"], 4)
        self.assertEqual(root["drawdown"]["trough_step"], 5)
        self.assertIsNone(root["drawdown"]["return_step"])
        self.assertFalse(root["drawdown"]["cross_gap_search"])

    def test_low_progress_requires_full_point03_window_and_records_confirmation(self):
        sample = sample_episode(18, dt=0.03)
        records = measure_episode(sample)
        low = measurements(records)["low_progress"]
        self.assertEqual((low["true_count"], low["false_count"], low["unknown_count"]), (1, 0, 17))
        only = active_intervals(low)[0]
        self.assertEqual(only["steps_half_open"], [18, 19])
        self.assertAlmostEqual(only["confirmation_time_s"], 0.54)
        low_record = next(r for r in records if r["record_type"] == "measurement" and r["key"] == "low_progress")
        definition = low_record["measurement_config"]["definition"]
        self.assertEqual(definition["lookback_samples"], 17)
        self.assertAlmostEqual(definition["actual_window_span_s"], 0.51)
        state = next(r for r in records if r["record_type"] == "state_interval"
                     and r["key"] == "low_progress" and r["value"]["state"] == "true")
        self.assertEqual(state["evidence_ref"]["steps_half_open"], [1, 19])
        self.assertEqual(state["evidence_ref"]["chunks"][0]["rows_half_open"], [0, 18])
        shorter = measurements(measure_episode(sample_episode(17, dt=0.03)))["low_progress"]
        self.assertEqual((shorter["valid_count"], shorter["unknown_count"]), (0, 17))

    def test_low_progress_strict_threshold_and_motion_are_separate(self):
        equal = sample_episode(26)
        equal.arrays["base_pos_w"][-1, 0] = 0.02
        at_boundary = measurements(measure_episode(equal))["low_progress"]
        self.assertEqual((at_boundary["true_count"], at_boundary["false_count"]), (0, 1))
        retreat = sample_episode(26)
        retreat.arrays["base_pos_w"][:, 0] = np.linspace(0, -1, 26)
        data = measurements(measure_episode(retreat))
        self.assertEqual(data["low_progress"]["true_count"], 1)
        self.assertEqual(data["root_progress"]["sampled_path_xy_m"], 1)
        self.assertEqual(data["root_progress"]["net_progress_m"], -1)
        gap = sample_episode(steps=[*range(1, 14), *range(15, 28)])
        self.assertEqual(measurements(measure_episode(gap))["low_progress"]["valid_count"], 0)

    def test_normalized_quaternion_controls_primary_sign_raw_is_separate(self):
        sample = sample_episode(2)
        angle = math.acos(1e-5)
        unit = np.array([math.cos(angle / 2), math.sin(angle / 2), 0, 0])
        sample.arrays["base_quat_w"][:] = unit * 1.00004
        sample.arrays["support_pos_w"][:, :, 1] = 1
        data = measurements(measure_episode(sample))
        up = data["body_up"]
        self.assertEqual(up["negative"]["true_count"], 0)
        self.assertAlmostEqual(up["up_z"]["first"], 1e-5, places=12)
        self.assertLess(up["raw_body_up_diagnostic"]["first"], 0)
        self.assertGreater(up["raw_vs_normalized_up_z_max_abs_delta"], 1e-5)
        np.testing.assert_allclose(data[f"{FEET[0]}.relative_position"]["first_body_xyz_m"],
                                   [0, math.cos(angle), -math.sin(angle)], atol=1e-12)

    def test_invalid_quaternion_only_blocks_its_dependent_measurements(self):
        sample = sample_episode(3)
        sample.arrays["base_quat_w"][1] = [1.0002, 0, 0, 0]
        data = measurements(measure_episode(sample))
        self.assertEqual(data["body_up"]["up_z"]["valid_count"], 2)
        self.assertEqual(data["body_up"]["negative"]["unknown_count"], 1)
        self.assertEqual(data[f"{FEET[0]}.relative_position"]["body_xyz_m"][0]["unknown_count"], 1)
        self.assertEqual(data["root_progress"]["valid_count"], 3)
        self.assertEqual(data[f"{FEET[0]}.marker"]["true_count"], 3)

    def test_strict_force_and_planar_speed_boundaries_discriminate_xyz(self):
        sample = sample_episode(4)
        sample.arrays["support_contact_forces_w"][:, 0, 2] = [0, 1, np.nextafter(1.0, np.inf), 0]
        sample.arrays["support_lin_vel_w"][:, 0] = [0.03, 0, 0.08]
        sample.arrays["support_lin_vel_w"][3, 0] = [0.05, 0, 0.08]
        data = measurements(measure_episode(sample))
        marker = data[f"{FEET[0]}.marker"]
        self.assertEqual((marker["true_count"], marker["false_count"]), (1, 3))
        low = data[f"{FEET[0]}.low_speed_off"]
        self.assertEqual((low["true_count"], low["false_count"]), (2, 2))
        self.assertEqual(low["longest"]["interval"]["steps_half_open"], [1, 3])
        self.assertAlmostEqual(low["longest"]["max_speed_xy_mps"], 0.03)
        self.assertGreater(low["longest"]["max_speed_xyz_mps"], 0.05)
        sample.arrays["support_contact_forces_w"][:, 0] = 0
        sample.arrays["support_lin_vel_w"][:, 0] = [0.05, 0, 0.08]
        equal = measurements(measure_episode(sample))[f"{FEET[0]}.low_speed_off"]
        self.assertEqual(equal["true_count"], 0)
        self.assertIsNone(equal["longest"])

    def test_planar_predicate_does_not_require_vertical_velocity_or_attitude(self):
        sample = sample_episode(3)
        sample.arrays["support_contact_forces_w"][:, 0] = 0
        sample.arrays["support_lin_vel_w"][:, 0] = [0.03, 0, np.nan]
        del sample.arrays["base_quat_w"]
        data = measurements(measure_episode(sample))
        low = data[f"{FEET[0]}.low_speed_off"]
        self.assertEqual((low["true_count"], low["unknown_count"]), (3, 0))
        self.assertIsNone(low["longest"]["max_speed_xyz_mps"])
        self.assertIsNone(low["longest"]["minimum_up_z"])
        self.assertEqual(low["longest"]["orientation_valid_count"], 0)

    def test_all_sixteen_contact_combinations_have_frozen_lsb_role_encoding(self):
        table = np.array([
            [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0],
            [0, 0, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0],
            [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1],
            [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]], dtype=bool)
        sample = sample_episode(16)
        sample.arrays["support_contact_forces_w"][:, :, 2] = table * 2
        value = measurements(measure_episode(sample))["contact_combinations"]
        self.assertEqual(value["encoding_version"], "role_index_lsb.v1")
        self.assertEqual(value["bit_order"], list(FEET))
        self.assertEqual(value["bit_weights"], [1, 2, 4, 8])
        self.assertEqual(value["counts"], [1] * 16)
        self.assertEqual([(c["step"], c["before"], c["after"]) for c in value["transitions"]],
                         [(step, step - 2, step - 1) for step in range(2, 17)])
        self.assertEqual(table[11].tolist(), [True, True, False, True])

    def test_equal_occupancy_different_order_retains_different_events(self):
        outputs = []
        for markers in ([0, 0, 1, 1], [0, 1, 0, 1]):
            sample = sample_episode()
            sample.arrays["support_contact_forces_w"][:, 0, 2] = np.asarray(markers) * 2
            records = measure_episode(sample)
            value = measurements(records)[f"{FEET[0]}.marker"]
            events = [r["value"]["step"] for r in records
                      if r["record_type"] == "event" and r["key"] == f"{FEET[0]}.marker.transition"]
            outputs.append((value["fraction"], events))
        self.assertEqual(outputs, [(0.5, [3]), (0.5, [2, 3, 4])])

    def test_force_norm_signed_vertical_share_and_trapezoid_are_distinct(self):
        sample = sample_episode(3, support=FEET[:2])
        sample.arrays["support_contact_forces_w"][:, :, 2] = [[0, 2], [10, 2], [0, 2]]
        force = measurements(measure_episode(sample))["force_summary"]
        np.testing.assert_allclose(force["sample_sum_norm_n"], [10, 6])
        np.testing.assert_allclose(force["norm_share"], [0.625, 0.375])
        np.testing.assert_allclose(force["rectangle_norm_ns"], [0.2, 0.12])
        integrals = force["trapezoid_sample_integral_proxy"]
        np.testing.assert_allclose([item["norm_ns"] for item in integrals], [0.2, 0.08])
        self.assertEqual([item["valid_edge_count"] for item in integrals], [2, 2])
        self.assertEqual(force["physical_impulse"], "unavailable")
        sample.arrays["support_contact_forces_w"][:, :, 2] = [[-2, 1], [-2, 1], [-2, 1]]
        negative = measurements(measure_episode(sample))["force_summary"]
        np.testing.assert_allclose(negative["norm_share"], [2 / 3, 1 / 3])
        np.testing.assert_allclose(negative["signed_fz_share"], [2, -1])
        self.assertEqual(negative["per_entity"][0]["signed_fz_n"]["minimum"], -2)

    def test_force_zero_denominator_is_unavailable_not_zero_share(self):
        sample = sample_episode(2, support=FEET[:2])
        sample.arrays["support_contact_forces_w"][:] = 0
        zero = measurements(measure_episode(sample))["force_summary"]
        self.assertEqual(zero["norm_share"], [None, None])
        self.assertEqual(zero["signed_fz_share"], [None, None])
        self.assertEqual([x["norm_ns"] for x in zero["trapezoid_sample_integral_proxy"]], [0, 0])
        sample.arrays["support_contact_forces_w"][:, :, 2] = [-2, 2]
        cancel = measurements(measure_episode(sample))["force_summary"]
        self.assertEqual(cancel["norm_share"], [0.5, 0.5])
        self.assertEqual(cancel["signed_fz_share"], [None, None])

    def test_force_integral_does_not_bridge_gap_or_invalid_foot_sample(self):
        sample = sample_episode(steps=[1, 2, 4, 5], support=FEET[:2])
        sample.arrays["support_contact_forces_w"][:, 0, 2] = [2, 4, 100, 102]
        sample.arrays["support_contact_forces_w"][1, 1, 0] = np.nan
        force = measurements(measure_episode(sample))["force_summary"]
        first, second = force["trapezoid_sample_integral_proxy"]
        self.assertEqual(first["valid_edge_count"], 2)
        self.assertAlmostEqual(first["covered_span_s"], 0.04)
        self.assertAlmostEqual(first["norm_ns"], 2.08)
        np.testing.assert_allclose(first["vector_ns"], [0, 0, 2.08])
        self.assertEqual(second["valid_edge_count"], 1)
        self.assertAlmostEqual(second["norm_ns"], 0.04)
        self.assertEqual((force["joint_valid_count"], force["unknown_joint_count"]), (3, 1))

    def test_full_body_rotation_and_world_translation_do_not_fake_relative_motion(self):
        sample = sample_episode(3)
        sample.arrays["base_quat_w"][:] = [math.sqrt(0.5), math.sqrt(0.5), 0, 0]
        sample.arrays["base_pos_w"][:, 0] = [0, 1, 2]
        sample.arrays["support_pos_w"][:] = sample.arrays["base_pos_w"][:, None, :] + [0, 1, 0]
        sample.arrays["support_lin_vel_w"][:, :, 0] = 50
        value = measurements(measure_episode(sample))[f"{FEET[0]}.relative_position"]
        np.testing.assert_allclose(value["first_body_xyz_m"], [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(value["last_body_xyz_m"], [0, 0, -1], atol=1e-12)
        self.assertAlmostEqual(value["relative_sample_path_m"], 0)
        self.assertEqual(value["world_xy_speed_mps"]["minimum"], 50)
        self.assertEqual(value["position_velocity_reference_equivalence"], "unverified")
        self.assertEqual(value["default_pose_reference"], "unavailable")

    def test_dynamic_entity_counts_and_joint_mapping_use_binding_order(self):
        sample = sample_episode(2)
        sample.arrays["joint_pos"][0] = np.arange(12)
        sample.arrays["joint_pos"][1] = np.arange(12) + 0.25
        data = measurements(measure_episode(sample))
        self.assertEqual(data["RL_calf_joint.pose"]["minimum"], 10)
        self.assertEqual(data["RL_calf_joint.pose"]["maximum"], 10.25)
        self.assertEqual(data["RL_calf_joint.pose"]["range"], 0.25)
        self.assertIn("unavailable", data["RL_calf_joint.pose"]["limit_proximity"])
        six = sample_episode(2, support=tuple(f"support_{i}" for i in range(6)), joints=("only_joint",))
        six.arrays["joint_pos"][:, 0] = [-0.3, 0.7]
        six.arrays["support_contact_forces_w"][0] = 0
        output = measurements(measure_episode(six))
        self.assertEqual(len(output["contact_combinations"]["counts"]), 64)
        self.assertEqual(output["contact_combinations"]["counts"][0], 1)
        self.assertEqual(output["contact_combinations"]["counts"][63], 1)
        self.assertEqual(output["only_joint.pose"]["range"], 1)

    def test_missing_force_and_invalid_single_foot_preserve_unaffected_root(self):
        sample = sample_episode(3)
        sample.arrays["base_pos_w"][:, 0] = [0, 1, 2]
        sample.arrays["support_contact_forces_w"][1, 0, 0] = np.nan
        data = measurements(measure_episode(sample))
        self.assertEqual((data[f"{FEET[0]}.marker"]["valid_count"], data[f"{FEET[0]}.marker"]["unknown_count"]), (2, 1))
        self.assertEqual(data[f"{FEET[1]}.marker"]["valid_count"], 3)
        self.assertEqual((data["contact_combinations"]["valid_count"], data["contact_combinations"]["unknown_count"]), (2, 1))
        self.assertEqual(data["root_progress"]["net_progress_m"], 2)
        del sample.arrays["support_contact_forces_w"]
        missing = measurements(measure_episode(sample))
        self.assertEqual(missing[f"{FEET[0]}.marker"]["unknown_count"], 3)
        self.assertIsNone(missing[f"{FEET[0]}.marker"]["fraction"])
        self.assertEqual(missing[f"{FEET[0]}.marker"]["missing_channels"], ["support_contact_forces_w"])
        self.assertEqual(missing["root_progress"]["net_progress_m"], 2)

    def test_current_history_and_terminal_manager_event_remain_separate(self):
        sample = sample_episode(4)
        sample.arrays["support_contact_forces_w"][:, 0] = 0
        sample.arrays["support_contact_forces_w_history"][-1, 1, 0, 2] = 4
        sample.arrays["forbidden_contact_forces_w_history"][-1, 2, 0, 2] = 3
        sample.arrays["done_mask"][-1] = True
        sample.arrays["wrapper_done_mask"][-1] = True
        sample.arrays["termination_1"][-1] = True
        sample.task["outcome"] = {"success": False, "primary_reason": "forbidden_contact", "episode_steps": 4}
        records = measure_episode(sample)
        data = measurements(records)
        self.assertEqual(data[f"{BASE}.current_marker"]["true_count"], 0)
        self.assertEqual(data[f"{BASE}.history_diagnostic"]["true_count"], 1)
        self.assertEqual(data[f"{FEET[0]}.marker"]["true_count"], 0)
        history = data[f"{FEET[0]}.history_diagnostic"]
        self.assertEqual(history["history_any_on_current_off_count"], 1)
        self.assertFalse(history["full_physics_coverage"] or history["integrable_as_substeps"])
        self.assertEqual((history["slot_timestamps"], history["slot_validity"]), ("not_recorded", "not_recorded"))
        terminal = data["task_endpoint"]
        self.assertTrue(terminal["terminal_flags"]["termination_1"])
        self.assertEqual(terminal["outcome"]["primary_reason"], "forbidden_contact")
        event = next(r for r in records if r["record_type"] == "event" and r["key"] == "termination_1.transition")
        self.assertEqual(event["value"]["step"], 4)
        self.assertIsNone(event["value"]["physical_event_range_s"])

    def test_endpoint_margin_records_include_position_operand_and_frozen_result(self):
        sample = sample_episode(3, dt=0.1)
        sample.arrays["base_pos_w"][:, 0] = [0, 1, 2]
        sample.task.update(goal_progress_m=5, deadline_s=2)
        records = measure_episode(sample)
        endpoint = next(r for r in records if r["record_type"] == "measurement" and r["key"] == "task_endpoint")
        self.assertEqual(endpoint["value"]["goal_margin_m"], 3)
        self.assertAlmostEqual(endpoint["value"]["deadline_margin_s"], 1.7)
        self.assertEqual(endpoint["value"]["terminal_step"], 3)
        self.assertIn("base_pos_w", endpoint["evidence_ref"]["channels"])
        self.assertEqual(endpoint["value"]["geometry_clearance"], "deferred_pending_independent_geometry_validation")

    def test_invalid_manager_flag_is_unknown_and_json_safe_without_losing_other_evidence(self):
        sample = sample_episode()
        sample.arrays["done_mask"] = np.array([0, np.nan, 0, np.nan])
        records = measure_episode(sample)
        json.dumps(records, allow_nan=False)
        endpoint = measurements(records)["task_endpoint"]
        self.assertIsNone(endpoint["terminal_flags"]["done_mask"])
        self.assertEqual(endpoint["tail_flags"]["done_mask"], [0, None, 0, None])
        self.assertEqual(endpoint["outcome"], sample.task["outcome"])
        unknown = [r for r in records if r["record_type"] == "state_interval"
                   and r["key"] == "done_mask" and r["value"]["state"] == "unknown"]
        self.assertEqual([r["value"]["steps_half_open"] for r in unknown], [[2, 3], [4, 5]])
        self.assertEqual(measurements(records)["root_progress"]["valid_count"], 4)

    def test_episode_equal_and_cell_time_weighted_fractions_have_explicit_units(self):
        first = sample_episode(1)
        second = sample_episode(9)
        second.arrays["support_contact_forces_w"][:, 0] = 0
        a = measurements(measure_episode(first))[f"{FEET[0]}.marker"]
        b = measurements(measure_episode(second))[f"{FEET[0]}.marker"]
        result = aggregate_fractions([a, b, {"numerator_cell_s": 0, "denominator_cell_s": 0}])
        self.assertEqual((result["total_episode_count"], result["valid_episode_count"], result["unknown_episode_count"]), (3, 2, 1))
        self.assertEqual(result["episode_equal_mean"], 0.5)
        self.assertAlmostEqual(result["sample_cell_duration_weighted_fraction"], 0.1)
        self.assertEqual(result["units"], {"episode_equal_mean": "episode_equal_fraction",
                                          "sample_cell_duration_weighted_fraction": "sample_cell_time_fraction"})
        # Different control periods require time weights, not pooled row counts.
        changed_dt = sample_episode(1, dt=0.04)
        c = measurements(measure_episode(changed_dt))[f"{FEET[0]}.marker"]
        weighted = aggregate_fractions([c, b])
        self.assertAlmostEqual(weighted["sample_cell_duration_weighted_fraction"], 0.04 / 0.22)
        empty = aggregate_fractions([])
        self.assertIsNone(empty["episode_equal_mean"])
        self.assertIsNone(empty["sample_cell_duration_weighted_fraction"])

    def test_all_records_are_json_safe_unique_and_relations_resolve_to_operands(self):
        sample = sample_episode(30)
        sample.arrays["support_contact_forces_w"][:, 0] = 0
        sample.arrays["base_quat_w"][3:] = [0, 1, 0, 0]
        sample.arrays["support_lin_vel_w"][:6, 0, 0] = 0.1
        records = measure_episode(sample)
        json.dumps(records, allow_nan=False)
        ids = {record["record_id"] for record in records}
        self.assertEqual(len(ids), len(records))
        relations = [record for record in records if record["record_type"] == "temporal_relation"
                     and "operand_record_ids" in record["value"]]
        self.assertTrue(relations)
        inversion_before = [record for record in relations if record["value"]["relation"] == "onset_before"
                            and record["key"].endswith(f"{FEET[0]}.low_speed_off")]
        self.assertEqual(len(inversion_before), 1)
        self.assertEqual(inversion_before[0]["value"]["status"], "true")
        for record in relations:
            self.assertTrue(set(record["value"]["operand_record_ids"]) <= ids)
            self.assertEqual(record["value"]["physical_relation"], "unknown")


if __name__ == "__main__":
    unittest.main()
