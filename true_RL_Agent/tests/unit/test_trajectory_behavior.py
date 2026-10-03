"""Independent observable-behavior requirements, not a template-count gate."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from tests.unit.trajectory_fixtures import sample_episode, FEET
from true_rl_agent.evaluation.trajectory_translation.measurement import measure_behavior, low_progress_track, position_motion
from true_rl_agent.evaluation.trajectory_translation.cohort import aggregate_behavior
from true_rl_agent.evaluation.trajectory_translation.cycles import measure_contact_cycles
from true_rl_agent.evaluation.trajectory_translation.joint_limits import measure_joint_limits
from true_rl_agent.tasks.stair_climbing.trajectory_cycles import episode_periods, summarize_periods
from true_rl_agent.tasks.stair_climbing.trajectory_production import _behavior_claims, PROCESS_BANDS


def behavior(ep):
    ep.task.update(stratum_id="fixture", cell_id="fixture")
    return measure_behavior(ep)


def final_text(ep):
    measured = behavior(ep)
    summary = {"behavior": aggregate_behavior([measured], motion_bands=PROCESS_BANDS),
        "cycle": summarize_periods([episode_periods(measure_contact_cycles(ep))]),
        "joint": None, "terrain": None, "capabilities": {}}
    rows, _, required = _behavior_claims(summary)
    return "\n".join(rows[cid][1]["text"] for cid in required)


class BehaviorRequirements(unittest.TestCase):
    def test_static_translation_leg_motion_and_high_foot_reach_text(self):
        ep = sample_episode(80)
        ep.arrays["support_lin_vel_w"][:] = .08  # Deliberately biased independent channel.
        static = final_text(ep)
        self.assertIn("相对点速0m/s", static)
        self.assertIn("世界点速0m/s", static)
        ep.arrays["base_pos_w"][:, 0] = ep.times*.2
        ep.arrays["support_pos_w"][:, :, 0] = ep.times[:, None]*.2
        translating = final_text(ep)
        self.assertIn("世界点速0.2m/s", translating)
        self.assertIn("相对点速0m/s", translating)
        ep.arrays["support_pos_w"][:, 0, 2] = .5
        raised = final_text(ep)
        self.assertIn("(x,y,z)=0/0/0.5m", raised)
        self.assertNotEqual(raised, translating)
        ep.arrays["support_pos_w"][:, 0, 2] = .1*np.sin(ep.times*12)
        self.assertNotEqual(final_text(ep), translating)
        clone = deepcopy(ep)
        clone.source.update(manifest_sha256="f"*64, trial_id="renamed")
        self.assertEqual(final_text(ep), final_text(clone))

    def test_initial_not_applicable_internal_missing_and_negative(self):
        ep = sample_episode(80)
        ep.arrays["base_pos_w"][:, 0] = ep.times*.2
        low = low_progress_track(ep, ep.arrays["base_pos_w"][:, 0])
        self.assertEqual(low.config["not_applicable_count"],25)
        self.assertTrue(np.all(low.values[25:] == 0))
        activity = behavior(ep)["facts"]["all"]["active"]
        self.assertEqual((activity["value"],activity["valid"],activity["unknown"],activity["not_applicable"]),(0,55,0,25))
        ep.arrays["base_pos_w"][40] = np.nan
        value=behavior(ep)["facts"]["all"]["low_progress"]
        self.assertEqual(value["unknown"],26)
        ep = sample_episode(10)
        self.assertIsNone(behavior(ep)["facts"]["all"]["low_progress"]["value"])

    def test_direction_and_quaternion_rotation_not_euler_range(self):
        ep=sample_episode(80)
        ep.arrays["base_pos_w"][:,0]=-ep.times*.2
        f=behavior(ep)["facts"]["all"]
        self.assertEqual(f["low_progress"]["value"],100)
        self.assertLess(f["net"]["value"],0)
        ep=sample_episode(3)
        angle=np.radians([179,181,183])
        ep.arrays["base_quat_w"]=np.column_stack((np.cos(angle/2),np.sin(angle/2),np.zeros((3,2))))
        self.assertAlmostEqual(behavior(ep)["facts"]["all"]["rotation"]["value"],4,places=7)
        angle=np.radians([89,90,91])
        ep.arrays["base_quat_w"]=np.column_stack((np.cos(angle/2),np.zeros(3),np.sin(angle/2),np.zeros(3)))
        self.assertAlmostEqual(behavior(ep)["facts"]["all"]["rotation"]["value"],2,places=7)
        ep.arrays["base_quat_w"][:]=[0.,1.,0.,0.]
        self.assertIn("已观察到上轴向下1/1回合",final_text(ep))
        ep.steps=np.array([1,2,4])
        self.assertTrue(np.isnan(position_motion(ep)["world_velocity"][2]).all())

    def test_two_feet_share_five_newton_contract(self):
        ep=sample_episode(80,support=FEET[:2])
        ep.task["contact_threshold_n"]=5
        ep.arrays["support_contact_forces_w"][:,:,2]=4
        text=final_text(ep)
        self.assertIn("力模>5N",text)
        self.assertIn("全部未触发100%",text)
        self.assertTrue(all(r["measurement_config"]["contact_threshold_n"]==5 for r in measure_contact_cycles(ep)))

    def test_nominal_roundoff_and_real_outside_are_separate(self):
        lower,upper=np.radians([-155.9992218,-48.0001068])
        ep=sample_episode(3,joints=("calf",))
        ep.arrays["joint_pos"]=np.array([[lower],[upper],[lower-.0856]],dtype=np.float32)
        r=measure_joint_limits(ep,{"calf":{"lower":lower,"upper":upper,"unit":"rad","coordinate_status":"qualified_nominal"}},q_unit="rad",asset_id="fixture")[0]["value"]
        self.assertEqual((r["counts"]["lower_edge_rows"],r["counts"]["upper_edge_rows"],r["counts"]["outside_rows"]),(1,1,1))
        self.assertLess(r["minimum_signed_margin_rad"],-.0855)

    def test_independent_push_raw_height_speed_mapping(self):
        from tests.unit.test_push_trajectory import episode
        from true_rl_agent.tasks.push_recovery import trajectory_measurement as m
        from true_rl_agent.tasks.push_recovery.trajectory_card import build_push_card
        ep=episode()
        ep.arrays["pose"][:,0]=.4
        ep.arrays["pose"][:,3]=.1
        def verify():
            card=build_push_card([m.measure_push_episode(ep)],"fixture",episodes=[ep])
            claims={c["key"]:c["text"] for c in card["claims"]}
            self.assertIn("全程根部高度：",claims["height_m"])
            self.assertIn("平均为0.4m；",claims["height_m"])
            self.assertIn("全程根部平面速度模：",claims["planar_speed_mps"])
            self.assertIn("平均为0.1m/s；",claims["planar_speed_mps"])
        verify()
        wrong=deepcopy(m.ROOT_METRICS)
        wrong["height_m"]=(3,"根部高度","m")
        wrong["planar_speed_mps"]=(0,"根部平面速度模","m/s")
        with patch.object(m,"ROOT_METRICS",wrong), self.assertRaises(AssertionError):
            verify()  # Independent expectation catches the common-mode mutation.
