"""Independent observable-behavior requirements, not a template-count gate."""
from copy import deepcopy
import re
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


def final_claims(ep):
    measured = behavior(ep)
    summary = {"behavior": aggregate_behavior([measured], motion_bands=PROCESS_BANDS),
        "cycle": summarize_periods([episode_periods(measure_contact_cycles(ep))]),
        "joint": None, "terrain": None, "capabilities": {}}
    rows, _, required = _behavior_claims(summary)
    return {cid:rows[cid][1]["text"] for cid in required}


def final_text(ep):
    return "\n".join(final_claims(ep).values())


class BehaviorRequirements(unittest.TestCase):
    def test_post_inversion_quiet_interval_is_not_hidden_by_longer_earlier_interval(self):
        ep = sample_episode(130)
        ep.arrays["base_quat_w"][79:] = [0.,1.,0.,0.]
        ep.arrays["support_contact_forces_w"][:] = 0.
        ep.arrays["support_pos_w"][65:90,:,0] = np.arange(1,26)[:,None]*.01
        ep.arrays["support_pos_w"][90:,:,0] = .25
        events = [r for r in behavior(ep)["events"] if r["kind"]=="down_before_low_off"]
        self.assertEqual(len(events),4)
        self.assertTrue(all(r["first_observed_down_step"]==80 and r["start_step"]==91
                            and r["end_step"]==130 and abs(r["span_s"]-.78)<1e-10 for r in events))

    def test_joint_same_side_outside_occupancy_controls_text_and_optional_order(self):
        ep = sample_episode(80,joints=("aa","bb","yy","zz"))
        for i,count in enumerate((16,32,48,80)):
            ep.arrays["joint_pos"][:count,i] = 1.0002 if i==3 else .95
        bounds = {name:{"lower":-1.,"upper":1.,"unit":"rad","coordinate_status":"qualified_nominal"} for name in ep.entities["joints"]}
        measured = measure_joint_limits(ep,bounds,q_unit="rad")
        joint = {"groups":{name:{"counts":value["counts"],
            "episode_equal":{"minimum_signed_margin_rad":{"minimum":value["minimum_signed_margin_rad"]}}} for name,value in measured.items()}}
        summary = {"behavior":aggregate_behavior([behavior(ep)],motion_bands=PROCESS_BANDS),
            "cycle":summarize_periods([episode_periods(measure_contact_cycles(ep))]),"joint":joint,"capabilities":{}}
        rows,trace,required = _behavior_claims(summary)
        self.assertEqual([key for key in required if key.startswith("claim:joint.")],["claim:joint.zz","claim:joint.yy"])
        text = rows["claim:joint.zz"][1]["text"]
        self.assertEqual(float(re.search(r"上侧([\d.]+)%",text)[1]),100.)
        self.assertEqual(float(re.search(r"最小有符号余量(-?[\d.]+)rad",text)[1]),-.0002)
        self.assertLess(trace["claim:joint.bb"]["priority"],trace["claim:joint.aa"]["priority"])

    def test_condition_concentration_keeps_full_concentration_and_ignores_speed_ratio(self):
        for sparse_failures in (0,1):
            rows = []
            for i in range(30):
                ep = sample_episode(40)
                condition = ("concentrated","sparse","other")[i//10]
                failure = i<8 or (i==10 and sparse_failures)
                ep.env_index=i
                ep.task.update(stratum_id=condition,cell_id=condition)
                ep.task["outcome"].update(primary_reason="forbidden_contact" if failure else "success",success=not failure)
                if i==10:
                    ep.arrays["base_pos_w"][:,0] = ep.times*100
                rows.append(measure_behavior(ep))
            summary = {"behavior":aggregate_behavior(rows,motion_bands=PROCESS_BANDS),
                "cycle":summarize_periods([episode_periods(measure_contact_cycles(ep))]),"joint":None,"capabilities":{}}
            claims,trace,_ = _behavior_claims(summary)
            text = claims["claim:conditions.forbidden_contact"][1]["text"]
            self.assertEqual(summary["behavior"]["groups"]["outcome:forbidden_contact"]["episodes"],8+sparse_failures)
            match = re.search(r"concentrated (总计|末前速近零)(\d+)/10",text)
            self.assertIsNotNone(match)
            self.assertEqual(match.groups(),("末前速近零" if sparse_failures else "总计","8"))
            cell = summary["behavior"]["groups"]["condition:concentrated"]
            subset = summary["behavior"]["groups"]["condition:concentrated/forbidden_contact"]
            self.assertEqual((cell["episodes"],subset["episodes"]),(10,8))
            if sparse_failures:
                self.assertEqual(subset["terminal_motion"]["near_zero"]["episodes"],8)
            self.assertNotIn("sparse ",text)
            self.assertEqual(trace["claim:conditions.forbidden_contact"]["operands"]["cells"],
                             {"concentrated":["near_zero"] if sparse_failures else []})

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
        foot = final_claims(ep)["claim:foot.0"]
        self.assertTrue(foot.startswith("FL："))
        xyz = re.search(r"相对\(x,y,z\)均值=([^m]+)m", foot)
        self.assertIsNotNone(xyz)
        self.assertEqual(tuple(map(float, xyz[1].split("/"))), (0.,0.,.5))
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
        claims = final_claims(ep)
        presence = re.search(r"上轴向下(\d+)/(\d+)回合",claims["claim:motion"])
        self.assertIsNotNone(presence)
        self.assertEqual(tuple(map(int,presence.groups())), (1,1))
        single = claims["claim:outcome.success"]
        self.assertIn("成功1回合（单例，不作群体推广）",single)
        self.assertIn("结束0.06s；净前移0m、横移0m；末窗前速0m/s",single)
        self.assertIn("上轴倾角180°",single)
        self.assertIn("方位不适用（有效0/1，不适用1、未知0）",single)
        self.assertIn("上轴向下1/1回合",single)
        self.assertEqual(behavior(ep)["facts"]["terminal"]["tilt_end"]["value"],180.)
        ep.steps=np.array([1,2,4])
        self.assertTrue(np.isnan(position_motion(ep)["world_velocity"][2]).all())

    def test_two_feet_share_five_newton_contract(self):
        ep=sample_episode(80,support=FEET[:2])
        ep.task["contact_threshold_n"]=5
        ep.arrays["support_contact_forces_w"][:,:,2]=4
        text=final_text(ep)
        self.assertIn("力模>5N",text)
        self.assertIn("全部未触发100%",text)
        self.assertEqual(measure_contact_cycles(ep), {name: [] for name in ep.entities["support"]})

    def test_nominal_roundoff_and_real_outside_are_separate(self):
        lower,upper=np.radians([-155.9992218,-48.0001068])
        ep=sample_episode(3,joints=("calf",))
        ep.arrays["joint_pos"]=np.array([[lower],[upper],[lower-.0856]],dtype=np.float32)
        r=measure_joint_limits(ep,{"calf":{"lower":lower,"upper":upper,"unit":"rad","coordinate_status":"qualified_nominal"}},q_unit="rad")["calf"]
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
