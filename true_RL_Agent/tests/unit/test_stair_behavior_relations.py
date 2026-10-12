"""Independent direction and observation requirements replacing relative-z holds."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np

from tests.unit.trajectory_fixtures import sample_episode
from true_rl_agent.evaluation.trajectory_translation.measurement import measure_behavior
from true_rl_agent.evaluation.trajectory_translation.cohort import aggregate_behavior
from true_rl_agent.tasks.stair_climbing import trajectory_production as production


def measured(ep):
    ep.task.update(stratum_id="fixture", cell_id="fixture")
    return measure_behavior(ep)


def summary(episodes):
    return {"behavior":aggregate_behavior([measured(e) for e in episodes],motion_bands=production.PROCESS_BANDS),
            "cycle":{"groups":{"contact_cycles:"+f:{"episodes":len(episodes),"episodes_with_qualified_windows":0,
                "episode_median_qualified_cycle_period_s":{"median":None}} for f in episodes[0].entities["support"]}},
            "joint":None,"capabilities":{}}


def text(episodes):
    rows,_,required=production._behavior_claims(summary(episodes))
    return "\n".join(rows[k][1]["text"] for k in required)


class BehaviorRelationsTests(unittest.TestCase):
    def test_circular_wrap_cancellation_and_projection_degeneracy(self):
        episodes=[sample_episode(40),sample_episode(40)]
        def headings(degrees):
            for ep,deg in zip(episodes,degrees):
                a=np.radians(deg)/2
                ep.arrays["base_quat_w"][:]=[np.cos(a),0,0,np.sin(a)]
            return summary(episodes)["behavior"]["groups"]["all"]["facts"]["all"]["heading_end"]
        f=headings([179,-179])
        self.assertAlmostEqual(abs(f["distribution"]["circular_mean"]),180)
        self.assertAlmostEqual(f["distribution"]["resultant"],.999847695,places=8)
        f=headings([90,-90])
        self.assertIsNone(f["distribution"]["circular_mean"])
        ep=episodes[0]
        ep.arrays["base_quat_w"][:]=[2**-.5,0,2**-.5,0]
        f=measured(ep)["facts"]["terminal"]["heading_end"]
        self.assertEqual((f["value"],f["unknown"],f["not_applicable"]),(None,0,1))
        ep.arrays["base_quat_w"][-1]=np.nan
        f=measured(ep)["facts"]["terminal"]["heading_end"]
        self.assertEqual((f["value"],f["unknown"],f["not_applicable"]),(None,1,0))
        ep.arrays["base_quat_w"][:]=[0,1,0,0]
        self.assertIsNone(measured(ep)["facts"]["terminal"]["up_direction_end"]["value"])

    def test_first_down_previous_sample_and_gap_are_observations(self):
        ep=sample_episode(3)
        ep.arrays["base_quat_w"][1]=[np.cos(np.pi/6),0,np.sin(np.pi/6),0]
        ep.arrays["base_quat_w"][2]=[0,0,1,0]
        f=measured(ep)["first_down_observation"]
        self.assertEqual((f["step"],f["time_s"],f["up_direction_deg"]),(3,.06,None))
        self.assertAlmostEqual(f["previous"]["tilt_deg"],60)
        self.assertEqual(f["previous"]["up_direction_deg"],0)
        ep.steps=np.array([1,2,4])
        self.assertIsNone(measured(ep)["first_down_observation"]["previous"])
        ep.arrays["base_quat_w"][:]=[0,1,0,0]
        f=measured(ep)["first_down_observation"]
        self.assertEqual((f["step"],f["time_s"],f["previous"]),(1,.02,None))

    def test_signed_motion_and_paired_attitude_reach_prose(self):
        ep=sample_episode(80)
        ep.arrays["base_pos_w"][:,0]=ep.times*.2
        ep.arrays["base_pos_w"][:,1]=ep.times*.7
        ep.arrays["base_quat_w"][:]=[np.cos(np.pi/6),0,np.sin(np.pi/6),0]
        episodes=[ep,deepcopy(ep)]
        def verify():
            value=text(episodes)
            for expected in ("净横向位移1.106m","前速0.2m/s、横速0.7m/s","上轴倾角60°、方位0°(R=1)"):
                self.assertIn(expected,value)
        verify()
        original=production._number
        with patch.object(production,"_number",side_effect=lambda value,unit=None:original(value*2 if value is not None and unit=="m/s" else value,unit)):
            with self.assertRaises(AssertionError):
                verify()  # Raw expected .2/.7 must reject a shared renderer error.

    def test_terminal_contact_denominators_short_history_and_unknown(self):
        ep=sample_episode(40)
        ep.arrays["support_contact_forces_w"][:]=0
        ep.arrays["support_contact_forces_w"][-1,0]=np.nan
        f=measured(ep)["facts"]["terminal"]["contact.0"]
        self.assertEqual((f["value"],f["valid"],f["unknown"]),(100,25,1))
        short=measured(sample_episode(10))["facts"]["terminal"]["contact.0"]
        self.assertEqual((short["value"],short["not_applicable"]),(None,10))
        short=sample_episode(2)
        short.dt_s=.3  # A .3s observation span is not a complete .5s history.
        self.assertIsNone(measured(short)["facts"]["terminal"]["contact.0"]["value"])


if __name__ == "__main__":
    unittest.main()
