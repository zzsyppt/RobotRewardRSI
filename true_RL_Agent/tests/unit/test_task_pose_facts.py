"""Displayed pose measurements keep their sign, scope and missing dimensions."""

import math
import struct
import unittest
from dataclasses import replace

from true_rl_agent.evaluation.visual_observer import MediaImage, MediaWindow
from true_rl_agent.tasks.push_recovery.visual_observation import (
    PUSH_S0_DEFINITIONS, push_pose_facts, push_s0_for_window,
)
from true_rl_agent.tasks.stair_climbing.visual_observation import (
    stair_pose_facts, stair_s0_definitions, stair_s0_for_window,
)


IDENTITY = {name: name + "-value" for name in (
    "session_id", "task_id", "robot_id", "protocol_id", "source_id",
    "checkpoint_id", "episode_id", "run_id")}
# This is the already-bound frame consumed by stair_s0_for_window; asset binding
# itself is covered by the existing task visual-observation tests.
ROOT_FRAME = {"robot_profile_id": IDENTITY["robot_id"], "body_name": "base",
              "forward_axis_body": [1., 0., 0.], "up_axis_body": [0., 0., 1.]}


def window(steps):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 1, 1)
    identity = {**IDENTITY, **{"numeric_" + key: value for key, value in IDENTITY.items()}}
    return MediaWindow("e1", "camera", "layout", "roi", ("root", "base"),
                       tuple(MediaImage(png, 1, 1, (step,), "v0") for step in steps),
                       identity=identity)


def stair_window(steps, *, quaternions=None, positions=None, origin=None, frame=ROOT_FRAME):
    result = window(reversed(steps))
    scalars = stair_s0_for_window(result, identity=IDENTITY, control_step=steps,
                                 base_pos_w=positions, base_quat_w=quaternions,
                                 terrain_origin_w=origin, root_frame=frame)
    return replace(result, s0=tuple(reversed(scalars)), s0_definitions=stair_s0_definitions(frame))


def push_window(steps, poses):
    result = window(reversed(steps))
    scalars = push_s0_for_window(result, identity=IDENTITY, control_step=steps,
                                pose=poses, planar_motion=None)
    return replace(result, s0=tuple(reversed(scalars)), s0_definitions=PUSH_S0_DEFINITIONS)


class TaskPoseFactsTests(unittest.TestCase):
    def test_stair_signed_elevation_and_inverted_up_are_distinct_dimensions(self):
        half = math.radians(15)
        media = stair_window([1, 7, 11], quaternions=[
            [math.cos(half), 0, math.sin(half), 0],
            [math.cos(half), 0, -math.sin(half), 0],
            [0, 1, 0, 0],
        ])
        facts = {fact["id"]: fact for fact in stair_pose_facts(media)}
        self.assertEqual(len(facts), 2)
        elevation = facts["stair.root_forward_elevation"]
        self.assertIn("1: -30.0; 7: 30.0; 11: 0.0", elevation["text"])
        self.assertIn("1: 30.0; 7: 30.0; 11: 180.0", facts["stair.root_up_world_z_angle"]["text"])
        self.assertEqual(elevation["scalar_refs"], [
            {"name": "root_forward_world_z", "step": step} for step in [1, 7, 11]])
        self.assertEqual(stair_pose_facts(media), stair_pose_facts(replace(media, s0=tuple(reversed(media.s0)))))

    def test_stair_missing_frame_does_not_invent_forward_or_clearance(self):
        media = stair_window([2], quaternions=[[1, 0, 0, 0]],
                             positions=[[10, 20, 5.12349]], origin=[10, 20, 5], frame=None)
        facts = {fact["id"]: fact for fact in stair_pose_facts(media)}
        self.assertEqual(set(facts), {"stair.root_up_world_z_angle", "stair.root_terrain_height"})
        self.assertIn("approx m): 2: 0.123", facts["stair.root_terrain_height"]["text"])
        self.assertIn("Not foot clearance", facts["stair.root_terrain_height"]["text"])
        self.assertEqual(stair_pose_facts(stair_window([2], frame=None)), ())

    def test_stair_only_roundoff_outside_unit_interval_is_clamped(self):
        media = stair_window([1], quaternions=[[1, 0, 0, 0]])
        for value, angle in [(1 + 5e-7, "0.0"), (-1 - 5e-7, "180.0")]:
            with self.subTest(value=value):
                scalars = tuple({**s, "value": value} if s["name"] == "root_up_world_z" else s for s in media.s0)
                fact = next(f for f in stair_pose_facts(replace(media, s0=scalars))
                            if f["id"] == "stair.root_up_world_z_angle")
                self.assertIn("approx deg): 1: " + angle, fact["text"])
        for name in ["root_up_world_z", "root_forward_world_z"]:
            for value in [1.01, -1.01, float("nan"), True]:
                with self.subTest(name=name, value=value):
                    scalars = tuple({**s, "value": value} if s["name"] == name else s for s in media.s0)
                    with self.assertRaises(ValueError):
                        stair_pose_facts(replace(media, s0=scalars))

    def test_push_signed_roll_pitch_use_present_columns_and_sorted_steps(self):
        media = push_window([1, 4], [[.8004, math.radians(15), math.radians(-30)],
                                     [.7994, math.radians(-15), math.radians(30)]])
        facts = {fact["id"]: fact for fact in push_pose_facts(media)}
        self.assertEqual(len(facts), 3)
        self.assertIn("1: 15.0; 4: -15.0", facts["push.root_roll"]["text"])
        self.assertIn("1: -30.0; 4: 30.0", facts["push.root_pitch"]["text"])
        self.assertIn("1: 0.800; 4: 0.799", facts["push.root_height"]["text"])
        self.assertIn("not an independent torso link or knee angle", facts["push.root_roll"]["text"])
        self.assertEqual(facts["push.root_pitch"]["scalar_refs"], [
            {"name": "root_pitch_rad", "step": 1}, {"name": "root_pitch_rad", "step": 4}])

    def test_push_legacy_narrow_arrays_leave_missing_angles_absent(self):
        for poses, expected in [(None, set()), ([[.8]], {"push.root_height"}),
                                ([[.8, .1]], {"push.root_height", "push.root_roll"})]:
            with self.subTest(poses=poses):
                self.assertEqual({f["id"] for f in push_pose_facts(push_window([1], poses))}, expected)

    def test_rounded_small_angle_is_explicitly_approximate(self):
        fact = next(f for f in push_pose_facts(push_window([1], [[.8, math.radians(.001)]]))
                    if f["id"] == "push.root_roll")
        self.assertIn("approx deg): 1: 0.0", fact["text"])
        self.assertNotIn("unchanged", fact["text"])
        self.assertNotIn("zero", fact["text"])

    def test_push_euler_boundaries_and_overflow(self):
        facts = {fact["id"]: fact for fact in push_pose_facts(
            push_window([1, 2], [[.8, -math.pi, -math.pi / 2], [.8, math.pi, math.pi / 2]]))}
        self.assertIn("1: -180.0; 2: 180.0", facts["push.root_roll"]["text"])
        self.assertIn("1: -90.0; 2: 90.0", facts["push.root_pitch"]["text"])
        with self.assertRaises(ValueError):
            push_pose_facts(push_window([1], [[.8, 1e308]]))

    def test_facts_reject_duplicate_or_undisplayed_scalar_references(self):
        for media, build in [(stair_window([1], quaternions=[[1, 0, 0, 0]]), stair_pose_facts),
                             (push_window([1], [[.8, .1, .2]]), push_pose_facts)]:
            with self.subTest(task=build.__name__):
                with self.assertRaises(ValueError):
                    build(replace(media, s0=media.s0 + (media.s0[0],)))
                with self.assertRaises(ValueError):
                    build(replace(media, s0=({**media.s0[0], "step": 2},)))
                with self.assertRaises(ValueError):
                    build(replace(media, s0=({**media.s0[0], "value": float("inf")},)))


if __name__ == "__main__":
    unittest.main()
