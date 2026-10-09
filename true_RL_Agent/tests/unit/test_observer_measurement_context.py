"""Measured context keeps exact values/definitions and explicit scalar provenance."""

from dataclasses import replace
import json
import unittest

from true_rl_agent.evaluation.visual_observer import prepare_observer_request
from tests.unit.test_task_visual_observation import window


class ObserverMeasurementContextTests(unittest.TestCase):
    def media(self):
        definition = {"name": "measured_angle", "body": "root", "unit": "rad",
                      "reference_frame": "fixture axis", "source": "fixture numeric array"}
        scalars = tuple({**definition, "step": step, "value": value}
                        for step, value in ((1, -.123456789012345), (80, .37602783780356275)))
        facts = ({"id": "fixture.angle", "body": "root",
                  "text": "Fixture angle at displayed steps (approx deg): 1: -7.1; 80: 21.5.",
                  "scalar_refs": [{"name": "measured_angle", "step": step} for step in (1, 80)]},)
        return replace(window([1, 80], ()), s0=scalars, s0_definitions=(definition,), measured_facts=facts)

    def public(self, media):
        prepared = prepare_observer_request([media], model="fixture", max_tokens=64)
        return json.loads(prepared["payload"]["messages"][1]["content"][0]["text"])

    def test_compact_scalars_reconstruct_exact_defined_measurements(self):
        media = self.media()
        public = self.public(media)
        by_name = {definition["name"]: definition for definition in public["s0_definitions"]}
        self.assertEqual([{**by_name[row["name"]], **row} for row in public["s0"]], list(media.s0))
        self.assertEqual(public["measured_facts"], list(media.measured_facts))
        self.assertLess(len(json.dumps(public["s0"])), len(json.dumps(media.s0)))

    def test_fact_cannot_cite_another_step_or_unprovided_measurement(self):
        media = self.media()
        for reference in ({"name": "measured_angle", "step": 2}, {"name": "unknown", "step": 1}):
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, "actual S0"):
                self.public(replace(media, measured_facts=({**media.measured_facts[0],
                                                          "scalar_refs": [reference]},)))

    def test_missing_measured_context_is_not_filled_in(self):
        public = self.public(window([1, 80], ()))
        self.assertEqual(public["s0"], [])
        self.assertEqual(public["measured_facts"], [])
        self.assertEqual(public["s0_definitions"], [])
        self.assertEqual(public["body_image_points"], [])

    def test_body_projection_is_bound_to_actual_image_and_does_not_claim_visibility(self):
        media = self.media()
        point = {"body_name": "root", "control_step": 1, "view_id": media.images[0].view_id,
                 "image_xy_px": [12.5, 9.25], "projection_status": "in_frame"}
        public = self.public(replace(media, body_image_points=(point,)))
        self.assertEqual(public["body_image_points"], [{"body": "root", "step": 1,
            "view_id": media.images[0].view_id, "xy_px": [12.5, 9.25], "status": "in_frame"}])
        for patch in ({"control_step": 99}, {"body_name": "unbound_link"}, {"view_id": "unshown"}):
            with self.subTest(patch=patch), self.assertRaisesRegex(ValueError, "displayed step/view"):
                self.public(replace(media, body_image_points=({**point, **patch},)))


if __name__ == "__main__":
    unittest.main()
