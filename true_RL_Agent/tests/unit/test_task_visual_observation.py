"""Numerical claims must describe only bound, displayed samples."""

import struct
import unittest
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from true_rl_agent.evaluation.visual_observer import (
    MediaImage, MediaWindow, audit_observer_output, prepare_observer_request,
)
from true_rl_agent.evaluation.visual_review import (
    REVIEW_CRITERION, SCHEMA_VERSION as REVIEW_SCHEMA, attach_visual_review, visual_review_binding,
)
from true_rl_agent.tasks.push_recovery.visual_observation import (
    PUSH_S0_DEFINITIONS,
    align_push_snapshot_samples,
    attach_push_visual_observations,
    push_s0_for_window,
    push_measurement_claims,
    push_numeric_check,
)
from true_rl_agent.tasks.stair_climbing.visual_observation import (
    STAIR_MEASUREMENT_CLAIMS,
    align_stair_snapshot_samples,
    STAIR_S0_DEFINITIONS,
    bind_stair_root_frame,
    stair_s0_definitions,
    stair_s0_for_window,
    attach_stair_visual_observations,
    stair_numeric_check,
)

IDENTITY = {key: key + "-value" for key in (
    "session_id", "task_id", "robot_id", "protocol_id", "source_id",
    "checkpoint_id", "episode_id", "run_id")}


def window(steps, claims, identity=IDENTITY, views=("side",)):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 1, 1)
    media_identity = {**identity, **{"numeric_" + k: v for k, v in identity.items()}}
    return MediaWindow("e1", "camera", "layout", "roi", ("root",),
                       tuple(MediaImage(png, 1, 1, tuple(steps), view) for view in views),
                       identity=media_identity, measurement_claims=claims)


def observation(claim, first=1, last=3, *, steps=None, views=("side",), relation="state", asserted=True):
    cited = sorted({first, last}) if steps is None else steps
    return {"numeric_claim": {"id": claim, "asserted": asserted}, "body": "root",
            "relation": relation, "spatial_visibility": "visible",
            "text": "Root pose differs across cited samples.", "uncertainty": "sampled views only",
            "evidence_refs": [{"step": step, "view_id": view} for step in cited for view in views]}


def test_stair_shown_minimum_and_endpoint_do_not_imply_stillness():
    steps = np.array([1, 2, 3])
    pos = np.array([[0., 0, 0], [4., 0, 0], [0.01, 0, 0]])
    quat = np.array([[1., 0, 0, 0], [0., 1, 0, 0], [1., 0, 0, 0]])
    check = stair_numeric_check(identity=IDENTITY, control_step=steps,
                                base_pos_w=pos, base_quat_w=quat)
    shown = window([1, 3], STAIR_MEASUREMENT_CLAIMS)
    assert check(observation("root_up_z_negative"), shown)["status"] == "conflict"
    assert check(observation("root_small_x_endpoint_delta"), shown)["status"] == "consistent"
    assert check(observation("root_up_z_negative"), window([1, 2, 3], STAIR_MEASUREMENT_CLAIMS))["status"] == "conflict"
    assert check(observation("root_up_z_negative", steps=[1, 2, 3]), window([1, 2, 3], STAIR_MEASUREMENT_CLAIMS))["status"] == "consistent"
    assert check(observation("root_small_x_endpoint_delta", 1, 1), shown) is None
    assert check(observation("unknown"), shown) is None


def test_push_shown_height_uses_origin_relative_pose_and_xy_endpoints():
    steps = [1, 2, 3]
    # pose[:,0] is root_pos_w.z - env_origin_w.z, not absolute world z.
    pose = np.array([[0.8, 0], [0.1, 0], [0.7, 0]])
    motion = np.array([[0, 0, 0, 0], [0, 0, 5, 0], [0, 0, .01, 0]])
    claims = push_measurement_claims(.5)
    assert "root_pos_w.z - env_origin_w.z" in claims[0]["definition"]
    check = push_numeric_check(identity=IDENTITY, control_step=steps, pose=pose,
                               planar_motion=motion, recovery_height_m=.5)
    shown = window([1, 3], claims)
    assert check(observation("root_low_height_sample"), shown)["status"] == "conflict"
    assert check(observation("root_low_height_sample"), window([1, 2, 3], claims))["status"] == "conflict"
    assert check(observation("root_low_height_sample", steps=[1, 2, 3]), window([1, 2, 3], claims))["status"] == "consistent"
    assert check(observation("root_small_xy_endpoint_delta"), shown)["status"] == "consistent"
    assert check(observation("root_small_xy_endpoint_delta", 1, 1), shown) is None
    assert check(observation("unknown"), shown) is None


def test_independent_claim_arrays_and_exact_definition():
    stair_window = window([1, 3], STAIR_MEASUREMENT_CLAIMS)
    stair = stair_numeric_check(identity=IDENTITY, control_step=[1, 3], base_pos_w=None,
                                base_quat_w=[[1, 0, 0, 0], [0, 1, 0, 0]])
    assert stair(observation("root_up_z_negative"), stair_window)["status"] == "consistent"
    assert stair(observation("root_small_x_endpoint_delta"), stair_window) is None
    wrong_stair = ({**STAIR_MEASUREMENT_CLAIMS[0], "definition": "different coordinate"},
                   STAIR_MEASUREMENT_CLAIMS[1])
    assert stair(observation("root_up_z_negative"), window([1, 3], wrong_stair)) is None

    threshold = 0.12345678912345678
    claims = push_measurement_claims(threshold)
    assert repr(threshold) in claims[0]["definition"]
    push = push_numeric_check(identity=IDENTITY, control_step=[1, 3],
                              pose=[[.2], [.1]], planar_motion=None,
                              recovery_height_m=threshold)
    assert push(observation("root_low_height_sample"), window([1, 3], claims))["status"] == "consistent"
    assert push(observation("root_small_xy_endpoint_delta"), window([1, 3], claims)) is None
    wrong_push = ({**claims[0], "definition": "different threshold"}, claims[1])
    assert push(observation("root_low_height_sample"), window([1, 3], wrong_push)) is None


def check_binding_mismatch_and_missing_displayed_step(test):
    cases = [
        (lambda: stair_numeric_check(identity=IDENTITY, control_step=[1],
                                     base_pos_w=[[0, 0, 0]], base_quat_w=[[1, 0, 0, 0]]), STAIR_MEASUREMENT_CLAIMS),
        (lambda: push_numeric_check(identity=IDENTITY, control_step=[1],
                                    pose=[[.8]], planar_motion=[[0, 0, 0, 0]],
                                    recovery_height_m=.5), push_measurement_claims(.5)),
    ]
    for factory, claims in cases:
        with test.subTest(claim=claims[0]["id"]):
            check = factory()
            altered = {**IDENTITY, "run_id": "another-run"}
            with test.assertRaisesRegex(ValueError, "identities differ"):
                check(observation(claims[0]["id"], 1, 1), window([1], claims, altered))
            test.assertIsNone(check(observation(claims[0]["id"], 1, 2), window([1, 2], claims)))


def audited(identity):
    media = replace(window([1], ()), identity=identity)
    raw = {"episodes": [{"alias": "e1", "observations": [{
        "body": "root", "spatial_visibility": "visible",
        "text": "Root leans left", "uncertainty": "sampled view",
        "relation": "state", "evidence_refs": [{"step": 1, "view_id": "side"}],
    }], "not_visible": []}]}
    return audit_observer_output(raw, [media])


def bound(**updates):
    value = {**IDENTITY, **updates}
    value.update({"numeric_" + key: value[key] for key in IDENTITY})
    return value


def reviewed_fixture(audit):
    # Synthetic projection fixture, not a model/human semantic accuracy result.
    return attach_visual_review(audit, {
        "schema_version": REVIEW_SCHEMA, "source_sha256": visual_review_binding(audit),
        "criterion": REVIEW_CRITERION, "reviewer": {"id": "synthetic-unit-fixture", "kind": "dev"},
        "decisions": [{"observation_index": i, "verdict": "supported", "reason": "Synthetic fixture only."}
                      for i in range(len(audit["observations"]))],
    })


def paired_reviewed_fixture(first_identity, second_identity):
    raw = audited(first_identity)["raw_output"]
    second = deepcopy(raw["episodes"][0])
    second["alias"] = "e2"
    raw["episodes"].append(second)
    media = [replace(window([1], ()), identity=first_identity),
             replace(window([1], ()), alias="e2", identity=second_identity)]
    return reviewed_fixture(audit_observer_output(raw, media))


def test_stair_attachment_preserves_numeric_and_replaces_gap():
    full = {"domain": {"archive_hash": "archive", "checkpoint_hash": "sha256:checkpoint",
                       "task_protocol_hash": "protocol"},
            "episodes": [{"episode_id": "r4.e2", "run": 4, "slot": 2, "measurements": [1, 2]}],
            "semantic": {"sampling": {"visual_observed_episode_n": 0, "numeric": [2]},
                         "sentences": [{"id": "visual-unavailable", "section": "visual"},
                                       {"id": "numeric", "section": "process"}]},
            "l1": {"unchanged": [1]}, "l2": {"unchanged": [2]}}
    original = deepcopy(full)
    valid = bound(source_id="archive", checkpoint_id="sha256:checkpoint",
                  protocol_id="protocol", run_id="4", episode_id="r4.e2")
    result = attach_stair_visual_observations(full, reviewed_fixture(audited(valid)), observer_version="v1")
    assert full == original
    assert result["episodes"] == full["episodes"]
    assert result["l1"] == full["l1"] and result["l2"] == full["l2"]
    assert [row["id"] for row in result["semantic"]["sentences"]] == ["numeric", "visual.observation", "visual.coverage"]
    assert result["semantic"]["sampling"]["visual_observed_episode_n"] == 1
    again = attach_stair_visual_observations(result, reviewed_fixture(audited(valid)), observer_version="v1")
    assert [row["id"] for row in again["semantic"]["sentences"]] == ["numeric", "visual.observation", "visual.coverage"]
    assert result["observer_source_binding"]["approved_aliases"] == {"e1": "r4.e2"}
    mixed = audited(valid)
    invalid = deepcopy(mixed["observations"][0])
    invalid["violations"] = ["bounded_text"]
    invalid["raw_observation"]["text"] = "x" * 201
    mixed["observations"].insert(0, invalid)
    mixed_result = attach_stair_visual_observations(full, reviewed_fixture(mixed), observer_version="v1")
    assert [row["id"] for row in mixed_result["semantic"]["sentences"]] == ["numeric", "visual.observation", "visual.coverage"]
    assert "2 statements" in mixed_result["semantic"]["sentences"][-1]["text"]
    assert "1 mechanically invalid" in mixed_result["semantic"]["sentences"][-1]["text"]
    assert mixed_result["observer_source_binding"]["rejected_observation_indices"] == []
    bad = attach_stair_visual_observations(full, audited({**valid, "source_id": "wrong"}), observer_version="v1")
    assert not any(row["id"] == "visual.observation" for row in bad["semantic"]["sentences"])
    assert "1 statements" in bad["semantic"]["sentences"][-1]["text"]
    assert "1 mechanically invalid" in bad["semantic"]["sentences"][-1]["text"]
    assert bad["observer_source_binding"]["rejected_observation_indices"] == [0]
    assert bad["observer_evidence"]["observations"][0]["media_identity"]["source_id"] == "wrong"


def test_push_attachment_requires_verified_actual_run():
    full = {"binding": {"checkpoint_hash": "sha256:checkpoint", "resolved_protocol_hash": "protocol"},
            "source_bindings": [{"source_index": 0, "logical_run": 1, "manifest_sha256": "archive",
                                 "actual_run_index": 4, "actual_run_status": "verified_archive_identity"}],
            "transcripts": [{"source": 0, "env_id": 2, "measurements": [1]}],
            "sentences": [{"id": "push.visual.missing", "section": "visual"},
                          {"id": "numeric", "section": "process"}],
            "evidence_catalog": {"numeric": {"pointer": "/transcripts/0"}},
            "l1": [1], "l2": [2]}
    original = deepcopy(full)
    valid = bound(source_id="archive", checkpoint_id="sha256:checkpoint",
                  protocol_id="protocol", run_id="4", episode_id="r4.e2")
    result = attach_push_visual_observations(full, reviewed_fixture(audited(valid)), observer_version="v1")
    assert full == original
    assert result["transcripts"] == full["transcripts"]
    assert result["l1"] == full["l1"] and result["l2"] == full["l2"]
    assert result["evidence_catalog"]["numeric"] == full["evidence_catalog"]["numeric"]
    assert [row["id"] for row in result["sentences"]] == ["numeric", "visual.observation", "visual.coverage"]
    assert "r1.e2/root" in result["sentences"][1]["text"]
    again = attach_push_visual_observations(result, reviewed_fixture(audited(valid)), observer_version="v1")
    assert [row["id"] for row in again["sentences"]] == ["numeric", "visual.observation", "visual.coverage"]
    assert result["observer_source_binding"]["approved_aliases"] == {"e1": "r1.e2"}
    full["source_bindings"][0]["actual_run_index"] = None
    bad = attach_push_visual_observations(full, audited(valid), observer_version="v1")
    assert not any(row["id"] == "visual.observation" for row in bad["sentences"])
    assert "1 statements" in bad["sentences"][-1]["text"]
    assert "1 mechanically invalid" in bad["sentences"][-1]["text"]
    assert bad["observer_source_binding"]["rejected_observation_indices"] == [0]


class TaskVisualObservationTests(unittest.TestCase):
    def test_stair_visual_episode_priority_reuses_numeric_exemplar_then_terminal_failure(self):
        full = {"domain": {"archive_hash": "archive", "checkpoint_hash": "sha256:checkpoint",
                           "task_protocol_hash": "protocol"},
                "episodes": [{"episode_id": "r4.e2", "run": 4, "slot": 2, "primary_reason": "success"},
                             {"episode_id": "r4.e3", "run": 4, "slot": 3, "primary_reason": "timeout"}],
                "semantic": {"sampling": {}, "sentences": []}}
        first = bound(source_id="archive", checkpoint_id="sha256:checkpoint",
                      protocol_id="protocol", run_id="4", episode_id="r4.e2")
        second = bound(source_id="archive", checkpoint_id="sha256:checkpoint",
                       protocol_id="protocol", run_id="4", episode_id="r4.e3")
        audit = paired_reviewed_fixture(first, second)
        audit_before = deepcopy(audit)
        for exemplar, expected in ((None, "r4.e3"), ("r4.e2", "r4.e2"),
                                   ("r4.e3", "r4.e3"), ("not_observed", "r4.e3")):
            with self.subTest(exemplar=exemplar):
                full["semantic"]["sentences"] = ([] if exemplar is None else [
                    {"id": "process-example", "section": "process", "scope": {"label": exemplar}}])
                before = deepcopy(full)
                result = attach_stair_visual_observations(full, audit, observer_version="v1")
                shown = next(row for row in result["semantic"]["sentences"] if row["id"] == "visual.observation")
                self.assertEqual(shown["scope"]["label"], expected)
                self.assertEqual(full, before)
                self.assertEqual(result["episodes"], full["episodes"])
                self.assertEqual(result["observer_evidence"], audit_before)
                self.assertEqual(audit, audit_before)
        full["semantic"]["sentences"] = []
        for reason in (None, "", " ", False, "success"):
            with self.subTest(reason=reason):
                full["episodes"][1]["primary_reason"] = reason
                result = attach_stair_visual_observations(full, audit, observer_version="v1")
                self.assertEqual(result["semantic"]["sentences"][0]["scope"]["label"], "r4.e2")
        full["episodes"][1]["primary_reason"] = "timeout"
        rejected = paired_reviewed_fixture(first, bound(source_id="wrong", checkpoint_id="sha256:checkpoint",
            protocol_id="protocol", run_id="4", episode_id="r4.e3"))
        result = attach_stair_visual_observations(full, rejected, observer_version="v1")
        self.assertEqual(result["semantic"]["sentences"][0]["scope"]["label"], "r4.e2")
        self.assertEqual(result["observer_evidence"], rejected)

    def test_push_visual_episode_priority_uses_logical_exemplar_and_known_terminal_failure(self):
        full = {"binding": {"checkpoint_hash": "sha256:checkpoint", "resolved_protocol_hash": "protocol"},
                "source_bindings": [{"source_index": 0, "logical_run": 1, "manifest_sha256": "archive",
                                     "actual_run_index": 4, "actual_run_status": "verified_archive_identity"}],
                "transcripts": [{"source": 0, "env_id": 2, "terminal_label": "success_recovered"},
                                {"source": 0, "env_id": 3, "terminal_label": "not_recovered"}],
                "sentences": [], "evidence_catalog": {}}
        first = bound(source_id="archive", checkpoint_id="sha256:checkpoint",
                      protocol_id="protocol", run_id="4", episode_id="r4.e2")
        second = bound(source_id="archive", checkpoint_id="sha256:checkpoint",
                       protocol_id="protocol", run_id="4", episode_id="r4.e3")
        audit = paired_reviewed_fixture(first, second)
        audit_before = deepcopy(audit)
        for exemplar, expected in ((None, "r1.e3"), ("r1.e2", "r1.e2"),
                                   ("r1.e3", "r1.e3"), ("r4.e2", "r1.e3")):
            with self.subTest(exemplar=exemplar):
                full["representative_selection"] = {} if exemplar is None else {"context_episode_id": exemplar}
                before = deepcopy(full)
                result = attach_push_visual_observations(full, audit, observer_version="v1")
                self.assertEqual(result["sentences"][0]["scope"]["label"], expected)
                self.assertEqual(full, before)
                self.assertEqual(result["transcripts"], full["transcripts"])
                self.assertEqual(result["observer_evidence"], audit_before)
                self.assertEqual(audit, audit_before)
        full.pop("representative_selection")
        for label, expected in ((None, "r1.e2"), ("no_push_control", "r1.e2"),
                                ("unknown", "r1.e2"), ("success_recovered", "r1.e2"),
                                ("fall_side", "r1.e3"), ("height_collapse", "r1.e3")):
            with self.subTest(label=label):
                full["transcripts"][1]["terminal_label"] = label
                result = attach_push_visual_observations(full, audit, observer_version="v1")
                self.assertEqual(result["sentences"][0]["scope"]["label"], expected)

    def test_stair_measurements(self):
        test_stair_shown_minimum_and_endpoint_do_not_imply_stillness()

    def test_push_measurements(self):
        test_push_shown_height_uses_origin_relative_pose_and_xy_endpoints()

    def test_binding(self):
        check_binding_mismatch_and_missing_displayed_step(self)

    def test_independent_claim_arrays(self):
        test_independent_claim_arrays_and_exact_definition()

    def test_explicit_polarity_and_strict_threshold_boundaries(self):
        cases = [
            (stair_numeric_check(identity=IDENTITY, control_step=[1, 3],
                                 base_pos_w=[[0, 0, 0], [.02, 0, 0]],
                                 base_quat_w=[[.5, .5, .5, .5], [.5, .5, .5, .5]]),
             STAIR_MEASUREMENT_CLAIMS, False),
            (stair_numeric_check(identity=IDENTITY, control_step=[1, 3],
                                 base_pos_w=[[0, 0, 0], [.01, 0, 0]],
                                 base_quat_w=[[0, 1, 0, 0], [0, 1, 0, 0]]),
             STAIR_MEASUREMENT_CLAIMS, True),
            (push_numeric_check(identity=IDENTITY, control_step=[1, 3],
                                pose=[[.5], [.8]], planar_motion=[[0, 0, 0, 0], [0, 0, .02, 0]],
                                recovery_height_m=.5), push_measurement_claims(.5), False),
            (push_numeric_check(identity=IDENTITY, control_step=[1, 3],
                                pose=[[.49], [.8]], planar_motion=[[0, 0, 0, 0], [0, 0, .01, 0]],
                                recovery_height_m=.5), push_measurement_claims(.5), True),
        ]
        for check, claims, observed in cases:
            media = window([1, 3], claims)
            for claim in claims:
                for asserted in (False, True):
                    with self.subTest(claim=claim["id"], observed=observed, asserted=asserted):
                        raw = observation(claim["id"], asserted=asserted)
                        before = deepcopy(raw)
                        result = check(raw, media)
                        self.assertEqual(result["status"], "consistent" if observed == asserted else "conflict")
                        evidence = result["numeric_evidence"]
                        self.assertEqual(evidence["claim"], claim["id"])
                        self.assertIs(evidence["asserted"], asserted)
                        self.assertIs(evidence["observed"], observed)
                        self.assertEqual(evidence["cited_steps"], [1, 3])
                        self.assertEqual(evidence["evidence_refs"], raw["evidence_refs"])
                        self.assertNotIn("displayed_steps", evidence)
                        self.assertFalse({"start", "end", "category", "visibility"} & set(raw))
                        self.assertEqual(raw, before)

    def test_missing_measurements_do_not_satisfy_negative_assertions(self):
        cases = [
            (stair_numeric_check(identity=IDENTITY, control_step=[1, 3],
                                 base_pos_w=None, base_quat_w=None), STAIR_MEASUREMENT_CLAIMS),
            (push_numeric_check(identity=IDENTITY, control_step=[1, 3], pose=None,
                                planar_motion=None, recovery_height_m=.5), push_measurement_claims(.5)),
            (stair_numeric_check(identity=IDENTITY, control_step=[1],
                                 base_pos_w=[[0, 0, 0]], base_quat_w=[[1, 0, 0, 0]]), STAIR_MEASUREMENT_CLAIMS),
            (push_numeric_check(identity=IDENTITY, control_step=[1], pose=[[.8]],
                                planar_motion=[[0, 0, 0, 0]], recovery_height_m=.5), push_measurement_claims(.5)),
        ]
        for check, claims in cases:
            for claim in claims:
                for asserted in (False, True):
                    with self.subTest(claim=claim["id"], asserted=asserted):
                        self.assertIsNone(check(observation(claim["id"], asserted=asserted), window([1, 3], claims)))

    def test_claim_requires_exact_object_and_boolean_without_legacy_strings(self):
        cases = [
            (stair_numeric_check(identity=IDENTITY, control_step=[1, 3],
                                 base_pos_w=[[0, 0, 0], [.01, 0, 0]],
                                 base_quat_w=[[1, 0, 0, 0], [1, 0, 0, 0]]), STAIR_MEASUREMENT_CLAIMS),
            (push_numeric_check(identity=IDENTITY, control_step=[1, 3], pose=[[.8], [.7]],
                                planar_motion=[[0, 0, 0, 0], [0, 0, .01, 0]],
                                recovery_height_m=.5), push_measurement_claims(.5)),
        ]
        for check, claims in cases:
            media = window([1, 3], claims)
            claim_id = claims[0]["id"]
            for invalid in (claim_id, None, {}, {"id": claim_id}, {"id": claim_id, "asserted": 0},
                            {"id": claim_id, "asserted": "false"},
                            {"id": claim_id, "asserted": False, "extra": "field"},
                            {"id": [claim_id], "asserted": False}, {"id": "unknown", "asserted": False}):
                with self.subTest(claim=claim_id, invalid=invalid):
                    raw = observation(claim_id)
                    raw["numeric_claim"] = invalid
                    self.assertIsNone(check(raw, media))
            raw = observation(claim_id)
            del raw["numeric_claim"]
            self.assertIsNone(check(raw, media))

    def test_stair_attachment(self):
        test_stair_attachment_preserves_numeric_and_replaces_gap()

    def test_push_attachment(self):
        test_push_attachment_requires_verified_actual_run()

    def test_stair_s0_reference_and_missing_fields(self):
        media = window([1, 3], STAIR_MEASUREMENT_CLAIMS, views=("side", "front"))
        kwargs = dict(identity=IDENTITY, control_step=[1, 2, 3],
                      base_pos_w=[[100, 8, 10], [104, 8, 11], [100.01, 8, 10.5]],
                      base_quat_w=[[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        scalars = stair_s0_for_window(media, terrain_origin_w=[100, 8, 9], **kwargs)
        values = {(row["name"], row["step"]): row["value"] for row in scalars}
        self.assertEqual(len(scalars), 6)  # one scalar per step, not per camera
        self.assertEqual(values[("root_terrain_x", 1)], 0)
        self.assertAlmostEqual(values[("root_terrain_x", 3)], .01)
        self.assertEqual(values[("root_terrain_z", 1)], 1)
        self.assertEqual(values[("root_up_world_z", 3)], 1)
        self.assertFalse(any(row["step"] == 2 for row in scalars))
        without_origin = stair_s0_for_window(media, **kwargs)
        self.assertEqual({row["name"] for row in without_origin}, {"root_up_world_z"})
        no_quat = stair_s0_for_window(media, **{**kwargs, "base_quat_w": None}, terrain_origin_w=[100, 8, 9])
        self.assertEqual({row["name"] for row in no_quat}, {"root_terrain_x", "root_terrain_z"})
        self.assertEqual(stair_s0_for_window(media, **{**kwargs, "control_step": None}), ())
        with self.assertRaisesRegex(ValueError, "terrain origin"):
            stair_s0_for_window(media, terrain_origin_w=[0, 0], **kwargs)
        # A fixed origin cancels: endpoint checking need not fabricate absolute S0.
        checker = stair_numeric_check(**kwargs)
        self.assertEqual(checker(observation("root_small_x_endpoint_delta"), media)["status"], "consistent")
        for row in scalars:
            self.assertEqual(set(row), {"name", "unit", "body", "reference_frame", "source", "step", "value"})
            self.assertEqual(row["body"], "root")

    def test_stair_asset_bound_forward_s0_and_blind_definition(self):
        configs = Path(__file__).resolve().parents[2] / "configs/robots"
        profile = json.loads((configs / "unitree_go2_12dof_isaaclab_v1.json").read_text())
        spec = json.loads((configs / "unitree_go2_12dof_isaaclab_v1.body_frame.json").read_text())
        projection = {key: spec[key] for key in ("robot_profile_id", "robot_profile_hash", "kinematic_base_source")}
        projection["resolved_roles"] = {"base": ["base"]}
        frame = bind_stair_root_frame(spec, runtime_projection=projection, robot_profile=profile)
        identity = {**IDENTITY, "robot_id": spec["robot_profile_id"]}
        media = window([99, 196], STAIR_MEASUREMENT_CLAIMS, identity)
        # Preserved development quaternions: front axis rises, despite negative qy.
        quat = [[0.9819588661193848, -0.022247588261961937, -0.17556169629096985, 0.06663438677787781],
                [0.9632313251495361, 0.019036561250686646, -0.248256653547287, 0.1009528785943985]]
        kwargs = dict(identity=identity, control_step=[99, 196], base_pos_w=None, base_quat_w=quat)
        scalars = stair_s0_for_window(media, root_frame=frame, **kwargs)
        forward = [row for row in scalars if row["name"] == "root_forward_world_z"]
        np.testing.assert_allclose([row["value"] for row in forward], [0.341823819645508, 0.48210076206067587], rtol=0, atol=1e-12)
        ready = replace(media, s0=scalars, s0_definitions=stair_s0_definitions(frame))
        prepared = prepare_observer_request([ready], model="offline-fixture")
        public = json.loads(prepared["payload"]["messages"][1]["content"][0]["text"])
        definition = public["s0_definitions"][-1]
        self.assertIn("root=base", definition["source"])
        self.assertIn("forward=[1.0, 0.0, 0.0]", definition["source"])
        self.assertIn("up=[0.0, 0.0, 1.0]", definition["source"])
        self.assertIn("positive points upward", definition["reference_frame"])
        definitions = {row["name"]: row for row in public["s0_definitions"]}
        self.assertEqual([{**definitions[row["name"]], **row} for row in public["s0"]], list(scalars))
        for private in (spec["robot_profile_id"], spec["robot_profile_hash"], spec["root_asset"]["sha256"]):
            self.assertNotIn(private, json.dumps(public))
        self.assertFalse(any(row["name"] == "root_forward_world_z" for row in stair_s0_for_window(media, **kwargs)))
        self.assertEqual(stair_s0_for_window(media, root_frame=frame, **{**kwargs, "base_quat_w": None}), ())
        self.assertIsNone(bind_stair_root_frame(None, runtime_projection=projection, robot_profile=profile))
        for changed in ({**spec, "robot_profile_hash": "0" * 64},
                        {**spec, "root_asset": {**spec["root_asset"], "sha256": "0" * 64}}):
            with self.assertRaisesRegex(ValueError, "profile|asset hash"):
                bind_stair_root_frame(changed, runtime_projection=projection, robot_profile=profile)
        with self.assertRaisesRegex(ValueError, "bound articulation root"):
            bind_stair_root_frame(spec, runtime_projection={**projection, "kinematic_base_source": "body_role"}, robot_profile=profile)

    def test_stair_forward_uses_supplied_axis_and_quaternion_sign(self):
        # A synthetic +Y-forward frame proves the computation never defaults to +X.
        frame = {"robot_profile_id": IDENTITY["robot_id"], "body_name": "synthetic_base",
                 "forward_axis_body": [0., 1., 0.], "up_axis_body": [0., 0., 1.]}
        media = window([1, 2, 3], ())
        q = np.array([np.sqrt(.5), np.sqrt(.5), 0., 0.])
        scalars = stair_s0_for_window(media, identity=IDENTITY, control_step=[1, 2, 3],
            base_pos_w=None, base_quat_w=[q, -q, q * [1, -1, -1, -1]], root_frame=frame)
        forward = [row["value"] for row in scalars if row["name"] == "root_forward_world_z"]
        np.testing.assert_allclose(forward, [1., 1., -1.], rtol=0, atol=1e-12)

    def test_push_s0_uses_recorded_references_not_velocity(self):
        media = window([1, 3], push_measurement_claims(.5))
        kwargs = dict(identity=IDENTITY, control_step=[1, 2, 3],
                      pose=[[.8, .1], [.1, .2], [.7, .3]],
                      planar_motion=[[80, 90, 4, -2], [800, 900, 8, -5], [81, 91, 4.01, -2.01]])
        scalars = push_s0_for_window(media, **kwargs)
        values = {(row["name"], row["step"]): row["value"] for row in scalars}
        self.assertEqual(len(scalars), 8)
        self.assertEqual(values[("root_roll_rad", 1)], .1)
        self.assertNotIn(("root_pitch_rad", 1), values)
        self.assertEqual(values[("root_height", 1)], .8)
        self.assertEqual(values[("root_reset_x", 1)], 4)
        self.assertEqual(values[("root_reset_y", 1)], -2)
        self.assertIn("environment origin", PUSH_S0_DEFINITIONS[0]["reference_frame"])
        self.assertIn("reset", PUSH_S0_DEFINITIONS[1]["reference_frame"])
        self.assertEqual({row["name"] for row in push_s0_for_window(media, **{**kwargs, "pose": None})},
                         {"root_reset_x", "root_reset_y"})
        self.assertEqual({row["name"] for row in push_s0_for_window(media, **{**kwargs, "planar_motion": None})},
                         {"root_height", "root_roll_rad"})
        self.assertEqual(push_s0_for_window(media, **{**kwargs, "pose": None, "planar_motion": None}), ())
        # A displayed step absent from the numeric slice gets no invented scalar.
        self.assertEqual(len(push_s0_for_window(window([1, 4], push_measurement_claims(.5)), **kwargs)), 4)

    def test_s0_source_binding_and_public_whitelist(self):
        for factory, definitions, claims, kwargs in [
            (stair_s0_for_window, STAIR_S0_DEFINITIONS, STAIR_MEASUREMENT_CLAIMS,
             dict(base_pos_w=[[9, 0, 3]], base_quat_w=[[1, 0, 0, 0]], terrain_origin_w=[8, 0, 1])),
            (push_s0_for_window, PUSH_S0_DEFINITIONS, push_measurement_claims(.5),
             dict(pose=[[.8]], planar_motion=[[90, 99, 2, -1]])),
        ]:
            media = window([1], claims)
            scalars = factory(media, identity=IDENTITY, control_step=[1], **kwargs)
            ready = replace(media, s0=scalars, s0_definitions=definitions)
            request = prepare_observer_request([ready], model="offline-fixture")
            public = json.loads(request["payload"]["messages"][1]["content"][0]["text"])
            definitions_by_name = {row["name"]: row for row in public["s0_definitions"]}
            self.assertEqual([{**definitions_by_name[row["name"]], **row}
                              for row in public["s0"]], list(scalars))
            self.assertEqual(public["s0_definitions"], list(definitions))
            self.assertNotIn(IDENTITY["source_id"], json.dumps(public))
            self.assertTrue(all("threshold" not in row and "outcome" not in row for row in public["s0"]))
            for key in ("episode_id", "source_id", "numeric_episode_id", "numeric_source_id"):
                with self.subTest(factory=factory.__name__, key=key):
                    changed = replace(media, identity={**media.identity, key: "different"})
                    with self.assertRaisesRegex(ValueError, "identities differ"):
                        factory(changed, identity=IDENTITY, control_step=[1], **kwargs)
            bad = ({**scalars[0], "reference_frame": "camera projection"}, *scalars[1:])
            with self.assertRaisesRegex(ValueError, "definition whitelist"):
                prepare_observer_request([replace(ready, s0=bad)], model="offline-fixture")

    def test_cited_view_pairs_and_no_time_from_two_views(self):
        cases = [
            (stair_numeric_check(identity=IDENTITY, control_step=[1, 3],
                                 base_pos_w=[[0, 0, 0], [.01, 0, 0]],
                                 base_quat_w=[[1, 0, 0, 0], [0, 1, 0, 0]]), STAIR_MEASUREMENT_CLAIMS),
            (push_numeric_check(identity=IDENTITY, control_step=[1, 3], pose=[[.8], [.1]],
                                planar_motion=[[0, 0, 0, 0], [0, 0, .01, 0]],
                                recovery_height_m=.5), push_measurement_claims(.5)),
        ]
        for check, claims in cases:
            media = window([1, 3], claims, views=("side", "front"))
            for refs in ([{"step": 2, "view_id": "side"}], [{"step": 1, "view_id": "missing"}],
                         [{"step": 1, "view_id": "side"}, {"step": 1, "view_id": "side"}], []):
                obs = observation(claims[0]["id"], 1, 1)
                obs["evidence_refs"] = refs
                self.assertIsNone(check(obs, media))
            same_step = observation(claims[1]["id"], 1, 1, views=("side", "front"), relation="temporal_change")
            self.assertIsNone(check(same_step, media))  # one numeric endpoint, not two
            same_step.update(text="Root moves between these views.", uncertainty="sampled images")
            raw = {"episodes": [{"alias": "e1", "observations": [same_step], "not_visible": []}]}
            audit = audit_observer_output(raw, [media], numeric_check=check)
            self.assertIn("temporal_change_requires_same_view_steps", audit["observations"][0]["violations"])
            self.assertEqual(audit["observations"][0]["numeric_status"], "visual_only")
            # Same-step multiview may support a state; the unreferenced later extreme cannot.
            state = observation(claims[0]["id"], 1, 1, views=("side", "front"))
            self.assertEqual(check(state, media)["status"], "conflict")

    def test_bound_array_snapshot_and_shape_validation(self):
        pos = np.array([[0., 0, 0], [.01, 0, 0]])
        quat = np.array([[1., 0, 0, 0], [1., 0, 0, 0]])
        check = stair_numeric_check(identity=IDENTITY, control_step=[1, 3], base_pos_w=pos, base_quat_w=quat)
        pos[1, 0] = 10
        quat[1] = [0, 1, 0, 0]
        media = window([1, 3], STAIR_MEASUREMENT_CLAIMS)
        self.assertEqual(check(observation("root_up_z_negative"), media)["status"], "conflict")
        self.assertEqual(check(observation("root_small_x_endpoint_delta"), media)["status"], "consistent")
        with self.assertRaisesRegex(ValueError, "valid-step arrays"):
            stair_numeric_check(identity=IDENTITY, control_step=[1], base_pos_w=[[0, 0, 0]], base_quat_w=[[2, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "valid-step arrays"):
            push_numeric_check(identity=IDENTITY, control_step=[1, 3], pose=[[.8]], planar_motion=None, recovery_height_m=.5)


class SnapshotNumericAlignmentTests(unittest.TestCase):
    @staticmethod
    def manifest(states):
        return {"schema_version": "native_rgb_snapshots.v1", "env_index": 2, "seed": 17,
                "state_conventions": {"root_quat_w": "root link orientation in world xyzw, unit quaternion"},
                "selected_control_steps": [1, 3, 9],
                "samples": [{"control_step": step, "state_before": deepcopy(state),
                             "state_after": deepcopy(state), "views": [{"view_id": "v0"}, {"view_id": "v1"}]}
                            for step, state in states]}

    @staticmethod
    def fields(result, sample=0):
        return {row["field"]: row for row in result["samples"][sample]["fields"]}

    def stair_fixture(self):
        # NPZ ordering differs from native articulation ordering; native has 8 joints.
        states = [(1, {"root_pos_w": [3., 4., 5.], "root_quat_w": [0., 0., 0., 1.],
                       "joint_pos": [float(v) for v in range(8)]}),
                  (3, {"root_pos_w": [6., 7., 8.], "root_quat_w": [1., 0., 0., 0.],
                       "joint_pos": [float(v + 10) for v in range(8)]})]
        indices = [4, 1, 7, 0, 6, 2, 5, 3]
        kwargs = dict(control_step=[1, 3], episode_length=np.array([1, 3]),
                      base_pos_w=np.array([[3, 4, 5], [6, 7, 8]], dtype=np.float32),
                      base_quat_w=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
                      joint_pos=np.array([np.arange(8)[indices], (np.arange(8)+10)[indices]], dtype=np.float32),
                      joint_ids=indices, kinematic_base_source="articulation_root")
        return self.manifest(states), kwargs

    def test_stair_actual_samples_step_lookup_and_joint_binding(self):
        manifest, kwargs = self.stair_fixture()
        result = align_stair_snapshot_samples(manifest, **kwargs)
        self.assertEqual(result["status"], "overlap_aligned")
        self.assertEqual([s["control_step"] for s in result["samples"]], [1, 3])  # selected 9 was never captured
        self.assertTrue(all(f["max_abs_error"] == 0 for s in result["samples"] for f in s["fields"]))
        # Row lookup uses explicit control_step, not sample index or step-1.
        reordered = {**kwargs, "control_step": [3, 1]}
        for key in ("episode_length", "base_pos_w", "base_quat_w", "joint_pos"):
            reordered[key] = kwargs[key][::-1]
        self.assertEqual(align_stair_snapshot_samples(manifest, **reordered)["status"], "overlap_aligned")
        self.assertEqual(self.fields(result)["joint_pos"]["comparison"], "dtype_exact")

    def test_stair_wrong_row_quaternion_order_and_joint_order_detected(self):
        manifest, kwargs = self.stair_fixture()
        variants = [
            ("base_pos_w", {**kwargs, "base_pos_w": kwargs["base_pos_w"][::-1]}),
            ("base_quat_w", {**kwargs, "base_quat_w": kwargs["base_quat_w"][:, [1, 2, 3, 0]]}),
            ("joint_pos", {**kwargs, "joint_ids": list(range(8))}),
            ("episode_length", {**kwargs, "episode_length": np.array([3, 1])}),
        ]
        for field, changed in variants:
            with self.subTest(field=field):
                result = align_stair_snapshot_samples(manifest, **changed)
                self.assertEqual(result["status"], "mismatch")
                self.assertEqual(self.fields(result)[field]["status"], "mismatch")
        # A one-ULP mismatch is not swallowed by a generic absolute tolerance.
        shifted = kwargs["base_pos_w"].copy()
        shifted[0, 0] = np.nextafter(shifted[0, 0], np.float32(np.inf))
        result = align_stair_snapshot_samples(manifest, **{**kwargs, "base_pos_w": shifted})
        self.assertEqual(self.fields(result)["base_pos_w"]["status"], "mismatch")
        self.assertGreater(self.fields(result)["base_pos_w"]["max_abs_error"], 0)

    def test_stair_missing_data_and_binding_never_pass(self):
        manifest, kwargs = self.stair_fixture()
        for missing in ("base_pos_w", "base_quat_w", "joint_pos", "joint_ids", "episode_length", "kinematic_base_source"):
            with self.subTest(missing=missing):
                result = align_stair_snapshot_samples(manifest, **{**kwargs, missing: None})
                self.assertEqual(result["status"], "incomplete")
                self.assertIn("missing", [r["status"] for r in result["samples"][0]["fields"]])
        self.assertEqual(align_stair_snapshot_samples(manifest, control_step=[])["status"], "incomplete")
        self.assertEqual(align_stair_snapshot_samples({"samples": []}, **kwargs)["status"], "incomplete")
        result = align_stair_snapshot_samples(manifest, **{**kwargs, "kinematic_base_source": "a_body"})
        self.assertEqual(self.fields(result)["base_pos_w"]["status"], "unsupported")

    def test_stair_recorded_joint_names_resolve_permutation_and_check_explicit_ids(self):
        manifest, kwargs = self.stair_fixture()
        manifest["schema_version"] = "native_rgb_snapshots.v2"
        native_names = [f"joint_{i}" for i in range(8)]
        manifest["articulation_joint_names"] = native_names
        numeric_names = [native_names[i] for i in kwargs["joint_ids"]]
        named = {**kwargs, "joint_ids": None, "numeric_joint_names": numeric_names}
        result = align_stair_snapshot_samples(manifest, **named)
        self.assertEqual(result["status"], "overlap_aligned")
        joints = self.fields(result)["joint_pos"]
        self.assertEqual(joints["joint_ids"], kwargs["joint_ids"])
        self.assertEqual(joints["joint_binding_source"], "recorded_joint_names")
        self.assertEqual(align_stair_snapshot_samples(manifest, **kwargs,
                         numeric_joint_names=numeric_names)["status"], "overlap_aligned")
        wrong = {**named, "joint_ids": list(range(8))}
        result = align_stair_snapshot_samples(manifest, **wrong)
        self.assertEqual(self.fields(result)["joint_pos"]["status"], "mismatch")
        self.assertIn("disagree", self.fields(result)["joint_pos"]["reason"])

    def test_stair_named_binding_missing_names_are_incomplete_and_bad_counts_fail(self):
        manifest, kwargs = self.stair_fixture()
        native_names = [f"joint_{i}" for i in range(8)]
        numeric_names = [native_names[i] for i in kwargs["joint_ids"]]
        named = {**kwargs, "joint_ids": None, "numeric_joint_names": numeric_names}
        for actual_names, wanted_names, status in (
                (None, numeric_names, "incomplete"),
                (native_names, None, "incomplete"),
                (["missing_joint", *native_names[1:]], numeric_names, "incomplete"),
                ([native_names[1], *native_names[1:]], numeric_names, "mismatch"),
                (native_names, numeric_names[:-1], "mismatch"),
                (native_names[:-1], numeric_names, "mismatch")):
            with self.subTest(native=actual_names, numeric=wanted_names):
                changed = {**manifest, "articulation_joint_names": actual_names}
                result = align_stair_snapshot_samples(changed, **{**named, "numeric_joint_names": wanted_names})
                self.assertEqual(result["status"], status)
                self.assertNotEqual(self.fields(result)["joint_pos"]["status"], "aligned")

    def push_fixture(self):
        # Nonzero roll and pitch make incorrect quaternion layouts observable.
        r, p = np.float32(.4), np.float32(-.3)
        cr, sr, cp, sp = np.cos(r/2), np.sin(r/2), np.cos(p/2), np.sin(p/2)
        quat = np.array([sr*cp, cr*sp, -sr*sp, cr*cp], dtype=np.float32)
        state = {"root_pos_w": [100., 200., 300.], "root_quat_w": quat.tolist(),
                 "root_lin_vel_w": [3., 4., 7.], "root_ang_vel_w": [0., 0., 2.],
                 "joint_pos": [7., 8.]}
        # Unrelated height/reset data cannot be validated from snapshot root_pos alone.
        return self.manifest([(3, state)]), dict(control_step=[1, 3],
            pose=np.array([[.8, 0, 0, 0, 0, 4], [.7, r, p, 5, 2, 99]], dtype=np.float32),
            planar_motion=np.array([[0, 0, 70, 80], [3, 4, 90, 100]], dtype=np.float32))

    def test_push_overlap_uses_named_velocity_layout_and_only_real_samples(self):
        manifest, kwargs = self.push_fixture()
        result = align_push_snapshot_samples(manifest, **kwargs)
        self.assertEqual(result["status"], "overlap_aligned")
        self.assertEqual(len(result["samples"]), 1)
        self.assertEqual(result["samples"][0]["numeric_row"], 1)
        fields = self.fields(result)
        self.assertEqual(fields["planar_velocity_xy"]["max_abs_error"], 0)
        self.assertEqual(fields["planar_speed"]["atol"], 1e-5)
        self.assertEqual(fields["angular_speed"]["rtol"], 0)
        self.assertIn("overlap_only", result["scope"])
        self.assertEqual({x["field"] for x in result["not_checked"]},
                         {"height", "reset_relative_xy_and_drift", "full_root_pose_and_joints", "PNG pixels"})

    def test_push_wrong_quaternion_velocity_or_rows_are_mismatches(self):
        manifest, kwargs = self.push_fixture()
        wrong_q = deepcopy(manifest)
        q = wrong_q["samples"][0]["state_before"]["root_quat_w"]
        wrong_q["samples"][0]["state_before"]["root_quat_w"] = [q[3], q[0], q[1], q[2]]
        self.assertEqual(align_push_snapshot_samples(wrong_q, **kwargs)["status"], "mismatch")
        wrong_velocity = deepcopy(manifest)
        st = wrong_velocity["samples"][0]["state_before"]
        st["root_lin_vel_w"], st["root_ang_vel_w"] = st["root_ang_vel_w"], st["root_lin_vel_w"]
        result = align_push_snapshot_samples(wrong_velocity, **kwargs)
        self.assertEqual(self.fields(result)["planar_velocity_xy"]["status"], "mismatch")
        self.assertEqual(self.fields(result)["angular_speed"]["status"], "mismatch")
        self.assertEqual(align_push_snapshot_samples(manifest, **{**kwargs, "pose": kwargs["pose"][::-1]})["status"], "mismatch")
        wrong_columns = kwargs["planar_motion"][:, [2, 3, 0, 1]]
        self.assertEqual(align_push_snapshot_samples(manifest, **{**kwargs, "planar_motion": wrong_columns})["status"], "mismatch")

    def test_push_missing_fields_and_tolerances(self):
        manifest, kwargs = self.push_fixture()
        for missing in ("pose", "planar_motion"):
            result = align_push_snapshot_samples(manifest, **{**kwargs, missing: None})
            self.assertEqual(result["status"], "incomplete")
        missing_native = deepcopy(manifest)
        del missing_native["samples"][0]["state_before"]["root_ang_vel_w"]
        self.assertEqual(align_push_snapshot_samples(missing_native, **kwargs)["status"], "incomplete")
        self.assertEqual(align_push_snapshot_samples(manifest, control_step=[])["status"], "incomplete")
        close_pose = kwargs["pose"].copy()
        close_pose[1, 3] += np.float32(5e-6)
        self.assertEqual(align_push_snapshot_samples(manifest, **{**kwargs, "pose": close_pose})["status"], "overlap_aligned")
        close_pose[1, 3] += np.float32(2e-5)
        self.assertEqual(align_push_snapshot_samples(manifest, **{**kwargs, "pose": close_pose})["status"], "mismatch")
        close_motion = kwargs["planar_motion"].copy()
        close_motion[1, 0] = np.nextafter(close_motion[1, 0], np.float32(np.inf))
        result = align_push_snapshot_samples(manifest, **{**kwargs, "planar_motion": close_motion})
        self.assertEqual(self.fields(result)["planar_velocity_xy"]["status"], "mismatch")
