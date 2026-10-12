"""Offline visual-observer contracts; no provider or simulator is invoked."""

from __future__ import annotations

import base64
import copy
from dataclasses import replace
import json
from pathlib import Path
import struct
import sys
import unittest
import zlib

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.visual_observer import (  # noqa: E402
    LAYOUT_VERSION,
    SCHEMA_VERSION,
    MediaWindow,
    ObserverBudget,
    audit_observer_output,
    audit_request_text,
    call_observer,
    candidate_name_entry,
    prepare_observer_request,
    render_frame_grid,
    visual_process_sentences,
)
from tests.unit.test_visual_review import synthetic_review  # noqa: E402


def window(alias="e1", steps=(10, 12), identity=None):
    frames = [(step, 4, 3, bytes((20 * index % 256, 50, 100)) * 12)
              for index, step in enumerate(steps)]
    media_identity = {"session_id": "session-private", "task_id": "task-private",
                      "robot_id": "robot-private", "protocol_id": "protocol-private",
                      "source_id": "source-private", "run_id": "private-r1",
                      "episode_id": "original-e91", "checkpoint_id": "private-ckpt"}
    media_identity.update({"numeric_" + key: value for key, value in list(media_identity.items())})
    media_identity.update(source="secret/path", success=False)
    definition = {"name": "z_axis", "unit": "unitless", "body": "root",
                  "reference_frame": "world", "source": "body_up_axis"}
    return MediaWindow(
        alias, "camera.v1", LAYOUT_VERSION, "full.v1", ("root", "leg"),
        (render_frame_grid(frames, columns=2, view_id="side"),),
        ({**definition, "step": steps[0], "value": 0.2},),
        identity=identity if identity is not None else media_identity,
        s0_definitions=(definition,),
    )


def output(**changes):
    observation = {"body": "root", "spatial_visibility": "visible",
                   "uncertainty": "only two displayed samples", "text": "The body points downward.",
                   "relation": "state", "evidence_refs": [
                       {"step": 10, "view_id": "side"}, {"step": 12, "view_id": "side"}]}
    observation.update(changes)
    return {"episodes": [{"alias": "e1", "observations": [observation],
                          "not_visible": ["The far-side leg is hidden."]}]}


def budget(**changes):
    values = {"max_pixels": 1_000_000, "max_image_bytes": 1_000_000,
              "max_cost_usd": 1.0, "max_calls": 30}
    values.update(changes)
    return ObserverBudget(**values)


class FakeTransport:
    def __init__(self, error=False, fingerprint="frozen-v1"):
        self.error = error
        self.fingerprint = fingerprint
        self.requests = []

    def post_json(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise TimeoutError("no result")
        return {"model": "configured-model", "system_fingerprint": self.fingerprint,
                "usage": {"prompt_tokens": 127, "completion_tokens": 55},
                "choices": [{"message": {"content": json.dumps(output())}}]}


class VisualObserverTests(unittest.TestCase):
    def test_provider_thinking_setting_is_sealed_and_sent(self):
        item = window()
        prepared = prepare_observer_request([item], model="configured-model",
            generation_options={"thinking": {"type": "disabled"}})
        transport = FakeTransport()
        call_observer(prepared, [item], budget=budget(), reserved_cost_usd=0.01,
            url="https://test.invalid", api_key="fixture", receipt_sink=lambda receipt: None,
            transport=transport, expected_fingerprint="frozen-v1")
        self.assertEqual(transport.requests[0]["payload"]["thinking"], {"type": "disabled"})
        with self.assertRaisesRegex(ValueError, "thinking/effort"):
            prepare_observer_request([item], model="configured-model", generation_options={"messages": []})

    def test_short_report_keeps_verbatim_conflict_and_all_output_denominator(self):
        item = replace(window(), measurement_claims=({"id": "axis_below_zero", "body": "root",
            "definition": "Minimum body-up world-z over displayed steps is below zero."},))
        raw = output()
        raw["episodes"][0]["observations"].append({**raw["episodes"][0]["observations"][0],
            "text": "Displayed body-up axis points below horizontal.",
            "numeric_claim": {"id": "axis_below_zero", "asserted": True}})
        checked = audit_observer_output(raw, [item], numeric_check=lambda observation, window:
            {"status": "conflict", "numeric_evidence": {"min": 0.2}} if observation.get("numeric_claim") else None)
        checked = synthetic_review(checked)
        before = copy.deepcopy(checked)
        rows = visual_process_sentences(checked, episode_labels={"e1": "r1.e91"},
            cohort_episode_n=12, observer_version="observer.v1", evidence_ref="observer")
        self.assertEqual(len(rows), 2)
        self.assertIn("Displayed body-up axis points below horizontal.", rows[0]["text"])
        self.assertIn("conflict", rows[0]["text"])
        self.assertIn("axis_below_zero=true: conflict (predicate only)", rows[0]["text"])
        self.assertIn("spatial: visible; sparse cited steps", rows[0]["text"])
        self.assertEqual(rows[0]["window"], {"start": 10, "end": 12})
        self.assertIn("2 statements", rows[1]["text"])
        self.assertIn("1 shown", rows[1]["text"])
        self.assertEqual(rows[0]["image_evidence"], checked["observations"][1]["image_evidence"])
        self.assertIn("observer/observations/1/image_evidence/0", rows[0]["evidence_refs"])
        self.assertEqual(checked, before)

    def test_short_report_prefers_process_over_scalar_checked_static_pose(self):
        item = replace(window(), measurement_claims=({"id": "axis_below_zero", "body": "root",
            "definition": "Minimum body-up world-z over displayed steps is below zero."},))
        raw = output(numeric_claim={"id": "axis_below_zero", "asserted": False})
        change = output(relation="temporal_change", text="The leg is more bent at step 12 than at step 10.")["episodes"][0]["observations"][0]
        raw["episodes"][0]["observations"].append(change)
        checked = audit_observer_output(raw, [item], numeric_check=lambda observation, window:
            {"status": "consistent", "numeric_evidence": {"min": 0.2}})
        checked = synthetic_review(checked)
        before = copy.deepcopy(checked)
        rows = visual_process_sentences(checked, episode_labels={"e1": "r1.e91"},
            cohort_episode_n=12, observer_version="observer.v1", evidence_ref="observer")
        self.assertEqual(rows[0]["relation"], "temporal_change")
        self.assertIn(change["text"], rows[0]["text"])
        self.assertIn("2 statements", rows[1]["text"])
        self.assertEqual(checked, before)

    def test_unqualified_normality_is_preserved_but_not_projected(self):
        checked = audit_observer_output(output(text="No anomalies were visible."), [window()])
        self.assertEqual(len(checked["observations"]), 1)
        self.assertIn("unqualified_normality", checked["observations"][0]["violations"])
        rows = visual_process_sentences(checked, episode_labels={"e1": "r1.e91"},
            cohort_episode_n=12, observer_version="observer.v1", evidence_ref="observer")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "S5")
        self.assertIn("1 mechanically invalid", rows[0]["text"])

    def test_explicit_episode_priority_precedes_temporal_and_keeps_original_evidence(self):
        raw = output(relation="temporal_change")
        second = output()["episodes"][0]
        second["alias"] = "e2"
        raw["episodes"].append(second)
        checked = synthetic_review(audit_observer_output(raw, [window(), window("e2")]))
        before = copy.deepcopy(checked)
        kwargs = dict(episode_labels={"e1": "first", "e2": "second"}, cohort_episode_n=12,
                      observer_version="observer.v1", evidence_ref="observer")
        legacy = visual_process_sentences(checked, **kwargs)
        self.assertEqual(legacy[0]["scope"]["label"], "first")
        self.assertEqual(visual_process_sentences(checked, episode_priority={}, **kwargs), legacy)
        ranked = visual_process_sentences(checked, episode_priority={"e2": 0}, **kwargs)
        self.assertEqual(ranked[0]["scope"]["label"], "second")
        self.assertEqual(ranked[0]["relation"], "state")
        self.assertIn(second["observations"][0]["text"], ranked[0]["text"])
        self.assertEqual(ranked[0]["image_evidence"], checked["observations"][1]["image_evidence"])
        self.assertIn("observer/observations/1", ranked[0]["evidence_refs"])
        self.assertEqual(ranked[1], legacy[1])
        self.assertEqual(checked, before)

    def test_episode_priority_does_not_override_conflict_or_semantic_admission(self):
        item = replace(window(), measurement_claims=({"id": "axis_below_zero", "body": "root",
            "definition": "Fixture predicate, not prose verification."},))
        raw = output(numeric_claim={"id": "axis_below_zero", "asserted": True})
        second = output(relation="temporal_change")["episodes"][0]
        second["alias"] = "e2"
        raw["episodes"].append(second)
        audit = audit_observer_output(raw, [item, window("e2")],
            numeric_check=lambda *_: {"status": "conflict"})
        kwargs = dict(episode_labels={"e1": "first", "e2": "second"}, cohort_episode_n=12,
                      observer_version="observer.v1", evidence_ref="observer")
        rows = visual_process_sentences(synthetic_review(audit), episode_priority={"e2": 0}, **kwargs)
        self.assertEqual(rows[0]["scope"]["label"], "first")
        for verdict in ("contradicted", "insufficient"):
            with self.subTest(verdict=verdict):
                reviewed = synthetic_review(audit, [verdict, "supported"])
                rows = visual_process_sentences(reviewed, episode_priority={"e1": 0}, **kwargs)
                self.assertEqual(rows[0]["scope"]["label"], "second")
        rows = visual_process_sentences(audit, episode_priority={"e1": 0}, **kwargs)
        self.assertFalse(any(row["kind"] == "S3" for row in rows))

    def test_episode_priority_only_accepts_approved_alias_integer_ranks(self):
        audit = synthetic_review(audit_observer_output(output(), [window()]))
        for priority in ({"foreign": 0}, {"e1": -1}, {"e1": True}, {"e1": 0.5}):
            with self.subTest(priority=priority), self.assertRaisesRegex(ValueError, "episode_priority"):
                visual_process_sentences(audit, episode_labels={"e1": "first"}, cohort_episode_n=1,
                    observer_version="observer.v1", evidence_ref="observer", episode_priority=priority)

    def test_invalid_containers_preserve_three_raw_observations_without_s3(self):
        for malformed in ("top_level_type", "episode_extra_field", "list_alias", "dict_alias"):
            with self.subTest(malformed=malformed):
                raw = output()
                raw["episodes"][0]["observations"] = [
                    output(text=f"The displayed body has pose {index}.")["episodes"][0]["observations"][0]
                    for index in range(3)]
                if malformed == "top_level_type":
                    raw["type"] = "json_object"
                elif malformed == "episode_extra_field":
                    raw["episodes"][0]["extra"] = "unrequested"
                else:
                    raw["episodes"][0]["alias"] = [] if malformed == "list_alias" else {}
                before = copy.deepcopy(raw)
                checked = audit_observer_output(raw, [window()])
                self.assertFalse(checked["passed"])
                self.assertEqual(len(checked["observations"]), 3)
                self.assertIs(checked["raw_output"], raw)
                self.assertEqual([record["raw_observation"] for record in checked["observations"]],
                                 before["episodes"][0]["observations"])
                if malformed == "top_level_type":
                    self.assertEqual(checked["violations"], ["output_schema"])
                    self.assertTrue(all(not record["violations"] for record in checked["observations"]))
                elif malformed == "episode_extra_field":
                    self.assertIn("episodes[0]:episode_schema", checked["violations"])
                rows = visual_process_sentences(checked, episode_labels={"e1": "r1.e91"},
                    cohort_episode_n=12, observer_version="observer.v1", evidence_ref="observer")
                self.assertEqual([row["kind"] for row in rows], ["S5"])
                self.assertIn("3 statements enumerated", rows[0]["text"])
                self.assertIn("Visual output not delivered", rows[0]["text"])
                self.assertIn(checked["violations"][0], rows[0]["text"])
                self.assertIn("0 shown", rows[0]["text"])
                self.assertEqual(rows[0]["scope"]["numerator"], 0)
                self.assertEqual(raw, before)

    def test_non_list_containers_are_not_coerced_or_allowed_to_hide_siblings(self):
        for raw in (None, [], {}, {"episodes": None}, {"episodes": {}}, {"episodes": "[]"}):
            with self.subTest(raw=raw):
                checked = audit_observer_output(raw, [window()])
                self.assertFalse(checked["passed"])
                self.assertIs(checked["raw_output"], raw)
                self.assertEqual(checked["observations"], [])
                self.assertEqual(checked["violations"], ["output_schema"])
        for invalid in (None, {}, "[]"):
            with self.subTest(observations=invalid):
                raw = output()
                raw["episodes"][0]["observations"] *= 3
                raw["episodes"].insert(0, {"alias": "e2", "observations": invalid, "not_visible": []})
                raw["episodes"].append("not an episode object")
                before = copy.deepcopy(raw)
                checked = audit_observer_output(raw, [window(), window("e2")])
                self.assertEqual(len(checked["observations"]), 3)
                self.assertIn("episodes[0]:observations_schema", checked["violations"])
                self.assertIn("episodes[2]:episode_schema", checked["violations"])
                rows = visual_process_sentences(checked, episode_labels={"e1": "r1.e91", "e2": "r1.e92"},
                    cohort_episode_n=12, observer_version="observer.v1", evidence_ref="observer")
                self.assertEqual([row["kind"] for row in rows], ["S5"])
                self.assertIn("3 statements", rows[0]["text"])
                self.assertIn("0 shown", rows[0]["text"])
                self.assertEqual(raw, before)

    def test_uncertainty_allows_120_characters_while_text_stays_at_200(self):
        for field, length, accepted in (("uncertainty", 120, True), ("uncertainty", 121, False),
                                        ("text", 200, True), ("text", 201, False)):
            with self.subTest(field=field, length=length):
                raw = output(**{field: "x" * length})
                checked = audit_observer_output(raw, [window()])
                self.assertEqual(checked["passed"], accepted)
                self.assertEqual(checked["observations"][0]["raw_observation"], raw["episodes"][0]["observations"][0])
                self.assertEqual("bounded_text" in checked["observations"][0]["violations"], not accepted)
        prepared = prepare_observer_request([window()], model="configured-model")
        instruction = prepared["payload"]["messages"][0]["content"]
        self.assertIn("only top-level key is episodes", instruction)
        self.assertIn("settings, not answer fields", instruction)
        self.assertIn("code computes bounds from references", instruction)
        self.assertIn("spatial_visibility concerns spatial detail, not time coverage", instruction)
        self.assertIn("boolean for whether that exact predicate holds at the cited steps", instruction)
        self.assertIn("at most 200 and 120 characters respectively", instruction)

    def test_measurement_claim_is_explicit_and_task_registered(self):
        item = replace(window(), measurement_claims=({"id": "axis_below_zero", "body": "root",
            "definition": "Minimum body-up world-z over displayed steps is below zero."},))
        prepared = prepare_observer_request([item], model="configured-model")
        self.assertIn(b"axis_below_zero", prepared["request_bytes"])
        calls = []
        def numeric_check(observation, selected):
            calls.append(observation["numeric_claim"])
            return {"status": "consistent", "numeric_evidence": {"min": -0.2}}
        claim = {"id": "axis_below_zero", "asserted": True}
        checked = audit_observer_output(output(numeric_claim=claim), [item], numeric_check=numeric_check)
        self.assertTrue(checked["passed"])
        self.assertEqual(calls, [claim])
        self.assertEqual(checked["observations"][0]["raw_observation"]["text"], "The body points downward.")
        for invalid in ("axis_below_zero", None, {"id": "invented", "asserted": True},
                        {"id": "axis_below_zero"}, {"id": "axis_below_zero", "asserted": 1},
                        {"id": "axis_below_zero", "asserted": "false"}):
            with self.subTest(claim=invalid):
                checked = audit_observer_output(output(numeric_claim=invalid), [item], numeric_check=numeric_check)
                self.assertFalse(checked["passed"])
                self.assertEqual(calls, [claim])
                self.assertIn("numeric_claim_definition", checked["observations"][0]["violations"])
        checked = audit_observer_output(output(), [item], numeric_check=numeric_check)
        self.assertTrue(checked["passed"])
        self.assertEqual(calls, [claim])
        self.assertEqual(checked["observations"][0]["numeric_status"], "visual_only")

    def test_actual_request_contains_images_but_no_private_identity(self):
        item = window()
        prepared = prepare_observer_request([item], model="configured-model")
        raw = prepared["request_bytes"]
        self.assertEqual(json.dumps(prepared["payload"], ensure_ascii=False).encode(), raw)
        self.assertNotIn(b"secret/path", raw)
        self.assertNotIn(b"private-r1", raw)
        self.assertNotIn(b"success", raw)
        self.assertTrue(prepared["audit"]["passed"])
        parts = json.loads(raw)["messages"][1]["content"]
        url = next(part["image_url"]["url"] for part in parts if part["type"] == "image_url")
        png = base64.b64decode(url.split(",")[1])
        self.assertEqual(png, item.images[0].data)
        image_index = next(i for i, part in enumerate(parts) if part["type"] == "image_url")
        label = json.loads(parts[image_index - 1]["text"])
        self.assertEqual(label["image_size_px"], list(struct.unpack(">II", png[16:24])))
        self.assertIn("original PNG's image_size_px [width, height]", json.loads(raw)["messages"][0]["content"])
        self.assertEqual(prepared["private_sidecar"][0]["identity"], dict(item.identity))
        self.assertEqual(prepared["private_sidecar"][0]["images"][0]["view_id"], "side")
        self.assertEqual(prepared["private_sidecar"][0]["images"][0]["displayed_steps"], [10, 12])

    def test_negative_assertion_rendering_does_not_certify_visual_prose(self):
        item = replace(window(), measurement_claims=({"id": "axis_below_zero", "body": "root",
            "definition": "Minimum body-up world-z over cited steps is below zero."},))
        raw = output(numeric_claim={"id": "axis_below_zero", "asserted": False})
        before = copy.deepcopy(raw)
        for result, expected in ((None, "not checked"), ({"status": "consistent"}, "consistent")):
            with self.subTest(numeric_status=expected):
                checked = audit_observer_output(raw, [item], numeric_check=lambda observation, media: result)
                self.assertTrue(checked["passed"])
                self.assertEqual(checked["semantic_support"], "not_evaluated")
                checked = synthetic_review(checked)
                rows = visual_process_sentences(checked, episode_labels={"e1": "r1.e91"},
                    cohort_episode_n=12, observer_version="observer.v3", evidence_ref="observer")
                self.assertIn(f"axis_below_zero=false: {expected} (predicate only)", rows[0]["text"])
                self.assertIn(before["episodes"][0]["observations"][0]["text"], rows[0]["text"])
                self.assertEqual(rows[0]["window"], checked["observations"][0]["derived_window"])
                self.assertEqual(raw, before)

    def test_outcome_tokens_are_rejected_in_public_text_and_s0_keys(self):
        original = window()
        leaking = MediaWindow(original.alias, "camera.success.v1", original.layout_version,
                              original.roi_version, original.body_names, original.images,
                              identity=original.identity)
        with self.assertRaisesRegex(ValueError, "blind observer"):
            prepare_observer_request([leaking], model="configured-model")
        leaking = MediaWindow(original.alias, "cam.v1", original.layout_version,
                              original.roi_version, original.body_names, original.images,
                              ({"name": "axis", "step": 10, "value": 0, "unit": "1", "score": 9},))
        with self.assertRaisesRegex(ValueError, "S0"):
            prepare_observer_request([leaking], model="configured-model")

    def test_json_escapes_are_scanned_after_decoding(self):
        result = audit_request_text(b'{"text":"\\u6210\\u529f"}')
        self.assertFalse(result["passed"])
        self.assertIn("成功", result["violations"])

    def test_incomplete_identity_or_foreign_numeric_input_is_rejected(self):
        for identity in ({}, dict(window().identity, numeric_run_id="other-run")):
            with self.subTest(identity=identity):
                with self.assertRaisesRegex(ValueError, "identity"):
                    prepare_observer_request([window(identity=identity)], model="configured-model")

    def test_base64_random_words_are_not_text_leaks(self):
        # "fail" is legal base64; decoded binary is not model-readable text.
        raw = b'{"image_url":{"url":"data:image/png;base64,fail"}}'
        result = audit_request_text(raw)
        self.assertTrue(result["passed"])
        self.assertEqual(result["image_proofs"][0]["bytes"], 3)
        self.assertFalse(audit_request_text(b'{"text":"fail"}')["passed"])

    def test_custom_outcome_enum_is_scanned(self):
        self.assertFalse(audit_request_text(b'{"text":"custom_end_kind"}', ["custom_end_kind"])["passed"])

    def test_budget_includes_retries_and_rejects_without_mutation(self):
        state = budget()
        media = [window()]
        for _ in range(24):
            state.reserve(media, 0.01)
        self.assertEqual((state.calls, state.images, state.subframes), (24, 24, 48))
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, "budget"):
            state.reserve(media, 0.01)
        self.assertEqual(state, before)

    def test_subframes_pixels_bytes_cost_and_episode_caps(self):
        cases = [(budget(max_pixels=1), [window()], 0),
                 (budget(max_image_bytes=1), [window()], 0),
                 (budget(max_cost_usd=0.1), [window()], 0.2),
                 (budget(), [window(f"e{index}") for index in range(5)], 0),
                 (budget(max_pixels=10_000_000), [window(steps=tuple(range(97)))], 0)]
        for state, media, cost in cases:
            with self.subTest(caps=state):
                with self.assertRaisesRegex(ValueError, "budget"):
                    state.reserve(media, cost)
                self.assertEqual(state.calls, 0)

    def test_alias_identity_cannot_change_between_calls(self):
        state = budget()
        state.reserve([window()], 0)
        with self.assertRaisesRegex(ValueError, "identity"):
            state.reserve([window(identity={"run_id": "different"})], 0)

    def test_window_and_sampling_are_derived_from_citations_not_all_input_frames(self):
        media = window(steps=(10, 11, 12))
        raw = output(spatial_visibility="visible")
        before = copy.deepcopy(raw)
        accepted = audit_observer_output(raw, [media])
        self.assertTrue(accepted["passed"])
        row = accepted["observations"][0]
        self.assertEqual(row["derived_window"], {"start": 10, "end": 12})
        self.assertEqual(row["cited_steps"], [10, 12])
        self.assertEqual(row["cited_step_gaps"], [[11, 11]])
        self.assertEqual(row["temporal_coverage"], "sparse_cited_steps")
        self.assertIn(11, media.images[0].displayed_steps)  # Uncited is not unshown.
        self.assertNotIn("unshown_ranges", row)
        self.assertEqual(raw, before)
        for steps, coverage in (((11,), "single_cited_step"), ((10, 11), "consecutive_cited_steps")):
            with self.subTest(steps=steps):
                checked = audit_observer_output(output(evidence_refs=[
                    {"step": step, "view_id": "side"} for step in steps]), [media])
                self.assertTrue(checked["passed"])
                row = checked["observations"][0]
                self.assertEqual(row["derived_window"], {"start": min(steps), "end": max(steps)})
                self.assertEqual(row["temporal_coverage"], coverage)
                self.assertEqual(row["cited_step_gaps"], [])
        outside = audit_observer_output(output(evidence_refs=[{"step": 13, "view_id": "side"}]), [media])
        self.assertIn("view_step_reference", outside["observations"][0]["violations"])
        self.assertNotIn("derived_window", outside["observations"][0])

    def test_v3_rejects_redundant_v2_fields_instead_of_repairing_them(self):
        for key, value in (("start", 10), ("end", 12), ("category", "visual_only"),
                           ("visibility", "sampled_frames_only")):
            with self.subTest(field=key):
                raw = output(**{key: value})
                checked = audit_observer_output(raw, [window()])
                self.assertFalse(checked["passed"])
                self.assertEqual(checked["observations"][0]["raw_observation"], raw["episodes"][0]["observations"][0])
                self.assertIn("observation_schema", checked["observations"][0]["violations"])
        checked = audit_observer_output(output(spatial_visibility="sampled_frames_only"), [window()])
        self.assertIn("spatial_visibility", checked["observations"][0]["violations"])

    def test_unknown_bodies_and_excess_observations_are_not_filtered(self):
        raw = output(body="unregistered")
        raw["episodes"][0]["observations"] *= 4
        result = audit_observer_output(raw, [window()])
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["observations"]), 4)
        self.assertEqual(result["raw_output"], raw)
        self.assertIn("episodes[0]:observation_count", result["violations"])
        self.assertTrue(all("unknown_body" in row["violations"] for row in result["observations"]))

    def test_task_mapping_preserves_conflict_without_rewriting_observation(self):
        raw = output(numeric_claim={"id": "axis_below_zero", "asserted": False})
        before = copy.deepcopy(raw)
        check = lambda observation, media: {"status": "conflict", "numeric_evidence": {"value": 0.9}}
        item = replace(window(), measurement_claims=({"id": "axis_below_zero", "body": "root",
            "definition": "Minimum body-up world-z over cited steps is below zero."},))
        result = audit_observer_output(raw, [item], numeric_check=check)
        self.assertTrue(result["passed"])
        row = result["observations"][0]
        self.assertEqual(row["numeric_status"], "conflict")
        self.assertEqual(row["numeric_check"]["numeric_evidence"]["value"], 0.9)
        self.assertEqual(raw, before)
        self.assertEqual(row["raw_observation"], before["episodes"][0]["observations"][0])

    def test_absent_mapping_is_visual_only_even_without_numeric_binding(self):
        result = audit_observer_output(output(), [window(identity={"is_replay": True})],
                                       numeric_check=lambda observation, media: None)
        self.assertTrue(result["passed"])
        self.assertEqual(result["observations"][0]["numeric_status"], "visual_only")
        self.assertTrue(result["observations"][0]["replay_observation"])

    def test_replay_cannot_borrow_original_episode_numeric_evidence(self):
        identity = dict(window().identity, is_replay=True, run_id="replay-r2")
        item = replace(window(identity=identity), measurement_claims=({"id": "axis_below_zero", "body": "root",
            "definition": "Minimum body-up world-z over cited steps is below zero."},))
        result = audit_observer_output(output(numeric_claim={"id": "axis_below_zero", "asserted": True}), [item],
            numeric_check=lambda observation, media: {"status": "consistent"})
        self.assertFalse(result["passed"])
        self.assertEqual(result["observations"][0]["numeric_status"], "visual_only")
        self.assertIn("numeric_identity_mismatch", result["observations"][0]["violations"])

    def test_forbidden_inferences_are_task_checked_before_numeric_mapping(self):
        result = audit_observer_output(output(), [window()],
            inference_check=lambda observation: ["task_inference_out_of_scope"])
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["observations"]), 1)

    def test_call_receipts_exact_request_usage_and_fingerprint(self):
        media = [window()]
        prepared = prepare_observer_request(media, model="configured-model")
        sink, transport = [], FakeTransport()
        receipt = call_observer(prepared, media, budget=budget(), reserved_cost_usd=0.05,
            url="https://operator-configured.invalid/chat/completions", api_key="test-key",
            receipt_sink=sink.append, transport=transport, expected_fingerprint="frozen-v1")
        self.assertEqual([row["status"] for row in sink], ["reserved", "received"])
        self.assertEqual(receipt["usage"]["prompt_tokens"], 127)
        self.assertTrue(receipt["eligible_for_confirmation"])
        self.assertEqual(transport.requests[0]["payload"], prepared["payload"])
        self.assertNotIn("test-key", json.dumps(sink))

    def test_transport_error_stays_charged_and_is_recorded(self):
        media, state, sink = [window()], budget(), []
        with self.assertRaises(TimeoutError):
            call_observer(prepare_observer_request(media, model="configured-model"), media,
                budget=state, reserved_cost_usd=0.1, url="https://operator-configured.invalid",
                api_key="test-key", receipt_sink=sink.append, transport=FakeTransport(error=True))
        self.assertEqual(state.images, 1)
        self.assertEqual(state.reserved_cost_usd, 0.1)
        self.assertEqual(sink[-1]["cost_status"], "reserved_unknown")

    def test_missing_or_changed_fingerprint_is_not_confirmation_evidence(self):
        media = [window()]
        for fingerprint in (None, "drifted-v2"):
            result = call_observer(prepare_observer_request(media, model="configured-model"), media,
                budget=budget(), reserved_cost_usd=0.1, url="https://operator-configured.invalid",
                api_key="test-key", receipt_sink=lambda record: None,
                transport=FakeTransport(fingerprint=fingerprint), expected_fingerprint="frozen-v1")
            self.assertFalse(result["eligible_for_confirmation"])

    def test_budget_media_and_prepared_request_cannot_disagree(self):
        media = [window()]
        transport = FakeTransport()
        with self.assertRaisesRegex(ValueError, "budget media"):
            call_observer(prepare_observer_request(media, model="configured-model"), [window("e2")],
                budget=budget(), reserved_cost_usd=0, url="https://operator-configured.invalid",
                api_key="test-key", receipt_sink=lambda row: None, transport=transport)
        self.assertEqual(transport.requests, [])

    def test_candidate_names_remain_proposals_and_merge_history_is_required(self):
        kwargs = dict(raw_name="unfamiliar shape", alias="e1", start=10, end=12,
                      observer_version="model.prompt.layout.v1")
        row = candidate_name_entry(**kwargs)
        self.assertEqual(row["status"], "candidate")
        self.assertIsNone(row["canonical_name"])
        with self.assertRaisesRegex(ValueError, "merge"):
            candidate_name_entry(**kwargs, canonical_name="shared name")
        row = candidate_name_entry(**kwargs, canonical_name="shared name",
                                   merge_history=[{"actor": "developer", "decision": "manual grouping"}])
        self.assertEqual(row["raw_name"], "unfamiliar shape")
        self.assertEqual(row["status"], "candidate")

    def test_grid_pixels_match_frame_order_roi_and_step_glyphs(self):
        red = bytes((255, 0, 0)) * 12
        blue = bytes((0, 0, 255)) * 12
        image = render_frame_grid([(10, 4, 3, red), (12, 4, 3, blue)], view_id="side", columns=2, roi=(1, 1, 3, 3))
        self.assertEqual(image.displayed_steps, (10, 12))
        self.assertEqual(image.view_id, "side")
        self.assertEqual((image.width, image.height), (56, 26))
        offset, compressed = 8, b""
        while offset < len(image.data):
            size = struct.unpack(">I", image.data[offset:offset + 4])[0]
            kind = image.data[offset + 4:offset + 8]
            data = image.data[offset + 8:offset + 8 + size]
            crc = struct.unpack(">I", image.data[offset + 8 + size:offset + 12 + size])[0]
            self.assertEqual(crc, zlib.crc32(kind + data))
            if kind == b"IDAT":
                compressed += data
            offset += size + 12
        decoded = zlib.decompress(compressed)
        stride = 1 + image.width * 3
        pixel = lambda x, y: decoded[y * stride + 1 + x * 3:y * stride + 4 + x * 3]
        self.assertEqual(pixel(0, 24), bytes((255, 0, 0)))
        self.assertEqual(pixel(28, 24), bytes((0, 0, 255)))
        # First digit is 1: its top middle pixel is ink, top left is white.
        self.assertEqual(pixel(5, 4), bytes((0, 0, 0)))
        self.assertEqual(pixel(2, 4), bytes((255, 255, 255)))
        # The second digit differs (0 vs 2), checking labels follow input steps.
        self.assertEqual(pixel(14, 7), bytes((0, 0, 0)))
        self.assertEqual(pixel(42, 7), bytes((255, 255, 255)))

    def test_v3_dual_view_references_bind_pixels_without_spending_extra_episodes(self):
        original = window()
        front = render_frame_grid([(10, 4, 3, bytes((200, 0, 0)) * 12),
                                   (12, 4, 3, bytes((0, 200, 0)) * 12)], view_id="front")
        item = replace(original, images=(*original.images, front))
        prepared = prepare_observer_request([item], model="configured-model")
        parts = prepared["payload"]["messages"][1]["content"]
        labels = [json.loads(part["text"]) for part in parts if part["type"] == "text"]
        self.assertEqual([label["view_id"] for label in labels if "view_id" in label], ["side", "front"])
        raw = output(spatial_visibility="visible", evidence_refs=[
            {"step": 10, "view_id": "side"}, {"step": 10, "view_id": "front"}])
        checked = audit_observer_output(raw, [item])
        self.assertTrue(checked["passed"])
        refs = checked["observations"][0]["image_evidence"]
        self.assertEqual([(ref["step"], ref["view_id"]) for ref in refs], [(10, "side"), (10, "front")])
        self.assertNotEqual(refs[0]["image_sha256"], refs[1]["image_sha256"])
        totals = budget().reserve([item], 0)
        self.assertEqual((totals["episodes"], totals["images"], totals["subframes"]), (1, 2, 4))
        self.assertEqual(SCHEMA_VERSION, "true_rl_agent.visual_observer.v3")

    def test_view_step_pair_must_exist_not_just_each_component(self):
        first = window(steps=(10,))
        second = render_frame_grid([(12, 4, 3, bytes((0, 200, 0)) * 12)], view_id="front")
        item = replace(first, images=(*first.images, second))
        for refs in ([{"step": 10, "view_id": "front"}],
                     [{"step": 12, "view_id": "invented"}],
                     [{"step": 10, "view_id": "side"}] * 2):
            raw = output(evidence_refs=refs)
            checked = audit_observer_output(raw, [item])
            self.assertIn("view_step_reference", checked["observations"][0]["violations"])
            self.assertEqual(checked["raw_output"], raw)
        checked = audit_observer_output(output(evidence_refs=[{"step": 10, "view_id": "side"}]), [window()])
        self.assertTrue(checked["passed"])
        self.assertEqual(checked["observations"][0]["derived_window"], {"start": 10, "end": 10})

    def test_temporal_change_needs_time_and_a_shared_view(self):
        original = window()
        front = replace(original.images[0], view_id="front")
        item = replace(original, images=(*original.images, front))
        for refs in (
            [{"step": 10, "view_id": "side"}, {"step": 10, "view_id": "front"}],
            [{"step": 10, "view_id": "side"}, {"step": 12, "view_id": "front"}],
        ):
            checked = audit_observer_output(output(relation="temporal_change", evidence_refs=refs), [item])
            self.assertIn("temporal_change_requires_same_view_steps", checked["observations"][0]["violations"])
        accepted = audit_observer_output(output(relation="temporal_change"), [item])
        self.assertTrue(accepted["passed"])

    def test_ambiguous_same_alias_windows_and_duplicate_view_steps_are_rejected(self):
        original = window()
        with self.assertRaisesRegex(ValueError, "one MediaWindow"):
            prepare_observer_request([original, original], model="configured-model")
        with self.assertRaisesRegex(ValueError, "step/view_id"):
            prepare_observer_request([replace(original, images=original.images * 2)], model="configured-model")
        checked = audit_observer_output(output(), [original, original])
        self.assertIn("e1:one_window_per_episode", checked["violations"])
        self.assertEqual(len(checked["observations"]), 1)

    def test_s0_whitelist_defines_unit_body_reference_frame_and_source(self):
        original = window()
        for changed in (replace(original, s0_definitions=()),
                        replace(original, s0_definitions=original.s0_definitions * 2)):
            with self.assertRaisesRegex(ValueError, "S0"):
                prepare_observer_request([changed], model="configured-model")
        for key, value in (("name", "unknown"), ("unit", "m"), ("body", "leg"),
                           ("reference_frame", "camera"), ("source", "different_measurement")):
            with self.subTest(field=key):
                scalar = {**original.s0[0], key: value}
                with self.assertRaisesRegex(ValueError, "whitelist"):
                    prepare_observer_request([replace(original, s0=(scalar,))], model="configured-model")
        definition = {"name": "custom_distance", "unit": "m", "body": "leg",
                      "reference_frame": "task_surface", "source": "task_distance_measurement"}
        custom = replace(original, s0_definitions=(definition,),
            s0=({**definition, "step": 10, "value": 0.3},))
        prepared = prepare_observer_request([custom], model="configured-model")
        public = json.loads(prepared["payload"]["messages"][1]["content"][0]["text"])
        self.assertEqual(public["s0_definitions"], [definition])

    def test_s0_binding_cannot_borrow_another_declared_source_or_episode(self):
        original = window()
        for key in ("run_id", "episode_id", "source_id", "checkpoint_id"):
            with self.subTest(field=key):
                identity = {**original.identity, "numeric_" + key: "foreign"}
                with self.assertRaisesRegex(ValueError, "numeric identity"):
                    prepare_observer_request([replace(original, identity=identity)], model="configured-model")

    def test_dual_view_subframes_stay_cumulative_across_calls(self):
        original = window(steps=tuple(range(8)))
        item = replace(original, images=(*original.images, replace(original.images[0], view_id="front")))
        state = budget()
        for _ in range(6):
            state.reserve([item], 0)
        self.assertEqual((state.images, state.subframes), (12, 96))
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, "budget"):
            state.reserve([item], 0)
        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
