"""Discriminating tests for the independent output-to-frozen-fact verifier."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("trajectory_verifier", ROOT / "scripts/verify_stair_trajectory_evidence.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
TOLERANCE = {"absolute": 1e-6, "relative": 1e-6}


def interval(first, stop, state, left=False, right=False):
    return {"steps_half_open": [first, stop], "state": state, "sample_count": stop - first,
            "sample_cell_duration_proxy_s": (stop - first) * .02,
            "observed_span_s": (stop - first - 1) * .02,
            "left_censored": left, "right_censored": right,
            "onset_range_s": None if left else [(first - 1) * .02, first * .02],
            "end_range_s": None if right else [(stop - 1) * .02, stop * .02],
            "complete_run": not left and not right,
            "physical_continuity": "unknown", "boundary_clock": "recorder_readout"}


def marker_output():
    key = "robot_0.support.FL_foot.marker"
    header = {"entities": {"support": ["robot_0.support." + s for s in verifier.SUPPORT], "forbidden": ["robot_0.body.base"]}, "coverage": {}}
    records = [{"record_id": "m", "record_type": "measurement", "key": key,
                "value": {"true_count": 1, "false_count": 2, "fraction": 1 / 3}}]
    for i, (first, stop, state, left, right) in enumerate(((1, 2, "false", True, False), (2, 3, "true", False, False), (3, 4, "false", False, True))):
        records.append({"record_id": f"s{i}", "record_type": "state_interval", "key": key,
                        "value": interval(first, stop, state, left, right)})
    for i, after in ((2, True), (3, False)):
        records.append({"record_id": f"e{i}", "record_type": "event", "key": key + ".transition", "value": {"step": i, "after": after}})
    return {"header": header, "records": records}


class ComparisonTests(unittest.TestCase):
    def test_integer_and_boolean_values_never_use_float_tolerance(self):
        for expected, actual in ((10000000, 10000001), (3, 3.0), (True, 1), (None, False)):
            changes, _ = verifier.compare_value(expected, actual, TOLERANCE)
            self.assertEqual(changes[0]["status"], "fail")

    def test_float_rounding_is_reported_and_outside_tolerance_fails(self):
        changes, _ = verifier.compare_value(.123456, .1234564, TOLERANCE)
        self.assertEqual(changes[0]["status"], "accepted_roundoff")
        changes, _ = verifier.compare_value(.123456, .123460, TOLERANCE)
        self.assertEqual(changes[0]["status"], "fail")

    def test_missing_field_is_unsupported_even_if_other_values_match(self):
        result = verifier.result("fact", {"a": 1, "b": False}, {"a": 1}, TOLERANCE, ["id"])
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["differences"][0]["path"], "value.b")
        self.assertEqual(result["record_ids"], ["id"])

    def test_short_contact_and_both_censored_sides_are_preserved(self):
        view = verifier.EpisodeView(marker_output())
        value = view.mapping("FL_foot.markers", {}, compact=False)
        self.assertEqual(value["observed_onset_steps"], [2])
        self.assertEqual(value["observed_offset_steps"], [3])
        self.assertEqual(value["on_runs"][0]["sample_count"], 1)
        self.assertEqual(value["on_runs"][0]["observed_endpoint_span_s"], 0)
        self.assertTrue(value["off_runs"][0]["left_censored"])
        self.assertTrue(value["off_runs"][1]["right_censored"])
        self.assertIsNone(value["off_runs"][1]["physical_continuous_duration_s"])
        self.assertEqual(view.used, {"m", "s0", "s1", "s2", "e2", "e3"})

    def test_deleted_short_transition_and_wrong_censor_fail(self):
        original = marker_output()
        expected = verifier.EpisodeView(original).mapping("FL_foot.markers", {}, compact=False)
        original["records"] = [r for r in original["records"] if r["record_id"] != "e2"]
        original["records"][1]["value"]["left_censored"] = False
        actual = verifier.EpisodeView(original).mapping("FL_foot.markers", {}, compact=False)
        changes, _ = verifier.compare_value(expected, actual, TOLERANCE)
        failures = {x["path"] for x in changes if x["status"] == "fail"}
        self.assertIn("value.observed_onset_steps", failures)
        self.assertIn("value.off_runs[0].left_censored", failures)

    def test_role_order_rejected_instead_of_relabeling(self):
        output = marker_output()
        output["header"]["entities"]["support"].reverse()
        with self.assertRaisesRegex(ValueError, "role order"):
            verifier.EpisodeView(output)

    def test_same_display_name_does_not_merge_sources(self):
        first = {"source": {"manifest_sha256": "a"}, "run_index": 1, "env_index": 60, "display_name": "e60"}
        second = {**first, "source": {"manifest_sha256": "b"}}
        self.assertNotEqual(verifier._identity(first), verifier._identity(second))

    def test_source_identity_rejects_float_and_boolean_indices(self):
        for run, env in ((1.0, 60), (True, 60), (1, 60.0)):
            with self.assertRaisesRegex(ValueError, "exact integers"):
                verifier._identity({"source": {"manifest_sha256": "a"}, "run_index": run, "env_index": env})

    def test_threshold_source_is_read_from_record_not_inferred_from_magnitude(self):
        output = marker_output()
        key = "robot_0.support.FL_foot.low_speed_off"
        record = {"record_id": "low", "record_type": "measurement", "key": key,
                  "measurement_config": {"low_horizontal_speed_mps": .05,
                                         "low_speed_and_progress_threshold_source": "task_spec",
                                         "definition": "M04 XYZ predicate"},
                  "value": {"longest": None}}
        output["records"].append(record)
        value = verifier.EpisodeView(output).mapping("FL_foot.low_speed_off", {}, compact=False)
        changes, _ = verifier.compare_value({"threshold_source": "development_calibration"}, value, TOLERANCE)
        self.assertTrue(any(d["path"] == "value.threshold_source" and d["status"] == "fail" for d in changes))
        record["measurement_config"]["low_speed_and_progress_threshold_source"] = "development_calibration"
        value = verifier.EpisodeView(output).mapping("FL_foot.low_speed_off", {}, compact=False)
        self.assertEqual(value["threshold_source"], "development_calibration")
        changes, _ = verifier.compare_value({"predicate_definition": "M04 XY predicate; XYZ diagnostics only"}, value, TOLERANCE)
        self.assertTrue(any(d["path"] == "value.predicate_definition" and d["status"] == "fail" for d in changes))


class InputIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def fixture(self, bad_ref=False):
        header = {"record_type": "episode", "episode_id": "ep", "run_index": 1, "env_index": 2,
                  "source": {"manifest_sha256": "source", "source_id": "sample", "acceptance_split": "public_dev", "trial_id": "trial", "checkpoint_hash": "checkpoint"},
                  "coverage": {"valid_rows": 1}}
        record = {"record_type": "measurement", "episode_id": "ep", "record_id": "record", "evidence_ref": {
            "manifest_sha256": "wrong" if bad_ref else "source", "run_index": 1, "chunks": [{"env_index": 2}]}}
        path = self.directory / "evidence.jsonl"
        path.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
        manifest = {"schema_version": "trajectory.numeric_manifest.v1", "complete": True,
                    "acceptance_split": "public_dev", "source_id": "sample", "source_manifest_ref": {"manifest_sha256": "source"},
                    "emitted_episodes": 1, "emitted_valid_rows": 1, "record_counts": {"episode": 1, "measurement": 1},
                    "evidence": {"path": "evidence.jsonl", "sha256": verifier.sha256(path), "bytes": path.stat().st_size}}
        (self.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_evidence_tamper_fails_before_numeric_mapping(self):
        path = self.fixture()
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            verifier.load_output(self.directory, {("source", 1, 2)})

    def test_record_source_splice_rejected_even_with_matching_file_hash(self):
        self.fixture(bad_ref=True)
        with self.assertRaisesRegex(ValueError, "reference differs"):
            verifier.load_output(self.directory, {("source", 1, 2)})

    def test_successful_stream_reader_preserves_selected_ids(self):
        self.fixture()
        _, selected = verifier.load_output(self.directory, {("source", 1, 2)})
        self.assertEqual(selected[("source", 1, 2)]["records"][0]["record_id"], "record")


if __name__ == "__main__":
    unittest.main()
