"""CPU-only transport tests: old completed probe plus neutral synthetic carriers."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import prepare_desktop_review as adapter

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1]
OLD_PROBE = RESULTS / "semantic_input_development_r1/probes/episode-13/gpt6.1-sol"
OLD_PNG = RESULTS / "n5_confirmation_execution_r2/sol_facts_media_bundled/media-04/materials/f5127e18bdfcb6c64d206287c91d1754aeb34a6ba41930551a311e67d66509c6.png"
ASSIGNMENT = adapter.REVIEW_FILE.with_name("N5_REVIEW_ASSIGNMENT_R3.json")


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = adapter.frozen_review()
        cls.png = OLD_PNG.read_bytes()
        cls.temp_parent = HERE / "test_tmp"
        cls.temp_parent.mkdir(exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_parent)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(HERE.resolve()))

    def tearDown(self):
        # The temporary target was resolved and checked under this helper directory.
        self.temp.cleanup()

    def save(self, name, value):
        path = self.root / name
        path.write_bytes(adapter.encoded(value))
        return path

    def fixtures(self):
        materials = []
        for name, role in (("protocol", "review_protocol"), ("questions", "focus_questions")):
            path = self.save(name + ".json", {"transport_test_only": True, "marker": name})
            materials.append({"id": name, "role": role, **self.review.ref(path), "mime_type": "application/json"})
        png = self.root / "original.png"
        png.write_bytes(self.png)
        # One existing original PNG repeated under neutral labels proves record order
        # and inline interleaving without adding another semantic image fixture.
        for name in ("z", "a"):
            materials.append({"id": name, "role": "raw_image", **self.review.ref(png), "mime_type": "image/png", "label": name})
        manifest = {"schema_version": "p3new.confirmation_review_materials.v1", "packet_id": "episode-02", "stage": "facts",
                    "purpose": "synthetic_transport_test_only_never_release", "max_output_tokens": 32768,
                    "materials": materials, **self.review.assignment_fields(ASSIGNMENT, "episode-02")}
        return self.save("facts.manifest.json", manifest), manifest

    def test_completed_push13_text_equivalence_and_source_unchanged(self):
        before = {name: adapter.ref(OLD_PROBE / name) for name in ("prepare.json", "request.http_body.json")}
        result = adapter.prepare(OLD_PROBE, self.root / "push13", allow_development_single=True)
        original = adapter.read(OLD_PROBE / "request.http_body.json")["messages"]
        restored = adapter.reconstruct_messages(self.root / "push13/blind/input.json")
        prior_input = adapter.read(adapter.read(OLD_PROBE / "desktop_input_preparation.json")["input_file"])
        self.assertEqual(adapter.encoded(restored), adapter.encoded(original))
        self.assertEqual(restored, prior_input["messages"])
        self.assertEqual(before, {name: adapter.ref(OLD_PROBE / name) for name in before})
        self.assertTrue(result["development_only"])
        self.assertEqual(result["actual_calls"], 0)

    def test_two_identities_image_order_and_full_roundtrip(self):
        manifest, _ = self.fixtures()
        for judge in self.review.SOL_PAIR:
            prepared = self.root / (judge + "-prepared")
            self.review.prepare(manifest, judge, prepared)
            output = self.root / judge
            result = adapter.prepare(prepared, output, assignment_file=ASSIGNMENT)
            self.assertEqual([item["material_id"] for item in result["images"]], ["z", "a"])
            self.assertEqual([item["image_index"] for item in result["images"]], [0, 1])
            source_messages = adapter.read(prepared / "request.http_body.json")["messages"]
            self.assertEqual(adapter.reconstruct_messages(output / "blind/input.json"), source_messages)
            child = adapter.read(output / "blind/input.json")
            self.assertNotIn("model", child)
            self.assertNotIn("review_assignment", child)
            self.assertNotIn("source_prepare", child)
            self.assertEqual([block["type"] for block in child["messages"][1]["content"]][-4:],
                             ["text", "image_file", "text", "image_file"])
            for item in result["images"]:
                self.assertEqual((output / "blind" / item["file"]).read_bytes(), self.png)
            with self.assertRaisesRegex(ValueError, "must be new"):
                adapter.prepare(prepared, output, assignment_file=ASSIGNMENT)
        self.assertTrue((self.root / "gpt6.1-sol-a/transport_receipt.json").is_file())
        self.assertTrue((self.root / "gpt6.1-sol-b/transport_receipt.json").is_file())

    def test_report_requires_the_same_judge_seal(self):
        facts_path, manifest = self.fixtures()
        answer_path = self.save("answer.json", {"judge_id": "gpt6.1-sol-a", "packet_id": "episode-02", "stage": "facts", "complete": True, "findings": []})
        seal = {"schema_version": "p3new.confirmation_facts_seal.v1", "status": "SEALED", "judge_id": "gpt6.1-sol-a", "packet_id": "episode-02",
                "facts_manifest_sha256": adapter.ref(facts_path)["sha256"], "answer": self.review.ref(answer_path)}
        seal_path = self.save("seal.json", seal)
        report = {**manifest, "stage": "report", "facts_manifest_sha256": seal["facts_manifest_sha256"]}
        report_path = self.save("report.manifest.json", report)
        prepared = self.root / "report-prepared"
        self.review.prepare(report_path, "gpt6.1-sol-a", prepared, seal_path)
        receipt = adapter.prepare(prepared, self.root / "report", assignment_file=ASSIGNMENT, own_facts_seal=seal_path)
        self.assertEqual(receipt["own_facts_seal_verified_local"]["sha256"], adapter.ref(seal_path)["sha256"])
        card_path, body_path = prepared / "prepare.json", prepared / "request.http_body.json"
        original_card, original_body = card_path.read_bytes(), body_path.read_bytes()
        changed = json.loads(original_body)
        for block in changed["messages"][1]["content"]:
            if block["type"] == "text":
                item = json.loads(block["text"])
                if item.get("role") == "own_sealed_facts":
                    item["data"]["findings"] = ["transport mutation"]
                    block["text"] = adapter.encoded(item).decode("utf-8")
        changed_body = adapter.encoded(changed)
        changed_card = json.loads(original_card)
        changed_card.update(request_sha256=adapter.sha(changed_body), request_bytes=len(changed_body))
        body_path.write_bytes(changed_body)
        card_path.write_bytes(adapter.encoded(changed_card))
        with self.assertRaisesRegex(ValueError, "sealed answer content differs"):
            adapter.prepare(prepared, self.root / "changed-inline", assignment_file=ASSIGNMENT, own_facts_seal=seal_path)
        card_path.write_bytes(original_card)
        body_path.write_bytes(original_body)
        wrong_seal = self.save("wrong-seal.json", {**seal, "judge_id": "gpt6.1-sol-b"})
        card = adapter.read(card_path)
        card["own_facts_seal"] = self.review.ref(wrong_seal)
        card_path.write_bytes(adapter.encoded(card))
        with self.assertRaisesRegex(ValueError, "another identity"):
            adapter.prepare(prepared, self.root / "wrong-report", assignment_file=ASSIGNMENT, own_facts_seal=wrong_seal)

    def test_request_fixture_assignment_and_identity_guards(self):
        manifest, _ = self.fixtures()
        prepared = self.root / "guards-prepared"
        self.review.prepare(manifest, "gpt6.1-sol-a", prepared)
        with self.assertRaisesRegex(ValueError, "frozen assignment"):
            adapter.prepare(prepared, self.root / "missing-assignment")
        card_path = prepared / "prepare.json"
        original = adapter.read(card_path)
        for field, value, message in (("request_sha256", "0" * 64, "SHA/size"),
                                      ("capacity_facts_fixture", {"synthetic_only": True}, "genuine prepared"),
                                      ("body_limit_bytes", 99, "ceiling"),
                                      ("judge_id", "gpt6-sol", "formal identity")):
            card_path.write_bytes(adapter.encoded({**original, field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                adapter.prepare(prepared, self.root / field, assignment_file=ASSIGNMENT)

    def test_bad_adjacent_png_hash_rejected(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": json.dumps({"material_id": "neutral", "role": "raw_image", "sha256": "0" * 64})},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(self.png).decode("ascii"), "detail": "high"}}]}]
        with self.assertRaisesRegex(ValueError, "adjacent label"):
            adapter.adapt_messages(messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
