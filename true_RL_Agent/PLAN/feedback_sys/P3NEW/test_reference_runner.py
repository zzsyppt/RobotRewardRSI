"""Offline integration checks for the versioned review reference contract."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_confirmation_review as runner
import prepare_confirmation_materials as packing


class ReferenceRunnerTests(unittest.TestCase):
    reference_contract = runner.coverage.ANCHOR_CONTRACT
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.materials = []
        for name, role, data in (
            ("protocol", "review_protocol", {"rules": "unit test fixture"}),
            ("questions", "focus_questions", {"questions": ["Describe observed motion"]}),
            ("raw", "raw_numeric", {"definitions": {"geometry": {"axis": "world"}},
                                    "direct_columns": {"x": {"components": ["0,1,2"]}}}),
        ):
            self.materials.append(self.material(name, role, data))
        self.facts = self.root / "facts.json"
        runner.save(self.facts, {"schema_version": "p3new.confirmation_review_materials.v1",
            "packet_id": "fixture", "stage": "facts", "purpose": "unit_test_only",
            "reference_contract": self.reference_contract, "materials": self.materials})

    def material(self, name, role, data):
        path = self.root / (name + ".json")
        runner.save(path, data)
        return {"id": name, "role": role, "mime_type": "application/json", **runner.ref(path)}

    def prepare_facts(self, judge_id="deepseek"):
        output = self.root / "prepared"
        card = runner.prepare(self.facts, judge_id, output)
        payload = runner.read(output / "request.http_body.json")
        catalog = runner.read(output / "reference_catalog.json")
        answer = {"judge_id": judge_id, "packet_id": "fixture", "stage": "facts", "complete": True,
            "reference_catalog_sha256": card["reference_catalog"]["sha256"],
            "findings": {"necessary_facts": [{"fact_id": "F1", "statement": "fixture only",
                "evidence_refs": [{"anchor_id": catalog["items"][0]["anchor_id"]}]}]}}
        runner.save(self.root / "answer.json", answer)
        seal = {"schema_version": "p3new.confirmation_facts_seal.v1", "status": "SEALED",
            "judge_id": judge_id, "packet_id": "fixture", "facts_manifest_sha256": runner.ref(self.facts)["sha256"],
            "answer": runner.ref(self.root / "answer.json"), "reference_contract": self.reference_contract,
            "reference_catalog": runner.ref(output / "reference_catalog.json")}
        runner.save(self.root / "seal.json", seal)
        return card, payload, catalog, answer, seal

    def report_manifest(self):
        path = self.root / "report.manifest.json"
        report = self.material("report", "automatic_report", {"report": {"S1": "fixture text"}})
        packing.prepare_report_manifest(self.facts, [report], path)
        return path

    def assign_pair(self):
        path = self.root / "assignment.json"
        row = {slot: {"judge_id": judge, "model": "gpt-6.1-sol", "carrier": "fresh_subagent"}
               for slot, judge in zip(("A", "B"), runner.SOL_PAIR)}
        runner.save(path, {"schema_version": runner.ASSIGNMENT_SCHEMA, "status": "FROZEN",
                           "packets": {"fixture": row}})
        facts = runner.read(self.facts)
        facts.update(runner.assignment_fields(path, "fixture"))
        runner.save(self.facts, facts)
        return path

    def test_two_sol_identities_preserve_same_materials_without_overwriting(self):
        path = self.assign_pair()
        cards, inputs = [], []
        for judge in runner.review_judges(runner.read(self.facts)):
            output = self.root / judge
            cards.append(runner.prepare(self.facts, judge, output))
            inputs.append(runner.read(output / "request.http_body.json"))
            self.assertEqual(runner.read(output / "prepare.json")["judge_id"], judge)
        self.assertEqual(len(cards), 2)
        self.assertNotEqual(cards[0]["request_sha256"], cards[1]["request_sha256"])
        self.assertEqual(inputs[0]["messages"][1], inputs[1]["messages"][1])
        for card in cards:
            self.assertEqual(card["review_assignment"]["source"], runner.ref(path))
            self.assertEqual(card["reviewer"]["model"], "gpt-6.1-sol")

    def test_assigned_report_preserves_roster_and_rejects_other_sol_seal(self):
        self.assign_pair()
        card, _, _, _, _ = self.prepare_facts(runner.SOL_PAIR[0])
        report = self.report_manifest()
        _, _, report_card = runner.build(report, runner.SOL_PAIR[0], self.root / "seal.json")
        self.assertEqual(card["review_assignment"], report_card["review_assignment"])
        with self.assertRaisesRegex(ValueError, "another judge"):
            runner.build(report, runner.SOL_PAIR[1], self.root / "seal.json")

    def test_missing_wrong_or_changed_assignment_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "requires a frozen"):
            runner.build(self.facts, runner.SOL_PAIR[0])
        with self.assertRaises(FileNotFoundError):
            runner.assignment_fields(self.root / "missing.json", "fixture")
        path = self.assign_pair()
        with self.assertRaisesRegex(ValueError, "lacks packet"):
            runner.assignment_fields(path, "another-packet")
        with self.assertRaisesRegex(ValueError, "not assigned"):
            runner.build(self.facts, "deepseek")
        data = runner.read(path)
        data["packets"]["fixture"]["B"] = {"judge_id": "deepseek", "model": runner.MODEL, "carrier": "deepseek_api"}
        runner.save(path, data)
        with self.assertRaisesRegex(ValueError, "hash or packet row changed"):
            runner.build(self.facts, runner.SOL_PAIR[0])
        data["packets"]["fixture"]["B"]["model"] = "gpt-6.1-sol"
        runner.save(path, data)
        with self.assertRaisesRegex(ValueError, "model/carrier"):
            runner.assignment_fields(path, "fixture")

    def test_prepared_catalog_bound_and_both_judges_see_same_blocks(self):
        card, payload, catalog, answer, _ = self.prepare_facts()
        self.assertEqual(card["reference_catalog"]["sha256"], hashlib.sha256(runner.coverage.canonical(catalog)).hexdigest())
        audit = runner.coverage.audit_answer(answer, payload, stage="facts")
        self.assertTrue(audit["review_complete"], audit)
        self.assertFalse(audit["semantic_support_proven"])
        for judge_id in ("gpt6-sol", "gpt6.1-sol"):
            other, instruction, other_card = runner.build(self.facts, judge_id)
            self.assertEqual(payload["messages"][1], json.loads(other)["messages"][1])
            self.assertEqual(other_card["judge_id"], judge_id)
            self.assertIn(f"judge_id={judge_id!r}", instruction)
        rebuilt, _, rebuilt_card = runner.build(self.facts, "deepseek")
        self.assertEqual(payload, json.loads(rebuilt))
        self.assertEqual(card, rebuilt_card)

    def test_report_preserves_original_anchor_identity_and_original_answer(self):
        _, _, catalog, answer, _ = self.prepare_facts()
        report_manifest = self.report_manifest()
        body, _, card = runner.build(report_manifest, "deepseek", self.root / "seal.json")
        materials, inventory = runner.coverage.request_materials(json.loads(body))
        report_catalog = runner.coverage.request_reference_catalog(json.loads(body))
        self.assertEqual(materials["own_sealed_facts"]["data"], answer)
        self.assertTrue(all(row in report_catalog["items"] for row in catalog["items"]))
        self.assertTrue(all(any(row["anchor_id"] == item["inventory_id"] and row["locator"] == item["locator"]
                                for row in report_catalog["items"]) for item in inventory["items"]))
        self.assertEqual(card["own_facts_reference_catalog"], runner.ref(self.root / "prepared/reference_catalog.json"))

    def test_report_rejects_missing_or_rebound_facts_catalog(self):
        _, _, _, _, seal = self.prepare_facts()
        report_manifest = self.report_manifest()
        seal["reference_contract"] = None
        runner.save(self.root / "wrong_seal.json", seal)
        with self.assertRaisesRegex(ValueError, "reference contracts differ"):
            runner.build(report_manifest, "deepseek", self.root / "wrong_seal.json")
        seal["reference_contract"] = self.reference_contract
        answer = runner.read(self.root / "answer.json")
        answer["reference_catalog_sha256"] = "0" * 64
        runner.save(self.root / "answer.json", answer)
        seal["answer"] = runner.ref(self.root / "answer.json")
        runner.save(self.root / "wrong_seal.json", seal)
        with self.assertRaisesRegex(ValueError, "original facts catalog"):
            runner.build(report_manifest, "deepseek", self.root / "wrong_seal.json")

    def test_legacy_manifest_stays_without_catalog(self):
        manifest = runner.read(self.facts)
        del manifest["reference_contract"]
        runner.save(self.facts, manifest)
        body, instruction, card = runner.build(self.facts, "deepseek")
        self.assertIsNone(runner.coverage.request_reference_catalog(json.loads(body)))
        self.assertNotIn("reference_contract", card)
        self.assertIn(runner.coverage.REFERENCE_INSTRUCTION, instruction)
        self.assertNotIn("reference_catalog_sha256=", instruction)


class ReferenceBlockRunnerTests(ReferenceRunnerTests):
    reference_contract = "p3new.reference_blocks.v2"

    def test_ids_and_values_are_adjacent_and_original_tree_is_preserved(self):
        _, payload, catalog, _, _ = self.prepare_facts()
        visible, _ = runner.coverage.request_materials(payload)
        self.assertEqual(visible["raw"]["data"], runner.read(self.root / "raw.json"))
        blocks = [json.loads(b["text"]) for b in payload["messages"][1]["content"] if b["type"] == "text"]
        leaves = [b for b in blocks if b.get("role") == "reference_material_block"]
        self.assertTrue(leaves)
        by_id = {row["anchor_id"]: row["locator"] for row in catalog["items"]}
        for leaf in leaves:
            self.assertEqual(leaf["locator"], by_id[leaf["anchor_id"]])
            self.assertEqual(leaf["value"], runner.coverage.resolve_reference(leaf["locator"], visible))
        self.assertFalse(any(b.get("material_id") == "raw" for b in blocks))


if __name__ == "__main__":
    unittest.main()
