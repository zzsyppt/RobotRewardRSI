"""Synthetic, offline boundary tests for the opt-in reference-anchor contract.

Run from the repository root with:
    python -m unittest PLAN.feedback_sys.P3NEW.test_reference_catalog
No formal answers, observer materials, model APIs, or experiment files are read.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "reference_catalog_test_coverage", Path(__file__).with_name("confirmation_review_coverage.py")
)
coverage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def material(identity, role, data):
    return {"material_id": identity, "role": role, "sha256": digest(data), "data": data}


def fixture_materials():
    return {
        "definition": material("definition", "task_definition", {
            "definitions": {
                "geometry": {"step_height_m": 0.2},
                "contact_history": {"frames": [{"contact": True}, {"contact": False}]},
                "protocol": {"episode_outcomes": {
                    "success": {"required_height_m": 0.4},
                    "failure_priority": ["fall", "timeout"],
                }},
                "forbidden_force_norm_n": "A force magnitude in newtons.",
                "escaped": {"a/b": {"~key": "escaped value"}},
            },
        }),
        "raw": material("raw", "raw_numeric", {
            "direct_columns": {
                "forbidden_force_norm_n": [0.0, 3.5],
                "success": [False, True],
            },
            "csv": "frame,force_n\n0,0.0\n1,3.5\n",
            "empty": {},
        }),
        "other": material("other", "raw_numeric", {"secret": "another material"}),
        "image": {"material_id": "image", "role": "raw_image", "sha256": "1" * 64,
                  "frame_id": "synthetic-frame-0", "width": 16, "height": 16},
    }


def request_for(materials, catalog=None, inventory=None):
    blocks = [{"type": "text", "text": canonical(value).decode()} for value in materials.values()]
    if inventory is not None:
        blocks.append({"type": "text", "text": canonical({
            "role": "review_text_inventory", "data": inventory}).decode()})
    if catalog is not None:
        blocks.append({"type": "text", "text": canonical({
            "role": "reference_catalog", "data": catalog}).decode()})
    return {"messages": [{"role": "user", "content": blocks}]}


class ReferenceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.materials = fixture_materials()

    def catalog(self, materials=None, inventory=None):
        return coverage.build_reference_catalog(
            self.materials if materials is None else materials, inventory=inventory)

    def locator(self, identity, pointer):
        return {"material_id": identity, "sha256": self.materials[identity]["sha256"],
                "json_pointer": pointer}

    def anchor(self, catalog, identity, pointer):
        matches = [item for item in catalog["items"] if item["locator"] == self.locator(identity, pointer)]
        self.assertEqual(len(matches), 1, (identity, pointer))
        return {"anchor_id": matches[0]["anchor_id"]}

    def facts_answer(self, reference, catalog=None):
        answer = {"complete": True, "findings": {"necessary_facts": [{
            "fact_id": "synthetic-fact-1", "statement": "Synthetic factual assertion.",
            "evidence_refs": [reference],
        }]}}
        if catalog is not None:
            answer["reference_catalog_sha256"] = digest(catalog)
        return answer

    def assertRejected(self, ref, catalog, materials=None):
        with self.assertRaises((ValueError, KeyError, IndexError, TypeError)):
            coverage.normalize_reference(ref, self.materials if materials is None else materials, catalog)

    def test_legacy_facts_result_is_exactly_unchanged(self):
        locator = self.locator("definition", "/data/definitions/geometry/step_height_m")
        result = coverage.audit_answer(self.facts_answer(locator), request_for(self.materials), stage="facts")
        self.assertEqual(result, {
            "stage": "facts", "mechanical_valid": True, "errors": [],
            "declared_complete": True, "review_complete": True, "references_checked": 1,
            "semantic_support_proven": False,
        })

    def test_legacy_invalid_reference_preserves_error_and_result_shape(self):
        locator = self.locator("definition", "/data/geometry")
        result = coverage.audit_answer(self.facts_answer(locator), request_for(self.materials), stage="facts")
        self.assertEqual(result, {
            "stage": "facts", "mechanical_valid": False,
            "errors": [{"answer_locator": "/findings/necessary_facts/0/evidence_refs/0",
                        "issue": "unresolvable_reference", "detail": "'geometry'"}],
            "declared_complete": True, "review_complete": False, "references_checked": 0,
            "semantic_support_proven": False,
        })

    def test_catalog_has_deterministic_frontier_and_locator_based_ids(self):
        catalog = self.catalog()
        self.assertEqual(coverage.ANCHOR_CONTRACT, "p3new.reference_anchors.v1")
        self.assertEqual(set(catalog), {"schema_version", "items"})
        self.assertEqual(catalog["schema_version"], coverage.ANCHOR_CONTRACT)
        self.assertEqual([item["anchor_id"] for item in catalog["items"]],
                         sorted(item["anchor_id"] for item in catalog["items"]))
        for item in catalog["items"]:
            self.assertEqual(set(item), {"anchor_id", "locator"})
            self.assertEqual(item["anchor_id"], "a-" + digest(item["locator"])[:16])
        expected = {
            ("definition", "/data/definitions/geometry"),
            ("definition", "/data/definitions/contact_history"),
            ("definition", "/data/definitions/protocol"),
            ("definition", "/data/definitions/forbidden_force_norm_n"),
            ("definition", "/data/definitions/escaped"),
            ("raw", "/data/direct_columns/forbidden_force_norm_n"),
            ("raw", "/data/direct_columns/success"),
            ("raw", "/data/csv"), ("raw", "/data/empty"),
            ("other", "/data/secret"), ("image", ""),
        }
        self.assertEqual({(item["locator"]["material_id"], item["locator"]["json_pointer"])
                          for item in catalog["items"]}, expected)
        reversed_materials = dict(reversed(list(self.materials.items())))
        reversed_materials["definition"] = copy.deepcopy(reversed_materials["definition"])
        definitions = reversed_materials["definition"]["data"]["definitions"]
        reversed_materials["definition"]["data"]["definitions"] = dict(reversed(list(definitions.items())))
        self.assertEqual(self.catalog(reversed_materials), catalog)

    def test_non_evidence_roles_are_not_ordinary_anchors(self):
        materials = copy.deepcopy(self.materials)
        excluded = {"review_protocol", "focus_questions", "own_sealed_facts"}
        for role in excluded:
            materials[role] = material(role, role, {"text": "Synthetic excluded material."})
        self.assertEqual(self.catalog(materials), self.catalog())

    def test_five_parent_paths_resolve_without_manually_recreating_parents(self):
        catalog = self.catalog()
        cases = [
            ("geometry", "definition", "/data/definitions/geometry", "/step_height_m", 0.2),
            ("contact_history", "definition", "/data/definitions/contact_history", "/frames/0/contact", True),
            ("forbidden_force_norm_n", "raw", "/data/direct_columns/forbidden_force_norm_n", "/1", 3.5),
            ("success", "definition", "/data/definitions/protocol", "/episode_outcomes/success/required_height_m", 0.4),
            ("failure_priority", "definition", "/data/definitions/protocol", "/episode_outcomes/failure_priority/0", "fall"),
        ]
        for field, identity, base, suffix, value in cases:
            with self.subTest(field=field):
                ref = {**self.anchor(catalog, identity, base), "within_anchor_pointer": suffix}
                normalized = coverage.normalize_reference(ref, self.materials, catalog)
                self.assertEqual(normalized, self.locator(identity, base + suffix))
                self.assertEqual(coverage.resolve_reference(normalized, self.materials), value)

    def test_unknown_anchor_and_mixed_legacy_fields_are_rejected(self):
        catalog = self.catalog()
        self.assertRejected({"anchor_id": "a-unknown"}, catalog)
        anchor = self.anchor(catalog, "definition", "/data/definitions/geometry")
        for extra in ({"material_id": "other"}, {"sha256": "0" * 64},
                      {"json_pointer": "/data/secret"}, {"decode_json_string_at": "/data/csv"},
                      {"unexpected": "ignored metadata must not pass"}):
            with self.subTest(extra=extra):
                self.assertRejected({**anchor, **extra}, catalog)
        self.assertRejected(self.locator("definition", "/data/definitions/geometry"), catalog)

    def test_local_pointer_cannot_escape_anchor_or_switch_material(self):
        catalog = self.catalog()
        anchor = self.anchor(catalog, "definition", "/data/definitions/geometry")
        for suffix in ("../other/data/secret", "/../../other/data/secret",
                       "/data/direct_columns/success", "/materials/other/data/secret"):
            with self.subTest(suffix=suffix):
                self.assertRejected({**anchor, "within_anchor_pointer": suffix}, catalog)

    def test_arrays_require_in_bounds_canonical_indices(self):
        catalog = self.catalog()
        anchor = self.anchor(catalog, "raw", "/data/direct_columns/forbidden_force_norm_n")
        for suffix in ("/2", "/-1", "/01", "/-", "/1.0", "/999999"):
            with self.subTest(suffix=suffix):
                self.assertRejected({**anchor, "within_anchor_pointer": suffix}, catalog)

    def test_csv_string_remains_atomic_and_is_not_implicitly_parsed(self):
        catalog = self.catalog()
        anchor = self.anchor(catalog, "raw", "/data/csv")
        normalized = coverage.normalize_reference(anchor, self.materials, catalog)
        self.assertEqual(coverage.resolve_reference(normalized, self.materials), self.materials["raw"]["data"]["csv"])
        self.assertRejected({**anchor, "within_anchor_pointer": "/0"}, catalog)
        self.assertRejected({**anchor, "within_anchor_pointer": "/rows/1/force_n"}, catalog)

    def test_pointer_escaping_is_strict_and_valid_escapes_work(self):
        catalog = self.catalog()
        anchor = self.anchor(catalog, "definition", "/data/definitions/escaped")
        ref = {**anchor, "within_anchor_pointer": "/a~1b/~0key"}
        normalized = coverage.normalize_reference(ref, self.materials, catalog)
        self.assertEqual(coverage.resolve_reference(normalized, self.materials), "escaped value")
        for suffix in ("/a~2b", "/a~", "/a~1b/~key"):
            with self.subTest(suffix=suffix):
                self.assertRejected({**anchor, "within_anchor_pointer": suffix}, catalog)

    def test_normalization_checks_material_sha(self):
        catalog = self.catalog()
        anchor = self.anchor(catalog, "definition", "/data/definitions/geometry")
        changed = copy.deepcopy(self.materials)
        changed["definition"]["sha256"] = "0" * 64
        self.assertRejected(anchor, catalog, changed)
        changed_catalog = copy.deepcopy(catalog)
        item = next(item for item in changed_catalog["items"] if item["anchor_id"] == anchor["anchor_id"])
        item["locator"]["sha256"] = "0" * 64
        self.assertRejected(anchor, changed_catalog)

    def test_equal_values_at_different_sources_keep_distinct_identities(self):
        materials = {"one": material("one", "raw_numeric", {"value": 7}),
                     "two": material("two", "raw_numeric", {"value": 7})}
        items = self.catalog(materials)["items"]
        self.assertEqual(len(items), 2)
        self.assertNotEqual(items[0]["anchor_id"], items[1]["anchor_id"])
        self.assertEqual({item["locator"]["material_id"] for item in items}, {"one", "two"})

    def test_request_catalog_is_optional_and_unique(self):
        catalog = self.catalog()
        self.assertIsNone(coverage.request_reference_catalog(request_for(self.materials)))
        request = request_for(self.materials, catalog)
        self.assertEqual(coverage.request_reference_catalog(request), catalog)
        request["messages"][0]["content"].append(copy.deepcopy(request["messages"][0]["content"][-1]))
        with self.assertRaises(ValueError):
            coverage.request_reference_catalog(request)
        result = coverage.audit_answer(self.facts_answer(self.anchor(catalog, "image", ""), catalog), request, stage="facts")
        self.assertFalse(result["mechanical_valid"])

    def test_catalog_hash_is_required_and_must_match(self):
        catalog = self.catalog()
        anchor = self.anchor(catalog, "definition", "/data/definitions/geometry")
        for supplied_hash in (None, "0" * 64):
            answer = self.facts_answer(anchor)
            if supplied_hash is not None:
                answer["reference_catalog_sha256"] = supplied_hash
            with self.subTest(supplied_hash=supplied_hash):
                result = coverage.audit_answer(answer, request_for(self.materials, catalog), stage="facts")
                self.assertFalse(result["mechanical_valid"])
                self.assertFalse(result["review_complete"])

    def test_catalog_rebuilt_from_actual_materials_rejects_self_consistent_drift(self):
        catalog = self.catalog()
        anchor = self.anchor(catalog, "definition", "/data/definitions/geometry")
        changed = copy.deepcopy(catalog)
        item = next(item for item in changed["items"] if item["anchor_id"] == anchor["anchor_id"])
        item["locator"]["json_pointer"] = "/data/definitions/contact_history"
        answer = self.facts_answer(anchor, changed)
        result = coverage.audit_answer(answer, request_for(self.materials, changed), stage="facts")
        self.assertFalse(result["mechanical_valid"])
        self.assertFalse(result["review_complete"])

    def test_catalog_mode_rejects_legacy_locator_even_if_it_resolves(self):
        catalog = self.catalog()
        answer = self.facts_answer(self.locator("definition", "/data/definitions/geometry"), catalog)
        result = coverage.audit_answer(answer, request_for(self.materials, catalog), stage="facts")
        self.assertFalse(result["mechanical_valid"])

    def test_valid_reference_can_pass_mechanically_with_false_semantic_claim(self):
        catalog = self.catalog()
        anchor = {**self.anchor(catalog, "definition", "/data/definitions/geometry"),
                  "within_anchor_pointer": "/step_height_m"}
        answer = self.facts_answer(anchor, catalog)
        answer["findings"]["necessary_facts"][0]["statement"] = "The step height is 999 metres."
        result = coverage.audit_answer(answer, request_for(self.materials, catalog), stage="facts")
        self.assertTrue(result["mechanical_valid"], result)
        self.assertTrue(result["review_complete"], result)
        self.assertFalse(result["semantic_support_proven"])
        self.assertEqual(result["resolved_references"], [{
            "answer_locator": "/findings/necessary_facts/0/evidence_refs/0",
            "locator": self.locator("definition", "/data/definitions/geometry/step_height_m"),
        }])

    def report_fixture(self, extra_fields=None):
        materials = copy.deepcopy(self.materials)
        report_data = json.loads(canonical({
            "report": {"observation": "Synthetic report text.", "terms": ["term one", "term two"],
                       **(extra_fields or {})}}))
        materials["report"] = material("report", "automatic_report", report_data)
        inventory = coverage.build_inventory(materials)
        catalog = self.catalog(materials, inventory)
        evidence = self.anchor(catalog, "definition", "/data/definitions/geometry")
        claims, review = [], []
        for index, item in enumerate(inventory["items"]):
            claim_id = "claim-" + str(index)
            claims.append({"claim_id": claim_id, "verdict": "unsupported",
                           "reason": "Synthetic fixture records no semantic support.",
                           "report_locator": {"anchor_id": item["inventory_id"]},
                           "evidence_refs": [evidence]})
            review.append({"inventory_id": item["inventory_id"], "status": "reviewed",
                           "claim_ids": [claim_id], "reason": "Synthetic review completed."})
        answer = {"complete": True, "reference_catalog_sha256": digest(catalog),
                  "findings": {"claims": claims, "inventory_review": review}}
        return materials, inventory, catalog, answer

    def test_report_reuses_exact_inventory_and_preserves_raw_anchor_ids(self):
        materials, inventory, catalog, answer = self.report_fixture()
        raw_items = [item for item in catalog["items"] if item["locator"]["material_id"] in self.materials]
        self.assertEqual(raw_items, self.catalog()["items"])
        report_items = [item for item in catalog["items"] if item["locator"]["material_id"] == "report"]
        self.assertEqual(report_items, sorted([
            {"anchor_id": item["inventory_id"], "locator": item["locator"]}
            for item in inventory["items"]], key=lambda item: item["anchor_id"]))
        for item in inventory["items"]:
            self.assertEqual(coverage.normalize_reference({"anchor_id": item["inventory_id"]}, materials, catalog), item["locator"])
        result = coverage.audit_answer(answer, request_for(materials, catalog, inventory), stage="report")
        self.assertTrue(result["mechanical_valid"], result)
        self.assertTrue(result["review_complete"], result)
        self.assertEqual(result["inventory_items_recorded"], len(inventory["items"]))
        self.assertFalse(result["semantic_support_proven"])
        self.assertFalse(result["atomic_claim_completeness_proven"])

    def test_string_ids_and_legacy_objects_resolve_same_original_locator(self):
        catalog = self.catalog()
        for item in catalog["items"]:
            for reference in (item["anchor_id"], {"anchor_id": item["anchor_id"]}):
                self.assertEqual(coverage.normalize_reference(reference, self.materials, catalog), item["locator"])
        reference = catalog["items"][0]["anchor_id"]
        result = coverage.audit_answer(self.facts_answer(reference, catalog),
                                       request_for(self.materials, catalog), stage="facts")
        self.assertTrue(result["mechanical_valid"], result)
        materials, inventory, catalog, answer = self.report_fixture()
        for claim in answer["findings"]["claims"]:
            claim["report_locator"] = claim["report_locator"]["anchor_id"]
            claim["evidence_refs"] = [ref["anchor_id"] for ref in claim["evidence_refs"]]
        result = coverage.audit_answer(answer, request_for(materials, catalog, inventory), stage="report")
        self.assertTrue(result["review_complete"], result)
        self.assertFalse(result["semantic_support_proven"])

    def test_string_unknown_ids_and_broad_report_anchor_still_rejected(self):
        catalog = self.catalog()
        identity = catalog["items"][0]["anchor_id"]
        for reference in ("a-unknown", "", " " + identity, identity + "/extra"):
            self.assertRejected(reference, catalog)
        materials, inventory, catalog, answer = self.report_fixture({"nested": {"text": "Synthetic leaf."}})
        leaf = next(item for item in inventory["items"]
                    if item["locator"]["json_pointer"] == "/data/report/nested/text")
        parent = next(item for item in catalog["items"]
                      if item["locator"]["material_id"] == "report"
                      and item["locator"]["json_pointer"] == "/data/report/nested")
        claim = next(claim for claim in answer["findings"]["claims"]
                     if claim["report_locator"]["anchor_id"] == leaf["inventory_id"])
        claim["report_locator"] = parent["anchor_id"]
        result = coverage.audit_answer(answer, request_for(materials, catalog, inventory), stage="report")
        self.assertFalse(result["mechanical_valid"], result)
        self.assertTrue(any(isinstance(error, dict) and error.get("issue") == "claim_must_point_to_exact_inventory_text"
                            for error in result["errors"]), result)
        claim["report_locator"] = leaf["inventory_id"]
        result = coverage.audit_answer(answer, request_for(materials, catalog, inventory), stage="report")
        self.assertTrue(result["review_complete"], result)

    def test_report_descendant_that_resolves_cannot_replace_exact_inventory_text(self):
        materials, inventory, catalog, answer = self.report_fixture()
        list_item = next(item for item in inventory["items"] if isinstance(item["original"], list))
        claim = next(claim for claim in answer["findings"]["claims"]
                     if claim["report_locator"]["anchor_id"] == list_item["inventory_id"])
        claim["report_locator"]["within_anchor_pointer"] = "/0"
        result = coverage.audit_answer(answer, request_for(materials, catalog, inventory), stage="report")
        self.assertFalse(result["mechanical_valid"])
        self.assertTrue(any(isinstance(error, dict) and error.get("issue") == "claim_must_point_to_exact_inventory_text"
                            for error in result["errors"]), result)

    def test_report_non_text_numeric_field_remains_available_as_evidence(self):
        materials, inventory, catalog, answer = self.report_fixture({"numeric_measurement": 0.75})
        locator = {"material_id": "report", "sha256": materials["report"]["sha256"],
                   "json_pointer": "/data/report/numeric_measurement"}
        self.assertFalse(any(item["locator"] == locator for item in inventory["items"]))
        matches = [item for item in catalog["items"] if item["locator"] == locator]
        self.assertEqual(len(matches), 1)
        reference = {"anchor_id": matches[0]["anchor_id"]}
        self.assertEqual(matches[0]["anchor_id"], "a-" + digest(locator)[:16])
        normalized = coverage.normalize_reference(reference, materials, catalog)
        self.assertEqual(normalized, locator)
        self.assertEqual(coverage.resolve_reference(normalized, materials), 0.75)
        answer["findings"]["claims"][0]["evidence_refs"] = [reference]
        result = coverage.audit_answer(answer, request_for(materials, catalog, inventory), stage="report")
        self.assertTrue(result["mechanical_valid"], result)
        self.assertTrue(result["review_complete"], result)
        self.assertFalse(result["semantic_support_proven"])

    def test_report_overlap_keeps_only_inventory_id_for_same_locator(self):
        materials, inventory, catalog, _ = self.report_fixture()
        ordinary_catalog = self.catalog(materials)
        locators = [canonical(item["locator"]) for item in catalog["items"]]
        self.assertEqual(len(locators), len(set(locators)))
        for item in inventory["items"]:
            with self.subTest(inventory_id=item["inventory_id"]):
                ordinary_id = "a-" + digest(item["locator"])[:16]
                self.assertIn({"anchor_id": ordinary_id, "locator": item["locator"]}, ordinary_catalog["items"])
                matches = [entry for entry in catalog["items"] if entry["locator"] == item["locator"]]
                self.assertEqual(matches, [{"anchor_id": item["inventory_id"], "locator": item["locator"]}])
                self.assertNotIn(ordinary_id, [entry["anchor_id"] for entry in catalog["items"]])

    def test_report_block_requires_local_pointer_to_exact_inventory_leaf(self):
        materials, inventory, catalog, answer = self.report_fixture({
            "nested": {"text": "Synthetic nested explanatory leaf."}})
        leaf = next(item for item in inventory["items"]
                    if item["locator"]["json_pointer"] == "/data/report/nested/text")
        block = next(item for item in catalog["items"]
                     if item["locator"]["material_id"] == "report"
                     and item["locator"]["json_pointer"] == "/data/report/nested")
        claim = next(claim for claim in answer["findings"]["claims"]
                     if claim["report_locator"]["anchor_id"] == leaf["inventory_id"])
        claim["report_locator"] = {"anchor_id": block["anchor_id"]}
        result = coverage.audit_answer(answer, request_for(materials, catalog, inventory), stage="report")
        self.assertFalse(result["mechanical_valid"])
        self.assertTrue(any(isinstance(error, dict) and error.get("issue") == "claim_must_point_to_exact_inventory_text"
                            for error in result["errors"]), result)
        claim["report_locator"]["within_anchor_pointer"] = "/text"
        normalized = coverage.normalize_reference(claim["report_locator"], materials, catalog)
        self.assertEqual(normalized, leaf["locator"])
        result = coverage.audit_answer(answer, request_for(materials, catalog, inventory), stage="report")
        self.assertTrue(result["mechanical_valid"], result)
        self.assertTrue(result["review_complete"], result)
        self.assertFalse(result["semantic_support_proven"])

    def test_duplicate_inventory_and_anchor_identities_are_rejected(self):
        materials, inventory, catalog, _ = self.report_fixture()
        duplicated_inventory = copy.deepcopy(inventory)
        duplicated_inventory["items"].append(copy.deepcopy(duplicated_inventory["items"][0]))
        with self.assertRaises(ValueError):
            coverage.build_reference_catalog(materials, inventory=duplicated_inventory)
        reference = self.anchor(catalog, "definition", "/data/definitions/geometry")
        duplicated_catalog = copy.deepcopy(catalog)
        entry = next(item for item in duplicated_catalog["items"] if item["anchor_id"] == reference["anchor_id"])
        duplicated_catalog["items"].append(copy.deepcopy(entry))
        self.assertRejected(reference, duplicated_catalog, materials)


if __name__ == "__main__":
    unittest.main()
