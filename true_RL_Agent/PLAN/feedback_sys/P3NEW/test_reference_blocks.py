"""Offline lossless presentation tests; no model calls or scientific judgments."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "reference_blocks_test_coverage", Path(__file__).with_name("confirmation_review_coverage.py")
)
coverage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage)


def text(value):
    return {"type": "text", "text": json.dumps(value, ensure_ascii=False, allow_nan=False)}


def material(identity, data, role="raw_numeric", **extra):
    # This original-file identity deliberately differs from canonical(data).
    return {"material_id": identity, "role": role, "sha256": "1" * 64, **extra, "data": data}


def request_for(materials, inventory=None, *, split=True):
    catalog = coverage.build_reference_catalog(materials, inventory)
    blocks = []
    for envelope in materials.values():
        blocks.extend(coverage.render_material_blocks(envelope, catalog) if split else [text(envelope)])
    if inventory is not None:
        blocks.append(text({"role": "review_text_inventory", "data": inventory}))
    blocks.append(text({"role": "reference_catalog", "data": catalog}))
    return {"messages": [{"role": "user", "content": blocks}]}, catalog


def block_indices(request, role):
    return [i for i, block in enumerate(request["messages"][0]["content"])
            if block["type"] == "text" and json.loads(block["text"]).get("role") == role]


def change(request, index, update):
    blocks = request["messages"][0]["content"]
    value = json.loads(blocks[index]["text"])
    update(value)
    blocks[index] = text(value)


class ReferenceBlocksTests(unittest.TestCase):
    def setUp(self):
        self.materials = {"raw": material("raw", {
            "z": {"later": {"nested": [1, 1.0, -0.0, True, False, None]}, "first": "unique-z-first"},
            "a": {"a/b": {"~key": "中文"}, "": [], "0": {}},
            "csv": "frame,force_n\n0,0.0\n1,3.5\n",
        }, source_path="original.json", extra_metadata={"retain": [True, None]})}

    def test_round_trip_preserves_full_envelope_types_and_original_tree_order(self):
        request, catalog = request_for(self.materials)
        restored, inventory = coverage.request_materials(request)
        self.assertEqual(restored, self.materials)
        self.assertIsNone(inventory)
        self.assertEqual(json.dumps(restored), json.dumps(self.materials))
        self.assertEqual(coverage.canonical(restored), coverage.canonical(self.materials))
        values = restored["raw"]["data"]["z"]["later"]["nested"]
        self.assertEqual([type(value) for value in values], [int, float, float, bool, bool, type(None)])
        indices = block_indices(request, "reference_material_block")
        rendered = [json.loads(request["messages"][0]["content"][index]["text"]) for index in indices]
        self.assertEqual([block["locator"]["json_pointer"] for block in rendered], [
            "/data/z/later", "/data/z/first", "/data/a/a~1b", "/data/a/", "/data/a/0", "/data/csv"])
        self.assertEqual(json.dumps(request).count("unique-z-first"), 1)
        for block in rendered:
            self.assertEqual(coverage.resolve_reference(block["locator"], restored), block["value"])
            self.assertEqual(coverage.normalize_reference({"anchor_id": block["anchor_id"]}, restored, catalog), block["locator"])
        header = json.loads(request["messages"][0]["content"][0]["text"])
        self.assertEqual(header["schema_version"], "p3new.reference_blocks.v2")
        self.assertEqual(header["envelope"], {key: value for key, value in self.materials["raw"].items() if key != "data"})
        self.assertEqual(header["data_canonical_sha256"], hashlib.sha256(coverage.canonical(self.materials["raw"]["data"])).hexdigest())
        self.assertNotEqual(header["data_canonical_sha256"], self.materials["raw"]["sha256"])

    def test_empty_containers_null_scalars_and_arrays_each_remain_one_complete_value(self):
        for value in ({}, [], None, False, 0, 0.0, "", [{"nested": {"deep": 4}}, None]):
            with self.subTest(value=value):
                materials = {"value": material("value", value)}
                request, _ = request_for(materials)
                self.assertEqual(len(block_indices(request, "reference_material_block")), 1)
                restored, _ = coverage.request_materials(request)
                self.assertEqual(coverage.canonical(restored), coverage.canonical(materials))

    def test_v1_catalog_and_anchor_ids_are_unchanged_by_encoding(self):
        old, old_catalog = request_for(self.materials, split=False)
        new, new_catalog = request_for(self.materials)
        self.assertEqual(coverage.request_materials(old), coverage.request_materials(new))
        self.assertEqual(old_catalog, new_catalog)
        self.assertEqual(coverage.ANCHOR_CONTRACT, "p3new.reference_anchors.v1")
        for item in new_catalog["items"]:
            self.assertEqual(item["anchor_id"], "a-" + hashlib.sha256(coverage.canonical(item["locator"])).hexdigest()[:16])
        answer = {"complete": True, "reference_catalog_sha256": hashlib.sha256(coverage.canonical(new_catalog)).hexdigest(),
                  "findings": {"necessary_facts": [{"fact_id": "F1", "evidence_refs": [{"anchor_id": new_catalog["items"][0]["anchor_id"]}]}]}}
        self.assertEqual(coverage.audit_answer(answer, old, stage="facts"), coverage.audit_answer(answer, new, stage="facts"))

    def test_report_inventory_reuse_and_nested_json_locators_survive_without_duplicate_values(self):
        materials = {**self.materials,
            "report": material("report", {"report": {"S1": "report-leaf-unique", "nested": {"S2": "nested-leaf"}}}, "automatic_report"),
            "bounded": material("bounded", {"encoded": json.dumps({"report": {"S3": "decoded-leaf"}})}, "bounded_request",
                review_text_roots=[{"decode_json_string_at": "/data/encoded", "json_pointer": "/report"}]),
        }
        inventory = coverage.build_inventory(materials)
        request, catalog = request_for(materials, inventory)
        restored, supplied = coverage.request_materials(request)
        self.assertEqual(restored, materials)
        self.assertEqual(supplied, inventory)
        self.assertEqual(coverage.build_inventory(restored), inventory)
        self.assertEqual(coverage.build_reference_catalog(restored, inventory), catalog)
        rendered = [json.loads(request["messages"][0]["content"][index]["text"])
                    for index in block_indices(request, "reference_material_block")]
        leaf = next(item for item in inventory["items"] if item["original"] == "report-leaf-unique")
        matching = [block for block in rendered if block["locator"] == leaf["locator"]]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["anchor_id"], leaf["inventory_id"])
        self.assertEqual(sum(block["value"] == "report-leaf-unique" for block in rendered), 1)
        decoded = next(item for item in inventory["items"] if item["original"] == "decoded-leaf")
        self.assertEqual(coverage.normalize_reference({"anchor_id": decoded["inventory_id"]}, restored, catalog), decoded["locator"])
        self.assertFalse(any(block["locator"] == decoded["locator"] for block in rendered))

    def test_images_and_excluded_messages_keep_original_metadata_and_png_adjacency(self):
        image = {"material_id": "image", "role": "raw_image", "sha256": "2" * 64,
                 "frame_id": "original-label", "width": 1, "height": 1}
        materials = {**self.materials, "image": image}
        for role in ("review_protocol", "focus_questions", "own_sealed_facts"):
            materials[role] = material(role, {"unmodified": ["a", None]}, role)
        request, catalog = request_for(materials)
        for envelope in list(materials.values())[1:]:
            blocks = coverage.render_material_blocks(envelope, catalog)
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks, [text(envelope)])
        png = {"type": "image_url", "image_url": {"url": "data:image/png;base64,synthetic"}}
        content = request["messages"][0]["content"]
        image_index = block_indices(request, "raw_image")[0]
        content.insert(image_index + 1, png)
        self.assertEqual(json.loads(content[image_index]["text"]), image)
        self.assertEqual(content[image_index + 1], png)
        self.assertEqual(coverage.request_materials(request)[0], materials)

    def test_missing_duplicate_and_parent_child_overlap_are_rejected(self):
        request, _ = request_for(self.materials)
        indices = block_indices(request, "reference_material_block")
        missing = copy.deepcopy(request)
        del missing["messages"][0]["content"][indices[-1]]
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            coverage.request_materials(missing)
        change(missing, 0, lambda item: item.update(block_count=item["block_count"] - 1))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            coverage.request_materials(missing)
        duplicate = copy.deepcopy(request)
        duplicate["messages"][0]["content"][indices[1]] = copy.deepcopy(duplicate["messages"][0]["content"][indices[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            coverage.request_materials(duplicate)
        overlap = copy.deepcopy(request)
        change(overlap, indices[1], lambda item: item["locator"].update(json_pointer="/data/z"))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            coverage.request_materials(overlap)

    def test_changed_data_wrong_anchor_source_or_frontier_are_rejected(self):
        request, _ = request_for(self.materials)
        index = block_indices(request, "reference_material_block")[0]
        for update in (
            lambda item: item.update(value={"changed": 999}),
            lambda item: item.update(anchor_id="a-unknown"),
            lambda item: item["locator"].update(sha256="3" * 64),
            lambda item: item["locator"].update(material_id="not-supplied"),
            lambda item: item["locator"].update(json_pointer="/data/z/later/nested"),
            lambda item: item["locator"].update(json_pointer="/data/z/~2bad"),
            lambda item: item["locator"].update(decode_json_string_at="/data/csv"),
        ):
            bad = copy.deepcopy(request)
            change(bad, index, update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                coverage.request_materials(bad)

    def test_malformed_headers_missing_catalog_and_duplicate_json_keys_are_rejected(self):
        request, _ = request_for(self.materials)
        for update in (
            lambda item: item.update(schema_version="p3new.reference_blocks.unknown"),
            lambda item: item.update(block_count=True),
            lambda item: item.update(block_count=-1),
            lambda item: item["envelope"].update(data={}),
            lambda item: item.update(data_canonical_sha256="bad"),
        ):
            bad = copy.deepcopy(request)
            change(bad, 0, update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                coverage.request_materials(bad)
        no_catalog = copy.deepcopy(request)
        no_catalog["messages"][0]["content"].pop()
        with self.assertRaises(ValueError):
            coverage.request_materials(no_catalog)
        duplicate = copy.deepcopy(request)
        block = duplicate["messages"][0]["content"][1]
        block["text"] = block["text"].replace('"value":', '"value": null, "value":', 1)
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            coverage.request_materials(duplicate)
        non_finite = copy.deepcopy(request)
        block = non_finite["messages"][0]["content"][1]
        block["text"] = block["text"].replace('"value":', '"bad": NaN, "value":', 1)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            coverage.request_materials(non_finite)

    def test_all_encodings_share_material_identity_namespace_across_messages(self):
        request, _ = request_for(self.materials)
        for duplicate in (text(self.materials["raw"]), request["messages"][0]["content"][0]):
            bad = copy.deepcopy(request)
            bad["messages"].append({"role": "user", "content": [copy.deepcopy(duplicate)]})
            with self.assertRaisesRegex(ValueError, "duplicate material identity"):
                coverage.request_materials(bad)
        orphan = copy.deepcopy(request)
        del orphan["messages"][0]["content"][0]
        with self.assertRaisesRegex(ValueError, "no preceding header"):
            coverage.request_materials(orphan)

    def test_legacy_requests_without_catalog_remain_accepted_and_missing_data_is_not_null(self):
        request = {"messages": [{"role": "user", "content": [text(value) for value in self.materials.values()]}]}
        self.assertEqual(coverage.request_materials(request), (self.materials, None))
        incomplete = material("missing", None)
        del incomplete["data"]
        with self.assertRaises(KeyError):
            coverage.render_material_blocks(incomplete, coverage.build_reference_catalog(self.materials))


if __name__ == "__main__":
    unittest.main()
