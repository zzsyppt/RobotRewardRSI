"""Contract routing and pre-write rejection; semantic verification stays task-owned."""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

HELPERS = Path(__file__).resolve().parents[2] / "PLAN/feedback_sys/P3NEW"
sys.path.insert(0, str(HELPERS))
try:
    import prepare_review_task_contract as contracts
    spec = importlib.util.spec_from_file_location("confirmation_contract_packer", HELPERS / "prepare_confirmation_materials.py")
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)
finally:
    sys.path.remove(str(HELPERS))


class ConfirmationTaskDefinitionTests(unittest.TestCase):
    def fixture(self, base):
        raw = base / "raw"
        sources, definitions = [], []
        for task, runs in (("stair", [1, 2, 3]), ("push", [1]), ("push", [2]), ("push", [3])):
            numeric = base / f"{task}-{runs[0]}.json"
            identity = {"task_id": "unitree_stair_climbing" if task == "stair" else "unitree_push_recovery"}
            if task == "push":
                identity["run_index"] = runs[0]
            packer.save(numeric, {"identity": identity, "metadata": {"seeds": [{"run_index": r} for r in runs]}})
            for run in runs:
                sources.append({"manifest_path": str(numeric), "manifest_sha256": packer.digest(numeric),
                                "identity": identity, "run": run})
            definition = base / f"contract-{task}-{runs[0]}.json"
            packer.save(definition, {"task": task, "numeric_manifest": {"sha256": packer.digest(numeric)}})
            definitions.append(definition)
        packer.save(raw / "_private/sources.json", {"sources": sources})
        selected = [{"episode_id": "push-example"}, {"episode_id": "stair-example"}]
        for alias, task, run in (("push-example", "push", "run_02"), ("stair-example", "stair", "run_03")):
            packer.save(raw / "episodes" / alias / "episode.json", {"episode": {"task": task, "run_group": run}})
        focuses = ["focus/push_run01.json", "focus/stair_all.json"]
        for path, task in zip(focuses, ("push", "stair")):
            packer.save(raw / path, {"task": task})
        cell = base / "cell.json"
        packer.save(cell, {})
        return raw, selected, focuses, definitions, cell

    def pack_inputs(self, raw, selected, focuses):
        protocol, card = raw.parent / "protocol.json", raw.parent / "card.json"
        packer.save(protocol, {})
        packer.save(card, {"status": "FROZEN"})
        packer.save(raw / "manifest.json", {"purpose": "confirmation", "stage": "facts",
            "private_source_manifest": {"file": "_private/sources.json", "sha256": packer.digest(raw / "_private/sources.json")},
            "frozen_inputs": {"capture_card": {"sha256": packer.digest(card)}, "review_protocol": {"sha256": packer.digest(protocol)}},
            "readable_materials": [{"file": f"focus/{i:02d}.json"} for i in range(12)]})
        packer.save(raw / "selected_episodes.json", selected * 12)
        return protocol, card

    def test_episode_own_run_focus_all_runs_and_stair_deduplication(self):
        with TemporaryDirectory() as tmp:
            raw, selected, focuses, definitions, cell = self.fixture(Path(tmp))
            # This test isolates wiring. The engineering integration uses real verify.
            with patch.object(contracts, "verify") as verify:
                routes, receipt = packer._prepare_task_definition_inputs(
                    raw, selected, focuses, definitions, cell, required=True)
            self.assertEqual(verify.call_count, 4)
            self.assertEqual([r["file"] for r in routes["episode-01"]], [str(definitions[2].resolve())])
            self.assertEqual([r["file"] for r in routes["episode-02"]], [str(definitions[0].resolve())])
            self.assertEqual([r["file"] for r in routes["focus-01"]], [str(p.resolve()) for p in definitions[1:]])
            self.assertEqual([r["file"] for r in routes["focus-02"]], [str(definitions[0].resolve())])
            self.assertEqual(len(receipt["task_run_bindings"]), 6)
            calls = {call.args[0]["task"]: call.args[-1] for call in verify.call_args_list}
            self.assertIsNone(calls["stair"])
            self.assertEqual(calls["push"], cell.resolve())

    def test_missing_or_wrong_numeric_contract_creates_no_output(self):
        with TemporaryDirectory() as tmp:
            raw, selected, focuses, definitions, cell = self.fixture(Path(tmp))
            protocol, card = self.pack_inputs(raw, selected, focuses)
            wrong = raw.parent / "unrelated-contract.json"
            packer.save(wrong, {"task": "push", "numeric_manifest": {"sha256": "0" * 64}})
            for index, paths in enumerate((None, definitions[:-1], definitions[:-1] + [wrong])):
                with self.subTest(paths=paths), patch.object(contracts, "verify") as verify:
                    output = raw.parent / f"rejected-{index}"
                    with self.assertRaisesRegex(ValueError, "requires explicit|do not cover"):
                        packer.pack(raw, output, protocol, card, task_definitions=paths, push_resolved_cell_manifest=cell)
                    self.assertFalse(output.exists())
                    verify.assert_not_called()

    def test_task_verifier_rejection_precedes_output_creation(self):
        with TemporaryDirectory() as tmp:
            raw, selected, focuses, definitions, cell = self.fixture(Path(tmp))
            protocol, card = self.pack_inputs(raw, selected, focuses)
            output = raw.parent / "rejected-source"
            with patch.object(contracts, "verify", side_effect=ValueError("numeric/source manifest drift")):
                with self.assertRaisesRegex(ValueError, "numeric/source manifest drift"):
                    packer.pack(raw, output, protocol, card, task_definitions=definitions, push_resolved_cell_manifest=cell)
            self.assertFalse(output.exists())

    def test_missing_or_wrong_export_source_binding_creates_no_output(self):
        with TemporaryDirectory() as tmp:
            raw, selected, focuses, definitions, cell = self.fixture(Path(tmp))
            protocol, card = self.pack_inputs(raw, selected, focuses)
            manifest = packer.read(raw / "manifest.json")
            for index, binding in enumerate((None, {"file": "_private/sources.json", "sha256": "0" * 64})):
                broken = dict(manifest)
                if binding is None:
                    broken.pop("private_source_manifest")
                else:
                    broken["private_source_manifest"] = binding
                packer.save(raw / "manifest.json", broken)
                output = raw.parent / f"rejected-binding-{index}"
                with self.subTest(binding=binding), patch.object(contracts, "verify") as verify:
                    with self.assertRaisesRegex(ValueError, "private_source_manifest binding"):
                        packer.pack(raw, output, protocol, card, task_definitions=definitions, push_resolved_cell_manifest=cell)
                    self.assertFalse(output.exists())
                    verify.assert_not_called()

    def test_legacy_development_has_no_new_private_manifest_dependency(self):
        with patch.object(contracts, "verify") as verify:
            self.assertEqual(packer._prepare_task_definition_inputs(
                Path("does-not-exist"), [], [], None, None, required=False), ({}, None))
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
