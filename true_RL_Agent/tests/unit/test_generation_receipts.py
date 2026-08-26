from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from true_rl_agent.agent_core.generation_contract import (
    GenerationLaneAssignment,
    GenerationManifest,
)
from true_rl_agent.agent_core.generation_receipts import (
    GenerationReceiptJournal,
    load_generation_manifest_from_receipt,
    load_generation_manifest_receipt,
    load_generation_selection_receipt,
)
from true_rl_agent.evaluation.binding import canonical_json_hash


def _manifest() -> GenerationManifest:
    return GenerationManifest(
        generation_id="generation-001",
        generation_index=1,
        session_genesis_hash="1" * 64,
        generation_session_contract_hash="2" * 64,
        ledger_epoch_hash="3" * 64,
        parent_program_source_checksum="sha256:" + "4" * 64,
        parent_checkpoint_hash="sha256:" + "5" * 64,
        current_best_evidence_hash="6" * 64,
        base_public_context_hash="7" * 64,
        evaluation_binding_hash="8" * 64,
        training_initialization_hash="9" * 64,
        guard_reference_evidence_hash="a" * 64,
        guard_reference_phase_receipt_hash="b" * 64,
        lanes=(
            GenerationLaneAssignment("lane-a", "cpu", None, "c" * 64),
            GenerationLaneAssignment("lane-b", "cpu", None, "d" * 64),
        ),
    )


class GenerationReceiptTests(unittest.TestCase):
    def test_manifest_selection_and_execution_are_create_once_and_bound(self) -> None:
        manifest = _manifest()
        screens = [
            {"trial_id": "trial-a", "score": 0.7},
            {"trial_id": "trial-b", "score": 0.6},
        ]
        guard = {"trial_id": "parent", "score": 0.5}
        selection = {
            "selected_trial_id": "trial-a",
            "screen_evidence_hashes": [canonical_json_hash(item) for item in screens],
            "guard_reference_evidence_hash": canonical_json_hash(guard),
            "acceptance_authority": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            journal = GenerationReceiptJournal(Path(tmp), manifest=manifest)
            manifest_ref = journal.bind_manifest()
            selection_ref = journal.publish_selection(
                selection=selection,
                screens=screens,
                guard_reference=guard,
                guard_reference_phase_receipt_checksum="sha256:" + "e" * 64,
            )
            record = {"schema_version": "test.generation.v1", "generation_id": "generation-001"}
            execution_ref = journal.publish_execution(record)

            self.assertEqual(
                load_generation_manifest_receipt(manifest_ref, manifest=manifest),
                manifest_ref,
            )
            self.assertEqual(
                load_generation_manifest_from_receipt(manifest_ref),
                manifest,
            )
            loaded_selection = load_generation_selection_receipt(
                selection_ref,
                manifest=manifest,
            )
            self.assertEqual(loaded_selection.selection, selection)
            self.assertEqual(journal.load_execution(), (record, execution_ref))
            self.assertEqual(journal.publish_execution(record), execution_ref)

            with self.assertRaisesRegex(ValueError, "differs"):
                journal.publish_execution({**record, "generation_id": "generation-002"})

    def test_selection_rejects_unbound_screen_order_and_tampering(self) -> None:
        manifest = _manifest()
        screens = [
            {"trial_id": "trial-a", "score": 0.7},
            {"trial_id": "trial-b", "score": 0.6},
        ]
        guard = {"trial_id": "parent", "score": 0.5}
        bad_selection = {
            "selected_trial_id": "trial-a",
            "screen_evidence_hashes": [
                canonical_json_hash(screens[1]),
                canonical_json_hash(screens[0]),
            ],
            "guard_reference_evidence_hash": canonical_json_hash(guard),
            "acceptance_authority": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            journal = GenerationReceiptJournal(Path(tmp), manifest=manifest)
            journal.bind_manifest()
            with self.assertRaisesRegex(ValueError, "screen order"):
                journal.publish_selection(
                    selection=bad_selection,
                    screens=screens,
                    guard_reference=guard,
                    guard_reference_phase_receipt_checksum="sha256:" + "e" * 64,
                )

            good_selection = {
                **bad_selection,
                "screen_evidence_hashes": [
                    canonical_json_hash(item) for item in screens
                ],
            }
            ref = journal.publish_selection(
                selection=good_selection,
                screens=screens,
                guard_reference=guard,
                guard_reference_phase_receipt_checksum="sha256:" + "e" * 64,
            )
            path = Path(ref.path)
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                load_generation_selection_receipt(ref, manifest=manifest)


if __name__ == "__main__":
    unittest.main()
