from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from true_rl_agent.agent_core.generation import _selected_state
from true_rl_agent.agent_core.generation_record import commit_generation_record

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LEDGER_RECORD_SCHEMA_VERSION,
    GenerationLaneAssignment,
    GenerationManifest,
    GenerationSessionContract,
    generation_resource_plan_hash,
    generation_training_initialization_hash,
    load_generation_session_contract,
    publish_generation_session_contract,
)


SHA = "a" * 64
CHECKSUM = "sha256:" + "b" * 64


def _bound_contract_hashes(*, resource_plan_hash: str) -> dict[str, str]:
    return {
        "authoring_policy": "1" * 64,
        "continuation_policy": "2" * 64,
        "evaluation_cost": "3" * 64,
        "evaluation_protocol": "4" * 64,
        "final_confirmation_plan": "5" * 64,
        "provider_policy": "6" * 64,
        "resource_plan": resource_plan_hash,
        "retry_policy": "7" * 64,
        "selector": "8" * 64,
    }


class GenerationContractTests(unittest.TestCase):
    def _lanes(self) -> tuple[GenerationLaneAssignment, GenerationLaneAssignment]:
        return (
            GenerationLaneAssignment(
                lane_id="lane-a",
                execution_device="cuda:0",
                resource_binding_hash="a" * 64,
                lane_intent_hash="c" * 64,
            ),
            GenerationLaneAssignment(
                lane_id="lane-b",
                execution_device="cuda:1",
                resource_binding_hash="b" * 64,
                lane_intent_hash="d" * 64,
            ),
        )

    def _contract(self) -> GenerationSessionContract:
        lanes = self._lanes()
        return GenerationSessionContract(
            session_id="generation-session",
            task_id="task",
            backend_id="backend",
            max_generations=15,
            max_trials=30,
            max_train_jobs=30,
            bound_contract_hashes=_bound_contract_hashes(
                resource_plan_hash=generation_resource_plan_hash(lanes)
            ),
        )

    def test_create_once_contract_round_trips_and_rejects_drift(self) -> None:
        contract = self._contract()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = publish_generation_session_contract(root, contract)
            second = publish_generation_session_contract(root, contract)
            loaded = load_generation_session_contract(root, first)
            self.assertEqual(first, second)
            self.assertEqual(loaded.to_dict(), contract.to_dict())

            drift = replace(contract, max_generations=14, max_trials=28, max_train_jobs=28)
            with self.assertRaises(FileExistsError):
                publish_generation_session_contract(root, drift)

    def test_cuda_lanes_require_two_distinct_bound_devices(self) -> None:
        lanes = self._lanes()
        manifest = GenerationManifest(
            generation_id="generation-001",
            generation_index=1,
            session_genesis_hash=SHA,
            generation_session_contract_hash="1" * 64,
            ledger_epoch_hash="2" * 64,
            parent_program_source_checksum=CHECKSUM,
            parent_checkpoint_hash=CHECKSUM,
            current_best_evidence_hash="3" * 64,
            base_public_context_hash="4" * 64,
            evaluation_binding_hash="5" * 64,
            training_initialization_hash="6" * 64,
            guard_reference_evidence_hash="7" * 64,
            guard_reference_phase_receipt_hash="8" * 64,
            lanes=lanes,
        )
        self.assertEqual(GenerationManifest.from_mapping(manifest.to_dict()), manifest)

        for bad_lanes in (
            (lanes[0], replace(lanes[1], execution_device="cuda:0")),
            (lanes[0], replace(lanes[1], resource_binding_hash="a" * 64)),
            (lanes[0], replace(lanes[1], execution_device="cpu", resource_binding_hash=None)),
        ):
            with self.subTest(bad_lanes=bad_lanes):
                with self.assertRaises(ValueError):
                    replace(manifest, lanes=bad_lanes)

    def test_training_initialization_hash_excludes_candidate_identity(self) -> None:
        first = generation_training_initialization_hash(
            policy_initialization_kind="random",
            training_binding_hash="a" * 64,
            expected_initial_policy_state_hash=None,
        )
        second = generation_training_initialization_hash(
            policy_initialization_kind="random",
            training_binding_hash="a" * 64,
            expected_initial_policy_state_hash=None,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            self._contract().ledger_record_schema,
            GENERATION_LEDGER_RECORD_SCHEMA_VERSION,
        )

    def test_selected_trial_must_name_exactly_one_lane(self) -> None:
        selection = SimpleNamespace(selected_trial_id="missing-trial")
        states = (
            SimpleNamespace(runtime=SimpleNamespace(trial_id="trial-a")),
            SimpleNamespace(runtime=SimpleNamespace(trial_id="trial-b")),
        )
        with self.assertRaisesRegex(ValueError, "unknown or duplicate trial"):
            _selected_state(states, selection)

        duplicate_states = (
            SimpleNamespace(runtime=SimpleNamespace(trial_id="trial-a")),
            SimpleNamespace(runtime=SimpleNamespace(trial_id="trial-a")),
        )
        selection = SimpleNamespace(selected_trial_id="trial-a")
        with self.assertRaisesRegex(ValueError, "unknown or duplicate trial"):
            _selected_state(duplicate_states, selection)

    def test_generation_commit_replays_before_lock_and_lock_only_checks_snapshot(
        self,
    ) -> None:
        snapshot = ({"generation_id": "generation-001"},)
        incoming = {"generation_id": "generation-002"}
        events: list[str] = []

        def _validate(*args, **kwargs) -> None:
            self.assertEqual(args[0], snapshot)
            self.assertEqual(kwargs["incoming"], incoming)
            events.append("validated")

        def _append(_path, record, *, pre_append_validator) -> bool:
            self.assertEqual(events, ["validated"])
            self.assertEqual(record, incoming)
            pre_append_validator(snapshot, record)
            with self.assertRaisesRegex(ValueError, "changed after commit prevalidation"):
                pre_append_validator(
                    ({"generation_id": "concurrent-generation"},),
                    record,
                )
            events.append("appended")
            return True

        with (
            patch(
                "true_rl_agent.agent_core.generation_record.load_ledger",
                return_value=list(snapshot),
            ),
            patch(
                "true_rl_agent.agent_core.generation_record."
                "_validate_generation_commit_snapshot",
                side_effect=_validate,
            ),
            patch(
                "true_rl_agent.agent_core.generation_record."
                "append_ledger_record_once",
                side_effect=_append,
            ),
        ):
            committed = commit_generation_record(
                ledger_path=Path("unused-ledger.jsonl"),
                record=incoming,
                initial_current_best=SimpleNamespace(),
                acceptance_protocol=SimpleNamespace(),
            )

        self.assertTrue(committed)
        self.assertEqual(events, ["validated", "appended"])


if __name__ == "__main__":
    unittest.main()
