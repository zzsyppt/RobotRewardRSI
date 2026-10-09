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
    GENERATION_MANIFEST_V2_SCHEMA_VERSION,
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationLaneAssignment,
    GenerationManifest,
    GenerationSessionContract,
    generation_resource_plan_hash,
    generation_training_initialization_hash,
    load_generation_session_contract,
    load_replay_generation_session_contract,
    latest_nonaccepted_trial,
    publish_generation_session_contract,
)
from true_rl_agent.runner.specs import GpuProcessBinding


SHA = "a" * 64
CHECKSUM = "sha256:" + "b" * 64


def _bound_contract_hashes(*, resource_plan_hash: str) -> dict[str, str]:
    return {
        "authoring_policy": "1" * 64,
        "continuation_policy": "2" * 64,
        "evaluation_cost": "3" * 64,
        "evaluation_protocol": "4" * 64,
        "exploration_projector": "e" * 64,
        "final_confirmation_plan": "5" * 64,
        "provider_policy": "6" * 64,
        "resource_plan": resource_plan_hash,
        "retry_policy": "7" * 64,
        "selector": "8" * 64,
    }


class GenerationContractTests(unittest.TestCase):
    def test_readonly_replay_uses_frozen_authoring_identity_but_execution_stays_strict(self):
        stored = self._v3_contract()
        current = replace(stored, bound_contract_hashes={**stored.bound_contract_hashes,
                                                        "authoring_policy": "a" * 64})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = publish_generation_session_contract(root, stored)
            path = root / ref.logical_name
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            self.assertEqual(load_replay_generation_session_contract(root, current), (stored, ref))
            with self.assertRaises(FileExistsError):
                publish_generation_session_contract(root, current, require_existing=True)
            for changed in (replace(current, session_id="other-session"),
                            replace(current, bound_contract_hashes={**current.bound_contract_hashes,
                                                                  "selector": "b" * 64}),
                            replace(current, bound_contract_hashes={**current.bound_contract_hashes,
                                                                  "provider_policy": "c" * 64})):
                with self.assertRaisesRegex(ValueError, "outside authoring_policy"):
                    load_replay_generation_session_contract(root, changed)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_latest_trained_nonaccepted_trial_crosses_incomplete_and_accepted_generations(self):
        trained = {"trial_id": "g1-b", "trial_status": "qualified_not_selected",
                   "candidate": {"candidate_id": "sibling"}, "train": {"completed": True}}
        accepted = {**trained, "trial_id": "g1-a", "trial_status": "accepted"}
        untrained = {**trained, "trial_id": "g2-b", "trial_status": "candidate_failed", "train": None}
        records = [{"generation": {"trials": [trained, accepted]}},
                   {"generation": {"trials": [untrained]}}]
        self.assertEqual(latest_nonaccepted_trial(records), (0, trained))
        self.assertIsNone(latest_nonaccepted_trial(records[1:]))

    def test_nonaccepted_source_prefers_task_selection_then_existing_fallback(self):
        selected = {"trial_id": "g1-a", "trial_status": "formal_rejected",
                    "candidate": {"candidate_id": "selected"}, "train": {"completed": True}}
        sibling = {"trial_id": "g1-b", "trial_status": "qualified_not_selected",
                   "candidate": {"candidate_id": "sibling"}, "train": {"completed": True}}
        for status, selection, expected in (
            ("formal_rejected", {"selected_trial_id": "g1-a"}, "g1-a"),
            ("accepted", {"selected_trial_id": "g1-a"}, "g1-b"),
            ("formal_rejected", None, "g1-b"),
        ):
            with self.subTest(status=status, selection=selection):
                records = [{"generation": {
                    "trials": [{**selected, "trial_status": status}, sibling],
                    "selection": selection,
                }}]
                result = latest_nonaccepted_trial(records)
                self.assertIsNotNone(result)
                self.assertEqual(result[0], 0)
                self.assertEqual(result[1]["trial_id"], expected)

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

    def _v3_contract(self) -> GenerationSessionContract:
        bound_contract_hashes = _bound_contract_hashes(
            resource_plan_hash=generation_resource_plan_hash(self._bound_lanes())
        )
        bound_contract_hashes["session_cost"] = "9" * 64
        return GenerationSessionContract(
            schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
            session_mode="development_generation_pilot",
            session_id="generation-pilot",
            task_id="task",
            backend_id="backend",
            max_generations=1,
            max_trials=2,
            max_train_jobs=2,
            bound_contract_hashes=bound_contract_hashes,
        )

    def _bound_lanes(
        self,
    ) -> tuple[GenerationLaneAssignment, GenerationLaneAssignment]:
        return (
            GenerationLaneAssignment(
                lane_id="lane-a",
                execution_device="cuda:0",
                resource_binding_hash="a" * 64,
                lane_intent_hash="c" * 64,
                gpu_process_binding=GpuProcessBinding(
                    host_execution_device="cuda:0",
                    process_execution_device="cuda:0",
                    resource_binding_hash="a" * 64,
                ),
            ),
            GenerationLaneAssignment(
                lane_id="lane-b",
                execution_device="cuda:0",
                resource_binding_hash="b" * 64,
                lane_intent_hash="d" * 64,
                gpu_process_binding=GpuProcessBinding(
                    host_execution_device="cuda:1",
                    process_execution_device="cuda:0",
                    resource_binding_hash="b" * 64,
                ),
            ),
        )

    def _manifest(
        self,
        lanes: tuple[GenerationLaneAssignment, GenerationLaneAssignment],
        *,
        schema_version: str | None = None,
    ) -> GenerationManifest:
        kwargs = {}
        if schema_version is not None:
            kwargs["schema_version"] = schema_version
        return GenerationManifest(
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
            **kwargs,
        )

    def test_create_once_contract_round_trips_and_rejects_drift(self) -> None:
        contract = self._contract()
        self.assertNotIn("session_mode", contract.to_dict())
        self.assertNotIn("session_cost", contract.bound_contract_hashes)
        self.assertEqual(
            contract.contract_hash,
            "7435d985e16a2e8b2bac25e650936830c27cb4a56e3efadcd9f63fe72c455b02",
        )
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

    def test_v3_contract_binds_session_mode_and_session_cost(self) -> None:
        contract = self._v3_contract()
        self.assertEqual(
            contract.to_dict()["session_mode"],
            "development_generation_pilot",
        )
        self.assertEqual(contract.bound_contract_hashes["session_cost"], "9" * 64)
        self.assertEqual(GenerationSessionContract.from_mapping(contract.to_dict()), contract)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = publish_generation_session_contract(root, contract)
            self.assertEqual(load_generation_session_contract(root, ref), contract)

        with self.assertRaisesRegex(ValueError, "session_mode"):
            replace(contract, session_mode=None)
        without_session_cost = dict(contract.bound_contract_hashes)
        without_session_cost.pop("session_cost")
        with self.assertRaisesRegex(ValueError, "hash fields mismatch"):
            replace(contract, bound_contract_hashes=without_session_cost)

        legacy = self._contract()
        with self.assertRaisesRegex(ValueError, "cannot carry session_mode"):
            replace(legacy, session_mode="development_generation_pilot")
        with self.assertRaisesRegex(ValueError, "hash fields mismatch"):
            replace(
                legacy,
                bound_contract_hashes={
                    **dict(legacy.bound_contract_hashes),
                    "session_cost": "9" * 64,
                },
            )

    def test_cuda_lanes_require_two_distinct_bound_devices(self) -> None:
        lanes = self._lanes()
        manifest = self._manifest(lanes)
        self.assertEqual(GenerationManifest.from_mapping(manifest.to_dict()), manifest)

        for bad_lanes in (
            (lanes[0], replace(lanes[1], execution_device="cuda:0")),
            (lanes[0], replace(lanes[1], resource_binding_hash="a" * 64)),
            (lanes[0], replace(lanes[1], execution_device="cpu", resource_binding_hash=None)),
        ):
            with self.subTest(bad_lanes=bad_lanes):
                with self.assertRaises(ValueError):
                    replace(manifest, lanes=bad_lanes)

    def test_legacy_lane_bytes_and_resource_hash_shape_remain_v1(self) -> None:
        lanes = self._lanes()
        self.assertEqual(
            lanes[0].to_dict(),
            {
                "lane_id": "lane-a",
                "execution_device": "cuda:0",
                "resource_binding_hash": "a" * 64,
                "lane_intent_hash": "c" * 64,
            },
        )
        self.assertNotIn("gpu_process_binding", self._manifest(lanes).to_dict()["lanes"][0])
        self.assertEqual(
            self._manifest(lanes).to_public_dict(),
            self._manifest(lanes).to_dict(),
        )
        self.assertEqual(
            generation_resource_plan_hash(lanes),
            "6243db4a5fea2696901bbd64534d51fe7b938a04fe5fd91f8e4f0756d01db6fc",
        )

    def test_v2_uses_host_binding_while_children_share_logical_cuda_zero(self) -> None:
        lanes = self._bound_lanes()
        manifest = self._manifest(
            lanes,
            schema_version=GENERATION_MANIFEST_V2_SCHEMA_VERSION,
        )
        self.assertEqual({lane.execution_device for lane in lanes}, {"cuda:0"})
        self.assertEqual(
            {
                lane.gpu_process_binding.host_execution_device
                for lane in lanes
                if lane.gpu_process_binding is not None
            },
            {"cuda:0", "cuda:1"},
        )
        self.assertEqual(GenerationManifest.from_mapping(manifest.to_dict()), manifest)
        self.assertEqual(
            manifest.to_public_dict()["lanes"][0]["gpu_process_binding"],
            lanes[0].gpu_process_binding.to_public_dict(),
        )
        self.assertNotIn(
            "host_execution_device",
            manifest.to_public_dict()["lanes"][0]["gpu_process_binding"],
        )
        self.assertEqual(
            manifest.to_dict()["lanes"][0]["gpu_process_binding"],
            lanes[0].gpu_process_binding.to_dict(),
        )
        self.assertNotEqual(
            generation_resource_plan_hash(lanes),
            generation_resource_plan_hash(self._lanes()),
        )

    def test_generation_binding_schema_mixes_fail_closed(self) -> None:
        legacy = self._lanes()
        bound = self._bound_lanes()
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            generation_resource_plan_hash((bound[0], legacy[1]))
        with self.assertRaisesRegex(ValueError, "v1 cannot carry"):
            self._manifest(bound)
        with self.assertRaisesRegex(ValueError, "v2 requires every"):
            self._manifest(
                legacy,
                schema_version=GENERATION_MANIFEST_V2_SCHEMA_VERSION,
            )

    def test_v2_requires_distinct_host_devices_and_resource_bindings(self) -> None:
        lanes = self._bound_lanes()
        same_host = replace(
            lanes[1],
            gpu_process_binding=replace(
                lanes[1].gpu_process_binding,
                host_execution_device="cuda:0",
            ),
        )
        with self.assertRaisesRegex(ValueError, "distinct host"):
            generation_resource_plan_hash((lanes[0], same_host))
        with self.assertRaisesRegex(ValueError, "distinct host"):
            self._manifest(
                (lanes[0], same_host),
                schema_version=GENERATION_MANIFEST_V2_SCHEMA_VERSION,
            )

        same_resource = replace(
            lanes[1],
            resource_binding_hash="a" * 64,
            gpu_process_binding=replace(
                lanes[1].gpu_process_binding,
                resource_binding_hash="a" * 64,
            ),
        )
        with self.assertRaisesRegex(ValueError, "distinct GPU"):
            generation_resource_plan_hash((lanes[0], same_resource))
        with self.assertRaisesRegex(ValueError, "distinct GPUs"):
            self._manifest(
                (lanes[0], same_resource),
                schema_version=GENERATION_MANIFEST_V2_SCHEMA_VERSION,
            )

    def test_gpu_process_binding_round_trips_exact_fields(self) -> None:
        binding = GpuProcessBinding(
            host_execution_device="cuda:1",
            process_execution_device="cuda:0",
            resource_binding_hash="a" * 64,
        )
        self.assertEqual(GpuProcessBinding.from_mapping(binding.to_dict()), binding)
        self.assertEqual(
            binding.to_public_dict(),
            {
                "process_execution_device": "cuda:0",
                "resource_binding_hash": "a" * 64,
            },
        )
        with self.assertRaisesRegex(ValueError, "must be a CUDA device"):
            replace(binding, host_execution_device="cpu")
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            GpuProcessBinding.from_mapping({**binding.to_dict(), "extra": True})

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
