from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

P6_DIR = Path(__file__).resolve().parent
if str(P6_DIR) not in sys.path:
    sys.path.insert(0, str(P6_DIR))

import verify_phase6 as verifier
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    StairGenerationActualCostJournal,
    StairGenerationActualCostUsage,
    StairGenerationSessionCostContract,
    StairGenerationSessionCostContractV2,
    StairGenerationSessionCostContractV4,
    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_OUTPUT_BYTES,
    STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS,
    STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES,
    build_stair_generation_scientific_cost_basis,
)


def _phase5_go() -> dict[str, object]:
    return {
        "status": "go",
        "cell_variant_id": verifier.GO2_CELL_VARIANT_ID,
    }


def _pilot_pass() -> dict[str, object]:
    return {
        "status": "pass",
        "loop_verified": True,
        "improvement_observed": False,
        "generation_count": 1,
        "session_genesis_hash": "a" * 64,
    }


def _validation_pass(*, improvement: bool = False) -> dict[str, object]:
    return {
        "status": "pass",
        "loop_verified": True,
        "improvement_observed": improvement,
        "generation_count": 3,
        "session_genesis_hash": "b" * 64,
    }


class Phase6VerifierTests(unittest.TestCase):
    def _validation_cost_contract(self, *, imported: bool):
        config = verifier.StairClimbingTaskConfig.defaults(verifier.GO2_PROTOCOL)
        basis = build_stair_generation_scientific_cost_basis(
            session_id="phase6-verifier-cost",
            config=config,
        )
        if imported:
            return StairGenerationSessionCostContractV4(
                session_mode=verifier.DEVELOPMENT_GENERATION_VALIDATION_MODE,
                max_generations=3,
                scientific_cost_basis=replace(basis, parent_bootstrap_seed_units=0),
                retry_train_jobs=0,
                retry_eval_seed_units=7,
                successful_allocated_active_gpu_hours=(
                    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
                ),
                retry_allocated_active_gpu_hours=(
                    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
                ),
                successful_output_bytes=(
                    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_SUCCESSFUL_OUTPUT_BYTES
                ),
                retry_output_bytes=(
                    STAIR_GENERATION_IMPORTED_REFERENCE_VALIDATION_RETRY_OUTPUT_BYTES
                ),
            )
        return StairGenerationSessionCostContractV2(
            session_mode=verifier.DEVELOPMENT_GENERATION_VALIDATION_MODE,
            max_generations=3,
            scientific_cost_basis=basis,
            retry_train_jobs=1,
            retry_eval_seed_units=7,
            successful_allocated_active_gpu_hours=(
                STAIR_GENERATION_VALIDATION_SUCCESSFUL_ALLOCATED_ACTIVE_GPU_HOURS
            ),
            retry_allocated_active_gpu_hours=(
                STAIR_GENERATION_VALIDATION_RETRY_ALLOCATED_ACTIVE_GPU_HOURS
            ),
            successful_output_bytes=(
                STAIR_GENERATION_VALIDATION_SUCCESSFUL_OUTPUT_BYTES
            ),
            retry_output_bytes=STAIR_GENERATION_VALIDATION_RETRY_OUTPUT_BYTES,
        )

    def _completed_validation(self, *, output_dir: Path, cost_contract):
        return SimpleNamespace(
            output_dir=output_dir,
            resolved_cell_manifest=object(),
            session_genesis=object(),
            session_contract=object(),
            cost_contract=cost_contract,
            continuation_policy=object(),
            ledger=(),
            report={},
            generation_execution_receipts=(),
            lifecycle=object(),
            actual_cost_receipts=(),
            source_checkout={"source_checkout_hash": "c" * 64},
        )

    def _write_index(self, payload: dict[str, object], root: Path) -> Path:
        body = copy.deepcopy(payload)
        body["index_hash"] = verifier.canonical_json_hash(
            {key: value for key, value in body.items() if key != "index_hash"}
        )
        path = root / "phase6-index.json"
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _p5_guard(self):
        payload = json.loads(verifier._P5_SESSION_GENESIS.read_text(encoding="utf-8"))
        genesis = verifier.SessionGenesisArtifact.from_mapping(payload)
        anchor = genesis.anchor
        config = verifier.StairClimbingTaskConfig.defaults(
            verifier.GO2_PROTOCOL,
            resolved_cell_task_protocol_hash=(
                genesis.evaluation_binding.task_protocol_hash
            ),
        )
        parent = verifier.CurrentBestEvalStats.from_reference_public_result_mapping(
            reference_id=anchor.reference_manifest.reference_id,
            reference_manifest_hash=str(anchor.reference_manifest.manifest_hash),
            program_source_checksum=anchor.reference_manifest.source_checksum,
            training_binding_hash=str(anchor.training_binding.binding_hash),
            capability_profile_hash=anchor.reference_manifest.capability_profile_hash,
            checkpoint_hash=anchor.checkpoint_hash,
            public_result=anchor.public_result,
            acceptance_protocol=config.acceptance_protocol,
            expected_evaluation_binding=genesis.evaluation_binding,
        )
        _adapter, guard = verifier._initial_stair_guard_reference(
            genesis=genesis,
            parent=parent,
            config=config,
        )
        return guard

    def test_current_index_contains_fresh_validation_and_failed_runs_are_excluded(
        self,
    ) -> None:
        index = verifier._load_index()
        self.assertEqual(
            index["validation"]["session_id"],
            "p6-go2-validation3-v4-successor",
        )
        self.assertEqual(
            index["validation"]["expected_generation_outcomes"],
            ["accepted", "no_selection", "no_selection"],
        )
        self.assertEqual(index["pilot"]["session_id"], "p6-go2-pilot-6508354")
        self.assertEqual(index["pilot"]["source_commit"][:7], "6508354")
        excluded = index["excluded_diagnostics"]
        self.assertEqual(
            {item["session_id"] for item in excluded},
            {
                "p6-go2-validation3-d8932b9",
                "p6-go2-validation3-ea089cc",
            },
        )
        self.assertTrue(all(item["status"] == "excluded" for item in excluded))

    def test_validation_rejects_legacy_v2_cost_contract(self) -> None:
        entry = {
            "session_id": "phase6-validation",
            "cell_variant_id": verifier.GO2_CELL_VARIANT_ID,
            "source_commit": "a" * 40,
        }
        with tempfile.TemporaryDirectory(prefix="phase6-validation-") as raw:
            output_dir = Path(raw).resolve()
            completed = self._completed_validation(
                output_dir=output_dir,
                cost_contract=self._validation_cost_contract(imported=False),
            )
            with (
                mock.patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "load_completed_stair_generation_session",
                    return_value=completed,
                ),
                self.assertRaisesRegex(AssertionError, "exact cost contract v4"),
            ):
                verifier._verify_validation(output_dir, entry)

    def test_validation_accepts_exact_v4_cost_contract(self) -> None:
        entry = {
            "session_id": "phase6-validation",
            "cell_variant_id": verifier.GO2_CELL_VARIANT_ID,
            "source_commit": "a" * 40,
            "source_provenance": "typed_source_checkout_receipt",
        }
        with tempfile.TemporaryDirectory(prefix="phase6-validation-") as raw:
            output_dir = Path(raw).resolve()
            completed = self._completed_validation(
                output_dir=output_dir,
                cost_contract=self._validation_cost_contract(imported=True),
            )
            with (
                mock.patch(
                    "true_rl_agent.tasks.stair_climbing.generation_session."
                    "load_completed_stair_generation_session",
                    return_value=completed,
                ),
                mock.patch.object(
                    verifier,
                    "_replay_scientific_session",
                    return_value={"status": "pass"},
                ),
            ):
                result = verifier._verify_validation(output_dir, entry)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["source_checkout_hash"], completed.source_checkout["source_checkout_hash"]
        )

    def test_pilot_only_replay_passes_without_granting_phase6_go(self) -> None:
        payload = json.loads(verifier.EVIDENCE_INDEX.read_text(encoding="utf-8"))
        payload["validation"] = None
        with tempfile.TemporaryDirectory(prefix="phase6-pilot-index-") as raw:
            index_path = self._write_index(payload, Path(raw))
            with (
                mock.patch.object(verifier, "_verify_phase5", return_value=_phase5_go()),
                mock.patch.object(verifier, "_verify_pilot", return_value=_pilot_pass()),
            ):
                result = verifier.verify(
                    pilot_output_dir=Path("C:/phase6-pilot"),
                    evidence_index_path=index_path,
                )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["pilot"]["loop_verified"])
        self.assertFalse(result["phase6_go"])
        self.assertFalse(result["loop_verified"])
        self.assertIsNone(result["improvement_observed"])
        self.assertEqual(result["validation"]["status"], "not_available")

    def test_require_go_rejects_pilot_only_evidence(self) -> None:
        with (
            mock.patch.object(verifier, "_verify_phase5", return_value=_phase5_go()),
            mock.patch.object(verifier, "_verify_pilot", return_value=_pilot_pass()),
            self.assertRaisesRegex(AssertionError, "fresh validation_3 closure"),
        ):
            verifier.verify(
                pilot_output_dir=Path("C:/phase6-pilot"),
                require_go=True,
            )

    def test_unindexed_validation_cannot_be_supplied_as_go_evidence(self) -> None:
        payload = json.loads(verifier.EVIDENCE_INDEX.read_text(encoding="utf-8"))
        payload["validation"] = None
        with tempfile.TemporaryDirectory(prefix="phase6-unindexed-") as raw:
            index_path = self._write_index(payload, Path(raw))
            with (
                mock.patch.object(verifier, "_verify_phase5", return_value=_phase5_go()),
                mock.patch.object(verifier, "_verify_pilot", return_value=_pilot_pass()),
                self.assertRaisesRegex(AssertionError, "unindexed validation"),
            ):
                verifier.verify(
                    pilot_output_dir=Path("C:/phase6-pilot"),
                    validation_output_dir=Path("C:/phase6-validation"),
                    evidence_index_path=index_path,
                )

    def test_index_hash_tamper_is_rejected(self) -> None:
        payload = json.loads(verifier.EVIDENCE_INDEX.read_text(encoding="utf-8"))
        payload["pilot"]["session_id"] = "forged-pilot"
        with tempfile.TemporaryDirectory(prefix="phase6-index-") as raw:
            path = Path(raw) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "index hash mismatch"):
                verifier._load_index(path)

    def test_excluded_diagnostic_cannot_be_promoted_to_validation(self) -> None:
        payload = json.loads(verifier.EVIDENCE_INDEX.read_text(encoding="utf-8"))
        validation = copy.deepcopy(payload["pilot"])
        validation.update(
            session_id="p6-go2-validation3-d8932b9",
            source_commit="d8932b9de70bbb08375b597bf0952a00f7780b88",
            session_mode=verifier.DEVELOPMENT_GENERATION_VALIDATION_MODE,
            expected_generation_count=3,
            expected_generation_outcomes=[
                "paired_authoring_incomplete",
                "candidate_execution_incomplete",
                "no_selection",
            ],
            source_provenance="typed_source_checkout_receipt",
        )
        validation.pop("generation_execution_receipt_checksum")
        payload["validation"] = validation
        with tempfile.TemporaryDirectory(prefix="phase6-index-") as raw:
            path = self._write_index(payload, Path(raw))
            with self.assertRaisesRegex(AssertionError, "validation is marked excluded"):
                verifier._load_index(path)

    def test_improvement_is_reported_but_not_a_go_gate(self) -> None:
        payload = json.loads(verifier.EVIDENCE_INDEX.read_text(encoding="utf-8"))
        validation = copy.deepcopy(payload["pilot"])
        validation.update(
            session_id="p6-go2-validation3-fresh",
            source_commit="ea089cce1ba90e4f7aa553012f8e0da581ea8f67",
            session_mode=verifier.DEVELOPMENT_GENERATION_VALIDATION_MODE,
            expected_generation_count=3,
            expected_generation_outcomes=[
                "formal_rejected",
                "no_selection",
                "formal_rejected",
            ],
            source_provenance="typed_source_checkout_receipt",
        )
        validation.pop("generation_execution_receipt_checksum")
        payload["validation"] = validation
        with tempfile.TemporaryDirectory(prefix="phase6-index-") as raw:
            index_path = self._write_index(payload, Path(raw))
            with (
                mock.patch.object(verifier, "_verify_phase5", return_value=_phase5_go()),
                mock.patch.object(verifier, "_verify_pilot", return_value=_pilot_pass()),
                mock.patch.object(
                    verifier,
                    "_verify_validation",
                    return_value=_validation_pass(improvement=False),
                ),
                mock.patch.object(
                    verifier,
                    "_git_is_ancestor",
                    return_value=True,
                ) as ancestry,
            ):
                result = verifier.verify(
                    pilot_output_dir=Path("C:/phase6-pilot"),
                    validation_output_dir=Path("C:/phase6-validation"),
                    evidence_index_path=index_path,
                    require_go=True,
                )
        self.assertTrue(result["phase6_go"])
        self.assertTrue(result["loop_verified"])
        self.assertFalse(result["improvement_observed"])
        self.assertEqual(ancestry.call_count, 2)
        ancestry.assert_any_call(payload["pilot"]["source_commit"], validation["source_commit"])
        ancestry.assert_any_call(validation["source_commit"])

    def test_validation_must_preserve_pilot_scientific_identity(self) -> None:
        payload = json.loads(verifier.EVIDENCE_INDEX.read_text(encoding="utf-8"))
        validation = copy.deepcopy(payload["pilot"])
        validation.update(
            session_id="p6-go2-validation3-fresh",
            source_commit="ea089cce1ba90e4f7aa553012f8e0da581ea8f67",
            session_mode=verifier.DEVELOPMENT_GENERATION_VALIDATION_MODE,
            expected_generation_count=3,
            expected_generation_outcomes=["formal_rejected"] * 3,
            source_provenance="typed_source_checkout_receipt",
            reference_checkpoint_hash="sha256:" + "5" * 64,
        )
        validation.pop("generation_execution_receipt_checksum")
        payload["validation"] = validation
        with tempfile.TemporaryDirectory(prefix="phase6-index-") as raw:
            path = self._write_index(payload, Path(raw))
            with self.assertRaisesRegex(
                AssertionError,
                "scientific identity differs",
            ):
                verifier._load_index(path)

    def test_pilot_execution_checksum_requires_the_loaded_journal_ref(self) -> None:
        checksum = "sha256:" + "1" * 64
        entry = {"generation_execution_receipt_checksum": checksum}
        report = {"generation_execution_receipt": {"checksum": checksum}}
        verifier._verify_pilot_execution_checksum(
            report=report,
            replay={"generation_execution_receipt_checksums": [checksum]},
            entry=entry,
        )
        with self.assertRaisesRegex(AssertionError, "execution receipt mismatch"):
            verifier._verify_pilot_execution_checksum(
                report=report,
                replay={
                    "generation_execution_receipt_checksums": [
                        "sha256:" + "2" * 64
                    ]
                },
                entry=entry,
            )

    def test_candidate_private_receipts_are_loaded_by_type(self) -> None:
        checksum = "sha256:" + "3" * 64
        with tempfile.TemporaryDirectory(prefix="phase6-receipts-") as raw:
            root = Path(raw).resolve()
            artifact_specs = [
                ("lane-a_train_receipt", "train.json"),
                ("lane-a_public_dev_phase_receipt", "public-dev.json"),
                ("lane-a_public_accept_phase_receipt", "accept.json"),
                ("lane-a_sealed_validation_phase_receipt", "sealed.json"),
                ("lane-a_trial_execution_receipt", "execution.json"),
            ]
            artifacts = []
            for kind, filename in artifact_specs:
                path = root / filename
                path.write_text("{}", encoding="utf-8")
                artifacts.append(
                    {
                        "kind": kind,
                        "uri": str(path),
                        "checksum": checksum,
                        "size_bytes": path.stat().st_size,
                    }
                )

            def loaded(ref):
                return SimpleNamespace(receipt_ref=ref)

            def loaded_execution(ref):
                return {}, SimpleNamespace(receipt_ref=ref)

            with (
                mock.patch.object(verifier, "load_trial_train_receipt", side_effect=loaded),
                mock.patch.object(verifier, "load_trial_eval_phase_receipt", side_effect=loaded),
                mock.patch.object(
                    verifier,
                    "load_trial_execution_receipt",
                    side_effect=loaded_execution,
                ),
            ):
                result = verifier._verify_candidate_private_receipts(
                    root,
                    [{"private_artifacts": artifacts}],
                )
        self.assertEqual(
            result["counts"],
            {"train": 1, "eval_phase": 3, "trial_execution": 1},
        )
        self.assertEqual(result["public_dev_receipt_count"], 1)
        self.assertEqual(result["public_accept_receipt_count"], 1)
        self.assertEqual(result["sealed_validation_receipt_count"], 1)

    def test_real_v1_pilot_cost_contract_uses_linear_replay(self) -> None:
        config = verifier.StairClimbingTaskConfig.defaults(verifier.GO2_PROTOCOL)
        basis = build_stair_generation_scientific_cost_basis(
            session_id="phase6-v1-replay-test",
            config=config,
        )
        contract = StairGenerationSessionCostContract(
            session_mode=verifier.DEVELOPMENT_GENERATION_PILOT_MODE,
            max_generations=1,
            scientific_cost_basis=basis,
            retry_train_jobs=1,
            retry_eval_seed_units=7,
            max_allocated_active_gpu_hours=8.0,
            max_simulated_control_steps=850_000_000,
            max_output_bytes=8 * 1024**3,
        )
        with tempfile.TemporaryDirectory(prefix="phase6-v1-cost-") as raw:
            journal = StairGenerationActualCostJournal(Path(raw), contract=contract)
            reservation = journal.publish_reservation()
            terminal = journal.publish_snapshot(
                event_id="pilot-terminal",
                previous_cost_chain_head_hash=reservation.payload_hash,
                usage=StairGenerationActualCostUsage(),
                finalization_output_reserve_bytes=0,
            )
            chain = journal.load_chain()
        lifecycle = SimpleNamespace(
            recovery_cost_edges=(),
            segment_count=1,
            cost_chain_head_hash=terminal.payload_hash,
        )
        self.assertIs(type(contract), StairGenerationSessionCostContract)
        self.assertEqual(
            verifier._replay_session_cost_chain(
                contract=contract,
                actual_cost_receipts=chain,
                lifecycle=lifecycle,
            ),
            terminal.payload_hash,
        )

    def test_recomputed_cost_binds_raw_counters_and_closed_attempts(self) -> None:
        counters = {
            "provider_attempts": 2,
            "job_attempts": 7,
            "train_jobs": 2,
            "eval_seed_units": 23,
            "observed_output_bytes": 90,
            "observed_output_files": 12,
            "job_cost_evidence_hash": "4" * 64,
        }
        usage = SimpleNamespace(to_dict=lambda: dict(counters))
        actual = {**counters, "accounted_output_bytes": 100}
        receipt = SimpleNamespace(
            payload={
                "actual": actual,
                "conservative_charges": ["provider-full-reserve:test"],
            }
        )
        genesis = SimpleNamespace(
            reference_genesis_import_ref=SimpleNamespace(),
            anchor=SimpleNamespace(
                private_eval_artifact=SimpleNamespace(uri="C:/reference.json")
            ),
        )
        with mock.patch.object(
            verifier,
            "collect_actual_cost_usage",
            return_value=(usage, ("provider-full-reserve:test",), ()),
        ):
            result = verifier._verify_recomputed_actual_cost(
                output_dir=Path("C:/phase6").resolve(),
                session_id="phase6-cost-test",
                genesis=genesis,
                session_contract=SimpleNamespace(),
                cost_contract=SimpleNamespace(),
                ledger=[],
                final_cost_receipt=receipt,
            )
        self.assertEqual(result["actual"]["provider_attempts"], 2)
        self.assertEqual(result["actual"]["job_cost_evidence_hash"], "4" * 64)
        self.assertEqual(result["open_job_attempt_reservation_count"], 0)

    def test_initial_current_best_is_rebuilt_from_genesis_public_result(self) -> None:
        payload = json.loads(verifier._P5_SESSION_GENESIS.read_text(encoding="utf-8"))
        source = verifier.SessionGenesisArtifact.from_mapping(payload)
        anchor = source.anchor
        config = verifier.StairClimbingTaskConfig.defaults(verifier.GO2_PROTOCOL)
        expected = verifier.CurrentBestEvalStats.from_reference_public_result_mapping(
            reference_id=anchor.reference_manifest.reference_id,
            reference_manifest_hash=str(anchor.reference_manifest.manifest_hash),
            program_source_checksum=anchor.reference_manifest.source_checksum,
            training_binding_hash=str(anchor.training_binding.binding_hash),
            capability_profile_hash=anchor.reference_manifest.capability_profile_hash,
            checkpoint_hash=anchor.checkpoint_hash,
            public_result=anchor.public_result,
            acceptance_protocol=config.acceptance_protocol,
            expected_evaluation_binding=source.evaluation_binding,
        )
        import_ref = SimpleNamespace(
            source_session_id=source.session_id,
            source_session_genesis_hash=source.session_genesis_hash,
            source_session_genesis_checksum="sha256:" + "6" * 64,
        )
        imported = SimpleNamespace(
            anchor=anchor,
            evaluation_binding=source.evaluation_binding,
            reference_genesis_import_ref=import_ref,
        )
        entry = {
            "reference_checkpoint_hash": anchor.checkpoint_hash,
            "reference_manifest_hash": anchor.reference_manifest.manifest_hash,
            "initial_policy_state_hash": anchor.initial_policy_state_hash,
            "source_reference_session_id": source.session_id,
            "source_reference_genesis_hash": source.session_genesis_hash,
            "source_reference_genesis_checksum": import_ref.source_session_genesis_checksum,
        }
        rebuilt = verifier._initial_current_best(
            imported,
            {"parent_current_best": expected.to_public_dict()},
            entry,
            acceptance_protocol=config.acceptance_protocol,
        )
        self.assertEqual(rebuilt, expected)

    def test_imported_genesis_bytes_are_bound_to_phase5_verified_file(self) -> None:
        p5_bytes = verifier._P5_SESSION_GENESIS.read_bytes()
        payload = json.loads(p5_bytes.decode("utf-8"))
        source = verifier.SessionGenesisArtifact.from_mapping(payload)
        checksum = "sha256:" + hashlib.sha256(p5_bytes).hexdigest()
        with tempfile.TemporaryDirectory(prefix="phase6-imported-genesis-") as raw:
            source_path = Path(raw).resolve() / "session_genesis.json"
            source_path.write_bytes(p5_bytes)
            imported = SimpleNamespace(
                anchor=source.anchor,
                runtime_snapshot_hash=source.runtime_snapshot_hash,
                evaluation_binding=source.evaluation_binding,
                reference_genesis_import_ref=SimpleNamespace(
                    source_session_id=source.session_id,
                    source_session_genesis_hash=source.session_genesis_hash,
                    source_session_genesis_uri=str(source_path),
                    source_session_genesis_checksum=checksum,
                ),
            )
            result = verifier._verify_imported_reference_genesis(
                imported,
                {
                    "source_reference_session_id": source.session_id,
                    "source_reference_genesis_hash": source.session_genesis_hash,
                    "source_reference_genesis_checksum": checksum,
                },
            )
        self.assertTrue(result["raw_bytes_match_phase5"])

    def test_generation_manifest_receipt_cannot_escape_session_root(self) -> None:
        checksum = "sha256:" + "7" * 64
        with tempfile.TemporaryDirectory(prefix="phase6-manifest-root-") as raw:
            workspace = Path(raw).resolve()
            output_dir = workspace / "session"
            output_dir.mkdir()
            inside = output_dir / "manifest.json"
            outside = workspace / "outside.json"
            inside.write_text("{}", encoding="utf-8")
            outside.write_text("{}", encoding="utf-8")
            record = {
                "manifest_receipt_checksum": checksum,
                "private_artifacts": [
                    {
                        "kind": "generation_manifest_receipt",
                        "uri": str(inside),
                        "checksum": checksum,
                    }
                ],
            }
            self.assertEqual(
                verifier._verify_generation_manifest_receipt_path(
                    output_dir,
                    record,
                ),
                inside,
            )
            record["private_artifacts"][0]["uri"] = str(outside)
            with self.assertRaisesRegex(AssertionError, "escaped its session output"):
                verifier._verify_generation_manifest_receipt_path(
                    output_dir,
                    record,
                )

    def test_generation_training_initialization_rejects_manifest_drift(self) -> None:
        payload = json.loads(verifier._P5_SESSION_GENESIS.read_text(encoding="utf-8"))
        source = verifier.SessionGenesisArtifact.from_mapping(payload)
        parent = SimpleNamespace(
            training_binding_hash=source.anchor.training_binding.binding_hash
        )
        genesis = SimpleNamespace(
            anchor=source.anchor,
            policy_initialization_kind=source.policy_initialization_kind,
        )
        with self.assertRaisesRegex(
            AssertionError,
            "training initialization differs from replayed parent",
        ):
            verifier._verify_generation_training_initialization(
                genesis=genesis,
                ledger=[{"trials": [], "private_artifacts": []}],
                replay_entries=[SimpleNamespace(current_best_before=parent)],
                generation_manifests=[
                    SimpleNamespace(training_initialization_hash="8" * 64)
                ],
            )

    def test_generation_guard_reference_is_rebuilt_from_p5_public_dev(self) -> None:
        guard = self._p5_guard()
        receipt = guard.phase_receipt_ref
        record = {
            "private_artifacts": [
                {
                    "kind": "generation_guard_reference_phase_receipt",
                    "uri": receipt.path,
                    "checksum": receipt.checksum,
                    "size_bytes": receipt.size_bytes,
                }
            ]
        }
        manifest = SimpleNamespace(
            guard_reference_evidence_hash=guard.evidence.evidence_hash,
            guard_reference_phase_receipt_hash=verifier.canonical_json_hash(
                receipt.to_dict()
            ),
        )
        self.assertEqual(
            verifier._assert_generation_guard_reference(
                output_dir=Path("C:/phase6").resolve(),
                record=record,
                manifest=manifest,
                guard=guard,
            ),
            receipt,
        )
        manifest.guard_reference_evidence_hash = "9" * 64
        with self.assertRaisesRegex(
            AssertionError,
            "guard reference differs from replayed current-best",
        ):
            verifier._assert_generation_guard_reference(
                output_dir=Path("C:/phase6").resolve(),
                record=record,
                manifest=manifest,
                guard=guard,
            )

    def test_selector_replay_rejects_tampered_candidate_screen(self) -> None:
        guard = self._p5_guard()
        screens = (
            SimpleNamespace(to_selector_dict=lambda: {"trial_id": "trial-a"}),
            SimpleNamespace(to_selector_dict=lambda: {"trial_id": "trial-b"}),
        )
        selection_payload = {"selected_trial_id": "trial-a"}
        rebuilt_selection = SimpleNamespace(
            to_public_dict=lambda: dict(selection_payload)
        )
        loaded = SimpleNamespace(
            screens=({"trial_id": "trial-a"}, {"trial_id": "trial-b"}),
            selection=dict(selection_payload),
            guard_reference=guard.evidence.to_selector_dict(),
            guard_reference_phase_receipt_checksum=guard.phase_receipt_ref.checksum,
        )
        verifier._assert_task_owned_selector_replay(
            loaded_selection=loaded,
            stored_selection=selection_payload,
            screens=screens,
            rebuilt_selection=rebuilt_selection,
            guard=guard,
            guard_receipt_ref=guard.phase_receipt_ref,
        )
        loaded.screens = ({"trial_id": "trial-a"}, {"trial_id": "tampered"})
        with self.assertRaisesRegex(
            AssertionError,
            "selector receipt differs from task-owned replay",
        ):
            verifier._assert_task_owned_selector_replay(
                loaded_selection=loaded,
                stored_selection=selection_payload,
                screens=screens,
                rebuilt_selection=rebuilt_selection,
                guard=guard,
                guard_receipt_ref=guard.phase_receipt_ref,
            )

    def test_formal_reassembly_binds_public_result_decision_and_current_best(self) -> None:
        before = {"mode": "reference", "public_score": 0.40}
        after = {"mode": "accepted", "public_score": 0.55}
        public_result = {"status": "pass", "public_score": 0.55, "metrics": {}}
        decision = {
            "accepted": True,
            "outcome": "policy_improvement_accept",
            "reason": "named_guards_paired_headroom_delta",
        }
        loaded_record = {
            "public_result": public_result,
            "decision": decision,
        }
        round_hash = verifier.canonical_json_hash(loaded_record)
        execution_checksum = "sha256:" + "1" * 64
        sealed_checksum = "sha256:" + "2" * 64
        trial = {
            "trial_id": "generation-001-lane-a",
            "lane_id": "lane-a",
            "execution_receipt_checksum": execution_checksum,
            "formal_record_hash": round_hash,
        }
        formal = {
            "selected_trial_id": trial["trial_id"],
            "selected_lane_id": trial["lane_id"],
            "trial_execution_receipt_checksum": execution_checksum,
            "sealed_phase_receipt_checksum": sealed_checksum,
            "round_record_hash": round_hash,
            "accepted": True,
            "decision_outcome": decision["outcome"],
        }
        typed_public_result = SimpleNamespace(
            to_public_dict=lambda: copy.deepcopy(public_result)
        )
        loaded = SimpleNamespace(
            public_result=typed_public_result,
            decision=SimpleNamespace(to_public_dict=lambda: copy.deepcopy(decision)),
            record=loaded_record,
        )
        replay_entry = SimpleNamespace(
            public_decision={
                **decision,
                "generation_id": "generation-001",
                "validation_status": "valid",
            },
            current_best_before=SimpleNamespace(
                to_public_dict=lambda: copy.deepcopy(before)
            ),
            current_best_after=SimpleNamespace(
                to_public_dict=lambda: copy.deepcopy(after)
            ),
        )
        record = {
            "generation_id": "generation-001",
            "parent_current_best": before,
            "resulting_current_best": after,
        }
        verifier._assert_reassembled_formal_chain(
            record=record,
            trial=trial,
            formal=formal,
            loaded_execution=loaded,
            reconstructed_public_result=typed_public_result,
            replay_entry=replay_entry,
            execution_ref=SimpleNamespace(checksum=execution_checksum),
            sealed_ref=SimpleNamespace(checksum=sealed_checksum),
        )
        forged = SimpleNamespace(
            to_public_dict=lambda: {
                "status": "pass",
                "public_score": 0.56,
                "metrics": {},
            }
        )
        with self.assertRaisesRegex(
            AssertionError,
            "reconstructed formal public result",
        ):
            verifier._assert_reassembled_formal_chain(
                record=record,
                trial=trial,
                formal=formal,
                loaded_execution=loaded,
                reconstructed_public_result=forged,
                replay_entry=replay_entry,
                execution_ref=SimpleNamespace(checksum=execution_checksum),
                sealed_ref=SimpleNamespace(checksum=sealed_checksum),
            )

    def test_private_feedback_scan_rejects_exact_cell_or_score_leak(self) -> None:
        private = {
            "split": "public_accept",
            "cell_ids": ["public_accept.height_ood_01"],
            "j_primary": 0.8123456789012345,
        }
        public = {
            "split": "public_dev",
            "cell_ids": ["public_dev.height_ood_01"],
            "j_primary": 0.7123456789012345,
        }
        verifier._assert_prompt_payload_private_feedback_absent(
            payload={
                "protocol": {
                    "cohorts": ["public_accept", "sealed_validation"],
                    "public_accept_seed_count": 5,
                },
                "public_score": public["j_primary"],
            },
            private_summaries=[private],
            public_authorities=[public],
        )
        with self.assertRaisesRegex(AssertionError, "private eval feedback"):
            verifier._assert_prompt_payload_private_feedback_absent(
                payload={"cell": private["cell_ids"][0]},
                private_summaries=[private],
                public_authorities=[public],
            )
        with self.assertRaisesRegex(AssertionError, "private eval feedback"):
            verifier._assert_prompt_payload_private_feedback_absent(
                payload={"score": private["j_primary"]},
                private_summaries=[private],
                public_authorities=[public],
            )
        verifier._assert_prompt_payload_private_feedback_absent(
            payload={"score": private["j_primary"]},
            private_summaries=[private],
            public_authorities=[public, {"published_score": private["j_primary"]}],
        )

    def test_prompt_privacy_enumerates_every_authoring_attempt(self) -> None:
        genesis_payload = json.loads(
            verifier._P5_SESSION_GENESIS.read_text(encoding="utf-8")
        )
        genesis = verifier.SessionGenesisArtifact.from_mapping(genesis_payload)
        reference_results = {
            verifier.EvaluationPhase.PUBLIC_DEV: SimpleNamespace(
                task_evidence={
                    "summary": {
                        "split": "public_dev",
                        "j_primary": 0.7123456789012345,
                    }
                }
            ),
            verifier.EvaluationPhase.PUBLIC_ACCEPT: SimpleNamespace(
                task_evidence={
                    "summary": {
                        "split": "public_accept",
                        "cell_ids": ["public_accept.height_ood_01"],
                        "j_primary": 0.8123456789012345,
                    }
                }
            ),
            verifier.EvaluationPhase.SEALED_VALIDATION: SimpleNamespace(
                task_evidence={
                    "summary": {
                        "split": "sealed_validation",
                        "cell_ids": ["sealed_validation.height_ood_01"],
                        "j_primary": 0.8223456789012345,
                    }
                }
            ),
        }
        with tempfile.TemporaryDirectory(prefix="phase6-prompt-attempts-") as raw:
            root = Path(raw).resolve()
            refs_by_trial = {}
            for trial_id, count in (("generation-001-lane-a", 1), ("generation-001-lane-b", 2)):
                refs = []
                for index in range(count):
                    path = root / f"{trial_id}-{index}.json"
                    path.write_text("{}", encoding="utf-8")
                    refs.append(
                        SimpleNamespace(
                            path=str(path),
                            checksum="sha256:" + f"{len(refs_by_trial) + index + 3:064x}",
                            size_bytes=path.stat().st_size,
                        )
                    )
                refs_by_trial[trial_id] = tuple(refs)

            def journal(_round_dir, *, round_id):
                return SimpleNamespace(
                    snapshot=lambda: SimpleNamespace(
                        prompt_payload_refs=refs_by_trial[round_id]
                    )
                )

            ledger = [
                {
                    "generation_id": "generation-001",
                    "selection": None,
                    "parent_current_best": {"public_score": 0.7},
                    "resulting_current_best": {"public_score": 0.7},
                    "private_artifacts": [],
                    "trials": [
                        {
                            "lane_id": "lane-a",
                            "trial_id": "generation-001-lane-a",
                        },
                        {
                            "lane_id": "lane-b",
                            "trial_id": "generation-001-lane-b",
                        },
                    ],
                }
            ]
            with (
                mock.patch.object(
                    verifier,
                    "_reference_eval_phase_results",
                    return_value=reference_results,
                ),
                mock.patch.object(
                    verifier,
                    "_optional_generation_phase_result",
                    return_value=None,
                ),
                mock.patch.object(
                    verifier,
                    "_optional_generation_execution_public_evidence",
                    return_value=(),
                ),
                mock.patch.object(
                    verifier,
                    "RoundAuthoringJournal",
                    side_effect=journal,
                ),
                mock.patch.object(
                    verifier,
                    "verify_prompt_payload",
                    return_value={"public_dev_score": 0.7123456789012345},
                ) as verify_prompt,
            ):
                result = verifier._verify_all_authoring_prompt_privacy(
                    output_dir=root,
                    genesis=genesis,
                    ledger=ledger,
                )
        self.assertEqual(result["prompt_payload_ref_count"], 3)
        self.assertEqual(verify_prompt.call_count, 3)
        self.assertEqual(result["private_feedback_leak_count"], 0)


if __name__ == "__main__":
    unittest.main()
