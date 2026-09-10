"""Pin persisted cost/closure formats to data emitted before the shared host."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import audit_stair_generation_history as entrypoint
from true_rl_agent.agent_core.generation_closure import publish_or_validate_generation_closure
from true_rl_agent.agent_core.generation_cost_journal import (
    GenerationActualCostUsage, GenerationActualCostReceiptRef,
    generation_cost_receipt_projection, project_actual_generation_cost,
)
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.tasks.stair_climbing.generation_session_cost import StairGenerationSessionCostContractV3

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/generation_cost_closure_cd48221.json"


class HistoricalWireTests(unittest.TestCase):
    def test_pre_p4_cost_arithmetic_and_closure_bytes_remain_identical(self):
        frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
        contract = StairGenerationSessionCostContractV3.from_mapping(frozen["cost_contract"])
        expected = frozen["receipt"]
        usage = GenerationActualCostUsage(**{name: expected["actual"][name]
                                            for name in GenerationActualCostUsage.__dataclass_fields__})
        reserve, actual, checks = project_actual_generation_cost(contract, usage=usage,
            finalization_output_reserve_bytes=expected["actual"]["finalization_output_reserve_bytes"])
        self.assertEqual(reserve, expected["reservation"])
        self.assertEqual(actual, expected["actual"])
        self.assertEqual(checks, expected["cap_checks"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            closure = frozen["closure"]
            # Only the machine-local path is relocated; frozen arithmetic, hashes,
            # schema and every other closure field stay independently prescribed.
            receipt = GenerationActualCostReceiptRef(path=root / "cost.json",
                payload_hash=closure["actual_cost_receipt"]["payload_hash"], payload=expected)
            projection = generation_cost_receipt_projection(receipt)
            self.assertEqual(projection, {**closure["actual_cost_receipt"], "path": str(receipt.path)})
            from true_rl_agent.evaluation.binding import canonical_json_hash
            body = {key: value for key, value in closure.items() if key != "closure_hash"}
            body["actual_cost_receipt"] = {**body["actual_cost_receipt"], "path": str(receipt.path)}
            expected_closure = {**body, "closure_hash": canonical_json_hash(body)}
            foundation = SimpleNamespace(cost_contract=contract,
                config=SimpleNamespace(output_dir=root, session_id=closure["session_id"]))
            with patch("true_rl_agent.agent_core.generation_closure.validate_stored_resource_closure_evidence"):
                published = publish_or_validate_generation_closure(foundation,
                    ledger=[frozen["record"]], generation_record=frozen["record"],
                    generation_execution_receipt=closure["generation_execution_receipt"],
                    actual_cost_receipt=receipt, resource_evidence=closure["resource_evidence"],
                    segment_id=closure["closed_by_segment_id"])
            self.assertEqual(published, expected_closure)
            path = root / "generation_closures" / (closure["generation_id"] + ".json")
            self.assertEqual(path.read_bytes(), canonical_json_bytes(expected_closure, trailing_newline=True))

    def test_history_cli_reads_frozen_caps_and_dispatches_only_the_reader(self):
        frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "generation_session_cost.json").write_text(json.dumps(frozen["cost_contract"]), encoding="utf-8")
            with (patch.object(entrypoint, "audit_completed_stair_formal_session",
                return_value={"session_id": "history", "schema_version": "true_rl_agent.stair.generation_session_report.v6"}) as audit,
                contextlib.redirect_stdout(io.StringIO())):
                status = entrypoint.main(["--output-dir", str(root), "--lease-root", str(root / "leases"),
                    "--session-id", "history", "--cell-variant-id", "stair.g1_37dof.unified_v3",
                    "--expected-source-commit", "a" * 40, "--machine-id", "fixture",
                    "--gpu0-uuid", "GPU-00000000-0000-0000-0000-000000000001",
                    "--gpu1-uuid", "GPU-00000000-0000-0000-0000-000000000002"])
            self.assertEqual(status, 0)
            config = audit.call_args.args[0]
            self.assertTrue(config.terminal_audit_only and config.resume_existing)
            self.assertEqual(config.max_output_bytes, frozen["cost_contract"]["physical_cap"]["output_bytes"])
            self.assertFalse((root / "leases").exists())


if __name__ == "__main__":
    unittest.main()
