"""Three CPU-only carrier checks; no subprocess, auth or runtime-home creation."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare_desktop_review as adapter
import run_confirmation_review as review
import run_sol_review as runner

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jhZkAAAAASUVORK5CYII=")
ROOT = Path(__file__).resolve().parents[3]
POLICY = {"schema_version": "p3new.sol_cli_policy.v1", "status": "FROZEN", "model": "gpt-6.1-sol",
          "reasoning_effort": "high", "model_context_window": 512000, "model_auto_compact_token_limit": 450000}


def write(path, value):
    Path(path).write_bytes(adapter.encoded(value))


class CarrierTests(unittest.TestCase):
    def test_roundtrip_order_and_assignment(self):
        blocks = []
        for index in range(2):
            blocks.extend([{"type": "text", "text": adapter.encoded({"role": "raw_image", "material_id": str(index), "sha256": adapter.sha(PNG)}).decode()},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode(), "detail": "original"}},
                           {"type": "text", "text": "逐块 text\n0.000000 exact"}])
        messages = [{"role": "system", "content": "original system\n"}, {"role": "user", "content": blocks}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            carrier, images, _ = adapter.adapt_messages(messages)
            (root / "images").mkdir()
            for record, raw in images:
                (root / record["file"]).write_bytes(raw)
            write(root / "input.json", carrier)
            self.assertEqual(adapter.encoded(adapter.reconstruct_messages(root / "input.json")), adapter.encoded(messages))
            carrier["block_order"].reverse()
            write(root / "input.json", carrier)
            with self.assertRaisesRegex(ValueError, "sequence changed"):
                adapter.reconstruct_messages(root / "input.json")
            assignment = {"schema_version": review.ASSIGNMENT_SCHEMA, "status": "FROZEN", "packets": {"p": {
                "A": {"judge_id": review.SOL_PAIR[0], "model": "gpt-6.1-sol", "carrier": "fresh_cli"},
                "B": {"judge_id": review.SOL_PAIR[1], "model": "gpt-6.1-sol", "carrier": "fresh_subagent"}}}}
            write(root / "assignment.json", assignment)
            self.assertEqual(review.assignment_fields(root / "assignment.json", "p")["review_assignment"]["reviewers"], assignment["packets"]["p"])
            assignment["packets"]["p"]["A"]["model"] = "gpt-6-sol"
            write(root / "assignment.json", assignment)
            with self.assertRaisesRegex(ValueError, "model/carrier"):
                review.assignment_fields(root / "assignment.json", "p")

    def test_only_own_seal_report_and_facts_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared"
            prepared.mkdir()
            write(root / "policy.json", POLICY)
            answer = {"judge_id": "gpt6.1-sol", "packet_id": "p", "stage": "facts", "complete": True, "findings": []}
            write(root / "answer.json", answer)
            seal = {"schema_version": "p3new.confirmation_facts_seal.v1", "status": "SEALED", "judge_id": "gpt6.1-sol", "packet_id": "p", "answer": adapter.ref(root / "answer.json"), "reference_contract": None}
            write(root / "seal.json", seal)
            own = {"role": "own_sealed_facts", "material_id": "own_sealed_facts", "sha256": seal["answer"]["sha256"], "judge_id": "gpt6.1-sol", "data": answer}
            payload = {"messages": [{"role": "system", "content": "You are independent judge gpt6.1-sol for packet p, stage report. original"}, {"role": "user", "content": [{"type": "text", "text": adapter.encoded(own).decode()}]}], "max_tokens": 32768}
            sources = [{"id": "own_sealed_facts", "role": "own_sealed_facts", "sha256": own["sha256"]}]
            card = {"schema_version": "p3new.confirmation_review_prepared.v1", "status": "PREPARED_NOT_RELEASED", "judge_id": "gpt6.1-sol", "packet_id": "p", "stage": "report", "body_limit_bytes": review.MAX_BODY, "max_png_count": review.MAX_IMAGES, "max_output_tokens": 32768, "materials": sources, "png_count": 0, "own_facts_seal": adapter.ref(root / "seal.json"), "reference_contract": None, "runner": {}, "transport": {}, "manifest": {}}
            def refresh():
                body = adapter.encoded(payload)
                (prepared / "request.http_body.json").write_bytes(body)
                card.update(request_sha256=adapter.sha(body), request_bytes=len(body))
                write(prepared / "prepare.json", card)
            refresh()
            fake_coverage = SimpleNamespace(__file__=review.coverage.__file__, request_materials=lambda body: ({"own_sealed_facts": own}, {}))
            fake_review = SimpleNamespace(__file__=review.__file__, coverage=fake_coverage, MAX_BODY=review.MAX_BODY, MAX_IMAGES=review.MAX_IMAGES, SOL_PAIR=review.SOL_PAIR, envelope=review.envelope)
            with patch.object(adapter, "frozen_review", return_value=fake_review):
                receipt = adapter.prepare(prepared, root / "good", source_root=ROOT, cli_policy=root / "policy.json", own_facts_seal=root / "seal.json", allow_development_single=True)
                self.assertTrue(receipt["request_reconstructed_exactly"])
                self.assertEqual(receipt["own_facts_answer_verified_local"]["sha256"], seal["answer"]["sha256"])
                seal["judge_id"] = "gpt6.1-sol-b"
                write(root / "seal.json", seal)
                card["own_facts_seal"] = adapter.ref(root / "seal.json")
                refresh()
                with self.assertRaisesRegex(ValueError, "another identity/packet"):
                    adapter.prepare(prepared, root / "bad", source_root=ROOT, cli_policy=root / "policy.json", own_facts_seal=root / "seal.json", allow_development_single=True)
                card["stage"] = "facts"
                card["own_facts_seal"] = None
                payload["messages"][0]["content"] = payload["messages"][0]["content"].replace("stage report", "stage facts")
                refresh()
                with self.assertRaisesRegex(ValueError, "facts cannot carry answers"):
                    adapter.prepare(prepared, root / "facts", source_root=ROOT, cli_policy=root / "policy.json", allow_development_single=True)

    def test_actual_image_result_and_private_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            blind = Path(temp)
            path = blind / "image.png"
            path.write_bytes(PNG)
            expected = [{"file": "image.png", "sha256": adapter.sha(PNG)}]
            rows = [{"type": "session_meta", "payload": {"id": "new"}},
                    {"type": "response_item", "payload": {"type": "reasoning", "text": "PRIVATE_SENTINEL"}},
                    {"type": "turn_context", "payload": {"model": "gpt-6.1-sol", "effort": "high"}},
                    {"type": "response_item", "payload": {"type": "function_call", "name": "view_image", "call_id": "c", "arguments": json.dumps({"path": str(path), "detail": "original"})}},
                    {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "c", "output": [{"type": "input_image", "detail": "original", "image_url": "data:application/octet-stream;base64," + base64.b64encode(PNG).decode()}]}}]
            def trace():
                (blind / "rollout.jsonl").write_bytes(b"\n".join(adapter.encoded(row) for row in rows))
                return runner.selective_trace(blind / "rollout.jsonl", expected, blind)
            result = trace()
            self.assertTrue(result["access_verified"])
            self.assertEqual(result["tools"][0]["result_image_size"], [1, 1])
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
            native_call, native_result = deepcopy(rows[-2]), deepcopy(rows[-1])
            rows[-2]["payload"] = {"type": "custom_tool_call", "name": "exec", "call_id": "c",
                                    "input": runner.code_view_template("image.png") + "\n"}
            rows[-1]["payload"]["type"] = "custom_tool_call_output"
            self.assertTrue(trace()["access_verified"])
            rows[-2]["payload"]["input"] += "text('extra JavaScript');"
            self.assertFalse(trace()["access_verified"])
            rows[-2], rows[-1] = native_call, native_result
            rows[-1]["payload"]["output"][0]["detail"] = "high"
            self.assertFalse(trace()["access_verified"])
            rows[-1]["payload"]["output"] = "tool failed"
            self.assertFalse(trace()["access_verified"])
            rows[-1]["payload"]["output"] = [{"type": "input_image", "detail": "original", "image_url": "data:image/png;base64," + base64.b64encode(PNG).decode()}]
            rows.append({"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "extra", "arguments": "PRIVATE_SENTINEL"}})
            self.assertFalse(trace()["access_verified"])
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(trace()))
            rows[-1] = {"type": "response_item", "payload": {"type": "web_search_call", "query": "PRIVATE_SENTINEL"}}
            self.assertFalse(trace()["access_verified"])
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(trace()))
            with patch.dict(runner.os.environ, {"CODEX_THREAD_ID": "private", "CODEX_APP_TOOLS_PIPE_PATH": "private"}):
                environment, names = runner.child_environment(blind / "nonexistent-home")
                self.assertIn("CODEX_THREAD_ID", names)
                self.assertNotIn("CODEX_THREAD_ID", environment)
                self.assertFalse((blind / "nonexistent-home").exists())


if __name__ == "__main__":
    unittest.main()
