from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.candidates.prompt_projection import build_prompt_payload
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload
from true_rl_agent.prompting.source_catalog import (
    MAX_PROMPT_SOURCE_INLINE_BYTES,
    MAX_PROMPT_SOURCE_READ_BYTES,
    build_prompt_source_context,
)


class PromptSourceCatalogTests(unittest.TestCase):
    def test_no_accepted_records_exports_pending_source_catalog_and_none_lineage(self) -> None:
        payload = PublicContextBuilder().build_prompt_payload([])

        catalog = payload["source_catalog"]
        lineage = payload["candidate_lineage"]
        frame = payload["authoring_context_frame"]

        logical_ids = [item["logical_id"] for item in catalog["items"]]
        self.assertEqual(
            logical_ids,
            [
                "api/candidate_components.py",
                "current_best/training_code/components.py",
                "examples/known_good_reward_terms.py",
                "attempt/latest_failed/training_code/components.py",
            ],
        )
        for item in catalog["items"]:
            self.assertIn("role", item)
            self.assertIn("candidate_use", item)
            self.assertIn("max_bytes", item)
            self.assertTrue(item.get("content_sha256") or item.get("pending_reason"))
        self.assertEqual(lineage["status"], "none")
        self.assertIsNone(lineage["current_best"])
        self.assertEqual(payload["source_payloads"], [])
        self.assertIn("source_catalog", frame["included_sections"])
        self.assertIn("candidate_lineage", frame["included_sections"])
        self.assertIn("source_payloads", frame["included_sections"])
        self.assertIn("exploration_directive", frame["included_sections"])
        assert_public_only_prompt_payload(payload)

    def test_current_best_source_is_inlined_from_latest_accepted_without_path_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / "candidate-old"
            latest_dir = root / "candidate-latest"
            for candidate_dir, term_name in ((old_dir, "old_reward"), (latest_dir, "latest_reward")):
                (candidate_dir / "training_code").mkdir(parents=True)
                (candidate_dir / "training_code" / "components.py").write_text(
                    "\n".join(
                        [
                            "from __future__ import annotations",
                            "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents",
                            "PATH_A = 'D:/humanoid_agent/private/raw.txt'",
                            "PATH_B = '/data/yuyangliu/secrets/key.env'",
                            f"def {term_name}(inp):",
                            "    return inp.base_height",
                            "def build_training_components() -> TrainingComponents:",
                            f"    return TrainingComponents(name='{term_name}', reward_terms=[RewardTerm(name='{term_name}', weight=1.0, fn={term_name})])",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

            records = [
                _accepted_record("round-001", "candidate-old", old_dir, 0.20),
                {
                    "round_id": "round-002",
                    "decision": {"accepted": False, "outcome": "reject", "public_score": 0.10},
                    "candidate": {
                        "candidate_id": "candidate-rejected",
                        "path": str(root / "candidate-rejected"),
                        "changed_components": ["reward_callable"],
                    },
                },
                _accepted_record("round-003", "candidate-latest", latest_dir, 0.35),
            ]

            payload = PublicContextBuilder().build_prompt_payload(records)

        text = json.dumps(payload, sort_keys=True)
        lineage = payload["candidate_lineage"]
        current_best = lineage["current_best"]
        source_payload = payload["source_payloads"][0]
        scrubbed_content = source_payload["content"]
        expected_hash = hashlib.sha256(scrubbed_content.encode("utf-8")).hexdigest()

        self.assertEqual(lineage["status"], "accepted_current_best")
        self.assertEqual(current_best["round_id"], "round-003")
        self.assertEqual(current_best["candidate_id"], "candidate-latest")
        self.assertEqual(current_best["public_score"], 0.35)
        self.assertEqual(current_best["source_logical_id"], "current_best/training_code/components.py")
        self.assertEqual(current_best["source_sha256"], expected_hash)
        self.assertEqual(current_best["source_delivery"], "inline")
        self.assertEqual(source_payload["logical_id"], "current_best/training_code/components.py")
        self.assertEqual(source_payload["content_sha256"], expected_hash)
        self.assertIn("def latest_reward", scrubbed_content)
        self.assertNotIn("def old_reward", scrubbed_content)
        self.assertIn("[path]", scrubbed_content)
        self.assertNotIn("D:/humanoid_agent", text)
        self.assertNotIn("/data/yuyangliu", text)
        self.assertNotIn("private/raw.txt", text)
        self.assertNotIn("/secrets/", text)
        self.assertNotIn(str(latest_dir), text)
        self.assertNotIn(str(old_dir), text)
        self.assertNotIn('"path"', json.dumps(payload["public_context"], sort_keys=True))
        assert_public_only_prompt_payload(payload)

    def test_oversized_current_best_source_is_omitted_but_hash_and_summary_remain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-large"
            (candidate_dir / "training_code").mkdir(parents=True)
            large_body = "x = 1\n" + ("# filler\n" * (MAX_PROMPT_SOURCE_INLINE_BYTES // 4))
            (candidate_dir / "training_code" / "components.py").write_text(
                large_body,
                encoding="utf-8",
            )

            payload = PublicContextBuilder().build_prompt_payload(
                [_accepted_record("round-001", "candidate-large", candidate_dir, 0.40)]
            )

        current_best = payload["candidate_lineage"]["current_best"]
        catalog_item = _catalog_item(payload, "current_best/training_code/components.py")
        text = json.dumps(payload, sort_keys=True)

        self.assertEqual(current_best["source_delivery"], "omitted_over_budget")
        self.assertEqual(catalog_item["status"], "omitted_over_budget")
        self.assertIsNotNone(catalog_item["content_sha256"])
        self.assertGreater(catalog_item["bytes"], MAX_PROMPT_SOURCE_INLINE_BYTES)
        self.assertEqual(payload["source_payloads"], [])
        self.assertIn("source omitted because it exceeds inline budget", current_best["source_summary"])
        self.assertNotIn("# filler", text)
        self.assertNotIn(str(candidate_dir), text)
        assert_public_only_prompt_payload(payload)

    def test_non_utf8_current_best_source_is_unavailable_not_crashing_prompt_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-bad-encoding"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_bytes(b"def bad(inp):\n    return \xff\n")

            payload = PublicContextBuilder().build_prompt_payload(
                [_accepted_record("round-001", "candidate-bad-encoding", candidate_dir, 0.40)]
            )

        current_best = payload["candidate_lineage"]["current_best"]
        catalog_item = _catalog_item(payload, "current_best/training_code/components.py")

        self.assertEqual(current_best["source_delivery"], "unavailable")
        self.assertEqual(current_best["source_unavailable_reason"], "candidate_components_source_read_failed")
        self.assertEqual(catalog_item["pending_reason"], "candidate_components_source_read_failed")
        self.assertEqual(payload["source_payloads"], [])
        assert_public_only_prompt_payload(payload)

    def test_candidate_public_projection_removes_path_but_private_artifact_keeps_source_ref(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "round-001",
                "decision": {"accepted": True, "outcome": "policy_improvement_accept", "public_score": 0.2},
                "candidate": {
                    "candidate_id": "candidate-001",
                    "path": "D:/humanoid_agent/private/candidates/candidate-001",
                    "changed_components": ["reward_callable"],
                },
            }
        )
        candidate = CandidatePackageRef(
            candidate_id="candidate-001",
            path="D:/humanoid_agent/private/candidates/candidate-001",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
        )

        public_record = record.ledger_public_record()

        self.assertNotIn("path", public_record["candidate"])
        self.assertEqual(record.private_artifacts[0].kind, "candidate_package")
        self.assertEqual(record.private_artifacts[0].uri, "D:/humanoid_agent/private/candidates/candidate-001")
        self.assertNotIn("path", candidate.to_public_dict())

    def test_source_resolver_does_not_consume_public_candidate_path_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-public-path-only"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def leaked_public_path_source(inp):\n    return 0.0\n",
                encoding="utf-8",
            )

            context = build_prompt_source_context([_accepted_record("round-001", "candidate-001", candidate_dir, 0.2)])

        self.assertEqual(context["candidate_lineage"]["status"], "accepted_current_best")
        current_best = context["candidate_lineage"]["current_best"]
        self.assertEqual(current_best["source_delivery"], "unavailable")
        self.assertEqual(current_best["source_unavailable_reason"], "accepted_record_has_no_candidate_package_source_ref")
        self.assertEqual(context["source_payloads"], [])

    def test_latest_failed_source_can_inline_without_accepted_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-failed"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "\n".join(
                    [
                        "PATH_A = 'D:/humanoid_agent/private/raw.txt'",
                        "def failed_reward(inp):",
                        "    return inp.base_height",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            failed_candidate = CandidatePackageRef(
                candidate_id="candidate-failed",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            context = build_prompt_source_context([], latest_failed_candidate=failed_candidate)

        payload = build_prompt_payload(public_context=context)
        failed_item = _catalog_item(payload, "attempt/latest_failed/training_code/components.py")
        failed_payload = payload["source_payloads"][0]
        text = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["candidate_lineage"]["status"], "none")
        self.assertEqual(failed_item["status"], "inline")
        self.assertEqual(failed_payload["logical_id"], "attempt/latest_failed/training_code/components.py")
        self.assertIn("def failed_reward", failed_payload["content"])
        self.assertIn("[path]", failed_payload["content"])
        self.assertNotIn("D:/humanoid_agent", text)
        self.assertNotIn("private/raw.txt", text)
        assert_public_only_prompt_payload(payload)

    def test_non_utf8_latest_failed_source_is_unavailable_not_crashing_prompt_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-failed-bad-encoding"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_bytes(b"def bad(inp):\n    return \xff\n")
            failed_candidate = CandidatePackageRef(
                candidate_id="candidate-failed-bad-encoding",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            payload = build_prompt_payload(
                public_context=build_prompt_source_context([], latest_failed_candidate=failed_candidate)
            )

        failed_item = _catalog_item(payload, "attempt/latest_failed/training_code/components.py")

        self.assertEqual(failed_item["status"], "pending")
        self.assertEqual(failed_item["pending_reason"], "candidate_components_source_read_failed")
        self.assertEqual(payload["source_payloads"], [])
        assert_public_only_prompt_payload(payload)

    def test_latest_runtime_failed_record_exposes_failed_source_without_public_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-runtime-failed"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def runtime_failed_reward(inp):\n    return inp.base_height\n",
                encoding="utf-8",
            )
            record = ExperimentRecord.from_ledger_record(
                {
                    "round_id": "round-001",
                    "decision": {
                        "accepted": False,
                        "outcome": "reject",
                        "reason": "candidate_runtime_failed",
                    },
                    "candidate": {
                        "candidate_id": "candidate-runtime-failed",
                        "path": str(candidate_dir),
                        "changed_components": ["reward_callable"],
                    },
                    "runner_error": {
                        "type": "CandidateRuntimeError",
                        "message": "candidate failed inside runner",
                    },
                }
            )

            payload = build_prompt_payload(public_context=build_prompt_source_context([record]))

        failed_item = _catalog_item(payload, "attempt/latest_failed/training_code/components.py")
        failed_payload = payload["source_payloads"][0]
        text = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["candidate_lineage"]["status"], "none")
        self.assertEqual(failed_item["status"], "inline")
        self.assertEqual(failed_payload["logical_id"], "attempt/latest_failed/training_code/components.py")
        self.assertIn("runtime_failed_reward", failed_payload["content"])
        self.assertNotIn(str(candidate_dir), text)
        self.assertNotIn('"path"', json.dumps(payload["public_context"], sort_keys=True))
        assert_public_only_prompt_payload(payload)

    def test_runtime_failed_source_is_dropped_after_intervening_accept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed_dir = root / "candidate-runtime-failed"
            accepted_dir = root / "candidate-accepted"
            for candidate_dir, term_name in (
                (failed_dir, "stale_runtime_failed_reward"),
                (accepted_dir, "accepted_success_reward"),
            ):
                (candidate_dir / "training_code").mkdir(parents=True)
                (candidate_dir / "training_code" / "components.py").write_text(
                    f"def {term_name}(inp):\n    return inp.base_height\n",
                    encoding="utf-8",
                )

            failed_record = {
                "round_id": "round-001",
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "candidate_runtime_failed",
                },
                "candidate": {
                    "candidate_id": "candidate-runtime-failed",
                    "path": str(failed_dir),
                    "changed_components": ["reward_callable"],
                },
                "runner_error": {
                    "type": "CandidateRuntimeError",
                    "message": "candidate failed inside runner",
                },
            }
            accepted_record = _accepted_record("round-002", "candidate-accepted", accepted_dir, 0.50)

            payload = PublicContextBuilder().build_prompt_payload([failed_record, accepted_record])

        failed_item = _catalog_item(payload, "attempt/latest_failed/training_code/components.py")
        failed_sources = [
            item
            for item in payload["source_payloads"]
            if item["logical_id"] == "attempt/latest_failed/training_code/components.py"
        ]
        text = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["candidate_lineage"]["status"], "accepted_current_best")
        self.assertEqual(failed_item["status"], "pending")
        self.assertEqual(failed_item["pending_reason"], "no_latest_failed_candidate")
        self.assertEqual(failed_sources, [])
        self.assertIn("accepted_success_reward", text)
        self.assertNotIn("stale_runtime_failed_reward", text)
        assert_public_only_prompt_payload(payload)

    def test_candidate_components_symlink_is_rejected_before_resolving_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-symlink"
            training_dir = candidate_dir / "training_code"
            training_dir.mkdir(parents=True)
            link_path = training_dir / "components.py"
            target_path = training_dir / "real_components.py"
            target_path.write_text("def symlink_target_reward(inp):\n    return inp.base_height\n", encoding="utf-8")
            failed_candidate = CandidatePackageRef(
                candidate_id="candidate-symlink",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            original_resolve = Path.resolve

            def fake_resolve(path: Path, *args, **kwargs) -> Path:
                if path == link_path:
                    return target_path
                return original_resolve(path, *args, **kwargs)

            def fake_is_symlink(path: Path) -> bool:
                return path == link_path

            with patch.object(Path, "resolve", fake_resolve), patch.object(Path, "is_symlink", fake_is_symlink):
                payload = build_prompt_payload(
                    public_context=build_prompt_source_context([], latest_failed_candidate=failed_candidate)
                )

        failed_item = _catalog_item(payload, "attempt/latest_failed/training_code/components.py")
        text = json.dumps(payload, sort_keys=True)

        self.assertEqual(failed_item["status"], "pending")
        self.assertEqual(failed_item["pending_reason"], "candidate_components_source_is_symlink")
        self.assertEqual(payload["source_payloads"], [])
        self.assertNotIn("symlink_target_reward", text)
        assert_public_only_prompt_payload(payload)

    def test_oversized_latest_failed_source_is_omitted_but_metadata_remains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-failed-large"
            (candidate_dir / "training_code").mkdir(parents=True)
            oversized = "def failed_large_reward(inp):\n    return inp.base_height\n" + (
                "# filler\n" * (MAX_PROMPT_SOURCE_INLINE_BYTES // 4)
            )
            (candidate_dir / "training_code" / "components.py").write_text(oversized, encoding="utf-8")
            failed_candidate = CandidatePackageRef(
                candidate_id="candidate-failed-large",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            payload = build_prompt_payload(
                public_context=build_prompt_source_context([], latest_failed_candidate=failed_candidate)
            )

        failed_item = _catalog_item(payload, "attempt/latest_failed/training_code/components.py")
        text = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["candidate_lineage"]["status"], "none")
        self.assertEqual(failed_item["status"], "omitted_over_budget")
        self.assertIsNotNone(failed_item["content_sha256"])
        self.assertGreater(failed_item["bytes"], MAX_PROMPT_SOURCE_INLINE_BYTES)
        self.assertEqual(payload["source_payloads"], [])
        self.assertNotIn("# filler", text)
        assert_public_only_prompt_payload(payload)

    def test_prompt_payload_rejects_source_payload_outside_catalog(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt_payload(
                public_context={
                    "source_payloads": [
                        {
                            "schema_version": "true_rl_agent.prompt_source_payload.v1",
                            "logical_id": "bad/source.py",
                            "role": "latest_accepted_candidate_source",
                            "delivery": "inline",
                            "content": "def bad(inp):\n    return 0.0\n",
                            "content_sha256": hashlib.sha256(b"def bad(inp):\n    return 0.0\n").hexdigest(),
                            "bytes": len(b"def bad(inp):\n    return 0.0\n"),
                            "summary": "1 def",
                            "public_safe": True,
                        }
                    ],
                }
            )

    def test_prompt_payload_rejects_source_payload_hash_or_byte_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-inline"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def stable_reward(inp):\n    return 0.0\n",
                encoding="utf-8",
            )
            sections = build_prompt_source_context(
                [ExperimentRecord.from_ledger_record(_accepted_record("round-001", "candidate-001", candidate_dir, 0.2))]
            )
            sections["source_payloads"][0]["bytes"] += 1

            with self.assertRaises(ValueError):
                build_prompt_payload(public_context=sections)

    def test_prompt_payload_rejects_lineage_source_payload_ref_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-inline"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def stable_reward(inp):\n    return 0.0\n",
                encoding="utf-8",
            )
            sections = build_prompt_source_context(
                [ExperimentRecord.from_ledger_record(_accepted_record("round-001", "candidate-001", candidate_dir, 0.2))]
            )
            sections["candidate_lineage"]["current_best"]["source_payload_ref"] = "attempt/latest_failed/training_code/components.py"

            with self.assertRaises(ValueError):
                build_prompt_payload(public_context=sections)

    def test_over_read_budget_current_best_source_keeps_hash_bytes_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-huge"
            (candidate_dir / "training_code").mkdir(parents=True)
            source = "def huge_reward(inp):\n    return 0.0\n" + ("# filler\n" * (MAX_PROMPT_SOURCE_READ_BYTES // 4))
            (candidate_dir / "training_code" / "components.py").write_text(source, encoding="utf-8")

            payload = PublicContextBuilder().build_prompt_payload(
                [_accepted_record("round-001", "candidate-huge", candidate_dir, 0.4)]
            )

        current_best = payload["candidate_lineage"]["current_best"]
        catalog_item = _catalog_item(payload, "current_best/training_code/components.py")

        self.assertEqual(current_best["source_delivery"], "omitted_over_budget")
        self.assertEqual(catalog_item["status"], "omitted_over_budget")
        self.assertIsNotNone(catalog_item["content_sha256"])
        self.assertIsNotNone(current_best["source_sha256"])
        self.assertEqual(catalog_item["content_sha256"], current_best["source_sha256"])
        self.assertGreater(catalog_item["bytes"], MAX_PROMPT_SOURCE_READ_BYTES)
        self.assertIn("read budget", current_best["source_summary"])
        self.assertEqual(payload["source_payloads"], [])

def _accepted_record(round_id: str, candidate_id: str, candidate_dir: Path, public_score: float) -> dict:
    return {
        "round_id": round_id,
        "decision": {
            "accepted": True,
            "outcome": "policy_improvement_accept",
            "public_score": public_score,
        },
        "candidate": {
            "candidate_id": candidate_id,
            "path": str(candidate_dir),
            "changed_components": ["reward_callable"],
        },
    }


def _catalog_item(payload: dict, logical_id: str) -> dict:
    for item in payload["source_catalog"]["items"]:
        if item["logical_id"] == logical_id:
            return item
    raise AssertionError(f"missing source catalog item: {logical_id}")


if __name__ == "__main__":
    unittest.main()
