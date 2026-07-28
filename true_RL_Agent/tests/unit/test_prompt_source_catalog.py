from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.prompt_projection import build_prompt_payload
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.evaluation.acceptance_protocol import (
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.memory import (
    ExperimentRecord,
)
from true_rl_agent.memory import (
    PublicContextBuilder as _PublicContextBuilder,
)
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload
from true_rl_agent.prompting.source_catalog import (
    MAX_PROMPT_SOURCE_INLINE_BYTES,
    MAX_PROMPT_SOURCE_READ_BYTES,
)
from true_rl_agent.prompting.source_catalog import (
    build_prompt_source_context as _build_prompt_source_context,
)

SOURCE_CATALOG_ACCEPTANCE_PROTOCOL = AcceptanceProtocol(
    protocol_id="source_catalog_acceptance_v1",
    score_source="public_accept.score_stats.mean",
    min_score_count=1,
    absolute_accept_floor=0.0,
    min_delta=0.0,
    public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
)
SOURCE_CATALOG_EVALUATION_PROTOCOL = EvaluationProtocol(
    eval_protocol_version="source_catalog_eval_v1",
    public_seed_count=1,
    public_accept_seed_count=1,
    sealed_validation_seed_count=7,
    eval_num_envs=64,
)
SOURCE_CATALOG_EVALUATION_BINDING = EvaluationBindingV1(
    eval_protocol_hash=SOURCE_CATALOG_EVALUATION_PROTOCOL.protocol_hash,
    acceptance_protocol_hash=SOURCE_CATALOG_ACCEPTANCE_PROTOCOL.protocol_hash,
    task_protocol_hash="d" * 64,
    seed_cohort_hash="e" * 64,
)
SOURCE_CATALOG_SESSION_GENESIS_HASH = "f" * 64
SOURCE_CATALOG_TRAINING_BINDING_HASH = "1" * 64
SOURCE_CATALOG_CAPABILITY_PROFILE_HASH = "2" * 64


def _bootstrap_current_best() -> CurrentBestEvalStats:
    return CurrentBestEvalStats.bootstrap(
        initial_best_score=0.0,
        acceptance_protocol=SOURCE_CATALOG_ACCEPTANCE_PROTOCOL,
        evaluation_binding=SOURCE_CATALOG_EVALUATION_BINDING,
        anchor_id="source-catalog-bootstrap",
    )


def _bind_source_records(records):
    bound = []
    for record in records:
        if isinstance(record, ExperimentRecord):
            public_record = record.ledger_public_record()
            public_record["session_genesis_hash"] = (
                SOURCE_CATALOG_SESSION_GENESIS_HASH
            )
            public_record["evaluation_binding"] = (
                SOURCE_CATALOG_EVALUATION_BINDING.to_public_dict()
            )
            bound.append(replace(record, public_record=public_record))
        else:
            item = dict(record)
            item["session_genesis_hash"] = SOURCE_CATALOG_SESSION_GENESIS_HASH
            item["evaluation_binding"] = (
                SOURCE_CATALOG_EVALUATION_BINDING.to_public_dict()
            )
            bound.append(item)
    return bound


class _BoundPublicContextBuilder(_PublicContextBuilder):
    def _prepare_records(self, records):
        return super()._prepare_records(_bind_source_records(records))


def PublicContextBuilder(*args, **kwargs):
    kwargs.setdefault(
        "acceptance_protocol",
        SOURCE_CATALOG_ACCEPTANCE_PROTOCOL,
    )
    kwargs.setdefault("initial_current_best", _bootstrap_current_best())
    kwargs.setdefault(
        "session_genesis_hash",
        SOURCE_CATALOG_SESSION_GENESIS_HASH,
    )
    return _BoundPublicContextBuilder(*args, **kwargs)


def build_prompt_source_context(*args, **kwargs):
    args = (_bind_source_records(args[0]), *args[1:])
    kwargs.setdefault(
        "acceptance_protocol",
        SOURCE_CATALOG_ACCEPTANCE_PROTOCOL,
    )
    kwargs.setdefault("initial_current_best", _bootstrap_current_best())
    kwargs.setdefault(
        "session_genesis_hash",
        SOURCE_CATALOG_SESSION_GENESIS_HASH,
    )
    return _build_prompt_source_context(*args, **kwargs)


class PromptSourceCatalogTests(unittest.TestCase):
    def test_no_accepted_records_exports_pending_source_catalog_and_none_lineage(self) -> None:
        payload = PublicContextBuilder().build_prompt_payload([])

        catalog = payload["source_catalog"]
        lineage = payload["candidate_lineage"]
        frame = payload["authoring_context_frame"]

        self.assertTrue(catalog["policy"]["source_request_enabled"])
        self.assertTrue(catalog["policy"]["ids_are_closed_allowlist"])
        self.assertFalse(catalog["policy"]["provider_filesystem_access"])
        logical_ids = [item["logical_id"] for item in catalog["items"]]
        self.assertEqual(
            logical_ids,
            [
                "api/candidate_components.py",
                "current_best/training_code/components.py",
                "memory/best_observed/training_code/components.py",
                "examples/known_good_reward_terms.py",
                "attempt/latest_failed/training_code/components.py",
            ],
        )
        for item in catalog["items"]:
            self.assertIn("role", item)
            self.assertIn("candidate_use", item)
            self.assertIn("max_bytes", item)
            self.assertIn("requestable", item)
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
                _accepted_record(
                    "round-003",
                    "candidate-latest",
                    latest_dir,
                    0.35,
                    current_best_score=0.20,
                    current_best_round_id="round-001",
                    current_best_candidate_id="candidate-old",
                ),
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

    def test_contradictory_accept_does_not_become_current_best_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-invalid-accept"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def invalid_accept_source(inp):\n    return 0.0\n",
                encoding="utf-8",
            )
            record = _accepted_record(
                "round-invalid",
                "candidate-invalid",
                candidate_dir,
                0.99,
            )
            record["decision"].update(
                accepted=True,
                outcome="reject",
                reason="candidate_verifier_failed",
            )

            payload = PublicContextBuilder().build_prompt_payload([record])

        self.assertEqual(payload["candidate_lineage"]["status"], "none")
        self.assertEqual(payload["source_payloads"], [])

    def test_sequence_invalid_accept_cannot_replace_lineage_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_dir = root / "candidate-first"
            forged_dir = root / "candidate-forged"
            for candidate_dir, function_name in (
                (first_dir, "first_valid_source"),
                (forged_dir, "forged_source"),
            ):
                (candidate_dir / "training_code").mkdir(parents=True)
                (candidate_dir / "training_code" / "components.py").write_text(
                    f"def {function_name}(inp):\n    return 0.0\n",
                    encoding="utf-8",
                )
            records = [
                _accepted_record(
                    "round-001", "candidate-first", first_dir, 0.50
                ),
                _accepted_record(
                    "round-002", "candidate-forged", forged_dir, 0.51
                ),
            ]

            payload = PublicContextBuilder().build_prompt_payload(records)

        lineage = payload["candidate_lineage"]
        source_text = json.dumps(payload["source_payloads"], sort_keys=True)
        self.assertEqual(lineage["current_best"]["round_id"], "round-001")
        self.assertEqual(
            lineage["current_best"]["candidate_id"], "candidate-first"
        )
        self.assertIn("first_valid_source", source_text)
        self.assertNotIn("forged_source", source_text)

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
                        "surface_classification": "code_level",
                        "public_score": None,
                        "comparison": {},
                        "consumed_gates": [],
                        "gate_statuses": {},
                    },
                    "candidate": {
                        "candidate_id": "candidate-runtime-failed",
                        "path": str(candidate_dir),
                        "changed_components": ["reward_callable"],
                    },
                    "candidate_surface": {"classification": "code_level"},
                    "verifier": {"status": "pass"},
                    "jobs": {
                        "train": {"job_id": "round-001-train"},
                        "eval": {"job_id": "round-001-eval"},
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

    def test_latest_evaluated_reject_exposes_source_for_next_round_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-evaluated-reject"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def evaluated_reward(inp):\n    return inp.base_height - 0.5 * inp.pitch * inp.pitch\n",
                encoding="utf-8",
            )
            rejected = _accepted_record(
                "round-001",
                "candidate-evaluated-reject",
                candidate_dir,
                0.20,
            )
            required = rejected["public_result"]["metrics"]["acceptance"][
                "required_pass_gates"
            ]
            required.append("behavior_quality_gate")
            rejected["public_result"]["metrics"]["behavior_quality_gate"] = {
                "status": "fail"
            }
            rejected["decision"].update(
                accepted=False,
                outcome="reject",
                reason="required_gate_failed",
            )
            rejected["decision"]["consumed_gates"].append("behavior_quality_gate")
            rejected["decision"]["gate_statuses"]["behavior_quality_gate"] = "fail"
            record = ExperimentRecord.from_ledger_record(rejected)

            payload = build_prompt_payload(
                public_context=build_prompt_source_context([record])
            )

        failed_item = _catalog_item(
            payload,
            "attempt/latest_failed/training_code/components.py",
        )
        failed_payload = payload["source_payloads"][0]
        payload_text = json.dumps(payload, sort_keys=True)

        lineage = payload["candidate_lineage"]
        observed = lineage["best_observed_experiment"]
        best_item = _catalog_item(
            payload,
            "memory/best_observed/training_code/components.py",
        )

        self.assertEqual(lineage["status"], "bootstrap_observed_anchor")
        self.assertIsNone(lineage["current_best"])
        self.assertFalse(observed["accepted"])
        self.assertEqual(
            observed["authority"],
            "authoring_anchor_only_not_policy_best",
        )
        self.assertEqual(observed["candidate_id"], "candidate-evaluated-reject")
        self.assertEqual(observed["source_logical_id"], failed_payload["logical_id"])
        self.assertEqual(best_item["status"], "alias")
        self.assertEqual(best_item["alias_of"], failed_payload["logical_id"])
        self.assertEqual(
            payload["exploration_directive"]["mode"],
            "bootstrap_refine",
        )
        self.assertEqual(failed_item["status"], "inline")
        self.assertEqual(failed_item["role"], "latest_nonaccepted_candidate_source")
        self.assertEqual(
            failed_payload["role"],
            "latest_nonaccepted_candidate_source",
        )
        self.assertIn("0.5 * inp.pitch", failed_payload["content"])
        self.assertNotIn(str(candidate_dir), payload_text)
        assert_public_only_prompt_payload(payload)

    def test_bootstrap_anchor_keeps_best_observed_source_not_worse_latest_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            better_dir = root / "candidate-better"
            worse_dir = root / "candidate-worse"
            for candidate_dir, term_name in (
                (better_dir, "better_reward"),
                (worse_dir, "worse_reward"),
            ):
                (candidate_dir / "training_code").mkdir(parents=True)
                (candidate_dir / "training_code" / "components.py").write_text(
                    f"def {term_name}(inp):\n    return inp.base_height\n",
                    encoding="utf-8",
                )

            better = _evaluated_reject_record(
                "round-001",
                "candidate-better",
                better_dir,
                0.40,
            )
            worse = _evaluated_reject_record(
                "round-002",
                "candidate-worse",
                worse_dir,
                0.10,
            )
            payload = PublicContextBuilder().build_prompt_payload(
                [
                    ExperimentRecord.from_ledger_record(better),
                    ExperimentRecord.from_ledger_record(worse),
                ]
            )

        lineage = payload["candidate_lineage"]
        observed = lineage["best_observed_experiment"]
        payloads = {
            item["logical_id"]: item for item in payload["source_payloads"]
        }

        self.assertEqual(lineage["status"], "bootstrap_observed_anchor")
        self.assertEqual(observed["round_id"], "round-001")
        self.assertEqual(observed["candidate_id"], "candidate-better")
        self.assertEqual(observed["public_score"], 0.40)
        self.assertEqual(
            observed["source_logical_id"],
            "memory/best_observed/training_code/components.py",
        )
        self.assertEqual(
            set(payloads),
            {
                "memory/best_observed/training_code/components.py",
                "attempt/latest_failed/training_code/components.py",
            },
        )
        self.assertIn(
            "better_reward",
            payloads[
                "memory/best_observed/training_code/components.py"
            ]["content"],
        )
        self.assertIn(
            "worse_reward",
            payloads[
                "attempt/latest_failed/training_code/components.py"
            ]["content"],
        )
        self.assertNotEqual(
            payloads["memory/best_observed/training_code/components.py"][
                "content_sha256"
            ],
            payloads["attempt/latest_failed/training_code/components.py"][
                "content_sha256"
            ],
        )
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

    def test_current_best_and_latest_evaluated_reject_sources_are_both_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accepted_dir = root / "candidate-accepted"
            rejected_dir = root / "candidate-rejected"
            for candidate_dir, source in (
                (
                    accepted_dir,
                    "def accepted_reward(inp):\n    return inp.base_height\n",
                ),
                (
                    rejected_dir,
                    "def rejected_reward(inp):\n    return inp.base_height - 0.5 * inp.pitch * inp.pitch\n",
                ),
            ):
                (candidate_dir / "training_code").mkdir(parents=True)
                (candidate_dir / "training_code" / "components.py").write_text(
                    source,
                    encoding="utf-8",
                )
            accepted = _accepted_record(
                "round-001",
                "candidate-accepted",
                accepted_dir,
                0.50,
            )
            rejected = _accepted_record(
                "round-002",
                "candidate-rejected",
                rejected_dir,
                0.40,
                current_best_score=0.50,
                current_best_round_id="round-001",
                current_best_candidate_id="candidate-accepted",
            )
            rejected["decision"].update(
                accepted=False,
                outcome="reject",
                reason="noise_aware_delta_not_met",
            )

            payload = PublicContextBuilder().build_prompt_payload(
                [accepted, rejected]
            )

        payloads = {
            item["logical_id"]: item for item in payload["source_payloads"]
        }
        lineage = payload["candidate_lineage"]

        self.assertEqual(lineage["status"], "accepted_current_best")
        self.assertEqual(
            lineage["current_best"]["candidate_id"],
            "candidate-accepted",
        )
        self.assertEqual(
            set(payloads),
            {
                "current_best/training_code/components.py",
                "attempt/latest_failed/training_code/components.py",
            },
        )
        self.assertIn(
            "accepted_reward",
            payloads["current_best/training_code/components.py"]["content"],
        )
        self.assertIn(
            "0.5 * inp.pitch",
            payloads[
                "attempt/latest_failed/training_code/components.py"
            ]["content"],
        )
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

    def test_prompt_payload_rejects_bootstrap_anchor_with_policy_best_authority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "candidate-rejected"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "training_code" / "components.py").write_text(
                "def rejected_reward(inp):\n    return inp.base_height\n",
                encoding="utf-8",
            )
            record = ExperimentRecord.from_ledger_record(
                _evaluated_reject_record(
                    "round-001",
                    "candidate-rejected",
                    candidate_dir,
                    0.2,
                )
            )
            sections = build_prompt_source_context([record])

        sections["candidate_lineage"]["best_observed_experiment"][
            "authority"
        ] = "accepted_policy_best"
        with self.assertRaisesRegex(ValueError, "invalid authority"):
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

def _accepted_record(
    round_id: str,
    candidate_id: str,
    candidate_dir: Path,
    public_score: float,
    *,
    current_best_score: float | None = None,
    current_best_round_id: str | None = None,
    current_best_candidate_id: str | None = None,
) -> dict:
    protocol_hash = SOURCE_CATALOG_EVALUATION_PROTOCOL.protocol_hash
    checkpoint_hash = "sha256:" + "f" * 64
    source_path = candidate_dir / "training_code" / "components.py"
    source_checksum = "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
    subject_manifest_hash = hashlib.sha256(
        f"candidate-manifest:{candidate_id}".encode("utf-8")
    ).hexdigest()
    sealed_seed_count = 7
    score_stats = {
        "count": 1,
        "mean": public_score,
        "std": 0.0,
        "standard_error": 0.0,
        "eval_protocol_hash": protocol_hash,
    }
    required_gates = [
        "accept_evidence_gate",
        "budget_accounting",
        "sealed_validation_gate",
    ]
    if current_best_score is None:
        current_best = _bootstrap_current_best()
    else:
        previous_candidate_dir = candidate_dir.parent / str(
            current_best_candidate_id
        )
        if not previous_candidate_dir.is_dir():
            previous_candidate_dir = candidate_dir
        previous = _accepted_record(
            str(current_best_round_id),
            str(current_best_candidate_id),
            previous_candidate_dir,
            current_best_score,
        )
        current_best = CurrentBestEvalStats.from_public_result_mapping(
            round_id=str(current_best_round_id),
            candidate_id=str(current_best_candidate_id),
            public_result=previous["public_result"],
            acceptance_protocol=SOURCE_CATALOG_ACCEPTANCE_PROTOCOL,
            subject_manifest_hash=previous["jobs"]["eval"][
                "subject_manifest_hash"
            ],
            program_source_checksum=previous["jobs"]["eval"][
                "program_source_checksum"
            ],
            training_binding_hash=previous["jobs"]["eval"][
                "training_binding_hash"
            ],
            capability_profile_hash=previous["jobs"]["eval"][
                "capability_profile_hash"
            ],
            checkpoint_hash=previous["jobs"]["eval"]["checkpoint"][
                "checksum"
            ],
            expected_evaluation_binding=SOURCE_CATALOG_EVALUATION_BINDING,
        )
    return {
        "round_id": round_id,
        "session_genesis_hash": SOURCE_CATALOG_SESSION_GENESIS_HASH,
        "evaluation_binding": (
            SOURCE_CATALOG_EVALUATION_BINDING.to_public_dict()
        ),
        "decision": {
            "accepted": True,
            "outcome": "policy_improvement_accept",
            "reason": (
                "absolute_accept_floor"
                if current_best_score is None
                else "noise_aware_delta"
            ),
            "surface_classification": "code_level",
            "public_score": public_score,
            "comparison": {
                "mode": "bootstrap" if current_best_score is None else "accepted",
                "score_source": SOURCE_CATALOG_ACCEPTANCE_PROTOCOL.score_source,
                "acceptance_protocol_hash": SOURCE_CATALOG_ACCEPTANCE_PROTOCOL.protocol_hash,
                "candidate_score": public_score,
                "current_best_score": current_best_score or 0.0,
                "delta": public_score - (current_best_score or 0.0),
                "threshold": 0.0,
                "delta_met": public_score - (current_best_score or 0.0) > 0.0,
                "absolute_accept_floor": 0.0,
                "candidate_score_stats": dict(score_stats),
                "best_score_stats": (
                    {}
                    if current_best_score is None
                    else {
                        "count": 1,
                        "mean": current_best_score,
                        "std": 0.0,
                        "standard_error": 0.0,
                        "eval_protocol_hash": protocol_hash,
                    }
                ),
                "candidate_evaluation_binding_hash": (
                    SOURCE_CATALOG_EVALUATION_BINDING.binding_hash
                ),
                "current_best": current_best.comparison_identity(),
                "public_score": public_score,
            },
            "consumed_gates": list(required_gates),
            "gate_statuses": {gate: "pass" for gate in required_gates},
        },
        "candidate": {
            "candidate_id": candidate_id,
            "path": str(candidate_dir),
            "changed_components": ["reward_callable"],
        },
        "candidate_surface": {
            "classification": "code_level",
            "eligible_for_policy_accept": True,
            "reason": "code_level_candidate",
        },
        "verifier": {"status": "pass"},
        "jobs": {
            "train": {
                "job_id": f"{round_id}-train",
                "subject": {
                    "schema_version": "true_rl_agent.experiment.subject.v2",
                    "kind": "candidate",
                    "id": candidate_id,
                },
                "subject_manifest_hash": subject_manifest_hash,
                "program_source_checksum": source_checksum,
                "training_binding": {
                    "binding_hash": SOURCE_CATALOG_TRAINING_BINDING_HASH,
                    "capability_profile_hash": (
                        SOURCE_CATALOG_CAPABILITY_PROFILE_HASH
                    ),
                },
            },
            "eval": {
                "job_id": f"{round_id}-eval",
                "subject": {
                    "schema_version": "true_rl_agent.experiment.subject.v2",
                    "kind": "candidate",
                    "id": candidate_id,
                },
                "subject_manifest_hash": subject_manifest_hash,
                "program_source_checksum": source_checksum,
                "training_binding_hash": SOURCE_CATALOG_TRAINING_BINDING_HASH,
                "capability_profile_hash": (
                    SOURCE_CATALOG_CAPABILITY_PROFILE_HASH
                ),
                **SOURCE_CATALOG_EVALUATION_PROTOCOL.to_public_dict(),
                "evaluation_binding": (
                    SOURCE_CATALOG_EVALUATION_BINDING.to_public_dict()
                ),
                "checkpoint": {"checksum": checkpoint_hash},
            },
        },
        "train_result": {"job_id": f"{round_id}-train"},
        "public_result": {
            "status": "pass",
            "public_score": public_score,
            "metrics": {
                "evaluation_binding": (
                    SOURCE_CATALOG_EVALUATION_BINDING.to_public_dict()
                ),
                "evaluation_subject": {
                    "schema_version": "true_rl_agent.experiment.subject.v2",
                    "kind": "candidate",
                    "id": candidate_id,
                },
                "subject_manifest_hash": subject_manifest_hash,
                "program_source_checksum": source_checksum,
                "training_binding_hash": SOURCE_CATALOG_TRAINING_BINDING_HASH,
                "capability_profile_hash": (
                    SOURCE_CATALOG_CAPABILITY_PROFILE_HASH
                ),
                "public_accept": {"score_stats": dict(score_stats)},
                "acceptance": {
                    "score_source": SOURCE_CATALOG_ACCEPTANCE_PROTOCOL.score_source,
                    "public_score_binding": SOURCE_CATALOG_ACCEPTANCE_PROTOCOL.public_score_binding,
                    "score_stats": dict(score_stats),
                    "min_score_count": 1,
                    "absolute_accept_floor": SOURCE_CATALOG_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
                    "min_delta": SOURCE_CATALOG_ACCEPTANCE_PROTOCOL.min_delta,
                    "acceptance_protocol_hash": SOURCE_CATALOG_ACCEPTANCE_PROTOCOL.protocol_hash,
                    "required_pass_gates": list(required_gates),
                },
                "accept_evidence_gate": {"status": "pass"},
                "budget_accounting": {"status": "pass"},
                "sealed_validation": {
                    "schema_version": "true_rl_agent.sealed_validation.evidence.v3",
                    "status": "pass",
                    "reason": "test_holdout_passed",
                    "expected_seed_count": sealed_seed_count,
                    "observed_seed_count": sealed_seed_count,
                    "eval_protocol_hash": protocol_hash,
                    "evaluation_binding_hash": (
                        SOURCE_CATALOG_EVALUATION_BINDING.binding_hash
                    ),
                    "evaluation_subject": {
                        "schema_version": "true_rl_agent.experiment.subject.v2",
                        "kind": "candidate",
                        "id": candidate_id,
                    },
                    "subject_manifest_hash": subject_manifest_hash,
                    "program_source_checksum": source_checksum,
                    "training_binding_hash": (
                        SOURCE_CATALOG_TRAINING_BINDING_HASH
                    ),
                    "capability_profile_hash": (
                        SOURCE_CATALOG_CAPABILITY_PROFILE_HASH
                    ),
                    "checkpoint_hash": checkpoint_hash,
                    "score_stats": {
                        "count": sealed_seed_count,
                        "mean": public_score,
                        "std": 0.0,
                        "standard_error": 0.0,
                        "eval_protocol_hash": protocol_hash,
                    },
                },
                "sealed_validation_gate": {
                    "status": "pass",
                    "checkpoint_safety_status": "pass",
                    "candidate_isolation_status": "pass",
                    "candidate_code_loaded_in_eval": False,
                    "candidate_package_loaded_in_eval": False,
                    "candidate_path_argument_present": False,
                    "eval_protocol_hash_match_status": "pass",
                    "evaluation_binding_hash_match_status": "pass",
                    "checkpoint_hash_match_status": "pass",
                    "holdout_rollout_status": "pass",
                    "reason": "sealed_validation_complete",
                },
            },
        },
    }


def _evaluated_reject_record(
    round_id: str,
    candidate_id: str,
    candidate_dir: Path,
    public_score: float,
) -> dict:
    record = _accepted_record(
        round_id,
        candidate_id,
        candidate_dir,
        public_score,
    )
    required = record["public_result"]["metrics"]["acceptance"][
        "required_pass_gates"
    ]
    required.append("behavior_quality_gate")
    record["public_result"]["metrics"]["behavior_quality_gate"] = {
        "status": "fail"
    }
    record["decision"].update(
        accepted=False,
        outcome="reject",
        reason="required_gate_failed",
    )
    record["decision"]["consumed_gates"].append("behavior_quality_gate")
    record["decision"]["gate_statuses"]["behavior_quality_gate"] = "fail"
    return record


def _catalog_item(payload: dict, logical_id: str) -> dict:
    for item in payload["source_catalog"]["items"]:
        if item["logical_id"] == logical_id:
            return item
    raise AssertionError(f"missing source catalog item: {logical_id}")


if __name__ == "__main__":
    unittest.main()
