"""Automatic profile boundaries, atomic artifacts and per-archive cache behavior."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.unit import test_stair_trajectory_input_inspection as fixture
from true_rl_agent.composition.base import TaskCompositionContext, resolve_task_composition
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing import trajectory_artifacts as artifacts
from true_rl_agent.tasks.stair_climbing.trajectory_pipeline import (
    TRAJECTORY_PROFILE, build_trial_trajectory_card, validate_feedback_profile,
    validate_build_receipt,
)
from true_rl_agent.tasks.stair_climbing.trajectory_report import build_stair_trajectory_report, project_trajectory_card

ROOT = Path(__file__).resolve().parents[2]


class AutomaticTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StairTrajectoryInputInspectionTests.setUpClass()
        cls.fixture = fixture.StairTrajectoryInputInspectionTests()
        cls.fixture.setUp()
        cls.root = cls.fixture.root

        def bind(identity, context, chunks):
            identity["evaluation_plan_hash"] = "e" * 64
            identity["evaluation_binding"]["evaluation_binding_hash"] = "f" * 64
        cls.reference = cls.fixture._archive(bind)
        cls.identity = json.loads(Path(cls.reference["manifest_path"]).read_text())["identity"]
        cls.card_ref, cls.receipt = build_trial_trajectory_card(
            cls.reference, expected_identity=cls.identity, output_dir=cls.root / "phase")
        cls.full = build_stair_trajectory_report(cls.card_ref, numeric_reference=cls.reference,
                                               expected_identity=cls.identity)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def test_normal_composition_wires_profile_and_preserves_old_defaults(self):
        provider = StairClimbingCompositionProvider()
        values = dict(provider.cli_defaults())
        self.assertNotIn("behavior_feedback_profile", values)
        values.update(start_mode="reward_cold_start", cell_variant_id="stair.go2_12dof.unified_v3")
        old = resolve_task_composition(provider=provider, values=values,
            context=TaskCompositionContext(project_root=ROOT, output_dir=self.root / "composition"))
        values.update(behavior_feedback_profile=TRAJECTORY_PROFILE, trajectory_report_max_bytes=8192)
        new = resolve_task_composition(provider=provider, values=values,
            context=TaskCompositionContext(project_root=ROOT, output_dir=self.root / "composition"))
        self.assertEqual(new.runner.eval_phase_summary_finalizer.keywords["behavior_feedback_profile"], TRAJECTORY_PROFILE)
        self.assertEqual(new.task_defaults["behavior_feedback_profile"], TRAJECTORY_PROFILE)
        self.assertNotIn("behavior_feedback_profile", old.task_defaults)
        self.assertEqual(new.task_plugin.config.formal_training_binding(), old.task_plugin.config.formal_training_binding())
        self.assertEqual(new.task_plugin.config.evaluation_protocol(), old.task_plugin.config.evaluation_protocol())
        self.assertEqual(new.task_plugin.config.acceptance_protocol, old.task_plugin.config.acceptance_protocol)
        self.assertNotEqual(new.genesis_identity_snapshot(), old.genesis_identity_snapshot())
        values["trajectory_report_max_bytes"] = 8000
        smaller = resolve_task_composition(provider=provider, values=values,
            context=TaskCompositionContext(project_root=ROOT, output_dir=self.root / "composition"))
        self.assertNotEqual(new.genesis_identity_snapshot(), smaller.genesis_identity_snapshot())

    def test_versions_robots_and_budget_reject_explicitly(self):
        for profile, budget, cell in [("unknown", 8192, "stair.go2_12dof.unified_v3"),
                                     (TRAJECTORY_PROFILE, 8192, "stair.g1_37dof.unified_v3"),
                                     (TRAJECTORY_PROFILE, 8193, "stair.go2_12dof.unified_v3"),
                                     ("legacy_process_v1", 8000, "stair.go2_12dof.unified_v3")]:
            with self.subTest(profile=profile, budget=budget, cell=cell), self.assertRaises(ValueError):
                validate_feedback_profile(profile, budget, cell)

    def test_automatic_selection_has_no_frozen_examples_or_other_source(self):
        manifest = json.loads(Path(self.card_ref["manifest_path"]).read_text())
        card = json.loads(Path(self.card_ref["manifest_path"]).with_name("policy_card.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["example_source_bindings"], [])
        self.assertEqual(manifest["selection_ref"], {"mode": "current_archive_deterministic_examples.v1", "frozen_examples": False})
        self.assertNotIn("冻结E1样本", " ".join(c["text"] for c in card["claims"]))
        self.assertEqual(len(self.full["audit"]["required_claim_ids"]), 18)
        self.assertTrue(any("该槽不可提供事实" in c["text"] for c in card["claims"]))
        self.assertEqual(self.full["public_report"]["domain"]["archive_hash"], self.reference["manifest_sha256"])

    def test_repeated_build_and_budget_change_reuse_artifacts(self):
        with patch.object(artifacts, "measure_episode", side_effect=AssertionError("must not rebuild numeric")):
            ref, receipt = build_trial_trajectory_card(self.reference, expected_identity=self.identity, output_dir=self.root / "phase")
        self.assertEqual(ref, self.card_ref)
        self.assertTrue(receipt["numeric_reused"] and receipt["card_reused"])
        full = build_stair_trajectory_report(ref, numeric_reference=self.reference, expected_identity=self.identity,
                                             max_bytes=self.full["audit"]["required_report_bytes"])
        self.assertEqual(full["audit"]["included_claim_ids"], full["audit"]["required_claim_ids"])
        with self.assertRaisesRegex(ValueError, "mandatory facts exceed"):
            build_stair_trajectory_report(ref, numeric_reference=self.reference, expected_identity=self.identity,
                                          max_bytes=self.full["audit"]["required_report_bytes"] - 1)

    def test_foreign_job_checkpoint_plan_never_reuses_card(self):
        for key in ("job_id", "checkpoint_hash", "evaluation_plan_hash"):
            identity = {**self.identity, key: "wrong"}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity mismatch"):
                build_trial_trajectory_card(self.reference, expected_identity=identity, output_dir=self.root / "phase")

    def test_missing_manifest_does_not_fallback(self):
        bad = {**self.reference, "manifest_path": str(self.root / "absent.json")}
        with self.assertRaises(Exception):
            build_trial_trajectory_card(bad, expected_identity=self.identity, output_dir=self.root / "phase")

    def test_private_receipt_replay_binds_configuration_source_and_public_bytes(self):
        public = self.full["public_report"]
        receipt = {**self.receipt, "projection": {
            "profile": TRAJECTORY_PROFILE, "max_report_bytes": 8192,
            "card_id": public["card_id"], "public_content_hash": public["report_projection"]["content_hash"],
            "required_claim_ids": self.full["audit"]["required_claim_ids"],
            "report_bytes": self.full["audit"]["report_bytes"]}}
        validate_build_receipt(receipt, public)
        for path, value in [(('complete',), False), (('cache_key',), 'bad'),
                (('inputs', 'archive_sha256'), 'bad'), (('inputs', 'profile'), 'legacy_process_v1'),
                (('projection', 'card_id'), 'bad'), (('projection', 'public_content_hash'), 'bad'),
                (('projection', 'max_report_bytes'), 1), (('projection', 'report_bytes'), 1),
                (('projection', 'required_claim_ids'), self.full['audit']['required_claim_ids'][:-1])]:
            bad = deepcopy(receipt)
            target = bad
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_build_receipt(bad, public)
        with self.assertRaises(ValueError):
            validate_build_receipt(receipt, {**public, 'schema_version': 'legacy'})

    def test_removed_unavailable_slot_rejected_even_with_consistent_projection_hashes(self):
        audit = self.full["audit"]
        claims = [deepcopy(audit["claim_trace"][cid]["claim"]) for cid in audit["claim_order"]]
        absent = next(c for c in claims if c["template_id"] == "definition"
                      and c["operands"]["text"]["path"][-1].startswith("unavailable_"))
        claims.remove(absent)
        projected = project_trajectory_card({"claims": claims}, json.loads(audit["derivation_records_json"]),
            domain=self.full["public_report"]["domain"], card_hash=self.full["public_report"]["card_id"][5:])
        self.assertEqual(len(projected["audit"]["required_claim_ids"]), 17)
        public = projected["public_report"]
        receipt = {**self.receipt, "projection": {
            "profile": TRAJECTORY_PROFILE, "max_report_bytes": 8192,
            "card_id": public["card_id"], "public_content_hash": public["report_projection"]["content_hash"],
            "required_claim_ids": projected["audit"]["required_claim_ids"],
            "report_bytes": projected["audit"]["report_bytes"]}}
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            validate_build_receipt(receipt, public)

    def test_changed_profile_and_nonpublic_archive_rejected(self):
        for field, value in (("profile", "stair.unsupported.v3"), ("splits", ["sealed_validation"])):
            m = json.loads(Path(self.reference["manifest_path"]).read_text())
            m["metadata" if field == "profile" else "identity"][field] = value
            path = self.root / (field + ".json")
            path.write_text(json.dumps(m))
            bad = {**self.reference, "manifest_path": str(path), "manifest_sha256": file_sha256(path)}
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_trial_trajectory_card(bad, expected_identity=self.identity, output_dir=self.root / "phase")

    def test_manifest_published_only_after_complete_write(self):
        path = self.root / "interrupted_manifest.json"
        with patch.object(artifacts.json, "dump", side_effect=OSError("fixture disk interruption")):
            with self.assertRaises(OSError):
                artifacts._write_json(path, {"complete": True})
        self.assertFalse(path.exists())
        self.assertTrue(path.with_name(path.name + ".partial").exists())

    def test_incomplete_owned_stage_restarts(self):
        from true_rl_agent.tasks.stair_climbing.trajectory_pipeline import _restart_incomplete
        root = self.root / "incomplete"
        numeric = root / "numeric"
        numeric.mkdir(parents=True)
        (numeric / "evidence.jsonl.partial").write_text("unfinished")
        _restart_incomplete(numeric, root)
        self.assertFalse(numeric.exists())
        numeric.mkdir()
        (numeric / "manifest.json").write_text('{"complete":true}')
        _restart_incomplete(numeric, root)
        self.assertTrue(numeric.exists())


if __name__ == "__main__":
    unittest.main()
