"""R1a CPU contracts/selection only; all added v3 metadata here is a fixture."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from tests.unit import test_stair_trajectory_input_inspection as fixture
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.tasks.stair_climbing import trajectory_production as production
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import _load_contract
from true_rl_agent.tasks.stair_climbing.trajectory_capabilities import (
    CAPTURE_PROFILE, CONTRACT_VERSION, cell_bindings, geometry_sources, resolve_capabilities,
)
from true_rl_agent.tasks.stair_climbing.trajectory_pipeline import build_trial_trajectory_card
from true_rl_agent.tasks.stair_climbing.trajectory_report import build_stair_trajectory_report


def cpu_contract(projection, cells):
    """Exercise the interface, never call this a newly recorded runtime contract."""
    package = Path(production.__file__).parent
    return {"schema_version": CONTRACT_VERSION, "sample_phase": "pre_reset_control_step_end",
            "provenance": "CPU_fixture_metadata_not_new_rollout",
            "joint_position": {"source": "robot.data.joint_pos[:, resolved.joint_ids]", "unit": "rad",
                "coordinate_basis": "USD_joint_local_frames_no_added_zero_offset",
                "joint_names": list(projection.joint_names), "articulation_joint_names": list(projection.joint_names),
                "resolved_joint_ids": list(range(projection.joint_count)),
                "runtime_projection_hash": projection.projection_hash,
                "recorder_source_sha256": file_sha256(package / "isaaclab_runtime_env.py")},
            "nominal_terrain": {"source_sha256": geometry_sources(), "frame": "world_z_up_translation_only",
                "placement": "nominal_expected_layout_with_recorded_origins", "cells": cell_bindings(cells)}}


class ProductionTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StairTrajectoryInputInspectionTests.setUpClass()
        cls.fixture = fixture.StairTrajectoryInputInspectionTests()
        cls.fixture.setUp()
        cls.root = cls.fixture.root
        def mutate(identity, context, chunks):
            identity["evaluation_plan_hash"] = "e" * 64
            identity["evaluation_binding"]["evaluation_binding_hash"] = "f" * 64
            context.update(profile=CAPTURE_PROFILE, measurement_contract=cpu_contract(cls.fixture.projection, cls.fixture.cells))
        cls.reference = cls.fixture._archive(mutate)
        cls.identity = json.loads(Path(cls.reference["manifest_path"]).read_bytes())["identity"]
        cls.card, cls.receipt = build_trial_trajectory_card(cls.reference, expected_identity=cls.identity,
            output_dir=cls.root / "production", profile=production.PROFILE)
        cls.base = build_stair_trajectory_report(cls.card, numeric_reference=cls.reference, expected_identity=cls.identity)
        cls.summary = production.load_measurements(cls.receipt["measurements_reference"], verify_records=True)
        cls.full = production.project_production_report(cls.base, cls.summary, max_bytes=12288)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def test_new_hash_v3_contract_and_short_unequal_episodes(self):
        self.assertNotIn(self.reference["manifest_sha256"], __import__(
            "true_rl_agent.tasks.stair_climbing.trajectory_joint_limits", fromlist=["QUALIFIED_SOURCES"]).QUALIFIED_SOURCES)
        self.assertEqual(self.summary["capabilities"]["nominal_joint_limits"]["status"], "available")
        self.assertEqual(self.summary["capabilities"]["runtime_surface"], "unknown_not_recorded")
        for g in self.summary["cycle"]["groups"].values():
            if "counts" in g:
                self.assertEqual(g["counts"]["qualified_unique_cycles"], 0)
        group = next(iter(self.summary["joint"]["groups"].values()))
        self.assertEqual(group["counts"]["total_rows"], 3 * len(self.fixture.cells) - 3)

    def test_normal_cli_default_budget_versions_and_session_identity(self):
        import argparse
        from true_rl_agent.composition.base import TaskCompositionContext, resolve_task_composition
        from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
        provider = StairClimbingCompositionProvider()
        parser = argparse.ArgumentParser()
        provider.add_cli_arguments(parser)
        args = parser.parse_args(["--behavior-feedback-profile", production.PROFILE,
                                  "--cell-variant-id", "stair.go2_12dof.unified_v3"])
        values = {**provider.cli_defaults(), **vars(args), "start_mode": "reward_cold_start"}
        context = TaskCompositionContext(project_root=Path(__file__).resolve().parents[2], output_dir=self.root / "cli")
        current = resolve_task_composition(provider=provider, values=values, context=context)
        self.assertEqual(current.task_defaults["trajectory_report_max_bytes"], 10240)
        self.assertEqual(current.runner.eval_phase_summary_finalizer.keywords["behavior_feedback_profile"], production.PROFILE)
        older = resolve_task_composition(provider=provider, values={**values, "behavior_feedback_profile": "trajectory_numeric_v1"}, context=context)
        self.assertEqual(older.task_defaults["trajectory_report_max_bytes"], 8192)
        self.assertNotEqual(current.genesis_identity_snapshot(), older.genesis_identity_snapshot())

    def test_v3_recompute_rejects_missing_contract(self):
        from true_rl_agent.tasks.stair_climbing.numeric_capture import recompute_stair_numeric_evidence
        manifest = json.loads(Path(self.reference["manifest_path"]).read_bytes())
        manifest["metadata"].pop("measurement_contract")
        path = Path(self.reference["manifest_path"]).with_name("missing_contract.json")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract version"):
            recompute_stair_numeric_evidence({**self.reference, "manifest_path": str(path), "manifest_sha256": file_sha256(path)})

    def test_v2_new_hash_enables_cycle_but_not_nominal_guesses(self):
        reference = self.fixture._archive()
        capabilities, bounds = resolve_capabilities(_load_contract(reference), reference)
        self.assertEqual(capabilities["cycle"]["status"], "available")
        self.assertIsNone(bounds)
        self.assertEqual(capabilities["nominal_terrain"]["status"], "unavailable")

    def test_unsupported_nominal_asset_disables_only_that_capability(self):
        archive = _load_contract(self.reference)
        archive.manifest["metadata"]["resolved_cell_manifest"]["robot_profile"]["asset_closure"]["root_identity"] = "unqualified_asset"
        capabilities, bounds = resolve_capabilities(archive, self.reference)
        self.assertIsNone(bounds)
        self.assertEqual(capabilities["nominal_joint_limits"]["reason"], "asset_outside_qualified_nominal_model")
        self.assertEqual(capabilities["cycle"]["status"], "available")
        self.assertEqual(capabilities["nominal_terrain"]["status"], "available")

    def test_bad_coordinate_order_source_geometry_rejected(self):
        for key, value in (("unit", "deg"), ("resolved_joint_ids", list(reversed(range(12)))),
                           ("recorder_source_sha256", "0" * 64)):
            archive = _load_contract(self.reference)
            archive.manifest["metadata"]["measurement_contract"]["joint_position"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve_capabilities(archive, self.reference)
        for key, value in (("frame", "rotated"), ("source_sha256", {}), ("cells", [])):
            archive = _load_contract(self.reference)
            archive.manifest["metadata"]["measurement_contract"]["nominal_terrain"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve_capabilities(archive, self.reference)

    def test_unified_selection_preserves_base_slots_and_new_facts(self):
        audit = self.full["audit"]
        self.assertEqual(len(audit["base_semantic_mapping"]), 18)
        required = set(audit["required_claim_ids"])
        self.assertTrue(required <= set(audit["included_claim_ids"]))
        selected = [audit["claim_trace"][cid] for cid in required]
        self.assertEqual({r["measurement"] for r in selected if "measurement" in r}, {"cycle", "joint", "terrain"})
        text = json.dumps(self.full["public_report"], ensure_ascii=False)
        self.assertNotIn("周期资格、有效/名义限位绑定、逐类地形几何资格", text)
        self.assertIn("Effective runtime limits", text)
        production.replay_production_report(self.full)

    def test_budget_never_silently_drops_required_facts(self):
        minimum = self.full["audit"]["required_report_bytes"]
        with self.assertRaisesRegex(ValueError, "mandatory facts exceed"):
            production.project_production_report(self.base, self.summary, max_bytes=minimum - 1)
        exact = production.project_production_report(self.base, self.summary, max_bytes=minimum)
        self.assertEqual(exact["audit"]["required_claim_ids"], exact["audit"]["included_claim_ids"])

    def test_full_replay_detects_resigned_sentence_or_entity(self):
        from true_rl_agent.evaluation.binding import canonical_json_hash
        wrong = deepcopy(self.full)
        public = wrong["public_report"]
        public["report"]["process"][0]["text"] = "fabricated finite denominator"
        public["report_projection"].pop("content_hash")
        public["report_projection"]["content_hash"] = canonical_json_hash(public)
        with self.assertRaisesRegex(ValueError, "differs from operands"):
            production.replay_production_report(wrong)
        wrong = deepcopy(self.full["audit"]["base_operands"])
        wrong["domain"]["checkpoint_hash"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "source binding"):
            production.project_production_report(wrong, self.summary, max_bytes=12288)

    def test_repeated_build_reuses_measurements(self):
        with patch.object(production, "build_measurements", side_effect=AssertionError("unexpected recompute")):
            card, receipt = build_trial_trajectory_card(self.reference, expected_identity=self.identity,
                output_dir=self.root / "production", profile=production.PROFILE)
        self.assertEqual(card, self.card)
        self.assertTrue(receipt["measurements_reused"])

    def test_completed_cache_seals_measurement_manifest(self):
        path = Path(self.receipt["measurements_reference"]["manifest_path"])
        original = path.read_bytes()
        try:
            data = json.loads(original)
            data["changed_after_completion"] = True
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "completed artifact binding changed"):
                build_trial_trajectory_card(self.reference, expected_identity=self.identity,
                    output_dir=self.root / "production", profile=production.PROFILE)
        finally:
            path.write_bytes(original)

    def test_replay_rejects_resigned_measurement_metadata(self):
        original = Path(self.receipt["measurements_reference"]["manifest_path"])
        for attack in ("unit", "source", "entity", "episode", "artifacts", "joint_names", "phase_operand"):
            target = self.root / ("tampered_" + attack)
            shutil.copytree(original.parent, target)
            manifest = json.loads((target / "manifest.json").read_bytes())
            lines = (target / "measurements.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[0])
            record = row["cycle"]["records"][0]
            if attack == "unit":
                record["unit"] = "milliseconds; physical gait"
            elif attack == "source":
                record["evidence_ref"]["manifest_sha256"] = "f" * 64
            elif attack == "entity":
                record["entity_ids"] = ["other_foot"]
            elif attack == "episode":
                row["cycle"]["env_index"] = 1
            elif attack == "joint_names":
                row["joint"]["joint_names"].pop()
            elif attack == "phase_operand":
                row["cycle"]["records"][4]["value"]["reference_record_id"] = "foreign"
            else:
                manifest["artifacts"] = {}
            lines[0] = json.dumps(row)
            (target / "measurements.jsonl").write_text("\n".join(lines)+"\n", encoding="utf-8")
            if attack != "artifacts":
                manifest["artifacts"]["measurements.jsonl"] = {"sha256": file_sha256(target / "measurements.jsonl"),
                    "bytes": (target / "measurements.jsonl").stat().st_size}
            (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            reference = {"manifest_path": str(target / "manifest.json"), "manifest_sha256": file_sha256(target / "manifest.json")}
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                production.load_measurements(reference, verify_records=True)

    def test_public_validator_keeps_old_trust_boundaries_and_explicit_budget(self):
        from true_rl_agent.evaluation.binding import canonical_json_hash
        from true_rl_agent.tasks.stair_climbing.trajectory_report import validate_stair_trajectory_report
        public = self.full["public_report"]
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_stair_trajectory_report(public, max_bytes=1)
        for attack in ("card", "digest", "negative", "domain", "extra", "sealed"):
            bad = deepcopy(public)
            if attack == "card": bad["card_id"] = "card:bad"
            elif attack == "digest": bad["report_projection"]["required_claim_ids_hash"] = "bad"
            elif attack == "negative": bad["report_projection"]["omitted_claim_count"] = -1
            elif attack == "domain": bad["domain"]["evaluation_job_id"] = {}
            elif attack == "extra": bad["report_projection"]["extra"] = 1
            else: bad["report"]["gaps"][0]["text"] = "sealed_validation"
            bad["report_projection"].pop("content_hash")
            bad["report_projection"]["content_hash"] = canonical_json_hash(bad)
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                validate_stair_trajectory_report(bad)

    def test_latest_lane_missing_sibling_and_parent_source_are_distinct(self):
        from true_rl_agent.evaluation.binding import canonical_json_hash
        from true_rl_agent.tasks.stair_climbing.prompt_adapter import _latest_behavior_feedback
        def trial(name):
            public = deepcopy(self.full["public_report"])
            public["domain"]["evaluation_job_id"] = name + "-eval-public-dev"
            public["report_projection"].pop("content_hash")
            public["report_projection"]["content_hash"] = canonical_json_hash(public)
            return {"trial_id": name, "lane_id": name, "public_dev": {
                "checkpoint_hash": public["domain"]["checkpoint_hash"],
                "evaluation_plan_hash": public["domain"]["evaluation_plan_hash"], "public_summary": {"behavior": public}}}
        a, b = trial("a"), trial("b")
        generation = {"trials": [a, b], "parent_public_dev": {"behavior": a["public_dev"]["public_summary"]["behavior"]},
                      "parent_current_best": {"checkpoint_hash": "wrong_parent"}}
        selected = _latest_behavior_feedback([{"generation": generation}])
        self.assertEqual(selected["source_trial_id"], "b")
        self.assertEqual(selected["diff"]["reason"], "parent_checkpoint_mismatch")
        broken = deepcopy(generation)
        broken["trials"][0]["trial_id"] = "wrong_sibling"
        with self.assertRaisesRegex(ValueError, "sibling/trial/phase"):
            _latest_behavior_feedback([{"generation": broken}])
        b["public_dev"]["public_summary"].pop("behavior")
        self.assertIsNone(_latest_behavior_feedback([{"generation": generation}]))


if __name__ == "__main__":
    unittest.main()
