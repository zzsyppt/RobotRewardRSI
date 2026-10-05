"""R1a CPU contracts/selection only; all added v3 metadata here is a fixture."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import numpy as np

from tests.unit import stair_archive_fixture as fixture
from true_rl_agent.runner.numeric_evidence import file_sha256
from true_rl_agent.tasks.stair_climbing import trajectory_production as production
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import _load_contract
from true_rl_agent.tasks.stair_climbing.trajectory_capabilities import (
    CAPTURE_PROFILE, CONTRACT_VERSION, cell_bindings, geometry_sources, resolve_capabilities,
)
from true_rl_agent.tasks.stair_climbing.trajectory_pipeline import build_phase_description


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
        fixture.StairArchiveFixture.setUpClass()
        cls.fixture = fixture.StairArchiveFixture()
        cls.fixture.setUp()
        cls.root = cls.fixture.root
        def mutate(identity, context, chunks):
            identity["evaluation_plan_hash"] = "e" * 64
            identity["evaluation_binding"]["evaluation_binding_hash"] = "f" * 64
            context.update(profile=CAPTURE_PROFILE, measurement_contract=cpu_contract(cls.fixture.projection, cls.fixture.cells))
        cls.reference = cls.fixture._archive(mutate)
        cls.identity = json.loads(Path(cls.reference["manifest_path"]).read_bytes())["identity"]
        cls.domain, cls.summary, cls.comparison = build_phase_description(cls.reference, expected_identity=cls.identity)
        cls.full = production.project_production_report(cls.domain, cls.summary)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def test_public_projection_retains_joint_processes_and_short_counterexamples(self):
        """Synthetic operands exercise the real budgeted projection, not raw-file validation."""
        from tests.unit.trajectory_fixtures import sample_episode
        from true_rl_agent.evaluation.trajectory_translation.measurement import measure_behavior
        from true_rl_agent.evaluation.trajectory_translation.cohort import aggregate_behavior
        episodes = []
        for index, (outcome, speed) in enumerate((("forbidden_contact",-.4),("forbidden_contact",.6),
                ("forbidden_contact",.1),("timeout",0),("timeout",.7),("success",0),("timeout",0))):
            ep = sample_episode(80)
            ep.env_index = index
            ep.task.update(stratum_id="fixture",cell_id="fixture")
            ep.task["outcome"].update(primary_reason=outcome,success=outcome=="success")
            ep.arrays["base_pos_w"][:,0] = ep.times*speed
            episodes.append(ep)
        episodes[0].arrays["base_quat_w"][:] = [0,1,0,0]
        episodes[0].arrays["support_contact_forces_w"][:] = 0
        episodes[2].arrays["forbidden_contact_forces_w_history"][-1,1,0,2] = 3
        episodes[2].arrays["termination_0"][-1] = True
        episodes[2].arrays["termination_1"][-1] = True
        episodes[5].arrays["base_pos_w"][:,0] = np.r_[np.linspace(0,.4,20),np.linspace(.4,.1,20),np.linspace(.1,.6,40)]
        unknown = episodes[-1]
        unknown.arrays["base_quat_w"][40:] = np.nan
        mask = unknown.steps != 35
        unknown.steps = unknown.steps[mask]
        unknown.arrays = {k:v[mask] for k,v in unknown.arrays.items()}
        unknown.chunk_indices, unknown.chunk_rows = unknown.chunk_indices[mask], unknown.chunk_rows[mask]
        down_missing_speed = deepcopy(episodes[0])
        down_missing_speed.env_index = len(episodes)
        down_missing_speed.arrays["base_pos_w"][-5:,0] = np.nan
        episodes.append(down_missing_speed)
        summary = {**self.summary, "joint":None,"terrain":None,"capabilities":{},
            "behavior":aggregate_behavior([measure_behavior(ep) for ep in episodes],motion_bands=production.PROCESS_BANDS),
            "cycle":{"groups":{"contact_cycles:"+foot:{"episodes":len(episodes),"episodes_with_qualified_windows":0,"episode_median_qualified_cycle_period_s":{"median":None}} for foot in episodes[0].entities["support"]}}}
        full = production.project_production_report(self.domain,summary,max_bytes=10240)
        public = production.validate_production_report(full["public_report"])
        self.assertEqual(production.replay_production_report(json.loads(json.dumps(full,sort_keys=True))),public)
        text = "\n".join(row["text"] for section in ("process","counterexample") for row in public["report"][section])
        for expected in ("已见上轴向下2回合（结束1.6s，末速-0.4m/s，末速有效1/2回合）", "末前速>0.2m/s1回合（结束1.6s，末速0.6m/s）",
                         "其余已知1回合", "|前向速度|<0.01m/s的", "速度中位0.7m/s",
                         "1回合因缺测不能判全程阴性", "history ANY仅终点1拍", "回撤0.3m", "首次观察到上轴向下"):
            self.assertIn(expected,text)
        self.assertNotIn("全程曾上轴向下0回合",text)
        from true_rl_agent.evaluation.process_report import report_bytes
        shown = production.author_behavior_payload({"source":{},"profile":public})
        self.assertLessEqual(report_bytes(shown),10240)

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
        for key, value in (("unit", "deg"), ("resolved_joint_ids", list(reversed(range(12))))):
            archive = _load_contract(self.reference)
            archive.manifest["metadata"]["measurement_contract"]["joint_position"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve_capabilities(archive, self.reference)
        for key, value in (("frame", "rotated"), ("cells", [])):
            archive = _load_contract(self.reference)
            archive.manifest["metadata"]["measurement_contract"]["nominal_terrain"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve_capabilities(archive, self.reference)

    def test_selection_covers_motion_all_feet_and_available_facts(self):
        audit = self.full["audit"]
        self.assertNotIn("base_semantic_mapping",audit)
        public = self.full["public_report"]
        required = set(public["report_projection"]["required_claim_ids"])
        shown = production.author_behavior_payload({"source":{},"profile":public})
        self.assertTrue(required <= {r['id'] for v in shown['profile']['report'].values() if isinstance(v,list) for r in v})
        self.assertTrue({"claim:motion", "claim:foot.0", "claim:foot.1", "claim:foot.2", "claim:foot.3", "claim:terrain"} <= required)
        text = json.dumps(self.full["public_report"], ensure_ascii=False)
        self.assertNotIn("周期资格、有效/名义限位绑定、逐类地形几何资格", text)
        self.assertIn("非运行时限位", text)
        production.replay_production_report(self.full)

    def test_budget_never_silently_drops_required_facts(self):
        from true_rl_agent.evaluation.process_report import ReportBudgetExceeded, report_bytes
        feedback = {"source":{},"profile":self.full["public_report"]}
        with self.assertRaises(ReportBudgetExceeded) as caught:
            production.author_behavior_payload(feedback,max_bytes=1)
        minimum = caught.exception.required_bytes
        with self.assertRaisesRegex(ValueError, "mandatory facts exceed"):
            production.author_behavior_payload(feedback,max_bytes=minimum-1)
        exact = production.author_behavior_payload(feedback,max_bytes=minimum)
        self.assertEqual(report_bytes(exact),minimum)

    def test_full_replay_detects_resigned_sentence_or_entity(self):
        from true_rl_agent.evaluation.binding import canonical_json_hash
        wrong = deepcopy(self.full)
        public = wrong["public_report"]
        public["report"]["process"][0]["text"] = "fabricated finite denominator"
        public["report_projection"].pop("content_hash")
        public["report_projection"]["content_hash"] = canonical_json_hash(public)
        with self.assertRaisesRegex(ValueError, "differs from summary"):
            production.replay_production_report(wrong)
        with self.assertRaisesRegex(ValueError, "source identity"):
            build_phase_description(self.reference, expected_identity={**self.identity,"checkpoint_hash":"wrong"})


if __name__ == "__main__":
    unittest.main()
