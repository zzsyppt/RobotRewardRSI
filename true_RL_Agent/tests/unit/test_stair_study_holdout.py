"""Study isolation uses the existing final plan, while training stays invariant."""

import json
import tempfile
import unittest
from pathlib import Path

from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    bind_stair_generation_final_config, create_stair_generation_final_confirmation_plan,
    load_stair_generation_final_confirmation_plan,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE, STAIR_FORMAL_EVAL_NAMESPACE

ROOT = Path(__file__).resolve().parents[2]


class StairStudyHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json")

    def make_study(self, name, replication=None):
        plan, path = create_stair_generation_final_confirmation_plan(
            output_dir=self.root / name, session_id=name, config=self.config,
            replication_source=replication,
        )
        return plan, path, bind_stair_generation_final_config(self.config, plan=plan, plan_path=path)

    def test_new_studies_change_final_geometry_and_seeds_without_changing_training_or_search(self):
        original = self.config.ordered_evaluation_suite.to_dict()
        plan_a, path_a, a = self.make_study("a")
        plan_b, _, b = self.make_study("b")
        self.assertNotEqual(plan_a.seed_cohort_hash, plan_b.seed_cohort_hash)
        for config in (a, b):
            self.assertEqual(config.formal_training_binding(), self.config.formal_training_binding())
            self.assertEqual(config.task_protocol_hash, self.config.task_protocol_hash)
            self.assertEqual(config.ordered_evaluation_suite.source_suite_hash, self.config.ordered_evaluation_suite.suite_hash)
            self.assertNotEqual(config.ordered_evaluation_suite.suite_hash, self.config.ordered_evaluation_suite.suite_hash)
            from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
            restored_suite = StairUnifiedEvaluationSuite.from_mapping(config.ordered_evaluation_suite.to_dict(),
                training_profile=config.unified_training_profile)
            self.assertEqual(restored_suite.cells, config.ordered_evaluation_suite.cells)
            self.assertEqual(restored_suite.to_dict(), config.ordered_evaluation_suite.to_dict())
            self.assertEqual(restored_suite.cohort_hash("sealed_validation"), config.ordered_evaluation_suite.cohort_hash("sealed_validation"))
            self.assertNotEqual(config.ordered_evaluation_suite.cell_set_hash, self.config.ordered_evaluation_suite.cell_set_hash)
            source = {cell.geometry_digest for cell in self.config.ordered_evaluation_suite.cells}
            self.assertFalse(source & {cell.geometry_digest for cell in config.ordered_evaluation_suite.cells_for_cohort("sealed_validation")})
            self.assertEqual(config.seed_resolver().derive_seed(split="public_dev", run_index=1, namespace=STAIR_FORMAL_EVAL_NAMESPACE),
                             self.config.seed_resolver().derive_seed(split="public_dev", run_index=1, namespace=STAIR_FORMAL_EVAL_NAMESPACE))
        self.assertNotEqual(a.seed_resolver().derive_seed(split="sealed_validation", run_index=1, namespace=STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE),
                            b.seed_resolver().derive_seed(split="sealed_validation", run_index=1, namespace=STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE))
        self.assertEqual(self.config.ordered_evaluation_suite.to_dict(), original)
        restored = load_stair_generation_final_confirmation_plan(path_a, expected_config=self.config)
        self.assertEqual(restored, plan_a)
        public = json.dumps(plan_a.to_dict())
        self.assertNotIn(a.final_holdout.nonce, public)
        self.assertFalse(plan_a.to_dict()["information_flow"]["feedback_allowed"])
        with self.assertRaises(FileExistsError):
            self.make_study("a")

    def test_replication_is_declared_and_commitment_tampering_cannot_redraw_final(self):
        plan, path, original = self.make_study("original")
        replica_plan, _, replica = self.make_study("replica", path.with_name("study_holdout.json"))
        self.assertEqual(replica_plan.replication_of, plan.study_holdout_commitment)
        self.assertNotEqual(replica_plan.plan_hash, plan.plan_hash)
        self.assertEqual(replica_plan.seed_cohort_hash, plan.seed_cohort_hash)
        self.assertEqual(replica.ordered_evaluation_suite.cells, original.ordered_evaluation_suite.cells)
        holdout_path = path.with_name("study_holdout.json")
        data = json.loads(holdout_path.read_text())
        data["nonce"] = "0" * 64
        from true_rl_agent.json_contract import canonical_json_bytes
        holdout_path.write_bytes(canonical_json_bytes(data, trailing_newline=True))
        with self.assertRaisesRegex(ValueError, "commitment mismatch"):
            load_stair_generation_final_confirmation_plan(path, expected_config=self.config)

    def test_creation_excludes_a_known_geometry_collision_before_commitment(self):
        from unittest.mock import patch
        from true_rl_agent.tasks.stair_climbing.study_holdout import StairStudyHoldout

        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_unified_protocol_v3.json")
        source = config.ordered_evaluation_suite
        # This nonce makes height_ood_level_3 equal calibration.height_ood_level_5.
        collision = "e468f961fdef3952173773b4007a143c5e37bfe00aeefec5a9d229dcb1babb00"
        invalid = StairStudyHoldout("invalid", source.suite_hash, collision)
        with self.assertRaisesRegex(ValueError, "overlaps a previously available course"):
            invalid.bind_suite(source, training_profile=config.unified_training_profile)
        path = self.root / "valid" / "study_holdout.json"
        with patch("true_rl_agent.tasks.stair_climbing.study_holdout.secrets.token_hex",
                   side_effect=[collision, "1" * 64]) as draw:
            valid = StairStudyHoldout.create(path=path, study_id="valid", source_suite=source,
                                            training_profile=config.unified_training_profile)
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(valid.nonce, "1" * 64)
        self.assertEqual(StairStudyHoldout.load(path), valid)
        realized = valid.bind_suite(source, training_profile=config.unified_training_profile)
        self.assertFalse({cell.geometry_digest for cell in source.cells} &
                         {cell.geometry_digest for cell in realized.cells_for_cohort("sealed_validation")})
        replica_path = self.root / "invalid-replica" / "study_holdout.json"
        with patch("true_rl_agent.tasks.stair_climbing.study_holdout.secrets.token_hex") as draw:
            with self.assertRaisesRegex(ValueError, "overlaps a previously available course"):
                StairStudyHoldout.create(path=replica_path, study_id="replica", source_suite=source,
                    training_profile=config.unified_training_profile, replication=invalid)
        draw.assert_not_called()
        self.assertFalse(replica_path.exists())

    def test_creation_redraws_internal_collisions_for_both_robots_but_not_replication(self):
        from unittest.mock import patch
        from true_rl_agent.tasks.stair_climbing.study_holdout import StairStudyHoldout

        collisions = {
            "stair": "9937857b534fddce97ba865bc6dceba758a7bf6e38f1a503637339695f46e533",
            "stair_go2_12dof": "548c07602e0daf4a507fa086b31b6ae06a738c9da4561e69f9d0f774131fbb4d",
        }
        for stem, nonce in collisions.items():
            with self.subTest(robot=stem):
                config = StairClimbingTaskConfig.defaults(ROOT / f"configs/tasks/{stem}_unified_protocol_v3.json")
                source = config.ordered_evaluation_suite
                invalid = StairStudyHoldout("invalid", source.suite_hash, nonce)
                with self.assertRaisesRegex(ValueError, "sealed cohort contains duplicate geometries"):
                    invalid.bind_suite(source, training_profile=config.unified_training_profile)
                path = self.root / stem / "study_holdout.json"
                kwargs = dict(path=path, study_id=stem, source_suite=source,
                              training_profile=config.unified_training_profile)
                with patch("true_rl_agent.tasks.stair_climbing.study_holdout.secrets.token_hex",
                           side_effect=[nonce, "1" * 64]) as draw:
                    result = StairStudyHoldout.create(**kwargs)
                self.assertEqual(draw.call_count, 2)
                self.assertEqual(StairStudyHoldout.load(path), result)
                sealed = result.bind_suite(source, training_profile=config.unified_training_profile).cells_for_cohort("sealed_validation")
                self.assertEqual(len(sealed), len({cell.geometry_digest for cell in sealed}))
                replica_path = self.root / stem / "replica.json"
                with patch("true_rl_agent.tasks.stair_climbing.study_holdout.secrets.token_hex") as draw:
                    with self.assertRaisesRegex(ValueError, "duplicate geometries"):
                        StairStudyHoldout.create(**{**kwargs, "path": replica_path}, replication=invalid)
                draw.assert_not_called()
                self.assertFalse(replica_path.exists())
                with patch("true_rl_agent.tasks.stair_climbing.study_holdout.secrets.token_hex", return_value=nonce) as draw:
                    with self.assertRaisesRegex(ValueError, "exhausted 64"):
                        StairStudyHoldout.create(**{**kwargs, "path": replica_path})
                self.assertEqual(draw.call_count, 64)
                self.assertFalse(replica_path.exists())

    def test_surface_ratio_uses_realized_minimum_riser_for_both_base_kinds(self):
        from true_rl_agent.tasks.stair_climbing.unified_evaluation import stair_geometry_variant, _resolve_geometry

        for stem in ("stair", "stair_go2_12dof"):
            config = StairClimbingTaskConfig.defaults(ROOT / f"configs/tasks/{stem}_unified_protocol_v3.json")
            payload = config.ordered_evaluation_suite.to_dict()
            surface = next(item["geometry"]["surface_profile"] for item in payload["cell_templates"]
                           if item["stratum_id"] == "c2_mechanism_ood")
            for kind in ("regular", "c1"):
                base = next(item["geometry"] for item in payload["cell_templates"] if item["geometry"]["kind"] == kind)
                def risers(geometry):
                    return _resolve_geometry(geometry, training_profile=config.unified_training_profile,
                                             step_count=8)[0]
                original = {"kind": "c2", "base_geometry": base, "surface_profile": surface}
                for shift in (-0.021, 0.023, 0.045):
                    with self.subTest(robot=stem, base=kind, shift=shift):
                        changed = stair_geometry_variant(original, stratum_id="c2_mechanism_ood",
                            training_profile=config.unified_training_profile,
                            support_length_m=payload["scale_context"]["support_length_m"],
                            height_fraction=shift, tread_fraction=0, surface_phase_shift=2)
                        self.assertAlmostEqual(changed["surface_profile"]["amplitude_m"] / min(risers(changed)),
                                               surface["amplitude_m"] / min(risers(original)))
                        self.assertNotEqual(risers(changed), risers(original))

    def test_frozen_legacy_final_keeps_absolute_amplitude(self):
        from true_rl_agent.tasks.stair_climbing.study_holdout import StairStudyHoldout, LEGACY_FINAL_HOLDOUT_POLICY
        from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite

        payload = self.config.ordered_evaluation_suite.to_dict()
        payload["final_holdout_policy"] = LEGACY_FINAL_HOLDOUT_POLICY
        source = StairUnifiedEvaluationSuite.freeze(payload, training_profile=self.config.unified_training_profile)
        holdout = StairStudyHoldout("legacy", source.suite_hash, "a" * 64)
        realized = holdout.bind_suite(source, training_profile=self.config.unified_training_profile)
        for cell in realized.cells_for_cohort("sealed_validation"):
            if cell.stratum_id == "c2_mechanism_ood":
                old = source.cell(cell.cell_id)
                self.assertEqual(cell.surface_profile["amplitude_m"], old.surface_profile["amplitude_m"])
                self.assertNotEqual(cell.riser_heights_m, old.riser_heights_m)

    def test_new_evaluation_reuses_the_immediate_checkpoint_and_original_training_domain(self):
        for stem, seed in (("stair", 1275486543), ("stair_go2_12dof", 1262835026)):
            old = StairClimbingTaskConfig.defaults(ROOT / f"configs/tasks/{stem}_unified_protocol_v2.json")
            new = StairClimbingTaskConfig.defaults(ROOT / f"configs/tasks/{stem}_unified_protocol_v3.json")
            self.assertEqual(old.formal_training_binding(), new.formal_training_binding())
            self.assertEqual(old.train_seed, new.train_seed)
            self.assertEqual(new.train_seed, seed)
            self.assertEqual(new.unified_protocol.training_lineage["source_cell_variant_id"], old.cell_variant_id)
            self.assertNotEqual(new.task_protocol_hash, old.task_protocol_hash)

    def test_evaluation_only_lineage_rejects_a_rehashed_training_change(self):
        from dataclasses import replace
        from unittest.mock import patch
        from true_rl_agent.json_contract import load_strict_json_object
        from true_rl_agent.tasks.stair_climbing.unified_protocol import StairUnifiedProtocol

        path = ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json"
        protocol = StairUnifiedProtocol.load(path)
        payload = protocol.to_dict()
        payload["training"]["command"]["forward_mps_range"][0] += 0.01
        altered = replace(protocol, _payload=payload)
        payload["task_protocol_hash"] = altered.task_protocol_hash
        payload["protocol_hash"] = replace(protocol, _payload=payload).protocol_hash
        def load(candidate, **kwargs):
            return payload if Path(candidate).resolve() == path else load_strict_json_object(candidate, **kwargs)
        with patch("true_rl_agent.tasks.stair_climbing.unified_protocol.load_strict_json_object", side_effect=load):
            with self.assertRaisesRegex(ValueError, "evaluation-only successor changed training"):
                StairUnifiedProtocol.load(path)


if __name__ == "__main__":
    unittest.main()
