from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import hashlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from tests.unit.test_stair_numeric_capture import _TwoStepEnv
from true_rl_agent.memory.context_builder import PublicContextBuilder
from true_rl_agent.tasks.stair_climbing.behavior import (
    _Episode, _representatives, build_stair_behavior, project_stair_behavior, compare_stair_behavior, MAX_PUBLIC_BYTES,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.evaluation_runtime import run_stair_split_execution
from true_rl_agent.tasks.stair_climbing.numeric_capture import StairNumericCapture, STAIR_NUMERIC_PROFILE_V1
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FORMAL_EVAL_NAMESPACE
from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_stair_summary, build_reference_objective_family_mask
from true_rl_agent.tasks.stair_climbing.prompt_adapter import StairClimbingPromptAdapter, _minimum_core_stair_prompt_sections
from true_rl_agent.runner.numeric_evidence import NumericEvidenceWriter, iter_numeric_evidence, load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.legacy_transition_reader import STAIR_CONTACT_HISTORY_TENSOR_FIELDS

ROOT = Path(__file__).resolve().parents[2]


class StairBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json")
        cls.manifest = resolve_stair_cell_manifest(cell_variant_id=cls.config.cell_variant_id,
            config=cls.config, plugin=StairClimbingTaskPlugin(cls.config))
        cls.config = replace(cls.config, resolved_cell_task_protocol_hash=cls.manifest.resolved_cell_task_protocol_hash)
        cls.projection = stair_runtime_projection_from_manifest(cls.manifest)
        cls.identity = {"checkpoint_hash": "a"*64, "splits": ["public_dev"], "seed_namespace": STAIR_FORMAL_EVAL_NAMESPACE,
            "environment_source_preflight": {"source_manifest_hash": "b"*64},
            "backend_execution_environment_verification": {"actual_manifest_hash": "c"*64}}
        capture = StairNumericCapture(Path(cls.temp.name) / "numeric", identity=cls.identity,
            config=cls.config, resolved_cell_manifest=cls.manifest)
        execution = run_stair_split_execution(
            env=_TwoStepEnv(cls.config.ordered_evaluation_suite.private_cell_assignment("public_dev"), cls.projection, cls.config),
            inference_policy=lambda x: x, protocol=None, semantics=cls.config.evaluation_semantics,
            seed_resolver=cls.config.seed_resolver(), split="public_dev", seed_count=cls.config.public_seed_count,
            seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE, runtime_projection=cls.projection, numeric_capture=capture)
        cls.execution = execution
        cls.summary = build_unified_stair_summary(suite=cls.config.ordered_evaluation_suite,
            semantics=cls.config.evaluation_semantics, split="public_dev", per_seed_outcomes=execution.per_seed_outcomes)
        cls.reference = capture.complete(cls.summary)
        cls.full = build_stair_behavior(cls.reference, expected_identity=cls.identity)
        cls.public = project_stair_behavior(cls.full)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_archive_derivation_missing_anchor_history_and_bounded_public(self):
        self.assertEqual(len(self.full["episodes"]), 864)
        self.assertEqual(self.public["profile"]["episode_count"], 864)
        self.assertTrue(all(e["anchor"]["status"] == "not_reached" for e in self.full["episodes"]))
        self.assertEqual(self.public["contrasts"]["matched_group_count"], 0)
        self.assertGreater(self.public["contrasts"]["missing_success_group_count"], 0)
        self.assertEqual(self.public["domain"]["history_status"], "recorded")
        self.assertEqual(self.public["domain"]["reward_input_schema_hash"], self.projection.reward_input_schema_hash_vnext)
        self.assertTrue(any(e["code"] == "contact_history" for e in self.full["episodes"][0]["events"]))
        self.assertFalse(any(e["code"] == "contact_instant" for e in self.full["episodes"][0]["events"]))
        self.assertLessEqual(len(json.dumps(self.public, separators=(",", ":")).encode()), MAX_PUBLIC_BYTES)
        self.assertEqual(project_stair_behavior(self.public), self.public)
        ids = [e["episode_id"] for e in self.public["transcripts"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn(self.temp.name, json.dumps(self.public))
        with self.assertRaisesRegex(ValueError, "identity"):
            build_stair_behavior(self.reference, expected_identity={"checkpoint_hash": "d"*64})

    def test_old_archive_does_not_invent_history_or_block_common_field_comparison(self):
        old = load_numeric_evidence_manifest(self.reference)
        writer = NumericEvidenceWriter(Path(self.temp.name) / "legacy", old["identity"])
        for header, arrays in iter_numeric_evidence(self.reference):
            transition = dict(header["transition_header"])
            transition.pop("contact_force_history")
            writer.write_chunk({k: v for k, v in arrays.items() if k not in STAIR_CONTACT_HISTORY_TENSOR_FIELDS},
                               {**header, "transition_header": transition})
        metadata = {**old["metadata"], "profile": STAIR_NUMERIC_PROFILE_V1}
        metadata.pop("contact_force_history")
        legacy = build_stair_behavior(writer.complete(metadata))
        self.assertEqual(legacy["domain"]["history_status"], "not_recorded")
        self.assertFalse(any(e["code"] == "contact_history" for e in legacy["episodes"][0]["events"]))
        self.assertEqual(project_stair_behavior(legacy)["profile"], self.public["profile"])
        self.assertEqual(compare_stair_behavior(self.public, project_stair_behavior(legacy),
            expected_parent_checkpoint_hash="a"*64)["status"], "available")

    def test_diff_checks_parent_source_cohort_and_unavailable(self):
        candidate = deepcopy(self.public)
        candidate["domain"]["checkpoint_hash"] = "d"*64
        diff = compare_stair_behavior(self.public, candidate, expected_parent_checkpoint_hash="a"*64)
        self.assertEqual(diff["status"], "available")
        self.assertTrue(all(all(value == 0 for value in row["rate_delta"]) for row in diff["rows"]))
        self.assertEqual(compare_stair_behavior(self.public, candidate, expected_parent_checkpoint_hash="e"*64)["reason"],
                         "parent_checkpoint_mismatch")
        for field in ("source_hash", "cohort_hash", "episode_keys_hash", "reward_input_schema_hash"):
            mutated = deepcopy(candidate)
            mutated["domain"][field] = "f"*64
            self.assertEqual(compare_stair_behavior(self.public, mutated, expected_parent_checkpoint_hash="a"*64)["status"], "not_comparable")
        missing = deepcopy(self.public)
        missing["domain"]["source_hash"] = None
        self.assertEqual(compare_stair_behavior(missing, missing, expected_parent_checkpoint_hash="a"*64)["reason"], "execution_source_unavailable")
        self.assertEqual(compare_stair_behavior(None, candidate, expected_parent_checkpoint_hash=None)["status"], "not_available")

    def test_representatives_do_not_substitute_unmatched_success_and_keep_denominators(self):
        episodes = deepcopy(self.full["episodes"][:2])
        episodes[1].update(primary_reason="success", cell="different-cell")
        shown = _representatives(episodes)
        self.assertEqual([e["relationship"] for e in shown], ["index_example"])
        full = deepcopy(self.full)
        reasons = self.config.evaluation_semantics.outcomes["primary_reasons"]
        for index, episode in enumerate(full["episodes"]):
            episode["primary_reason"] = reasons[index % len(reasons)]
        projected = project_stair_behavior(full)
        self.assertEqual(sum(row[2] for row in projected["profile"]["rows"])
                         + projected["profile"]["omitted_group_episodes"], len(full["episodes"]))
        totals = projected["profile"]["stratum_totals"]
        self.assertEqual(sum(row[totals["columns"].index("n")] for row in totals["rows"]), len(full["episodes"]))
        self.assertGreater(projected["profile"]["omitted_group_rows"], 0)
        compared = compare_stair_behavior(projected, projected, expected_parent_checkpoint_hash="a"*64)
        self.assertEqual(sum(row["parent"][0] for row in compared["rows"]), len(full["episodes"]))
        self.assertTrue(all(value == 0 for row in compared["rows"] for value in row["rate_delta"]))
        self.assertGreater(projected["omitted"]["episodes"], 0)
        self.assertEqual(len(projected["transcripts"]), 2)
        self.assertNotEqual(projected["transcripts"][0]["primary_reason"], "success")
        self.assertEqual(projected["transcripts"][1]["primary_reason"], "success")
        self.assertTrue(all(row["events"] for row in projected["transcripts"]))
        self.assertLessEqual(len(json.dumps(projected, separators=(",", ":")).encode()), MAX_PUBLIC_BYTES)

    def test_g1_binding_and_actionability_are_derived_from_its_actual_abi(self):
        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_unified_protocol_v3.json")
        manifest = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id, config=config, plugin=StairClimbingTaskPlugin(config))
        config = replace(config, resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash)
        projection = stair_runtime_projection_from_manifest(manifest)
        with tempfile.TemporaryDirectory() as directory:
            capture = StairNumericCapture(Path(directory) / "g1", identity=self.identity, config=config, resolved_cell_manifest=manifest)
            execution = run_stair_split_execution(env=_TwoStepEnv(config.ordered_evaluation_suite.private_cell_assignment("public_dev"), projection, config),
                inference_policy=lambda x: x, protocol=None, semantics=config.evaluation_semantics,
                seed_resolver=config.seed_resolver(), split="public_dev", seed_count=config.public_seed_count,
                seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE, runtime_projection=projection, numeric_capture=capture)
            summary = build_unified_stair_summary(suite=config.ordered_evaluation_suite, semantics=config.evaluation_semantics,
                split="public_dev", per_seed_outcomes=execution.per_seed_outcomes)
            full = build_stair_behavior(capture.complete(summary))
            public = project_stair_behavior(full)
        self.assertEqual(public["semantics"]["support_bodies"], ["left_ankle_roll_link", "right_ankle_roll_link"])
        self.assertEqual(public["semantics"]["forbidden_bodies"], ["torso_link"])
        self.assertEqual(public["semantics"]["kinematic_body"], "articulation_root")
        self.assertEqual(full["semantics"]["actionability"]["contact_edge"], ["not_in_reward_ABI"])
        self.assertNotEqual(public["domain"]["reward_input_schema_hash"], self.public["domain"]["reward_input_schema_hash"])

    def test_actionability_separates_instantaneous_inputs_from_diagnostic_history(self):
        actionability = self.full["semantics"]["actionability"]
        for key in ("tilt", "slip_body", "slip_any", "unloaded_motion"):
            self.assertEqual(actionability[key][0], "instant_only")
        self.assertIn("global/body/vertical streaks", actionability["streaks"][0])
        self.assertIn("absent pose/velocity history", actionability["streaks"][0])
        self.assertIn("force buffer insufficient", actionability["streaks"][0])
        self.assertEqual(actionability["forbidden_manager"], ["available", "forbidden_contact"])
        self.assertEqual(actionability["forbidden_history"],
                         ["buffer_not_in_ABI; instant_proxy_only", "forbidden_contact_force"])
        self.assertEqual(actionability["contact_history"],
                         ["available", "support_contact", "support_contact_force_history"])
        abi_fields = {field.name for field in self.projection.reward_input_fields_vnext}
        for key in ("tilt", "slip_body", "unloaded_motion", "contact_history", "forbidden_manager"):
            self.assertTrue(set(actionability[key][1:]) <= abi_fields)
        self.assertIn("forbidden_contact_force", abi_fields)
        self.assertNotIn("forbidden_contact_force_history", abi_fields)
        self.assertNotIn("support_lin_vel_history", abi_fields)
        self.assertNotIn("roll_history", abi_fields)
        self.assertIn("no new state/input", actionability["authority"][0])
        self.assertEqual(self.public["domain"]["reward_input_schema_hash"], self.projection.reward_input_schema_hash_vnext)
        public_actions = self.public["semantics"]["actionability"]
        self.assertEqual(public_actions["loaded_point_motion_any"], ["instant_only", "support_contact_force", "support_lin_vel"])
        self.assertEqual(public_actions["unloaded_low_motion"][1:], ["support_contact_force", "support_lin_vel"])
        self.assertNotIn("contact_history", public_actions)  # unselected event catalogue stays in full

    def _root_motion_sequence(self, *, speeds, contacts=None, rolls=None, progress=None, history=False):
        cell = self.config.ordered_evaluation_suite.private_cell_assignment("public_dev")[0]
        state = _Episode(cell, 1, 0, self.config.evaluation_semantics.outcomes["signals"], ("left", "right"), history)
        for i, speed in enumerate(speeds):
            step = i+1
            roll = rolls[i] if rolls else 0.
            contact = contacts[i] if contacts else (False, False)
            t = SimpleNamespace(episode_step=step, progress_m=progress[i] if progress else 0.,
                support_contacts=contact, support_planar_speeds_mps=(speed, speed),
                support_surface_clearances_m=(.1, .1), forbidden_contact=False)
            raw = {"base_quat_w": torch.tensor([[math.cos(roll/2), math.sin(roll/2), 0., 0.]], dtype=torch.float64),
                   "support_lin_vel_w": torch.tensor([[[speed, 0., 0.], [speed, 0., 0.]]], dtype=torch.float64)}
            if history:
                raw["support_contact_forces_w_history"] = torch.ones((1, 3, 2, 3), dtype=torch.float64)*10
                raw["forbidden_contact_forces_w_history"] = torch.zeros((1, 3, 1, 3), dtype=torch.float64)
            result = state.observe(t, raw, SimpleNamespace(primary_reason="timeout") if step==len(speeds) else None)
        return result

    def test_direction_extrema_and_retreat_keep_terminal_time_and_signed_orientation(self):
        row = self._root_motion_sequence(speeds=[.1]*4, rolls=[0., -.8, .8, math.pi], progress=[0., .4, .3, -.2])
        motion = row["root_motion"]
        self.assertEqual(motion["extrema"]["roll_rad"], [-.8, 2, round(math.pi, 6), 4])
        self.assertEqual(motion["extrema"]["up_z"], [-1., 4, 1., 1])
        self.assertEqual(motion["first_up_z_negative_step"], 4)
        self.assertEqual(motion["max_x_retreat"], [2, 4, .6])
        self.assertEqual(motion["last"][3], -1.)

    def test_unloaded_low_motion_duration_threshold_history_and_overlap(self):
        short = self._root_motion_sequence(speeds=[.05]*24)
        exact = self._root_motion_sequence(speeds=[.05]*25)
        sustained = self._root_motion_sequence(speeds=[.05]*60, history=True)
        above = self._root_motion_sequence(speeds=[.050001]*60)
        alternating = self._root_motion_sequence(speeds=[.01]*60,
            contacts=[(i%2==0, i%2!=0) for i in range(60)])
        self.assertFalse(short["before_terminal"]["unloaded_low_motion"])
        self.assertFalse(exact["before_terminal"]["unloaded_low_motion"])  # confirms AT terminal
        self.assertEqual([e["confirmed"] for e in exact["events"] if e["code"]=="unloaded_low_motion"], [25, 25])
        self.assertTrue(sustained["before_terminal"]["unloaded_low_motion"])
        self.assertFalse(above["before_terminal"]["unloaded_low_motion"])
        self.assertFalse(alternating["before_terminal"]["unloaded_low_motion"])
        for row in (sustained, above):
            self.assertTrue(row["before_terminal"]["low_progress"])
        slow = [e for e in sustained["events"] if e["code"]=="unloaded_low_motion"]
        self.assertEqual(slow[0]["measured"], {"max_speed_mps": .05, "root_x_gain_m": 0.})
        self.assertEqual((slow[0]["start"], slow[0]["end"]), (1, 60))
        self.assertTrue(any(e["code"]=="contact_history" for e in sustained["events"]))
        self.assertFalse(any(e["code"]=="contact_instant" for e in sustained["events"]))
        selected = _representatives([sustained], enhanced=True)[0]
        self.assertTrue(any(e["code"]=="unloaded_low_motion" for e in selected["events"]))
        self.assertTrue(any(e["code"]=="low_progress" for e in selected["events"]))

    def _named_motion_episode(self, *, order=(0, 1, 2, 3), mirror=False, reorder_names=True):
        """Synthetic measured fields, not mirrored robot dynamics or terrain."""
        names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        reflected_names = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
        labels = reflected_names if mirror else names
        bound_names = tuple(labels[i] for i in order) if reorder_names else labels
        cell = self.config.ordered_evaluation_suite.private_cell_assignment("public_dev")[0]
        state = _Episode(cell, 1, 0, self.config.evaluation_semantics.outcomes["signals"], bound_names, True)
        velocity = torch.tensor([[.6, .2, .12], [-.3, .55, -.08], [.1, -.15, .09], [-.2, -.1, -.11]], dtype=torch.float64)
        history = torch.zeros((3, 4, 3), dtype=torch.float64)
        clearance = (.012, .04, .014, .06)
        for step in range(1, 46):
            contact = [((step+i*3) % 16) < 7 for i in range(4)]
            forces = torch.tensor([[2.+i, .1*(i+1), .2] if contact[i] else [0., 0., 0.]
                                   for i in range(4)], dtype=torch.float64)
            history = torch.cat((forces[None], history[:2]), dim=0)
            roll, pitch, yaw = (.8 if 4 <= step <= 17 else .2), .1, .35
            cr, sr, cp, sp, cy, sy = (math.cos(roll/2), math.sin(roll/2), math.cos(pitch/2),
                                     math.sin(pitch/2), math.cos(yaw/2), math.sin(yaw/2))
            quat = torch.tensor([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                                 cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy], dtype=torch.float64)
            v, h = velocity.clone(), history.clone()
            forbidden = torch.zeros((3, 1, 3), dtype=torch.float64)
            if step == 45:
                forbidden[0, 0] = torch.tensor([2., -1., .5])
            if mirror:
                # R' = M R M, M=diag(1,-1,1): roll/yaw change sign,
                # pitch and world-Z do not. Left/right names change with bodies.
                v[:, 1] *= -1
                h[:, :, 1] *= -1
                forbidden[:, :, 1] *= -1
                quat[[1, 3]] *= -1
            transition = SimpleNamespace(episode_step=step, progress_m=.08*min(step, 12),
                support_contacts=tuple(contact[i] for i in order),
                support_planar_speeds_mps=tuple(float(torch.linalg.vector_norm(v[i, :2])) for i in order),
                support_surface_clearances_m=tuple(clearance[i] for i in order), forbidden_contact=step == 45)
            raw = {"base_quat_w": quat[None], "support_lin_vel_w": v[list(order)][None],
                   "support_contact_forces_w_history": h[:, list(order), :][None],
                   "forbidden_contact_forces_w_history": forbidden[None]}
            episode = state.observe(transition, raw, SimpleNamespace(primary_reason="forbidden_contact") if step == 45 else None)
        return episode

    @staticmethod
    def _normalize_named_episode(episode, *, reflect_names=False):
        row = deepcopy(episode)
        swap = {"FL_foot": "FR_foot", "FR_foot": "FL_foot", "RL_foot": "RR_foot", "RR_foot": "RL_foot"} if reflect_names else {}
        for event in row["events"]:
            event["body"] = swap.get(event["body"], event["body"])
        row["events"].sort(key=lambda e: (e["start"], e["body"], e["code"]))
        row["body_extrema"] = sorted([[swap.get(e[0], e[0]), *e[1:]] for e in row["body_extrema"]])
        return row

    def test_named_body_reorder_preserves_events_and_extrema_with_bound_names(self):
        original = self._normalize_named_episode(self._named_motion_episode())
        reordered = self._normalize_named_episode(self._named_motion_episode(order=(2, 0, 3, 1)))
        self.assertEqual(reordered, original)
        # This control proves the fixture distinguishes correct named binding
        # from a tensor permutation with stale names.
        stale_names = self._normalize_named_episode(self._named_motion_episode(order=(2, 0, 3, 1), reorder_names=False))
        self.assertNotEqual(stale_names["events"], original["events"])
        self.assertNotEqual(stale_names["body_extrema"], original["body_extrema"])
        self.assertTrue(any(e["code"] == "slip_body" and e["confirmed"] is not None for e in original["events"]))
        self.assertTrue(any(e["code"] == "low_clearance_body" and e["confirmed"] is not None for e in original["events"]))

    def test_world_y_roll_reflection_keeps_scalar_rules_and_reflects_signed_pose(self):
        original = self._normalize_named_episode(self._named_motion_episode())
        reflected = self._normalize_named_episode(self._named_motion_episode(mirror=True), reflect_names=True)
        self.assertEqual(reflected["ranges"]["roll_rad"], [-original["ranges"]["roll_rad"][1], -original["ranges"]["roll_rad"][0]])
        reflected["ranges"]["roll_rad"] = original["ranges"]["roll_rad"]
        a, b = original["root_motion"], reflected["root_motion"]
        for name in ("first", "last"):
            self.assertEqual(b[name][1], -a[name][1])
            b[name][1] = a[name][1]
        self.assertEqual(b["extrema"]["roll_rad"], [-a["extrema"]["roll_rad"][2], a["extrema"]["roll_rad"][3],
                                                       -a["extrema"]["roll_rad"][0], a["extrema"]["roll_rad"][1]])
        b["extrema"]["roll_rad"] = a["extrema"]["roll_rad"]
        self.assertEqual(reflected, original)
        self.assertTrue(original["before_terminal"]["tilt"])
        self.assertTrue(any(e["code"] == "unloaded_rising" and e["confirmed"] is not None for e in original["events"]))
        self.assertTrue(any(e["code"] == "unloaded_lowering" and e["confirmed"] is not None for e in original["events"]))

    def test_parallel_per_body_streak_window_and_terminal_order(self):
        cell = self.config.ordered_evaluation_suite.private_cell_assignment("public_dev")[0]
        state = _Episode(cell, 0, 0, self.config.evaluation_semantics.outcomes["signals"], ("left", "right"), False)
        for step in range(1, 61):
            # Alternating feet sustain the global rule without a same-foot streak.
            contact = (step % 2 == 0, step % 2 != 0) if step <= 6 else (False, False)
            t = SimpleNamespace(episode_step=step, progress_m=0., support_contacts=contact,
                support_planar_speeds_mps=(.6, .6), support_surface_clearances_m=(.1, .1),
                forbidden_contact=step == 61)
            raw = {"base_quat_w": torch.tensor([[1., 0., 0., 0.]]),
                   "support_lin_vel_w": torch.tensor([[[0., 0., .1], [0., 0., -.1]]])}
            episode = state.observe(t, raw, SimpleNamespace(primary_reason="timeout") if step == 60 else None)
        self.assertTrue(episode["before_terminal"]["slip_any"])
        self.assertFalse(episode["window_present"]["slip_any"])
        self.assertFalse(any(e["confirmed"] is not None for e in episode["events"] if e["code"] == "slip_body"))
        self.assertTrue(any(e["code"] == "unloaded_rising" for e in episode["events"]))
        self.assertTrue(any(e["code"] == "unloaded_lowering" for e in episode["events"]))
        self.assertEqual(episode["coverage"]["window_first_step"], 11)

    def test_phase_sealing_portable_replay_and_actual_author_payload(self):
        from tests.unit.test_stair_phased_evaluation import StairPhasedEvaluationTests
        from true_rl_agent.evaluation.phases import EvaluationPhase
        from true_rl_agent.evaluation.binding import canonical_json_hash
        from true_rl_agent.runner.specs import EvalPhaseJobSpec
        from true_rl_agent.tasks.stair_climbing.phased_evaluation import (
            build_stair_eval_phase_summary, finalize_stair_eval_phase_summary,
        )
        from true_rl_agent.tasks.stair_climbing.phased_adapter import StairPhasedEvaluationAdapter
        helper = StairPhasedEvaluationTests()
        helper.config = self.config
        plan = helper._eval_plan("behavior")
        base = helper._execution({EvaluationPhase.PUBLIC_DEV: [0, 0, 0]})
        base = replace(base, summaries={"public_dev": self.summary},
                       per_seed_outcomes={"public_dev": self.execution.per_seed_outcomes})
        identity = {**plan.job_identity().to_public_dict(), "checkpoint_hash": base.checkpoint_hash,
            "evaluation_plan_hash": canonical_json_hash(plan.to_public_dict()),
            "evaluation_binding": plan.evaluation_binding.to_public_dict(), "seed_namespace": STAIR_FORMAL_EVAL_NAMESPACE,
            "splits": ["public_dev"], "environment_source_preflight": dict(base.source_manifest_preflight),
            "backend_execution_environment_verification": dict(base.backend_execution_environment_verification)}
        with tempfile.TemporaryDirectory() as directory:
            writer = NumericEvidenceWriter(Path(directory) / "archive", identity)
            for header, arrays in iter_numeric_evidence(self.reference):
                writer.write_chunk(arrays, header)
            ref = writer.complete(load_numeric_evidence_manifest(self.reference)["metadata"])
            raw = build_stair_eval_phase_summary(config=self.config, identity=plan.job_identity(),
                evaluation_plan_hash=identity["evaluation_plan_hash"], phase=EvaluationPhase.PUBLIC_DEV,
                execution=replace(base, numeric_evidence=ref), evaluation_binding=plan.evaluation_binding,
                checkpoint_safety_report=helper._checkpoint_report())
            self.assertEqual(raw["status"], "postprocessing")
            self.assertNotIn("behavior", raw["task_evidence"])
            raw = finalize_stair_eval_phase_summary(EvalPhaseJobSpec(
                job_id=plan.job_id, evaluation_plan=plan, phase=EvaluationPhase.PUBLIC_DEV),
                raw, config=self.config, output_dir=Path(directory))
            self.assertEqual(raw["status"], "pass")
            costs = raw["evaluation_execution_cost"]["splits"]["public_dev"]
            self.assertGreaterEqual(costs["postprocessing_wall_time_s"], 0)
            self.assertGreaterEqual(costs["postprocessing_cpu_time_s"], 0)
            phase = helper._phase_result(plan=plan, phase=EvaluationPhase.PUBLIC_DEV, job_id=plan.job_id, summary=raw)
            self.assertIn("behavior", phase.task_evidence)
            full_reference = phase.task_evidence["behavior_full"]
            full_path = Path(directory) / full_reference["file"]
            self.assertEqual(full_reference["sha256"], hashlib.sha256(full_path.read_bytes()).hexdigest())
            self.assertEqual(full_reference["size_bytes"], full_path.stat().st_size)
            complete_behavior = json.loads(full_path.read_text(encoding="utf-8"))
            self.assertEqual(project_stair_behavior(complete_behavior), phase.task_evidence["behavior"])
        # No archive read while projecting historical sealed phase receipts.
        adapter = StairPhasedEvaluationAdapter(self.config)
        public_summary = adapter.project_public_phase_summary(phase_result=phase)
        self.assertEqual(public_summary["behavior"], raw["task_evidence"]["behavior"])
        legacy = replace(phase, task_evidence={k: v for k, v in phase.task_evidence.items() if k not in {"behavior", "behavior_full"}})
        self.assertEqual(adapter.project_public_phase_summary(phase_result=legacy), self.summary)

        accept = helper._execution({EvaluationPhase.PUBLIC_ACCEPT: [6]*5}).summaries["public_accept"]
        mask = build_reference_objective_family_mask(suite=self.config.ordered_evaluation_suite, public_accept_summary=accept)
        task = StairClimbingPromptAdapter(self.config, resolved_cell_manifest=self.manifest,
                                          objective_family_mask_provider=lambda: mask)
        parent_summary = deepcopy(public_summary)
        parent_summary["behavior"]["domain"]["checkpoint_hash"] = "parent-checkpoint"
        record = {"round_id": "g1", "generation": {
            "generation_id": "g1", "generation_index": 1, "generation_outcome": "incomplete", "formal": None,
            "parent_current_best": {"checkpoint_hash": "parent-checkpoint"}, "parent_public_dev": parent_summary,
            "trials": [{"trial_id": "g1-a", "lane_id": "lane-a", "trial_status": "public_dev_complete",
                        "candidate": {"candidate_id": "candidate-a"}, "public_dev": {"public_summary": public_summary}},
                       {"trial_id": "g1-b", "lane_id": "lane-b", "trial_status": "candidate_failed",
                        "candidate": {"candidate_id": "candidate-b"}, "status_reason": "candidate_runtime_failed"}]}}
        projected = task.project_prompt_public_history(public_records=[record], initial_anchor_public_evidence=None)
        self.assertIn("behavior", projected.records[0]["generation"]["parent_public_dev"])
        builder = PublicContextBuilder(task_prompt_adapter=task)
        payload = builder.build_prompt_payload([record])
        shown = payload["public_feedback_summary"]["behavior"]
        self.assertEqual(shown["source_trial_id"], "g1-a")
        self.assertEqual(shown["diff"]["status"], "available")
        expected_profile = deepcopy(public_summary["behavior"])
        expected_profile["report"]["parent_diff"] = [{"id": shown["diff"]["statement"]["id"],
            "text": shown["diff"]["statement"]["text"]}]
        self.assertEqual(shown["profile"], expected_profile)
        self.assertIn("confirmed strictly before terminal", shown["profile"]["report"]["parent_diff"][0]["text"])
        sections = task.build_prompt_sections(public_records=[record], public_context={})
        minimum = _minimum_core_stair_prompt_sections(sections, target_bytes=56*1024)
        self.assertEqual(minimum["public_feedback_summary"]["behavior"], shown)
        encoded = json.dumps(payload)
        self.assertNotIn("manifest_path", encoded)
        self.assertNotIn("behavior_full", encoded)
        self.assertNotIn(self.temp.name, encoded)
        self.assertLessEqual(len(json.dumps(shown, separators=(",", ":")).encode()), 12*1024)
        self._assert_authoring_evidence_delivery(builder, record, payload)

    def _assert_authoring_evidence_delivery(self, builder, record, draft):
        from tests.unit.test_candidate_authoring_loop import _write_candidate_package
        from true_rl_agent.agent_core.openai_compatible_provider import (
            OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig,
        )
        from true_rl_agent.prompting.task_sections import validate_provider_task_prompt_receipt
        with tempfile.TemporaryDirectory() as directory:
            failed = _write_candidate_package(Path(directory), "failed-evidence-proposal",
                "def reward(inp):\n    return inp.base_height\n")
            provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(
                model="offline", output_dir=Path(directory) / "provider"))
            for category, inline, kind in (
                ("verifier_error", True, "static_repair"),
                ("authoring_policy_violation", True, "static_repair"),
                ("reasoning_output_exhausted", False, "static_repair"),
                ("response_truncated", True, "static_repair"),
                ("verifier_error", True, "source_followup"),
            ):
                with self.subTest(category=category, inline=inline, kind=kind):
                    payload = builder.build_prompt_payload([record],
                        latest_failed_candidate=failed if inline else None,
                        candidate_repair_feedback={"failure_category": category, "reason": "repair fixture"},
                        frame_kind=kind)
                    request = provider._chat_completion_request(payload)
                    text = request["messages"][1]["content"]
                    delivered = json.loads(text)
                    for section in ("public_feedback_summary", "selection_guidance", "exploration_directive"):
                        self.assertEqual(delivered[section], draft[section])
                    self.assertIn("behavior", delivered["public_feedback_summary"])
                    budget = delivered["authoring_context_frame"]["budget_metadata"]
                    self.assertEqual(budget["actual_prompt_payload_bytes"], len(text.encode()))
                    validate_provider_task_prompt_receipt(budget["provider_task_sections"], prompt_payload=delivered)
                    if inline and kind == "static_repair":
                        self.assertNotIn("public_history", delivered)

    def test_qualified_reference_assembly_reaches_empty_history_author_without_fake_trial(self):
        from tests.unit.test_stair_phased_evaluation import StairPhasedEvaluationTests
        from true_rl_agent.evaluation.binding import canonical_json_hash
        from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
        from true_rl_agent.evaluation.phases import EvaluationPhase
        from true_rl_agent.evaluation.subject import ExperimentSubject
        from true_rl_agent.runner.budget import build_budget_accounting
        from true_rl_agent.runner.subprocess_runner import SubprocessRunner
        from true_rl_agent.tasks.stair_climbing.phased_evaluation import assemble_stair_reference_eval_phase_results
        from true_rl_agent.tasks.stair_climbing.public_projection import project_stair_public_result
        helper = StairPhasedEvaluationTests()
        helper.config = self.config
        reference_source = "# Reference source for the CPU receipt fixture.\n"
        plan = replace(helper._eval_plan("reference-behavior"), subject=ExperimentSubject.reference("behavior-reference"),
                       program_source_checksum="sha256:"+hashlib.sha256(reference_source.encode()).hexdigest())
        phases = [helper._phase_result_for(plan=plan, phase=phase, successes=[12]*helper._seed_count(phase))
                  for phase in EvaluationPhase]
        dev = next(phase for phase in phases if phase.phase is EvaluationPhase.PUBLIC_DEV)
        behavior = deepcopy(self.public)
        behavior["domain"]["checkpoint_hash"] = dev.checkpoint_hash
        dev = replace(dev, task_evidence={**dev.task_evidence, "summary": self.summary,
                                         "numeric_evidence": self.reference, "behavior": behavior})
        phases = [dev if phase.phase is EvaluationPhase.PUBLIC_DEV else phase for phase in phases]
        with tempfile.TemporaryDirectory() as directory:
            receipts = helper._publish_phase_receipts(root=Path(directory), plan=plan, phase_results=phases)
            assembly = assemble_stair_reference_eval_phase_results(config=self.config, eval_plan=plan,
                phase_results=phases, phase_receipt_refs=receipts)
            self.assertEqual(assembly.summary["metrics"]["behavior"], behavior)
            runner_public = SubprocessRunner(allowed_executables=("python",)).project_assembled_eval_summary(plan, assembly)
        # Complete but synthetic CPU receipts: this tests the real qualification
        # restoration/consumer path, and is not evidence of simulator training.
        calls = self.config.train_steps_per_iteration*self.config.train_max_iterations+1
        train = {"environment_source_preflight": dict(dev.environment_source_preflight),
            "program_source_checksum": plan.program_source_checksum, "subject_manifest_hash": plan.subject_manifest_hash,
            "reward_program_qualification_hash": "9"*64, "subject_kind": "reference", "subject_id": plan.subject.subject_id,
            "training": {"budget_accounting": build_budget_accounting(budget_steps=self.config.budget_steps,
                num_envs=self.config.train_num_envs, num_steps_per_env=self.config.train_steps_per_iteration,
                max_iterations=self.config.train_max_iterations)},
            "runtime_activation_summary": {"status": "pass",
                "runtime_status": self.config.train_runtime_profile().to_public_dict()["complete_reward_runtime"],
                "dispatcher_calls": calls, "term_calls": calls, "declared_term_count": 1, "reward_term_calls": calls,
                "diagnostic_calls": 0, "nonzero_reward_contribution": True,
                "input_guard": "pass", "rng_guard": "pass", "source_alias_guard": "pass",
                "training_recorder_enabled": False, "reward_manager_active_terms": ["complete_reward"],
                "reward_manager_term_weight": 1., "component_activation_records": [{"name": "fixture_term"}],
                "capability_activation_records": ["complete_reward_program"],
                "program": {"term_names": ["fixture_term"], "program_source_checksum": plan.program_source_checksum,
                    "subject_manifest_hash": plan.subject_manifest_hash, "qualification_hash": "9"*64,
                    "input_schema_hash": self.projection.reward_input_schema_hash_vnext,
                    "subject": {"kind": "reference", "id": plan.subject.subject_id}},
                "activation_preflight": {"schema_version": "true_rl_agent.task.stair_climbing.reward_activation_preflight.v2",
                    "status": "pass", "real_transition_dispatcher_calls": 1, "transition_common_step": 1,
                    "repeat_count": self.config.vnext_reward_authority.runtime_authority_payload()["dispatcher"]["determinism_preflight_repeats"],
                    "term_names": ["fixture_term"]}}}
        public = project_stair_public_result(config=self.config, public_result=runner_public, train_summary=train,
                                              resolved_cell_manifest=self.manifest)
        self.assertEqual(public.metrics["behavior"], behavior)
        best_args = dict(reference_id=plan.subject.subject_id, reference_manifest_hash=plan.subject_manifest_hash,
            program_source_checksum=plan.program_source_checksum, training_binding_hash=plan.training_binding_hash,
            capability_profile_hash=plan.capability_profile_hash, checkpoint_hash=dev.checkpoint_hash,
            acceptance_protocol=self.config.acceptance_protocol, expected_evaluation_binding=plan.evaluation_binding)
        best = CurrentBestEvalStats.from_reference_public_result_mapping(public_result=public.to_public_dict(), **best_args)
        self.assertEqual(best.mode, "reference")
        legacy_public = public.to_public_dict()
        legacy_public["metrics"].pop("behavior")
        self.assertEqual(CurrentBestEvalStats.from_reference_public_result_mapping(public_result=legacy_public, **best_args), best)
        anchor = {"anchor_kind": "reference_program", "public_result": public.to_public_dict(),
                  "public_result_hash": canonical_json_hash(public.to_public_dict())}
        task = StairClimbingPromptAdapter(self.config, resolved_cell_manifest=self.manifest,
            objective_family_mask_provider=lambda: public.metrics["objective_family_mask"])
        payload = PublicContextBuilder(task_prompt_adapter=task, initial_current_best=best,
            session_genesis_hash="a"*64, acceptance_protocol=self.config.acceptance_protocol,
            initial_reference_program_source=reference_source,
            initial_anchor_public_evidence=anchor).build_prompt_payload([])
        shown = payload["public_feedback_summary"]["behavior"]
        self.assertEqual(shown["source_kind"], "reference_anchor")
        self.assertEqual(shown["source_public_result_hash"], anchor["public_result_hash"])
        self.assertEqual(shown["profile"], behavior)
        self.assertEqual(shown["diff"]["status"], "not_available")
        self.assertNotIn("source_trial_id", shown)
        self.assertEqual(payload["public_history"]["recent_outcomes"], [])
        old_anchor = {**anchor, "public_result": legacy_public, "public_result_hash": canonical_json_hash(legacy_public)}
        old_payload = PublicContextBuilder(task_prompt_adapter=task, initial_current_best=best,
            session_genesis_hash="a"*64, acceptance_protocol=self.config.acceptance_protocol,
            initial_reference_program_source=reference_source,
            initial_anchor_public_evidence=old_anchor).build_prompt_payload([])
        self.assertNotIn("behavior", old_payload["public_feedback_summary"])


if __name__ == "__main__":
    unittest.main()
