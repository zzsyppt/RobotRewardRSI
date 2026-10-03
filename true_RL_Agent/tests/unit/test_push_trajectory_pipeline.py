"""Normal Push profile and CPU boundaries; synthetic capture is not policy evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from true_rl_agent.composition.base import TaskCompositionContext, resolve_task_composition
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, report_bytes
from true_rl_agent.runner.numeric_evidence import file_sha256
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA, NumericEvidenceError, load_numeric_evidence_manifest
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.trajectory_pipeline import (
    build_phase_trajectory, replay_full_report, validate_build_receipt, validate_feedback_profile, TRAJECTORY_PROFILE,
)
from true_rl_agent.tasks.push_recovery.trajectory_report import require_trial_binding, validate_push_trajectory_report
from true_rl_agent.tasks.push_recovery.trajectory_card import json_bytes
from true_rl_agent.tasks.push_recovery.rollout import aggregate_recovery_episodes

ROOT = Path(__file__).resolve().parents[2]


def seed_from_manifest(manifest, reference=None, *, split="public_dev"):
    """Archive-backed rollout projection; no invented action/reward observations."""
    m, i = manifest["metadata"], manifest["identity"]
    rows = deepcopy(m["runtime_episode_records"])
    force = "force" in m
    result = {"schema_version": "true_rl_agent.push_recovery.rollout.v3" if force else "true_rl_agent.push_recovery.rollout.v2",
        "status": "pass", "run": {"split": split, "run_index": i["run_index"]},
        "task": {"mechanism": i["mechanism"], "diagnostic_control_only": m["control_only"]},
        "environment": {"num_envs": m["num_envs"]}, "per_episode": rows,
        "aggregate": aggregate_recovery_episodes(rows, force_sequence=force),
        "evidence": {**{k: True for k in ("reset_ok", "observation_finite", "reward_finite",
            "events_recorded_at_injection", "outcomes_captured_before_reset", "all_first_episodes_closed")},
            "resolved_protocol_hash": i["resolved_protocol_hash"]},
        "execution_cost": {"evaluation_env_count": m["num_envs"],
            "vector_env_step_calls": max(r["episode_steps"] for r in rows),
            "closed_first_episode_control_steps": sum(r["episode_steps"] for r in rows)}}
    if reference:
        result["numeric_evidence_ref"] = reference
    return result


def rebind_archives(paths, root, identity, context):
    """Explicit fixture rebinding; chunks unchanged and original manifests untouched."""
    seeds, mappings = [], []
    for index, source in enumerate(paths, 1):
        source = Path(source)
        original = json.loads(source.read_bytes())
        manifest = deepcopy(original)
        manifest["identity"].update(identity, resolved_cell_manifest_hash=context["resolved_cell_manifest_hash"],
                                     run_index=index)
        target = Path(root) / f"public_dev-{index:02d}"
        target.mkdir(parents=True)
        for chunk in manifest["chunks"]:
            shutil.copyfile(source.parent / chunk["file"], target / chunk["file"])
            assert file_sha256(target / chunk["file"]) == chunk["sha256"]
        path = target / "manifest.json"
        path.write_bytes(json_bytes(manifest))
        ref = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA, "manifest_path": str(path), "manifest_sha256": file_sha256(path)}
        seeds.append(seed_from_manifest(manifest, ref))
        mappings.append({"original_manifest": str(source), "original_sha256": file_sha256(source),
            "original_identity": original["identity"], "fixture_reference": ref, "fixture_identity": manifest["identity"],
            "unchanged_chunks": [c["sha256"] for c in manifest["chunks"]]})
    return seeds, mappings


class PushTrajectoryPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.unit.test_push_numeric_capture import PushNumericCaptureTests
        cls.fixture = PushNumericCaptureTests()
        cls.fixture.setUp()
        f = cls.fixture
        capture = f.capture(f.force, "p2b2-force")
        f.force.until(f.force.env.max_episode_length)
        ref = capture.complete(f.force.state.episode_records())
        cls.raw = Path(ref["manifest_path"])
        manifest = load_numeric_evidence_manifest(ref)
        cls.identity = {"job_id": "trial-a-eval-public-dev", "task_id": "unitree_push_recovery",
            "subject_kind": "candidate", "subject_id": "candidate-a", "subject_manifest_hash": "1" * 64,
            "program_source_checksum": "sha256:" + "2" * 64, "training_binding_hash": "3" * 64,
            "capability_profile_hash": "4" * 64, "checkpoint_hash": "sha256:" + "5" * 64,
            "evaluation_plan_hash": "6" * 64, "evaluation_binding": {"evaluation_binding_hash": "7" * 64},
            "runtime_dynamics_hash": "sha256:" + "8" * 64}
        cls.context = {"cell_variant_id": "push.go2_12dof.recovery_v3", "resolved_cell_manifest_hash": "9" * 64,
            "resolved_protocol_hash": manifest["identity"]["resolved_protocol_hash"]}

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.seeds, self.mapping = rebind_archives([self.raw] * 3, self.root / "raw", self.identity, self.context)

    def build(self, **kwargs):
        return build_phase_trajectory(kwargs.pop("seeds", self.seeds), expected_identity=kwargs.pop("identity", self.identity),
            context=self.context, expected_seed_count=3, output_dir=self.root / "phase", **kwargs)

    def test_budget_degradation_keeps_push_results_and_visible_reason(self):
        from types import SimpleNamespace
        from true_rl_agent.evaluation.phases import EvaluationPhase
        from true_rl_agent.tasks.push_recovery.phased_evaluation import finalize_push_eval_phase_summary
        from true_rl_agent.tasks.push_recovery.prompt_adapter import _v2_dev_summary
        identity = self.identity
        spec = SimpleNamespace(phase=EvaluationPhase.PUBLIC_DEV,
            job_identity=lambda:SimpleNamespace(to_public_dict=lambda:identity),
            execution_checkpoint=lambda:SimpleNamespace(checksum=identity["checkpoint_hash"]),
            evaluation_plan_hash=identity["evaluation_plan_hash"],
            program_source_checksum=identity["program_source_checksum"],
            training_binding_hash=identity["training_binding_hash"],
            evaluation_binding=SimpleNamespace(to_public_dict=lambda:identity["evaluation_binding"]))
        context = {**self.context, **{k:identity[k] for k in ("checkpoint_hash","evaluation_plan_hash","program_source_checksum","runtime_dynamics_hash")}}
        summary = {"status":"postprocessing", "task_evidence":{"seed_summaries":self.seeds,"behavior_context":context},
            "environment_source_preflight":{"runtime_dynamics_hash":identity["runtime_dynamics_hash"]},
            "evaluation_execution_cost":{"splits":{"public_dev":{}}}}
        final = finalize_push_eval_phase_summary(spec,summary,output_dir=self.root/"small",
            behavior_feedback_profile=TRAJECTORY_PROFILE,trajectory_report_max_bytes=128,expected_seed_count=3)
        self.assertEqual(final["status"],"pass")
        evidence = final["task_evidence"]
        self.assertEqual(evidence["seed_summaries"],self.seeds)
        self.assertNotIn("behavior",evidence)
        self.assertNotIn("trajectory_build",evidence)
        self.assertEqual(_v2_dev_summary({"description_status":evidence["description_status"]})["description_status"]["reason"],"budget_exceeded")

    def test_complete_three_archive_group_and_public_replay_without_raw_io(self):
        full, public, receipt = self.build()
        self.assertEqual(receipt["episodes"], 12)
        self.assertEqual(len(public["domain"]["manifest_sha256"]), 3)
        self.assertEqual(len(receipt["required_claim_ids"]), 20)
        self.assertLessEqual(report_bytes(public), 12288)
        self.assertEqual(public["domain"]["root_signed_xy"], "recorded_world_xy")
        self.assertEqual(public["domain"]["mechanism"], "world_planar_external_force")
        with patch("true_rl_agent.runner.numeric_evidence.load_numeric_evidence_manifest", side_effect=AssertionError("prompt opened raw")):
            self.assertEqual(public, validate_push_trajectory_report(public))
            validate_build_receipt(receipt, public)
        self.assertEqual(public, replay_full_report(full, expected_sources=full["sources"], expected_identity=self.identity))

    def test_retry_reuses_bytes_but_checks_raw_and_configuration(self):
        first = self.build()
        marker = Path(first[2]["artifact_manifest"]["manifest_path"])
        before = marker.read_bytes(), marker.stat().st_mtime_ns
        again = self.build()
        self.assertEqual(first[:2], again[:2])
        self.assertTrue(again[2]["artifacts_reused"])
        self.assertEqual(before, (marker.read_bytes(), marker.stat().st_mtime_ns))
        smaller = self.build(max_bytes=12287)
        self.assertNotEqual(first[2]["cache_key"], smaller[2]["cache_key"])
        marker.with_name("policy_card.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.build()

    def test_missing_duplicate_or_foreign_seed_cannot_claim_complete_group(self):
        for seeds in (self.seeds[:-1], [self.seeds[0]] * 3,
                      [{k: v for k, v in self.seeds[0].items() if k != "numeric_evidence_ref"}, *self.seeds[1:]]):
            with self.subTest(seeds=len(seeds)), self.assertRaises(ValueError):
                self.build(seeds=seeds)
        for k in ("job_id", "checkpoint_hash", "evaluation_plan_hash", "subject_kind", "program_source_checksum"):
            with self.subTest(k=k), self.assertRaisesRegex(NumericEvidenceError, "another phase"):
                self.build(identity={**self.identity, k: "foreign"})
        Path(self.seeds[1]["numeric_evidence_ref"]["manifest_path"]).unlink()
        with self.assertRaisesRegex(NumericEvidenceError, "not available"):
            self.build()

    def test_mandatory_budget_and_resigned_semantics_rejected(self):
        with self.assertRaisesRegex(ValueError, "mandatory"):
            self.build(max_bytes=100)
        full, public, receipt = self.build()
        from true_rl_agent.tasks.push_recovery.trajectory_card import _render_claims
        from true_rl_agent.evaluation.trajectory_translation.contracts import stable_id
        for name in ("metric", "unit", "prepost"):
            broken = deepcopy(full)
            card = broken["card"]
            if name == "metric":
                metrics = card["cohorts"]["all"]["pose"]
                metrics["height_m"], metrics["planar_speed_mps"] = metrics["planar_speed_mps"], metrics["height_m"]
            else:
                event = card["examples"][0]["value"]["events"][0]
                if name == "unit":
                    event["vector_unit"] = "wrong"
                else:
                    w = event["windows"]
                    w["pre"], w["post"] = w["post"], w["pre"]
            card["claims"] = _render_claims(card["domain"], card["cohorts"], card["examples"], card["selection"], card["parameters"], card["timing"])
            card["card_id"] = stable_id("push-card", {k: v for k, v in card.items() if k not in {"card_id", "public_report", "projection_audit"}})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "semantic operands"):
                replay_full_report(broken, expected_sources=full["sources"], expected_identity=self.identity)
        bad = deepcopy(public)
        victim = receipt["required_claim_ids"][0]
        for section in SENTENCE_SECTIONS:
            bad["report"][section] = [r for r in bad["report"][section] if r["id"] != victim]
        bad["report_projection"]["included_claim_count"] -= 1
        bad["report_projection"]["omitted_claim_count"] += 1
        bad["report_projection"].pop("content_hash")
        bad["report_projection"]["content_hash"] = canonical_json_hash(bad)
        with self.assertRaises(ValueError):
            validate_build_receipt(receipt, bad)

    def test_sibling_and_role_binding(self):
        _, public, _ = self.build()
        trial = {"trial_id": "trial-a", "candidate": {"candidate_id": "candidate-a",
                 "program_source_checksum": self.identity["program_source_checksum"]},
                 "public_dev": {k: self.identity[k] for k in ("checkpoint_hash", "evaluation_plan_hash")}}
        require_trial_binding(public, trial)
        with self.assertRaisesRegex(ValueError, "sibling/trial/role"):
            require_trial_binding(public, {**trial, "trial_id": "trial-b"})

    def test_force_v1_retains_missing_direction_and_mixed_capabilities_reject(self):
        ref = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA, "manifest_path": str(self.raw), "manifest_sha256": file_sha256(self.raw)}
        old = self.fixture.rewrite_archive(ref, "p2b2-legacy", legacy=True)
        seeds, _ = rebind_archives([old["manifest_path"]] * 3, self.root / "v1", self.identity, self.context)
        _, public, _ = self.build(seeds=seeds)
        self.assertEqual(public["domain"]["root_signed_xy"], "not_recorded")
        with self.assertRaisesRegex(ValueError, "no cross-condition pooling"):
            self.build(seeds=[seeds[0], *self.seeds[1:]])

    def test_feedback_uses_selected_generation_and_keeps_accepted_sibling_distinct(self):
        from true_rl_agent.tasks.push_recovery.prompt_adapter import _v2_prompt_record, _v2_feedback_summary, _minimum_core_push_prompt_sections
        full, a, _ = self.build()
        identity = {**self.identity, "job_id": "trial-b-eval-public-dev", "subject_id": "candidate-b"}
        seeds, _ = rebind_archives([self.raw] * 3, self.root / "b", identity, self.context)
        _, b, _ = self.build(seeds=seeds, identity=identity)
        def trial(report, name, status):
            return {"trial_id": "trial-" + name, "lane_id": "lane-" + name, "trial_status": status, "train": {},
                "candidate": {"candidate_id": "candidate-" + name, "program_source_checksum": identity["program_source_checksum"]},
                "public_dev": {"checkpoint_hash": identity["checkpoint_hash"], "evaluation_plan_hash": identity["evaluation_plan_hash"],
                    "public_summary": {"behavior": report, "score_stats": {"mean": .7}}}}
        record = {"decision": {"accepted": True}, "generation": {"generation_id": "g1",
            "trials": [trial(a, "a", "accepted"), trial(b, "b", "not_selected")],
            "selection": {"selected_trial_id": "trial-a"}, "formal": {"accepted": True, "selected_trial_id": "trial-a"}}}
        projected = _v2_prompt_record(record)
        feedback = _v2_feedback_summary([projected, {"generation": {"generation_id": "g2", "trials": []}}])
        self.assertEqual(feedback["source_round_id"], "trial-b")
        self.assertEqual(feedback["trajectory_sources"]["generation_id"], "g1")
        self.assertEqual(feedback["current_best"]["source_round_id"], "trial-a")
        self.assertEqual(feedback["current_best"]["public_dev"]["behavior"]["source"]["card_id"], a["card_id"])
        self.assertEqual(len(feedback["trajectory_sources"]["siblings"]), 2)
        minimum = _minimum_core_push_prompt_sections({"public_feedback_summary": feedback})
        self.assertEqual(minimum["public_feedback_summary"]["trajectory_sources"], feedback["trajectory_sources"])
        bad = deepcopy(record)
        bad["generation"]["trials"][0]["public_dev"]["public_summary"]["behavior"] = b
        with self.assertRaisesRegex(ValueError, "sibling/trial/role"):
            _v2_prompt_record(bad)

    def test_reference_and_parent_role_cannot_be_changed_at_same_checkpoint(self):
        from true_rl_agent.tasks.push_recovery.prompt_adapter import _v2_reference_feedback, _v2_prompt_record
        full, public, receipt = self.build()
        result = {"metrics": {"public_dev": {"behavior": public}, "trajectory_phase_binding": receipt["phase_binding"]}}
        anchor = {"anchor_kind": "reference_program", "public_result": result, "public_result_hash": canonical_json_hash(result)}
        with self.assertRaisesRegex(ValueError, "reference role"):
            _v2_reference_feedback(anchor)
        record = {"generation": {"trials": [], "parent_public_dev": {"behavior": public},
            "parent_current_best": {"anchor_kind": "reference", "anchor_id": "candidate-a",
                **{k: self.identity[k] for k in ("checkpoint_hash", "program_source_checksum", "subject_manifest_hash", "training_binding_hash")},
                "evaluation_binding_hash": "7" * 64}}}
        with self.assertRaisesRegex(ValueError, "parent identity"):
            _v2_prompt_record(record)

    def test_full_replay_source_id_is_external_and_prompt_rejects_private_full(self):
        from true_rl_agent.tasks.push_recovery.behavior import project_push_behavior
        from true_rl_agent.tasks.push_recovery.trajectory_card import build_push_card
        from true_rl_agent.tasks.push_recovery.trajectory_adapter import iter_push_episodes
        from true_rl_agent.tasks.push_recovery.trajectory_measurement import measure_push_episode
        full, _, _ = self.build()
        with self.assertRaisesRegex(ValueError, "explicit CPU source"):
            project_push_behavior(full)
        episodes = [e for s in full["sources"] for e in iter_push_episodes(s["reference"], expected_identity=s["identity"])]
        full["card"] = build_push_card([measure_push_episode(e) for e in episodes], "forged-source", episodes=episodes)
        with self.assertRaisesRegex(ValueError, "semantic operands"):
            replay_full_report(full, expected_sources=full["sources"], expected_identity=self.identity)

    def test_normal_composition_profile_and_genesis_binding(self):
        provider = PushRecoveryCompositionProvider()
        values = {**provider.cli_defaults(), "cell_variant_id": self.context["cell_variant_id"]}
        context = TaskCompositionContext(project_root=ROOT, output_dir=self.root / "composition")
        old = resolve_task_composition(provider=provider, values=values, context=context)
        new = resolve_task_composition(provider=provider, values={**values, "behavior_feedback_profile": TRAJECTORY_PROFILE}, context=context)
        self.assertNotIn("behavior_feedback_profile", old.task_defaults)
        self.assertNotEqual(new.genesis_identity_snapshot(), old.genesis_identity_snapshot())
        self.assertEqual(new.backend_adapter.config.training_agent_configuration, old.backend_adapter.config.training_agent_configuration)
        self.assertEqual(new.task_plugin.config.evaluation_protocol(), old.task_plugin.config.evaluation_protocol())
        self.assertEqual(new.runner.eval_phase_summary_finalizer.keywords["expected_seed_count"], 3)
        self.assertEqual(new.runner.eval_phase_summary_finalizer.keywords["trajectory_report_max_bytes"], 12288)

    def test_profile_support_and_budget_fail_explicitly(self):
        for profile, budget, cell in (("wrong", 12288, self.context["cell_variant_id"]),
            (TRAJECTORY_PROFILE, 12289, self.context["cell_variant_id"]),
            (TRAJECTORY_PROFILE, 12288, "push.g1_29dof.reference_v1"),
            ("legacy_process_v1", 100, self.context["cell_variant_id"])):
            with self.subTest(profile=profile, budget=budget, cell=cell), self.assertRaises(ValueError):
                validate_feedback_profile(profile, budget, cell)
