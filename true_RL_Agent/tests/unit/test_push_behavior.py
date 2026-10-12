"""CPU archive semantics and the real task prompt path; no provider or simulator."""
from __future__ import annotations

import copy
import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import time
from dataclasses import replace
from types import SimpleNamespace

from tests.unit import test_push_numeric_capture
from tests.unit.test_push_recovery_v2_prompt_smoke import _dev
from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL, TEST_SESSION_GENESIS_HASH, make_bootstrap_current_best, valid_accepted_ledger_record,
)
from true_rl_agent.memory import PublicContextBuilder
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.tasks.push_recovery.behavior import (
    build_push_behavior, compare_push_behavior, project_push_behavior, push_behavior_context,
)
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.numeric_capture import PushNumericCapture, PushNumericCaptureOptions
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.prompt_adapter import PushRecoveryPromptAdapter, _minimum_core_push_prompt_sections
from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest


class PushBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_push_numeric_capture.PushNumericCaptureTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tmp.cleanup)

    def capture(self, *, force=False, name="behavior", extra_identity=None, control=False):
        config = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v3" if force else "push.go2_12dof.recovery_v2")
        manifest = resolve_push_cell_manifest(cell_variant_id=config.cell_variant_id, config=config, plugin=PushRecoveryTaskPlugin(config))
        context = push_behavior_context(manifest, evaluation_plan_hash="a" * 64,
            checkpoint_hash="sha256:" + "b" * 64, program_source_checksum="sha256:" + "c" * 64,
            runtime_dynamics_hash="sha256:" + "d" * 64, execution_source_hash="f" * 64)
        identity = {key: context[key] for key in ("evaluation_plan_hash", "checkpoint_hash", "program_source_checksum", "runtime_dynamics_hash", "resolved_cell_manifest_hash")}
        identity.update(evaluation_binding={"evaluation_binding_hash": "e" * 64},
            backend_execution_environment_verification={"status": "pass", "actual_manifest_hash": "9" * 64})
        identity.update(extra_identity or {})
        runtime = self.fixture.force if force else self.fixture.velocity
        if not force:
            runtime.state.planned[:] = 2
        if control:
            runtime.state.control_only = True
            runtime.state.planned[:] = 0
            runtime.state.stratum[:] = 0
        capture = PushNumericCapture(PushNumericCaptureOptions(Path(self.fixture.tmp.name) / name, identity, chunk_steps=17),
                                     runtime.state, split_name="public_dev", run_index=1)
        runtime.state.numeric_capture = capture
        return runtime, capture, context

    def finish(self, runtime, capture, context):
        ref = capture.complete(runtime.state.episode_records())
        return build_push_behavior([{"numeric_evidence_ref": ref}], context=context), ref

    def velocity_behavior(self):
        runtime, capture, context = self.capture()
        first = runtime.state.event_times[0]
        def change(step):
            runtime.data.root_ang_vel_w.zero_()
            if step == first + 3:
                runtime.data.root_lin_vel_w[:, 0] = 1
                runtime.data.root_ang_vel_w[:, 0] = 10
            if step == runtime.env.max_episode_length:
                runtime.data.root_lin_vel_w[0, 0] = 1
        self.fixture.velocity_until(runtime.env.max_episode_length, change)
        return self.finish(runtime, capture, context)

    def test_dwell_concurrent_conditions_success_counterexample_and_final_state(self):
        behavior, _ = self.velocity_behavior()
        failed, success = behavior["transcripts"]
        self.assertEqual([row["terminal_label"] for row in (failed, success)], ["not_recovered", "success_recovered"])
        for row in (failed, success):
            event = row["recoveries"][0]["dwell_interruptions"][0]
            self.assertEqual(event["failed_conditions"], ["planar_speed", "angular_speed"])
            self.assertEqual(event["after_anchor_steps"], 3)
            self.assertEqual(event["preceding_dwell_steps"], 2)
            self.assertEqual(row["recoveries"][0]["condition_violation_spans"]["planar_speed"][0]["observed_steps"], 1)
        self.assertTrue(failed["all_required_recoveries_latched"])
        self.assertEqual(failed["final_failed_conditions"], ["planar_speed"])
        self.assertEqual(failed["unstable_steps_after_all_recoveries"], 1)
        comparison = behavior["profile"]["success_failure_comparison"]
        self.assertEqual(comparison["success_recovered"]["interrupted_episodes"], 1)
        self.assertEqual(comparison["not_recovered"]["all_recovered_then_final_unstable"], 1)

    def test_force_last_pulse_anchor_closed_padding_and_partial_coverage(self):
        runtime, capture, context = self.capture(force=True)
        runtime.until(runtime.state.event_times[0])
        runtime.data.root_pos_w[0, 2] = .1
        runtime.step()
        runtime.data.root_pos_w[0, 2] = .4
        runtime.state.reset(runtime.torch.tensor([0]))
        while int(runtime.env.episode_length_buf[0]) < runtime.env.max_episode_length:
            runtime.data.root_pos_w[3, 0] = runtime.state.initial_xy[3, 0] + .0001 * (int(runtime.env.episode_length_buf[0]) + 1)
            runtime.step()
        behavior, _ = self.finish(runtime, capture, context)
        fallen, repeated = behavior["transcripts"][0], behavior["transcripts"][3]
        self.assertEqual(fallen["terminal_label"], "height_collapse")
        self.assertIsNone(fallen["recoveries"][0]["anchor_control_step"])
        self.assertEqual(fallen["valid_steps"], runtime.state.event_times[0] + 1)
        self.assertEqual(fallen["recoveries"][0]["process"]["status"], "no_verified_observed_window")
        unit = repeated["recoveries"][0]
        self.assertEqual(unit["identity"], "sequence")
        self.assertEqual(unit["anchor_control_step"], runtime.state.event_times[1] + runtime.state.pulse_steps)
        self.assertEqual(unit["process"]["recovery_anchor_kind"], "last_pulse_end")
        self.assertEqual(unit["process"]["reference_event_index"], 1)
        self.assertEqual(unit["process"]["injection_start_sample"]["control_step"], runtime.state.event_times[1])
        self.assertEqual(unit["process"]["first_observed_response"]["control_step"], runtime.state.event_times[1] + 1)
        self.assertEqual(unit["process"]["window_first"]["control_step"], unit["anchor_control_step"] + 1)
        self.assertAlmostEqual(unit["process"]["window_net_displacement_world_xy_m"][0], .0001 * runtime.state.window, places=6)

    def test_signed_motion_and_old_profile_missing_direction(self):
        runtime, capture, context = self.capture()
        def change(step):
            runtime.data.root_lin_vel_w[0, :2] = runtime.torch.tensor([.03, -.04])
            runtime.data.root_lin_vel_w[1, :2] = runtime.torch.tensor([-.03, .04])
            runtime.data.root_pos_w[0, :2] = runtime.state.initial_xy[0] + runtime.torch.tensor([.05, -.02])
            runtime.data.root_pos_w[1, :2] = runtime.state.initial_xy[1] + runtime.torch.tensor([-.05, .02])
        self.fixture.velocity_until(runtime.env.max_episode_length, change)
        behavior, ref = self.finish(runtime, capture, context)
        directions = [row["recoveries"][0]["process"]["window_last"] for row in behavior["transcripts"]]
        self.assertLess(directions[0]["velocity_world_xy_mps"][1], 0)
        self.assertGreater(directions[1]["velocity_world_xy_mps"][1], 0)
        old_ref = self.fixture.rewrite_archive(ref, "old-direction", legacy=True)
        old = build_push_behavior([{"numeric_evidence_ref": old_ref}], context=context)
        for key in ("episodes", "terminal_counts", "observed_window_steps", "stable_window_steps", "failed_condition_steps"):
            self.assertEqual(behavior["profile"]["all"][key], old["profile"]["all"][key])
        self.assertEqual(old["profile"]["all"]["window_measurements"]["window_net_displacement_m"],
                         {"observed_units": 0, "median": None})
        self.assertEqual(old["transcripts"][0]["recoveries"][0]["process"]["displacement_status"], "not_collected_in_v1")

    def test_event_relative_displacement_peak_and_unexecuted_second_event(self):
        runtime, capture, context = self.capture()
        first = runtime.state.event_times[0]
        def change(step):
            runtime.data.root_pos_w[1, 0] = runtime.state.initial_xy[1, 0] + .0001 * step
            if step == first + 3:
                runtime.data.root_pos_w[0, 2] = .01
            if step == first + 1:
                runtime.data.root_lin_vel_w[1, 0] = .4
            if step == first + 2:
                runtime.data.root_lin_vel_w[1, 0] = .8
        self.fixture.velocity_until(runtime.env.max_episode_length, change)
        full, _ = self.finish(runtime, capture, context)
        fallen, success = full["transcripts"]
        self.assertEqual(fallen["event_conditions"]["planned_events"], 2)
        self.assertEqual(fallen["event_conditions"]["executed_events"], 1)
        self.assertEqual(fallen["recoveries"][1]["coverage"], "anchor_not_observed")
        self.assertIsNone(fallen["recoveries"][1]["anchor_control_step"])
        process = success["recoveries"][0]["process"]
        self.assertEqual(process["window_first"]["control_step"], first + 1)
        self.assertEqual(process["window_speed_peak"]["control_step"], first + 2)
        self.assertAlmostEqual(process["peak_after_anchor_s"], 2 * runtime.state.dt)
        second = success["recoveries"][1]["process"]
        self.assertAlmostEqual(second["window_net_displacement_world_xy_m"][0], .0001 * runtime.state.window, places=6)
        self.assertGreater(second["window_last"]["reset_relative_position_world_xy_m"][0],
                           second["window_net_displacement_world_xy_m"][0])
        public = project_push_behavior(full)
        # No success with just one executed event: don't silently substitute the
        # otherwise successful two-event episode or drop its planned denominator.
        self.assertEqual(public["selection"]["selected_episodes"], 1)
        self.assertEqual(len(public["representative_transcripts"][0]["recoveries"]), 2)

    def test_selection_uses_failure_coverage_and_diff_refuses_semantics_change(self):
        from true_rl_agent.tasks.push_recovery.behavior import _matched_profiles, _process_representatives
        full, _ = self.velocity_behavior()
        failed, success = full["transcripts"]
        rows = []
        for index, (stratum, original) in enumerate((("low", failed), ("high", failed), ("high", failed), ("high", success))):
            row = copy.deepcopy(original)
            row.update(stratum=stratum, env_id=index)
            row["event_conditions"]["stratum"] = stratum
            rows.append(row)
        selected = _process_representatives(rows, _matched_profiles(rows))
        self.assertEqual([row["stratum"] for row in selected], ["high", "high"])
        self.assertEqual(selected[0]["event_conditions"], selected[1]["event_conditions"])
        old = copy.deepcopy(full)
        old.update(schema_version="true_rl_agent.push_recovery.behavior.v1", rule_version="push_process_v1")
        self.assertEqual(compare_push_behavior(full, old,
            expected_parent_checkpoint_hash=old["binding"]["checkpoint_hash"])["reason"], "behavior_semantics_version_mismatch")

    def test_force_control_does_not_count_reducer_storage_slot_as_failed_recovery(self):
        runtime, capture, context = self.capture(force=True, control=True)
        runtime.until(runtime.env.max_episode_length)
        full, _ = self.finish(runtime, capture, context)
        self.assertEqual(full["profile"]["all"]["terminal_counts"], {"no_push_control": runtime.state.n})
        self.assertEqual(full["profile"]["all"]["recovery_units"], 0)
        self.assertEqual(full["profile"]["all"]["zero_stable_units"], 0)
        self.assertEqual(project_push_behavior(full)["representative_transcripts"][0]["relationship"], "no_failure_in_cohort")

    def test_parent_diff_requires_actual_parent_same_domain_and_complete_coverage(self):
        behavior, _ = self.velocity_behavior()
        parent = copy.deepcopy(behavior)
        parent["binding"]["checkpoint_hash"] = "sha256:" + "1" * 64
        expected = parent["binding"]["checkpoint_hash"]
        diff = compare_push_behavior(behavior, parent, expected_parent_checkpoint_hash=expected)
        self.assertEqual(diff["status"], "available")
        self.assertEqual(compare_push_behavior(project_push_behavior(behavior), project_push_behavior(parent),
                                              expected_parent_checkpoint_hash=expected), diff)
        self.assertTrue(all(value == 0 for value in diff["count_delta"].values()))
        self.assertEqual(compare_push_behavior(behavior, parent, expected_parent_checkpoint_hash="wrong")["status"], "not_comparable")
        parent["comparison_domain"]["execution_source_hash"] = None
        self.assertEqual(compare_push_behavior(behavior, parent, expected_parent_checkpoint_hash=expected)["reason"], "execution_source_unavailable")
        self.assertEqual(compare_push_behavior(behavior, None, expected_parent_checkpoint_hash=expected)["status"], "missing")
        parent = copy.deepcopy(behavior)
        parent["coverage"]["complete_episodes"] -= 1
        self.assertEqual(compare_push_behavior(behavior, parent, expected_parent_checkpoint_hash=parent["binding"]["checkpoint_hash"])["reason"], "incomplete_episode_coverage")

    def test_public_keeps_every_stratum_and_does_not_fabricate_unmatched_success(self):
        runtime, capture, context = self.capture(force=True)
        runtime.until(runtime.env.max_episode_length - 1)
        runtime.data.root_lin_vel_w[0, 0] = 1.
        runtime.step()
        full, _ = self.finish(runtime, capture, context)
        public = project_push_behavior(full)
        # This fixture has one episode per intensity. Its other successes are
        # not legitimate counterexamples for the low-stratum failure.
        self.assertEqual([row["terminal_label"] for row in public["representative_transcripts"]], ["not_recovered"])
        self.assertEqual(public["selection"]["selected_episodes"], 1)
        self.assertEqual(public["profile"]["matched_failure_profiles"][0]["success_status"], "missing_matching_events")
        self.assertLessEqual(canonical_prompt_json_bytes(public), 12288)
        self.assertEqual(project_push_behavior(public), public)
        self.assertNotIn("resolved_protocol_hash", public["binding"])
        self.assertIn("resolved_protocol_hash", public["comparison_domain"])
        for stratum, expected in full["profile"]["by_stratum"].items():
            row = public["profile"]["by_stratum"][stratum]
            self.assertEqual(row["episodes"], expected["episodes"])
            self.assertEqual(row["terminal_counts"], expected["terminal_counts"])
            self.assertEqual(row["events"]["executed"], expected["executed_events"])
            self.assertEqual(row["windows"]["no_anchor"], expected["unobserved_anchor_units"])
        self.assertIn("stop at first latch", public["definitions"]["counting"])

    def test_prompt_replay_and_final_provider_payload_keep_bounded_facts(self):
        from true_rl_agent.agent_core.openai_compatible_provider import (
            OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig,
        )
        behavior, _ = self.velocity_behavior()
        behavior = project_push_behavior(behavior)
        self.assertLessEqual(canonical_prompt_json_bytes(behavior), 12288)
        self.assertGreater(behavior["selection"]["selected_episodes"], 0)
        record = valid_accepted_ledger_record(round_id="behavior", candidate_id="winner", public_score=.75)
        record["public_result"]["metrics"]["public_dev"] = {**_dev(), "behavior": behavior}
        record["public_result"]["metrics"]["public_accept"]["secret"] = "accept-canary-push-behavior"
        config = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2")
        adapter = PushRecoveryPromptAdapter(config)
        original = copy.deepcopy(record)
        projection = adapter.project_prompt_public_history(public_records=[record], initial_anchor_public_evidence=None)
        self.assertEqual(projection.records, adapter.project_prompt_public_history(public_records=projection.records, initial_anchor_public_evidence=None).records)
        builder = PublicContextBuilder(task_prompt_adapter=adapter, acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(), session_genesis_hash=TEST_SESSION_GENESIS_HASH)
        built = builder.build_prompt([record])
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(model="offline", output_dir=Path(tmp)))
            request = provider._chat_completion_request(built.provider_payload)
        delivered = json.loads(request["messages"][1]["content"])
        public = delivered["public_feedback_summary"]["public_dev"]["behavior"]
        self.assertEqual(public["profile"], behavior["profile"])
        self.assertEqual(public["actionability"]["reset_relative_displacement"]["status"], "requires_new_observation")
        self.assertEqual(public["sources"], behavior["sources"])
        self.assertNotIn("manifest_path", json.dumps(delivered))
        self.assertNotIn("accept-canary", json.dumps(delivered))
        minimum = _minimum_core_push_prompt_sections(adapter.build_prompt_sections(public_records=[record], public_context={}))
        self.assertLess(canonical_prompt_json_bytes(minimum), 56 * 1024)
        self.assertEqual(record, original)
        # Exercise the provider serialization after each existing frame boundary,
        # including a verifier rejection with a materialized failed source.
        from tests.unit.test_candidate_authoring_loop import _write_candidate_package
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
                    repaired = json.loads(text)
                    for section in ("public_feedback_summary", "selection_guidance", "exploration_directive"):
                        self.assertEqual(repaired[section], delivered[section])
                    self.assertEqual(repaired["public_feedback_summary"]["public_dev"]["behavior"], public)
                    budget = repaired["authoring_context_frame"]["budget_metadata"]
                    self.assertEqual(budget["actual_prompt_payload_bytes"], len(text.encode()))
                    validate_provider_task_prompt_receipt(budget["provider_task_sections"], prompt_payload=repaired)
                    if inline and kind == "static_repair":
                        self.assertNotIn("public_history", repaired)

    def test_parent_diff_can_omit_optional_pose_without_losing_examples_or_counts(self):
        full, _ = self.velocity_behavior()
        public = project_push_behavior(full)
        self.assertEqual(project_push_behavior(public), public)
        parent = copy.deepcopy(public)
        parent["binding"]["checkpoint_hash"] = "sha256:" + "1" * 64
        public["parent_diff"] = compare_push_behavior(public, parent,
            expected_parent_checkpoint_hash=parent["binding"]["checkpoint_hash"])
        original = copy.deepcopy(public)
        near_budget = canonical_prompt_json_bytes(public) - 19
        compact = project_push_behavior(public, max_bytes=near_budget)
        self.assertLessEqual(canonical_prompt_json_bytes(compact), near_budget)
        self.assertEqual(compact["profile"], original["profile"])
        self.assertEqual(compact["parent_diff"], original["parent_diff"])
        self.assertEqual(compact["selection"], original["selection"])
        self.assertEqual(len(compact["representative_transcripts"]), 2)
        first, second = compact["representative_transcripts"]
        expected_first = copy.deepcopy(original["representative_transcripts"][0])
        expected_first.pop("final_pose")
        expected_first["final_pose_omitted"] = "public_byte_budget"
        self.assertEqual(first, expected_first)
        self.assertEqual(second, original["representative_transcripts"][1])
        self.assertEqual(project_push_behavior(compact, max_bytes=near_budget), compact)
        self.assertEqual(public, original)

    def test_phase_seals_behavior_only_when_produced_and_history_parent_diff_is_idempotent(self):
        from true_rl_agent.evaluation.binding import canonical_json_hash
        from true_rl_agent.evaluation.phases import EvaluationPhase
        from true_rl_agent.tasks.push_recovery.rollout import aggregate_recovery_episodes
        from true_rl_agent.tasks.push_recovery.phased_evaluation import build_push_phase_summary, finalize_push_eval_phase_summary
        from true_rl_agent.tasks.push_recovery.phased_adapter import PushPhasedEvaluationAdapter
        from true_rl_agent.runner.job_identity import JobIdentity
        from true_rl_agent.evaluation.subject import ExperimentSubject
        job_identity = JobIdentity(job_id="behavior-test", task_id="unitree_push_recovery",
            subject=ExperimentSubject.reference("reference"), subject_manifest_hash="1" * 64)
        runtime, capture, context = self.capture(extra_identity=job_identity.to_public_dict())
        self.fixture.velocity_until(runtime.env.max_episode_length)
        rows = runtime.state.episode_records()
        ref = capture.complete(rows)
        summary = {"schema_version": "true_rl_agent.push_recovery.rollout.v2", "status": "pass", "per_episode": rows,
            "aggregate": aggregate_recovery_episodes(rows), "numeric_evidence_ref": ref,
            "run": {"split": "public_dev", "run_index": 1}, "task": {"mechanism": runtime.state.parameters["mechanism"]},
            "environment": {"num_envs": 2}, "evidence": {"resolved_protocol_hash": context["resolved_protocol_hash"], "all_first_episodes_closed": True},
            "execution_cost": {"evaluation_env_count": 2, "vector_env_step_calls": runtime.env.max_episode_length,
                "closed_first_episode_control_steps": sum(row["episode_steps"] for row in rows)}}
        kwargs = dict(phase=EvaluationPhase.PUBLIC_DEV, evaluation_plan_hash=context["evaluation_plan_hash"],
            seed_summaries=[summary], expected_seed_count=1, wall_time_ns=1,
            identity={"program_source_checksum": context["program_source_checksum"]}, execution_device="cpu",
            checkpoint_safety_report={"checkpoint_hash": context["checkpoint_hash"]}, runtime_dynamics_hash=context["runtime_dynamics_hash"])
        old = build_push_phase_summary(**kwargs)
        new = build_push_phase_summary(**kwargs, behavior_context=context)
        self.assertEqual(new["status"], "postprocessing")
        self.assertNotIn("behavior", new["task_evidence"])
        phase_spec = SimpleNamespace(phase=EvaluationPhase.PUBLIC_DEV,
            evaluation_plan_hash=context["evaluation_plan_hash"], program_source_checksum=context["program_source_checksum"],
            execution_checkpoint=lambda: SimpleNamespace(checksum=context["checkpoint_hash"]),
            job_identity=lambda: job_identity,
            evaluation_binding=SimpleNamespace(to_public_dict=lambda: {"evaluation_binding_hash": "e" * 64}))
        new = finalize_push_eval_phase_summary(phase_spec, new, output_dir=Path(self.fixture.tmp.name) / "phase")
        saved_bytes = (Path(self.fixture.tmp.name) / "phase" / "behavior_full.json").read_bytes()
        saved_full = json.loads(saved_bytes)
        self.assertEqual(new["task_evidence"]["behavior_full"]["size_bytes"], len(saved_bytes))
        self.assertEqual(new["task_evidence"]["behavior_full"]["sha256"], canonical_json_hash(saved_full))
        self.assertEqual(new["task_evidence"]["behavior"]["full_derivation_sha256"], canonical_json_hash(saved_full))
        self.assertEqual(saved_full["source_bindings"][0]["actual_run_index"], 1)
        self.assertNotIn("behavior", old["task_evidence"])
        self.assertEqual(new["task_evidence_hash"], canonical_json_hash(new["task_evidence"]))
        # Execute the production producer body without importing Isaac Sim.
        # JobIdentity's public mapping intentionally has no program lineage;
        # the validated CLI field is also the numeric archive's source.
        script = Path(__file__).resolve().parents[2] / "scripts" / "isaaclab_rsl_rl_eval.py"
        functions = [node for node in ast.parse(script.read_text(encoding="utf-8")).body
                     if isinstance(node, ast.FunctionDef) and node.name in {"run_public_eval", "eval_summary_binding"}]
        namespace = {"argparse": argparse, "Any": object, "time": time,
            "evaluation_protocol_from_args": lambda args: SimpleNamespace(public_seed_count=1,
                public_accept_seed_count=1, sealed_validation_seed_count=1, to_public_dict=lambda: {}),
            "evaluation_binding_from_args": lambda args: SimpleNamespace(to_public_dict=lambda: {"evaluation_binding_hash": "e" * 64}),
            "public_runtime_dynamics": lambda config: {"runtime_dynamics_hash": context["runtime_dynamics_hash"]},
            "run_rollout": lambda *args, **kwargs: summary}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(script), "exec"), namespace)
        config = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2")
        manifest = resolve_push_cell_manifest(cell_variant_id=config.cell_variant_id,
            config=config, plugin=PushRecoveryTaskPlugin(config))
        args = SimpleNamespace(evaluation_phase="public_dev", evaluation_plan_hash=context["evaluation_plan_hash"],
            resolved_cell_manifest=manifest, program_source_checksum=context["program_source_checksum"], device="cpu",
            backend_execution_environment_verification={"status": "pass", "actual_manifest_hash": "9" * 64},
            evaluation_job_identity=job_identity)
        self.assertNotIn("program_source_checksum", namespace["eval_summary_binding"](args))
        with patch("true_rl_agent.runner.source_snapshot.running_project_source_snapshot", return_value={"files": {}}):
            produced = namespace["run_public_eval"](None, {}, args,
                checkpoint_safety_report={"checkpoint_hash": context["checkpoint_hash"]})
        self.assertEqual(produced["status"], "postprocessing")
        produced = finalize_push_eval_phase_summary(phase_spec, produced, output_dir=Path(self.fixture.tmp.name) / "produced")
        self.assertEqual(produced["status"], "pass")
        self.assertEqual(produced["task_evidence"]["behavior"]["binding"]["program_source_checksum"], context["program_source_checksum"])
        self.assertEqual(produced["task_evidence_hash"], canonical_json_hash(produced["task_evidence"]))
        config = replace(PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2"), public_seed_count=1)
        adapter = PushPhasedEvaluationAdapter(config, "0" * 64)
        phase = SimpleNamespace(task_id="unitree_push_recovery", phase=EvaluationPhase.PUBLIC_DEV,
            eval_protocol_hash=config.evaluation_protocol().protocol_hash, task_evidence=new["task_evidence"], checkpoint_hash=context["checkpoint_hash"])
        public = adapter.project_public_phase_summary(phase_result=phase)
        self.assertNotIn("behavior_full", public)
        self.assertNotIn("transcripts", json.dumps(public))
        invalid = copy.deepcopy(phase)
        invalid.task_evidence["behavior_full"]["sha256"] = None
        invalid.task_evidence["behavior"]["full_derivation_sha256"] = None
        with self.assertRaisesRegex(ValueError, "sealed full derivation"):
            adapter.project_public_phase_summary(phase_result=invalid)
        before = copy.deepcopy(new)
        prompt_adapter = PushRecoveryPromptAdapter(config)
        record = {"generation": {"parent_current_best": {"checkpoint_hash": context["checkpoint_hash"]},
            "parent_public_dev": public, "trials": [{"trial_id": "measured", "trial_status": "not_selected",
            "candidate": {"candidate_id": "candidate"}, "train": {}, "public_dev": {"public_summary": public}}]}}
        projected = prompt_adapter.project_prompt_public_history(public_records=[record], initial_anchor_public_evidence=None)
        derived = projected.records[0]["generation"]["trials"][0]["public_dev"]["public_summary"]["behavior"]
        self.assertEqual(derived["parent_diff"]["status"], "available")
        again = prompt_adapter.project_prompt_public_history(public_records=projected.records, initial_anchor_public_evidence=None)
        self.assertEqual(projected.records, again.records)
        feedback = prompt_adapter.build_prompt_sections(public_records=again.records, public_context={})["public_feedback_summary"]
        self.assertEqual(feedback["public_dev"]["behavior"]["parent_diff"], derived["parent_diff"])
        self.assertEqual(new, before)
        phase.task_evidence = old["task_evidence"]
        self.assertNotIn("behavior", adapter.project_public_phase_summary(phase_result=phase))


if __name__ == "__main__":
    unittest.main()
