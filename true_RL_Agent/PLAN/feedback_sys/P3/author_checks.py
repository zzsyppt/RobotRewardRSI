"""CPU checks for the PLAN experiment; real saved qualification, fake transport."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import author_experiment as exp


def qualified_spec():
    q = ROOT / "PLAN/2new_world/P3/results/cohort_alignment_20260909/go2_12dof_qualification"
    train = ROOT / "PLAN/2new_world/P2/results/audit3_go2_full_train/runner/audit3_go2_full_train/train/summary.json"
    def ref(path):
        return {"path": str(path), "sha256": exp.digest(path)}
    return {"task": "stair", "mode": "existing_stair_qualification", "composition_values": {"cell_variant_id": "stair.go2_12dof.unified_v3"},
        "reference_program_source_checksum": "sha256:10b1a34218d7fbb3ed124a6913e6d45d1462e5508d062510f31a3d1d7ac272cb",
        "checkpoint_hash": "sha256:7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c",
        "qualification_result": ref(q / "result.json"), "qualification_assembly": ref(q / "assembly.json"), "training_summary": ref(train)}


class ExperimentChecks(unittest.TestCase):
    def test_reference_locator_preserves_visibility_and_does_not_grade_meaning(self):
        from blind_export import reference_locations
        seen = {"public_feedback_summary": {"behavior": {"representative_transcripts": [
            {"source": 0, "env_id": 3, "terminal_label": "not_recovered"}]}},
            "latest_failed_source": "r1.e999 invented_claim"}
        truth = {"episodes": [{"episode_id": "r1.e48", "terminal_label": "timeout", "count": 95}],
            "transcripts": [{"source": 0, "env_id": 3, "terminal_label": "not_recovered"}]}
        report = reference_locations("s0.e3 not_recovered; r1.e48; r1.e999 invented_claim 95%", seen, truth)
        self.assertEqual(report["seen_feedback"]["s0.e3"]["total_matches"], 1)
        self.assertEqual(report["seen_feedback"]["r1.e48"]["total_matches"], 0)
        self.assertEqual(report["full_truth"]["r1.e48"]["total_matches"], 1)
        self.assertEqual(report["seen_feedback"]["r1.e999"]["total_matches"], 0)
        self.assertEqual(report["numeric_claims"], "NOT_GRADED_HUMAN_REQUIRED")
        self.assertEqual(report["numeric_exact_locations"]["full_truth"]["95"]["total_matches"], 1)

    def test_balanced_schedule_and_correlated_cases(self):
        stage = exp.read(Path(__file__).with_name("author_stage.json"))
        cases = exp.read(Path(__file__).with_name("case_design.json"))["cases"]
        blocks = exp.schedule(stage, cases)
        self.assertEqual(len(blocks), 24)
        self.assertEqual(blocks, exp.schedule(stage, cases))
        for case in cases:
            pair = [b for b in blocks if b["case_id"] == case["case_id"]]
            self.assertEqual(pair[0]["arm_order"], list(reversed(pair[1]["arm_order"])))
        self.assertEqual(len({c["cluster_id"] for c in cases}), 2)

    def test_bound_source_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.json"
            exp.write(path, {"fact": 1})
            ref = {"path": str(path), "sha256": exp.digest(path)}
            self.assertEqual(exp.bound_read(ref, Path(temp)), {"fact": 1})
            exp.write(path, {"fact": 2})
            with self.assertRaisesRegex(ValueError, "changed"):
                exp.bound_read(ref, Path(temp))

    def test_fairness_rejects_score_and_source_permission_tampering(self):
        a = {"public_feedback_summary": {"score": 0.5}, "authoring_context_frame": {
            "source_hashes": ["parent"], "frame_contract": {"source_followup_enabled": True},
            "budget_metadata": {"hard_limits": {"max": 40000}, "estimated_prompt_bytes": 10}}}
        t = copy.deepcopy(a)
        t["public_feedback_summary"]["behavior"] = {"measured": True}
        t["authoring_context_frame"]["budget_metadata"]["estimated_prompt_bytes"] = 20
        exp.assert_arm_fairness("stair", a, t)
        t["public_feedback_summary"]["score"] = 0.6
        with self.assertRaisesRegex(ValueError, "outside"):
            exp.assert_arm_fairness("stair", a, t)
        t["public_feedback_summary"]["score"] = 0.5
        t["authoring_context_frame"]["frame_contract"]["source_followup_enabled"] = False
        with self.assertRaisesRegex(ValueError, "outside"):
            exp.assert_arm_fairness("stair", a, t)
        t["authoring_context_frame"]["frame_contract"]["source_followup_enabled"] = True
        a["public_feedback_summary"]["feedback_entries"] = [{"source_public_result_hash": "source-A"}]
        t["public_feedback_summary"]["feedback_entries"] = [{"source_public_result_hash": "source-T"}]
        exp.assert_arm_fairness("stair", a, t, projection_hashes={"A": "source-A", "T": "source-T"})
        t["public_feedback_summary"]["feedback_entries"][0]["source_public_result_hash"] = "tampered"
        with self.assertRaisesRegex(ValueError, "canonical source"):
            exp.assert_arm_fairness("stair", a, t, projection_hashes={"A": "source-A", "T": "source-T"})

    def test_existing_push_velocity_qualification_keeps_original_domain(self):
        from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
        from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
        path = ROOT / "PLAN/2new_world/P6/g1_qualified_01/formal/session_genesis.json"
        anchor = exp.read(path)["anchor"]
        cfg = PushRecoveryTaskConfig.from_manifest(load_resolved_cell_manifest(path.with_name("resolved_cell_manifest.json")))
        spec = {"task": "push", "mode": "existing_reference_genesis", "genesis": {"path": str(path), "sha256": exp.digest(path)},
            "checkpoint_hash": anchor["checkpoint_hash"], "reference_program_source_checksum": anchor["reference_manifest"]["source_checksum"],
            "composition_values": {"cell_variant_id": "push.g1_37dof.recovery_v2", "push_protocol": cfg.push_protocol.to_dict(),
                "reference_program_id": cfg.reference_program_id, "reference_admission_policy": None, "eval_base_seed": 1000000000}}
        with tempfile.TemporaryDirectory() as temp:
            _, best, *_ = exp.reference_context(spec, root=ROOT, base=ROOT, output=Path(temp))
            self.assertEqual(best.evaluation_binding_hash, "fa86907f5ca4f7b7dd2891640495c7e925ab89f5cf9eb92cfa63f9d2ab0c8428")
            spec["composition_values"]["eval_base_seed"] = 1500000000
            with self.assertRaisesRegex(ValueError, "domains differ"):
                exp.reference_context(spec, root=ROOT, base=ROOT, output=Path(temp) / "wrong")

    def test_real_reference_builder_static_loop_and_budget_journal(self):
        from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
        from true_rl_agent.agent_core.generation_provider_budget import CostGuardedProvider
        from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig
        from true_rl_agent.memory.context_builder import PublicContextBuilder
        from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
        from true_rl_agent.prompting import assert_public_only_prompt_payload
        from true_rl_agent.prompt_artifacts import verify_authoring_evidence
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            comp, best, evidence, source, namespace, _ = exp.reference_context(qualified_spec(), root=ROOT, base=ROOT, output=out)
            original = comp.task_plugin.prompt_adapter()
            base_sections = original.build_prompt_sections(public_records=(), public_context={"initial_anchor_public_evidence": evidence})
            view = exp.FrozenFeedbackAdapter(original, base_sections["public_feedback_summary"], "Focus on the public-dev c1_conditional_frontier stratum; use only visible evidence.")
            builder = PublicContextBuilder(task_memory_adapter=comp.task_memory_adapter, task_prompt_adapter=view,
                phased_evaluation_adapter=comp.task_plugin.phased_evaluation_adapter(),
                acceptance_protocol=comp.task_plugin.task_spec.acceptance_protocol, initial_current_best=best,
                session_genesis_hash=namespace,
                candidate_capability_profile=comp.task_plugin.task_spec.candidate_capability_profile,
                initial_reference_program_source=source, initial_anchor_public_evidence=evidence)
            readiness = comp.authoring_capability_readiness.to_prompt_mapping()
            payload = builder.build_prompt([], authoring_capability_readiness=readiness,
                candidate_verifier_prompt_rules=comp.candidate_admission_policy.authoring_prompt_rules()).provider_payload
            assert_public_only_prompt_payload(payload)
            self.assertEqual(select_authoring_anchor_source(payload)[0], source)
            self.assertEqual(best.checkpoint_hash, qualified_spec()["checkpoint_hash"])
            candidate = source.replace("return inputs.projected_gravity[:, :2].square().sum(dim=1)",
                "return inputs.projected_gravity[:, :2].square().sum(dim=1) * (1.0 + inputs.base_lin_vel[:, 0].abs())")
            self.assertNotEqual(candidate, source)
            class FakeTransport:
                calls = 0
                def post_json(self, **kwargs):
                    self.calls += 1
                    return {"model": "fake-offline-transport", "usage": {"prompt_tokens": 15000, "completion_tokens": 1000, "total_tokens": 16000},
                        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"candidate_id": "p3_cpu_contract_probe",
                            "declared_changed_components": ["reward_callable"], "components_py": candidate})}}]}
            transport = FakeTransport()
            provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(model="offline-fixture", output_dir=out / "candidates",
                api_key="offline-test-only", base_url="https://example.invalid"), transport=transport)
            generation = "p3-cpu-generation-001"
            guarded = CostGuardedProvider(delegate=provider, output_dir=out, generation_id=generation, lane_id="lane-a",
                provider_caps={"attempts_per_lane_per_generation": 5, "input_units_per_attempt_reserve": 40000, "output_units_per_attempt_cap": 32768}, max_generations=24)
            result = CandidateAuthoringLoop(context_builder=builder, provider=guarded, candidate_admission_policy=comp.candidate_admission_policy,
                round_dir=out / "authoring" / generation / "lane-a", max_attempts=5,
                component_smoke_adapter=comp.task_plugin.component_smoke_adapter(), authoring_capability_readiness=readiness
            ).run_round(round_id=generation + "-lane-a", records=[])
            self.assertIsNotNone(result.candidate, result.static_failure)
            self.assertEqual(transport.calls, 1)
            self.assertEqual(result.parent_program_source_checksum, qualified_spec()["reference_program_source_checksum"])
            verify_authoring_evidence(result.authoring_evidence_ref, expected_round_id=generation + "-lane-a",
                expected_selected_prompt=result.prompt_payload_ref)
            accounting = exp.usage_report(out, exp.read(Path(__file__).with_name("author_stage.json")))
            self.assertEqual(accounting["attempts"], 1)
            self.assertEqual(accounting["response_models"], ["fake-offline-transport"])
            self.assertAlmostEqual(accounting["usd_peak_actual_or_reserved"], 0.0057)
            from blind_export import reward_checks
            facts = reward_checks(candidate, payload, result.attempt_records)
            self.assertEqual(facts["status"], "projected")
            self.assertEqual(facts["after"]["weight_sign"], "negative")
            self.assertEqual(facts["finite_fixture_sign_arithmetic"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
