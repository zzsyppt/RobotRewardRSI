"""P1c source binding, atomic UTF-8 budgets and versioned delivery boundaries."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from tests.unit import test_stair_trajectory_production as production_fixture
from true_rl_agent.tasks.stair_climbing.trajectory_production import project_production_report, author_behavior_payload
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import report_bytes, SENTENCE_SECTIONS
from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior, compare_stair_behavior
from true_rl_agent.tasks.stair_climbing.trajectory_report import (
    PUBLIC_SCHEMA, MAX_PUBLIC_BYTES, validate_stair_trajectory_report,
    require_trajectory_trial_binding, require_trajectory_phase_binding,
)
from true_rl_agent.tasks.stair_climbing.prompt_adapter import _latest_behavior_feedback, _reference_behavior_feedback


def resign(report):
    report["report_projection"].pop("content_hash", None)
    report["report_projection"]["content_hash"] = canonical_json_hash(report)
    return report


class StairTrajectoryReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        production_fixture.ProductionTrajectoryTests.setUpClass()
        cls.summary = production_fixture.ProductionTrajectoryTests.summary
        cls.domain = deepcopy(production_fixture.ProductionTrajectoryTests.domain)
        cls.domain["evaluation_job_id"] = "generation-1-lane-a-eval-public-dev"
        cls.full = cls.project()
        cls.public = cls.full["public_report"]

    @classmethod
    def tearDownClass(cls):
        production_fixture.ProductionTrajectoryTests.tearDownClass()

    @classmethod
    def project(cls, budget=MAX_PUBLIC_BYTES):
        return project_production_report(cls.domain, cls.summary, max_bytes=budget)

    def trial(self, report=None, name="generation-1-lane-a"):
        report = deepcopy(report or self.public)
        return {"trial_id": name, "lane_id": name.rsplit("-", 1)[-1], "public_dev": {
            "checkpoint_hash": report["domain"]["checkpoint_hash"],
            "evaluation_plan_hash": report["domain"]["evaluation_plan_hash"],
            "public_summary": {"behavior": report}}}

    def test_exact_final_bytes_and_one_byte_below_required_fail(self):
        from true_rl_agent.evaluation.process_report import ReportBudgetExceeded
        from unittest.mock import patch
        from true_rl_agent.evaluation.process_report import select_whole_claims
        feedback = {"source":{"role":"当前编辑基准"},"profile":self.public}
        with self.assertRaises(ReportBudgetExceeded) as caught:
            author_behavior_payload(feedback,max_bytes=1)
        n = caught.exception.required_bytes
        tight = author_behavior_payload(feedback,max_bytes=n)
        self.assertEqual(report_bytes(tight), n)
        with self.assertRaisesRegex(ValueError, "mandatory facts exceed"):
            author_behavior_payload(feedback,max_bytes=n-1)
        with patch('true_rl_agent.tasks.stair_climbing.trajectory_production.select_whole_claims',wraps=select_whole_claims) as select:
            full = self.project()
            self.assertEqual(select.call_count,0)
            shown = author_behavior_payload({**feedback,"profile":json.loads(json.dumps(full['public_report'],sort_keys=True))})
            self.assertEqual(select.call_count,1)
        self.assertEqual(list(shown['profile']['report']),[*SENTENCE_SECTIONS,'actionability'])




    def test_source_sibling_job_checkpoint_and_plan_mismatches_rejected(self):
        require_trajectory_trial_binding(self.public, self.trial())
        for field in ("trial_id", "checkpoint_hash", "evaluation_plan_hash"):
            trial = self.trial()
            if field == "trial_id":
                trial[field] = "generation-1-lane-b"
            else:
                trial["public_dev"][field] = "x"*64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "sibling/trial/phase"):
                require_trajectory_trial_binding(self.public, trial)
        phase = dict(job_id=self.domain["evaluation_job_id"], checkpoint_hash=self.domain["checkpoint_hash"],
                     evaluation_plan_hash=self.domain["evaluation_plan_hash"], evaluation_binding_hash=self.domain["evaluation_binding_hash"],
                     archive_hash=self.domain["archive_hash"], seed_namespace=self.domain["seed_namespace"])
        require_trajectory_phase_binding(self.public, **phase)
        for field in phase:
            bad = {**phase, field: "wrong"}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "phase source binding"):
                require_trajectory_phase_binding(self.public, **bad)

    def test_wrong_unselected_sibling_is_not_hidden_by_last_lane(self):
        trial = self.trial()
        bad = self.trial(name="generation-1-lane-b")
        with self.assertRaisesRegex(ValueError, "sibling/trial/phase"):
            _latest_behavior_feedback([{"generation": {"trials": [bad, trial]}}])

    def test_editing_baseline_survives_later_rejected_trials_and_accepts_bound_parent_observation(self):
        first = self.trial()
        first["candidate"] = {"candidate_id":"accepted-a","program_source_checksum":"1"*64}
        best = {"anchor_kind":"candidate","anchor_id":"accepted-a","program_source_checksum":"1"*64,
                "accepted_round_id":first["trial_id"], "checkpoint_hash":self.domain["checkpoint_hash"]}
        other = deepcopy(self.public)
        other["domain"].update(evaluation_job_id="generation-2-lane-b-eval-public-dev", checkpoint_hash="another-checkpoint")
        resign(other)
        second = self.trial(other, "generation-2-lane-b")
        second["candidate"] = dict(first["candidate"])  # Same source/ID cannot substitute another checkpoint/trial.
        records = [{"generation":{"generation_index":1,"trials":[first]}},
                   {"generation":{"generation_index":2,"trials":[second]}}]
        result = _latest_behavior_feedback(records,current_best=best)
        self.assertEqual(result["source_trial_id"],first["trial_id"])
        self.assertEqual(result["profile"],self.public)
        observed = deepcopy(self.public)
        observed["domain"]["evaluation_job_id"] = "new-incumbent-public-dev-observation"
        resign(observed)
        records[-1]["generation"].update(parent_current_best=best,
            parent_public_dev={"behavior":observed})
        result = _latest_behavior_feedback(records,current_best=best)
        self.assertEqual(result["profile"],observed)
        self.assertEqual(result["diff"]["reason"],"missing_explicit_parent")
        records[-1]["generation"]["parent_current_best"] = {**best,"accepted_round_id":"another-trial"}
        self.assertEqual(_latest_behavior_feedback(records,current_best=best)["profile"],self.public)
        with self.assertRaisesRegex(ValueError,'predecessor identity'):
            _latest_behavior_feedback(records,current_best={**best,"checkpoint_hash":"unmeasured"})

    def test_paths_sealed_text_and_changed_sentence_rejected_before_scrubbing(self):
        for text in ("C:\\private\\original.npz", "/private/archive/data.npz", "sealed_validation"):
            bad = deepcopy(self.public)
            bad["report"]["gaps"][0]["text"] = text
            resign(bad)
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_stair_trajectory_report(bad)
        bad = deepcopy(self.public)
        bad["report"]["process"][0]["text"] = "lost the denominator"
        with self.assertRaisesRegex(ValueError, "content hash"):
            project_stair_behavior(bad)

    def test_diagnostic_source_not_promoted_to_reference_or_round_trial(self):
        observation = {"status": "pass", "metrics": {"behavior": self.public}}
        binding = {"job_id": self.domain["evaluation_job_id"], **{k: self.domain[k] for k in
            ("checkpoint_hash", "evaluation_plan_hash", "evaluation_binding_hash", "archive_hash", "seed_namespace")}}
        anchor = {"anchor_kind": "reference_program", "public_result_hash": canonical_json_hash(observation),
            "public_result": observation, "trajectory_phase_binding": binding}
        result = _reference_behavior_feedback({"initial_anchor_public_evidence": anchor})
        self.assertEqual(result["profile"], self.public)
        self.assertEqual(result["source"]["role"], "初始参考观测")
        for field in binding:
            bad = deepcopy(anchor)
            bad["trajectory_phase_binding"][field] = "unrelated-source"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "phase source binding"):
                _reference_behavior_feedback({"initial_anchor_public_evidence": bad})
        bad = deepcopy(anchor)
        bad["public_result_hash"] = "a"*64
        with self.assertRaisesRegex(ValueError, "result hash mismatch"):
            _reference_behavior_feedback({"initial_anchor_public_evidence": bad})
        bad = deepcopy(anchor)
        bad.pop("trajectory_phase_binding")
        with self.assertRaisesRegex(ValueError, "explicit phase binding"):
            _reference_behavior_feedback({"initial_anchor_public_evidence": bad})
        with self.assertRaisesRegex(ValueError, "explicit phase"):
            _latest_behavior_feedback([{"round_id": "unbound-round", "public_result": {"metrics": {"behavior": self.public}}}])


    def test_actual_prompt_reference_across_generations_and_final_visible_budget(self):
        from dataclasses import replace
        from types import SimpleNamespace
        from tests.unit.test_stair_phase4_contract import StairPhase4ContractTests
        from true_rl_agent.tasks.stair_climbing.prompt_adapter import StairClimbingPromptAdapter
        from true_rl_agent.tasks.stair_climbing.unified_scoring import build_reference_objective_family_mask
        from true_rl_agent.evaluation.subject import ExperimentSubject
        from true_rl_agent.prompting.payload import build_prompt_payload, serialize_prompt_payload
        fixture = production_fixture.ProductionTrajectoryTests.fixture
        config = replace(fixture.config, resolved_cell_task_protocol_hash=fixture.resolved.resolved_cell_task_protocol_hash)
        summaries = SimpleNamespace(config=config)
        make = StairPhase4ContractTests._summary_for_split.__func__
        progress = {cell.stratum_id:1. for cell in config.ordered_evaluation_suite.private_cell_assignment("public_dev")}
        successes = {key:round(config.ordered_evaluation_suite.episodes_per_seed_per_cell*2/3) for key in progress}
        public_dev = make(summaries,"public_dev",progress=progress,success_counts=successes)
        mask = build_reference_objective_family_mask(suite=config.ordered_evaluation_suite,
            public_accept_summary=make(summaries,"public_accept",progress=progress,success_counts=successes))
        adapter = StairClimbingPromptAdapter(config,resolved_cell_manifest=fixture.resolved,
            objective_family_mask_provider=lambda:mask)
        binding = {"job_id":self.domain["evaluation_job_id"], **{k:self.domain[k] for k in
            ("checkpoint_hash","evaluation_plan_hash","evaluation_binding_hash","archive_hash","seed_namespace")}}
        observation = {"status":"pass","metrics":{"behavior":self.public,"public_dev":public_dev,
            "trajectory_phase_binding":binding,"evaluation_subject":ExperimentSubject.reference("initial-ref").to_public_dict(),
            "program_source_checksum":"1"*64}}
        context = {"current_best_identity":{"anchor_kind":"reference","anchor_id":"initial-ref",
            "checkpoint_hash":self.domain["checkpoint_hash"],"program_source_checksum":"1"*64},
            "initial_anchor_public_evidence":{"anchor_kind":"reference_program","public_result":observation,
                "public_result_hash":canonical_json_hash(observation)}}
        runtime = {"step_calls":8,"reward_term_count":1,"reward_term_calls":8,
            "component_activation_records":[{"name":"fixture_reward","component_kind":"reward_term",
                "last_weighted_mean":-.125,"max_abs_weighted_contribution":3.5,
                "last_sample_common_step":7,"sampled_env_count":16}]}
        context['initial_anchor_public_evidence']['training_runtime'] = {
            'evaluation_subject':ExperimentSubject.reference('initial-ref').to_public_dict(),
            'checkpoint_hash':self.domain['checkpoint_hash'],'program_source_checksum':'1'*64,
            'runtime_activation_summary':runtime}
        reference_sections = adapter.build_prompt_sections(public_records=[],public_context=context)
        reference_payload = json.loads(serialize_prompt_payload(build_prompt_payload(public_context={'task_prompt_sections':reference_sections})))
        # A renderer fault after source validation must not erase measured feedback.
        cause = ArithmeticError("column mapping unavailable: " + "x" * 600)
        def broken_render(*args, **kwargs):
            raise RuntimeError("author renderer failed") from cause
        with patch("true_rl_agent.tasks.stair_climbing.trajectory_production.select_whole_claims", side_effect=broken_render) as render:
            failed_sections = adapter.build_prompt_sections(public_records=[], public_context=context)
            failed_payload = json.loads(serialize_prompt_payload(build_prompt_payload(public_context={"task_prompt_sections":failed_sections})))
            failed_behavior = failed_payload["public_feedback_summary"]["behavior"]
            status = failed_behavior["description_status"]
            self.assertEqual(status["stage"], "author_rendering")
            self.assertEqual(status["error_type"], "ArithmeticError")
            self.assertEqual(status["error_message"], str(cause)[:512])
            self.assertEqual(failed_behavior["source"], reference_payload["public_feedback_summary"]["behavior"]["source"])
            self.assertEqual(failed_payload["runtime_feedback_summary"], reference_payload["runtime_feedback_summary"])
            self.assertEqual({k:v for k,v in failed_payload["public_feedback_summary"].items() if k != "behavior"},
                             {k:v for k,v in reference_payload["public_feedback_summary"].items() if k != "behavior"})
            from true_rl_agent.prompt_artifacts import publish_prompt_payload, verify_prompt_payload
            ref = publish_prompt_payload(fixture.root / "render-failure-prompt.json", failed_payload)
            self.assertEqual(verify_prompt_payload(ref), failed_payload)
            render.reset_mock()
            bad = deepcopy(context)
            anchor = bad["initial_anchor_public_evidence"]
            anchor["public_result"]["metrics"]["trajectory_phase_binding"]["checkpoint_hash"] = "wrong-source"
            anchor["public_result_hash"] = canonical_json_hash(anchor["public_result"])
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                adapter.build_prompt_sections(public_records=[], public_context=bad)
            render.assert_not_called()
        context_path = fixture.root/'reference-context.json'
        context_path.write_text(json.dumps(context,sort_keys=True),encoding='utf-8')
        restored_sections = adapter.build_prompt_sections(public_records=[],public_context=json.loads(context_path.read_bytes()))
        restored_payload = json.loads(serialize_prompt_payload(build_prompt_payload(public_context={'task_prompt_sections':restored_sections})))
        self.assertEqual(reference_payload,restored_payload)
        reference_runtime = reference_payload['runtime_feedback_summary']
        self.assertEqual(reference_runtime['source_kind'],'reference_anchor')
        self.assertEqual(reference_runtime['source_candidate_id'],'initial-ref')
        item = reference_runtime['latest_runtime_activation']['components'][0]
        self.assertEqual((item['last_mean'],item['last_sample_common_step'],item['sampled_env_count']),(-.125,7,16))
        self.assertIn('不是训练全程均值',reference_runtime['scope'])
        self.assertTrue(any(e.get('source_kind')=='reference_anchor' for e in reference_payload['public_feedback_summary']['feedback_entries']))
        self.assertFalse(any('J_primary=(2*J_OOD+J_ID)/3' in s for s in reference_payload['selection_guidance']))
        records = [{"generation":{"generation_index":2,"trials":[{"trial_id":"g2-b","lane_id":"lane-b",
            "trial_status":"candidate_failed","status_reason":"train_incomplete",
            "candidate":{"candidate_id":"unfinished"},"cost":{"charged_attempts":1,"gpu_hours":.2},
            "runtime_activation_summary":runtime}]}}]
        before_records = deepcopy(records)
        sections = adapter.build_prompt_sections(public_records=records,public_context=context)
        delivered = json.loads(serialize_prompt_payload(build_prompt_payload(public_context={"task_prompt_sections":sections})))
        behavior = delivered["public_feedback_summary"]["behavior"]
        self.assertEqual(behavior["source"]["anchor_kind"],"reference")
        self.assertEqual(behavior["secondary"][0]["reason"],"train_incomplete")
        self.assertEqual(behavior['secondary'][0]['comparison_status'],'not_evaluated')
        self.assertIn('尚无策略效果结论',behavior['secondary'][0]['text'])
        self.assertEqual(records,before_records)
        from true_rl_agent.agent_core.experiment_memory import _trial_outcome
        self.assertEqual(_trial_outcome(records[0]['generation']['trials'][0])[4],'execution_incomplete')
        missing_runtime = deepcopy(context)
        missing_runtime['initial_anchor_public_evidence']['training_runtime']['checkpoint_hash']='other-checkpoint'
        no_borrow = adapter.build_prompt_sections(public_records=records,public_context=missing_runtime)['runtime_feedback_summary']
        self.assertEqual(no_borrow['status'],'not_available')
        self.assertEqual(no_borrow['recent_trials'][0]['source_candidate_id'],'unfinished')
        self.assertEqual(set(behavior["profile"]),{"kind","report"})
        self.assertLessEqual(len(json.dumps(behavior,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()),10240)
        wrong = deepcopy(context)
        wrong["current_best_identity"]["anchor_id"] = "another-ref"
        self.assertIsNone(_reference_behavior_feedback(wrong))
        tight = deepcopy(context)
        anchor = tight["initial_anchor_public_evidence"]
        anchor["public_result"]["metrics"]["behavior"] = self.project(128)["public_report"]
        anchor["public_result_hash"] = canonical_json_hash(anchor["public_result"])
        bounded = adapter.build_prompt_sections(public_records=records,public_context=tight)["public_feedback_summary"]["behavior"]
        self.assertEqual(bounded["description_status"]["reason"],"budget_exceeded")
        self.assertEqual(bounded["description_status"]["max_bytes"],128)
        self.assertGreater(bounded["description_status"]["required_bytes"],128)
        self.assertEqual(bounded["description_status"]["error_type"],"ReportBudgetExceeded")
        self.assertEqual(bounded["source"]["anchor_kind"],"reference")
        # The complete source + nonempty diff + secondary is charged, not the internal job/digests.
        feedback = {"source":behavior["source"],"secondary":behavior["secondary"],"profile":self.public,
            "diff":{"status":"available","text":"对照", "composition":{"parent":{"episodes":5},"candidate":{"episodes":7}},
                    "rows":[{"text":"RL接触占比20→60%，差40个百分点。"}]}}
        visible = author_behavior_payload(feedback)
        n = len(json.dumps(visible,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())
        self.assertLessEqual(n,10240)
        renamed = deepcopy(feedback)
        renamed["profile"]["domain"]["evaluation_job_id"] = "very-long-internal-job-"*50
        resign(renamed["profile"])
        self.assertEqual(author_behavior_payload(renamed),visible)

        # Accepted candidate through the same adapter and final serializer, with measured operands.
        from tests.unit.trajectory_fixtures import sample_episode
        from true_rl_agent.evaluation.trajectory_translation.measurement import measure_behavior
        from true_rl_agent.tasks.stair_climbing.behavior import comparison_operands
        parent = deepcopy(self.public)
        parent["domain"]["checkpoint_hash"] = "predecessor-checkpoint"
        resign(parent)
        operands = []
        for speed,card in ((1.,parent),(2.,self.public)):
            ep = sample_episode(80)
            ep.task.update(cell_id="same-geometry",stratum_id="fixture")
            ep.arrays["base_pos_w"][:,0] = ep.times*speed
            operand = comparison_operands([measure_behavior(ep)])
            operand["domain"] = card["domain"]
            operands.append(operand)
        trial = self.trial()
        trial.update(lane_id="lane-a",trial_status="accepted",candidate={"candidate_id":"accepted-a","program_source_checksum":"1"*64})
        trial['runtime_activation_summary'] = deepcopy(runtime)
        trial["public_dev"]["public_summary"] = {**public_dev,"behavior":self.public,"behavior_comparison":operands[1]}
        best = {"anchor_kind":"candidate","anchor_id":"accepted-a","accepted_round_id":trial["trial_id"],
                "checkpoint_hash":self.domain["checkpoint_hash"],"program_source_checksum":"1"*64}
        records = [{"generation":{"generation_index":1,"parent_current_best":{"anchor_kind":"reference",
            "anchor_id":"initial-ref","checkpoint_hash":parent["domain"]["checkpoint_hash"]},
            "parent_public_dev":{**public_dev,"behavior":parent,"behavior_comparison":operands[0]},"trials":[trial]}}]
        sections = adapter.build_prompt_sections(public_records=records,public_context={"current_best_identity":best})
        delivered = json.loads(serialize_prompt_payload(build_prompt_payload(public_context={"task_prompt_sections":sections})))
        self.assertEqual(delivered['runtime_feedback_summary']['source_candidate_id'],'accepted-a')
        self.assertEqual(delivered['runtime_feedback_summary']['lane_id'],'lane-a')
        behavior = delivered["public_feedback_summary"]["behavior"]
        self.assertEqual((behavior["source"]["generation_index"],behavior["source"]["lane_id"]),(1,"lane-a"))
        self.assertEqual(behavior["diff"]["status"],"available")
        self.assertTrue(any("1→2m/s" in row["text"] for row in behavior["secondary"]))
        self.assertLessEqual(len(json.dumps(behavior,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()),10240)
        self.assertEqual(set(behavior["profile"]),{"kind","report"})

        # Once accepted, a later generation observes the same incumbent. The
        # accepting generation remains the predecessor, not that self-observation.
        later = {'generation':{'generation_index':3,'parent_current_best':best,
            'parent_public_dev':trial['public_dev']['public_summary'],'trials':[]}}
        stored_path = fixture.root/'author-history.json'
        stored_path.write_text(json.dumps([*records,later],sort_keys=True),encoding='utf-8')
        sections = adapter.build_prompt_sections(public_records=json.loads(stored_path.read_bytes()),public_context={'current_best_identity':best})
        payload = json.loads(serialize_prompt_payload(build_prompt_payload(public_context={'task_prompt_sections':sections})))
        current = payload['public_feedback_summary']['behavior']
        self.assertEqual(current['source']['observation_generation'],3)
        self.assertEqual(current['diff']['source'],{'generation_index':1,'lane_id':'lane-a'})
        self.assertIn('1→2m/s',current['diff']['core_text'])
        self.assertEqual(list(current['profile']['report']),[*SENTENCE_SECTIONS,'actionability'])
        self.assertLessEqual(report_bytes(current),10240)

        # Two rejected synthetic candidates retain their own frozen-parent
        # effects; a failed-outcome main-card sentence cannot fund their space.
        rejected_a = deepcopy(trial)
        rejected_a.update(trial_status='not_selected',status_reason='public_dev_below_guard')
        rejected_b = deepcopy(rejected_a)
        rejected_b.update(trial_id='generation-1-lane-b',lane_id='lane-b',candidate={'candidate_id':'rejected-b','program_source_checksum':'2'*64})
        bcard = deepcopy(self.public)
        bcard['domain'].update(evaluation_job_id='generation-1-lane-b-eval-public-dev',checkpoint_hash='checkpoint-b')
        resign(bcard)
        # Independent expected means: B is slower than the frozen parent.
        ep = sample_episode(80)
        ep.task.update(cell_id="same-geometry",stratum_id="fixture")
        ep.arrays['base_pos_w'][:,0] = ep.times*.5
        bop = comparison_operands([measure_behavior(ep)])
        bop['domain'] = bcard['domain']
        rejected_b['public_dev'].update(checkpoint_hash='checkpoint-b',public_summary={**public_dev,'behavior':bcard,'behavior_comparison':bop})
        both = deepcopy(records[0]); both['generation']['trials']=[rejected_a,rejected_b]
        baseline = both['generation']['parent_current_best']
        shown = adapter.build_prompt_sections(public_records=[json.loads(json.dumps(both,sort_keys=True))],public_context={'current_best_identity':baseline})
        payload = json.loads(serialize_prompt_payload(build_prompt_payload(public_context={'task_prompt_sections':shown})))
        current=payload['public_feedback_summary']['behavior']
        self.assertEqual(current['source']['anchor_kind'],'reference')
        self.assertEqual([r['lane_id'] for r in current['secondary']],['lane-a','lane-b'])
        self.assertIn('1→2m/s',current['secondary'][0]['text'])
        self.assertIn('1→0.5m/s',current['secondary'][1]['text'])
        claims={r['id']:r['text'] for v in current['profile']['report'].values() if isinstance(v,list) for r in v}
        self.assertIn('claim:outcome.forbidden_contact',claims)
        self.assertIn('claim:outcome.timeout',claims)
        self.assertEqual({e['lane_id'] for e in payload['public_feedback_summary']['feedback_entries'] if 'lane_id' in e},{'lane-a','lane-b'})
        self.assertLessEqual(report_bytes(current),10240)

    def test_selection_membership_and_budget_reject_rehashed_invalid_values(self):
        for key, value in (("max_bytes", -1), ("optional_claim_ids", ["missing-claim"])):
            bad = deepcopy(self.public)
            bad["report_projection"][key] = value
            resign(bad)
            with self.subTest(key=key), self.assertRaises(ValueError):
                project_stair_behavior(bad)

    def test_failure_condition_process_precedes_diff_and_extra_secondary(self):
        from tests.unit.trajectory_fixtures import sample_episode
        from true_rl_agent.evaluation.trajectory_translation.measurement import measure_behavior
        from true_rl_agent.evaluation.trajectory_translation.cohort import aggregate_behavior
        from true_rl_agent.evaluation.trajectory_translation.cycles import measure_contact_cycles
        from true_rl_agent.tasks.stair_climbing.trajectory_cycles import episode_periods, summarize_periods
        from true_rl_agent.tasks.stair_climbing.trajectory_production import PROCESS_BANDS
        measured,periods = [],[]
        for i in range(30):
            ep = sample_episode(80)
            ep.env_index = i
            condition = ("concentrated","rare","other")[i//10]
            reason = "timeout" if i<8 else "off_course" if i==10 else "success"
            ep.task.update(stratum_id=condition,cell_id=condition)
            ep.task["outcome"].update(primary_reason=reason,success=reason=="success")
            if i in (6,7):
                ep.arrays["base_pos_w"][:,0] = ep.times
            measured.append(measure_behavior(ep))
            periods.append(episode_periods(measure_contact_cycles(ep)))
        summary = {"catalog_version":self.summary["catalog_version"],"archive_sha256":self.domain["archive_hash"],
            "behavior":aggregate_behavior(measured,motion_bands=PROCESS_BANDS),
            "cycle":summarize_periods(periods),"joint":None,"capabilities":{}}
        public = project_production_report(self.domain,summary)["public_report"]
        required = public["report_projection"]["required_claim_ids"]
        condition = next(row for row in public["report"]["process"] if row["id"]=="claim:conditions.timeout")
        self.assertIn(condition["id"],required)
        source = {"role":"当前编辑基准","anchor_kind":"candidate","generation_index":1,"lane_id":"lane-a"}
        diff = {"status":"available","reason":"matched_explicit_parent"}
        budget = 10240
        feedback = {"source":source,"profile":public,"diff":{**diff,"text":"条件配对的其他差分。",
            "rows":[{"condition":"other","text":"其他条件的较低优先级差异。"*20}]},
            "secondary":[{"generation_index":2,"lane_id":"lane-a","episodes":30,"outcomes":{"timeout":8,"success":21,"off_course":1}}]}
        feedback["secondary"].append({**feedback["secondary"][0],"lane_id":"lane-b"})
        second = author_behavior_payload(feedback,max_bytes=budget)
        claims = {row["id"]:row["text"] for section in SENTENCE_SECTIONS for row in second["profile"]["report"][section]}
        self.assertTrue(all(value in claims["claim:conditions.timeout"] for value in
                            ("concentrated ","末前速近零6/10","末前进2/10")))
        self.assertIn("claim:outcome.off_course",claims)
        self.assertEqual([r['lane_id'] for r in second['secondary']],['lane-a','lane-b'])
        self.assertLessEqual(report_bytes(second),budget)


if __name__ == "__main__":
    unittest.main()
