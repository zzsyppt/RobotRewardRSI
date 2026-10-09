"""V2 recovery closure and qualification; no simulator or policy claims."""

import copy
import unittest
from unittest.mock import patch

from true_rl_agent.evaluation.derived_statistics import DERIVED_STATISTICS_VERSION
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery import metrics


def rollout(successes=3, *, pushed=4):
    return {
        "schema_version": metrics.RECOVERY_ROLLOUT_SCHEMA, "status": "pass",
        "aggregate": {
            "episodes": 4, "eligible_episode_count": 4, "control_episode_count": 0,
            "planned_push_event_count": 4, "pushed_episode_count": pushed,
            "push_event_count": pushed, "recovered_event_count": successes,
            "invalid_injection_count": 0, "successes": successes,
            "success_rate": successes / 4,
            "failure_breakdown": {"counts": {"success_recovered": successes,
                                             "fall_height_collapse": 4 - successes}, "total": 4},
        },
        # A small body must not inherit the old absolute G1 height gate.
        "secondary_metrics": {"mean_min_base_height_m": 0.12},
    }


class RecoveryV2MetricsTests(unittest.TestCase):
    def test_reference_dev_screen_uses_frozen_band_and_actual_recovery_evidence(self):
        from types import SimpleNamespace
        from true_rl_agent.evaluation.phases import EvaluationPhase
        from true_rl_agent.tasks.push_recovery.phased_adapter import PushPhasedEvaluationAdapter, PUSH_PHASE_EVIDENCE_SCHEMA
        from true_rl_agent.tasks.push_recovery.reference_program import reference_dev_evidence_passes

        config = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2")
        for score, expected in ((None, False), (0.1999, False), (0.2, True), (0.9, True), (0.9001, False)):
            self.assertEqual(reference_dev_evidence_passes(config, {"score_stats": {"mean": score}}), expected)
        adapter = PushPhasedEvaluationAdapter(config=config, task_protocol_hash="a" * 64)
        for sample, expected in ((rollout(1), "pass"), (rollout(4), "fail"), (rollout(0, pushed=0), "fail")):
            phase = SimpleNamespace(task_id="unitree_push_recovery", phase=EvaluationPhase.PUBLIC_DEV,
                eval_protocol_hash=config.evaluation_protocol().protocol_hash,
                task_evidence={"schema_version": PUSH_PHASE_EVIDENCE_SCHEMA, "phase": "public_dev", "seed_summaries": [sample] * 3})
            gate = adapter.project_reference_public_dev_gate(phase_result=phase)
            self.assertEqual(gate["status"], expected)
            self.assertEqual(gate["public_dev_reference_band"], [0.2, 0.9])

    def test_planned_early_falls_are_valid_failures_and_keep_the_denominator(self):
        failed = rollout(0, pushed=0)
        self.assertEqual(metrics.push_recovery_evidence_gate([failed])["status"], "pass")
        projected = metrics.project_push_split([failed], split_name="public_dev", errors=[])
        self.assertEqual(projected["score_stats"]["episode_count"], 4)
        self.assertEqual(projected["score_stats"]["mean"], 0)

    def test_public_strata_are_counts_only_without_raw_scenario_details(self):
        raw = rollout()
        raw["aggregate"]["stratum_results"] = {"low": {"episodes": 4, "successes": 3,
            "future_direction": 1.75}, "control": {"episodes": 0, "successes": 0}}
        result = metrics.project_push_split([raw], split_name="public_dev", errors=[])
        self.assertEqual(result["seed_results"][0]["stratum_results"],
                         {"low": {"episodes": 4, "successes": 3}})

    def test_no_push_control_cannot_be_formal_recovery_success(self):
        control = rollout()
        control["aggregate"].update(control_episode_count=4, eligible_episode_count=0,
                                    planned_push_event_count=0)
        self.assertEqual(metrics.push_recovery_evidence_gate([control])["status"], "fail")

    def test_success_requires_injection_and_integer_closure(self):
        for field, value in (("pushed_episode_count", 0), ("success_rate", 1.0),
                             ("invalid_injection_count", 1), ("recovered_event_count", 5)):
            with self.subTest(field=field):
                corrupt = rollout()
                corrupt["aggregate"][field] = value
                self.assertEqual(metrics.push_recovery_evidence_gate([corrupt])["status"], "fail")

    def test_v2_uses_one_recovery_gate_and_canonical_statistics(self):
        config = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2")
        result = metrics.build_push_public_result_from_summaries(
            public_dev_summaries=[rollout(), rollout(2), rollout(1)],
            public_accept_summaries=[rollout()] * 3,
            budget_summary={"status": "pass"},
            acceptance_protocol=config.acceptance_protocol(),
            evaluation_protocol=config.evaluation_protocol())
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.metrics["recovery_evidence_gate"]["status"], "pass")
        self.assertNotIn("behavior_quality_gate", result.metrics)
        self.assertNotIn("motion_quality_gate", result.metrics)
        self.assertEqual(result.metrics["public_dev"]["score_stats"]["statistics_version"],
                         DERIVED_STATISTICS_VERSION)
        attached = metrics.attach_push_budget_gate(result, budget_summary={"status": "pass"})
        self.assertNotIn("behavior_quality_gate", attached.metrics["acceptance"]["required_pass_gates"])

    def test_new_holdout_drop_rule_is_executed_and_legacy_keeps_its_rule(self):
        config = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2")
        def evaluate(summaries, accept):
            with patch.object(metrics, "build_sealed_rollout_evidence", side_effect=lambda **kw: kw):
                return metrics.build_push_sealed_rollout_evidence(
                    summaries=summaries, evaluation_protocol=config.evaluation_protocol(),
                    checkpoint_hash="unused", evaluation_binding_hash="unused", evaluation_subject=None,
                    capability_profile_hash="unused", acceptance_protocol=config.acceptance_protocol(),
                    public_accept_score=accept)
        self.assertEqual(evaluate([rollout(2)] * 7, 0.75)["task_validation_status"], "fail")
        self.assertEqual(evaluate([rollout(3)] * 7, 0.80)["task_validation_status"], "pass")
        legacy = copy.deepcopy(rollout(2))
        legacy.pop("schema_version")
        self.assertEqual(evaluate([legacy] * 7, None)["task_validation_status"], "pass")


if __name__ == "__main__":
    unittest.main()
