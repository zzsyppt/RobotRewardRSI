from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.candidates.authoring_policy import (  # noqa: E402
    CANDIDATE_MECHANISM_OPERATION_DELTA_SCHEMA_VERSION,
    MAX_CANONICAL_COMPONENT_NAME_CHARS,
)
from true_rl_agent.agent_core.generation_contract import (  # noqa: E402
    GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
)
from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats  # noqa: E402
from true_rl_agent.evaluation.public_result import PublicEvalResult  # noqa: E402
from true_rl_agent.evaluation.subject import ExperimentSubject  # noqa: E402
from true_rl_agent.prompting.serialization import (  # noqa: E402
    canonical_prompt_json_bytes,
)
from true_rl_agent.runner.budget import build_budget_accounting  # noqa: E402
from true_rl_agent.tasks.prompt_adapter import (  # noqa: E402
    MAX_TASK_PROMPT_SECTIONS_BYTES,
)
from true_rl_agent.tasks.stair_climbing.component_smoke import (  # noqa: E402
    STAIR_SMOKE_NUM_ENVS,
    StairClimbingComponentSmokeAdapter,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import (  # noqa: E402
    StairEpisodeOutcome,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (  # noqa: E402
    build_stair_sealed_rollout_evidence,
    build_stair_split_summary,
    validate_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.generalization_contract import (  # noqa: E402
    build_stair_generalization_summary,
)
from true_rl_agent.tasks.stair_climbing.memory import (  # noqa: E402
    StairTaskMemoryAdapter,
)
from true_rl_agent.tasks.stair_climbing.exploration import (  # noqa: E402
    StairExplorationProjector,
)
from true_rl_agent.tasks.stair_climbing.prompt_adapter import (  # noqa: E402
    StairClimbingPromptAdapter,
)
from true_rl_agent.tasks.stair_climbing.public_projection import (  # noqa: E402
    build_stair_candidate_runtime_effect_evidence,
    build_stair_prompt_feedback,
    build_stair_prompt_score_semantics,
    build_stair_runtime_activation_gate,
    project_stair_public_result,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (  # noqa: E402
    build_stair_execution_source_preflight_summary,
    stair_training_distribution_upstream_source_evidence,
)


class StairPublicProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        self.public_dev = _summary(self.config, "public_dev")
        self.public_accept = _summary(self.config, "public_accept")
        self.source_preflight = _source_preflight(self.config)

    def test_split_validator_recomputes_statistics_and_closed_counts(self) -> None:
        canonical = validate_stair_split_summary(
            self.config.protocol_v1,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="public_accept",
            summary=self.public_accept["anchor"],
        )
        self.assertEqual(canonical["score_stats"]["mean"], 0.5)
        self.assertEqual(
            sum(canonical["primary_outcome_counts"].values()),
            canonical["total_episode_count"],
        )
        self.assertEqual(
            canonical["evaluation_semantics_hash"],
            self.config.evaluation_semantics.protocol_hash,
        )

        changed = copy.deepcopy(self.public_accept["anchor"])
        changed["evaluation_semantics_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "evaluation semantics"):
            validate_stair_split_summary(
                self.config.protocol_v1,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                summary=changed,
            )

        changed = copy.deepcopy(self.public_accept["anchor"])
        changed["score_stats"]["mean"] = 0.9
        with self.assertRaisesRegex(ValueError, "score mean"):
            validate_stair_split_summary(
                self.config.protocol_v1,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                summary=changed,
            )

        changed = copy.deepcopy(self.public_accept["anchor"])
        changed["primary_outcome_counts"]["success"] += 1
        changed["primary_outcome_counts"]["timeout"] -= 1
        with self.assertRaisesRegex(ValueError, "success counts"):
            validate_stair_split_summary(
                self.config.protocol_v1,
                semantics=self.config.evaluation_semantics,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                summary=changed,
            )

    def test_exploration_value_uses_stair_owned_public_objective_and_guard(self) -> None:
        projector = StairExplorationProjector(self.config)
        parent = CurrentBestEvalStats(
            mode="incomplete",
            anchor_kind="bootstrap",
            anchor_id="stair-exploration-parent",
            public_accept_stats={"mean": 0.2, "score_samples": [0.2] * 5},
            guard_score_stats={"mean": 0.5, "score_samples": [0.5] * 5},
        )
        trial = {
            "trial_status": "qualified_not_selected",
            "public_dev": {"public_summary": self.public_dev},
            "public_accept": {"public_summary": self.public_accept},
        }
        generation = {
            "guard_reference": {
                "generalization_score_stats": {"mean": 0.2},
                "anchor_score_stats": {"mean": 0.5},
            }
        }
        positive = projector.assess_trial(
            generation=generation,
            trial=trial,
            parent=parent,
        )
        self.assertEqual(positive.observed_effect, "strong_positive")
        self.assertEqual(positive.value_label, "high")
        self.assertTrue(positive.public_evidence_summary["guard"]["guard_met"])
        self.assertGreater(positive.revisit_rank_key[0], 0)

        missing_reference = projector.assess_trial(
            generation={"guard_reference": None},
            trial=trial,
            parent=parent,
        )
        self.assertEqual(missing_reference.observed_effect, "unknown")
        self.assertEqual(missing_reference.value_label, "unknown")
        self.assertEqual(
            missing_reference.reason_codes,
            ("parent_public_dev_reference_missing",),
        )

        conflicting = projector.assess_trial(
            generation={
                "guard_reference": {
                    "generalization_score_stats": {"mean": 0.6},
                    "anchor_score_stats": {"mean": 0.5},
                }
            },
            trial=trial,
            parent=parent,
        )
        self.assertEqual(conflicting.observed_effect, "mixed")
        self.assertEqual(
            conflicting.reason_codes,
            ("public_dev_accept_direction_conflict",),
        )

        guard_parent = CurrentBestEvalStats(
            mode="incomplete",
            anchor_kind="bootstrap",
            anchor_id="stair-exploration-guard-parent",
            public_accept_stats={"mean": 0.2, "score_samples": [0.2] * 5},
            guard_score_stats={"mean": 0.6, "score_samples": [0.6] * 5},
        )
        unsafe = projector.assess_trial(
            generation=generation,
            trial=trial,
            parent=guard_parent,
        )
        self.assertEqual(unsafe.observed_effect, "negative")
        self.assertEqual(unsafe.value_label, "low")

    def test_prompt_keeps_latest_paired_objective_guard_comparisons(self) -> None:
        generation_id = "g" * 128
        eval_protocol_hash = self.config.evaluation_protocol().protocol_hash
        parent = CurrentBestEvalStats(
            mode="incomplete",
            anchor_kind="bootstrap",
            anchor_id="stair-prompt-comparison-parent",
            public_accept_stats={
                "count": 5,
                "mean": 0.6,
                "std": 0.0,
                "standard_error": 0.0,
                "eval_protocol_hash": eval_protocol_hash,
                "score_samples": [0.6] * 5,
            },
            guard_score_stats={
                "count": 5,
                "mean": 0.4,
                "std": 0.0,
                "standard_error": 0.0,
                "eval_protocol_hash": eval_protocol_hash,
                "score_samples": [0.4] * 5,
            },
            task_protocol_hash=self.config.task_protocol_hash,
            eval_protocol_hash=eval_protocol_hash,
            acceptance_protocol_hash=(
                self.config.acceptance_protocol.protocol_hash
            ),
            min_score_count=5,
        )
        trials = [
            {
                "trial_id": generation_id + "-lane-a",
                "lane_id": "lane-a",
                "candidate": {
                    "candidate_id": "paired-candidate-a",
                    "program_source_checksum": "sha256:" + ("a" * 64),
                },
                "trial_status": "qualified_not_selected",
                "public_dev": {"public_summary": self.public_dev},
                "public_accept": {"public_summary": self.public_accept},
            },
            {
                "trial_id": generation_id + "-lane-b",
                "lane_id": "lane-b",
                "candidate": {
                    "candidate_id": "paired-candidate-b",
                    "program_source_checksum": "sha256:" + ("b" * 64),
                },
                "trial_status": "accepted",
                "public_dev": {"public_summary": self.public_dev},
                "public_accept": {"public_summary": self.public_accept},
            },
        ]
        generation = {
            "schema_version": GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "generation_index": 2,
            "parent_current_best": parent.to_public_dict(),
            "guard_reference": {
                "generalization_score_stats": {"mean": 0.5},
                "anchor_score_stats": {"mean": 0.5},
            },
            "trials": trials,
            "selection": {"selected_trial_id": generation_id + "-lane-b"},
        }
        newer_incomplete_generation = {
            "schema_version": GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
            "generation_id": "generation-003",
            "generation_index": 3,
            "parent_current_best": parent.to_public_dict(),
            "trials": [
                {
                    "trial_id": "generation-003-lane-a",
                    "lane_id": "lane-a",
                    "trial_status": "candidate_failed",
                    "public_dev": None,
                    "public_accept": None,
                },
                {
                    "trial_id": "generation-003-lane-b",
                    "lane_id": "lane-b",
                    "trial_status": "not_started",
                    "public_dev": None,
                    "public_accept": None,
                },
            ],
            "selection": None,
        }
        newer_partial_generation = {
            "schema_version": GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
            "generation_id": "generation-004",
            "generation_index": 4,
            "parent_current_best": parent.to_public_dict(),
            "guard_reference": {
                "generalization_score_stats": {"mean": 0.5},
                "anchor_score_stats": {"mean": 0.5},
            },
            "trials": [
                {
                    "trial_id": "generation-004-lane-a",
                    "lane_id": "lane-a",
                    "candidate": {
                        "candidate_id": "partial-candidate-a",
                        "program_source_checksum": "sha256:" + ("c" * 64),
                    },
                    "trial_status": "qualified_not_selected",
                    "public_dev": {"public_summary": self.public_dev},
                    "public_accept": {"public_summary": self.public_accept},
                },
                {
                    "trial_id": "generation-004-lane-b",
                    "lane_id": "lane-b",
                    "trial_status": "candidate_failed",
                    "public_dev": None,
                    "public_accept": None,
                },
            ],
            "selection": None,
        }

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[
                {"generation": generation},
                {"generation": newer_incomplete_generation},
                {"generation": newer_partial_generation},
            ],
            public_context={},
        )

        comparisons = sections["public_feedback_summary"][
            "latest_generation_trial_comparisons"
        ]
        self.assertEqual(
            sections["public_feedback_summary"][
                "latest_generation_trial_comparison_basis"
            ],
            "public_accept_vs_parent",
        )
        self.assertEqual(len(comparisons), 2)
        by_lane = {value["lane_id"]: value for value in comparisons}
        self.assertEqual(by_lane["lane-a"]["selection_status"], "not_selected")
        self.assertEqual(by_lane["lane-b"]["selection_status"], "selected")
        self.assertEqual(by_lane["lane-a"]["candidate_id"], "paired-candidate-a")
        self.assertEqual(
            by_lane["lane-b"]["program_source_checksum"],
            "sha256:" + ("b" * 64),
        )
        self.assertEqual(
            by_lane["lane-a"]["trial_status"],
            "qualified_not_selected",
        )
        self.assertAlmostEqual(by_lane["lane-a"]["objective_delta"], -0.1)
        self.assertAlmostEqual(by_lane["lane-a"]["guard_delta"], 0.1)
        self.assertFalse(by_lane["lane-a"]["objective_materiality_met"])
        self.assertTrue(by_lane["lane-a"]["guard_met"])
        self.assertIn(
            "public_guard_improved_objective_not_improved",
            by_lane["lane-a"]["reason_codes"],
        )
        expected_comparison_fields = {
            "generation_id",
            "trial_id",
            "lane_id",
            "candidate_id",
            "program_source_checksum",
            "trial_status",
            "selection_status",
            "observed_effect",
            "objective_delta",
            "objective_materiality_threshold",
            "objective_materiality_met",
            "guard_delta",
            "guard_max_regression",
            "guard_met",
            "reason_codes",
        }
        self.assertTrue(
            all(set(value) == expected_comparison_fields for value in comparisons)
        )
        self.assertNotIn("checkpoint", repr(comparisons).lower())

        too_long_generation = copy.deepcopy(generation)
        too_long_generation["generation_id"] = "g" * 129
        with self.assertRaisesRegex(ValueError, "generation prompt identity"):
            StairClimbingPromptAdapter(self.config).build_prompt_sections(
                public_records=[{"generation": too_long_generation}],
                public_context={},
            )

        too_long_trial = copy.deepcopy(generation)
        too_long_trial["trials"][0]["trial_id"] = "t" * 136
        with self.assertRaisesRegex(ValueError, "trial identity"):
            StairClimbingPromptAdapter(self.config).build_prompt_sections(
                public_records=[{"generation": too_long_trial}],
                public_context={},
            )

    def test_projection_overwrites_forged_task_gates(self) -> None:
        raw = PublicEvalResult(
            status="pass",
            public_score=0.5,
            metrics={
                "environment_source_preflight": self.source_preflight,
                "stair_evaluation": {
                    "public_dev": self.public_dev,
                    "public_accept": self.public_accept,
                },
                "stair_behavior_gate": {
                    "status": "pass",
                    "reason": "forged_candidate_gate",
                },
                "runtime_activation_gate": {"status": "pass"},
                "budget_accounting": {"status": "pass"},
                "execution_device": "cuda:1",
            },
        )
        projected = project_stair_public_result(
            config=self.config,
            public_result=raw,
            train_summary=_train_summary(self.config),
        )

        self.assertEqual(
            projected.metrics["stair_behavior_gate"]["reason"],
            "task_recorder_public_split_evidence_complete",
        )
        self.assertEqual(projected.metrics["budget_accounting"]["status"], "pass")
        self.assertEqual(
            projected.metrics["environment_source_preflight"],
            self.source_preflight,
        )
        self.assertEqual(
            projected.metrics["runtime_activation_gate"]["status"],
            "fail",
        )
        self.assertEqual(
            projected.metrics["acceptance"]["required_pass_gates"],
            self.config.protocol_v1.to_dict()["acceptance"]["required_pass_gates"],
        )
        self.assertEqual(
            projected.metrics["acceptance"]["acceptance_protocol_hash"],
            self.config.acceptance_protocol.protocol_hash,
        )
        self.assertEqual(
            projected.metrics["acceptance"]["score_samples"],
            self.public_accept["generalization"]["seed_macro_scores"],
        )
        self.assertEqual(
            projected.metrics["acceptance"]["guard_score_samples"],
            self.public_accept["anchor"]["seed_macro_scores"],
        )
        self.assertEqual(
            projected.metrics["acceptance"]["guard_regression_margin"],
            0.02,
        )
        self.assertEqual(projected.metrics["execution_device"], "cuda:1")
        self.assertNotIn("stair_evaluation", projected.metrics)

    def test_projection_rejects_unknown_non_task_metric(self) -> None:
        raw = PublicEvalResult(
            status="pass",
            public_score=0.5,
            metrics={
                "environment_source_preflight": self.source_preflight,
                "stair_evaluation": {
                    "public_dev": self.public_dev,
                    "public_accept": self.public_accept,
                },
                "unknown_runner_identity": "forged",
            },
        )
        with self.assertRaisesRegex(
            ValueError,
            "unsupported field.*unknown_runner_identity",
        ):
            project_stair_public_result(
                config=self.config,
                public_result=raw,
                train_summary=_train_summary(self.config),
            )

    def test_projection_recomputes_task_owned_sealed_score_verdict(self) -> None:
        evidence = build_stair_sealed_rollout_evidence(
            self.config.protocol_v1,
            semantics=self.config.evaluation_semantics,
            evaluation_protocol=self.config.evaluation_protocol(),
            summary=_summary(self.config, "sealed_validation")["generalization"],
            checkpoint_hash="sha256:" + "1" * 64,
            evaluation_binding_hash="2" * 64,
            evaluation_subject=ExperimentSubject(
                kind="candidate",
                subject_id="candidate-1",
            ),
            subject_manifest_hash="3" * 64,
            program_source_checksum="sha256:" + "4" * 64,
            training_binding_hash="5" * 64,
            capability_profile_hash="6" * 64,
            acceptance_protocol=self.config.acceptance_protocol,
            evaluation_profile=self.config.evaluation_profile,
            public_objective_summary=self.public_accept["generalization"],
        )
        self.assertEqual(evidence["status"], "pass")

        for field, forged_value in (
            ("status", "fail"),
            ("reason", "stair_generalization_holdout_floor_met"),
        ):
            with self.subTest(field=field):
                forged = copy.deepcopy(evidence)
                forged[field] = forged_value
                with self.assertRaisesRegex(
                    ValueError,
                    f"sealed validation {field}",
                ):
                    project_stair_public_result(
                        config=self.config,
                        public_result=PublicEvalResult(
                            status="pass",
                            public_score=0.5,
                            metrics={
                                "environment_source_preflight": self.source_preflight,
                                "stair_evaluation": {
                                    "public_dev": self.public_dev,
                                    "public_accept": self.public_accept,
                                },
                                "sealed_validation": forged,
                            },
                        ),
                        train_summary=_train_summary(self.config),
                    )

        missing_hash = copy.deepcopy(evidence)
        missing_hash.pop("task_evidence_hash")
        with self.assertRaisesRegex(
            ValueError,
            "requires a canonical task evidence hash",
        ):
            project_stair_public_result(
                config=self.config,
                public_result=PublicEvalResult(
                    status="pass",
                    public_score=0.5,
                    metrics={
                        "environment_source_preflight": self.source_preflight,
                        "stair_evaluation": {
                            "public_dev": self.public_dev,
                            "public_accept": self.public_accept,
                        },
                        "sealed_validation": missing_hash,
                    },
                ),
                train_summary=_train_summary(self.config),
            )

    def test_projection_rejects_score_and_budget_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "public score"):
            project_stair_public_result(
                config=self.config,
                public_result=PublicEvalResult(
                    status="pass",
                    public_score=0.7,
                    metrics={
                        "environment_source_preflight": self.source_preflight,
                        "stair_evaluation": {
                            "public_dev": self.public_dev,
                            "public_accept": self.public_accept,
                        }
                    },
                ),
                train_summary=_train_summary(self.config),
            )

        train_summary = _train_summary(self.config)
        train_summary["training"]["budget_accounting"]["budget_steps"] = (
            self.config.budget_steps + 1
        )
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    }
                },
            ),
            train_summary=train_summary,
        )
        self.assertEqual(projected.metrics["budget_accounting"]["status"], "fail")

    def test_projection_rejects_missing_or_tampered_source_preflight(self) -> None:
        mutations = {
            "missing": None,
            "schema": {**self.source_preflight, "schema_version": "forged.v1"},
            "protocol": {**self.source_preflight, "task_protocol_hash": "0" * 64},
            "manifest": {**self.source_preflight, "source_manifest_hash": "0" * 64},
            "extra_field": {**self.source_preflight, "unexpected": True},
        }
        for stage in ("train", "eval"):
            for mutation, changed in mutations.items():
                with self.subTest(stage=stage, mutation=mutation):
                    train_summary = _train_summary(self.config)
                    metrics = {
                        "environment_source_preflight": self.source_preflight,
                        "stair_evaluation": {
                            "public_dev": self.public_dev,
                            "public_accept": self.public_accept,
                        },
                    }
                    target = train_summary if stage == "train" else metrics
                    if changed is None:
                        target.pop("environment_source_preflight")
                    else:
                        target["environment_source_preflight"] = changed

                    with self.assertRaisesRegex(
                        ValueError,
                        f"Stair {stage} environment source preflight",
                    ):
                        project_stair_public_result(
                            config=self.config,
                            public_result=PublicEvalResult(
                                status="pass",
                                public_score=0.5,
                                metrics=metrics,
                            ),
                            train_summary=train_summary,
                        )

    def test_runtime_gate_requires_complete_execution_evidence(self) -> None:
        train_summary = _train_summary(self.config)
        source_checksum = "sha256:" + "1" * 64
        subject_manifest_hash = "2" * 64
        qualification_hash = "3" * 64
        train_summary["program_source_checksum"] = source_checksum
        train_summary["subject_kind"] = "candidate"
        train_summary["subject_id"] = "candidate-runtime-test"
        train_summary["subject_manifest_hash"] = subject_manifest_hash
        train_summary["reward_program_qualification_hash"] = qualification_hash
        term_names = [f"term_{index}" for index in range(15)]
        expected_dispatcher_calls = (
            self.config.train_steps_per_iteration * self.config.train_max_iterations
        ) + 1
        train_summary["runtime_activation_summary"] = {
            "status": "pass",
            "runtime_status": "active_pre_reset_complete_reward_v1",
            "dispatcher_calls": expected_dispatcher_calls,
            "term_calls": expected_dispatcher_calls * 15,
            "declared_term_count": 15,
            "input_guard": "pass",
            "rng_guard": "pass",
            "source_alias_guard": "pass",
            "reward_manager_active_terms": ["complete_reward"],
            "reward_manager_term_weight": 1.0,
            "training_recorder_enabled": False,
            "reward_term_calls": expected_dispatcher_calls * 15,
            "diagnostic_calls": 0,
            "nonzero_reward_contribution": True,
            "component_activation_records": [
                {"name": name} for name in term_names
            ],
            "capability_activation_records": ["complete_reward_program"],
            "activation_preflight": {
                "schema_version": STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION,
                "status": "pass",
                "real_transition_dispatcher_calls": 1,
                "transition_common_step": 1,
                "repeat_count": 2,
                "term_names": term_names,
            },
            "program": {
                "program_source_checksum": source_checksum,
                "subject_manifest_hash": subject_manifest_hash,
                "qualification_hash": qualification_hash,
                "subject": {
                    "kind": "candidate",
                    "id": "candidate-runtime-test",
                },
                "input_schema_hash": (
                    self.config.protocol_v2.complete_reward_runtime.input_schema_hash
                ),
                "term_names": term_names,
            },
        }

        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    }
                },
            ),
            train_summary=train_summary,
        )

        self.assertEqual(
            projected.metrics["runtime_activation_gate"]["status"],
            "pass",
        )
        forged = copy.deepcopy(train_summary)
        forged["runtime_activation_summary"]["term_calls"] -= 1
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    }
                },
            ),
            train_summary=forged,
        )
        self.assertEqual(
            projected.metrics["runtime_activation_gate"]["status"],
            "fail",
        )
        self.assertIn(
            "term_call_accounting",
            projected.metrics["runtime_activation_gate"]["failed_checks"],
        )

        forged = copy.deepcopy(train_summary)
        forged["runtime_activation_summary"]["activation_preflight"][
            "real_transition_dispatcher_calls"
        ] = 0
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    }
                },
            ),
            train_summary=forged,
        )
        self.assertIn(
            "activation_preflight",
            projected.metrics["runtime_activation_gate"]["failed_checks"],
        )
        forged = copy.deepcopy(train_summary)
        forged["runtime_activation_summary"]["program"][
            "subject_manifest_hash"
        ] = "4" * 64
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    }
                },
            ),
            train_summary=forged,
        )
        self.assertIn(
            "program_identity",
            projected.metrics["runtime_activation_gate"]["failed_checks"],
        )

    def test_runtime_gate_observes_candidate_reward_operation(self) -> None:
        train_summary = _train_summary(self.config)
        source_checksum = "sha256:" + "1" * 64
        subject_manifest_hash = "2" * 64
        qualification_hash = "3" * 64
        train_summary.update(
            {
                "program_source_checksum": source_checksum,
                "subject_kind": "candidate",
                "subject_id": "candidate-runtime-test",
                "subject_manifest_hash": subject_manifest_hash,
                "reward_program_qualification_hash": qualification_hash,
            }
        )
        term_names = ["reference_term", "candidate_term"]
        expected_dispatcher_calls = (
            self.config.train_steps_per_iteration * self.config.train_max_iterations
        ) + 1
        train_summary["runtime_activation_summary"] = {
            "status": "pass",
            "runtime_status": "active_pre_reset_complete_reward_v1",
            "dispatcher_calls": expected_dispatcher_calls,
            "term_calls": expected_dispatcher_calls * len(term_names),
            "declared_term_count": len(term_names),
            "input_guard": "pass",
            "rng_guard": "pass",
            "source_alias_guard": "pass",
            "reward_manager_active_terms": ["complete_reward"],
            "reward_manager_term_weight": 1.0,
            "training_recorder_enabled": False,
            "reward_term_calls": expected_dispatcher_calls * len(term_names),
            "diagnostic_calls": 0,
            "nonzero_reward_contribution": True,
            "component_activation_records": [
                {
                    "component_kind": "reward_term",
                    "name": name,
                    "max_abs_weighted_contribution": (
                        0.5 if name == "reference_term" else 2.0e-12
                    ),
                }
                for name in term_names
            ],
            "capability_activation_records": ["complete_reward_program"],
            "activation_preflight": {
                "schema_version": STAIR_REWARD_ACTIVATION_PREFLIGHT_SCHEMA_VERSION,
                "status": "pass",
                "real_transition_dispatcher_calls": 1,
                "transition_common_step": 1,
                "repeat_count": 2,
                "term_names": term_names,
            },
            "program": {
                "program_source_checksum": source_checksum,
                "subject_manifest_hash": subject_manifest_hash,
                "qualification_hash": qualification_hash,
                "subject": {
                    "kind": "candidate",
                    "id": "candidate-runtime-test",
                },
                "input_schema_hash": (
                    self.config.protocol_v2.complete_reward_runtime.input_schema_hash
                ),
                "term_names": term_names,
            },
        }
        operation_delta = {
            "schema_version": CANDIDATE_MECHANISM_OPERATION_DELTA_SCHEMA_VERSION,
            "operation": "add",
            "before_component": None,
            "after_component": {
                "mechanism_id": "RewardTerm:candidate_term",
                "structure_hash": "4" * 64,
                "parameter_normalized_structure_hash": "5" * 64,
                "input_fields": ["goal_progress_fraction"],
            },
            "anchor_source_checksum": "sha256:" + "6" * 64,
            "candidate_source_checksum": "sha256:" + "7" * 64,
        }

        gate = build_stair_runtime_activation_gate(
            self.config,
            train_summary,
            mechanism_operation_delta=operation_delta,
        )
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(
            build_stair_candidate_runtime_effect_evidence(
                train_summary,
                mechanism_operation_delta=operation_delta,
            )["status"],
            "pass",
        )
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=train_summary,
            mechanism_operation_delta=operation_delta,
        )
        self.assertEqual(
            projected.metrics["runtime_activation_gate"]["status"],
            "pass",
        )

        for contribution in (0.0, 1.0e-12, float("nan"), None):
            with self.subTest(contribution=contribution):
                changed = copy.deepcopy(train_summary)
                changed["runtime_activation_summary"][
                    "component_activation_records"
                ][1]["max_abs_weighted_contribution"] = contribution
                gate = build_stair_runtime_activation_gate(
                    self.config,
                    changed,
                    mechanism_operation_delta=operation_delta,
                )
                self.assertEqual(gate["status"], "fail")
                self.assertEqual(
                    build_stair_candidate_runtime_effect_evidence(
                        changed,
                        mechanism_operation_delta=operation_delta,
                    )["status"],
                    "fail",
                )
                self.assertIn(
                    "candidate_reward_operation_runtime_effect",
                    gate["failed_checks"],
                )

        removal_delta = {
            **operation_delta,
            "operation": "remove",
            "before_component": {
                **operation_delta["after_component"],
                "mechanism_id": "RewardTerm:removed_term",
            },
            "after_component": None,
        }
        gate = build_stair_runtime_activation_gate(
            self.config,
            train_summary,
            mechanism_operation_delta=removal_delta,
        )
        self.assertEqual(gate["status"], "pass")

        still_present = copy.deepcopy(removal_delta)
        still_present["before_component"]["mechanism_id"] = (
            "RewardTerm:reference_term"
        )
        gate = build_stair_runtime_activation_gate(
            self.config,
            train_summary,
            mechanism_operation_delta=still_present,
        )
        self.assertEqual(gate["status"], "fail")
        self.assertIn(
            "candidate_reward_operation_runtime_effect",
            gate["failed_checks"],
        )

        for reward_delta in (operation_delta, removal_delta):
            with self.subTest(
                unsupported_operation=reward_delta["operation"],
            ):
                unsupported = copy.deepcopy(reward_delta)
                changed_component = (
                    unsupported["before_component"]
                    if unsupported["operation"] == "remove"
                    else unsupported["after_component"]
                )
                changed_component["mechanism_id"] = "DiagnosticTerm:future_probe"
                with self.assertRaisesRegex(
                    ValueError,
                    "supports RewardTerm deltas only",
                ):
                    build_stair_runtime_activation_gate(
                        self.config,
                        train_summary,
                        mechanism_operation_delta=unsupported,
                    )

        with self.assertRaisesRegex(
            ValueError,
            "invalid verifier-owned mechanism operation delta",
        ):
            build_stair_runtime_activation_gate(
                self.config,
                train_summary,
                mechanism_operation_delta={"operation": "add"},
            )

    def test_prompt_memory_exposes_only_the_phase_c_public_abi(self) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    }
                },
            ),
            train_summary=_train_summary(self.config),
        )
        record = {
            "task": {"task_id": STAIR_TASK_ID},
            "public_result": projected.to_public_dict(),
            "raw_seed": 123,
        }
        memory = StairTaskMemoryAdapter().build_public_summary([record])
        self.assertEqual(
            memory["metric_scope"],
            "stable_top_generalization_objective",
        )
        self.assertEqual(
            memory["score_source"],
            "acceptance_objective.score_stats.mean",
        )
        self.assertEqual(
            memory["failure_summary_source"],
            "generalization_public_dev",
        )
        self.assertEqual(
            memory["stable_top_success"]["latest"],
            projected.metrics["acceptance_objective"]["score_stats"]["mean"],
        )
        self.assertNotIn("macro_success", memory)
        legacy_memory = StairTaskMemoryAdapter().build_public_summary(
            [
                {
                    "task": {"task_id": STAIR_TASK_ID},
                    "public_result": {
                        "metrics": {
                            "public_accept": self.public_accept["anchor"],
                            "public_dev": self.public_dev["anchor"],
                        }
                    },
                }
            ]
        )
        self.assertEqual(legacy_memory, {})

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[record],
            public_context={"task_public_summary": memory},
        )
        self.assertEqual(sections["reward_inputs_contract"]["status"], "available")
        height_scan_semantics = sections["reward_inputs_contract"][
            "field_semantics_constraints"
        ]["height_scan"]
        self.assertEqual(
            height_scan_semantics["flattened_index_to_ray_mapping"],
            "not_exposed",
        )
        self.assertIn("positive=lower/farther", height_scan_semantics["sensor_semantics"])
        self.assertIn("no overhead or torso", height_scan_semantics["sensor_semantics"])
        self.assertIn("order-invariant", height_scan_semantics["candidate_usage"])
        termination_masks = sections["reward_inputs_contract"][
            "field_semantics_constraints"
        ]["termination_masks"]
        self.assertEqual(
            termination_masks["equivalent_fields"],
            {"terminated_non_timeout": "base_contact"},
        )
        self.assertIn("identical masks", termination_masks["candidate_usage"])
        progress_fraction = sections["reward_inputs_contract"][
            "field_semantics_constraints"
        ]["progress_fraction"]
        self.assertEqual(
            progress_fraction["spatial_alternative"],
            "goal_progress_fraction",
        )
        self.assertIn("proximity to the top", progress_fraction["candidate_usage"])
        base_height = sections["reward_inputs_contract"][
            "field_semantics_constraints"
        ]["base_height"]
        self.assertFalse(base_height["local_clearance"])
        self.assertIn("staircase elevation", base_height["candidate_usage"])
        projected_gravity = sections["reward_inputs_contract"][
            "field_semantics_constraints"
        ]["projected_gravity"]
        self.assertEqual(
            projected_gravity["small_angle_xyz"],
            ["pitch", "-roll", "-1"],
        )
        guidance = " ".join(sections["selection_guidance"]).lower()
        self.assertIn("use current_best as the default parent", guidance)
        self.assertIn("keep each candidate attributable", guidance)
        raw_policy = sections["reward_inputs_contract"][
            "field_semantics_constraints"
        ]["raw_physical_magnitudes"]
        self.assertEqual(
            raw_policy["enforcement"],
            "active_reward_term_final_clip",
        )
        self.assertIn("clip_min", raw_policy["candidate_usage"])
        self.assertEqual(
            repr(sections).count("active_reward_term_final_clip"),
            1,
        )
        self.assertIn("authoritative over prose rationale", guidance)
        self.assertIn("maximum weighted contribution", guidance)
        self.assertNotIn("tune one existing mechanism", guidance)
        self.assertIn("goal_not_reached_detected is frequent", guidance)
        self.assertIn(
            "objective materiality and the guard cap are conjunctive",
            guidance,
        )
        self.assertIn("neither can compensate", guidance)
        self.assertEqual(
            sections["task"]["candidate_authoring_status"],
            "governed_by_shared_candidate_capabilities",
        )
        train_summary = sections["task"]["fixed_train_protocol_summary"]
        self.assertTrue(train_summary["complete_reward_runtime_available"])
        self.assertNotIn("candidate_can_change", train_summary)
        success_criteria = sections["task"]["fixed_eval_protocol_summary"][
            "success_criteria"
        ]
        self.assertEqual(
            success_criteria["completion_mode"],
            "stable_top_traversal_after_first_valid_top_arrival",
        )
        self.assertEqual(
            success_criteria["top_platform_stability_requirement"][
                "consecutive_control_steps"
            ],
            20,
        )
        self.assertEqual(
            success_criteria["top_platform_stability_requirement"][
                "maximum_support_effector_planar_speed_mps"
            ],
            0.5,
        )
        self.assertEqual(success_criteria["maximum_abs_roll_or_pitch_rad"], 0.65)
        self.assertEqual(
            success_criteria["minimum_kinematic_base_clearance_above_top_m"],
            0.45,
        )
        self.assertEqual(
            [
                field["name"]
                for field in sections["reward_inputs_contract"]["fields"]
            ],
            list(self.config.protocol_v2.complete_reward_runtime.field_names),
        )
        self.assertNotIn("raw_seed", str(sections))
        self.assertNotIn("seed_summaries", str(sections))
        self.assertNotIn("sealed_validation_summary", str(sections))
        self.assertNotIn("[redacted]", str(sections))
        prompt_feedback = projected.metrics["prompt_feedback"]
        self.assertEqual(
            prompt_feedback["schema_version"],
            "true_rl_agent.task.stair_climbing.prompt_feedback.v9",
        )
        self.assertEqual(
            set(prompt_feedback["mean_success_rate_by_bin"]),
            {str(item["bin_id"]) for item in self.config.protocol_v1.eval_bins},
        )
        self.assertEqual(
            prompt_feedback["generalization_summary"]["grid_axes"][
                "count_denominator"
            ],
            self.config.public_seed_count
            * self.config.evaluation_profile.episodes_per_seed_per_cell,
        )
        success_grid = prompt_feedback["generalization_summary"][
            "success_count_grid"
        ]
        self.assertEqual([len(row) for row in success_grid], [5] * 7)
        outcome_grids = prompt_feedback["generalization_summary"][
            "primary_failure_count_grids"
        ]
        self.assertEqual(set(outcome_grids), {"timeout"})
        self.assertNotIn(
            "success",
            prompt_feedback["generalization_summary"][
                "primary_failure_reasons"
            ],
        )
        self.assertTrue(
            all(
                rate == 0.5
                for rate in prompt_feedback["mean_success_rate_by_bin"].values()
            )
        )
        self.assertEqual(
            prompt_feedback["primary_failure_breakdown"],
            self.public_dev["generalization"]["primary_failure_breakdown"],
        )
        self.assertIn("dominant_failure", prompt_feedback)
        self.assertEqual(
            prompt_feedback["stable_success_summary"]["stable_top_success_rate"],
            0.5,
        )
        self.assertEqual(
            prompt_feedback["stable_success_summary"][
                "top_arrival_without_stable_success_rate"
            ],
            0.0,
        )
        self.assertEqual(
            prompt_feedback["diagnostic_flag_rates"][
                "goal_not_reached_detected"
            ],
            0.5,
        )
        fixture = StairClimbingComponentSmokeAdapter(
            self.config
        ).build_component_smoke_fixture()
        self.assertEqual(fixture.task_id, "unitree_stair_climbing")
        self.assertEqual(
            fixture.expected_reward_output_shape,
            (STAIR_SMOKE_NUM_ENVS,),
        )
        self.assertEqual(
            fixture.public_summary["input_schema_hash"],
            self.config.protocol_v2.complete_reward_runtime.input_schema_hash,
        )

    def test_component_smoke_env_axis_does_not_alias_input_axes(self) -> None:
        trailing_dimensions = {
            dimension
            for field in self.config.protocol_v2.complete_reward_runtime.input_fields
            for dimension in field.trailing_shape
        }

        self.assertNotIn(STAIR_SMOKE_NUM_ENVS, trailing_dimensions)

    def test_prompt_score_semantics_names_objective_guard_and_public_dev_roles(
        self,
    ) -> None:
        semantics = build_stair_prompt_score_semantics(
            self.config.acceptance_protocol
        )

        self.assertEqual(
            semantics,
            {
                "score_sources_by_role": {
                    "objective_score": "acceptance_objective.score_stats.mean",
                    "public_dev_objective_score": (
                        "generalization_public_dev.score_stats.mean"
                    ),
                    "guard_score": "public_accept.score_stats.mean",
                },
                "diagnostic_only_roles": ["public_dev_objective_score"],
            },
        )

    def test_prompt_feedback_keeps_r1_objective_guard_and_public_dev_distinct(
        self,
    ) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        projected["public_score"] = 0.575
        projected["metrics"]["acceptance"]["score_stats"]["mean"] = (
            0.5385714285714285
        )
        projected["metrics"]["prompt_feedback"]["generalization_summary"][
            "public_dev_objective_score"
        ] = 0.5607142857142857
        result_hash = canonical_json_hash(projected)
        candidate = copy.deepcopy(projected)

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[
                {
                    "round_id": "stair-round-r1-shape",
                    "candidate": {"candidate_id": "r1-shaped-candidate"},
                    "decision": {
                        "accepted": False,
                        "public_score": 0.57,
                        "comparison": {
                            "score_source": (
                                "acceptance_objective.score_stats.mean"
                            ),
                            "candidate_score": 0.52,
                            "guard_score_source": (
                                "public_accept.score_stats.mean"
                            ),
                            "candidate_guard_score": 0.57,
                        },
                    },
                    "public_result": candidate,
                }
            ],
            public_context={
                "initial_anchor_public_evidence": {
                    "anchor_kind": "reference_program",
                    "public_result": projected,
                    "public_result_hash": result_hash,
                }
            },
        )

        entries = sections["public_feedback_summary"]["feedback_entries"]
        reference, candidate_entry = entries
        self.assertEqual(
            reference["objective_score"],
            0.5385714285714285,
        )
        self.assertEqual(reference["guard_score"], 0.575)
        self.assertEqual(candidate_entry["objective_score"], 0.52)
        self.assertEqual(candidate_entry["guard_score"], 0.57)
        for entry in entries:
            self.assertNotIn("public_score", entry)
            generalization = entry["feedback"]["generalization_summary"]
            self.assertEqual(
                generalization["public_dev_objective_score"],
                0.5607142857142857,
            )
            self.assertNotIn("score", generalization)
        self.assertEqual(
            sections["public_eval_contract"]["prompt_score_semantics"],
            build_stair_prompt_score_semantics(self.config.acceptance_protocol),
        )
        self.assertNotIn(
            "objective_score_source",
            sections["public_eval_contract"],
        )
        self.assertNotIn(
            "guard_score_source",
            sections["public_eval_contract"],
        )

    def test_reference_anchor_feedback_seeds_first_candidate_prompt(self) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        )
        public_result = projected.to_public_dict()
        public_result_hash = canonical_json_hash(public_result)
        adapter = StairClimbingPromptAdapter(self.config)
        sections = adapter.build_prompt_sections(
            public_records=[],
            public_context={
                "initial_anchor_public_evidence": {
                    "anchor_kind": "reference_program",
                    "public_result": public_result,
                    "public_result_hash": public_result_hash,
                }
            },
        )
        summary = sections["public_feedback_summary"]
        self.assertEqual(
            summary["schema_version"],
            "true_rl_agent.task.stair_climbing.public_feedback_summary.v10",
        )
        self.assertEqual(len(summary["feedback_entries"]), 1)
        anchor_entry = summary["feedback_entries"][0]
        self.assertEqual(anchor_entry["source_kind"], "reference_anchor")
        self.assertEqual(anchor_entry["objective_score"], 0.5)
        self.assertEqual(anchor_entry["guard_score"], 0.5)
        self.assertNotIn("public_score", anchor_entry)
        self.assertEqual(
            anchor_entry["source_public_result_hash"],
            public_result_hash,
        )
        self.assertEqual(
            anchor_entry["feedback"],
            projected.metrics["prompt_feedback"],
        )
        self.assertNotIn("stair_evaluation", summary)

        candidate_feedback = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.prompt_feedback.v9"
            ),
            "source_split": "public_dev",
            "generalization_summary": {
                "public_dev_objective_score": 0.5,
            },
        }
        candidate_result = copy.deepcopy(public_result)
        candidate_result["metrics"].pop("public_dev")
        candidate_result["metrics"]["prompt_feedback"] = candidate_feedback
        sections = adapter.build_prompt_sections(
            public_records=[
                {
                    "round_id": "stair-round-001",
                    "candidate": {"candidate_id": "candidate-001"},
                    "decision": {
                        "accepted": False,
                        "public_score": 0.5,
                    },
                    "train_result": {
                        "program_source_checksum": "sha256:" + "1" * 64
                    },
                    "public_result": candidate_result,
                }
            ],
            public_context={
                "initial_anchor_public_evidence": {
                    "anchor_kind": "reference_program",
                    "public_result": public_result,
                    "public_result_hash": public_result_hash,
                }
            },
        )
        summary = sections["public_feedback_summary"]
        self.assertEqual(len(summary["feedback_entries"]), 2)
        candidate_entry = summary["feedback_entries"][-1]
        self.assertEqual(candidate_entry["candidate_id"], "candidate-001")
        self.assertIn(
            "best_observed_nonaccepted",
            candidate_entry["selection_basis"],
        )
        self.assertEqual(candidate_entry["feedback"], candidate_feedback)

    def test_unknown_prompt_feedback_schema_is_not_exposed(self) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        projected["metrics"]["prompt_feedback"] = {
            "schema_version": "test.stair.prompt_feedback.unknown",
            "generalization_summary": {"score": 0.99},
        }

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[
                _prompt_record(
                    "stair-round-001",
                    "unknown-feedback",
                    "1",
                    0.5,
                    projected,
                )
            ],
            public_context={},
        )

        self.assertEqual(
            sections["public_feedback_summary"].get("feedback_entries", []),
            [],
        )

    def test_conflicting_legacy_score_alias_is_not_exposed(self) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        projected["metrics"]["prompt_feedback"] = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.prompt_feedback.v8"
            ),
            "source_split": "public_dev",
            "generalization_summary": {
                "score": 0.1,
                "public_dev_objective_score": 0.9,
            },
        }

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[
                _prompt_record(
                    "stair-round-001",
                    "conflicting-feedback",
                    "1",
                    0.5,
                    projected,
                )
            ],
            public_context={},
        )

        self.assertEqual(
            sections["public_feedback_summary"].get("feedback_entries", []),
            [],
        )

    def test_prompt_feedback_exposes_arrival_without_stable_success(self) -> None:
        public_dev = copy.deepcopy(self.public_dev["anchor"])
        public_dev["primary_failure_breakdown"] = {
            "timeout": 0.9,
            "forbidden_contact": 0.05,
        }
        public_dev["diagnostic_flag_rates"] = {
            "goal_not_reached_detected": 0.06,
            "first_valid_top_arrival_achieved": 0.2,
        }

        feedback = build_stair_prompt_feedback(public_dev)

        self.assertAlmostEqual(
            feedback["stable_success_summary"][
                "top_arrival_without_stable_success_rate"
            ],
            0.15,
        )

    def test_prompt_feedback_nonaccepted_ties_use_recency_not_mean_progress(
        self,
    ) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        older_high_progress_result = copy.deepcopy(projected)
        older_high_progress_result["metrics"]["prompt_feedback"][
            "primary_failure_breakdown"
        ] = {
            "base_orientation_instability": 0.6,
            "forbidden_contact": 0.1,
        }
        older_high_progress_result["metrics"]["prompt_feedback"][
            "dominant_failure"
        ] = {
            "reason": "base_orientation_instability",
            "rate": 0.6,
        }
        older_high_progress_result["metrics"]["prompt_feedback"][
            "diagnostic_flag_rates"
        ] = {"goal_not_reached_detected": 0.75}
        older_high_progress_result["metrics"]["prompt_feedback"][
            "progress_summary"
        ][
            "mean_max_progress_m"
        ] = 0.64
        latest_low_progress_result = copy.deepcopy(projected)
        latest_low_progress_result["metrics"]["prompt_feedback"][
            "primary_failure_breakdown"
        ] = {
            "forbidden_contact": 1.0,
        }
        latest_low_progress_result["metrics"]["prompt_feedback"][
            "dominant_failure"
        ] = {
            "reason": "forbidden_contact",
            "rate": 1.0,
        }
        latest_low_progress_result["metrics"]["prompt_feedback"][
            "diagnostic_flag_rates"
        ] = {"goal_not_reached_detected": 1.0}
        latest_low_progress_result["metrics"]["prompt_feedback"][
            "progress_summary"
        ][
            "mean_max_progress_m"
        ] = 0.13
        records = [
            _prompt_record(
                "stair-round-001",
                "older-high-progress",
                "1",
                0.0,
                older_high_progress_result,
            ),
            _prompt_record(
                "stair-round-002",
                "latest-low-progress",
                "2",
                0.0,
                latest_low_progress_result,
            ),
        ]

        summary = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=records,
            public_context={},
        )["public_feedback_summary"]
        entries = {
            entry["candidate_id"]: entry
            for entry in summary["feedback_entries"]
        }

        self.assertEqual(
            set(entries),
            {"older-high-progress", "latest-low-progress"},
        )
        adapter = StairClimbingPromptAdapter(self.config)
        self.assertEqual(
            adapter.nonaccepted_authoring_tiebreak_key(records[0]),
            (),
        )
        self.assertEqual(
            adapter.nonaccepted_authoring_tiebreak_key(records[1]),
            (),
        )
        self.assertNotIn(
            "best_observed_nonaccepted",
            entries["older-high-progress"]["selection_basis"],
        )
        self.assertIn(
            "best_observed_nonaccepted",
            entries["latest-low-progress"]["selection_basis"],
        )
        self.assertEqual(
            entries["older-high-progress"]["feedback"]["dominant_failure"],
            {"reason": "base_orientation_instability", "rate": 0.6},
        )
        self.assertEqual(
            entries["older-high-progress"]["feedback"]["diagnostic_flag_rates"][
                "goal_not_reached_detected"
            ],
            0.75,
        )
        self.assertIn(
            "latest_nonaccepted",
            entries["latest-low-progress"]["selection_basis"],
        )
        self.assertEqual(
            entries["latest-low-progress"]["feedback"][
                "primary_failure_breakdown"
            ],
            {"forbidden_contact": 1.0},
        )

    def test_prompt_feedback_history_stays_bounded_without_losing_source_anchors(
        self,
    ) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        records = [
            _prompt_record(
                f"stair-round-{index:03d}",
                f"candidate-{index:03d}",
                f"{index % 10}",
                0.9 if index == 0 else 0.01 * index,
                copy.deepcopy(projected),
            )
            for index in range(15)
        ]
        records[4]["decision"]["accepted"] = True
        records[-1]["decision"]["gate_statuses"] = {
            "runtime_activation_gate": "pass"
        }
        records[-1]["runtime_activation_summary"] = {
            "reward_term_calls": 48001 * 18,
            "nonzero_reward_contribution": True,
            "capability_activation_records": ["complete_reward_program"],
            "component_activation_records": [
                {
                    "component_kind": "reward_term",
                    "name": f"candidate_term_{index}",
                    "call_count": 48001,
                    "observation_count": 2001,
                    "last_weighted_mean": 0.01 * index,
                    "max_abs_weighted_contribution": 0.1 + index,
                }
                for index in range(18)
            ],
        }

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=records,
            public_context={},
        )
        summary = sections["public_feedback_summary"]
        entries = summary["feedback_entries"]
        candidate_ids = {entry["candidate_id"] for entry in entries}

        self.assertLessEqual(len(entries), 5)
        self.assertIn("candidate-000", candidate_ids)
        self.assertIn("candidate-004", candidate_ids)
        self.assertIn("candidate-014", candidate_ids)
        self.assertEqual(len(candidate_ids), len(entries))
        self.assertTrue(summary["byte_budget_compaction_applied"])
        self.assertEqual(
            summary["omitted_candidate_feedback_entry_count"],
            len(records) - len(entries),
        )
        self.assertLessEqual(
            canonical_prompt_json_bytes(sections),
            MAX_TASK_PROMPT_SECTIONS_BYTES
            - summary["limits"]["reserved_task_prompt_headroom_bytes"],
        )

    def test_required_feedback_anchors_use_core_projection_before_overflow(
        self,
    ) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        feedback = projected["metrics"]["prompt_feedback"]

        def use_long_runtime_floats(value: object) -> object:
            if isinstance(value, dict):
                return {
                    key: (
                        item
                        if key
                        in {"success_count_grid", "primary_failure_count_grids"}
                        else use_long_runtime_floats(item)
                    )
                    for key, item in value.items()
                }
            if (
                not isinstance(value, bool)
                and isinstance(value, int | float)
                and float(value) != 0.0
            ):
                return 0.12345678901234568
            return value

        projected["metrics"]["prompt_feedback"] = use_long_runtime_floats(
            feedback
        )
        expected_progress_by_tread = projected["metrics"]["prompt_feedback"][
            "progress_summary"
        ]["mean_max_progress_m_by_tread"]
        self.assertEqual(len(expected_progress_by_tread), 5)
        expected_compact_progress_by_tread = {
            key: round(float(value), 8)
            for key, value in expected_progress_by_tread.items()
        }
        records = [
            _prompt_record(
                "stair-round-001",
                "accepted-current-best",
                "1",
                0.9,
                copy.deepcopy(projected),
            ),
            _prompt_record(
                "stair-round-002",
                "best-observed-nonaccepted",
                "2",
                0.95,
                copy.deepcopy(projected),
            ),
            _prompt_record(
                "stair-round-003",
                "latest-evaluated",
                "3",
                0.8,
                copy.deepcopy(projected),
            ),
        ]
        records[0]["decision"]["accepted"] = True
        records[-1]["decision"]["gate_statuses"] = {
            "runtime_activation_gate": "pass"
        }
        records[-1]["runtime_activation_summary"] = {
            "reward_term_calls": 48001 * 17,
            "nonzero_reward_contribution": True,
            "capability_activation_records": ["complete_reward_program"],
            "component_activation_records": [
                {
                    "component_kind": "reward_term",
                    "name": f"runtime_term_{index}",
                    "call_count": 48001,
                    "observation_count": 2001,
                    "last_weighted_mean": 0.12345678901234568,
                    "max_abs_weighted_contribution": (
                        0.12345678901234568 + index
                    ),
                }
                for index in range(17)
            ],
        }

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=records,
            public_context={},
        )
        summary = sections["public_feedback_summary"]
        entries = summary["feedback_entries"]

        self.assertEqual(
            {entry["candidate_id"] for entry in entries},
            {
                "accepted-current-best",
                "best-observed-nonaccepted",
                "latest-evaluated",
            },
        )
        self.assertTrue(summary["byte_budget_compaction_applied"])
        self.assertEqual(
            summary["feedback_entry_projection"],
            {
                "detail_level": "core",
                "schema_version": (
                    "true_rl_agent.task.stair_climbing."
                    "feedback_entry_projection.v6"
                ),
                "source_split": "public_dev",
                "omitted_rates_are_zero": True,
                "generalization_grid": {
                    "layout": "height_rows_x_tread_columns",
                    "height_bin_ids": [
                        cell.bin_id
                        for cell in self.config.evaluation_profile.anchor_cells
                    ],
                    "tread_depths_m": list(
                        self.config.evaluation_profile.tread_depths_m
                    ),
                    "value_encoding": "episode_count",
                    "count_denominator": 24,
                },
                "count_grid_string_encoding": {
                    "fields": [
                        "success_count_grid_string",
                        "primary_failure_count_grid_strings.<reason>",
                    ],
                    "row_order": "generalization_grid.height_bin_ids",
                    "column_order": "generalization_grid.tread_depths_m",
                    "row_separator": ";",
                    "column_separator": ",",
                    "value_encoding": "episode_count",
                },
            },
        )
        for entry in entries:
            compact = entry["feedback"]
            self.assertNotIn("schema_version", compact)
            self.assertNotIn("source_split", compact)
            self.assertIn("dominant_failure", compact)
            self.assertNotIn(
                "difficulty_diagnostics",
                compact["generalization_summary"],
            )
            encoded_grid = compact["generalization_summary"][
                "success_count_grid_string"
            ]
            grid = [
                [int(value) for value in row.split(",")]
                for row in encoded_grid.split(";")
            ]
            self.assertEqual(len(grid), 7)
            self.assertTrue(all(len(row) == 5 for row in grid))
            self.assertNotIn(
                "success_rate_by_cell",
                compact["generalization_summary"],
            )
            self.assertEqual(
                compact["progress_summary"]["mean_max_progress_m_by_tread"],
                expected_compact_progress_by_tread,
            )
            self.assertNotIn(
                "mean_max_progress_m",
                compact["progress_summary"],
            )
            self.assertNotIn(
                "first_valid_top_arrival_achieved",
                compact["diagnostic_flag_rates"],
            )
            self.assertTrue(
                all(
                    float(rate) != 0.0
                    for rate in compact["primary_failure_breakdown"].values()
                )
            )
        self.assertLessEqual(
            canonical_prompt_json_bytes(sections),
            MAX_TASK_PROMPT_SECTIONS_BYTES
            - summary["limits"]["reserved_task_prompt_headroom_bytes"],
        )

    def test_required_eval_v8_feedback_is_normalized_before_compaction(self) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        legacy = copy.deepcopy(projected)
        feedback = legacy["metrics"]["prompt_feedback"]
        feedback["schema_version"] = (
            "true_rl_agent.task.stair_climbing.prompt_feedback.v8"
        )
        generalization = feedback["generalization_summary"]
        generalization["score"] = generalization.pop(
            "public_dev_objective_score"
        )
        records = [
            _prompt_record(
                "stair-round-001",
                "legacy-latest",
                "1",
                0.5,
                copy.deepcopy(legacy),
            )
        ]
        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=records,
            public_context={
                "initial_anchor_public_evidence": {
                    "anchor_kind": "reference_program",
                    "public_result": copy.deepcopy(legacy),
                    "public_result_hash": canonical_json_hash(legacy),
                }
            },
        )

        summary = sections["public_feedback_summary"]
        entries = summary["feedback_entries"]
        self.assertEqual(len(entries), 2)
        for entry in entries:
            compact_feedback = entry["feedback"]
            compact_generalization = compact_feedback["generalization_summary"]
            self.assertIn(
                "public_dev_objective_score",
                compact_generalization,
            )
            self.assertNotIn("score", compact_generalization)
        self.assertEqual(
            summary["feedback_entry_projection"]["schema_version"],
            "true_rl_agent.task.stair_climbing.feedback_entry_projection.v6",
        )
        self.assertLessEqual(
            canonical_prompt_json_bytes(sections),
            MAX_TASK_PROMPT_SECTIONS_BYTES
            - summary["limits"]["reserved_task_prompt_headroom_bytes"],
        )

    def test_small_eval_v8_feedback_is_normalized_without_compaction(self) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        legacy = copy.deepcopy(projected)
        legacy["metrics"]["prompt_feedback"] = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.prompt_feedback.v8"
            ),
            "source_split": "public_dev",
            "generalization_summary": {"score": 0.205},
        }
        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[
                _prompt_record(
                    "stair-round-001",
                    "legacy-small",
                    "1",
                    0.5,
                    legacy,
                )
            ],
            public_context={},
        )

        summary = sections["public_feedback_summary"]
        self.assertFalse(summary["byte_budget_compaction_applied"])
        feedback = summary["feedback_entries"][0]["feedback"]
        self.assertEqual(
            feedback["schema_version"],
            "true_rl_agent.task.stair_climbing.prompt_feedback.v9",
        )
        self.assertEqual(
            feedback["generalization_summary"],
            {"public_dev_objective_score": 0.205},
        )

    def test_prompt_overflow_summarizes_only_historical_comparison(self) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        generalization = projected["metrics"]["prompt_feedback"][
            "generalization_summary"
        ]
        generalization["primary_failure_count_grids"] = {
            reason: [[1] * 5 for _ in range(7)]
            for reason in self.config.episode_outcomes()["failure_priority"]
        }
        generalization["zero_count_grids_omitted"] = []
        records = [
            _prompt_record(
                f"stair-round-{'a' * 90}",
                "a" * 70,
                "1",
                0.9,
                copy.deepcopy(projected),
            ),
            _prompt_record(
                f"stair-round-{'b' * 90}",
                "b" * 70,
                "2",
                0.95,
                copy.deepcopy(projected),
            ),
            _prompt_record(
                f"stair-round-{'c' * 90}",
                "c" * 70,
                "3",
                0.8,
                copy.deepcopy(projected),
            ),
        ]
        records[0]["decision"]["accepted"] = True
        records[-1]["decision"]["gate_statuses"] = {
            "runtime_activation_gate": "pass"
        }
        records[-1]["runtime_activation_summary"] = {
            "reward_term_calls": 48001 * 17,
            "nonzero_reward_contribution": True,
            "capability_activation_records": ["complete_reward_program"],
            "component_activation_records": [
                {
                    "component_kind": "reward_term",
                    "name": f"runtime_term_{index}",
                    "call_count": 48001,
                    "observation_count": 2001,
                    "last_weighted_mean": 0.12345678901234568,
                    "max_abs_weighted_contribution": (
                        0.12345678901234568 + index
                    ),
                }
                for index in range(17)
            ],
        }

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=records,
            public_context={},
        )
        summary = sections["public_feedback_summary"]
        entries = {
            entry["candidate_id"][0]: entry
            for entry in summary["feedback_entries"]
        }

        self.assertEqual(set(entries), {"a", "b", "c"})
        self.assertEqual(
            summary["feedback_entry_projection"][
                "historical_comparison_detail_level"
            ],
            "summary",
        )
        for role in ("a", "c"):
            generalization = entries[role]["feedback"][
                "generalization_summary"
            ]
            self.assertIn("success_count_grid_string", generalization)
            self.assertEqual(
                set(generalization["primary_failure_count_grid_strings"]),
                set(self.config.episode_outcomes()["failure_priority"]),
            )
            self.assertNotIn("detail_omissions", entries[role])
        historical = entries["b"]
        self.assertEqual(
            historical["detail_omissions"],
            [
                "historical_cell_grids",
                "historical_completion_and_diagnostic_summaries",
                "historical_progress_summary",
            ],
        )
        self.assertEqual(
            set(historical["feedback"]),
            {
                "dominant_failure",
                "generalization_summary",
                "primary_failure_breakdown",
            },
        )
        self.assertEqual(
            set(historical["feedback"]["generalization_summary"]),
            {"public_dev_objective_score"},
        )
        self.assertLessEqual(
            canonical_prompt_json_bytes(sections),
            MAX_TASK_PROMPT_SECTIONS_BYTES
            - summary["limits"]["reserved_task_prompt_headroom_bytes"],
        )

    def test_legal_identity_maxima_fit_required_feedback_under_adversarial_load(
        self,
    ) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        feedback = projected["metrics"]["prompt_feedback"]
        generalization = feedback["generalization_summary"]
        generalization["primary_failure_count_grids"] = {
            reason: [[1] * 5 for _ in range(7)]
            for reason in self.config.episode_outcomes()["failure_priority"]
        }
        generalization["zero_count_grids_omitted"] = []

        def use_long_floats(value: object) -> object:
            if isinstance(value, dict):
                return {
                    key: (
                        item
                        if key
                        in {"success_count_grid", "primary_failure_count_grids"}
                        else use_long_floats(item)
                    )
                    for key, item in value.items()
                }
            if (
                not isinstance(value, bool)
                and isinstance(value, int | float)
                and float(value) != 0.0
            ):
                return 0.12345678901234568
            return value

        projected["metrics"]["prompt_feedback"] = use_long_floats(feedback)
        candidate_ids = [character * 80 for character in "abcde"]
        records = [
            _prompt_record(
                f"stair-round-{index + 1:03d}",
                candidate_id,
                str(index + 1),
                (0.9, 0.95, 0.7, 0.75, 0.8)[index],
                copy.deepcopy(projected),
            )
            for index, candidate_id in enumerate(candidate_ids)
        ]
        records[0]["decision"]["accepted"] = True
        records[-1]["decision"]["gate_statuses"] = {
            "runtime_activation_gate": "pass"
        }
        records[-1]["runtime_activation_summary"] = {
            "reward_term_calls": 48001 * 17,
            "nonzero_reward_contribution": True,
            "capability_activation_records": ["complete_reward_program"],
            "component_activation_records": [
                {
                    "component_kind": "reward_term",
                    "name": (
                        f"{index:02d}"
                        + "n" * (MAX_CANONICAL_COMPONENT_NAME_CHARS - 2)
                    ),
                    "call_count": 48001,
                    "observation_count": 2001,
                    "last_weighted_mean": 0.12345678901234568,
                    "max_abs_weighted_contribution": (
                        index + 0.12345678901234568
                    ),
                }
                for index in range(17)
            ],
        }

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=records,
            public_context={},
        )
        summary = sections["public_feedback_summary"]
        target = (
            MAX_TASK_PROMPT_SECTIONS_BYTES
            - summary["limits"]["reserved_task_prompt_headroom_bytes"]
        )

        self.assertEqual(
            {entry["candidate_id"] for entry in summary["feedback_entries"]},
            {candidate_ids[0], candidate_ids[-1]},
        )
        self.assertTrue(summary["byte_budget_compaction_applied"])
        self.assertLessEqual(canonical_prompt_json_bytes(sections), target)

    def test_reference_and_latest_feedback_retain_compact_failure_grids(
        self,
    ) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "environment_source_preflight": self.source_preflight,
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    },
                },
            ),
            train_summary=_train_summary(self.config),
        ).to_public_dict()
        record = _prompt_record(
            "stair-round-001",
            "latest-evaluated",
            "1",
            0.4,
            copy.deepcopy(projected),
        )
        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[record],
            public_context={
                "initial_anchor_public_evidence": {
                    "anchor_kind": "reference_program",
                    "public_result": copy.deepcopy(projected),
                    "public_result_hash": canonical_json_hash(projected),
                }
            },
        )
        summary = sections["public_feedback_summary"]
        self.assertTrue(summary["byte_budget_compaction_applied"])
        self.assertEqual(len(summary["feedback_entries"]), 2)
        expected_timeout_grid = projected["metrics"]["prompt_feedback"][
            "generalization_summary"
        ]["primary_failure_count_grids"]["timeout"]
        expected_success_grid = projected["metrics"]["prompt_feedback"][
            "generalization_summary"
        ]["success_count_grid"]
        for entry in summary["feedback_entries"]:
            self.assertNotIn("detail_omissions", entry)
            generalization = entry["feedback"]["generalization_summary"]
            success_rows = generalization["success_count_grid_string"].split(
                ";"
            )
            self.assertEqual(
                [
                    [int(value) for value in row.split(",")]
                    for row in success_rows
                ],
                expected_success_grid,
            )
            encoded = generalization["primary_failure_count_grid_strings"]
            self.assertEqual(set(encoded), {"timeout"})
            rows = encoded["timeout"].split(";")
            self.assertEqual(len(rows), 7)
            self.assertEqual(
                [[int(value) for value in row.split(",")] for row in rows],
                expected_timeout_grid,
            )
        self.assertLessEqual(
            canonical_prompt_json_bytes(sections),
            MAX_TASK_PROMPT_SECTIONS_BYTES
            - summary["limits"]["reserved_task_prompt_headroom_bytes"],
        )

    def test_prompt_projects_latest_validated_complete_reward_runtime(self) -> None:
        components = [
            {
                "component_kind": "reward_term",
                "name": f"reference_term_{index}",
                "call_count": 48001,
                "observation_count": 2001,
                "last_weighted_mean": 0.01,
                "max_abs_weighted_contribution": 0.1 + index,
            }
            for index in range(10)
        ]
        components.append(
            {
                "component_kind": "reward_term",
                "name": "late_dominant_complete_reward_term",
                "call_count": 48001,
                "observation_count": 2001,
                "last_weighted_mean": 2.0,
                "max_abs_weighted_contribution": 120.0,
            }
        )
        record = {
            "round_id": "stair-round-runtime",
            "candidate": {"candidate_id": "stair-runtime-candidate"},
            "train_result": {"job_id": "train-runtime"},
            "public_result": {"status": "pass", "metrics": {}},
            "decision": {
                "reason": "noise_aware_delta_not_met",
                "gate_statuses": {"runtime_activation_gate": "pass"},
            },
            "runtime_activation_summary": {
                "reward_term_calls": 528011,
                "nonzero_reward_contribution": True,
                "capability_activation_records": ["complete_reward_program"],
                "component_activation_records": components,
            },
        }

        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[record],
            public_context={},
        )
        runtime = sections["runtime_feedback_summary"]
        activation = runtime["latest_runtime_activation"]

        self.assertEqual(runtime["status"], "available")
        self.assertEqual(runtime["source_round_id"], "stair-round-runtime")
        self.assertEqual(runtime["source_candidate_id"], "stair-runtime-candidate")
        self.assertEqual(activation["component_count"], 11)
        self.assertEqual(activation["omitted_component_count"], 0)
        self.assertEqual(
            activation["components"][0]["name"],
            "late_dominant_complete_reward_term",
        )
        incomplete = copy.deepcopy(record)
        incomplete["decision"]["gate_statuses"]["runtime_activation_gate"] = (
            "fail"
        )
        runtime = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[incomplete],
            public_context={},
        )["runtime_feedback_summary"]
        self.assertEqual(runtime["status"], "not_available")

def _prompt_record(
    round_id: str,
    candidate_id: str,
    checksum_digit: str,
    public_score: float,
    public_result: dict[str, object],
) -> dict[str, object]:
    return {
        "round_id": round_id,
        "candidate": {"candidate_id": candidate_id},
        "candidate_surface": {"classification": "code_level"},
        "decision": {
            "accepted": False,
            "reason": "required_gate_failed",
            "public_score": public_score,
            "comparison": {"candidate_score": public_score},
        },
        "train_result": {
            "program_source_checksum": "sha256:" + checksum_digit * 64
        },
        "public_result": public_result,
    }


def _summary(config: StairClimbingTaskConfig, split: str) -> dict[str, object]:
    seed_count = {
        "public_dev": config.public_seed_count,
        "public_accept": config.public_accept_seed_count,
        "sealed_validation": config.sealed_validation_seed_count,
    }[split]
    per_seed = []
    for _ in range(seed_count):
        outcomes = []
        for env_index, cell in enumerate(
            config.evaluation_profile.private_cell_assignment()
        ):
            success = env_index % 2 == 0
            outcomes.append(
                StairEpisodeOutcome(
                    env_index=env_index,
                    bin_id=cell.bin_id,
                    episode_steps=600,
                    primary_reason="success" if success else "timeout",
                    success=success,
                    diagnostic_flags=(
                        ("first_valid_top_arrival_achieved",)
                        if success
                        else ("goal_not_reached_detected",)
                    ),
                    max_progress_m=cell.progress_goal_m if success else 0.5,
                    progress_fraction=1.0 if success else 0.11494253,
                    active_terminal_terms=() if success else ("time_out",),
                    evaluation_cell_id=cell.cell_id,
                )
            )
        per_seed.append(outcomes)
    anchor_ids = {cell.cell_id for cell in config.evaluation_profile.anchor_cells}
    anchor_outcomes = [
        [
            outcome
            for outcome in seed_outcomes
            if outcome.evaluation_cell_id in anchor_ids
        ]
        for seed_outcomes in per_seed
    ]
    return {
        "anchor": build_stair_split_summary(
            config.protocol_v1,
            semantics=config.evaluation_semantics,
            evaluation_protocol=config.evaluation_protocol(),
            split=split,
            per_seed_outcomes=anchor_outcomes,
        ),
        "generalization": build_stair_generalization_summary(
            profile=config.evaluation_profile,
            semantics=config.evaluation_semantics,
            evaluation_protocol=config.evaluation_protocol(),
            split=split,
            per_seed_outcomes=per_seed,
        ),
    }


def _train_summary(config: StairClimbingTaskConfig) -> dict[str, object]:
    return {
        "environment_source_preflight": _source_preflight(config),
        "training": {
            "budget_accounting": build_budget_accounting(
                budget_steps=config.budget_steps,
                num_envs=config.train_num_envs,
                num_steps_per_env=config.train_steps_per_iteration,
                max_iterations=config.train_max_iterations,
            )
        },
    }


def _source_preflight(config: StairClimbingTaskConfig) -> dict[str, str]:
    contract = config.protocol_v2.runner_owned_training_distribution
    if contract is None:
        raise ValueError("Stair test config requires a training distribution")
    upstream = stair_training_distribution_upstream_source_evidence(contract)
    return build_stair_execution_source_preflight_summary(
        task_protocol_hash=config.task_protocol_hash,
        source_manifest_hash=str(upstream["source_manifest_hash"]),
    )


if __name__ == "__main__":
    unittest.main()
