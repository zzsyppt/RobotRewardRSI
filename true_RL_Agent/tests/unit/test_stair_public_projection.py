from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.public_result import PublicEvalResult  # noqa: E402
from true_rl_agent.runner.budget import build_budget_accounting  # noqa: E402
from true_rl_agent.tasks.stair_climbing.component_smoke import (  # noqa: E402
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
    build_stair_split_summary,
    validate_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.memory import (  # noqa: E402
    StairTaskMemoryAdapter,
)
from true_rl_agent.tasks.stair_climbing.prompt_adapter import (  # noqa: E402
    StairClimbingPromptAdapter,
)
from true_rl_agent.tasks.stair_climbing.public_projection import (  # noqa: E402
    project_stair_public_result,
)


class StairPublicProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
        )
        self.public_dev = _summary(self.config, "public_dev")
        self.public_accept = _summary(self.config, "public_accept")

    def test_split_validator_recomputes_statistics_and_closed_counts(self) -> None:
        canonical = validate_stair_split_summary(
            self.config.protocol_v1,
            evaluation_protocol=self.config.evaluation_protocol(),
            split="public_accept",
            summary=self.public_accept,
        )
        self.assertEqual(canonical["score_stats"]["mean"], 0.5)
        self.assertEqual(
            sum(canonical["primary_outcome_counts"].values()),
            canonical["total_episode_count"],
        )

        changed = copy.deepcopy(self.public_accept)
        changed["score_stats"]["mean"] = 0.9
        with self.assertRaisesRegex(ValueError, "score mean"):
            validate_stair_split_summary(
                self.config.protocol_v1,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                summary=changed,
            )

        changed = copy.deepcopy(self.public_accept)
        changed["primary_outcome_counts"]["success"] += 1
        changed["primary_outcome_counts"]["timeout"] -= 1
        with self.assertRaisesRegex(ValueError, "success counts"):
            validate_stair_split_summary(
                self.config.protocol_v1,
                evaluation_protocol=self.config.evaluation_protocol(),
                split="public_accept",
                summary=changed,
            )

    def test_projection_overwrites_forged_task_gates(self) -> None:
        raw = PublicEvalResult(
            status="pass",
            public_score=0.5,
            metrics={
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
            projected.metrics["runtime_activation_gate"]["status"],
            "fail",
        )
        self.assertEqual(
            projected.metrics["acceptance"]["required_pass_gates"],
            self.config.protocol_v1.to_dict()["acceptance"]["required_pass_gates"],
        )
        self.assertNotIn("stair_evaluation", projected.metrics)

    def test_projection_rejects_score_and_budget_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "public score"):
            project_stair_public_result(
                config=self.config,
                public_result=PublicEvalResult(
                    status="pass",
                    public_score=0.7,
                    metrics={
                        "stair_evaluation": {
                            "public_dev": self.public_dev,
                            "public_accept": self.public_accept,
                        }
                    },
                ),
                train_summary=_train_summary(self.config),
            )

        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
                    "stair_evaluation": {
                        "public_dev": self.public_dev,
                        "public_accept": self.public_accept,
                    }
                },
            ),
            train_summary={
                "training": {
                    "budget_accounting": {
                        **_train_summary(self.config)["training"]["budget_accounting"],
                        "budget_steps": self.config.budget_steps + 1,
                    }
                }
            },
        )
        self.assertEqual(projected.metrics["budget_accounting"]["status"], "fail")

    def test_prompt_memory_and_phase_b_smoke_remain_public_and_fail_closed(self) -> None:
        projected = project_stair_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=0.5,
                metrics={
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
        self.assertEqual(memory["macro_success"]["latest"], 0.5)
        sections = StairClimbingPromptAdapter(self.config).build_prompt_sections(
            public_records=[record],
            public_context={"task_public_summary": memory},
        )
        self.assertEqual(sections["reward_inputs_contract"]["status"], "not_available")
        self.assertNotIn("raw_seed", str(sections))
        self.assertNotIn("[redacted]", str(sections))
        self.assertIsNone(
            StairClimbingComponentSmokeAdapter(
                self.config
            ).build_component_smoke_fixture()
        )


def _summary(config: StairClimbingTaskConfig, split: str) -> dict[str, object]:
    seed_count = {
        "public_dev": config.public_seed_count,
        "public_accept": config.public_accept_seed_count,
    }[split]
    per_seed = []
    for _ in range(seed_count):
        outcomes = []
        for env_index in range(config.eval_num_envs):
            bin_id = str(config.protocol_v1.eval_bins[env_index // 8]["bin_id"])
            success = env_index % 2 == 0
            outcomes.append(
                StairEpisodeOutcome(
                    env_index=env_index,
                    bin_id=bin_id,
                    episode_steps=600,
                    primary_reason="success" if success else "timeout",
                    success=success,
                    diagnostic_flags=() if success else ("stalled_progress_detected",),
                    max_progress_m=4.35 if success else 0.5,
                    progress_fraction=1.0 if success else 0.11494253,
                    active_terminal_terms=("time_out",),
                )
            )
        per_seed.append(outcomes)
    return build_stair_split_summary(
        config.protocol_v1,
        evaluation_protocol=config.evaluation_protocol(),
        split=split,
        per_seed_outcomes=per_seed,
    )


def _train_summary(config: StairClimbingTaskConfig) -> dict[str, object]:
    return {
        "training": {
            "budget_accounting": build_budget_accounting(
                budget_steps=config.budget_steps,
                num_envs=config.train_num_envs,
                num_steps_per_env=config.train_steps_per_iteration,
                max_iterations=config.train_max_iterations,
            )
        }
    }


if __name__ == "__main__":
    unittest.main()
