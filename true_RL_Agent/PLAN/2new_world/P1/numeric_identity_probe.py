"""P1 acceptance probe; CPU only, prints hashes for the same raw episode facts."""
from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.derived_statistics import sample_statistics
from true_rl_agent.evaluation.acceptance_protocol import acceptance_threshold
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_unified_stair_summary, validate_unified_stair_summary,
    build_unified_score_projection,
)


def probe() -> dict:
    rows = {}
    for protocol in ("stair_unified_protocol_v1.json", "stair_go2_12dof_unified_protocol_v2.json"):
        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks" / protocol)
        suite = config.ordered_evaluation_suite
        assignment = suite.private_cell_assignment("public_accept")
        seeds = []
        for seed_index in range(config.public_accept_seed_count):
            outcomes = []
            for index, cell in enumerate(assignment):
                success = (index + seed_index) % 5 < 2
                progress = 1.0 if success else ((index * 7 + seed_index * 3) % 19) / 19.0
                outcomes.append(StairEpisodeOutcome(
                    env_index=index, bin_id=cell.cell_id,
                    episode_steps=cell.episode_horizon_control_steps,
                    primary_reason="success" if success else "timeout", success=success,
                    diagnostic_flags=(), max_progress_m=progress * cell.progress_goal_m,
                    progress_fraction=progress, active_terminal_terms=() if success else ("time_out",),
                    evaluation_cell_id=cell.cell_id,
                ))
            seeds.append(outcomes)
        raw = [[asdict(outcome) for outcome in seed] for seed in seeds]
        summary = build_unified_stair_summary(suite=suite, semantics=config.evaluation_semantics, split="public_accept", per_seed_outcomes=seeds)
        canonical = validate_unified_stair_summary(suite=suite, semantics=config.evaluation_semantics, split="public_accept", summary=json.loads(json.dumps(summary)))
        if canonical != summary:
            raise AssertionError("summary changed on read")
        legacy = json.loads(json.dumps(summary))
        legacy["schema_version"] = "true_rl_agent.task.stair_climbing.unified_evaluation_summary.v1"
        legacy_ordered = legacy["ordered_summary"]
        legacy_ordered["schema_version"] = "true_rl_agent.task.stair_climbing.ordered_evaluation_summary.v2"
        legacy_stats = legacy_ordered["score_stats"]
        legacy_stats.pop("statistics_version")
        # Historical readers preserve the stored one-ULP representation.
        legacy_stats["std"] = math.nextafter(legacy_stats["std"], math.inf)
        legacy_hash = canonical_json_hash(legacy)
        restored_legacy = validate_unified_stair_summary(
            suite=suite, semantics=config.evaluation_semantics,
            split="public_accept", summary=legacy,
        )
        if canonical_json_hash(restored_legacy) != legacy_hash:
            raise AssertionError("legacy evidence was rewritten")
        projection = build_unified_score_projection(summary)
        values = projection["j_primary"]["seed_macro_scores"]
        best = [value * 0.8 for value in values]
        stats = sample_statistics(values)
        threshold = acceptance_threshold(config.acceptance_protocol, candidate_score=stats["mean"], best_score=sample_statistics(best)["mean"], candidate_score_samples=values, best_score_samples=best, statistics_version=stats["statistics_version"])
        rows[protocol] = {
            "raw_hash": canonical_json_hash(raw), "summary_hash": canonical_json_hash(summary),
            "projection_hash": canonical_json_hash(projection),
            "legacy_read_hash": legacy_hash,
            "paired_threshold": threshold.to_public_dict(),
            "stats": stats,
        }
    stress = sample_statistics([0.13690476190476192, 0.303968253968254, 0.44920634920634916, 0.9])
    body = {"rows": rows, "stress_statistics": stress}
    return {"python": platform.python_version(), **body, "canonical_identity": canonical_json_hash(body)}


if __name__ == "__main__":
    print(json.dumps(probe(), sort_keys=True, indent=2))
