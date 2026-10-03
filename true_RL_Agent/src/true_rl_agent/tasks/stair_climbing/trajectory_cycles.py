"""Only the qualified per-foot periods consumed by the behavior description."""
from statistics import median
from true_rl_agent.evaluation.trajectory_translation.cohort import distribution


def episode_periods(records):
    result = {}
    for r in records:
        v = r["value"]
        periods = [v["complete_cycles"][i]["raw_inter_onset_interval_s"] for i in v["qualified_cycle_indices"]]
        result[r["entity_ids"][0]] = median(periods) if periods else None
    return result


def summarize_periods(episodes):
    groups = {}
    for entity in episodes[0]:
        values = [e[entity] for e in episodes if e[entity] is not None]
        groups["contact_cycles:"+entity] = {"episodes":len(episodes),
            "episodes_with_qualified_windows":len(values),
            "episode_median_qualified_cycle_period_s":distribution(values)}
    return {"groups":groups}
