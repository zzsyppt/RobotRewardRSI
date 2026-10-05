"""Only the qualified per-foot periods consumed by the behavior description."""
from statistics import median


def episode_periods(periods_by_entity):
    """Give each episode one median, regardless of its qualified cycle count."""
    return {entity: median(periods) if periods else None for entity, periods in periods_by_entity.items()}


def summarize_periods(episodes):
    groups = {}
    for entity in episodes[0]:
        values = [e[entity] for e in episodes if e[entity] is not None]
        groups["contact_cycles:"+entity] = {"episodes":len(episodes),
            "episodes_with_qualified_windows":len(values),
            "episode_median_qualified_cycle_period_s":{"median":median(values) if values else None}}
    return {"groups":groups}
