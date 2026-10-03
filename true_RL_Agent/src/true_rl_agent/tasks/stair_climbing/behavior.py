"""Current behavior projection and explicitly bound, condition-matched comparison."""
from collections import Counter
from statistics import median
from typing import Mapping

from true_rl_agent.evaluation.process_report import format_number
from .trajectory_production import PUBLIC_SCHEMA, FULL_SCHEMA, replay_production_report

COMPARISON_VERSION = "stair.behavior_comparison.v1"
COMPARISON_METRICS = {"all": ("net", "speed", "active"),
                      "terminal": ("progress", "forward_speed", "height", "tilt_end")}


def project_stair_behavior(value):
    return replay_production_report(value)


def comparison_operands(episodes):
    """Reuse measured facts; store no per-episode ledger or alternate definitions."""
    descriptors = {}
    def facts(members, window):
        result = {}
        for key in COMPARISON_METRICS[window]:
            entries = [e["facts"][window][key] for e in members]
            definition = {"label": entries[0]["label"], "unit": entries[0]["unit"],
                          "statistic": "episode_median", "window": ("endpoint" if key in {"progress","tilt_end"} else "terminal_0.5s_complete" if window == "terminal" else "whole_episode")}
            if any((e["label"], e["unit"]) != (definition["label"], definition["unit"]) for e in entries):
                raise ValueError("inconsistent comparison fact definitions")
            descriptors[window+"."+key] = definition
            # Endpoint quantities remain valid for short episodes. Window means
            # require a complete common window and finite samples.
            accepted = [entry for episode,entry in zip(members,entries) if entry["value"] is not None
                and (window != "terminal" or key in {"progress","tilt_end"}
                     or episode["terminal_motion_complete"] and entry["unknown"] == 0)]
            values = [e["value"] for e in accepted]
            result[key] = [median(values) if values else None, len(values),
                           sum(e["unknown"] for e in entries), sum(e["not_applicable"] for e in entries)]
        return result
    grouped = {}
    for e in episodes:
        grouped.setdefault(e["cell"], []).append(e)
    conditions = {}
    for cell, members in sorted(grouped.items()):
        outcomes = dict(Counter(e["outcome"]["primary_reason"] for e in members))
        terminal = {}
        for outcome, n in sorted(outcomes.items()):
            terminal[outcome] = {"n":n, "facts":facts([e for e in members if e["outcome"]["primary_reason"] == outcome], "terminal")}
        conditions[cell] = {"stratum":members[0]["stratum"], "n":len(members),
            "outcomes":outcomes, "all":facts(members,"all"), "terminal":terminal}
    return {"version":COMPARISON_VERSION,"descriptors":descriptors,"conditions":conditions,
            "fact_columns":["value","valid_episodes","unknown_samples","not_applicable_samples"],
            "contact_threshold_n":episodes[0]["contact_threshold_n"], "terminal_window_s":.5}


def compare_stair_behavior(parent, candidate, *, expected_parent_checkpoint_hash,
                           parent_comparison=None, candidate_comparison=None):
    if not expected_parent_checkpoint_hash or (not isinstance(parent, Mapping) and parent_comparison is None):
        return {"status":"not_available","reason":"missing_explicit_parent",
                "text":"未提供显式父代描述，不能计算父代差分。"}
    left = project_stair_behavior(parent) if isinstance(parent, Mapping) else {"domain":parent_comparison["domain"]}
    right = project_stair_behavior(candidate)
    if left["domain"]["checkpoint_hash"] != expected_parent_checkpoint_hash:
        raise ValueError("explicit parent checkpoint identity mismatch")
    if parent_comparison is None or candidate_comparison is None:
        return {"status":"not_available","reason":"comparison_operands_not_recorded",
                "text":"已提供显式父代，但一侧未记录当前比较所需摘要。"}
    a, b = parent_comparison, candidate_comparison
    if a.get("domain") != left["domain"] or b.get("domain") != right["domain"]:
        raise ValueError("comparison operands/report source identity mismatch")
    fields = ("task_id","task_protocol_hash","evaluation_binding_hash","seed_namespace","projection_hash","catalog_version")
    unequal = [k for k in fields if left["domain"][k] != right["domain"][k]]
    unequal += [k for k in ("version","descriptors","fact_columns","contact_threshold_n","terminal_window_s") if a[k] != b[k]]
    if unequal:
        return {"status":"not_comparable","reason":"comparison_domain_or_definition_mismatch", "fields":unequal,
                "text":"父代与当前的协议、投影或描述定义不同，不计算差值。"}
    ac, bc = a["conditions"], b["conditions"]
    common = sorted(set(ac) & set(bc))
    if not common:
        return {"status":"not_comparable","reason":"no_matching_conditions",
                "text":"父代与当前没有相同的评估条件，不计算差值。"}
    rows, unavailable = [], 0
    def add(cell, outcome, av, bv, an, bn, definition):
        nonlocal unavailable
        if av[0] is None or bv[0] is None:
            unavailable += 1
            return
        delta = bv[0]-av[0]
        rows.append({"condition":cell,"stratum":ac[cell]["stratum"],"outcome":outcome,
            **definition,"parent":av,"candidate":bv,"parent_n":an,"candidate_n":bn,"delta":delta,
            "score":abs(delta)/max(abs(av[0]),abs(bv[0]),1e-9)*min(av[1],bv[1])})
    for cell in common:
        x,y=ac[cell],bc[cell]
        if x["stratum"] != y["stratum"]:
            raise ValueError("same cell has inconsistent condition stratum")
        for outcome in sorted(set(x["outcomes"]) | set(y["outcomes"])):
            an,bn=x["n"],y["n"]
            add(cell,outcome,[100*x["outcomes"].get(outcome,0)/an,an,0,0],
                [100*y["outcomes"].get(outcome,0)/bn,bn,0,0],an,bn,
                {"label":"结局占比","unit":"%","statistic":"episode_rate","window":"all"})
        for key in COMPARISON_METRICS["all"]:
            add(cell,None,x["all"][key],y["all"][key],x["n"],y["n"],a["descriptors"]["all."+key])
        for outcome in sorted(set(x["terminal"]) | set(y["terminal"])):
            if outcome not in x["terminal"] or outcome not in y["terminal"]:
                unavailable += len(COMPARISON_METRICS["terminal"])
                continue
            ax,by=x["terminal"][outcome],y["terminal"][outcome]
            for key in COMPARISON_METRICS["terminal"]:
                add(cell,outcome,ax["facts"][key],by["facts"][key],ax["n"],by["n"],a["descriptors"]["terminal."+key])
    selected=sorted(rows,key=lambda r:(-r["score"],r["condition"],r["outcome"] or "",r["label"]))[:3]
    for row in selected:
        row.pop("score")
        av,bv=row["parent"],row["candidate"]
        label=row["stratum"]+"/"+row["condition"]+("/"+row["outcome"] if row["outcome"] else "")
        row["text"]=(label+"："+row["label"]+format_number(av[0])+"→"+format_number(bv[0])+row["unit"]+
            "，差"+format_number(row["delta"])+("个百分点" if row["unit"]=="%" else row["unit"])+
            f"；有效回合{av[1]}/{row['parent_n']}→{bv[1]}/{row['candidate_n']}，缺测读数{av[2]}→{bv[2]}。")
    def composition(cells):
        outcomes=Counter()
        for c in cells.values(): outcomes.update(c["outcomes"])
        return {"episodes":sum(c["n"] for c in cells.values()),"outcomes":dict(outcomes)}
    return {"status":"available","reason":"matched_explicit_parent",
        "parent_checkpoint_hash":expected_parent_checkpoint_hash,"candidate_checkpoint_hash":right["domain"]["checkpoint_hash"],
        "matched_conditions":len(common),"parent_only_conditions":len(set(ac)-set(bc)),"candidate_only_conditions":len(set(bc)-set(ac)),
        "composition":{"parent":composition(ac),"candidate":composition(bc)},
        "comparable_facts":len(rows),"unavailable_facts":unavailable,"value_columns":a["fact_columns"],"rows":selected,
        "text":"按相同条件比较；末窗另按相同结局匹配。列出相对差异乘有效回合数最大的事实，保留绝对值和分母；不是逐回合配对或因果改善。"}
