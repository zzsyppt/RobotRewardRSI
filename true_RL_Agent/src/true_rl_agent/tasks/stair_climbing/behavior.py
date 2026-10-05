"""Current behavior projection and explicitly bound, condition-matched comparison."""
from collections import Counter
from statistics import median
from typing import Mapping

from true_rl_agent.evaluation.process_report import format_quantity
from true_rl_agent.evaluation.trajectory_translation.measurement import NUMERIC_CONFIG
from .trajectory_production import PUBLIC_SCHEMA, FULL_SCHEMA, replay_production_report

COMPARISON_VERSION = "stair.behavior_comparison.v2"
COMPARISON_METRICS = {"all": ("net", "speed", "active"),
                      "terminal": ("progress", "forward_speed", "height", "tilt_end")}


def project_stair_behavior(value):
    return replay_production_report(value)


def comparison_operands(episodes):
    """Reuse measured facts; store no per-episode ledger or alternate definitions."""
    descriptors = {}
    def facts(members, window, *, include_support=False):
        result = {}
        keys = list(COMPARISON_METRICS[window])
        if include_support and window == "all":
            keys += [entity+".contact" for entity in members[0]["support"]]
        for key in keys:
            entries = [e["facts"][window][key] for e in members]
            definition = {"label": entries[0]["label"], "unit": entries[0]["unit"],
                          "statistic": "episode_median", "window": ("endpoint" if key in {"progress","tilt_end"} else f"terminal_{NUMERIC_CONFIG['terminal_window_s']:g}s_complete" if window == "terminal" else "whole_episode")}
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
    def summarize(members, *, include_support=False):
        outcomes = dict(Counter(e["outcome"]["primary_reason"] for e in members))
        terminal = {}
        for outcome, n in sorted(outcomes.items()):
            terminal[outcome] = {"n":n, "facts":facts([e for e in members if e["outcome"]["primary_reason"] == outcome], "terminal")}
        return {"n":len(members), "outcomes":outcomes,
                "all":facts(members,"all",include_support=include_support), "terminal":terminal}
    conditions = {cell:{"stratum":members[0]["stratum"], **summarize(members)}
                  for cell,members in sorted(grouped.items())}
    cohort = summarize(episodes, include_support=True)
    return {"version":COMPARISON_VERSION,"descriptors":descriptors,"conditions":conditions,
            "cohort":cohort,
            "fact_columns":["value","valid_episodes","unknown_samples","not_applicable_samples"],
            "contact_threshold_n":episodes[0]["contact_threshold_n"], "terminal_window_s":NUMERIC_CONFIG["terminal_window_s"]}


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
    # Unit scales describe readable absolute effects; neither baseline-relative
    # ratios nor sub-resolution changes receive an artificial maximum score.
    scales = {"%":10., "m":.5, "m/s":.1, "°":10.}
    resolution = {"%":.1, "m":.001, "m/s":.001, "°":.1}
    cohort_size = max(a["cohort"]["n"], b["cohort"]["n"])
    def add(cell, outcome, key, av, bv, an, bn, definition):
        nonlocal unavailable
        if av[0] is None or bv[0] is None:
            unavailable += 1
            return
        delta = bv[0]-av[0]
        if abs(delta) < resolution[definition["unit"]]:
            return
        rows.append({"condition":cell,"stratum":"全队列" if cell is None else ac[cell]["stratum"],"outcome":outcome,
            "metric":key,
            **definition,"parent":av,"candidate":bv,"parent_n":an,"candidate_n":bn,"delta":delta,
            "score":abs(delta)/scales[definition["unit"]]*min(av[1],bv[1])/cohort_size})
    x,y = a["cohort"],b["cohort"]
    for key in x["all"]:
        add(None,None,key,x["all"][key],y["all"][key],x["n"],y["n"],a["descriptors"]["all."+key])
    for cell in common:
        x,y=ac[cell],bc[cell]
        if x["stratum"] != y["stratum"]:
            raise ValueError("same cell has inconsistent condition stratum")
        for outcome in sorted(set(x["outcomes"]) | set(y["outcomes"])):
            an,bn=x["n"],y["n"]
            add(cell,outcome,"outcome_rate",[100*x["outcomes"].get(outcome,0)/an,an,0,0],
                [100*y["outcomes"].get(outcome,0)/bn,bn,0,0],an,bn,
                {"label":"结局占比","unit":"%","statistic":"episode_rate","window":"all"})
        for key in COMPARISON_METRICS["all"]:
            add(cell,None,key,x["all"][key],y["all"][key],x["n"],y["n"],a["descriptors"]["all."+key])
        for outcome in sorted(set(x["terminal"]) | set(y["terminal"])):
            if outcome not in x["terminal"] or outcome not in y["terminal"]:
                unavailable += len(COMPARISON_METRICS["terminal"])
                continue
            ax,by=x["terminal"][outcome],y["terminal"][outcome]
            for key in COMPARISON_METRICS["terminal"]:
                add(cell,outcome,key,ax["facts"][key],by["facts"][key],ax["n"],by["n"],a["descriptors"]["terminal."+key])
    ranked=sorted(rows,key=lambda r:(-r["score"],r["condition"] or "",r["outcome"] or "",r["metric"]))
    # Rank all changes by the same evidence scale. Several foot-contact changes
    # share a single sentence rather than taking one leading slot per foot.
    feet = [r for r in ranked if r["condition"] is None and r["metric"].endswith(".contact")]
    details = [r for r in ranked if r not in feet]
    if feet:
        contact = {"condition":None, "outcome":None, "metric":"support.contact",
            "members":[{k:v for k,v in r.items() if k != "score"} for r in feet],
            "score":max(r["score"] for r in feet)}
        contact["text"] = "全队列各足接触占比："+"；".join(
            r["label"].replace("robot_0.support.", "").replace("接触标记占比", "")+
            format_quantity(r["parent"][0],"%")+"→"+format_quantity(r["candidate"][0],"%")+"%"
            +f"（有效{r['parent'][1]}/{r['parent_n']}→{r['candidate'][1]}/{r['candidate_n']}回合）" for r in feet)+"。"
        details.append(contact)
    details = sorted(details,key=lambda r:(-r["score"],r["condition"] or "",r["outcome"] or "",r["metric"]))
    for row in details:
        row.pop("score")
        if "text" in row:
            continue
        av,bv=row["parent"],row["candidate"]
        label=row["stratum"]+("/"+row["condition"] if row["condition"] else "")+("/"+row["outcome"] if row["outcome"] else "")
        row["text"]=(label+"："+row["label"].replace("robot_0.support.", "")+format_quantity(av[0],row["unit"])+"→"+format_quantity(bv[0],row["unit"])+row["unit"]+
            "，差"+format_quantity(row["delta"],row["unit"])+("个百分点" if row["unit"]=="%" else row["unit"])+
            f"；有效回合{av[1]}/{row['parent_n']}→{bv[1]}/{row['candidate_n']}，缺测读数{av[2]}→{bv[2]}。")
    def composition(cells):
        outcomes=Counter()
        for c in cells.values(): outcomes.update(c["outcomes"])
        return {"episodes":sum(c["n"] for c in cells.values()),"outcomes":dict(outcomes)}
    names = {"success":"成功", "forbidden_contact":"禁用部位接触", "off_course":"偏离路线",
             "base_orientation_instability":"姿态失稳", "slip":"滑移", "insufficient_clearance":"间隙不足", "timeout":"超时"}
    x, y = a["cohort"], b["cohort"]
    core = f"父代→候选（{x['n']}→{y['n']}回合）："+"、".join(
        names.get(k,k)+f"{x['outcomes'].get(k,0)}→{y['outcomes'].get(k,0)}" for k in sorted(set(x["outcomes"])|set(y["outcomes"])))+"。"
    for key in ("net", "speed"):
        av,bv=x["all"][key],y["all"][key]
        definition=a["descriptors"]["all."+key]
        core += definition["label"]+format_quantity(av[0],definition["unit"])+"→"+format_quantity(bv[0],definition["unit"])+definition["unit"]
        if av[1] != x["n"] or bv[1] != y["n"]:
            core += f"（有效{av[1]}/{x['n']}→{bv[1]}/{y['n']}）"
        core += "；"
    return {"status":"available","reason":"matched_explicit_parent",
        "parent_checkpoint_hash":expected_parent_checkpoint_hash,"candidate_checkpoint_hash":right["domain"]["checkpoint_hash"],
        "matched_conditions":len(common),"parent_only_conditions":len(set(ac)-set(bc)),"candidate_only_conditions":len(set(bc)-set(ac)),
        "composition":{"parent":composition(ac),"candidate":composition(bc)},
        "comparable_facts":len(rows),"unavailable_facts":unavailable,"value_columns":a["fact_columns"],"rows":details,
        "core_text":core.rstrip("；")+"。",
        "text":("可比事实在显示分辨率内无差异。" if not rows else "按绝对变化和有效回合覆盖选择细节。")+
            "条件与末窗结局分别匹配，非逐回合配对或因果改善。"}
