"""Direct archive-to-behavior summaries; no intermediate card or episode ledger."""
from __future__ import annotations
from copy import deepcopy
from itertools import zip_longest
import numpy as np
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, ReportBudgetExceeded, report_bytes, format_quantity, select_whole_claims
from true_rl_agent.evaluation.trajectory_translation.cycles import measure_contact_cycles
from true_rl_agent.evaluation.trajectory_translation.joint_limits import measure_joint_limits, measure_runtime_joint_limits
from true_rl_agent.evaluation.trajectory_translation.measurement import measure_behavior, NUMERIC_CONFIG, terminal_window_mask
from true_rl_agent.evaluation.trajectory_translation.cohort import aggregate_behavior, process_partition
from .trajectory_adapter import _load_contract, iter_stair_episodes
from .trajectory_capabilities import resolve_capabilities
from .trajectory_cycles import episode_periods, summarize_periods
from . import trajectory_terrain as terrain

PROFILE = "trajectory_numeric_v2"
CATALOG_VERSION = "stair.trajectory.production_catalog.v7"
PUBLIC_SCHEMA = "true_rl_agent.stair.behavior.public.v8"
FULL_SCHEMA = "true_rl_agent.stair.behavior.full.v8"
SELECTION_VERSION = "stair.trajectory.behavior_selection.v7"
LEXICON_VERSION = "stair.trajectory_terms.v9"
MAX_PUBLIC_BYTES = 10240
DEFAULT_PUBLIC_BYTES = MAX_PUBLIC_BYTES
PROCESS_BANDS = {"near_zero_mps": .01, "forward_mps": .2,
                 "source": "go2_stair_descriptive_bands_not_runtime_or_causal_thresholds"}
VERSIONS = {"catalog": CATALOG_VERSION, "public": PUBLIC_SCHEMA,
            "selection": SELECTION_VERSION, "lexicon": LEXICON_VERSION}
_number = format_quantity


class DescriptionComputationError(RuntimeError):
    """Description computation failed after source admission; not an archive error."""

    def __init__(self, stage):
        super().__init__(f"Stair description {stage} failed")
        self.stage = stage


def build_summary(reference: dict) -> dict:
    """Keep source admission/iteration outside the description failure boundary."""
    archive = _load_contract(reference)
    capabilities, bounds = resolve_capabilities(archive, reference)
    runtime = archive.manifest["metadata"].get("runtime_joint_position_limits")
    buffers = {kind:np.asarray(runtime["values"][key], dtype=float) for kind,key in
               (("hard","joint_pos_limits"),("soft","soft_joint_pos_limits"))} if runtime is not None else None
    behavior_rows, cycle_periods, joint_values = [], [], {}
    runtime_rows = []
    failure = None
    for ep in iter_stair_episodes(reference, source_id="behavior"):
        # Iterator advancement (including the second file read) and all source
        # qualification still run after a computation failure. A later corrupt
        # chunk must win over the unavailable-description result.
        if capabilities["nominal_terrain"]["status"] == "available":
            cell = archive.cells[ep.env_index]
            geometry = cell.spec.geometry_payload()
            terrain_inputs = terrain.prepare_terrain_inputs(ep, geometry, geometry_digest=cell.geometry_digest)
        if failure is not None:
            continue
        try:
            behavior = measure_behavior(ep)
            behavior_rows.append(behavior)
            cycle_periods.append(episode_periods(measure_contact_cycles(ep)))
            if buffers is not None:
                runtime_rows.append((ep.task["outcome"]["primary_reason"], measure_runtime_joint_limits(ep, buffers)))
            if bounds is not None:
                for entity, value in measure_joint_limits(ep, bounds, q_unit="rad").items():
                    joint_values.setdefault(entity, []).append((value["counts"], value["minimum_signed_margin_rad"]))
            if capabilities["nominal_terrain"]["status"] == "available":
                _, relative, statuses = terrain.measure_terrain(ep, geometry, terrain_inputs)
                for window, mask in (("all", ep.steps > 0), ("terminal", terminal_window_mask(ep))):
                    for column, entity in enumerate((*ep.entities["root"], *ep.entities["support"])):
                        valid = mask & (statuses[:, column] == 0)
                        behavior["facts"][window][entity+".terrain"] = {"label": entity+"参考点地形相对高度", "unit": "m",
                            "value": float(relative[valid, column].mean()) if valid.any() else None,
                            "valid": int(valid.sum()), "unknown": int(mask.sum()-valid.sum()), "not_applicable": 0}
        except Exception as exc:
            failure = exc
    if failure is not None:
        raise DescriptionComputationError("measurement") from failure
    try:
        return _aggregate_summary(reference, capabilities, behavior_rows, cycle_periods, joint_values, runtime_rows)
    except Exception as exc:
        raise DescriptionComputationError("aggregation") from exc


def _aggregate_summary(reference, capabilities, behavior_rows, cycle_periods, joint_values, runtime_rows=()):
    joint = {"groups": {name:{"counts":{k:sum(c[k] for c,_ in values) for k in values[0][0]},
        "episode_equal":{"minimum_signed_margin_rad":{"minimum":min((v for _,v in values if v is not None),default=None)}}}
        for name,values in joint_values.items()}} if joint_values else None
    summary = {"catalog_version":CATALOG_VERSION, "archive_sha256":reference["manifest_sha256"],
        "capabilities": capabilities, "behavior":aggregate_behavior(behavior_rows,motion_bands=PROCESS_BANDS),
        "cycle":summarize_periods(cycle_periods), "joint":joint,
        "runtime_joint":_aggregate_runtime_limits(runtime_rows) if runtime_rows else None}

    from .behavior import comparison_operands
    summary["comparison"] = comparison_operands(behavior_rows)
    summary["comparison"]["cohort"]["failure_process"] = failure_process_operands(summary["behavior"])
    return summary


def _aggregate_runtime_limits(rows):
    groups = {}
    for group in ("all", *sorted({reason for reason,_ in rows})):
        members = [values for reason,values in rows if group == "all" or reason == group]
        facts = {}
        for kind in members[0]:
            facts[kind] = {}
            for name in members[0][kind]:
                values = [r[kind][name] for r in members]
                counts = {key:sum(v[key] for v in values) for key in ("valid","unknown","lower_outside","upper_outside")}
                facts[kind][name] = {"label":values[0]["label"], "unit":"rad", **counts,
                    "bounds":{side:[min(v[side] for v in values),max(v[side] for v in values)] for side in ("lower","upper")},
                    "minimum_margin":min((v["minimum_margin"] for v in values if v["minimum_margin"] is not None),default=None),
                    "outside_percent":{"label":"有效记录点界外占比", "unit":"%",
                        **{side:100*counts[side+"_outside"]/counts["valid"] if counts["valid"] else None for side in ("lower","upper")}}}
        groups[group] = {"episodes":len(members), "facts":facts}
    return {"groups":groups, "basis":"startup_buffers_per_environment; pooled_finite_recorded_points"}


def source_domain(reference, archive):
    identity = archive["identity"]
    return {"evaluation_job_id":identity["job_id"], "evaluation_plan_hash":identity["evaluation_plan_hash"],
        "checkpoint_hash":identity["checkpoint_hash"], "task_id":identity["task_id"],
        **{k:identity["evaluation_binding"][k] for k in ("task_protocol_hash","evaluation_binding_hash")},
        "seed_namespace":identity["seed_namespace"], "archive_hash":reference["manifest_sha256"],
        "projection_hash":archive["metadata"]["runtime_projection"]["projection_hash"],
        "catalog_version":CATALOG_VERSION,"acceptance_split":"public_dev"}


OUTCOME_NAMES = {"success":"成功", "timeout":"超时", "time_out":"超时", "forbidden_contact":"禁用部位接触标签",
         "base_orientation_instability":"姿态失稳标签", "off_course":"偏离路线标签", "slip":"滑移标签",
         "insufficient_clearance":"间隙不足标签", "bad_orientation":"姿态失稳标签", "orientation":"姿态失稳标签"}

def _fact_text(facts, key, label=None, statistic="median"):
    f = facts[key]
    if "circular_mean" in f["distribution"]:
        d = f["distribution"]
        text = (f["label"] if label is None else label)
        text += (_number(d["circular_mean"], "°")+"°" if d["circular_mean"] is not None else
                 "无均向" if f["valid_episodes"] else "不适用" if f["not_applicable"] and not f["unknown"] else "未知")
        if d["resultant"] is not None:
            text += "(R="+_number(d["resultant"])+")"
        if f["valid_episodes"] != f["episodes"]:
            text += f"（有效{f['valid_episodes']}/{f['episodes']}，不适用{f['not_applicable']}、未知{f['unknown']}）"
        return text
    text = (f["label"] if label is None else label)+_number(f["distribution"][statistic], f["unit"])+f["unit"]
    if f["valid_episodes"] != f["episodes"] or f["unknown"]:
        text += f"（有效{f['valid_episodes']}/{f['episodes']}回合，缺失{f['unknown']}读数"
        text += (f"、不适用{f['not_applicable']}读数" if f["not_applicable"] else "")+"）"
    return text

def _presence_text(facts):
    p = facts["inverted"]["presence"]
    text = f"上轴向下{p['observed_positive']}/{facts['inverted']['episodes']}回合"
    if p["unknown"]:
        text += f"；完整记录未见{p['complete_negative']}回合，{p['unknown']}回合因缺测不能判全程阴性"
    return text

def _joint_text(group, bands, *, compact=False):
    parts = group.get("joint_process", {})
    if process_partition(group) == "joint_process":
        labels = {"observed_down":"已见上轴向下", "no_down_forward":f"完整未见向下且末前速>{bands['forward_mps']:g}m/s",
                  "remaining":"其余已知", "unknown":"姿态或末窗未定"}
        return "组内："+"；".join(labels[k]+f"{parts[k]['episodes']}回合（"+
            ("" if compact else "结束"+_number(parts[k]["end_time_s"]["median"])+"s，")+"末速"+
            _number(parts[k]["forward_speed_mps"]["median"])+"m/s"+
            (f"，末速有效{parts[k]['forward_speed_mps']['count']}/{parts[k]['episodes']}回合"
             if parts[k]['forward_speed_mps']['count'] != parts[k]['episodes'] else "")+
            "）" for k in labels if k in parts)+"。"
    motion = group.get("terminal_motion", {})
    if process_partition(group) == "terminal_motion":
        labels = {"near_zero":f"|前向速度|<{bands['near_zero_mps']:g}m/s", "forward":f"前向速度>{bands['forward_mps']:g}m/s",
                  "backward":f"前向速度<{-bands['forward_mps']:g}m/s", "remaining":"其余已知", "unknown":"末窗未定"}
        return "末窗分布："+"；".join(labels[k]+f"的{motion[k]['episodes']}回合（速度中位"+
            _number(motion[k]["forward_speed_mps"]["median"])+"m/s"+
            (f"，末速有效{motion[k]['forward_speed_mps']['count']}/{motion[k]['episodes']}回合"
             if motion[k]["forward_speed_mps"]["count"] != motion[k]["episodes"] else "")+"）" for k in labels if k in motion)+"。"
    return ""


def failure_process_operands(behavior):
    """Project existing aggregate facts, without another measurement or full card."""
    groups = {}
    for name, group in sorted(behavior["groups"].items()):
        if not name.startswith("outcome:") or name == "outcome:success":
            continue
        partition = process_partition(group)
        groups[name.split(":", 1)[1]] = {
            "episodes": group["episodes"],
            "facts": {window: {key: group["facts"][window][key] for key in keys}
                      for window, keys in (("all", ("time", "net", "net_lateral", "inverted")),
                                           ("terminal", ("forward_speed", "tilt_end")))},
            **({partition: group[partition]} if partition else {}),
        }
    return {"motion_bands": behavior["motion_bands"], "groups": groups}


def _outcome_text(reason, group, bands, *, compact=False):
    whole, f = group["facts"]["all"], group["facts"]["terminal"]
    single = group["episodes"] == 1
    text = OUTCOME_NAMES.get(reason, reason)+("1回合（单例，不作群体推广）" if single else f"组{group['episodes']}回合")+"："
    text += _fact_text(whole, "time", "结束")
    text += "；"+_fact_text(whole, "net", "净前移")+"、"+_fact_text(whole, "net_lateral", "横移")
    text += "；末窗"+_fact_text(f, "forward_speed", "前速")
    if compact:
        text += "；末态上轴"+_fact_text(f, "tilt_end", "倾角")
    else:
        text += "、"+"、".join(_fact_text(f, k, label) for k, label in (("lateral_speed", "横速"), ("height", "根部z")))
        text += "；末态"+_fact_text(f, "heading_end", "前轴朝向")+"，上轴"+_fact_text(f, "tilt_end", "倾角")+"、"+_fact_text(f, "up_direction_end", "方位")
    return text+"；"+_presence_text(whole)+"。"+_joint_text(group, bands, compact=compact)


def recent_failure_process(comparison):
    process = (comparison or {}).get("cohort", {}).get("failure_process")
    if process is None:
        return "候选绝对失败过程摘要未记录。"
    return "候选自身失败过程："+("".join(_outcome_text(reason, group, process["motion_bands"], compact=True)
        for reason, group in process["groups"].items()) or "本队列无失败标签回合。")


def _behavior_claims(summary):
    """Semantic order and priorities never depend on archive IDs."""
    behavior = summary["behavior"]
    groups = behavior["groups"]
    rows, trace, required = {}, {}, []

    def add(key, section, text, operands, mandatory=True, priority=0):
        cid = "claim:"+key
        rows[cid] = (section, {"id": cid, "text": text})
        trace[cid] = {"operands": operands, "priority": priority}
        if mandatory:
            required.append(cid)

    def low_process(group):
        facet = group["low_progress_facet"]
        parts = []
        for key, label in (("up_nonnegative","上轴z≥0"),("up_negative","上轴向下"),("posture_unknown","姿态未知")):
            c = facet["counts"][key]
            if sum(c.values()):
                parts.append(label+f"活动/低活动{c['active']}/{c['quiet']}点"+
                             (f"、未知{c['unknown']}点" if c['unknown'] else ""))
        text = f"低进度{facet['positive']}点"
        if parts:
            text += "："+"；".join(parts)
        return text+(f"；内部未知{facet['unknown']}点" if facet['unknown'] else "")+"。"

    bands = behavior["motion_bands"] or PROCESS_BANDS
    n = groups["all"]["episodes"]
    outcomes = groups["all"]["outcomes"]
    names = OUTCOME_NAMES
    window_s = NUMERIC_CONFIG["terminal_window_s"]
    add("scope", "results", f"本队列{n}回合，存档结局："+"、".join(f"{names.get(k,k)}{v}" for k,v in sorted(outcomes.items()))+f"。标签不等于终止触发器；量取回合中位数，方向取圆均值+集中度R；末窗为结束前{window_s:g}s，短回合仅已有末段。", outcomes)
    f = groups["all"]["facts"]["all"]
    add("motion", "process", "总体："+"；".join(_fact_text(f,k) for k in ("net","net_lateral","speed","backward","active","heading_end"))+"；"+_presence_text(f)+"。", f)
    for name, group in sorted(groups.items()):
        if not name.startswith("outcome:"):
            continue
        reason = name.split(":",1)[1]
        f = group["facts"]["terminal"]
        whole = group["facts"]["all"]
        text = _outcome_text(reason, group, bands)
        down = whole["first_down_time"]
        if down["valid_episodes"] >= 2:
            text += "首次观察向下时刻中位"+_number(down["distribution"]["median"])+f"s（{down['valid_episodes']}阳性回合）。"
        contacts = sorted((k for k in f if k.startswith("contact.") and (f[k]["distribution"]["mean"] or 0)>0),
                          key=lambda k:(-f[k]["distribution"]["mean"],k))[:2]
        if contacts:
            text += "末窗力标记："+"、".join(_fact_text(f,k,label="+".join(e.rsplit(".",1)[-1].removesuffix("_foot") for e in f[k]["label"].split("+")),statistic="mean") for k in contacts)+"。"
        if reason != "success" and group["low_progress_facet"]["positive"]:
            text += low_process(group)
        add("outcome."+reason, "counterexample" if reason=="success" else "process", text, group)
        observed = group.get("first_down_observation")
        if observed:
            previous = observed["previous"]
            text = f"{names.get(reason,reason)}组首次向下时间居中例r{observed['run_index']}/e{observed['env_index']}：步{observed['step']}（{_number(observed['time_s'])}s）首次观察上轴向下，倾角{_number(observed['tilt_deg'])}°"
            text += "、方位"+(_number(observed["up_direction_deg"])+"°" if observed["up_direction_deg"] is not None else "不适用")+"；"
            if previous:
                text += f"相邻前拍{previous['step']}倾角{_number(previous['tilt_deg'])}°、方位"+(_number(previous["up_direction_deg"])+"°" if previous["up_direction_deg"] is not None else "不适用")+"。"
            else:
                text += "缺少相邻有效未向下前拍。"
            add("first_down."+reason,"process",text,observed,False,.05)
        if (whole["rotation"]["distribution"]["median"] or 0)>5:
            add("attitude."+reason,"process",names.get(reason,reason)+"组：前轴仰角极小/极大值的中位数"+_number(whole["forward_elevation_min"]["distribution"]["median"])+"/"+_number(whole["forward_elevation"]["distribution"]["median"])+"°；"+_fact_text(whole,"rotation")+"（只累计有效连续边）。",whole,False,.3)
    f = groups["all"]["facts"]["all"]
    for i, entity in enumerate(behavior["support"]):
        short = entity.rsplit(".",1)[-1].removesuffix("_foot")
        cycle = summary["cycle"]["groups"]["contact_cycles:"+entity]
        period = cycle["episode_median_qualified_cycle_period_s"]["median"]
        text = short+"："+"、".join(_fact_text(f, entity+"."+key, label) for key,label in
            (("contact","接触占比"),("world","世界点速"),("relative","相对点速")))
        text += "；相对(x,y,z)均值="+"/".join(_number(f[entity+"."+axis]["distribution"]["median"]) for axis in "xyz")+"m"
        text += "；末窗相对点速"+_number(groups["all"]["facts"]["terminal"][entity+".relative"]["distribution"]["median"])+"m/s"
        text += (f"；局部周期{_number(period)}s" if period is not None else "；局部周期未检出")+f"（{cycle['episodes_with_qualified_windows']}/{cycle['episodes']}回合）。"
        add("foot."+str(i), "process", text, {"facts":{k:v for k,v in f.items() if k.startswith(entity+".")},"cycle":cycle})
    contacts = sorted((k for k in f if k.startswith("contact.") and (f[k]["distribution"]["mean"] or 0)>0), key=lambda k:(-f[k]["distribution"]["mean"], k))[:3]
    if contacts:
        add("contacts", "process", "全程接触组合："+"；".join(_fact_text(f,k,label="+".join(e.rsplit(".",1)[-1].removesuffix("_foot") for e in f[k]["label"].split("+")),statistic="mean") for k in contacts)+"。", {k:f[k] for k in contacts})
    joint = summary["joint"]
    if joint:
        def edge_counts(name):
            c = joint["groups"][name]["counts"]
            return c["lower_edge_rows"]+c["lower_outside_rows"], c["upper_edge_rows"]+c["upper_outside_rows"]
        ranked = sorted(joint["groups"], key=lambda k: (-max(edge_counts(k))/max(1,joint["groups"][k]["counts"]["finite_rows"]),k))
        for i, name in enumerate(ranked):
            g = joint["groups"][name]; c = g["counts"]; total = c["finite_rows"]
            if not c["inside_edge_band_rows"] and not c["outside_rows"]:
                continue
            lower, upper = edge_counts(name)
            text = f"{name}名义限位：有效{total}点，下侧5%带内或界外{_number(100*lower/total, '%') if total else '未知'}%，上侧{_number(100*upper/total, '%') if total else '未知'}%；其中超出下/上界{_number(100*c['lower_outside_rows']/total, '%') if total else '未知'}%/{_number(100*c['upper_outside_rows']/total, '%') if total else '未知'}%；最小有符号余量{_number(g['episode_equal']['minimum_signed_margin_rad']['minimum'], 'rad')}rad。"
            add("joint."+name,"process",text,g,mandatory=i<2,priority=3+i/1000)
    runtime_joint = summary.get("runtime_joint")
    if runtime_joint:
        limit_groups = runtime_joint["groups"]
        overall = limit_groups["all"]["facts"]
        def outside(v):
            return v["lower_outside"]+v["upper_outside"]
        selected = {min(records,key=lambda name:(-outside(records[name]),name))
                    for records in overall.values() if any(outside(v) for v in records.values())}
        def percentages(facts, name):
            def percent(value):
                return f"{value:.1g}" if value is not None and 0 < value < .05 else _number(value, "%")
            return "、".join(kind+" "+"/".join(percent(facts[kind][name]["outside_percent"][side])
                for side in ("lower","upper"))+"%" for kind in ("hard","soft"))
        for name in sorted(selected):
            first = overall["hard"][name]
            text = name.removesuffix("_joint")+f"界外下/上：有效{first['valid']}点，"+percentages(overall,name)
            text += "；最小余量(hard/soft)"+"/".join(f"{overall[kind][name]['minimum_margin']:.4g}" for kind in ("hard","soft"))+"rad。"
            if first["unknown"]:
                text += f"缺测{first['unknown']}点。"
            candidates = [k for k,g in limit_groups.items() if k not in ("all","success") and g["episodes"] >= 2]
            def difference(group):
                f = limit_groups[group]["facts"]
                return max(abs(outside(f[k][name])-outside(overall[k][name])*f[k][name]["valid"]/max(1,overall[k][name]["valid"])) for k in ("hard","soft"))
            contrasts = (["success"] if limit_groups.get("success",{}).get("episodes",0) >= 2 else [])
            contrasts += sorted(candidates,key=lambda key:(-difference(key),key))[:2-len(contrasts)]
            text += "；".join(names.get(key,key)+f"{limit_groups[key]['episodes']}回合/{limit_groups[key]['facts']['hard'][name]['valid']}点："+
                              percentages(limit_groups[key]["facts"],name) for key in contrasts)+("。" if contrasts else "")
            add("runtime_joint."+name,"process",text,{"joint":name,"groups":{k:limit_groups[k] for k in ("all",*contrasts)}})
        if not selected:
            totals = {kind:{key:sum(v[key] for v in facts.values()) for key in ("valid","unknown")} for kind,facts in overall.items()}
            add("runtime_joint.none","process","启动hard/soft上下界外均0点；"+"；".join(
                f"{kind}有效{v['valid']}、未知{v['unknown']}关节点" for kind,v in totals.items())+"。",runtime_joint)
    terrain_keys = [k for k in f if k.endswith(".terrain")]
    if terrain_keys:
        add("terrain", "process", "参考点相对名义地形高度："+"；".join(_fact_text(f,k,label=k.removesuffix('.terrain').rsplit('.',1)[-1]) for k in terrain_keys)+"。",{k:f[k] for k in terrain_keys})
        for name,g in sorted(groups.items()):
            if name.startswith("outcome:") and name != "outcome:success" and g["episodes"] > 1:
                t = g["facts"]["terminal"]
                known = [k for k in terrain_keys if t[k]["valid_episodes"]]
                if not known:
                    continue
                foot_keys = [k for k in known if k != "robot_0.root.terrain"]
                contrasting = min(foot_keys, key=lambda k: (-t[k]["valid_episodes"],-abs(t[k]["distribution"]["median"]),k), default=None)
                selected = [k for k in ("robot_0.root.terrain",contrasting) if k in known]
                text = names.get(name.split(':',1)[1],name.split(':',1)[1])+"组末窗地形参考差："
                text += "；".join(_fact_text(t,k,label=k.removesuffix('.terrain').rsplit('.',1)[-1]) for k in selected)+"。"
                add("terrain."+name,"process",text,{"selected":{k:t[k] for k in selected}},False,
                    .15+1/(100+g['episodes']*abs(t[contrasting]['distribution']['median'])) if contrasting else .16)
    for outcome, parent in ((k.split(":",1)[1],g) for k,g in sorted(groups.items()) if k.startswith("outcome:") and k!="outcome:success"):
        if parent["episodes"] < 2:
            continue
        candidates = []
        for key, g in groups.items():
            if not key.startswith("condition:") or not key.endswith("/"+outcome) or g["episodes"] < 2:
                continue
            cell = key.split(":",1)[1].rsplit("/",1)[0]
            total = groups["condition:"+cell]["episodes"]
            excess = g["episodes"] - total*parent["episodes"]/n
            candidates.append((cell,g,total,excess))
        chosen = {c:[] for c,g,total,excess in sorted(candidates,key=lambda r:(-r[3],-r[1]["episodes"],r[0]))[:2] if excess > 1e-9}
        outcome_cells = set(chosen)
        partition = process_partition(parent)
        if partition:
            # Cover every described nonsingleton process, even outside the two
            # cells with the largest outcome excess. Merge shared cell rows.
            for part, values in parent[partition].items():
                if values["episodes"] < 2:
                    continue
                represented = [row for row in candidates if row[1].get(partition,{}).get(part,{}).get("episodes",0) >= 2]
                if represented:
                    cell,g,total,_ = min(represented,key=lambda r:(-r[1][partition][part]["episodes"],r[0]))
                    chosen.setdefault(cell,[]).append(part)
        labels = {"observed_down":"向下", "no_down_forward":"完整未见向下且末前进", "remaining":"其余", "unknown":"未知",
                  "near_zero":"末前速近零", "forward":"末前进", "backward":"末倒退"}
        parts = []
        for cell,g,total,_ in candidates:
            if cell not in chosen:
                continue
            process_rows = [labels[part]+f"{g[partition][part]['episodes']}/{total}" for part in sorted(chosen[cell])]
            process_count = sum(g[partition][part]["episodes"] for part in chosen[cell])
            values = ([f"总计{g['episodes']}/{total}"] if cell in outcome_cells and process_count != g["episodes"] else [])
            text = cell.removeprefix("public_dev.")+" "+"".join(values)
            text += ("（其中"+"、".join(process_rows)+"）" if values and process_rows else "、".join(process_rows))
            parts.append((cell,text))
        if parts:
            text = names.get(outcome,outcome)+"条件格："+"；".join(text for _,text in sorted(parts))+"回合。"
            add("conditions."+outcome,"process",text,{"cells":chosen,"counts":candidates},bool(partition),.1)
    for event in behavior["witnesses"]:
        loc = f"r{event['run_index']}/e{event['env_index']}"
        kind = event["kind"]
        if kind == "success_drawdown_return":
            text = (f"成功组回撤≥{NUMERIC_CONFIG['progress_increment_m']:g}m后返回{event['qualifying_episodes']}回合；最大回撤例{loc}：步{event['peak_step']}→{event['trough_step']}回撤"
                    f"{_number(event['max_drawdown_m'])}m，步{event['return_step']}返回此前进度，最终成功；实例不代表发生率。")
            section = "counterexample"
        elif kind == "terminal_history_difference":
            pattern = ("完整记录中，当前力全程越阈0拍、history ANY仅终点1拍" if
                       (event['current_positive'], event['current_unknown'], event['history_positive'], event['history_unknown']) == (0,0,1,0)
                       else f"呈现当前力越阈{event['current_positive']}拍、history ANY越阈{event['history_positive']}拍")
            text = (f"终点缓冲差异：{event['matching_episodes']}回合"+pattern+f"。例{loc}步{event['step']}，{event['entity'].rsplit('.',1)[-1]}当前"
                    f"{_number(event['current_terminal_n'], 'N')}N，history槽力模"+"/".join(_number(v, 'N') for v in event["history_terminal_n"])+"N；")
            flags = sorted(k for k,v in event["terminal_flags"].items() if v)
            text += ("同拍记录"+"、".join(flags)+"为真；" if flags else "")+"槽时标未记录，不能推定物理接触起点或持续时长。"
            section = "process"
        else:
            text = (f"上轴向下后出现持续低速未触发段{event['qualifying_episodes']}回合；最长例{loc}：步{event['first_observed_down_step']}首次观察到上轴向下，随后{event['entity'].rsplit('.',1)[-1]}"
                    f"最长低速未触发段在步{event['start_step']}–{event['end_step']}（{event['samples']}记录点，世界XY点速最大"
                    f"{_number(event['max_world_xy_speed_mps'])}m/s，当前力模≤{behavior['contact_threshold_n']:g}N）。不能由后段推断直立举足；先后不证明因果。")
            section = "process"
        add("witness."+kind,section,text,event,False,.01)
    threshold=behavior["contact_threshold_n"]
    missing=[]
    if not joint and not runtime_joint: missing.append("名义关节坐标绑定")
    if not runtime_joint: missing.append("启动hard/soft缓冲区")
    if not terrain_keys: missing.append("名义地形绑定")
    definition=(f"定义：主卡与最近尝试各述自身队列、共用下列量定义。任务+X前、+Y横；净移从首记录点起，根部z相对场景原点，足相对量用身体坐标。力模>{threshold:g}N为接触标记；点速=相邻有效位置差/时间，非迈步；差分归后端、不跨缺步。低进度=完整{NUMERIC_CONFIG['progress_window_s']:g}s窗最大新增<{NUMERIC_CONFIG['progress_increment_m']:g}m（含倒退），确认非起点；任一足相对点速≥{NUMERIC_CONFIG['low_horizontal_speed_mps']:g}m/s为活动，全低于为低活动。初窗不足不适用、内部缺测未知，比例用有效点。倾角用上轴；方位/朝向为对应轴相对+X的水平投影，非运动/翻转方向；圆均值配R（1一致、近0无均向），退化不适用、坏值未知。首次向下仅首次观察，旋转累计四元数短弧。地形差=参考点z−名义高度，非脚底净空。")
    if runtime_joint:
        binding = summary["capabilities"]["runtime_joint_limits"]["binding"]
        end = binding.get("end_values_equal")
        snapshots = "记录称初末相同" if end is True else "记录称初末不同" if end is False else "末快照未知"
        definition += f"界外占比按各组全程有效q点、各环境启动界计算；{snapshots}，中途未监测。soft非仿真约束，界外非失败判据；仅扣边界量化差，余量留原值。"
    elif joint:
        definition += "名义边带非运行时限位；仅扣量化误差，余量保留原值。"
    definition += f"接触组合：全程/末窗各取回合内有效拍占比再等权平均；不足{window_s:g}s的回合末窗不适用，缺测另计；非承重/腾空。"
    definition += f"Go2描述分段：末前速{bands['near_zero_mps']:g}/{bands['forward_mps']:g}m/s，回撤≥{NUMERIC_CONFIG['progress_increment_m']:g}m，低速未触发≥{NUMERIC_CONFIG['progress_window_s']:g}s（世界XY点速<{NUMERIC_CONFIG['low_horizontal_speed_mps']:g}m/s且力≤{threshold:g}N）；非物理失败判据。"
    add("definitions","gaps",definition,{"numeric":NUMERIC_CONFIG,"contact_threshold_n":threshold})
    missing_steps=groups["all"]["facts"]["all"]["duration"]["unknown"]
    low=groups["all"]["low_progress_facet"]
    add("gaps","gaps",f"全队列低进度窗：初始不适用{low['not_applicable']}、内部未知{low['unknown']}点。"+(f"记录跨度内缺{missing_steps}步；" if missing_steps else "")+("缺少"+"、".join(missing)+"；" if missing else "")+"动作力矩、奖励分项未记录；有效限位/碰撞面未验证；过程不证明原因。"+"各结局报绝对量，条件未逐实例配对。",summary["capabilities"])
    add("parent","parent_diff","本卡描述当前队列。",{})
    return rows, trace, required


def project_production_report(domain: dict, summary: dict, *, max_bytes: int = DEFAULT_PUBLIC_BYTES) -> dict:
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PUBLIC_BYTES:
        raise ValueError("production trajectory report budget must be in 1..10240")
    if domain["archive_hash"] != summary["archive_sha256"] or summary["catalog_version"] != CATALOG_VERSION:
        raise ValueError("production summary/source mismatch")
    from .trajectory_report import validate_trajectory_domain
    validate_trajectory_domain(domain, catalog_version=CATALOG_VERSION)
    try:
        return _render_production_report(domain, summary, max_bytes=max_bytes)
    except Exception as exc:
        raise DescriptionComputationError("rendering") from exc


def _render_production_report(domain, summary, *, max_bytes):
    rows, trace, required = _behavior_claims(summary)
    optional = sorted(set(rows)-set(required),key=lambda cid:(trace[cid]["priority"],cid))
    card_id = "card:" + canonical_json_hash([domain, summary, VERSIONS])
    # Store each statement once. Capacity belongs to the eventual author payload,
    # where its baseline, predecessor, and recent trials are all known.
    report = {s:[rows[cid][1] for cid in (*required,*optional) if rows[cid][0] == s] for s in SENTENCE_SECTIONS}
    report["visual"] = [{"id":"trajectory.visual", "text":"仅数值记录，未检查媒体。"}]
    if not report["counterexample"]:
        report["counterexample"] = [{"id":"trajectory.counterexample_unavailable", "text":"本队列无成功回合。"}]
    report["actionability"] = {"available":["现有任务奖励接口"],"diagnostic_only":["行为描述"],"authority":"评测契约不变"}
    projection = {"max_bytes":max_bytes, "required_claim_ids":required, "optional_claim_ids":optional}
    public = {"schema_version":PUBLIC_SCHEMA,"domain":domain,"derivation_version":SELECTION_VERSION,
              "lexicon_version":LEXICON_VERSION,"card_id":card_id,"report":report,"report_projection":projection}
    projection["content_hash"] = canonical_json_hash(public)
    validate_production_report(public)
    return {"schema_version":FULL_SCHEMA,"public_report":public,
        "audit":{"summary":summary,"max_report_bytes":max_bytes}}


def author_behavior_payload(feedback: dict, *, max_bytes: int = MAX_PUBLIC_BYTES) -> dict:
    """Select once against the actual visible source, comparison, and whole prose."""
    public = feedback["profile"]
    projection = public["report_projection"]
    max_bytes = min(max_bytes, projection["max_bytes"])
    required = list(projection["required_claim_ids"])
    rows = {row["id"]: (section, row) for section in SENTENCE_SECTIONS
            for row in public["report"][section] if row["id"].startswith("claim:")}
    # Long identities remain bound in the internal profile/trial, never in the
    # byte budget that decides which physical facts the author can read.
    source = deepcopy(feedback["source"])
    difference = feedback.get("diff", {})
    diff_identity = {k:difference[k] for k in ("status", "reason", "source") if k in difference}
    shared_pair = difference.get("source") and any(all(row.get(k) == v for k,v in difference["source"].items())
        for row in feedback.get("secondary", []))
    diff_summary = {k:deepcopy(difference[k]) for k in
        ("core_text", "text", "matched_conditions", "parent_only_conditions", "candidate_only_conditions")
        if k in difference}
    # The short comparison and each real recent attempt are core feedback.
    # They are never silently traded away for additional baseline detail.
    diff_rows = {f"diff:row.{i}":row for i,row in enumerate(difference.get("rows", []))}
    if shared_pair:
        diff_summary = {"text":"接受时的父代变化见下列同代、同lane的最近尝试。"}
        diff_rows = {}
    secondary = deepcopy(feedback.get("secondary", []))
    trial_details = {f"trial:{i}.row.{j}":(i,row)
        for i,trial in enumerate(secondary) for j,row in enumerate(trial.pop("detail_rows", []))}
    # Interleave existing semantic orders; no cross-family floating priority scale.
    main_details = projection["optional_claim_ids"]
    detail_orders = [main_details, list(diff_rows)] + [
        [key for key,(owner,_) in trial_details.items() if owner == i] for i in range(len(secondary))]
    optional = [key for group in zip_longest(*detail_orders) for key in group if key is not None]
    def render(included):
        report = {section:[deepcopy(row) for row in public["report"][section] if not row["id"].startswith("claim:")]
                  + [rows[cid][1] for cid in included if cid in rows and rows[cid][0] == section]
                  for section in SENTENCE_SECTIONS}
        report["actionability"] = deepcopy(public["report"]["actionability"])
        selected_diff = [{"text":row["text"]} for key,row in diff_rows.items() if key in included]
        shown_diff = {**diff_identity, **diff_summary}
        if selected_diff:
            shown_diff["rows"] = selected_diff
        shown_trials = deepcopy(secondary)
        for key in included:
            if key in trial_details:
                owner,row = trial_details[key]
                shown_trials[owner].setdefault("details", []).append(row["text"])
        return {"source":source, "diff":shown_diff,
                "secondary":shown_trials,
                "profile": {"kind":"numeric_behavior", "report":report}}
    result, _, _ = select_whole_claims(required, optional, render, max_bytes=max_bytes)
    return result


def validate_production_report(value):
    from .trajectory_report import _validate_trajectory_public
    return _validate_trajectory_public(value, public_schema=PUBLIC_SCHEMA,
        derivation_version=SELECTION_VERSION, lexicon_version=LEXICON_VERSION,
        catalog_version=CATALOG_VERSION, max_public_bytes=MAX_PUBLIC_BYTES)


def replay_production_report(value):
    if value["schema_version"] == PUBLIC_SCHEMA:
        return validate_production_report(value)
    if value["schema_version"] != FULL_SCHEMA:
        raise ValueError("unsupported Stair behavior full schema")
    audit = value["audit"]
    expected = project_production_report(value["public_report"]["domain"], audit["summary"], max_bytes=audit["max_report_bytes"])
    if value["public_report"] != expected["public_report"] or any(audit.get(k) != v for k,v in expected["audit"].items()):
        raise ValueError("production report differs from summary/selection")
    return validate_production_report(value["public_report"])
