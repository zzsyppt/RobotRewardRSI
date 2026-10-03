"""Direct archive-to-behavior summaries; no intermediate card or episode ledger."""
from __future__ import annotations
from copy import deepcopy
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, report_bytes, format_number, select_whole_claims
from true_rl_agent.evaluation.trajectory_translation.cycles import measure_contact_cycles
from true_rl_agent.evaluation.trajectory_translation.joint_limits import measure_joint_limits
from true_rl_agent.evaluation.trajectory_translation.measurement import measure_behavior, NUMERIC_CONFIG
from true_rl_agent.evaluation.trajectory_translation.cohort import aggregate_behavior, distribution
from .trajectory_adapter import _load_contract, iter_stair_episodes
from .trajectory_capabilities import resolve_capabilities
from .trajectory_cycles import episode_periods, summarize_periods
from . import trajectory_joint_limits as limits
from . import trajectory_terrain as terrain

PROFILE = "trajectory_numeric_v2"
CATALOG_VERSION = "stair.trajectory.production_catalog.v4"
PUBLIC_SCHEMA = "true_rl_agent.stair.behavior.public.v6"
FULL_SCHEMA = "true_rl_agent.stair.behavior.full.v6"
SELECTION_VERSION = "stair.trajectory.behavior_selection.v4"
LEXICON_VERSION = "stair.trajectory_terms.v2"
MAX_PUBLIC_BYTES = 10240
DEFAULT_PUBLIC_BYTES = MAX_PUBLIC_BYTES
PROCESS_BANDS = {"near_zero_mps": .01, "forward_mps": .2,
                 "source": "go2_stair_descriptive_bands_not_runtime_or_causal_thresholds"}
VERSIONS = {"catalog": CATALOG_VERSION, "public": PUBLIC_SCHEMA,
            "selection": SELECTION_VERSION, "lexicon": LEXICON_VERSION}
_number = format_number


def build_summary(reference: dict) -> dict:
    """Validate the raw archive once and retain only consumed aggregates/events."""
    archive = _load_contract(reference)
    capabilities, bounds = resolve_capabilities(archive, reference)
    behavior_rows, cycle_periods, joint_values = [], [], {}
    for ep in iter_stair_episodes(reference, source_id="behavior"):
        behavior = measure_behavior(ep)
        behavior_rows.append(behavior)
        cycle_periods.append(episode_periods(measure_contact_cycles(ep)))
        if bounds is not None:
            for r in measure_joint_limits(ep, bounds, q_unit="rad", asset_id=limits.ASSET_SHA):
                v = r["value"]
                joint_values.setdefault(r["entity_ids"][0], []).append((v["counts"],v["minimum_signed_margin_rad"]))
        if capabilities["nominal_terrain"]["status"] == "available":
            cell = archive.cells[ep.env_index]
            _, arrays = terrain.measure_terrain(ep, cell.spec.geometry_payload(),
                geometry_digest=cell.geometry_digest, terrain_type=cell.terrain_type_index, regime=cell.terrain_regime_id)
            _, relative, statuses = arrays
            for window, mask in (("all", ep.steps > 0), ("terminal", ep.times >= ep.times[-1]-.5-1e-9)):
                for column, entity in enumerate((*ep.entities["root"], *ep.entities["support"])):
                    valid = mask & (statuses[:, column] == 0)
                    behavior["facts"][window][entity+".terrain"] = {"label": entity+"参考点地形相对高度", "unit": "m",
                        "value": float(relative[valid, column].mean()) if valid.any() else None,
                        "valid": int(valid.sum()), "unknown": int(mask.sum()-valid.sum()), "not_applicable": 0}
    joint = {"groups": {name:{"counts":{k:sum(c[k] for c,_ in values) for k in values[0][0]},
        "episode_equal":{"minimum_signed_margin_rad":distribution([v for _,v in values if v is not None])}}
        for name,values in joint_values.items()}} if joint_values else None
    summary = {"catalog_version":CATALOG_VERSION, "archive_sha256":reference["manifest_sha256"],
        "capabilities": capabilities, "behavior":aggregate_behavior(behavior_rows,motion_bands=PROCESS_BANDS),
        "cycle":summarize_periods(cycle_periods), "joint":joint}

    # The author and selection consume facets, not a repeated roster per facet.
    for group in summary["behavior"]["groups"].values():
        group.pop("members", None)
    # Persist the statistics actually consumed by prose/selection, not repeated
    # all-channel vectors for every condition facet.
    for name,g in summary["behavior"]["groups"].items():
        for window,facts in g["facts"].items():
            if name == "all":
                keys = ({"net","speed","backward","active","inverted","duration"} |
                        {k for k in facts if "." in k}) if window == "all" else {k for k in facts if k.endswith((".relative",".terrain"))}
            elif name.startswith("outcome:"):
                keys = {"time","inverted","tilt","rotation","forward_elevation_min","forward_elevation"} if window == "all" else (
                    {"progress","forward_speed","height","tilt_end","low_progress"} | {k for k in facts if k.endswith(".terrain")})
            else:
                keys = {"forward_speed"} if window == "terminal" and "/" in name else set()
            g["facts"][window] = {k:v for k,v in facts.items() if k in keys}
        for facts in g["facts"].values():
            for fact in facts.values():
                fact.pop("positive_episodes",None)
                fact["distribution"] = {k:fact["distribution"][k] for k in ("median","mean")}
        for facet in ("joint_process","terminal_motion"):
            for part in g.get(facet,{}).values():
                part.pop("conditions",None)
                for key in ("forward_speed_mps","end_time_s"):
                    part[key] = {k:part[key][k] for k in ("median","count")}
    from .behavior import comparison_operands
    summary["comparison"] = comparison_operands(behavior_rows)
    return summary


def source_domain(reference, archive):
    identity = archive["identity"]
    return {"evaluation_job_id":identity["job_id"], "evaluation_plan_hash":identity["evaluation_plan_hash"],
        "checkpoint_hash":identity["checkpoint_hash"], "task_id":identity["task_id"],
        **{k:identity["evaluation_binding"][k] for k in ("task_protocol_hash","evaluation_binding_hash")},
        "seed_namespace":identity["seed_namespace"], "archive_hash":reference["manifest_sha256"],
        "projection_hash":archive["metadata"]["runtime_projection"]["projection_hash"],
        "catalog_version":CATALOG_VERSION,"acceptance_split":"public_dev"}


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

    def fact(facts, key, label=None, statistic="median"):
        f = facts[key]
        text = (f["label"] if label is None else label)+_number(f["distribution"][statistic])+f["unit"]
        if f["valid_episodes"] != f["episodes"] or f["unknown"]:
            text += f"（有效{f['valid_episodes']}/{f['episodes']}回合，缺失{f['unknown']}读数）"
        return text

    def presence_text(facts):
        p = facts["inverted"]["presence"]
        text = f"已观察到上轴向下{p['observed_positive']}/{facts['inverted']['episodes']}回合"
        if p["unknown"]:
            text += f"；完整记录未见{p['complete_negative']}回合，{p['unknown']}回合因缺测不能判全程阴性"
        return text

    bands = behavior["motion_bands"] or PROCESS_BANDS
    def joint_text(group):
        parts = group.get("joint_process", {})
        if parts.get("observed_down") and len(parts) > 1:
            labels = {"observed_down":"已见上轴向下", "no_down_forward":f"完整记录未见上轴向下且末速>{bands['forward_mps']:g}m/s的",
                      "remaining":"其余已知", "unknown":"姿态或末窗未定"}
            return "同标签内："+"；".join(labels[k]+f"{parts[k]['episodes']}回合（结束"+
                _number(parts[k]["end_time_s"]["median"])+"s，末速"+
                _number(parts[k]["forward_speed_mps"]["median"])+"m/s"+
                (f"，末速有效{parts[k]['forward_speed_mps']['count']}/{parts[k]['episodes']}回合"
                 if parts[k]['forward_speed_mps']['count'] != parts[k]['episodes'] else "")+
                "）" for k in labels if k in parts)+"。"
        motion = group.get("terminal_motion", {})
        if "near_zero" in motion and ("forward" in motion or "backward" in motion):
            labels = {"near_zero":f"|有向速度|<{bands['near_zero_mps']:g}m/s", "forward":f"有向速度>{bands['forward_mps']:g}m/s",
                      "backward":f"有向速度<{-bands['forward_mps']:g}m/s", "remaining":"其余已知", "unknown":"末窗未定"}
            return "末窗分布："+"；".join(labels[k]+f"的{motion[k]['episodes']}回合（速度中位"+
                _number(motion[k]["forward_speed_mps"]["median"])+"m/s）" for k in labels if k in motion)+"。"
        return ""

    n = groups["all"]["episodes"]
    outcomes = groups["all"]["outcomes"]
    names = {"success":"成功", "timeout":"超时", "time_out":"超时", "forbidden_contact":"禁用部位接触标签",
             "base_orientation_instability":"姿态失稳标签", "off_course":"偏离路线标签", "slip":"滑移标签",
             "bad_orientation":"姿态失稳标签", "orientation":"姿态失稳标签"}
    add("scope", "results", f"本队列{n}回合，存档结局："+"、".join(f"{names.get(k,k)}{v}" for k,v in sorted(outcomes.items()))+"。标签不等于运行时终止触发器。下述先逐回合统计再取中位数，接触组合另报回合均值；末窗是记录结束前最多0.5s。", outcomes)
    f = groups["all"]["facts"]["all"]
    add("motion", "process", "总体："+"；".join(fact(f,k) for k in ("net","speed","backward","active"))+"；"+presence_text(f)+"。", f)
    for name, group in sorted(groups.items()):
        if not name.startswith("outcome:"):
            continue
        reason = name.split(":",1)[1]
        f = group["facts"]["terminal"]
        whole = group["facts"]["all"]
        text = f"{names.get(reason,reason)}组{group['episodes']}回合："+fact(whole,"time")+"；末窗"+"、".join(fact(f,k) for k in ("progress","forward_speed","height","tilt_end","low_progress"))+"；"+presence_text(whole)+"，"+fact(whole,"tilt")+"。"+joint_text(group)
        add("outcome."+reason, "counterexample" if reason=="success" else "process", text, group)
        if (whole["rotation"]["distribution"]["median"] or 0)>5:
            add("attitude."+reason,"process",names.get(reason,reason)+"组：前轴仰角极小/极大值的中位数"+_number(whole["forward_elevation_min"]["distribution"]["median"])+"/"+_number(whole["forward_elevation"]["distribution"]["median"])+"°；"+fact(whole,"rotation")+"（只累计有效连续边）。",whole,False,.3)
    f = groups["all"]["facts"]["all"]
    for i, entity in enumerate(behavior["support"]):
        short = entity.rsplit(".",1)[-1]
        cycle = summary["cycle"]["groups"]["contact_cycles:"+entity]
        period = cycle["episode_median_qualified_cycle_period_s"]["median"]
        text = short+"："+"、".join(fact(f, entity+"."+key, label) for key,label in
            (("contact","接触占比"),("world","世界点速"),("relative","相对点速")))
        text += "；身体相对位置均值(x,y,z)="+"/".join(_number(f[entity+"."+axis]["distribution"]["median"]) for axis in "xyz")+"m"
        text += "；末窗相对点速"+_number(groups["all"]["facts"]["terminal"][entity+".relative"]["distribution"]["median"])+"m/s"
        text += (f"；局部周期{_number(period)}s" if period is not None else "；未检出合格局部周期")+f"，{cycle['episodes_with_qualified_windows']}/{cycle['episodes']}回合有合格重复窗。"
        add("foot."+str(i), "process", text, {"facts":{k:v for k,v in f.items() if k.startswith(entity+".")},"cycle":cycle})
    contacts = sorted((k for k in f if k.startswith("contact.") and (f[k]["distribution"]["mean"] or 0)>0), key=lambda k:(-f[k]["distribution"]["mean"], k))[:3]
    if contacts:
        add("contacts", "process", "主要接触组合（按回合占比均值）："+"；".join(fact(f,k,statistic="mean") for k in contacts)+"。", {k:f[k] for k in contacts})
    joint = summary["joint"]
    if joint:
        ranked = sorted(joint["groups"], key=lambda k: (-joint["groups"][k]["counts"]["inside_edge_band_rows"]/max(1,joint["groups"][k]["counts"]["finite_rows"]),k))
        for i, name in enumerate(ranked):
            g = joint["groups"][name]; c = g["counts"]; total = c["finite_rows"]
            if not c["inside_edge_band_rows"] and not c["outside_rows"]:
                continue
            text = f"{name}名义限位：全队列有效{total}点，下/上5%边带{_number(100*c['lower_edge_rows']/total) if total else '未知'}%/{_number(100*c['upper_edge_rows']/total) if total else '未知'}%；超出名义下/上界{c['lower_outside_rows']}/{c['upper_outside_rows']}点；最小有符号余量{_number(g['episode_equal']['minimum_signed_margin_rad']['minimum'])}rad。"
            add("joint."+name,"process",text,g,mandatory=i<2,priority=3)
    terrain_keys = [k for k in f if k.endswith(".terrain")]
    if terrain_keys:
        add("terrain", "process", "参考点相对名义地形高度："+"；".join(fact(f,k,label=k.removesuffix('.terrain').rsplit('.',1)[-1]) for k in terrain_keys)+"。",{k:f[k] for k in terrain_keys})
        for name,g in sorted(groups.items()):
            if name.startswith("outcome:") and name != "outcome:success":
                t = g["facts"]["terminal"]
                known = [k for k in terrain_keys if t[k]["valid_episodes"]]
                if not known:
                    continue
                reference = groups.get("outcome:success",groups["all"])["facts"]["terminal"]
                foot_keys = [k for k in known if k != "robot_0.root.terrain"]
                contrasting = max(foot_keys, key=lambda k: abs(t[k]["distribution"]["median"]-(reference[k]["distribution"]["median"] or 0)), default=None)
                selected = [k for k in ("robot_0.root.terrain",contrasting) if k in known]
                text = names.get(name.split(':',1)[1],name.split(':',1)[1])+"组末窗地形参考差："
                text += "；".join(fact(t,k,label=k.removesuffix('.terrain').rsplit('.',1)[-1])+
                    ("，对应成功组"+fact(reference,k,label="") if "outcome:success" in groups else "") for k in selected)+"。"
                add("terrain."+name,"process",text,{"selected":{k:t[k] for k in selected},
                    "reference":{k:reference[k] for k in selected}},False,.25)
    for outcome, parent in ((k.split(':',1)[1],g) for k,g in sorted(groups.items()) if k.startswith("outcome:") and k!="outcome:success"):
        candidates = []
        for key, g in groups.items():
            if not key.startswith("condition:") or not key.endswith("/"+outcome) or g["episodes"] == parent["episodes"]:
                continue
            condition = key.split(':',1)[1].rsplit('/',1)[0]
            count = g["episodes"]
            gain = abs(count/groups["condition:"+condition]["episodes"] - parent["episodes"]/n)
            for facet in ("joint_process", "terminal_motion"):
                for part in sorted(set(g.get(facet, {})) | set(parent.get(facet, {}))):
                    gain += abs(g.get(facet, {}).get(part, {}).get("episodes",0)/count -
                                parent.get(facet, {}).get(part, {}).get("episodes",0)/parent["episodes"])
            speed = g["facts"]["terminal"]["forward_speed"]["distribution"]["median"]
            base_speed = parent["facts"]["terminal"]["forward_speed"]["distribution"]["median"]
            if speed is not None and base_speed is not None:
                gain += abs(speed-base_speed)/max(abs(base_speed), bands["forward_mps"])
            if count*gain >= 1:
                candidates.append((count*gain, condition, g))
        chosen = sorted(candidates, key=lambda x:(-x[0],x[1]))[:2]
        if chosen:
            parts = []
            mixed = parent.get("joint_process", {}).get("observed_down") and len(parent["joint_process"]) > 1
            for _, condition, g in chosen:
                text = condition+f"{g['episodes']}回合"
                if mixed:
                    text += "（按上述联合分组顺序计"+"/".join(str(g.get("joint_process", {}).get(k,{}).get("episodes",0))
                        for k in ("observed_down","no_down_forward","remaining","unknown") if k in parent["joint_process"])+"）"
                elif "near_zero" in parent.get("terminal_motion", {}) and any(k in parent["terminal_motion"] for k in ("forward","backward")):
                    text += "（按上述末窗分段顺序计"+"/".join(str(g.get("terminal_motion", {}).get(k,{}).get("episodes",0))
                        for k in ("near_zero","forward","backward","remaining","unknown") if k in parent["terminal_motion"])+"）"
                else:
                    text += "，末速"+_number(g["facts"]["terminal"]["forward_speed"]["distribution"]["median"])+"m/s"
                parts.append(text)
            add("conditions."+outcome,"process",names.get(outcome,outcome)+"组的条件差异："+"；".join(parts)+"。",chosen,False,.2)
    for event in behavior["witnesses"]:
        loc = f"r{event['run_index']}/e{event['env_index']}"
        kind = event["kind"]
        if kind == "success_drawdown_return":
            text = (f"成功反例{loc}：一个回撤并返回的实例，步{event['peak_step']}→{event['trough_step']}回撤"
                    f"{_number(event['max_drawdown_m'])}m，步{event['return_step']}返回此前进度，最终成功；实例不代表发生率。")
            section = "counterexample"
        elif kind == "terminal_history_difference":
            pattern = ("完整记录中，当前力全程越阈0拍、history ANY仅终点1拍" if
                       (event['current_positive'], event['current_unknown'], event['history_positive'], event['history_unknown']) == (0,0,1,0)
                       else f"呈现当前力越阈{event['current_positive']}拍、history ANY越阈{event['history_positive']}拍")
            text = (f"终点缓冲差异：{event['matching_episodes']}回合"+pattern+f"。例{loc}步{event['step']}，{event['entity'].rsplit('.',1)[-1]}当前"
                    f"{_number(event['current_terminal_n'])}N，history槽力模"+"/".join(_number(v) for v in event["history_terminal_n"])+"N；")
            flags = sorted(k for k,v in event["terminal_flags"].items() if v)
            text += ("同拍记录"+"、".join(flags)+"为真；" if flags else "")+"槽时标未记录，不能推定物理接触起点或持续时长。"
            section = "process"
        else:
            text = (f"先后反例{loc}：步{event['first_observed_down_step']}首次观察到上轴向下，随后{event['entity'].rsplit('.',1)[-1]}"
                    f"最长低速未触发段在步{event['start_step']}–{event['end_step']}（{event['samples']}记录点，世界XY点速最大"
                    f"{_number(event['max_world_xy_speed_mps'])}m/s，当前力模≤{behavior['contact_threshold_n']:g}N）。不能由后段推断直立举足；先后不证明因果。")
            section = "process"
        add("witness."+kind,section,text,event,False,.01)
    threshold=behavior["contact_threshold_n"]
    missing=[]
    if not joint: missing.append("名义关节坐标绑定")
    if not terrain_keys: missing.append("名义地形绑定")
    definition=(f"定义：接触标记为记录力模>{threshold:g}N；点速为相邻有效位置差/时间，分世界与身体坐标，不等于迈步。低正向进度为完整{NUMERIC_CONFIG['progress_window_s']:g}s窗最大新增<{NUMERIC_CONFIG['progress_increment_m']:g}m，含倒退；相对点速≥{NUMERIC_CONFIG['low_horizontal_speed_mps']:g}m/s为变化。初始不足窗不适用，内部缺步/非有限数未知；比例仅用有效点，确认时刻不是起点。差分归后端时标，不跨缺步。倾角用身体上轴，旋转用相邻四元数短弧，不用Euler范围推断翻转。名义边带非运行时限位；越界计数仅扣除边界转为记录精度的量化误差，有符号余量保留原值。地形差为参考点z减名义高度，非脚底净空。")
    definition += f"末速分段{bands['near_zero_mps']:g}/{bands['forward_mps']:g}m/s仅用于本Go2阶梯摘要，非任务或因果阈值；低速未触发指世界XY点速<{NUMERIC_CONFIG['low_horizontal_speed_mps']:g}m/s且当前力≤{threshold:g}N。"
    add("definitions","gaps",definition,{"numeric":NUMERIC_CONFIG,"contact_threshold_n":threshold})
    missing_steps=groups["all"]["facts"]["all"]["duration"]["unknown"]
    add("gaps","gaps",(f"记录跨度内缺{missing_steps}步；" if missing_steps else "")+("缺少"+"、".join(missing)+"；" if missing else "")+"动作力矩、奖励分项、运行时限位/表面未记录；过程不证明原因。成功组为当前队列对照，条件未逐实例配对。",summary["capabilities"])
    add("parent","parent_diff","父代匹配与数值差分见本次反馈；本卡描述当前队列。",{})
    return rows, trace, required


def project_production_report(domain: dict, summary: dict, *, max_bytes: int = DEFAULT_PUBLIC_BYTES) -> dict:
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PUBLIC_BYTES:
        raise ValueError("production trajectory report budget must be in 1..10240")
    if domain["archive_hash"] != summary["archive_sha256"] or summary["catalog_version"] != CATALOG_VERSION:
        raise ValueError("production summary/source mismatch")
    rows, trace, required = _behavior_claims(summary)
    optional = sorted(set(rows)-set(required),key=lambda cid:(trace[cid]["priority"],cid))
    card_id = "card:" + canonical_json_hash([domain, summary, VERSIONS])
    def render(included):
        report = {s: [rows[cid][1] for cid in included if rows[cid][0] == s] for s in SENTENCE_SECTIONS}
        report["visual"] = [{"id": "trajectory.visual", "text": "仅使用数值记录，未检查媒体。"}]
        if not report["counterexample"]:
            report["counterexample"] = [{"id": "trajectory.counterexample_unavailable", "text": "本队列无成功对照；保留绝对过程描述。"}]
        report["actionability"] = {"available":["现有任务允许的奖励输入和组件"],"diagnostic_only":["轨迹诊断不扩展奖励接口"],"authority":"不改变任务结局、控制器或训练权限"}
        receipt = {"selection_version": SELECTION_VERSION, "claim_count": len(rows),
                   "included_claim_count": len(included), "omitted_claim_count": len(rows)-len(included),
                   "omitted_claim_ids_hash": canonical_json_hash([cid for cid in rows if cid not in included]),
                   "required_claim_ids_hash": canonical_json_hash(required)}
        value = {"schema_version": PUBLIC_SCHEMA, "domain": domain, "derivation_version": SELECTION_VERSION,
                 "lexicon_version": LEXICON_VERSION, "card_id": card_id, "report": report, "report_projection": receipt}
        receipt["content_hash"] = canonical_json_hash(value)
        return value


    public, included, required_bytes = select_whole_claims(required, optional, render, max_bytes=max_bytes)
    validate_production_report(public, max_bytes=max_bytes)
    return {"schema_version":FULL_SCHEMA,"public_report":public,
        "audit":{"summary":summary,"max_report_bytes":max_bytes,
            "required_claim_ids":required,"included_claim_ids":included,
            "required_report_bytes":required_bytes,"report_bytes":report_bytes(public)}}


def validate_production_report(value, *, max_bytes=MAX_PUBLIC_BYTES):
    from .trajectory_report import _validate_trajectory_public
    return _validate_trajectory_public(value, max_bytes=max_bytes, public_schema=PUBLIC_SCHEMA,
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
