"""Independent NPZ oracle and audit. No production translator/renderer imports.

Original prepare preceded new card output. Corrected prepare is post-review.
verify checks every raw episode,
all enabled measurements, short transitions, nearest onsets, compact metrics,
cohort facets and every final sentence's operands and projection text.
"""
import argparse
from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_policy_behavior_card as old

OUT = ROOT / "outputs/p2a"
LOCK = Path(__file__).with_name("source_lock.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def rng(v):
    return {"valid_count": len(v), "unknown_count": 0, "minimum": float(min(v)), "maximum": float(max(v)),
        "first": float(v[0]), "last": float(v[-1]), "range": float(np.ptp(v)),
        "minimum_step": int(np.argmin(v)) + 1, "maximum_step": int(np.argmax(v)) + 1,
        "first_valid_step": 1, "last_valid_step": len(v)}


def stats(v, dt):
    result = {"true_count": int(sum(v == 1)), "false_count": int(sum(v == 0)), "unknown_count": int(sum(v == -1))}
    valid = result["true_count"] + result["false_count"]
    result.update(valid_count=valid, fraction=result["true_count"] / valid if valid else None,
        numerator_cell_s=result["true_count"] * dt, denominator_cell_s=valid * dt,
        observed_onset_count=int(sum((v[:-1] == 0) & (v[1:] == 1))),
        observed_offset_count=int(sum((v[:-1] == 1) & (v[1:] == 0))))
    result["intervals"] = sorted([{**r, "state": label} for state, label in [(1, "true"), (0, "false"), (-1, "unknown")]
        for r in old.runs(v, dt, state)], key=lambda r: r["steps_half_open"][0])
    return result


RETRIGGER_QUALIFICATION = (
    "post-review clarification: same-entity finite current force norm >1N; "
    "preceding observed off N*dt>=0.5s, immediately one complete on, "
    "then >=1 observed off; no gap/unknown inside this witness; "
    "preceding off may be left-censored; following off need not reach 0.5s"
)


def single_current_on_witnesses(steps, forces, entities, dt):
    """Raw readout witnesses, sorted by on step then stable entity ID.

    Short-on examples remain valid even when qualifies_long_off is false.
    Unknowns/gaps split runs; they cannot bridge either side of the on.
    """
    steps = np.asarray(steps)
    forces = np.asarray(forces, dtype=float)
    markers = np.where(np.isfinite(forces).all(axis=2),
                       (np.linalg.norm(forces, axis=2) > 1).astype(int), -1)
    result = []
    for entity_index, entity in enumerate(entities):
        values = markers[:, entity_index]
        cuts = [0, *(np.flatnonzero((np.diff(values) != 0) | (np.diff(steps) != 1)) + 1).tolist(), len(steps)]
        runs = list(zip(cuts[:-1], cuts[1:]))

        def interval(start, stop):
            left = start == 0 or steps[start] != steps[start-1] + 1 or values[start-1] == -1
            right = stop == len(steps) or steps[stop] != steps[stop-1] + 1 or values[stop] == -1
            return {"steps_half_open": [int(steps[start]), int(steps[stop-1]) + 1],
                "sample_count": stop-start, "sample_cell_duration_proxy_s": (stop-start)*dt,
                "observed_span_s": float((steps[stop-1]-steps[start])*dt),
                "left_censored": bool(left), "right_censored": bool(right), "complete_run": not (left or right)}

        for j in range(1, len(runs)-1):
            start, stop = runs[j]
            before, after = runs[j-1], runs[j+1]
            if (stop-start != 1 or values[start] != 1 or values[before[0]] != 0 or values[after[0]] != 0
                    or steps[start] != steps[start-1] + 1 or steps[stop] != steps[start] + 1):
                continue
            result.append({"entity": entity, "entity_index": entity_index, "on_step": int(steps[start]),
                "dt_s": dt, "off_before": interval(*before), "on": interval(start, stop),
                "off_after": interval(*after), "qualifies_long_off": (before[1]-before[0])*dt >= .5})
    return sorted(result, key=lambda r: (r["on_step"], r["entity"]))


def raw_facts(m, a, dt, origin, outcome):
    n = len(a["episode_length"])
    projection = m["metadata"]["runtime_projection"]
    feet = ["robot_0.support." + x for x in projection["resolved_roles"]["support_effector"]]
    bodies = ["robot_0.body." + x for x in projection["resolved_roles"]["forbidden_contact"]]
    joints = projection["joint_names"]
    p, q = a["base_pos_w"], a["base_quat_w"]
    q = q / np.sqrt((q*q).sum(axis=1))[:, None]
    up = 1 - 2 * (q[:, 1]**2 + q[:, 2]**2)
    x = p[:, 0] - origin[0]
    dd = np.maximum.accumulate(x) - x
    trough = int(np.argmax(dd)); peak = int(np.argmax(x[:trough+1]))
    returned = next((i+1 for i in range(trough+1, n) if x[i] >= x[peak]), None)
    drawdown = {"max_drawdown_m": float(dd[trough]), "peak_step": peak+1, "trough_step": trough+1,
        "return_step": returned, "return_observation_right_censored": returned is None, "search_end_step": n, "cross_gap_search": False}
    delta = np.diff(p, axis=0)
    facts = {"root_progress": {"progress": rng(x), "net_progress_m": float(x[-1]-x[0]),
        "net_world_xyz_m": (p[-1]-p[0]).tolist(), "sampled_path_xy_m": float(np.linalg.norm(delta[:, :2],axis=1).sum()),
        "sampled_path_xyz_m": float(np.linalg.norm(delta,axis=1).sum()), "drawdown": drawdown,
        "positive_progress_variation_m": float(np.maximum(np.diff(x),0).sum()),
        "negative_progress_variation_m": float(np.maximum(-np.diff(x),0).sum()), "valid_count": n, "unknown_count": 0},
        "body_up": {"up_z": rng(up), "negative": stats((up < 0).astype(int), dt)}}
    width = int(np.ceil(.5/dt)); low = np.full(n,-1)
    for i in range(width,n):
        low[i] = max(x[i-width:i+1])-x[i-width] < .02
    facts["low_progress"] = stats(low,dt)
    tracks = {"low_progress":low,"body_up_negative":(up < 0).astype(int)}
    f = a["support_contact_forces_w"]; fn = np.sqrt((f*f).sum(axis=2)); marker = fn > 1
    vxy = np.sqrt((a["support_lin_vel_w"][:,:,:2]**2).sum(axis=2))
    vxyz = np.sqrt((a["support_lin_vel_w"]**2).sum(axis=2))
    relative = np.empty_like(a["support_pos_w"])
    for i,(w,u,v,z) in enumerate(q):
        rot = np.array([[1-2*(v*v+z*z),2*(u*v-w*z),2*(u*z+w*v)],
            [2*(u*v+w*z),1-2*(u*u+z*z),2*(v*z-w*u)],
            [2*(u*z-w*v),2*(v*z+w*u),1-2*(u*u+v*v)]])
        relative[i] = (a["support_pos_w"][i]-p[i]) @ rot
    for i,e in enumerate(feet):
        tracks[e+".marker"] = marker[:,i].astype(int)
        tracks[e+".low_speed_off"] = ((~marker[:,i]) & (vxy[:,i]<.05)).astype(int)
        facts[e+".marker"] = stats(tracks[e+".marker"],dt)
        v = stats(tracks[e+".low_speed_off"],dt)
        longest = max([r for r in v["intervals"] if r["state"]=="true"],key=lambda r:r["sample_count"],default=None)
        v["longest"] = None
        if longest:
            start,stop = longest["steps_half_open"]; sl=slice(start-1,stop-1)
            v["longest"] = {"interval":longest,"max_speed_xy_mps":float(max(vxy[sl,i])),
                "median_speed_xy_mps":float(np.median(vxy[sl,i])),"max_speed_xyz_mps":float(max(vxyz[sl,i])),
                "median_speed_xyz_mps":float(np.median(vxyz[sl,i])),"max_force_norm_n":float(max(fn[sl,i])),
                "minimum_up_z":float(min(up[sl])),"maximum_up_z":float(max(up[sl])),"orientation_valid_count":stop-start}
        facts[e+".low_speed_off"]=v
        facts[e+".relative_position"]={"body_xyz_m":[rng(relative[:,i,j]) for j in range(3)],
            "first_body_xyz_m":relative[0,i].tolist(),"last_body_xyz_m":relative[-1,i].tolist(),
            "mean_body_xyz_m":relative[:,i].mean(axis=0).tolist(),"world_xy_speed_mps":rng(vxy[:,i]),"world_xyz_speed_mps":rng(vxyz[:,i])}
    codes=sum(marker[:,i].astype(int)*(2**i) for i in range(len(feet)))
    facts["contact_combinations"]={"counts":[int(sum(codes==c)) for c in range(2**len(feet))],
        "bit_order":feet,"bit_weights":[2**i for i in range(len(feet))],"valid_count":n,"unknown_count":0,
        "transitions":[{"step":i+1,"before":int(codes[i-1]),"after":int(codes[i])} for i in range(1,n) if codes[i]!=codes[i-1]]}
    norms=fn.sum(axis=0);zs=f[:,:,2].sum(axis=0)
    facts["force_summary"]={"sample_sum_norm_n":norms.tolist(),"sample_sum_signed_fz_n":zs.tolist(),
        "rectangle_norm_ns":(norms*dt).tolist(),"rectangle_signed_fz_ns":(zs*dt).tolist(),
        "norm_share":(norms/sum(norms)).tolist() if abs(sum(norms))>1e-12 else [None]*len(feet),
        "signed_fz_share":(zs/sum(zs)).tolist() if abs(sum(zs))>1e-12 else [None]*len(feet),
        "per_entity":[{"entity":e,"norm_n":rng(fn[:,i]),"signed_fz_n":rng(f[:,i,2])} for i,e in enumerate(feet)]}
    for i,e in enumerate(joints):facts[e+".pose"]=rng(a["joint_pos"][:,i])
    for prefix,entities in [("support",feet),("forbidden",bodies)]:
        channel=prefix+"_contact_forces_w"; hname=channel+"_history"
        current=np.sqrt((a[channel]**2).sum(axis=2))>1
        h = (np.sqrt((a[hname]**2).sum(axis=3))>1).any(axis=1).astype(int) if hname in a else np.full((n,len(entities)),-1)
        for i,e in enumerate(entities):
            tracks[e+".history_any_marker"]=h[:,i]
            facts[e+".history_diagnostic"]={**stats(h[:,i],dt),"joint_valid_count":n if hname in a else 0,
                "history_any_on_current_off_count":int(sum((h[:,i]==1)&~current[:,i])),
                "current_on_history_all_off_count":int(sum(current[:,i]&(h[:,i]==0))),"missing_channels":[] if hname in a else [hname]}
            if prefix=="forbidden":
                tracks[e+".current_marker"]=current[:,i].astype(int)
                facts[e+".current_marker"]={**stats(tracks[e+".current_marker"],dt),"force_norm_n":rng(np.linalg.norm(a[channel][:,i],axis=1))}
    flags=["done_mask","wrapper_done_mask","termination_0","termination_1"]
    for key in flags:tracks[key]=a[key].astype(int)
    facts["task_endpoint"]={"outcome":outcome,"terminal_step":n,"terminal_time_s":n*dt,
        "terminal_flags":{k:bool(a[k][-1]) for k in flags},"tail_steps":list(range(max(1,n-4),n+1)),
        "tail_flags":{k:a[k][-5:].astype(bool).tolist() for k in flags},
        "tail_forbidden_norms_n":np.linalg.norm(a["forbidden_contact_forces_w"][-5:],axis=2).tolist(),
        "tail_support_norms_n":fn[-5:].tolist(),"tail_forbidden_history_norms_n":np.linalg.norm(a["forbidden_contact_forces_w_history"][-5:],axis=3).tolist() if "forbidden_contact_forces_w_history" in a else None}
    return facts,tracks,up,vxy,vxyz


def prepare():
    # Original necessary_raw.json is retained. A corrected prepare run writes
    # a new ledger and must not relabel the initial selection as qualified.
    result={"source_lock_sha256":old.sha(LOCK),"producer_imports":False,"sources":[],
        "retrigger_qualification": RETRIGGER_QUALIFICATION,
        "selection_status": "targeted post-review correction; not unseen confirmation"}
    for source in read(LOCK)["sources"]:
        chosen={(r["run_index"],r["env_index"]):[r["selection_reason"]] for r in source["predeclared_examples"]}
        cases={};examples={};counts=Counter();retrigger_witness=None
        for m,run,env,dt,origin,outcome,a in old.raw_episodes(source):
            facts,tracks,up,vxy,vxyz=raw_facts(m,a,dt,origin,outcome)
            counts[outcome["primary_reason"]]+=1
            base=facts["robot_0.body.base.current_marker"]
            h=facts["robot_0.body.base.history_diagnostic"]
            conditions={"manager_without_current":base["true_count"]==0 and bool(a["termination_1"][-1]),
                "history_only_single_readout":base["true_count"]==0 and h["true_count"]==1,
                "successful_retreat_return":outcome["success"] and facts["root_progress"]["drawdown"]["max_drawdown_m"]>0 and facts["root_progress"]["drawdown"]["return_step"] is not None,
                "xy_xyz_disagreement":bool(np.any((vxy<.05)&(vxyz>=.05)&(np.linalg.norm(a["support_contact_forces_w"],axis=2)<=1)))}
            inv=np.flatnonzero(up<0)
            conditions["inverted_before_long_lowoff"]=bool(len(inv) and any(v.get("longest") and v["longest"]["interval"]["sample_cell_duration_proxy_s"]>=.5 and inv[0]+1<v["longest"]["interval"]["steps_half_open"][0] for k,v in facts.items() if k.endswith(".low_speed_off")))
            entities = ["robot_0.support." + e for e in m["metadata"]["runtime_projection"]["resolved_roles"]["support_effector"]]
            short_ons = single_current_on_witnesses(a["episode_length"], a["support_contact_forces_w"], entities, dt)
            qualified = [w for w in short_ons if w["qualifies_long_off"]]
            conditions["isolated_current_on"] = bool(short_ons)
            conditions["single_current_retrigger"] = bool(qualified)
            if qualified and retrigger_witness is None:
                retrigger_witness = {"run": run, "env": env, **qualified[0]}
            for key,present in conditions.items():
                if present and key not in cases:
                    cases[key]=[run,env];chosen.setdefault((run,env),[]).append(key)
            if (run,env) in chosen:
                examples[f"r{run}.e{env}"]={"run":run,"env":env,"reasons":chosen[(run,env)],"facts":facts,
                    "raw_terminal":{k:a[k][-1].tolist() for k in ["base_quat_w","support_contact_forces_w","forbidden_contact_forces_w","termination_1"]}}
        result["sources"].append({"source_id":source["source_id"],"manifest_sha256":source["manifest_sha256"],
            "outcome_counts":dict(counts),"counterexamples":cases,"absent_cases":[k for k in conditions if k not in cases],"examples":examples,
            "single_current_retrigger_witness":retrigger_witness})
        print(source["source_id"],len(examples),dict(counts),flush=True)
    write(OUT/"necessary_raw_retrigger_corrected.json",result)


def numeric_episodes(path):
    header=None;rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            r=json.loads(line)
            if r["record_type"]=="episode":
                if header is not None:yield header,rows
                header=r;rows=[]
            else:rows.append(r)
        if header is not None:yield header,rows


def selection_for_verify(frozen, name):
    pre = deepcopy(next(s for s in frozen["sources"] if s["source_id"] == name))
    if "retrigger_qualification" not in frozen:
        # The original ledger established isolated on, not long preceding off.
        def label(value):
            return "isolated_current_on" if value == "single_current_retrigger" else value
        pre["counterexamples"] = {label(k): v for k, v in pre["counterexamples"].items()}
        pre["absent_cases"] = [label(k) for k in pre["absent_cases"]]
        for example in pre["examples"].values():
            example["reasons"] = [label(k) for k in example["reasons"]]
        pre["long_off_qualification_status"] = "not_established_by_original_ledger"
    else:
        old.check(frozen["retrigger_qualification"], RETRIGGER_QUALIFICATION, "selection qualification")
        pre["long_off_qualification_status"] = "evaluated_by_post_review_selector"
    return pre


def verify(name, selection=OUT/"necessary_raw.json", output=None):
    output = Path(output) if output is not None else OUT/(name+"_audit_selection_corrected.json")
    if output.exists():
        raise FileExistsError(output)  # Refuse before expensive verification; preserve old audits.
    started=time.perf_counter();source=next(s for s in read(LOCK)["sources"] if s["source_id"]==name)
    frozen=read(selection);old.check(frozen["source_lock_sha256"],old.sha(LOCK),"frozen selection")
    pre=selection_for_verify(frozen,name)
    old.check(pre["manifest_sha256"],source["manifest_sha256"],"selection source")
    replay=read(OUT/name/"replay.json");directory=Path(replay["card_reference"]["manifest_path"]).parent
    manifest=read(directory/"manifest.json");card=read(directory/"policy_card.json")
    for p,h in manifest["code_sha256"].items():old.check(old.sha(ROOT/p),h,"code hash")
    for p,v in manifest["artifacts"].items():old.check(old.sha(directory/p),v["sha256"],"artifact hash")
    records={};summaries={};groups={}
    for line in (directory/"evidence.jsonl").open(encoding="utf-8"):
        r=json.loads(line)
        if r["record_type"]=="claim":continue
        records[r["record_id"]]=r
        if r["record_type"]=="episode_summary":summaries[(r["run_index"],r["env_index"])]=r
        if r["record_type"]=="cohort_summary":groups[r["group"]]=r
    nm=read(replay["numeric_manifest"]);npz_evidence=Path(replay["numeric_manifest"]).with_name("evidence.jsonl")
    old.check(old.sha(npz_evidence),nm["evidence"]["sha256"],"whole numeric evidence")
    numeric=iter(numeric_episodes(npz_evidence));totals=Counter();sample_checks={};raw_manifest=None
    for m,run,env,dt,origin,outcome,a in old.raw_episodes(source):
        raw_manifest=m;s=summaries[(run,env)];header,rows=next(numeric)
        old.check([header["run_index"],header["env_index"]],[run,env],"numeric episode order")
        facts,tracks,_,_,_=raw_facts(m,a,dt,origin,outcome)
        measured={r["key"]:r for r in rows if r["record_type"]=="measurement"}
        old.check(set(measured)==set(facts),True,"all descriptor keys covered")
        for key,value in facts.items():old.check(measured[key]["value"],value,key)
        totals["measurements"]+=len(facts)
        # Independently compare every transition, including manager; missing history cannot supply events.
        for key,v in tracks.items():
            expected=[{"step":i+1,"before":bool(v[i-1]),"after":bool(v[i])} for i in range(1,len(v)) if v[i]>=0 and v[i-1]>=0 and v[i]!=v[i-1]]
            actual=[r["value"] for r in rows if r["record_type"]=="event" and r["key"]==key+".transition"]
            old.check(actual,expected,key+" events");totals["events"]+=len(expected)
        for r in rows:
            if r["value"].get("relation")!="nearest_readout_onset":continue
            left,right=r["entity_ids"]
            l=tracks[left+".marker"];rr=tracks[right+".marker"]
            candidates=[i+1 for i in range(1,len(rr)) if rr[i-1]==0 and rr[i]==1]
            step=r["value"]["reference_step"]
            target=min(candidates,key=lambda k:(abs(k-step),k)) if candidates else None
            old.check(r["value"],{"target_step":target,"candidate_count":len(candidates)},"nearest candidates")
            old.check(r["evidence_ref"]["steps_half_open"],[1,len(l)+1],"nearest complete dependency")
            totals["nearest_pairs"]+=1
        for r in rows:
            ref=r["evidence_ref"]
            old.check(ref["manifest_sha256"],source["manifest_sha256"],"source ref")
            old.check(set(ref["channels"])<=set(a),True,"actual raw channel refs")
            if ref.get("missing_channels"):
                old.check(all(c not in a for c in ref["missing_channels"]),True,"missing channel inventory")
                old.check(bool(ref.get("missing_basis")),True,"missing basis")
        info=old.audit_episode(s,a,dt,origin,outcome,manifest["primary_binding"]["entities"])
        totals["metric_checks"]+=info["metrics_checked"];totals["episodes"]+=1
        label=f"r{run}.e{env}"
        if label in pre["examples"]:
            old.check(facts,pre["examples"][label]["facts"],"frozen necessary facts "+label)
            sample_checks[label]={"reasons":pre["examples"][label]["reasons"],"necessary_descriptors":len(facts),
                "passed":True,"reading_example":s["episode_id"] in card["example_claims"],"summary_id":s["record_id"]}
    old.check(next(numeric,None),None,"no extra numeric episode")
    old.check(len(sample_checks),len(pre["examples"]),"all necessary samples")
    totals["aggregate_metrics"]=old.audit_groups(groups,summaries,source["manifest_sha256"])
    for _,v in manifest["source_files"].items():old.check(old.sha(v["evidence_path"]),v["sha256"],"source evidence hash")
    totals["references"]=old.audit_references({s["episode_id"]:s for s in summaries.values()},records,manifest["source_files"],{source["manifest_sha256"]:raw_manifest})
    public=read(OUT/name/"public.json");full=read(OUT/name/"behavior_full.json")
    public_rows=[r for section in ["results","process","counterexample","visual","gaps","parent_diff"] for r in public["report"][section]]
    by_id={c["claim_id"]:c for c in card["claims"]};md=(directory/"policy_card_zh.md").read_text(encoding="utf-8")
    claim_reports=[]
    for c in card["claims"]:
        operands={k:old.follow(records[v["record_id"]],v["path"]) for k,v in c["operands"].items()}
        old.check(c["text"] in md,True,"claim in view")
        old.check(md.count(c["claim_id"]),1,"once in view")
        old.check(any(bad in c["text"] for bad in ["D:\\","C:\\",".npz","manifest_path"]),False,"path leak")
        claim_reports.append({"claim_id":c["claim_id"],"template":c["template_id"],"scope":c["scope"],"text":c["text"],
            "operand_ids":[v["record_id"] for v in c["operands"].values()],"operand_trace_resolved":bool(operands)})
    for r in public_rows:
        if r["id"] in by_id:old.check(by_id[r["id"]]["text"] in r["text"],True,"projection exact original sentence")
    old.check(full["audit"]["report_bytes"]<=8192,True,"public budget")
    old.check(len(full["audit"]["required_claim_ids"]),18,"mandatory slots")
    unavailable_cases = (["history_only_single_readout"]
        if raw_manifest["metadata"]["profile"] == "stair.unified_first_episode.v1" else [])
    report={"passed":True,"source_id":name,"source_manifest_sha256":source["manifest_sha256"],"totals":dict(totals),
        "selection_ledger":{"path":str(selection),"sha256":old.sha(selection),
            "qualification":frozen.get("retrigger_qualification"),
            "long_off_qualification_status":pre["long_off_qualification_status"],
            "status":frozen.get("selection_status","original ledger; isolated-on labels corrected after review")},
        "selected_examples":sample_checks,"necessary_descriptor_checks":sum(v["necessary_descriptors"] for v in sample_checks.values()),
        "claim_count":len(claim_reports),"public_statement_count":len(public_rows),"claim_reports":claim_reports,
        "public_sentences":public_rows,"raw_outcomes":pre["outcome_counts"],"counterexamples":pre["counterexamples"],
        "absent_cases":[c for c in pre["absent_cases"] if c not in unavailable_cases],
        "capability_unavailable_cases":unavailable_cases,
        "absence_scope":"no qualifying observed witness under the selected ledger's selector; legacy long-off qualification is not established; not proof of physical absence",
        "gaps":"missing history excluded from physical fact accuracy; full E2 unseen gate not met",
        "all_group_metrics":groups["all"]["metrics"],"wording_review":"requires separate human/agent semantic review; source arrays and operands all checked",
        "producer_or_renderer_imports":False,"wall_seconds":time.perf_counter()-started}
    write(output,report)
    print(json.dumps({k:report[k] for k in ["passed","source_id","totals","necessary_descriptor_checks","claim_count","public_statement_count","wall_seconds"]},indent=2),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","verify"]);p.add_argument("--source",choices=["k1","k2","k3"])
    p.add_argument("--selection",type=Path,default=OUT/"necessary_raw.json")
    p.add_argument("--output",type=Path,help="New audit path; existing files are never overwritten")
    args=p.parse_args(argv)
    if args.mode=="prepare":prepare()
    else:
        if args.source is None:p.error("verify requires --source")
        verify(args.source,args.selection,args.output)


if __name__=="__main__":
    main()
