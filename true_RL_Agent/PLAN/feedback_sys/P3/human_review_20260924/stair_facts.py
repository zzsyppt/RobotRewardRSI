import json, statistics as st, sys
from collections import Counter, defaultdict
R="D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3/results/author_final_bundle/author_blind_review/reviewer/"
d=json.load(open(R+"evidence-stair.json",encoding="utf-8"))
eps=d["episodes"]
geo={c['template_id']:c['geometry'] for c in json.load(open("D:/humanoid_agent/true_RL_Agent/configs/tasks/stair_go2_12dof_unified_evaluation_suite_v2.json",encoding="utf-8"))["cell_templates"]}
def q(xs):
    if not xs: return None
    xs=sorted(xs); n=len(xs)
    return [round(xs[0],3), round(xs[n//4],3), round(xs[n//2],3), round(xs[(3*n)//4],3), round(xs[-1],3)]
strata=sorted(set(e['stratum'] for e in eps))
for s in strata:
    E=[e for e in eps if e['stratum']==s]
    print("="*100); print("STRATUM",s,"n",len(E))
    cells=sorted(set(e['cell'] for e in E))
    for c in cells:
        C=[e for e in E if e['cell']==c]
        cnt=Counter(e['primary_reason'] for e in C)
        print("  ",c.replace('public_dev.',''),geo[c.replace('public_dev.','')].get('riser_height_m',geo[c.replace('public_dev.','')].get('riser_base_m')), dict(cnt),
              "by run", {r:dict(Counter(e['primary_reason'] for e in C if e['run']==r)) for r in (1,2,3)})
    for reason in sorted(set(e['primary_reason'] for e in E)):
        G=[e for e in E if e['primary_reason']==reason]
        bt=Counter(); wp=Counter()
        for e in G:
            for k,v in e['before_terminal'].items(): bt[k]+=bool(v)
            for k,v in e['window_present'].items(): wp[k]+=bool(v)
        anc=Counter(e['anchor']['status'] for e in G)
        steps=[e['steps'] for e in G]
        pmin=[e['ranges']['pitch_rad'][0] for e in G]; pmax=[e['ranges']['pitch_rad'][1] for e in G]
        rmin=[e['ranges']['roll_rad'][0] for e in G]; rmax=[e['ranges']['roll_rad'][1] for e in G]
        prog=[e['ranges']['progress_m'][1] for e in G]
        print(f"   [{reason}] n={len(G)} before_terminal={dict(bt)} window={dict(wp)} anchor={dict(anc)}")
        print(f"       steps q={q(steps)} pitch_min q={q(pmin)} pitch_max q={q(pmax)} roll_min q={q(rmin)} roll_max q={q(rmax)} progress_max q={q(prog)}")
        # event codes present anywhere before terminal, by body, count of episodes
        codes=Counter(); lcb=Counter(); slb=Counter(); tiltsign=Counter()
        for e in G:
            seen=set()
            for ev in e['events']:
                key=(ev['code'], ev['body'] if ev['code'] in ('low_clearance_body','slip_body','tilt') else '')
                seen.add(key)
            for k in seen: codes[k]+=1
        print("       episodes-with-event:", {f"{k[0]}{'/'+k[1] if k[1] else ''}":v for k,v in sorted(codes.items()) if k[0] not in ('contact_history','contact_instant','unloaded_rising','unloaded_lowering')})
