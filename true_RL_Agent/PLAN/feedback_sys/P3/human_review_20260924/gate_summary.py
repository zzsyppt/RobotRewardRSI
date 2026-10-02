"""Unblinded summary of the frozen review sheet (arithmetic only)."""
import json, sys
from collections import defaultdict
B="D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3/results/author_final_bundle/author_blind_review/"
sheet=json.load(open(B+"reviewer/review_sheet.json",encoding="utf-8"))
wh=json.load(open(B+"withhold_from_reviewer.json",encoding="utf-8"))
cat={c['case_id']:c['process_category'] for c in json.load(open("D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3/case_design.json",encoding="utf-8"))['cases']}
m={r['blind_id']:r for r in wh['mapping']}
items={p['blind_id']:p for p in sheet['proposals']}
seen=set(); uniq=[]; dups=[]
for bid in sorted(items):
    rp=m[bid]['result_path']
    if rp in seen: dups.append(bid); continue
    seen.add(rp); uniq.append(bid)
print("unique episodes", len(uniq), "duplicates", dups)
dup_pairs=[(b,[u for u in uniq if m[u]['result_path']==m[b]['result_path']][0]) for b in dups]
print("dup pairs (dup, original):", [(d,o,m[o]['arm']) for d,o in dup_pairs])
KEY={'numeric','fabricated_observation','contradicted_fact','causal_overreach','code_misdescription'}
def stats(bids):
    n=len(bids); g=t=0; macro=[]; mg=mt=0
    he_items=0; he_total=0; types=defaultdict(int); contra_hyp=0; contra_hyp_items=0
    AR=[];TT=[];RS=[]; auth=0; beh_used=defaultdict(int); mech=defaultdict(int); focus=defaultdict(int)
    for b in bids:
        p=items[b]; fs=p['fact_support']; g+=fs['grounded']; t+=fs['total']
        if fs['total']: macro.append(fs['grounded']/fs['total'])
        for c in p['claims']:
            if c['type']=='measured_fact':
                mt+=1; mg+= c['grounded'] is True
        hes=[h for h in p['hard_errors'] if h['type'] in KEY]
        he_total+=len(hes); he_items+= bool(hes)
        for h in p['hard_errors']: types[h['type']]+=1
        ch=[c for c in p['claims'] if c['type']=='explanatory_hypothesis' and c['truth']=='contradicted']
        contra_hyp+=len(ch); contra_hyp_items+=bool(ch)
        AR.append(p['action_relevance']); TT.append(p['testability']); RS.append(p['restraint'])
        auth+= p['authorization']['verdict']=='pass'
        beh_used[p['behavior_evidence_used']]+=1; mech[p['mechanism_matches_code']['verdict']]+=1; focus[p['focus_addressed']['verdict']]+=1
    avg=lambda x: round(sum(x)/len(x),3) if x else None
    return dict(n=n, fact_support_micro=f"{g}/{t}={g/t:.3f}", fact_support_macro=avg(macro), measured_fact_support=f"{mg}/{mt}={mg/mt:.3f}" if mt else None,
        items_with_key_HE=f"{he_items}/{n}", key_HE_total=he_total, HE_types=dict(types), items_with_contradicted_hypothesis=f"{contra_hyp_items}/{n}", contradicted_hypotheses=contra_hyp,
        AR_mean=avg(AR), AR_dist={k:AR.count(k) for k in (0,1,2)}, testability_mean=avg(TT), T_dist={k:TT.count(k) for k in (0,1,2)}, restraint_mean=avg(RS), RS_dist={k:RS.count(k) for k in (0,1,2)},
        authorization_pass=f"{auth}/{n}", behavior_used=dict(beh_used), mechanism_matches=dict(mech), focus=dict(focus))
out={}
for arm in ('A','T'):
    out[arm]=stats([b for b in uniq if m[b]['arm']==arm])
for task in ('stair','push'):
    for arm in ('A','T'):
        out[f"{task}/{arm}"]=stats([b for b in uniq if m[b]['arm']==arm and m[b]['case_id'].startswith(task)])
for pc in sorted(set(cat.values())):
    for arm in ('A','T'):
        out[f"{pc}/{arm}"]=stats([b for b in uniq if m[b]['arm']==arm and cat[m[b]['case_id']]==pc])
# per-case table
per_case=defaultdict(dict)
for b in uniq:
    p=items[b]; r=m[b]
    per_case[r['case_id']].setdefault(r['arm'],[]).append(dict(id=b,rep=r['repetition'],fs=f"{p['fact_support']['grounded']}/{p['fact_support']['total']}",HE=[h['type'] for h in p['hard_errors']],AR=p['action_relevance'],T=p['testability'],RS=p['restraint'],beh=p['behavior_evidence_used'],input=p['input_kind']))
out['per_case']=per_case
json.dump(out,open(sys.argv[1],'w',encoding='utf-8'),ensure_ascii=False,indent=1)
for k in ['A','T','stair/A','stair/T','push/A','push/T']+[f"{pc}/{a}" for pc in sorted(set(cat.values())) for a in 'AT']:
    s=out[k]; print(f"{k:28} n={s['n']} fs={s['fact_support_micro']} meas={s['measured_fact_support']} keyHE_items={s['items_with_key_HE']} keyHE={s['key_HE_total']} contraHyp={s['items_with_contradicted_hypothesis']} AR={s['AR_mean']} T={s['testability_mean']} RS={s['restraint_mean']} auth={s['authorization_pass']} beh={s['behavior_used']}")
