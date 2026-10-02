import json, random, statistics
from collections import defaultdict, Counter
B="D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3/results/author_final_bundle/author_blind_review/"
sheet=json.load(open(B+"reviewer/review_sheet.json",encoding="utf-8"))
wh=json.load(open(B+"withhold_from_reviewer.json",encoding="utf-8"))
cases=json.load(open("D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3/case_design.json",encoding="utf-8"))['cases']
cat={c['case_id']:c['process_category'] for c in cases}
m={r['blind_id']:r for r in wh['mapping']}
items={p['blind_id']:p for p in sheet['proposals']}
print("sheet keys", list(sheet.keys()))
p0=sheet['proposals'][0]; print("item keys", list(p0.keys()))
by_path=defaultdict(list)
for b,r in m.items(): by_path[r['result_path']].append(b)
dups={tuple(sorted(v)) for v in by_path.values() if len(v)>1}
print("n items", len(items), "unique result paths", len(by_path), "dup groups", sorted(dups))
def sig(p): return (p['fact_support']['grounded'],p['fact_support']['total'],tuple(sorted(h['type'] for h in p['hard_errors'])),p['action_relevance'],p['testability'],p['restraint'])
for g in sorted(dups):
    print(" dup", g, [sig(items[b]) for b in g], "identical" if len({sig(items[b]) for b in g})==1 else "DIFFERENT")
uniq=[]; seen=set()
for b in sorted(items):
    rp=m[b]['result_path']
    if rp in seen: continue
    seen.add(rp); uniq.append(b)
KEY={'numeric','fabricated_observation','contradicted_fact','causal_overreach','code_misdescription'}
def keyhe(p): return [h for h in p['hard_errors'] if h['type'] in KEY]
def nonkey(p): return [h for h in p['hard_errors'] if h['type'] not in KEY]
print("non-key HE types:", Counter(h['type'] for b in uniq for h in nonkey(items[b])))
def seen_info(b):
    s=json.load(open(B+f"reviewer/seen-{b[-3:]}.json",encoding="utf-8"))
    txt=json.dumps(s)
    return ('"behavior"' in txt, '"public_feedback_summary"' in txt, len(txt))
rows=[]
for b in uniq:
    p=items[b]; r=m[b]
    src=open(B+f"reviewer/{b}.py",encoding="utf-8").read()
    rat=p.get('rationale') or ''
    hb,hf,slen=seen_info(b)
    rows.append(dict(id=b,arm=r['arm'],case=r['case_id'],rep=r['repetition'],cat=cat[r['case_id']],task=r['case_id'].split('-')[0],
        g=p['fact_support']['grounded'],t=p['fact_support']['total'],nclaims=len(p['claims']),
        he=len(keyhe(p)),hetypes=[h['type'] for h in keyhe(p)],AR=p['action_relevance'],RS=p['restraint'],TT=p['testability'],
        beh=p['behavior_evidence_used'],input=p['input_kind'],src_len=len(src),rat_len=len(rat),seen_has_behavior=hb,seen_has_feedback=hf,seen_len=slen,
        claim_types=Counter(c['type'] for c in p['claims']),truth=Counter(c.get('truth') for c in p['claims']),
        unavailable=sum(1 for c in p['claims'] if c.get('truth')=='unavailable')))
print("\nBlinding leak: seen file shows behavior block?")
for arm in 'AT':
    rs=[r for r in rows if r['arm']==arm]
    print(f" arm {arm}: behavior visible in seen file {sum(r['seen_has_behavior'] for r in rs)}/{len(rs)}; feedback summary present {sum(r['seen_has_feedback'] for r in rs)}/{len(rs)}")
def agg(rs,label):
    g=sum(r['g'] for r in rs); t=sum(r['t'] for r in rs); he=sum(r['he'] for r in rs)
    print(f"{label:34} n={len(rs):2d} fs={g}/{t}={g/t if t else 0:.3f} facts/item={t/len(rs):.1f} keyHE={he} ({he/len(rs):.2f}/item) AR={statistics.mean(r['AR'] for r in rs):.2f} RS={statistics.mean(r['RS'] for r in rs):.2f} src_chars={statistics.mean(r['src_len'] for r in rs):.0f} rat_chars={statistics.mean(r['rat_len'] for r in rs):.0f} facts/1k_src={1000*t/sum(r['src_len'] for r in rs):.2f} facts/1k_rat={1000*t/max(1,sum(r['rat_len'] for r in rs)):.2f}")
print()
for arm in 'AT': agg([r for r in rows if r['arm']==arm],f"arm {arm}")
for arm in 'AT':
    for ik in ('first_draft','static_repair','other'):
        rs=[r for r in rows if r['arm']==arm and r['input']==ik]
        if rs: agg(rs,f"arm {arm} / {ik}")
for arm in 'AT':
    for beh in ('not_visible','no','partly','yes'):
        rs=[r for r in rows if r['arm']==arm and r['beh']==beh]
        if rs: agg(rs,f"arm {arm} / beh={beh}")
for arm in 'AT':
    for hb in (False,True):
        rs=[r for r in rows if r['arm']==arm and r['seen_has_behavior']==hb]
        if rs: agg(rs,f"arm {arm} / seen_behavior={hb}")
for arm in 'AT':
    for hf in (False,True):
        rs=[r for r in rows if r['arm']==arm and r['seen_has_feedback']==hf]
        if rs: agg(rs,f"arm {arm} / seen_feedback={hf}")
print()
for arm in 'AT':
    print(arm, "key HE types", dict(Counter(t for r in rows if r['arm']==arm for t in r['hetypes'])))
for arm in 'AT':
    srt=sorted([r for r in rows if r['arm']==arm], key=lambda r:-r['he'])
    tot=sum(r['he'] for r in srt)
    print(arm, "HE per item sorted", [r['he'] for r in srt], "items with 0 HE", sum(1 for r in srt if r['he']==0), "top4 share", round(sum(r['he'] for r in srt[:4])/tot,2) if tot else None)
print("\nPaired by case (T minus A):")
wins=losses=ties=0; diffs=[]; hediffs=[]
for c in sorted(cat):
    a=[r for r in rows if r['case']==c and r['arm']=='A']; t=[r for r in rows if r['case']==c and r['arm']=='T']
    fa=sum(r['g'] for r in a)/sum(r['t'] for r in a); ft=sum(r['g'] for r in t)/sum(r['t'] for r in t)
    ha=sum(r['he'] for r in a); ht=sum(r['he'] for r in t)
    diffs.append(ft-fa); hediffs.append(ht-ha)
    wins+= ft>fa; losses+= ft<fa; ties+= ft==fa
    print(f" {c:9} {cat[c]:22} A fs={fa:.3f} HE={ha} in={[r['input'][:5] for r in a]}  T fs={ft:.3f} HE={ht} in={[r['input'][:5] for r in t]} beh={[r['beh'][:4] for r in t]}  d_fs={ft-fa:+.3f} d_HE={ht-ha:+d}")
print("cases T better fs:",wins,"worse:",losses,"tie:",ties, "mean d_fs", round(statistics.mean(diffs),3), "mean d_HE", round(statistics.mean(hediffs),2))
print("cases T fewer HE:", sum(1 for d in hediffs if d<0), "more:", sum(1 for d in hediffs if d>0), "tie:", sum(1 for d in hediffs if d==0))
random.seed(20260929)
A=[r for r in rows if r['arm']=='A']; T=[r for r in rows if r['arm']=='T']
def boot(stat, N=20000):
    vals=[]
    for _ in range(N):
        a=[random.choice(A) for _ in A]; t=[random.choice(T) for _ in T]
        vals.append(stat(t)-stat(a))
    vals.sort(); return (round(vals[int(0.025*N)],3), round(vals[int(0.975*N)],3), round(statistics.mean(vals),3))
fs=lambda rs: sum(r['g'] for r in rs)/sum(r['t'] for r in rs)
he=lambda rs: sum(r['he'] for r in rs)/len(rs)
ar=lambda rs: statistics.mean(r['AR'] for r in rs)
he_nocode=lambda rs: sum(1 for r in rs for t in r['hetypes'] if t!='code_misdescription')/len(rs)
he_code=lambda rs: sum(1 for r in rs for t in r['hetypes'] if t=='code_misdescription')/len(rs)
print("\nitem-cluster bootstrap 95% CI for T-A (lo, hi, mean):")
print(" fact support:", boot(fs))
print(" key HE/item:", boot(he))
print(" key HE/item excl code_misdescription:", boot(he_nocode), " A", round(he_nocode(A),2), "T", round(he_nocode(T),2))
print(" code_misdescription/item:", boot(he_code), " A", round(he_code(A),2), "T", round(he_code(T),2))
print(" AR mean:", boot(ar))
def perm(d, N=20000):
    obs=abs(sum(d)); cnt=0
    for _ in range(N):
        s=sum(x if random.random()<0.5 else -x for x in d)
        if abs(s)>=obs-1e-12: cnt+=1
    return cnt/N
print("sign-flip permutation p, case-paired fs diff:", perm(diffs), " HE diff:", perm(hediffs))
Tv=[r for r in T if r['seen_has_behavior']]; Tn=[r for r in T if not r['seen_has_behavior']]
def boot2(X,Y,stat,N=20000):
    vals=[]
    for _ in range(N):
        vals.append(stat([random.choice(Y) for _ in Y])-stat([random.choice(X) for _ in X]))
    vals.sort(); return (round(vals[int(0.025*N)],3), round(vals[int(0.975*N)],3))
print("T(block visible) minus T(block not visible) fs CI:", boot2(Tn,Tv,fs), "n", len(Tv), len(Tn))
print("T(block not visible) minus A fs CI:", boot2(A,Tn,fs))
print()
for arm in 'AT':
    tr=Counter(); ct=Counter()
    for r in rows:
        if r['arm']==arm: tr.update(r['truth']); ct.update(r['claim_types'])
    print(arm, "truth", dict(tr), "types", dict(ct))
for arm in 'AT':
    ung=Counter()
    for b in uniq:
        if m[b]['arm']!=arm: continue
        for c in items[b]['claims']:
            if c['type'] in ('measured_fact','code_fact') and c['grounded'] is not True: ung[(c['type'],c.get('truth'),str(c['grounded']))]+=1
    print(arm, "ungrounded facts by (type,truth,grounded)", dict(ung))
json.dump(rows,open("C:/Users/lover/AppData/Local/Temp/claude/D--humanoid-agent/d96a789a-0d19-4cbb-9841-28f35468b4f0/scratchpad/audit_rows.json","w"),default=str,indent=0)
