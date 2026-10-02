import json, math
from collections import Counter, defaultdict
R="D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3/results/author_final_bundle/author_blind_review/reviewer/"
d=json.load(open(R+"evidence-push.json",encoding="utf-8"))
T=d["transcripts"]
print(json.dumps(T[0]["recoveries"][0],indent=0)[:1500])
def q(xs):
    if not xs: return None
    xs=sorted(xs); n=len(xs)
    return [round(xs[0],3), round(xs[n//4],3), round(xs[n//2],3), round(xs[(3*n)//4],3), round(xs[-1],3)]
for s in ['low','medium','high','repeated','ALL']:
    E=[t for t in T if s=='ALL' or t['stratum']==s]
    print("="*90); print("STRATUM",s,len(E), dict(Counter(t['terminal_label'] for t in E)))
    for lab in sorted(set(t['terminal_label'] for t in E)):
        G=[t for t in E if t['terminal_label']==lab]
        ffc=Counter(tuple(sorted(t['final_failed_conditions'])) for t in G)
        cov=Counter(r['coverage'] for t in G for r in t['recoveries'])
        mag=[math.hypot(*r['direction']['reference_world_xy']) for t in G for r in t['recoveries'] if r.get('direction',{}).get('status')=='available']
        par=[r['direction']['velocity_parallel_mps'] for t in G for r in t['recoveries'] if r.get('direction',{}).get('status')=='available']
        lat=[r['direction']['velocity_lateral_mps'] for t in G for r in t['recoveries'] if r.get('direction',{}).get('status')=='available']
        dpar=[r['direction']['reset_displacement_parallel_m'] for t in G for r in t['recoveries'] if r.get('direction',{}).get('status')=='available']
        rec_lat=[ (r['recovery_control_step']-r['anchor_control_step'])*0.02 for t in G for r in t['recoveries'] if r.get('recovery_control_step') is not None and r.get('anchor_control_step') is not None]
        fin=[t['final_pose'] for t in G]
        final_unst=Counter(t['final_unstable_after_any_recovery'] for t in G)
        fi=[t['first_instability_after_all_recoveries'] for t in G if t.get('first_instability_after_all_recoveries')]
        ficond=Counter(tuple(x['failed_conditions']) for x in fi)
        print(f"  [{lab}] n={len(G)} final_failed={dict(ffc)} unit_coverage={dict(cov)} final_unstable_after_rec={dict(final_unst)}")
        print(f"      push |dv| q={q(mag)} vel_parallel q={q(par)} vel_lateral q={q(lat)} disp_parallel q={q(dpar)} recovery_latency_s q={q(rec_lat)}")
        print(f"      final height q={q([f['height_m'] for f in fin])} final planar q={q([f['planar_speed_mps'] for f in fin])} drift q={q([f['xy_drift_m'] for f in fin])} final pitch q={q([f['pitch_rad'] for f in fin])} final roll q={q([f['roll_rad'] for f in fin])}")
        print(f"      first instability conds={dict(ficond)} after_last_recovery_s q={q([x['after_last_recovery_s'] for x in fi])}")
        print(f"      valid_steps q={q([t['valid_steps'] for t in G])} unstable_steps_after_all q={q([t['unstable_steps_after_all_recoveries'] for t in G if t['unstable_steps_after_all_recoveries'] is not None])}")
