"""Independent raw-NPZ audit; no project measurement/report imports, no writes."""
import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np

def quant(v):
 a=np.asarray(v,float);a=a[np.isfinite(a)]
 return dict(zip(('min','q25','median','q75','max'),map(float,np.quantile(a,[0,.25,.5,.75,1])))) if len(a) else None

def analyze(folder, nominal=False):
 p=Path(folder);raw=(p/'manifest.json').read_bytes();m=json.loads(raw);meta=m['metadata']
 footnames=meta['runtime_projection']['resolved_roles']['support_effector'];jnames=meta['runtime_projection']['joint_names']
 strata={cell:st for st,cells in meta['summary']['stratum_cell_ids'].items() for cell in cells}
 thresh=meta['semantics']['episode_outcomes']['signals']['slip']['contact_force_threshold']
 rows=[];contact=np.zeros(len(footnames),int);combos=Counter();N=0;low_stats=np.zeros(5,int);jointlow=np.zeros(len(jnames),int);jointhi=jointlow.copy();jointout=jointlow.copy()
 for seed in meta['seeds']:
  run=seed['run_index'];dt=seed['step_dt'];chunks=sorted((x for x in m['chunks'] if x['metadata']['run_index']==run),key=lambda x:x['metadata']['step_start'])
  keys=('base_pos_w','base_quat_w','support_pos_w','support_lin_vel_w','support_contact_forces_w','joint_pos','episode_length','valid')
  parts={k:[] for k in keys};steps=[];origin=None
  for c in chunks:
   with np.load(p/c['file']) as a:
    for k in keys:parts[k].append(a[k])
    if 'terrain_env_origins_w' in a:origin=np.asarray(a['terrain_env_origins_w'],float)
   start=c['metadata']['step_start'];steps.extend(range(start,start+c['metadata']['steps']))
  arr={k:np.concatenate(v,axis=0) for k,v in parts.items()};steps=np.asarray(steps)
  for env,outcome in enumerate(seed['outcomes']):
   valid=arr['valid'][:,env];ix=np.flatnonzero(valid);s=steps[ix];a={k:np.asarray(v[ix,env],float) for k,v in arr.items() if k!='valid'}
   assert len(ix)==outcome['episode_steps'] and np.all(np.diff(s)==1)
   assert np.all(np.diff(a['episode_length'])==1)
   assert all(np.isfinite(v).all() for v in a.values())
   n=len(ix);t=s*dt;root=a['base_pos_w'];feet=a['support_pos_w'];q=a['base_quat_w'];q=q/np.linalg.norm(q,axis=1,keepdims=True);w,x,y,z=q.T
   up=1-2*(x*x+y*y);pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1));forward_z=2*(x*z-w*y)
   R=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(n,3,3)
   rel=np.einsum('nji,nkj->nki',R,feet-root[:,None,:])
   force=np.linalg.norm(a['support_contact_forces_w'],axis=-1);on=force>thresh;contact+=on.sum(axis=0);N+=n
   code=(on*(2**np.arange(len(footnames)))).sum(axis=1);combos.update(map(int,code))
   cv=np.linalg.norm(a['support_lin_vel_w'][...,:2],axis=-1);pv=np.linalg.norm(np.diff(feet[...,:2],axis=0),axis=-1)/dt;rv=np.linalg.norm(np.diff(rel,axis=0),axis=-1)/dt
   window=int(round(.5/dt));vlast=(root[-1,0]-root[-1-window,0])/(window*dt) if n>window else None
   prog=root[:,0]-origin[env,0]
   if n>window:
    low=np.array([np.max(prog[i-window:i+1])-prog[i-window]<.02 for i in range(window,n)])
    low_stats+=np.array([len(low),low.sum(),(low&(cv[window:]>=.05).any(axis=1)).sum(),(low&(pv[window-1:]>=.05).any(axis=1)).sum(),(low&(rv[window-1:]>=.05).any(axis=1)).sum()])
   tail=t>=2.;tailp=t[1:]>=2.;ft=feet[tail];rt=rel[tail];qt=a['joint_pos'][tail]
   row={'run':run,'env':env,'reason':outcome['primary_reason'],'stratum':strata[outcome['evaluation_cell_id']],'cell':outcome['evaluation_cell_id'],'n':n,'end_s':float(t[-1]),'net_x_m':float(root[-1,0]-root[0,0]),'end_progress_m':float(prog[-1]),'net_y_m':float(root[-1,1]-root[0,1]),'up_min':float(up.min()),'up_end':float(up[-1]),'up_negative':bool((up<0).any()),'first_up_negative_s':float(t[np.flatnonzero(up<0)[0]]) if (up<0).any() else None,'last_half_s_signed_vx':vlast,'end_z_above_origin_m':float(root[-1,2]-origin[env,2]),'foot_contact_fraction':on.mean(axis=0).tolist(),'foot_relative_z_median':np.median(rel[:,:,2],axis=0).tolist(),'pitch_max_before_up_negative':float(pitch[:np.flatnonzero(up<0)[0]+1].max()) if (up<0).any() else None,'forward_axis_world_z_max':float(forward_z.max())}
   if tail.sum()>1:
    row.update(after2s_world_foot_span_max_m=float(np.linalg.norm(np.ptp(ft,axis=0),axis=1).max()),after2s_body_foot_span_max_m=float(np.linalg.norm(np.ptp(rt,axis=0),axis=1).max()),after2s_joint_span_max_rad=float(np.ptp(qt,axis=0).max()),after2s_root_span_m=float(np.linalg.norm(np.ptp(root[tail],axis=0))),after2s_channel_xy_median=float(np.median(cv[tail])),after2s_world_fd_xy_median=float(np.median(pv[tailp])),after2s_body_fd_xyz_median=float(np.median(rv[tailp])))
   if run==1 and env==0:
    row['witness_after2s']={'steps':[int(s[tail][0]),int(s[-1])],'foot_world_first':ft[0].tolist(),'foot_world_last':ft[-1].tolist(),'channel_xy_median_each_foot':np.median(cv[tail],axis=0).tolist(),'force_norm_median_each_foot':np.median(force[tail],axis=0).tolist(),'root_first':root[tail][0].tolist(),'root_last':root[-1].tolist()} if tail.sum()>1 else None
   rows.append(row)
   lo=np.deg2rad(-155.9992218017578);hi=np.deg2rad(-48.00010681152344);band=.05*(hi-lo);tlo=abs(float(np.float32(lo))-lo);thi=abs(float(np.float32(hi))-hi)
   for j,name in enumerate(jnames):
    if name.endswith('calf_joint'):
     v=a['joint_pos'][:,j];jointlow[j]+=((v>=lo-tlo)&(v<=lo+band)).sum();jointhi[j]+=((v<=hi+thi)&(v>=hi-band)).sum();jointout[j]+=((v<lo-tlo)|(v>hi+thi)).sum()
 def group(rr):
  return {'n':len(rr),'outcomes':dict(Counter(x['reason'] for x in rr)),'strata':dict(Counter(x['stratum'] for x in rr)),**{k:quant([x[k] for x in rr if x.get(k) is not None]) for k in ('net_x_m','end_progress_m','end_s','last_half_s_signed_vx','up_min','up_end','end_z_above_origin_m','after2s_world_foot_span_max_m','after2s_body_foot_span_max_m','after2s_joint_span_max_rad','after2s_root_span_m','after2s_channel_xy_median','after2s_world_fd_xy_median','after2s_body_fd_xyz_median')}}
 out={'path':str(p),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'development_data':True,'all_valid_rows_finite_contiguous':True,'foot_order':footnames,'joint_order':jnames,'contact_threshold_n':thresh,'valid_samples':N,'outcomes':dict(Counter(x['reason'] for x in rows)),'contact_fraction_pooled':(contact/N).tolist(),'contact_combinations_top':[(c,v/N) for c,v in combos.most_common(4)],'low_progress_full_window_counts':dict(zip(('full_windows','low_progress','low_and_channel_any','low_and_world_fd_any','low_and_body_fd_any'),map(int,low_stats))),'calf_nominal_5pct_pooled':{name:{'lower':jointlow[j]/N,'upper':jointhi[j]/N,'outside_beyond_bound_float32_quantization':jointout[j]/N} for j,name in enumerate(jnames) if name.endswith('calf_joint')},'all':group(rows),'by_outcome':{reason:group([x for x in rows if x['reason']==reason]) for reason in sorted(set(x['reason'] for x in rows))},'up_negative':group([x for x in rows if x['up_negative']]),'failure_upright_forward_terminal':group([x for x in rows if x['reason']!='success' and not x['up_negative'] and x['last_half_s_signed_vx'] is not None and x['last_half_s_signed_vx']>.2]),'rows':rows}
 if not nominal:out.pop('calf_nominal_5pct_pooled')
 return out

print(json.dumps([analyze(p, '--nominal' in sys.argv) for p in sys.argv[1:] if p!='--nominal'],ensure_ascii=True,allow_nan=False))
