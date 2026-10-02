import pickle,json,sys,io
import numpy as np
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
P=pickle.load(open(sys.argv[1],'rb'))
def col(v,T):
    shape=v['component_shape']; n=int(np.prod(shape)) if shape else 1
    if 'by_c_order_component' in v:
        comps=v['by_c_order_component']
        if isinstance(comps,str): comps=[comps]
        arr=np.array([[float(x) if x not in('null','') else np.nan for x in c.split(',')] for c in comps]).T
    elif 'inclusive_constant_intervals_by_component' in v:
        arr=np.zeros((T,n))
        for j,ivs in enumerate(v['inclusive_constant_intervals_by_component']):
            for a,b,val in ivs: arr[a-1:b,j]=val
    else: raise KeyError(list(v))
    return arr.reshape((T,*shape)) if shape else arr.reshape(T)
FEET=['FL','FR','RL','RR']
def quat_axes(q):
    w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3]
    bx=np.stack([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)],1)   # body x in world
    bz=np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],1)   # body z in world
    return bx,bz
for name,ep in P.items():
    b=ep['blocks']
    if ep['q']['task']!='stair': continue
    T=b['/data/episode/original_terminal_record']['episode_steps']
    rec=b['/data/episode/original_terminal_record']
    g=b['/data/definitions/geometry']
    origin=np.array(b['/data/static_values/terrain_env_origins_w'])
    pos=col(b['/data/raw_columns/base_pos_w'],T)-origin
    q=col(b['/data/raw_columns/base_quat_w'],T)
    rp=col(b['/data/direct_columns/root_roll_pitch_rad'],T)
    feet=col(b['/data/raw_columns/support_pos_w'],T)-origin
    surf=col(b['/data/direct_columns/support_surface_height_terrain_m'],T)
    clr=col(b['/data/direct_columns/support_reference_clearance_m'],T)
    F=col(b['/data/direct_columns/support_force_norm_n'],T)
    fb=col(b['/data/direct_columns/forbidden_buffer_max_force_norm_n'],T).reshape(T)
    levels=np.concatenate([[0],np.cumsum(g['riser_heights_m'])])
    idx=np.abs(surf[:,:,None]-levels[None,None,:]).argmin(2)   # stair index per foot
    bx,bz=quat_axes(q)
    upz=bz[:,2]
    x=pos[:,0]; 
    # progress stall: last step at which running max of x increased
    runmax=np.maximum.accumulate(x); last_gain=int(np.max(np.nonzero(np.diff(runmax,prepend=runmax[0]-1)>0)[0]))+1
    # mean surface height under feet in contact
    contact=F>1.0
    # body height above mean foot surface
    hsurf=surf.mean(1); belly=pos[:,2]-hsurf
    # slope pitch: stair slope angle
    slope=np.arctan2(np.mean(g['riser_heights_m']),np.mean(g['tread_depths_m']))
    print(f"\n=== {name} {ep['q']['evidence_scope']} cell={g['cell_id'].split('.')[-1]} T={T} outcome={rec['primary_reason']} progress={rec['max_progress_m']:.2f} m ({rec['progress_fraction']:.0%}) risers~{np.mean(g['riser_heights_m']):.3f} treads~{np.mean(g['tread_depths_m']):.3f} n_steps={len(g['riser_heights_m'])}")
    if rec['primary_reason']=='success':
        print(f"   success: belly-height above mean foot surface min {belly.min():.3f} med {np.median(belly):.3f}; max|pitch| {np.abs(rp[:,1]).max():.2f}; max|roll| {np.abs(rp[:,0]).max():.2f}; pitch med {np.median(rp[:,1]):.2f}")
        continue
    def snap(t,label):
        i=t-1
        heading=np.arctan2(bx[i,1],bx[i,0])
        # lean direction of body up axis in heading frame
        hz=bz[i,:2]; c,s=np.cos(-heading),np.sin(-heading); lean=np.array([c*hz[0]-s*hz[1], s*hz[0]+c*hz[1]])
        print(f"   [{label} t={t} ({t*0.02:.2f}s)] root x={x[i]:.2f} z={pos[i,2]:.2f} roll={rp[i,0]:+.2f} pitch={rp[i,1]:+.2f} up_z={upz[i]:+.2f} lean(fwd,left)=({lean[0]:+.2f},{lean[1]:+.2f}) "
              f"feet on stair idx {dict(zip(FEET,idx[i].tolist()))} contact {dict(zip(FEET,contact[i].astype(int).tolist()))} foot-clr {np.round(clr[i],3).tolist()} root-above-feet-surface={belly[i]:.3f}")
    print(f"   last forward gain at step {last_gain} ({(T-last_gain)*0.02:.2f}s before end); running-max x={runmax[-1]:.2f}; final x={x[-1]:.2f}")
    # first tilt>0.65
    tilt=np.nonzero(np.maximum(np.abs(rp[:,0]),np.abs(rp[:,1]))>0.65)[0]
    neg=np.nonzero(upz<0)[0]
    print(f"   first |roll|or|pitch|>0.65 at {tilt[0]+1 if len(tilt) else None}; first up_z<0 at {neg[0]+1 if len(neg) else None}; forbidden-buffer first>1N at {int(np.nonzero(fb>1)[0][0])+1 if (fb>1).any() else None}")
    for t,l in ((max(1,last_gain-50),'1s before stall'),(last_gain,'stall'),(min(T,last_gain+25),'stall+0.5s'),(max(1,T-25),'end-0.5s'),(T,'end')):
        snap(t,l)
    # feet stepping activity after stall: number of touchdown events per foot
    td=((contact[1:]&~contact[:-1])[last_gain-1:]).sum(0)
    print(f"   touchdowns after stall per foot {dict(zip(FEET,td.tolist()))}; fraction of time each foot unloaded after stall {np.round(1-contact[last_gain-1:].mean(0),2).tolist()}")
    # pitch statistics during climbing (before stall)
    print(f"   pitch before stall: median {np.median(rp[:last_gain,1]):+.2f} (stair slope {slope:.2f} rad => nose-up expected ~ -{slope:.2f} if pitch sign is nose-down positive); min/max {rp[:last_gain,1].min():+.2f}/{rp[:last_gain,1].max():+.2f}")
