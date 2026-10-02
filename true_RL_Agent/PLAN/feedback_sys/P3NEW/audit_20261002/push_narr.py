import pickle,json,sys,io
import numpy as np
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
P=pickle.load(open(sys.argv[1],'rb'))
def col(v,T):
    shape=v['component_shape']; n=int(np.prod(shape)) if shape else 1
    comps=v['by_c_order_component']
    arr=np.array([[float(x) for x in c.split(',')] for c in comps]).T
    return arr.reshape((T,*shape)) if shape else arr.reshape(T)
for name,ep in P.items():
    b=ep['blocks']
    if ep['q']['task']!='push': continue
    T=b['/data/episode/steps']; rec=b['/data/episode/original_terminal_record']
    pose=col(b['/data/raw_columns/pose'],T)       # height, roll, pitch, planar speed, ang speed, xy drift
    pm=col(b['/data/raw_columns/planar_motion'],T) # vx, vy, x, y (reset-relative)
    ev=rec['push_events']
    anchors=[e['control_step'] for e in ev if e.get('control_step',-1)>0]
    out=b['/data/episode/original_outcome']
    thr=0.215546
    h,roll,pitch,spd,ang,drift=pose.T
    vx,vy,x,y=pm.T
    print(f"\n=== {name} {ep['q']['evidence_scope']} stratum={rec['stratum']} outcome={out} pushes at {anchors}")
    e0=ev[0]
    dv=np.array(e0.get('requested_delta_v_world_mps') or e0.get('requested_delta_v_world_m_per_s') or [np.nan,np.nan])[:2] if isinstance(e0,dict) else None
    print('   event keys',list(e0.keys())[:12])
    for a_i,a in enumerate(anchors):
        end=anchors[a_i+1] if a_i+1<len(anchors) else T
        seg=slice(a,end)   # steps a+1..end (0-based a..end-1)
        s=spd[seg]; 
        # time to first return below threshold, fraction below, last-2s stats
        below=s<=thr
        first_below=int(np.argmax(below))+1 if below.any() else None
        # push direction from velocity jump
        jump=np.array([vx[a]-vx[a-1],vy[a]-vy[a-1]]); jn=np.linalg.norm(jump); u=jump/jn
        vpar=vx[seg]*u[0]+vy[seg]*u[1]; vperp=-vx[seg]*u[1]+vy[seg]*u[0]
        disp=np.array([x[end-1]-x[a-1],y[end-1]-y[a-1]]); dpar=disp@u; dperp=-disp[0]*u[1]+disp[1]*u[0]
        tail=slice(max(a,end-100),end)
        # dominant oscillation frequency of roll in tail (stepping cadence proxy)
        r=roll[tail]-roll[tail].mean(); f=np.fft.rfftfreq(len(r),0.02); A=np.abs(np.fft.rfft(r)); k=A[1:].argmax()+1
        sp=spd[tail]
        print(f"   push{a_i} @ {a}: Δv={jn:.2f} m/s; speed first ≤{thr:.3f} after {first_below} steps ({(first_below or 0)*0.02:.2f}s); frac below in segment {below.mean():.2f}; "
              f"net disp along push {dpar:+.2f} m, lateral {dperp:+.2f} m; v_par last2s mean {vpar[-100:].mean():+.2f}, v_perp {vperp[-100:].mean():+.2f}; "
              f"speed last2s mean {sp.mean():.2f} (min {sp.min():.2f}, max {sp.max():.2f}); roll osc {f[k]:.2f} Hz amp {2*A[k]/len(r):.3f} rad; "
              f"height last2s {h[tail].mean():.3f}±{h[tail].std():.3f}; max|roll| {np.abs(roll[seg]).max():.2f} max|pitch| {np.abs(pitch[seg]).max():.2f} min h {h[seg].min():.2f}")
    print(f"   final: speed {spd[-1]:.2f} ang {ang[-1]:.2f} roll {roll[-1]:+.2f} pitch {pitch[-1]:+.2f} h {h[-1]:.3f} drift {drift[-1]:.2f}; pre-push speed mean {spd[20:anchors[0]-1].mean():.3f} (steps 21..{anchors[0]-1}); pre-push height {h[20:anchors[0]-1].mean():.3f}")
