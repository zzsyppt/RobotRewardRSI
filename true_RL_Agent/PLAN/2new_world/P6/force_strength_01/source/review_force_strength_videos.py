"""Compare native replays with frozen numeric runs and build actual-frame review sheets."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / '.venv/Lib/site-packages'))
import imageio_ffmpeg

P6 = ROOT / 'PLAN/2new_world/P6'
D = P6 / 'force_strength_01'
FONT = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
SMALL = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path) as a:
        return {k:a[k] for k in a.files}


def sheet(case, steps, pictures, facts, title, path):
    w, h, cols = 360, 348, 3
    canvas = Image.new('RGB', (w*cols, h*math.ceil(len(steps)/cols)+60), '#101820')
    draw = ImageDraw.Draw(canvas)
    draw.text((12,8), title, font=FONT, fill='white')
    draw.text((12,32), f"{case['case_id']} | {case['selection_reason']} | outcome: {case['expected_episode']['failure_label']}", font=SMALL, fill='#d5dde5')
    for i, step in enumerate(steps):
        x, y = (i%cols)*w, (i//cols)*h+60
        # A fixed center crop of the actual native frame; original MP4 is retained.
        panel = pictures[step].crop((280,90,1000,630)).resize((w,270))
        canvas.paste(panel, (x,y))
        f = facts[step-1]
        lines = [f"step {step} / {step*.02:.2f}s | force {f['force_n']:.1f} N",
                 f"tilt {f['tilt_deg']:.1f} deg | height {f['height_m']:.3f} m",
                 f"speed {f['speed_mps']:.3f} m/s | stable {f['stable']}"]
        for j, line in enumerate(lines):
            draw.text((x+8,y+275+22*j),line,font=FONT,fill='#ffa552' if j==0 and f['force_n']>0 else 'white')
    canvas.save(path, quality=92)


reports=[]
comparison_images=[]
for case in read(D/'records/video_selection.json')['cases']:
    folder=D/('videos_v3' if case['seed_index']>1 else 'videos_v2')/case['case_id']
    if not (folder/'video.json').exists():
        continue
    meta=read(folder/'video.json')
    index,env=case['seed_index'],case['env_index']
    source=P6/'force_01'/case['robot'] if case['force_scale']==1 else D/('scale_'+str(case['force_scale']).replace('.','p'))
    original=read(source/f'seed_{index:02d}.json'); replay=read(folder/f'seed_{index:02d}.json')
    a=arrays(source/f'seed_{index:02d}.npz'); b=arrays(folder/f'seed_{index:02d}.npz')
    p=read(folder/'parameters.json'); definition=read(folder/'definition.json')
    assert read(folder/'source_identity.json')['checkpoint_sha256']==read(source/'source_identity.json')['checkpoint_sha256']
    assert sha(folder/f'seed_{index:02d}.npz')==replay['raw_sha256']
    assert np.array_equal(a['valid'],b['valid']) and np.array_equal(a['steps'],b['steps'])
    assert np.array_equal(a['physics_valid'],b['physics_valid'])
    assert (meta['seed_index'],meta['seed'],meta['env_index'])==(index,case['seed'],env)
    assert case['expected_episode']==original['episodes'][env]
    outcomes=[]
    for x,y in zip(original['episodes'],replay['episodes'],strict=True):
        keys=('env_id','stratum','episode_steps','planned_push_count','success','failure_label','recovered_event_count','event_recovery_control_steps','invalid_injection_count')
        outcomes.append(all(x[k]==y[k] for k in keys))
    differences={}
    for k in a:
        mask=b['physics_valid'] if k.startswith('physics_') else b['valid']
        x,y=a[k],b[k]
        if x.shape[:2]==mask.shape:
            x,y=x[mask],y[mask]
        differences[k]={'exact':bool(np.array_equal(x,y)), 'max_abs_difference':float(np.max(np.abs(x.astype(float)-y.astype(float))))}
    rows=replay['episodes']; selected=rows[env]; n=selected['episode_steps']
    assert selected==meta['selected_episode'] and n==meta['frame_count']
    q=b['quaternion_xyzw'][:n,env];x,y,z,w=q.T
    roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
    tilt=np.maximum(abs(roll),abs(pitch));height=b['position'][:n,env,2]-replay['origins'][env][2]
    speed=np.linalg.norm(b['linear'][:n,env,:2],axis=1);angular=np.linalg.norm(b['angular'][:n,env],axis=1)
    stable=(tilt<=p['recovery_tilt_rad'])&(height>=p['recovery_height_m'])&(speed<=p['recovery_velocity_mps'])&(angular<=p['recovery_angular_velocity_radps'])
    decimation=round(definition['control_dt_s']/definition['physics_dt_s'])
    force=np.linalg.norm(b['physics_requested_force_w'].reshape(-1,decimation,64,3)[:n,0,env],axis=1)
    facts=[{'step':i+1,'force_n':float(force[i]),'tilt_deg':float(np.degrees(tilt[i])), 'height_m':float(height[i]),'speed_mps':float(speed[i]),'stable':bool(stable[i])} for i in range(n)]
    keys={1,n,int(tilt.argmax())+1,int(height.argmin())+1,int(speed.argmax())+1}
    for e,s in zip(selected['force_events'],selected['event_recovery_control_steps']):
        keys.update([max(1,e['start_control_step']), min(n,e['end_control_step'])])
        if s is not None:keys.add(s)
    keys=sorted(keys)
    timeline=sorted(set(range(1,n+1,8))|{n})
    wanted=set(keys)|set(timeline); pictures={}
    video=folder/'video'/Path(meta['file']).name
    assert sha(video)==meta['sha256']
    stream=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    header=next(stream); count=0
    for count,frame in enumerate(stream,1):
        if count in wanted:
            pictures[count]=Image.frombytes('RGB',header['size'],frame)
    assert count==n and tuple(header['size'])==(1280,720) and abs(header['fps']-50)<1e-6
    out=D/'visual_review'/case['case_id'];out.mkdir(parents=True,exist_ok=True)
    sheet(case,keys,pictures,facts,'Native simulation - push / worst posture / recovery / terminal',out/'keyframes.jpg')
    if case['robot']=='go2' and index==1 and env==35:
        comparison_path=out/'same_case_comparison_row.jpg'
        sheet(case,[51,62,n],pictures,facts,
              f"Same checkpoint / seed 10042 / env 35 | force x{case['force_scale']} | before / force end / episode end",
              comparison_path)
        comparison_images.append(Image.open(comparison_path).copy())
    for i in range(0,len(timeline),12):
        sheet(case,timeline[i:i+12],pictures,facts,'Native simulation - chronological frames every 0.16 seconds',out/f'timeline_{i//12+1:02d}.jpg')
    report={'case_id':case['case_id'],'all_64_outcomes_equal':all(outcomes), 'equal_outcome_count':sum(outcomes),
        'all_valid_trajectory_arrays_exact':all(v['exact'] for v in differences.values()), 'differences':differences,
        'decoded_frames':count,'metadata':header,'selected_episode':selected,
        'keyframe_steps':keys,'frame_facts':facts,'native_video_sha256':sha(video)}
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    reports.append({k:v for k,v in report.items() if k not in ('differences','frame_facts','selected_episode','metadata')})
    print(case['case_id'],sum(outcomes),'exact',report['all_valid_trajectory_arrays_exact'],'frames',count,flush=True)
result={'status':'PASS' if all(r['all_64_outcomes_equal'] and r['all_valid_trajectory_arrays_exact'] for r in reports) else 'DIFFERENCES',
    'cases':reports,'replayed_episodes_not_new_samples':64*len(reports),'total_decoded_frames':sum(r['decoded_frames'] for r in reports),
    'review_script_sha256':sha(Path(__file__))}
if len(reports)!=8:
    result['status']='INCOMPLETE'
(D/'video_comparison.json').write_text(json.dumps(result,indent=2))
canvas=Image.new('RGB',(1080,sum(im.height for im in comparison_images)),'#101820')
y=0
for im in comparison_images:
    canvas.paste(im,(0,y));y+=im.height
canvas.save(D/'visual_review/same_case_strength_comparison.jpg',quality=94)
print(result['status'],result['total_decoded_frames'])
