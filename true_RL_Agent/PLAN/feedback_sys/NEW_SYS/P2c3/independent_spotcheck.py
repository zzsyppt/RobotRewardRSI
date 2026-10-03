"""Read-only Astra reviewer's independent C2 face probe, preserved for repetition.

This selects a non-flat first-tread example after implementation; it is an
additional audit, not the predeclared lexicographic example selection.
No production or full-array oracle import.
"""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path('.')
history_path=root/'outputs/p2c3/historical_02/stdout.json'
history=json.loads(history_path.read_text(encoding='utf-8'))
lock=json.loads((root/'PLAN/feedback_sys/NEW_SYS/P2a/source_lock.json').read_text(encoding='utf-8'))['sources'][0]
manifest_path=root/lock['manifest_relative_path']; manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==lock['manifest_sha256']
source=history['sources'][0]; cells={c['cell_id']:c for c in source['cells']}
rough_envs=[e for e,cid in enumerate(source['env_cell_ids']) if cells[cid]['regime']=='c2_rough']
origins=None; pick=None
for chunk in manifest['chunks']:
 if chunk['metadata']['run_index']!=1: continue
 cp=manifest_path.parent/chunk['file']; assert hashlib.sha256(cp.read_bytes()).hexdigest()==chunk['sha256']
 with np.load(cp,allow_pickle=False) as z:
  if origins is None: origins=z['terrain_env_origins_w'].astype(float); types=z['terrain_types'].copy()
  for row in range(len(z['valid'])):
   for env in rough_envs:
    if not z['valid'][row,env]: continue
    cell=cells[source['env_cell_ids'][env]]; g=cell['geometry']; o=origins[env]
    for col,p in enumerate(z['support_pos_w'][row,env].astype(float)):
     local=p-o+cell['mesh_origin']; x,y=local[:2]
     if not (g['start_platform_length_m']+.015<x<g['start_platform_length_m']+g['tread_depths_m'][0]-.015 and .01<y<g['size_m'][1]-.01):continue
     best=None
     for mi,mesh in enumerate(cell['meshes']):
      vertices=np.array(mesh['vertices'],dtype=float)
      for fi,face in enumerate(mesh['faces']):
       tri=vertices[face]; normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
       if normal[2]<=0:continue
       w=np.linalg.solve(np.vstack((tri[:,:2].T,np.ones(3))),np.array([x,y,1.]))
       if min(w)<-1e-12 or max(w)>1+1e-12:continue
       hz=float(w@tri[:,2])
       if best is None or hz>best['mesh_height']:
        best={'mesh_index':mi,'face_index':fi,'triangle':tri.tolist(),'barycentric':w.tolist(),'mesh_height':hz}
     if best and best['mesh_index']==1 and abs(best['mesh_height']-g['riser_heights_m'][0])>1e-5:
      pick={'source':'k1','manifest_sha256':lock['manifest_sha256'],'history_file':str(history_path),'history_sha256':hashlib.sha256(history_path.read_bytes()).hexdigest(),'chunk':chunk['file'],'chunk_sha256':chunk['sha256'],'run':1,'env':env,'step':chunk['metadata']['step_start']+row,'row':row,'support_column':col,'support_name':manifest['metadata']['runtime_projection']['resolved_roles']['support_effector'][col],'terrain_type':int(types[env]),'cell_id':cell['cell_id'],'geometry_digest':cell['geometry_digest'],'point_world':p.tolist(),'origin_world':o.tolist(),'mesh_xy':local[:2].tolist(),**best,'height_world':best['mesh_height']+o[2],'relative_z':float(p[2]-o[2]-best['mesh_height'])}
      break
    if pick:break
   if pick:break
 if pick:break
print(json.dumps(pick,indent=2))
cell=cells[pick['cell_id']]; edge=cell['geometry']['start_platform_length_m']; guard=1e-4
for x in [edge-guard,np.nextafter(edge-guard,-np.inf),edge+guard,np.nextafter(edge+guard,np.inf)]:
 print('BOUNDARY_LOCAL_X',repr(float(x)),'in_closed_guard',bool(edge-guard<=x<=edge+guard))
