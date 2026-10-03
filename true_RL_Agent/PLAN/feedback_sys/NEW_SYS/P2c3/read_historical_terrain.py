"""Read approved historical sources; reconstruct nominal meshes on CPU, no simulator."""
import ast
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import trimesh

started, cpu = time.perf_counter(), time.process_time()
ROOT = Path('/data/yuyangliu/AGENT')
SOURCES = [
    ('k1', 'feedback_p2_audit_20260924/source', 'feedback_p2_audit_20260924/stair_parent_on',
     '0389a10d7cb9dbe14fff063f43c56482fcc6fbd918281a37beacb59c043db8d3'),
    ('k2', 'feedback_p2_20260923/source_r2', 'feedback_p2_20260923/r2_gen8_on',
     'fe488cb848dbe61454c0b9cc4fa6d8f8fc278deb1157065170577d9e91be5f20'),
    ('k3', 'feedback_p3new_20260929/n5_confirmation_source_r4_final',
     'feedback_p3new_20260929/n5_confirmation_20261001_r4/stair',
     '8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83'),
]
FILES = ('isaaclab_evaluation_terrain.py', 'isaaclab_irregular_terrain.py',
         'isaaclab_rough_terrain.py', 'rough_surface.py', 'ordered_evaluation.py',
         'unified_evaluation.py', 'unified_training.py', 'isaaclab_runtime_env.py',
         'isaaclab_reward_runtime.py', 'numeric_capture.py', 'runtime_projection.py')


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def excerpt_file(p):
    return {'path': str(p), 'sha256': digest(p), 'text': p.read_bytes().decode('utf-8')}


def mesh_functions(source):
    """Execute only CPU mesh functions, preserving their original AST verbatim."""
    folder = source/'src/true_rl_agent/tasks/stair_climbing'
    scope = {'np': np, 'trimesh': trimesh, '__name__': 'historical_mesh_cpu'}
    exec(compile((folder/'rough_surface.py').read_text(), str(folder/'rough_surface.py'), 'exec'), scope)
    for name, names in [('isaaclab_irregular_terrain.py', {'build_irregular_stair_meshes', '_box'}),
                        ('isaaclab_rough_terrain.py', {'build_rough_stair_meshes', '_rough_tread_solid', '_box'})]:
        tree = ast.parse((folder/name).read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(tree, str(folder/name), 'exec'), scope)
    return scope


rows = []
for sid, source_name, job_name, sha in SOURCES:
    source, jobdir = ROOT/source_name, ROOT/job_name
    manifest_path = jobdir/'numeric_evidence/manifest.json'
    assert digest(manifest_path) == sha
    manifest = json.loads(manifest_path.read_text())
    job, result = [json.loads((jobdir/n).read_text()) for n in ('public_dev_job.json', 'result.json')]
    assert job['metadata']['work_dir'] == str(source)
    assert result['numeric_evidence_ref']['manifest_sha256'] == sha
    # Reload each historical CPU-only suite implementation, not the current checkout.
    for key in list(sys.modules):
        if key == 'true_rl_agent' or key.startswith('true_rl_agent.'):
            del sys.modules[key]
    sys.path.insert(0, str(source/'src'))
    from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
    from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
    md = manifest['metadata']
    training = StairUnifiedTrainingProfile.from_mapping(md['training_profile'])
    suite = StairUnifiedEvaluationSuite.from_mapping(md['evaluation_suite'], training_profile=training)
    scope = mesh_functions(source)
    cells = []
    for c in suite.runtime_cells('public_dev'):
        g = c.spec.geometry_payload()
        kwargs = dict(size=tuple(c.size_m), floor_depth=c.floor_depth_m,
                      start_platform_length=c.start_platform_length_m,
                      riser_heights_m=tuple(c.riser_heights_m), tread_depths_m=tuple(c.tread_depths_m),
                      top_platform_length=c.top_platform_length_m)
        if g['surface_profile']['kind'] == 'flat':
            meshes, origin = scope['build_irregular_stair_meshes'](**kwargs)
        else:
            meshes, origin = scope['build_rough_stair_meshes'](**kwargs, surface_profile=g['surface_profile'])
        cells.append({'cell_id': c.cell_id, 'terrain_type': c.terrain_type_index,
                      'regime': c.terrain_regime_id, 'geometry': g, 'geometry_digest': c.geometry_digest,
                      'expected_origin_w': list(c.expected_origin_m), 'mesh_origin': origin.tolist(),
                      'meshes': [{'vertices': m.vertices.tolist(), 'faces': m.faces.tolist()} for m in meshes]})
    rows.append({'source_id': sid, 'manifest_path': str(manifest_path), 'manifest_sha256': sha,
                 'job_path': str(jobdir/'public_dev_job.json'), 'job_sha256': digest(jobdir/'public_dev_job.json'),
                 'result_path': str(jobdir/'result.json'), 'result_sha256': digest(jobdir/'result.json'),
                 'job_work_dir': str(source), 'files': [excerpt_file(source/'src/true_rl_agent/tasks/stair_climbing'/n) for n in FILES],
                 'cells': cells, 'env_cell_ids': [c.cell_id for c in suite.private_cell_assignment('public_dev')],
                 'metadata_geometry': {k: md[k] for k in ('training_profile', 'evaluation_suite', 'coordinates', 'runtime_projection')}})
    sys.path.pop(0)
print(json.dumps({'scope': 'historical source and reconstructed nominal meshes only; no physics runtime',
                  'sources': rows, 'server_writes': 0, 'gpu_runs': 0,
                  'wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-cpu,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
