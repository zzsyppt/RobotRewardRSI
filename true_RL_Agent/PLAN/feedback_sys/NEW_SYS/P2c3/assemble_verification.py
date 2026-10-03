"""Index immutable run outputs and create compact reviewer deliverables once."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT/'outputs/p2c3'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def info(p): return {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f: f.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


history = read(OUT/'historical_02/stdout.json')
vendor = read(OUT/'vendor_01/stdout.json')
sources = []
tokens = ('num_rows=1', 'num_cols=', 'border_width=0', 'use_cache=False', 'cell.terrain_regime_id',
          'terrain_env_origins_w', 'terrain_types', 'support_pos_w', 'body_pos_w',
          'start_platform_length / 2.0', 'start_platform_length_m / 2.0', 'expected_origin_m=',
          'faces.extend(((v00', 'terrain_mesh.apply_transform', 'mesh.apply_transform',
          'self.terrain_origins', 'create_prim_from_mesh(', 'faceVertexIndices', 'collider_cfg', 'contact_offset')


def source_info(item):
    lines = item['text'].splitlines(); chosen = set()
    for i, line in enumerate(lines):
        if any(token in line for token in tokens): chosen.update(range(max(0, i-2), min(len(lines), i+4)))
    return {'path': item['path'], 'sha256': item['sha256'],
            'anchors': [{'line': i+1, 'text': lines[i]} for i in sorted(chosen)]}


for source in history['sources']:
    sources.append({k: source[k] for k in ('source_id', 'manifest_path', 'manifest_sha256', 'job_path', 'job_sha256',
                                          'result_path', 'result_sha256', 'job_work_dir', 'env_cell_ids')})
    sources[-1]['files'] = [source_info(f) for f in source['files']]
    sources[-1]['cells'] = [{**{k: v for k, v in c.items() if k != 'meshes'},
        'meshes_sha256_canonical_json': hashlib.sha256(json.dumps(c['meshes'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'mesh_count': len(c['meshes']), 'vertices': sum(len(m['vertices']) for m in c['meshes']),
        'triangles': sum(len(m['faces']) for m in c['meshes'])} for c in source['cells']]
write(HERE/'TERRAIN_SOURCE_EVIDENCE.json', {
    'scope': 'historical task source/config mesh reconstruction plus recorded origins defines a nominal reference model',
    'historical_sources': sources, 'original_historical_stdout': info(OUT/'historical_02/stdout.json'),
    'current_vendor_only': {'scope': vendor['scope'], 'git': vendor['git'],
                          'files': [source_info(f) for f in vendor['files']], 'stdout': info(OUT/'vendor_01/stdout.json')},
    'qualification_boundaries': ['historical importer bytes/revision not verified', 'runtime/cooked collision shape not archived',
        'contact/rest offsets not inferred from None defaults', 'nominal expected world-axis layout only',
        'offline source locks are not a production collection/asset/config admission contract']})

examples, results = {}, []
for sid in ('k1', 'k2', 'k3'):
    directory = OUT/'real_03'/sid
    audit = read(OUT/'raw_audit_03'/(sid+'_audit.json'))
    assert audit['status'] == 'PASS' and audit['independent_extreme_first_index_differences'] == []
    manifest = read(directory/'manifest.json')
    examples[sid] = {k: audit[k] for k in ('examples', 'absent_categories', 'all_statements')}
    results.append({'source_id': sid, 'build_manifest': manifest,
                    'audit_summary': {k: v for k, v in audit.items() if k not in ('examples', 'all_statements', 'summary')},
                    'files': [info(p) for p in sorted(directory.iterdir()) if p.is_file()],
                    'final_oracle': info(OUT/'raw_audit_03'/(sid+'_audit.json'))})
write(HERE/'REAL_EXAMPLES.json', examples)

selection = {'tests_03': {'compatibility', 'frozen_artifacts'}, 'real_03': {'build_k1', 'build_k2', 'build_k3'},
             'tests_04': {'terrain'}, 'final_audit_01': {'independent_spotcheck'}, 'raw_audit_03': {'oracle_k1', 'oracle_k2', 'oracle_k3'}}
commands, all_commands = [], []
for folder in sorted(OUT.iterdir()):
    receipt = folder/'command_receipts.json'
    if not receipt.exists(): continue
    for row in read(receipt):
        row['receipt_file'] = info(receipt)
        row['logs'] = [info(folder/(row['name']+suffix)) for suffix in ('.stdout.txt', '.stderr.txt', '.memory.json')]
        all_commands.append(row)
        if row['name'] in selection.get(folder.name, set()):
            assert row['exit_code'] == 0
            commands.append(row)
remote = [dict(read(OUT/folder/'receipt.json'), receipt_file=info(OUT/folder/'receipt.json'),
               original_stdout=info(OUT/folder/'stdout.json'), original_stderr=info(OUT/folder/'stderr.txt'))
          for folder in ('historical_01', 'historical_02', 'vendor_01')]
new_tests = re.search(r'Ran (\d+) tests', (OUT/'tests_04/terrain.stderr.txt').read_text()).group(1)
compat_tests = re.search(r'Ran (\d+) tests', (OUT/'tests_03/compatibility.stderr.txt').read_text()).group(1)
assert (new_tests, compat_tests) == ('18', '171')
files = [*HERE.glob('*.py'), HERE/'FREEZE_ZH.md', HERE/'DEVELOPMENT_ZH.md', HERE/'TERRAIN_SOURCE_EVIDENCE.json',
         HERE/'REAL_EXAMPLES.json', ROOT/'scripts/build_stair_terrain_supplement.py',
         ROOT/'src/true_rl_agent/tasks/stair_climbing/trajectory_terrain.py',
         *(ROOT/'tests/unit'/name for name in ('terrain_mesh_oracle.py', 'test_terrain_reference.py', 'test_terrain_offline.py'))]
write(HERE/'verification.json', {
    'status': 'development_and_internal_readonly_review_PASS_pending_parent_independent_gate',
    'baseline_commit': '68a60710a9d13534460a7f53755106189ad20bd3', 'prefreeze_commit': '574b841c',
    'qualification': 'nominal configuration/reference-point heights only; not actual runtime surface or production delivery',
    'selected_final_commands': commands, 'all_preserved_local_commands': all_commands, 'remote_readonly_commands': remote,
    'sources': results, 'checks': {'new_tests': 18, 'compatibility_tests': 171, 'failures': 0, 'errors': 0, 'skips': 0,
        'episodes': 2592, 'valid_rows': 851583, 'point_samples': 4257915, 'records': 12960, 'independent_first_extrema': 51840,
        'known_nominal_points': 4253328, 'unknown_nominal_points': 4587,
        'maximum_triangle_formula_error_m': max(s['audit_summary']['maximum_triangle_formula_error_m'] for s in results),
        'new_real_preview_sentences': 0, 'new_candidate_sentences': 45, 'p2a_frozen_files': 33, 'p1b_frozen_card_files': 3,
        'push_frozen_files': 22, 'old_v2_actual_replays': 3},
    'cost': {'selected_final_local_wall_seconds': sum(c['wall_seconds'] for c in commands),
        'all_preserved_local_wall_seconds': sum(c['wall_seconds'] for c in all_commands),
        'peak_working_set_bytes': max(c['memory']['peak_working_set_bytes'] for c in commands),
        'final_build_artifact_bytes': sum(sum(f['bytes'] for f in s['build_manifest']['artifacts'].values()) for s in results),
        'remote_wall_seconds': sum(r['wall_seconds'] for r in remote), 'api_calls': 0, 'functional_llm_calls': 0,
        'training_runs': 0, 'gpu_runs': 0, 'server_writes': 0, 'new_collection_runs': 0,
        'scope': 'measured subprocess receipts; interactive source reads and reviewer exploration excluded'},
    'development_corrections': ['unknown witness position/origin nonfinite components now null',
        'offline qualification and historical public content locked externally; no caller-resigned text admission',
        'first historical read_text normalized mixed newlines; historical_02 preserves original decoded bytes',
        'runpy relative __file__ resolved before manifest relative path calculation',
        'test hash-mutation patched actual file_sha256 path instead of unused read_bytes',
        'final raw oracle uses independent mesh first indices; trace sorting remains an extra check'],
    'remaining': ['parent independent stage gate', 'historical importer revision/bytes not verified',
        'runtime cooked geometry and actual offsets absent', 'normal production collection qualification replaces historical fixture locks later',
        'production sentence budgeting/delivery separate stage', 'K5/LLM results untouched'],
    'files': [info(p) for p in files]})
print(json.dumps({'status': 'PASS', 'verification': info(HERE/'verification.json')}))
