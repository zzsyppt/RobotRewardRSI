"""Offline M10 qualification/replay, not a production collection-contract adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import _load_contract, iter_stair_episodes
from true_rl_agent.tasks.stair_climbing.trajectory_terrain import CATALOG_VERSION, CONFIG, STATUS, measure_terrain, summarize, statements
from true_rl_agent.tasks.stair_climbing.trajectory_report import report_bytes, validate_stair_trajectory_report
from true_rl_agent.tasks.stair_climbing.trajectory_artifacts import file_sha256, peak_memory_bytes

# Audit fixtures, not production admission. Future sources require a recorded
# collection/asset/config contract; this offline command does not implement it.
OFFLINE_QUALIFICATION_SHA = '84aecf9e412be7578ee567ca65cc1c6ff48351f8f45c1fb789a1a61913e8a48f'
OFFLINE_PUBLIC_SHA = {
    'k1': '2e1e66a218dc6e1e60bbd005b27167c546b1199d7b4e763b943eb7e7cf05b331',
    'k2': 'b9e41070b79c4b102e960af6a1807781a6129dbc339f43241727f7269a504e36',
    'k3': '48e3bb615a55100f3190b7ca6b1256e9057c7086cc95ee24d4ced6cb123ea4b6',
}

def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode())


def build(reference, source_id, output, qualification, qualification_sha256, base_public):
    start, cpu = time.perf_counter(), time.process_time()
    if file_sha256(qualification) != qualification_sha256 or qualification_sha256 != OFFLINE_QUALIFICATION_SHA:
        raise ValueError('historical qualification artifact hash mismatch')
    archive = _load_contract(reference)
    history = json.loads(qualification.read_text())
    historical = next((s for s in history['sources'] if s['manifest_sha256'] == reference['manifest_sha256']), None)
    if historical is None:
        raise ValueError('no offline historical source qualification; production contract not implemented')
    md = archive.manifest['metadata']
    if any(md[k] != v for k, v in historical['metadata_geometry'].items()):
        raise ValueError('historical geometry metadata differs')
    # CPU formulas and spec expansion must match the actual historical source bytes.
    frozen_names = ('rough_surface.py', 'ordered_evaluation.py', 'unified_evaluation.py', 'unified_training.py',
                    'isaaclab_irregular_terrain.py', 'isaaclab_rough_terrain.py', 'isaaclab_evaluation_terrain.py')
    for item in historical['files']:
        name = Path(item['path']).name
        if hashlib.sha256(item['text'].encode()).hexdigest() != item['sha256']:
            raise ValueError('historical source text/hash mismatch')
        if name in frozen_names and file_sha256(ROOT/'src/true_rl_agent/tasks/stair_climbing'/name) != item['sha256']:
            raise ValueError('current nominal geometry implementation differs from historical source')
    if historical['env_cell_ids'] != [c.cell_id for c in archive.cells]:
        raise ValueError('historical environment/cell mapping differs')
    cells = {c['cell_id']: c for c in historical['cells']}
    for cell in archive.cells:
        c = cells[cell.cell_id]
        if (c['geometry'] != cell.spec.geometry_payload() or c['geometry_digest'] != cell.geometry_digest
                or c['terrain_type'] != cell.terrain_type_index or c['expected_origin_w'] != list(cell.expected_origin_m)):
            raise ValueError('historical spec/type/origin mapping differs')
    # This lock is intentionally confined to the offline runner, never a production admission rule.
    approved_base = ROOT/'outputs/p2a'/source_id/'public.json'
    if file_sha256(approved_base) != OFFLINE_PUBLIC_SHA[source_id]:
        raise ValueError('offline historical public hash differs from P2a freeze')
    if base_public is not None and base_public.read_bytes() != approved_base.read_bytes():
        raise ValueError('offline preview base differs from frozen historical public')
    base = validate_stair_trajectory_report(json.loads(approved_base.read_text()), max_bytes=8192)
    if base['domain']['archive_hash'] != reference['manifest_sha256']:
        raise ValueError('offline base/source mismatch')
    binding = {'manifest_sha256': reference['manifest_sha256'], 'qualification_sha256': qualification_sha256,
               'source_id': source_id, 'identity': archive.manifest['identity'],
               'projection': archive.projection.to_summary_dict(),
               'placement': 'nominal_expected_layout_with_recorded_origins; historical_importer_bytes_unverified',
               'runtime_surface': 'unknown_not_recorded', 'base_file_sha256': file_sha256(approved_base)}
    output.mkdir(parents=True, exist_ok=False)
    episodes, traces, offset = [], [[], [], []], 0
    with (output/'terrain.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for ep in iter_stair_episodes(reference, source_id=source_id):
            cell = archive.cells[ep.env_index]
            records, arrays = measure_terrain(ep, cell.spec.geometry_payload(), geometry_digest=cell.geometry_digest,
                terrain_type=cell.terrain_type_index, regime=cell.terrain_regime_id)
            row = {'episode_id': ep.episode_id, 'run_index': ep.run_index, 'env_index': ep.env_index,
                   'runner_seed': ep.source['runner_seed'], 'dt_s': ep.dt_s,
                   'trace_rows_half_open': [offset, offset+len(ep.steps)], 'records': records}
            offset += len(ep.steps)
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
            episodes.append(row)
            for buffer, array in zip(traces, arrays, strict=True): buffer.append(array)
    np.savez_compressed(output/'traces.npz', **{k: np.concatenate(v) for k, v in
                       zip(('height_world_m', 'relative_z_m', 'status'), traces, strict=True)})
    summary = summarize(episodes)
    claims = statements(binding, summary)
    report = {'schema_version': 'stair.nominal_terrain_supplement.v1', 'binding': binding,
              'catalog_version': CATALOG_VERSION, 'base_catalog_version': 'trajectory.p0.measurements.v2',
              'measurement_config': CONFIG, 'status_codes': list(STATUS), 'summary': summary,
              'optional_statements': claims, 'status': 'offline_not_production_delivery'}
    write(output/'report.json', report)
    preview, included, omitted = base, [], []
    for claim in claims:
        candidate = {'schema_version': 'stair.nominal_terrain_preview.v1', 'status': 'offline_not_production_delivery',
                     'base_report': base, 'optional_nominal_terrain_statements': [*included, claim]}
        if report_bytes(candidate) <= 8192:
            included.append(claim); preview = candidate
        else:
            omitted.append({'id': claim['id'], 'reason': 'optional_whole_sentence_combined_budget'})
    write(output/'preview.json', preview)
    write(output/'preview_receipt.json', {'status': 'offline_not_production_delivery',
          'binding': binding, 'included_ids': [c['id'] for c in included], 'omitted': omitted,
          'combined_bytes': report_bytes(preview), 'max_bytes': 8192})
    package = ROOT/'src/true_rl_agent'
    dependencies = [Path(__file__).resolve(), *sorted((package/'tasks/stair_climbing').glob('trajectory_*.py')),
                    *sorted((package/'evaluation/trajectory_translation').glob('*.py')),
                    *(package/'tasks/stair_climbing'/name for name in (*frozen_names, 'runtime_projection.py', 'evaluation_semantics.py')),
                    *(package/name for name in ('evaluation/binding.py', 'evaluation/process_report.py', 'runner/numeric_evidence.py',
                                                 'composition/cell_manifest.py'))]
    manifest = {'source_manifest_ref': reference, 'binding': binding, 'complete': True,
                'artifacts': {p.name: {'sha256': file_sha256(p), 'bytes': p.stat().st_size}
                              for p in sorted(output.iterdir()) if p.is_file()},
                'code_sha256': {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in dependencies},
                'episode_count': len(episodes), 'trace_rows': offset, 'wall_seconds': time.perf_counter()-start,
                'cpu_seconds': time.process_time()-cpu, 'peak_working_set_bytes': peak_memory_bytes(),
                'api_calls': 0, 'gpu_runs': 0, 'training_runs': 0}
    write(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--source-id', choices=['k1', 'k2', 'k3'], required=True)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--qualification-sha256', required=True)
    p.add_argument('--base-public', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reference = {'schema_version': NUMERIC_EVIDENCE_REF_SCHEMA, 'manifest_path': str(a.manifest.resolve()),
                 'manifest_sha256': a.manifest_sha256}
    print(json.dumps(build(reference, a.source_id, a.output, a.qualification,
                           a.qualification_sha256, a.base_public), indent=2))
