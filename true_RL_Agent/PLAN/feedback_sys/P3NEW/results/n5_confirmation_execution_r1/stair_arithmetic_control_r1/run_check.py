"""One-off CPU control: frozen N5 Stair selection and arithmetic, no review calls."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
SOURCE = BASE / 'n5_postprocess_source_r1'
OUT = BASE / 'n5_execution_20260930/control/stair_arithmetic_r1'
SOURCE_MANIFEST = BASE / 'n5_postprocess_source_r1_inputs/source_manifest.json'
CARD = BASE / 'n5_execution_20260930/capture_card.json'
CAPTURE = BASE / 'n5_confirmation_20260930/stair'
MANIFEST = CAPTURE / 'numeric_evidence/manifest.json'
FULL = CAPTURE / 'evaluation/public_dev/behavior_full.json'
EXPECTED = {
    SOURCE_MANIFEST: '9b19b13e42299bc8b78a9a34e332fc99a0dea7e5ac945f8ea077d8425f8b4078',
    CARD: '2921fa1a73b021529f9565a696449fed3336d7675183b78282d8460ddf604fed',
    MANIFEST: 'e8520d580f53408f1c21c7ad9056af8fde167135cc03d5251a019f903753b4a1',
    FULL: '4761c25458a3817b4e9a3c20893f749067ef0d70c40166d30c25387a6745b40a',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_bytes())


def load(name):
    path = SOURCE / 'PLAN/feedback_sys/P3NEW' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, str(path)
    source_manifest = read(SOURCE_MANIFEST)
    for relative, entry in source_manifest['files'].items():
        assert sha(SOURCE / relative) == entry['sha256'], relative
    assert source_manifest['file_count'] == len(source_manifest['files']) == 445
    sys.path.insert(0, str(SOURCE / 'src'))
    sys.path.insert(0, str(SOURCE / 'PLAN/feedback_sys/P3NEW'))
    E = load('export_confirmation_raw')
    P = load('prepare_confirmation_materials')
    V = load('verify_stair_confirmation_arithmetic')
    import numpy as np
    from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
    from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
    from true_rl_agent.tasks.stair_climbing.terrain_evidence import stair_surface_query_for_ordered_cell

    manifest, card = E.read_manifest(MANIFEST), read(CARD)
    assert card['status'] == 'FROZEN'
    E.verify_confirmation_source('stair', manifest, card['tasks']['stair'])
    rows = E.source_rows('stair', manifest, 'stair_source')
    selected = E.legacy.select(rows, 'stair')  # Same frozen selection; full not loaded yet.
    assert len(selected) == len({row['id'] for row in selected}) == 12
    aliases = {row['id']: f'stair-{i:04d}' for i, row in enumerate(
        sorted(rows, key=lambda row: (row['run'], row['env'])), 1)}
    full = read(FULL)
    assert full['domain']['archive_hash'] == EXPECTED[MANIFEST]
    indexed = {row['episode_id']: (i, row) for i, row in enumerate(full['episodes'])}
    assert len(indexed) == len(rows) == len(full['episodes'])
    assert set(indexed) == {row['id'] for row in rows}
    shared = {'domain': full['domain'], 'semantics': full['semantics'], 'contract': full['semantic']['contract']}
    meta = manifest['metadata']
    cells = StairUnifiedEvaluationSuite.from_mapping(meta['evaluation_suite'],
        training_profile=StairUnifiedTrainingProfile.from_mapping(meta['training_profile'])).private_cell_assignment('public_dev')
    OUT.mkdir(parents=True, exist_ok=False)
    material_root = OUT / 'control_materials'
    E.write(OUT / 'selected_episodes.json', selected)
    results, loaded_sources = [], []
    for run in [seed['run_index'] for seed in meta['seeds']]:
        arrays, static, terms, missing, chunks = E.load_run(MANIFEST, manifest, 'stair', run)
        loaded_sources.append({'run': run, 'missing_fields': missing, 'chunks': chunks})
        for record in [row for row in selected if row['run'] == run]:
            env = record['env']
            raw, positions = E.episode_arrays(arrays, env, 'episode_length', record['steps'])
            per_static = {key: value[env].copy() for key, value in static.items()}
            cell = cells[env]
            assert cell.cell_id == record['cell']
            assert int(per_static['terrain_types']) == cell.terrain_type_index
            origin = per_static['terrain_env_origins_w']
            assert np.allclose(origin, cell.expected_origin_m, rtol=0, atol=1e-5)
            dt = next(seed['step_dt'] for seed in meta['seeds'] if seed['run_index'] == run)
            curves, summary = E.stair_measurements(raw, origin, stair_surface_query_for_ordered_cell(cell.spec),
                meta['semantics']['episode_outcomes']['signals'], dt, terms)
            definitions = E.stair_definitions(meta, cell, terms)
            terminal = {key: value for key, value in record['terminal'].items() if key not in ('bin_id', 'evaluation_cell_id')}
            alias = aliases[record['id']]
            row = {'episode_id': alias, 'task': 'stair', 'run_group': f'run_{run:02d}',
                'scope': record['scope'], 'cell': record['cell'], 'steps': record['steps'], 'dt_s': dt,
                'original_success': record['success'], 'original_outcome': record['outcome'],
                'original_terminal_record': terminal, 'missing_fields': missing, 'direct_measurements': summary}
            roundtrip = E.export_episode(material_root, row, raw, per_static, curves, definitions, positions, 100)
            dest = material_root / 'episodes' / alias
            header = read(dest / 'episode.json')
            parts = [read(dest / item['file']) for item in header['parts']]
            related = P.related_episode(header, parts, dest / header['original_values_npz']['file'])
            index, episode = indexed[record['id']]
            derivation = {'source': {'full': ref(FULL), 'raw_manifest': ref(MANIFEST)},
                'source_pointer': f'/episodes/{index}', 'episode_id': alias,
                'original_logical_id': record['id'], 'shared': shared, 'complete_episode_derivation': episode}
            E.write(dest / 'related_numeric.json', related)
            E.write(dest / 'full_derivation.json', derivation)
            checked = V.verify_stair_episode(related, derivation)
            result_path = OUT / 'results' / (alias + '.json')
            E.write(result_path, checked)
            results.append({'episode_id': alias, 'original_id': record['id'], 'steps': record['steps'],
                'selection_role': record['selection_role'], 'status': checked['status'],
                'checked_n': checked['checked_n'], 'failed_n': checked['failed_n'],
                'unchecked_n': checked['unchecked_n'], 'failed_fields': [c['field'] for c in checked['checks'] if c['status'] == 'fail'],
                'result': ref(result_path), 'related_numeric': ref(dest / 'related_numeric.json'),
                'full_derivation': ref(dest / 'full_derivation.json'), 'slice_roundtrip': roundtrip})
        print(json.dumps({'run_complete': run, 'episodes_checked': len(results)}), flush=True)
    assert len(results) == 12
    prohibited = [name for name in sys.modules if name.endswith(('.behavior', '.process_report', '.visual_observation', '.visual_review'))]
    assert not prohibited, prohibited
    actual_modules = {}
    for name, module in sys.modules.items():
        path = getattr(module, '__file__', None)
        if path and Path(path).resolve().is_relative_to(SOURCE):
            actual_modules[name] = ref(Path(path))
    for path, expected in EXPECTED.items():
        assert sha(path) == expected, str(path)
    for relative, entry in source_manifest['files'].items():
        assert sha(SOURCE / relative) == entry['sha256'], relative
    receipt = {'schema_version': 'p3new.n5_stair_arithmetic_control.v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'FAIL' if any(r['failed_n'] for r in results) else 'PASS_CHECKED_FIELDS_WITH_UNCHECKED',
        'scope': 'CPU arithmetic only for the existing fixed 12 Stair numeric selections; not formal material packing, semantic acceptance, complete independent L0 coverage or full 864-episode coverage',
        'selection': 'unchanged frozen export_confirmation_raw.source_rows -> P3R.select/rank; made before reading full',
        'model_calls': 0, 'gpu_calls': 0, 'observer_answers_read': False, 'formal_judgements_read': False,
        'input_refs': {path.name if path != MANIFEST else 'numeric_manifest': ref(path) for path in EXPECTED},
        'execution_script': ref(Path(__file__)), 'source_files_verified_before_and_after': 445,
        'source_material_strategy': 'Existing lower-level exporter and related_episode functions, saved as control materials only; no fake formal export manifest or packing receipt',
        'source_operand_link': 'NOT_RUN: frozen source-link verifier needs actual final raw_export manifest and packing receipt, which depend on both tasks. Force/surface remain supplied direct operands in this limited arithmetic result.',
        'source_raw_inputs_unchanged': True, 'pool_count': len(rows), 'selected_count': len(results),
        'checked_total': sum(r['checked_n'] for r in results), 'failed_total': sum(r['failed_n'] for r in results),
        'results': results, 'loaded_chunk_sources': loaded_sources, 'actual_source_modules': actual_modules,
        'loaded_helper_refs': {name: ref(Path(module.__file__)) for name, module in
            [('exporter', E), ('packing', P), ('arithmetic', V), ('selection', E.legacy)]}}
    E.write(OUT / 'receipt.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'selected_count', 'checked_total', 'failed_total')}), flush=True)


if __name__ == '__main__':
    main()
