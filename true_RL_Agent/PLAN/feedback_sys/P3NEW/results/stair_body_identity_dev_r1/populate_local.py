"""Reuse existing local original PNGs and verify the downloaded CPU controls."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

D = Path(__file__).resolve().parent
OLD = D.parent / 'n5_confirmation_execution_r3/stair_observer_inputs'


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    geometry_path = OLD / 'geometry_identity_check/geometry_receipt.json'
    geometry = {(row['alias'], row['step'], row['view_id'], link['body_name']): link
                for row in read(geometry_path)['rows'] for link in row['named_link_projection']}
    cpu_path = D / 'cpu_preparation_receipt.json'
    cpu = read(cpu_path)
    require(cpu['scope'] == 'development_after_r3' and cpu['status'] == 'PREPARED_NOT_CALLED', 'Wrong preparation scope/status')
    output = []
    for packet in cpu['packets']:
        alias = packet['alias']
        blind, old_blind = D / alias / 'blind_inputs', OLD / alias / 'blind_inputs'
        manifest_path = blind / 'materials.json'
        require(ref(manifest_path)['sha256'] == packet['materials']['sha256'], 'Downloaded manifest differs')
        materials = read(manifest_path)['materials']
        old_materials = read(old_blind / 'materials.json')['materials']
        require(materials[1:] == old_materials[1:], 'Original image list changed')
        require(ref(D / alias / 'selection.json')['sha256'] == packet['original_selection']['sha256'], 'Selection changed')
        text_path = blind / 'blind_input_text.json'
        require(ref(text_path)['sha256'] == materials[0]['sha256'], 'Downloaded text differs')
        new_text, old_text = read(text_path), read(old_blind / 'blind_input_text.json')
        new_public = json.loads(new_text['messages'][1]['text_blocks'][0])
        old_public = json.loads(old_text['messages'][1]['text_blocks'][0])
        normalized_public = deepcopy(new_public)
        normalized_public['body_image_points'] = []
        require(normalized_public == old_public, 'Fields other than body_image_points changed')
        normalized_text = deepcopy(new_text)
        normalized_text['messages'][1]['text_blocks'][0] = old_text['messages'][1]['text_blocks'][0]
        require(normalized_text == old_text, 'Other text or envelope changed')
        image_dir = blind / 'images'
        image_dir.mkdir(exist_ok=False)
        local_sources, image_keys = [], []
        for row in materials[1:]:
            source, target = old_blind / row['path'], blind / row['path']
            source_ref = ref(source)
            require(source_ref['sha256'] == row['sha256'], 'Existing original PNG changed')
            shutil.copyfile(source, target)
            require(ref(target)['sha256'] == row['sha256'], 'Copied PNG bytes differ')
            local_sources.append(source_ref)
            image_keys.append((row['step'], row['view_id']))
        points = new_public['body_image_points']
        counts = Counter((p['step'], p['view_id']) for p in points)
        require(set(counts) == set(image_keys) and all(counts[key] == 4 for key in image_keys), 'Not four points per actual image')
        max_error = 0.0
        for point in points:
            prior = geometry[(alias, point['step'], point['view_id'], point['body'])]
            require(point['status'] == prior['projection_status'], 'Geometry status differs')
            require(point['xy_px'] == prior['image_xy_px'], 'Geometry projection differs')
            max_error = max(max_error, *(abs(a-b) for a, b in zip(point['xy_px'], prior['image_xy_px'])))
        whitelist = [ref(manifest_path), *[ref(blind / row['path']) for row in materials]]
        require(len(whitelist) == 14 and len(list(blind.rglob('*.*'))) == 14, 'Wrong whitelist file count')
        output.append({'alias': alias, 'scope': 'development_after_r3', 'blind_directory': str(blind),
            'manifest': ref(manifest_path), 'whitelist': whitelist, 'local_original_image_sources': local_sources,
            'displayed_steps': new_public['displayed_steps'], 'image_count': len(image_keys),
            'body_image_points': len(points), 'points_per_image': 4, 'original_png_bytes_unchanged': True,
            'image_metadata_order_unchanged': True, 'same_text_except_body_image_points': True,
            'same_system_prompt': True, 'same_measurements_claims': True,
            'geometry_receipt_points_compared': len(points), 'max_abs_coordinate_difference_px': max_error})
    receipt = {'scope': 'development_after_r3', 'status': 'PREPARED_NOT_CALLED',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'script': ref(Path(__file__)),
        'remote_cpu_preparation_receipt': ref(cpu_path), 'independent_prior_geometry_receipt': ref(geometry_path),
        'packets': output, 'PNG_network_download_bytes': 0, 'new_model_calls': 0,
        'new_API_calls': 0, 'gpu_calls': 0, 'confirmation_admitted': False,
        'scope_limit': 'Named-origin geometric projection only; no visibility or semantic judgment. Original R3 selection scope is preserved provenance.'}
    with (D / 'local_preparation_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'scope': receipt['scope'], 'packets': len(output), 'images': 24,
        'matched_geometry_points': sum(row['geometry_receipt_points_compared'] for row in output),
        'max_error_px': max(row['max_abs_coordinate_difference_px'] for row in output),
        'manifest_sha256': {row['alias']: row['manifest']['sha256'] for row in output}}))


if __name__ == '__main__':
    main()
