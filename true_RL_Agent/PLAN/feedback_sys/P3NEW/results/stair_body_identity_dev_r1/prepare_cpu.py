"""One-off CPU preparation using unchanged R3 images/selection; no model calls."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
D = B / 'stair_body_identity_dev_r1'
S = D / 'source'
F = B / 'n5_confirmation_source_r3_final'
C = B / 'n5_execution_20260930_r3'
CAP = B / 'n5_confirmation_20260930_r3'
SCOPE = 'development_after_r3'
CHANGED = {
    'src/true_rl_agent/tasks/stair_climbing/visual_observation.py',
    'configs/robots/unitree_go2_12dof_isaaclab_v1.body_frame.json',
    'PLAN/feedback_sys/P3NEW/prepare_native_observer.py',
}


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def save(path, value):
    with path.open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def require(value, reason):
    if not value:
        raise ValueError(reason)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def changed_paths(a, b, path=''):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        return [p for key in sorted(a.keys() | b.keys()) for p in
                ([path + '/' + key] if key not in a or key not in b else changed_paths(a[key], b[key], path + '/' + key))]
    if isinstance(a, list):
        if len(a) != len(b):
            return [path]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in changed_paths(x, y, path + '/' + str(i))]
    return [] if a == b else [path]


def main():
    require(not any((D / alias).exists() for alias in ('e1', 'e2')), 'Preserve existing outputs')
    base_manifest_path = C / 'runtime_source_manifest.json'
    base = read(base_manifest_path)
    actual, differences = {}, []
    for relative, expected in base.items():
        frozen = ref(F / relative)
        require(frozen['sha256'] == expected['sha256'], 'Frozen R3 source changed: ' + relative)
        current = ref(S / relative)
        actual[relative] = {key: current[key] for key in ('sha256', 'size_bytes')}
        if current['sha256'] != frozen['sha256']:
            differences.append(relative)
    require(set(differences) == CHANGED, 'Development source differs outside the three reviewed files')
    save(D / 'development_source_manifest.json', actual)
    source_receipt = {'scope': SCOPE, 'base_source_manifest': ref(base_manifest_path),
        'base_source_root': str(F), 'development_source_root': str(S),
        'development_source_manifest': ref(D / 'development_source_manifest.json'),
        'verified_frozen_files': len(base), 'changed_files': [
            {'relative_path': relative, 'before': ref(F / relative), 'after': ref(S / relative)}
            for relative in sorted(differences)], 'frozen_source_unchanged': True}
    save(D / 'source_receipt.json', source_receipt)
    sys.path[:0] = [str(S / 'src'), str(S / 'PLAN/feedback_sys/P3NEW'), str(S)]
    native = load('prepare_native_observer', S / 'PLAN/feedback_sys/P3NEW/prepare_native_observer.py')
    carrier = load('existing_sol_observer_control', C / 'postcapture_preparation/sol_observer_control.py')
    carrier.CARD_SHA = ref(C / 'capture_card.json')['sha256']
    carrier.SOURCE_SHA = ref(D / 'development_source_manifest.json')['sha256']
    original_save = carrier.save

    def save_development_receipt(path, value):
        # Reuse the unchanged material generator; stamp its new control receipt
        # before its first write. Never edit the R3 helper or previous receipts.
        if path.name == 'carrier_preparation.json':
            value = {**value, 'scope': SCOPE, 'confirmation_admitted': False,
                'source_selection_scope': 'n5_confirmation',
                'source_scope_is_provenance_only': True, 'reasoning_effort': 'high'}
        original_save(path, value)

    carrier.save = save_development_receipt
    packets = []
    for alias in ('e1', 'e2'):
        old = CAP / 'observers/stair' / alias
        built = native.build(CAP / 'exports/stair', old / 'selection.json', episode_alias=alias)
        before, after = read(old / 'request.json'), built['prepared']['payload']
        public_before = json.loads(before['messages'][1]['content'][0]['text'])
        public_after = json.loads(after['messages'][1]['content'][0]['text'])
        require(public_before['body_image_points'] == [], 'Original input already has points')
        without_points = deepcopy(public_after)
        without_points['body_image_points'] = []
        require(without_points == public_before, 'Public fields other than body_image_points changed')
        normalized_after = deepcopy(after)
        normalized_after['messages'][1]['content'][0]['text'] = before['messages'][1]['content'][0]['text']
        require(normalized_after == before, 'Request envelope, other text or PNG data changed')
        images = [image for window in built['windows'] for image in window.images]
        require(len(images) == 12, 'Must preserve twelve images per packet')
        names = ['FR_calf', 'FL_calf', 'RR_calf', 'RL_calf']
        points = public_after['body_image_points']
        require(len(points) == 48, 'Must have four points per displayed image')
        for image in images:
            subset = [p for p in points if p['step'] == image.displayed_steps[0] and p['view_id'] == image.view_id]
            require([p['body'] for p in subset] == names, 'Wrong named points, step or view')
        packet = D / alias
        packet.mkdir()
        (packet / 'request.json').write_bytes(built['prepared']['request_bytes'])
        (packet / 'selection.json').write_bytes(built['selection_bytes'])
        for name in ('card', 'private_sidecar', 'mechanical_audit', 'alignment'):
            value = {**built[name], 'scope': SCOPE, 'source_selection_scope': 'n5_confirmation',
                'source_scope_is_provenance_only': True, 'confirmation_admitted': False}
            save(packet / (name + '.json'), value)
        binding = {'episode_alias': alias, 'request_sha256': built['card']['request_sha256']}
        carrier.prepare_packet(packet, built, binding)
        old_materials, new_materials = read(old / 'blind_inputs/materials.json'), read(packet / 'blind_inputs/materials.json')
        old_images = [row for row in old_materials['materials'] if row['role'] == 'original_image']
        new_images = [row for row in new_materials['materials'] if row['role'] == 'original_image']
        require(new_images == old_images, 'Carrier PNG metadata/hash/order changed')
        require(len(list((packet / 'blind_inputs').rglob('*.*'))) == 14, 'Expected 14 whitelist files')
        packets.append({'alias': alias, 'scope': SCOPE, 'status': 'PREPARED_NOT_CALLED',
            'original_request': ref(old / 'request.json'), 'new_request': ref(packet / 'request.json'),
            'original_selection': ref(old / 'selection.json'), 'selection': ref(packet / 'selection.json'),
            'materials': ref(packet / 'blind_inputs/materials.json'),
            'carrier_preparation': ref(packet / 'carrier_preparation.json'),
            'same_request_except_body_image_points': True, 'same_system_prompt': True,
            'same_original_png_bytes_and_order': True, 'same_s0_definitions_measurements_claims': True,
            'selected_steps': built['card']['selected_steps'], 'view_ids': built['card']['view_ids'],
            'image_count': len(images), 'points_count': len(points), 'points_per_image': 4,
            'whitelist_file_count': 14,
            'private_changed_paths': changed_paths(read(old / 'private_sidecar.json'), built['private_sidecar'])})
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith('true_rl_agent') or name == 'prepare_native_observer':
            path = getattr(module, '__file__', None)
            if path:
                path = Path(path).resolve()
                require(path.is_relative_to(S), 'Loaded module outside development source: ' + name)
                item = ref(path)
                require(item['sha256'] == actual[path.relative_to(S).as_posix()]['sha256'], 'Loaded source changed')
                loaded[name] = item
    for relative, expected in base.items():
        require(ref(F / relative)['sha256'] == expected['sha256'], 'Frozen source changed during preparation')
    save(D / 'cpu_preparation_receipt.json', {'scope': SCOPE, 'status': 'PREPARED_NOT_CALLED',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'script': ref(Path(__file__)),
        'source_receipt': ref(D / 'source_receipt.json'), 'loaded_modules': loaded,
        'existing_material_generator': ref(Path(carrier.__file__)), 'packets': packets,
        'new_model_calls': 0, 'new_API_calls': 0, 'gpu_calls': 0, 'confirmation_admitted': False,
        'selection_scope_note': 'Original n5_confirmation selection preserved byte-for-byte for provenance; this entire preparation is development_after_r3.',
        'geometry_visibility_claim': 'Projection locates named origins only; visibility and semantic support are not established.'})
    print(json.dumps({'scope': SCOPE, 'packets': len(packets), 'images': 24,
                      'new_model_calls': 0, 'receipt': ref(D / 'cpu_preparation_receipt.json')}))


if __name__ == '__main__':
    main()
