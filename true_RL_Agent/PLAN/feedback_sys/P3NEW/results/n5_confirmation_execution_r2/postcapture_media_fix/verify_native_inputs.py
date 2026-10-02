"""Read-only native rebuild against actual observer inputs, not corrected facts execution."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
S, C = B / 'n5_confirmation_source_r2', B / 'n5_execution_20260930_r2'
CAP = B / 'n5_confirmation_20260930_r2'
P = S / 'PLAN/feedback_sys/P3NEW'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path[:0] = [str(S / 'src'), str(P), str(C), str(S)]
import prepare_native_observer as native
from sol_observer_control import blind_text
from true_rl_agent.evaluation.visual_observer import _public_window


def ref(path):
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(path.read_bytes())


def main():
    output = Path(__file__).with_name('native_input_comparison.json')
    assert not output.exists()
    rows = []
    for task in ('stair', 'push'):
        selection_path = CAP / 'observer_selection' / task / 'selection.json'
        for chosen in read(selection_path)['episodes']:
            alias = chosen['alias']
            packet = CAP / 'observers' / task / alias
            built = native.build(CAP / 'exports' / task, selection_path, episode_alias=alias)
            window, = built['windows']
            assert window.alias == alias
            assert (packet / 'selection.json').read_bytes() == selection_path.read_bytes()
            assert built['prepared']['request_bytes'] == (packet / 'request.json').read_bytes()
            assert blind_text(built['prepared']['payload']) == read(packet / 'blind_inputs/blind_input_text.json')
            materials = read(packet / 'blind_inputs/materials.json')['materials']
            images = [item for item in materials if item['role'] == 'original_image']
            assert len(images) == len(window.images) == 12
            for image, actual in zip(window.images, images, strict=True):
                path = packet / 'blind_inputs' / actual['path']
                assert image.data == path.read_bytes()
                assert hashlib.sha256(image.data).hexdigest() == actual['sha256']
                assert image.displayed_steps == (actual['step'],) and image.view_id == actual['view_id']
                assert [image.width, image.height] == actual['image_size_px']
            public = _public_window(window)
            encoded = json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            rows.append({'task': task, 'alias': alias, 'source_run': chosen['run_index'],
                'source_env': chosen['env_index'], 'selected_steps': chosen['selected_steps'],
                'selection': ref(selection_path), 'request': ref(packet / 'request.json'),
                'blind_text': ref(packet / 'blind_inputs/blind_input_text.json'),
                'materials': ref(packet / 'blind_inputs/materials.json'),
                'canonical_complete_public_window_sha256': hashlib.sha256(encoded).hexdigest(),
                'complete_request_and_blind_text_identical': True, 'all_images_byte_and_order_identical': True,
                'image_count': len(images)})
            print(json.dumps({'task': task, 'alias': alias, 'input_comparison': 'PASS'}), flush=True)
    result = {'schema_version': 'p3new.n5_media_native_input_comparison.v1',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS_NATIVE_INPUT_REBUILD_ONLY',
        'script': ref(Path(__file__)), 'native_source': ref(Path(native.__file__)), 'units': rows,
        'corrected_media_generator_executed': False, 'model_requests': 0, 'gpu_runs': 0,
        'observer_or_root_answers_read': False, 'input_files_modified': False,
        'scope': 'The corrected formal call uses a byte-identical copy of this actual v2 selection and the same alias. This proves native input reconstruction; corrected media facts artifacts have not yet been generated or compared.'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
