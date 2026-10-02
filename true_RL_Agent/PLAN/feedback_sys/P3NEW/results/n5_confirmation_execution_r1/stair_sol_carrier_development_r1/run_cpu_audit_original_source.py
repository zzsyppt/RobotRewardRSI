"""One-off CPU audit of two unchanged desktop answers against original packets."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
SOURCE = BASE / 'n5_postprocess_source_r1'
ORIGINAL = BASE / 'n5_confirmation_20260930/observers/stair'
WORK = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'PLAN/feedback_sys/P3NEW')]
import prepare_native_observer as native
from true_rl_agent.evaluation.visual_observer import SCHEMA_VERSION, audit_observer_output
from true_rl_agent.evaluation.visual_review import visual_review_binding

def read(path):
    return json.loads(path.read_bytes())

def ref(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

registration = read(WORK / 'registration.json')
invocations = read(WORK / 'desktop_invocations.json')
assert invocations['registration_sha256'] == ref(WORK / 'registration.json')['sha256']
inputs = [ref(WORK / name) for name in ('registration.json', 'desktop_invocations.json')]
raw_calls, observations, issues, calls = [], [], [], []
for index, alias in enumerate(('e1', 'e2')):
    original, response = ORIGINAL / alias, WORK / alias / 'sol_response'
    case = registration['cases'][index]
    assert case['alias'] == alias
    assert ref(original / 'request.json')['sha256'] == case['original_request_sha256']
    materials_path = WORK / alias / 'materials.json'
    assert ref(materials_path)['sha256'] == case['blind_manifest_sha256']
    private = read(original / 'private_sidecar.json')
    built = native.build(Path(private['source_export']), original / 'selection.json', episode_alias=alias)
    assert built['prepared']['request_bytes'] == (original / 'request.json').read_bytes()
    for name in ('card', 'private_sidecar', 'mechanical_audit', 'alignment'):
        assert built[name] == read(original / (name + '.json'))
    image_rows = [row for row in read(materials_path)['materials'] if row['role'] == 'original_image']
    assets = [asset for window in built['windows'] for asset in window.images]
    assert len(image_rows) == len(assets) == 12
    assert all(row['sha256'] == hashlib.sha256(asset.data).hexdigest()
               and row['step'] == asset.displayed_steps[0] and row['view_id'] == asset.view_id
               for row, asset in zip(image_rows, assets))
    answer = read(response / 'raw_answer.txt')
    audit = audit_observer_output(answer, built['windows'], numeric_check=built['numeric_check'])
    save(response / 'mechanical_audit.json', audit)
    raw_calls.append(answer)
    for record in audit['observations']:
        row = deepcopy(record)
        row['reference'] = f'calls[{index}].' + row['reference']
        row['source_call_index'] = index
        observations.append(row)
    issues.extend(f'calls[{index}]:{item}' for item in audit['violations'])
    calls.append({'alias': alias, 'raw_answer': ref(response / 'raw_answer.txt'),
                  'access_receipt': ref(response / 'access_receipt.json'),
                  'audit': ref(response / 'mechanical_audit.json'),
                  'original_request': ref(original / 'request.json')})
    inputs.extend([ref(materials_path), *[ref(response / name) for name in ('raw_answer.txt', 'access_receipt.json')]])
combined = {'schema_version': SCHEMA_VERSION, 'scope': registration['scope'],
            'raw_output': {'calls': raw_calls}, 'observations': observations, 'violations': issues,
            'passed': not issues and not any(row['violations'] for row in observations),
            'semantic_support': 'not_evaluated', 'all_raw_observations_retained': True}
save(WORK / 'combined_audit.json', combined)
receipt = {'scope': registration['scope'], 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
           'original_n5_status': 'INCONCLUSIVE', 'cpu_model_or_gpu_calls': 0,
           'inputs': inputs, 'calls': calls, 'script': ref(Path(__file__)),
           'source': [ref(Path(module.__file__)) for module in (native, sys.modules[audit_observer_output.__module__])],
           'combined': ref(WORK / 'combined_audit.json'), 'visual_review_binding': visual_review_binding(combined),
           'passed': combined['passed'], 'observations': len(observations)}
save(WORK / 'audit_receipt.json', receipt)
print(json.dumps({key: receipt[key] for key in ('passed', 'observations', 'visual_review_binding')}))
