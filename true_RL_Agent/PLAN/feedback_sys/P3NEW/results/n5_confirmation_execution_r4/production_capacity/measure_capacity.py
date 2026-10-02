"""R4 production request measurement only: no network, credentials, or model calls."""
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import zipfile
import tokenizers
from tokenizers import Tokenizer

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
CAP = B / 'n5_confirmation_20261001_r4'
S = B / 'n5_confirmation_source_r4_final'
sys.path.insert(0, str(S / 'src'))
from true_rl_agent.prompt_serialization import canonical_prompt_json

def sha(raw): return hashlib.sha256(raw).hexdigest()
def ref(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)}
def read(path): return json.loads(path.read_bytes())
def size(value): return len(canonical_prompt_json(value).encode('utf-8'))

tokenizer_path = B / 'n5_execution_20260930/control/numeric_materials_r1/token_capacity_r1/deepseek_v4_tokenizer.zip'
blob = tokenizer_path.read_bytes()
assert sha(blob) == 'e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495'
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    member = 'deepseek_v4_tokenizer/tokenizer.json'
    raw = archive.read(member)
assert sha(raw) == '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614'
tokenizer = Tokenizer.from_str(raw.decode('utf-8'))
count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)

def measure(path):
    request = read(path)
    blocks, images = [], []
    for mi, message in enumerate(request['messages']):
        content = message['content']
        for bi, block in enumerate([{'type': 'text', 'text': content}] if isinstance(content, str) else content):
            if block['type'] == 'text':
                text = block['text']
                blocks.append({'message_index': mi, 'block_index': bi, 'role': message['role'],
                               'utf8_bytes': len(text.encode('utf-8')), 'sha256': sha(text.encode('utf-8')), 'tokens': count(text)})
            elif block['type'] == 'image_url':
                url = block['image_url']['url']
                assert url.startswith('data:image/png;base64,')
                pixels = base64.b64decode(url.split(',', 1)[1], validate=True)
                assert pixels[:8] == b'\x89PNG\r\n\x1a\n'
                images.append({'message_index': mi, 'block_index': bi, 'sha256': sha(pixels), 'bytes': len(pixels),
                               'size_px': list(struct.unpack('>II', pixels[16:24]))})
            else:
                raise ValueError(block['type'])
    total = sum(block['tokens'] for block in blocks)
    return {'request': ref(path), 'model': request['model'], 'max_output_tokens': request['max_tokens'],
            'thinking': request.get('thinking'), 'reasoning_effort': request.get('reasoning_effort'),
            'message_count': len(request['messages']), 'text_blocks': blocks, 'images': images, 'image_count': len(images),
            'all_message_text_utf8_bytes': sum(block['utf8_bytes'] for block in blocks),
            'complete_message_content_text_tokens': total, 'text_plus_output_tokens': total + request['max_tokens']}

assembly_path = CAP / 'delivery_offline_r1/assembly_receipt.json'
assembly = read(assembly_path)
authors, observers = [], []
for task in ('stair', 'push'):
    root = CAP / 'delivery_offline_r1' / task
    row = measure(root / 'request.http_body.json')
    payload = read(root / 'user_payload.json')
    assert read(root / 'request.http_body.json')['messages'][1]['content'].encode('utf-8') == (root / 'user_payload.json').read_bytes()
    budget = payload['authoring_context_frame']['budget_metadata']
    assert size(payload) == budget['actual_prompt_payload_bytes']
    for key, expected in budget['section_serialized_bytes'].items():
        assert size(payload[key]) == expected
    task_receipt = budget['provider_task_sections']
    task_bytes = size({key: payload[key] for key in task_receipt['section_keys']})
    assert task_bytes == task_receipt['serialized_bytes']
    history = payload['public_history']
    memory = history['experiment_memory']
    section_max = max(budget['section_serialized_bytes'].values())
    row.update(task=task, payload=ref(root / 'user_payload.json'), public_report=ref(root / 'public_report.json'),
               public_report_limit_bytes=assembly['tasks'][task]['public_limit_bytes'], budget_metadata=budget,
               independently_recomputed={'payload_bytes': size(payload), 'largest_section_bytes': section_max,
                                         'task_sections_bytes': task_bytes, 'public_history_bytes': size(history),
                                         'recent_outcomes_bytes': size(history['recent_outcomes']),
                                         'recent_outcomes_count': len(history['recent_outcomes']),
                                         'experiment_memory_bytes': size(memory), 'memory_experiments': len(memory['experiments']),
                                         'memory_omitted_experiments': memory['omitted_experiment_count']},
               byte_boundaries_pass=(size(payload) <= 262144 and section_max <= 98304 and task_bytes <= 57344 and
                                     size(memory) <= 14336 and len(memory['experiments']) <= 16 and
                                     size(history['recent_outcomes']) <= 8192 and len(history['recent_outcomes']) <= 8 and
                                     (root / 'public_report.json').stat().st_size <= assembly['tasks'][task]['public_limit_bytes']))
    assert row['byte_boundaries_pass'] and row['image_count'] == 0
    authors.append(row)
    for alias in ('e1', 'e2'):
        observer = measure(CAP / 'observers' / task / alias / 'request.json')
        assert observer['image_count'] == 12
        observer.update(task=task, alias=alias, recorded_image_ceiling_per_image_tokens=1024,
                        image_token_ceiling=observer['image_count'] * 1024,
                        text_plus_recorded_image_ceiling_plus_output=observer['text_plus_output_tokens'] + observer['image_count'] * 1024,
                        recorded_http_body_limit_bytes=48 * 1024 * 1024,
                        http_body_within_recorded_limit=observer['request']['bytes'] <= 48 * 1024 * 1024)
        observers.append(observer)
result = {'status': 'R4_CPU_CAPACITY_ONLY_NOT_SENT_NOT_PROVIDER_USAGE', 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
          'script': ref(Path(__file__)), 'assembly': ref(assembly_path), 'tokenizer_zip': ref(tokenizer_path),
          'tokenizer_json_member': member, 'tokenizer_json_sha256': sha(raw), 'tokenizers_version': tokenizers.__version__,
          'method': 'Same frozen tokenizer and per-complete-message-text-block encode(add_special_tokens=False) convention as existing production_author_capacity.json and measure_token_capacity.py.',
          'authors': authors, 'observers': observers, 'model_calls': 0, 'api_calls': 0, 'gpu_calls': 0, 'input_mutations': 0,
          'limits': ['Requests were not sent; no new provider prompt_tokens, image_tokens, framing, fingerprint or usage.',
                     '1024/image and 48 MiB are inherited registered official-doc ceilings, not re-fetched specifications or measured usage.',
                     'Current memory contains zero experiments; no full 16-round capacity demonstration.',
                     'N5 facts judges and their large inputs are outside this production author/observer audit.']}
out = Path(__file__).with_name('production_capacity.json')
with out.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
print(json.dumps({'output': ref(out), 'authors': [(r['task'], r['complete_message_content_text_tokens']) for r in authors],
                  'observers': [(r['task'], r['alias'], r['complete_message_content_text_tokens']) for r in observers]}))
