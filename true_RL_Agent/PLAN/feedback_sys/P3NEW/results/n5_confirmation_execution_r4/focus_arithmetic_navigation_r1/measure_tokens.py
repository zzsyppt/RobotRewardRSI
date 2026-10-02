"""CPU-only paired text-token proxy; reuse existing tokenizer ZIP, no install."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import tokenizers
from tokenizers import Tokenizer

sys.dont_write_bytecode = True
root = Path(sys.argv[1])
cache = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930/control/numeric_materials_r1/token_capacity_r1/deepseek_v4_tokenizer.zip')
blob = cache.read_bytes()
assert hashlib.sha256(blob).hexdigest() == 'e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495'
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    raw = archive.read('deepseek_v4_tokenizer/tokenizer.json')
assert hashlib.sha256(raw).hexdigest() == '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614'
tokenizer = Tokenizer.from_str(raw.decode())
def count(text):
    return len(tokenizer.encode(text, add_special_tokens=False).ids)
def measure(path):
    request = json.loads(path.read_bytes())
    texts = []
    for message in request['messages']:
        content = message['content']
        if isinstance(content, str):
            texts.append(content)
        else:
            assert all(block['type'] == 'text' for block in content)
            texts.extend(block['text'] for block in content)
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'messages_text_utf8_bytes': sum(len(text.encode()) for text in texts),
        'messages_text_proxy_tokens': sum(count(text) for text in texts),
        'serialized_messages_proxy_tokens': count(json.dumps(request['messages'], ensure_ascii=False, indent=2)),
        'text_blocks': len(texts), 'original_max_output_tokens': request['max_tokens']}
rows = []
for old_path in sorted((root / 'capacity_inputs').glob('*.old.json')):
    new_path = old_path.with_name(old_path.name.replace('.old.json', '.new.json'))
    old, new = measure(old_path), measure(new_path)
    rows.append({'packet_id': old_path.name.split('.')[0], 'old': old, 'new': new,
        'messages_text_proxy_delta': new['messages_text_proxy_tokens']-old['messages_text_proxy_tokens'],
        'serialized_messages_proxy_delta': new['serialized_messages_proxy_tokens']-old['serialized_messages_proxy_tokens']})
result = {'status': 'MEASURED_CPU_STATIC_ONLY', 'tokenizer': {'zip_path': str(cache),
    'zip_sha256': hashlib.sha256(blob).hexdigest(), 'json_sha256': hashlib.sha256(raw).hexdigest(),
    'library_version': tokenizers.__version__}, 'rows': rows,
    'basis': 'DeepSeek tokenizer text proxy, not actual SOL token usage. Both rows use all original messages; serialized_messages is a separate conservative outer-JSON representation, never add it to messages_text. No images in these six focus requests. No model execution, semantic score or release.',
    'max_new_messages_text_proxy_tokens': max(row['new']['messages_text_proxy_tokens'] for row in rows),
    'max_new_serialized_messages_proxy_tokens': max(row['new']['serialized_messages_proxy_tokens'] for row in rows)}
(root / 'capacity_measurement.json').write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode()+b'\n')
print(json.dumps({'status': result['status'], 'rows': len(rows),
    'messages_text_deltas': [row['messages_text_proxy_delta'] for row in rows],
    'serialized_deltas': [row['serialized_messages_proxy_delta'] for row in rows],
    'max_new_text': result['max_new_messages_text_proxy_tokens'],
    'max_new_serialized': result['max_new_serialized_messages_proxy_tokens']}))
