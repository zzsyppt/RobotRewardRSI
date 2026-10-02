"""Measure the actual unsent CLI carrier on CPU with the existing tokenizer."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import tokenizers
from tokenizers import Tokenizer

root = Path(sys.argv[1])
cache = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930/control/numeric_materials_r1/token_capacity_r1/deepseek_v4_tokenizer.zip')
blob = cache.read_bytes()
assert hashlib.sha256(blob).hexdigest() == 'e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495'
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    raw = archive.read('deepseek_v4_tokenizer/tokenizer.json')
assert hashlib.sha256(raw).hexdigest() == '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614'
tokenizer = Tokenizer.from_str(raw.decode())
def count(text): return len(tokenizer.encode(text, add_special_tokens=False).ids)
def file_measure(name):
    data = (root / name).read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'utf8_bytes': len(data), 'proxy_tokens': count(data.decode())}
request = json.loads((root / 'request.http_body.json').read_bytes())
texts = [block['text'] for message in request['messages']
    for block in ([{'type': 'text', 'text': message['content']}] if isinstance(message['content'], str) else message['content'])]
policy = json.loads((root / 'N5_SOL_CLI_POLICY_R5.json').read_bytes())
carrier, stdin = file_measure('input.json'), file_measure('stdin.txt')
reserve = sum(policy[name] for name in ('output_reservation_tokens', 'fixed_overhead_reservation', 'proxy_difference_reservation'))
result = {'status': 'STATIC_CPU_MEASURED_NO_CALL', 'request_sha256': hashlib.sha256((root / 'request.http_body.json').read_bytes()).hexdigest(),
    'messages_text_proxy_tokens': sum(count(text) for text in texts), 'text_blocks': len(texts),
    'messages_text_utf8_bytes': sum(len(text.encode()) for text in texts), 'blind_input': carrier, 'full_cli_stdin': stdin,
    'wrapper_proxy_delta': stdin['proxy_tokens']-carrier['proxy_tokens'], 'image_count': 0, 'image_tokens': 0,
    'policy_sha256': hashlib.sha256((root / 'N5_SOL_CLI_POLICY_R5.json').read_bytes()).hexdigest(),
    'process_context_window_requested': policy['model_context_window'], 'process_auto_compact_requested': policy['model_auto_compact_token_limit'],
    'reserved_overhead_output_proxy_tokens': reserve, 'full_stdin_proxy_plus_reserve': stdin['proxy_tokens']+reserve,
    'static_full_stdin_within_input_proxy_limit': stdin['proxy_tokens'] <= policy['input_proxy_limit'],
    'static_proxy_plus_reserve_within_requested_window': stdin['proxy_tokens']+reserve <= policy['model_context_window'],
    'tokenizer': {'zip_path': str(cache), 'zip_sha256': hashlib.sha256(blob).hexdigest(),
        'json_sha256': hashlib.sha256(raw).hexdigest(), 'version': tokenizers.__version__},
    'basis': 'DeepSeek tokenizer proxy only, not SOL tokenizer or runtime usage. Original message text, full blind JSON and full stdin are alternative carriers; never add them. stdin includes the actual unchanged CLI wrapper. Reserved overhead/output/margin is a separate conservative policy reserve, not measured usage.',
    'model_calls': 0, 'actual_usd_cost': None, 'actual_context_usage': None, 'actual_compaction': None,
    'no_release_or_semantic_claim': True}
(root / 'capacity_receipt.json').write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode()+b'\n')
print(json.dumps({key: result[key] for key in ('status', 'messages_text_proxy_tokens', 'full_cli_stdin', 'full_stdin_proxy_plus_reserve', 'static_full_stdin_within_input_proxy_limit', 'static_proxy_plus_reserve_within_requested_window')}))
