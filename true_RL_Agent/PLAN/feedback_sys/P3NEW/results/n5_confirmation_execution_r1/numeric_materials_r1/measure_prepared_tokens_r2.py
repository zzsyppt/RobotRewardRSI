"""Count only existing N5 text blocks with the previously frozen static tokenizer."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
CONTROL = BASE / 'n5_execution_20260930/control/numeric_materials_r1'
OUT = CONTROL / 'token_capacity_r2'
TOKENIZER_SHA = '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614'
ZIP_SHA = 'e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495'
URL = 'https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def ref(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': digest(data), 'bytes': len(data)}


def main():
    import tokenizers
    from tokenizers import Tokenizer
    binding_path = CONTROL / 'wiring_arithmetic_receipt.json'
    binding = json.loads(binding_path.read_bytes())
    assert binding['package_confirmation_status'] == 'INCONCLUSIVE'
    requests = []
    for unit in binding['wiring']:
        for judge, variant in unit['variants'].items():
            path = Path(variant['request']['path'])
            assert ref(path) == variant['request']
            requests.append((unit, judge, path))
    assert len(requests) == 72 and len({str(row[2]) for row in requests}) == 72
    OUT.mkdir(exist_ok=False)
    # Reuse the exact static ZIP retained by the first attempt; no network.
    cached_zip = CONTROL / 'token_capacity_r1/deepseek_v4_tokenizer.zip'
    blob = cached_zip.read_bytes()
    assert digest(blob) == ZIP_SHA, 'unexpected static tokenizer ZIP'
    with (OUT / 'deepseek_v4_tokenizer.zip').open('xb') as stream:
        stream.write(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        candidates = [(name, archive.read(name)) for name in archive.namelist() if name.endswith('tokenizer.json')]
    matching = [(name, data) for name, data in candidates if digest(data) == TOKENIZER_SHA]
    assert len(matching) == 1
    tokenizer_member, tokenizer_bytes = matching[0]
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    records, grouped = [], {}
    for unit, judge, path in requests:
        blob = path.read_bytes()
        assert digest(blob) == unit['variants'][judge]['request']['sha256']
        body = json.loads(blob)
        assert body['max_tokens'] == 32768 and len(body['messages']) == 2
        system, user = body['messages']
        assert system['role'] == 'system' and user['role'] == 'user'
        assert isinstance(system['content'], str) and isinstance(user['content'], list)
        parts = [{'role': 'system', 'tokens': count(system['content']),
                  'text_utf8_bytes': len(system['content'].encode('utf-8')),
                  'text_sha256': digest(system['content'].encode('utf-8'))}]
        for block in user['content']:
            assert block['type'] == 'text'
            text = block['text']
            parts.append({'role': json.loads(text)['role'], 'tokens': count(text),
                'text_utf8_bytes': len(text.encode('utf-8')), 'text_sha256': digest(text.encode('utf-8'))})
        total = sum(part['tokens'] for part in parts)
        row = {'packet_id': unit['packet_id'], 'task': unit['task'], 'axis': unit['axis'], 'judge': judge,
            'request': ref(path), 'body_model': body['model'], 'by_role': parts,
            'serialized_text_tokens_deepseek_v4': total,
            'meaning': 'exact frozen-tokenizer count of supplied text blocks, excluding chat framing/special tokens' if judge == 'deepseek' else 'DeepSeek-tokenizer proxy only; not an actual SOL token count',
            'body_model_visible_utf8_bytes': sum(part['text_utf8_bytes'] for part in parts),
            'output_reserve': 32768, 'system_tools_reserve': 32768, 'margin': 8192,
            'conservative_reserved_total_using_this_count': total + 32768 + 32768 + 8192,
            'within_258400_under_proxy_assumption_only': total + 32768 + 32768 + 8192 <= 258400,
            'provider_usage': None, 'actual_agent_peak_tokens': None, 'capacity_release': False}
        grouped.setdefault(unit['packet_id'], []).append(row)
        records.append(row)
    for rows in grouped.values():
        assert len(rows) == 2 and rows[0]['by_role'][1:] == rows[1]['by_role'][1:]
    result = {'schema_version': 'p3new.n5_prepared_static_text_tokens.v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'STATIC_TEXT_COUNT_ONLY_NOT_RELEASED', 'package_confirmation_status': 'INCONCLUSIVE',
        'model_calls': 0, 'gpu_runs': 0, 'facts_admitted': False,
        'tokenizer_acquisition': 'existing_local_static_file_from_preserved_first_attempt; zero network in this count',
        'cached_tokenizer_input': ref(cached_zip),
        'previous_failed_count_execution': ref(CONTROL / 'measure_tokens.execution.json'),
        'tokenizer_url': URL, 'tokenizer_zip': ref(OUT / 'deepseek_v4_tokenizer.zip'),
        'tokenizer_zip_member': tokenizer_member, 'tokenizer_json_sha256': digest(tokenizer_bytes),
        'runtime': {'tokenizers_version': tokenizers.__version__, 'module': ref(Path(tokenizers.__file__))},
        'method': 'Same sum of complete system and each serialized text block as capacity_actual_facts_r1/measure_token_capacity.py; add_special_tokens=False. No HTTP JSON escaping is counted. SOL remains a proxy.',
        'reference_counter': {'repository_file': 'PLAN/feedback_sys/P3NEW/results/capacity_actual_facts_r1/measure_token_capacity.py',
            'sha256': '6c14e3f71ab588da261b97e10e5c27bbd0952cb740c78f33120aea9dddb532e3'},
        'unmeasured': ['provider chat-template/special-token overhead', 'actual SOL tokenizer', 'agent/tool/runtime shell and actual peak context', 'semantic completeness or confirmation admission'],
        'wiring_receipt': ref(binding_path), 'execution_script': ref(Path(__file__)),
        'requests_counted': len(records), 'packet_count': len(grouped),
        'judge_material_text_blocks_identical_for_all_packets': True, 'records': records}
    with (OUT / 'tokenizer_counts.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    largest = sorted(records, key=lambda row: row['serialized_text_tokens_deepseek_v4'], reverse=True)
    print(json.dumps({'status': result['status'], 'package_confirmation_status': 'INCONCLUSIVE',
        'requests': len(records), 'packets': len(grouped), 'tokenizer_sha256': TOKENIZER_SHA,
        'max_text_tokens': largest[0]['serialized_text_tokens_deepseek_v4'],
        'max_reserved_total': largest[0]['conservative_reserved_total_using_this_count'],
        'largest': [{k: row[k] for k in ('packet_id', 'task', 'judge', 'serialized_text_tokens_deepseek_v4')} for row in largest[:6]],
        'result': ref(OUT / 'tokenizer_counts.json')}))


if __name__ == '__main__':
    main()
