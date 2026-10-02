"""Read-only prepared-request text counts; writes one new control JSON, never release."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
CARD = BASE / 'n5_execution_20260930_r2/capture_card.json'
CARD_SHA = '6bea98ed63e1f1e516631e58205e3da93406311d1b87466e445fe754b0c22794'
CACHED_ZIP = BASE / 'n5_execution_20260930/control/numeric_materials_r1/token_capacity_r1/deepseek_v4_tokenizer.zip'
ZIP_SHA = 'e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495'
TOKENIZER_SHA = '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    data = path.read_bytes()
    return data, {'path': str(path.resolve()), 'sha256': digest(data), 'bytes': len(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True,
                        help='Directory recursively containing actual prepare.json/request.http_body.json pairs.')
    parser.add_argument('--output', type=Path, required=True, help='New control JSON file; must not exist.')
    args = parser.parse_args()
    require(args.prepared_root.is_dir(), 'prepared-root must be an existing directory')
    require(not args.output.exists(), 'Preserve existing output; choose a new file')
    prepared_paths = sorted(args.prepared_root.rglob('prepare.json'))
    require(prepared_paths, 'No prepare.json files found')
    card_bytes, card_ref = read(CARD)
    require(card_ref['sha256'] == CARD_SHA, 'R2 capture card SHA differs')
    card = json.loads(card_bytes)
    require(card['status'] == 'FROZEN', 'Expected frozen R2 card')
    materials, sol = card['materials'], card['model_roles']['gpt6_sol_judge']
    rules = {key: materials[key] for key in ('input_token_proxy_limit', 'system_tools_reserved',
             'output_reserved', 'proxy_margin_reserved', 'total_proxy_reserved')}
    rules.update(context_window=sol['context_window'], auto_compact_limit=sol['auto_compact_limit'])
    reserve = rules['system_tools_reserved'] + rules['output_reserved'] + rules['proxy_margin_reserved']
    require(rules['total_proxy_reserved'] == rules['input_token_proxy_limit'] + reserve,
            'Card reservation arithmetic differs')

    # Reuse the original cached ZIP in memory. No extraction, copy, or download.
    import tokenizers
    from tokenizers import Tokenizer
    blob, zip_ref = read(CACHED_ZIP)
    require(zip_ref['sha256'] == ZIP_SHA, 'Frozen tokenizer ZIP SHA differs')
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        candidates = [(name, archive.read(name)) for name in archive.namelist() if name.endswith('tokenizer.json')]
    matching = [(name, data) for name, data in candidates if digest(data) == TOKENIZER_SHA]
    require(len(matching) == 1, 'Expected exactly one frozen tokenizer.json member')
    member, tokenizer_bytes = matching[0]
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode('utf-8'))
    records = []
    for prepared_path in prepared_paths:
        prepared_bytes, prepared_ref = read(prepared_path)
        prepared = json.loads(prepared_bytes)
        require(prepared['schema_version'] == 'p3new.confirmation_review_prepared.v1',
                'Unexpected prepare schema: ' + str(prepared_path))
        body_bytes, request_ref = read(prepared_path.with_name('request.http_body.json'))
        require(request_ref['sha256'] == prepared['request_sha256']
                and request_ref['bytes'] == prepared['request_bytes'],
                'Prepared/request SHA or size differs: ' + str(prepared_path))
        body = json.loads(body_bytes)
        parts, images = [], []
        for message_index, message in enumerate(body['messages']):
            content = message['content']
            blocks = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
            require(isinstance(blocks, list), 'Unsupported message content')
            for block_index, block in enumerate(blocks):
                location = {'message_index': message_index, 'block_index': block_index, 'role': message['role']}
                if block['type'] == 'text':
                    text = block['text']
                    require(isinstance(text, str), 'Non-string text block')
                    raw = text.encode('utf-8')
                    part = {**location, 'text_sha256': digest(raw), 'text_utf8_bytes': len(raw),
                            'tokens': len(tokenizer.encode(text, add_special_tokens=False).ids)}
                    try:
                        label = json.loads(text)
                    except (ValueError, TypeError):
                        label = None
                    if isinstance(label, dict) and isinstance(label.get('role'), str):
                        part['material_role'] = label['role']
                    parts.append(part)
                elif block['type'] == 'image_url':
                    url = block['image_url']['url']
                    image = {**location, 'detail': block['image_url'].get('detail'),
                             'url_sha256': digest(url.encode('utf-8')), 'bytes': None, 'image_tokens': None}
                    if url.startswith('data:'):
                        header, encoded = url.split(',', 1)
                        require(header.startswith('data:image/') and header.endswith(';base64'),
                                'Unsupported inline image encoding')
                        data = base64.b64decode(encoded, validate=True)
                        image.update(bytes=len(data), sha256=digest(data), media_type=header[5:-7])
                    else:
                        image['byte_count_limit'] = 'External image not fetched; byte count unavailable.'
                    images.append(image)
                else:
                    raise ValueError('Uncounted content type: ' + str(block['type']))
        require(len(images) == prepared['png_count'], 'Prepared/request image count differs')
        total = sum(part['tokens'] for part in parts)
        reserved_total = total + reserve
        records.append({'prepared': prepared_ref, 'request': request_ref,
            'packet_id': prepared['packet_id'], 'judge_id': prepared['judge_id'], 'stage': prepared['stage'],
            'prepared_status': prepared['status'], 'body_model': body.get('model'),
            'body_max_tokens': body.get('max_tokens'),
            'body_output_cap_matches_card_reservation': body.get('max_tokens') == rules['output_reserved'],
            'text_blocks': parts, 'text_block_count': len(parts),
            'serialized_text_tokens_deepseek_v4': total,
            'text_utf8_bytes': sum(part['text_utf8_bytes'] for part in parts),
            'images': images, 'image_count': len(images),
            'known_image_bytes': sum(image['bytes'] for image in images if image['bytes'] is not None),
            'images_with_unknown_bytes': sum(image['bytes'] is None for image in images),
            'image_tokens': None, 'image_tokens_counted': False,
            'conservative_reserved_total_using_text_proxy_only': reserved_total,
            'text_only_proxy_comparisons': {
                'within_input_token_proxy_limit': total <= rules['input_token_proxy_limit'],
                'reserved_total_within_context_window': reserved_total <= rules['context_window'],
                'reserved_total_below_auto_compact_limit': reserved_total < rules['auto_compact_limit']},
            'measurement_scope': 'TEXT_ONLY_IMAGES_UNCOUNTED' if images else 'SUPPLIED_TEXT_ONLY',
            'provider_usage': None, 'actual_agent_peak_tokens': None, 'capacity_release': False})
    result = {'schema_version': 'p3new.n5_prepared_static_text_tokens.v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'STATIC_TEXT_COUNT_ONLY_NOT_RELEASED',
        'prepared_root': str(args.prepared_root.resolve()), 'capture_card': card_ref, 'reservation_rules': rules,
        'cached_tokenizer_input': zip_ref, 'tokenizer_zip_member': member, 'tokenizer_json_sha256': TOKENIZER_SHA,
        'runtime': {'tokenizers_version': tokenizers.__version__, 'module': read(Path(tokenizers.__file__))[1]},
        'execution_script': read(Path(__file__))[1],
        'method': 'Sum complete system and every supplied text block using frozen DeepSeek V4 tokenizer, add_special_tokens=False; excludes HTTP JSON escaping and chat framing. SOL counts are a proxy, not actual SOL tokens.',
        'unmeasured': ['provider chat-template and special-token overhead', 'actual SOL tokenizer',
            'image tokens', 'agent/tool/runtime overhead and actual peak context', 'actual provider usage',
            'semantic completeness, necessary facts, or confirmation admission'],
        'output_reservation_limit': 'SOL output reservation is not a CLI-enforced output cap.',
        'multimodal_limit': 'Text-only comparisons never release a multimodal request; all image tokens are uncounted.',
        'model_calls': 0, 'gpu_runs': 0, 'network_requests': 0, 'tokenizer_copies_created': 0,
        'input_mutations': 0, 'facts_admitted': False, 'semantic_review_performed': False, 'capacity_release': False,
        'requests_counted': len(records), 'text_block_count': sum(row['text_block_count'] for row in records),
        'image_count': sum(row['image_count'] for row in records),
        'max_text_tokens': max(row['serialized_text_tokens_deepseek_v4'] for row in records),
        'max_reserved_text_proxy_total': max(row['conservative_reserved_total_using_text_proxy_only'] for row in records),
        'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'requests_counted': len(records),
                      'max_text_tokens': result['max_text_tokens'], 'output': read(args.output)[1]}))


if __name__ == '__main__':
    main()
