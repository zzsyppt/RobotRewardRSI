"""Fixed R4 CPU export/local carrier preparation; never invoke a model or seal facts."""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
C, CAP, S = B / 'n5_execution_20261001_r4', B / 'n5_confirmation_20261001_r4', B / 'n5_confirmation_source_r4_final'
E = Path(__file__).resolve().parent
LOCAL_SOURCE = E.parent / 'n5_confirmation_source_r4/source'
CLI = Path('C:/Users/lover/AppData/Local/OpenAI/Codex/bin/c6fe824d725f02d7/codex.exe')
CARD_SHA = '811d7f140a5b3015f66a5093ba516c60bdc4a05c9ac827461d19bb05b8be3041'
SOURCE_SHA = '1b7cb40e5b1d1780011e1748a9e6adae88517dd880b74c8b0b5f23df66a1a645'
ASSIGNMENT_SHA = '4b3a3aceb695f52ccd081547693bdd004e4002297bf8f9c6e6f10c563a75e12d'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(path.read_bytes())
def ref(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': sha(raw), 'size_bytes': len(raw)}
def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def modules(source):
    p = source / 'PLAN/feedback_sys/P3NEW'
    sys.path[:0] = [str(p), str(source / 'src')]
    review = importlib.import_module('run_confirmation_review')
    cli = importlib.import_module('run_sol_review')
    require(Path(review.__file__).resolve() == (p / 'run_confirmation_review.py').resolve(), 'Wrong frozen runner')
    require(Path(cli.__file__).resolve() == (p / 'run_sol_review.py').resolve(), 'Wrong frozen CLI carrier')
    return p, review, cli


def verify_source(source, manifest_path):
    require(ref(manifest_path)['sha256'] == SOURCE_SHA, 'Final R4 manifest differs')
    manifest = read(manifest_path)
    require(len(manifest) == 492, 'Final source count differs')
    for relative, expected in manifest.items():
        actual = ref(source / relative)
        require(actual['sha256'] == expected['sha256'] and actual['size_bytes'] == expected['size_bytes'], 'Source differs: ' + relative)
    return manifest


def export():
    out = C / 'facts_offline_preparation'
    require(not out.exists(), 'Preserve prior export attempt')
    manifest = verify_source(S, C / 'runtime_source_manifest.json')
    require(ref(C / 'capture_card.json')['sha256'] == CARD_SHA, 'R4 card differs')
    p, review, cli = modules(S)
    assignment_path, policy_path = p / 'N5_REVIEW_ASSIGNMENT_R4.json', p / 'N5_SOL_CLI_POLICY_R4.json'
    require(ref(assignment_path)['sha256'] == ASSIGNMENT_SHA, 'R4 assignment differs')
    assignment, policy = read(assignment_path)['packets'], read(policy_path)
    numeric = read(CAP / 'numeric_materials/packing_receipt.json')
    media = read(CAP / 'media_materials/confirmation_media_receipt.json')
    units = [(r['packet_id'], r['task'], CAP / 'numeric_materials' / r['facts_manifest'],
              CAP / 'numeric_materials/prepared' / r['packet_id']) for r in numeric['packets']]
    units += [(r['unit_id'], r['task'], Path(r['facts_manifest']['file']),
               CAP / 'media_materials' / r['unit_id'] / 'prepared_facts') for r in media['units']]
    require(len(units) == 40 and {r[0] for r in units} == set(assignment), 'Actual 40-unit mapping differs')
    ds = {packet for packet, slots in assignment.items() if slots['B']['judge_id'] == 'deepseek'}
    first = {min(packet for packet, t, _, _ in units if t == task and packet.startswith(axis + '-'))
             for task in ('stair', 'push') for axis in ('episode', 'focus')}
    require(ds == first | {'media-01', 'media-02', 'media-03', 'media-04'} and len(ds) == 8,
            'Actual per-task first episode/focus routing differs')
    # Same cached tokenizer bytes and encode convention as count_prepared_text.py.
    import tokenizers
    from tokenizers import Tokenizer
    tokenizer_path = B / 'n5_execution_20260930/control/numeric_materials_r1/token_capacity_r1/deepseek_v4_tokenizer.zip'
    blob = tokenizer_path.read_bytes()
    require(sha(blob) == 'e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495', 'Tokenizer ZIP differs')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        candidates = [(n, z.read(n)) for n in z.namelist() if n.endswith('tokenizer.json')]
    matched = [(n, raw) for n, raw in candidates if sha(raw) == '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614']
    require(len(matched) == 1, 'Tokenizer member differs')
    tokenizer = Tokenizer.from_str(matched[0][1].decode('utf-8'))
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    records, members = [], []
    out.mkdir()
    with zipfile.ZipFile(out / 'sol_original_prepared.zip', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for packet, task, facts, prepared_root in sorted(units):
            facts_data = read(facts)
            require(facts_data['stage'] == 'facts' and facts_data['purpose'] == 'confirmation', 'Not actual facts')
            require(review.review_judges(facts_data) == tuple(assignment[packet][slot]['judge_id'] for slot in ('A', 'B')), 'Assignment row differs')
            for slot in ('A', 'B'):
                reviewer = assignment[packet][slot]
                judge = reviewer['judge_id']
                prepared = prepared_root / judge
                card, raw = read(prepared / 'prepare.json'), (prepared / 'request.http_body.json').read_bytes()
                require(card['stage'] == 'facts' and card['judge_id'] == judge and card['packet_id'] == packet and
                        card['status'] == 'PREPARED_NOT_RELEASED' and card['capacity_facts_fixture'] is None and
                        card['own_facts_seal'] is None and card['max_output_tokens'] == 32768 and
                        card['reviewer'] == reviewer and card['manifest']['sha256'] == ref(facts)['sha256'], 'Original facts identity differs')
                require(sha(raw) == card['request_sha256'] and len(raw) == card['request_bytes'] <= review.MAX_BODY, 'Request bytes differ')
                for key in ('runner', 'transport', 'coverage_helper'):
                    source_path = Path(card[key]['file'])
                    require(card[key]['sha256'] == manifest[source_path.relative_to(S).as_posix()]['sha256'], 'Prepared source differs')
                body = json.loads(raw)
                require(review.encoded(body) == raw and body['max_tokens'] == 32768, 'Request serialization/output differs')
                for item in card['materials']:
                    require(sha(Path(item['file']).read_bytes()) == item['sha256'], 'Actual material changed')
                adapted, images, texts = cli.adapter.adapt_messages(body['messages'])
                require(len(images) == card['png_count'] <= review.MAX_IMAGES, 'Original image count differs')
                by_id = {item['id']: item for item in card['materials']}
                for image, image_raw in images:
                    require(image['sha256'] == by_id[image['material_id']]['sha256'] and
                            len(image_raw) == by_id[image['material_id']]['size_bytes'], 'Image order/hash differs')
                text_tokens = sum(count(block['text']) for _, _, _, block in cli.adapter.blocks(body['messages']) if block['type'] == 'text')
                record = {'packet_id': packet, 'task': task, 'slot': slot, 'judge_id': judge, 'reviewer': reviewer,
                          'remote_manifest': ref(facts), 'remote_prepare': ref(prepared / 'prepare.json'),
                          'remote_request': ref(prepared / 'request.http_body.json'), 'png_count': len(images),
                          'images': [x for x, _ in images], 'original_text_blocks_deepseek_tokens': text_tokens,
                          'output_reservation': 32768, 'image_tokens': None, 'actual_SOL_tokens': None,
                          'actual_usage': None, 'capacity_release': False}
                if judge in review.SOL_PAIR:
                    stdin = cli.WRAPPER + cli.adapter.encoded(adapted) + b'\nEND ORIGINAL CARRIER JSON\n'
                    proxy = count(stdin.decode('utf-8'))
                    reserved = proxy + policy['fixed_overhead_reservation'] + policy['proxy_difference_reservation'] + 32768
                    record.update(full_cli_stdin_sha256=sha(stdin), full_cli_stdin_bytes=len(stdin),
                                  full_cli_stdin_deepseek_token_proxy=proxy, reserved_text_proxy=reserved,
                                  text_proxy_within_input_limit=proxy <= policy['input_proxy_limit'],
                                  reserved_text_proxy_below_autocompact=reserved < policy['model_auto_compact_token_limit'],
                                  reserved_text_proxy_within_context=reserved <= policy['model_context_window'])
                    for name in ('prepare.json', 'request.http_body.json', 'reference_catalog.json'):
                        path = prepared / name
                        arc = f'prepared/{packet}/{judge}/{name}'
                        z.write(path, arc)
                        members.append({'member': arc, **ref(path)})
                records.append(record)
                print(json.dumps({'prepared_checked': len(records), 'packet': packet, 'judge': judge}), flush=True)
    require(len(records) == 80 and len(members) == 216, '80 identities / 72 SOL contexts required')
    verify_source(S, C / 'runtime_source_manifest.json')
    result = {'schema_version': 'p3new.r4_facts_offline_export.v1', 'status': 'EXPORTED_NOT_RELEASED',
              'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'n5_confirmation',
              'script': ref(Path(__file__)), 'capture_card': ref(C / 'capture_card.json'),
              'source_manifest': ref(C / 'runtime_source_manifest.json'), 'assignment': ref(assignment_path),
              'policy': ref(policy_path), 'policy_values': policy, 'tokenizer_zip': ref(tokenizer_path),
              'tokenizer_member': matched[0][0], 'tokenizer_json_sha256': sha(matched[0][1]),
              'tokenizers_version': tokenizers.__version__, 'actual_deepseek_units': sorted(ds),
              'archive': ref(out / 'sol_original_prepared.zip'), 'members': members, 'records': records,
              'actual_calls': 0, 'reports_prepared': 0, 'facts_sealed': 0, 'source_unchanged': True,
              'capacity_release': False, 'limits': ['Full CLI stdin count uses frozen DeepSeek tokenizer as SOL proxy, not actual SOL tokens.',
                  'Image tokens, provider framing and runtime peak remain unmeasured; no multimodal release.',
                  '32768 is SOL output reservation, not a CLI-enforced cap. No semantic judgment.']}
    save(out / 'export_receipt.json', result)
    print(json.dumps({'status': result['status'], 'archive': result['archive'], 'receipt': ref(out / 'export_receipt.json')}), flush=True)


def prepare_local():
    root = E / 'facts_offline_preparation'
    export_path = root / 'export_receipt.json'
    exported = read(export_path)
    archive = root / 'sol_original_prepared.zip'
    require(ref(archive)['sha256'] == exported['archive']['sha256'], 'Transfer archive differs')
    require(not (root / 'prepared').exists() and not (root / 'transport').exists(), 'Preserve prior local attempt')
    verify_source(LOCAL_SOURCE, E / 'runtime_source_manifest.json')
    p, review, cli = modules(LOCAL_SOURCE)
    require(ref(CLI)['sha256'] == 'cbafb6422bca005b94c12d105b1a16a0474219e24ea4893f85846409c464f5a1', 'Verified CLI differs')
    require(ref(E / 'capture_card.json')['sha256'] == exported['capture_card']['sha256'] == CARD_SHA and
            ref(p / 'N5_REVIEW_ASSIGNMENT_R4.json')['sha256'] == exported['assignment']['sha256'] == ASSIGNMENT_SHA,
            'Local card/assignment differs')
    members = {r['member']: r for r in exported['members']}
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(members) == 216 and set(z.namelist()) == set(members), 'Archive members differ')
        for name, expected in members.items():
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'prepared', 'Unsafe archive path')
            raw = z.read(name)
            require(sha(raw) == expected['sha256'] and len(raw) == expected['size_bytes'], 'Archive bytes differ')
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f: f.write(raw)
    results, launches = [], []
    for row in exported['records']:
        if row['judge_id'] not in review.SOL_PAIR:
            continue
        packet, judge = row['packet_id'], row['judge_id']
        prepared = root / 'prepared' / packet / judge
        output = root / 'transport' / packet / judge
        cli.run(prepared, output, source_root=LOCAL_SOURCE, cli_policy=p / 'N5_SOL_CLI_POLICY_R4.json',
                assignment_file=p / 'N5_REVIEW_ASSIGNMENT_R4.json', execute=False, cli=CLI)
        receipt = read(output / 'transport_receipt.json')
        require(ref(prepared / 'request.http_body.json')['sha256'] == row['remote_request']['sha256'] and
                ref(prepared / 'prepare.json')['sha256'] == row['remote_prepare']['sha256'] and
                ref(output / 'stdin.txt')['sha256'] == row['full_cli_stdin_sha256'] and
                receipt['messages_reconstructed_exactly'] and receipt['request_reconstructed_exactly'] and
                receipt['images'] == row['images'], 'Original request/stdin/image parity failed')
        results.append({'packet_id': packet, 'judge_id': judge, 'transport_receipt': ref(output / 'transport_receipt.json'),
                        'stdin': ref(output / 'stdin.txt'), 'original_request_sha256': row['remote_request']['sha256'],
                        'original_prepare_sha256': row['remote_prepare']['sha256'], 'exact_parity': True})
        call_output = E / 'formal_facts_calls' / packet / judge
        runtime = E / 'private_cli_runtime' / packet / judge
        require(not call_output.exists() and not runtime.exists(), 'Planned launch context is not fresh')
        argv = [sys.executable, '-B', str(p / 'run_sol_review.py'), '--prepared', str(prepared), '--output', str(call_output),
                '--source-root', str(LOCAL_SOURCE), '--cli-policy', str(p / 'N5_SOL_CLI_POLICY_R4.json'),
                '--review-assignment', str(p / 'N5_REVIEW_ASSIGNMENT_R4.json'), '--cli', str(CLI),
                '--auth-source-home', 'C:/Users/lover/.codex', '--runtime-home', str(runtime), '--execute-one-call']
        launches.append({'packet_id': packet, 'judge_id': judge, 'status': 'NOT_RELEASED_NOT_EXECUTED', 'argv': argv})
    require(len(results) == 72 and len({(r['packet_id'], r['judge_id']) for r in results}) == 72, 'Independent SOL outputs collided')
    verify_source(LOCAL_SOURCE, E / 'runtime_source_manifest.json')
    receipt = {'status': 'PREPARED_IDENTICAL_TRANSPORT_NOT_RELEASED', 'scope': 'n5_confirmation',
               'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'export': ref(export_path), 'archive': ref(archive),
               'script': ref(Path(__file__)), 'contexts': results, 'source_unchanged': True, 'actual_calls': 0,
               'facts_sealed': 0, 'reports_prepared': 0, 'capacity_release': False,
               'limits': 'Original remote metadata paths remain unchanged. Local adapter verifies identity/source/assignment and reconstructs original request. No raw-source path rewrite, actual image consumption or semantic pass claimed.'}
    save(root / 'local_preparation_receipt.json', receipt)
    save(root / 'pending_sol_launch_commands.json', {'status': 'NOT_RELEASED_NOT_EXECUTED', 'calls': launches,
                                                    'barrier': 'All 80 actual facts answers must be sealed before any report.'})
    print(json.dumps({'status': receipt['status'], 'contexts': len(results), 'receipt': ref(root / 'local_preparation_receipt.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'prepare-local'))
    args = parser.parse_args()
    export() if args.mode == 'export' else prepare_local()
