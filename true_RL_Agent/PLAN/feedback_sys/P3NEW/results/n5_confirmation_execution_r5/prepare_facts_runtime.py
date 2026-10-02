"""R5 facts-only mirror and CLI registration. No model call, home, seal or report."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.dont_write_bytecode = True
SOURCE_SHA = 'ca9e11f079200d26545f75a2550bcf0901afd59c02adee3d418695cb4ccfa2bd'
ASSIGNMENT_SHA = '7ce626e648a1b85b22099444ba27b49ce120f4663b08e96afd1e981f999b2c7c'
POLICY_SHA = '07993fda0d35755d8186f51a1c8ae319c9e649544497f8a6e8720d8b05fe1bc0'
DS_SLOTS = ('episode-01', 'episode-13', 'focus-01', 'focus-07', 'media-01', 'media-02', 'media-03', 'media-04')
C = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20261001_r5'
CAP = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r5'
NAVIGATION_SCRIPT_SHA = 'e90eb0caa603ef86f7d3db3f2fd6eeee81be44b6549adbed3a9be604b1f27284'


def now(): return datetime.now(timezone.utc).isoformat()
def require(ok, message):
    if not ok: raise ValueError(message)
def read(path): return json.loads(Path(path).read_bytes())
def sha(raw): return hashlib.sha256(raw).hexdigest()
def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': sha(raw), 'size_bytes': len(raw)}
def check(item, path=None):
    actual = ref(path or item.get('path') or item['file'])
    require(actual['sha256'] == item['sha256'], 'Bound SHA differs: ' + actual['path'])
    require('size_bytes' not in item or actual['size_bytes'] == item['size_bytes'], 'Bound size differs')
    return actual
def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def modules(source, manifest):
    source = Path(source).resolve()
    require(check(manifest)['sha256'] == SOURCE_SHA, 'Only the frozen R5 source is accepted')
    inventory = read(manifest.get('path', manifest.get('file')))
    require(len(inventory) == 500, 'R5 source roster differs')
    for relative, expected in inventory.items(): check(expected, source / relative)
    p = source / 'PLAN/feedback_sys/P3NEW'
    sys.path[:0] = [str(p), str(source / 'src')]
    review = importlib.import_module('run_confirmation_review')
    cli = importlib.import_module('run_sol_review')
    audit = importlib.import_module('audit_confirmation_facts')
    require(Path(review.__file__).resolve() == p / 'run_confirmation_review.py' and
            Path(cli.__file__).resolve() == p / 'run_sol_review.py' and
            Path(audit.__file__).resolve() == p / 'audit_confirmation_facts.py', 'Wrong imported source')
    require(ref(p / 'N5_REVIEW_ASSIGNMENT_R5.json')['sha256'] == ASSIGNMENT_SHA and
            ref(p / 'N5_SOL_CLI_POLICY_R5.json')['sha256'] == POLICY_SHA, 'R5 assignment/policy differs')
    return p, review, cli, audit


def gates(inputs):
    """Bind existing receipts; no new semantic/approval platform or report input."""
    for key in ('capture_card', 'source_manifest', 'preparation', 'assembly_execution'): check(inputs[key])
    require(inputs['source_manifest']['sha256'] == SOURCE_SHA, 'Wrong source identity')
    card = read(inputs['capture_card']['path'])
    require(card['status'] == 'FROZEN' and card['tasks']['stair']['cohort']['diagnostic_cohort'] == 'n5_r5_20261001' and
            card['execution_control_root'] == C and card['review_assignment']['sha256'] == ASSIGNMENT_SHA,
            'Not the R5 capture/assignment')
    delivery = read(inputs['assembly_execution']['path'])
    require(delivery['execution_status'] == 'COMPLETED' and delivery['status'] == 'offline_serialized_not_sent'
            and delivery['returncode'] == 0 and delivery['capture_card']['sha256'] == inputs['capture_card']['sha256']
            and delivery['source_manifest']['sha256'] == SOURCE_SHA, 'Offline delivery is not complete/bound')
    bound = {r['sha256'] for r in delivery['actual_module_and_trace_inputs']}
    require(set(inputs['module_seals']) == {'stair', 'push'}, 'Both module receipts required')
    for item in inputs['module_seals'].values():
        require(check(item)['sha256'] in bound, 'Module receipt not used by actual delivery')
    require(set(inputs['cpu_receipts']) == {'numeric', 'media', 'validate', 'audit'}, 'Four CPU receipts required')
    for item in inputs['cpu_receipts'].values():
        receipt = read(check(item)['path'])
        require(receipt['status'] == 'COMPLETED_CPU_ONLY_NOT_RELEASED' and
                receipt['capture_card']['sha256'] == inputs['capture_card']['sha256'] and
                receipt['source_manifest']['sha256'] == SOURCE_SHA, 'CPU status/binding differs')
    navigation = read(check(inputs['navigation_audit'])['path'])
    require(navigation['status'] == 'PASS_INDEPENDENT_NAVIGATION_ARITHMETIC_NOT_SEMANTIC' and
            navigation['capture_card_sha256'] == inputs['capture_card']['sha256'] and
            check(navigation['script'])['sha256'] == NAVIGATION_SCRIPT_SHA, 'Actual R5 navigation supplement required')
    nav_inputs = {r.get('path', r.get('file')): r for r in navigation['inputs']}
    require(nav_inputs[C + '/postcapture/audit/receipt.json']['sha256'] == inputs['cpu_receipts']['audit']['sha256'], 'Navigation used a different original audit')
    for path in (CAP + '/numeric_raw/manifest.json', CAP + '/numeric_materials/packing_receipt.json'):
        check(nav_inputs[path])
    for item in navigation['inputs']: check(item)
    nav_rows = navigation['rows']
    require(len(nav_rows) == 6 and len({r['packet_id'] for r in nav_rows}) == 6 and
            all(r['packet_id'].startswith('focus-') and r['all_expected_groups_components_counts_equal'] is True for r in nav_rows), 'All six actual Stair navigation views required')
    prepared = read(inputs['preparation']['path'])
    require(prepared['status'] == 'PREPARED_NOT_RELEASED' and prepared['api_calls'] == prepared['facts_sealed'] == 0
            and prepared['assignment']['sha256'] == ASSIGNMENT_SHA, 'Original R5 facts preparation required')
    return card, prepared


def export(inputs_path, inputs_sha, output):
    require(ref(inputs_path)['sha256'] == inputs_sha, 'Explicit inputs SHA differs')
    inputs = read(inputs_path)
    _, prepared = gates(inputs)
    p, review, cli, _ = modules(inputs['source_root'], inputs['source_manifest'])
    rows = prepared['rows']
    expected = {(packet, item['judge_id']) for packet, slots in read(p / 'N5_REVIEW_ASSIGNMENT_R5.json')['packets'].items()
                for item in slots.values()}
    require(len(rows) == 80 and {(r['packet_id'], r['judge_id']) for r in rows} == expected, 'Original 40/80 roster differs')
    require(tuple(sorted(r['packet_id'] for r in rows if r['judge_id'] == 'deepseek')) == tuple(sorted(DS_SLOTS)), 'DS8 routing differs')
    output = Path(output).resolve()
    require(not output.exists(), 'New export only; preserve every previous attempt')
    output.mkdir(parents=True)
    records, members = [], []
    with zipfile.ZipFile(output / 'sol_original_prepared.zip', 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for row in sorted(rows, key=lambda r: (r['packet_id'], r['judge_id'])):
            original = check(row['prepared'])
            directory = Path(original['path']).parent
            card = read(directory / 'prepare.json')
            request = ref(directory / 'request.http_body.json')
            cap = 65536 if row['judge_id'] == 'deepseek' else 32768
            require(card['stage'] == 'facts' and card['status'] == 'PREPARED_NOT_RELEASED' and
                    card['packet_id'] == row['packet_id'] and card['judge_id'] == row['judge_id'] and
                    card['max_output_tokens'] == row['max_output_tokens'] == cap and
                    card['own_facts_seal'] is None and card['capacity_facts_fixture'] is None and not card.get('synthetic_only') and
                    request['sha256'] == card['request_sha256'] and request['size_bytes'] == card['request_bytes'], 'Original facts identity differs')
            check(row['facts_manifest'])
            for item in card['materials']: check(item)
            require(card['runner']['sha256'] == ref(p / 'run_confirmation_review.py')['sha256'] and
                    card['coverage_helper']['sha256'] == ref(p / 'confirmation_review_coverage.py')['sha256'] and
                    card['review_assignment'] == review.assignment_fields(p / 'N5_REVIEW_ASSIGNMENT_R5.json', row['packet_id'])['review_assignment'], 'Prepared source/assignment differs')
            record = dict(packet_id=row['packet_id'], judge_id=row['judge_id'], slot=row['slot'],
                          remote_prepare=original, remote_request=request, remote_manifest=check(row['facts_manifest']),
                          output_reservation=cap, png_count=card['png_count'])
            if row['judge_id'] in review.SOL_PAIR:
                carrier, images, _ = cli.adapter.adapt_messages(read(directory / 'request.http_body.json')['messages'])
                stdin = cli.WRAPPER + cli.adapter.encoded(carrier) + b'\nEND ORIGINAL CARRIER JSON\n'
                record.update(full_cli_stdin_sha256=sha(stdin), full_cli_stdin_bytes=len(stdin), images=[r for r, _ in images])
                for name in ('prepare.json', 'request.http_body.json', 'reference_catalog.json'):
                    path = directory / name
                    member = f"prepared/{row['packet_id']}/{row['judge_id']}/{name}"
                    archive.write(path, member)
                    members.append({'member': member, **ref(path)})
            records.append(record)
    receipt = dict(status='EXPORTED_NOT_RELEASED', scope='n5_r5_facts_only', recorded_at_utc=now(),
                   inputs=ref(inputs_path), source_manifest=inputs['source_manifest'], capture_card=inputs['capture_card'],
                   preparation=inputs['preparation'], records=records, members=members,
                   archive=ref(output / 'sol_original_prepared.zip'), script=ref(__file__),
                   dispatch_script=ref(Path(__file__).with_name('dispatch_facts.py')),
                   actual_calls=0, facts_sealed=0, reports_prepared=0, capacity_release=False)
    save(output / 'export_receipt.json', receipt)
    return receipt


def local_prepare(args):
    exported = read(args.export)
    require(ref(args.export)['sha256'] == args.export_sha256 and exported['scope'] == 'n5_r5_facts_only' and
            exported['source_manifest']['sha256'] == SOURCE_SHA, 'Original R5 export SHA/identity required')
    check(exported['archive'], args.archive)
    p, review, cli, _ = modules(args.source_root, ref(args.source_manifest))
    output = args.output.resolve()
    require(not output.exists(), 'New local preparation only')
    require(args.runtime_root.resolve() != output and not args.runtime_root.resolve().is_relative_to(output), 'Private runtime must be outside exported preparation')
    require(args.cli.is_file() and (args.auth_source_home / 'auth.json').is_file(), 'Explicit CLI/auth source must exist; credential contents are not read')
    output.mkdir(parents=True)
    expected = {r['member']: r for r in exported['members']}
    require(len(expected) == 216, 'Exactly 72 contexts with three original files required')
    with zipfile.ZipFile(args.archive) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected), 'Transferred archive roster differs')
        for name, item in expected.items():
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'prepared', 'Unsafe archive member')
            raw = archive.read(name)
            require(sha(raw) == item['sha256'] and len(raw) == item['size_bytes'], 'Transferred original bytes differ')
            target = output.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write(raw)
    calls = []
    for row in exported['records']:
        if row['judge_id'] not in review.SOL_PAIR: continue
        packet, judge = row['packet_id'], row['judge_id']
        prepared, preview = output / 'prepared' / packet / judge, output / 'transport' / packet / judge
        cli.run(prepared, preview, source_root=args.source_root, cli_policy=p / 'N5_SOL_CLI_POLICY_R5.json',
                assignment_file=p / 'N5_REVIEW_ASSIGNMENT_R5.json', execute=False, cli=args.cli)
        transport = read(preview / 'transport_receipt.json')
        require(ref(preview / 'stdin.txt')['sha256'] == row['full_cli_stdin_sha256'] and
                transport['request_reconstructed_exactly'] and transport['messages_reconstructed_exactly'] and
                transport['images'] == row['images'], 'Carrier/request/original PNG parity differs')
        actual_output, runtime = args.calls_root.resolve() / packet / judge, args.runtime_root.resolve() / packet / judge
        require(not actual_output.exists() and not runtime.exists(), 'Fresh single-call output/home required')
        argv = [sys.executable, '-B', str(p / 'run_sol_review.py'), '--prepared', str(prepared), '--output', str(actual_output),
                '--source-root', str(args.source_root.resolve()), '--cli-policy', str(p / 'N5_SOL_CLI_POLICY_R5.json'),
                '--review-assignment', str(p / 'N5_REVIEW_ASSIGNMENT_R5.json'), '--cli', str(args.cli.resolve()),
                '--auth-source-home', str(args.auth_source_home.resolve()), '--runtime-home', str(runtime), '--execute-one-call']
        calls.append({**row, 'argv': argv, 'preview_stdin': ref(preview / 'stdin.txt'),
                      'preview_transport': ref(preview / 'transport_receipt.json'),
                      'local_prepare': ref(prepared / 'prepare.json'), 'local_request': ref(prepared / 'request.http_body.json')})
    require(len(calls) == 72, 'Fixed 72 SOL slots required')
    registration = dict(status='PREPARED_NOT_RELEASED', scope='n5_r5_facts_only', created_utc=now(),
                        export=ref(args.export), source_root=str(args.source_root.resolve()), source_manifest=ref(args.source_manifest),
                        cli=ref(args.cli), calls=calls, remote_control=C, script=ref(__file__),
                        dispatch_script=ref(Path(__file__).with_name('dispatch_facts.py')),
                        max_parallel=2, application_retries=0, facts_sealed=0, reports_prepared=0,
                        capacity_release=False, fees_usd=None,
                        limits='No real capacity gate yet. DS tokenizer counts are a SOL proxy. Image/runtime peak tokens and fees remain unknown; output reservation is not a hard CLI cap.')
    save(output / 'pending_sol_launch_commands.json', registration)
    return registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    remote = sub.add_parser('export')
    remote.add_argument('--inputs', type=Path, required=True)
    remote.add_argument('--inputs-sha256', required=True)
    remote.add_argument('--output', type=Path, required=True)
    local = sub.add_parser('prepare-local')
    for name in ('export', 'archive', 'source-root', 'source-manifest', 'output', 'calls-root', 'runtime-root', 'cli', 'auth-source-home'):
        local.add_argument('--' + name, type=Path, required=True)
    local.add_argument('--export-sha256', required=True)
    args = parser.parse_args()
    result = export(args.inputs, args.inputs_sha256, args.output) if args.mode == 'export' else local_prepare(args)
    print(json.dumps({'status': result['status'], 'model_calls': 0, 'reports_prepared': 0}))


if __name__ == '__main__': main()
