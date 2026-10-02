"""R5 CPU control only. Explicit command; no capture, model call or report reveal."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
S = B / 'n5_confirmation_source_r5_final'
C = B / 'n5_execution_20261001_r5'
CAP = B / 'n5_confirmation_20261001_r5'
P = S / 'PLAN/feedback_sys/P3NEW'
SOURCE_SHA = 'ca9e11f079200d26545f75a2550bcf0901afd59c02adee3d418695cb4ccfa2bd'
CARD = C / 'capture_card.json'
MANIFEST = C / 'runtime_source_manifest.json'
ASSIGNMENT = P / 'N5_REVIEW_ASSIGNMENT_R5.json'
PROTOCOL = P / 'N5_FACT_REVIEW_PROTOCOL_ZH.md'
RAW, PACK, MEDIA = (CAP / name for name in ('numeric_raw', 'numeric_materials', 'media_materials'))
DEFS = C / 'numeric_task_definitions'

def now(): return datetime.now(timezone.utc).isoformat()
def require(ok, message):
    if not ok: raise ValueError(message)
def read(path): return json.loads(Path(path).read_bytes())
def ref(path):
    path = Path(path).resolve()
    with path.open('rb') as stream: sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'file': str(path), 'sha256': sha, 'size_bytes': path.stat().st_size}
def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')

def check_freeze(card_sha):
    require(ref(MANIFEST)['sha256'] == SOURCE_SHA, 'Wrong R5 source manifest')
    require(ref(CARD)['sha256'] == card_sha, 'Actual frozen capture-card SHA differs')
    card, manifest = read(CARD), read(MANIFEST)
    require(card['status'] == 'FROZEN' and card['output_root'] == str(CAP)
            and card['source']['remote_root'] == str(S)
            and card['source']['manifest']['sha256'] == SOURCE_SHA, 'Wrong R5 roots/card')
    for relative, expected in manifest.items():
        actual = ref(S / relative)
        require(actual['sha256'] == expected['sha256'] and actual['size_bytes'] == expected['size_bytes'],
                'Frozen source differs: ' + relative)
    require(card['review_protocol']['sha256'] == ref(PROTOCOL)['sha256'], 'Protocol/card mismatch')
    require(read(ASSIGNMENT)['status'] == 'FROZEN', 'Reviewer assignment is not frozen')
    require(card['review_assignment']['sha256'] == ref(ASSIGNMENT)['sha256']
            and card['review_assignment']['remote_path'] == str(ASSIGNMENT), 'Assignment/card mismatch')
    return card

def completed(task, card):
    require(read(C / (task + '.exit.json'))['returncode'] == 0, 'Capture exit failed: ' + task)
    require(read(CAP / task / 'result.json')['status'] == 'PASS', 'Capture wrapper failed: ' + task)
    phase = read(CAP / task / 'public_dev_result.json')
    require(phase['evaluation_binding_hash'] == card['tasks'][task]['current_evaluation_binding']['evaluation_binding_hash'],
            'Capture identity is not the actual R5 card: ' + task)

def commands_for(args, card, out):
    commands = []
    if args.mode == 'task':
        t = args.task
        completed(t, card)
        for path in (CAP/'exports'/t, CAP/'observer_selection'/t, CAP/'observers'/t):
            require(not path.exists(), 'Preserve previous output: ' + str(path))
        commands += [('export', [P/'extract_dual_view_evidence.py', '--summary', CAP/t/'evaluation/public_dev/summary.json',
            '--recordings', CAP/t/'video/recordings.json', '--resolved-cell-manifest', CAP/t/'composition/resolved_cell_manifest.json',
            '--stdout-log', C/(t+'.stdout.log'), '--output', CAP/'exports'/t, '--png-storage', 'hardlink']),
            ('selection', [P/'prepare_n5_media_selection.py', '--export', CAP/'exports'/t,
            '--full', CAP/t/'evaluation/public_dev/behavior_full.json', '--capture-card', CARD, '--output', CAP/'observer_selection'/t]),
            ('observer_prepare', [P/'prepare_native_observer.py', '--export', CAP/'exports'/t,
            '--selection', CAP/'observer_selection'/t, '--capture-card', CARD, '--output', CAP/'observers'/t])]
    else:
        for t in ('stair', 'push'): completed(t, card)
    if args.mode == 'numeric':
        require(not any(path.exists() for path in (RAW, PACK, DEFS)), 'Numeric destinations must be new')
        stair = CAP/'stair/numeric_evidence/manifest.json'
        push = [CAP/f'push/numeric/public_dev-{n:02d}/manifest.json' for n in (1,2,3)]
        cell = CAP/'push/composition/resolved_cell_manifest.json'
        commands.append(('raw_export', [P/'export_confirmation_raw.py', '--stair', stair, '--push', *push,
            '--output', RAW, '--purpose', 'confirmation', '--capture-card', CARD, '--review-protocol', PROTOCOL,
            '--profile', 'related_v1', '--part-steps', '64']))
        definitions = [('stair', stair, DEFS/'stair.json')]
        definitions += [('push', path, DEFS/f'push_run{n:02d}.json') for n,path in enumerate(push,1)]
        for t, numeric, target in definitions:
            argv = [P/'prepare_review_task_contract.py', '--task', t, '--numeric-manifest', numeric,
                    '--source-manifest', RAW/'_private/sources.json', '--output', target]
            if t == 'push': argv += ['--resolved-cell-manifest', cell]
            commands.append(('definition_'+target.stem, argv))
        argv = [P/'prepare_confirmation_materials.py', '--raw-export', RAW, '--output', PACK,
                '--protocol', PROTOCOL, '--capture-card', CARD, '--profile', 'related_v2',
                '--max-output-tokens', '32768', '--review-assignment', ASSIGNMENT,
                '--push-resolved-cell-manifest', cell]
        for _,_,target in definitions: argv += ['--task-definition', target]
        commands.append(('numeric_pack', argv))
    if args.mode == 'media':
        require(not MEDIA.exists() and not (C/'media_inputs.json').exists(), 'Media destinations must be new')
        config = {'purpose':'confirmation', 'capture_card':ref(CARD), 'selections':{}, 'units':[]}
        for t in ('stair','push'):
            path = CAP/'observer_selection'/t/'selection.json'
            selected, gate = read(path), read(path.with_name('card.json'))
            require(gate['observer_profile'] == 'READY_FOR_PREPARATION' and gate['pool_complete'] is True
                    and not gate['missing_classes'], 'Selection pool incomplete: ' + t)
            require(gate['selection_sha256'] == ref(path)['sha256']
                    and selected['source']['capture_card_sha256'] == args.card_sha256, 'Selection binding mismatch')
            require([row['alias'] for row in selected['episodes']] == ['e1','e2'], 'Unexpected selection aliases')
            observer = read(CAP/'observers'/t/'report.json')
            require(observer['status'] == 'PREPARED_NOT_RELEASED' and observer['provider_requests'] == 0
                    and observer['selection_sha256'] == ref(path)['sha256'], 'Observer input not prepared')
            config['selections'][t] = ref(path)
            for row in selected['episodes']:
                config['units'].append({'id':f'media-{len(config["units"])+1:02d}', 'export':str(CAP/'exports'/t),
                    'run_index':row['run_index'], 'env_index':row['env_index'],
                    'full':ref(CAP/t/'evaluation/public_dev/behavior_full.json')})
        save(C/'media_inputs.json', config)
        commands.append(('media_pack', [P/'prepare_confirmation_media.py', '--inputs', C/'media_inputs.json',
            '--output', MEDIA, '--protocol', PROTOCOL, '--review-assignment', ASSIGNMENT]))
    if args.mode == 'audit':
        require(read(C/'postcapture/validate/receipt.json')['status'] == 'COMPLETED_CPU_ONLY_NOT_RELEASED',
                'Validate the real 40-packet routing before audit')
        require(not any(path.exists() for path in (C/'control/numeric_arithmetic', C/'control/input_arithmetic', C/'control/input_arithmetic_source_binding')),
                'Preserve previous arithmetic/binding outputs')
        h = Path(__file__).resolve().parent
        commands += [
            ('full_arithmetic_24_and_stair_sources', [h/'verify_numeric_arithmetic.py','--card-sha256',args.card_sha256]),
            ('push_archive_source_links', [h/'verify_push_source_links.py','--card-sha256',args.card_sha256]),
            ('source_header_snapshot', [h/'bind_source_headers.py','--stage','snapshot','--card-sha256',args.card_sha256]),
            ('input_arithmetic_28', [h/'verify_input_arithmetic.py','--attempt',CAP,'--original-capture',CAP,
                                    '--frozen-source',S,'--output',C/'control/input_arithmetic']),
            ('source_header_binding', [h/'bind_source_headers.py','--stage','bind','--card-sha256',args.card_sha256]),
            ('focus_scope_residual', [h/'verify_scope_residual_coverage.py']),
            ('focus_representatives', [h/'verify_representatives.py']),
            ('focus_stair_operands', [h/'verify_stair_focus_operands.py']),
            ('focus_push_operands', [h/'verify_push_focus_operands_v2.py']),
        ]
    return [(name, [sys.executable, '-B', *map(str, argv)]) for name,argv in commands]

def validate_packing():
    numeric, media = read(PACK/'packing_receipt.json'), read(MEDIA/'confirmation_media_receipt.json')
    require((numeric['episode_count'],numeric['focus_count'],len(numeric['packets'])) == (24,12,36), 'Wrong numeric counts')
    require(numeric['material_profile'] == 'related_v2' and numeric['actual_calls'] == media['actual_calls'] == 0,
            'Wrong profile or non-offline receipt')
    require(len(media['units']) == 4 and not any(media['missing_outcome_aliases'].values()), 'Incomplete media')
    assignment = read(ASSIGNMENT)['packets']
    packets = [(row['packet_id'],row['axis'],row['task'],PACK/row['facts_manifest']) for row in numeric['packets']]
    packets += [(row['unit_id'],'media',row['task'],Path(row['facts_manifest']['file'])) for row in media['units']]
    require({p[0] for p in packets} == set(assignment) and len(packets) == 40, 'Assignment packet set differs')
    ds = []
    for packet,axis,task,path in packets:
        value = read(path)
        require(value['stage'] == 'facts' and value['reference_contract'] == 'p3new.reference_blocks.v2'
                and value['material_profile'] == 'related_v2', 'Wrong facts representation')
        bound = value['review_assignment']
        require(bound['reviewers'] == assignment[packet] and bound['source']['sha256'] == ref(ASSIGNMENT)['sha256'],
                'Assignment row/source differs')
        require(assignment[packet]['A']['judge_id'] == 'gpt6.1-sol-a', 'Wrong A identity')
        if assignment[packet]['B']['judge_id'] == 'deepseek': ds.append(packet)
    expected = {packet for packet,axis,_,_ in packets if axis == 'media'}
    for task in ('stair','push'):
        for axis in ('episode','focus'):
            expected.add(min(packet for packet,a,t,_ in packets if (a,t) == (axis,task)))
    require(set(ds) == expected and len(ds) == 8, 'Actual task routing differs from frozen DeepSeek roster')
    return {'status':'OFFLINE_PACK_COUNTS_AND_ROLE_ROUTING_CHECKED', 'units':40, 'expected_facts_answers':80,
            'deepseek_units':sorted(ds), 'actual_model_calls':0, 'facts_sealed':False, 'report_reveal_allowed':False}

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--mode', choices=('task','numeric','media','validate','audit'), required=True)
    parser.add_argument('--task', choices=('stair','push'))
    parser.add_argument('--card-sha256', required=True, help='Actual R5 frozen card SHA; never a guessed value')
    args = parser.parse_args()
    if (args.mode == 'task') != (args.task is not None): parser.error('--task is required only with --mode task')
    require(len(args.card_sha256) == 64 and all(c in '0123456789abcdef' for c in args.card_sha256), 'Invalid actual SHA')
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    card = check_freeze(args.card_sha256)
    out = C/'postcapture'/(args.task if args.mode == 'task' else args.mode)
    require(not out.exists(), 'Preserve prior/incomplete control output')
    out.mkdir(parents=True)
    receipt = {'status':'IN_PROGRESS', 'started_at_utc':now(), 'script':ref(__file__), 'capture_card':ref(CARD),
               'source_manifest':ref(MANIFEST), 'steps':[], 'model_calls':0, 'gpu_runs':0, 'report_requests':0}
    try:
        commands = commands_for(args,card,out)
        save(out/'commands.json', {'commands':commands, 'cwd':str(S)})
        env = os.environ.copy()
        env['PYTHONPATH'] = f'{S}/src:{S}:{P}'
        for name,argv in commands:
            step = {'name':name, 'argv':argv, 'started_at_utc':now()}
            save(out/(name+'.started.json'), step)
            with (out/(name+'.stdout.log')).open('xb') as stdout, (out/(name+'.stderr.log')).open('xb') as stderr:
                process = subprocess.run(argv, cwd=S, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            step.update(returncode=process.returncode, finished_at_utc=now(),
                        stdout=ref(out/(name+'.stdout.log')), stderr=ref(out/(name+'.stderr.log')))
            save(out/(name+'.exit.json'),step)
            receipt['steps'].append(step)
            require(process.returncode == 0, name+' failed; stop downstream and preserve outputs')
        if args.mode == 'validate': receipt['packing'] = validate_packing()
        check_freeze(args.card_sha256)
        receipt.update(status='COMPLETED_CPU_ONLY_NOT_RELEASED', finished_at_utc=now())
        save(out/'receipt.json',receipt)
        return 0
    except Exception as error:
        receipt.update(status='ERROR', finished_at_utc=now(), error_type=type(error).__name__, error=str(error))
        save(out/'receipt.json',receipt)
        with (out/'exception.traceback.txt').open('x',encoding='utf-8') as stream: traceback.print_exc(file=stream)
        traceback.print_exc()
        return 1

if __name__ == '__main__': raise SystemExit(main())
