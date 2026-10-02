"""Future fixed N5 facts preparation only; no transport, retry, seal or scoring.

The historical results/*/prepare_facts_offline.py files remain replay artifacts.
Input: assignment ref and 40 units (packet_id, task, axis, scope, facts_manifest ref).
"""
from copy import deepcopy
import argparse
from pathlib import Path
import shutil
import run_confirmation_review as runner


def checked(reference):
    path = Path(reference.get('file', reference.get('path', ''))).resolve()
    runner.require(path.is_file() and runner.ref(path)['sha256'] == reference['sha256'], 'Input SHA differs: ' + str(path))
    runner.require('size_bytes' not in reference or path.stat().st_size == reference['size_bytes'], 'Input size differs')
    return path


def output_cap(judge):
    runner.require(judge in ('deepseek', *runner.SOL_PAIR), 'Future N5 identity required')
    return 65536 if judge == 'deepseek' else 32768


def roster(index):
    assignment = checked(index['assignment'])
    packets = runner.read(assignment)['packets']
    units = index['units']
    expected = {f'{axis}-{i:02d}' for axis, count in (('episode', 24), ('focus', 12), ('media', 4)) for i in range(1, count + 1)}
    runner.require(len(units) == 40 and len({u['packet_id'] for u in units}) == 40
                   and {u['packet_id'] for u in units} == set(packets) == expected
                   and all(u['packet_id'].startswith(u['axis'] + '-') for u in units), 'Exactly the fixed 40 units required')
    for task in ('stair', 'push'):
        for axis, count in (('episode', 12), ('focus', 6), ('media', 2)):
            runner.require(sum(u['task'] == task and u['axis'] == axis for u in units) == count,
                           'Fixed per-task axis denominator differs')
    rows = {u['packet_id']: runner.assignment_fields(assignment, u['packet_id'])['review_assignment'] for u in units}
    ds = {p for p, binding in rows.items() if binding['reviewers']['B']['judge_id'] == 'deepseek'}
    first = {min(u['packet_id'] for u in units if u['task'] == task and u['axis'] == axis)
             for task in ('stair', 'push') for axis in ('episode', 'focus')}
    runner.require(ds == first | {u['packet_id'] for u in units if u['axis'] == 'media'}, 'Fixed DS eight-unit routing differs')
    return rows


def manifest_for_judge(original, judge, output):
    original, output = Path(original).resolve(), Path(output).resolve()
    value = deepcopy(runner.read(original))
    runner.require(value['stage'] == 'facts' and judge in runner.review_judges(value), 'Wrong facts identity')
    for item in value['materials']:
        item['file'] = str(runner.referenced(original.parent, item)[0])
    value['max_output_tokens'] = output_cap(judge)
    output.parent.mkdir(parents=True, exist_ok=True)
    runner.save(output, value, exclusive=True)
    return output


def prepare(index, output):
    rows = roster(index)
    output = Path(output).resolve()
    runner.require(not output.exists(), 'Fresh output required; never resume or overwrite')
    output.mkdir(parents=True)
    try:
        result = []
        for unit in index['units']:
            packet = unit['packet_id']
            original = checked(unit['facts_manifest'])
            facts = runner.read(original)
            runner.require(facts.get('packet_id') == packet and facts.get('review_assignment') == rows[packet], 'Original facts assignment differs')
            for slot, reviewer in rows[packet]['reviewers'].items():
                judge = reviewer['judge_id']
                dest = output / packet / judge
                manifest = manifest_for_judge(original, judge, dest / 'facts.manifest.json')
                card = runner.prepare(manifest, judge, dest / 'prepared')
                result.append({'packet_id': packet, 'judge_id': judge, 'slot': slot,
                               'facts_manifest': runner.ref(manifest), 'prepared': runner.ref(dest / 'prepared/prepare.json'),
                               'max_output_tokens': card['max_output_tokens']})
        receipt = {'status': 'PREPARED_NOT_RELEASED', 'assignment': index['assignment'], 'rows': result,
                   'api_calls': 0, 'facts_sealed': 0, 'capacity_release': False,
                   'limits': 'DS 65536 hard output cap; SOL 32768 reservation only. Measure actual requests before dispatch.'}
        runner.save(output / 'preparation.json', receipt, exclusive=True)
        return receipt
    except BaseException:
        # Only this explicitly new, exclusively owned directory; no source links are created.
        runner.require(output.is_dir() and not output.is_symlink(), 'Unsafe cleanup target')
        shutil.rmtree(output)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(runner.read(args.index), args.output)
