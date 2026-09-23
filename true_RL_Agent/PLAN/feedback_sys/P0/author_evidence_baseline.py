"""Extract literal historical rationales and check selected citations. No candidate execution.

Run from any directory with Python 3.10+. Formal source files are the read-only SSH
copies in author_evidence_raw; this script performs no network or provider calls.
The ten Push annotations are a disclosed human review, not an automatic judge.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FORMAL_NAME = 'formal_go2_stair_15round_20260901_c6ab8ba'
FORMAL = OUT / 'inventory_remote_raw/outputs' / FORMAL_NAME
FORMAL_SOURCE = OUT / 'author_evidence_raw/formal_c6/provider_candidates'
PUSH_ROOT = ROOT / 'PLAN/2new_world/Go2_push/attempt_08/session'
REMOTE = '/data/yuyangliu/AGENT/true_RL_Agent/outputs/' + FORMAL_NAME




def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def walk(value, path=''):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, path + '/' + key.replace('~', '~0').replace('/', '~1'))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, path + '/' + str(index))


def literal_rationale(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n.value for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == 'rationale']
    assert len(nodes) == 1, str(path)
    value = ast.literal_eval(nodes[0])
    assert isinstance(value, str)
    return value, nodes[0].lineno


def extraction(source_root, session, family):
    rows = []
    for source in sorted(source_root.rglob('components.py')):
        generation, lane, folder, _, _ = source.relative_to(source_root).parts
        attempt = 'attempt_' + folder.split('_', 1)[0].zfill(6)
        candidate_id = folder.split('_', 1)[1]
        receipt_path = session / 'authoring' / generation / lane / 'authoring_attempts' / attempt / 'attempt_result.json'
        payload_path = receipt_path.with_name('prompt_payload.json')
        receipt, payload = read(receipt_path), read(payload_path)
        provider = receipt['provider_attempt']
        assert provider['candidate_id'] == candidate_id
        assert receipt['prompt_payload_checksum'] == 'sha256:' + sha(payload_path)
        rationale, line = literal_rationale(source)
        checksum = 'sha256:' + sha(source)
        historical_hash = provider.get('candidate_program_source_checksum')
        repair_bindings = []
        if historical_hash is not None:
            assert historical_hash == checksum
            binding = 'same_attempt_source_checksum'
        else:
            # Failed static verification omits the program hash; the subsequent
            # repair input carries the exact failed source as a source_payload.
            for later in sorted(receipt_path.parents[1].glob('attempt_*/prompt_payload.json')):
                if later.parent.name <= attempt:
                    continue
                later_receipt = read(later.with_name('attempt_result.json'))
                assert later_receipt['prompt_payload_checksum'] == 'sha256:' + sha(later)
                for item in read(later).get('source_payloads', []):
                    if hashlib.sha256(item.get('content', '').encode()).hexdigest() == sha(source):
                        repair_bindings.append({**ref(later), 'logical_id': item['logical_id']})
            assert repair_bindings, str(source)
            binding = 'same_attempt_identity_plus_later_repair_exact_source'
        round_id = generation + '-' + lane
        train = session / 'runner' / round_id / 'train/summary.json'
        dev = session / 'runner' / round_id / 'eval/public_dev/summary.json'
        outcome = {'status': 'no_matching_completed_training'}
        if train.exists():
            train_obj = read(train)
            if train_obj.get('program_source_checksum') == checksum:
                outcome = {'status': 'training_' + train_obj['status'], 'train': ref(train)}
                if dev.exists():
                    d = read(dev)
                    assert d['subject_manifest_hash'] == train_obj['subject_manifest_hash']
                    outcome.update(public_dev=ref(dev), cell=d['resolved_cell_identity']['cell_variant_id'])
        temporal = {'per_episode', 'force_events', 'push_events', 'dwell_break_reason',
                    'sequence_recovery_control_step', 'post_push_trace', 'stable_dwell_trace', 'video_frames'}
        present = sorted({p.rsplit('/', 1)[-1] for p, _ in walk(payload)} & temporal)
        rows.append({'family': family, 'round_id': round_id, 'generation': int(generation[-3:]),
                     'lane': lane, 'attempt': attempt, 'candidate_id': candidate_id,
                     'source': ref(source), 'remote_source': REMOTE + '/provider_candidates/' + source.relative_to(source_root).as_posix() if family == 'formal_c6' else None,
                     'rationale_line': line, 'rationale': rationale, 'payload': ref(payload_path),
                     'receipt': ref(receipt_path), 'prompt_hash_matches_receipt': True,
                     'source_binding': binding, 'repair_source_bindings': repair_bindings,
                     'provider_status': provider.get('provider_status'),
                     'verifier_status': provider.get('verifier_status'),
                     'frame_kind': payload.get('authoring_context_frame', {}).get('frame_kind', 'legacy_unspecified'),
                     'prompt_schema': payload.get('schema_version'), 'temporal_field_keys_found': present,
                     'outcome': outcome})
    return rows


def json_pointer(obj, path):
    for part in path.strip('/').split('/'):
        key = part.replace('~1', '/').replace('~0', '~')
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def push_review(row, annotation):
    payload = read(ROOT / row['payload']['path'])
    rationale = row['rationale']
    assert annotation['candidate_id'] == row['candidate_id']
    assert annotation['source_sha256'] == row['source']['sha256']
    assert annotation['payload_sha256'] == row['payload']['sha256']
    strata, negatives = annotation['strata'], annotation['negatives']
    mechanism, judgment = annotation['mechanism_quote'], annotation['assessment_zh']
    assert mechanism in rationale
    root = '/public_feedback_summary'
    feedback = payload['public_feedback_summary']
    if isinstance(feedback.get('current_best'), dict):
        root += '/current_best'
        parent = feedback['current_best']
    else:
        parent = feedback
    references = []
    score = re.search(r'public_dev mean ([0-9.]+)', rationale)
    if score:
        actual = parent['public_dev']['score_stats']['mean']
        assert math.isclose(float(score[1]), actual, abs_tol=0.00005)
        references.append({'type': 'parent_score', 'quote': score[0], 'pointer': root + '/public_dev/score_stats/mean', 'value': actual})
    labels = r'(success_recovered|fall_side|fall_forward|not_recovered|height_collapse)'
    counts = [(m[0], m[1], m[2]) for m in re.finditer(r'(\d+) ' + labels + r'\b', rationale)]
    counts += [(m[0], m[2], m[1]) for m in re.finditer(labels + r' (\d+)\b', rationale)]
    for quote, text, label in counts:
        actual = parent['public_dev']['failure_breakdown']['counts'][label]
        assert int(text) == actual
        references.append({'type': 'parent_failure_count', 'quote': quote,
                           'pointer': root + '/public_dev/failure_breakdown/counts/' + label, 'value': actual})
    for name, text in strata:
        assert text in rationale
        actual = parent['public_dev']['stratum_results'][name]['n_over_N']
        assert actual == text
        references.append({'type': 'stratum_count', 'quote': text, 'stratum': name,
                           'pointer': root + '/public_dev/stratum_results/' + name + '/n_over_N', 'value': actual})
    trials = [(p, v) for p, v in walk(payload.get('public_history', {}), '/public_history')
              if isinstance(v, list) and len(v) == 9 and isinstance(v[1], str) and v[0] in ('a', 'b')]
    for negative in negatives:
        candidate_id, quoted_score = negative['candidate_id'], negative['quoted_score']
        match = [(p, v) for p, v in trials if v[1] == candidate_id]
        assert len(match) == 1, candidate_id
        pointer, trial = match[0]
        claim = {'type': 'negative_history', 'candidate_id': candidate_id,
                 'pointer': pointer, 'status': trial[4], 'public_dev_delta': trial[5],
                 'claim_limit': 'observed outcome/status, not causal disproof of a mechanism'}
        if quoted_score is not None:
            assert quoted_score in rationale
            basis = negative['score_basis']
            assert basis
            value = sum(json_pointer(payload, p) for p in basis)
            assert math.isclose(float(quoted_score), value, abs_tol=0.000051)
            claim.update(quoted_score=quoted_score, computed_score=value, score_basis=basis)
        references.append(claim)
    for annotation_claim in annotation['additional_claims']:
        claim = dict(annotation_claim)
        assert claim['quote'] in rationale
        value = json_pointer(payload, claim['pointer'])
        if claim['type'] == 'negative_candidate_failure_count':
            assert value == claim['value']
            assert json_pointer(payload, claim['identity_pointer']) == claim['candidate_id']
        elif claim['type'] == 'closed_count_table_zero':
            assert sum(value['counts'].values()) == value['total']
            assert all(value['counts'].get(label, 0) == 0 for label in claim['labels'])
        else:
            raise ValueError('Unknown human claim check: ' + claim['type'])
        references.append(claim)
    row['review'] = {'review_method': 'human-selected claims; numeric and identity references checked by this script',
                     'verified_aggregate_references': references,
                     'explicit_parent_mean': bool(score), 'explicit_failure_counts': bool(counts),
                     'explicit_stratum_n_over_N': bool(strata), 'negative_history_discussed': bool(negatives),
                     'negative_score_quoted': any(n['quoted_score'] is not None for n in negatives),
                     'process_mechanism_claim': {'quote': mechanism, 'status': 'not_supported_by_process_observation_in_this_payload', 'assessment_zh': judgment},
                     'source_based_claims': 'kernel/weight/clip/slope descriptions require source/arithmetic checks; are not episode observations'}


def source_semantic_review(rows, annotation):
    target = next(r for r in rows if r['candidate_id'] == annotation['candidate_id'])
    tree = ast.parse((ROOT / target['source']['path']).read_text(encoding='utf-8'))
    terms = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'RewardTerm' and any(k.arg == 'name' and ast.literal_eval(k.value) == annotation['term_name'] for k in n.keywords)]
    assert len(terms) == 1
    term = terms[0]
    settings = {k.arg: ast.literal_eval(k.value) for k in term.keywords if k.arg in ('weight', 'clip_min', 'clip_max')}
    assert settings == annotation['settings']
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == annotation['function_name'])
    returned = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    assert ast.unparse(returned.value) == annotation['expected_return_expression']
    consumer = next(r for r in rows if r['candidate_id'] == annotation['consumer_candidate_id'])
    activation = json_pointer(read(ROOT / consumer['payload']['path']), annotation['activation_pointer'])
    assert activation['mechanism'] == 'RewardTerm:' + annotation['term_name']
    assert activation['weight'] == settings['weight']
    interval = sorted(settings['weight'] * settings[k] for k in ('clip_min', 'clip_max'))
    assert interval[0] >= 0 and activation['last_mean'] > 0
    return {'candidate_id': target['candidate_id'], 'source': target['source'],
            'return_line': returned.lineno, 'term_line': term.lineno, 'settings': settings,
            'weighted_reward_rate_interval_before_dt': interval, 'runtime_rule': annotation['runtime_rule'],
            'next_actual_payload': consumer['payload'], 'activation_pointer': annotation['activation_pointer'],
            'observed_last_mean': activation['last_mean'], 'finding': annotation['finding'],
            'causal_claim': annotation['causal_claim']}


def main():
    push = extraction(PUSH_ROOT / 'provider_candidates', PUSH_ROOT, 'push_attempt_08')
    formal = extraction(FORMAL_SOURCE, FORMAL, 'formal_c6')
    assert len(push) == 10 and len(formal) == 36
    annotations_path = OUT / 'author_review_annotations.json'
    annotations = read(annotations_path)
    by_candidate = {r['candidate_id']: r for r in annotations['push_reviews']}
    assert len(by_candidate) == len(annotations['push_reviews']) == len(push)
    for row in push:
        push_review(row, by_candidate[row['candidate_id']])
    for row in formal:
        text = row['rationale']
        start = text.lower().find('hypothesis')
        row['mechanism_fixture'] = {'literal_hypothesis_tail': text[start:] if start >= 0 else None,
                                   'review_status': 'literal_and_input_identity_checked; not_an_exhaustive_numeric_or_semantic_review',
                                   'use': 'retrospective_rule_fixture_only; no_prediction_hit_label'}
    flags = ['explicit_parent_mean', 'explicit_failure_counts', 'explicit_stratum_n_over_N', 'negative_history_discussed', 'negative_score_quoted']
    totals = {key: sum(r['review'][key] for r in push) for key in flags}
    totals['all_four_explicit_numeric_families'] = sum(all(r['review'][key] for key in (flags[0], flags[1], flags[2], flags[4])) for r in push)
    counterexample = source_semantic_review(push, annotations['source_semantic_review'])
    counts = {'push_rationales': len(push), 'formal_rationales': len(formal),
              'push_actual_payloads': len(list((PUSH_ROOT / 'authoring').rglob('prompt_payload.json'))),
              'formal_actual_payloads': len(list((FORMAL / 'authoring').rglob('prompt_payload.json'))),
              'formal_source_binding': dict(Counter(r['source_binding'] for r in formal)),
              'formal_outcome_status': dict(Counter(r['outcome']['status'] for r in formal)),
              'formal_with_matching_public_dev': sum('public_dev' in r['outcome'] for r in formal),
              'push_citation_counts': totals}
    result = {'method': 'AST literal extraction; same-attempt payload/checksum binding; ten human Push annotations with mechanical citation checks',
              'scope': 'historical author baseline and retrospective fixture inventory; no new model response or policy rollout',
              'counts': counts, 'push': push, 'formal': formal,
              'human_annotations': ref(annotations_path), 'source_semantic_counterexample': counterexample,
              'design_revision': ['Historical A payloads do not replace fresh A outputs in a matched A/T study.',
                  'Never append later v3 replays to historical v2 A as if available at its decision time; build both arms from one declared current replay checkpoint/case cutoff.',
                  'Use historical rationale/outcome pairs only to test retrospective adjudication rules; separate no-training, failed-training and completed public-dev outcomes.',
                  'One human judges interpretation/action quality; mechanical checks validate cited IDs, JSON pointers, units, counts, source/protocol and observation cutoff.',
                  'Mechanical value matching does not establish entailment, temporal order, causality or grader independence. Randomize arm labels/order and preserve the rubric and raw judgments.']}
    (OUT / 'author_evidence_baseline.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == '__main__':
    main()
