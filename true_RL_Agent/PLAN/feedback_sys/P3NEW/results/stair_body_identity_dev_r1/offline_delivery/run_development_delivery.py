"""One CPU-only development serialization with the existing two-task assembler."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
DEV = B / 'stair_body_identity_dev_r1'
S, O = DEV / 'source', DEV / 'offline_delivery'
C = B / 'n5_execution_20260930_r3'
CAP = B / 'n5_confirmation_20260930_r3'


def read(p): return json.loads(Path(p).read_bytes())
def ref(p):
    p = Path(p).resolve(); raw = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
def save(name, data):
    with (O / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n')
def strings(value):
    if isinstance(value, str): yield value
    elif isinstance(value, dict):
        for v in value.values(): yield from strings(v)
    elif isinstance(value, list):
        for v in value: yield from strings(v)


def main():
    assert not (O / 'serialized').exists() and not (O / 'execution_receipt.json').exists()
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f'{S}/src:{S}')
    sys.path[:0] = [str(S / 'src'), str(S)]
    derivation = read(O / 'assembler_derivation.json')
    original = S / 'PLAN/feedback_sys/P3NEW/results/n5_confirmation_execution_r1/assemble_delivery.py'
    driver = O / 'assemble_delivery_development.py'
    assert ref(original)['sha256'] == derivation['original_sha256']
    assert driver.read_bytes() == original.read_bytes().replace(derivation['before'].encode(), derivation['after'].encode())
    assert ref(driver)['sha256'] == derivation['development_copy_sha256']
    manifest_path = DEV / 'development_source_manifest.json'; manifest = read(manifest_path)
    def source_check():
        for relative, expected in manifest.items():
            actual = ref(S / relative)
            assert actual['sha256'] == expected['sha256'] and actual['size_bytes'] == expected['size_bytes'], relative
    source_check()
    combined = read(DEV / 'combined_audit.json')
    binding = read(DEV / 'visual_review_binding.json')
    review_path, seal_path = O / 'inputs/root_module_review.json', O / 'inputs/root_module_review_receipt.json'
    review, seal = read(review_path), read(seal_path)
    assert combined['scope'] == 'development_after_r3' and seal['scope'] == 'DEVELOPMENT_AFTER_R3_NOT_CONFIRMATION'
    assert len(combined['observations']) == len(review['decisions']) == seal['raw_observations'] == 5
    assert all(d['verdict'] == 'supported' for d in review['decisions'])
    assert seal['review']['sha256'] == ref(review_path)['sha256']
    assert seal['combined_audit']['sha256'] == binding['combined_audit']['sha256'] == ref(DEV / 'combined_audit.json')['sha256']
    assert seal['visual_review_binding'] == binding['source_sha256'] == review['source_sha256']
    prior_inputs = C / 'delivery_inputs.ready.json'
    ready = deepcopy(read(prior_inputs))
    ready.update(scope='development_after_r3', confirmation_admitted=False,
                 status='READY_FOR_DEVELOPMENT_OFFLINE_SERIALIZATION_NOT_RELEASED',
                 development_source_manifest=ref(manifest_path), visual_review_binding=ref(DEV / 'visual_review_binding.json'))
    item = ready['tasks']['stair']
    item.update(audits=[ref(DEV / a / 'sol_response/mechanical_audit.json') for a in ('e1', 'e2')],
                combined_audit=ref(DEV / 'combined_audit.json'), review=ref(review_path),
                as_of_cutoff=seal['recorded_at_utc'], as_of_cutoff_ref={**ref(seal_path), 'pointer': '/recorded_at_utc'})
    assert ready['tasks']['push'] == read(prior_inputs)['tasks']['push']
    save('delivery_inputs.ready.json', ready)
    receipt = {'scope': 'development_after_r3', 'confirmation_admitted': False, 'status': 'STARTED',
               'started_at_utc': datetime.now(timezone.utc).isoformat(), 'script': ref(__file__),
               'source_manifest': ref(manifest_path), 'source_receipt': ref(DEV / 'source_receipt.json'),
               'assembler': ref(driver), 'assembler_derivation': ref(O / 'assembler_derivation.json'),
               'inputs': ref(O / 'delivery_inputs.ready.json'), 'root_review': ref(review_path),
               'root_seal': ref(seal_path), 'visual_review_binding': ref(DEV / 'visual_review_binding.json'),
               'model_calls': 0, 'API_calls': 0, 'author_calls': 0, 'gpu_calls': 0, 'automatic_retries': 0}
    save('started.json', receipt)
    try:
        spec = importlib.util.spec_from_file_location('development_delivery', driver)
        A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
        result = A.assemble(O / 'delivery_inputs.ready.json', S, B / 'text_author_smoke_r1', O / 'serialized')
        assert result['status'] == 'offline_serialized_not_sent'
        assert result['model_calls'] == result['new_author_calls'] == result['gpu_calls'] == 0
        assert result['semantic_decisions_created'] is False
        row = result['tasks']['stair']
        full, public, payload = (read(row[k]['path']) for k in ('full', 'public', 'payload'))
        originals = [r['raw_observation']['text'] for r in combined['observations']]
        old_combined = read(CAP / 'observers/stair/combined_audit.json')
        old_review = read(CAP / 'observers/stair/root_module_review.json')
        old_rejected = [old_combined['observations'][d['observation_index']]['raw_observation']['text']
                        for d in old_review['decisions'] if d['verdict'] != 'supported']
        assert len(old_rejected) == 2 and all(t not in originals for t in old_rejected)
        for obj in (full, public, payload):
            assert not any(t in s for t in old_rejected for s in strings(obj))
        evidence = full['observer_evidence']
        admitted = [r['raw_observation']['text'] for r in evidence['observations']]
        assert admitted == originals
        assert evidence['semantic_review']['source_sha256'] == review['source_sha256']
        assert row['nonvisual_unchanged'] and row['public_limit_bytes'] == 8192 and row['public_bytes'] <= 8192
        assert row['payload']['size_bytes'] <= 256 * 1024
        push_old = read(CAP / 'delivery_offline_r1/assembly_receipt.json')['tasks']['push']
        push_equal = {k: Path(result['tasks']['push'][k]['path']).read_bytes() == Path(push_old[k]['path']).read_bytes()
                      for k in ('public', 'payload', 'http_request', 'full')}
        assert all(push_equal.values()), push_equal
        source_check()
        receipt.update(status='DEVELOPMENT_OFFLINE_DELIVERY_VERIFIED_NOT_SENT',
                       finished_at_utc=datetime.now(timezone.utc).isoformat(),
                       assembly_receipt=ref(O / 'serialized/assembly_receipt.json'),
                       source_files_verified_unchanged=len(manifest),
                       stair={'new_root_supported_observations': 5, 'new_observation_texts_sha256':
                              [hashlib.sha256(t.encode()).hexdigest() for t in originals],
                              'raw_observations_in_full_exactly_match_new_five': True,
                              'old_two_nonadmitted_clauses_absent_from_full_public_payload': True,
                              'old_two_clause_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in old_rejected],
                              'nonvisual_full_and_public_sections_unchanged': True,
                              'public_bytes': row['public_bytes'], 'public_limit_bytes': 8192,
                              'author_payload_bytes': row['payload']['size_bytes'], 'author_payload_limit_bytes': 256*1024,
                              'shown_S3': row['shown_S3'], 'source_binding': row['source_binding'],
                              'public': row['public'], 'payload': row['payload'], 'http_request': row['http_request'],
                              'budget': row['budget']},
                       push={'same_original_six_and_review_reused': True, 'outputs_byte_identical': push_equal},
                       limits='Development regression only; existing semantic verdicts consumed, no new model/score/author delivery/formal N5.')
        save('execution_receipt.json', receipt)
        print(json.dumps({'status': receipt['status'], 'stair_public_bytes': row['public_bytes'],
                          'stair_author_payload_bytes': row['payload']['size_bytes'], 'push_byte_identical': push_equal}), flush=True)
    except Exception as e:
        receipt.update(status='ERROR_PRESERVED_NO_RETRY', error_type=type(e).__name__, error=str(e))
        save('execution_receipt.json', receipt); traceback.print_exc(); raise


if __name__ == '__main__': main()
