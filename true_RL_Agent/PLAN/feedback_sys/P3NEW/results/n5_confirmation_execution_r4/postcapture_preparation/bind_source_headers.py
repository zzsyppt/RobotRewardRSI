"""R4 header/file joins around the existing 28-packet audit; no arithmetic claims."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from postcapture_cpu import CAP, C, PACK, RAW, MEDIA, CARD, check_freeze, read, ref, require, save

OUT = C/'control/input_arithmetic_source_binding'
AUDIT = C/'control/input_arithmetic/receipt.json'

def bound(path, expected):
    actual = ref(path)
    require(actual['sha256'] == expected['sha256'], 'Bound source hash differs: '+str(path))
    return actual

def snapshot(card_sha):
    check_freeze(card_sha)
    require(not OUT.exists(), 'Preserve prior source snapshot/binding')
    manifest = read(RAW/'manifest.json')
    catalogue = {r['file']:r for r in manifest['readable_materials']}
    packing, media = read(PACK/'packing_receipt.json'), read(MEDIA/'confirmation_media_receipt.json')
    require(packing['source_sha256']['raw_manifest'] == ref(RAW/'manifest.json')['sha256'], 'Packing/raw manifest mismatch')
    inputs = [ref(CARD),ref(RAW/'manifest.json'),ref(PACK/'packing_receipt.json'),ref(MEDIA/'confirmation_media_receipt.json')]
    rows = []
    for packet in packing['packets']:
        if packet['axis'] != 'episode': continue
        path = PACK/packet['facts_manifest']
        inputs.append(bound(path,{'sha256':packet['facts_manifest_sha256']}))
        facts = read(path)
        raw, = [r for r in facts['materials'] if r['role'] == 'raw_numeric']
        related = bound(path.parent/raw['file'],raw)
        header_path = RAW/'episodes'/packet['scope']/'episode.json'
        header = bound(header_path,catalogue[header_path.relative_to(RAW).as_posix()])
        npz = bound(header_path.parent/read(header_path)['original_values_npz']['file'],read(header_path)['original_values_npz'])
        rows.append({'packet':packet['packet_id'],'header':header,'related':related,'npz':npz,
                     'numeric_raw_manifest_header_bound':True,'basis':'actual_numeric_raw_manifest_and_facts_material'})
    for packet in media['units']:
        path = Path(packet['facts_manifest']['file'])
        inputs.append(bound(path,packet['facts_manifest']))
        facts = read(path)
        raw, = [r for r in facts['materials'] if r['role'] == 'raw_numeric']
        related = bound(path.parent/raw['file'],raw)
        header_path = MEDIA/packet['unit_id']/'raw/episodes'/packet['packet_alias']/'episode.json'
        header = ref(header_path)
        npz = bound(header_path.parent/read(header_path)['original_values_npz']['file'],read(header_path)['original_values_npz'])
        source_path = MEDIA/packet['unit_id']/'raw/source_numeric.json'
        source = read(source_path)
        inputs += [ref(source_path),bound(source['manifest']['file'],source['manifest'])]
        rows.append({'packet':packet['unit_id'],'header':header,'related':related,'npz':npz,
                     'source_numeric':ref(source_path),'numeric_raw_manifest_header_bound':False,
                     'basis':'current_media_header_snapshot_before_independent_arithmetic; source_numeric_manifest_identity_retained'})
    require(len(rows) == 28 and sum(r['numeric_raw_manifest_header_bound'] for r in rows) == 24, 'Wrong header counts')
    save(OUT/'snapshot.json', {'status':'PRE_AUDIT_SOURCE_SNAPSHOT_NOT_A_VERDICT','script':ref(__file__),
        'inputs':inputs,'headers':rows,'model_calls':0,'gpu_calls':0,
        'limits':['Four media headers are not declared by numeric_raw/manifest.json.',
                  'Snapshot/hash joins do not independently verify original archive run/env slicing or physical truth.']})

def bind(card_sha):
    check_freeze(card_sha)
    before, audit = read(OUT/'snapshot.json'), read(AUDIT)
    require(audit['status'] == 'PASS_WITH_EXPLICIT_LIMITS' and len(audit['packets']) == 28, '28-packet arithmetic is not complete/pass')
    checked = {r['path']:r for r in audit['checked_inputs']}
    results = []
    for row in before['headers']:
        for key in ('header','related','npz'):
            record = row[key]
            item = checked[record['file']]
            require(record['sha256'] == item['before_sha256'] == item['after_sha256'] == ref(record['file'])['sha256'],
                    'Snapshot/audit/current bytes differ: '+record['file'])
        results.append({'packet':row['packet'],'header':row['header'],
                        'numeric_raw_manifest_header_bound':row['numeric_raw_manifest_header_bound'],
                        'snapshot_to_audit_hashes_equal':True,'basis':row['basis']})
    for item in before['inputs']: bound(item['file'],item)
    save(OUT/'binding.json', {'status':'PASS_FILE_BINDING_WITH_EXPLICIT_LIMITS','snapshot':ref(OUT/'snapshot.json'),
        'audit':ref(AUDIT),'headers':results,'header_count':28,'numeric_manifest_bound_count':24,
        'media_snapshot_bound_count':4,'limits':before['limits'],'model_calls':0,'gpu_calls':0})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--stage',choices=('snapshot','bind'),required=True)
    parser.add_argument('--card-sha256',required=True)
    args = parser.parse_args()
    (snapshot if args.stage == 'snapshot' else bind)(args.card_sha256)
