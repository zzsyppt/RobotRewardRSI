"""Prepare R3 copies of existing full/source controls; never run data checks."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3 = HERE.parents[2]
E2 = P3/'results/n5_confirmation_execution_r2'
SOURCE = P3/'results/n5_confirmation_source_r3'
FROZEN = SOURCE/'source_final/PLAN/feedback_sys/P3NEW'
ROOT = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r3_final'
CONTROL = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930_r3'
rows = []

def ref(path):
    data = path.read_bytes()
    return {'file':str(path),'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}
def transform(source,name,replacements):
    text = source.read_text(encoding='utf-8')
    edits = []
    for old,new in replacements:
        n = text.count(old)
        if not n: raise ValueError('Missing expected old text '+name+' '+old)
        text = text.replace(old,new)
        edits.append({'old':old,'new':new,'occurrences':n})
    ast.parse(text)
    target = HERE/name
    with target.open('x',encoding='utf-8',newline='\n') as stream: stream.write(text)
    rows.append({'original':ref(source),'adapted':ref(target),'changes':edits,'executed_on_R3_inputs':False})

# Copy only this already frozen function. Force, surface and declaration arithmetic
# remain byte-for-byte source segments; only projection binding and repo location differ.
source_path = FROZEN/'verify_stair_confirmation_arithmetic.py'
source_text = source_path.read_text(encoding='utf-8')
tree = ast.parse(source_text)
function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'verify_stair_operand_sources')
body = ast.get_source_segment(source_text,function)
old = '    record("episode_metadata_mapping", encoded(raw["episode"]) == encoded({k: v for k, v in header["episode"].items() if k != "direct_measurements"}))'
new = '''    projection = verify_episode_projection(raw["episode"], header["episode"], task="stair", contract=load_contract(FROZEN_SOURCE_ROOT, SOURCE_MANIFEST))
    record("episode_metadata_mapping", projection["status"] == "PASS_EXACT_RETAINED_FIELDS", projection=projection)'''
assert body.count(old) == 1
body = body.replace(old,new)
assert body.count('repo = Path(__file__).resolve().parents[3]') == 1
body = body.replace('repo = Path(__file__).resolve().parents[3]','repo = FROZEN_SOURCE_ROOT')
text = '''"""R3 control copy: frozen Stair operand arithmetic; exact terminal projection binding."""
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_stair_confirmation_arithmetic import encoded, decode_column
from terminal_projection_check import load_contract, verify_episode_projection
'''+f"FROZEN_SOURCE_ROOT = Path({ROOT!r})\nSOURCE_MANIFEST = Path({(CONTROL+'/runtime_source_manifest.json')!r})\n\n"+body+'\n'
ast.parse(text)
target = HERE/'stair_operand_source_adapter.py'
with target.open('x',encoding='utf-8',newline='\n') as stream: stream.write(text)
rows.append({'original':ref(source_path),'adapted':ref(target),'extracted_function':'verify_stair_operand_sources',
    'changes':[{'old':old,'new':new,'occurrences':1},
               {'old':'repo = Path(__file__).resolve().parents[3]','new':'repo = FROZEN_SOURCE_ROOT','occurrences':1}],
    'preserved_arithmetic':'All original declared fields, NPZ bytes, source raw columns, force/history norms and normative surface operations',
    'executed_on_R3_inputs':False})

roots = [('n5_confirmation_20260930_r2','n5_confirmation_20260930_r3'),
         ('n5_confirmation_source_r2','n5_confirmation_source_r3_final'),
         ('n5_execution_20260930_r2','n5_execution_20260930_r3')]
transform(E2/'verify_numeric_arithmetic.py','verify_numeric_arithmetic.py',roots+[
    ("CARD_SHA = '6bea98ed63e1f1e516631e58205e3da93406311d1b87466e445fe754b0c22794'",'CARD_SHA = None'),
    ('497d14b6a133d1ba3afa97e15c39b00515479692e528aac648970851245f3b76','5cb1b36c5ca2e7af40d62c51e7cf1fe9aae4790e080353b8947cc6e14e5902d5'),
    ("and raw_receipt['material_profile'] == packing['material_profile'] == 'related_v1'",
     "and raw_receipt['material_profile'] == 'related_v1' and packing['material_profile'] == 'related_v2'"),
    ('Expected untouched related_v1 confirmation facts pack','Expected untouched related_v1 raw / related_v2 confirmation facts pack'),
    ("        stair, push = modules.values()",'''        stair, push = modules.values()
        import stair_operand_source_adapter as source_adapter
        import terminal_projection_check as projection_check
        for control_module in (source_adapter, projection_check):
            control_ref = ref(control_module.__file__)
            watched[control_ref['path']] = control_ref
            receipt.setdefault('control_modules', {})[control_module.__name__] = control_ref
        projection_contract = projection_check.load_contract(S, SOURCE_MANIFEST)
        frozen_ref(HELPERS / 'export_confirmation_raw.py', source)
        raw_catalogue = {r['file']: r for r in raw_receipt['readable_materials']}'''),
    ("            logical_id = aliases[task, scope]",'''            require(facts['reference_contract'] == 'p3new.reference_blocks.v2'
                    and facts['material_profile'] == raw['profile'] == 'related_v2'
                    and raw['schema_version'] == 'p3new.related_episode.v2', 'Wrong R3 facts representation')
            header_name = f'episodes/{scope}/episode.json'
            header_path = RAW / header_name
            header = read(header_path, raw_catalogue[header_name]['sha256'])
            model_header_ref, = [r for r in raw['source_materials'] if r['file'] == header_name]
            require(model_header_ref['sha256'] == raw_catalogue[header_name]['sha256'], 'Actual raw header reference differs')
            projection = projection_check.verify_episode_projection(raw['episode'], header['episode'],
                task=task, contract=projection_contract)
            projection_path = OUT / 'terminal_projection' / (packet + '.json')
            save(projection_path, projection)
            require(projection['status'] == 'PASS_EXACT_RETAINED_FIELDS', 'Actual terminal projection differs: ' + packet)
            logical_id = aliases[task, scope]'''),
    ("'full_derivation': ref(derivation_path), 'result': ref(result_path), 'status': result['status'],",
     "'full_derivation': ref(derivation_path), 'terminal_projection': ref(projection_path), 'result': ref(result_path), 'status': result['status'],"),
    ('link = stair.verify_stair_operand_sources(raw,','link = source_adapter.verify_stair_operand_sources(raw,'),
    ('Push has no frozen independent archive-to-derivation source-link verifier. Supplied-byte/metadata arithmetic does not authenticate physical construction or terminal meaning.',
     'Push archive-to-derivation source links are a separate control supplement after this receipt. Neither step authenticates physical construction or terminal meaning.'),
    ('No independent original NPZ-to-derivation source link; supplied-byte/metadata arithmetic only.',
     'This receipt contains only supplied-byte/metadata arithmetic; run the separate R3 Push archive source-link control and retain its own receipt.'),
    ("if __name__ == '__main__':\n    raise SystemExit(main())",'''if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--card-sha256', required=True)
    args = parser.parse_args()
    CARD_SHA = args.card_sha256
    require(len(CARD_SHA) == 64 and all(c in '0123456789abcdef' for c in CARD_SHA), 'Invalid actual card SHA')
    raise SystemExit(main())'''),
    ('R2','R3'),('p3new.n5_r2_numeric_arithmetic.v1','p3new.n5_r3_numeric_arithmetic.v1'),
])

transform(E2/'source_link_audit_r1/verify_push_source_links.py','verify_push_source_links.py',[
    ('n5_confirmation_20260930_r2','n5_confirmation_20260930_r3'),
    ('n5_execution_20260930_r2','n5_execution_20260930_r3'),
    ('import numpy as np','import numpy as np\nfrom terminal_projection_check import load_contract, verify_episode_projection\nfrom postcapture_cpu import check_freeze'),
    ("    export_manifest = load(C/'numeric_raw/manifest.json')",f'''    source_root = Path({ROOT!r})
    contract = load_contract(source_root, E/'runtime_source_manifest.json')
    read_bytes(Path(__file__))
    read_bytes(Path(__file__).with_name('terminal_projection_check.py'))
    read_bytes(Path(contract['policy_source']['path']), contract['policy_source']['sha256'])
    export_manifest = load(C/'numeric_raw/manifest.json')'''),
    ("    check('packing/private SHA', packing['raw_private_source_binding']['sha256'], private_ref['sha256'])",'''    check('packing/private SHA', packing['raw_private_source_binding']['sha256'], private_ref['sha256'])
    check('R3 raw profile', export_manifest['material_profile'], 'related_v1')
    check('R3 packed profile', packing['material_profile'], 'related_v2')'''),
    ("        check(f'{pid}/related header', related['episode'], {k:v for k,v in header['episode'].items() if k!='direct_measurements'})",'''        projection = verify_episode_projection(related['episode'], header['episode'], task='push', contract=contract)
        check(f'{pid}/related exact retained header fields', projection['status'], 'PASS_EXACT_RETAINED_FIELDS')
        check(f'{pid}/related profile', related['profile'], 'related_v2')
        check(f'{pid}/facts reference contract', fact['reference_contract'], 'p3new.reference_blocks.v2')'''),
    ("'field_byte_checks':fields,'difference_count':len(differences)-before,",
     "'field_byte_checks':fields,'terminal_projection':projection,'difference_count':len(differences)-before,"),
    ("if __name__=='__main__':\n    sys.exit(main())",'''if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--card-sha256', required=True)
    args = parser.parse_args()
    check_freeze(args.card_sha256)
    sys.exit(main())'''),
    ('R2','R3'),
])

with (HERE/'full_source_control_adaptations.json').open('x',encoding='utf-8') as stream:
    json.dump({'status':'PREPARED_NOT_EXECUTED','frozen_source_manifest':ref(SOURCE/'source_manifest.final.json'),
       'adaptations':rows,'projection_checker':ref(HERE/'terminal_projection_check.py'),
       'model_calls':0,'gpu_calls':0,'raw_inputs_read':False},stream,indent=2,ensure_ascii=False)
    stream.write('\n')
print(json.dumps({'prepared_files':len(rows),'formal_runs':0,'production_source_edits':0}))
