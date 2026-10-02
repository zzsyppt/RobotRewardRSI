"""Local preparation only: copy existing CPU algorithms, record exact adaptations."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
P3 = HERE.parents[2]
E2 = P3 / 'results/n5_confirmation_execution_r2'
DEV = P3 / 'results/semantic_input_development_r1/arithmetic_input_audit'
SOURCE = P3 / 'results/n5_confirmation_source_r3'
rows = []

def ref(path):
    data = path.read_bytes()
    return {'file':str(path), 'sha256':hashlib.sha256(data).hexdigest(), 'size_bytes':len(data)}

def copy(source, name, changes):
    original = source.read_text(encoding='utf-8')
    changed = original
    applied = []
    for old,new in changes:
        count = changed.count(old)
        if not count: raise ValueError('Expected source text missing: '+name+' '+old)
        changed = changed.replace(old,new)
        applied.append({'old':old,'new':new,'occurrences':count})
    ast.parse(changed,filename=name)
    target = HERE/name
    with target.open('x',encoding='utf-8',newline='\n') as stream: stream.write(changed)
    rows.append({'original':ref(source),'adapted':ref(target),'changes':applied,
                 'execution':'NOT_RUN_ON_R3_INPUTS','production_source_modified':False})

roots = [('n5_confirmation_20260930_r2','n5_confirmation_20260930_r3'),
         ('n5_confirmation_source_r2','n5_confirmation_source_r3_final')]
controls = roots + [('n5_execution_20260930_r2','n5_execution_20260930_r3')]
for old,new,changes in [
    ('focus_deterministic_audit_r1/verify_scope_residual_coverage.py','verify_scope_residual_coverage.py',roots),
    ('focus_representative_audit_r1/verify_representatives.py','verify_representatives.py',roots),
    ('focus_operand_audit_r1/stair/verify_stair_focus_operands.py','verify_stair_focus_operands.py',roots),
    ('focus_operand_audit_r1/push/verify_push_focus_operands_v2.py','verify_push_focus_operands_v2.py',controls),
]: copy(E2/old,new,changes)

copy(DEV/'verify_input_arithmetic.py','verify_input_arithmetic.py',[
    ('glob("media-*/related_numeric.json")','glob("media-*/raw/related_numeric.json")'),
    ('args.original_capture / "media_materials_control_corrected" / path.parent.name / "raw/episodes"',
     'args.original_capture / "media_materials" / path.parent.parent.name / "raw/episodes"'),
    ('audit_packet(path.parent.name, path, folders[0], geometry_classes)',
     'audit_packet(path.parent.parent.name, path, folders[0], geometry_classes)'),
])
copy(E2/'sol_observer_control.py','sol_observer_control.py', controls + [
    ("CARD_SHA = '6bea98ed63e1f1e516631e58205e3da93406311d1b87466e445fe754b0c22794'", 'CARD_SHA = None'),
    ('497d14b6a133d1ba3afa97e15c39b00515479692e528aac648970851245f3b76',
     '5cb1b36c5ca2e7af40d62c51e7cf1fe9aae4790e080353b8947cc6e14e5902d5'),
    ('gpt-6-sol','gpt-6.1-sol'),
    ("    args = parser.parse_args()", "    parser.add_argument('--card-sha256', required=True, help='Actual R3 frozen card SHA')\n    args = parser.parse_args()\n    CARD_SHA = args.card_sha256\n    require(len(CARD_SHA) == 64 and all(c in '0123456789abcdef' for c in CARD_SHA), 'Invalid card SHA')"),
    ('R2','R3'),
])

manifest = json.loads((SOURCE/'source_manifest.final.json').read_bytes())
required = [
    'PLAN/feedback_sys/P3NEW/'+name for name in (
    'extract_dual_view_evidence.py','prepare_n5_media_selection.py','prepare_native_observer.py',
    'export_confirmation_raw.py','prepare_review_task_contract.py','prepare_confirmation_materials.py',
    'prepare_confirmation_media.py','run_confirmation_review.py','confirmation_review_coverage.py',
    'verify_stair_confirmation_arithmetic.py','verify_push_confirmation_arithmetic.py',
    'recompute_residuals_from_raw.py','N5_REVIEW_ASSIGNMENT_R3.json')]
required += ['PLAN/feedback_sys/P3R/confirmation_raw_material.py',
             'PLAN/feedback_sys/P3NEW/results/n5_confirmation_execution_r1/assemble_delivery.py']
checks = []
for relative in required:
    path = SOURCE/'source_final'/relative
    actual = ref(path)
    expected = manifest[relative]
    if (actual['sha256'],actual['size_bytes']) != (expected['sha256'],expected['size_bytes']):
        raise ValueError('Frozen source differs: '+relative)
    checks.append({'relative':relative,**actual,'manifest_match':True})
with (HERE/'control_adaptations.json').open('x',encoding='utf-8') as stream:
    json.dump({'status':'PREPARED_NOT_EXECUTED','production_source_manifest':ref(SOURCE/'source_manifest.final.json'),
      'frozen_entrypoints':checks,'adaptations':rows,'model_calls':0,'gpu_calls':0,
      'raw_capture_inputs_read':False,'card_sha256':None,'packet_members_and_hashes':None},stream,indent=2,ensure_ascii=False)
    stream.write('\n')
print(json.dumps({'adapted_files':len(rows),'frozen_entrypoints_verified':len(checks),'actual_raw_runs':0}))
