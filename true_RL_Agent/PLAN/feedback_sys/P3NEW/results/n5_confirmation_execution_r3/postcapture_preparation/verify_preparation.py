"""Static local preparation checks; never imports or runs capture/focus scripts."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
P3 = HERE.parents[2]
SOURCE = P3/'results/n5_confirmation_source_r3'

def ref(path):
    data = path.read_bytes()
    return {'file':str(path),'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}
def save(name,value):
    with (HERE/name).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False)
        stream.write('\n')

parsed = {}
for path in sorted(HERE.glob('*.py')):
    parsed[path.name] = ast.parse(path.read_text(encoding='utf-8'),filename=str(path))

source_manifest = json.loads((SOURCE/'source_manifest.final.json').read_bytes())
for name,expected in source_manifest.items():
    actual = ref(SOURCE/'source_final'/name)
    assert (actual['sha256'],actual['size_bytes']) == (expected['sha256'],expected['size_bytes']),name

adaptations = json.loads((HERE/'control_adaptations.json').read_bytes())
preserved = []
for row in adaptations['adaptations']:
    old = Path(row['original']['file'])
    new = Path(row['adapted']['file'])
    assert ref(old) == row['original'] and ref(new) == row['adapted']
    if new.name == 'sol_observer_control.py': continue
    before = ast.parse(old.read_text(encoding='utf-8'))
    old_functions = {n.name:ast.dump(n,include_attributes=False) for n in before.body if isinstance(n,ast.FunctionDef)}
    new_functions = {n.name:ast.dump(n,include_attributes=False) for n in parsed[new.name].body if isinstance(n,ast.FunctionDef)}
    for name in old_functions:
        if new.name == 'verify_input_arithmetic.py' and name == 'main': continue
        assert old_functions[name] == new_functions[name],(new.name,name)
    preserved.append({'script':new.name,'algorithm_functions_unchanged':True})

help_results = []
for name in ('postcapture_cpu.py','sol_observer_control.py','verify_input_arithmetic.py','bind_source_headers.py'):
    result = subprocess.run([sys.executable,'-B',str(HERE/name),'--help'],capture_output=True)
    help_results.append({'script':name,'exit_code':result.returncode,
                         'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),
                         'stderr':result.stderr.decode('utf-8','replace')})
    assert result.returncode == 0,(name,result.stderr)

save('preserve_old_inputs.json', {
    'scope':'Dependencies/provenance required by this preparation; not an authorization to delete other paths',
    'local_original_controls':[row['original'] for row in adaptations['adaptations']],
    'additional_local_preserve':[
        str(P3/'results/n5_confirmation_execution_r2/postcapture_commands.md'),
        str(P3/'results/n5_confirmation_execution_r2/verify_numeric_arithmetic.py'),
        str(P3/'results/n5_confirmation_execution_r2/source_link_audit_r1'),
        str(P3/'results/n5_confirmation_execution_r2/prepare_numeric_control.py'),
        str(P3/'results/n5_confirmation_execution_r2/prepare_media_control.py'),
        str(P3/'results/n5_confirmation_execution_r2/count_prepared_text.py'),
        str(P3/'results/semantic_input_development_r1/arithmetic_input_audit'),
    ],
    'remote_preserve':[
        '/data/yuyangliu/AGENT/feedback_p3new_20260929/text_author_smoke_r1',
        '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930/control/numeric_materials_r1/token_capacity_r1/deepseek_v4_tokenizer.zip',
    ],
    'preserve_prior_answers_receipts_and_source_freezes':True,
})
save('preparation_manifest.json', {'status':'PREPARED_NOT_EXECUTED',
    'frozen_source_manifest':ref(SOURCE/'source_manifest.final.json'),
    'control_files':{p.name:ref(p) for p in sorted(HERE.glob('*.py'))},
    'commands':ref(HERE/'postcapture_commands.md'),'adaptation_registration':ref(HERE/'control_adaptations.json'),
    'actual_card_sha256':None,'actual_raw_inputs':None,'actual_slot_identity':None})
save('static_verification.json', {'status':'PASS_STATIC_PREPARATION_ONLY', 'script':ref(Path(__file__)),
    'syntax_checked_files':sorted(parsed),'source_files_unchanged':len(source_manifest),
    'unchanged_algorithm_checks':preserved,'help_checks':help_results,
    'raw_capture_inputs_read':False,'model_calls':0,'gpu_calls':0,
    'production_or_formal_execution':False,'new_material_arithmetic_status':'NOT_EXECUTED'})
print(json.dumps({'status':'PASS_STATIC_PREPARATION_ONLY','syntax_files':len(parsed),
                  'source_files_unchanged':len(source_manifest),'algorithm_scripts_unchanged':len(preserved),
                  'help_checks':len(help_results),'raw_inputs_read':False}))
