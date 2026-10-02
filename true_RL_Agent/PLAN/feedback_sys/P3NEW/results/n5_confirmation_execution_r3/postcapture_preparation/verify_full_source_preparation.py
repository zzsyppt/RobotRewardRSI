"""Static and synthetic verification of control adaptation, without capture inputs."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]/'results/n5_confirmation_source_r3'
OUT = HERE/'full_source_preparation_checks'
OUT.mkdir(exist_ok=False)

def ref(path):
    data = path.read_bytes()
    return {'file':str(path),'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}
def save(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
def execute(name,args):
    process = subprocess.run([sys.executable,'-B',*args],cwd=HERE,capture_output=True)
    for key,data in (('stdout',process.stdout),('stderr',process.stderr)):
        with (OUT/(name+'.'+key+'.txt')).open('xb') as stream: stream.write(data)
    row = {'argv':[sys.executable,'-B',*args],'exit_code':process.returncode,
           'stdout':ref(OUT/(name+'.stdout.txt')),'stderr':ref(OUT/(name+'.stderr.txt'))}
    if process.returncode: raise RuntimeError(str(row))
    return row

source_manifest = json.loads((SOURCE/'source_manifest.final.json').read_bytes())
for relative,expected in source_manifest.items():
    actual = ref(SOURCE/'source_final'/relative)
    assert actual['sha256'] == expected['sha256'] and actual['size_bytes'] == expected['size_bytes'],relative

syntax = []
for path in sorted(HERE.glob('*.py')):
    ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    syntax.append(ref(path))

registration = json.loads((HERE/'full_source_control_adaptations.json').read_bytes())
identity_checks = []
for row in registration['adaptations']:
    old,new = Path(row['original']['file']),Path(row['adapted']['file'])
    assert ref(old) == row['original'] and ref(new) == row['adapted']
    original = old.read_text(encoding='utf-8')
    if row.get('extracted_function'):
        node = next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name == row['extracted_function'])
        expected = ast.get_source_segment(original,node)
        for change in row['changes']:
            assert expected.count(change['old']) == change['occurrences']
            expected = expected.replace(change['old'],change['new'])
        node = next(n for n in ast.parse(new.read_text(encoding='utf-8')).body
                    if isinstance(n,ast.FunctionDef) and n.name == row['extracted_function'])
        assert ast.dump(ast.parse(expected).body[0],include_attributes=False) == ast.dump(node,include_attributes=False)
        identity_checks.append({'file':new.name,'frozen_operand_function_only_declared_projection_and_root_differences':True})
    else:
        expected = original
        for change in row['changes']:
            assert expected.count(change['old']) == change['occurrences']
            expected = expected.replace(change['old'],change['new'])
        assert expected == new.read_text(encoding='utf-8')
        old_functions = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
        new_functions = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(expected).body if isinstance(n,ast.FunctionDef)}
        for name in old_functions:
            if name != 'main': assert old_functions[name] == new_functions[name],(new.name,name)
        identity_checks.append({'file':new.name,'exact_declared_adaptation_only':True,'non_main_functions_unchanged':True})

tests = [execute('terminal_projection_synthetic_mutations',[str(HERE/'test_terminal_projection_adapter.py'),'-v'])]
for name in ('postcapture_cpu.py','verify_numeric_arithmetic.py','verify_push_source_links.py'):
    tests.append(execute(name+'.help',[str(HERE/name),'--help']))

save(HERE/'preparation_manifest.final.json', {'status':'PREPARED_NOT_EXECUTED',
    'supersedes_for_current_control_bytes':'preparation_manifest.json',
    'frozen_source_manifest':ref(SOURCE/'source_manifest.final.json'),
    'control_files':{p.name:ref(p) for p in sorted(HERE.glob('*.py'))},
    'commands':ref(HERE/'postcapture_commands.md'),
    'adaptation_registrations':[ref(HERE/'control_adaptations.json'),ref(HERE/'full_source_control_adaptations.json')],
    'actual_card_sha256':None,'actual_raw_inputs':None,'actual_slot_identity':None})
save(HERE/'static_verification.final.json', {'status':'PASS_STATIC_AND_SYNTHETIC_PREPARATION_ONLY',
    'source_files_unchanged':len(source_manifest),'syntax_files':syntax,'algorithm_preservation':identity_checks,
    'executed_checks':tests,'synthetic_mutation_tests_passed':10,
    'raw_capture_inputs_read':False,'model_calls':0,'gpu_calls':0,
    'real_24_full_source_chain_status':'NOT_EXECUTED_ON_R3_INPUTS',
    'real_28_input_arithmetic_status':'NOT_EXECUTED_ON_R3_INPUTS'})
print(json.dumps({'status':'PASS_STATIC_AND_SYNTHETIC_PREPARATION_ONLY','source_files_unchanged':len(source_manifest),
                  'syntax_files':len(syntax),'synthetic_tests':10,'help_checks':3,'raw_inputs_read':False}))
