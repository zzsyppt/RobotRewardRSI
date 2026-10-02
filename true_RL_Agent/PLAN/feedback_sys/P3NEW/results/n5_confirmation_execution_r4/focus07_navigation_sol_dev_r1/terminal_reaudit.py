"""Collect this one probe's public terminal evidence; never export reasoning."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parents[2]
sys.path.insert(0, str(P))
import confirmation_review_coverage as coverage
import run_sol_review as sol

def read(path): return json.loads(Path(path).read_bytes())
def ref(path):
    path = Path(path).resolve()
    # Streaming hash binds the private original without exposing its contents.
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}
def save(path, value):
    with Path(path).open('xb') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode()+b'\n')

registration = read(D / 'registration.draft.json')
assert ref(D / 'registration.draft.json')['sha256'] == '2d8283aaaa185efb5ee39856d4601ae486761a73aaeac622dfb27e8bc97c94aa'
receipt = read(D / 'execution/receipt.json')
actual_registration = read(D / 'execution/registration.json')
trace = read(D / 'execution/access_trace.json')
policy = read(D / 'N5_SOL_CLI_POLICY_R5.json')
runtime_home = Path(registration['runtime_home'])
rollouts = list((runtime_home / 'sessions').rglob('*.jsonl'))
assert len(rollouts) == 1
rollout_ref = ref(rollouts[0])
assert not (runtime_home / 'auth.json').exists()
assert ref(D / 'execution/stdin.txt')['sha256'] == registration['stdin_preview']['sha256']
assert ref(D / 'execution/executed_driver.py')['sha256'] == registration['source']['run_sol_review.py']['sha256']
assert ref(D / 'execution/blind/input.json')['sha256'] == read(D / 'capacity_receipt.json')['blind_input']['sha256']
request = read(D / 'prepared/request.http_body.json')
assert sol.adapter.reconstruct_messages(D / 'execution/blind/input.json') == request['messages']
windows = sorted({v['model_context_window'] for v in trace['usage'] if v.get('model_context_window') is not None})
summary = {'scope': 'one_old_focus07_navigation_gpt61sol_nonblind_development_check_only',
    'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'root_release': ref(D / 'root_release.json'),
    'registration_draft': ref(D / 'registration.draft.json'), 'original_execution_registration': ref(D / 'execution/registration.json'),
    'executed_driver': ref(D / 'execution/executed_driver.py'), 'original_transport_receipt': ref(D / 'execution/receipt.json'),
    'access_trace': ref(D / 'execution/access_trace.json'), 'private_rollout_original_hash_only': rollout_ref,
    'private_stdout_hash_only': receipt.get('private_stdout'), 'private_stderr_hash_only': receipt.get('private_stderr'),
    'source_bindings': registration['source'], 'cli_policy': registration['cli_policy'], 'assignment': registration['assignment'],
    'request': ref(D / 'prepared/request.http_body.json'), 'raw_answer': ref(D / 'execution/raw_answer.json'),
    'exit_code': receipt.get('exit_code'), 'transport_status': receipt['status'],
    'actual_models': receipt.get('actual_models'), 'actual_contexts': trace['contexts'],
    'usage_from_turn_completed': receipt.get('usage'), 'context_window_values_from_token_count': windows,
    'compaction_events': trace['compactions'], 'tool_calls': trace['tools'], 'tool_call_count': len(trace['tools']),
    'access_verified': trace['access_verified'], 'stdin_matches_registered_preview': True, 'messages_reconstructed_exactly': True,
    'one_new_session': len(trace['session_ids']) == 1 and trace['session_ids'] == receipt.get('thread_ids'),
    'copied_auth_removed': True, 'removed_environment_names': actual_registration['removed_inherited_environment_names'],
    'application_retries': 0, 'hidden_transport_retries': 'unobserved; no zero-retry claim',
    'actual_usd_cost': None, 'cost_basis': 'CLI invoice/cost not available; usage is not a dollar bill. No DeepSeek or author dispatch or ledger debit.',
    'private_reasoning_or_base_instructions_exported': False, 'formal_score_or_seal': None}
try:
    answer = read(D / 'execution/raw_answer.json')
    spec = importlib.util.spec_from_file_location('existing_facts_structure_audit', P / 'audit_confirmation_facts.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    # Use the current shared checker with this actual prepared helper/protocol.
    # Never mutate the historical R4 pinned checker or any original receipt.
    audit.HELPER = P / 'confirmation_review_coverage.py'
    audit.PROTOCOL = D / 'N5_FACT_REVIEW_PROTOCOL_ZH.md'
    structural = audit.audit_contract(D / 'prepared/prepare.json', D / 'execution/raw_answer.json')
    structural['checker_function_source'] = ref(P / 'audit_confirmation_facts.py')
    structural['runtime_binding_note'] = 'Current audit_contract function explicitly bound to this prepared helper and protocol; no historical R4 checker or receipt changed.'
    save(D / 'facts_contract_audit.json', structural)
    facts = answer.get('findings', {}).get('necessary_facts', [])
    refs = [reference for fact in facts if isinstance(fact, dict) for reference in fact.get('evidence_refs', [])]
    materials, _ = coverage.request_materials(request)
    catalog = coverage.request_reference_catalog(request)
    identities = {row['anchor_id'] for row in catalog['items']}
    summary.update(answer_declared_complete=answer.get('complete'), necessary_fact_count=len(facts),
        structural_errors=len(structural['errors']), structural_audit=ref(D / 'facts_contract_audit.json'),
        reference_count=len(refs), string_reference_count=sum(isinstance(v, str) for v in refs),
        every_reference_exact_catalog_string=bool(refs) and all(isinstance(v, str) and v in identities for v in refs),
        frozen_reference_audit_complete=structural['frozen_reference_audit']['review_complete'], semantic_support_proven=False)
except (ValueError, TypeError, KeyError) as error:
    summary['answer_mechanical_audit_error'] = type(error).__name__ + ': ' + str(error)
    summary['semantic_support_proven'] = False
save(D / 'terminal_evidence.json', summary)
print(json.dumps({key: summary.get(key) for key in ('transport_status', 'exit_code', 'actual_models',
    'usage_from_turn_completed', 'context_window_values_from_token_count', 'tool_call_count',
    'answer_declared_complete', 'necessary_fact_count', 'structural_errors', 'reference_count',
    'every_reference_exact_catalog_string', 'answer_mechanical_audit_error')}))
