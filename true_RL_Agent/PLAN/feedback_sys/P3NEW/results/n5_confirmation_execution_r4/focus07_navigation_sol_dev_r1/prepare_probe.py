"""Prepare one old focus-07 development CLI probe; this never executes a model."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parents[2]
REPO = P.parents[2]
sys.path.insert(0, str(P))
import confirmation_review_coverage as coverage
import prepare_confirmation_materials as packing
import run_confirmation_review as review
import run_sol_review as sol

def sha(raw): return hashlib.sha256(raw).hexdigest()
def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": sha(raw), "size_bytes": len(raw)}
def save(path, value):
    with path.open('xb') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode()+b'\n')

old_path = D.parent / 'focus_arithmetic_navigation_r1/capacity_inputs/focus-07.old.json'
old_request = json.loads(old_path.read_bytes())
old_materials, _ = coverage.request_materials(old_request)
assert old_request['messages'][0]['content'].startswith('You are independent judge gpt6.1-sol-a for packet focus-07, stage facts. ')
related_path = D.parent / 'focus_arithmetic_navigation_r1/rebuilt/focus-07.json'
related = json.loads(related_path.read_bytes())
old_related = old_materials['related-raw']['data']
preserved = deepcopy(related)
preserved.pop('arithmetic_navigation')
assert preserved == old_related
assert sha((D / 'materials/stair.task-definition.original.json').read_bytes()) == old_materials['task-definition-stair-8c5f2d68db3d']['sha256']
assert json.loads((D / 'materials/stair.task-definition.original.json').read_bytes()) == old_materials['task-definition-stair-8c5f2d68db3d']['data']

for name in ('N5_SOL_CLI_POLICY_R5.json', 'N5_REVIEW_ASSIGNMENT_R5.json', 'N5_FACT_REVIEW_PROTOCOL_ZH.md'):
    shutil.copyfile(P / name, D / name)
policy_path, assignment_path = D / 'N5_SOL_CLI_POLICY_R5.json', D / 'N5_REVIEW_ASSIGNMENT_R5.json'
assignment = review.assignment_fields(assignment_path, 'focus-07')
assert assignment['review_assignment']['reviewers']['A'] == {'judge_id': 'gpt6.1-sol-a', 'model': 'gpt-6.1-sol', 'carrier': 'fresh_cli'}
protocol = review.packet_protocol(D / 'N5_FACT_REVIEW_PROTOCOL_ZH.md')
save(D / 'materials/review_protocol.json', protocol)
save(D / 'materials/questions.json', old_materials['questions']['data'])
assert ref(D / 'materials/questions.json')['sha256'] == old_materials['questions']['sha256']
shutil.copyfile(related_path, D / 'materials/related.focus07.json')

paths = {'protocol': D / 'materials/review_protocol.json', 'questions': D / 'materials/questions.json',
    'related-raw': D / 'materials/related.focus07.json',
    'task-definition-stair-8c5f2d68db3d': D / 'materials/stair.task-definition.original.json'}
manifest_materials = [{"id": identity, "role": old_materials[identity]['role'],
    "file": str(path.relative_to(D)), "sha256": ref(path)['sha256'], "mime_type": "application/json"}
    for identity, path in paths.items()]
save(D / 'facts.manifest.json', {"schema_version": "p3new.confirmation_review_materials.v1", "packet_id": "focus-07",
    "stage": "facts", "purpose": "old_stair_focus_arithmetic_navigation_development_feasibility_only",
    "reference_contract": coverage.BLOCK_CONTRACT, "material_profile": "related_v2", "max_output_tokens": 32768,
    **assignment, "materials": manifest_materials})
review.prepare(D / 'facts.manifest.json', 'gpt6.1-sol-a', D / 'prepared')
cli = Path(shutil.which('codex'))
sol.run(D / 'prepared', D / 'carrier_preview', source_root=REPO, cli_policy=policy_path,
        assignment_file=assignment_path, cli=cli, execute=False)
request = json.loads((D / 'prepared/request.http_body.json').read_bytes())
new_materials, _ = coverage.request_materials(request)
assert set(new_materials) == set(old_materials)
for identity in ('questions', 'task-definition-stair-8c5f2d68db3d'):
    assert new_materials[identity] == old_materials[identity]
assert new_materials['related-raw']['data'] == related
assert sol.adapter.reconstruct_messages(D / 'carrier_preview/blind/input.json') == request['messages']
runtime_home = Path('C:/Users/lover/AppData/Local/Temp/p3new-focus07-navigation-sol-dev-r1-20261001')
assert not runtime_home.exists()
argv, overrides = sol.cli_argv(cli, D / 'execution', D / 'execution/blind', json.loads(policy_path.read_bytes()))
_, removed_names = sol.child_environment(runtime_home)
launch = [sys.executable, '-B', str(P / 'run_sol_review.py'), '--prepared', str(D / 'prepared'),
    '--output', str(D / 'execution'), '--source-root', str(REPO), '--cli-policy', str(policy_path),
    '--review-assignment', str(assignment_path), '--cli', str(cli), '--auth-source-home', 'C:/Users/lover/.codex',
    '--runtime-home', str(runtime_home), '--execute-one-call']
save(D / 'request_diff.json', {"old_request": ref(old_path), "new_request": ref(D / 'prepared/request.http_body.json'),
    "old_material_identities": {k: {"role": v['role'], "sha256": v['sha256']} for k, v in old_materials.items()},
    "new_material_identities": {k: {"role": v['role'], "sha256": v['sha256']} for k, v in new_materials.items()},
    "changes": ["Current generic string-ID system/packet protocol", "related-raw adds only the verified neutral arithmetic_navigation; original values and definitions unchanged", "Source/catalog IDs and hashes change accordingly; original five questions and exact task-definition bytes unchanged"],
    "no_targeted_omission_prompt_or_manual_answer": True, "complete_original_related_values_preserved": True,
    "automatic_report_or_other_answers_supplied": False, "material_count": len(new_materials), "image_count": 0})
save(D / 'registration.draft.json', {"status": "PREPARED_NOT_DISPATCHED_CAPACITY_PENDING", "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "one_old_focus07_navigation_gpt61sol_development_feasibility_only", "judge_id": "gpt6.1-sol-a", "packet_id": "focus-07", "stage": "facts",
    "calls_authorized_here": 0, "max_future_calls": 1, "application_retries": 0, "resume": False, "ephemeral": False,
    "model": "gpt-6.1-sol", "reasoning_effort": "high", "assignment": ref(assignment_path), "cli_policy": ref(policy_path),
    "source_root": str(REPO), "source": {name: ref(P / name) for name in ('run_sol_review.py', 'run_confirmation_review.py', 'prepare_desktop_review.py', 'confirmation_review_coverage.py', 'prepare_confirmation_materials.py', 'N5_FACT_REVIEW_PROTOCOL_ZH.md')},
    "prepared": ref(D / 'prepared/prepare.json'), "request": ref(D / 'prepared/request.http_body.json'),
    "request_diff": ref(D / 'request_diff.json'), "transport_preview": ref(D / 'carrier_preview/transport_receipt.json'),
    "stdin_preview": ref(D / 'carrier_preview/stdin.txt'), "cli": ref(cli), "future_cli_argv": argv, "process_config": overrides,
    "launch_after_root_approval_only": launch, "runtime_home": str(runtime_home), "runtime_home_created": False,
    "auth_source_home": "C:/Users/lover/.codex", "credential_contents_read_by_preparation": False,
    "auth_handling_if_executed": "copy existing auth.json into new empty runtime home; unlink copied credential in finally; never export contents",
    "removed_inherited_environment_names_preview": removed_names, "allowed_model_tools": [],
    "forbidden_model_reads": ["input.json reread", "repository", "results/control/registration", "other answers or reports", "old contexts/memory/skills/MCP/web", "auth source/runtime home"],
    "access_boundary": "model receives full stdin once; this focus has no images and permits no tool calls. Whitelist/trace audit, not OS filesystem isolation.",
    "actual_model_usage_compaction_or_tool_trace": None, "actual_usd_cost": None, "cost_basis": "SOL CLI cost/invoice unavailable; no dollar estimate or free-call claim. No DeepSeek/provider or author call and no provider ledger mutation.",
    "formal_n5_score_or_seal": None, "limitations": ["Not an original-versus-new or cross-model causal comparison", "Does not restore stopped R4 slots or count toward formal N5", "No promise all five questions will be completed", "Hidden provider transport retries unobserved; only application retries fixed at zero", "512000/450000 requested per process, not observed runtime usage" ]})
print(json.dumps({"status": "PREPARED_NO_CALL", "request": ref(D / 'prepared/request.http_body.json'), "stdin": ref(D / 'carrier_preview/stdin.txt')}))
