"""Read-only A02 ledger replay; writes only the requested audit output."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument("--project-root", type=Path, required=True)
p.add_argument("--session-dir", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--prompt-adapter-source", type=Path)
a = p.parse_args()
sys.path.insert(0, str(a.project_root / "src"))

from true_rl_agent.agent_core.ledger import load_ledger, PROVIDER_AUTHORING_MODE, primary_ledger_epoch_hash
from true_rl_agent.agent_core.session_genesis import SessionGenesisArtifact
from true_rl_agent.agent_core.generation_contract import GenerationSessionContract
from true_rl_agent.evaluation.acceptance_replay import project_public_history
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.memory import PublicContextBuilder
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.memory import PushTaskMemoryAdapter
from true_rl_agent.tasks.push_recovery.phased_adapter import PushPhasedEvaluationAdapter

if a.prompt_adapter_source:
    name = "true_rl_agent.tasks.push_recovery.prompt_adapter"
    spec = importlib.util.spec_from_file_location(name, a.prompt_adapter_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from true_rl_agent.tasks.push_recovery.prompt_adapter import PushRecoveryPromptAdapter
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin

def read(name):
    return json.loads((a.session_dir / name).read_text(encoding="utf-8"))

raw_genesis = read("session_genesis.json")
g = SessionGenesisArtifact.from_mapping(raw_genesis)
anchor = raw_genesis["anchor"]
manifest = anchor["reference_manifest"]
config = PushRecoveryTaskConfig.for_cell(read("resolved_cell_manifest.json")["cell_variant_id"])
protocol = config.acceptance_protocol()
initial = CurrentBestEvalStats.from_reference_public_result_mapping(
    reference_id=manifest["reference_id"], reference_manifest_hash=manifest["manifest_hash"],
    program_source_checksum=manifest["source_checksum"],
    training_binding_hash=anchor["training_binding"]["binding_hash"],
    capability_profile_hash=manifest["capability_profile_hash"], checkpoint_hash=anchor["checkpoint_hash"],
    public_result=anchor["public_result"], acceptance_protocol=protocol,
    expected_evaluation_binding=g.evaluation_binding)
adapter = PushPhasedEvaluationAdapter(config, g.evaluation_binding.task_protocol_hash)
ledger = load_ledger(a.session_dir / "ledger.jsonl")
kwargs = dict(initial_current_best=initial, expected_session_genesis_hash=g.session_genesis_hash,
    acceptance_protocol=protocol, expected_initial_policy_state_hash=anchor["initial_policy_state_hash"],
    expected_authoring_mode=PROVIDER_AUTHORING_MODE,
    expected_ledger_epoch_hash=primary_ledger_epoch_hash(g.session_genesis_hash),
    phased_evaluation_adapter=adapter)
history = project_public_history(ledger, **kwargs)
result = {"scope": "Read-only receipt-bound replay; no simulation or session mutation.",
    "records": history.records, "initial_current_best": initial.to_public_dict(),
    "session_genesis_hash": g.session_genesis_hash, "acceptance_protocol_hash": protocol.protocol_hash,
    "current_best": history.current_best.to_public_dict(), "public_score_source": history.public_score_source}
builder = PublicContextBuilder(task_prompt_adapter=PushRecoveryPromptAdapter(config),
    task_memory_adapter=PushTaskMemoryAdapter(config=config), acceptance_protocol=protocol,
    initial_current_best=initial, session_genesis_hash=g.session_genesis_hash,
    candidate_capability_profile=PushRecoveryTaskPlugin(config).task_spec.candidate_capability_profile,
    initial_reference_program_source=read("reference_eval_artifact.json")["program_source"],
    initial_anchor_public_evidence={k: anchor[k] for k in ("public_result", "public_result_hash")}
        | {"anchor_kind": "reference_program"},
    expected_initial_policy_state_hash=anchor["initial_policy_state_hash"],
    ledger_authoring_mode=PROVIDER_AUTHORING_MODE, expected_ledger_epoch_hash=kwargs["expected_ledger_epoch_hash"],
    generation_session_contract=GenerationSessionContract.from_mapping(read("generation_session_contract.json")),
    phased_evaluation_adapter=adapter)
try:
    built = builder.build_prompt(ledger)
    result.update(prompt_status="PASS", provider_payload=built.provider_payload,
                  round_public_context=built.round_public_context)
except Exception as exc:
    result.update(prompt_status="FAIL", prompt_error_type=type(exc).__name__, prompt_error=str(exc))
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"history_records": len(history.records), "prompt_status": result["prompt_status"],
                  "prompt_error": result.get("prompt_error")}, ensure_ascii=False))
