"""Experiment-only check of reuse cost projection through the real consumer."""
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
from tests.unit.test_completed_reference_training import CompletedReferenceTrainingTests, _EnvironmentBackend
from true_rl_agent.agent_core.session_genesis import prepare_session_genesis
from true_rl_agent.agent_core.generation_job_evidence import validated_job_cost_artifacts
from true_rl_agent.agent_core.generation_accounting import collect_actual_cost_usage
from true_rl_agent.runner.job_attempts import JobAttemptJournal

class ProductionPhaseNamesBackend(_EnvironmentBackend):
    def materialize_eval_phase(self, **kwargs):
        spec = super().materialize_eval_phase(**kwargs)
        return replace(spec, job_id=f"{spec.evaluation_plan.job_id}-{spec.phase.value.replace('_', '-')}")

with TemporaryDirectory() as directory:
    fixture = CompletedReferenceTrainingTests()
    source, spec, result, runner, kwargs = fixture.source_fixture(Path(directory))
    kwargs["backend_adapter"] = ProductionPhaseNamesBackend("b" * 64)
    runner.phase_calls.clear()
    with patch.object(runner, "recover_completed_train", create=True, return_value=result), patch.object(runner, "run_train", side_effect=AssertionError("unexpected train")):
        created = prepare_session_genesis(**kwargs)
        loaded = prepare_session_genesis(**kwargs, require_existing=True)
    scopes = kwargs["reference_eval_phase_attempt_scopes"]
    scientific = SimpleNamespace(train_steps_per_job=spec.budget_steps, seed_units_for_phase=lambda phase: 1, max_eval_control_steps_per_seed_unit=100, reference_eval_seed_units=3)
    foundation = SimpleNamespace(
        config=SimpleNamespace(output_dir=kwargs["output_dir"], session_id=kwargs["session_id"]),
        cost_contract=SimpleNamespace(scientific_cost_basis=scientific, contract_hash="a"*64, uses_operation_reserves=False, to_dict=lambda: {"provider": {}}),
        session_contract=SimpleNamespace(contract_hash=next(iter(scopes.values())).session_contract_hash, max_generations=1),
    )
    train_cost, eval_cost, evidence_hash = validated_job_cost_artifacts(foundation, genesis=loaded)
    for phase, scope in scopes.items():
        job_id = next(job_id for job_id in eval_cost if job_id.endswith(phase.value.replace("_","-")))
        journal = JobAttemptJournal(scope=scope, logical_job_id=job_id, job_kind="eval_phase", job_input_hash="1"*64, environment_hash="b"*64, execution_device="cuda:0", resource_binding_hash=scope.resource_binding_hash)
        reservation = journal.reserve(0, previous_result=None)
        journal.finish(reservation, outcome="completed", failure_kind=None, retryable=False, output_hash="2"*64, returncode=0, resource_lease_wall_duration_ns=1000)
    usage, conservative, open_reservations = collect_actual_cost_usage(foundation, genesis=loaded, ledger=[])
    assert usage.train_jobs == 0 and usage.train_steps == 0
    assert usage.job_attempts == 3 and usage.eval_seed_units == 3 and usage.simulated_control_steps == 30
    assert usage.resource_lease_wall_duration_ns == 3000
    assert not conservative and not open_reservations
    assert created.session_genesis_hash == loaded.session_genesis_hash
    assert len(runner.phase_calls) == 3
    assert not (kwargs["output_dir"] / "ledger.jsonl").exists()
    print(json.dumps({"status":"PASS","usage":asdict(usage),"fresh_phases":[p.value for p in runner.phase_calls],"train_cost_keys":list(train_cost),"eval_cost_keys":list(eval_cost),"source_train_job_id":spec.job_id,"limitations":["Reference training and GPU evaluations use existing unit fixture objects; no physical run claimed.","Actual cost consumer and on-disk attempt journals are production functions; operation-allocation v2 layer is reviewed statically and awaits target runtime."]},ensure_ascii=False,indent=2))
