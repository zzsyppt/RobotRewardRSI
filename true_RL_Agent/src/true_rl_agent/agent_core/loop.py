"""Multi-round AgentCore loop skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.provider import CandidateProvider
from true_rl_agent.agent_core.round import RoundInputs, RoundResult, run_candidate_round
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.verifier import CandidateVerifier
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import Runner
from true_rl_agent.tasks.base import TaskPlugin


@dataclass(frozen=True)
class LoopInputs:
    session_id: str
    round_count: int
    task_plugin: TaskPlugin
    backend_adapter: BackendAdapter
    runner: Runner
    verifier: CandidateVerifier
    provider: CandidateProvider
    policy_initialization: PolicyInitialization
    ledger_path: Path
    initial_best_score: float
    policy_accept_delta: float


class AgentLoop:
    def __init__(self, inputs: LoopInputs) -> None:
        self.inputs = inputs

    def run(self) -> list[RoundResult]:
        results: list[RoundResult] = []
        current_best_score = self.inputs.initial_best_score
        context_builder = PublicContextBuilder(task_prompt_adapter=self.inputs.task_plugin.prompt_adapter())
        for index in range(self.inputs.round_count):
            ledger = load_ledger(self.inputs.ledger_path)
            records = [ExperimentRecord.from_ledger_record(record) for record in ledger]
            prompt_payload = context_builder.build_prompt_payload(records)
            public_context = prompt_payload["public_context"]
            candidate = self.inputs.provider.next_candidate(prompt_payload=prompt_payload)
            round_id = f"{self.inputs.session_id}-round-{index + 1:03d}"
            result = run_candidate_round(
                RoundInputs(
                    round_id=round_id,
                    task_plugin=self.inputs.task_plugin,
                    backend_adapter=self.inputs.backend_adapter,
                    runner=self.inputs.runner,
                    verifier=self.inputs.verifier,
                    candidate=candidate,
                    policy_initialization=self.inputs.policy_initialization,
                    ledger_path=self.inputs.ledger_path,
                    public_context=public_context,
                    current_best_score=current_best_score,
                    policy_accept_delta=self.inputs.policy_accept_delta,
                )
            )
            results.append(result)
            if result.decision.accepted and result.decision.public_score is not None:
                current_best_score = max(current_best_score, result.decision.public_score)
        return results
