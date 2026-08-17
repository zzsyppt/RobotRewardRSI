"""Replay-only multi-round AgentCore loop.

Production live authoring goes through ``live_loop.run_live_loop`` and
``CandidateAuthoringLoop``. This legacy loop remains only for replay-style
tests and in-memory candidates that already exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from true_rl_agent.agent_core.ledger import (
    NON_PROVIDER_AUTHORING_MODE,
    load_ledger,
)
from true_rl_agent.agent_core.provider import CandidateProvider
from true_rl_agent.agent_core.round import RoundInputs, RoundResult, run_candidate_round
from true_rl_agent.agent_core.session_genesis import (
    VerifiedSessionGenesis,
    verify_session_genesis_for_runtime,
)
from true_rl_agent.backends.base import BackendAdapter
from true_rl_agent.candidates.verifier import CandidateVerifier
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
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
    candidate_admission_policy: CandidateVerifier
    provider: CandidateProvider
    policy_initialization: PolicyInitialization
    ledger_path: Path
    session_genesis: VerifiedSessionGenesis
    policy_accept_delta: float


class AgentLoop:
    def __init__(self, inputs: LoopInputs) -> None:
        _reject_authoring_capable_provider(inputs.provider)
        self.inputs = inputs

    def run(self) -> list[RoundResult]:
        results: list[RoundResult] = []
        acceptance_protocol = self.inputs.task_plugin.task_spec.acceptance_protocol
        if acceptance_protocol is None:
            raise ValueError("task does not define a fixed acceptance protocol")
        if not math.isclose(
            self.inputs.policy_accept_delta,
            acceptance_protocol.min_delta,
        ):
            raise ValueError(
                "policy_accept_delta cannot override the task acceptance protocol"
            )
        verify_session_genesis_for_runtime(
            self.inputs.session_genesis,
            output_dir=self.inputs.ledger_path.parent,
            session_id=self.inputs.session_id,
            task_plugin=self.inputs.task_plugin,
            backend_adapter=self.inputs.backend_adapter,
            policy_initialization=self.inputs.policy_initialization,
        )
        initial_current_best = self.inputs.session_genesis.initial_current_best
        session_genesis_hash = self.inputs.session_genesis.session_genesis_hash
        backend_execution_environment_manifest_hash = (
            self.inputs.session_genesis.backend_execution_environment_manifest_hash
        )
        evaluation_binding = self.inputs.session_genesis.evaluation_binding
        context_builder = PublicContextBuilder(
            task_prompt_adapter=self.inputs.task_plugin.prompt_adapter(),
            acceptance_protocol=acceptance_protocol,
            initial_current_best=initial_current_best,
            session_genesis_hash=session_genesis_hash,
            candidate_capability_profile=(
                self.inputs.task_plugin.task_spec.candidate_capability_profile
            ),
            initial_reference_program_source=(
                self.inputs.session_genesis.reference_program_source()
            ),
            ledger_authoring_mode=NON_PROVIDER_AUTHORING_MODE,
        )
        for index in range(self.inputs.round_count):
            ledger = load_ledger(self.inputs.ledger_path)
            replay = replay_acceptance_ledger(
                ledger,
                initial_current_best=initial_current_best,
                expected_session_genesis_hash=session_genesis_hash,
                acceptance_protocol=acceptance_protocol,
            )
            replay.raise_for_continuation()
            current_best_eval_stats = replay.current_best
            records = [ExperimentRecord.from_ledger_record(record) for record in ledger]
            prompt_build = context_builder.build_prompt(records)
            prompt_payload = prompt_build.provider_payload
            candidate = self.inputs.provider.next_candidate(prompt_payload=prompt_payload)
            round_id = f"{self.inputs.session_id}-round-{index + 1:03d}"
            result = run_candidate_round(
                RoundInputs(
                    round_id=round_id,
                    task_plugin=self.inputs.task_plugin,
                    backend_adapter=self.inputs.backend_adapter,
                    runner=self.inputs.runner,
                    candidate_admission_policy=(
                        self.inputs.candidate_admission_policy
                    ),
                    candidate=candidate,
                    policy_initialization=self.inputs.policy_initialization,
                    ledger_path=self.inputs.ledger_path,
                    public_context=prompt_build.round_public_context,
                    current_best_eval_stats=current_best_eval_stats,
                    session_genesis_hash=session_genesis_hash,
                    evaluation_binding=evaluation_binding,
                    authoring_mode=NON_PROVIDER_AUTHORING_MODE,
                    backend_execution_environment_manifest_hash=(
                        backend_execution_environment_manifest_hash
                    ),
                    parent_program_source_checksum=(
                        prompt_build.parent_program_source_checksum
                    ),
                )
            )
            results.append(result)
        final_replay = replay_acceptance_ledger(
            load_ledger(self.inputs.ledger_path),
            initial_current_best=initial_current_best,
            expected_session_genesis_hash=session_genesis_hash,
            acceptance_protocol=acceptance_protocol,
        )
        final_replay.raise_for_continuation()
        return results


def _reject_authoring_capable_provider(provider: CandidateProvider) -> None:
    method = getattr(provider, "next_authoring_response", None)
    if callable(method):
        raise TypeError(
            "AgentLoop is replay-only and must not be used with authoring-capable providers; "
            "use live_loop.run_live_loop for provider authoring, source follow-up, static repair, "
            "component smoke, and anti-degenerate checks"
        )
