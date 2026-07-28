from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.evaluation import make_verified_candidate_artifact
from tests.fakes.fake_components import (
    FAKE_ACCEPTANCE_PROTOCOL,
    FAKE_CAPABILITY_PROFILE,
    FakeBackendAdapter,
    FakeRunner,
)
from true_rl_agent.agent_core.session_genesis import (
    REFERENCE_EVAL_ARTIFACT_FILENAME,
    REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
    REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
    SESSION_GENESIS_FILENAME,
    ReferenceProgramGenesisAnchor,
    SessionGenesisArtifact,
    prepare_session_genesis,
)
from true_rl_agent.agent_core.source_request_limits import CURRENT_BEST_SOURCE_ID
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.decision_record_contract import validate_decision_record
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.memory.records import PrivateArtifactRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.prompting.source_catalog import build_prompt_source_context
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.safety.eval_seal import build_trusted_eval_seal
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.training_binding import TrainingBinding
from true_rl_agent.training_programs import ReferenceProgramManifest


class _ReferenceTaskPlugin(TaskPlugin):
    @property
    def task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="fake_push_recovery",
            name="Fake Reference Task",
            supported_policy_initializations=frozenset({"random"}),
            genesis_anchor_policy={"random": "reference_program"},
            candidate_capability_profile=FAKE_CAPABILITY_PROFILE,
            invariants=TaskInvariantSpec(
                {
                    "episode_length": 512,
                    "reference_fixture": "fixed_v1",
                }
            ),
            acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
        )


def _checksum(source: str) -> str:
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def _reference_fixture(tmp_path: Path) -> dict[str, object]:
    task = _ReferenceTaskPlugin()
    backend = FakeBackendAdapter()
    runner = FakeRunner()
    binding = backend.evaluation_binding_for(task.task_spec)
    source = (
        "from true_rl_agent.candidates.components import TrainingComponents\n\n"
        "def build_training_components() -> TrainingComponents:\n"
        "    return TrainingComponents(name='official-reference-r0')\n"
    )
    manifest = ReferenceProgramManifest(
        reference_id="official-reference-r0",
        task_id=task.task_spec.task_id,
        source_uri="upstream://official/reference/components.py",
        source_checksum=_checksum(source),
        source_size_bytes=len(source.encode("utf-8")),
        authorship_origin="official_upstream",
        upstream_project="official-locomotion",
        upstream_revision="revision-1",
        upstream_source_path="rewards/components.py",
        capability_profile_hash=str(FAKE_CAPABILITY_PROFILE.profile_hash),
    )
    training_binding = TrainingBinding(
        initialization_mode="random",
        policy_architecture_hash=canonical_json_hash({"policy": "fake_v1"}),
        agent_configuration_hash=canonical_json_hash(
            {"agent_configuration": "fake_v1"}
        ),
        initialization_seed_schedule_hash=canonical_json_hash(
            {"initialization_seeds": [42]}
        ),
        optimizer_training_profile_hash=canonical_json_hash(
            {"optimizer": "fake_v1"}
        ),
        budget_steps=128,
        private_training_seed_resolver_hash=canonical_json_hash(
            {"resolver": "private_reference_v1"}
        ),
        capability_profile_hash=str(FAKE_CAPABILITY_PROFILE.profile_hash),
    )
    checkpoint = CheckpointRef(
        uri="artifact://official-reference-r0/eval.pt",
        checksum=_checksum("reference-checkpoint-bytes"),
        role="eval",
        sealed=True,
    )
    eval_spec = EvalJobSpec(
        job_id="official-reference-r0-eval",
        task_id=task.task_spec.task_id,
        subject=ExperimentSubject.reference(manifest.reference_id),
        subject_manifest_hash=str(manifest.manifest_hash),
        program_source_checksum=manifest.source_checksum,
        training_binding_hash=str(training_binding.binding_hash),
        capability_profile_hash=manifest.capability_profile_hash,
        checkpoint=checkpoint,
        public_seed_count=4,
        public_accept_seed_count=3,
        sealed_validation_seed_count=8,
        eval_num_envs=64,
        evaluation_binding=binding,
        metadata={"fake_public_score": 0.25},
    )
    raw_result = runner.run_eval(eval_spec)
    qualification_gate = {
        "schema_version": REFERENCE_QUALIFICATION_GATE_SCHEMA_VERSION,
        "status": "pass",
        "protocol_hash": binding.task_protocol_hash,
        "evidence_hash": canonical_json_hash(
            {"task_owned_reference_evidence": "fixture-pass"}
        ),
    }
    metrics = dict(raw_result.metrics)
    metrics["reference_capability_gate"] = qualification_gate
    acceptance = dict(metrics["acceptance"])
    acceptance["required_pass_gates"] = [
        *acceptance["required_pass_gates"],
        "reference_capability_gate",
    ]
    metrics["acceptance"] = acceptance
    public_result = PublicEvalResult(
        status=raw_result.status,
        public_score=raw_result.public_score,
        metrics=metrics,
    )
    seal = build_trusted_eval_seal(
        eval_spec=eval_spec,
        task_invariants=task.task_spec.invariants,
        checkpoint_safety_report=metrics["checkpoint_safety_report"],
        candidate_isolation_status="pass",
    )

    private_path = tmp_path / REFERENCE_EVAL_ARTIFACT_FILENAME
    private_payload = {
        "schema_version": REFERENCE_EVAL_ARTIFACT_SCHEMA_VERSION,
        "reference_manifest": manifest.to_public_dict(),
        "training_binding": training_binding.to_private_dict(),
        "eval_job": eval_spec.to_public_dict(),
        "checkpoint_hash": checkpoint.checksum,
        "public_result": public_result.to_public_dict(),
        "trusted_eval_seal": seal.to_public_dict(),
        "qualification_gate": qualification_gate,
        "program_source": source,
    }
    private_path.write_bytes(
        canonical_json_bytes(private_payload, trailing_newline=True)
    )
    private_checksum = file_sha256(private_path)
    assert private_checksum is not None
    anchor = ReferenceProgramGenesisAnchor(
        reference_manifest=manifest,
        training_binding=training_binding,
        checkpoint_hash=checkpoint.checksum,
        public_result=public_result.to_public_dict(),
        public_result_hash=canonical_json_hash(public_result.to_public_dict()),
        trusted_eval_seal=seal.to_public_dict(),
        qualification_gate=qualification_gate,
        private_eval_artifact=PrivateArtifactRef(
            kind="reference_eval_artifact",
            uri=str(private_path.resolve()),
            checksum=private_checksum,
            reason="session_genesis_reference_evidence",
        ),
    )
    runtime_snapshot = {"fixture": "reference-session-v1"}
    artifact = SessionGenesisArtifact(
        session_id="reference-session",
        task_id=task.task_spec.task_id,
        backend_id=backend.backend_id,
        policy_initialization_kind="random",
        runtime_snapshot_hash=canonical_json_hash(runtime_snapshot),
        evaluation_binding=binding,
        anchor=anchor,
    )
    (tmp_path / SESSION_GENESIS_FILENAME).write_bytes(
        canonical_json_bytes(artifact.to_dict(), trailing_newline=True)
    )
    return {
        "task": task,
        "backend": backend,
        "runner": runner,
        "binding": binding,
        "source": source,
        "manifest": manifest,
        "training_binding": training_binding,
        "runtime_snapshot": runtime_snapshot,
        "private_path": private_path,
    }


def _load_reference_genesis(
    tmp_path: Path,
    fixture: dict[str, object],
):
    return prepare_session_genesis(
        output_dir=tmp_path,
        session_id="reference-session",
        task_plugin=fixture["task"],
        backend_adapter=fixture["backend"],
        runner=fixture["runner"],
        policy_initialization=PolicyInitialization.random(),
        runtime_snapshot=fixture["runtime_snapshot"],
        reference_program_manifest=fixture["manifest"],
    )


def _accepted_candidate_record(
    fixture: dict[str, object],
    *,
    genesis_hash: str,
    current_best,
) -> dict[str, object]:
    candidate_id = "candidate-after-reference"
    candidate_artifact = make_verified_candidate_artifact(candidate_id)
    candidate_manifest_hash = candidate_artifact.subject_manifest_hash
    candidate_source_checksum = candidate_artifact.program_source_checksum
    checkpoint = CheckpointRef(
        uri="artifact://candidate-after-reference/eval.pt",
        checksum=_checksum("candidate-checkpoint-bytes"),
        role="eval",
        sealed=True,
    )
    training_binding = fixture["training_binding"]
    train_spec = TrainJobSpec(
        job_id="round-001-train",
        task_id=fixture["task"].task_spec.task_id,
        subject=ExperimentSubject.candidate(candidate_id),
        subject_manifest_hash=candidate_manifest_hash,
        program_source_checksum=candidate_source_checksum,
        policy_initialization=PolicyInitialization.random(),
        training_binding=training_binding,
        budget_steps=training_binding.budget_steps,
        candidate_artifact=candidate_artifact,
    )
    eval_spec = EvalJobSpec(
        job_id="round-001-eval",
        task_id=fixture["task"].task_spec.task_id,
        subject=ExperimentSubject.candidate(candidate_id),
        subject_manifest_hash=candidate_manifest_hash,
        program_source_checksum=candidate_source_checksum,
        training_binding_hash=str(training_binding.binding_hash),
        capability_profile_hash=training_binding.capability_profile_hash,
        checkpoint=checkpoint,
        public_seed_count=4,
        public_accept_seed_count=3,
        sealed_validation_seed_count=8,
        eval_num_envs=64,
        evaluation_binding=fixture["binding"],
        metadata={"fake_public_score": 0.50},
    )
    public_result = fixture["runner"].run_eval(eval_spec)
    raw_score_stats = public_result.metrics["acceptance"]["score_stats"]
    score_stats = {
        key: raw_score_stats[key]
        for key in (
            "count",
            "mean",
            "std",
            "standard_error",
            "eval_protocol_hash",
        )
    }
    required_gates = list(
        public_result.metrics["acceptance"]["required_pass_gates"]
    )
    delta = 0.50 - current_best.score_for_comparison()
    return {
        "round_id": "round-001",
        "session_genesis_hash": genesis_hash,
        "evaluation_binding": fixture["binding"].to_public_dict(),
        "candidate": {
            "candidate_id": candidate_id,
            "changed_components": ["reward_callable"],
        },
        "verifier": {"status": "pass"},
        "candidate_surface": {
            "classification": "code_level",
            "eligible_for_policy_accept": True,
            "reason": "code_level_candidate",
        },
        "jobs": {
            "train": train_spec.to_public_dict(),
            "eval": eval_spec.to_public_dict(),
        },
        "train_result": {"job_id": train_spec.job_id},
        "public_result": public_result.to_public_dict(),
        "decision": {
            "outcome": "policy_improvement_accept",
            "accepted": True,
            "reason": "noise_aware_delta",
            "surface_classification": "code_level",
            "public_score": 0.50,
            "comparison": {
                "mode": "reference",
                "score_source": FAKE_ACCEPTANCE_PROTOCOL.score_source,
                "acceptance_protocol_hash": (
                    FAKE_ACCEPTANCE_PROTOCOL.protocol_hash
                ),
                "candidate_score": 0.50,
                "current_best_score": current_best.score_for_comparison(),
                "delta": delta,
                "threshold": FAKE_ACCEPTANCE_PROTOCOL.min_delta,
                "delta_met": True,
                "absolute_accept_floor": (
                    FAKE_ACCEPTANCE_PROTOCOL.absolute_accept_floor
                ),
                "candidate_score_stats": score_stats,
                "best_score_stats": dict(current_best.public_accept_stats),
                "candidate_evaluation_binding_hash": (
                    fixture["binding"].binding_hash
                ),
                "current_best": current_best.comparison_identity(),
                "public_score": 0.50,
            },
            "consumed_gates": required_gates,
            "gate_statuses": {gate: "pass" for gate in required_gates},
        },
    }


class ReferenceGenesisContractTests(unittest.TestCase):
    def test_restores_current_best_and_source_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            genesis = _load_reference_genesis(tmp_path, fixture)

            self.assertEqual(genesis.initial_current_best.mode, "reference")
            self.assertEqual(genesis.initial_current_best.anchor_kind, "reference")
            self.assertTrue(genesis.initial_current_best.has_full_eval_stats())
            self.assertEqual(genesis.reference_program_source(), fixture["source"])

            source_context = build_prompt_source_context(
                [],
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                initial_current_best=genesis.initial_current_best,
                session_genesis_hash=genesis.session_genesis_hash,
                initial_reference_program_source=genesis.reference_program_source(),
            )
            self.assertEqual(
                source_context["candidate_lineage"]["status"],
                "reference_current_best",
            )
            current = source_context["candidate_lineage"]["current_best"]
            self.assertEqual(current["reference_id"], "official-reference-r0")
            self.assertEqual(
                current["program_source_checksum"],
                fixture["manifest"].source_checksum,
            )
            payload = next(
                item
                for item in source_context["source_payloads"]
                if item["logical_id"] == CURRENT_BEST_SOURCE_ID
            )
            self.assertEqual(payload["role"], "reference_program_anchor")
            self.assertEqual(payload["content"], fixture["source"])

    def test_first_candidate_advances_without_sequence_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            genesis = _load_reference_genesis(tmp_path, fixture)
            record = _accepted_candidate_record(
                fixture,
                genesis_hash=genesis.session_genesis_hash,
                current_best=genesis.initial_current_best,
            )

            validation = validate_decision_record(
                record,
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                expected_session_genesis_hash=genesis.session_genesis_hash,
                expected_evaluation_binding=fixture["binding"],
            )
            replay = replay_acceptance_ledger(
                [record],
                initial_current_best=genesis.initial_current_best,
                expected_session_genesis_hash=genesis.session_genesis_hash,
                acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
            )

            self.assertTrue(validation.valid_policy_accept, validation.reasons)
            self.assertEqual(
                replay.entries[0].current_best_before.mode,
                "reference",
            )
            self.assertEqual(replay.entries[0].current_best_after.mode, "accepted")
            self.assertEqual(replay.current_best.anchor_kind, "candidate")
            self.assertEqual(replay.current_best.accepted_round_id, "round-001")
            self.assertEqual(
                replay.current_best.accepted_candidate_id,
                "candidate-after-reference",
            )

    def test_rejects_private_eval_job_or_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            private_path = fixture["private_path"]
            payload = json.loads(private_path.read_text(encoding="utf-8"))
            payload["eval_job"]["job_id"] = "forged-eval-job"
            private_path.write_bytes(
                canonical_json_bytes(payload, trailing_newline=True)
            )
            private_checksum = file_sha256(private_path)
            self.assertIsNotNone(private_checksum)
            genesis_path = tmp_path / SESSION_GENESIS_FILENAME
            genesis_payload = json.loads(genesis_path.read_text(encoding="utf-8"))
            genesis_payload["anchor"]["private_eval_artifact"]["checksum"] = (
                private_checksum
            )
            genesis_payload["session_genesis_hash"] = canonical_json_hash(
                {
                    key: value
                    for key, value in genesis_payload.items()
                    if key != "session_genesis_hash"
                }
            )
            genesis_path.write_bytes(
                canonical_json_bytes(genesis_payload, trailing_newline=True)
            )

            with self.assertRaisesRegex(ValueError, "trusted eval seal mismatch"):
                _load_reference_genesis(tmp_path, fixture)

    def test_source_projection_rejects_unbound_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = _reference_fixture(tmp_path)
            genesis = _load_reference_genesis(tmp_path, fixture)

            with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
                build_prompt_source_context(
                    [],
                    acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
                    initial_current_best=genesis.initial_current_best,
                    session_genesis_hash=genesis.session_genesis_hash,
                    initial_reference_program_source="forged source",
                )


if __name__ == "__main__":
    unittest.main()
