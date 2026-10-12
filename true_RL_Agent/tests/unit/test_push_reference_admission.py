"""Observed-outcome reference admission and frozen pre-policy readers."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.unit.test_push_force_consumers import episode
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.tasks.push_recovery import metrics
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionConfig, PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.config import PUSH_ACTIVE_CELLS, PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.phased_adapter import PushPhasedEvaluationAdapter, PUSH_PHASE_EVIDENCE_SCHEMA
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.reference_program import (
    qualify_push_reference, reference_dev_evidence_passes, reference_qualification_contract,
)
from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest
from true_rl_agent.tasks.push_recovery.rollout import aggregate_recovery_episodes

ROOT = Path(__file__).resolve().parents[2]
POLICY = "observed_outcomes_v1"


def _rollout(successes):
    rows = []
    for env_id in range(64):
        row = episode(success=env_id < successes, env_id=env_id)
        row["stratum"] = ("low", "medium", "high", "repeated")[env_id // 16]
        if row["stratum"] != "repeated":
            row["force_events"] = row["force_events"][:1]
            for name in ("planned_push_count", "push_event_count", "completed_push_event_count", "verified_push_event_count"):
                row[name] = 1
        rows.append(row)
    return {"schema_version": metrics.FORCE_ROLLOUT_SCHEMA, "status": "pass", "per_episode": rows,
            "aggregate": aggregate_recovery_episodes(rows, force_sequence=True)}


def _dev_gate(config, counts):
    phase = SimpleNamespace(task_id="unitree_push_recovery", phase=EvaluationPhase.PUBLIC_DEV,
        eval_protocol_hash=config.evaluation_protocol().protocol_hash,
        task_evidence={"schema_version": PUSH_PHASE_EVIDENCE_SCHEMA, "phase": "public_dev",
                       "seed_summaries": [_rollout(count) for count in counts]})
    return PushPhasedEvaluationAdapter(config, "a" * 64).project_reference_public_dev_gate(phase_result=phase)


def _manifest(config):
    return resolve_push_cell_manifest(cell_variant_id=config.cell_variant_id, config=config,
                                     plugin=PushRecoveryTaskPlugin(config))


class PushReferenceAdmissionTests(unittest.TestCase):
    def test_new_dev_requires_observed_success_and_failure_without_a_replacement_ratio(self):
        for cell in PUSH_ACTIVE_CELLS:
            config = PushRecoveryTaskConfig.for_cell(cell)
            self.assertEqual(config.reference_admission_policy, POLICY)
            for counts, expected in (((11, 11, 11), "pass"), ((46, 49, 46), "pass"),
                                     ((1, 0, 0), "pass"), ((64, 64, 63), "pass"),
                                     ((0, 0, 0), "fail"), ((64, 64, 64), "fail")):
                with self.subTest(cell=cell, counts=counts):
                    gate = _dev_gate(config, counts)
                    self.assertEqual(gate["recovery_evidence_status"], "pass")
                    self.assertEqual(gate["status"], expected)
            self.assertFalse(reference_dev_evidence_passes(config, {"score_stats": {"mean": .5}, "seed_results": []}))
            # These endpoint exclusions do not claim that 191/192 is far from saturation.

    def test_explicit_old_policy_keeps_the_frozen_band_and_absolute_floor(self):
        for cell in PUSH_ACTIVE_CELLS:
            config = PushRecoveryTaskConfig.for_cell(cell, reference_admission_policy=None)
            self.assertIsNone(config.reference_admission_policy)
            self.assertEqual(config.absolute_accept_floor, .2)
            self.assertEqual(reference_qualification_contract(config)["public_dev_reference_band"], [.2, .9])
            self.assertEqual(_dev_gate(config, (11, 11, 11))["status"], "fail")
            self.assertEqual(_dev_gate(config, (46, 49, 46))["status"], "pass")
            self.assertEqual(_dev_gate(config, (64, 64, 63))["status"], "fail")
            with self.assertRaisesRegex(ValueError, "floor"):
                replace(config, reference_admission_policy=POLICY)

    def test_policy_changes_identity_but_not_physics_or_candidate_relative_rule(self):
        for cell in PUSH_ACTIVE_CELLS:
            new = PushRecoveryTaskConfig.for_cell(cell)
            old = PushRecoveryTaskConfig.for_cell(cell, reference_admission_policy=None)
            self.assertEqual(new.train_runtime_dynamics(), old.train_runtime_dynamics())
            self.assertEqual(new.eval_runtime_dynamics(), old.eval_runtime_dynamics())
            self.assertEqual(new.push_protocol, old.push_protocol)
            self.assertEqual(new.push_evaluation_protocol, old.push_evaluation_protocol)
            new_accept, old_accept = new.acceptance_protocol().to_public_dict(), old.acceptance_protocol().to_public_dict()
            self.assertEqual(new_accept.pop("absolute_accept_floor"), 0)
            self.assertEqual(old_accept.pop("absolute_accept_floor"), .2)
            self.assertNotEqual(new_accept.pop("acceptance_protocol_hash"), old_accept.pop("acceptance_protocol_hash"))
            self.assertEqual(new_accept.pop("protocol_id"), "unitree_push_recovery_acceptance_v3")
            self.assertEqual(old_accept.pop("protocol_id"), "unitree_push_recovery_acceptance_v2")
            self.assertEqual(new_accept, old_accept)
            new_contract, old_contract = reference_qualification_contract(new), reference_qualification_contract(old)
            self.assertEqual(new_contract["required_gates"], old_contract["required_gates"])
            self.assertEqual(new_contract["independent_validation"], old_contract["independent_validation"])
            self.assertEqual(new_contract["train_budget_steps"], old_contract["train_budget_steps"])
            new_manifest, old_manifest = _manifest(new), _manifest(old)
            self.assertEqual(new_manifest.robot_task_binding["task_projection"]["reference_admission_policy"], POLICY)
            self.assertNotIn("reference_admission_policy", old_manifest.robot_task_binding["task_projection"])
            self.assertNotEqual(new_manifest.manifest_hash, old_manifest.manifest_hash)
            self.assertNotEqual(new_manifest.resolved_cell_task_protocol_hash, old_manifest.resolved_cell_task_protocol_hash)
            for config, manifest in ((new, new_manifest), (old, old_manifest)):
                self.assertEqual(manifest.contract_hashes["reference_capability_gate_hash"],
                                 canonical_json_hash(reference_qualification_contract(config)))

    def test_real_frozen_force_manifests_keep_original_admission_and_content_hash(self):
        for robot in ("g1", "go2"):
            with self.subTest(robot=robot):
                path = ROOT / f"PLAN/2new_world/P6/force_unified_01/{robot}_public_dev_03/composition/resolved_cell_manifest.json"
                frozen = load_resolved_cell_manifest(path)
                config = PushRecoveryTaskConfig.from_manifest(frozen)
                self.assertIsNone(config.reference_admission_policy)
                self.assertEqual(config.absolute_accept_floor, .2)
                contract = reference_qualification_contract(config)
                self.assertEqual(contract["public_dev_reference_band"], [.2, .9])
                self.assertEqual(canonical_json_hash(contract), frozen.contract_hashes["reference_capability_gate_hash"])
                self.assertEqual(config.acceptance_protocol().protocol_hash, frozen.contract_hashes["acceptance_protocol_hash"])
                rebuilt = _manifest(replace(config, resolved_cell_task_protocol_hash=None))
                self.assertEqual(rebuilt.to_dict(), frozen.to_dict())
                # A historical argv inherits today's CLI defaults, but read-only
                # reconstruction must take admission from the frozen manifest.
                provider = PushRecoveryCompositionProvider()
                with tempfile.TemporaryDirectory() as directory:
                    copied = Path(directory) / "resolved_cell_manifest.json"
                    copied.write_bytes(path.read_bytes())
                    before = copied.read_bytes()
                    loaded = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": frozen.cell_variant_id},
                        context=TaskCompositionContext(project_root=ROOT, output_dir=Path(directory), read_only=True))
                    self.assertIsNone(loaded.task_plugin.config.reference_admission_policy)
                    self.assertEqual(loaded.resolved_cell_manifest.to_dict(), frozen.to_dict())
                    self.assertEqual(copied.read_bytes(), before)

    def test_composition_freezes_new_policy_and_omitted_snapshot_reads_old_policy(self):
        provider = PushRecoveryCompositionProvider()
        for cell in PUSH_ACTIVE_CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                current = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": cell},
                    context=TaskCompositionContext(project_root=ROOT, output_dir=Path(directory)))
                snapshot = dict(current.task_defaults)
                self.assertEqual(snapshot["reference_admission_policy"], POLICY)
                parsed = PushRecoveryCompositionConfig.from_values(snapshot)
                self.assertEqual(parsed.to_task_config().reference_admission_policy, POLICY)
                self.assertEqual(parsed.to_task_defaults(), snapshot)
                old_snapshot = {key: value for key, value in snapshot.items() if key != "reference_admission_policy"}
                old = PushRecoveryCompositionConfig.from_values(old_snapshot)
                self.assertIsNone(old.to_task_config().reference_admission_policy)
                self.assertEqual(old.to_task_config().absolute_accept_floor, .2)
                self.assertNotIn("reference_admission_policy", old.to_task_defaults())
                self.assertEqual(old.to_task_defaults(), old_snapshot)
                self.assertIsNone(PushRecoveryCompositionConfig.from_values(old.to_task_defaults()).to_task_config().reference_admission_policy)

    def test_final_reference_keeps_evidence_budget_and_sealed_gates_with_no_point_two_floor(self):
        config = PushRecoveryTaskConfig.for_cell("push.g1_37dof.recovery_v3")
        def qualify(dev_counts=(1, 0, 0), accept_counts=(1, 0, 0), *, failed_gate=None):
            result = metrics.build_push_public_result_from_summaries(
                public_dev_summaries=[_rollout(count) for count in dev_counts],
                public_accept_summaries=[_rollout(count) for count in accept_counts],
                budget_summary={"status": "pass"}, acceptance_protocol=config.acceptance_protocol(),
                evaluation_protocol=config.evaluation_protocol())
            projected = copy.deepcopy(dict(result.metrics))
            projected["sealed_validation_gate"] = {"status": "pass"}
            if failed_gate is not None and failed_gate != "budget_accounting":
                projected[failed_gate]["status"] = "fail"
            return qualify_push_reference(config=config,
                public_result=PublicEvalResult(status=result.status, public_score=result.public_score, metrics=projected),
                train_summary={"status": "fail" if failed_gate == "budget_accounting" else "pass"},
                task_protocol_hash="a" * 64).gate["status"]
        self.assertEqual(qualify(), "pass")
        self.assertEqual(qualify(accept_counts=(64, 64, 64)), "pass")
        self.assertEqual(qualify(accept_counts=(0, 0, 0)), "fail")
        self.assertEqual(qualify(dev_counts=(0, 0, 0)), "fail")
        self.assertEqual(qualify(dev_counts=(64, 64, 64)), "fail")
        for gate in reference_qualification_contract(config)["required_gates"]:
            with self.subTest(failed_gate=gate):
                self.assertEqual(qualify(failed_gate=gate), "fail")

    def test_sealed_positive_outcome_requirement_is_versioned_without_restoring_point_two_floor(self):
        from tests.unit.test_push_recovery_v2_metrics import rollout as old_rollout

        old = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v2")
        current = PushRecoveryTaskConfig.for_cell("push.go2_12dof.recovery_v3")
        old_zero_floor = replace(old.acceptance_protocol(), absolute_accept_floor=0)

        def sealed(config, protocol, sample, accept=.05):
            with patch.object(metrics, "build_sealed_rollout_evidence", side_effect=lambda **values: values):
                return metrics.build_push_sealed_rollout_evidence(
                    summaries=[sample] * config.sealed_validation_seed_count,
                    evaluation_protocol=config.evaluation_protocol(), acceptance_protocol=protocol,
                    public_accept_score=accept, checkpoint_hash="unused", evaluation_binding_hash="unused",
                    evaluation_subject=None, capability_profile_hash="unused")["task_validation_status"]

        self.assertEqual(sealed(old, old_zero_floor, old_rollout(0)), "pass")
        self.assertEqual(sealed(current, current.acceptance_protocol(), _rollout(0)), "fail")
        self.assertEqual(sealed(current, current.acceptance_protocol(), _rollout(1)), "pass")
        self.assertEqual(sealed(current, current.acceptance_protocol(), _rollout(1), accept=.2), "fail")


if __name__ == "__main__":
    unittest.main()
