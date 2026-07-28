from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.backends.rsl_rl_training_config import (
    RslRlAgentConfiguration,
)
from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.training_binding import (
    TrainingBinding,
    initialization_seed_schedule_hash,
    validate_training_execution_argv,
)
from true_rl_agent.training_programs import ReferenceProgramManifest


def _digest(label: str) -> str:
    return canonical_json_hash({"label": label})


def _checksum(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _profile() -> CandidateCapabilityProfile:
    return CandidateCapabilityProfile.build(
        profile_id="phase_a_profile_v1",
        capability_ids=("reward_term", "diagnostic_term"),
        known_capability_ids=("diagnostic_term", "reward_term"),
    )


def _training_binding(profile: CandidateCapabilityProfile) -> TrainingBinding:
    return TrainingBinding(
        initialization_mode="random",
        policy_architecture_hash=_digest("architecture"),
        agent_configuration_hash=_digest("agent-configuration"),
        initialization_seed_schedule_hash=_digest("initialization-seeds"),
        optimizer_training_profile_hash=_digest("optimizer"),
        budget_steps=128,
        private_training_seed_resolver_hash=_digest("private-resolver"),
        capability_profile_hash=str(profile.profile_hash),
    )


class PhaseAContractTests(unittest.TestCase):
    def test_rsl_rl_execution_config_is_applied_and_bound_to_argv(self) -> None:
        configuration = RslRlAgentConfiguration(
            policy={"class_name": "ActorCritic", "hidden_dims": [64, 32]},
            algorithm={"class_name": "PPO", "learning_rate": 0.001},
        )

        class Section:
            pass

        class AgentCfg:
            def __init__(self) -> None:
                self.policy = Section()
                self.policy.class_name = "DriftedPolicy"
                self.policy.hidden_dims = [1]
                self.algorithm = Section()
                self.algorithm.class_name = "DriftedOptimizer"
                self.algorithm.learning_rate = 1.0

            def to_dict(self) -> dict:
                return {
                    "policy": dict(vars(self.policy)),
                    "algorithm": dict(vars(self.algorithm)),
                }

        agent_cfg = AgentCfg()
        configuration.apply_and_verify(agent_cfg)
        self.assertEqual(
            agent_cfg.to_dict()["policy"],
            configuration.to_dict()["policy"],
        )
        binding = TrainingBinding(
            initialization_mode="random",
            policy_architecture_hash=str(configuration.policy_architecture_hash),
            agent_configuration_hash=str(configuration.agent_configuration_hash),
            initialization_seed_schedule_hash=initialization_seed_schedule_hash(42),
            optimizer_training_profile_hash=str(
                configuration.optimizer_training_profile_hash
            ),
            budget_steps=128,
            private_training_seed_resolver_hash=_digest("private-resolver"),
            capability_profile_hash=str(_profile().profile_hash),
        )
        argv = [
            "python",
            "train.py",
            "--training-binding-hash",
            str(binding.binding_hash),
            "--capability-profile-hash",
            binding.capability_profile_hash,
            "--policy-architecture-hash",
            binding.policy_architecture_hash,
            "--optimizer-training-profile-hash",
            binding.optimizer_training_profile_hash,
            "--agent-configuration-hash",
            binding.agent_configuration_hash,
            "--agent-configuration-json",
            configuration.to_canonical_json(),
            "--initialization-seed-schedule-hash",
            binding.initialization_seed_schedule_hash,
            "--seed",
            "42",
        ]
        self.assertEqual(
            validate_training_execution_argv(argv, binding=binding),
            configuration.to_dict(),
        )
        with self.assertRaisesRegex(ValueError, "--seed must appear exactly once"):
            validate_training_execution_argv(
                [*argv, "--seed", "7"],
                binding=binding,
            )

    def test_experiment_subject_is_the_only_exact_structured_identity(self) -> None:
        subject = ExperimentSubject.reference("official-reference-r0")

        self.assertEqual(
            ExperimentSubject.from_mapping(subject.to_public_dict()),
            subject,
        )
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            ExperimentSubject.from_mapping(
                {**subject.to_public_dict(), "candidate_id": "forged"}
            )
        with self.assertRaisesRegex(ValueError, "schema_version mismatch"):
            ExperimentSubject.from_mapping(
                {**subject.to_public_dict(), "schema_version": "legacy.v1"}
            )
        with self.assertRaisesRegex(
            ValueError,
            "unsupported experiment subject kind",
        ):
            ExperimentSubject(kind="genesis", subject_id="forged")

    def test_capability_profile_is_closed_sorted_and_self_hashed(self) -> None:
        profile = _profile()

        self.assertEqual(profile.capability_ids, ("diagnostic_term", "reward_term"))
        self.assertEqual(
            CandidateCapabilityProfile.from_mapping(
                profile.to_public_dict(),
                known_capability_ids=("diagnostic_term", "reward_term"),
            ),
            profile,
        )

        unordered = profile.to_public_dict()
        unordered["capability_ids"] = ["reward_term", "diagnostic_term"]
        unordered["profile_hash"] = canonical_json_hash(
            {
                key: value
                for key, value in unordered.items()
                if key != "profile_hash"
            }
        )
        with self.assertRaisesRegex(ValueError, "must be sorted"):
            CandidateCapabilityProfile.from_mapping(
                unordered,
                known_capability_ids=("diagnostic_term", "reward_term"),
            )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            CandidateCapabilityProfile.build(
                profile_id="duplicate-profile",
                capability_ids=("reward_term", "reward_term"),
                known_capability_ids=("reward_term",),
            )
        with self.assertRaisesRegex(ValueError, "unknown capability"):
            CandidateCapabilityProfile.build(
                profile_id="unknown-profile",
                capability_ids=("future_permission",),
                known_capability_ids=("reward_term",),
            )

    def test_training_binding_has_private_payload_and_strict_types(self) -> None:
        binding = _training_binding(_profile())

        self.assertEqual(TrainingBinding.from_mapping(binding.to_private_dict()), binding)
        self.assertNotIn(
            "private_training_seed_resolver_hash",
            binding.to_public_dict(),
        )
        malformed = binding.to_private_dict()
        malformed["budget_steps"] = True
        malformed["binding_hash"] = canonical_json_hash(
            {
                key: value
                for key, value in malformed.items()
                if key != "binding_hash"
            }
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            TrainingBinding.from_mapping(malformed)

        drifted = binding.to_private_dict()
        drifted["optimizer_training_profile_hash"] = _digest(
            "different-optimizer"
        )
        with self.assertRaisesRegex(ValueError, "binding hash mismatch"):
            TrainingBinding.from_mapping(drifted)

    def test_reference_manifest_binds_canonical_source_bytes(self) -> None:
        profile = _profile()
        source = b"def build_training_components():\n    return None\n"
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "components.py"
            source_path.write_bytes(source)
            manifest = ReferenceProgramManifest(
                reference_id="official-reference-r0",
                task_id="fake-task",
                source_uri=str(source_path),
                source_checksum=_checksum(source),
                source_size_bytes=len(source),
                authorship_origin="official_upstream",
                upstream_project="official-project",
                upstream_revision="revision-1",
                upstream_source_path="upstream/rewards.py",
                capability_profile_hash=str(profile.profile_hash),
            )

            manifest.verify_source_file(source_path)
            self.assertEqual(
                ReferenceProgramManifest.from_mapping(manifest.to_public_dict()),
                manifest,
            )
            source_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "source size mismatch|source checksum mismatch",
            ):
                manifest.verify_source_file(source_path)

            drifted = manifest.to_public_dict()
            drifted["upstream_revision"] = "revision-2"
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                ReferenceProgramManifest.from_mapping(drifted)


if __name__ == "__main__":
    unittest.main()
