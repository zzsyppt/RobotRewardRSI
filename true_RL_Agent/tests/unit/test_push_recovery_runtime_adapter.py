from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.artifact import (  # noqa: E402
    CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
    seal_candidate_package,
)
from true_rl_agent.candidates.authoring_policy import (  # noqa: E402
    MAX_CANONICAL_COMPONENT_NAME_CHARS,
)
from true_rl_agent.candidates.components import (  # noqa: E402
    RewardTerm,
    TrainingComponents,
)
from true_rl_agent.candidates.loader import (  # noqa: E402
    MAX_CANDIDATE_EXECUTION_FILE_BYTES,
)
from true_rl_agent.candidates.mechanism import RuntimeActivationEvidence  # noqa: E402
from true_rl_agent.candidates.package_contract import (  # noqa: E402
    CANDIDATE_PACKAGE_FILE_MAX_BYTES,
    preflight_candidate_package,
)
from true_rl_agent.candidates.schema import (  # noqa: E402
    CandidateChangeSet,
    CandidatePackageRef,
)
from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.tasks.push_recovery import PushRecoveryTaskPlugin  # noqa: E402
from true_rl_agent.tasks.push_recovery.candidate_failure import (  # noqa: E402
    PushCandidateProgramFailure,
)
from true_rl_agent.tasks.push_recovery.runtime_adapter import (  # noqa: E402
    CandidateRewardVecEnvWrapper,
    CandidateRuntimeAdapter,
    RuntimeComponentError,
    guard_vector,
    runtime_contract,
    validate_training_components,
)
from true_rl_agent.training_programs import (  # noqa: E402
    resolve_candidate_program_identity,
)


class _FakeVecEnv:
    def __init__(self, *, extras: dict | None = None) -> None:
        self.unwrapped = self
        self.step_calls = 0
        self.extras = extras or {}

    def step(self, actions):
        self.step_calls += 1
        return ({"obs": self.step_calls}, [1.0, 1.0], [False, False], self.extras)


class _VelocityDeltaVecEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.step_calls = 0
        self.robot_data = type("RobotData", (), {})()
        self.robot = type("Robot", (), {"data": self.robot_data})()
        self.scene = {"robot": self.robot}
        self.velocities = [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.22, 0.0, 0.0], [0.03, 0.0, 0.0]],
        ]

    def step(self, actions):
        index = min(self.step_calls, len(self.velocities) - 1)
        self.robot_data.root_lin_vel_w = self.velocities[index]
        self.step_calls += 1
        return ({"obs": self.step_calls}, [1.0, 1.0], [False, False], {})


class PushRecoveryRuntimeAdapterTests(unittest.TestCase):
    def test_reward_peak_frame_uses_same_env_and_only_replaces_on_larger_peak(self) -> None:
        import torch

        for tensor in (False, True):
            with self.subTest(torch_tensor=tensor):
                rows = [[1.0, 2.0], [1.0, 5.0]]

                def vector(index):
                    return torch.tensor(rows[index]) if tensor else list(rows[index])

                adapter = CandidateRuntimeAdapter(
                    manifest=None,
                    components=TrainingComponents(
                        name="reward-peak-probe",
                        reward_terms=[
                            RewardTerm(name="benefit", weight=2.0, fn=lambda _: vector(0), clip_max=3.0),
                            RewardTerm(name="penalty", weight=-3.0, fn=lambda _: vector(1)),
                        ],
                    ),
                    num_envs=2,
                    planned_training_steps=8,
                    diagnostic_interval_steps=24,
                )
                adapter.enable_physical_program()
                wrapper = CandidateRewardVecEnvWrapper(_FakeVecEnv(), adapter)
                self.assertEqual(wrapper.step([[0.0], [0.0]])[1], [0.0, -10.0])
                first = adapter.runtime_summary()["max_abs_candidate_reward_frame"]
                self.assertEqual(first["control_step_1based"], 1)
                self.assertEqual(first["env_index_0based"], 1)
                self.assertEqual(first["signed_total"], -11.0)
                self.assertEqual(first["units"], "reward_rate_before_control_dt_integration")
                self.assertEqual(first["terms"], [
                    {"name": "benefit", "effective_weight": 2.0, "signed_weighted_contribution": 4.0},
                    {"name": "penalty", "effective_weight": -3.0, "signed_weighted_contribution": -15.0},
                ])
                self.assertEqual(adapter.runtime_summary()["max_abs_candidate_reward"], 11.0)

                rows[:] = [[3.0, 3.0], [2.0, 1.0]]
                self.assertEqual(wrapper.step([[0.0], [0.0]])[1], [1.0, 4.0])
                self.assertEqual(adapter.runtime_summary()["max_abs_candidate_reward_frame"], first)
                rows[:] = [[1.0, 1.0], [-3.0, 0.0]]
                self.assertEqual(wrapper.step([[0.0], [0.0]])[1], [12.0, 3.0])
                self.assertEqual(adapter.runtime_summary()["max_abs_candidate_reward_frame"], first)

                rows[:] = [[9.0, 1.0], [7.0, 1.0]]
                self.assertEqual(wrapper.step([[0.0], [0.0]])[1], [-14.0, 0.0])
                runtime = adapter.runtime_summary()
                peak = runtime["max_abs_candidate_reward_frame"]
                self.assertEqual(peak["control_step_1based"], 4)
                self.assertEqual(peak["env_index_0based"], 0)
                self.assertEqual(peak["signed_total"], -15.0)
                self.assertEqual(runtime["max_abs_candidate_reward"], 15.0)
                self.assertEqual([term["signed_weighted_contribution"] for term in peak["terms"]], [6.0, -21.0])
                self.assertEqual(sum(term["signed_weighted_contribution"] for term in peak["terms"]), peak["signed_total"])
                self.assertTrue(all(row["observation_count"] == 0 for row in runtime["component_activation_records"]))
                json.dumps(peak)

    def test_runtime_reuses_shared_component_name_bound(self) -> None:
        def reward(_inputs):
            return [0.0, 0.0]

        valid = "n" * MAX_CANONICAL_COMPONENT_NAME_CHARS
        validate_training_components(
            TrainingComponents(
                name=valid,
                reward_terms=[RewardTerm(name=valid, weight=1.0, fn=reward)],
            )
        )
        for components in (
            TrainingComponents(name=valid + "n"),
            TrainingComponents(
                name=valid,
                reward_terms=[
                    RewardTerm(name=valid + "n", weight=1.0, fn=reward)
                ],
            ),
        ):
            with self.subTest(program_name=components.name):
                with self.assertRaisesRegex(RuntimeComponentError, "name is invalid"):
                    validate_training_components(components)

    def test_candidate_callable_failure_is_typed_but_os_failure_is_not(self) -> None:
        def adapter_for(error: Exception) -> CandidateRuntimeAdapter:
            def fail(_inputs):
                raise error

            return CandidateRuntimeAdapter(
                manifest=None,
                components=TrainingComponents(
                    name="failure-probe",
                    reward_terms=(
                        RewardTerm(name="failure", weight=1.0, fn=fail),
                    ),
                ),
                num_envs=2,
                planned_training_steps=8,
                diagnostic_interval_steps=1,
            )

        with self.assertRaises(PushCandidateProgramFailure) as context:
            CandidateRewardVecEnvWrapper(
                _FakeVecEnv(),
                adapter_for(ValueError("private candidate error")),
            ).step([[0.0], [0.0]])
        self.assertNotIn("private candidate error", str(context.exception))

        with self.assertRaisesRegex(OSError, "filesystem unavailable"):
            CandidateRewardVecEnvWrapper(
                _FakeVecEnv(),
                adapter_for(OSError("filesystem unavailable")),
            ).step([[0.0], [0.0]])

    def test_shared_reward_adapter_failure_is_infrastructure_owned(self) -> None:
        adapter = CandidateRuntimeAdapter(
            manifest=None,
            components=TrainingComponents(
                name="adapter-failure-probe",
                reward_terms=(
                    RewardTerm(
                        name="valid-candidate-output",
                        weight=1.0,
                        fn=lambda _inputs: [0.0, 0.0],
                    ),
                ),
            ),
            num_envs=2,
            planned_training_steps=8,
            diagnostic_interval_steps=1,
        )

        with patch(
            "true_rl_agent.tasks.push_recovery.runtime_adapter._scale_like",
            side_effect=RuntimeError("shared adapter failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "shared adapter failed"):
                CandidateRewardVecEnvWrapper(_FakeVecEnv(), adapter).step(
                    [[0.0], [0.0]]
                )

    def test_runtime_adapter_rejects_noncanonical_integer_inputs(self) -> None:
        components = TrainingComponents(name="strict-runtime-inputs")
        valid = {
            "manifest": None,
            "components": components,
            "num_envs": 2,
            "planned_training_steps": 8,
            "diagnostic_interval_steps": 1,
        }
        for field_name, invalid in (
            ("num_envs", True),
            ("num_envs", 2.5),
            ("planned_training_steps", "8"),
            ("planned_training_steps", 0),
            ("diagnostic_interval_steps", False),
            ("diagnostic_interval_steps", -1),
        ):
            kwargs = {**valid, field_name: invalid}
            with self.subTest(field_name=field_name, invalid=invalid):
                with self.assertRaisesRegex(RuntimeComponentError, field_name):
                    CandidateRuntimeAdapter(**kwargs)

    def _write_candidate(self, root: Path) -> Path:
        candidate = root / "candidate"
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            textwrap.dedent(
                """
                schema_version: true_rl_agent.candidate.v1
                candidate_id: runtime_reward
                entrypoint: training_code.components:build_training_components
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (candidate / "training_code" / "components.py").write_text(
            textwrap.dedent(
                """
                from true_rl_agent.candidates.components import DiagnosticTerm, RewardTerm, ScheduleTerm, TrainingComponents


                def reward(inputs):
                    return [-0.25, 0.25]


                def diagnostic(inputs):
                    return {"progress": inputs.progress_fraction}


                def build_training_components():
                    return TrainingComponents(
                        name="runtime_reward",
                        reward_terms=[RewardTerm(name="recovery_bonus", weight=2.0, fn=reward)],
                        diagnostics=[DiagnosticTerm(name="progress_probe", fn=diagnostic)],
                        schedules=[ScheduleTerm(name="scale", key="reward_weight_scale", value=0.5)],
                    )
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return candidate

    def _seal_candidate(self, root: Path):
        candidate = self._write_candidate(root)
        (candidate / "rationale.md").write_text(
            "runtime reward\n",
            encoding="utf-8",
        )
        for filename in ("operation_diff.json", "diagnostics.json"):
            (candidate / filename).write_text("{}\n", encoding="utf-8")
        return seal_candidate_package(
            CandidatePackageRef(
                candidate_id="runtime_reward",
                path=str(candidate),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            ),
            artifact_root=root / "artifacts",
        )

    def _rewrite_legacy_artifact(
        self,
        artifact_path: Path,
        *,
        source_payload: bytes,
        sparse: bool = False,
    ) -> dict[str, object]:
        source_path = artifact_path / "training_code" / "components.py"
        source_path.write_bytes(source_payload)
        if sparse:
            for relative_path in (
                "rationale.md",
                "operation_diff.json",
                "diagnostics.json",
            ):
                (artifact_path / relative_path).unlink()

        identity = resolve_candidate_program_identity(
            artifact_path,
            expected_candidate_id="runtime_reward",
            sealed_artifact=True,
            enforce_package_size_limits=False,
        )
        manifest_path = artifact_path / CANDIDATE_ARTIFACT_MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sparse:
            manifest["files"] = [
                item
                for item in manifest["files"]
                if item["path"]
                in {"candidate_manifest.yaml", "training_code/components.py"}
            ]
        for item in manifest["files"]:
            if item["path"] == "training_code/components.py":
                item["checksum"] = (
                    "sha256:" + hashlib.sha256(source_payload).hexdigest()
                )
                item["size_bytes"] = len(source_payload)
        manifest["subject_manifest_hash"] = identity.manifest_hash
        manifest["program_source_checksum"] = identity.source_checksum
        manifest["program_source_size_bytes"] = identity.source_size_bytes
        manifest["artifact_hash"] = canonical_json_hash(
            {key: value for key, value in manifest.items() if key != "artifact_hash"}
        )
        manifest_path.write_bytes(canonical_json_bytes(manifest, trailing_newline=True))
        return manifest

    def _write_push_signal_candidate(self, root: Path) -> Path:
        candidate = root / "candidate"
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            textwrap.dedent(
                """
                schema_version: true_rl_agent.candidate.v1
                candidate_id: push_signal_reward
                entrypoint: training_code.components:build_training_components
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (candidate / "training_code" / "components.py").write_text(
            textwrap.dedent(
                """
                from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


                def reward(inputs):
                    return [
                        float(active) + float(vec[0])
                        for active, vec in zip(inputs.push_active, inputs.push_vec_public)
                    ]


                def build_training_components():
                    return TrainingComponents(
                        name="push_signal_reward",
                        reward_terms=[RewardTerm(name="push_signal_reward", weight=1.0, fn=reward)],
                    )
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return candidate

    def test_candidate_reward_adapter_adds_guarded_rewards_and_records_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CandidateRuntimeAdapter.from_candidate_dir(
                self._write_candidate(Path(tmp)),
                num_envs=2,
                planned_training_steps=8,
                diagnostic_interval_steps=1,
            )
            wrapper = CandidateRewardVecEnvWrapper(_FakeVecEnv(), adapter)

            result = wrapper.step([[0.0], [0.0]])

        self.assertEqual(result[1], [0.75, 1.25])
        runtime = adapter.runtime_summary()
        self.assertEqual(runtime["reward_term_calls"], 1)
        self.assertEqual(runtime["diagnostic_calls"], 1)
        self.assertTrue(runtime["nonzero_reward_contribution"])
        self.assertEqual(runtime["last_candidate_reward_mean"], 0.0)
        self.assertGreater(runtime["max_abs_candidate_reward"], 0.0)
        self.assertEqual(
            runtime["component_activation_records"],
            [
                {
                    "name": "recovery_bonus",
                    "observation_count": 1,
                    "nonzero_observation_count": 1,
                    "max_abs_weighted_contribution": 0.25,
                    "last_weighted_mean": 0.0,
                    "effective_weight": 1.0,
                    "sampling_interval_steps": 1,
                    "component_kind": "reward_term",
                    "nonzero_observation_fraction": 1.0,
                }
            ],
        )
        self.assertIn("reward_callable", runtime["capability_activation_records"])
        self.assertIn("reward_weight_scale", runtime["capability_activation_records"])
        evidence = RuntimeActivationEvidence.from_mapping(runtime)
        self.assertEqual(
            evidence.to_public_dict()["component_activation_records"],
            runtime["component_activation_records"],
        )
        self.assertIn("progress_probe", evidence.diagnostic_outputs)

    def test_candidate_runtime_adapter_binds_expected_id_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))

            with self.assertRaisesRegex(ValueError, "candidate manifest identity mismatch"):
                CandidateRuntimeAdapter.from_candidate_dir(
                    candidate_dir,
                    num_envs=2,
                    planned_training_steps=8,
                    expected_candidate_id="different_candidate",
                )

    def test_sealed_runtime_executes_bound_bytes_not_post_verification_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._seal_candidate(root)
            artifact_path = Path(artifact.path)
            source_path = artifact_path / "training_code" / "components.py"
            marker_path = root / "malicious-source-executed"
            replacement = root / "replacement-components.py"
            replacement.write_text(
                textwrap.dedent(
                    f"""
                    from pathlib import Path
                    Path({str(marker_path)!r}).write_text("executed", encoding="utf-8")
                    raise RuntimeError("malicious source executed")
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            real_preflight = preflight_candidate_package
            swapped = False

            def swap_then_preflight(candidate_root, **kwargs):
                nonlocal swapped
                if not swapped:
                    source_path.unlink()
                    replacement.replace(source_path)
                    swapped = True
                return real_preflight(candidate_root, **kwargs)

            with patch(
                "true_rl_agent.tasks.push_recovery.runtime_adapter."
                "preflight_candidate_package",
                side_effect=swap_then_preflight,
            ), self.assertRaisesRegex(
                ValueError,
                "source (size|checksum) mismatch before execution",
            ):
                CandidateRuntimeAdapter.from_candidate_dir(
                    artifact_path,
                    num_envs=2,
                    planned_training_steps=8,
                    expected_candidate_id=artifact.candidate_id,
                    expected_subject_manifest_hash=artifact.subject_manifest_hash,
                    expected_program_source_checksum=artifact.program_source_checksum,
                    expected_candidate_artifact_hash=artifact.artifact_hash,
                )

        self.assertTrue(swapped)
        self.assertFalse(marker_path.exists())

    def test_sparse_legacy_sealed_runtime_keeps_source_over_new_provider_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._seal_candidate(root)
            legacy_path = root / "legacy-sparse-artifact"
            shutil.copytree(artifact.path, legacy_path)
            source_path = legacy_path / "training_code" / "components.py"
            source_payload = source_path.read_bytes()
            target_size = (
                CANDIDATE_PACKAGE_FILE_MAX_BYTES["training_code/components.py"] + 1
            )
            source_payload += b"\n#" + b"x" * (target_size - len(source_payload) - 2)
            manifest = self._rewrite_legacy_artifact(
                legacy_path,
                source_payload=source_payload,
                sparse=True,
            )

            adapter = CandidateRuntimeAdapter.from_candidate_dir(
                legacy_path,
                num_envs=2,
                planned_training_steps=8,
                expected_candidate_id="runtime_reward",
                expected_subject_manifest_hash=manifest["subject_manifest_hash"],
                expected_program_source_checksum=manifest[
                    "program_source_checksum"
                ],
                expected_candidate_artifact_hash=manifest["artifact_hash"],
            )

        self.assertEqual(adapter.manifest.candidate_id, "runtime_reward")
        self.assertEqual(adapter.components.name, "runtime_reward")
        self.assertGreater(
            len(source_payload),
            CANDIDATE_PACKAGE_FILE_MAX_BYTES["training_code/components.py"],
        )

    def test_sealed_runtime_rejects_source_above_absolute_execution_bound_before_read(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._seal_candidate(root)
            legacy_path = root / "oversized-legacy-artifact"
            shutil.copytree(artifact.path, legacy_path)
            source_path = legacy_path / "training_code" / "components.py"
            source_payload = source_path.read_bytes()
            target_size = MAX_CANDIDATE_EXECUTION_FILE_BYTES + 1
            source_payload += b"\n#" + b"x" * (target_size - len(source_payload) - 2)
            manifest = self._rewrite_legacy_artifact(
                legacy_path,
                source_payload=source_payload,
            )

            with patch(
                "true_rl_agent.tasks.push_recovery.runtime_adapter."
                "open_preflighted_candidate_file",
                side_effect=AssertionError("oversized execution file must not be read"),
            ), self.assertRaisesRegex(ValueError, "absolute byte limit"):
                CandidateRuntimeAdapter.from_candidate_dir(
                    legacy_path,
                    num_envs=2,
                    planned_training_steps=8,
                    expected_candidate_id="runtime_reward",
                    expected_subject_manifest_hash=manifest[
                        "subject_manifest_hash"
                    ],
                    expected_program_source_checksum=manifest[
                        "program_source_checksum"
                    ],
                    expected_candidate_artifact_hash=manifest["artifact_hash"],
                )

    def test_guard_vector_rejects_wrong_length_and_nonfinite_outputs(self) -> None:
        with self.assertRaisesRegex(RuntimeComponentError, "must return 2 values"):
            guard_vector([1.0], name="bad", kind="reward", num_envs=2)
        with self.assertRaisesRegex(RuntimeComponentError, "non-finite"):
            guard_vector([1.0, float("nan")], name="bad", kind="reward", num_envs=2)

    def test_runtime_contract_reserved_reward_inputs_match_prompt_contract(self) -> None:
        runtime = runtime_contract()
        prompt = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[],
            public_context={"previous_rounds": 0, "public_only": True},
        )["reward_inputs_contract"]

        self.assertEqual(runtime["reserved_zero_fallback_inputs"], list(prompt["reserved_fields"]))
        self.assertEqual(runtime["candidate_receives_prompt_effective"], prompt["field_order"])

    def test_runtime_contract_exposes_public_push_signal_as_effective_inputs(self) -> None:
        runtime = runtime_contract()
        prompt = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[],
            public_context={"previous_rounds": 0, "public_only": True},
        )["reward_inputs_contract"]

        self.assertIn("push_active", runtime["candidate_receives_prompt_effective"])
        self.assertIn("push_vec_public", runtime["candidate_receives_prompt_effective"])
        self.assertEqual(runtime["public_push_signal_inputs"], prompt["public_push_signal_fields"])
        self.assertIn("push_active", prompt["fields"])
        self.assertIn("push_vec_public", prompt["fields"])
        self.assertNotIn("push_active", prompt["reserved_fields"])
        self.assertNotIn("push_vec_public", prompt["reserved_fields"])

    def test_runtime_uses_explicit_public_push_signal_from_step_extras(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CandidateRuntimeAdapter.from_candidate_dir(
                self._write_push_signal_candidate(Path(tmp)),
                num_envs=2,
                planned_training_steps=8,
            )
            wrapper = CandidateRewardVecEnvWrapper(
                _FakeVecEnv(
                    extras={
                        "push_active": [0.0, 1.0],
                        "push_vec_public": [[0.0, 0.0], [0.25, -0.1]],
                    }
                ),
                adapter,
            )

            result = wrapper.step([[0.0], [0.0]])

        self.assertEqual(result[1], [1.0, 2.25])
        provenance = adapter.runtime_summary()["input_provenance"]
        self.assertEqual(provenance["push_active"], "observed")
        self.assertEqual(provenance["push_vec_public"], "observed")

    def test_runtime_derives_public_push_signal_from_observed_velocity_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CandidateRuntimeAdapter.from_candidate_dir(
                self._write_push_signal_candidate(Path(tmp)),
                num_envs=2,
                planned_training_steps=8,
                diagnostic_interval_steps=1,
            )
            wrapper = CandidateRewardVecEnvWrapper(_VelocityDeltaVecEnv(), adapter)

            first = wrapper.step([[0.0], [0.0]])
            second = wrapper.step([[0.0], [0.0]])

        self.assertEqual(first[1], [1.0, 1.0])
        self.assertAlmostEqual(second[1][0], 2.22)
        self.assertEqual(second[1][1], 1.0)
        provenance = adapter.runtime_summary()["input_provenance"]
        self.assertEqual(provenance["push_active"], "observed")
        self.assertEqual(provenance["push_vec_public"], "observed")
        activation = adapter.runtime_summary()["component_activation_records"][0]
        self.assertEqual(activation["observation_count"], 2)
        self.assertEqual(activation["nonzero_observation_count"], 1)
        self.assertEqual(activation["nonzero_observation_fraction"], 0.5)

    def test_runtime_keeps_public_push_signal_zero_when_unobserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CandidateRuntimeAdapter.from_candidate_dir(
                self._write_push_signal_candidate(Path(tmp)),
                num_envs=2,
                planned_training_steps=8,
            )
            wrapper = CandidateRewardVecEnvWrapper(_FakeVecEnv(), adapter)

            result = wrapper.step([[0.0], [0.0]])

        self.assertEqual(result[1], [1.0, 1.0])
        provenance = adapter.runtime_summary()["input_provenance"]
        self.assertEqual(provenance["push_active"], "zero_fallback")
        self.assertEqual(provenance["push_vec_public"], "zero_fallback")

    def test_input_provenance_tracks_current_action_history(self) -> None:
        observed_inputs = []

        def action_delta_reward(inputs):
            observed_inputs.append((inputs.prev_action, inputs.action_delta))
            return [row[0] for row in inputs.action_delta]

        adapter = CandidateRuntimeAdapter(
            manifest=None,
            components=TrainingComponents(
                name="action-history-probe",
                reward_terms=[RewardTerm(name="delta", weight=1.0, fn=action_delta_reward)],
            ),
            num_envs=2,
            planned_training_steps=8,
        )
        wrapper = CandidateRewardVecEnvWrapper(_FakeVecEnv(), adapter)

        self.assertEqual(wrapper.step([[2.0], [4.0]])[1], [3.0, 5.0])
        first = adapter.runtime_summary()["input_provenance"]
        self.assertEqual(first["last_action"], "observed")
        self.assertEqual(first["prev_action"], "zero_fallback")
        self.assertEqual(first["action_delta"], "zero_fallback")

        self.assertEqual(wrapper.step([[3.0], [6.0]])[1], [2.0, 3.0])
        second = adapter.runtime_summary()["input_provenance"]
        self.assertEqual(second["prev_action"], "observed")
        self.assertEqual(second["action_delta"], "observed")
        self.assertEqual(first["prev_action"], "zero_fallback")
        self.assertEqual(observed_inputs, [
            ([[0.0], [0.0]], [[2.0], [4.0]]),
            ([[2.0], [4.0]], [[1.0], [2.0]]),
        ])


if __name__ == "__main__":
    unittest.main()
