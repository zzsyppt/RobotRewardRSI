from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.candidates.components import RewardTerm  # noqa: E402
from true_rl_agent.candidates.mechanism import (  # noqa: E402
    MechanismEvidenceGate,
    RuntimeActivationEvidence,
)
from true_rl_agent.candidates.schema import CandidateSurfaceAssessment  # noqa: E402
from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_training import (  # noqa: E402
    configure_stair_rsl_rl_agent,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (  # noqa: E402
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.reward_inputs import (  # noqa: E402
    StairRewardInputsV2,
    stair_reward_input_prompt_contract,
)
from true_rl_agent.tasks.stair_climbing.reward_performance import (  # noqa: E402
    STAIR_REWARD_BENCHMARK_CASE_SCHEMA_VERSION,
    compile_stair_reward_performance_report,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (  # noqa: E402
    QualifiedStairRewardProgram,
    StairCompleteRewardDispatcher,
    StairRewardProgramError,
    assert_stair_reward_program_deterministic,
    evaluate_stair_reward_program,
    qualify_stair_reference_reward_program,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
)


class StairCompleteRewardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
        )

    def test_canonical_abi_preserves_common_prefix_and_public_boundary(self) -> None:
        contract = self.config.protocol_v2.complete_reward_runtime
        common = tuple(
            self.config.protocol_v1.to_dict()["candidate_boundary"][
                "common_reward_input_fields"
            ]
        )
        prompt = stair_reward_input_prompt_contract(contract)

        self.assertEqual(contract.field_names[: len(common)], common)
        self.assertEqual(len(contract.field_names), len(set(contract.field_names)))
        self.assertEqual(prompt["input_schema_hash"], contract.input_schema_hash)
        self.assertEqual(
            [item["name"] for item in prompt["fields"]],
            list(contract.field_names),
        )
        serialized = repr(prompt).lower()
        for forbidden in ("raw_seed", "bin_identity", "checkpoint_local_path"):
            self.assertNotIn(forbidden, serialized)

    def test_reward_inputs_are_task_builder_owned_and_immutable(self) -> None:
        contract = self.config.protocol_v2.complete_reward_runtime
        marker = object()
        values = {name: marker for name in contract.field_names}
        inputs = StairRewardInputsV2._validated(
            contract=contract,
            values=values,
        )

        self.assertIs(inputs.base_height, marker)
        self.assertEqual(inputs.schema_hash, contract.input_schema_hash)
        with self.assertRaisesRegex(AttributeError, "immutable"):
            inputs.base_height = object()
        with self.assertRaisesRegex(TypeError, "task-owned builder"):
            StairRewardInputsV2(
                contract=contract,
                values=tuple(values.values()),
            )

    def test_official_reference_binds_real_bytes_and_complete_term_set(self) -> None:
        resolved = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        )
        equivalence = self.config.protocol_v2.to_public_dict()[
            "complete_reward_runtime"
        ]["reference_equivalence"]

        self.assertLess(resolved.artifact.manifest.source_size_bytes, 24 * 1024)
        self.assertEqual(
            resolved.reward_program.term_names,
            tuple(equivalence["active_term_names"]),
        )
        self.assertEqual(
            resolved.artifact.manifest.manifest_hash,
            resolved.reward_program.subject_manifest_hash,
        )
        self.assertIs(copy.deepcopy(resolved.reward_program), resolved.reward_program)
        self.assertIsInstance(resolved.reward_program.reward_terms, tuple)
        self.assertFalse(hasattr(resolved.reward_program, "components"))
        with self.assertRaisesRegex(TypeError, "task verifier"):
            QualifiedStairRewardProgram(
                reward_terms=resolved.reward_program.reward_terms,
                contract=resolved.reward_program.contract,
                subject=resolved.reward_program.subject,
                subject_manifest_hash=resolved.reward_program.subject_manifest_hash,
                program_source_checksum=resolved.reward_program.program_source_checksum,
                program_source_size_bytes=(
                    resolved.reward_program.program_source_size_bytes
                ),
                static_verifier_policy_hash=(
                    resolved.reward_program.static_verifier_policy_hash
                ),
                qualification_hash=resolved.reward_program.qualification_hash,
            )

    def test_qualification_requires_a_verified_role_artifact(self) -> None:
        resolved = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        )
        with self.assertRaisesRegex(TypeError, "verified artifact"):
            qualify_stair_reference_reward_program(
                artifact=object(),
                contract=resolved.reward_program.contract,
            )

    def test_runtime_profile_is_bound_to_the_active_phase_c_contract(self) -> None:
        profile = self.config.train_runtime_profile().to_public_dict()

        self.assertEqual(
            profile["complete_reward_runtime"],
            STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
        )
        self.assertEqual(
            profile["draft_protocol_hash"],
            self.config.protocol_v2.protocol_hash,
        )

    def test_rsl_rl_execution_configuration_is_strict_and_single_source(self) -> None:
        configuration = self.config.training_agent_configuration()

        class FakeAgentCfg:
            def __init__(self) -> None:
                self.policy = SimpleNamespace(**dict(configuration.policy))
                self.algorithm = SimpleNamespace(**dict(configuration.algorithm))

            def to_dict(self) -> dict[str, object]:
                return {
                    "policy": dict(vars(self.policy)),
                    "algorithm": dict(vars(self.algorithm)),
                }

        agent_cfg = FakeAgentCfg()
        effective = configure_stair_rsl_rl_agent(
            agent_cfg,
            configuration=configuration,
            seed=17,
            device="cuda:0",
            num_steps_per_env=8,
            max_iterations=1,
            save_interval=1,
            run_name="phase_c_smoke",
            experiment_name="stair_test",
        )
        self.assertEqual(effective["seed"], 17)
        self.assertEqual(effective["num_steps_per_env"], 8)
        self.assertEqual(
            effective["agent_configuration_hash"],
            configuration.agent_configuration_hash,
        )
        with self.assertRaisesRegex(ValueError, "seed"):
            configure_stair_rsl_rl_agent(
                FakeAgentCfg(),
                configuration=configuration,
                seed=True,
                device="cuda:0",
                num_steps_per_env=8,
                max_iterations=1,
                save_interval=1,
                run_name="phase_c_smoke",
                experiment_name="stair_test",
            )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_reward_output_fault_is_normal_and_tensor_runtime_survives(self) -> None:
        import torch

        resolved = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        )
        contract = resolved.reward_program.contract
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        values = {}
        for field in contract.input_fields:
            dtype = torch.bool if field.dtype == "bool" else torch.float32
            values[field.name] = torch.zeros(
                (2, *field.trailing_shape),
                dtype=dtype,
                device=device,
            )
        values["projected_gravity"][:, 2] = -1.0
        values["base_quat_world"][:, 0] = 1.0
        values["remaining_goal_distance_m"].fill_(1.0)
        values["ankle_soft_joint_pos_limits"][:, :, 0] = -1.0
        values["ankle_soft_joint_pos_limits"][:, :, 1] = 1.0
        values["hip_knee_joint_acc"].fill_(1.0e6)
        inputs = StairRewardInputsV2._validated(
            contract=contract,
            values=values,
        )

        with self.assertRaisesRegex(
            StairRewardProgramError,
            "reward_term_weighted:dof_acc_l2",
        ):
            evaluate_stair_reward_program(
                resolved.reward_program,
                inputs,
                num_envs=2,
            )
        self.assertEqual(
            float((torch.arange(32, device=device) * 2).sum().item()),
            992.0,
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_runtime_guards_input_mutation_and_restores_rng(self) -> None:
        import torch

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def build_inputs(program: QualifiedStairRewardProgram) -> StairRewardInputsV2:
            values = {}
            for field in program.contract.input_fields:
                dtype = torch.bool if field.dtype == "bool" else torch.float32
                values[field.name] = torch.zeros(
                    (2, *field.trailing_shape),
                    dtype=dtype,
                    device=device,
                )
            return StairRewardInputsV2._validated(
                contract=program.contract,
                values=values,
            )

        mutation_program = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        ).reward_program
        object.__setattr__(
            mutation_program,
            "reward_terms",
            (
                RewardTerm(
                    name="forced_mutation",
                    weight=1.0,
                    fn=lambda inputs: inputs.base_height.zero_(),
                ),
            ),
        )
        with self.assertRaisesRegex(StairRewardProgramError, "mutated its input"):
            evaluate_stair_reward_program(
                mutation_program,
                build_inputs(mutation_program),
                num_envs=2,
            )

        rng_program = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        ).reward_program
        object.__setattr__(
            rng_program,
            "reward_terms",
            (
                RewardTerm(
                    name="forced_rng",
                    weight=1.0,
                    fn=lambda inputs: torch.rand_like(inputs.base_height),
                ),
            ),
        )
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else ()
        )
        with self.assertRaisesRegex(StairRewardProgramError, "changed process RNG"):
            evaluate_stair_reward_program(
                rng_program,
                build_inputs(rng_program),
                num_envs=2,
            )
        self.assertTrue(torch.equal(cpu_rng_before, torch.random.get_rng_state()))
        cuda_rng_after = (
            tuple(torch.cuda.get_rng_state_all())
            if torch.cuda.is_available() and torch.cuda.is_initialized()
            else ()
        )
        self.assertEqual(len(cuda_rng_before), len(cuda_rng_after))
        for before, after in zip(
            cuda_rng_before,
            cuda_rng_after,
            strict=True,
        ):
            self.assertTrue(torch.equal(before, after))

        deterministic_program = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        ).reward_program
        self.assertEqual(
            assert_stair_reward_program_deterministic(
                deterministic_program,
                lambda: build_inputs(deterministic_program),
            )["status"],
            "pass",
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is unavailable")
    def test_stair_runtime_evidence_satisfies_shared_mechanism_contract(self) -> None:
        import torch

        program = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        ).reward_program
        values = {}
        for field in program.contract.input_fields:
            dtype = torch.bool if field.dtype == "bool" else torch.float32
            values[field.name] = torch.zeros(
                (2, *field.trailing_shape),
                dtype=dtype,
            )
        inputs = StairRewardInputsV2._validated(
            contract=program.contract,
            values=values,
        )
        dispatcher = StairCompleteRewardDispatcher(program, num_envs=2)
        dispatcher.compute(inputs, common_step=1)

        summary = dispatcher.runtime_summary()
        evidence = RuntimeActivationEvidence.from_mapping(summary)
        decision = MechanismEvidenceGate().evaluate(
            surface=CandidateSurfaceAssessment(
                classification="code_level",
                eligible_for_policy_accept=True,
                reason="complete_reward_program",
            ),
            evidence=evidence,
        )

        self.assertEqual(evidence.reward_term_calls, len(program.term_names))
        self.assertTrue(evidence.nonzero_reward_contribution)
        self.assertEqual(decision.status, "pass")
        self.assertEqual(
            evidence.capability_activation_records,
            ("complete_reward_program",),
        )

    def test_performance_compiler_rejects_semantic_or_hash_drift(self) -> None:
        resolved = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        )
        performance = self.config.protocol_v2.complete_reward_runtime.to_public_dict()[
            "performance_gate"
        ]
        equivalence = self.config.protocol_v2.complete_reward_runtime.to_public_dict()[
            "reference_equivalence"
        ]
        measured_steps = int(performance["measured_control_steps"])
        num_envs = int(performance["num_envs"])

        def case_report(name: str, throughput: float) -> dict[str, object]:
            elapsed = round(num_envs * measured_steps / throughput, 8)
            dispatcher = name != "official"
            expected_calls = int(performance["warmup_control_steps"]) + measured_steps
            report: dict[str, object] = {
                "schema_version": STAIR_REWARD_BENCHMARK_CASE_SCHEMA_VERSION,
                "status": "pass",
                "case": name,
                "protocol_v2_hash": self.config.protocol_v2.protocol_hash,
                "num_envs": num_envs,
                "warmup_control_steps": int(performance["warmup_control_steps"]),
                "measured_control_steps": measured_steps,
                "elapsed_seconds": elapsed,
                "env_steps_per_second": round(
                    num_envs * measured_steps / elapsed,
                    3,
                ),
                "reward_manager_active_terms": (
                    ["complete_reward"]
                    if dispatcher
                    else list(equivalence["active_term_names"])
                    + list(equivalence["zero_weight_term_names"])
                ),
                "training_recorder_enabled": (
                    name == "dispatcher_with_eval_recorder"
                ),
                "program": (
                    resolved.reward_program.to_public_dict()
                    if dispatcher
                    else None
                ),
                "runtime": (
                    {
                        "status": "pass",
                        "dispatcher_calls": expected_calls,
                        "term_calls": expected_calls
                        * len(resolved.reward_program.term_names),
                    }
                    if dispatcher
                    else {}
                ),
                "hardware_fingerprint": {
                    "device": "cuda:0",
                    "gpu_name": "test-gpu",
                    "torch": "test",
                    "simulation_dt_s": 0.005,
                    "control_dt_s": 0.02,
                },
            }
            report["report_hash"] = canonical_json_hash(report)
            return report

        cases = [
            case_report("official", 1000.0),
            case_report("dispatcher", 900.0),
            case_report("dispatcher_with_eval_recorder", 800.0),
        ]
        compiled = compile_stair_reward_performance_report(
            cases,
            protocol=self.config.protocol_v2,
            reference_program=resolved.reward_program,
        )
        self.assertEqual(compiled["status"], "pass")
        self.assertAlmostEqual(
            compiled["gate"]["observed_dispatcher_throughput_ratio"],
            0.9,
        )
        self.assertAlmostEqual(
            compiled[
                "diagnostic_recorder_to_dispatcher_throughput_ratio"
            ],
            8.0 / 9.0,
        )

        forged = copy.deepcopy(cases)
        forged[1]["training_recorder_enabled"] = True
        forged[1]["report_hash"] = canonical_json_hash(
            {key: value for key, value in forged[1].items() if key != "report_hash"}
        )
        with self.assertRaisesRegex(ValueError, "recorder state mismatch"):
            compile_stair_reward_performance_report(
                forged,
                protocol=self.config.protocol_v2,
                reference_program=resolved.reward_program,
            )


if __name__ == "__main__":
    unittest.main()
