from __future__ import annotations

import copy
import importlib.util
import math
import sys
import textwrap
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
from true_rl_agent.evaluation.subject import ExperimentSubject  # noqa: E402
from true_rl_agent.tasks.stair_climbing.candidate_failure import (  # noqa: E402
    STAIR_CANDIDATE_FAILURE_CONTRACT,
    StairCandidateProgramFailure,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import (  # noqa: E402
    _execute_stair_reward_program_boundary,
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
    StairRewardOutputContractError,
    StairRewardProgramError,
    assert_stair_reward_program_deterministic,
    evaluate_stair_reward_program,
    qualify_stair_reference_reward_program,
)
from true_rl_agent.tasks.stair_climbing.reward_source_semantics import (  # noqa: E402
    STAIR_REWARD_SOURCE_SEMANTICS_SCHEMA_VERSION,
    audit_stair_reward_source_semantics,
    stair_reward_source_semantics_policy_hash,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
)


def _single_reward_source(
    return_statement: str,
    *,
    term_args: str = "",
    helper: str = "",
) -> str:
    weight = "" if "weight=" in term_args else ", weight=-0.01"
    sections = [
        "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents",
    ]
    if helper.strip():
        sections.append(textwrap.dedent(helper).strip())
    sections.extend(
        (
            "def reward(inputs):\n"
            + textwrap.indent(textwrap.dedent(return_statement).strip(), "    "),
            textwrap.dedent(
                f"""
                def build_training_components():
                    return TrainingComponents(
                        name="raw-physical-probe",
                        reward_terms=[RewardTerm(name="probe", fn=reward{weight}{term_args})],
                        diagnostics=[],
                        schedules=[],
                        metadata={{}},
                        rationale="test",
                    )
                """
            ).strip(),
        )
    )
    return "\n\n\n".join(sections) + "\n"


def _project_world_gravity_into_base(
    *,
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float]:
    """Independently apply R_WB.T to the normalized world gravity vector."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr))
    rotation_y = ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp))
    rotation_z = ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0))

    def multiply(
        left: tuple[tuple[float, ...], ...],
        right: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(
                sum(left[row][inner] * right[inner][column] for inner in range(3))
                for column in range(3)
            )
            for row in range(3)
        )

    rotation_world_from_base = multiply(rotation_z, multiply(rotation_y, rotation_x))
    gravity_world = (0.0, 0.0, -1.0)
    return tuple(
        sum(
            rotation_world_from_base[row][column] * gravity_world[row]
            for row in range(3)
        )
        for column in range(3)
    )


class StairCompleteRewardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        cls.official_source = (
            ROOT
            / "src"
            / "true_rl_agent"
            / "tasks"
            / "stair_climbing"
            / "reference_programs"
            / "official_g1_rough_v1"
            / "training_code"
            / "components.py"
        ).read_text(encoding="utf-8")

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
            prompt["schema_version"],
            "true_rl_agent.reward_inputs_contract.stair.v12",
        )
        self.assertLess(
            list(prompt).index("field_semantics_constraints"),
            list(prompt).index("fields"),
        )
        self.assertEqual(
            [item["name"] for item in prompt["fields"]],
            list(contract.field_names),
        )
        raw_magnitudes = prompt["field_semantics_constraints"][
            "raw_physical_magnitudes"
        ]
        self.assertEqual(raw_magnitudes["normalization_status"], "raw_not_normalized")
        self.assertIn("base_contact_force", raw_magnitudes["fields"])
        self.assertEqual(
            raw_magnitudes["enforcement"],
            "active_reward_term_final_clip",
        )
        self.assertIn("clip_min", raw_magnitudes["candidate_usage"])
        self.assertIn("clip_max", raw_magnitudes["candidate_usage"])
        axes = prompt["field_semantics_constraints"][
            "high_risk_axis_semantics"
        ]
        self.assertIn("independent environments", axes["environment"])
        self.assertIn("output[i] only from input[i]", axes["environment"])
        self.assertIn("declared history fields", axes["environment"])
        self.assertIn("0 newest", axes["foot_contact_force_history"])
        self.assertIn("0 left", axes["foot_contact_force_history"])
        self.assertIn("max all 3 history", axes["foot_contact_force_history"])
        self.assertIn("does not identify every onset", axes["foot_contact_force_history"])
        self.assertIn("world +x stair-forward", axes["spatial_vectors"])
        remaining = prompt["field_semantics_constraints"][
            "remaining_goal_distance_m"
        ]
        self.assertEqual(remaining["source_equivalence"], "goal_x-current_x")
        self.assertIn("negative beyond", remaining["sign"])
        projected_gravity = prompt["field_semantics_constraints"][
            "projected_gravity"
        ]
        self.assertEqual(
            projected_gravity["reference_frame"],
            "FLU base frame (+x forward,+y left,+z up)",
        )
        self.assertEqual(
            projected_gravity["definition"],
            (
                "right-handed R_WB=Rz(yaw)@Ry(pitch)@Rx(roll); "
                "g_b=R_WB.T@[0,0,-1]"
            ),
        )
        self.assertEqual(
            projected_gravity["components_xyz"],
            [
                "sin(pitch)",
                "-sin(roll)*cos(pitch)",
                "-cos(roll)*cos(pitch)",
            ],
        )
        self.assertEqual(
            projected_gravity["small_angle_xyz"],
            ["pitch", "-roll", "-1"],
        )
        angle = 0.2
        cases = {
            "positive_pitch": ((0.0, angle), (0.19866933, 0.0, -0.98006658)),
            "negative_pitch": ((0.0, -angle), (-0.19866933, 0.0, -0.98006658)),
            "positive_roll": ((angle, 0.0), (0.0, -0.19866933, -0.98006658)),
            "negative_roll": ((-angle, 0.0), (0.0, 0.19866933, -0.98006658)),
        }
        for (roll, pitch), expected in cases.values():
            actual = _project_world_gravity_into_base(
                roll=roll,
                pitch=pitch,
                yaw=0.73,
            )
            for actual_component, expected_component in zip(
                actual,
                expected,
                strict=True,
            ):
                self.assertAlmostEqual(actual_component, expected_component, places=8)
        self.assertIn("inversion", projected_gravity["candidate_usage"])
        height_scan = prompt["field_semantics_constraints"]["height_scan"]
        self.assertEqual(
            height_scan["source_transform_ref"],
            "source_transforms.height_scan",
        )
        self.assertIn("downward terrain scan", height_scan["sensor_semantics"])
        self.assertIn("positive=lower/farther", height_scan["sensor_semantics"])
        self.assertIn("negative=higher/closer", height_scan["sensor_semantics"])
        self.assertIn("no overhead or torso", height_scan["sensor_semantics"])
        self.assertIn("order-invariant", height_scan["candidate_usage"])
        base_contact = prompt["field_semantics_constraints"]["base_contact_force"]
        self.assertEqual(base_contact["binary_alternative"], "base_contact")
        self.assertIn("raw_physical_magnitudes", base_contact["candidate_usage"])
        termination_masks = prompt["field_semantics_constraints"][
            "termination_masks"
        ]
        self.assertEqual(
            termination_masks["fixed_active_terms"],
            ["time_out", "base_contact"],
        )
        self.assertEqual(
            termination_masks["equivalent_fields"],
            {"terminated_non_timeout": "base_contact"},
        )
        self.assertIn(
            "identical masks",
            termination_masks["candidate_usage"],
        )
        progress_fraction = prompt["field_semantics_constraints"][
            "progress_fraction"
        ]
        self.assertEqual(
            progress_fraction["spatial_alternative"],
            "goal_progress_fraction",
        )
        self.assertIn("episode-time", progress_fraction["candidate_usage"])
        self.assertIn("does not", progress_fraction["candidate_usage"])
        goal_progress = prompt["field_semantics_constraints"][
            "goal_progress_fraction"
        ]
        self.assertIn("instantaneous", goal_progress["semantics"])
        self.assertIn("may decrease", goal_progress["temporal_behavior"])
        self.assertIn("not latched", goal_progress["temporal_behavior"])
        base_height = prompt["field_semantics_constraints"]["base_height"]
        self.assertEqual(
            base_height["source_equivalence"],
            "base_pos_relative_to_terrain_origin[:, 2]",
        )
        self.assertFalse(base_height["local_clearance"])
        self.assertIn("supporting step", base_height["candidate_usage"])
        self.assertIn("goal_progress_fraction", base_height["candidate_usage"])
        opaque_axes = prompt["opaque_axis_contract"]
        self.assertEqual(
            opaque_axes["schema_version"],
            STAIR_REWARD_SOURCE_SEMANTICS_SCHEMA_VERSION,
        )
        groups = opaque_axes["opaque_trailing_axis_groups"]
        self.assertEqual([group["axes"] for group in groups], [[0]])
        self.assertIn("joint_pos_error", groups[0]["fields"])
        self.assertIn("height_scan", groups[0]["fields"])
        self.assertIn("Member mapping is hidden", opaque_axes["usage_rule"])
        self.assertIn("fixed or partial indexing", opaque_axes["usage_rule"])
        serialized = repr(prompt).lower()
        for forbidden in ("raw_seed", "bin_identity", "checkpoint_local_path"):
            self.assertNotIn(forbidden, serialized)

    def test_opaque_axis_source_policy_rejects_invented_member_semantics(self) -> None:
        contract = self.config.protocol_v2.complete_reward_runtime
        rejected_sources = {
            "direct_joint_slice": """
                def reward(inputs):
                    return inputs.joint_pos_error[:, :12].abs().sum(dim=1)
            """,
            "alias_joint_slice": """
                def reward(inputs):
                    values = inputs.joint_pos_error
                    return values[:, :12].abs().sum(dim=1)
            """,
            "helper_joint_slice": """
                def first_members(values):
                    return values[:, :12]

                def reward(inputs):
                    return first_members(inputs.joint_pos_error).abs().sum(dim=1)
            """,
            "nested_helper_joint_slice": """
                def reward(inputs):
                    def first_members(values):
                        return values[:, :12]
                    return first_members(inputs.joint_pos_error).abs().sum(dim=1)
            """,
            "fixed_height_ray": """
                def reward(inputs):
                    return inputs.height_scan[:, 4]
            """,
            "positional_method": """
                def reward(inputs):
                    return inputs.height_scan.topk(4, dim=1).values.sum(dim=1)
            """,
            "unlisted_axis_method": """
                def reward(inputs):
                    return inputs.joint_pos_error.roll(1, dims=1).abs().sum(dim=1)
            """,
            "positional_weight_constructor": """
                def reward(inputs):
                    weights = inputs.base_height.new_tensor(
                        [1.0] * 12 + [0.0] * 25
                    )
                    return (inputs.joint_pos_error * weights).abs().sum(dim=1)
            """,
        }
        for name, source in rejected_sources.items():
            with self.subTest(name=name):
                issues = audit_stair_reward_source_semantics(
                    textwrap.dedent(source),
                    contract=contract,
                    trusted_reference_source_text=self.official_source,
                )
                self.assertTrue(issues)
                self.assertTrue(
                    any(issue.startswith("opaque_input_") for issue in issues)
                )

        accepted_sources = (
            """
                def reward(inputs):
                    return inputs.joint_pos_error.abs().sum(dim=1)
            """,
            """
                def reward(inputs):
                    lower = inputs.ankle_soft_joint_pos_limits[:, :, 0]
                    upper = inputs.ankle_soft_joint_pos_limits[:, :, 1]
                    error = (inputs.ankle_joint_pos - lower).clamp(max=0.0)
                    return (error + upper * 0.0).sum(dim=1)
            """,
            """
                def reward(inputs):
                    return inputs.commanded_base_vel[:, 0]
            """,
        )
        for source in accepted_sources:
            self.assertEqual(
                audit_stair_reward_source_semantics(
                    textwrap.dedent(source),
                    contract=contract,
                    trusted_reference_source_text=self.official_source,
                ),
                (),
            )

        self.assertEqual(
            audit_stair_reward_source_semantics(
                self.official_source,
                contract=contract,
                trusted_reference_source_text=self.official_source,
            ),
            (),
        )

    def test_raw_physical_reward_terms_require_a_final_declarative_bound(self) -> None:
        contract = self.config.protocol_v2.complete_reward_runtime
        cases = {
            "r2_unbounded": (
                "return (inputs.base_contact_force.norm(dim=-1) - 1.0).clamp(min=0.0)",
                "",
                "raw_physical_reward_requires_finite_two_sided_clip",
            ),
            "tiny_weight_is_not_a_bound": (
                "return inputs.base_contact_force.norm(dim=-1)",
                ", weight=-1e-12",
                "raw_physical_reward_requires_finite_two_sided_clip",
            ),
            "callable_tanh_is_not_the_final_contract": (
                "return inputs.base_contact_force.norm(dim=-1).tanh()",
                "",
                "raw_physical_reward_requires_finite_two_sided_clip",
            ),
            "one_sided_clip": (
                "return inputs.base_contact_force.norm(dim=-1)",
                ", clip_max=100.0",
                "raw_physical_reward_requires_finite_two_sided_clip",
            ),
            "equal_clip": (
                "return inputs.base_contact_force.norm(dim=-1)",
                ", clip_min=1.0, clip_max=1.0",
                "raw_physical_reward_clip_order_invalid",
            ),
            "output_contract_exceeded": (
                "return inputs.base_contact_force.norm(dim=-1)",
                ", weight=100001.0, clip_min=-1.0, clip_max=1.0",
                "raw_physical_reward_weighted_bound_exceeds_output_contract",
            ),
        }
        for name, (body, term_args, expected_issue) in cases.items():
            with self.subTest(name=name):
                issues = audit_stair_reward_source_semantics(
                    _single_reward_source(body, term_args=term_args),
                    contract=contract,
                    trusted_reference_source_text=self.official_source,
                )
                self.assertTrue(
                    any(issue.startswith(expected_issue) for issue in issues),
                    issues,
                )

        dynamic_clip = _single_reward_source(
            "return inputs.base_contact_force.norm(dim=-1)",
            term_args=", clip_min=LOWER, clip_max=UPPER",
            helper="LOWER = 0.0\nUPPER = 100.0",
        )
        self.assertIn(
            "raw_physical_reward_component_projection_unavailable",
            audit_stair_reward_source_semantics(
                dynamic_clip,
                contract=contract,
                trusted_reference_source_text=self.official_source,
            ),
        )

        bounded = _single_reward_source(
            "return inputs.base_contact_force.norm(dim=-1)",
            term_args=", clip_min=0.0, clip_max=100.0",
        )
        self.assertEqual(
            audit_stair_reward_source_semantics(
                bounded,
                contract=contract,
                trusted_reference_source_text=self.official_source,
            ),
            (),
        )
        nonraw = _single_reward_source("return inputs.goal_progress_fraction")
        self.assertEqual(
            audit_stair_reward_source_semantics(
                nonraw,
                contract=contract,
                trusted_reference_source_text=self.official_source,
            ),
            (),
        )

    def test_raw_physical_reference_exception_is_exact_not_name_based(self) -> None:
        contract = self.config.protocol_v2.complete_reward_runtime
        modified = self.official_source.replace(
            "weight=-1.25e-7,",
            "weight=-2.5e-7,",
            1,
        )
        self.assertNotEqual(modified, self.official_source)
        issues = audit_stair_reward_source_semantics(
            modified,
            contract=contract,
            trusted_reference_source_text=self.official_source,
        )
        self.assertTrue(
            any(
                issue.startswith(
                    "raw_physical_reward_requires_finite_two_sided_clip:"
                    "RewardTerm:dof_acc_l2"
                )
                for issue in issues
            ),
            issues,
        )
        self.assertNotEqual(
            stair_reward_source_semantics_policy_hash(
                contract,
                trusted_reference_source_text=self.official_source,
            ),
            stair_reward_source_semantics_policy_hash(
                contract,
                trusted_reference_source_text=modified,
            ),
        )

    def test_raw_physical_dependency_through_helper_is_bounded(self) -> None:
        contract = self.config.protocol_v2.complete_reward_runtime
        source = _single_reward_source(
            "return force_norm(inputs)",
            term_args=", clip_min=0.0, clip_max=50.0",
            helper="""
                def force_norm(inputs):
                    return inputs.base_contact_force.norm(dim=-1)
            """,
        )
        self.assertEqual(
            audit_stair_reward_source_semantics(
                source,
                contract=contract,
                trusted_reference_source_text=self.official_source,
            ),
            (),
        )

    def test_raw_physical_input_aliases_cannot_bypass_final_clip(self) -> None:
        contract = self.config.protocol_v2.complete_reward_runtime
        direct_alias = _single_reward_source(
            "alias = inputs\nreturn alias.base_contact_force.norm(dim=-1)",
        )
        helper_alias = _single_reward_source(
            "alias = inputs\nreturn force_norm(alias)",
            helper="""
                def force_norm(inputs):
                    alias = inputs
                    return alias.base_contact_force.norm(dim=-1)
            """,
        )
        nonfirst_helper_argument = _single_reward_source(
            "return force_norm(1.0, inputs)",
            helper="""
                def force_norm(scale, value):
                    return value.base_contact_force.norm(dim=-1) * scale
            """,
        )
        keyword_helper_argument = _single_reward_source(
            "return force_norm(scale=1.0, value=inputs)",
            helper="""
                def force_norm(scale, value):
                    return value.base_contact_force.norm(dim=-1) * scale
            """,
        )
        list_alias = _single_reward_source(
            "alias = [inputs][0]\nreturn alias.base_contact_force.norm(dim=-1)",
        )
        tuple_alias = _single_reward_source(
            "alias, = (inputs,)\nreturn alias.base_contact_force.norm(dim=-1)",
        )
        conditional_alias = _single_reward_source(
            "alias = inputs if inputs.base_height > 0 else inputs\n"
            "return alias.base_contact_force.norm(dim=-1)",
        )
        varargs_helper = _single_reward_source(
            "return force_norm(inputs)",
            helper="""
                def force_norm(*values):
                    return values[0].base_contact_force.norm(dim=-1)
            """,
        )

        for name, source in {
            "direct_alias": direct_alias,
            "helper_alias": helper_alias,
            "nonfirst_helper_argument": nonfirst_helper_argument,
            "keyword_helper_argument": keyword_helper_argument,
            "list_alias": list_alias,
            "tuple_alias": tuple_alias,
            "conditional_alias": conditional_alias,
            "varargs_helper": varargs_helper,
        }.items():
            with self.subTest(name=name):
                issues = audit_stair_reward_source_semantics(
                    source,
                    contract=contract,
                    trusted_reference_source_text=self.official_source,
                )
                self.assertTrue(
                    any(
                        issue.startswith(
                            "raw_physical_reward_requires_finite_two_sided_clip:"
                        )
                        for issue in issues
                    ),
                    issues,
                )

        dead_raw_helper = _single_reward_source(
            "return inputs.goal_progress_fraction",
            helper="""
                def unused(inputs):
                    return inputs.base_contact_force.norm(dim=-1)
            """,
        )
        self.assertEqual(
            audit_stair_reward_source_semantics(
                dead_raw_helper,
                contract=contract,
                trusted_reference_source_text=self.official_source,
            ),
            (),
        )

    def test_reward_failure_boundary_is_subject_and_stage_exact(self) -> None:
        reference_program = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        ).reward_program
        candidate_program = resolve_official_stair_reference_program(
            self.config,
            project_root=ROOT,
        ).reward_program
        object.__setattr__(
            candidate_program,
            "subject",
            ExperimentSubject.candidate("candidate-failure-probe"),
        )

        def rejected() -> None:
            raise StairRewardProgramError("private candidate exception")

        expected = {
            "activation_preflight": False,
            "ppo_reward_execution": True,
        }
        for stage, requires_initial_state in expected.items():
            with self.subTest(stage=stage):
                with self.assertRaises(StairCandidateProgramFailure) as context:
                    _execute_stair_reward_program_boundary(
                        rejected,
                        reward_program=candidate_program,
                        stage=stage,
                    )
                evidence = context.exception.evidence
                self.assertEqual(evidence.failure_stage, stage)
                rule = STAIR_CANDIDATE_FAILURE_CONTRACT.rule_for(
                    stage=evidence.failure_stage,
                    reason_code=evidence.reason_code,
                )
                self.assertIs(
                    rule.requires_initial_policy_state,
                    requires_initial_state,
                )
                self.assertNotIn(
                    "private candidate exception",
                    str(context.exception),
                )

        with self.assertRaisesRegex(
            StairRewardProgramError,
            "private candidate exception",
        ):
            _execute_stair_reward_program_boundary(
                rejected,
                reward_program=reference_program,
                stage="ppo_reward_execution",
            )

        with self.assertRaisesRegex(RuntimeError, "CUDA error"):
            _execute_stair_reward_program_boundary(
                lambda: (_ for _ in ()).throw(
                    RuntimeError("CUDA error: device-side assert")
                ),
                reward_program=candidate_program,
                stage="ppo_reward_execution",
            )

        with self.assertRaises(StairCandidateProgramFailure) as context:
            _execute_stair_reward_program_boundary(
                lambda: (_ for _ in ()).throw(
                    StairRewardOutputContractError("private output details")
                ),
                reward_program=candidate_program,
                stage="ppo_reward_execution",
            )
        self.assertEqual(
            context.exception.evidence.reason_code,
            "reward_program_output_contract_rejected",
        )
        self.assertNotIn("private output details", str(context.exception))

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

    def test_runtime_profiles_separate_training_and_evaluation_identity(self) -> None:
        train_profile = self.config.train_runtime_profile().to_public_dict()
        eval_profile = self.config.eval_runtime_profile().to_public_dict()

        self.assertEqual(
            train_profile["complete_reward_runtime"],
            STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
        )
        self.assertEqual(
            train_profile["task_protocol_hash"],
            self.config.protocol_v2.protocol_hash,
        )
        self.assertEqual(
            eval_profile["task_protocol_hash"],
            self.config.task_protocol_hash,
        )

    def test_rsl_rl_execution_configuration_is_strict_and_single_source(self) -> None:
        configuration = self.config.training_agent_configuration()

        class FakeAgentCfg:
            def __init__(self) -> None:
                self.policy = SimpleNamespace(**dict(configuration.policy))
                self.algorithm = SimpleNamespace(**dict(configuration.algorithm))

            def to_dict(self) -> dict[str, object]:
                return {
                    "actor": self.actor,
                    "critic": self.critic,
                    "algorithm": self.algorithm,
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
            execution_contract=self.config.protocol_v2.to_public_dict()[
                "training_execution"
            ],
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
                execution_contract=self.config.protocol_v2.to_public_dict()[
                    "training_execution"
                ],
            )

    def test_production_protocol_explicitly_binds_runner_observation_groups(self) -> None:
        config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        configuration = config.training_agent_configuration()

        class FakeAgentCfg:
            def __init__(self) -> None:
                self.policy = SimpleNamespace(**dict(configuration.policy))
                self.algorithm = SimpleNamespace(**dict(configuration.algorithm))

            def to_dict(self) -> dict[str, object]:
                return {
                    "actor": self.actor,
                    "critic": self.critic,
                    "algorithm": self.algorithm,
                }

        execution = config.protocol_v2.to_public_dict()["training_execution"]
        effective = configure_stair_rsl_rl_agent(
            FakeAgentCfg(),
            configuration=configuration,
            seed=17,
            device="cuda:0",
            num_steps_per_env=8,
            max_iterations=1,
            save_interval=1,
            run_name="phase_d_smoke",
            experiment_name="stair_test",
            execution_contract=execution,
        )
        self.assertEqual(effective["runner_contract_mode"], "explicit")
        self.assertEqual(
            effective["observation_groups"],
            {"policy": ["policy"], "critic": ["policy"]},
        )
        self.assertIsNone(effective["clip_actions"])
        self.assertFalse(effective["resume"])

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
            StairRewardOutputContractError,
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
    def test_resource_failures_are_not_wrapped_as_reward_program_failures(self) -> None:
        import torch

        def inputs_for(program: QualifiedStairRewardProgram) -> StairRewardInputsV2:
            values = {}
            for field in program.contract.input_fields:
                dtype = torch.bool if field.dtype == "bool" else torch.float32
                values[field.name] = torch.zeros(
                    (2, *field.trailing_shape),
                    dtype=dtype,
                )
            return StairRewardInputsV2._validated(
                contract=program.contract,
                values=values,
            )

        cases = (
            MemoryError("allocator exhausted"),
            OSError("filesystem unavailable"),
            RuntimeError("CUDA error: illegal memory access"),
        )
        for error in cases:
            with self.subTest(error=type(error).__name__):
                program = resolve_official_stair_reference_program(
                    self.config,
                    project_root=ROOT,
                ).reward_program

                def fail(_inputs, *, selected=error):
                    raise selected

                object.__setattr__(
                    program,
                    "reward_terms",
                    (RewardTerm(name="forced_failure", weight=1.0, fn=fail),),
                )
                with self.assertRaises(type(error)):
                    evaluate_stair_reward_program(
                        program,
                        inputs_for(program),
                        num_envs=2,
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
        self.assertTrue(all(record["last_sample_common_step"] == 1 and record["sampled_env_count"] == 2
                            for record in evidence.component_activation_records))
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
