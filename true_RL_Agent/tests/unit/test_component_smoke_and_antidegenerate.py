from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.authoring_checks import (  # noqa: E402
    assess_anti_degenerate_source,
    run_component_smoke_check,
)
from true_rl_agent.candidates.schema import (  # noqa: E402
    CandidateChangeSet,
    CandidatePackageRef,
)
from true_rl_agent.tasks.component_smoke import (  # noqa: E402
    ComponentSmokeFixture,
    SmokeTensor,
)
from true_rl_agent.tasks.push_recovery.reward_inputs import (
    PushRecoveryRewardInputs,  # noqa: E402
)


def _write_candidate(root: Path, candidate_id: str, components_py: str) -> CandidatePackageRef:
    candidate_dir = root / candidate_id
    (candidate_dir / "training_code").mkdir(parents=True)
    (candidate_dir / "candidate_manifest.yaml").write_text(
        "\n".join(
            [
                "schema_version: true_rl_agent.candidate.v1",
                f"candidate_id: {candidate_id}",
                "entrypoint: training_code.components:build_training_components",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (candidate_dir / "training_code" / "components.py").write_text(
        components_py,
        encoding="utf-8",
    )
    return CandidatePackageRef(
        candidate_id=candidate_id,
        path=str(candidate_dir),
        change_set=CandidateChangeSet(frozenset({"reward_callable"})),
    )


class _SmokeAdapter:
    task_id = "unit_test_task"

    def build_component_smoke_fixture(self) -> ComponentSmokeFixture:
        num_envs = 3
        vector = SmokeTensor.vector([0.35, 0.4, 0.45])
        matrix2 = SmokeTensor.matrix(
            [
                [0.0, 0.1],
                [0.1, 0.0],
                [0.0, -0.1],
            ]
        )
        matrix3 = SmokeTensor.matrix(
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ]
        )
        return ComponentSmokeFixture(
            task_id=self.task_id,
            reward_inputs=PushRecoveryRewardInputs(
                base_height=vector,
                roll=SmokeTensor.vector([0.01, -0.02, 0.03]),
                pitch=SmokeTensor.vector([0.02, 0.01, -0.01]),
                base_pos_xy=matrix2,
                base_lin_vel=matrix3,
                base_ang_vel=matrix3,
                projected_gravity=matrix3,
                commanded_base_vel=matrix3,
                joint_pos=matrix3,
                joint_vel=matrix3,
                joint_pos_error=matrix3,
                last_action=matrix3,
                prev_action=matrix3,
                action_delta=matrix3,
                push_active=SmokeTensor.vector([0.0, 1.0, 0.0]),
                push_vec_public=matrix2,
                progress_fraction=SmokeTensor.vector([0.1, 0.5, 0.9]),
            ),
            expected_reward_output_shape=(num_envs,),
            public_summary={"fixture_owner": "task_owned_test", "num_envs": num_envs},
        )


class ComponentSmokeAndAntiDegenerateTests(unittest.TestCase):
    def test_smoke_tensor_supports_torch_like_masks_and_common_math(self) -> None:
        heights = SmokeTensor.vector([0.2, 0.3, 0.4])

        low_mask = (heights < 0.3).float()
        high_mask = (heights >= 0.3).float()
        positive_excess = (heights - 0.25 > 0.0).float()
        square = heights.square()
        roots = SmokeTensor.vector([4.0, 9.0]).sqrt()
        pow_method = heights.pow(2)
        relu = (heights - 0.3).relu()
        clamp_min = (0.3 - heights).clamp_min(0.0)
        clamp_max = heights.clamp_max(0.3)
        row_norm = SmokeTensor.matrix([[3.0, 4.0], [5.0, 12.0]]).norm(p=2, dim=-1)
        row_sum = SmokeTensor.matrix([[1.0, 2.0], [3.0, 4.0]]).sum(dim=-1, keepdim=False)
        row_mean = SmokeTensor.matrix([[1.0, 3.0], [3.0, 7.0]]).mean(dim=-1, keepdim=False)

        self.assertEqual(low_mask.values, (1.0, 0.0, 0.0))
        self.assertEqual(high_mask.values, (0.0, 1.0, 1.0))
        self.assertEqual(positive_excess.values, (0.0, 1.0, 1.0))
        self.assertEqual(square.values, (0.04000000000000001, 0.09, 0.16000000000000003))
        self.assertEqual(roots.values, (2.0, 3.0))
        self.assertEqual(pow_method.values, square.values)
        self.assertEqual(relu.values, (0.0, 0.0, 0.10000000000000003))
        self.assertEqual(clamp_min.values, (0.09999999999999998, 0.0, 0.0))
        self.assertEqual(clamp_max.values, (0.2, 0.3, 0.3))
        self.assertEqual(row_norm.values, (5.0, 13.0))
        self.assertEqual(row_sum.values, (3.0, 7.0))
        self.assertEqual(row_mean.values, (2.0, 5.0))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            bool(heights > 0.3)

    def test_smoke_tensor_supports_standard_ellipsis_indexing(self) -> None:
        values = SmokeTensor.matrix(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )

        self.assertEqual(values[..., :2], values[:, :2])
        self.assertEqual(values[..., 1], SmokeTensor.vector([2.0, 5.0]))
        self.assertEqual(values[1, ...], SmokeTensor.vector([4.0, 5.0, 6.0]))
        self.assertEqual(values[1, 2], 6.0)

    def test_component_smoke_accepts_valid_ellipsis_tensor_slices(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def velocity_reward(inp):
    error = inp.commanded_base_vel[..., :2] - inp.base_lin_vel[..., :2]
    return -(error * error).sum(dim=-1)

def build_training_components():
    return TrainingComponents(
        name="ellipsis_velocity_reward",
        reward_terms=[RewardTerm(name="velocity_reward", weight=1.0, fn=velocity_reward)],
        rationale="Use normal PyTorch ellipsis slicing.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-ellipsis-ok", source)
            result = run_component_smoke_check(candidate, _SmokeAdapter())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checked_reward_terms"], 1)

    def test_component_smoke_executes_task_owned_fixture(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def height_reward(inp):
    return inp.base_height * 0.5

def build_training_components():
    return TrainingComponents(
        name="height_reward",
        reward_terms=[RewardTerm(name="height_reward", weight=1.0, fn=height_reward)],
        rationale="Smoke-valid reward callable.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-ok", source)
            result = run_component_smoke_check(candidate, _SmokeAdapter())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checked_reward_terms"], 1)
        self.assertEqual(result["task_id"], "unit_test_task")
        self.assertEqual(result["fixture_summary"]["fixture_owner"], "task_owned_test")

    def test_component_smoke_accepts_torch_like_comparison_masks(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def comparison_mask_reward(inp):
    low_height = (inp.base_height < 0.38).float()
    exceed = (inp.base_height - 0.37 > 0.0).float()
    return low_height * 0.25 + exceed * 0.1

def build_training_components():
    return TrainingComponents(
        name="comparison_mask_reward",
        reward_terms=[RewardTerm(name="comparison_mask_reward", weight=1.0, fn=comparison_mask_reward)],
        rationale="Smoke-valid torch-like comparison mask reward callable.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-mask-ok", source)
            result = run_component_smoke_check(candidate, _SmokeAdapter())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checked_reward_terms"], 1)

    def test_component_smoke_accepts_common_torch_tensor_method_aliases(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def method_alias_reward(inp):
    speed = inp.base_lin_vel[:, :2].norm(p=2, dim=-1)
    speed_excess = (speed - 0.15).clamp_min(0.0)
    tilt_energy = (inp.roll.pow(2) + inp.pitch.pow(2)).sqrt().clamp_max(0.5)
    smoothness = inp.action_delta.pow(2).sum(dim=-1, keepdim=False)
    return speed_excess.relu() + tilt_energy + smoothness * 0.01

def build_training_components():
    return TrainingComponents(
        name="method_alias_reward",
        reward_terms=[RewardTerm(name="method_alias_reward", weight=1.0, fn=method_alias_reward)],
        rationale="Smoke-valid torch-like tensor method aliases.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-method-alias-ok", source)
            result = run_component_smoke_check(candidate, _SmokeAdapter())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checked_reward_terms"], 1)

    def test_component_smoke_failure_feedback_is_public_safe(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def broken_reward(inp):
    raise RuntimeError("D:/humanoid_agent/private/raw.txt /data/yuyangliu/secrets/key.env sk-live-secret")

def build_training_components():
    return TrainingComponents(
        name="broken_reward",
        reward_terms=[RewardTerm(name="broken_reward", weight=1.0, fn=broken_reward)],
        rationale="Broken reward callable.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-bad", source)
            result = run_component_smoke_check(candidate, _SmokeAdapter())

        text = json.dumps(result, sort_keys=True)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["reason"], "reward_callable_failed")
        self.assertNotIn("D:/humanoid_agent", text)
        self.assertNotIn("/data/yuyangliu", text)
        self.assertNotIn("sk-live-secret", text)

    def test_component_smoke_rejects_diagnostic_callable_failure(self) -> None:
        source = """
from true_rl_agent.candidates.components import DiagnosticTerm, RewardTerm, TrainingComponents

def height_reward(inp):
    return inp.base_height * 0.5

def broken_diagnostic(inp):
    return {"mean_velocity": missing_velocity.mean()}

def build_training_components():
    return TrainingComponents(
        name="broken_diagnostic",
        reward_terms=[RewardTerm(name="height_reward", weight=1.0, fn=height_reward)],
        diagnostics=[DiagnosticTerm(name="broken_diagnostic", fn=broken_diagnostic)],
        rationale="A broken diagnostic must be repaired before training.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-broken-diagnostic", source)
            result = run_component_smoke_check(candidate, _SmokeAdapter())

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["reason"], "diagnostic_callable_failed")
        self.assertEqual(result["checked_reward_terms"], 1)
        self.assertEqual(result["checked_diagnostics"], 1)
        self.assertEqual(result["diagnostic_results"][0]["status"], "fail")
        self.assertEqual(result["diagnostic_results"][0]["error_type"], "NameError")

    def test_component_smoke_unsupported_output_becomes_repair_feedback(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def unsupported_reward(inp):
    return object()

def build_training_components():
    return TrainingComponents(
        name="unsupported_reward",
        reward_terms=[RewardTerm(name="unsupported_reward", weight=1.0, fn=unsupported_reward)],
        rationale="Unsupported output should fail smoke without raising.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-unsupported", source)
            result = run_component_smoke_check(candidate, _SmokeAdapter())

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["reason"], "reward_output_unsupported")

    def test_component_smoke_timeout_becomes_repair_feedback(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def hanging_reward(inp):
    while True:
        pass

def build_training_components():
    return TrainingComponents(
        name="hanging_reward",
        reward_terms=[RewardTerm(name="hanging_reward", weight=1.0, fn=hanging_reward)],
        rationale="Infinite loop should be terminated by smoke timeout.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-hanging", source)
            result = run_component_smoke_check(candidate, _SmokeAdapter(), timeout_s=0.25)

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["reason"], "reward_callable_timeout")
        self.assertEqual(result["reward_results"][0]["status"], "fail")
        self.assertEqual(result["reward_results"][0]["reason"], "reward_callable_timeout")
        self.assertLessEqual(result["reward_results"][0]["timeout_s"], 0.25)

    def test_anti_degenerate_source_classification(self) -> None:
        current_best = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def reward(inp):
    return inp.base_height * 0.5

def build_training_components():
    return TrainingComponents(name="best", reward_terms=[RewardTerm(name="height", weight=1.0, fn=reward)])
"""
        identical = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

# formatting-only comment
def reward(inp):
    return inp.base_height * 0.5

def build_training_components():
    return TrainingComponents(name="best", reward_terms=[RewardTerm(name="height", weight=1.0, fn=reward)])
"""
        numeric_only = current_best.replace("0.5", "0.7").replace("1.0", "2.0")
        mechanism = current_best.replace(
            "return inp.base_height * 0.5",
            "return inp.base_height * 0.5 - abs(inp.roll)",
        )

        self.assertEqual(
            assess_anti_degenerate_source(
                candidate_source=identical,
                current_best_source=current_best,
            )["classification"],
            "identical_or_format_only",
        )
        numeric_assessment = assess_anti_degenerate_source(
            candidate_source=numeric_only,
            current_best_source=current_best,
        )
        self.assertEqual(numeric_assessment["classification"], "parameter_only_delta")
        self.assertEqual(numeric_assessment["status"], "warning")
        numeric_and_rationale_only = (
            current_best.replace("0.5", "0.7")
            .replace('name="best"', 'name="renamed"')
            .replace('name="height"', 'name="renamed_height"')
        )
        mixed_assessment = assess_anti_degenerate_source(
            candidate_source=numeric_and_rationale_only,
            current_best_source=current_best,
        )
        self.assertEqual(
            mixed_assessment["classification"],
            "parameter_and_rationale_only_delta",
        )
        self.assertEqual(mixed_assessment["status"], "warning")
        self.assertEqual(
            assess_anti_degenerate_source(
                candidate_source=mechanism,
                current_best_source=current_best,
            )["classification"],
            "mechanism_delta_present",
        )


if __name__ == "__main__":
    unittest.main()
