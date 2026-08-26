from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.authoring_checks import (  # noqa: E402
    ANTI_DEGENERATE_SCHEMA_VERSION,
    COMPONENT_SMOKE_SCHEMA_VERSION,
    SingleComponentDeltaComparisonUnavailable,
    assess_anti_degenerate_candidate,
    assess_anti_degenerate_source,
    compare_single_component_deltas,
    run_component_smoke_check,
)
from true_rl_agent.candidates.authoring_policy import (  # noqa: E402
    ANTI_DEGENERATE_ASSESSMENT_SCHEMA_VERSION,
    canonical_mechanism_runtime_identity,
    format_canonical_mechanism_id,
    require_canonical_mechanism_delta,
)
from true_rl_agent.candidates.component_smoke_runtime import (  # noqa: E402
    COMPONENT_SMOKE_SCHEMA_VERSION as RUNTIME_COMPONENT_SMOKE_SCHEMA_VERSION,
)
from true_rl_agent.candidates.component_source_semantics import (  # noqa: E402
    extract_training_components_rationale,
    project_single_component_operation_delta,
)
from true_rl_agent.candidates.isolated_worker import (  # noqa: E402
    IsolatedWorkerStartupError,
)
from true_rl_agent.candidates.loader import load_training_components  # noqa: E402
from true_rl_agent.candidates.package_contract import (  # noqa: E402
    open_preflighted_candidate_file,
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
    (candidate_dir / "rationale.md").write_text(
        "Candidate test rationale.\n",
        encoding="utf-8",
    )
    (candidate_dir / "operation_diff.json").write_text("{}\n", encoding="utf-8")
    (candidate_dir / "diagnostics.json").write_text("{}\n", encoding="utf-8")
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


def _restore_delayed_reward_inputs(delay_s: float, reward_inputs: object) -> object:
    time.sleep(delay_s)
    return reward_inputs


class _DelayedPickleRewardInputs:
    def __init__(self, delay_s: float, reward_inputs: object) -> None:
        self.delay_s = delay_s
        self.reward_inputs = reward_inputs

    def __reduce__(self):
        return (
            _restore_delayed_reward_inputs,
            (self.delay_s, self.reward_inputs),
        )


class _DelayedUnpickleSmokeAdapter(_SmokeAdapter):
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def build_component_smoke_fixture(self) -> ComponentSmokeFixture:
        fixture = super().build_component_smoke_fixture()
        return ComponentSmokeFixture(
            task_id=fixture.task_id,
            reward_inputs=_DelayedPickleRewardInputs(
                self.delay_s,
                fixture.reward_inputs,
            ),
            expected_reward_output_shape=fixture.expected_reward_output_shape,
            public_summary=fixture.public_summary,
        )


class _DelayedOperationTensor:
    def __init__(self, value: SmokeTensor, delay_s: float) -> None:
        self.value = value
        self.delay_s = delay_s

    def __mul__(self, other: object) -> object:
        time.sleep(self.delay_s)
        return self.value * other


class _DelayedCallableSmokeAdapter(_SmokeAdapter):
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def build_component_smoke_fixture(self) -> ComponentSmokeFixture:
        fixture = super().build_component_smoke_fixture()
        reward_inputs = replace(
            fixture.reward_inputs,
            base_height=_DelayedOperationTensor(
                fixture.reward_inputs.base_height,
                self.delay_s,
            ),
        )
        return replace(fixture, reward_inputs=reward_inputs)


class ComponentSmokeAndAntiDegenerateTests(unittest.TestCase):
    def test_stable_authoring_checks_facade_reexports_schema_versions(self) -> None:
        self.assertEqual(
            COMPONENT_SMOKE_SCHEMA_VERSION,
            RUNTIME_COMPONENT_SMOKE_SCHEMA_VERSION,
        )
        self.assertEqual(
            ANTI_DEGENERATE_SCHEMA_VERSION,
            ANTI_DEGENERATE_ASSESSMENT_SCHEMA_VERSION,
        )

    def test_candidate_source_swap_after_preflight_is_not_parsed(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def reward(inputs):
    return inputs.base_height

def build_training_components():
    return TrainingComponents(
        name="original",
        reward_terms=[RewardTerm(name="reward", weight=1.0, fn=reward)],
        rationale="Original source.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "swap-candidate", source)
            target = Path(candidate.path) / "training_code" / "components.py"
            replacement = Path(tmp) / "replacement.py"
            replacement.write_text(
                source.replace('name="original"', 'name="swapped"'),
                encoding="utf-8",
            )
            swapped = False

            @contextmanager
            def swap_before_open(preflight, relative_path):
                nonlocal swapped
                if not swapped and relative_path == "training_code/components.py":
                    replacement.replace(target)
                    swapped = True
                with open_preflighted_candidate_file(
                    preflight,
                    relative_path,
                ) as handle:
                    yield handle

            with patch(
                "true_rl_agent.candidates.component_source_semantics."
                "open_preflighted_candidate_file",
                side_effect=swap_before_open,
            ):
                assessment = assess_anti_degenerate_candidate(
                    candidate,
                    {"source_payloads": []},
                )

        self.assertTrue(swapped)
        self.assertFalse(assessment["blocking"])
        self.assertEqual(
            assessment["reason"],
            "candidate_source_not_available",
        )

    def test_canonical_mechanism_identity_is_shared_across_term_kinds(self) -> None:
        mechanism = format_canonical_mechanism_id(
            "ScheduleTerm",
            "height-scale",
        )
        self.assertEqual(mechanism, "ScheduleTerm:height-scale")
        self.assertEqual(
            require_canonical_mechanism_delta([mechanism]),
            (mechanism,),
        )
        self.assertEqual(
            canonical_mechanism_runtime_identity(mechanism),
            ("schedule_term", "height-scale"),
        )
        with self.assertRaises(ValueError):
            require_canonical_mechanism_delta([{}])
        with self.assertRaises(ValueError):
            format_canonical_mechanism_id("RewardTerm", "x" * 129)

    def test_dynamic_canonical_term_name_is_a_repairable_blocker(self) -> None:
        anchor = """
def reward(inputs):
    return inputs.base_height

components = [RewardTerm(name="height_reward", weight=1.0, fn=reward)]
"""
        candidate = """
def reward(inputs):
    return inputs.base_height

term_name = "height_reward"
components = [RewardTerm(name=term_name, weight=1.0, fn=reward)]
"""
        assessment = assess_anti_degenerate_source(
            candidate_source=candidate,
            current_best_source=anchor,
        )

        self.assertEqual(assessment["status"], "fail")
        self.assertTrue(assessment["blocking"])
        self.assertEqual(
            assessment["reason"],
            "canonical_component_mechanisms_cannot_be_identified",
        )

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

    def test_worker_startup_time_is_not_charged_to_candidate_load(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def height_reward(inp):
    return inp.base_height * 0.5

def build_training_components():
    return TrainingComponents(
        name="fast_reward",
        reward_terms=[RewardTerm(name="height", weight=1.0, fn=height_reward)],
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-slow-start", source)
            with patch(
                "true_rl_agent.candidates.authoring_checks."
                "DEFAULT_COMPONENT_SMOKE_LOAD_TIMEOUT_S",
                0.25,
            ):
                result = run_component_smoke_check(
                    candidate,
                    _DelayedUnpickleSmokeAdapter(0.6),
                    timeout_s=0.25,
                    worker_startup_timeout_s=2.0,
                )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["reason"], "component_smoke_passed")

    def test_worker_startup_timeout_is_an_infrastructure_error(self) -> None:
        source = """
from true_rl_agent.candidates.components import TrainingComponents

def build_training_components():
    return TrainingComponents(name="never_started")
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-startup-timeout", source)
            with self.assertRaises(IsolatedWorkerStartupError):
                run_component_smoke_check(
                    candidate,
                    _DelayedUnpickleSmokeAdapter(0.5),
                    worker_startup_timeout_s=0.05,
                )

    def test_each_candidate_callable_receives_its_own_timeout_budget(self) -> None:
        source = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def first_reward(inp):
    return inp.base_height * 0.5

def second_reward(inp):
    return inp.base_height * 0.25

def build_training_components():
    return TrainingComponents(
        name="two_rewards",
        reward_terms=[
            RewardTerm(name="first", weight=1.0, fn=first_reward),
            RewardTerm(name="second", weight=1.0, fn=second_reward),
        ],
        rationale="Exercise two independently bounded pure reward callables.",
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-two-rewards", source)
            result = run_component_smoke_check(
                candidate,
                _DelayedCallableSmokeAdapter(0.2),
                timeout_s=0.3,
            )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checked_reward_terms"], 2)

    def test_component_smoke_drains_large_result_before_worker_exit(self) -> None:
        reward_terms = ",\n".join(
            "            RewardTerm(name=\"reward_{index:03d}\", "
            "weight=1.0, fn=height_reward)".format(index=index)
            for index in range(400)
        )
        source = f"""
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def height_reward(inp):
    return inp.base_height * 0.5

def build_training_components():
    return TrainingComponents(
        name="large_smoke_result",
        reward_terms=[
{reward_terms}
        ],
    )
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-large-result", source)
            result = run_component_smoke_check(candidate, _SmokeAdapter())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checked_reward_terms"], 400)

    def test_component_load_timeout_remains_a_candidate_failure(self) -> None:
        source = """
from true_rl_agent.candidates.components import TrainingComponents

def build_training_components():
    while True:
        pass
    return TrainingComponents(name="unreachable")
"""
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "candidate-load-timeout", source)
            with patch(
                "true_rl_agent.candidates.authoring_checks."
                "DEFAULT_COMPONENT_SMOKE_LOAD_TIMEOUT_S",
                0.25,
            ):
                result = run_component_smoke_check(
                    candidate,
                    _SmokeAdapter(),
                    timeout_s=0.1,
                )

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["reason"], "component_load_timeout")
        self.assertEqual(result["timeout_s"], 0.25)

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

    def test_compound_component_mechanism_delta_is_blocking(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def contact(inp):
    return inp.base_contact.float()

def settle(inp):
    return inp.goal_progress_fraction

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[
            RewardTerm(name="contact", weight=-1.0, fn=contact, clip_min=0.0, clip_max=1.0),
            RewardTerm(name="settle", weight=0.2, fn=settle, clip_min=0.0, clip_max=1.0),
        ],
    )
"""
        compound = anchor.replace(
            "return inp.base_contact.float()",
            "return inp.base_contact.float() * inp.goal_progress_fraction",
        ).replace("weight=0.2, fn=settle", "weight=0.5, fn=settle")

        assessment = assess_anti_degenerate_source(
            candidate_source=compound,
            current_best_source=anchor,
        )

        self.assertTrue(assessment["blocking"])
        self.assertEqual(
            assessment["schema_version"],
            "true_rl_agent.anti_degenerate_assessment.v4",
        )
        self.assertEqual(assessment["classification"], "compound_mechanism_delta")
        self.assertEqual(
            assessment["changed_mechanisms"],
            ["RewardTerm:contact", "RewardTerm:settle"],
        )

    def test_one_term_function_and_weight_change_remains_attributable(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def contact(inp):
    return inp.base_contact.float()

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="contact", weight=-1.0, fn=contact)],
    )
"""
        candidate = anchor.replace(
            "return inp.base_contact.float()",
            "return inp.base_contact.float() * inp.goal_progress_fraction",
        ).replace("weight=-1.0", "weight=-2.0")

        assessment = assess_anti_degenerate_source(
            candidate_source=candidate,
            current_best_source=anchor,
        )

        self.assertFalse(assessment["blocking"])
        self.assertEqual(assessment["classification"], "mechanism_delta_present")
        self.assertEqual(assessment["changed_mechanisms"], ["RewardTerm:contact"])

    def test_numeric_changes_across_two_terms_are_compound(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def first(inp):
    return inp.roll

def second(inp):
    return inp.pitch

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[
            RewardTerm(name="first", weight=-1.0, fn=first),
            RewardTerm(name="second", weight=-1.0, fn=second),
        ],
    )
"""
        candidate = anchor.replace(
            'name="first", weight=-1.0',
            'name="first", weight=-2.0',
        ).replace(
            'name="second", weight=-1.0',
            'name="second", weight=-3.0',
        )

        assessment = assess_anti_degenerate_source(
            candidate_source=candidate,
            current_best_source=anchor,
        )

        self.assertTrue(assessment["blocking"])
        self.assertEqual(assessment["classification"], "compound_mechanism_delta")
        self.assertEqual(assessment["changed_mechanism_count"], 2)

    def test_shared_helper_change_across_two_terms_is_compound(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def shared_scale(inp):
    return inp.goal_progress_fraction

def first(inp):
    return shared_scale(inp) * inp.roll

def second(inp):
    return shared_scale(inp) * inp.pitch

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[
            RewardTerm(name="first", weight=-1.0, fn=first),
            RewardTerm(name="second", weight=-1.0, fn=second),
        ],
    )
"""
        candidate = anchor.replace(
            "return inp.goal_progress_fraction",
            "return inp.goal_progress_fraction.square()",
        )

        assessment = assess_anti_degenerate_source(
            candidate_source=candidate,
            current_best_source=anchor,
        )

        self.assertTrue(assessment["blocking"])
        self.assertEqual(assessment["classification"], "compound_mechanism_delta")
        self.assertEqual(
            assessment["changed_mechanisms"],
            ["RewardTerm:first", "RewardTerm:second"],
        )

    def test_helper_change_for_one_term_remains_attributable(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def scale(inp):
    return inp.goal_progress_fraction

def progress(inp):
    return scale(inp) * inp.roll

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="progress", weight=1.0, fn=progress)],
    )
"""
        candidate = anchor.replace(
            "return inp.goal_progress_fraction",
            "return inp.goal_progress_fraction.square()",
        )

        assessment = assess_anti_degenerate_source(
            candidate_source=candidate,
            current_best_source=anchor,
        )

        self.assertFalse(assessment["blocking"])
        self.assertEqual(assessment["classification"], "mechanism_delta_present")
        self.assertEqual(assessment["changed_mechanisms"], ["RewardTerm:progress"])

    def test_unused_helper_is_not_a_component_mechanism_delta(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def progress(inp):
    return inp.goal_progress_fraction

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="progress", weight=1.0, fn=progress)],
    )
"""
        candidate = anchor.replace(
            "def build_training_components():",
            "def unused(inp):\n"
            "    ghost = RewardTerm(name=\"fake_delta\", weight=9.0, fn=progress)\n"
            "    return inp.roll\n\n"
            "def build_training_components():",
        )

        assessment = assess_anti_degenerate_source(
            candidate_source=candidate,
            current_best_source=anchor,
        )

        self.assertTrue(assessment["blocking"])
        self.assertEqual(
            assessment["classification"],
            "no_component_mechanism_delta",
        )
        self.assertEqual(assessment["changed_mechanism_count"], 0)

    def test_builder_name_rebinding_is_repairable_not_misattributed(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def reward(inp):
    return inp.base_height

def build_training_components():
    term = RewardTerm(name="active", weight=1.0, fn=reward)
    terms = [term]
    return TrainingComponents(name="anchor", reward_terms=terms)
"""
        rebound = anchor.replace(
            '    return TrainingComponents(name="anchor", reward_terms=terms)',
            '    term = RewardTerm(name="fake_delta", weight=9.0, fn=reward)\n'
            '    return TrainingComponents(name="anchor", reward_terms=terms)',
        )
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "rebound", rebound)
            loaded = load_training_components(Path(candidate.path))
            self.assertEqual(
                [term.name for term in loaded.reward_terms],
                ["active"],
            )

        assessment = assess_anti_degenerate_source(
            candidate_source=rebound,
            current_best_source=anchor,
        )
        self.assertTrue(assessment["blocking"])
        self.assertEqual(
            assessment["reason"],
            "canonical_component_mechanisms_cannot_be_identified",
        )

    def test_builder_callable_alias_does_not_fabricate_a_delta(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def active(inp):
    return inp.base_height

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="active", weight=1.0, fn=active)],
    )
"""
        alias = anchor.replace(
            "def build_training_components():\n",
            "def build_training_components():\n    fn_alias = active\n",
        ).replace("fn=active)", "fn=fn_alias)")

        assessment = assess_anti_degenerate_source(
            candidate_source=alias,
            current_best_source=anchor,
        )
        self.assertTrue(assessment["blocking"])
        self.assertEqual(
            assessment["classification"],
            "no_component_mechanism_delta",
        )
        self.assertEqual(assessment["changed_mechanism_count"], 0)

    def test_callable_identifier_renames_do_not_fabricate_a_delta(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def scale(value):
    scaled = value * 0.5
    return scaled

def active(inp):
    value = scale(inp.base_height)
    return value

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="active", weight=1.0, fn=active)],
    )
"""
        variants = {
            "local": anchor.replace(
                "value = scale(inp.base_height)\n    return value",
                "renamed = scale(inp.base_height)\n    return renamed",
            ),
            "helper": anchor.replace("scale", "renamed_scale"),
        }
        for label, candidate in variants.items():
            with self.subTest(label=label):
                assessment = assess_anti_degenerate_source(
                    candidate_source=candidate,
                    current_best_source=anchor,
                )
                self.assertTrue(assessment["blocking"])
                self.assertEqual(
                    assessment["classification"],
                    "no_component_mechanism_delta",
                )
                self.assertEqual(assessment["changed_mechanism_count"], 0)

        keyword_anchor = anchor.replace(
            "scale(inp.base_height)",
            "scale(value=inp.base_height)",
        )
        keyword_candidate = keyword_anchor.replace(
            "def scale(value):\n    scaled = value * 0.5",
            "def scale(renamed):\n    scaled = renamed * 0.5",
        ).replace("scale(value=inp.base_height)", "scale(renamed=inp.base_height)")
        keyword_assessment = assess_anti_degenerate_source(
            candidate_source=keyword_candidate,
            current_best_source=keyword_anchor,
        )
        self.assertTrue(keyword_assessment["blocking"])
        self.assertEqual(
            keyword_assessment["classification"],
            "no_component_mechanism_delta",
        )

    def test_nonruntime_callable_syntax_does_not_fabricate_a_delta(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def active(inp):
    value = inp.base_height * 0.5
    return value

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="active", weight=1.0, fn=active)],
    )
"""
        variants = {
            "pass": anchor.replace("    return value", "    pass\n    return value"),
            "ellipsis": anchor.replace("    return value", "    ...\n    return value"),
            "if_true": anchor.replace(
                "    return value",
                "    if True:\n        return value",
            ),
            "if_literal_truthy": anchor.replace(
                "    return value",
                "    if 1:\n        return value",
            ),
            "if_false": anchor.replace(
                "    return value",
                "    if False:\n        return inp.base_height * 9.0\n"
                "    return value",
            ),
            "if_expression": anchor.replace(
                "    return value",
                "    return value if True else inp.base_height * 9.0",
            ),
            "unreachable_return": anchor.replace(
                "    return value",
                "    return value\n    return inp.roll.square()",
            ),
            "annotations": anchor.replace(
                "def active(inp):\n    value =",
                "def active(inp: object) -> object:\n    value: object =",
            ),
        }
        for label, candidate in variants.items():
            with self.subTest(label=label):
                assessment = assess_anti_degenerate_source(
                    candidate_source=candidate,
                    current_best_source=anchor,
                )
                self.assertTrue(assessment["blocking"])
                self.assertEqual(
                    assessment["classification"],
                    "no_component_mechanism_delta",
                )

    def test_component_callable_extra_parameters_are_not_canonical(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def active(inp):
    return inp.base_height

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="active", weight=1.0, fn=active)],
    )
"""
        variants = {
            "default": anchor.replace("active(inp)", "active(inp, unused=0)"),
            "vararg": anchor.replace("active(inp)", "active(inp, *unused)"),
            "kwarg": anchor.replace("active(inp)", "active(inp, **unused)"),
        }
        for label, candidate in variants.items():
            with self.subTest(label=label):
                assessment = assess_anti_degenerate_source(
                    candidate_source=candidate,
                    current_best_source=anchor,
                )
                self.assertTrue(assessment["blocking"])
                self.assertEqual(
                    assessment["reason"],
                    "canonical_component_mechanisms_cannot_be_identified",
                )

    def test_helper_fake_api_and_alias_forms_are_not_attributed(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def scale(value):
    return value * 0.5

def active(inp):
    return scale(inp.base_height)

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="active", weight=1.0, fn=active)],
    )
"""
        extra_argument_variants = {
            "default": anchor.replace("scale(value)", "scale(value, unused=0)"),
            "vararg": anchor.replace("scale(value)", "scale(value, *unused)"),
            "kwarg": anchor.replace("scale(value)", "scale(value, **unused)"),
        }
        for label, candidate in extra_argument_variants.items():
            with self.subTest(label=label):
                assessment = assess_anti_degenerate_source(
                    candidate_source=candidate,
                    current_best_source=anchor,
                )
                self.assertTrue(assessment["blocking"])
                self.assertEqual(
                    assessment["reason"],
                    "canonical_component_mechanisms_cannot_be_identified",
                )

        alias_anchor = anchor.replace(
            "    return scale(inp.base_height)",
            "    alias = scale\n    return alias(inp.base_height)",
        )
        alias_variants = {
            "rename": alias_anchor.replace("scale", "renamed_scale"),
            "body_change": alias_anchor.replace("value * 0.5", "value * 0.75"),
        }
        for label, candidate in alias_variants.items():
            with self.subTest(label=f"alias_{label}"):
                assessment = assess_anti_degenerate_source(
                    candidate_source=candidate,
                    current_best_source=alias_anchor,
                )
                self.assertTrue(assessment["blocking"])
                self.assertEqual(
                    assessment["reason"],
                    "canonical_component_mechanisms_cannot_be_identified",
                )

    def test_recursive_helper_and_pattern_renames_do_not_fabricate_delta(self) -> None:
        recursive_anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def scale(inp, step=0):
    if step == 1:
        return inp.base_height
    return scale(inp, step=1)

def active(inp):
    return scale(inp)

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="active", weight=1.0, fn=active)],
    )
"""
        recursive_candidate = recursive_anchor.replace("scale", "renamed_scale")
        recursive_assessment = assess_anti_degenerate_source(
            candidate_source=recursive_candidate,
            current_best_source=recursive_anchor,
        )
        self.assertTrue(recursive_assessment["blocking"])
        self.assertEqual(
            recursive_assessment["classification"],
            "no_component_mechanism_delta",
        )

        pattern_anchor = recursive_anchor.replace(
            "def active(inp):\n    return scale(inp)",
            "def active(inp):\n"
            "    match inp:\n"
            "        case value:\n"
            "            return value.base_height",
        )
        pattern_candidate = pattern_anchor.replace(
            "case value:\n            return value.base_height",
            "case renamed:\n            return renamed.base_height",
        )
        pattern_assessment = assess_anti_degenerate_source(
            candidate_source=pattern_candidate,
            current_best_source=pattern_anchor,
        )
        self.assertTrue(pattern_assessment["blocking"])
        self.assertEqual(
            pattern_assessment["classification"],
            "no_component_mechanism_delta",
        )

    def test_explicit_reward_clip_defaults_do_not_fabricate_delta(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def active(inp):
    return inp.base_height

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="active", weight=1.0, fn=active)],
    )
"""
        for field_name in ("clip_min", "clip_max"):
            with self.subTest(field_name=field_name):
                candidate = anchor.replace(
                    "weight=1.0, fn=active)",
                    f"weight=1.0, fn=active, {field_name}=None)",
                )
                assessment = assess_anti_degenerate_source(
                    candidate_source=candidate,
                    current_best_source=anchor,
                )
                self.assertTrue(assessment["blocking"])
                self.assertEqual(
                    assessment["classification"],
                    "no_component_mechanism_delta",
                )

    def test_component_name_only_rename_is_not_a_mechanism_delta(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def active(inp):
    return inp.base_height

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="active", weight=1.0, fn=active)],
    )
"""
        candidate = anchor.replace('name="active"', 'name="renamed"')
        assessment = assess_anti_degenerate_source(
            candidate_source=candidate,
            current_best_source=anchor,
        )
        self.assertTrue(assessment["blocking"])
        self.assertEqual(
            assessment["classification"],
            "no_component_mechanism_delta",
        )
        self.assertEqual(assessment["changed_mechanism_count"], 0)

    def test_nested_callable_scopes_are_not_claimed_as_canonical_deltas(self) -> None:
        templates = {
            "nested_name": (
                "def active(inp):\n"
                "    def inner(value):\n"
                "        return value * 0.5\n"
                "    return inner(inp.base_height)\n",
                "def active(inp):\n"
                "    def renamed_inner(value):\n"
                "        return value * 0.5\n"
                "    return renamed_inner(inp.base_height)\n",
            ),
            "nested_argument": (
                "def active(inp):\n"
                "    def inner(value):\n"
                "        return value * 0.5\n"
                "    return inner(inp.base_height)\n",
                "def active(inp):\n"
                "    def inner(renamed):\n"
                "        return renamed * 0.5\n"
                "    return inner(inp.base_height)\n",
            ),
            "lambda_argument": (
                "def active(inp):\n"
                "    helper = lambda value: value * 0.5\n"
                "    return helper(inp.base_height)\n",
                "def active(inp):\n"
                "    helper = lambda renamed: renamed * 0.5\n"
                "    return helper(inp.base_height)\n",
            ),
        }
        suffix = (
            "\ndef build_training_components():\n"
            "    return TrainingComponents(\n"
            "        name='anchor',\n"
            "        reward_terms=[RewardTerm(name='active', weight=1.0, fn=active)],\n"
            "    )\n"
        )
        prefix = (
            "from true_rl_agent.candidates.components import "
            "RewardTerm, TrainingComponents\n\n"
        )
        for label, (anchor_callable, candidate_callable) in templates.items():
            with self.subTest(label=label):
                assessment = assess_anti_degenerate_source(
                    candidate_source=prefix + candidate_callable + suffix,
                    current_best_source=prefix + anchor_callable + suffix,
                )
                self.assertTrue(assessment["blocking"])
                self.assertEqual(
                    assessment["reason"],
                    "canonical_component_mechanisms_cannot_be_identified",
                )

    def test_one_removed_and_added_term_is_one_replacement(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def old_contact(inp):
    return inp.base_contact.float()

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="old_contact", weight=-1.0, fn=old_contact)],
    )
"""
        candidate = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def horizontal_contact(inp):
    return inp.base_contact_force[..., :2].norm(dim=-1).squeeze(-1)

def build_training_components():
    return TrainingComponents(
        name="candidate",
        reward_terms=[
            RewardTerm(name="horizontal_contact", weight=-1.0, fn=horizontal_contact),
        ],
    )
"""

        assessment = assess_anti_degenerate_source(
            candidate_source=candidate,
            current_best_source=anchor,
        )

        self.assertFalse(assessment["blocking"])
        self.assertEqual(assessment["changed_mechanism_count"], 1)
        self.assertEqual(
            assessment["changed_mechanisms"],
            ["RewardTerm:old_contact->horizontal_contact"],
        )

    def test_rejected_same_parent_input_variant_records_nonblocking_evidence(
        self,
    ) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def progress(inp):
    return inp.goal_progress_fraction

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="progress", weight=0.2, fn=progress)],
    )
"""
        rejected = anchor.replace(
            "def build_training_components():",
            "def base_force(inp):\n"
            "    return inp.base_contact_force.norm(dim=-1).squeeze(-1).clamp(max=5000.0) / 1000.0\n\n"
            "def build_training_components():",
        ).replace(
            "reward_terms=[RewardTerm(name=\"progress\", weight=0.2, fn=progress)]",
            "reward_terms=[\n"
            "            RewardTerm(name=\"progress\", weight=0.2, fn=progress),\n"
            "            RewardTerm(name=\"base_force\", weight=-0.01, fn=base_force),\n"
            "        ]",
        )
        variant = anchor.replace(
            "def build_training_components():",
            "def torso_force(inp):\n"
            "    return (inp.base_contact_force.norm(dim=-1).squeeze(-1) / 100.0).clamp(max=1.0)\n\n"
            "def build_training_components():",
        ).replace(
            "reward_terms=[RewardTerm(name=\"progress\", weight=0.2, fn=progress)]",
            "reward_terms=[\n"
            "            RewardTerm(name=\"progress\", weight=0.2, fn=progress),\n"
            "            RewardTerm(name=\"torso_force\", weight=-0.05, fn=torso_force),\n"
            "        ]",
        )
        variant_without_prior = variant.replace(
            '        name="anchor",',
            '        name="anchor",\n'
            '        rationale="Add a bounded torso force penalty.",',
        )
        variant_with_prior = variant_without_prior.replace(
            'rationale="Add a bounded torso force penalty."',
            (
                'rationale="Unlike add_base_contact_force_penalty, saturate at '
                '100 N so the signal acts before termination."'
            ),
        )
        anchor_digest = hashlib.sha256(anchor.encode("utf-8")).hexdigest()
        prompt = {
            "source_payloads": [
                {
                    "logical_id": "current_best/training_code/components.py",
                    "content": anchor,
                    "content_sha256": anchor_digest,
                }
            ]
        }
        history = [
            {
                "round_id": "round-contact-rejected",
                "candidate_id": "add_base_contact_force_penalty",
                "accepted": False,
                "parent_program_source_checksum": f"sha256:{anchor_digest}",
                "content": rejected,
                "content_sha256": hashlib.sha256(rejected.encode("utf-8")).hexdigest(),
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(
                Path(tmp),
                "contact-variant",
                variant_without_prior,
            )
            (Path(candidate.path) / "rationale.md").write_text(
                "Standalone legacy view names add_base_contact_force_penalty.",
                encoding="utf-8",
            )
            without_named_candidate = assess_anti_degenerate_candidate(
                candidate,
                prompt,
                evaluated_history_sources=history,
            )
            (Path(candidate.path) / "training_code" / "components.py").write_text(
                variant_with_prior,
                encoding="utf-8",
            )
            (Path(candidate.path) / "rationale.md").write_text(
                "Standalone legacy view deliberately omits the prior id.",
                encoding="utf-8",
            )
            with_named_candidate = assess_anti_degenerate_candidate(
                candidate,
                prompt,
                evaluated_history_sources=history,
            )
        self.assertFalse(without_named_candidate["blocking"])
        without_name_evidence = without_named_candidate[
            "historical_variant_assessment"
        ]
        self.assertEqual(
            without_name_evidence["matched_candidate_id"],
            "add_base_contact_force_penalty",
        )
        self.assertEqual(
            without_name_evidence["shared_input_fields"],
            ["base_contact_force"],
        )
        self.assertFalse(without_name_evidence["blocking"])
        self.assertEqual(
            without_name_evidence["hypothesis_source"],
            "training_code/components.py:TrainingComponents.rationale",
        )
        self.assertFalse(
            without_name_evidence["literal_candidate_id_mentioned_in_rationale"]
        )
        self.assertFalse(with_named_candidate["blocking"])
        with_name_evidence = with_named_candidate["historical_variant_assessment"]
        self.assertEqual(
            with_name_evidence["matched_candidate_id"],
            "add_base_contact_force_penalty",
        )
        self.assertTrue(
            with_name_evidence["literal_candidate_id_mentioned_in_rationale"]
        )

    def test_history_repeat_uses_each_candidates_own_anchor_and_ignores_name(
        self,
    ) -> None:
        def source(*, include_upright: bool, goal_name: str | None, weight: float) -> str:
            upright_fn = (
                "\n\ndef upright(inp):\n    return -inp.projected_gravity[:, 2]\n"
                if include_upright
                else ""
            )
            goal_fn = (
                f"\n\ndef {goal_name}(inp):\n    return inp.goal_progress_fraction\n"
                if goal_name is not None
                else ""
            )
            terms = [
                'RewardTerm(name="base_height", weight=1.0, fn=base_height)'
            ]
            if include_upright:
                terms.append(
                    'RewardTerm(name="upright", weight=0.1, fn=upright)'
                )
            if goal_name is not None:
                terms.append(
                    f'RewardTerm(name="{goal_name}", weight={weight}, '
                    f'fn={goal_name}, clip_min=0.0, clip_max=1.0)'
                )
            return (
                "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n"
                "\n\ndef base_height(inp):\n    return inp.base_height\n"
                f"{upright_fn}{goal_fn}\n\ndef build_training_components():\n"
                "    return TrainingComponents(\n"
                '        name="candidate",\n'
                f"        reward_terms=[{', '.join(terms)}],\n"
                '        rationale="One attributable mechanism.",\n'
                "    )\n"
            )

        reference = source(include_upright=False, goal_name=None, weight=0.0)
        evolved_anchor = source(include_upright=True, goal_name=None, weight=0.0)
        historical = source(
            include_upright=False,
            goal_name="goal_progress",
            weight=0.5,
        )
        renamed_same_anchor = source(
            include_upright=False,
            goal_name="goal_progress_reward",
            weight=0.25,
        )
        renamed_new_anchor = source(
            include_upright=True,
            goal_name="goal_progress_reward",
            weight=0.25,
        )
        history = [
            {
                "round_id": "round-goal-progress",
                "candidate_id": "goal-progress",
                "accepted": False,
                "content": historical,
                "content_sha256": hashlib.sha256(
                    historical.encode("utf-8")
                ).hexdigest(),
                "anchor_content": reference,
                "anchor_content_sha256": hashlib.sha256(
                    reference.encode("utf-8")
                ).hexdigest(),
                "anchor_role": "current_best",
            }
        ]

        self.assertEqual(
            compare_single_component_deltas(
                candidate_source=renamed_same_anchor,
                candidate_anchor_source=reference,
                comparison_source=historical,
                comparison_anchor_source=reference,
            ),
            "parameter_only_delta",
        )
        self.assertEqual(
            compare_single_component_deltas(
                candidate_source=renamed_new_anchor,
                candidate_anchor_source=evolved_anchor,
                comparison_source=historical,
                comparison_anchor_source=reference,
            ),
            "parameter_only_delta",
        )

        for label, candidate_source, anchor in (
            ("rename", renamed_same_anchor, reference),
            ("anchor_drift", renamed_new_anchor, evolved_anchor),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                candidate = _write_candidate(Path(tmp), label, candidate_source)
                anchor_digest = hashlib.sha256(anchor.encode("utf-8")).hexdigest()
                assessment = assess_anti_degenerate_candidate(
                    candidate,
                    {
                        "source_payloads": [
                            {
                                "logical_id": (
                                    "current_best/training_code/components.py"
                                ),
                                "content": anchor,
                                "content_sha256": anchor_digest,
                            }
                        ]
                    },
                    evaluated_history_sources=history,
                )
            self.assertTrue(assessment["blocking"])
            self.assertEqual(
                assessment["classification"],
                "historical_mechanism_repeat",
            )
            self.assertEqual(
                assessment["equivalence_classification"],
                "parameter_only_delta",
            )
            self.assertEqual(
                assessment["comparison_scope"],
                "single_component_deltas_against_own_anchors",
            )
            self.assertEqual(
                assessment["candidate_changed_mechanism"],
                "RewardTerm:goal_progress_reward",
            )
            self.assertEqual(
                assessment["matched_changed_mechanism"],
                "RewardTerm:goal_progress",
            )
            self.assertEqual(assessment["history_sources_with_anchor"], 1)
            self.assertEqual(assessment["history_sources_without_anchor"], 0)

        legacy_history = [
            {
                key: value
                for key, value in history[0].items()
                if not key.startswith("anchor_")
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(
                Path(tmp),
                "legacy-anchor-missing",
                renamed_new_anchor,
            )
            assessment = assess_anti_degenerate_candidate(
                candidate,
                {
                    "source_payloads": [
                        {
                            "logical_id": (
                                "current_best/training_code/components.py"
                            ),
                            "content": evolved_anchor,
                            "content_sha256": hashlib.sha256(
                                evolved_anchor.encode("utf-8")
                            ).hexdigest(),
                        }
                    ]
                },
                evaluated_history_sources=legacy_history,
            )
        self.assertFalse(assessment["blocking"])
        self.assertEqual(assessment["history_sources_with_anchor"], 0)
        self.assertEqual(assessment["history_sources_without_anchor"], 1)
        self.assertTrue(assessment["candidate_authoring_anchor_available"])

        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(
                Path(tmp),
                "legacy-exact-repeat",
                historical,
            )
            assessment = assess_anti_degenerate_candidate(
                candidate,
                {
                    "source_payloads": [
                        {
                            "logical_id": (
                                "current_best/training_code/components.py"
                            ),
                            "content": reference,
                            "content_sha256": hashlib.sha256(
                                reference.encode("utf-8")
                            ).hexdigest(),
                        }
                    ]
                },
                evaluated_history_sources=legacy_history,
            )
        self.assertTrue(assessment["blocking"])
        self.assertEqual(
            assessment["comparison_scope"],
            "evaluated_history_legacy_full_source",
        )
        self.assertEqual(assessment["history_sources_with_anchor"], 0)
        self.assertEqual(assessment["history_sources_without_anchor"], 1)

    def test_component_comparison_preserves_structural_numeric_positions(
        self,
    ) -> None:
        def source(name: str | None, expression: str, *, weight: float = 1.0) -> str:
            mechanism = (
                ""
                if name is None
                else (
                    f"\n\ndef {name}(inputs):\n    return {expression}\n"
                )
            )
            terms = ['RewardTerm(name="base", weight=1.0, fn=base)']
            if name is not None:
                terms.append(
                    f'RewardTerm(name="{name}", weight={weight}, fn={name})'
                )
            return (
                "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n"
                "\n\ndef base(inputs):\n    return inputs.base_height\n"
                f"{mechanism}\n\ndef build_training_components():\n"
                "    return TrainingComponents(\n"
                '        name="candidate",\n'
                f"        reward_terms=[{', '.join(terms)}],\n"
                "    )\n"
            )

        anchor = source(None, "inputs.base_height")
        comparisons = (
            (
                "subscript_axis",
                "inputs.projected_gravity[:, 2]",
                "inputs.projected_gravity[:, 0]",
                None,
            ),
            (
                "signed_subscript_axis",
                "inputs.projected_gravity[:, -1]",
                "inputs.projected_gravity[:, 1]",
                None,
            ),
            (
                "keyword_dim",
                "inputs.base_contact_force.sum(dim=1)",
                "inputs.base_contact_force.sum(dim=2)",
                None,
            ),
            (
                "signed_keyword_dim",
                "inputs.base_contact_force.sum(dim=-1)",
                "inputs.base_contact_force.sum(dim=1)",
                None,
            ),
            (
                "positional_dim",
                "inputs.base_contact_force.sum(1)",
                "inputs.base_contact_force.sum(2)",
                None,
            ),
            (
                "signed_positional_dim",
                "inputs.base_contact_force.sum(-1)",
                "inputs.base_contact_force.sum(1)",
                None,
            ),
            (
                "sign",
                "-inputs.base_height",
                "inputs.base_height",
                None,
            ),
            (
                "arithmetic_scale",
                "inputs.goal_progress_fraction * 0.5",
                "inputs.goal_progress_fraction * 0.25",
                "parameter_only_delta",
            ),
            (
                "signed_arithmetic_scale",
                "inputs.goal_progress_fraction * -0.5",
                "inputs.goal_progress_fraction * 0.25",
                "parameter_only_delta",
            ),
            (
                "signed_threshold",
                "inputs.goal_progress_fraction > -0.85",
                "inputs.goal_progress_fraction > 0.85",
                "parameter_only_delta",
            ),
            (
                "method_arithmetic_scale",
                "inputs.goal_progress_fraction.mul(0.5)",
                "inputs.goal_progress_fraction.mul(0.25)",
                "parameter_only_delta",
            ),
            (
                "inline_clamp_threshold",
                "inputs.goal_progress_fraction.clamp(max=0.5)",
                "inputs.goal_progress_fraction.clamp(max=0.25)",
                "parameter_only_delta",
            ),
        )
        for label, first_expression, second_expression, expected in comparisons:
            with self.subTest(label=label):
                self.assertEqual(
                    compare_single_component_deltas(
                        candidate_source=source("first", first_expression),
                        candidate_anchor_source=anchor,
                        comparison_source=source("second", second_expression),
                        comparison_anchor_source=anchor,
                    ),
                    expected,
                )

        self.assertEqual(
            compare_single_component_deltas(
                candidate_source=source(
                    "first_weight",
                    "inputs.goal_progress_fraction",
                    weight=0.5,
                ),
                candidate_anchor_source=anchor,
                comparison_source=source(
                    "second_weight",
                    "inputs.goal_progress_fraction",
                    weight=0.25,
                ),
                comparison_anchor_source=anchor,
            ),
            "parameter_only_delta",
        )
        self.assertEqual(
            compare_single_component_deltas(
                candidate_source=source(
                    "first_signed_weight",
                    "inputs.goal_progress_fraction",
                    weight=-0.5,
                ),
                candidate_anchor_source=anchor,
                comparison_source=source(
                    "second_signed_weight",
                    "inputs.goal_progress_fraction",
                    weight=0.25,
                ),
                comparison_anchor_source=anchor,
            ),
            "parameter_only_delta",
        )

        compound = source("first", "inputs.base_height").replace(
            "def build_training_components():",
            "def extra(inputs):\n    return inputs.roll\n\n"
            "def build_training_components():",
        ).replace(
            'RewardTerm(name="first", weight=1.0, fn=first)]',
            'RewardTerm(name="first", weight=1.0, fn=first), '
            'RewardTerm(name="extra", weight=1.0, fn=extra)]',
        )
        with self.assertRaises(SingleComponentDeltaComparisonUnavailable):
            compare_single_component_deltas(
                candidate_source=compound,
                candidate_anchor_source=anchor,
                comparison_source=source("second", "inputs.base_height"),
                comparison_anchor_source=anchor,
            )

        structural_history_source = source(
            "first_index",
            "inputs.projected_gravity[:, 2]",
        )
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(
                Path(tmp),
                "legacy-structural-index",
                source(
                    "second_index",
                    "inputs.projected_gravity[:, 0]",
                ),
            )
            assessment = assess_anti_degenerate_candidate(
                candidate,
                {
                    "source_payloads": [
                        {
                            "logical_id": (
                                "current_best/training_code/components.py"
                            ),
                            "content": anchor,
                            "content_sha256": hashlib.sha256(
                                anchor.encode("utf-8")
                            ).hexdigest(),
                        }
                    ]
                },
                evaluated_history_sources=[
                    {
                        "round_id": "legacy-structural-index",
                        "candidate_id": "first-index",
                        "accepted": False,
                        "content": structural_history_source,
                        "content_sha256": hashlib.sha256(
                            structural_history_source.encode("utf-8")
                        ).hexdigest(),
                    }
                ],
            )
        self.assertFalse(assessment["blocking"])
        self.assertEqual(assessment["history_sources_with_anchor"], 0)
        self.assertEqual(assessment["history_sources_without_anchor"], 1)

    def test_single_component_operation_delta_uses_the_shared_analyzer(
        self,
    ) -> None:
        def source(
            *,
            extra_name: str | None,
            expression: str = "inputs.roll",
            weight: float = 1.0,
        ) -> str:
            extra_function = (
                ""
                if extra_name is None
                else (
                    f"\n\ndef {extra_name}(inputs):\n"
                    f"    return {expression}\n"
                )
            )
            extra_term = (
                ""
                if extra_name is None
                else (
                    f", RewardTerm(name=\"{extra_name}\", "
                    f"weight={weight}, fn={extra_name})"
                )
            )
            return (
                "from true_rl_agent.candidates.components import "
                "RewardTerm, TrainingComponents\n"
                "\n\ndef base(inputs):\n    return inputs.base_height\n"
                f"{extra_function}\n\ndef build_training_components():\n"
                "    return TrainingComponents(\n"
                '        name="candidate",\n'
                "        reward_terms=[RewardTerm(name=\"base\", "
                f"weight=1.0, fn=base){extra_term}],\n"
                "    )\n"
            )

        base = source(extra_name=None)
        with_extra = source(extra_name="extra")
        cases = (
            ("add", with_extra, base, None, "RewardTerm:extra"),
            ("remove", base, with_extra, "RewardTerm:extra", None),
            (
                "modify",
                source(extra_name="extra", weight=0.5),
                with_extra,
                "RewardTerm:extra",
                "RewardTerm:extra",
            ),
            (
                "rename",
                source(
                    extra_name="renamed",
                    expression="inputs.pitch",
                ),
                with_extra,
                "RewardTerm:extra",
                "RewardTerm:renamed",
            ),
        )
        for operation, candidate, anchor, before_id, after_id in cases:
            with self.subTest(operation=operation):
                projected = project_single_component_operation_delta(
                    candidate_source=candidate,
                    anchor_source=anchor,
                )
                self.assertEqual(projected["operation"], operation)
                self.assertEqual(
                    projected["before_component"]["mechanism_id"]
                    if projected["before_component"] is not None
                    else None,
                    before_id,
                )
                self.assertEqual(
                    projected["after_component"]["mechanism_id"]
                    if projected["after_component"] is not None
                    else None,
                    after_id,
                )
                self.assertEqual(
                    projected["candidate_source_checksum"],
                    "sha256:"
                    + hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
                )

        with self.assertRaises(SingleComponentDeltaComparisonUnavailable):
            project_single_component_operation_delta(
                candidate_source=source(extra_name="renamed"),
                anchor_source=with_extra,
            )
        with self.assertRaises(SingleComponentDeltaComparisonUnavailable):
            project_single_component_operation_delta(
                candidate_source=with_extra.replace(
                    'RewardTerm(name="extra", weight=1.0, fn=extra)]',
                    'RewardTerm(name="extra", weight=1.0, fn=extra), '
                    'RewardTerm(name="second", weight=1.0, fn=extra)]',
                ),
                anchor_source=base,
            )

    def test_cross_parent_same_input_structural_variant_remains_advisory(
        self,
    ) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def base_height(inp):
    return inp.base_height

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="base_height", weight=1.0, fn=base_height)],
    )
"""
        evolved_anchor = anchor.replace(
            "def build_training_components():",
            "def upright(inp):\n    return -inp.projected_gravity[:, 2]\n\n"
            "def build_training_components():",
        ).replace(
            'reward_terms=[RewardTerm(name="base_height", weight=1.0, fn=base_height)]',
            'reward_terms=[RewardTerm(name="base_height", weight=1.0, fn=base_height), '
            'RewardTerm(name="upright", weight=0.1, fn=upright)]',
        )
        rejected = anchor.replace(
            "def build_training_components():",
            "def force_sum(inp):\n"
            "    return (inp.base_contact_force.norm(dim=-1).sum(dim=1) / 100.0).clamp(0.0, 1.0)\n\n"
            "def build_training_components():",
        ).replace(
            'reward_terms=[RewardTerm(name="base_height", weight=1.0, fn=base_height)]',
            'reward_terms=[RewardTerm(name="base_height", weight=1.0, fn=base_height), '
            'RewardTerm(name="force_sum", weight=-2.0, fn=force_sum)]',
        )
        variant = evolved_anchor.replace(
            "def build_training_components():",
            "def force_norm(inp):\n"
            "    return inp.base_contact_force.norm(dim=-1).squeeze(-1)\n\n"
            "def build_training_components():",
        ).replace(
            'RewardTerm(name="upright", weight=0.1, fn=upright)]',
            'RewardTerm(name="upright", weight=0.1, fn=upright), '
            'RewardTerm(name="force_norm", weight=-0.01, fn=force_norm, '
            'clip_min=0.0, clip_max=50.0)]',
        )
        self.assertIsNone(
            compare_single_component_deltas(
                candidate_source=variant,
                candidate_anchor_source=evolved_anchor,
                comparison_source=rejected,
                comparison_anchor_source=anchor,
            )
        )
        history = [
            {
                "round_id": "round-force-sum",
                "candidate_id": "force-sum",
                "accepted": False,
                "content": rejected,
                "content_sha256": hashlib.sha256(
                    rejected.encode("utf-8")
                ).hexdigest(),
                "anchor_content": anchor,
                "anchor_content_sha256": hashlib.sha256(
                    anchor.encode("utf-8")
                ).hexdigest(),
                "anchor_role": "current_best",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "force-variant", variant)
            evolved_digest = hashlib.sha256(
                evolved_anchor.encode("utf-8")
            ).hexdigest()
            assessment = assess_anti_degenerate_candidate(
                candidate,
                {
                    "source_payloads": [
                        {
                            "logical_id": "current_best/training_code/components.py",
                            "content": evolved_anchor,
                            "content_sha256": evolved_digest,
                        }
                    ]
                },
                evaluated_history_sources=history,
            )

        self.assertFalse(assessment["blocking"])
        variant_evidence = assessment["historical_variant_assessment"]
        self.assertEqual(
            variant_evidence["classification"],
            "rejected_input_dependency_variant",
        )
        self.assertEqual(
            variant_evidence["comparison_scope"],
            "rejected_history_relative_to_own_anchor",
        )
        self.assertEqual(
            variant_evidence["shared_input_fields"],
            ["base_contact_force"],
        )
        self.assertFalse(variant_evidence["blocking"])

    def test_canonical_hypothesis_extractor_accepts_only_static_literal(self) -> None:
        bound_literal = """
from true_rl_agent.candidates.components import TrainingComponents

def build_training_components():
    hypothesis = "Bounded upright shaping should improve stable ascent."
    return TrainingComponents(name="candidate", rationale=hypothesis)
"""
        computed = bound_literal.replace(
            'hypothesis = "Bounded upright shaping should improve stable ascent."',
            'hypothesis = "Bounded " + make_suffix()',
        )

        self.assertEqual(
            extract_training_components_rationale(bound_literal),
            "Bounded upright shaping should improve stable ascent.",
        )
        self.assertIsNone(extract_training_components_rationale(computed))

    def test_legacy_candidate_without_embedded_hypothesis_is_not_newly_blocked(
        self,
    ) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def progress(inp):
    return inp.base_height

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[RewardTerm(name="progress", weight=0.2, fn=progress)],
    )
"""
        rejected = anchor.replace(
            "return inp.base_height",
            "return inp.base_height + inp.roll",
        )
        variant = anchor.replace(
            "return inp.base_height",
            "return inp.base_height - inp.roll",
        )
        anchor_digest = hashlib.sha256(anchor.encode("utf-8")).hexdigest()
        history = [
            {
                "round_id": "legacy-rejected",
                "candidate_id": "legacy-prior",
                "accepted": False,
                "parent_program_source_checksum": f"sha256:{anchor_digest}",
                "content": rejected,
                "content_sha256": hashlib.sha256(
                    rejected.encode("utf-8")
                ).hexdigest(),
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "legacy-variant", variant)
            (Path(candidate.path) / "rationale.md").write_text(
                "Standalone legacy hypothesis.",
                encoding="utf-8",
            )
            assessment = assess_anti_degenerate_candidate(
                candidate,
                {
                    "source_payloads": [
                        {
                            "logical_id": (
                                "current_best/training_code/components.py"
                            ),
                            "content": anchor,
                            "content_sha256": anchor_digest,
                        }
                    ]
                },
                evaluated_history_sources=history,
            )

        historical = assessment["historical_variant_assessment"]
        self.assertFalse(assessment["blocking"])
        self.assertFalse(historical["blocking"])
        self.assertFalse(historical["canonical_hypothesis_available"])
        self.assertEqual(
            historical["hypothesis_source"],
            "unavailable_legacy_candidate",
        )

    def test_cold_start_compares_against_best_observed_anchor(self) -> None:
        anchor = """
from true_rl_agent.candidates.components import RewardTerm, TrainingComponents

def progress(inp):
    return inp.goal_progress_fraction

def contact(inp):
    return inp.base_contact.float()

def build_training_components():
    return TrainingComponents(
        name="anchor",
        reward_terms=[
            RewardTerm(name="progress", weight=1.0, fn=progress),
            RewardTerm(name="contact", weight=-1.0, fn=contact),
        ],
    )
"""
        compound = anchor.replace(
            "return inp.goal_progress_fraction",
            "return inp.goal_progress_fraction.square()",
        ).replace("weight=-1.0", "weight=-2.0")
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _write_candidate(Path(tmp), "compound", compound)
            assessment = assess_anti_degenerate_candidate(
                candidate,
                {
                    "candidate_lineage": {
                        "status": "bootstrap_observed_anchor",
                        "best_observed_experiment": {
                            "source_logical_id": (
                                "attempt/latest_failed/training_code/components.py"
                            ),
                        },
                    },
                    "source_payloads": [
                        {
                            "logical_id": (
                                "attempt/latest_failed/training_code/components.py"
                            ),
                            "content": anchor,
                            "content_sha256": hashlib.sha256(
                                anchor.encode("utf-8")
                            ).hexdigest(),
                        }
                    ],
                },
            )

        self.assertTrue(assessment["blocking"])
        self.assertEqual(
            assessment["comparison_anchor"],
            "best_observed_nonaccepted",
        )
        self.assertEqual(assessment["changed_mechanism_count"], 2)


if __name__ == "__main__":
    unittest.main()
