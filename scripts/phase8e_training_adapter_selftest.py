#!/usr/bin/env python3
"""Self-test for the Phase 8E-3 training-loop adapter."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.code_sandbox.components import DiagnosticTerm, LossTerm, RewardInputs, RewardTerm, ScheduleTerm, TrainingComponents  # noqa: E402
from humanoid_agent.cold_start_lab.recipe_adapter import ComponentContractError, precompute_schedule_table  # noqa: E402
from humanoid_agent.cold_start_lab.push_signal import install_phase8e_push_signal_probe  # noqa: E402
from humanoid_agent.cold_start_lab.training_adapter import (  # noqa: E402
    REWARD_WEIGHT_SCALE_MAX,
    Phase8ETrainingAdapter,
    Phase8EVecEnvRewardWrapper,
    control_effectiveness_contract,
    reward_inputs_contract,
    reward_inputs_field_expansion_report,
    reward_inputs_field_status,
)


class FakeVecEnv:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, _actions):
        self.calls += 1
        return {"obs": self.calls}, [1.0, 1.0], [False, False], {"source": "fake"}


class FakeVecEnvWithLaterNumericVector:
    def step(self, _actions):
        return {"obs": "later_numeric"}, [1.0, 1.0], [9.0, 9.0], {"source": "fake"}


class FakeVecEnvWithInvalidRewardSlot:
    def step(self, _actions):
        return {"obs": "bad_reward_slot"}, [False, False], [1.0, 1.0], {"source": "fake"}


class FakeRobotData:
    root_pos_w = [[1.0, 2.0, 0.8], [1.5, -0.5, 0.9]]
    joint_pos = [[0.2, 0.4], [0.6, 0.8]]
    joint_vel = [[0.0, 0.1], [0.2, 0.3]]
    default_joint_pos = [[0.1, 0.1]]


class FakeRobot:
    data = FakeRobotData()


class FakeScene:
    env_origins = [[0.5, 1.0, 0.0], [1.0, -1.0, 0.0]]
    robot = FakeRobot()

    def __getitem__(self, key: str):
        if key == "robot":
            return self.robot
        raise KeyError(key)


class FakeCommandManager:
    def get_command(self, name: str):
        if name == "base_velocity":
            return [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0]]
        raise KeyError(name)


class FakeObservedVecEnv(FakeVecEnv):
    scene = FakeScene()
    command_manager = FakeCommandManager()


class FakePushSignalBuffer:
    def __init__(self) -> None:
        self.consume_calls = 0

    def consume(self, rewards):
        self.consume_calls += 1
        _assert(rewards == [1.0, 1.0], f"push signal buffer should receive reward vector shape source: {rewards}")
        return {
            "active": [1.0, 0.0],
            "vector": [[0.25, -0.5], [0.0, 0.0]],
            "provenance": "runner_owned_push_event_instrumentation",
        }


class FakeDelayedPushSignalBuffer:
    def __init__(self) -> None:
        self.consume_calls = 0

    def consume(self, _rewards):
        self.consume_calls += 1
        if self.consume_calls == 1:
            return {
                "active": [0.0, 0.0],
                "vector": [[0.0, 0.0], [0.0, 0.0]],
                "provenance": "zero_fallback_no_push_event_observed",
            }
        return {
            "active": [0.0, 1.0],
            "vector": [[0.0, 0.0], [-0.1, 0.2]],
            "provenance": "runner_owned_push_event_instrumentation",
        }


class FakePushObservedVecEnv(FakeObservedVecEnv):
    def __init__(self) -> None:
        super().__init__()
        self.phase8e_push_signal_buffer = FakePushSignalBuffer()


class FakeDelayedPushObservedVecEnv(FakeObservedVecEnv):
    def __init__(self) -> None:
        super().__init__()
        self.phase8e_push_signal_buffer = FakeDelayedPushSignalBuffer()


class FakeResetVecEnv:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, _actions):
        self.calls += 1
        dones = [True, False] if self.calls == 1 else [False, False]
        return {"obs": self.calls}, [0.0, 0.0], dones, {"source": "reset"}


class FakeEventTermCfg:
    def __init__(self):
        self.func = lambda *_args, **_kwargs: None
        self.params = {"velocity_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2)}}


class FakeEventManager:
    def __init__(self):
        self.term = FakeEventTermCfg()
        self.set_calls = 0

    def get_term_cfg(self, name: str):
        _assert(name == "push_robot", f"unexpected event term lookup: {name}")
        return self.term

    def set_term_cfg(self, name: str, cfg):
        _assert(name == "push_robot", f"unexpected event term write: {name}")
        self.term = cfg
        self.set_calls += 1


class FakeInstrumentedEnv:
    scene = type("FakeInstrumentedScene", (), {"num_envs": 2})()

    def __init__(self):
        self.event_manager = FakeEventManager()


def _reward(inp: RewardInputs) -> list[float]:
    _assert(inp.last_action == [[0.1, 0.2], [0.3, 0.4]], f"last_action should stay vector-valued: {inp.last_action}")
    _assert(inp.prev_action == [[0.0, 0.0], [0.0, 0.0]], f"prev_action first-step fallback mismatch: {inp.prev_action}")
    _assert(inp.action_delta == [[0.1, 0.2], [0.3, 0.4]], f"action_delta first-step mismatch: {inp.action_delta}")
    _assert(inp.base_pos_xy == [[0.0, 0.0], [0.0, 0.0]], f"base_pos_xy fallback mismatch: {inp.base_pos_xy}")
    _assert(inp.projected_gravity == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], f"projected_gravity fallback mismatch: {inp.projected_gravity}")
    _assert(inp.commanded_base_vel == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], f"commanded_base_vel fallback mismatch: {inp.commanded_base_vel}")
    _assert(inp.joint_pos == [[0.0, 0.0], [0.0, 0.0]], f"joint_pos fallback must stay rank-2: {inp.joint_pos}")
    _assert(inp.joint_vel == [[0.0, 0.0], [0.0, 0.0]], f"joint_vel fallback must stay rank-2: {inp.joint_vel}")
    _assert(inp.joint_pos_error == [[0.0, 0.0], [0.0, 0.0]], f"joint_pos_error fallback must stay rank-2: {inp.joint_pos_error}")
    _assert(inp.push_active == [0.0, 0.0], f"push_active no-event fallback mismatch: {inp.push_active}")
    _assert(
        inp.push_vec_public == [[0.0, 0.0], [0.0, 0.0]],
        f"push_vec_public no-event fallback mismatch: {inp.push_vec_public}",
    )
    return [0.25, 0.5]


def _loss(_inp: RewardInputs) -> list[float]:
    return [-0.1, 0.1]


def _diagnostic(inp: RewardInputs) -> dict[str, object]:
    return {
        "height_margin": inp.base_height,
        "stable": True,
    }


def _empty_diagnostic(_inp: RewardInputs) -> dict[str, float]:
    return {}


def _too_many_diagnostics(_inp: RewardInputs) -> dict[str, float]:
    return {f"field_{index}": float(index) for index in range(1500)}


def _bad_reward(_inp: RewardInputs) -> list[float]:
    return [0.0, math.inf]


def _observed_reward(inp: RewardInputs) -> list[float]:
    _assert(inp.base_height == [0.8, 0.9], f"base_height should come from root_pos_w z: {inp.base_height}")
    _assert(inp.base_pos_xy == [[0.5, 1.0], [0.5, 0.5]], f"base_pos_xy should be env-origin-relative xy: {inp.base_pos_xy}")
    _assert(inp.commanded_base_vel == [[0.0, 0.0, 0.0], [0.1, -0.1, 0.0]], f"commanded_base_vel mismatch: {inp.commanded_base_vel}")
    _assert_nested_close(inp.joint_pos_error, [[0.1, 0.3], [0.5, 0.7]], f"joint_pos_error mismatch: {inp.joint_pos_error}")
    return [0.0, 0.0]


def _push_signal_reward(inp: RewardInputs) -> list[float]:
    _assert(inp.push_active == [1.0, 0.0], f"push_active should come from runner instrumentation: {inp.push_active}")
    _assert(
        inp.push_vec_public == [[0.25, -0.5], [0.0, 0.0]],
        f"push_vec_public should expose public xy velocity kick: {inp.push_vec_public}",
    )
    return [0.0, 0.0]


def _delayed_push_signal_reward(_inp: RewardInputs) -> list[float]:
    return [0.0, 0.0]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_nested_close(actual: list[list[float]], expected: list[list[float]], message: str) -> None:
    if len(actual) != len(expected):
        raise AssertionError(message)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        if len(actual_row) != len(expected_row):
            raise AssertionError(message)
        for actual_value, expected_value in zip(actual_row, expected_row, strict=True):
            if abs(float(actual_value) - float(expected_value)) > 1.0e-9:
                raise AssertionError(message)


def main() -> int:
    control_contract = control_effectiveness_contract()
    _assert(
        set(control_contract["schedule_controls"]["effective_keys"]) == {"reward_weight_scale"},
        f"only reward_weight_scale should be effective in current training adapter: {control_contract}",
    )
    ineffective_schedule_keys = set(control_contract["schedule_controls"]["accepted_but_ineffective_keys"])
    _assert(
        {"push_magnitude_scale", "push_start_fraction", "push_interval_s", "domain_randomization_scale"}.issubset(ineffective_schedule_keys),
        f"push/domain schedule keys must be accepted but ineffective: {control_contract}",
    )
    _assert(
        control_contract["component_controls"]["loss_terms"]["effectiveness"] == "guarded_but_not_inserted_into_ppo_optimizer",
        f"LossTerm effectiveness must be explicit: {control_contract}",
    )

    contract = reward_inputs_contract()
    _assert(contract["schema_version"] == "humanoid_agent.phase8e.reward_inputs_contract.v1", f"unexpected contract schema: {contract}")
    _assert(contract["field_order"] == list(RewardInputs.field_names()), f"contract field order must follow RewardInputs: {contract}")
    _assert(set(contract["fields"]) == set(RewardInputs.field_names()), f"contract fields must exactly match RewardInputs: {contract}")
    _assert(contract["fields"]["base_pos_xy"]["shape"] == ["N", 2], f"base_pos_xy shape missing from contract: {contract}")
    _assert(
        contract["fields"]["base_pos_xy"]["coordinate_frame"] == "env_origin_relative_world_xy",
        f"base_pos_xy coordinate frame must prevent world-absolute drift misuse: {contract}",
    )
    _assert(
        "env_origins" in contract["fields"]["base_pos_xy"]["fallback_behavior"],
        f"base_pos_xy fallback must document env origin dependency: {contract}",
    )
    _assert(
        "reset" in contract["fields"]["action_delta"]["fallback_behavior"],
        f"action_delta contract must document reset semantics: {contract}",
    )
    _assert(
        contract["fields"]["push_active"]["source_status"] == "runner_owned_push_event_instrumentation",
        f"push_active must be advertised only after runner instrumentation is live: {contract}",
    )
    _assert(
        contract["fields"]["push_vec_public"]["coordinate_frame"] == "world_xy_velocity_delta",
        f"push_vec_public must document the public coordinate frame: {contract}",
    )
    for name, meta in contract["fields"].items():
        expected_status = f"{meta['source_status']}; fallback={meta['fallback_behavior']}"
        _assert(
            reward_inputs_field_status()[name] == expected_status,
            f"field status must be derived from contract metadata for {name}: {reward_inputs_field_status()[name]} != {expected_status}",
        )
    contract_text = json.dumps(contract, ensure_ascii=False, sort_keys=True)
    _assert("safe_usage" not in contract_text and "unsafe_usage" not in contract_text, "contract must stay schema-only")
    _assert("success_metric" not in contract_text and "checkpoint_selector" not in contract_text, "contract must not expose eval decision state")

    expansion_report = reward_inputs_field_expansion_report()
    implemented_fields = set(expansion_report["implemented_public_fields"])
    first_batch = {"base_pos_xy", "projected_gravity", "commanded_base_vel", "prev_action", "action_delta", "joint_pos_error"}
    _assert(first_batch.issubset(implemented_fields), f"first-batch fields missing from expansion report: {implemented_fields}")
    _assert("feet_contact" in expansion_report["planned_public_fields"], "feet/contact fields must stay planned until instrumented")
    _assert(expansion_report["field_shapes"]["base_pos_xy"] == ["N", 2], f"field_shapes must be machine-readable: {expansion_report}")
    _assert(expansion_report["field_shapes"]["joint_pos_error"] == ["N", "num_joints"], f"joint_pos_error shape missing: {expansion_report}")
    report_text = json.dumps(expansion_report, ensure_ascii=False, sort_keys=True)
    _assert("success_metric" not in report_text and "checkpoint_selector" not in report_text, "field expansion report must stay training-side public")

    components = TrainingComponents(
        name="phase8e_adapter_selftest",
        reward_terms=[RewardTerm("public_bonus", fn=_reward, weight=2.0, clip_min=0.0, clip_max=1.0)],
        loss_terms=[LossTerm("guarded_aux_loss", fn=_loss, weight=0.5)],
        diagnostics=[DiagnosticTerm("public_diag", fn=_diagnostic)],
    )
    adapter = Phase8ETrainingAdapter(
        manifest=None,
        components=components,
        num_envs=2,
        num_joints=2,
        planned_training_steps=4,
    )
    wrapped = Phase8EVecEnvRewardWrapper(FakeVecEnv(), adapter)
    obs, rewards, dones, extras = wrapped.step([[0.1, 0.2], [0.3, 0.4]])
    _assert(obs == {"obs": 1}, "obs passthrough failed")
    _assert(dones == [False, False], "done vector passthrough failed")
    _assert(extras == {"source": "fake"}, "extras passthrough failed")
    _assert(rewards == [1.5, 2.0], f"candidate reward was not added correctly: {rewards}")
    runtime = adapter.runtime_summary()
    _assert(runtime["step_calls"] == 1, "step count missing")
    _assert(runtime["reward_term_calls"] == 1, "reward term call count missing")
    _assert(runtime["loss_term_calls"] == 1, "loss term should be guarded even though PPO insertion is disabled")
    _assert(runtime["diagnostic_calls"] == 1, "diagnostic term call count missing")
    _assert(runtime["last_candidate_reward_mean"] == 0.75, "weighted candidate reward mean mismatch")
    field_status = adapter.public_summary()["runtime_contract"]["input_field_status"]
    _assert(
        field_status["push_active"].startswith("runner_owned_push_event_instrumentation"),
        "push field status must reflect live runner instrumentation",
    )
    _assert(
        adapter.public_summary()["runtime_contract"]["diagnostic_limits"]["overflow_behavior"] == "reject_candidate_payload",
        "public runtime contract must expose diagnostic payload limits",
    )
    _assert(
        adapter.public_summary()["runtime_contract"]["diagnostic_limits"]["max_diagnostic_terms"] > 0,
        "public runtime contract must expose diagnostic term-count limits",
    )
    try:
        Phase8ETrainingAdapter(
            manifest=None,
            components=TrainingComponents(
                name="too_many_empty_diag_terms",
                diagnostics=[DiagnosticTerm(f"empty_diag_{index}", fn=_empty_diagnostic) for index in range(1500)],
            ),
            num_envs=2,
            num_joints=2,
            planned_training_steps=1,
        )
    except ComponentContractError:
        pass
    else:
        raise AssertionError("too many empty DiagnosticTerm components must fail during adapter construction")
    oversized_diag_adapter = Phase8ETrainingAdapter(
        manifest=None,
        components=TrainingComponents(name="oversized_diag", diagnostics=[DiagnosticTerm("too_many_diag_fields", fn=_too_many_diagnostics)]),
        num_envs=2,
        num_joints=2,
        planned_training_steps=1,
    )
    try:
        oversized_diag_adapter.apply_step_rewards(env=object(), actions=[[0.0], [0.0]], rewards=[0.0, 0.0])
    except ComponentContractError:
        pass
    else:
        raise AssertionError("oversized diagnostic payload must fail before reaching runtime summary")
    _assert(
        oversized_diag_adapter.runtime_summary()["last_diagnostics"] == [],
        "oversized diagnostic payload must not be stored in runtime summary",
    )
    # Runtime provenance: this fake env exposes no robot data, so env-derived
    # fields must be marked zero_fallback, while last_action (supplied) is observed.
    provenance = runtime["input_provenance"]
    _assert(provenance["last_action"] == "observed", f"last_action should be observed: {provenance}")
    _assert(provenance["prev_action"] == "zero_fallback", f"prev_action first-step provenance mismatch: {provenance}")
    _assert(provenance["action_delta"] == "derived_from_current_action_and_zero_previous_action", f"action_delta first-step provenance mismatch: {provenance}")
    _assert(provenance["base_height"] == "zero_fallback", f"base_height should be zero_fallback without robot data: {provenance}")
    _assert(provenance["base_pos_xy"] == "zero_fallback", f"base_pos_xy should be zero_fallback without robot data: {provenance}")
    _assert(provenance["base_lin_vel"] == "zero_fallback", f"base_lin_vel should be zero_fallback without robot data: {provenance}")
    _assert(provenance["commanded_base_vel"] == "zero_fallback", f"commanded_base_vel should be zero_fallback without command manager: {provenance}")
    _assert(
        provenance["push_active"] == "zero_fallback_no_push_event_observed",
        f"push_active fallback provenance must be explicit when no event fires: {provenance}",
    )

    push_components = TrainingComponents(
        name="phase8e_push_signal_selftest",
        reward_terms=[RewardTerm("push_signal_probe", fn=_push_signal_reward, weight=1.0)],
    )
    push_adapter = Phase8ETrainingAdapter(
        manifest=None,
        components=push_components,
        num_envs=2,
        num_joints=2,
        planned_training_steps=4,
    )
    push_env = FakePushObservedVecEnv()
    Phase8EVecEnvRewardWrapper(push_env, push_adapter).step([[0.1, 0.2], [0.3, 0.4]])
    _assert(push_env.phase8e_push_signal_buffer.consume_calls == 1, "adapter must consume push signal once per reward step")
    _assert(
        push_adapter.runtime.input_provenance["push_active"] == "runner_owned_push_event_instrumentation",
        f"push signal provenance should record instrumentation source: {push_adapter.runtime.input_provenance}",
    )
    delayed_components = TrainingComponents(
        name="phase8e_delayed_push_signal_selftest",
        reward_terms=[RewardTerm("delayed_push_signal_probe", fn=_delayed_push_signal_reward, weight=1.0)],
    )
    delayed_adapter = Phase8ETrainingAdapter(
        manifest=None,
        components=delayed_components,
        num_envs=2,
        num_joints=2,
        planned_training_steps=4,
    )
    delayed_env = FakeDelayedPushObservedVecEnv()
    delayed_wrapper = Phase8EVecEnvRewardWrapper(delayed_env, delayed_adapter)
    delayed_wrapper.step([[0.1, 0.2], [0.3, 0.4]])
    delayed_wrapper.step([[0.1, 0.2], [0.3, 0.4]])
    _assert(
        delayed_adapter.runtime.input_provenance["push_active"] == "runner_owned_push_event_instrumentation",
        f"push provenance must upgrade after a later real event: {delayed_adapter.runtime.input_provenance}",
    )
    install_env = FakeInstrumentedEnv()
    install_summary = install_phase8e_push_signal_probe(install_env)
    _assert(install_summary["status"] == "installed", f"push signal probe should install: {install_summary}")
    _assert(install_env.event_manager.set_calls == 1, "installer must update EventManager term cfg")
    _assert(
        install_env.event_manager.term.func.__name__ == "phase8e_recording_push_by_setting_velocity",
        "installer must replace push_robot with the recording wrapper",
    )
    _assert(provenance["progress_fraction"] == "runner_owned", f"progress_fraction is runner-owned: {provenance}")

    captured_inputs: list[RewardInputs] = []

    def _capture(inp: RewardInputs) -> list[float]:
        captured_inputs.append(inp)
        return [0.0, 0.0]

    history_adapter = Phase8ETrainingAdapter(
        manifest=None,
        components=TrainingComponents(name="history", reward_terms=[RewardTerm("capture", fn=_capture)]),
        num_envs=2,
        planned_training_steps=2,
    )
    history_env = Phase8EVecEnvRewardWrapper(FakeVecEnv(), history_adapter)
    history_env.step([[0.1, 0.2], [0.3, 0.4]])
    history_env.step([[0.2, 0.1], [0.2, 0.6]])
    _assert(captured_inputs[1].prev_action == [[0.1, 0.2], [0.3, 0.4]], f"prev_action must track previous step: {captured_inputs[1].prev_action}")
    _assert_nested_close(
        captured_inputs[1].action_delta,
        [[0.1, -0.1], [-0.1, 0.2]],
        f"action_delta must be current minus previous: {captured_inputs[1].action_delta}",
    )

    reset_inputs: list[RewardInputs] = []

    def _capture_reset(inp: RewardInputs) -> list[float]:
        reset_inputs.append(inp)
        return [0.0, 0.0]

    reset_adapter = Phase8ETrainingAdapter(
        manifest=None,
        components=TrainingComponents(name="reset_history", reward_terms=[RewardTerm("capture_reset", fn=_capture_reset)]),
        num_envs=2,
        planned_training_steps=2,
    )
    reset_env = Phase8EVecEnvRewardWrapper(FakeResetVecEnv(), reset_adapter)
    reset_env.step([[0.5, 0.0], [0.2, 0.2]])
    reset_env.step([[0.1, 0.1], [0.4, 0.0]])
    _assert(reset_inputs[1].prev_action == [[0.0, 0.0], [0.2, 0.2]], f"reset envs must clear previous action: {reset_inputs[1].prev_action}")
    _assert_nested_close(
        reset_inputs[1].action_delta,
        [[0.1, 0.1], [0.2, -0.2]],
        f"action_delta must use reset-cleared previous action: {reset_inputs[1].action_delta}",
    )

    observed_adapter = Phase8ETrainingAdapter(
        manifest=None,
        components=TrainingComponents(name="observed", reward_terms=[RewardTerm("observed", fn=_observed_reward)]),
        num_envs=2,
        planned_training_steps=1,
    )
    observed_env = Phase8EVecEnvRewardWrapper(FakeObservedVecEnv(), observed_adapter)
    _assert(observed_adapter.num_joints == 2, f"wrapper should infer num_joints from robot data: {observed_adapter.num_joints}")
    observed_env.step([[0.0, 0.0], [0.0, 0.0]])
    observed_provenance = observed_adapter.runtime_summary()["input_provenance"]
    _assert(observed_provenance["base_height"] == "observed", f"base_height provenance mismatch: {observed_provenance}")
    _assert(observed_provenance["base_pos_xy"] == "observed", f"base_pos_xy provenance mismatch: {observed_provenance}")
    _assert(observed_provenance["commanded_base_vel"] == "observed", f"commanded_base_vel provenance mismatch: {observed_provenance}")
    _assert(observed_provenance["joint_pos_error"] == "observed", f"joint_pos_error provenance mismatch: {observed_provenance}")

    later_numeric_adapter = Phase8ETrainingAdapter(
        manifest=None,
        components=TrainingComponents(name="later_numeric", reward_terms=[RewardTerm("one", fn=lambda _inp: [1.0, 1.0])]),
        num_envs=2,
        planned_training_steps=1,
    )
    later_result = Phase8EVecEnvRewardWrapper(FakeVecEnvWithLaterNumericVector(), later_numeric_adapter).step([[0.0], [0.0]])
    _assert(later_result[1] == [2.0, 2.0], f"reward slot at index 1 was not updated: {later_result}")
    _assert(later_result[2] == [9.0, 9.0], "later numeric vector should not be treated as reward")

    try:
        Phase8EVecEnvRewardWrapper(FakeVecEnvWithInvalidRewardSlot(), later_numeric_adapter).step([[0.0], [0.0]])
    except ComponentContractError:
        pass
    else:
        raise AssertionError("non-numeric RSL-RL reward slot must fail instead of scanning later tuple items")

    bad_adapter = Phase8ETrainingAdapter(
        manifest=None,
        components=TrainingComponents(name="bad", reward_terms=[RewardTerm("bad", fn=_bad_reward)]),
        num_envs=2,
        planned_training_steps=1,
    )
    try:
        Phase8EVecEnvRewardWrapper(FakeVecEnv(), bad_adapter).step([0.0, 0.0])
    except ComponentContractError:
        pass
    else:
        raise AssertionError("non-finite runtime reward must fail")

    # reward_weight_scale curriculum: schedule scales candidate reward weights
    # over training progress (training-only, runner-clamped, eval unaffected).
    scaled_components = TrainingComponents(
        name="scaled",
        reward_terms=[RewardTerm("unit", fn=lambda inp: [1.0 for _ in inp.base_height])],
        schedules=[ScheduleTerm("curric", fn=lambda _p: {"reward_weight_scale": 3.0}, allowed_keys=("reward_weight_scale",))],
    )
    scaled_table = precompute_schedule_table(scaled_components, progress_points=[0.0, 0.5, 1.0], allowed_keys=("reward_weight_scale",))
    scaled_adapter = Phase8ETrainingAdapter(
        manifest=None, components=scaled_components, num_envs=2, planned_training_steps=4, schedule_table=scaled_table
    )
    scaled_out = scaled_adapter.apply_step_rewards(env=object(), actions=None, rewards=[0.0, 0.0])
    _assert(scaled_out == [3.0, 3.0], f"reward_weight_scale curriculum not applied: {scaled_out}")
    _assert(scaled_adapter.runtime_summary()["last_reward_weight_scale"] == 3.0, "reward_weight_scale not recorded")

    clamp_components = TrainingComponents(
        name="clamp",
        reward_terms=[RewardTerm("unit", fn=lambda inp: [1.0 for _ in inp.base_height])],
        schedules=[ScheduleTerm("curric", fn=lambda _p: {"reward_weight_scale": 1e6}, allowed_keys=("reward_weight_scale",))],
    )
    clamp_table = precompute_schedule_table(clamp_components, progress_points=[0.0, 1.0], allowed_keys=("reward_weight_scale",))
    clamp_adapter = Phase8ETrainingAdapter(
        manifest=None, components=clamp_components, num_envs=2, planned_training_steps=1, schedule_table=clamp_table
    )
    clamp_out = clamp_adapter.apply_step_rewards(env=object(), actions=None, rewards=[0.0, 0.0])
    _assert(clamp_out == [REWARD_WEIGHT_SCALE_MAX, REWARD_WEIGHT_SCALE_MAX], f"reward_weight_scale not clamped: {clamp_out}")

    print(json.dumps({"status": "pass", "test": "phase8e_training_adapter_selftest"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
