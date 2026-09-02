from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.tasks.stair_climbing.termination_capture import (
    StairTerminationCaptureError,
    build_termination_mask_snapshot,
    capture_current_termination_masks,
    install_stair_termination_capture,
    stair_termination_manager_resolution_contract,
    verify_stair_termination_manager_resolution,
)


class StairTerminationCaptureTests(unittest.TestCase):
    def test_captures_current_returns_even_when_manager_term_buffers_are_stale(
        self,
    ) -> None:
        current_masks = {
            "time_out": _FakeMask(False, True),
            "base_contact": _FakeMask(False, True),
            "future_term": _FakeMask(True, False),
        }
        stale_masks = {name: _FakeMask(False, False) for name in current_masks}
        env, cfg = _instrumented_env(
            current_masks,
            dones=_FakeMask(True, True),
            stale_manager_masks=stale_masks,
        )

        returned = _execute_terms(env, cfg)
        snapshot = capture_current_termination_masks(env)

        self.assertIs(returned["time_out"], current_masks["time_out"])
        self.assertEqual(
            snapshot.term_order,
            ("time_out", "base_contact", "future_term"),
        )
        self.assertEqual(snapshot.term_masks["time_out"].values, (False, True))
        self.assertEqual(snapshot.done_mask.values, (True, True))
        self.assertEqual(snapshot.time_out_mask.values, (False, True))
        self.assertEqual(snapshot.terminated_non_timeout_mask.values, (True, True))
        self.assertIsNot(snapshot.term_masks["time_out"], current_masks["time_out"])
        self.assertEqual(
            env.termination_manager.get_term("time_out").values,
            (False, False),
        )

    def test_timeout_and_contact_concurrency_is_preserved(self) -> None:
        env, cfg = _instrumented_env(
            {
                "time_out": _FakeMask(False, True),
                "base_contact": _FakeMask(False, True),
            },
            dones=_FakeMask(False, True),
        )
        _execute_terms(env, cfg)

        snapshot = capture_current_termination_masks(env)

        self.assertTrue(snapshot.term_masks["time_out"].values[1])
        self.assertTrue(snapshot.term_masks["base_contact"].values[1])

    def test_post_reset_step_replaces_historical_term_reason(self) -> None:
        current_masks = {
            "time_out": _FakeMask(False),
            "base_contact": _FakeMask(True),
        }
        cfg = SimpleNamespace(
            time_out=_term_cfg(
                lambda env: current_masks["time_out"],
                time_out=True,
            ),
            base_contact=_term_cfg(lambda env: current_masks["base_contact"]),
        )
        historical_masks = {
            "time_out": _FakeMask(False),
            "base_contact": _FakeMask(True),
        }
        manager = SimpleNamespace(
            active_terms=["time_out", "base_contact"],
            dones=_FakeMask(True),
            get_term=lambda name: historical_masks[name],
            get_term_cfg=lambda name: getattr(cfg, name),
        )
        env = SimpleNamespace(common_step_counter=7, termination_manager=manager)
        install_stair_termination_capture(cfg)

        _execute_terms(env, cfg)
        terminal = capture_current_termination_masks(env)
        self.assertEqual(terminal.term_masks["base_contact"].values, (True,))

        env.common_step_counter += 1
        current_masks["base_contact"] = _FakeMask(False)
        manager.dones = _FakeMask(False)
        _execute_terms(env, cfg)
        post_reset = capture_current_termination_masks(env)

        self.assertEqual(post_reset.term_masks["base_contact"].values, (False,))
        self.assertEqual(post_reset.terminated_non_timeout_mask.values, (False,))
        self.assertEqual(manager.get_term("base_contact").values, (True,))

    def test_done_union_mismatch_fails_at_capture_source(self) -> None:
        env, cfg = _instrumented_env(
            {"time_out": _FakeMask(False), "base_contact": _FakeMask(False)},
            dones=_FakeMask(True),
        )
        _execute_terms(env, cfg)

        with self.assertRaisesRegex(
            StairTerminationCaptureError,
            "does not equal OR",
        ):
            capture_current_termination_masks(env)

    def test_missing_or_stale_current_step_capture_is_rejected(self) -> None:
        env, cfg = _instrumented_env(
            {"time_out": _FakeMask(False), "base_contact": _FakeMask(False)},
            dones=_FakeMask(False),
        )
        _execute_terms(env, cfg)
        env.common_step_counter += 1

        with self.assertRaisesRegex(
            StairTerminationCaptureError,
            "not captured in the current step",
        ):
            capture_current_termination_masks(env)

        env, cfg = _instrumented_env(
            {"time_out": _FakeMask(False), "base_contact": _FakeMask(False)},
            dones=_FakeMask(False),
        )
        cfg.time_out.func(env, **cfg.time_out.params)
        with self.assertRaisesRegex(
            StairTerminationCaptureError,
            "does not match the active term set",
        ):
            capture_current_termination_masks(env)

    def test_installer_is_strict_and_does_not_touch_non_terms(self) -> None:
        marker = object()
        cfg = SimpleNamespace(
            time_out=_term_cfg(lambda env: _FakeMask(False)),
            disabled=None,
            metadata=marker,
        )

        self.assertEqual(
            install_stair_termination_capture(cfg),
            ("time_out",),
        )
        self.assertIs(cfg.metadata, marker)
        with self.assertRaisesRegex(
            StairTerminationCaptureError,
            "already instrumented",
        ):
            install_stair_termination_capture(cfg)

    def test_source_params_and_return_identity_are_preserved(self) -> None:
        expected = _FakeMask(True)

        def source(env: object, *, threshold: float) -> _FakeMask:
            self.assertEqual(threshold, 2.5)
            return expected

        cfg = SimpleNamespace(term=_term_cfg(source, {"threshold": 2.5}))
        cfg.time_out = _term_cfg(lambda env: _FakeMask(False), time_out=True)
        manager = SimpleNamespace(
            active_terms=["time_out", "term"],
            dones=_FakeMask(True),
            get_term=lambda name: _FakeMask(False),
            get_term_cfg=lambda name: getattr(cfg, name),
        )
        env = SimpleNamespace(common_step_counter=3, termination_manager=manager)
        install_stair_termination_capture(cfg)

        cfg.time_out.func(env, **cfg.time_out.params)
        actual = cfg.term.func(env, **cfg.term.params)

        self.assertIs(actual, expected)
        self.assertEqual(
            capture_current_termination_masks(env).term_masks["term"].values,
            (True,),
        )

    def test_installer_preserves_manager_resolvable_params_at_top_level(self) -> None:
        sensor_cfg = SimpleNamespace(body_ids=None)
        expected = _FakeMask(True)

        def source(env: object, *, sensor_cfg: object, threshold: float) -> _FakeMask:
            self.assertEqual(getattr(sensor_cfg, "body_ids"), (4,))
            self.assertEqual(threshold, 1.0)
            return expected

        cfg = SimpleNamespace(
            time_out=_term_cfg(lambda env: _FakeMask(False), time_out=True),
            base_contact=_term_cfg(
                source,
                {"sensor_cfg": sensor_cfg, "threshold": 1.0},
            ),
        )
        install_stair_termination_capture(cfg)

        self.assertEqual(
            set(cfg.base_contact.params),
            {"sensor_cfg", "threshold"},
        )
        self.assertIs(cfg.base_contact.params["sensor_cfg"], sensor_cfg)
        self.assertEqual(
            inspect.signature(cfg.base_contact.func),
            inspect.signature(source),
        )

        # Mirrors Isaac Lab's deferred top-level SceneEntityCfg resolution.
        sensor_cfg.body_ids = (4,)
        env = SimpleNamespace(common_step_counter=0)
        actual = cfg.base_contact.func(env, **cfg.base_contact.params)
        self.assertIs(actual, expected)

    def test_runtime_resolution_is_bound_to_protocol_and_scene(self) -> None:
        protocol = _termination_protocol()
        sensor_cfg = SimpleNamespace(
            name="contact_forces",
            body_names=["torso_link"],
            body_ids=[14],
        )
        time_out = _term_cfg(lambda env: _FakeMask(False), time_out=True)
        base_contact = _term_cfg(
            lambda env, **params: _FakeMask(False),
            {"sensor_cfg": sensor_cfg, "threshold": 1.0},
        )
        base_contact.time_out = False
        install_stair_termination_capture(
            SimpleNamespace(time_out=time_out, base_contact=base_contact)
        )
        manager = SimpleNamespace(
            active_terms=["time_out", "base_contact"],
            get_term_cfg=lambda name: {
                "time_out": time_out,
                "base_contact": base_contact,
            }[name],
        )
        scene_sensor = SimpleNamespace(
            find_bodies=lambda names, preserve_order: ([14], ["torso_link"])
        )
        env = SimpleNamespace(
            termination_manager=manager,
            scene={"contact_forces": scene_sensor},
        )

        expected = stair_termination_manager_resolution_contract(protocol)
        self.assertEqual(
            verify_stair_termination_manager_resolution(env, protocol=protocol),
            expected,
        )
        sensor_cfg.body_ids = [0]
        with self.assertRaisesRegex(
            StairTerminationCaptureError,
            "scene body resolution drifted",
        ):
            verify_stair_termination_manager_resolution(env, protocol=protocol)

    def test_shape_dtype_identity_and_duplicate_execution_are_strict(self) -> None:
        bad_dtype = _FakeMask(False)
        bad_dtype.dtype = "float32"
        with self.assertRaisesRegex(StairTerminationCaptureError, "boolean dtype"):
            build_termination_mask_snapshot(
                term_order=("time_out", "base_contact"),
                timeout_term_names=("time_out",),
                term_masks={
                    "time_out": bad_dtype,
                    "base_contact": _FakeMask(False),
                },
                done_mask=_FakeMask(False),
            )

        with self.assertRaisesRegex(StairTerminationCaptureError, "term set is invalid"):
            build_termination_mask_snapshot(
                term_order=("time_out", "time_out"),
                timeout_term_names=("time_out",),
                term_masks={"time_out": _FakeMask(False)},
                done_mask=_FakeMask(False),
            )

        env, cfg = _instrumented_env(
            {"time_out": _FakeMask(False)},
            dones=_FakeMask(False),
        )
        cfg.time_out.func(env, **cfg.time_out.params)
        with self.assertRaisesRegex(
            StairTerminationCaptureError,
            "more than once in one step",
        ):
            cfg.time_out.func(env, **cfg.time_out.params)


class _FakeScalar:
    def __init__(self, value: bool) -> None:
        self.value = value

    def item(self) -> bool:
        return self.value


class _FakeMask:
    dtype = "bool"

    def __init__(self, *values: bool) -> None:
        self.values = tuple(values)
        self.shape = (len(values),)

    def detach(self) -> _FakeMask:
        return self

    def clone(self) -> _FakeMask:
        return _FakeMask(*self.values)

    def __or__(self, other: _FakeMask) -> _FakeMask:
        return _FakeMask(
            *(left or right for left, right in zip(self.values, other.values, strict=True))
        )

    def __eq__(self, other: object) -> _FakeMask:  # type: ignore[override]
        if not isinstance(other, _FakeMask):
            return _FakeMask(*(False for _ in self.values))
        return _FakeMask(
            *(left == right for left, right in zip(self.values, other.values, strict=True))
        )

    def all(self) -> _FakeScalar:
        return _FakeScalar(all(self.values))


def _term_cfg(
    source: object,
    params: dict[str, object] | None = None,
    *,
    time_out: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(func=source, params=params or {}, time_out=time_out)


def _termination_protocol() -> dict[str, object]:
    return {
        "phase0_probe": {
            "termination_schema": {
                "schema_hash": "1" * 64,
                "terms": [
                    {
                        "manager_term": "time_out",
                        "parameters": {},
                        "time_out": True,
                    },
                    {
                        "manager_term": "base_contact",
                        "parameters": {
                            "sensor_cfg": {
                                "name": "contact_forces",
                                "body_names": ["torso_link"],
                            },
                            "threshold": 1.0,
                        },
                        "time_out": False,
                    },
                ],
            }
        },
    }


def _instrumented_env(
    masks: dict[str, _FakeMask],
    *,
    dones: _FakeMask,
    stale_manager_masks: dict[str, _FakeMask] | None = None,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    cfg = SimpleNamespace(
        **{
            name: _term_cfg(
                lambda env, value=value: value,
                time_out=name == "time_out",
            )
            for name, value in masks.items()
        }
    )
    manager_masks = stale_manager_masks or masks
    manager = SimpleNamespace(
        active_terms=list(masks),
        dones=dones,
        get_term=lambda name: manager_masks[name],
        get_term_cfg=lambda name: getattr(cfg, name),
    )
    env = SimpleNamespace(common_step_counter=7, termination_manager=manager)
    install_stair_termination_capture(cfg)
    return env, cfg


def _execute_terms(env: SimpleNamespace, cfg: SimpleNamespace) -> dict[str, _FakeMask]:
    return {
        name: getattr(cfg, name).func(env, **getattr(cfg, name).params)
        for name in env.termination_manager.active_terms
    }


if __name__ == "__main__":
    unittest.main()
