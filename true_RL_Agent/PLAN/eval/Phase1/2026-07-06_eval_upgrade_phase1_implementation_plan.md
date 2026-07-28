# Phase1 Eval CLI Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove candidate package paths from the eval execution boundary and make sealed validation candidate isolation an explicit, public-safe gate field.

**Architecture:** Keep the existing train/eval split: training still receives the candidate package, eval receives only runner-owned checkpoint material plus a candidate id for accounting. Add a small task-neutral runner helper for eval isolation checks so candidate-path rejection and gate shape converge in one place without leaking push-recovery-specific semantics into shared code.

**Tech Stack:** Python dataclasses, `argparse`, `unittest`, current `IsaacLabRslRlBackendAdapter`, `SubprocessRunner`, and the existing `scripts/isaaclab_rsl_rl_eval.py` entrypoint.

---

## Boundary Decisions

- Train remains candidate-code capable: `scripts/isaaclab_rsl_rl_train.py` and backend train argv may keep `--candidate <candidate.path>`.
- Eval becomes checkpoint-only plus identity: `scripts/isaaclab_rsl_rl_eval.py` accepts `--candidate-id <id>` and rejects `--candidate`, `--candidate-path`, and `--candidate-dir`.
- Candidate id is not a path substitute: values containing `/`, `\`, `://`, empty strings, or all-whitespace are invalid at the eval boundary.
- Shared layer stays task-neutral: the new helper only knows about candidate-path argv rejection, candidate-id validation, and isolation gate shape.
- Phase1 does not implement byte-level checkpoint safety report, sealed holdout rollout, current-best stats, or decision-gate consumption. Those remain Phase2+.

## Files

- Create: `D:\humanoid_agent\true_RL_Agent\src\true_rl_agent\runner\eval_isolation.py`
  - Defines candidate-path argv rejection, candidate-id validation, and `sealed_validation_gate` construction/validation.
- Create: `D:\humanoid_agent\true_RL_Agent\tests\unit\test_eval_isolation.py`
  - Focused red tests for the new shared helper.
- Modify: `D:\humanoid_agent\true_RL_Agent\src\true_rl_agent\backends\isaac_lab_rsl_rl.py`
  - Eval argv uses `--candidate-id`, not `--candidate`.
- Modify: `D:\humanoid_agent\true_RL_Agent\scripts\isaaclab_rsl_rl_eval.py`
  - Parser removes candidate package path input, validates candidate id, and records `sealed_validation_gate`.
- Modify: `D:\humanoid_agent\true_RL_Agent\src\true_rl_agent\runner\subprocess_runner.py`
  - Projects `sealed_validation_gate` into public eval metrics and fail-closes malformed or failing isolation gates.
- Modify tests:
  - `D:\humanoid_agent\true_RL_Agent\tests\unit\test_isaac_lab_backend.py`
  - `D:\humanoid_agent\true_RL_Agent\tests\unit\test_materialize_job_script.py`
  - `D:\humanoid_agent\true_RL_Agent\tests\unit\test_push_recovery_task.py`
  - `D:\humanoid_agent\true_RL_Agent\tests\unit\test_subprocess_runner.py`

## Task 1: Red Tests For Eval Isolation Helper

- [ ] **Step 1: Add failing helper tests**

Create `tests/unit/test_eval_isolation.py` with tests that expect:

```python
from __future__ import annotations

import unittest

from true_rl_agent.runner.eval_isolation import (
    build_candidate_isolation_gate,
    reject_candidate_path_arguments,
    validate_candidate_id,
    validate_sealed_validation_gate,
)


class EvalIsolationTests(unittest.TestCase):
    def test_rejects_candidate_path_arguments(self) -> None:
        for argv in (
            ["--candidate", "artifact://candidate-001"],
            ["--candidate=artifact://candidate-001"],
            ["--candidate-path", "D:/tmp/candidate"],
            ["--candidate-dir", "candidate_pkg"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(ValueError, "candidate path argument is not allowed"):
                    reject_candidate_path_arguments(argv)

    def test_allows_candidate_id_but_rejects_path_like_identity(self) -> None:
        self.assertEqual(validate_candidate_id("candidate-001"), "candidate-001")
        for value in ("", "   ", "artifact://candidate-001", "dir/candidate", r"dir\\candidate"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "candidate_id"):
                    validate_candidate_id(value)

    def test_builds_and_validates_candidate_isolation_gate(self) -> None:
        gate = build_candidate_isolation_gate()
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["candidate_isolation_status"], "pass")
        self.assertIs(validate_sealed_validation_gate(gate), gate)

        failed = build_candidate_isolation_gate(candidate_code_loaded=True)
        self.assertEqual(failed["status"], "fail")
        self.assertEqual(failed["candidate_isolation_status"], "fail")
        with self.assertRaisesRegex(ValueError, "candidate isolation failed"):
            validate_sealed_validation_gate(failed)
```

- [ ] **Step 2: Run and confirm RED**

Run:

```powershell
python -m unittest tests.unit.test_eval_isolation -v
```

Expected: import failure for `true_rl_agent.runner.eval_isolation`.

- [ ] **Step 3: Implement helper**

Add `src/true_rl_agent/runner/eval_isolation.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CANDIDATE_PATH_ARGUMENTS = ("--candidate", "--candidate-path", "--candidate-dir")


def reject_candidate_path_arguments(argv: Sequence[str]) -> None:
    for part in argv:
        if part in CANDIDATE_PATH_ARGUMENTS or any(
            part.startswith(f"{flag}=") for flag in CANDIDATE_PATH_ARGUMENTS
        ):
            raise ValueError(f"candidate path argument is not allowed in eval: {part}")


def validate_candidate_id(candidate_id: str) -> str:
    value = str(candidate_id).strip()
    if not value:
        raise ValueError("candidate_id must be non-empty")
    if "://" in value or "/" in value or "\\" in value:
        raise ValueError("candidate_id must be an identifier, not a path")
    return value


def build_candidate_isolation_gate(
    *,
    candidate_code_loaded: bool = False,
    candidate_package_loaded: bool = False,
    candidate_path_argument_present: bool = False,
) -> dict[str, Any]:
    failed = candidate_code_loaded or candidate_package_loaded or candidate_path_argument_present
    return {
        "status": "fail" if failed else "pass",
        "candidate_isolation_status": "fail" if failed else "pass",
        "candidate_code_loaded_in_eval": bool(candidate_code_loaded),
        "candidate_package_loaded_in_eval": bool(candidate_package_loaded),
        "candidate_path_argument_present": bool(candidate_path_argument_present),
        "reason": (
            "candidate_isolation_failed"
            if failed
            else "candidate_path_not_accepted_and_candidate_code_not_loaded"
        ),
    }


def validate_sealed_validation_gate(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(gate, Mapping):
        raise ValueError("sealed_validation_gate must be an object")
    if gate.get("candidate_isolation_status") != "pass":
        raise ValueError("candidate isolation failed")
    if gate.get("status") == "fail":
        raise ValueError("sealed validation gate failed")
    return gate
```

- [ ] **Step 4: Run and confirm GREEN**

Run:

```powershell
python -m unittest tests.unit.test_eval_isolation -v
```

Expected: all helper tests pass.

## Task 2: Backend Eval Argv Stops Passing Candidate Path

- [ ] **Step 1: Add failing backend/materialize assertions**

In `test_isaac_lab_backend.py`, extend `test_materializes_random_train_and_checkpoint_eval_cli_specs`:

```python
        eval_argv = bundle.eval_spec.metadata["cli_argv"]
        self.assertIn("--candidate-id", eval_argv)
        self.assertEqual(eval_argv[eval_argv.index("--candidate-id") + 1], "candidate-001")
        self.assertNotIn("--candidate", eval_argv)
        self.assertNotIn("artifact://candidate-001", eval_argv)
        self.assertNotIn("--candidate artifact://candidate-001", bundle.eval_spec.metadata["cli_command"])
        self.assertEqual(
            bundle.eval_spec.metadata["candidate_identity"],
            {"candidate_id": "candidate-001"},
        )
```

In `test_materialize_job_script.py`, extend the materialized output assertions:

```python
        self.assertIn("--candidate-id", runner_execution["eval"]["cli_argv"])
        self.assertEqual(
            runner_execution["eval"]["cli_argv"][runner_execution["eval"]["cli_argv"].index("--candidate-id") + 1],
            "candidate-001",
        )
        self.assertNotIn("artifact://candidate-001", runner_execution["eval"]["cli_argv"])
        self.assertNotIn("--candidate artifact://candidate-001", runner_execution["eval"]["cli_command"])
```

In `test_push_recovery_task.py`, extend `test_push_task_materializes_isaac_jobs_with_fixed_protocol_metadata`:

```python
        self.assertIn("--candidate-id", bundle.eval_spec.metadata["cli_argv"])
        self.assertEqual(
            bundle.eval_spec.metadata["cli_argv"][bundle.eval_spec.metadata["cli_argv"].index("--candidate-id") + 1],
            "push-code-candidate",
        )
        self.assertNotIn("memory://push-code-candidate", bundle.eval_spec.metadata["cli_argv"])
        self.assertNotIn("--candidate memory://push-code-candidate", bundle.eval_spec.metadata["cli_command"])
```

- [ ] **Step 2: Run and confirm RED**

Run:

```powershell
python -m unittest tests.unit.test_isaac_lab_backend tests.unit.test_materialize_job_script tests.unit.test_push_recovery_task -v
```

Expected: failures showing eval argv still contains `--candidate` and lacks `--candidate-id`.

- [ ] **Step 3: Modify backend eval argv**

Change `_eval_command()` and `_eval_argv()` in `src/true_rl_agent/backends/isaac_lab_rsl_rl.py` so they accept `candidate_id: str` instead of `candidate: CandidatePackageRef`, and materialize eval metadata with:

```python
                "candidate_identity": {"candidate_id": candidate.candidate_id},
```

The eval argv must contain:

```python
            "--candidate-id",
            candidate_id,
```

and must not read or pass `candidate.path`.

- [ ] **Step 4: Run and confirm GREEN**

Run the same unittest command from Step 2.

## Task 3: Eval Script Enforces Checkpoint-Only Candidate Isolation

- [ ] **Step 1: Add failing source-level assertions**

In `test_push_recovery_task.py`, rename or extend `test_isaac_trusted_eval_summary_records_no_candidate_code_loaded` with:

```python
        self.assertIn("--candidate-id", source)
        self.assertIn("reject_candidate_path_arguments", source)
        self.assertIn("validate_candidate_id", source)
        self.assertIn('"candidate_package_loaded_in_eval": False', source)
        self.assertIn('"sealed_validation_gate":', source)
        self.assertNotIn('parser.add_argument("--candidate", required=True)', source)
```

- [ ] **Step 2: Run and confirm RED**

Run:

```powershell
python -m unittest tests.unit.test_push_recovery_task.PushRecoveryTaskTests.test_isaac_trusted_eval_summary_records_no_candidate_code_loaded -v
```

Expected: failure because eval script still has `--candidate` and no gate helper usage.

- [ ] **Step 3: Modify eval script**

In `scripts/isaaclab_rsl_rl_eval.py`:

- Import helper functions from `true_rl_agent.runner.eval_isolation`.
- Call `reject_candidate_path_arguments(sys.argv[1:])` before `parse_args()`, and exit with code `2` on `ValueError`.
- Replace parser `--candidate` with required `--candidate-id`.
- Validate `args_cli.candidate_id` immediately after parsing.
- Add `candidate_id` to the `run` object for accounting.
- Add top-level:

```python
            "candidate_package_loaded_in_eval": False,
            "sealed_validation_gate": build_candidate_isolation_gate(),
```

to pass and fail summaries. Keep `candidate_code_loaded_in_eval = False`.

- [ ] **Step 4: Run and confirm GREEN**

Run the focused source-level test from Step 2.

## Task 4: Runner Projects And Validates Sealed Validation Gate

- [ ] **Step 1: Add failing runner test**

Extend `tests/unit/test_subprocess_runner.py` with a test whose eval summary contains:

```python
                    "sealed_validation_gate": {
                        "status": "pass",
                        "candidate_isolation_status": "pass",
                        "candidate_code_loaded_in_eval": False,
                        "candidate_package_loaded_in_eval": False,
                        "candidate_path_argument_present": False,
                    }
```

Assert:

```python
        self.assertEqual(public_result.metrics["sealed_validation_gate"]["candidate_isolation_status"], "pass")
```

Add a second test where `candidate_isolation_status` is `"fail"` and assert `RunnerExecutionError` includes `candidate isolation failed`.

- [ ] **Step 2: Run and confirm RED**

Run:

```powershell
python -m unittest tests.unit.test_subprocess_runner -v
```

Expected: missing projection or missing fail-close behavior.

- [ ] **Step 3: Modify runner summary projection**

In `src/true_rl_agent/runner/subprocess_runner.py`, import `validate_sealed_validation_gate`. In `_public_result_from_summary()`:

```python
        gate = summary.get("sealed_validation_gate")
        if gate is not None:
            if not isinstance(gate, Mapping):
                raise RunnerExecutionError(f"{job_id}: eval summary sealed_validation_gate must be an object")
            try:
                validate_sealed_validation_gate(gate)
            except ValueError as exc:
                raise RunnerExecutionError(f"{job_id}: {exc}") from exc
            public_metrics["sealed_validation_gate"] = dict(gate)
```

Also project `candidate_package_loaded_in_eval` when present, while keeping compatibility with the existing `candidate_package_loaded` field.

- [ ] **Step 4: Run and confirm GREEN**

Run the same runner unittest command from Step 2.

## Task 5: Full Focused Verification And Phase1 Record

- [ ] **Step 1: Run focused Phase1 suite**

Run:

```powershell
python -m unittest tests.unit.test_eval_isolation tests.unit.test_isaac_lab_backend tests.unit.test_materialize_job_script tests.unit.test_push_recovery_task tests.unit.test_subprocess_runner -v
```

- [ ] **Step 2: Run repository-level verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

- [ ] **Step 3: Write implementation record**

Create `PLAN/eval/Phase1/2026-07-06_eval_upgrade_phase1_implementation_record.md` summarizing:

- files changed;
- final eval argv contract;
- candidate isolation gate fields;
- tests run and exact outcomes;
- known Phase2+ remaining work.

## Self-Review

- Phase1 plan implements every listed Phase1 deliverable: eval CLI no candidate path, backend eval argv no candidate package path, fail-closed candidate path arguments, explicit candidate isolation gate summary.
- Plan intentionally does not consume gate in final accept decision yet; the global eval plan assigns that to Phase5.
- Shared helper is task-neutral and does not know push-recovery failure modes.
- Candidate identity is preserved as a non-path id only; candidate package path remains train-only.
