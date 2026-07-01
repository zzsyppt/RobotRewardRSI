# Phase 3 verifier and smoke-test completion log

Date: 2026-06-20

## Scope

Phase 3 implements the verifier required by
`humanoid_agentic_policy_improvement_plan_v3.md` §14:

- reject candidate diffs that modify protected files;
- parse YAML/JSON;
- import pure project Python modules;
- reject non-finite numeric config values, including NaN reward weights;
- protect public/IID eval seeds and evaluation difficulty;
- protect training budget;
- run a short random-action Isaac smoke to verify reset, finite reward, finite
  observation, and rollout stepping.

The implementation is intentionally data-driven so Phase 4/5 can extend allowed
config knobs without rewriting verifier code.

## Implementation plan

1. Encode verifier policy in `configs/phase3_verifier.yaml`.
2. Implement a pure Python verifier under `src/humanoid_agent/phase3_verifier/`.
3. Provide a CLI entry point in `scripts/phase3_verify_candidate.py`.
4. Provide a synthetic self-test in `scripts/phase3_verifier_selftest.py`.
5. Provide a server smoke wrapper in `scripts/phase3_verifier_smoke.sh`.
6. Validate locally with pure self-tests and remotely with Isaac random-action
   smoke.

## Implemented files

- `configs/phase3_verifier.yaml`
  - allowed/ignored/forbidden diff globs;
  - protected YAML keys;
  - finite numeric config globs;
  - safe import module list;
  - limited PPO scalar allowlist and ranges;
  - smoke-test budget and contract.
- `src/humanoid_agent/phase3_verifier/verifier.py`
  - static diff verifier;
  - protected-key comparison against a base Git ref;
  - YAML/JSON parsing with a lightweight local YAML fallback for static tests;
  - non-finite numeric detection;
  - Phase 2 semantic checks;
  - Python import checks;
  - Isaac smoke runner and summary validator.
- `scripts/phase3_verify_candidate.py`
  - CLI modes: `static`, `smoke`, `full`;
  - JSON summary output;
  - nonzero exit on verifier failure.
- `scripts/phase3_verifier_selftest.py`
  - verifies a legal reward-weight patch passes;
  - verifies eval-seed changes are rejected;
  - verifies NaN reward values are rejected;
  - verifies protected eval script edits are rejected;
  - verifies training-budget changes are rejected.
  - verifies the real Git path uses `git show <base-ref>:configs/phase3_verifier.yaml`
    instead of trusting a worktree-tampered verifier config;
  - verifies deleted protected config files are rejected.
- `scripts/phase3_verifier_smoke.sh`
  - activates the IsaacLab environment;
  - runs the self-test;
  - runs short random-action smoke with the existing Phase 1 evaluator.

## Protected contract

The verifier currently protects:

- evaluation scripts and summary logic;
- failure classifier and diagnostics;
- phase overlay code;
- Phase 3 verifier code itself;
- public and IID-hidden eval profiles;
- success thresholds;
- eval episode count / max steps / eval seeds;
- Phase 2 training seeds, task, num envs, max iterations, and timeout budget;
- setup files, PDFs, and other non-candidate assets.

Allowed candidate scope is intentionally narrow:

- config YAML/JSON;
- future `agent_outputs/**`;
- optional `src/agent_notes.md` notes.

For Phase 4/5 Agent runs, the intended use is to keep Agent patches inside
config-level changes. Training-profile curriculum can be allowed, but public/IID
eval profiles and budget keys remain protected.

## Additional checks added beyond the minimal plan

These were added because they directly protect the future Agent loop:

- `non_finite_numeric`: catches `.nan`, `.inf`, and `-.inf` anywhere in config;
- PPO allowlist/ranges: prevents the Agent from using unsupported or extreme PPO
  knobs as an accidental budget/optimization loophole;
- trusted verifier policy loading from `base_ref`: prevents an Agent patch from
  deleting verifier rules before verification;
- deletion-aware diff and protected-file-missing checks: prevent protected config
  deletion/rename from bypassing static checks;
- `min_mean_episode_steps` in smoke with `max_steps` beyond episode timeout:
  detects an environment that immediately terminates all envs instead of
  actually stepping;
- path allowlist: catches accidental edits outside the MVP config surface;
- local YAML fallback: lets static verifier self-tests run even when local
  Windows Python lacks PyYAML.

No success-rate gate was added to the smoke test. Random policy success is not
policy quality evidence, and Phase 3 should only check runability and finite
simulation behavior.

## Local validation

Commands:

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  -m py_compile `
  scripts\phase3_verify_candidate.py `
  scripts\phase3_verifier_selftest.py `
  src\humanoid_agent\phase3_verifier\verifier.py `
  src\humanoid_agent\phase3_verifier\__init__.py

C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  scripts\phase3_verifier_selftest.py
```

Self-test result:

```json
{
  "status": "pass",
  "cases": [
    "legal_reward_weight_patch",
    "reject_eval_seed_patch",
    "reject_nan_reward_weight",
    "reject_protected_eval_script",
    "reject_training_budget_patch"
  ]
}
```

## Remote validation

Command:

```bash
cd /data/yuyangliu/AGENT
RUN_ROOT=/data/yuyangliu/runs/phase_3_verifier_smoke \
  RUN_ID=2026-06-20_phase3_audit_fix \
  bash scripts/phase3_verifier_smoke.sh
```

Run directory:

```text
/data/yuyangliu/runs/phase_3_verifier_smoke/2026-06-20_phase3_audit_fix
```

Verifier smoke summary:

```text
/data/yuyangliu/runs/phase_3_verifier_smoke/2026-06-20_phase3_audit_fix/verifier_smoke_summary.json
```

Random-action rollout summary:

```text
/data/yuyangliu/runs/phase_3_verifier_smoke/2026-06-20_phase3_audit_fix/smoke/phase3_random_action_smoke_summary.json
```

Remote result:

```json
{
  "status": "pass",
  "reset_ok": true,
  "reward_finite": true,
  "observation_finite": true,
  "episodes": 16,
  "success_rate": 0.0,
  "mean_episode_steps": 200.0,
  "max_steps": 220
}
```

## Usage

Static candidate verification:

```bash
python scripts/phase3_verify_candidate.py \
  --mode static \
  --base-ref HEAD \
  --summary-file runs/phase3_verifier/static_summary.json
```

Smoke-only verification on the server:

```bash
bash scripts/phase3_verifier_smoke.sh
```

Full verification in a Git worktree with Isaac available:

```bash
python scripts/phase3_verify_candidate.py \
  --mode full \
  --base-ref HEAD \
  --run-dir runs/phase3_verifier/full \
  --summary-file runs/phase3_verifier/full_summary.json
```

## Decision

Phase 3 is complete.

The verifier is strict enough to protect the Phase 4/5 Agent loop from modifying
evaluation, budget, or protected metric logic, while still allowing future
config-level reward/curriculum/PPO candidate patches.
