# true_RL_Agent

Independent humanoid reinforcement-learning policy improvement agent.

This project is intentionally separate from the parent `humanoid_agent` codebase.
Old Phase 8E code, local Codex source, and prior audit artifacts are reference
material only. Runtime code in this project must not import parent project
packages, call parent scripts, or treat parent configs as hidden dependencies.

## Scope

- First task: Unitree G1/H1 push recovery from random policy initialization.
- First end-to-end loop: candidate -> verify -> train -> eval -> decision -> ledger.
- LLM role: propose code-level research hypotheses and candidate packages.
- Runner role: own budgets, checkpoint selection, training/eval execution, decisions,
  artifacts, public projection, and ledger records.

## Non-Goals

- This is not a generic autonomous agent framework.
- The first version must not optimize curriculum schedules as LLM-controlled knobs.
- Candidate acceptance must not reduce to reward-weight, terrain-range, PPO-hyperparameter,
  or eval-threshold tuning.
- Stair climbing is an extension target, not part of the first runtime deliverable.

## Current Stage

Phase 0 is the project boundary scaffold:

- installable package metadata under `pyproject.toml`
- source package under `src/true_rl_agent`
- local configs/scripts that fail closed until implemented
- boundary tests proving the project does not import the parent package

