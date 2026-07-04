# Subprocess Runner Execution Gap

## Context

Review after the initial 15-round audit found that the prior run was still a dry-run audit, not a real training/eval execution path. This pass keeps the existing dry-run audit harness but adds the missing runner-owned subprocess execution boundary needed for real backend jobs.

## Changes

- Added `SubprocessRunner` for runner-owned structured subprocess execution.
- Required `cli_argv` instead of shell command strings, with executable allowlisting and `shell=False`.
- Added train/eval summary JSON consumption, runner-owned checkpoint sealing, eval public-result parsing, timeout handling, and redacted failure output.
- Extended the Isaac Lab + RSL-RL adapter to emit structured `cli_argv`, `summary_path`, and train `sealed_checkpoint_uri`.
- Updated round execution so eval CLI metadata is rebound from the backend planned checkpoint to the runner-sealed train checkpoint before eval runs.
- Updated `materialize_job.py` to export runner execution metadata for server preflight.
- Updated configs from scaffold/materialization-only to implemented subprocess-runner boundaries while still marking external runtime as required.

## Bugs Caught During Review

- Eval `EvalJobSpec.checkpoint` was replaced with the sealed train checkpoint, but `metadata.cli_argv` still pointed at the backend planned checkpoint.
- External train/eval jobs had no `summary_path`, so `SubprocessRunner` could not consume real job summaries.
- Secret redaction in subprocess failure output was double-escaped and leaked non-`sk-` tokens.

## Verification

- `python -m unittest discover -s tests -v`: 89 tests passed.
- `python -m compileall -q src tests scripts`: passed.
- `python -m pip install --dry-run -e .`: passed.
- Static scans found no parent project runtime references, no shell execution, no scaffold/materialization-only residue, and only one approved `subprocess.run` call in `runner/subprocess_runner.py`.
- `scripts/materialize_job.py` now exports train/eval `cli_argv`, `summary_path`, and train `sealed_checkpoint_uri`.

## Remaining Runtime Status

The code path is ready to invoke real structured train/eval commands, but the external server runtime still needs actual `isaaclab-rsl-rl-train`, `isaaclab-rsl-rl-eval`, and Isaac Lab Python availability before a real 15-round training run can complete.
