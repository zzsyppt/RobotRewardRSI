# Phase 6 public-only multi-round loop implementation

Date: 2026-06-21

## Scope

Phase 6 implements the minimum multi-round LLM improvement loop from the project
plan. This log records code implementation and smoke validation. It does not
claim that the expensive live 3+ round training loop has already been run.

## Design decisions

- Start point is always the Phase 2 human baseline:
  `configs/phase1_push_recovery.yaml`, `configs/phase2_baseline.yaml`, and
  `/data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json`.
- Phase 6 feedback to the next Agent prompt is public-only:
  - uses a public prompt projection;
  - uses operation-only diffs;
  - excludes raw decision files and raw eval summaries;
  - excludes held-out/IID diagnostics and exact eval seed config.
- Candidate comparison uses fixed paired seeds:
  - initial set: `[101, 102, 103]`;
  - expanded set: `[101, 102, 103, 104, 105]`;
  - expand to 5 only when the 3-seed public delta is within
    `0.5 * pooled_standard_error` of the acceptance threshold.
- Acceptance rule is public-only:
  `candidate_public_mean - best_public_mean > max(0.05, 1.0 * pooled_standard_error)`.
- If a candidate is accepted after a 5-seed disambiguation, the next round still
  compares on the canonical 3-seed subset; the 5-seed result remains available
  as the current-best full summary for future ambiguity resolution.

## Implemented files

- `configs/phase6_loop.yaml`
- `src/humanoid_agent/phase6_loop/`
  - `stats.py`
  - `summaries.py`
  - `decision.py`
  - `feedback.py`
- `scripts/phase6_collect_public_summary.py`
- `scripts/phase6_seed_subset_summary.py`
- `scripts/phase6_public_decision.py`
- `scripts/phase6_run_candidate_public_eval.sh`
- `scripts/phase6_multi_round_loop.py`
- `scripts/phase6_llm_loop_selftest.py`

Phase 5 support changes:

- `src/humanoid_agent/phase5_llm_agent/prompting.py`
  preserves safe `phase6_feedback` through clipping and rejects forbidden prompt
  tokens before they can enter the Agent prompt.
- `src/humanoid_agent/phase5_llm_agent/runner.py`
  verifies candidate configs against the canonical protected Phase 1/2 document
  keys even when the current-best files live under a run directory.

## Validation

Local validation used the bundled Codex Python because the default Windows
`python.exe` was unavailable in this shell session.

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile `
  scripts\phase6_collect_public_summary.py `
  scripts\phase6_seed_subset_summary.py `
  scripts\phase6_public_decision.py `
  scripts\phase6_multi_round_loop.py `
  scripts\phase6_llm_loop_selftest.py `
  src\humanoid_agent\phase6_loop\__init__.py `
  src\humanoid_agent\phase6_loop\stats.py `
  src\humanoid_agent\phase6_loop\summaries.py `
  src\humanoid_agent\phase6_loop\decision.py `
  src\humanoid_agent\phase6_loop\feedback.py `
  src\humanoid_agent\phase5_llm_agent\prompting.py `
  src\humanoid_agent\phase5_llm_agent\runner.py
```

Result: pass.

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\phase6_llm_loop_selftest.py
```

Result: pass.

Covered checks:

- clear 3-seed accept does not expand to 5;
- near-threshold 3-seed result expands to 5;
- clear-margin 3-seed result stays at 3;
- Phase 6 feedback survives into the next prompt;
- prompt excludes held-out/IID and eval-seed tokens;
- feedback anonymizes seed IDs before prompt use;
- dry-run loop starts from Phase 2 baseline;
- dry-run feedback file is public-only.

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\phase5_llm_agent_selftest.py
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\phase5_accept_reject.py --selftest
```

Result: both pass.

## Reproduction commands

Dry-run plan on the server:

```bash
cd /data/yuyangliu/AGENT
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase6_llm_loop_selftest.py
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase6_multi_round_loop.py --rounds 3
```

Live Phase 6 execution:

```bash
cd /data/yuyangliu/AGENT
source /data/yuyangliu/secrets/deepseek.env
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase6_multi_round_loop.py \
  --execute \
  --provider deepseek \
  --allow-api-spend \
  --rounds 3
```

The live command calls the provider, trains/evaluates each accepted candidate
under fixed public seeds, and writes results under
`/data/yuyangliu/runs/phase_6_multi_round_loop/`.

## Remaining risks

- The expensive live multi-round run has not been executed in this implementation
  log. It should be run only when GPU time/API spend is intended.
- Final IID-hidden validation remains milestone/final audit only. Its results
  must not be written into any later Agent prompt.
- Phase 6 is still config-level only. More invasive reward-code or observation
  changes remain Phase 7/extension work.

## Pre-run audit follow-up

Date: 2026-06-21

Two run-blocking issues were found and fixed before starting the expensive live
loop.

1. 5-seed disambiguation could pollute the next round's best-summary path after
   a rejected candidate. The 5-seed comparator path is now local to that
   disambiguation step; the canonical next-round best summary remains the fixed
   3-seed public subset unless a candidate is explicitly accepted and promoted.
2. Public training/eval subprocess failures could previously raise out of the
   loop and abort all remaining rounds. Public eval or decision failures now
   write a rejected failure decision for that round and continue to the next
   round with the previous best unchanged. Best recheck failures after an accept
   are recorded while keeping the accepted best.

Additional hardening:

- `build_phase5_messages()` now performs a final forbidden-token scan on the
  exact messages that will be written and sent to the provider.
- `scripts/phase6_llm_loop_selftest.py` now includes orchestrator-level tests
  for:
  - rejected 5-seed expansion preserving the previous 3-seed best path;
  - public eval failure rejecting the candidate and continuing the loop.

Validation after this follow-up:

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile `
  scripts\phase6_multi_round_loop.py `
  scripts\phase6_llm_loop_selftest.py `
  src\humanoid_agent\phase5_llm_agent\prompting.py

& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\phase6_llm_loop_selftest.py
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\phase5_llm_agent_selftest.py
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\phase5_accept_reject.py --selftest
```

Result: all pass.

## Live 5-round DeepSeek run

Date: 2026-06-21

Run:

```bash
cd /data/yuyangliu/AGENT
ROUNDS=5 bash scripts/phase6_live_launch.sh 2026-06-21_phase6_live_deepseek_001
```

Server artifacts:

- run directory:
  `/data/yuyangliu/runs/phase_6_multi_round_loop/2026-06-21_phase6_live_deepseek_001`
- launch log:
  `/data/yuyangliu/runs/phase_6_multi_round_loop/2026-06-21_phase6_live_deepseek_001.launch.log`
- final summary:
  `/data/yuyangliu/runs/phase_6_multi_round_loop/2026-06-21_phase6_live_deepseek_001/phase6_loop_summary.json`
- accepted best configs:
  - `/data/yuyangliu/runs/phase_6_multi_round_loop/2026-06-21_phase6_live_deepseek_001/current_best/round_03_phase1_push_recovery.yaml`
  - `/data/yuyangliu/runs/phase_6_multi_round_loop/2026-06-21_phase6_live_deepseek_001/current_best/round_03_phase2_baseline.yaml`

Final loop status: `pass`.

All five rounds used public-only 3-seed evaluation. No round entered 5-seed
expansion because the observed deltas were either clearly accepted or clearly
rejected under the configured threshold.

| round | decision | public success values | mean | delta vs current best | threshold |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | rejected | 0.656250, 0.500000, 0.507812 | 0.554687 | -0.002604 | 0.114228 |
| 2 | rejected | 0.023438, 0.015625, 0.085938 | 0.041667 | -0.515625 | 0.104686 |
| 3 | accepted | 0.882812, 0.859375, 0.820312 | 0.854166 | +0.296875 | 0.103906 |
| 4 | rejected | 0.890625, 0.804688, 0.789062 | 0.828125 | -0.026041 | 0.050000 |
| 5 | rejected | 0.570312, 0.539062, 0.585938 | 0.565104 | -0.289062 | 0.050000 |

The only accepted candidate was round 3. Its semantic operation diff was:

- `phase1.profiles.phase2_train.push.velocity_range_mps.y`:
  `[0.0, 0.0] -> [0.0, 0.3]`
- `phase1.profiles.phase2_train.push.interval_range_s`:
  `[1.0, 1.0] -> [0.8, 1.2]`

This is consistent with the Phase 6 scope: the loop started from the Phase 2
baseline, used LLM-proposed config/reward changes, trained/evaluated each
candidate on public seeds only, and accepted only statistically clear public
improvement.

Security / anti-cheating checks:

- The final `phase6_loop_summary.json` reports `start_from: phase2_baseline`
  and `status: pass`.
- A recursive forbidden-token scan over all five live prompt files found no
  occurrences of hidden/eval seed fields or protected decision artifacts:
  `iid_hidden`, `iid-hidden`, `phase2_iid_hidden`, `iid_hidden_eval`,
  `iid_hidden_success_rate`, `accept_reject`, `"seed":`, `"evaluation":`,
  or `phase2_public`.
- The final best pointer remained on round 3 after rounds 4 and 5 were
  rejected, confirming that rejected candidates did not pollute the best config.

## Live launch wrapper

Date: 2026-06-21

Added `scripts/phase6_live_launch.sh` so the server can start the live Phase 6
loop with a simple reproducible command while keeping provider secrets in the
server-side environment file. The wrapper records a launch log and PID under
`/data/yuyangliu/runs/phase_6_multi_round_loop/`.

Example:

```bash
cd /data/yuyangliu/AGENT
ROUNDS=5 bash scripts/phase6_live_launch.sh 2026-06-21_phase6_live_deepseek_001
```

`ROUNDS=5` is still within the Phase 6 configured maximum and does not enter
Phase 7 extension work.

## Prompt scan false-positive follow-up

Date: 2026-06-21

A second pre-run audit found that the final prompt text scan was too coarse:
bare numeric tokens `1001` and `2001` were treated as forbidden substrings. That
can falsely reject ordinary floating-point public metrics such as `0.820012` or
`0.210015`, because the rendered JSON prompt is just text.

Fix:

- removed bare `1001` and `2001` from text-substring forbidden-token lists;
- kept named high-risk tokens such as `iid_hidden`, `accept_reject`, and
  `heldout_mean`;
- added structural selftest checks that prompt/public feedback do not contain
  `"seed":`, `"evaluation":`, or `phase2_public`;
- added a regression check that benign public floats containing those digit
  sequences do not trigger the final prompt guard.

Validation:

```powershell
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile `
  scripts\phase6_llm_loop_selftest.py `
  scripts\phase6_multi_round_loop.py `
  src\humanoid_agent\phase5_llm_agent\prompting.py `
  src\humanoid_agent\phase6_loop\feedback.py

& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\phase6_llm_loop_selftest.py
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\phase5_llm_agent_selftest.py
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\phase5_accept_reject.py --selftest
```

Result: all pass.
