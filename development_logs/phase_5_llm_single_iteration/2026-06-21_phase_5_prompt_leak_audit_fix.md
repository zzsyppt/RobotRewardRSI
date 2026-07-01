# Phase 5 prompt leak audit and public-only rerun

Date: 2026-06-21

## Scope

This audit checks whether Phase 5 leaked hidden evaluation information into the
LLM proposal prompt, fixes any real issues, and reruns the single-candidate
Phase 5 flow when needed.

## Findings

### 1. IID-hidden and eval seed information leaked into the original LLM prompt

Status: real issue; fixed.

The original live run:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001
```

stored a `prompt.txt` of 27,441 characters.  It contained raw Phase 2 summary
fields that should not have been proposal input, including:

- `iid_hidden_eval`;
- per-seed `iid_hidden_success_rate` and hidden failure fields;
- `phase2_iid_hidden`;
- eval seeds `1001` and `2001`;
- raw `config.evaluation`.

Root cause:

- `src/humanoid_agent/phase5_llm_agent/prompting.py` passed the raw summary
  through `_clip_json`.
- `_clip_json` only projected public-only fields when the JSON exceeded the
  size cap.  The real Phase 2 summary did not exceed the cap, so the hidden
  fields passed through unchanged.

Consequence:

- The original proposal may still be useful as an engineering result, but it is
  not a clean LLM-Agent result.
- Its IID-hidden improvement must not be used as independent evidence, because
  the Agent saw IID-hidden data before proposing the patch.

Fix:

- `build_phase5_messages` now calls `build_public_prompt_summary` before any
  clipping.
- The prompt summary is an allowlisted projection containing:
  - public aggregate success/failure/condition-flag statistics;
  - public per-run success/failure/condition-flag diagnostics, with only
    `run_index` identifiers;
  - compute summary;
  - workflow-complete/difficulty-calibration flags.
- The prompt projection excludes:
  - `iid_hidden_eval`;
  - any `iid_hidden_*` per-seed fields;
  - raw `config.evaluation`;
  - eval seeds such as `1001` and `2001`;
  - exact hidden profile names.

Regression test:

```bash
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_llm_agent_selftest.py
```

Server result:

```text
prompt_redacts_iid_hidden_and_eval_seed_fields: pass
```

Additional server probe on the real Phase 2 summary:

```bash
cd /data/yuyangliu/AGENT
rm -rf /tmp/phase5_prompt_redaction_probe
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_llm_single_iteration.py \
  --provider dry_run \
  --summary-file /data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json \
  --output-dir /tmp/phase5_prompt_redaction_probe

for token in iid_hidden iid-hidden phase2_iid_hidden iid_hidden_success_rate iid_hidden_eval 2001 1001; do
  grep -q "$token" /tmp/phase5_prompt_redaction_probe/prompt.txt && echo "FOUND:$token"
done
```

No tokens were found.

### 2. The Phase 5 formal protocol used five seeds, not two

Status: not a bug in the executed result.

The implementation ran the fixed Phase 2 training/eval contract, whose protected
training seeds are:

```text
101, 102, 103, 104, 105
```

This is stricter than the plan's minimum `2 seeds + optional 3rd seed` rule and
matches the Phase 2 baseline seed count.  No change was needed.

### 3. `candidate.diff` contained YAML serialization noise

Status: real issue for future Agent feedback; fixed.

The full-file `candidate.diff` is still retained for audit, but Phase 5 now also
writes:

```text
operation_diff.json
operation_diff.md
```

These files record only semantic config operations and exclude YAML
reserialization noise.  `scripts/phase5_llm_agent_selftest.py` now checks that
the operation-only diff is written.

## Clean public-only rerun

After the prompt fix, a fresh live DeepSeek run was executed:

```bash
cd /data/yuyangliu/AGENT
RUN_ID=2026-06-21_deepseek_v4_pro_public_only_001 \
PHASE5_PROVIDER=deepseek \
bash scripts/phase5_live_provider.sh
```

Run directory:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_public_only_001
```

Prompt audit:

```text
No matches for: iid_hidden, iid-hidden, phase2_iid_hidden,
iid_hidden_success_rate, iid_hidden_eval, 2001, 1001.
```

Verifier status:

```text
pass
```

Operation-only diff:

| # | document | path | old | new |
| ---: | --- | --- | --- | --- |
| 1 | `phase1` | `profiles.phase2_train.push.velocity_range_mps.x` | `[0.0, 0.0]` | `[-0.3, 0.3]` |
| 2 | `phase1` | `profiles.phase2_train.push.velocity_range_mps.y` | `[0.0, 0.0]` | `[-0.3, 0.3]` |

Train-start smoke:

```bash
cd /data/yuyangliu/AGENT
CANDIDATE_DIR=/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_public_only_001 \
SEED=501 \
bash scripts/phase5_train_start_smoke.sh
```

Status: `pass`.

Formal five-seed evaluation:

```bash
cd /data/yuyangliu/AGENT
CANDIDATE_DIR=/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_public_only_001 \
RUN_NAME=formal_5seed_eval \
bash scripts/phase5_run_candidate_eval.sh
```

Summary:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_public_only_001/formal_5seed_eval/summary.json
```

Decision:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_public_only_001/accept_reject.json
```

## Clean public-only results

Baseline public mean:

```text
0.521875
```

Clean candidate public results:

| Seed | Baseline public | Clean candidate public | Delta |
| --- | ---: | ---: | ---: |
| 101 | 0.492188 | 0.679688 | +0.187500 |
| 102 | 0.421875 | 0.757812 | +0.335937 |
| 103 | 0.757812 | 0.726562 | -0.031250 |
| 104 | 0.414062 | 0.734375 | +0.320313 |
| 105 | 0.523438 | 0.656250 | +0.132812 |

Aggregate:

```text
candidate_public_mean = 0.7109374
candidate_public_std  = 0.0417070559
candidate_public_SE   = 0.0186519624
public_delta          = 0.1890624
pooled_standard_error = 0.0652424654
acceptance_threshold  = 0.0652424654
decision              = accept
```

IID-hidden diagnostics, recorded after the proposal and not used for decision:

```text
baseline_mean  = 0.465625
candidate_mean = 0.7109372
delta          = 0.2453122
```

## Final decision after audit

Phase 5 remains complete, but the valid clean result is now:

```text
2026-06-21_deepseek_v4_pro_public_only_001
```

The earlier contaminated run:

```text
2026-06-21_deepseek_v4_pro_live_001
```

is superseded for clean Phase 5 claims.

## Remaining carry-forward

- Phase 6 should use the public-only prompt projection and operation-only diff
  artifacts.
- IID-hidden diagnostics must stay outside Agent prompts.
- If a future Agent feedback summary includes prior candidate results, it must
  include public metrics and operation diffs only.
