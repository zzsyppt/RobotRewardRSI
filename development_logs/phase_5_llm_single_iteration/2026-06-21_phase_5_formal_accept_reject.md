# Phase 5 formal accept/reject log

Date: 2026-06-21

Audit note added 2026-06-21: this original formal run is superseded by
`2026-06-21_phase_5_prompt_leak_audit_fix.md`.  The original proposal prompt
included IID-hidden/eval-seed fields from the raw Phase 2 summary, so this run
must not be used as the clean Phase 5 Agent result.  The corrected public-only
rerun is:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_public_only_001
```

## Scope

This log completes Phase 5 for one LLM-generated candidate.  It does not start
the Phase 6 multi-round loop.

Phase 5 requirements checked here:

1. use the live LLM proposal generated in the previous DeepSeek validation;
2. run the candidate through fixed-budget multi-seed training and evaluation;
3. compare against the Phase 2 five-seed baseline using public evaluation only;
4. record IID-hidden results only as diagnostics, not as an Agent feedback
   signal;
5. make a noise-aware accept/reject decision.

## Candidate

Candidate directory:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001
```

The accepted candidate changed only the Phase 2 training push profile:

```yaml
profiles.phase2_train.push.velocity_range_mps.x: [-0.3, 0.3]
profiles.phase2_train.push.velocity_range_mps.y: [-0.1, 0.1]
```

No reward, eval, seed, budget, metric, failure-classifier, or verifier edits
were used.

## Added tooling

- `scripts/phase5_accept_reject.py`
  - compares one candidate summary against the baseline summary;
  - enforces protected training/evaluation contract equality;
  - applies the plan rule:

    ```text
    accept iff candidate_public_mean - baseline_public_mean
      > max(0.05, 1.0 * pooled_standard_error)
    ```

  - records IID-hidden as diagnostic-only.
- `scripts/phase5_run_candidate_eval.sh`
  - runs a single Phase 5 candidate through the existing fixed-budget
    `phase2_baseline_variance.sh` workflow;
  - writes `accept_reject.json`;
  - explicitly stops at one candidate decision.

Local self-tests:

```bash
python scripts/phase5_accept_reject.py --selftest
python scripts/phase5_llm_agent_selftest.py
```

Both passed locally with the bundled Codex Python.  The server also passed:

```bash
cd /data/yuyangliu/AGENT
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_accept_reject.py --selftest
```

## Formal run

Command:

```bash
cd /data/yuyangliu/AGENT
CANDIDATE_DIR=/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001 \
RUN_NAME=formal_5seed_eval \
bash scripts/phase5_run_candidate_eval.sh
```

Run directory:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001/formal_5seed_eval
```

Summary:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001/formal_5seed_eval/summary.json
```

Decision:

```text
/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001/accept_reject.json
```

One wrapper issue occurred after all training/eval finished: the first wrapper
revision called `python`, while the server exposes `python3`.  The run summary
was already complete; `accept_reject.json` was generated manually with:

```bash
cd /data/yuyangliu/AGENT
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_accept_reject.py \
  --baseline-summary /data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json \
  --candidate-summary /data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001/formal_5seed_eval/summary.json \
  --proposal-summary /data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001/run_summary.json \
  --analysis-file /data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001/analysis.json \
  --output-file /data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_v4_pro_live_001/accept_reject.json
```

`scripts/phase5_run_candidate_eval.sh` was fixed to use `${PYTHON:-python3}`.

## Results

Baseline public success, from Phase 2:

```text
values = [0.492188, 0.421875, 0.757812, 0.414062, 0.523438]
mean   = 0.521875
SE     = 0.0625194657
```

Candidate public success:

| Seed | Baseline public | Candidate public | Delta |
| --- | ---: | ---: | ---: |
| 101 | 0.492188 | 0.859375 | +0.367187 |
| 102 | 0.421875 | 0.820312 | +0.398437 |
| 103 | 0.757812 | 0.789062 | +0.031250 |
| 104 | 0.414062 | 0.734375 | +0.320313 |
| 105 | 0.523438 | 0.812500 | +0.289062 |

Aggregate candidate public success:

```text
mean = 0.8031248
std  = 0.0460208555
SE   = 0.0205811523
```

Noise-aware comparison:

```text
public_delta          = 0.2812498
pooled_standard_error = 0.0658199622
acceptance_threshold = 0.0658199622
decision             = accept
```

IID-hidden diagnostics, not used for the decision:

```text
baseline_mean  = 0.465625
candidate_mean = 0.721875
delta          = 0.256250
```

Compute accounting:

```text
candidate total GPU-hours estimate = 1.1233333333
```

## Decision

Phase 5 single-candidate result: **accept**.

The accepted Phase 5 candidate materially improves public push-recovery success
under the protected fixed-budget protocol.  The public mean improved by
`+28.1 pp`, exceeding the noise-aware acceptance threshold of `6.58 pp`.

Because this candidate improved clearly, no prompt/input/architecture repair
rerun was needed.  If a later Phase 5 candidate fails, the allowed next action is
to revise Phase 5 prompt/input summarization or patch constraints and rerun one
single-candidate flow.  Multi-round automatic improvement remains Phase 6.

## Carry-forward

- The accepted config has not been copied over the baseline config files.  The
  candidate remains in its run directory until Phase 6 chooses how to represent
  `best`.
- IID-hidden was recorded only for audit.  It must not be included in the next
  Agent prompt.
- The candidate `candidate.diff` still contains YAML reserialization noise from
  dictionary-based YAML dumping.  The semantic change is only the training push
  range above.
