# Humanoid Control Agentic Self-Improvement MVP

The project follows `humanoid_agentic_policy_improvement_plan_v3.md`. The
current completed implementation milestone is Phase 6: a public-only
multi-round LLM improvement loop scaffold that starts from the Phase 2
baseline, uses fixed paired seed subsets, and only expands 3 seeds to 5 seeds
when the public decision is near the acceptance threshold. The expensive live
multi-round training run is intentionally a separate execution step.

Start here:

- Full environment installation and reproduction: `setup/environment_reproduction.md`
- Phase -1 evidence and decisions:
  `development_logs/phase_-1_environment_bootstrap/2026-06-20_phase_-1_completion.md`
- Phase 0 official baseline evidence:
  `development_logs/phase_0_official_baseline/2026-06-20_phase_0_official_baseline.md`
- Phase 1 standing push-recovery evidence:
  `development_logs/phase_1_push_recovery_environment/2026-06-20_phase_1_push_recovery.md`
- Phase 1 push-gate audit correction:
  `development_logs/phase_1_push_recovery_environment/2026-06-20_phase_1_gate_audit_fix.md`
- Phase 2 baseline variance evidence:
  `development_logs/phase_2_baseline_variance/2026-06-20_phase_2_baseline_variance.md`
- Phase 3 verifier and smoke evidence:
  `development_logs/phase_3_verifier_smoke/2026-06-20_phase_3_verifier_smoke.md`
- Phase 4 rule-baseline evidence:
  `development_logs/phase_4_rule_baseline/2026-06-20_phase_4_rule_baseline.md`
- Phase 5 LLM-agent scaffold evidence:
  `development_logs/phase_5_llm_single_iteration/2026-06-21_phase_5_llm_agent_scaffold.md`
- Phase 5 DeepSeek live validation:
  `development_logs/phase_5_llm_single_iteration/2026-06-21_phase_5_deepseek_live_validation.md`
- Phase 5 formal accept/reject:
  `development_logs/phase_5_llm_single_iteration/2026-06-21_phase_5_formal_accept_reject.md`
- Phase 5 prompt-leak audit and clean public-only rerun:
  `development_logs/phase_5_llm_single_iteration/2026-06-21_phase_5_prompt_leak_audit_fix.md`
- Phase 6 public-only multi-round loop implementation:
  `development_logs/phase_6_multi_round_loop/2026-06-21_phase_6_loop_implementation.md`
- Server status: `setup/server_check.md`
- Direct version pins: `setup/locks/`

Quick validation on the server:

```bash
cd /data/yuyangliu/AGENT
source scripts/activate_environment.sh
python scripts/smoke_torch_cuda.py
bash scripts/smoke_isaac_headless.sh
bash scripts/smoke_dual_gpu.sh
bash scripts/phase0_official_baseline.sh
bash scripts/phase1_push_recovery_smoke.sh
DEVICE=cuda:0 RUN_ROOT=/data/yuyangliu/runs/phase_2_baseline_variance_5seed \
  bash scripts/phase2_baseline_variance.sh
RUN_ROOT=/data/yuyangliu/runs/phase_3_verifier_smoke \
  bash scripts/phase3_verifier_smoke.sh
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase4_rule_baseline_selftest.py
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase4_rule_baseline.py \
  --summary-file /data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json \
  --output-dir /data/yuyangliu/runs/phase_4_rule_baseline/manual_repro
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_llm_agent_selftest.py
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase6_llm_loop_selftest.py
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_llm_single_iteration.py \
  --provider dry_run \
  --summary-file /data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json \
  --output-dir /data/yuyangliu/runs/phase_5_llm_single_iteration/manual_dry_run
# Live provider run requires a server-side secret file such as
# /data/yuyangliu/secrets/deepseek.env and explicit spend authorization.
RUN_ID=manual_deepseek_live PHASE5_PROVIDER=deepseek bash scripts/phase5_live_provider.sh
CANDIDATE_DIR=/data/yuyangliu/runs/phase_5_llm_single_iteration/manual_deepseek_live \
  RUN_NAME=formal_5seed_eval \
  bash scripts/phase5_run_candidate_eval.sh
# Phase 6 dry-run planning does not call the provider or train.
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase6_multi_round_loop.py \
  --rounds 3
# Live Phase 6 requires the server-side provider secret and explicit spend flag.
source /data/yuyangliu/secrets/deepseek.env
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase6_multi_round_loop.py \
  --execute \
  --provider deepseek \
  --allow-api-spend \
  --rounds 3
```

Large environments, caches, raw logs, and checkpoints live under
`/data/yuyangliu` and are intentionally excluded from Git.
