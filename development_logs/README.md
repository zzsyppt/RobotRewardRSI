# Development logs

Each project phase has an isolated directory. When a phase is complete, its
directory receives one final Markdown log containing:

- scope and acceptance criteria;
- exact dependency and Git versions;
- commands executed;
- validation evidence;
- failures, workarounds, and remaining risks;
- final go/no-go decision.

Large raw logs and checkpoints remain on the training server. The phase log
records their absolute paths and the commands required to reproduce them.

Completed phases:

- Phase -1: `phase_-1_environment_bootstrap/2026-06-20_phase_-1_completion.md`
- Phase 0: `phase_0_official_baseline/2026-06-20_phase_0_official_baseline.md`
- Phase 1: `phase_1_push_recovery_environment/2026-06-20_phase_1_push_recovery.md`
- Phase 1 gate audit correction: `phase_1_push_recovery_environment/2026-06-20_phase_1_gate_audit_fix.md`
- Phase 2: `phase_2_baseline_variance/2026-06-20_phase_2_baseline_variance.md`
- Phase 2 audit follow-up: `phase_2_baseline_variance/2026-06-20_phase_2_audit_followup.md`
- Phase 3: `phase_3_verifier_smoke/2026-06-20_phase_3_verifier_smoke.md`
- Phase 4: `phase_4_rule_baseline/2026-06-20_phase_4_rule_baseline.md`
- Phase 4 audit follow-up: `phase_4_rule_baseline/2026-06-20_phase_4_audit_followup.md`
- Phase 5 LLM-agent scaffold: `phase_5_llm_single_iteration/2026-06-21_phase_5_llm_agent_scaffold.md`
- Phase 5 DeepSeek live validation: `phase_5_llm_single_iteration/2026-06-21_phase_5_deepseek_live_validation.md`
- Phase 5 formal accept/reject: `phase_5_llm_single_iteration/2026-06-21_phase_5_formal_accept_reject.md`
- Phase 5 prompt-leak audit and clean public-only rerun: `phase_5_llm_single_iteration/2026-06-21_phase_5_prompt_leak_audit_fix.md`
- Phase 6 public-only multi-round loop implementation: `phase_6_multi_round_loop/2026-06-21_phase_6_loop_implementation.md`
