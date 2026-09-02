# Phase 1 Reference Inventory And Extraction Map

This document is the Phase 1 audit artifact for `true_RL_Agent`. Its purpose is
to decide what can be copied or rewritten from the parent project, what is only a
design-only reference, and what must stay out of the new runtime.

## Runtime Dependency Boundary

- New runtime code must not import the parent `humanoid_agent` package.
- New runtime code must not call parent scripts from `D:\humanoid_agent\scripts`
  or `/data/yuyangliu/AGENT/scripts`.
- New runtime configs must not treat parent configs as effective configs.
- New runtime code must not require Codex, the local `codex/` checkout, or Codex
  source files on the server.
- Parent project and Codex source can inform design, but copied code must be
  renamed into the `true_rl_agent` namespace and tested there.

## Parent Asset Map

| Area | Parent source | Reuse decision | copy target |
| --- | --- | --- | --- |
| candidate schema | `src/humanoid_agent/code_sandbox/loader.py`, `src/humanoid_agent/code_sandbox/components.py` | Rewrite the stable contract. Keep the proven `candidate_manifest.yaml` + `training_code/components.py` shape, but do not import parent dataclasses. | `src/true_rl_agent/candidates/schema.py` and later loader/verifier modules |
| verifier | `src/humanoid_agent/code_sandbox/verifier.py`, `src/humanoid_agent/code_sandbox/isolation.py`, `configs/cold_start/phase8d_verifier.yaml` | Copy concepts only first: static file allowlist, isolated load, public verifier summary. Reimplement after candidate schema is local. | `src/true_rl_agent/candidates/verifier.py` |
| safe checkpoint | `src/humanoid_agent/immutable_runner/safe_checkpoint.py`, `src/humanoid_agent/immutable_runner/checkpoints.py`, `scripts/phase8e_eval_load_safety_selftest.py` | Rewrite into runner-owned checkpoint seal/safe-load modules. Keep failure taxonomy and safe-load tests as reference. | `src/true_rl_agent/runner/checkpoint.py`, `src/true_rl_agent/safety/eval_seal.py` |
| ledger | `src/humanoid_agent/cold_start_lab/phase8e.py`, `src/humanoid_agent/cold_start_lab/phase8e_projection.py` | Copy the append-only JSONL pattern and public-only projection principle, not the old schema names. | `src/true_rl_agent/agent_core/ledger.py` |
| public projection | `src/humanoid_agent/cold_start_lab/phase8e_projection.py`, `src/humanoid_agent/cold_start_lab/phase8e_prompt.py` | Design-only reference for filtering hidden/private data before feedback. New project needs task-agnostic projection. | `src/true_rl_agent/evaluation/public_result.py`, `src/true_rl_agent/memory/public_context.py` |
| training adapter | `src/humanoid_agent/cold_start_lab/training_adapter.py`, `scripts/phase8e_training_adapter_selftest.py` | Do not copy as a universal adapter. Extract runner/backend separation: candidate components feed a backend materializer, runner owns execution. | `src/true_rl_agent/backends/isaac_lab_rsl_rl.py`, `src/true_rl_agent/runner/*` |
| provider/source pack | `src/humanoid_agent/cold_start_lab/phase8e_provider.py`, `src/humanoid_agent/cold_start_lab/phase8e_source_pack.py` | Design-only reference. Reuse the ideas of replay provider, source-read budget, JSON repair, and sensitive-value filtering later. | `src/true_rl_agent/agent_core/provider.py`, `src/true_rl_agent/candidates/prompt_projection.py` |
| full-round sequencing | `src/humanoid_agent/cold_start_lab/phase8e_full_round.py`, `scripts/phase8e_full_round_selftest.py` | Design-only reference for stage ordering and failure classification. Do not copy old command construction into AgentCore. | `src/true_rl_agent/agent_core/round.py`, `src/true_rl_agent/agent_core/loop.py` |
| curriculum compiler | `src/humanoid_agent/cold_start_lab/curriculum_compiler.py`, `scripts/phase8e_curriculum_compiler_selftest.py` | push-specific. Do not copy into the first runtime path. Only keep frozen/runner-owned/task-default lineage interfaces later. | `src/true_rl_agent/curriculum/schema.py`, `src/true_rl_agent/curriculum/lineage.py` |

## Push-Specific Code To Keep Out Of Generic Core

- Phase 8E curriculum payloads with `external_push_curriculum`, `train_only_dr_scale`,
  `reward_only_push_signal`, `public_push_probe`, and target push consolidation.
- Phase 1/Phase 2 push recovery YAML compiler details.
- Push rollout seed lists, push magnitude bins, push timing, and target distribution
  mechanics.
- Any acceptance rule that is calibrated only for the old push recovery run.

These belong in the future push recovery TaskPlugin or BackendAdapter dry-run
phase, not in AgentCore, generic candidate schema, or general runner contracts.

## Random-Only Limitation From The Old Path

The old cold-start path mostly assumes random policy training. The new project
must preserve `random` as the first executable path, but it must also define a
stable warm-start contract early:

- `random`: no baseline checkpoint and every candidate trains from runner-owned
  task defaults.
- `warm_start`: baseline checkpoint is only the training start, never the eval
  checkpoint and never an automatic accept signal.

This Phase 1 document records the limitation; implementation belongs to Phase 2
and Phase 6.

## Codex Source Reference

The local `codex/` checkout is a design-only reference for agent orchestration,
tool execution discipline, sandbox language, review workflow, and provider-style
turn management. It is not a dependency and is intentionally ignored by Git.

Do not vendor Codex source into `true_RL_Agent`. Do not require Codex packages on
the server. Any useful pattern must be re-expressed as local Python code under
`src/true_rl_agent` with tests.

## Minimal Copy Target List

The first implementation passes should copy or rewrite only these concepts:

1. Local candidate identity and manifest shape.
2. Runner-owned public result projection.
3. Append-only ledger records with public-only feedback flags.
4. Task invariant schema and compatibility checks.
5. Train/eval job specs that separate warm-start checkpoint from eval checkpoint.
6. Registry boundaries for TaskPlugin and BackendAdapter.

Everything else stays as design-only reference until its planned phase.

