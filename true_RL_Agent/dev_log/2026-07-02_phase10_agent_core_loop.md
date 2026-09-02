# 2026-07-02 Phase 10 AgentCore 最小闭环

## 本阶段目标

按计划完成 Phase 10：跑通任务无关的单轮和多轮 self-evolution skeleton，不启动真实训练也能 dry-run/replay 完整流程。

## 实现内容

- 新增 `AgentSession`，管理 session 目录、ledger 路径和 round artifact 目录。
- 新增 `ReplayProvider` 和 `CandidateProvider` 协议，用于 replay/dry-run candidate response。
- 新增 `build_public_context` 与 `sanitize_public_context`，从 ledger 构造下一轮 public context，并过滤 private/hidden/secret/token 字段。
- 新增 `AgentLoop`，按 provider -> round -> ledger -> next context 跑多轮 skeleton。
- 扩展 `RuntimeActivationEvidence`，记录 reward/diagnostic 调用次数、reward 输出统计、diagnostic 输出、feature/capability activation。
- 扩展 `run_candidate_round`：
  - 编译并记录 curriculum lineage，但不把 LLM curriculum proposal 接入训练。
  - 写入 `code_change_classification`、`runtime_activation_summary`、`ablation_summary`、`mechanism_attribution`、`anti_tuning_decision`。
  - 对 potential accept candidate 生成 knob-only shadow summary。
  - policy accept 必须通过 runner-owned runtime activation evidence 和 `MechanismEvidenceGate`。
- 扩展 fake backend/runner，使 dry-run 训练产生 runner-owned activation evidence。

## Review 结果

- 确认 parameter-only candidate 即使 public score 很高，也仍然 reject。
- 确认 DiagnosticTerm-only / engineering-only candidate 只能 record，不能 policy accept。
- 确认 code-level candidate 如果缺少 runner activation evidence，会被 `runner_activation_missing` 拒绝。
- 确认 public context 与 ledger 中的 `public_context` 不保留 private seed/token/key 字段。

## 验证

- `python -m unittest tests.integration.test_agent_core_loop -v`
- `python -m unittest tests.integration.test_fake_round -v`
- `python -m unittest tests.unit.test_prompt_projection_and_mechanism -v`
- `python -m unittest discover -s tests -v`
- `rg` 旧工程路径和父项目 runtime 依赖扫描无命中。
- `rg` 危险 shell/subprocess 调用扫描无命中。
- `git diff --check -- true_RL_Agent`
- `python -m compileall -q src tests`
- `python -m pip install --dry-run -e .`

## 结论

Phase 10 已完成。AgentCore 已有可 replay 的 session/loop/round/provider/context/ledger skeleton，且 anti-tuning 和 mechanism attribution 不依赖 candidate 自报。
