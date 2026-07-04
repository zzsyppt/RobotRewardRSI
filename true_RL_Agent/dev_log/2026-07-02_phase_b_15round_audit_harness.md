# 2026-07-02 Phase B 15-round Audit Harness

## 背景

Phase 14 后检查 B 阶段验收入口时发现 `scripts/run_loop.py`、`scripts/run_round.py`、`scripts/materialize_job.py` 仍是 scaffold，会直接阻断 15-round 验收。

当前 BackendAdapter 仍按计划处在 dry-run materialization 边界：生成 IsaacLab/RSL-RL CLI job spec，但不在 adapter 内启动真实训练。因此本次补的是安全 dry-run audit harness，用于验证 15 轮 AgentCore 决策闭环、prompt payload、curriculum 禁用态和 anti-tuning gate，不冒充 GPU 训练。

## 实现内容

- 新增 `true_rl_agent.audit.run_loop_audit`：
  - 生成 15 轮混合 candidate replay。
  - 每轮保存 prompt payload。
  - 运行 PushRecoveryTaskPlugin + IsaacLabRslRlBackendAdapter + DryRunAuditRunner。
  - 写出 ledger 和 `audit_summary.json`。
  - 检查 prompt public-only、curriculum 非 LLM 控制、训练 job 无 LLM schedule、parameter-only 不被 accept、存在 code-level accept。
- 新增 `DryRunAuditRunner`：
  - 不启动真实训练。
  - 由 runner-owned dry-run metadata 生成 runtime activation evidence 和 public push eval result。
- 替换 scaffold scripts：
  - `scripts/run_loop.py`：运行多轮 dry-run audit，默认 15 轮。
  - `scripts/run_round.py`：运行单轮 dry-run audit smoke。
  - `scripts/materialize_job.py`：输出 dry-run job bundle。
- 更新 `.gitignore`，忽略 `true_RL_Agent/runs/` 运行产物。
- 新增 `tests/integration/test_dry_run_audit_loop.py` 覆盖 15-round audit、CLI 入口、output-dir 复用保护、materialize component 显式选择。

## Review 结果

- 确认 15-round audit 每轮 prompt payload 都由 public projection 构造并保存。
- 确认 prompt payload 中不出现 private seed、hidden eval、trusted eval/seed、checkpoint selector 或 API key 风险字符串。
- 确认课程仍是 task_default / `llm_controlled=false`，没有把 LLM curriculum schedule 写入训练 job。
- 确认 parameter-only candidates 有高分也不会进入 accept。
- 确认 code-level candidates 只有在 runner-owned runtime activation evidence 存在且 public score delta 达标时才 accept。
- 确认 run output 目录不能复用已有 ledger，避免污染下一次 audit。

## 验证

- `python -m unittest tests.integration.test_dry_run_audit_loop -v`
- `python -m unittest discover -s tests -v`
- `python scripts/run_loop.py --rounds 15 --session-id local-final-15 --output-dir runs/local_audit_final_15`
- `rg` 旧工程路径和父项目 runtime 依赖扫描无命中。
- `rg` 危险 shell/subprocess 调用扫描无命中。
- `git diff --check -- true_RL_Agent .gitignore`
- `python -m compileall -q src tests scripts`
- `python -m pip install --dry-run -e .`

## 结论

B 阶段正式服务器验收前的 15-round dry-run audit harness 已完成。它验证的是 AgentCore/prompt/curriculum/anti-tuning 边界，不是实际 Isaac/RSL-RL GPU 训练。
