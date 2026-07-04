# 2026-07-02 Phase 11 Push Recovery 参考任务

## 本阶段目标

按计划完成 Phase 11：把 push recovery 作为第一版参考任务，验证新项目的 TaskPlugin、固定协议、backend job materialization、public metrics、invariant gate 和 anti-tuning gate。

## 实现内容

- 新增 `PushRecoveryTaskConfig`，固定 fall threshold、episode length、push magnitude/timing、success predicate、eval seed visibility、budget、checkpoint selector、accept threshold。
- 新增 `PushRecoveryTaskPlugin`：
  - 只声明任务语义、`TaskInvariantSpec`、固定 train protocol、固定 eval protocol。
  - 支持 `random` 与 `warm_start` initialization。
  - 预留 push curriculum capability，但 `effective_now=false`。
  - 不启动训练，不拼后端命令。
- 新增 push public metrics：
  - `build_push_public_result` 输出 success rate、failure breakdown、motion quality。
  - `classify_push_failure` 给出公开 failure 分类。
- Isaac Lab dry-run backend job metadata 记录 task-owned fixed train/eval protocol。
- 新增 push recovery 单测，覆盖 invariant weakening、public result、failure classifier、Isaac job materialization、schedule-only reject、code-level reward callable dry-run accept。

## Review 结果

- 确认 push task 不依赖父项目 package。
- 确认 push plugin 不持有 backend 命令拼接和训练执行权。
- 确认任何改 task invariant 的 bundle 会被现有 invariant gate 拒绝。
- 确认 `reward_weight_scale` / schedule-only candidate 即使 public score 高，也仍然不能成为 accepted improvement。
- 确认 code-level candidate 必须有 runner-owned runtime activation evidence 才能 accept。

## 验证

- `python -m unittest tests.unit.test_push_recovery_task -v`
- `python -m unittest tests.unit.test_isaac_lab_backend tests.unit.test_push_recovery_task -v`
- `python -m unittest discover -s tests -v`
- `rg` 旧工程路径和父项目 runtime 依赖扫描无命中。
- `rg` 危险 shell/subprocess 调用扫描无命中。
- `git diff --check -- true_RL_Agent`
- `python -m compileall -q src tests`
- `python -m pip install --dry-run -e .`

## 结论

Phase 11 已完成。Push recovery 现在是新项目内独立的参考任务，能通过 IsaacLabRslRlBackendAdapter 生成 train/eval job，并通过 dry-run runner 跑通 candidate -> verify -> train -> checkpoint -> eval -> decision -> ledger。
