# Phase 1：本地 Codex 参考与取舍

2026-09-23。参考对象为本地 `D:/humanoid_agent/codex` 当前源码；先读取 AGENTS 并使用 CodeGraph 定位。它提供实现思路，不构成本项目运行或验收证据。

| 借鉴 | 源头与真实消费者 | 本项目的有限落点 |
|---|---|---|
| 实时与重放共用转换规则 | `codex-rs/app-server-protocol/src/protocol/thread_history.rs:355` 的 `ThreadHistoryBuilder`；实时消费者 `app-server/src/thread_state.rs:211` 调用 `handle_event`，重放消费者 `app-server/src/request_processors.rs:717` 调用 `handle_rollout_item` 后 `finish` | 复用绑定 phase receipt 和 task-owned 公开投影，使 live/replay 同义；ledger 的 `public_summary` 是派生表示。保留现有 journal，不另建事件总线或日志。 |
| 来源记录与模型视图分离 | `codex-rs/history/src/lib.rs:46` 的 `ResponseItemEnvelope` 分存 item 与 metadata；`core/src/context_manager/history.rs:422` 的 `record_items_with_metadata` 处理副本，`:468` 的 `for_prompt` 生成模型输入 | 保留 `PublicContextBuilder` 的 validated history，给任务 projector 独立副本；身份比较依赖未被投影修改的记录。沿用现有结构，不引入通用 envelope 平台。 |
| 缺失不能凭当前状态补齐 | `codex-rs/history/src/compaction_checkpoint.rs:1` 保留最新无效 checkpoint、未知 producer，不静默退回较早有效项；`history/src/reconciled_retained_context.rs:1` 恢复部分消息不能清除原缺失标记 | 缺 runtime、未知字段、无 term、未执行、实测零和容量遗漏分别保留；不从其他 trial 补测量。不复制 Codex 的授权、加密 checkpoint 或消息恢复体系。 |

裁剪只借鉴一个关系约束：`core/src/context_manager/history.rs:540` 的 `remove_first_item` 会同步处理配对项。本项目沿用任务最小投影及 capacity receipt，避免裁剪出脱离身份、分母或来源的数值；不复制其具体 token 数字。

## 本次独立回归

正常测试文件：`tests/unit/test_feedback_evidence_boundary.py`。测试内任务 ID 为 `fixture_orbit_tracking`，仅复用通用 runner 决策 fixture；没有新增生产任务、任务公式或服务。

- 已验证公共记录 → 任务投影 → 任务记忆 → `PublicContextBuilder.build_prompt` → `OpenAICompatibleCandidateProvider._chat_completion_request` → 最终 JSON。自有误差 `null`/`0.0`、覆盖计数 4/6、round/candidate 身份和证据编号保持一致。任务反馈评分 0.75 与内部接受评分 0.25 分开。
- 私有字段在共享过滤时去除；accept/sealed 详细标记由任务 public-dev allowlist 去除。测试确认标记确实存在于输入，而不是只对空输出检查。
- 合法的原地反馈删减不改变调用者及已验证历史；重复构建相同。原地修改 candidate 身份或 accepted 决定均被拒绝，且不能污染已验证历史。

执行命令（仓库根目录）：

```powershell
& 'C:/Users/lover/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m unittest tests.unit.test_feedback_evidence_boundary -v
```

结果：3 个测试通过。另用内存 monkeypatch 将该模块 `deepcopy` 暂换成恒等函数，仅运行身份篡改测试：candidate 与 decision 两个子例均失败、0 个异常，证明回归能检出此前的原地修改漏洞；未改动生产文件。

范围：这是合成任务的 CPU 边界回归，起点是现有 ledger 公共结果经真实 acceptance replay 得到的记录；没有运行第三任务的训练或 phase evaluator，没有调用 provider，也不代替绑定 receipt 真实性、完整生命周期、容量压力或模型诊断质量的其他验收。
