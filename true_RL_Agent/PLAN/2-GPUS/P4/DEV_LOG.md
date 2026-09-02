# P4 开发日志

## 2026-08-17

### 基线与约束

- 在 `true-rl-agent-refactor`、基线 `3af99dd6999f5119c5458fd6218efe22e34febda` 上实施。
- 重读主计划和 P4 handoff，固定目标为“两条 sibling trial、一次 selector、一次 ledger commit”，不引入通用 DAG/Graph 框架。
- 全程仅本地 CPU/fake；未调用 provider、GPU 或服务器。

### 合同与身份

- 增加 generation session contract、manifest、lane assignment、trial status 与 create-once ref。
- 增加 Session Genesis v4 generation capability；保持 v2/v3 width=1 读取和运行边界。
- 增加第一代 Genesis parent public-dev bootstrap，并将 Stair phased cost 升为 v2。

### 执行与证据

- 将 P2 round 进一步收敛为 prepare/train/finalize 原语，旧 width=1 wrapper 继续走同源实现。
- 增加固定 Width=2 GenerationCoordinator：串行 A→B author，ready 后并行 train/dev，封存 selection，再并行 accept，仅 winner sealed。
- 增加 generation manifest/selection/execution receipts 和 generation-v1 record。
- 增加 train/eval phase receipt 的独立加载器，使 generation replay 能从落盘证据重算，而不是只信 record 内投影。
- 增加 generation replay、head-transition projection、public history/source delivery 和 mixed-writer 拒绝。

### 收敛修正

- 修复 Session Genesis v2 历史错误信息兼容，避免新 v4 capability 检查改变旧测试语义。
- 将 lane intent 与 B sibling exclusion 真实注入 provider prompt，并在 coordinator 与 ledger replay 两个决定边界验证；manifest 不再只是声明未被消费的合同。
- 将 B exclusion 与 A verifier-owned operation delta 交叉绑定，篡改 record 或 prompt 均失败。
- 删除 Stair phased cost v2 内 `phases/max_seed_units/max_episode_executions` 兼容别名，只保留 bootstrap 与 per-generation 两套不同语义的权威字段。
- 将 shared ledger 的报错和注释从 round-only 改为 round/generation 共用措辞，未改变 commit identity。

### 验收记录

- 核心：44 tests OK。
- 扩展联合：219 tests OK。
- Stair cost 清理后：35 tests OK。
- 全量：1217 tests OK，21 skipped。
- Python 编译检查与 `git diff --check` 通过。

### 交接

- P4 只完成本地 scientific generation 单元；没有 production live 入口，也未跑真机双卡。
- P5 继续实现 generation lifecycle/continuation 与 ledger-derived exploration index；P6 才做资源和成本 pilot。
- 未经用户验收不得进入 P5、同步服务器或启动实验。

### 用户审计后的修正

- 纠正旧 author-only 探针与生产 exclusion payload 不同却共用 `86.7%` 结论的问题；旧值降为上界。
- 按用户授权仅复查 R01/R04/R08/R11/R14 五份生产同形状请求，逐份人工核对；没有运行 GPU、train、eval 或 scientific ledger。
- 结果为 1 份完整且 distinct 候选、3 份 `finish_reason=length`、1 份最终 content 为空。R01 又在服务器 `3af99dd` 的 Isaac/Torch CPU 环境通过全部 16 个 RewardTerm smoke；失败集中在 `deepseek-v4-flash + reasoning_effort=max + O=16,384` 的成稿边界，不是已观察到的 exclusion 改名穿透。
- 后续 provider 默认改为 `deepseek-v4-flash`、`reasoning_effort=max`，thinking 请求省略 temperature；默认输出 cap 最小幅度提升为 32,768。32K 仍未实测闭合，不授权 GPU。
- 主计划中所有依赖 `O` 的 response-unit 上限同步重算：`5O=163,840`、`10O=327,680`、`30O=983,040`、`150O=4,915,200`，避免代码默认与成本表再次分叉。
- generation commit 的完整 receipt replay 移到写锁外；锁内只做 snapshot compare-and-swap 与 exactly-once append。
- width=1 provider/static failure writer 统一改用幂等 append；selector 未知/重复 trial 改为可诊断 `ValueError`。
- 端到端 generation 测试中的 13 个场景改为具名 `subTest`，并为 selected-state 和锁外 replay 增加独立单元测试。
- `acceptance_replay -> agent_core.generation_record` 的 lazy import 记为 P5 分层清理，不在 P4 为低风险依赖做大范围搬迁。
- 最终定向集合 84 项通过；provider receipt 边界另增 1 项；全量 `Ran 1221 tests`，`OK (skipped=21)`；编译与 `git diff --check` 通过。
- 服务器 `/tmp/production_exclusion_recheck_20260817` 已清理并确认不存在；未启动 GPU 作业。
