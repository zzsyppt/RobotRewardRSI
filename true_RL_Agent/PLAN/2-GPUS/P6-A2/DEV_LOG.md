# P6-A2 author-only 探针开发日志

> 2026-08-17 纠偏：本日志记录的是旧的高信息 exclusion 探针，不是 P4 最终生产 payload。`86.7%` 只能作为上界；其中包含最终合同禁止向 B 暴露的 A candidate/round ID 和逐轮手写语义。生产同形状复查及新结论见 `PLAN/2-GPUS/P4/PRODUCTION_EXCLUSION_RECHECK_REPORT_ZH.md`。

## 2026-08-17

### 用户授权范围

- 运行约 15 次 provider 调用，回答 A 的机制被排除后 B 是否还能提出真正不同的有效机制；
- 禁止 GPU；
- 每个样本由主执行者亲自阅读和确认，脚本只作传输、物化和现有合同投影参考。

### 实际执行

- 从正式 session `stair-reference-formal-20260813T125345Z` 只读取得 15 份真实 prompt、候选、own anchor 和 ledger；
- 对每轮 A 使用现有 verifier-owned operation delta 生成 public-safe exclusion；
- 使用 `deepseek-v4-pro / temperature=0.2 / adaptive reasoning / max_output=16384`；
- R01–R15 各发出一次实际 HTTP 请求，严格串行，不重试；
- 每个可解析 B 均人工阅读 A exclusion、B changed function 与组件结构，再在服务器执行 CPU-only static/component smoke；
- 未运行 train/eval/sealed，未写 primary ledger。

### 结果

- 有效且实质不同：13
- provider 传输失败：1（R02）
- output length 截断：1（R05）
- 等价机制漏放：0
- paired-ready：13/15 = 86.7%
- 预注册工程门：PASS

### 边界说明

当前树没有 P4 production `single_component_semantic_exclusion` 与 `GenerationCoordinator`。本次是用户授权的一次性 author-only 可行性探针，不得冒充正式 P6-A2 生产接线验收，也不授权 P6-B GPU pilot。

### 证据

- 完整输出：`D:\humanoid_agent\true_RL_Agent\outputs\author_only_paired_ready_probe_20260817`
- 结构化结果：`RESULT.json`
- 逐轮人工表：`MANUAL_REVIEW_ZH.md`
- 正式报告：`AUTHOR_ONLY_PAIRED_READY_PROBE_REPORT_ZH.md`
