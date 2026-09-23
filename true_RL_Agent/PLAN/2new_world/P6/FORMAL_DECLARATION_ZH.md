# P6 正式协议与资源声明

2026-09-12 北京时间 00:32:15 签发，对应 UTC `2026-09-11T16:32:15.626103+00:00`。权威记录为 [formal_declaration_01.json](formal_declaration_01.json)。签发时尚未消费本轮 holdout。

`active` 表示协议与源码身份已冻结；签发时两格 reference 的物理资格尚待验证；当前完整测量已为两格 NO-GO，结果见 [验收](ACCEPTANCE_ZH.md)。此声明不构成 reference 通过、正式训练已执行或 P7 搜索 GO 的证据。

| 身份 | SHA-256 |
|---|---|
| 声明 | `9c0e7e04f6eba63d2759bb18ef4c54b9fa39c3ff3eb139c02350c35d107e9872` |
| Push 协议 | `a51f190bab162281d6d014285dde612f73763fb20457bc690a5f7c3a5aaeb778` |
| 项目源码快照 | `d54c6eec4aeda12a76402e97cf5a316b51eef154d9780ed348c92d122887bdcc` |

## 固定样本与资格门

仅覆盖 `push.g1_37dof.recovery_v2` 和 `push.go2_12dof.recovery_v2`。每格一次 random-init reference 训练，train seed=42，4096 env × 24 steps/iteration × 500 iterations = **49,152,000 env-steps**；两格合计98,304,000。不因性能增加 iterations、改初始化或重训。

评测 base seed=42；dev/accept/sealed 为3/3/7 seeds，每 seed 64 episodes，即每格192/192/448，两格首次三卷共1664 episodes。正式分母包含施扰前早跌的计划受扰 episode；no-push 对照只作 diagnostic。

Reference 必须满足 dev 分数位于 `[0.2, 0.9]`、accept 与 sealed 各不低于0.2、accept→sealed下降不超过0.1，并通过 accept evidence、budget accounting、recovery evidence、sealed validation 四门。候选最小改进参数固定0.05，继续服从既有噪声感知接受规则；P6 不启动候选搜索。

## 执行上限

- 最多并用2张 GPU，使用现有 lease 和作业归属。每格完整资格流程由现有 owned-process wrapper 限制在 **14400秒（4小时）**，另有60秒终止宽限；两格 GPU allocation 合计上限8小时。
- 单个 train/eval 作业的 runner timeout 固定 **36000秒**。外层4小时总上限更紧，单作业设置不能延长整格预算。
- 基础设施重试 **0次**，provider 请求 **0次**；执行前可用磁盘至少40 GiB，两格资格输出总量上限20 GiB。
- 视频与矩阵 diagnostic 另计最多2 GPU小时。视频仅重放同一 reference checkpoint 的 public-dev；每格最多2轮、每轮192 episodes、最多6段录制。成功/失败实例由任务原始分类确认，不作为新的资格样本。

## 停止与 NO-GO

技术失败立即停止该格，保留原始日志、receipt、失败原因及已花成本，不自动重试。任务资格失败如实给出 **NO-GO**，不合成 Genesis、不补训练刷过门；超时和资源上限也不能通过重新启动绕开。

消费 accept/sealed 后禁止回调协议参数、reference 奖励、阈值或资格门；已经观察的 holdout 不能再用于调参并重报本次资格。若后续确需修订，须另行声明新身份和新实验，原失败证据保留。

开发接线、短生产流水线和本地回归的当前证据见 [DEVELOPMENT_LOG_ZH.md](DEVELOPMENT_LOG_ZH.md)。完整资格、独立 load-only 重载及原生视频核验仍须以各自真实回执关闭。
