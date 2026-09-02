# P6-D 终局 follow-up 逐项审计与修复记录

## 1. 范围与判断准则

本轮只处理 `P6D_RUNTIME_OBSERVATIONS_p6d-c582eca-20260823T194930Z_ZH.md`
中的 N1–N23。冻结实验提交仍为
`c582eca04dbac7d5ee5e2df0f56c646500ead55f`；正式 output、ledger、receipt、checkpoint
和 sealed 原始证据均不改写。本轮不启动 provider、Isaac、GPU 或任何新真机实验。
冻结 report SHA256 为
`6b7e324c09b58c62a1e450308080be5b9cef533b5737a74933ef389f2ebe0897`；15 代公共分类为
`accepted=4 / formal_rejected=7 / no_selection=4`，因此 N17 的正确完成科学代数是 15。

一项问题只有同时满足以下条件才进入产品修改：

1. 当前代码或冻结公共证据能够复现，而非只靠推测；
2. 会影响科学机会、事实准确性、GPU 前拦截或终局可独立复核；
3. 根修可以复用现有边界，新增维护面明显小于预期收益；
4. Stair 特殊语义留在 Stair，公共层只承载任务无关的最小扩展；
5. 不要求改写既有不可变证据，不降低旧任务插件默认行为。

## 2. 逐项结论

| 项 | 真实性与证据 | 决定 | 本轮处理 |
| --- | --- | --- | --- |
| N1 preflight 计划文字漂移 | **真实**。产品 preflight 不执行 terminal Reservation/Result 或 readiness subject 往返。 | 需要纠正文档，不能伪装成产品缺陷。 | 已按真实执行边界改计划 §8.1；terminal/readiness 仍由各自测试覆盖。 |
| N2 launcher timeout | **真实但仅属操作参数**。`43200s` 没有成为冻结 schema。 | 不值得新增合同。 | 在本文明确记录为一次运行参数；后续 runbook 可按预计墙钟调整。 |
| N3 Iray/libGLU 告警 | **证据不足**。没有对应 job failure、证据缺失或清理失败。 | 不改。 | 保留为运维观察，不安装依赖、不新增 readiness 门。 |
| N4 formal-rejected cooldown 优先级 | 有 3 个公开负样本；截至已有独立 exact-key 复核的 Gen10 未见 exact-equivalent 重训，全程也未观测到实际 GPU 损失。 | 证据不足，不改共享探索语义。 | 保留现状；未来只有出现精确重复并造成实际浪费才重审。 |
| N5 lane-A 终败导致 lane-B 不启动 | **设计风险真实**，但 P6-D 15 代没有 paired-authoring-incomplete；修改会改变 width/预算权限。 | 本轮不改。 | 明确延期，不把失败率模型当成已发生缺陷。 |
| N6 27 行私有墓碑 | **真实且零调用**。 | 删除收益明确。 | 删除 readiness 常量、launcher helper 和 CLI 私有 alias，共 27 个非空代码行（Git diff 为 31 行，含空白）；不留兼容壳。 |
| N7 再删约 1,300 行脚手架 | 旧估算已过期；当前调用图显示相关路径仍在主线。 | 不按行数驱动删除。 | 不处理。 |
| N8 task prompt 余量紧 | **真实近界**，但冻结运行无关键 section omission。 | 不扩 cap、不加新压缩层。 | 与 N12/N15 文案压缩及 N16 重排一起解决；容量/锚点测试保持原上限。 |
| N9 高层假设家族重复 | 至少 6/15 代有公共人工复核证据，但没有稳定 classifier。 | 只值得一条软提示。 | Stair guidance 增加“优先近期未试 causal family”；明确永不作为 gate。 |
| N10 重型 retry 余量紧 | 合同确实只剩一组，但 P6-D `retry=0` 且 fail-closed。 | 不改 cap。 | 只记录下一合同应按真实 retry 率重新预算。 |
| N11 rationale 与程序语义不一致 | **真实**，且错误 history 叙述跨代传播；历史 bytes 不可改。 | 不重写旧证据，不引入自动因果归因。 | 用 N12 防止新误读，用 N13 让后代可定位 exact candidate source identity。 |
| N12 输入轴/位置语义不完整 | **真实且重复发生**：contact history 方向、环境轴、foot 轴、空间正向和 goal-distance 符号均有样本。 | 需要 task-owned 根修。 | reward-input prompt v12 前置精简的高风险轴语义；不进入共享 verifier。 |
| N13 latest pair 缺 candidate identity | **真实**；loser 有结果但作者无法定位其公开 source identity。 | 需要小型投影修复。 | 只增加经验证的 `candidate_id` 与 `program_source_checksum`；feedback v10、minimum-core v3，不增加 raw source。 |
| N14 memory 先删后摘要 | **真实**；早期 accepted 记录在仍可容纳摘要时整条消失。 | 需要修压缩顺序。 | 固定 cap 内先摘要最老非保护记录，再删除；最新/最新 accepted 仍受保护，幂等。 |
| N15 明文字段语义被长合同淹没 | **真实**，但不支持 AST/taint 或新语义执行器。 | 与 N12 合并。 | 高风险语义前置，并压缩重复 guidance；总容量不扩张。 |
| N16 lane-specific package 过早打断前缀 | **真实字节结构事实**；provider cache 命中率仍非产品可控。 | 一次纯顺序调整值得做。 | `required_candidate_package` 后移至共同 task/guidance 后；字段和 schema 不变，并验证双 lane 前缀稳定。 |
| N17 report 少计 no-selection | **确定 bug**。4 个完整双 lane 科学代被漏计。 | 必修。 | formal report v6 将 `accepted + formal_rejected + no_selection` 计为完成科学代。 |
| N18 `max` 可能挤占 final | 冻结终态 41 次 attempt 中有 `38 stop / 3 length`，另有 empty-final；但 `high` 的质量与成功率收益没有闭合证据，一次 replay 被人工停止，未形成两样本报告。 | **证据不足，不切 high**。 | 继续 `max`；仅收成一个 Stair-owned 常量，避免多源漂移。 |
| N19 no-selection 标成 sibling_not_selected | **确定事实错误**。没有 winner 时不存在被 sibling 击败。 | 必修。 | shared generation 透传 adapter 的 selection reason；有 winner 的 loser 仍为 `sibling_not_selected`。 |
| N20 terminal audit 未重做 final private lineage | **真实**。创建期闭合，但事后 audit 未把 reservation anchor 与 replay head/私有 checkpoint 再绑定。 | 必修。 | Stair terminal audit 复用现有 ledger replay、final-pair builder、train receipt loader 和 checkpoint checksum；不建第二审计框架。 |
| N21 parent term alias 重复加权 | **确定且已有一整 lane GPU 损失**。 | 必修，且应在 GPU 前。 | 共享层只加默认 no-op authoring-delta hook；Stair 对 canonical pure-add 使用现有 structure signature 拒绝 parent exact alias。 |
| N22 跨 environment 运算可 shape-correct | **真实近失和纯内存复现**。现有 shape/finite/determinism 不足。 | 必修，保持最小变形检查。 | Stair fixture v2 使用非均匀 row marker；每 term 仅一次 reverse permutation 与一次 single-row locality，并沿用 qualification worker/same-round repair。 |
| N23 零 activation 候选可抢 formal 名额 | **决定性 Gen15 证据**。formal gate 保证不误晋级，但另一有效 lane 丧失 formal 机会。 | 必修。 | Stair 在 selector 前投影 task-owned runtime-effect eligibility；共享 generation 只传权威 `TrainJobResult`、train receipt checksum 与 mechanism operation delta，formal gate 保留为第二道复核。 |

## 3. 实现边界与复杂度选择

- 公共接口有三处最小变化：默认 no-op 的 authoring-delta hook；phased screen 接收权威
  `TrainJobResult`、train receipt checksum 与 mechanism operation delta；selection result 暴露任务给出的
  reason。另有两处不新增 schema 的共享行为收敛：memory 先摘要再删除，以及 prompt 公共 section 顺序
  调整。以上共享代码均不认识 Stair 字段、RewardTerm、环境轴或 15 代语义。
- Stair-owned 改动复用既有 component semantics、qualification worker、public projection、selector、
  ledger replay 和 final-pair builder；没有新增 daemon、receipt 家族、readiness 版本、AST blacklist、
  缓存系统或独立审计框架。
- screen v2→v3、selection contract/result v4→v5、formal report v5→v6、admission v3→v4、fixture
  v1→v2、reward-input v11→v12、public-feedback v9→v10、minimum-core v2→v3；新增的
  runtime-effect/environment-axis evidence 均为 v1。开发期旧 selection v3/v4 parser 被删除，不继续
  累积兼容分支；冻结 report v5/selection v4 只在提交 `c582eca` 上独立复核，新主线不迁移也不重写。
- prompt 不扩 32,768-byte task cap。新增语义通过压缩旧重复文案吸收，容量压力用现有 minimum-core
  规则处理。

## 4. 相邻同类审计

本轮沿修改调用链继续检查后，只发现并处理两项有直接证据的相邻问题：

1. Stair mutable candidate 在静态验证后有直接 `read_text/read_bytes` 路径。现统一复用既有 candidate
   package preflight 与 preflighted bounded open，再按 program identity 校验 candidate ID、size、checksum；
   authoring-delta 复读阶段的超限或竞态替换在 seal/smoke/GPU 前进入原 repair 路径；初次 verify
   读取异常保持 fail-closed，不额外增加推测性 failure 映射。
2. 初版 bool row marker 在 5 行 reverse 时具有对称性，可能漏掉只依赖 bool 字段的跨环境耦合。fixture
   改为非回文 modulo-3 marker；没有增加第二组排列或测试矩阵。

没有证据支持继续扩大清理或保护面：readiness 活路径、lane 全有全无、cooldown、retry cap、provider
reasoning policy 和 Iray 依赖均保持不变。

## 5. 验证与停止门

工程验证分三层：

1. prompt/memory/capacity/Push 隔离与旧 task plugin 回归；
2. N17–N23 的 focused unit/integration/production-shaped fake 测试；
3. Windows 全量与 Linux 全量，其中 Linux 必须真实执行两条 Torch CPU 环境轴检查。

验证对象与产品实施提交为
`ab84aa3350700a2f7d46231afefaec4425ffa2ef`，已推送至
`origin/true-rl-agent-refactor`，并在服务器
`/data/yuyangliu/AGENT` 以 clean worktree、同一 HEAD 验证：

- Windows 全量：`1483 tests / 0 failures / 0 errors / 29 skipped`（`844.185s`）；
- Linux 全量：`1483 tests / 0 failures / 0 errors / 2 skipped`（`1572.430s`）；
- Linux Torch CPU 环境轴两条聚焦检查：`2/2` 通过（`2.716s`），且两条也包含在上述全量中；
- 首次 Linux 全量曾人为设置 `CUDA_VISIBLE_DEVICES=''`，产品按合同拒绝父级 GPU remap，得到
  `3 failures / 5 errors`。该运行不是有效产品门禁；移除 remap 后所有对应错误消失，因此没有放宽合同或修改代码。

本文之后的纯文档收口提交不改变上述已验证的产品与测试 bytes，其 SHA 以 Git history 为准。至此工程门
完成并立即停止；下一轮 provider、Isaac、GPU 或正式 P6-D 后续实验必须等待用户静态审计与新的明确批准。
