# 55 项问题的 P0 初始处置

日期：2026-09-05。背景和原始证据只读 [PHASE6_UNRESOLVED_FOLLOWUP_ZH.md](../../Variant_construction/15_rounds_formal/PHASE6_UNRESOLVED_FOLLOWUP_ZH.md)。本表与原文编号集合一一对应，不纳入已关闭的 F1–F6、F8、F9、F12、F47。

原清单与 P0 起点同在 `0773a9a`，P0 前没有更晚的产品修复提交。未在 P0 改动的项沿用原审计为开发输入，在责任 Phase 重新做行为验收；“承接”不表示已经修复。涉及新协议的工作不能回写旧会话。

本表保留 P0 时点的初始分工；后续实现与当前状态见 [P1 验收](../P1/ACCEPTANCE_ZH.md)和 [P2 验收](../P2/ACCEPTANCE_ZH.md)，不能把本表的“待修”继续当作实时状态。F39 原描述曾误读为力作用点问题，现按原始问题更正。

| 编号 | P0 处置与边界 | 后续责任 |
|---|---|---|
| F7 | 保留至尺度链设计；有唯一行为消费者则接入，否则从新身份删除 | P3 |
| F10 | 暂保留公开导出；退役前查实际外部/文档消费者，不新建兼容包装 | P8 |
| F11 | 派生浮点身份仍按原版本读取；跨解释器规范化未实施 | P1 |
| F13 | scientific identity 与 execution placement 尚未拆分 | P1 |
| F14 | cost capability 分派留在现有实现，两任务共同需求形成后收敛 | P4 |
| F15 | Stair session 仍是当前装配；本阶段不抽取共享 session | P4 |
| F16 | 保留 completed-session reader；Phase 6 验收器/测试仍消费 | P4、P8 |
| F17 | 保留旧 Plan 字段与 hash；新版本再改字段语义 | P4 |
| F18 | 跨 boot resume 尚未闭合；同 boot 历史证据保留 | P1 |
| F19 | final metadata 副本尚未收敛 | P1 |
| F20 | authoring failure 的有界跨代摘要待补 | P5 |
| F21 | 诊断判定语义由 Stair task-owned prompt 补全 | P3、P5 |
| F22 | prompt 中重复 cell success 待同源投影收敛 | P5 |
| F23 | no-selection 分层失败证据待补，不能伪造已评测机制 | P5 |
| F24 | transport failure 与既有 smoke failure 的原因链待保留 | P5 |
| F25 | 双 lane authoring 拓扑与共享预算边界待协议化 | P4、P5 |
| F26 | 外部 P5 reference 实物及 SHA256 已核实并保护；尚未复制为新 session 自有 anchor | P1 |
| F27 | 不更改旧完整评测拓扑；新成本合同再定义合法早拒 | P4 |
| F28 | smoke 的零贡献/未覆盖 fixture 反例仍待修复 | P5 |
| F29 | README/runbook 指向正式双 lane launcher；概览与 Stair 指南已改为 active unified V2 的 task-owned outcomes、suite 和 acceptance，移除误标 active 的 v6/35-cell 规则；旧 wrapper 仍有 compatibility 消费 | P8 完成入口退役 |
| F30 | **retired**：删除 replay-only loop/session；有效 sanitizer/activation 测试迁移，现有 live/Genesis/replay 检查继续覆盖正式边界 | P0 验收 |
| F31 | 删除 7 个无当前消费者的一次性脚本；保留 manifest validator、pilot 依赖链、V1 predecessor 和 transition normalizer | P8 抽出最小 validator 后再删生成栈 |
| F32 | provider legacy alias 尚未移除；本次边界测试改为检查当前 live/generation 路径不读该 alias | P5、P8 |
| F33 | **retired**：删除一次性 P6-B historical-loader 审计脚本；保留实际历史报告和当前正式 reader | P0 验收 |
| F34 | Gym ID 身份尚未改动 | P2 |
| F35 | robot runtime source 与 task reference 的 source closure 待收敛 | P2 |
| F36 | 不清空共享 lease 根目录；沿用现有 owner 清理边界，残留元数据归最终清理 | P8 |
| F37 | PPO/Policy 重复配置由运行时迁移统一来源 | P2 |
| F38 | active V2 的重复任务不变量由 task-owned 协议收回 | P3、P4 |
| F39 | 验证 `support_effector.targets ⊆ support_body_names`；P2 在机器人层完成包含约束，不额外引入力作用点等同规则 | P2 |
| F40 | 双 lane Kit 日志路径在统一运行时阶段收敛 | P2 |
| F41 | forbidden-contact 身体/频率诊断由同一 episode evidence 派生 | P3、P5 |
| F42 | source recoverability 元数据与实际裁剪帧尚未闭合 | P5 |
| F43 | rationale 与权威 delta 的矛盾、latest-failed 来源待修 | P5 |
| F44 | **部分完成**：1321 个历史 model 中已实际删除 892 个、6,632,805,920 B；429 个保留，其中 9 个旧正式失败模型从候选转为失败证据保留；删除后 3214 个受保护文件哈希不变 | P0 首批清理完成；P8 收尾 |
| F45 | **fixed**：P1 三份文档的 Go2 × Push 责任统一为 Extension A，原证据数值不改 | P0 验收 |
| F46 | sibling/no-selection 的有界机制历史待保留 | P5 |
| F48 | eval_incomplete 恢复时 train 投影丢失待修 | P1 |
| F49 | authoring、runtime failure、public-dev infrastructure 原因分类待修 | P5 |
| F50 | child admission 尚未验证父会话冻结源码字节 | P1 |
| F51 | opaque repair placeholder 的 source follow-up 入口待修 | P5 |
| F52 | 非父代重复机制须接入现有 canonical delta 与有界历史 | P5 |
| F53 | 正式 c6ab8ba 与旧 d444f5b 失败 TRAIN 模型保留，冻结 cost/resource accounting 与失败证据不改写；本次未引入自动 retention 规则 | P8 |
| F54 | morphology/nominal scale 由 RobotProfile 单源投影，prompt/smoke 待消费 | P3、P5 |
| F55 | 全程 reference-head 的正式终态测试仍待补；P0 不删现有 candidate-head 测试 | P1 |
| F56 | planned URI 与实际 private artifacts 仍未区分；P0 不将原本未产出的文件记为删除造成的悬空 | P1 |
| F57 | minimum-core 成功条件字段待补全 | P5 |
| F58 | 权威 token-window/tokenizer 能力待绑定；不把 bytes/4 当成权威容量 | P5 |
| F59 | 控制 dt 与 weighted rate / step reward 单位待同源解释 | P2、P5 |
| F60 | 同名 modify 的 parent-relative delta activation 待补；非零整项不等于差分生效 | P5 |
| F61 | 删除无消费者的 legacy 导出，只保留有明确历史审计用途的 R1 两组视频及其他既有迁移材料；当前指南明确 legacy CLI 不能重放 active Go2-V2；原生录像/训练对齐尚未完成 | P2 |
| F62 | 纯改名/纯数值修改的 canonical classification 待修 | P5 |
| F63 | clearance 的观察性含义与成功/失败条件命中率待 task-owned 投影 | P3、P5 |
| F64 | 新实验 final seed 留出与未见题池必须重新冻结；旧 final 保持诊断身份 | P3、P7 |
| F65 | 终态重复 replay 与报告性能待收敛；P0 保留现存独立终态 replay 证据 | P4 |

`retired`/`fixed` 仅指表内明确列出的改动。整体 P0 的 GO/NO-GO、测试和实际删除量以 [验收记录](ACCEPTANCE_ZH.md) 为准。F29/F31/F44 的部分进展不能写成整项关闭。
