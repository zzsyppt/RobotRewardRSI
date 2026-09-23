# P5 开发与验收记录

2026-09-10 至 09-11，基线 `de9a9776d4e3c8cd75079a9a54a4b692b8049c92`。最终结论见 `ACCEPTANCE_ZH.md`；本文件记录有后续维护价值的取舍。

## 收敛方式

先处理 P4 四项复审，再处理连续修复、同源反馈、容量与物理解释、canonical 准入、runtime 差分及训练后的任务门。共享层只增加两个 task hook 和诊断身份字段；没有新建任务编排框架、机器人特例、自然语言机制判定器或另一份源码分析器。

从本地 Codex 借鉴的是所属任务终止/清理、持久化、再报告关闭的顺序。新会话可在第 1–14 个完整代边界暂停；已有段数和成本上限继续有效，没有加入任意指令点抢占或后台恢复服务。

## 版本与保留消费者

| 新写入 | 旧版本的实际消费者 / 保留边界 |
|---|---|
| prompt payload v5，紧凑 JSON，无 `public_context` 别名 | 已封存 prompt 的 checksum 与 replay 仍依赖 v4 缩进字节；`sealed_prompt_json` 只用于历史字节验证；当前 writer 与各层预算使用 compact，容量回执显式标明格式。 |
| authoring continuation v2 | 删除无活动消费者的 v1→v2 恢复重解释。已封存历史 artifact 仍按原始字节读取；不把它当作可继续执行的 authoring 状态。 |
| source catalog / lineage v4 | v3 历史 prompt 的验证仍保留其逻辑 ID；新 writer 分离当前 repair 与历史未接受源码。 |
| generation authoring policy v2 / generation ledger v3 | P4 及原实验的 v1/v2 记录需要旧双 lane 拓扑解码和 replay；新入口保留各 lane 的额度，A 失败时 B 不再启动；旧记录不改写。 |
| provider response receipt v4（有模型容量声明） | 原 provider 回执、离线 fixture 和未声明容量的旧配置仍能读取；新共享正式准备入口要求显式容量声明。 |
| Stair operation-effect v1，带诊断 context hash | 没有该绑定的旧训练摘要只能按原合同读取，不能冒充新“全 transition 同状态差分”证据。 |

旧 exploration card 的 exact hash 算法保持原样，已有 ledger/index 回放需要它。新 authoring 准入从同一 canonical delta 派生 exact 实现集合，未改变旧 card/hash。保留的是确有历史消费者的 reader/算法，不新增第二套科学接受语义。

## 验收发现与修复

- Ruff 清理时发现 `source_checkout` 与地形模块的转发导出仍有实际消费者，改成显式导出；不根据 F401 单独判定可删除。实际 Isaac 首次启动暴露的缺项已经真实重跑关闭。
- 真实 provider 第一次推理耗尽、第二次多机制被拒、第三次有效，保留完整回执。没有临时抬高尝试数或输出上限刷过门。
- 最终 admission 另以实际 provider 的候选封存执行一次，防止仅证明 authoring 的预验证，而遗漏训练授权入口。
- 最早完整回归暴露旧测试仍索引废弃别名、历史 source ID、缩进 JSON 和旧 feedback 字段，已对齐当前合同。修复帧仍断言修复源/原因同一、不可见私有路径、不额外消费正式轮次。
- Stair 模拟训练器现在从真实 TrainJobSpec sidecar 读取并核对诊断 hash，正向作用数据明确为模拟。真实差分由 Go2/G1 Isaac 作业测量，职责分开。
- 记录脚本最初在顶层直接加载 unittest，干扰 spawn；作废该次结果，外层改为仅启动标准 unittest 子进程和记录日志/退出码。该脚本只在本验收目录，不进入产品依赖。

保留失败记录以解释验收过程；没有触碰原完整 15 代 checkpoint，也没有扩展到 P6/P7 的训练搜索。

## 原批次验收（复审前）

本地完整 1606 项零失败、19 项平台/安装条件跳过；服务器完整 1606 项零失败、2 项显式 opt-in 跳过，这两项 Isaac 动态测试已另行启用并全部通过。一次本地运行遭会话中断，保留原记录后重跑了完整 discovery；最终回归中相关三项同样通过。最后只补充 runbook 的段数/成本边界和验收记录，没有在测试后继续修改产品代码。

## 2026-09-11 复审取舍

- H1 的最终 trial 回执是恢复权威：已完成训练也不能让终态重新运行 task gate。没有 public-dev 的终态直接恢复原判决，已有评测阶段沿用原恢复路径。
- 同一 selector 找到最近有训练证据的未接受 trial；源码、runtime 与详细反馈共用身份。紧凑 trial 表保留两条 lane 的机制/结果；先压缩重复详情再丢代，14 KiB 上限不扩大。
- active unified 的 prompt feedback 从原始 public-dev 派生，不再写入新的科学结果。pre-unified reader/writer 仍有明确历史消费者，未借此次修改迁移整套老协议。
- 从本地 Codex `codex-rs/core/src/context_manager/history.rs` 借鉴“持久化历史与模型可见窗口分开”，从 `compact_token_budget.rs` 借鉴确定性压缩可以复用既有生命周期。没有增加 LLM 摘要器、记忆数据库或新的编排框架。
- 真实候选复核发现 Python 3.11/3.12 的 AST dump 字段差异会改变旧结构 hash（例如空 type_params）。这在进场分析器中已存在；本次在同一 Python 对照进场版本，37 份分析完全相同。保留旧 ledger 的封存身份，未迁写为新 hash，也未加入跨后端机制去重。
- 最终自审补齐了容量回执的格式版本：当前 task receipt 标识 compact，capacity receipt v3 按 compact 计量；旧缩进 task receipt 与 capacity v2 仅在 reader 中按旧格式验证。P4 的 15 份原始 prompt/回执已验证，避免“全局改计量却遗留旧 reader”的同类问题。
- 完整 discovery 暴露三处旧断言后，保留原失败日志，改为验证“未训练失败留在错误历史、训练后拒绝源码跨接受轮保留”以及实际发送字节的裁剪边界。未降低断言或退回旧选择器；两端重新执行完整 1610 项均通过。
- 原真实 provider 产出的单机制候选在本轮最终代码下再次通过作者检查、封存资格化和执行授权，原源码校验值保持不变；只复验已有候选，不发新生成请求。

本轮最终结果以 AUDIT_FOLLOWUP_ZH.md 与 ACCEPTANCE_ZH.md 为准；原批次通过记录不替代本轮完整回归。

第一次复审终验：本地约 917 s、目标约 2724 s，均零失败/错误；目标两项 opt-in 动态检查另行通过。516 个交付源文件与测试逐字节一致，完整实验 932 文件及 71 checkpoint 未变。当时的进程及 lease 已闭合；该批通过不替代后续问题的复验。

## 2026-09-11 历史回放与内容取舍复审

冻结 P4 正式 CLI 暴露 authoring policy 演进导致的只读回放回归，改为共享 reader 沿用冻结 authoring 身份，实际续跑保持严格；删除无消费者的 P6-B 私有兼容分支及其专属测试，有效约束迁入共享契约和真实 preparation/replay 集成验证。

历史 prompt 的详细表只给当前最优与最新已测候选，合并公共表头和评测身份；旧条目保留关键成绩与失败。真实 G3-A 进一步发现没有 selector 时 parent dev 未投影，由共享层读取已绑定回执、任务层解释摘要，在模型视图中恢复已测分差。trial reason 随紧凑记忆保留，不包含原始私有错误。

本轮实施、内容字节对照、自审和最终验收见 [REPLAY_CONTEXT_FOLLOWUP_ZH.md](REPLAY_CONTEXT_FOLLOWUP_ZH.md)。没有新增 provider 请求、RL 训练或科学规则修订，P6/P7 保持未完成。

本轮终验完成：本地与目标各一次完整 1611 项，零失败/错误，分别跳过 19/2 项；真实 P4 正式 CLI 报告及 251 文件不变，完整 c6 的 932 文件/71 checkpoint 与 c8 的原始身份文件不变。515 文件源码清单两端一致，测试进程及 lease 已收尾。P5 恢复工程 GO，最终交付见 `replay_deployment_receipt.json`。
