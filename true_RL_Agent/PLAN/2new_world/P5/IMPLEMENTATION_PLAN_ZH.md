# P5 实施计划：authoring、反馈选择与机制准入

2026-09-10。进场提交：`de9a9776d4e3c8cd75079a9a54a4b692b8049c92`，进场工作树干净。本文落实主计划 P5 及进场四项复审，完成状态由验收记录与实际回执判定。

## 目标与边界

LLM 在任务 × 机器人 cell 的冻结研究权限内提交可解释的程序改动。共享层负责 authoring 生命周期、证据身份、容量、双 lane 和成本；Stair/Push 负责物理信号、反馈含义与 runtime 作用解释；RobotProfile 提供形态事实。保留一套现有 verifier、canonical delta 分析器和 runtime evaluator，不增加平行算法。

本阶段修复消息与证据链、机制改动权限及低成本止损，不以更改评测题池或接受阈值制造改进。原完整 15 代 checkpoint 和权威文件继续保护。P4 的固定 runner 通过不代替 P5 的真实 provider probe；P5 的 runtime 非零也不代表性能收益。

## 0. 先闭合四项复审

1. 用 Ruff `--select F` 取得全仓基线，清理确定无消费者的导入和局部量。历史转发模块确有消费者的导出要显式声明，不能由自动修复删除。将同一 lint 命令纳入日常开发验收。
2. 两处 `_provider_verifier_rules` 收到现有 prompt 证据读取所有者，当前执行和历史重放共用一个公开函数。
3. 删除 `prospective_terminal` 的假完整结构。预关闭检查只检查实际 ledger、高度、成本及资源关闭；真正终态仍在 lifecycle 持久化后由唯一 finalizer 构造。
4. 去掉 `(3,6,9,12)` 固定暂停点。新会话允许在任意非终末的完整 generation 关闭后暂停，沿用既有段数、成本和恢复上限；恢复旧会话读取其已冻结暂停边界，不回写旧合同。保留 P1 的真实中断清理与崩溃恢复，不增加实时抢占服务。

参考本地 Codex `core/src/tasks/mod.rs` 的具体顺序：通知中断、等待/清理所属任务、持久化中断记录，再发出完成状态。只借鉴生命周期所有权与持久化顺序，不移植 Tokio、事件总线或通用任务框架。

验收：lint；共享 CLI 与冻结恢复；关闭前预算/资源检查；真实暂停、恢复和终态回放；旧 v6 成本/closure 固定样本字节不变。

## 1. 先建立问题—来源—所有者清单

主问题背景继续引用 `../P0/ISSUE_DISPOSITION_ZH.md` 和 `PLAN/Variant_construction/15_rounds_formal/PHASE6_UNRESOLVED_FOLLOWUP_ZH.md`，不复制全部正文。读取正式 Gen1/5/6/7/10/11/13/14/15 的必要 prompt、authoring、公开评测及 runtime 片段；以 session/trial/program/source 身份区分，私有 accept/sealed/final 不进入作者反馈。

| 工作组 | 承接原编号 | 当前落点 |
|---|---|---|
| 修复连续性、双 lane 不完整记录 | F20/F24/F25/F46/F49 | `authoring_loop`、generation trial/record、公共历史 |
| 同源源码、反馈、最终可恢复性 | F23/F42/F43/F46/F51 | `PublicContextBuilder`、source catalog、payload、Stair memory/selection |
| 容量、别名和物理语义 | F21/F22/F32/F41/F54/F57/F58/F59/F63 | task prompt、RobotProfile 投影、provider policy/receipt |
| 机制重复、数值改动、smoke/runtime | F28/F52/F54/F60/F62 | 既有 canonical delta、candidate admission、task smoke/runtime evaluator |

已由 P2/P3 修复的部分只复核消费者与现状，不重复开发。跨 Phase 项仅关闭本阶段已证明的部分。

## 2. 修复连续性和同源反馈

- 尚未关闭的 candidate repair 与最近 provider transport failure 分开保存；成功验证才关闭 candidate repair。shape→transport→repair 的真实历史顺序作为回归。
- 双 lane 各保留冻结的尝试额度；A authoring 失败时，B 记 not_started，避免在本代无法训练的条件下继续消耗 provider。保留双候选 ready barrier；不引入单 lane 训练/选择。
- 从已存在的 generation trial 证据生成紧凑机制结果表，覆盖 no-selection、sibling 和不完整代。历史索引优先于重复详细状态；共享层不解释 Stair family。
- source、delta、public-dev 细节和 runtime 摘要使用同一 trial 选择结果。区分本轮 latest-failed 与较早未接受程序，避免 catalog 承诺错误的反馈来源。
- 最终裁剪完成后，重新派生 catalog/lineage/recoverability；不可请求且未内联的源码必须明确不可用。

验收：有限历史 fixture 穿过真实 projection/payload/authoring 链；逐个核对 trial/source checksum；candidate、provider、infrastructure 与科学拒绝不混淆；没有私有分数或未训练性能卡。

## 3. 上下文足量且有界

- 在现有 section 预算内保留权威整数计数、family/stratum、必要 progress、当前源码与未解决 repair；删除逐 cell success 重复表示和 deprecated `public_context` 别名。
- 从同一 task/runtime/profile 数据投影成功条件、坐标与 opaque 轴、接触窗口/onset、标称形态尺度、role 关系与积分单位。禁止把具体 G1/Go2 规则写进 shared prompting。
- provider、当前 section 预算及保存端共用 compact JSON；历史缩进回执由标识明确的 reader 验证。记录实际 tokenizer/context 能力依据、requested output、实际 usage 与 finish reason，不能把 bytes/4 当精确 token 真相。
- 分别记录 HTTP 终态错误、empty-final、reasoning 用尽输出；不猜测未验证的供应商错误码。显式启用的推理耗尽降级在同一 lane 的后续尝试及恢复中持续生效，保留冻结尝试/输出预算。

验收：初代、晚代、修复与 source follow-up 四类帧；容量上限、遗漏计数与可恢复性一致；公开字段 canary；原对象/发送对象内容一致；真实 provider probe 得到完整且可验证候选，准确记录失败与花费。

## 4. 机制准入和低成本止损

- 复用 canonical delta 的结构/数值身份，识别纯改名、numeric-only 与结构改动；依现行 reward-only 研究权限将不合格项送入 repair。
- 在同 parent、seed/comparison domain 下查询已有机制索引，去重历史 exact 程序，包括非负结果；与参数化负向 suppression 分开，不能扩大到未经验证的“概念重复”。
- 同一 task smoke 加入 profile 派生的标称姿态、接触过渡、直立/倒置等有限组合。报告样本未覆盖、观察恒零和 clip 饱和的事实与分母，不宣称数学恒零。
- 同一 runtime recorder 增加非零频率与覆盖分母；modify 比较同 transition 的 candidate-parent 差值，不用新项的绝对非零冒充改动生效。
- train receipt 完整闭合后调用同一 runtime-effect evaluator，明确不合格才在 public-dev 前止损；未测与稀疏覆盖不足保留可解释状态。

验收：改名 exact repeat、numeric-only rename、结构变化、标称姿态反号/饱和、modify 差分恒零；真实目标环境的采样/receipt/早拒链；旧任务主路径不退化。非零证据无 acceptance 权，不生成性能正卡。

## 5. 最终验收和交付

每个工作组先运行对应的行为回归，全部改动完成后冻结最终代码并跑本地与目标完整 discovery，目标环境执行本地受限的 Isaac/进程检查。接口迁移后执行 `ruff check --select F src scripts tests`，不以定制 F821 扫描代替。

最后逐项复审所有 P5 原编号和四项意见：标出 fixed/already_fixed/retired/具体保留消费者，提供对应测试或实际产物。同步核验过的代码和文档，再查原实验文件及 71 个 checkpoint、运行进程/GPU/lease。原失败和修正后的证据分开，不拼接旧成功日志声称最终全量。

验收记录保存在本目录，至少包含实施变化与边界、问题处置、容量/身份核对、provider probe、目标 runtime 证据、最终全量/lint、历史保护及部署回执。存在任何未实现或未取得实际证据的必要项时，P5 保持未完成。
