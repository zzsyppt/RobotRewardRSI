# P5 问题处置

本文件记录实现边界；最终验收状态见 `ACCEPTANCE_ZH.md`。原问题正文继续引用 `../P0/ISSUE_DISPOSITION_ZH.md` 与 `PLAN/Variant_construction/15_rounds_formal/PHASE6_UNRESOLVED_FOLLOWUP_ZH.md`，不把历史建议直接当成当前事实。

## P4 四项复审

| 项目 | 处置 |
|---|---|
| 未用导入 / Ruff F | 删除确认无消费者的导入；保留 `source_checkout` 的历史导出。实际 Isaac 启动发现 `goal_progress_m_by_terrain_type` 也是活导出，已显式保留并重新运行。仅看 F401 不能证明可删除。 |
| 重复 verifier rules | `prompt_artifacts.authoring_prompt_verifier_rules` 是唯一投影；generation 与 record 共用。 |
| 假 prospective terminal | 预校验直接接收实际消费的字段，不再制造零 hash 的完整终态对象。 |
| 固定 3/6/9/12 暂停 | 新合同允许任一完整代边界 1–14；旧会话继续读取自身冻结边界。保留既有段数、成本、进程所有权和恢复规则，不新增抢占服务。 |

参考本地 Codex 的任务取消与完成顺序，具体依据见实施计划；没有移植其事件总线或异步框架。

## P5 处置与所有者

| 原编号 | 当前处理与边界 |
|---|---|
| F20/F24/F49 | 未关闭的候选修复和最近 provider 失败分开持久化；中途 transport/source-request 不清除修复。训练成功后的任务门拒绝记 `post_train_task_gate_failed`，不伪造子进程失败，也不生成评测结果。 |
| F25/F46 | 新 generation ledger v3 保留双 lane 各自尝试额度及双就绪门；A authoring 失败即停止，B 记 not_started。删除没有执行收益的独立 authoring 分支。旧 v1/v2 记录保持原样读取。 |
| F23/F43/F46 | 最新未接受 trial 的源码、delta、public-dev 细节与 runtime 用同一选择结果，包括 no-selection/sibling；先跨过未训练代，选定最近有训练证据的未接受 trial；选定后即便 runtime/源码缺失，也不借另一 trial 的数据充数。rationale 不能覆盖 verifier 派生 delta。 |
| F42/F51 | 当前失败源码与历史未接受源码使用不同逻辑 ID。最终裁剪后重算可恢复性；仅有 checksum/大小匹配的已存 source 才能声明可请求。既有 source-request 次数和字节上限保留。 |
| F22/F32 | 新 prompt v5 去掉 deprecated `public_context` 别名，发送与保存使用同一个 compact JSON 对象。Stair 单份逐 cell 整数成功计数及分母替代重复浮点成功率；family、stratum、必要进度继续保留。旧 prompt 序列化不迁写。 |
| F21/F41/F57/F63 | 保留任务成功条件的完整 minimum-core；沿用 P3 的 first-valid-top-arrival、诊断计数和角色语义。观察到的 clearance 不冒充完成事件或新的接受指标。本阶段不改地形、难度或资格门。 |
| F54/F59 | task adapter 从绑定 profile/scale context 投影标称尺度、坐标、opaque 轴、contact onset/history/time 和 reward rate→step 积分单位。只说明该机器人实际具备的输入字段；没有机器人名字分支进入共享层。 |
| F58 | 新准备流程冻结 provider 模型窗口、输出上限及官方依据；预约输入与请求输出共同校验。实际 token 数由 provider usage 返回。HTTP 400 保留真实状态、终态处理；删除未向供应商验证的 input_context_overflow 码表。推理耗尽和 empty-final 按实际返回分类；同 lane 推理降级持续生效，固定额度内修复。 |
| F28 | 五个有限 smoke 场景覆盖标称支撑、落地、腾空、倾斜和倒置。保留不同环境行的数值差异；返回每场景输出、非零/饱和计数及分母。零值/饱和仅保存在观察记录中，不触发额外 provider 修复；raw 等于 clip 边界不算被截断。 |
| F52/F62 | 共用 canonical delta：纯改名、纯数值 modify/rename、新增数值克隆不作为新结构机制；准入比较将同方向 < 与 <=（> 与 >=）归入阈值参数族，不改冻结 delta/hash。新 authoring 在同 parent、同会话比较域内排除已评测 exact 实现，包含非负结果；未训练条目不占用该排除集。旧探索 card/hash 计算保留，避免破坏回放。 |
| F60 | 任务侧封存实际 parent source/manifest 和 canonical delta，经 TrainJobSpec 的诊断 hash、CLI、训练回执贯通。实际 dispatcher 对同一 transition 比较候选与 parent 的加权 rate 差；共享层仅消费任务门结论。整项非零不能代替改动差分非零。 |

## 证据解释

真实 authoring 探针从冻结的 P3 Go2 public 证据构造当前 prompt，不启动 RL，也不产生接受记录。首轮 32768 个输出 token 全被推理用尽；第二轮按已授权策略关闭推理并取得 JSON，但多机制变更被现有门拒绝；第三轮取得通过验证的单机制候选，但旧实现重新开启了思考。本次改为同 lane 持续降级，并验证中断恢复；原探针只证明当时的 provider/候选链，不冒充新版降级的实测。所有失败仍保留。

四个实际 Isaac 短 PPO 诊断覆盖 Go2/G1 × 零/非零 modify。每个作业有 128 个 PPO 环境步；dispatcher 还包含 1 次激活 preflight，所以差分覆盖为 9 × 16 = 144 个环境 transition。该证据证明执行和止损方向，不证明策略收益或完整训练性能。完整 generation 的“保留 train receipt、跳过评测”由真实编排链回归覆盖，未将短训练包装成正式 generation。

历史摘要脚本读的是受保护 `formal_go2_stair_15round_20260901_c6ab8ba` 的指定代。该会话 Gen1 本身是 formal rejected，不能用它冒充另一轮报告所述的 Gen1 authoring 不完整；连续性回归与 Gen6/13/14 的实际失败类型分别注明来源。

原 capacity_01 只是一组初代/预检组合，不能支持晚代容量结论。本次由真实两场 ledger 和 P4 Genesis 重建；修复帧取实际失败源码与原因，source-followup 是对真实晚代上下文的受控分支，未声称历史上发生过该请求。当前测量见 AUDIT_FOLLOWUP_ZH.md。

本轮 H1/H2、M1–M6 与低优先级清理的逐项验收见 [AUDIT_FOLLOWUP_ZH.md](AUDIT_FOLLOWUP_ZH.md)。

后续正式历史回放、反馈内容取舍、无 selector 的已测 dev 与失败 reason 四项复审见 [REPLAY_CONTEXT_FOLLOWUP_ZH.md](REPLAY_CONTEXT_FOLLOWUP_ZH.md)。前一批回归不替代这四项的验收，最新状态统一见 [ACCEPTANCE_ZH.md](ACCEPTANCE_ZH.md)。
