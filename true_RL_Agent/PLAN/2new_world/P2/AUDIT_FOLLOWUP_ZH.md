# P2 审计复核与修复

本轮按当前任务 × 机器人边界核实附件，不把建议直接视为事实。旧 15 rounds 的 checkpoint、协议、ledger 和原始证据继续保护。2026-09-08 已完成源头修复、有效完整训练、新 checkpoint 闭环、正式部署与交付核验，**P2 工程 GO**；最终结论见 [验收记录](ACCEPTANCE_ZH.md)。旧/新数值等价仍未成立，此前失败和中断运行保留各自原始证据。

## 实施顺序与边界

1. **接触采样源头。** 已安装 PhysX 实现将 `force_threshold=None` 和 `max_contact_data_count_per_prim=None` 分别解析为 1 N、4，两项不是有效参数漂移。实质问题是新 SensorBase 移除了旧版针对 history 的即时刷新，导致 lazy scene 下接触历史/计时采样频率变化。在后端 ContactSensor 边界恢复有状态采样；不复制物理内核、不修改奖励公式，保留 reset 后不读陈旧接触的上游修复。
2. **归属与证据。** Go2 配置恢复放在其机器人 factory，公共实例化流程不按机器人 ID 分支。冻结协议超时与实际进程 deadline 明确命名。RSL 配置映射不再把字典回读叫作 learner 验证；检查实际构造的模型/优化器，并保留科学合同与原生字段的映射。显式记录 checkpoint 转换和新模型 hash 格式。
3. **必要收尾。** 统一四元数/数组的消费边界；逐项判断地形颜色和 RGB 预热是否有实际替代方案；删除无消费者的后端占位 YAML。提供可实际执行编码和 tensor 测试的日常 CPU 依赖，不增加新的测试编排框架。
4. **严格复验。** 先执行同初态/同动作接触对照，逐项比较历史、接触计时和 reference 奖励项，不能只检查加权总奖励。短训练/恢复覆盖 G1、Go2、G1-29 Push；实际 learner 配置检查必须通过。修复后的 Go2 完整预算训练、checkpoint 评测及原生视频闭环使用新输出目录。视频开/关及编码顺序复验；全量回归、残留入口审计与历史保护检查后再给结论。

## 本轮诊断边界与预算

- 沿用原 P2 的公开 seed、初态与固定 32 控制步动作，不选择新题目改善对照。相同瞬时力的受控轨迹要求接触历史/计时/奖励项按各自物理单位对齐；浮点计时绝对误差上限 1e-6 s，布尔事件一致。reset 陈旧力抑制单独列明。
- 新增一份完整 Go2 训练（196608000 control steps、2000 iterations），随机初始化，最长 10 h；修复前完整训练不得用于修复后资格。新增短诊断/录像预算 2 GPU-h，使用进程实际时长记录。无 LLM/API 调用。
- 无关测试、语义与评分不扩张。本文件记录每项最终处置与证据路径，不再复制原始审计全文。

## 逐项核实与处置

| 附件项 | 判断与源头处置 |
|---|---|
| 1 接触传感器 | **实质问题成立。** 目标 PhysX 构造器将两个 `None` 解析为 1 N / 4；仍显式序列化有效值。真正漂移来自 SensorBase 删除了旧版 history 强制更新条件。新增薄 `SampledContactSensor`，对 history / air-time 状态按配置周期更新，沿用上游力、历史和计时内核。没有修改 reward 公式。 |
| 2 Go2 共享分支 | **成立，已移除。** G1 / Go2 各自 factory 负责原生机器人配置，公共实例化只应用 profile 资产根和采样周期。未再增加一份覆盖参数 spec。这些源码已被 backend execution manifest 的 project snapshot 冻结；Genesis 同时绑定任务科学快照与该 manifest，导入/恢复检查二者。不能因为单独的 cell 科学摘要不变，就忽略执行域约束；本轮不扩张 P1 身份模型。 |
| 3 两种 timeout | **命名问题成立，部分事实需纠正。** Python 字段改为 `protocol_runner_timeout_s`，只读取并验证旧 JSON 中的 7200；实际 process deadline 仍由 composition placement 的 36000 控制。附件所指 `config.py:344` 是构造校验，并非 summary 写入。Push 的旧合同是 500 iterations，不能凭 Stair 的 2000-iteration 耗时一律提高；Go2 Push 的真实成本另在其资格化阶段签发，本轮不假称完整 Push 预算已验证。 |
| 4 本地跳过两项回归 | **成立，已闭合。** 安装 CPU Torch / NumPy / FFmpeg，增加 `[test]` 依赖和 README 命令，取消两个新测试的缺依赖跳过。实际执行完整 469 帧编号校验及 checkpoint tensor 回归。三代生产形态测试需要 Isaac 源码才运行；其目标环境真实结果单独列出，不把本地 skip 当通过。 |
| 5 配置回读恒真 | **成立。** `apply_and_verify` 退役为诚实的 `apply_to`；`verify_runner` 检查实际构造的 actor / critic / distribution / PPO / Adam 和学习率。原生 learner 中人为改变 learning_rate 的反例被拒绝。Adam 是冻结 v1 所依据的 RSL-RL 3 PPO 硬编码行为，映射版本显式记录这一来源，不重写历史优化器 hash。 |
| 6a 占位 backend YAML | **成立，已删除。** 无活消费者；保留历史 PLAN 中对原文件的描述，不再维护一份假的预算入口。 |
| 6b 四元数两套惯用法 | **成立，已统一。** Push 通过 `quaternion_wxyz` 转换，与 Stair 同一 ABI。未增加通用归一化/非法四元数防御层。 |
| 6c 数组边界零散 | **成立，已收敛。** Stair reward 输入从 robot/contact/height 的原生数组进入 Torch 运算前显式转换，Push 复用既有 `first_attr` 消费入口。未包装整个环境或复制 ProxyArray。 |
| 6d 地形颜色 | **视觉变更申报不足成立。** 活动 builder 使用 `none`，因为该目标 trimesh 不提供 Lab `height` 分支所需的 jet。策略没有 RGB 输入，几何和碰撞 mesh 不变；改动已进入项目执行源码 hash，附件“没有进任何 hash”不准确。旧 `isaaclab_terrain.py` 仍是冻结 Phase-0 source artifact 的实物，修改会破坏真实校验，故保留原字节；它不是活动训练 builder。 |
| 6e 八次 render | **不采用所建议的直接替换。** 上游 `num_rerenders_on_reset` 只在 scene 有 RTX sensor 时执行；这里的 viewport RGB annotator 在首次 `render()` 才建立。保留八次不推进物理的预热，并注明与上游机制的差别。 |
| 6f 初始化 hash | **成立。** 新容器使用 `true_rl_agent.initial_policy_state.rsl_rl_5.v2`，训练摘要显式记录版本；恢复非空 state 检查。历史 hash 不改写。 |
| 6g 转换信息丢失 | **成立，已保存。** 两种任务的 eval 摘要记录 `checkpoint_loading`，Stair 录像旁车保留同一记录。它是执行诊断，不进入任务评分、episode outcome 或原有 execution_identity 的严格字段集合。 |
| 6h observation 词汇 | **成立，已恢复协议词汇。** 训练摘要继续写 `policy/critic`；RSL-RL 5 的 `actor/critic` 只在显式映射和版本化 constructed-learner 证据中出现。 |
| 6i 元组位置切片 | **成立，已改为 logical_name 映射与具名构造。** 不再依赖条目位置。 |
| 6j support-effector 约束 | **无需新增测试。** 现有 robot-matrix 测试已构造 torso + ankle 的无效支撑集合并断言拒绝；现有 profile 全部满足约束不等于不可证伪。 |

相邻审计另外修复了旧 checkpoint `log_std` 映射：应写入 `distribution.log_std_param`，此前误写为 scalar 参数名。当前 profile 都使用 scalar，尚未影响活跃训练；增加一个 tensor 映射断言即可，没有扩展 checkpoint 框架。

## 接触对照结果

完整字段及原 NPZ 的 SHA256 见 `results/audit_contact_sampling_comparison.json`，可用 `compare_contact_sampling.py` 复算。

- Go2：名义质量、惯量、材料和 32 步运动学相同。修复前 history 最大差 82.603 N、contact-time 差 0.005 s、last-air-time 差 0.015 s；修复后上述量、首次接触、瞬时力及 `feet_air_time` 分项差异全部为 0。总 reward 最大残差为 1.863e-8，未以总 reward 掩盖分项偏移。
- reset 的零时刻另列：旧缓冲有 35.458 N 陈旧力，新内核返回零；第一步之后瞬时力差为 0。保留上游 timestamp-zero 修复，不重新引入陈旧接触。
- G1：接触计时、`feet_air_time` 和 1024 个历史接触判定一致；受力/运动学仍分叉，`feet_slide` 数值随支撑速度变化。没有把它写成 G1 整体物理等价。
- 四份 train/eval 配置共 318 处差异，逐项归因见 `results/audit_environment_config_disposition.json`，没有未处置项。该报告是完整差异清单，不是数值等价证书；容量、有效接触参数、运行设备和视觉颜色均明确列明。

上游源码核对范围：旧、新 `SensorBase.update`；新 `isaaclab_physx/sensors/contact_sensor/contact_sensor.py` 的 1 N / 4 默认解析；`contact_sensor_kernels.py` 的零时刻处理；`manager_based_env.py` 的 `has_rtx_sensors` reset 分支。材料的 compliant-contact 默认零值另与 [PhysX 材料 schema](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_material_a_p_i.html) 对照。

## 追加发现：PhysX 容量溢出使旧完整训练资格失效

修复后首份完整训练 `audit_go2_full_train_01` 在最后记录 iteration 87 后出现 aggregate-pair 容量错误，日志共 362 条，所报告需求 2097155–2106194，高于默认 2097152。确认 PID / 作业 ID / process group 后终止该作业；子进程消失，runner 留下失败回执，不计入完整训练验收。权威失败日志在服务器 `new_world_p2_audit/true_RL_Agent/PLAN/2new_world/P2/results/audit_go2_full_train_01/`。

进一步检查旧 `go2_full_train_03`：同类错误 669 条，需求 2097153–2105894。**原来的完整训练 GO 撤回。** 正常退出、步数达标和可以加载 checkpoint 不能证明物理交互完整；这份 checkpoint 仅保留诊断用途，不能用于修复后资格。没有改写原始日志或删除旧产物。

最小修正是后端 PhysX 配置把 aggregate-pair 容量下限设为 `2**23`，保留调用者更大的已有设置；不改碰撞过滤、场景布局、机器人数量、动作、reward 或 PPO。四个生产 train/eval CLI 在发布成功摘要前检查自己的 Kit 日志，核心 PhysX error 会成为失败，不再被进程 exit 0 掩盖。使用上述真实失败日志验证了拒绝行为，另有一个本地正反例回归。[上游缓冲区说明](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/core-concepts/physical-backends/physx/configuration.html) 明确要求将溢出视为失败。

容量修复使用新的冻结副本 `new_world_p2_audit_final/true_RL_Agent`，不覆盖前一个副本的源码与失败证据。完整预算重新从随机初始化开始；结束必须同时满足 2000 iterations / 196608000 steps、无 PhysX error、实际 checkpoint / public-dev / 原生录像闭环。`audit_go2_native_frames_c1` 曾在 900 s deadline 退出，保留该失败；不能未经定位便归因为容量问题。

## 当前验收状态

最终修复源码的本地全量为 **1611 项、0 failure/error、19 skip**；目标运行时 **56 项、0 failure/error、0 skip**，含三代 pause / crash / resume。真实 469 帧逐帧编码和 checkpoint tensor 回归已运行。准确命令、日志与校验和见 `results/audit_final_tests.json`，不把 Isaac 专项的本地 skip 当通过。

最终后端上的 Go2 C0/C1/C2 与 G1 共 12 部影片完整解码，帧差 0、50 FPS，36 个首/中/末帧已查看。四组录像的完整 public-dev 摘要、执行身份、checkpoint 加载记录分别与关录像对照完全一致，见 `results/audit_final_video_on_off_comparison.json` 与 `results/audit_final_video_inspection.json`。G1 选中样本的两次失败原样保留；影片仅呈现任务 outcome，没有重新判分或挑选成功样本。

旧 anchor 的新后端公开结果为 G1 **118/216，J_primary=0.7124098944855968**，Go2 **192/216，J_primary=0.9490025176954733**。相对于旧后端，部分 strata 仍超出预先固定的 0.10 成功率差阈值，详见 `results/audit_final_public_backend_comparison.json`。这不是策略改进、新 reference 资格或多训练 seed 泛化结论。

容量修复后的 G1 / G1-29 Push 又分别完成 384 control steps、保存/加载与实际续跑；前者实际 learner 偏移反例被拒绝，后者 100 步评测覆盖施扰。见 `results/audit_final_short_resumes.json`。之前的 Go2 短续跑证据保留。双 GPU 的实际训练重叠与独立日志证据见 `results/audit_final_concurrent_training.json`，它不代表完整训练已完成。

正式 15 rounds 的 138 份物理日志另作只读扫描，没有本次发现的核心 PhysX error，见 `results/audit_formal_history_physics_logs.json`。旧训练资格被撤回的范围限于明确出错的 P2 运行，不扩张为否定正式历史。

最终运行源码与 manifest 分别冻结在 `results/audit_final_source_identity.json`、`results/audit_final_runtime_manifest.json`。中断时尚缺的完整训练、生产准入评测/录像和部署核验，已在下述恢复验收中闭合；原中断状态仅解释对应历史运行。

### 用户停止后的收尾

`audit2_go2_full_train` 已按用户要求停止，最后日志为 iteration **630/2000**，保存的最新文件为 `model_600.pt`；截至停止，核心 PhysX error 为 **0**。这只是未完成的训练前缀，不能据此签发完整预算资格。checkpoint 与原始日志保留在 `new_world_p2_audit_final` 执行副本中。

停止回执为 `results/audit_user_stop.json`。runner 的原始 `launch.json` 仍为 `FAIL`，attempt result 保留 `infrastructure_failure / summary_missing`（中断后没有成功训练摘要）；外层启动摘要 `results/audit_final_full_training_launch.json` 明确标为 `STOPPED_BY_USER`，实际 lease 为 5246.150893621 s。后续新 checkpoint 评测/录像没有启动，其回执 `results/audit_final_checkpoint_followup.json` 为 `CANCELLED_BY_USER`。

本次再次核对时，训练、控制器和后续任务 PID 均已不存在。本地与最终执行副本的 326 个执行文件均匹配冻结 snapshot；正式目录涉及本轮修复的 33 个文件仍匹配修复前 preimage，确认尚未部署。本地全量、目标测试日志及影片校验报告/抽帧文件的 SHA256 与回执一致。仅同步停止状态和既有证据，不重新启动 GPU 作业，不部署产品修复。修正了 README、主计划和本 Phase 文档中残留的“运行中”描述；原始失败回执不改写为通过。

### 2026-09-08 继续剩余验收

用户释放资源并授权完成剩余验收。再次核对两块 GPU 无计算进程、本地/最终执行副本源码均匹配上述冻结身份后，在新目录 `results/audit3_go2_full_train` 使用原 seed 和随机初始化执行完整 2000 iterations / 196608000 control steps，单作业仍上限 36000 s。原中断运行不拼入这份完整回执，其 checkpoint 和停止记录保留。

本次按顺序完成：完整训练与 PhysX 日志验收 → 新 checkpoint 经 production runner 完成 public-dev / 原生录像并解码、抽帧检查 → 按已核对的 33 文件差异部署正式目录 → 正式目录执行身份/入口检查、关闭录像的完整公开评测对照，以及 71 个 checkpoint、801 份正式证据、6 份 reference 的 SHA256 核对。已有源码未变的全量、ABI、短续跑、接触和历史 anchor 录像验收复用。本次不调用 LLM、不推进 P3/P4，不删除历史 checkpoint。新训练入口仅按既有 retention 清理自身运行目录的中间模型，保留最终迭代及权威副本。

完整训练 `audit3_go2_full_train` 使用 seed **1262835026**，完成 **2000 iterations / 196608000 control steps**；最终 `model_1999.pt` 与权威副本 SHA256 相同，核心 PhysX error 为 **0**。新 checkpoint 公开评测为 **165/216 成功，J_primary=0.883938290781893**；3 个评测 seed 的原生录像共 309/304/300 帧，均 50 FPS，完整解码帧差 0，9 张首/中/末帧已实际查看。训练、两份评测均走正式 runner 准入，旧 reference 未被提升或覆盖。

正式目录关闭录像重跑后，完整任务摘要、执行身份、checkpoint 加载记录和后端核验均与开录像运行相同。部署后的入口顺序测试 1 项通过；全部 71 个正式 checkpoint、801 份正式证据、6 份 reference 和 2333 个 Lab 源码文件 SHA256 不变。进程组无残留、GPU 无计算进程、lease 均未持锁，未清空共享锁目录。

本次 GPU lease 为训练 **8472.390 s**、原生评测 **261.606 s**、正式目录关录像评测 **129.145 s**，共 **2.461984 GPU-h**，不冒充整个 P2 历史开发成本。总回执为 `results/audit_completion_acceptance.json`，实际命令与原始回执分别见其引用。**剩余验收已完成，工程 GO；旧/新数值等价、新协议 reference 资格及新搜索成本合同的边界不变。**
