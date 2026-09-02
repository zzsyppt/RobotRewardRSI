# Phase 3A 历史验收报告：C1 irregular、cell-local 门与复杂闭环前置

> 日期：2026-08-27
> 验收对象：`stair.g1_37dof.irregular_v1`
> 当前结论：**ARCHIVED / SUPERSEDED_BY_PHASE3R**
> 派生状态：不再为独立 C1 cell 签发 `cell_loop_ready`；Phase 4 等待统一 G1 Stair 协议，而不是等待本报告转为 GO
> 当前入口：[Phase 3R 修正计划](../P3R/PHASE3R_CORRECTION_PLAN_ZH.md)

## 1. 结论与因果纠正

原 Phase 3A attempt1/attempt2 的真实训练与评测没有伪造，但它们不能继续作为当前 C1 reference 的能力结论。独立审计确认，当时同时改变了地形与奖励，且所谓 “role-adapted official G1 rough reward” 并不等价于官方奖励：15 项被压成 12 项，四类关节分组被压平为全部 37 个关节，`feet_air_time` 的 single-stance 门控被删除。该缺陷显著削弱运动相对静止的净收益，是 160 个 episode 全部 timeout、课程停在 level 0 的首要解释。

课程还存在第二个独立问题：旧训练范围入口过陡、跨度过窄，且最大坡度低于评测上界。即使奖励正确，from-scratch 训练也缺少稳定起步台阶。

相反，既有 clean reference checkpoint 在同一 C1 calibration cells 上取得约 `0.81–0.996` 的 mean progress，说明 C1 几何本身可行。因此旧 `NO-GO` 只对旧奖励、旧课程和旧 manifest 字节成立，不能再解释为“官方奖励在 C1 上失败”或“C1 地形不可学”。随后架构决定又取消了地形独立 cell：本阶段不再等待 fresh qualification，而是把正确奖励、C1 geometry/query 与 ordered evaluation 能力迁入统一 G1 Stair protocol draft，待 Phase 3B 补齐 C2 后再激活。

后续 Phase 3R 审计又确认：P3A 修正后的 C1 profile 虽补了 easy/hard grade 覆盖，但仍用同一个 difficulty 同时增大 riser、减小 tread，并继续依赖 raw-distance `terrain_levels_vel`。因此它只是一份历史中间配置，不是统一 D_train 的最终训练支持或课程。

同轮边界复审还确认，P3A 的 resolved task hash 强制链在该历史 cell 内有效，但其 base hash 仍传递性包含 `StairProtocolV2` 的固定 G1-37 reward schema，部分 materialization 仍读取 V1 `policy_compatibility`。因此它证明了 cell-local 身份闭合，不能证明 task protocol 已与机器人轴分离；该迁移由 Phase 3R 的 canonical Stair task projection 接管。

## 2. 源头修复

- 保留 legacy `unitree_g1_37dof_isaaclab_v1` 不变；新增版本化 `unitree_g1_37dof_isaaclab_v2`，两者物理/runtime facts 完全相同，v2 只增加机器人拥有的关节组角色。
- v2 明确声明 `ankle / arm / finger / hip_knee / hip_yaw_roll / locomotion / torso` 七个 `joint_group.*`；具体关节名只存在于 RobotProfile，Stair runtime 与 reward 只消费角色派生切片。
- C1 RobotTaskBinding 只声明八个 `(joint_group role, runtime source)` 小映射；通用投影据此生成按组 VNext 输入，不含 G1 组名或关节名。由此恢复官方 15 项有效 reward、原始权重、原始作用域与 single-stance 门控。reference 与未来 candidate 仍走同一 ABI、训练分布、预算、seed 域和评测绑定。
- 历史中间 profile 将 riser 随难度从 `0.05 m` 增至 `0.25 m`、tread 从 `0.48 m` 减至 `0.33 m`；它通过了当时的 grade 端点检查，但 riser/tread 仍完全相关，已由 Phase 3R 的二维支持与 normalized curriculum 取代。
- ordered 18-cell suite、public/sealed cohorts、acceptance floor、reference gate、训练预算、PPO 和 LLM 权限均未放宽。

## 3. 当前验收状态

| 验收域 | 当前结果 | 说明 |
|---|---|---|
| 18-cell ordered suite、C1 几何与解析 surface | PASS | suite/cell-set 未改变；mesh 与解析几何仍共源 |
| RobotProfile 关节组与 VNext ABI | PASS | 7 个 profile-owned groups；8 个 group-derived reward inputs |
| 官方 reward 结构等价 | PASS | 15 项名称、权重及函数 AST 逐项等价；允许差异仅为 role 字段重命名 |
| 官方 reward 数值等价 | PASS | 服务器真 torch、128 组合成状态逐项 `max_abs_diff=0.0`；双足同时接触 air-time 为 0，单支撑为 0.4 |
| 课程入口与评测覆盖 | HISTORICAL_STATIC_PASS / SUPERSEDED | 只证明 grade 端点覆盖；未证明 h/t 独立或 raw-distance 晋级公平，不可作为 unified protocol 的门 |
| C0 / Push / 通用 composition 回归 | PASS | 服务器全量单元测试 `1438 passed, 2 skipped, 1140 subtests passed` |
| 独立 C1 manifest 的 calibration | **CANCELLED_BY_ARCHITECTURE** | 旧 calibration 只保留为 C1 geometry feasibility；新校准属于 unified protocol |
| 独立 C1 evaluation / PPO smoke | **CANCELLED_BY_ARCHITECTURE** | 不再为已退出 catalog 的协议生成新运行回执 |
| 独立 C1 reference / sealed validation | **CANCELLED_BY_ARCHITECTURE** | attempt1/attempt2 只读保留，完整重训不再执行 |
| Phase 3A 总门 | **SUPERSEDED_BY_PHASE3R** | 由 Phase 3R 的统一身份、分布、OOD 和迁移门接管 |

## 4. 历史冻结身份（只用于解释 P3A 回执）

- RobotProfile：`unitree_g1_37dof_isaaclab_v2`，hash `1bc05d6728d1eac4b30dd24eb3c20cc44ed51573b10a494ea41fbed276861478`
- training profile hash：`165dfbe2d95ad6d53b8a509ce2d67b9dd7ae9bdd2b3697e3ce6fb2754531e724`
- ordered suite hash：`9c3db7e3fdd75fa1260909ba21b9048c1b2022ea34cf27757bdc67fbdfd98a38`
- cell-set hash：`aa2a57b0b0fe4b1ea26a9079e766e6d32f73cf757ebefc5cbf157002a34b4034`
- acceptance protocol hash：`4a41bcc09b559d8e0181e7b920e9bed5985b7ee851ce3347bb343ffb620e96a6`
- reference gate hash：`17c16ec71dbf578a9402090e95d7cd3214e998850ea89fb72485f2fa8739e7f6`
- VNext reward input schema hash：`1e8010af97ca3d384316e9c4987a9093bd719e72fa3adecf379cf7d2d51d9b56`
- runtime projection hash：`a78f664a30203edbe67062ab62be06df856ce7014e19f58bc09d0086b63840ac`
- resolved manifest hash：`f28a3455f6ade585eb33b50efa97cb8b963b1685e4e68776b2bc0bde3818e088`
- effective task protocol hash：`79779f793a295884a0e3b8b6d35c963c38a4970631ca0143d1bb1b50dd53fbb8`
- reference manifest hash：`c78e1ac00cf8c99e29863c7fec0e4ac37b79251d496ca3d3a4e2928390233dcb`

legacy clean manifest 仍为 `af7590cee2960116a24902c5b4baac7641551d166b302f6611bf225e11a282d7`，没有因 C1 增加关节组而改变。

## 5. 历史证据如何解释

旧 attempt2 确实完成 `4096 x 24 x 2000 = 196,608,000` env steps，并得到 public-accept success `0`、mean progress `0.06064`；旧 canonical receipt hash 为 `cee369bd2f4bc77ff0e33a50b928ed54da1ab561848f6aed3831defd0251befe`。它现在只保留为“错误 reward + 旧 curriculum”的失败审计，不得作为当前 reference 回执或地形能力结论。

旧 calibration receipt hash `330f9b52ac1ccb5d7cb6a3873946986f897aeee2fb4e8e9461437f5a76480ca4` 的四个角色结果仍是有价值的地形可行性证据：easy/target/hard/coverage-probe mean progress 分别约为 `0.9493 / 0.9507 / 0.8053 / 0.9961`。但由于 RobotProfile、reward ABI、课程和 manifest 身份已经版本化，它不能直接授权当前 cell。

旧 `P3A_C1_IRREGULAR_SNAPSHOT.json` 与旧 runtime receipts 现在只作为历史 NO-GO 证据验其 canonical receipt、记录的 source closure 结构和原始回执字节，不再拿当前产品文件逐字节对比。当前 unified Stair 是否回归由 P3B/P4/P5 的活契约门重算；不得通过重签旧 snapshot 或复制旧 checkpoint 把 P3A 恢复为当前能力证据。

## 6. 迁移后的严格处置

1. 不重跑独立 C1 manifest 的 calibration、PPO smoke 或 `196,608,000` env-step reference qualification；
2. 旧 identity、snapshot、checkpoint 和 receipts 保持只读，不通过重算 hash 伪装为 unified protocol 证据；
3. G1 v2 joint groups、官方 15-term reward、C1 generator/query、ordered summary 与 lineage 强制必须进入 Phase 3R 的迁移回归；
4. 新的 calibration、reference、sealed decision 和 `cell_loop_ready` 只能由唯一 G1 Stair active protocol 产生；
5. Phase 4 的 reference 资格化入口门为 Phase 3R GO + Phase 3B GO；Phase 4 candidate LOOP 还必须等待 unified reference readiness 通过。

因此，GPU 是否空闲不再影响本历史 Phase 的结论；本报告不会从 ARCHIVED 转回 PENDING 或 GO。
