# Phase 3R 开发日志

> 日期：2026-08-28
> 结论：**工程 GO；统一 Stair 协议保持 `draft`**
> 实施计划：[PHASE3R_CORRECTION_PLAN_ZH.md](PHASE3R_CORRECTION_PLAN_ZH.md)
> 验收报告：[PHASE3R_ACCEPTANCE_REPORT_ZH.md](PHASE3R_ACCEPTANCE_REPORT_ZH.md)

## 1. 完成边界

本阶段没有重建系统，也没有把地形保留为第三根矩阵轴。当前新会话只解析一个 `G1-37 × Stair` 单格：

- `cell_variant_id = stair.g1_37dof.unified_v1`；
- C0 regular 与 C1 irregular-flat 是同一训练分布内的逐样本 `terrain_regime_id`；
- C2 尚未实现，训练暴露冻结为零；
- LLM 仍只拥有完整 reward program 的修改权；terrain、课程、PPO、seed、预算、评测与验收全部由 runner/task 持有；
- protocol 保持 `draft`，正式 train/eval、reference 资格化和 candidate LOOP 均 fail-closed。

Phase 3A 的 joint groups、C1 mesh/surface query、ordered evidence 与 identity enforcement 被复用；“C1 独立 cell、独立 reference/ledger/current-best/LOOP”被撤销。

## 2. 实施记录

### 2.1 单格身份与语义边界

- 新增 canonical Stair task projection，直接哈希任务拥有的训练、地形、评测、失败分类与 acceptance 语义，不再传递性包裹含 G1-37 schema 的旧 V1/V2 protocol hash。
- RobotProfile/RuntimeProjection 继续负责 body、joint group、action/observation 与 VNext reward input 的 cell-local 解析；TrainingBindingV2、EvaluationBinding、manifest、checkpoint lineage 和子进程重载共同强制 resolved identity。
- catalog 只保留 `draft / active / archived` 这一项必要生命周期状态。当前 draft 对正式入口拒绝，不引入新工作流引擎。

### 2.2 统一训练分布与课程

- terrain grid 为 10 rows × 10 columns：1 个 C0 column，9 个 C1 `3 pattern family × 3 tread band` columns。
- 4096 个正式训练环境按冻结 seed 分配为 C0/C1 各 2048；C0 初始最大 level=5，C1=1。terrain type 固定，课程只移动 row。
- C0 riser 支持 `[0.05, 0.35] m`、tread 固定 `0.40 m`；C1 riser base 支持 `[0.08, 0.20] m`、tread bases 为 `0.37/0.40/0.43 m`，高度/踏深起伏分别为 `±0.012/±0.018 m`。
- 课程依据实际几何产生的 `goal_progress_fraction`，阈值为 up `0.80`、down `0.40`，不再用与踏深耦合的固定绝对位移。
- `size/floor_depth/start_platform/step_count/top_goal_offset` 只由 training profile 提供；evaluation suite 不再复制这些常量。训练 forward command 改为数值范围 `[0.0, 0.8] m/s`，运行时直接读取并与评测 `0.8 m/s` 交叉校验。
- `training_exposure.allowed_regimes` 只有 C0/C1，`c2_rough` 明确禁止。不存在的 `unseen_final_holdout` 已从活动训练 profile 删除，避免把未来 Phase 7 概念伪装成当前套卷。

### 2.3 评测、目标与 acceptance

- 每个 formal split 使用 7 个 stratum、每层 3 个显式 cell、每 cell 4 episodes，共 84 env/seed；另有独立 calibration cohort。
- public_dev/public_accept/sealed_validation 各自使用 21 道不同几何；geometry digest 不含 cell/cohort 标签，三组展开后 63 个 digest 两两不交。每个 seed 只运行自己的 21 道题，因此总成本仍为 `(3+5+7)×21×4=1260 episodes`。
- `c0_id/c1_id` 表示分布内能力；`height_ood/tread_ood/joint_ood` 表示受控标量外推；`phase_composition_ood` 表示已知 family 内留出的组合；`c1_conditional_frontier` 只声明条件前沿，不冒充全局标量 OOD。
- held-out phase 由 riser 与反向 tread pattern 的实际旋转序列判定。alternating 只保留真实不同的 phase 1，front/back-loaded 使用 phase 2/4/6；成员标签无法再把训练几何伪装成 OOD。
- 主目标冻结为 `J_primary=(2*J_OOD+J_ID)/3`；`J_OOD`、`C0-ID`、`C1-ID` 是三个独立 named guards。
- generic acceptance v6 只处理具名标量、paired samples 与通用非退化规则，不读取 Stair、C0/C1、riser 或 tread 术语。具体 score/stratum 语义全部留在 Stair TaskPlugin。
- `stair_010` 的历史非单调异常没有被臆断为 classifier 缺陷；当前统一套卷不让该点进入 guard 或 reference 支持性判断。
- LLM prompt 只投影 public_dev 的逐题/分层反馈；sealed 的 21 个 cell id 与 cohort override 不进入 prompt，source request 也只能访问 prompt catalog 中的候选奖励源码。当前 sealed 因而是 geometry-held-out；`final_holdout` 明确留到 Phase 7 新增。
- 后续等难度审计确认三套题虽已几何隔离，但 sealed 的 height/tread/joint/C1-frontier 外推距离均系统性大于 public_accept；这会污染 `max_public_to_sealed_score_drop=0.05` 的解释。评测 JSON 已改为相邻低/中/高距离带：public_dev 为 `25/50/75 mm`，public_accept 为 `20/45/80 mm`，sealed 为 `30/55/70 mm`；四层 cohort 平均差最大仅 `4.714 mm`，不再存在单向难度支配。
- tread 与 joint 三套卷统一使用“浅/深/浅”方向；C0-ID 平均 riser 均为 `0.22 m`；C1-ID 与 phase-OOD 的 riser/tread 边际集合分别完全相同，只通过 pattern/phase 组合保持 63 道题面不重合。无序 phase 采用平衡不完全区组，`p2/p4/p6` 各出现于两个 cohort，不把 phase 数值伪装成难度距离；三套卷平均 episode horizon 范围仅为 `587.429–589.524` control steps。reference 上限实际按 OOD family 聚合判断，而非逐题剔除，但 public_accept 偏易仍会让有用 family 被误判饱和。
- 本次语义修改只发生在 `stair_unified_evaluation_suite_v1.json` 的几何数值；protocol/component hash 与测试中的物理成本上限随身份链机械重签。最长单 seed 评测从 `63,336` 降到 `59,976` control steps，没有修改评分、acceptance、prompt、训练分布或共享层代码。

### 2.5 协议加载与未来机器人行

- unified protocol 用 schema 分派，并通过协议内相对 `component_refs` 加载训练、评测、acceptance 和 reward authority；加载器不再写死 G1 cell id 或文件名。
- Phase 5 增加 Go2 时只需新增机器人 profile/runtime adapter、薄 binding/registry 项和 Go2 协议配置，不应修改 Stair TaskPlugin 或 `unified_protocol.py`。

### 2.4 Reward ABI 与 reference

- 新协议只走 VNext reward ABI 与 TrainingBindingV2。
- 官方 G1 rough reference 保留 15 个有效 term、single-stance air-time 门控及 profile-owned joint groups；运行源目录收敛为 `official_g1_rough_vnext_v1`，不再使用带有独立 C1 生命周期含义的 `g1_irregular_v1` 名称。
- 真实 smoke 中 15 个 term 全部进入统一 dispatcher，产生有限且非零的 reward contribution。该证据只证明运行时接通，不证明策略学习效果。

## 3. 退役代码清理

完工后删除了只服务于旧独立 C1 cell 的 5 份配置、3 个运行模块、旧 reward 目录与一套 P3A 开发测试，共 10 个退役路径；同时删除 prompt、selection、public projection、phased evaluation、reference qualification 等模块内被统一条件遮住的不可达分支。

保留下来的 `isaaclab_irregular_terrain.py` 只提供统一任务仍实际消费的 C1 mesh/evaluation primitive；`ordered_evaluation.py` 与 `ordered_summary.py` 只保留统一显式套卷所需的通用 ordered-cell 聚合。P3A 文档、回执和历史 ledger 保持只读，未伪造为新协议证据。

自动边界门只检查两类已知退役对象：确切旧路径不得存在，确切旧 cell/套件标识不得进入活动 `src/tests/scripts/configs`。没有扩展成宽泛字符串防御框架。

完工后二次审计按产品入口、配置加载者和不可替代覆盖三项检查剩余代码。结果是：活动树没有旧 Phase 3A 入口或配置引用；所有 unified 配置均被运行时直接加载；P3R 新模块的公开 API 均有多处调用。除 `f8c5348` 已完成的 10 路径/3512 行退役清理外，没有新的无主代码或重复套件，因此没有继续删除仍被统一 cell 消费的底层 primitive 和验收器。

后续 Phase 5 生命周期审计进一步删除了 P3R verifier 中 400 余行已不可达的“重建当前 draft / 重签 snapshot”脚手架。保留的 verifier 只验证历史 snapshot 与真实 smoke 回执自洽；P3B/P4/P5 分别承担当前 active protocol、G1 LOOP 与机器人行的活语义回归，避免把合法扩展误判成历史 Phase 失效。

## 4. 真实运行时验证

服务器环境：Isaac Lab 2.3.0 对应运行环境、RTX 5090、`cuda:0`。运行 `capture_phase3r_runtime_smoke.py` 得到：

- 16 env，C0/C1 各 8；
- 实际预算 128 env steps，占正式 196,608,000 steps 的 `6.510416666666666e-7`；
- 37 维 action、310 维 policy observation；
- 完整 reward 由单一 dispatcher 调用，15 个 term 均执行；
- curriculum manager 激活，独立探针验证 level 的 up/hold/down 且 terrain type 不变；
- smoke checkpoint 不具备 reference/current-best/candidate 资格。

回执：`PHASE3R_RUNTIME_SMOKE_RECEIPT.json`，等难度修正后的内容身份 hash 为 `a1589b721ce2a036296bec8f0ce96b1a9d405e4388c901453629ef859d18692a`。烟测前还真实生成了 public_dev/public_accept/sealed_validation 各一块代表性 phase-OOD mesh，geometry digest 分别为 `bd58c9d9…`、`d5842a17…`、`48722c13…`，且都读取统一的 `floor_depth_m=0.5`。

## 5. 验证与提交

- 定向 Stair/矩阵测试：84 tests，PASS，2 skipped；
- Phase 3R 验收器自测与契约：Windows/Linux 各 13 tests，PASS；
- 服务器生产形态三代暂停/恢复与硬崩溃恢复：1 test，479.321 s，PASS；
- 等难度修正受影响范围：76 tests，PASS，3 skipped；评分/资格化/分阶段评测补充集 58 tests，PASS；
- 等难度修正后本地全量：1525 tests，256.525 s，PASS，29 skipped；
- 服务器同产品提交受影响集：123 tests，492.744 s，PASS，1 skipped；
- 此前 P3R 服务器全量：1525 tests，902.470 s，PASS，3 skipped；
- Phase 3R 自动验收：Windows/Linux 均 PASS；
- 清理提交：`f8c5348fc6842ab87c87e9b367f2cb3a10440e08`；
- 本轮源修复提交：`b3cda57208764df1ed2107d1baf72a6cc95d04b5`；真实 mesh 探针修复提交：`df895eb9f01725141408acca7c1e2359f102e6bb`；证据冻结随本文提交。
- formal cohort 等难度源修复提交：`4f3e83eb9cc45856075043ff320d4c387c0f7a86`；该提交已在服务器同 SHA 上完成真实 Isaac smoke，证据重签随本文提交。

生产形态测试逐步暴露并修复了旧 fixture 直接读取 legacy protocol、旧 G1 足端字段、4 episode 分数饱和、JSON 对象顺序、sealed task evidence 形状，以及命名 guard 在 decision/replay 中的 tuple/list 不一致。修复均落在各自权威边界，没有给统一 cell 增加特殊旁路。跳过项是既有平台/可选依赖条件，不是为 Phase 3R 新增的豁免。

本轮独立审计另外闭合了 8 个问题：phase OOD 改为展开序列判真；三个 formal cohort 改为 63 道几何隔离题池；共享几何常量和训练指令分别收敛为单一数值来源；协议组件改为相对路径与 schema 分派；未使用的 held-out phase 被实际覆盖；分配逻辑中的 `% 9` 改为由当前 C1 列数派生。其余固定基数继续作为冻结套卷的拒漂移断言，不抽象成通用网格框架。reference 时冻结 family mask 会降低接近饱和层的单轮灵敏度，这是保持同一轮候选可比性的有意取舍，已写入主计划，不修改目标函数。

## 6. 交接

Phase 3B 只需在同一 draft protocol 中加入 C2 generator/query/evidence 与 C2 OOD cells，验证 scanner 可见性及 failure-classification drift，然后重签完整 suite/manifest 并决定是否转 `active`。不得恢复地形 cell、独立 C2 reference 或另一套 LOOP。

Phase 4 才执行完整 official reference 训练、生成 objective family mask 并启动 reward LOOP。Phase 3R 不报告 `loop_verified`、`improvement_observed` 或 reference readiness。
