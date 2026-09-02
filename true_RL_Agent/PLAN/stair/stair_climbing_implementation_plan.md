# 上楼梯任务详细实施计划

- 制定日期：2026-07-23
- 代码基线：`810b760`（`true-rl-agent-refactor`）
- 目标任务：`unitree_stair_climbing`
- 初始策略候选：Isaac Lab 2.3.0 官方 `Isaac-Velocity-Rough-G1-v0` RSL-RL checkpoint
- 当前执行点：Phase 0 经阻塞问题复审与加固后完成，Phase 1A 的共享组合入口与 baseline eval-only 基础设施已完成；下一阶段为 Phase 1B。stair 仍未注册生产 `TaskPlugin`，也未创建 Session Genesis

## 1. 核心设计决定

### 1.1 选择 G1，不选择 H1

现有系统、服务器资产和历史训练均以 Unitree G1 为主。官方 G1 Rough checkpoint 的 robot、37 维动作、310 维 policy observation 和 RSL-RL 网络结构可保持原样复用。H1 会同时引入 robot、动作、观测和 checkpoint 迁移，不能帮助验证楼梯任务边界，因此首版不采用。

### 1.2 官方 Rough checkpoint 只作为 warm-start

官方 checkpoint 在 rough terrain 上训练，因此是合理的初始化策略候选；但它没有经过本系统冻结后的七桶楼梯协议评估，Phase 0 不能把它声明成 accepted current-best。Phase 0 首 episode 探针进一步确认：它能稳定朝楼梯前进并到达第一阶，但 56/56 episode 随后均发生 base contact/fall，成功数为 0。因此它只是可用于定向 fine-tune 的 warm-start，不是“已经会爬楼梯”的 baseline。

完整状态转换必须是：

`官方 checkpoint -> Phase 0 warm-start 候选准入 -> Phase 1 baseline eval-only -> Session Genesis -> baseline current-best`

禁止为它伪造 candidate、train result 或 acceptance ledger record。

### 1.3 不直接复用官方金字塔楼梯作为目标环境

Isaac Lab 官方 `MeshPyramidStairsTerrainCfg` 返回的 terrain origin 位于中央高台。官方 Rough task 会围绕该 origin 重置机器人，因此它验证的是 rough locomotion，不是“从底部朝固定方向上楼梯”。

目标环境采用任务自有的单向上升楼梯：

- 楼梯沿局部 `+x` 方向上升；
- origin 位于底部起步平台；
- reset yaw 固定为 `0`；
- 前进命令固定为正向，关闭 heading/yaw 随机化；
- 训练高度为 `0.05-0.20 m`；
- 评估高度固定为 `0.10/0.15/0.20/0.25/0.30/0.35/0.40 m`；
- 每个评估 bin 使用独立 terrain type，env 到 bin 的映射确定且可审计；
- 不允许 flat terrain fallback。

环境 overlay 必须继承官方 `G1RoughEnvCfg`，保持 observation、action、robot 和控制频率不变。楼梯进度、终点和失败分类全部留在任务包中。

### 1.4 借鉴 Codex 的边界方法，不复制其实现

参考 `D:\humanoid_agent\codex` 的扩展解析方式，采用三条原则：

1. **先解析为惰性描述符，再激活**：Phase 0 只加载并验证协议、baseline 和探针配置；导入 `true_rl_agent.tasks.stair_climbing` 不启动 Isaac、不注册 Gym、不创建进程。
2. **资源受所属根目录约束**：协议和 manifest ref 必须解析到项目根目录内；绝对路径、`..` 逃逸和 symlink 逃逸均拒绝。
3. **运行使用不可变快照**：一次加载得到 `StairPhase0Bundle`；后续校验和探针使用该快照，不在运行中重新读取可变配置。

## 2. 所有权边界

| 内容 | 所有者 | 禁止位置 |
| --- | --- | --- |
| 单向楼梯 mesh、origin、bin 映射 | `tasks/stair_climbing` | `runner`、`evaluation`、`agent_core` |
| 楼梯成功/失败语义与阈值 | `StairProtocolV1` 和楼梯任务包 | 共享协议中的 task-specific 字段 |
| 官方 G1 checkpoint 兼容描述 | 楼梯 baseline manifest | CLI 默认值、backend 隐式值 |
| 通用 eval-only、Genesis、subject/binding | Phase 1 共享合同 | 楼梯插件私有复制 |
| 候选训练执行安全、checkpoint 隔离 | 共享 Runner/Backend | TaskPlugin 直接执行命令 |
| 楼梯 RewardInputs 投影与字段激活证据 | 楼梯 runtime adapter | 通用 `RewardInputs` |

Phase 0 允许新增：

- 任务自有纯数据合同；
- 任务自有 Isaac Lab overlay；
- 显式调用才注册的 Phase 0 probe 环境；
- 薄探针脚本；
- 防止共享层吸收楼梯语义的边界测试。

Phase 0 禁止新增：

- 生产 `TaskPlugin` 注册；
- `BackendAdapter` 楼梯分支；
- baseline acceptance 或 Genesis 的临时替代物；
- 共享 current-best/replay 修改；
- 候选可写的地形、seed、观测、动作、预算或评估协议入口。

## 3. Phase 0 实施

**状态：已于 2026-07-24 完成**

冻结入口与证据：

- 唯一不可变信任锚：`configs/tasks/stair_extension_schema.json` 的 descriptor 自哈希；
- protocol、baseline manifest、checkpoint 和 report 的当前 hash 只存在于 canonical JSON/真实文件及其绑定链中，由 `StairPhase0Bundle` 和报告校验器自动验证，计划不复制派生值；
- 绑定报告：`PLAN/stair/phase0_report.json`

报告已验证官方 checkpoint 的共享安全加载、完整 G1 body/joint 顺序、真实 observation manager term/callable、action manager resolved joint 顺序、terrain grid/env origin 和 `56 env` 首 episode。每档 8 个 episode 均到达第一阶，随后均以 `base_contact` 结束并触发 fall 诊断，七档成功数均为 0。该报告只证明 warm-start 候选具备结构一致性与面向目标任务的前进可行性，不是 baseline eval 或 acceptance evidence。

### 3.1 协议与 baseline 资产

新增：

- `configs/tasks/stair_protocol_v1.json`
- `configs/tasks/stair_extension_schema.json`
- `configs/baselines/isaaclab_g1_rough_stair_v1.json`
- `src/true_rl_agent/tasks/stair_climbing/phase0_contract.py`

`StairProtocolV1` 必须完整冻结：

- robot、控制周期、observation/action schema；
- 单向楼梯几何、训练范围和七个评估 bin；
- reset pose/velocity、command 三轴值与重采样模式、heading/standing 比例、外力事件和 episode timeout；
- 固定训练预算与 checkpoint selector；
- 三个评估 split、seed 数、episode 数、env/bin 映射；
- seed 级七桶宏平均的统计定义；
- 成功、互斥 primary reason、重叠 diagnostic flag 和优先级；
- acceptance 常量与 required gates；
- 共享 RewardInputs 身份、计划中的楼梯字段、字段激活门、parameter-only 项和禁止项。Phase 0 的 task-specific prompt-effective 集合必须为空。

加载器采用 exact-key 校验；拒绝未知字段、缺字段、未允许的 `null`、`TODO/TBD`、非有限值、重复 bin、非递增高度、budget 算术错误和 hash 错误。代码只冻结 descriptor；protocol 与 manifest 验证结构、自哈希、内部关系及执行安全约束，随后由 bundle 对 descriptor 的 ref/id/hash 逐项绑定。修改 protocol 或 manifest 后重算自哈希可以作为独立结构对象解析，但不能进入冻结 bundle。共享 `EvaluationProtocol` 与 `AcceptanceProtocol` 只能由该文件构造并交叉校验。

baseline manifest 必须绑定：

- 官方 task 和 checkpoint 来源；
- checkpoint SHA-256、文件大小和 iteration；
- Isaac Lab、Isaac Lab Tasks、Isaac Lab RL、RSL-RL、Torch 版本；
- robot、policy observation、action 和网络结构；
- robot、observation/action schema hash；
- 角色 `warm_start`，并明确 `accepted_current_best=false`。

manifest 不写服务器绝对路径；本地 checkpoint 路径只允许作为探针 CLI 参数，并由 runner-owned hash 校验。

### 3.2 单向楼梯环境 overlay

新增：

- `src/true_rl_agent/tasks/stair_climbing/isaaclab_terrain.py`
- `src/true_rl_agent/tasks/stair_climbing/isaaclab_env_cfg.py`

实现要求：

- 自有 `OneWayStairsTerrainCfg` 和 mesh generator；
- 起步平台、固定数量台阶和顶部平台均为真实碰撞 mesh；
- terrain origin 位于起步平台，不位于顶部；
- eval generator 为 `1 row x 7 cols`，每列唯一对应一个高度；
- `56 env` 时必须得到每个 bin 连续 `8 env`；
- train generator 只在任务拥有的 `0.05-0.20 m` 范围采样；
- 显式关闭 terrain-level curriculum、随机 yaw、横向命令、旋转命令、外力和推扰；
- 保留官方 G1 Rough 的 robot、37 维动作、310 维 policy observation 和 `dt=0.005/decimation=4`；
- 包导入不触发 Gym 注册；只有 probe 明确调用注册函数时才注册 Phase 0 ID。

### 3.3 真实探针

新增：

- `scripts/isaaclab_stair_phase0_probe.py`
- `src/true_rl_agent/tasks/stair_climbing/isaaclab_phase0_probe.py`

探针必须：

1. 在启动 Isaac Sim 前解析 CLI；
2. 显式加载 protocol 与 manifest；
3. 通过共享 `build_checkpoint_safety_report()` 校验 checkpoint 的 hash 与 `weights_only=True` 安全加载，再校验任务 manifest 的文件大小；
4. 创建 `56 env` 的七桶 eval overlay；
5. 导出并校验完整 body/joint 顺序、observation manager term/callable/shape/params、action manager resolved joint 顺序、terrain grid/env origin 和 env/bin 映射；
6. 加载官方 RSL-RL checkpoint 并完成前向；
7. 跟踪每个 env 的首 episode，最多运行完整 `600 control steps = 12 s`；提前终止必须记录真实 base-contact/timeout、进度、fall、top hold 和 success；
8. 检查 observation、action、reward 和关键 robot state 无 NaN/Inf；
9. 输出 canonical Phase 0 report，包含版本、hash、runtime schema、映射、几何、逐 env 首 episode 与严格派生的每档摘要；
10. 在 `finally` 中关闭 env 与 simulation app。

Phase 0 report 是探针证据，不是 public eval result，不得用于 acceptance。

### 3.4 本地合同与边界测试

新增/修改：

- `tests/unit/test_stair_phase0_contract.py`
- `tests/unit/test_stair_extension_guide.py`
- `tests/unit/test_project_boundaries.py`

覆盖：

- 正确协议/manifest 可严格加载；
- 任一字段、hash、预算、bin、schema 或角色被篡改均失败；语义修改后重算自哈希仍失败；
- resource ref 无法逃出项目根；
- shared protocol 从 StairProtocolV1 单源派生；
- 任务包存在但 production runtime/training 开关仍为 false；
- 通用包不能导入楼梯包，也不能出现楼梯高度、bin、toe collision、terrain progress 等任务语义；
- push-recovery candidate policy 不得拥有 `stair_height_range`；楼梯 candidate surface 只能从 `StairProtocolV1` 派生；
- 包导入不加载 Isaac Lab、不注册 Phase 0 Gym 环境。
- 计划中的 `foot_clearance_m`、`foot_contact` 和 `terrain_progress_delta` 在 Phase 3 通过真实 shape/finite/nonzero/reset 探针前不得进入 prompt-effective ABI。

### 3.5 Phase 0 严格完成标准

必须同时满足：

- 七个高度均具有可辨认、非平地的真实 mesh；
- terrain origin、上升方向、spawn 和目标方向一致；
- 每个 bin 恰好 8 env，映射与 protocol 一致；
- 56 env 的完整 robot/observation/action runtime schema 与 manifest 一致；
- checkpoint hash、版本、iteration 和网络结构与 manifest 一致；
- checkpoint 完成推理，56 个首 episode 全部被闭合记录；完整 12 秒 horizon 之前的提前终止不得被重复 reset 样本掩盖；
- 训练分布内三个 bin 每档至少 50% episode 到达第一阶；成功不作为 Phase 0 warm-start 准入条件，但成功数必须如实记录；
- 纯合同测试、边界测试和项目全量测试通过；
- Phase 0 资产没有启用 production plugin/training command；
- report 明确区分 checkpoint 结构、runtime schema、terrain binding、首 episode 完整性、task viability 和 `warm_start_candidate_admitted`；同时保持 `accepted_current_best=false`、`report_is_acceptance_evidence=false`。

任一项失败，Phase 1 不得开始。

## 4. Phase 1：共享 baseline 闭环

### Phase 1A：组合入口与 eval-only

状态：已完成（2026-07-25）。

实现统一 task composition resolver、`EvaluationSubject`、`EvaluationBindingV1`、
`BaselineEvalJobSpec` 和 Runner baseline eval。`run_loop.py` 与 `materialize_job.py`
使用相同的两阶段 task/default 解析。Runner 在 eval/warm-start train 子进程启动前校验
sealed checkpoint 的实际字节与 hash；raw summary 只接受一套扁平 subject identity。
Phase 0 source manifest 通过楼梯任务包内 adapter 单向派生并绑定 runtime baseline manifest，
共享层只理解通用 source binding，不引用楼梯语义。

详细步骤与硬验收见 `PLAN/stair/phase1a_implementation_plan.md`。

验收：

- baseline eval 没有 candidate identity、candidate source 或 train result；
- task/backend/schema/checkpoint/binding 任一不符均在 baseline eval subprocess 前失败；
- 错误 checkpoint 字节在 eval 与 warm-start train 子进程启动前拒绝；
- raw summary/metrics 第二身份、source/runtime manifest 链漂移和 initialization/anchor
  manifest hash 不一致均 fail closed；
- push 行为无回归。

### Phase 1B：Genesis、current-best 与 replay

实现 create-once `SessionGenesisArtifact`、`VerifiedSessionGenesis`、完整 `CurrentBestEvalStats`、baseline anchor prompt 状态和 Genesis-aware replay。

验收：

- baseline 由 runner-owned eval 证据推导为 current-best；
- live loop 只接收一份 Verified Genesis；
- ledger 全部绑定同一 Genesis 和 EvaluationBinding；
- 篡改与跨协议比较 fail closed；
- session 恢复不重抽 seed、不覆盖 Genesis。

## 5. Phase 2：楼梯插件与公开合同

实现楼梯 `TaskPlugin`、TaskSpec、candidate surface、prompt/smoke/memory adapter、`StairClimbingRewardInputs` 和评估聚合器。注册发生在 composition root，不通过共享层的 task ID 条件分支。

验收：

- 只支持 `warm_start` 且强制 baseline Genesis；
- 正式 ID 为 `unitree_stair_climbing`，旧 ID 明确拒绝；
- 失败计数闭合，缺桶/重复桶/警告/非有限值均不能 pass；
- prompt 不暴露 seed、checkpoint 本地路径、hidden rollout 或 eval-only 信号。

## 6. Phase 3：共享候选运行时

从 push adapter 中只提取任务中立的候选加载、shape/finite guard、激活统计和诊断执行。push/stair 分别提供 task-owned RewardInputs builder。

验收：

- 新增 `RewardTerm` 始终是 code-level；
- schedule、已有 term scalar weight 和楼梯范围修改保持 parameter-only；
- 楼梯字段无 zero fallback；不可用字段不进入 prompt-effective ABI；
- push runtime 结果和摘要无回归。

## 7. Phase 4：正式 train/eval backend

实现薄 train/eval entrypoint 和 backend materialization。每轮训练固定从 Genesis baseline 开始；七桶 public/sealed eval 只加载 trusted checkpoint，不加载候选代码。

验收：

- argv、落盘 config、summary 和 checkpoint lineage 绑定同一 protocol；
- 训练预算算术严格相等；
- baseline/candidate subject 和 seal 正确；
- split/seed/bin/episode 计数完整；
- accepted checkpoint 不成为下一轮训练起点。

## 8. Phase 5-6：联调与启用

本地全量回归后，在服务器执行 baseline Genesis、两 iteration warm-start smoke、七桶 eval 和失败注入。最后运行 1-3 个真实候选轮次并核对在线状态与 replay。

只有完整链路通过后才设置：

- `runtime_plugin_enabled=true`
- `training_command_enabled=true`

`llm_stair_height_progression_enabled` 在 V1 始终保持 false。

## 9. 实施顺序

1. 固化并测试 Phase 0 合同。
2. 实现单向楼梯 mesh 与 eval overlay。
3. 在服务器运行 geometry/config 探针。
4. 加载官方 checkpoint，运行七桶首 episode 探针；每个 episode 最多允许完整 `600 control steps = 12 s`，提前结束时记录真实终止原因。
5. 回填 Phase 0 report、实测版本、shape 和吞吐量。
6. 运行局部及全量测试。
7. 对共享层做楼梯语义泄漏复审。
8. Phase 0 全部通过后，才开始 Phase 1A。
