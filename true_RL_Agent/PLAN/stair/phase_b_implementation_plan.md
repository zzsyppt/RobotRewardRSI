# Stair 完整奖励自进化 Phase B 实施计划

## 1. 阶段定位

本阶段对应 `revised_plan.md` 的“阶段 B：生产 Stair 任务链”。实施基线为提交 `ae2260e`。

Phase B 的目标是让 `unitree_stair_climbing` 拥有自己的严格配置、插件、组合 provider、Isaac Lab 环境入口、七桶评估证据链和 prompt/memory 边界，并能通过现有 backend 物化结构完整的 candidate train/eval job。

本阶段不实现完整奖励 dispatcher，不运行 R0 训练，不创建 Reference Genesis，也不允许 Stair 进入默认生产 registry。这些能力分别属于阶段 C、D、E。Phase B 不能临时回退到 warm-start、bootstrap 或 Push additive reward。

## 2. 当前代码结论

### 2.1 可复用边界

- `TaskPlugin`、`TaskCompositionProvider`、`ResolvedTaskComposition`、`IsaacLabRslRlBackendAdapter` 已是任务中立组合边界；
- backend 已绑定预算、RSL-RL policy/PPO 配置、training/evaluation binding、结构化 argv 和 runner 预检；
- Phase 0 已验证 G1 observation/action schema、单向楼梯 geometry、train/eval env overlay 和 auto-reset 前 recorder 时序；
- V1 已冻结七桶、episode horizon、成功/失败阈值和 acceptance；draft V2 已冻结完整奖励 capability、random-init reference 方向和 formal seed resolver 版本；
- shared prompt/memory/smoke 接口允许任务自己生成公开语义，不要求共享层认识楼梯字段。

### 2.2 必须替换或扩展的局部实现

- 当前通用名字的 Isaac train/eval 脚本实际是 Push 专用，不能让 Stair 复用其 overlay、seed policy 或 additive wrapper；
- `StairPhase0TransitionRecorder` 假设恰好一个 timeout term 和一个非-timeout term；服务器 Isaac Lab 2.3.0 的 `TerminationManager.get_term()` 会跨 reset 留存旧值，生产 recorder 必须在 term callable 的本步真实执行点捕获返回 mask；
- 现有 `evaluation_contract.py` 只有成功率聚合和总数断言，没有完整 episode tracker、diagnostic flags 和 public projection；
- 生产 registry 只有 Push，这是当前正确状态；Stair 只能通过显式 staging root 注册；
- canonical R0 源码尚未在阶段 C 产生，因此 Phase B 不能构造虚假 reference manifest。

## 3. 架构决定

### 3.1 单一配置真相

新增 task-owned `StairClimbingTaskConfig`。它从 canonical V1 和 draft V2 文件加载语义，只接收组合层已有的任务中立运行参数：预算、env 数、迭代数、seed 数、profile、entrypoint、timeout。

配置对象负责一次性验证：

- `budget_steps == train_num_envs * train_steps_per_iteration * train_max_iterations`；
- 当前预算属于 draft V2 的允许 pilot 档位；
- `eval_num_envs == 七桶数 * episodes_per_seed_per_bin`；
- public-dev/public-accept/sealed seed 数与 V1/V2 一致；
- episode horizon 与 timeout signal 一致；
- train/eval profile 名、V1 hash、V2 hash 和 G1 observation/action schema 一致；
- RSL-RL 网络与 PPO 配置采用官方 `G1RoughPPORunnerCfg` 的冻结等价值，实际执行进程必须重新应用并校验 canonical hash。

`TaskSpec`、backend config、runtime snapshot 和 argv 全部由这一已验证对象派生，不在 provider 中进行 `int()`/`str()` 容错转换。

### 3.2 Reference 尚未产生时 fail closed

`StairClimbingCompositionProvider` 构造时必须显式接收 runner-owned resolver 从真实源码字节生成的 `VerifiedReferenceProgramArtifact`。provider 不接受裸 `ReferenceProgramManifest`，不生成 planned manifest，也不把 warm-start baseline 当作 reference。provider 不用 `authorship_origin` 推断 session role 或 reference 资格；两者按总计划保持正交。

新增 task-owned staging registry builder，只有传入已绑定源码 bytes/hash/size/upstream provenance 的 verified reference artifact 才注册同一个 provider。默认 `build_application_composition_registry()` 不变。阶段 C 产生 canonical R0 源码后，由同一个可信 resolver 生成 artifact 并交给同一 provider；不另建 D/E/F 专用 provider。

### 3.3 显式生命周期门

参考 `D:/humanoid_agent/codex/codex-rs/features/src/lib.rs` 对 `UnderDevelopment / Experimental / Stable` 的显式分层思想，本项目不依赖隐式 import 激活：

- default registry：只包含已验收的 Push；
- stair staging registry：显式调用才包含 Stair；
- Stair runtime profile：明确记录 `complete_reward_runtime=not_activated_until_phase_c`；
- Phase B train job 可完整物化，但 Stair train entrypoint 在 reward dispatcher 激活前必须在 PPO 启动前 fail closed；
- eval entrypoint、环境与 task recorder 在 Phase B 可独立验收。

### 3.4 私有 seed/bin resolver

新增 task-owned resolver，按 `task protocol hash + evaluation binding namespace + split + seed index` 派生运行 seed，并生成公开可绑定的 cohort hash。

公开层只允许出现：resolver version、split 名、seed 数、run index、七桶覆盖状态和 cohort hash。禁止出现 raw seed、nonce、env-to-bin 私有数组。每个 seed 的 env 布局必须由 V1 的七桶顺序和 `envs_per_bin` 唯一生成。

pilot、formal reference 和 formal candidate 使用不同 namespace；Phase B 先实现并测试 namespace 不相交，阶段 D 冻结 final V2 时再绑定正式值。

### 3.5 生产 recorder 与闭合分类

新增 Stair task-owned production transition recorder：

- 只在 eval cfg 显式启用；train cfg 和 Phase0 production-check cfg 均不得启用；
- 仅在 eval config 构造时包装所有 termination term callable，在 Isaac auto-reset 前读取带当前 `common_step_counter` stamp 的本步返回 mask；
- 同时克隆 done union、root/ankle/contact/terrain identity 和 episode length；
- 对 `done == OR(all term masks)` 做强一致性校验；
- recorder 输出由 runner-owned task code 产生，candidate diagnostics 不进入分类输入；训练 config 不安装 callable wrapper 或 recorder。

纯 Python `StairEpisodeTracker` 消费这些 transition：

- 按 V1 阈值维护 max progress、top hold、fall/pitch/slip/clearance streak 和 diagnostic latch；
- success 优先满足完整成功谓词；失败按 V1 `failure_priority` 选唯一 primary reason；
- `stalled_progress_detected` 定义为 episode 结束时未成功且 `max_progress < goal`，不引入新的数值阈值；
- 每个完成 episode 输出一个 primary reason 和可重叠 diagnostic flags；
- 任意未知 termination term、缺字段、NaN/Inf、done/mask 不一致或无法分类都产生 classification warning，并使 behavior gate fail。

Phase0 recorder 改为复用同一逐 term capture helper，但保持 Phase0 extras key、报告 schema 和已冻结 protocol/descriptor hash 不变。

### 3.6 七桶统计与公开投影

每个 split 的每个 seed 必须恰好收集八个 episode/桶。统计顺序固定为：

1. 每桶 episode success rate；
2. 每 seed 对七桶等权宏平均；
3. 多 seed 形成 mean/sample std/standard error。

公开结果保留桶 ID、聚合计数、primary failure breakdown、diagnostic flag rates、progress 摘要和分类 warning 数；不保留 raw seed、逐 env 私有映射、terrain origin 或 sealed episode。

`stair_behavior_gate` 只从 eval entrypoint 的 task-recorder evidence 构造。TaskPlugin 仅验证/投影该证据并附加训练预算 gate，不读取 candidate runtime diagnostic 作为通过依据。

### 3.7 Prompt、memory 与 smoke 边界

- Prompt adapter 公开任务目标、固定七桶统计定义、primary/diagnostic 语义、失败反馈和不可修改项；
- raw seed/bin assignment、sealed validation rollout、checkpoint 路径和 terrain raw map 必须 scrub；
- Memory adapter 只压缩已验证 public result，保留 accepted/current source 之外的公开失败趋势，不复制私有 recorder 数据；
- Phase B smoke adapter 明确返回“完整奖励 ABI 尚未激活”，不伪造 `StairRewardInputsV2`；阶段 C 在同一 adapter 上补齐真实 detached-clone fixture。

## 4. 实施步骤

### B1. 配置、profile 和 provider

新增：

- `tasks/stair_climbing/config.py`；
- `tasks/stair_climbing/runtime_profile.py`；
- `tasks/stair_climbing/plugin.py`；
- `tasks/stair_climbing/composition.py`；
- `tasks/stair_climbing/staging.py`。

完成严格 config、官方 G1 PPO canonical payload、TaskSpec、evaluation/training binding、same-provider staging registry 和 snapshot 一致性测试。

验收门：错误类型、预算关系、seed/env/bin 关系、profile/hash 漂移均在 backend materialization 前失败；默认 registry 仍只含 Push。

### B2. Seed/bin 与 episode 证据核心

新增：

- `tasks/stair_climbing/seed_policy.py`；
- `tasks/stair_climbing/episode_evidence.py`。

扩展 `evaluation_contract.py`，建立完整 split accumulator、宏平均、闭合计数和 public-safe projection。

验收门：每 seed 七桶完整；少桶、多桶、重复 episode、raw seed 投影、未知终止 term、done/mask 漂移全部拒绝。

### B3. Isaac env 与 recorder

保持 Phase 0 的已冻结 `isaaclab_env_cfg.py` 不变，在独立 production 模块中：

- 新增 eval-only termination callable capture；
- 新增 production eval recorder/cfg；
- 新增显式 production Gym registration；
- train builder 不安装 capture，也不启用任何 recorder。

验收门：动态 fake-env 覆盖终止步跨阶、timeout+contact 并发优先级、任意数量 term、done/mask 不一致和 reset 后状态污染；Phase0 现有行为测试继续通过。

### B4. 正式 Stair entrypoint 与 backend 物化

新增 Stair 专用 train/eval entrypoint，复用 backend 的任务中立 argv 合同，但不导入 Push overlay/runtime。

- train entrypoint：完成 identity、sealed candidate、TrainingBinding、profile、预算和 env cfg preflight；在 Phase C dispatcher 未激活时于 PPO 启动前结构化 fail closed；
- eval entrypoint：加载 runner-sealed checkpoint，使用生产 eval cfg、private resolver、task recorder 和七桶 aggregator，生成 candidate-isolated summary/seal evidence；
- backend 仍只接收 opaque profile/hash，不增加 stair 参数。

验收门：materialized argv 指向 Stair entrypoint；task/protocol/profile/budget/identity hash 一致；argv 中无 Push profile；错误 config 在子进程前或 entrypoint preflight 阶段明确失败。

### B5. Prompt、memory、smoke 和 public gate

新增 Stair task-owned adapter，并在 plugin 中接入。测试 prompt/public result 不含 raw seed、env mapping、terrain origins、checkpoint path 或 candidate diagnostics 伪造的 gate。

验收门：相同公开记录得到确定性摘要；candidate 注入同名 `stair_behavior_gate` 不会覆盖 recorder gate；Phase C 前 smoke 明确不可用而不是伪造输入。

### B6. 回归与相邻审计

至少执行：

- Phase B 新增 unit/integration tests；
- Stair Phase0/V2 全部既有测试；
- composition/backend/materializer/runner/prompt/memory tests；
- Push 全路径和全仓 613+ tests；
- Ruff `E9,F,I`、compileall、`git diff --check`；
- server 同提交回归。

相邻审计检查：

- shared schema 是否出现 `stair/bin/height/failure` 字段；
- default registry 是否意外注册 Stair；
- train cfg 是否启用 recorder；
- materializer、snapshot、backend 或 argv 是否复制第二份协议事实；
- prompt/public 是否泄漏 private resolver 输入；
- Phase B 是否误启 complete reward/additive reward/reference Genesis。

## 5. 严格完成标准

Phase B 只有同时满足以下条件才可标记完成：

1. 同一个 `StairClimbingCompositionProvider` 在显式 staging registry 中 resolve 一次并产生一致 snapshot/backend/argv；
2. 没有由真实源码解析出的 verified reference artifact 时 provider/registry fail closed，不接受裸 manifest，不生成 placeholder R0；
3. 默认生产 registry 仍只有 Push；
4. train/eval 预算和七桶关系在物化前严格成立；
5. 每个完成 episode 恰好一个 primary outcome，所有计数闭合；
6. timeout+contact、terminal transition、任意 term mask 和 done/mask 漂移有行为测试；
7. 每个 seed 完整覆盖七桶，统计单位是一 seed 一个七桶宏平均；
8. public/prompt 无 raw seed、私有 bin mapping 和 sealed episode；
9. `stair_behavior_gate` 只能由 task recorder evidence 满足；
10. train 配置无 recorder，complete reward runtime 在 Phase C 前不能被执行；
11. Push materialize/dry-run/run-round/live-loop 与全量测试不退化；
12. 本地、origin、服务器 HEAD 一致且服务器回归通过。

## 6. 预计工作量

- B1 配置与组合：0.5–1 人日；
- B2 seed/evidence/aggregation：1 人日；
- B3 recorder/env：1–1.5 人日；
- B4 entrypoint/materialization：1–1.5 人日；
- B5 adapters/gates：0.5 人日；
- B6 回归与审计：0.5–1 人日。

总计约 4.5–6.5 人日。GPU 正式 R0 pilot 不计入 Phase B。

## 7. 实施与验收记录

- 生产 eval termination 证据不再读取会跨 reset 陈旧的 `get_term()`；capture scheme 已进入 eval runtime dynamics hash；
- 真实 Isaac Lab 2.3.0、56 env 的首 episode smoke 完整闭合，七桶各 8 个 episode；
- 使用 Phase 0 官方 G1 checkpoint 跑通 3 个 public-dev、5 个 public-accept、7 个 sealed-validation seed：对应 168、280、392 个 episode，分类 warning 均为 0；
- 该 checkpoint 在三组 split 的成功率均为 0，sealed gate 按协议返回 fail；它只用于验证评估执行链，不作为 R0、Genesis 或能力证据；
- staging/provider 已拒绝裸 manifest；通用 resolver 从实际 UTF-8 源码字节原子派生 content-addressed URI、checksum 和 size，并将 upstream provenance 一并绑定到 manifest/verification hash；
- 已增加 `run_all_stair_splits` 的 recorder-schema → decoder → tracker → 七桶聚合回归，以及服务器 Isaac Lab 下真实 recorder 和 train/eval env builder 的动态回归；
- Phase B train entrypoint 在完整 preflight 后、PPO 启动前按设计 fail closed，等待 Phase C 激活 complete-reward dispatcher。
