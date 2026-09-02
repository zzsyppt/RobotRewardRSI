# Phase 2 实际实施计划：Stair 任务语义与机器人形态解耦

> 日期：2026-08-26  
> 实施基线：`true-rl-agent-refactor` / `5513076d5cbfe279987310c12c89938ff1a7691e`  
> 状态：已实施；最终验收事实见 `PHASE2_ACCEPTANCE_REPORT_ZH.md`  
> 本阶段范围：只完成 Stair task-owned runtime/reward/evidence 的形态投影与 G1-37 legacy 兼容，不实现 Go2 cell、六足 cell、C1/C2 地形或新的 LOOP 权限

## 1. 阶段目标与不可突破的边界

本阶段把当前 G1-37 专用的 Stair 执行数据链改造成：

```text
ResolvedCellManifest
  -> StairRuntimeProjection（task-owned，显式 frame + 角色到运行时量的唯一解释）
  -> morphology-neutral reward / transition evidence
  -> Stair 成功、失败、surface 与评测语义

legacy G1-37 only:
  morphology-neutral inputs -> G1-37 compatibility adapter -> frozen V2 reward ABI
  frozen V2 recorder bytes  -> legacy transition reader     -> generic transition
```

边界固定如下：

- `robots/` 与 composition 公共层只知道 profile、role、capability 与通用 cardinality 约束，不知道楼梯、surface、足端成功判据或奖励字段；
- Stair 插件独占上楼方向、surface query、support/clearance/slip、成功与失败分类；运动学基座由 binding 明示，不能由 legacy adapter 或某个 role 名暗推；
- `RobotProfile` 继续只保存真实关节、刚体、执行器与共享角色事实；具体 Stair sensor、policy I/O 与兼容投影仍由 `RobotTaskBinding` 提供；
- G1-37 legacy reward ABI、官方 reference program 与历史 transition schema 不删除、不改写；新机器人只消费 VNext，不伪造 G1 的 arm/finger/torso 字段；
- Phase 2 不顺手建设通用 TerrainPlugin、通用机器人插件、动态 curriculum、非矩形考卷或 Go2 EnvCfg。

参考 `D:\humanoid_agent\codex` 的适用思想仅限三点：严格 descriptor、集中 resolver、解析后不可变身份。不会复制其 agent registry、生命周期或权限系统。

## 2. 实际工作包

### P2-W1：可变角色基数契约

在现有共享 `SemanticRoleRequirement` 上增加一个最小、任务无关的“binding 必须选择 profile 为该角色声明的全部目标”约束。Stair 的 `support_effector` 改为至少一个、无固定上限、必须全选；compiler 继续只做组合可行性，不解释 support 的任务含义。

完成定义：2/4/6 个 support profile 使用同一 Stair requirement 均可解析；漏选任一 support 得到确定的 `unsupported`；Push 与两个 legacy profile 的其它角色行为不变。

### P2-W2：StairRuntimeProjection 与 VNext reward ABI

新增 task-owned 的不可变 `StairRuntimeProjection`，只从 profile/binding 读取：显式 kinematic-base source、base role、全部 support、forbidden-contact 目标、全部真实关节、contact/height sensor 投影和相关内容 hash。由这些事实生成 VNext reward input schema hash、capability identity 与 runtime projection hash；manifest hash 只报告来源，不进入 effective projection hash。

VNext 只暴露通用 base/joint/action/support/contact/terrain/progress/termination 量，support 轴长度来自 projection；不存在 arm/finger/torso 等虚构字段。不同 support/joint/action 形状必须产生不同 schema/capability identity。

完成定义：解析不依赖 G1 raw-name 猜测；summary 可报告 resolved targets、support count、schema hash 与 projection hash；同一有效内容无论是否由 manifest 进入都具有同一个 projection hash，相关 binding/profile 内容变化才改变 identity。

### P2-W3：G1-37 reward 兼容适配

Isaac runtime 先构造 morphology-neutral VNext tensor snapshot，再由单独的 G1-37 compatibility adapter 投影到冻结的 `StairRewardInputsV2`。G1 特有 joint group 只存在于该兼容模块，并以 profile/binding 提供的真实 ordered joint set 为输入；通用 builder 不出现 G1 link/joint 名或双足常量。

完成定义：官方 reference reward 仍读取原 42-field V2 ABI；字段 shape/dtype/device、term 集、输入隔离和 deterministic guard 不变；VNext 本身不包含 G1 专属字段。

### P2-W4：变长 transition/evidence 与 surface query

生产 recorder 通过 `StairRuntimeProjection` 解析 body/sensor IDs。新增 VNext transition schema 使用 `support_*` 与 `forbidden_contact_*` 字段；legacy V2 原始 payload 由独立 reader 规范化后进入同一 decoder。`StairEnvTransition`、episode tracker 和 supporting-surface 计算统一消费同长度、非空 support tuples，不再假设 pair。

活跃 evaluation semantics 使用 morphology-neutral 的 `kinematic_base`、`support_effector`、`forbidden_contact` 描述；历史 v6 原样保留。通用 decoder 的 joint width 来自 runtime tensor，并在有 projection 时交叉校验，不再从历史 V1 的 G1 schema 回退。

surface 侧建立 task-owned query interface，并保留当前平踏面解析实现；decoder 只消费 query 返回的 level/height。Phase 3B 可替换 query 实现，但本阶段不实现 rough mesh。

完成定义：2/4/6 support evidence 均能完成 decode、surface projection 与 episode classification；legacy 与 VNext G1 输入产生相同 transition/outcome；成功阈值、稳定步数与 failure priority 不变。

### P2-W5：生产路径接线与身份回写

train/eval child 已重载的 resolved manifest 显式传入 Stair execution，生成 runtime projection 后交给 reward builder 与 recorder；未携带 manifest 的历史/测试入口仅允许使用由当前 G1 profile + legacy binding 解析出的兼容 projection。

完成定义：训练 runtime activation summary 写入 projection/schema identity；eval recorder payload 自证其 projection identity；manifest/profile/binding 与实际 Isaac asset 的解析不一致时在首个环境构造阶段失败，而不是静默回退到常量。

### P2-W6：回归、验收与冻结回执

只增加能证明上述边界的测试：cardinality、2/4/6 projection/evidence、legacy/VNext parity、真实 surface query、summary identity、源码边界，以及已有 reward/35-cell/replay/Push 回归。不开辟与本阶段无关的安全脚手架。

完成定义：目标测试、全仓静态测试和可用的 Isaac 动态 smoke 均有原始记录；形成机器可复算 snapshot、验收报告和开发日志。

## 3. 严格验收门

| Gate | 必须满足的事实 | 失败处理 |
|---|---|---|
| P2-G1 ownership | 公共 `robots/`/composition 无 Stair import 或楼梯语义；从 RobotRegistry 自动派生全部真实 body/joint 名扫描通用 Stair runtime/decoder/evidence/evaluation/plugin，零命中；另检查 `range(2)` 或 support 最大值 2 | NO-GO，回到边界设计，不加例外名单 |
| P2-G2 cardinality | 2/4/6 support 全选均 supported；漏选与 support/forbidden 冲突被 compiler 确定拒绝 | NO-GO |
| P2-G3 projection identity | kinematic-base source/body/joint/sensor 全由 profile/binding 解析；manifest provenance 不改变 effective projection hash；summary 含 resolved targets、projection/schema/capability hash；篡改身份被拒绝 | NO-GO |
| P2-G4 reward ABI | VNext 无 G1 专属字段；2/4/6 schema identity 不同；G1 adapter 对冻结 V2 的 42 fields、shape/dtype 与 reference terms 保持一致 | NO-GO |
| P2-G5 evidence/surface | 2/4/6 support 完成 decode/tracker；legacy/VNext G1 的物理 kinematic-base 量与 transition/outcome parity；surface query 与当前 flat oracle 一致 | NO-GO |
| P2-G6 legacy regression | G1-37 observation/action、reference reward、termination、35-cell outcomes 与历史 Genesis/ledger/reference replay 均不退化；Push tests 不受影响 | NO-GO |
| P2-G7 runtime smoke | 在冻结 Isaac Lab 2.3.0 环境完成实际 projection/body/sensor 解析和最小 reward/recorder smoke；若环境暂不可用，Phase 2 只能标记 code-complete，不能宣称 runtime GO | NO-GO for final Phase 2 GO |
| P2-G8 reproducibility | P2 receipt/verifier 在提交或换行转换后仍可复算；Git HEAD 仅记录 provenance，不作为产品内容门 | NO-GO |

## 4. 明确不做

- 不创建 Go2/六足 RobotProfile 或 cell；2/4/6 只用无 simulator 的契约 fixture 证明轴已打开；
- 不修改 C0 楼梯几何、35-cell 套卷、成功阈值、acceptance 或训练预算；
- 不让 VNext 替换历史 V2 artifact；它们并存，由 resolved cell 明确选择；
- 不把 surface、support 成功判据或 Stair sensor 约定搬进公共 RobotProfile/compiler；
- 不新增数据库、服务、第二套 LOOP、第二套 runner 或宽泛防御框架。
