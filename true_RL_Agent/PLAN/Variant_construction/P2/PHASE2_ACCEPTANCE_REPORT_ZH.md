# Phase 2 验收报告：Stair 任务语义与机器人形态解耦

> 验收日期：2026-08-27  
> 实施入口：`5513076d5cbfe279987310c12c89938ff1a7691e`（provenance only）  
> 结论：**GO**

## 1. 交付结论

Phase 2 已将 Stair reward/evidence/evaluation 的形态解释收敛到 task-owned `StairRuntimeProjection`：公共机器人层只提供真实 profile/roles，composition 只做通用可行性编译，Stair 任务层负责将已解析 roles 投影为 support/forbidden-contact/surface 语义。运动学基座不再从 legacy 分支或 `base` role 暗推，而是由 `RobotTaskBinding` 显式声明 `articulation_root` 或 `body_role`，reward 与 recorder 共用同一读取函数。

```text
RobotProfile + RobotTaskBinding + ResolvedCellManifest
                        |
                        v
              StairRuntimeProjection
                 /              \
       reward VNext              transition VNext
            |                          |
 G1 legacy adapter only       legacy V2 reader only
            |                          |
 frozen 42-field V2           generic decoder/tracker/surface query
```

这不是“Go2 已接入”或“复杂地形已实现”。它只证明 Stair 的任务语义核心不再依赖双足/G1 常量，并保留现有 G1-37 旧行为，为 Phase 3A/3B/5 提供可组合入口。

## 2. 严格验收门

| Gate | 结果 | 权威证据 |
|---|---|---|
| P2-G1 ownership | PASS | `robots/` 与 shared compiler 对 Stair 零 import；9 个通用 Stair 文件以 registry 中 104 个真实 body/joint 名的并集做 AST 字符串扫描，零命中；`range(2)`、support max 2 和旧展示名也零命中；G1 入口只在 `legacy_*` 模块 |
| P2-G2 cardinality | PASS | 2/4/6 support profiles 用同一 requirement/compiler 全部 supported；漏选任一 support 均稳定拒绝为 `role_selection_incomplete:support_effector`；support/forbidden 冲突规则保持 |
| P2-G3 projection identity | PASS | profile/binding 决定 kinematic-base source、role/joint/action/sensor；manifest hash 只作 provenance，不再污染 effective projection hash；有/无 manifest 的同一 cell 报告相同 projection hash；真实 Isaac summary 给出独立 robot/contact-sensor IDs |
| P2-G4 reward ABI | PASS | VNext field schema 无 G1 形态字段且 2/4/6 identity 不同；真实 G1 通用 snapshot 经唯一 legacy adapter 后仍是 42-field V2，并成功执行原 15-term reference reward |
| P2-G5 evidence/surface | PASS | variable support transition/tracker/surface query 的 2/4/6 tests 通过；通用 decoder 从 runtime tensor 推导 joint width，不再读 V1 G1 schema；旧 flat surface oracle 与 query 一致 |
| P2-G6 legacy regression | PASS | 最终稳定字节全仓 1505 tests 全部通过（29 skipped，987.665 s）；既有 Stair/Push、35-cell、Genesis/ledger/replay/reference 测试无退化 |
| P2-G7 runtime smoke | PASS | RTX 5090 / Isaac Lab 环境真实构造一 env；同一帧只移除 legacy adapter 标识后，pos/quat、世界/体坐标线角速度和 projected gravity 七组量的最大绝对差全部为 0；原 V2 recorder/reward 继续运行 |
| P2-G8 reproducibility | PASS | snapshot/verifier 使用 normalized-LF source hashes + canonical JSON；HEAD 仅作可空 provenance；5 个 verifier 自测覆盖当前快照、内容篡改、HEAD 非门、CRLF/LF 等价和 registry 派生扫描，且无 `.git` 的服务器归档可复跑 |

## 3. 真实 runtime 回执

`PHASE2_RUNTIME_PROBE_RECEIPT.json` 的核心事实：

| 项目 | 结果 |
|---|---|
| cell | `stair.g1_37dof.clean_legacy_v1` |
| manifest | `4f3cba01f11abcfadea7f262df9a1001a217c0654e8ff050a4b38d1dc4a6b3ca` |
| projection | `69405038c92b4a679cd58205a95d003f22af7cb174a079eb574111fe5e6367ff`（有/无 manifest 相同） |
| kinematic base | `articulation_root`；legacy/VNext 七组量 exact parity |
| robot IDs | base `[4]`；supports `[24,25]`；joints `[0..36]` |
| contact-sensor IDs | supports `[6,13]`；forbidden `[14]` |
| policy I/O | observation `[1,310]`；action `[1,37]` |
| transition | frozen V2；normalized support shape `[1,2,3]` |
| reward | 42-field V2，schema `2953831c…`，15 terms，finite output |
| receipt hash | `bea2c4678e82c10e2a220ab12077f3238979b2d5f4b5527834a4f2d34c963e26` |

probe 在隔离内容覆盖目录运行，未修改服务器主 checkout；完成后无 GPU compute process。冻结主机缺少可选 Iray `libGLU.so.1` 仍产生 warning，但不影响 physics/contact/ray/reward 路径；脚本在回执 fsync 后以 one-shot process 退出，失败分支保留非零退出码。

## 4. 兼容性与身份迁移

保持不变的是 G1 的物理/策略行为与历史数据 reader：V2 reward/transition、官方 reward terms、35-cell、evaluation 数值阈值、budget/seed/acceptance 均未重定义。历史 `stair_evaluation_semantics_v6.json` 保留为旧身份。

当前语义身份有意升级到 morphology-neutral v7：公开失败名 `torso_contact` 改为 `forbidden_contact`，descriptor 使用 `kinematic_base` / `support_effector` / resolved forbidden role，不再向 Go2/六足的 LLM feedback 声称 ankle 或 torso。其 protocol hash 为 `df35aa0…`，当前 effective task protocol hash 为 `d82aae04…`；历史 v6 artifact 不被重写。

发生预期变化的是当前组合身份：support requirement 从固定 2 改为 profile-exact variable cardinality，因而 manifest/grant/resolved protocol hash 必须变化。P1 冻结 hash 仍只描述当时身份，但其验收器现将历史源码/manifest hash 视为 provenance，并继续守护 legacy G1 profiles、cell axes 与 Push dry-run；P2 验收器负责当前 Stair 角色、2/4/6 基数、ownership 和真实 runtime 语义。旧 session 只在其冻结源码和 manifest 下审计，不能用当前 Phase 2 identity 续跑。

## 5. 回归与可复算验收

- P2 snapshot verifier：PASS，snapshot receipt hash `077a3dc3a6c9113492ca90b92a8379f6cc65a2c29b45010ae45484b36dab8654`；
- verifier self-tests：7/7 PASS；冻结源码闭包只作 provenance，当前 G1 legacy 语义与 registry 扩展边界分别有正/负测试；
- 本阶段相关本地回归：100 tests PASS，2 skipped；
- server static contract tests：42/42 PASS；
- 真实 Isaac 单场景 capture：PASS，legacy/VNext kinematic-base exact parity；
- 全量仓库：1505 tests PASS，29 skipped，987.665 s，exit code 0；
- `git diff --check`：PASS；本轮没有把本机未安装的 Ruff 伪写成验收证据，语法与执行回归由上述 1505 项测试及本阶段定向测试覆盖。

当前 snapshot 文件为 `P2_MORPHOLOGY_DECOUPLING_SNAPSHOT.json`；冻结的 29 文件 product-tree hash 为 `148984daa4fa917f104bcb8a65bce348fc2c7760d472f24b11df1cb356bc4a11`。验收器复核该历史闭包自身、真实 GPU receipt 及当前 legacy G1 的角色/关节/动作/基数/ownership 语义，不再提供 `--emit-current` 重签入口，也不把后续 Phase 的合法源码演进判成 P2 回归。

## 6. 不应误读为已完成的能力

- 没有创建 Go2/六足 profile 或 EnvCfg；2/4/6 是 simulator-free contract fixtures；
- VNext reward data ABI、projection identity 与 generic builder 已完成，但现有 candidate authoring/qualified program 仍是冻结 V2。non-legacy projection 会在 program schema 不匹配时早拒绝，不会伪装可运行；
- current G1 EnvCfg/RSL-RL agent selection 属于 legacy cell execution factory，Go2 的 cell-owned factory 与 reference reward 在 Phase 5 接入；
- prompt 展示名已去 G1，但 VNext reward prompt/qualified contract 必须在首个 non-legacy cell 启用前按同一 projection hash 接线；
- C1/C2、非矩形 evaluation suite 与 rough local-surface query 尚未实施。
- `StairProtocolV1.policy_compatibility.robot_schema` 仍作为历史 G1 协议字节保留，但通用 decoder 已不再消费；其彻底退出 VNext TaskSpec/EvaluationBinding 属于 Phase 3A 的 cell-resolved 协议工作，而不是本阶段重写历史协议。

## 7. 最终判定

**Phase 2 判定为 GO。** 全仓回归无旧插件退化，P2 verifier 在最终产品字节上通过，服务器动态回执 source closure 与本地一致。下一阶段可进入 Phase 3A；策略是否改善不是 Phase 2 工程门，也没有用增加 generation、放宽阈值或伪造新机器人输入换取通过。
