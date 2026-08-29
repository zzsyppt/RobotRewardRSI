# Phase 3R 验收报告

> 日期：2026-08-28
> 判定：**GO（工程与协议边界）**
> protocol 状态：**draft**
> 下一门：Phase 3B C2 roughness OOD 与 failure-feedback 校准

## 1. 判定摘要

Phase 3R 已把 G1-37 × Stair 从“C0/C1 各自形成系统身份”收敛为一个 robot-task cell。统一训练分布、目标函数、三个 ID/OOD guards、VNext reward ABI、身份强制与 fail-closed 生命周期均已接通；退役独立 C1 套件和不可达开发分支已删除。

本 GO 不表示策略效果通过、reference 已资格化或 LLM 已观察到改善。协议保持 `draft`，Phase 3B 补齐 C2 并重签最终身份前，正式 train/eval 与 candidate LOOP 均不得启动。

## 2. 严格验收

| 门 | 结果 | 证据 |
|---|---|---|
| P3R-G1 身份与启动 | PASS | 唯一 `stair.g1_37dof.unified_v1`；manifest/TrainingBindingV2/EvaluationBinding/child/checkpoint identity 闭合；live 与 formal generation 对 draft 均 fail-closed |
| P3R-G2 训练支持与课程 | PASS | 10×10 terrain、C0/C1 50/50、正式 2048/2048、初始 level 5/1、goal-progress up/down；训练 exposure 不含 C2；训练/评测 terrain 共有常量只有 training profile 一个来源，forward command 为可机读 `[0.0, 0.8] m/s` |
| P3R-G3 评测与 acceptance | PASS | 每 formal split 7 strata × 3 cells × 4 episodes = 84 env/seed；三组共 63 道互不重合几何，正式总成本仍为 1260 episodes；四个 OOD 层均使用相邻低/中/高距离带，cohort 平均差最大 4.714 mm；phase OOD 由展开序列判真；`J_primary=(2*J_OOD+J_ID)/3` 与三个 guards 可复算；generic v6 无任务术语 |
| P3R-G4 ABI 与历史边界 | PASS | 统一路径只用 VNext + TrainingBindingV2；官方 15-term reference 由 RobotProfile joint groups 解析；legacy C0/Push 回归通过，旧 candidate 不获新身份资格 |
| P3R-G5 所有权与清理 | PASS | active `src/scripts` 中 `is_irregular_cell` 为 0；10 个旧独立 C1 路径不存在；旧 cell/套件标识在活动树为 0；任务特殊语义未进入 generic acceptance |
| P3R-G6 定向与回归 | PASS | 本轮 76 项受影响范围测试通过、3 skipped，评分/资格化补充集 58 项通过；本地全量 1525 项通过、29 skipped；服务器同产品 SHA 受影响集 123 项通过、1 skipped（492.744 s），既有 P3R 服务器全量 1525 项通过、3 skipped（902.470 s）；真实 Isaac smoke 与三组代表性评测 mesh 探针通过 |
| P3R-G7 成本与 GO | PASS | 只执行 128-step draft smoke，预算占比 `6.510416666666666e-7`；未运行旧 C1 完整资格化、完整 reference 或 candidate |

## 3. 真实运行时回执

`PHASE3R_RUNTIME_SMOKE_RECEIPT.json` 由服务器真实 headless Isaac/RSL-RL 执行生成，不是 mock：

- receipt hash：`a1589b721ce2a036296bec8f0ce96b1a9d405e4388c901453629ef859d18692a`；
- resolved manifest hash：`0b5666a4aadce985babb46b4664e8e7f8ff8e6e0738ddb2d1100cacc97f5e936`；
- resolved task protocol hash：`a1600975892e5c44484d246e8dd7db6ae26efaf6aa31adb73f898c2bfc48b463`；
- training profile hash：`d222a1ddc2ed52ed6feae43d8413b6364576880ac4febbc09e6ac7a9122028d1`；
- evaluation suite hash：`7fe5911b8b2fa235d2d4d080b0a82d8c4d637f123388ac6bb9bd4fc0a15e4152`；
- C0/C1 运行分配为 8/8，reward term 数为 15，curriculum probe 为 PASS；public_dev/public_accept/sealed_validation 各一块代表性 phase-OOD mesh 均真实生成成功，且三者 geometry digest 不同；
- eligibility 中 reference qualification、current-best、candidate acceptance 全部为 false。

128 steps 只验证 terrain、reward、PPO、课程和身份链在真实运行时非结构性断开；不据此推断成功率、泛化或学习趋势。

## 4. 可复算快照

执行：

```powershell
$env:PYTHONPATH='src'
python PLAN/Variant_construction/P3R/verify_phase3r.py
```

当前冻结值：

- snapshot receipt hash：`d73ed3b32d5964bce62077a5258141ff998bd9d0a0d1fd50c8f6bb4184742164`；
- product tree hash：`fd1a954100eb7ab582499ae367ad0fd70484dc00415d30ce092b2227f6a8c47b`；
- product closure：53 个关键文件；runtime source closure：21 个文件；
- hash 模式：normalized LF SHA-256，提交/检出不会因 CRLF 改写而失效。

验收器同时重算 protocol、manifest、bindings、训练分配、suite strata、score、named guards、reference manifest、draft 正式入口拒绝、退役路径与 generic ownership 边界，并校验真实 smoke 的源文件闭包。

它还机械检查 formal cohort 内/间 geometry digest 不重合、geometry identity 不含 cell/cohort 标签、phase OOD 相对训练相位确实改变 riser/tread 序列、训练/评测共享地形常量不分叉，以及训练 forward 上限与评测命令一致。

独立难度审计发现初版 63 题虽几何隔离，却让 sealed 在 height/tread/joint/C1-frontier 四层上系统性更远，因而会把正常难度差混进 `max_public_to_sealed_score_drop=0.05`。源修复只重排 task-owned 评测 JSON 的几何数值：public_dev 使用 `25/50/75 mm`，public_accept 使用 `20/45/80 mm`，sealed 使用 `30/55/70 mm`；标量层 cohort 均值最大差为 3.334 mm，joint 层最大差为 4.714 mm。三套卷同时对齐 tread 外推方向、C0-ID 平均高度、C1-ID 与 phase-OOD 边际集合，63 个 geometry digest 仍全部唯一。无序 phase 类别不虚构标量距离，而以平衡不完全区组分配：`p2/p4/p6` 各跨两个 cohort 出现，三套卷都有 alternating/front/back-loaded；三套卷平均 episode horizon 为 587.429–589.524 control steps。reference 支持性判断实际按 OOD family 聚合，而不是单题直接剔除；Phase 4 随后把双边区间收敛为 floor-only 规则，该变更不改变“public_accept 不得系统性偏易”的结论。

## 5. 清理结论

已删除：

- 5 份 `stair_g1_37dof_irregular_*` 独立 cell 配置；
- `irregular_training.py`、`irregular_calibration.py`、`irregular_reference_gate.py`；
- `g1_irregular_v1` 奖励源名称与 `test_stair_phase3a_contract.py` 开发套件；
- prompt/selection/public projection/phased evaluation/reference 路径中的旧 C1 不可达分支及无人调用 score-view。

历史 P3A 文档和回执保留，因为它们是根因与迁移证据；它们不进入新协议 hash，也不能作为当前 PASS 复用。

完工后的第二轮只读清理审计没有发现新的可删对象：活动 `src/scripts/configs/tests` 中不存在旧 Phase 3A cell、旧配置名或旧入口引用；5 份 unified 配置各有真实运行时加载者；6 个 P3R 新模块的公开符号均有多处产品/验收引用。`is_irregular_cell` 仅保留在一条禁止其回流的负向测试中，不进入产品代码。继续删除将开始损失统一 C1 primitive、身份复算或真实闭环覆盖，因此未制造第二轮“为清理而清理”的提交。

服务器生产形态三代测试还验证了 JSON 往返后的 ledger 真实性。审计过程中从源头修正了 ordered evidence 对 JSON 对象顺序的依赖、fixture 对旧 G1 足端字段和饱和分数的依赖、sealed task evidence 形状，以及 generic decision/replay 对命名 guard 的 tuple/list 表示分歧；没有新增 P3R 特判或兼容旁路。

## 6. 非声明与 Phase 3B 入口

Phase 3R 没有证明：

- official reference 在统一分布上已收敛；
- 任一 OOD family 已达到 Phase 4 reference 支持地板；
- C2 geometry、scanner observability 或 failure taxonomy 已可靠；
- LLM LOOP 已运行或 reward candidate 已改善策略。

Phase 3B 必须在同一 cell 内加入 C2，并继续保证 PPO exposure 为零；完成 C2 量级/分类漂移对照和最终 suite 身份后，才能考虑把 protocol 从 `draft` 转为 `active`。

当前 `sealed_validation` 已是 geometry-held-out，而不只是 seed-held-out：它的 21 道几何与 public_dev/public_accept 均不重合，且 prompt 只投影 public_dev 结果。`final_holdout` 仍是 Phase 7 才新增的一次性研究套卷，不把尚不存在的配置写成当前能力。
