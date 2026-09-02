# P3 验收报告：分阶段评测、selector 与 eval 成本收敛

> 验收日期：2026-08-17  
> 入场 HEAD：`d38385c922d3594b443508abdcd99d33e8290cf8`  
> 结论：**P3 本地实现与严格回归通过；未启动 P4、GPU、API 或服务器实验。**

## 1. 最终系统形状

```text
完整 EvalJobSpec
  ├─ public_dev        -> private phase result + create-once receipt
  ├─ public_accept     -> private phase result + create-once receipt
  └─ sealed_validation -> private phase result + create-once receipt

A/B public_dev screen + current-parent public_dev guard reference
  -> Stair selector：guard 硬上限预筛 + objective 排序
  -> frozen winner / exact-tie-or-no-eligible
  -> winner 的 selection + 三份 phase receipts
  -> typed EvaluationAssembly
  -> runner 重验 receipt 后进入既有 PublicEvalResult / sealed / acceptance
```

这不是通用 Graph 或 Agent 框架。公共层只有固定 phase、执行身份、资源绑定、receipt 和 assembly；Stair 的 score、guard、35-cell 与 sealed 语义仍全部属于 Stair 插件。

## 2. 对本轮审计问题的逐项结论

### 2.1 selector 弃权率

审计结论真实。旧实现把 `abs(paired mean delta) > 1.5×paired_SE` 当成是否给出 winner 的门；这会把“不确定性判断”错误用成资源分配门，在真实 sibling 小差距下浪费已完成的双训练和 16 个公开 seed-units。

已源头改为：

- 对 guard-eligible 候选始终按 public-dev generalization objective 排序；
- `1.5×paired_SE` 仅形成 `outside_estimated_noise_band / within_estimated_noise_band` 标签，不再决定是否弃权；
- objective 完全相同才用 public-dev guard 次序；objective 与 guard 均完全相同才 no-selection；
- 输入无效仍 fail closed，不伪造 selection。

回归覆盖了均匀 `0.001` 差和位于估计噪声带内的非零差：两者都冻结 winner，但不会被表述成 acceptance materiality 或统计证明。

### 2.2 guard 缺席

审计结论真实。screen evidence 原先保留 `anchor_score_stats`，selector 却完全不消费。

已改为：

- screen evidence 同时保存 public-dev anchor 的 per-seed macro scores；
- selector 要求同域 current-parent public-dev guard reference；
- guard reference 的 evidence 与 subject 必须不同于 A/B，P4 还须由 generation manifest 将它绑定到共同 frozen parent；
- 相对 parent 越过 acceptance protocol 已冻结 `guard_max_regression=0.05` 硬上限的候选不得成为 winner；
- 仅剩一条 guard-eligible candidate 时直接选它；两条均不合格时 no-selection。

这不是第二套 acceptance：selector 不复制 public-accept 的动态 paired guard、absolute floor、headroom 或 sealed 门。正式 winner 仍须重新通过完整验收。

### 2.3 screen 与 acceptance 阈值量级不同

事实真实，但不应把 headroom 复制到 selector。headroom回答“是否足以更新 head”，selector回答“把唯一 sealed 查询分配给谁”。

因此合同现在显式声明：

- `acceptance_headroom_role=not_evaluated_by_selector`；
- paired SE 只描述估计噪声关系；
- selector 的选择不代表 material improvement，`acceptance_authority=false` 保持不变。

### 2.4 assembled summary 缺执行证明

审计结论真实。旧 `project_assembled_eval_summary(eval_spec, mapping)` 虽校验 summary 内部身份，却不能证明 selection 已冻结或三 phase receipt真实存在。

已改为：

- Stair assembler 返回 typed `EvaluationAssembly`，绑定完整 plan、selected trial、完整 selection对象和三份 phase receipt binding；
- runner拒绝裸 mapping；
- runner重新读取三个 receipt，核验 canonical JSON、byte size、checksum、共同 trial contract、phase job spec、phase result hash、plan、checkpoint、device与 resource binding；
- checksum、job spec、result、plan 或 winner binding漂移均 fail closed。

phase-result/spec 身份校验已收敛为 `runner.specs` 中的单一 task-neutral validator，由 receipt loader 与 runner 共同复用，没有留下两套字段表。

### 2.5 generation final-confirmation 二源

审计结论真实。未接线的 generation 文件重述了现有 v1 的部分语义，却缺少 `single_use`、同 cohort pairing/order、paired statistics 和 `acceptance_authority=false`。

最小源修是删除该无生产消费者的副本及其测试，不继续补强第二套合同。主计划现已明确：待 P4/P5 的 generation session、continuation chain 和 replay audit真实存在后，只在现有 Stair final-confirmation 模块中版本化扩展并复用冻结块。

### 2.6 弱边界测试

审计结论真实。按 import 模块名搜索 `sealed/private` 子串不能证明数据流边界，已删除。

现在的有效边界是：

- selector 参数必须是 `StairPublicDevScreenEvidence`；
- projector 强制 `phase=public_dev`；
- public-accept phase 输入、普通 mapping 和错误 phase 都被行为测试拒绝；
- formal assembler重放完整 selection并核对 winner evidence hash。

### 2.7 device 静默排除

审计结论真实。跨卡比较必须允许 `cuda:0/cuda:1` 不同，但不能让差异从证据中消失。

已改为 screen/phase receipt 显式记录：

- `execution_device`；
- P1 物理 GPU `resource_binding_hash`。

selection contract 将它们与 trial/plan/checkpoint identity一起声明为允许差异；科学 comparison domain 仍要求 task、协议、training、capability 和软件执行环境相同。

## 3. 其余 P3 交付

- 固定 `EvaluationPhase` 与 private `EvaluationPhaseResult`；
- `EvalPhaseJobSpec` 始终引用完整 `EvalJobSpec`；
- backend/runner支持单 split 执行并复用 P1 attempt、GPU lease和进程治理；
- `TrialReceiptJournal` 提供 create-once phase receipt；
- 单体完整 eval 和三 phase assembly 共用 Stair summary builder；
- task-owned成本上限仍为 `23 seed-units = 6,440 episodes`；
- Push/Neutral 默认不获得 Stair phased capability；
- width=1 round-v7、ledger replay、20-step stable success、acceptance协议和现有 final-confirmation v1均未改。

## 4. 验收证据

项目解释器：`D:\humanoid_agent\true_RL_Agent\.venv\Scripts\python.exe`

| 验收 | 结果 |
|---|---:|
| P3/runner/backend/receipt/旧 round 聚焦回归 | `190 tests; OK (skipped=2)` |
| acceptance/replay/现有 final-confirmation 兼容回归 | `63 tests; OK` |
| 全仓最终回归 | `1209 tests in 238.763s; OK (skipped=21)` |
| `compileall` | 通过 |
| `git diff --check` | 通过 |
| 旧 selector/noise-gate、弱终局副本、虚假 import 扫描残留 | 0 |

全量日志中的 `close failed`、summary publication failed 和 environment drift 文本来自既有故障注入用例；最终退出为成功。

## 5. 当前边界

已实现并由本地测试验证：phase spec/argv/runner/parser/receipt、Stair selector、guard reference、typed assembly、runner receipt重验、科学 summary等价和成本合同。

仍待 P4：A→B authoring、双 lane coordinator、selection create-once持久化、双 public-accept调度、winner-only sealed、generation ledger和单次 head transition。

仍待 P6 真机验证：双 GPU并行占用、CPU/RAM竞争、真实 Isaac phase/monolithic等价、墙钟收益和真实 selector/public-accept方向统计。

## 6. 关键文件指纹

| 文件 | SHA-256 |
|---|---|
| `src/true_rl_agent/evaluation/phases.py` | `396253956542f3ed609e5ad59a5ba6dec53a92f00aac0929f19589fa10e2393e` |
| `src/true_rl_agent/runner/specs.py` | `1c16c2c4caa1c17b7d3f51c6f89515d76600c7bc25c483d2ce448570efa2d578` |
| `src/true_rl_agent/runner/subprocess_runner.py` | `486e286c0916c9da1767a926e9858082a6cd66f518cd613ca70ea3a54b619564` |
| `src/true_rl_agent/agent_core/trial_receipts.py` | `e4ec147ad8d3a630c585eebc342c204cb131965b3d8c0bb5a78137963ffbc9ad` |
| `src/true_rl_agent/tasks/stair_climbing/phased_evaluation.py` | `1a4066671070ada6021f6819ebc344e7b987b7bf6966d6b43c123b4d0b702a32` |
| `src/true_rl_agent/tasks/stair_climbing/selection.py` | `49f422ceed3b063922f7b45d6072becde6fe5b3ae242e8d750c56c934df5df3a` |
| `src/true_rl_agent/tasks/stair_climbing/phased_cost.py` | `69e7dcaeee844edd9fc92489142553cd31432409ab0ac26bd36080e2db4d543e` |
| `scripts/isaaclab_stair_rsl_rl_eval.py` | `2e046c29fc4484b01370323b3773af31605602b46722eb1280dde79fe61e3f8c` |

这些指纹对应验收时工作树。P3 尚未提交或同步服务器；未经用户验收不进入 P4，也不启动实验。
