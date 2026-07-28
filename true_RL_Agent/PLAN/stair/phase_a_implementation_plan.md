# 楼梯奖励自进化阶段 A 详细实施计划

## 1. 阶段目标

阶段 A 只完成“系统能够无歧义地表达 reference program、固定训练绑定、任务权限快照和 reference Genesis”的共享合同，以及 task-owned `StairProtocolV2 draft`。

本阶段不实现生产 Stair 环境、R0 训练、完整奖励 runtime 或正式 reference Genesis 创建。未达到后续阶段所需运行条件时必须 fail closed，不能回退到 bootstrap、baseline 或 candidate 语义。

当前基线：

- Phase1B 工作区全量测试：`574 passed, 4 skipped, 415 subtests passed`；
- production composition registry 仍只有 Push Recovery；
- Session Genesis 已有 bootstrap/baseline 两类 anchor；
- candidate ledger 从第一轮 candidate 开始；
- evaluation subject 只有 candidate/baseline；
- train job 仍以独立 `candidate_id` 表达身份；
- prompt capability registry 是共享闭合 registry，但缺少 task-owned profile snapshot；
- V1 Stair protocol 仍是 Phase0 warm-start 证据，不能原地改写。

---

## 2. 核心设计决定

### 2.1 Genesis 选择使用闭合策略，不增加布尔开关

删除 `TaskSpec.requires_baseline_genesis`，替换为：

```text
genesis_anchor_policy:
  policy_initialization_kind -> bootstrap | policy_baseline | reference_program
```

原因：Push Recovery 同一个 task 同时支持：

- random -> bootstrap；
- warm_start -> policy_baseline。

因此单个静态 `genesis_anchor_kind` 仍不足以表达现有能力。`TaskSpec` 保存闭合策略，`ResolvedTaskComposition` 只保存本次初始化解析出的唯一 `genesis_anchor_kind`，并严格验证两者一致。

Stair V2 draft 只允许：

```text
random -> reference_program
```

reference task 在 reference Genesis materializer 尚未实现时必须拒绝创建，禁止隐式生成 bootstrap Genesis。

### 2.2 Subject 只有一套结构化身份

将 `EvaluationSubject` 升级并重命名为 `ExperimentSubject`：

```text
schema_version = true_rl_agent.experiment.subject.v2
kind = baseline | candidate | reference
id = bounded ASCII id
```

`TrainJobSpec`、`EvalJobSpec`、`BaselineEvalJobSpec` 和 summary identity 全部直接保存 subject。删除 job 中独立的 `candidate_id`/`baseline_id` 身份字段，不保留可序列化兼容别名。

`JobIdentity` 统一 train/eval identity；reference/baseline 必须携带 canonical `subject_manifest_hash`，candidate 首版不要求 manifest hash。

### 2.3 Task-owned capability profile 是 registry 的不可变选择快照

复用现有 `CandidateAuthoringCapabilityRegistry` 的 capability IDs，新增 task-neutral：

```text
CandidateCapabilityProfile
  profile_id
  capability_ids             # 有序、唯一、闭合
  profile_hash               # canonical JSON hash
```

规则：

- profile 不复制 capability 的描述和 runtime 语义；
- profile 只能选择 registry 已知 ID；
- TaskSpec 持有已验证 profile；
- composition runtime snapshot、train job、eval job、seal 和 current-best 绑定 profile hash；
- Push profile 保持当前有效的 reward term、diagnostic term、reward-weight schedule；
- Stair draft profile 只选择 `complete_reward_program`；
- `complete_reward_program` 在阶段 C runtime/verifier 未完成前保持非 prompt-visible、非 writable、非 effective；
- 未选择 capability 不能因共享默认 registry 而进入该任务 prompt。

### 2.4 TrainingBinding 绑定公平训练事实

新增 task-neutral `TrainingBinding`：

- initialization mode；
- initialization checkpoint hash（仅 warm-start）；
- 实际执行的 policy architecture canonical hash；
- 实际执行的 policy/PPO combined configuration hash；
- initialization seed schedule hash；
- 实际执行的 PPO/optimizer canonical hash；
- fixed budget steps；
- private training seed resolver hash；
- capability profile hash；
- canonical binding hash。

`PolicyInitialization` 仍是 checkpoint 执行对象；`TrainingBinding` 是不可变比较/lineage 摘要。`TrainJobSpec` 同时持有两者并严格交叉校验，不允许不一致。

RSL-RL backend 将 task-owned canonical policy/algorithm 配置作为结构化 argv 传给训练进程，在 subprocess 启动前与 binding 逐项校验；训练进程应用该配置后再从实际 `agent_cfg` 回显 canonical hash。Push 固定 seed 同样由 backend 显式写入 argv 和 binding，训练脚本不保留默认 seed。

### 2.5 ReferenceProgramManifest 只描述可信源码

新增 task-neutral `ReferenceProgramManifest`，canonical 源码布局固定为：

```text
training_code/components.py
training_code.components:build_training_components
```

manifest 绑定：

- reference ID、task ID；
- source URI、source byte checksum、source byte count；
- authorship origin；
- upstream project/revision/source path；
- capability profile hash；
- canonical manifest hash。

manifest 提供严格 exact-field loader 和 source-byte verification。共享层不解释楼梯 reward 输入、七桶或能力门。

### 2.6 ReferenceProgramGenesisAnchor 只新增第三类信任根

Session Genesis schema 升级为 V2，支持：

- bootstrap anchor；
- policy baseline anchor；
- reference program anchor。

`ReferenceProgramGenesisAnchor` 绑定：

- reference manifest；
- TrainingBinding；
- runner-sealed checkpoint；
- public result、trusted seal、private artifact；
- capability gate 的通用 pass/evidence/protocol 绑定。

阶段 A 只实现：

- exact serialization/deserialization；
- contextual validation；
- 从合法 reference anchor 派生 initial current-best；
- load/replay；
- reference task 在 Genesis 缺失时 fail closed。

阶段 A 不实现 R0 train+eval 和 reference anchor materialization。正式创建流程留给阶段 E。

### 2.7 Current-best 显式区分 reference 与 accepted

扩展 `CurrentBestEvalStats`：

- mode：`bootstrap | reference | accepted | incomplete`；
- anchor kind：`bootstrap | baseline | reference | candidate`；
- program source checksum；
- program manifest hash；
- training binding hash；
- capability profile hash。

规则：

- reference anchor -> `mode=reference`；
- baseline 继续是已评估 policy best，但不是程序源码 reference；
- 第一条 accepted candidate 才进入 `mode=accepted`；
- candidate decision record 必须绑定 source/training/profile hashes，replay 才能推进 source head；
- bootstrap/baseline 不伪造 program source；
- source catalog 增加 `reference_current_best`，只能从 Genesis reference artifact 解析完整源码。

### 2.8 V1 不变，V2 draft 为 task-owned overlay

新增：

- `configs/tasks/stair_protocol_v2_draft.json`；
- `tasks/stair_climbing/protocol_v2.py`。

V2 draft 通过 canonical hash 绑定不可变 V1 task protocol，并只声明 V2 改动与新增合同，避免复制整份 V1 成为第二来源。

draft 固定：

- Genesis：random -> reference_program；
- canonical complete reward source/entrypoint；
- Stair capability profile；
- pilot 允许预算：49,152,000 / 98,304,000 / 147,456,000 steps；
- reference capability gate：前三个低阶桶、5 个 public-accept seed 完整覆盖，低阶宏平均成功率不少于 0.05，低阶宏平均最大进度不少于 0.50；
- pilot 最大预算不得超过 147,456,000；
- pilot 不得放宽 capability floor；
- V1 七桶、episode outcome 和评估统计定义按 base protocol hash 继承。

loader 必须 exact-field、严格类型、严格关系验证；修改内容并重算 self hash 仍不能绕过冻结语义。

---

## 3. 实施顺序

### A1. 通用不可变合同

涉及文件：

- `evaluation/subject.py`；
- 新增 `candidates/capability_profile.py`；
- 新增 `training_programs.py`；
- 新增 `training_binding.py`；
- `tasks/base.py`。

先完成纯数据模型、canonical hash、exact loader 和负向单测，不改运行路径。

### A2. Composition 与 job identity 收敛

涉及文件：

- `composition/base.py`；
- `tasks/push_recovery/plugin.py`；
- `tasks/push_recovery/composition.py`；
- `runner/job_identity.py`；
- `runner/specs.py`；
- `backends/base.py`；
- `backends/isaac_lab_rsl_rl.py`；
- train/eval scripts；
- dry-run/fake runners。

完成 subject、profile、training binding 从 composition 到 argv/summary/checkpoint/eval spec 的单链传播。移除独立 candidate/baseline job identity 字段。

### A3. Seal、public result 与 current-best lineage

涉及文件：

- `runner/subprocess_runner.py`；
- `safety/eval_seal.py`；
- `evaluation/sealed_validation.py`；
- `evaluation/current_best.py`；
- `evaluation/acceptance_replay.py`；
- `evaluation/decision_record_contract.py`；
- `agent_core/round.py`。

runner 注入并验证 profile/training/source 绑定；candidate ledger 继续保持同样的轮次与决策结构，但 accepted record 缺少新 lineage 字段时 fail closed。

### A4. Reference Genesis 与 source projection

涉及文件：

- `agent_core/session_genesis.py`；
- `agent_core/live_loop.py`；
- `prompting/source_catalog.py`；
- memory/context 投影。

Session Genesis V2 解析三类 anchor。reference Genesis 可加载、验证并派生 initial current-best；缺失时拒绝启动。candidate ledger 不新增 sequence 0。

### A5. StairProtocolV2 draft

涉及文件：

- `configs/tasks/stair_protocol_v2_draft.json`；
- `tasks/stair_climbing/protocol_v2.py`；
- task-owned unit tests。

只在 Stair 包解释低阶桶、进度、能力门和 pilot 预算，不把这些名称加入共享合同。

### A6. 清理与全量验收

删除迁移后无调用者的：

- `EvaluationJobIdentity`；
- train/eval job 的独立 candidate/baseline identity；
- `requires_baseline_genesis`；
- 仅为旧字段存在的兼容转换和测试 fixture。

不保留双写、deprecated alias 或静默迁移。

---

## 4. 严格验收

### 4.1 身份与 hash

- reference 合同、已有可信 eval artifact、Genesis、current-best、source projection 与 replay 身份完全一致；
- 任意 subject kind/id、manifest hash、profile hash、training binding hash 漂移均 fail closed；
- Phase A 不宣称 reference train/eval argv 执行链完成；该链由阶段 E 的 materializer 验收；
- serialized train/eval job 无 `candidate_id`/`baseline_id` 第二身份；
- unknown subject kind/schema/field 被拒绝。

### 4.2 Genesis 与 replay

- Push random 创建 bootstrap Genesis；
- Push warm-start 创建 policy-baseline Genesis；
- reference task 没有已验证 reference anchor 时拒绝，不生成 bootstrap；
- 合法 reference anchor 恢复 `mode=reference` initial current-best；
- candidate ledger 第一条仍是 round 1 candidate；
- accepted candidate 才把 mode/source head 推进为 accepted/candidate；
- 修改 Genesis/source/training/profile 任一 hash 后 load/replay 拒绝。

### 4.3 Capability profile

- profile ID 重复、未知 capability、乱序/重复 IDs、错误 self hash 被拒绝；
- Push prompt-visible 能力不退化；
- Stair draft 只选择 complete reward，课程/diagnostic/schedule 不在该 profile；
- complete reward runtime 未 ready 时不进入 prompt。

### 4.4 Draft V2

- V1 文件与 hash 不变；
- draft V2 base hash 与当前 V1 精确一致；
- 修改七桶继承、能力门、预算集合、Genesis policy、canonical entrypoint 后即使重算 self hash仍被语义 validator 拒绝；
- draft hash 可重复加载并得到同一 resolved hash；
- 共享包搜索不到 Stair bin/reason/capability-floor 字符串。

### 4.5 旧功能回归

- Push materialize、dry-run、run-round、live-loop、baseline eval-only 全部通过；
- checkpoint preflight、structured argv、trusted seal 安全测试全部通过；
- Phase0 Stair bundle/report/probe tests 不退化；
- 项目标准 `unittest discover` 无 import error/warning；Phase A 测试不得要求项目未声明的 pytest 依赖；
- Ruff E9/F 对全仓通过，Ruff I 对本阶段变更集通过；compileall、git diff --check 通过。全仓历史文件的 import-only 格式债不纳入本阶段修改，避免制造无关 diff。

---

## 5. 本阶段明确不做

- 不注册生产 Stair provider；
- 不创建生产 Stair TaskPlugin/config；
- 不运行 R0 pilot 或正式训练；
- 不实现 pre-reset reward dispatcher 和新 RewardInputs；
- 不实现 reference train/eval materializer；
- 不改 candidate acceptance 算法和 round 编号；
- 不开放 curriculum 等未来权限；
- 不把 Phase0 checkpoint 变成 reference/current-best；
- 不保留已被新单源合同替代且无生产调用者的兼容代码。

---

## 6. 实施结果（2026-07-27）

**状态：阶段 A 已完成。**

实际落地与本计划一致，并在行为测试中额外发现、从源头修复了三处 reference 信任链问题：

1. reference 私有证据现在保存 canonical `eval_job`，Genesis 加载时从该 job、task invariant、checkpoint safety report 重建 trusted eval seal；仅重算 artifact/Genesis hash 不能伪造 eval job 身份。
2. acceptance decision contract 与 ledger replay 的 delta 比较原先仍有 `accepted-only` 分支，现统一允许完整的 `reference` current-best；首条 candidate 记录可从 `mode=reference` 推进到 `mode=accepted`，不新增 sequence 0。
3. task capability profile 没有已开放能力时，prompt component contract 明确为 `unavailable`；旧的 RewardTerm、DiagnosticTerm、ScheduleTerm 及其 verifier/example/loop 语义不会从共享默认 registry 泄漏进该任务 prompt。

本轮边界审计又完成五项收敛：

1. `TrainingBinding` 现在绑定实际执行的 policy、PPO/optimizer 和 combined agent configuration hash；task-owned 配置通过 structured argv 进入训练进程，Runner 在启动前校验，训练进程应用后回显同一组 hash。
2. train/eval seed 不再依赖脚本默认值；训练 seed 的 argv 值必须与 seed schedule hash 一致。
3. candidate identity resolver 复用正式 manifest loader；unsupported schema、重复/未知字段不能再生成可物化 job。
4. Phase A 的 reference 验收明确收窄为合同、可信 artifact、Genesis 与 replay；reference train/eval materializer 仍属于阶段 E。
5. Stair V2 只保留 canonical JSON、一个可信 draft hash 锚和结构/关系校验；capability ID 从共享 registry 解析，计划文档不保存派生 hash。

冻结结果只由 canonical JSON 与代码中的单一可信 draft hash 锚承载；计划文档不复制派生 hash。

验收结果：

- 标准入口：`python -m unittest discover -s tests -p "test*.py"`，结果为 `595 tests passed, 4 skipped`；
- reference Genesis、source projection、reference -> accepted replay、V2 防篡改和 capability 隔离均有动态测试；
- 变更集 Ruff `E9/F/I` 通过，全仓 Ruff `E9/F` 通过；
- `compileall`、`git diff --check` 通过；
- 共享源码扫描未发现 Stair bin、能力阈值、失败原因或 V2 task ID 泄漏。
- 本地 bundled Python 未安装 pytest；Phase A 新测试已改为 `unittest`，不再依赖 pytest 收集或外部 `PYTHONPATH`。

阶段边界保持不变：尚未注册生产 Stair provider，未实现完整奖励 runtime、R0 materializer、reference train/eval argv 执行链，也未运行 GPU pilot；这些工作从阶段 B 开始，不能由阶段 A 的 draft 或手工 Genesis 代替。
