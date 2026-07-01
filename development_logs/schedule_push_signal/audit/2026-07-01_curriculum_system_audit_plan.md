# Phase 8E 课程系统分阶段审计计划

## 目的

本计划用于系统审计当前 Phase 8E 课程系统，目标是尽量全面地发现设计、接线、安全边界、运行时行为、反馈闭环和可复现性中的潜在问题。审计按六个必做阶段推进，最后保留一个可选的行为效果阶段。

本轮审计优先判断“系统是否可信、边界是否正确、证据是否充分”，不把训练性能提升作为前置成功标准。

## 总体原则

- 先审安全边界，再审行为效果。
- 先审 prompt/provider 输入输出，再审 runner 编译和训练执行。
- 所有结论必须能落到当前代码、配置、测试、日志或产物。
- 区分 LLM 可提议内容、runner 可执行内容、trusted eval 可裁决内容。
- 区分 public feedback、public accept、hidden/private eval 和 runner 内部状态。
- 审计应覆盖正向路径、拒绝路径、脱敏路径、失败反馈路径和 accepted snapshot 路径。

## Phase 1: Prompt 与 Capability Registry 审计

### 目标

确认 LLM 在课程设计阶段看到的 prompt-facing 信息准确、足够、不过度暴露，并且 capability registry 与当前真实执行能力一致。

### 做什么

- 审计 `public_curriculum_evidence_bundle` 的 public-only 边界。
- 审计 `curriculum_design_contract` 是否足以约束 LLM 输出结构化课程。
- 审计固定 capability registry 是否与当前实现一致。
- 审计 prompt 是否隐藏未接通或不应广告的控制面。
- 审计 `RewardInputs`、schedule key、candidate controls 的 prompt-facing projection 是否真实反映运行时能力。
- 审计 prompt 历史摘要是否只包含下一轮 LLM 允许使用的 public feedback。

### 产出

一份 Prompt/Registry 审计结论，明确 LLM 现在能看见什么、能提出什么、不能提出什么，以及是否存在 prompt-facing contract 与真实执行不一致的问题。

## Phase 2: Provider 与 Candidate Artifact 审计

### 目标

确认 LLM 输出在进入训练前必须经过完整的 candidate package、课程 artifact、静态/动态 gate 和错误脱敏检查。

### 做什么

- 审计 candidate package 文件边界和路径安全。
- 审计 `operation_diff.json.curriculum_design` 与 `diagnostics.json.curriculum_hypothesis` 的必需性。
- 审计退化课程、单阶段课程、reward-only attribution、非法 capability、错误 final consolidation 是否会被拒绝。
- 审计 provider inner loop 的 source request、JSON repair、static repair 和 selected attempt 行为。
- 审计 provider 错误、repair feedback、verifier feedback 是否会泄露 API key、本地路径、hidden token、private eval 信息。
- 审计 provider summary 中用于后续 lineage 的 prompt payload 指针是否完整且安全。

### 产出

一份 Provider/Candidate 审计结论，明确坏 proposal 是否会在训练前被挡住，失败反馈是否足够帮助下一轮修复且不会泄露受保护信息。

## Phase 3: Curriculum Compiler 审计

### 目标

确认 LLM 的课程设计不会被原样信任，而是被 runner-owned compiler 转换成受控、可审计、训练侧专用的 effective curriculum。

### 做什么

- 审计 compiler 对 stage 数量、progress coverage、final consolidation、capability 使用和 attribution 的校验。
- 审计 raw curriculum 到 effective curriculum 的转换是否符合 runner 约束。
- 审计 final stage 是否强制回到 runner-owned target push 和 target robustness。
- 审计 push payload、robustness payload、scale、clamp 和 target distribution 的生成逻辑。
- 审计 compiled phase1/phase2 config 是否只改变训练侧配置。
- 审计 eval config、seed、budget、success metric、checkpoint selector 是否保持不可被 candidate/LLM 修改。
- 审计 compiler reject 路径是否会阻止训练并写出可追踪 summary。

### 产出

一份 Compiler 审计结论，明确 LLM 设计如何被翻译成 runner-owned effective curriculum，以及是否存在越权、缺少 clamp、错误回到目标分布或审计产物不完整的问题。

## Phase 4: Training Runtime、Reward Adapter 与 Push Signal 审计

### 目标

确认 compiled curriculum 真实进入训练路径，同时 candidate reward 代码仍只通过受限 `RewardInputs` 和 guarded components 影响训练。

### 做什么

- 审计训练脚本是否读取 compiled config 中的 `runner_curriculum`。
- 审计 curriculum runtime 是否按训练 progress 选择 stage 并在 env step 前应用 runner-owned push schedule。
- 审计 runtime wrapper 与 candidate reward wrapper 的顺序是否正确。
- 审计 candidate `RewardTerm` 是否经过 shape、finite、clip、weight scale guard 后才加入 reward。
- 审计 `ScheduleTerm` 是否只有当前真实有效的 key 影响训练。
- 审计 `LossTerm` 是否仍未插入 PPO optimizer。
- 审计 `DiagnosticTerm` 是否只作为 public numeric summary。
- 审计 `push_active` 和 `push_vec_public` 是否来自 runner-owned push event instrumentation，并在无事件或失败时安全 fallback。
- 审计 runtime summary 是否足以证明 stage transition、push application、reward contribution 和 input provenance。

### 产出

一份 Training Runtime 审计结论，明确课程是否真的执行、candidate 是否仍无 env/runner/optimizer/eval 控制权、push signal 是否真实且可追踪。

## Phase 5: Full-Round、隔离训练与 Trusted Eval 边界审计

### 目标

确认课程只影响训练侧，accept/reject 仍由 trusted eval、固定预算、checkpoint safety 和 public result gate 裁决。

### 做什么

- 审计 full-round 执行顺序是否固定为 provider gate、compiler、isolated training、checkpoint resolver、public rollout、linked decision。
- 审计 compiler reject、provider reject、training failure、trusted eval failure 的失败分类和中止位置。
- 审计 fixed budget、num envs、iterations、steps per iteration、timeout 和 runner config 是否不可被 candidate 修改。
- 审计训练子进程与 trusted eval 之间的边界，尤其是 trusted eval 不 import candidate code。
- 审计 checkpoint safety 是否由 trusted side 判断，不信任 child-reported safety。
- 审计 public dev 与 public accept 的用途、seed visibility 和 prompt projection 边界。
- 审计 motion quality、budget accounting、OOD/not-run gates 等 public gate 是否保持原有语义。

### 产出

一份 Full-Round/Trusted Eval 审计结论，明确课程系统是否破坏了原有固定预算、checkpoint safety、eval 独立性或 accept/reject 边界。

## Phase 6: Ledger、Lineage、Immutable Context 与 Next-Round Feedback 审计

### 目标

确认 accepted curriculum 能被复现和追责，下一轮 LLM 只能看到 public-safe 的 lineage 与 attribution feedback。

### 做什么

- 审计 `curriculum_lineage.json` 是否记录 public summary、raw curriculum、effective curriculum、clamps 和 attribution。
- 审计 accepted recipe snapshot 是否保存 runner curriculum snapshot。
- 审计 immutable context hash 是否覆盖 capability registry、prompt-facing contract、compiler、public evidence、training config 和 eval config。
- 审计 ledger record 与 prompt projection 是否只暴露 public-safe lineage。
- 审计 `curriculum_attribution_summary` 是否正确统计 accepted curriculum lineage、pure knob streak、gain attribution 和 latest accepted attribution。
- 审计 accepted snapshot、current best recipe、ledger inheritance 在多轮 autoresearch 中是否一致。
- 审计 lineage 缺失、summary 缺失、snapshot 写入失败时是否有可发现的失败信号。

### 产出

一份 Ledger/Lineage 审计结论，明确课程改动是否可追踪、可复现、可投影到下一轮 prompt，且不会泄露 runner-private 或 hidden eval 信息。

## Phase 7: 可选行为与效果审计

### 目标

在前六阶段确认系统可信之后，再审计课程是否真的按预期改变训练行为，以及是否有早期效果或退化信号。

### 做什么

- 运行小预算 smoke，确认 stage transition、push schedule application、push signal provenance 和 runtime summary。
- 对比有无 compiled curriculum 的训练侧行为差异。
- 对比有无 reward-only push signal 的 reward shaping 行为差异。
- 检查 final consolidation 是否实际回到 target distribution。
- 检查 public dev success、motion quality、action magnitude、fall breakdown 是否出现明显退化。
- 检查课程是否存在“看似通过但实际长期降低难度”的风险。

### 产出

一份行为效果审计结论，明确当前课程系统是否不仅安全接通，而且在小预算或受控对比中表现出合理训练行为。

## 建议执行顺序

1. Phase 1: Prompt 与 Capability Registry
2. Phase 2: Provider 与 Candidate Artifact
3. Phase 3: Curriculum Compiler
4. Phase 4: Training Runtime、Reward Adapter 与 Push Signal
5. Phase 5: Full-Round、隔离训练与 Trusted Eval
6. Phase 6: Ledger、Lineage、Immutable Context 与 Next-Round Feedback
7. Phase 7: 行为与效果审计，仅在前六阶段没有 P0/P1 问题后执行

## 每阶段审计记录模板

每个阶段完成后建议写一份独立审计记录，格式保持简短一致：

- 审计阶段
- 审计对象
- 关键风险
- 已检查内容
- 证据路径
- 发现的问题
- 风险等级
- 建议修复顺序
- 结论
