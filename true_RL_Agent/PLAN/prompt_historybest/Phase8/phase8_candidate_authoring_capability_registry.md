# Phase8: Candidate Authoring Capability Registry 泛化

## 目标

Phase8 将候选可写 surface 从分散的 prompt 字段收敛成公共 `candidate_capabilities` 合同。当前真实可写能力仍只包括 `RewardTerm`、`DiagnosticTerm`、`ScheduleTerm`；课程和环境只作为禁用或非 prompt-visible 的预留能力出现，不能变成可写 surface。

## 设计参考

Codex 中工具能力的关键模式是 registry 与 model-visible spec 分离：内部 registry 可以知道更多执行能力，但只有通过 provider/model gating 的 direct exposure 才进入模型可见合同。Phase8 借鉴这个单源 exposure 模式，但不引入 Codex 的 shell、任意工具调用或任意文件读取。

## 边界

- `candidate_capabilities` 是 prompt-facing 的候选作者合同，不是 runner 权限系统。
- 公共层只描述通用 capability、组件类、schema、materializer/verifier/runtime 边界，不解释任务失败语义。
- 任务层后续可以贡献 task-specific capability metadata，但必须在任务命名空间下，不能污染共享 capability 定义。
- `RewardTerm`、`DiagnosticTerm`、`ScheduleTerm` 的现有效果不能降低：reward 仍训练生效，diagnostic 仍 public feedback only，schedule 仍只允许 `reward_weight_scale`。
- `reward_weight_scale` 这类候选 `ScheduleTerm` key 必须从 `candidates.schedule_contract` 单源投影到 prompt、verifier、change-set 与 runtime，不能在各层重复维护。
- `candidate_component_contract`、`candidate_allowed_controls`、`control_effectiveness_contract` 作为兼容投影保留，但必须从同一个 capability registry 派生。
- 任务层可以通过 `task_capability_metadata` 贡献任务命名空间下的只读补充信息；共享 registry 只做 public-safe 搬运，不解释或合并任务语义。
- `curriculum_design`、`train_environment_design`、observation/feature projection 默认不可写，不进入 `prompt_visible_capabilities`。
- 不扩大 provider `declared_changed_components` 的 prompt advertised 值；verifier-derived change set 仍是权威。
- 不允许 `operation_diff`、runner patch、eval/budget/seed/checkpoint/PPO 修改重新进入 prompt 可写面。

## 调用流程

1. `prompting.capabilities` 构造默认 `CandidateAuthoringCapabilityRegistry`。
2. `payload.build_prompt_payload()` 生成 canonical `candidate_capabilities` section。
3. `TaskPromptAdapter` 如需提供任务专属 capability 说明，只能写入 `task_capability_metadata`，由 sanitizer 控制大小和 public-only 边界。
4. `candidates.schedule_contract` 提供候选 schedule key、change component、effective keys、runtime bounds 与 activation record。
5. `required_candidate_package.component_contract`、`candidate_allowed_controls`、`control_effectiveness_contract` 从 capability registry 和 schedule contract 投影，且不读取 `task_capability_metadata`。
6. `authoring_context_frame` 将 `candidate_capabilities` 作为稳定合同 section 计入 included sections 和 budget。
7. legacy alias 继续由 canonical section 派生，避免当前 provider 和测试断裂。

## 验收

- prompt payload 新增 `candidate_capabilities`，且 section 顺序位于 `required_candidate_package` 之后、reward/eval contract 之前。
- `prompt_visible_capabilities` 只包含当前真实可写的 reward、diagnostic、schedule 三类 capability。
- `disabled_or_reserved_capabilities` 可记录 curriculum/environment 预留，但必须声明 `prompt_visible=false`、`candidate_writable=false`、`effective_now=false`。
- `candidate_allowed_controls` 与 `candidate_capabilities.prompt_visible_capabilities[].component_class` 保持一致。
- `candidate_component_contract.classes` 与 capability registry 中的 component signatures 保持一致。
- `control_effectiveness_contract.component_controls` 与 capability registry 中的 effectiveness/runtime effect 保持一致。
- `reward_weight_scale` 的 allowed/effective key、change-set 映射和 runtime bounds 来自同一个 candidate schedule contract。
- prompt 中不出现可写 curriculum/environment surface、runner patch、eval patch、budget、seed、checkpoint、PPO。
- source_followup frame 仍保留 canonical `candidate_capabilities`，但不重复 legacy alias。
- task-specific capability metadata 只能出现在 `candidate_capabilities.task_specific_metadata`，不能污染 `prompt_visible_capabilities`、`candidate_allowed_controls` 或 component/effectiveness 投影。
- 任务特殊字段和 push recovery 语义不进入共享 capability registry。
- focused tests、full tests、compileall、diff-check 通过。
