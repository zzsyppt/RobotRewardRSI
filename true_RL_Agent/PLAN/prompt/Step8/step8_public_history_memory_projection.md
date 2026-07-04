# Step 8 Public History 与 Memory 投影升级

## Scope

本记录完成 `PLAN/prompt/prompt_upgrade_execution_plan.md` 的 Step8：升级下一轮 prompt 可见的 public history 与 memory 投影，让 LLM 能看到上一轮候选是否真正运行、是否进入 eval、公开分数/拒绝原因、runtime activation、runner/provider 失败摘要，同时保持信息有硬上限。

Step8 不改变 accept/reject 逻辑、不改变任务 success 判定、不暴露 private/trusted/raw seed/checkpoint selector，也不引入课程 stage 或 LLM curriculum control。

## Codex 设计参考

参考 `D:\humanoid_agent\codex\AGENTS.md` 的 model-visible context 约束：

- model context 必须增量构建，不重写历史。
- 注入模型上下文的内容必须有硬上限。
- 单个注入片段不能无限增长，超过较大上下文片段需要单独审查。
- 上下文片段应结构化，而不是临时拼接文本。

对应到新项目，本步骤没有把完整 ledger 原样塞进 prompt，而是在 `agent_core.context` 中生成结构化、按轮数和序列化字节数双重限制的 `recent_outcomes`。

## Implementation Plan Used

1. 先写失败测试，证明当前 `public_history` 只有薄的 round/outcome/surface，缺少 reason、public score、changed components、runtime/load/error evidence 和 history limits。
2. 在共享层 `build_public_context()` 源头生成 bounded recent history，而不是在 provider 或 prompt 表层补字段。
3. 投影跨任务通用字段：`round_id`、`outcome`、`reason`、`surface_classification`、`public_score`、`changed_components`。
4. 投影跨任务 runtime evidence：candidate package 是否加载、candidate code 是否进入 eval、reward/diagnostic calls、nonzero reward、capability/feature activation records、mechanism attribution。
5. 投影 public-safe runner/provider/repair failure 摘要，并统一走 prompt-public path scrub。
6. 让 `prompting.payload._public_history()` 直接复用 `agent_core.context.build_public_history_section()`，不再硬编码 `recent_outcomes[-5:]`，也不信任外部传入的放宽 limits。
7. 将 `PublicContextBuilder` 的 task memory summary 和最终 context 返回边界升级为 `sanitize_prompt_public_context()`。

## Files Changed

- `true_RL_Agent/src/true_rl_agent/agent_core/context.py`
- `true_RL_Agent/src/true_rl_agent/memory/context_builder.py`
- `true_RL_Agent/src/true_rl_agent/prompting/payload.py`
- `true_RL_Agent/tests/unit/test_memory_context_phase13.py`

`true_RL_Agent/src/true_rl_agent/tasks/push_recovery/memory.py` 已审计但本步未改：push-specific success/failure taxonomy 已由 task-owned memory/prompt adapter 负责，Step8 的新增字段属于跨任务 public history，不应把 push 语义写入共享投影。

## Boundary Decisions

- 共享层只理解跨任务 public facts，不解释 push recovery success。
- `candidate_package_loaded` 和 `candidate_code_loaded_in_eval` 从 public metrics、train summary、candidate summary 中读取，但缺失时只给 `unknown`，不推断任务语义。
- `runtime_activation` 只保留短摘要，不复制完整 train summary。
- runner/provider/repair errors 只保留公开类型、分类、原因和短 details，路径统一 scrub 为 `[path]`。
- history 使用 `PUBLIC_HISTORY_MAX_RECENT_ROUNDS = 8` 和 `PUBLIC_HISTORY_MAX_SERIALIZED_BYTES = 8192` 双重上限；超出时丢弃更旧条目并记录 `omitted_round_count`。
- task memory summary 进入 prompt 前也走 prompt-public sanitizer，防止路径值只在 payload 表层被 scrub。

## Red Tests Observed

实现前 `test_memory_context_phase13` 新增的 3 个 Step8 测试失败：

- `test_prompt_public_history_projects_runtime_and_error_evidence` 因 `recent_outcomes` 缺少 `reason` 失败。
- `test_prompt_public_history_projects_provider_repair_failure` 因 `recent_outcomes` 缺少 `provider_failure` 失败。
- `test_prompt_public_history_is_count_and_byte_bounded` 因 `public_history` 缺少 `limits` 失败。
- `test_direct_prompt_payload_public_history_is_still_bounded` 因直接 assembler 入口可绕过 history budget 失败。

这些失败对应 Step8 的真实缺口，而不是测试环境问题。

## Verification

Focused red-green command:

```powershell
$env:PYTHONPATH='D:\humanoid_agent\true_RL_Agent\src'
& 'C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.unit.test_memory_context_phase13
```

Current focused result:

- `test_memory_context_phase13`: `Ran 12 tests`, `OK`

## Review Follow-up

后续 review 发现两个真实问题，均已修复：

- `PublicContextBuilder` 的 raw `Mapping` 输入分支仍调用已删除 import 的 `sanitize_public_context()`，会在 `build_prompt_payload([dict_record])` 时 NameError。现在该分支直接使用 `sanitize_prompt_public_context()`，并有回归测试覆盖。
- direct assembler 入口可以通过外部 `public_context.recent_outcomes` 注入任意 history key，例如 `push_active`、`push_vec_public`、`raw_seed`。现在 public history 使用固定 allowlist 重投影，嵌套 details 和自由文本也过滤 seed/push 专属内容；同时 legacy `public_context` alias 回写同一份 bounded/sanitized history，避免 canonical section 干净但 alias 泄漏。

Full verification is recorded in the final turn report for this step.

## Remaining Scope

- Step9/10 still own final provider role split and readability cleanup.
- Step12/13 must validate a real generated multi-round prompt artifact and provider user payload capture.
- Future tasks can add task-specific memory adapters without changing shared public history semantics.
