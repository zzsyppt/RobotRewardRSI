# Phase 8E Prompt 与单 Round 内循环升级计划

日期：2026-06-28

状态：review 修订版

## 背景

`live_15round_20260628_024450` 这次 run 暴露的问题很集中：系统已经能完整训练和评测一个有效 candidate，但 LLM 拿到的训练侧 contract 信息太压缩，导致它经常写出“思路看起来对，代码实际跑不起”的训练侧组件。

第 4 轮 `upright_push_recovery_v1` 成功 accept，说明 runner、训练、checkpoint、public eval、decision 这条链路可以工作。但第 4 轮之后，大部分尝试没有真正进入有效评测，主要浪费在廉价错误上：

- 训练错误里反复出现 `base_lin_vel [N, 3]` 与 `push_vec_public [N, 2]` 的形状不匹配。这个现象现在只能作为 public training error 和高概率 root cause；在 per-term 捕获上线前，不能写成“已确认某个 term 崩了”的反馈。
- `push_active` 和 `push_vec_public` 虽然在 `RewardInputs` 里，但当前 adapter 中仍属于 runner 合成的 public push-state 字段，不应被 LLM 当成主要真实环境信号。
- training-side error 没有足够具体地进入下一次生成上下文，LLM 看不到具体是哪个 term、哪一组 shape、哪一层 contract 崩了。
- 部分 provider response 是坏 JSON，导致一个 round 在候选落盘前就浪费掉。

本文只规划前两层升级：

1. Prompt / 输入质量升级。
2. 单 round 内多次 LLM 调用 + 安全廉价校验 + 修复闭环。

暂时不把 multi-agent strategy tournament 放进第一阶段。当前证据表明，最高 ROI 是先避免 prompt / shape / JSON / contract 错误消耗 20 分钟级 Isaac 训练。

## Review 后的必须修正

这版计划吸收了三类 review 反馈。

### Critical：候选代码执行边界

任何会 import candidate package、调用 `build_training_components(...)`、执行 reward/loss/diagnostic callable 的逻辑，都不能在 orchestrator 父进程里运行。

正确边界是：

1. 父进程只做 JSON 解析、文件 staging、Phase 8D static 检查。
2. Phase 8D static 只能做纯静态检查，包括文件清单、manifest policy、AST/import policy、文本 policy，不执行候选代码。
3. 真正执行候选 callable 的 shape/contract 校验，必须作为 Phase 8D dynamic 的 fast mode，复用现有 namespace / Landlock 隔离子进程边界。
4. 父进程只读取 isolated child 产出的 public-safe summary，绝不 import 候选包。

因此本文后续不再使用“在 Phase 8D 之前新增一个会执行候选代码的 cheap validator”这个设计。cheap validation 只能拆成两段：父进程里的 no-exec static gate，加隔离子进程里的 fast dynamic check。

### High：source-read 必须是闭合 allowlist

Layer 2 允许 LLM 请求训练侧源信息，但这是新的权限面。它必须遵守：

- 以 allowlist 为准，默认拒绝，不靠 denylist 兜底。
- allowlist 应是具体 logical file id 或具体文件，不给前缀、目录通配或“训练目录全部”。
- `source_requests` 里的路径是非可信输入，必须 resolve、canonicalize、containment check，并拒绝绝对路径、`..`、符号链接、非普通文件。
- 返回内容前要做内容审计，不只是路径审计。重点确认没有 eval、metric、seed、hidden decision、checkpoint selector、secret、provider key 等泄漏。
- 第一阶段不默认给完整 `training_adapter.py`。写出 shape 正确 reward 通常只需要 contract、shapes、最小示例、accepted recipe 源码和 public failure record。

### Medium：字段与 shape 必须程序化生成

`RewardInputs` 当前真实字段是 11 个：

```text
base_height, roll, pitch, base_lin_vel, base_ang_vel,
joint_pos, joint_vel, last_action, push_active,
push_vec_public, progress_fraction
```

计划不能手写一份容易遗漏的字段表。`reward_inputs_contract` 和 fast dynamic synthetic inputs 必须从 runner-owned 来源程序化生成：

- 字段名来自 `RewardInputs.field_names()` 或 dataclass fields。
- shape/provenance 来自 runner-owned contract builder，底层可以复用 `Phase8ETrainingAdapter._build_inputs` 的真实 shape 逻辑或专门的 shape probe。
- `num_joints`、`num_actions` 来自当前 G1 runner metadata 或 adapter smoke sample，不能让 prompt 里手填。
- `progress_fraction` 要明确双重语义：作为 `RewardInputs` 字段时是每个 env 的 `[N]` public tensor；作为 `ScheduleTerm.fn(progress_fraction)` 入参时是 runner 预计算 schedule 的 scalar progress point。

## 核心原则

评测侧继续封闭。LLM 不应拿到 private eval seeds、hidden metrics、success metric 实现细节、checkpoint selector 权限，也不能修改 trusted eval。

训练侧可以更开放，但必须通过受控接口开放。这里的重点不是把信息总结得更漂亮，而是给 LLM 更少损失的源信息通道：

- `RewardInputs` 每个字段的 shape、值类型、语义、来源、instrumentation 状态。
- 形状正确、可运行的 `RewardTerm` 示例。
- 当前 best accepted recipe 的 `training_code/components.py` 源码。
- public failure record 和候选代码自己的 public-safe training error。
- 训练侧日志摘要、TensorBoard scalar 名称、训练 summary。
- 经内容审计后的 runner-owned contract/source 摘录。

源信息开放的边界是 training-side only 和 public-safe only。评测侧代码和逻辑仍然不进入 LLM 上下文。

## 第一层：Prompt / 输入质量升级

### 1. 先做 `RewardInputs` 字段扩充与字段准入

在生成 `reward_inputs_contract` 之前，应该先审一轮当前字段是否足够。现在 11 个字段能写基本 upright / velocity / action penalty，但对一个要自己创造 reward 的 Agent 来说，信息仍然偏窄：它缺少命令、足端、动作变化、默认姿态偏差、位移等训练侧 public 信号，导致 LLM 很容易围绕少数 proxy 反复改。

字段扩充的原则：

- 只加 training-side public state，不加 eval metric、seed、success threshold、checkpoint selector、runner/env handle。
- 优先加 runner 能稳定构造、shape 明确、每 step 都可用的字段。
- 优先加低维派生字段，而不是把整块 env/robot object 或超大 raw tensors 暴露给 candidate。
- 每个新增字段必须有 provenance：`observed`、`derived_from_observed`、`zero_fallback`、`not_instrumented`。
- 每个字段必须有机器可读的 shape，例如 `["N", 3]`，不能只把 shape 写成人读文本。
- 位移/姿态/命令类字段必须写清坐标系和 fallback；不能让 LLM 猜字段是 world、env-relative 还是 base frame。
- action history 类字段必须写清 episode reset 后的行为，避免把 reset 边界上的伪差分当成 smoothness 信号。
- 新字段必须先进入 synthetic inputs 和 isolated dynamic fast validation，再进入 prompt。

第一批建议增加的必要字段：

```text
base_pos_xy: [N, 2]
projected_gravity: [N, 3]
commanded_base_vel: [N, 3]
prev_action: [N, num_actions]
action_delta: [N, num_actions]
joint_pos_error: [N, num_joints]
```

这些字段的 ROI 比较高：

- `base_pos_xy`：让 LLM 能写 standing-in-place / drift penalty，但语义必须是每个 env 原点下的相对 xy，不能暴露 Isaac 世界绝对坐标。实现上取 `root_pos_w[:, :2] - scene.env_origins[:, :2]`；缺少 env origins 时走 zero fallback，不把 world absolute 伪装成 drift 信号。
- `projected_gravity`：比 roll/pitch 更通用，适合 upright reward，也能减少姿态公式错误。
- `commanded_base_vel`：让同一套 reward 可以区分 zero-command standing 和 velocity tracking，避免把“站稳”和“跟踪命令”混在一起。
- `prev_action` / `action_delta`：让 LLM 直接写 smoothness / action-rate penalty。当前 `last_action` 实际是 wrapper step 收到的 current action，名字容易误导；显式提供 previous/action_delta 更清楚。episode reset 后对应 env 的 previous action 必须清零；首步或 reset 后 `action_delta` 的语义是 `current_action - zero_previous`，不是跨 episode 的真实差分。
- `joint_pos_error`：比裸 `joint_pos` 更适合写默认姿态保持，减少 LLM 猜 G1 默认关节角。语义是 `joint_pos - default_joint_pos`；`default_joint_pos` 为 `[1, num_joints]` 时允许按 env 维广播到 `[N, num_joints]`。

第二批建议增加，但需要先确认 instrumentation 的字段：

```text
feet_contact: [N, num_feet]
feet_pos_z: [N, num_feet]
feet_vel_xy: [N, num_feet, 2]
joint_limit_margin: [N, num_joints]
push_active: [N]              # 从 placeholder 升级为真实 push state
push_vec_public: [N, 2]       # 从 placeholder 升级为真实 public push vector
```

这些字段能明显扩大 reward 空间，尤其是 foot slip、feet clearance、gait stability、push recovery。但它们依赖足端 body id/contact sensor/真实 push event 的可靠映射，不能在没验证前写成 observed。第一阶段可以把它们列为 `planned_public_fields`，只有 probe 通过后才进入 `RewardInputs`。

不建议增加的字段：

- raw env object、runner object、optimizer、policy、critic、PPO internals。
- eval success flag、private failure classifier、accept/reject threshold。
- private eval seeds、rollout seed、hidden metric 细节。
- 任意文件路径、checkpoint selector 状态。
- 完整 observation dict 的无过滤 dump。

字段扩充应产出一份 `reward_inputs_field_expansion_report.json`，列出：

- 当前 11 个 baseline 字段。
- 第一批新增字段是否已实现。
- 第二批 planned 字段为什么暂缓。
- 每个字段的 shape、来源、provenance、fallback 行为。
- 哪些字段不允许加入以及原因。

### 2. 增加一等公民的 `reward_inputs_contract`

`reward_inputs_contract` 不是“可选 reward 列表”，而是 LLM 可以读的输入 API schema。它告诉模型有哪些 public training inputs、每个张量是什么形状、当前是否真实 instrumented。新的 reward 仍然由 LLM 自由设计，只是必须写成这些输入上的合法公式。

这份 contract 本体应该保持精简、机器可校验，不承担“教 LLM 怎么写 reward”的职责。`safe_usage`、`unsafe_usage`、常见 bug 写法、推荐公式等内容不放进 contract schema，而是放到 known-good examples、self-audit checklist、失败反馈里。这样 contract 不会变成难维护的 prompt 教程，也更适合被 validator 和 synthetic input builder 复用。

这份 contract 必须程序化生成，不能手写。建议输出包括：

- `fields`：由 `RewardInputs.field_names()` 生成。
- `shape`：由 runner-owned shape probe 生成。
- `dtype_or_value_kind`：tensor-like numeric。
- `semantics`：字段语义。
- `source_status`：真实观测、runner 合成、placeholder、not instrumented 等。
- `provenance`：observed、derived_from_observed、zero_fallback、not_instrumented 等。
- `units`：如果明确，例如 m、m/s、rad、rad/s。
- `coordinate_frame`：如果适用，例如 world、base、joint-order。
- `fallback_behavior`：不可观测时如何填充。
- `shape_source`：用于审计的生成来源，例如 `RewardInputs + Phase8ETrainingAdapter._build_inputs probe`。

baseline 字段必须覆盖当前完整 11 项；如果字段扩充阶段已通过，contract 还应覆盖新增字段：

```text
base_height: [N]
roll: [N]
pitch: [N]
base_pos_xy: [N, 2]
base_lin_vel: [N, 3]
base_ang_vel: [N, 3]
projected_gravity: [N, 3]
commanded_base_vel: [N, 3]
joint_pos: [N, num_joints]
joint_vel: [N, num_joints]
joint_pos_error: [N, num_joints]
last_action: [N, num_actions]
prev_action: [N, num_actions]
action_delta: [N, num_actions]
push_active: [N]
push_vec_public: [N, 2]
progress_fraction: [N]
```

示例：

```json
{
  "base_lin_vel": {
    "shape": "[num_envs, 3]",
    "semantics": "base linear velocity, xyz",
    "source_status": "observed_when_robot_attr_available",
    "provenance": "observed",
    "units": "m/s",
    "coordinate_frame": "base_or_world_as_reported_by_adapter",
    "fallback_behavior": "zero_fallback_when_robot_attr_missing"
  },
  "base_pos_xy": {
    "shape": "[num_envs, 2]",
    "semantics": "env-origin-relative base xy position",
    "source_status": "derived_from_root_pos_w_and_scene_env_origins_when_available",
    "provenance": "observed",
    "units": "m",
    "coordinate_frame": "env_origin_relative_world_xy",
    "fallback_behavior": "zero_fallback_when_root_pos_or_env_origins_missing"
  },
  "action_delta": {
    "shape": "[num_envs, num_actions]",
    "semantics": "current action minus previous action",
    "source_status": "runner_owned_action_history",
    "provenance": "derived_from_action_history_or_current_action_and_zero_previous_action",
    "fallback_behavior": "previous action is zero on first step and for envs reset by the latest step result"
  },
  "push_vec_public": {
    "shape": "[num_envs, 2]",
    "semantics": "public xy push vector field",
    "source_status": "runner_synthesized_not_primary_signal",
    "provenance": "not_instrumented",
    "units": "unitless_or_mps_like_public_xy",
    "coordinate_frame": "xy",
    "fallback_behavior": "zero_placeholder_until_push_instrumentation_is_live"
  },
  "progress_fraction": {
    "shape": "[num_envs]",
    "semantics": "per-env public training progress inside RewardInputs",
    "provenance": "runner_owned",
    "schedule_semantics": "ScheduleTerm.fn receives scalar progress_fraction during runner-owned precompute"
  }
}
```

### 3. 区分 allowed controls 和 effective controls

当前 prompt 说了 schedule keys，但没有足够硬地区分“允许返回”和“现在真的生效”。应明确告诉 LLM：

- `reward_weight_scale`：当前真的生效，会缩放 candidate reward weights。
- `push_magnitude_scale`、`push_start_fraction`、`push_interval_s`：可以被接受和预计算，但当前还不会真正改变 env push curriculum。
- `domain_randomization_scale`：可以被接受和预计算，但当前还不会真正改变 env domain randomization。
- `LossTerm`：会被 guard 和执行检查，但当前还没有插入 PPO optimizer。

Prompt 里应直接写：

```text
Do not rely on currently ineffective schedule keys as the main causal mechanism.
The candidate must be able to improve through RewardTerm callables and
reward_weight_scale alone.
```

### 4. 提供最小但真实的 source pack

Prompt 应优先给 LLM 不经二次总结的源信息，但第一阶段的 source pack 要收敛在最小有用集合：

- generated `reward_inputs_contract.json`。
- `reward_inputs_field_expansion_report.json`。
- 1-2 个 runner-owned known-good component examples。
- 当前 accepted recipe 的完整 `training_code/components.py` 源码。
- accepted recipe 的 public eval summary。
- 最近失败候选的 public-safe error records。
- 必要的 contract/source 摘录，而不是默认提供完整 runner adapter 源码。

完整 `training_adapter.py` 不应作为第一阶段默认上下文。它包含 reward weight scale、provenance、runner glue 等逻辑，虽然属于训练侧，但仍要先做内容审计，再决定是否以摘录形式提供。

### 5. 放入 1-2 个最小可运行 `RewardTerm` 示例

示例必须是代码，不是概念描述。建议包含：

- 一个保守的 upright / height / action_delta penalty reward。
- 一个正确切片向量的例子，例如 `base_lin_vel[:, :2]`。
- 一个利用 `commanded_base_vel` 或 `base_pos_xy` 的例子。
- 一个明确返回 `[num_envs]` vector 的例子。

示例要短，并且不把当前不可靠字段作为主信号。示例本身应进入测试，确保能被当前 contract 和 isolated fast dynamic check 跑通。

### 6. 提高失败反馈保真度

对候选代码训练失败，下一次 LLM 调用应收到：

- 失败阶段：JSON staging、8D static、8D dynamic fast validation、fixed-budget training。
- 如果知道，给出失败的 component name。
- exception type 和 public-safe message。
- 如果 public-safe，给出 candidate code 内的 stack frame。
- checkpoint 缺失是否由训练崩溃导致。
- 如果是 shape mismatch，给出两边 shape。

在 per-term 捕获上线前，反馈只能说“训练错误报告了某组 shape mismatch”。上线后，反馈才能接近：

```text
RewardTerm <name> failed on isolated public synthetic inputs:
RuntimeError: tensor shape mismatch. lhs has shape [N, 3],
rhs has shape [N, 2]. Use lhs[:, :2] or avoid mixing these fields.
```

这比只给 `RuntimeError` 或 `fixed_budget_training_reject` 有用得多，同时不会伪造当前还没捕获到的 per-term 证据。

### 7. 增加 candidate 自检清单

在输出 JSON 前，prompt 应要求模型显式遵守一份 checklist：

- 每个 `RewardTerm`、`LossTerm`、`DiagnosticTerm` 都返回 public numeric vector 或 summary，shape 正确。
- 不重复 component name。
- 不使用 unsupported imports、runner hooks。
- 不依赖 hidden eval 或 success metric 细节。
- 不依赖当前未生效的 schedule key 作为主要机制。
- JSON 只包含要求的文件，不在 JSON 外输出 prose。
- 除非有明确理由，否则从 current best recipe 源码小步修改。

这份 checklist 也应在 repair prompt 里复用，使生成和修复使用同一套 contract。

## 第二层：单 Round 内多次 LLM 调用 + 安全廉价校验

### 目标

不要在 candidate 还没通过毫秒到秒级检查时，就启动一次 20 分钟级 Isaac 训练。但这个检查不能破坏安全模型：父进程不能执行候选代码。

修订后的流程是：

```text
for attempt in 1..K:
    optional source-read turns through closed allowlist
    author candidate
    parse/stage JSON without executing candidate code
    run Phase 8D static gate in parent process
    if static issues:
        repair prompt += exact static issues
        continue
    run Phase 8D dynamic fast validation in isolated child
    if no issues:
        select attempt
        break
    repair prompt += exact isolated dynamic issues

if no selected attempt:
    reject before training
else:
    run isolated fixed-budget training
    run trusted public eval
```

建议默认 `K = 3`，昂贵 live run 可设 `K = 5`。

### 1. Candidate 生成前允许 source-read turns

第一次 LLM 调用不一定直接产出 candidate。它可以先请求读取 allowlist 内的源信息。

允许读取的训练侧源信息应使用 logical ids，而不是任意路径：

```text
contract/reward_inputs_contract.json
examples/known_good_reward_terms.py
current_best/training_code/components.py
current_best/public_eval_summary.json
public_failures/recent_training_errors.json
public_ledger/recent_accepted_recipes.json
```

禁止读取：

- hidden eval configs 或 seeds。
- private eval results。
- success metric 实现。
- checkpoint selector 内部逻辑，除 public safe-load status 外。
- secrets、provider keys、无关系统路径。
- 未经内容审计的 runner adapter 全文件。

实现上不需要给 provider shell 权限。LLM 返回结构化 `source_requests`，orchestrator 根据闭合 allowlist 填回原始源代码/日志片段。

`source_requests` 处理规则：

- 只接受 allowlist 里的 logical id。
- 如果内部映射到文件，必须 canonicalize 后确认仍在允许 root 内。
- 拒绝绝对路径、`..`、符号链接、非普通文件。
- 返回前做内容审计和大小限制。
- 每次返回的 source 都写入 artifact，方便事后审计。

示例：

```json
{
  "source_requests": [
    {
      "id": "current_best/training_code/components.py",
      "reason": "reuse shape-safe RewardTerm patterns"
    },
    {
      "id": "contract/reward_inputs_contract.json",
      "reason": "confirm tensor shapes and instrumentation status"
    }
  ]
}
```

### 2. Author call

源信息阅读完成后，LLM 输出正常 candidate package JSON。此阶段仍然是单 agent，不引入独立 reviewer / scientist persona。

Author call 应收到：

- 增强后的 prompt contract。
- generated `reward_inputs_contract`。
- current best recipe 源码。
- public raw feedback。
- 前面 source-read turn 请求到的源代码片段。
- candidate self-audit checklist。

### 3. JSON staging 与 JSON repair

Provider response 如果 JSON 解析失败，不应直接浪费一个 round。应增加单独 JSON repair：

- 保存 raw response text 到 public run artifact。
- JSON parsing fail 时，让 provider 再调用一次“只把这段内容修成 valid JSON”的 prompt。
- JSON repair 不允许语义重写 candidate。
- 最大 retry 1-2 次。

JSON staging 只能落盘和做文件结构准备，不能 import candidate package。

### 4. Phase 8D static gate

任何候选代码执行前，必须先跑 8D static gate。父进程可做的事情只有：

- file inventory。
- manifest schema 和 policy。
- Python AST/import policy。
- text policy。
- package fingerprint。

如果 static gate 不通过，直接把 exact static issues 送给 repair call。此时没有执行候选代码，也不需要启动训练。

### 5. Phase 8D dynamic fast validation

shape/contract 校验必须作为 Phase 8D dynamic 的 fast mode，在隔离子进程里完成。它可以复用现有 isolated component loader，也可以扩展一个 sibling child mode，但安全属性必须一致：

- 网络隔离、PID namespace、Landlock 等 server enforcement 与现有 8D dynamic 一致。
- child 进程 import candidate package。
- child 进程调用 `build_training_components(...)`。
- child 进程运行 component structure validation。
- child 进程预计算 schedules。
- child 进程用程序化生成的 synthetic `RewardInputs` 执行每个 reward/loss/diagnostic callable。
- child 进程强制 reward/loss 输出 `[num_envs]` 且 finite。
- child 进程返回 component-specific public-safe issue records。
- 父进程只读 summary，并记录 `parent_imported_candidate_code: false`。

Synthetic inputs 不能手写列表，必须由同一个 `reward_inputs_contract` builder 生成。它应覆盖完整字段：

```text
base_height [N]
roll [N]
pitch [N]
base_lin_vel [N, 3]
base_ang_vel [N, 3]
joint_pos [N, num_joints]
joint_vel [N, num_joints]
last_action [N, num_actions]
push_active [N]
push_vec_public [N, 2]
progress_fraction [N]
```

这一步不替代 fixed-budget training。它只是在安全隔离内先拦截 cheap shape / contract 错误。

### 6. Repair call

如果 static gate 或 isolated fast validation 失败，下一次 LLM 调用收到：

- 上一次 candidate 的完整源代码。
- 精确 issue records。
- 严格要求只修 candidate package。
- 不给新的 eval 信息。
- 同一份 self-audit checklist。

Repair prompt 应偏向最小改动：

```text
Do not redesign the whole reward unless necessary. Fix the smallest code region
that causes the reported issue. Preserve working terms from the current best
recipe.
```

这应该能在一次额外调用里修掉重复出现的 shape bug、重复 component name、坏 JSON 等廉价问题。

### 7. Inner-loop artifacts

每个 attempt 都应留下可审计文件：

```text
provider_candidate/
  attempts/
    attempt_001/
      prompt_messages.json
      source_requests.json
      source_payloads.json
      raw_response.txt
      json_repair_summary.json
      staged_candidate/
      phase8d_static_summary.json
      phase8d_dynamic_fast_summary.json
      repair_feedback.json
    attempt_002/
      ...
  selected_attempt.json
  phase8e_provider_candidate_summary.json
```

顶层 provider summary 应记录：

- 尝试次数。
- 每次失败原因。
- 最终选中哪个 attempt。
- 是否经过 repair。
- 进入训练前 static 和 isolated dynamic fast validation 是否通过。
- 父进程是否 import 过 candidate code，必须为 false。

### 8. 停止条件

inner loop 在以下情况停止：

- candidate 通过 JSON staging、8D static、isolated dynamic fast validation。
- max attempts 用完。
- provider 返回 unsafe content。
- source request 超预算或请求 forbidden logical id。
- static/dynamic issue 重复且 repair 不再改变 candidate。

如果所有 attempt 都在训练前失败，该 round 应为 `reject`，reason 为 `candidate_failed_pretraining_inner_loop`，并且不启动 Isaac 训练。

## 与现有 Phase 8D/8E 边界的关系

这个计划不削弱 safety model。

新顺序是：

1. Prompt / source-read inner loop。
2. Candidate JSON staging。
3. Phase 8D static gate，父进程 no-exec。
4. Phase 8D dynamic fast validation，隔离子进程执行候选 callable。
5. 现有 isolated fixed-budget training。
6. 现有 safe checkpoint resolution。
7. 现有 trusted public eval。
8. 现有 public decision 和 ledger append。

评测仍由 runner 拥有。candidate 仍然只控制训练侧 components。

## 预期收益

收益主要是工程效率：

- shape / contract bug 在几秒内修复，而不是浪费整轮训练。
- invalid JSON 大概率在 staging 前修复。
- LLM 从可运行 accepted recipe 上迭代，而不是凭字段名猜 contract。
- public training error 能指向具体阶段和具体 component。
- 只有至少能在隔离 fast validation 中执行的 candidate 才进入昂贵训练。

对 `live_15round_20260628_024450` 来说，这大概率会节省第 4 轮之后的大部分失败轮，尤其是重复 shape mismatch 和坏 JSON。

## 实现草图

### Prompt / payload changes

扩展 `build_phase8e_prompt_package(...)`：

- `reward_inputs_field_expansion_report`。
- `reward_inputs_contract`，程序化生成。
- `effective_controls`。
- `current_best_recipe.components_source`。
- `known_good_examples`。
- `candidate_self_audit_checklist`。
- `source_access_policy`。

### Contract generator changes

新增 runner-owned contract builder：

- 字段名来自 `RewardInputs` dataclass。
- shape 来自 adapter-compatible probe。
- `num_joints` / `num_actions` 来自当前 G1 runner metadata。
- 输出 contract JSON 和 synthetic input builder 共用同一份 schema。
- contract builder selftest 要防止漏掉 `base_height`、`roll`、`pitch` 这类 baseline 字段。
- contract schema 不包含 `safe_usage` / `unsafe_usage` 这类教程字段；这些内容只允许出现在 examples、checklist、repair feedback 中。

### RewardInputs field expansion changes

在 contract builder 前新增字段扩充 pass：

- 对 current adapter 可稳定读取的 public state 做 probe。
- 第一批实现 `base_pos_xy`、`projected_gravity`、`commanded_base_vel`、`prev_action`、`action_delta`、`joint_pos_error`。
- 对足端/contact/push 相关字段先做 instrumentation probe，未通过则写入 `planned_public_fields`，不进入 `RewardInputs`。
- 每个新增字段必须同步进入 dataclass、adapter mapping、provenance、synthetic inputs、contract JSON、selftest。
- 对 `last_action` 的语义做澄清；如果它实际表示当前 step action，则 contract 里不要继续让 LLM 误解为 previous action。
- `base_pos_xy` 必须是 env-origin-relative xy；selftest 要覆盖 `root_pos_w[:, :2] - scene.env_origins[:, :2]`，防止回退成 world absolute。
- `prev_action` / `action_delta` 必须由 runner-owned action history 产生，并接收 VecEnv step result 的 done/reset mask；reset env 的 previous action 清零。
- `joint_pos_error` 必须支持 `default_joint_pos` 的 `[1, num_joints]` 到 `[N, num_joints]` 广播，形状不匹配时才走 zero fallback。
- `reward_inputs_field_expansion_report` 必须包含机器可读 `field_shapes`，并和 `RewardInputs.field_names()` 保持一致。

### Source-read service changes

新增 source-read 服务，而不是把文件系统交给 LLM：

- logical id allowlist。
- canonical path containment。
- symlink / non-regular file rejection。
- 内容审计。
- 大小和 turn 数预算。
- source payload artifact 记录。

### Provider orchestration changes

把单次 provider call 替换成 attempt loop：

- author 前支持 `source_requests`。
- 每个 attempt 保存到 `provider_candidate/attempts/`。
- parse failure 走 JSON repair call。
- static / isolated dynamic fast failure 走 candidate repair call。
- 最终只把 selected attempt 交给后续训练。

### Phase 8D dynamic fast mode changes

扩展现有 isolated dynamic 子进程能力：

- 在 child 内执行 component load 和 callable shape check。
- 复用 guard helpers 和 contract synthetic inputs。
- 输出 component-specific issue records。
- 输出 `parent_imported_candidate_code: false`。
- 可对 archived failed candidates 做 replay regression test。

### Config changes

建议新增：

```json
{
  "provider_inner_loop": {
    "enabled": true,
    "max_attempts": 3,
    "max_source_read_turns": 2,
    "json_repair_attempts": 1,
    "static_gate_required": true,
    "isolated_dynamic_fast_validation_required": true
  }
}
```

## 测试计划

下一次 live run 前至少要测：

- `reward_inputs_contract` 自动包含 11 个字段，尤其覆盖 `base_height`、`roll`、`pitch`。
- 字段扩充 pass 生成 `reward_inputs_field_expansion_report.json`。
- 第一批新增字段的 machine-readable shape/provenance 与 synthetic inputs 一致。
- `base_pos_xy` 验证为 env-origin-relative xy，而不是 world absolute xy。
- `prev_action` / `action_delta` 验证首步和 episode reset 后都使用 zero previous action，不跨 episode 泄露 action history。
- `joint_pos_error` 验证 `default_joint_pos` 的 `[1, num_joints]` 可广播到 `[N, num_joints]`。
- `base_pos_xy`、`projected_gravity`、`commanded_base_vel`、`prev_action`、`action_delta`、`joint_pos_error` 在缺失真实 env attr 时有明确 fallback。
- 足端/contact/push 相关字段在 instrumentation 未验证前只能出现在 `planned_public_fields`，不能进入 live `RewardInputs`。
- contract JSON 不包含 `safe_usage` / `unsafe_usage`，避免 schema 和 prompt 教程混在一起。
- contract JSON 和 synthetic input builder 来自同一 schema。
- `num_joints`、`num_actions` 使用当前 G1 runner 真实值。
- `progress_fraction` 的 RewardInputs `[N]` 语义和 ScheduleTerm scalar 语义都在文档和测试里区分。
- prompt payload 包含字段 shape 和 instrumentation status。
- 有 accepted recipe 时，prompt 包含 current best 源码。
- source request allowlist 拒绝 eval/private paths。
- source request 拒绝 `..`、绝对路径、符号链接、非普通文件。
- source payload 内容审计能拦截 eval/metric/seed/secret 关键词或未批准文件。
- Phase 8D static gate 不执行候选代码。
- isolated dynamic fast validation 只在 static pass 后运行。
- 父进程在 fast validation 中不 import candidate package。
- isolated dynamic fast validation 能抓 duplicate component names。
- isolated dynamic fast validation 能抓 `[N, 3] * [N, 2]` reward bug。
- JSON repair 能把 malformed provider text 修成 valid candidate package。
- repair attempt 能收到 exact issue records 并产出修复候选。
- max-attempt exhaustion 会在 Isaac 训练前 reject。
- 现有 Phase 8D、isolated training、trusted eval 测试仍然通过。

## 风险

- LLM 调用次数增加，但和浪费一次 20 分钟训练相比成本低。
- 原始 source context 可能过大；应使用 allowlist、大小上限和 source-read turns。
- source access 过宽可能泄露评测侧逻辑；必须使用闭合 allowlist 和内容审计。
- 如果 fast validation 被错误放在父进程，会破坏安全模型；必须用测试锁死 `parent_imported_candidate_code: false`。
- repair call 可能只修语法不改策略质量；通过 current best 源码、小步改动和后续 public eval 控制这个风险。

## 非目标

- 暂不做 multi-agent strategy tournament。
- 不改 trusted eval metric 或 accept/reject 逻辑。
- 不改 checkpoint selection。
- 不给 candidate env object、runner hooks、filesystem、network、process control。
- 不做大范围 repo refactor。

## 推荐落地顺序

1. 实现 Layer 1 prompt/payload changes，并保存一份 prompt artifact 人工检查。
2. 实现 RewardInputs 字段扩充 pass：先加第一批低风险字段，足端/contact/push 字段只做 probe 和 planned report。
3. 实现 contract generator 和 synthetic input builder，先把字段/shape 漂移问题消掉。
4. 实现 source-read service 的 closed allowlist、路径加固和内容审计。
5. 扩展 Phase 8D dynamic fast mode，用 archived failed candidates 做 replay 测试。
6. 加 static / dynamic failure 的 repair call。
7. 加 JSON repair path。
8. 跑 dry-run / replay selftest。
9. 跑一个短 live loop，`max_attempts=3`，验证只有 isolated dynamic fast validation pass 后才启动 Isaac training。

## 逐层逐条 Review

### 第一层 Review

1. `RewardInputs` 字段扩充：合理且应该放在 contract 前。否则 contract 只是把贫瘠输入描述得更清楚，不能从根上提升 reward 创造空间。第一批新增字段应偏保守，足端/contact/push 先 probe 再开放。
2. `reward_inputs_contract`：合理且必要。修订后它是机器可校验输入 schema，不是 reward 菜单，也不是 prompt 教程；新 reward 仍由 LLM 自由设计。`safe_usage` / `unsafe_usage` 不放进 contract 是正确的。
3. allowed vs effective controls：合理且必要。它直接减少 LLM 把未生效 schedule key 当主机制的概率。
4. 最小真实 source pack：合理，但必须受限。accepted recipe 源码可以给；完整 runner adapter 源码第一阶段不默认给，只给审计后的 contract/source 摘录。
5. known-good `RewardTerm` 示例：合理且高 ROI。示例必须是当前测试能跑通的真实代码，否则会变成新的误导源。
6. 失败反馈保真：合理。修订后不再把未捕获的 per-term 信息当事实；只有 per-term capture 上线后才给 component-level 失败。
7. candidate 自检清单：合理，但只能作为 prompt 约束。真正约束仍靠 static gate、isolated dynamic fast validation 和训练/eval。

### 第二层 Review

1. source-read turns：有必要，但权限风险最高。修订后采用 closed allowlist、logical id、路径加固和内容审计，风险可控。
2. Author call：合理。保持单 agent，避免第一阶段引入 multi-agent 调度复杂度。
3. JSON staging / repair：合理。它不执行候选代码，能低成本修第 9/11 轮那类坏 JSON。
4. Phase 8D static gate：必须保留在任何候选执行之前。这是候选代码进入隔离执行前的唯一父进程门禁。
5. Phase 8D dynamic fast validation：合理且是第二层的核心。修订后它位于隔离子进程中，和现有 8D dynamic 安全边界一致。
6. Repair call：合理。它的输入必须只包含 public-safe issue 和上次 candidate，不给新 eval 信息。
7. Inner-loop artifacts：合理。没有这些 artifact，后续很难审计为什么某个 attempt 被选中或拒绝。
8. 停止条件：合理。max attempts、unsafe content、source request 越界、重复失败都应在训练前终止。

总体结论：前两层升级应该做。优先级顺序是 Critical 边界修正、RewardInputs 字段扩充、contract 程序化生成、source-read 加固、isolated fast validation、repair loop。multi-agent 可以继续推迟；当前失败主要不是策略搜索不够多，而是单个 candidate 在获得足够 public training state、真实 contract 和廉价修复前就进入了昂贵训练。
