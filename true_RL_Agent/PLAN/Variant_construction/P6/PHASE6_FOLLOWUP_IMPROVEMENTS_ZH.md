# Phase 6 非阻断可提升项（终态审计）

> 证据范围：`7e5fa25b607ac118c64424d3e137fa6ce713b2c0` 与服务器终态输出
> `outputs/p6_go2_validation3_v4_successor`。本文不重复
> `PHASE6_FOLLOWUP_OBSERVATIONS_ZH.md` 的 O1–O15，只记录不影响本轮 GO 证据成立、但会降低后续
> LOOP 信息质量、报告可信度或维护效率的事实。

## F1：`joint_ood` 名称会把几何联合外推误读成机器人关节外推

**问题与证据**

- `src/true_rl_agent/tasks/stair_climbing/unified_evaluation.py:382-420` 将 `joint_ood` 定义为
  `joint_scalar_extrapolation`，其真实条件是 `(height_in, tread_in) == (False, False)`；它是台阶高度与踏深
  同时越出训练支持集，不涉及机器人 joint、动作或执行器参数。
- 生产 prompt 直接暴露这一 claim 名称。第三代 lane-a 的
  `go2_stair_wb_joint_limit_margin_v1` 因而在 rationale 中解释为“joint scalar extrapolation 会把 commanded
  joints 推向 soft limits”，并新增 `joint_limit_margin_l2`。候选源码位于
  `candidate_artifacts/...generation-003/lane-a/go2_stair_wb_joint_limit_margin_v1/.../training_code/components.py`。
- 同一候选的 `public_dev` 中，`joint_ood` 从 parent 的 `0.7189674265` 降至 `0.4228627820`，该层成功数从
  `18/27` 降至 `7/27`，forbidden-contact 从 `6/27` 升至 `18/27`；`J_primary` 从 parent 的
  `0.9791827723` 降至 `0.9339485899`。

**非阻断事实**

- 固定 selector 以 guard regression `0.067851 > 0.05` 拒绝了该候选，本轮 current-best 与 GO 结论未被污染。

## F2：诊断 flag 是历史相关信号，不是失败因果标签

**问题与证据**

- `src/true_rl_agent/tasks/stair_climbing/episode_evidence.py:417-465` 在 episode 任意时刻达到连续步数后永久锁存
  orientation/slip/clearance diagnostics；`episode_evidence.py:589-653` 则允许成功覆盖这些历史 diagnostics，
  且失败主因中的 orientation/slip/clearance 使用终止时的当前 streak。二者语义并不等价。
- `src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py:230-243` 的 `support_contact` 不是当前帧接触：
  它对 `net_forces_w_history` 的 history 维取最大值。Go2 binding 的 contact sensor 为
  `history_length=3`、`update_period_s=0.005`，因此该布尔量表示最近约 `15 ms` 内出现过接触。
- 第三代 lane-b rationale 把全局高 `slip_detected` 解释成失稳与 forbidden-contact 的“causal precursor”，并以
  `support_contact * support_lin_vel` 新增 `contact_slip_excess`。训练终态证明该项真实激活：
  `call_count=48,001`、`observation_count=2,001`、`last_weighted_mean=-0.0287608`、
  `max_abs_weighted_contribution=1.5`。
- 候选确实大幅压低历史 slip flag：相对 parent，八层每 `27` 局的 slip 计数依次由
  `22,22,18,23,27,19,26,25` 变为 `0,1,5,0,5,22,0,9`；但 `J_primary` 同时从
  `0.9791827723` 降至 `0.9362235077`，`joint_ood` 成功数从 `18/27` 降至 `5/27`，forbidden-contact
  从 `6/27` 升至 `20/27`。
- 这比 O5 的“flag 是否饱和”更窄且更强：本次实验证明 LLM 可以成功优化诊断计数，却反向伤害任务目标，因此
  flag 下降不能单独构成机制有效或失败因果已解除的证据。

**非阻断事实**

- selector 以 guard regression `0.064439 > 0.05` 拒绝 lane-b；诊断 flag 没有 acceptance authority。

## F3：`goal_progress_fraction` 是当前空间位置，不是吸收式任务事件

**问题与证据**

- `src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py:255-264` 每个控制步按当前
  `base_relative[:, 0]` 计算并 clamp `goal_progress_fraction`；机器人后退时该值会下降。它不是历史最大进度、
  首次到顶 latch 或 absorbing completion event。
- 对外 ABI 仅在 `runtime_projection.py:361-367` 给出 `float/fraction`；
  `reward_inputs.py:350-356` 只澄清 `progress_fraction` 是 episode 时间并把
  `goal_progress_fraction` 称作 spatial progress，没有声明其非锁存时间行为。
- 第二代 lane-a 的 `go2_stair_wb_prog_gait_v2` rationale 明确把该输入称为 “the absorbing task event”，并新增
  `goal_progress_shaping = goal_progress_fraction`。其 `public_dev J_primary=0.8899429524`、
  `J_ood=0.8349144285`，显著低于 parent 的 `0.9791827723`。

**非阻断事实**

- 该候选被固定 guard 拒绝，没有进入 current-best；runtime 数值本身与评测空间进度同源，问题只在 LLM 可见的
  时间语义不完整。

## F4：已接受 delta 的人类可读记忆把语义修改压成了 `rename`

**问题与证据**

- 第一代被接受候选的 `mechanism_operation_delta` 记录
  `RewardTerm:feet_air_time -> RewardTerm:support_air_time`、`operation="rename"`。before/after 的
  `structure_hash` 分别为 `7e3524d0...` / `174712ba...`，normalized hash 分别为
  `2b95230e...` / `a1c0878a...`，输入也从
  `{commanded_base_vel,support_first_contact,support_last_air_time}` 变为
  `{support_first_contact,support_last_air_time}`。
- 实际源码不只是改名：parent 为
  `(support_last_air_time - 0.5) * support_first_contact * moving`，其中
  `moving = norm(commanded_base_vel[:,:2]) > 0.1`；accepted candidate 为
  `support_last_air_time * support_first_contact`，同时删除 `-0.5` 阈值与 moving gate。
- `src/true_rl_agent/candidates/component_source_semantics.py:1707-1723` 只要 canonical identity 带
  `renamed_to` 就固定令 operation 为 `rename`；结构 hash 能显示差异，但 operation 标签不表达“改名并修改语义”。
- 第三代真实 prompt 的 `public_history.experiment_memory.experiments[0].candidate_components` 最终只显示
  `mechanism="RewardTerm:feet_air_time->support_air_time"`，机制归因为
  `not_attributed/shadow_ablation_not_run`。当前源码另行内联，因而 LLM 仍能读到函数本体；人类终态摘要与压缩记忆
  则无法仅凭标签还原本次被接受改动。

**非阻断事实**

- acceptance、current-best checksum 与后续训练均绑定完整候选源码和结构 hash；失真只存在于可读 operation/memory
  投影，不改变本轮执行或判决。

## F5：终态报告丢失已选 lane，且 no-selection 仍消耗两条 `public_accept`

**问题与证据**

- `src/true_rl_agent/tasks/stair_climbing/generation_session.py:7562-7566` 从
  `record.selection.selected_lane_id` 构建终态报告；selection v5 只存 `selected_trial_id`。因此
  `generation_session_report.json` 的第一代 `generation_outcome="accepted"` 但
  `selected_lane_id=null`，而权威 ledger 的 `formal.selected_lane_id="lane-a"`、
  `formal.selected_trial_id="...generation-001-lane-a"`、`accepted=true`。
- `src/true_rl_agent/agent_core/generation.py:436-449` 在解析 selected state 之前，对所有已通过 screen 的 lane
  无条件执行 `PUBLIC_ACCEPT`。第二、三代 selection 均为
  `reason="no_guard_eligible_candidate"`、`selected_trial_id=null`，但每代仍执行两条 lane × `5` seeds。
- 两代报告各记 `eval_seed_units=16`，其中 `6` 是两条 `public_dev`、`10` 是两条 `public_accept`；两代合计
  `20` 个 no-selection public-accept seed-units，即 `1,440` episodes。四个 attempt 的 lease duration 合计
  `929,549,373,943 ns = 0.258208 allocated GPU-h`。
- 同一终态报告的 `immutable_prompt_payload_bytes=412,951` 只覆盖最终进入六条 trial 引用的 prompt；六个
  `RoundAuthoringJournal` 实际共有 `10` 次 provider prompt、合计 `682,434 B`。Gen1 lane-b 四次失败/重试
  payload 没有进入该字段，字段名表达的“immutable prompt 总量”与真实会话总量不一致。
- `selector_reversal.complete_dual_public_generation_count=1`、一致性分母也是 `1`；但 Gen2/Gen3 的四条
  候选都真实完成 public_dev 和 public_accept。由于两代最终为 no-selection，它们没有进入这项人工报告指标，
  所以该指标不能覆盖本轮已经付费的全部 dual-public 比较。

**非阻断事实**

- ledger/formal receipt 中的接受对象与判决完整；独立验收器从 journal 计到全部 `10` 次 prompt，provider 成本也
  按 `10` 次结算；额外 public-accept 没有进入 selection，所有成本仍在 `55 eval_seed_units` 总账与 GPU cap 内。
  这里分别是人工报告投影/覆盖缺口和已记账的非必要评测成本，不是证据污染。

## F6：`final_iteration` 训练仍保留全部中间 checkpoint，占终态树的绝大多数

**问题与证据**

- `configs/tasks/stair_go2_12dof_unified_protocol_v2.json:1` 同时声明
  `checkpoint_selector="final_iteration"`、`resume=false`、`checkpoint_save_interval_iterations=50`。
- 六个正式 train 的 RSL-RL 目录各有 `41` 个 `model_*.pt`（`model_0.pt` 至 `model_1999.pt`），每目录约
  `284,406,26x B`；单个最终模型为 `6,882,293 B`。
- 六个 RSL-RL 目录合计 `1,706,437,600 B`，占终态普通文件总量 `1,766,955,537 B` 的 `96.58%`；
  其中 `246` 个 checkpoint 文件合计 `1,693,030,794 B`。

**非阻断事实**

- `accounted_output_bytes` 未超过冻结输出上限，final checkpoint 的 checksum 与评测引用均完整；问题只体现为终态
  存储与同步成本。

## F7：`task_scales` 进入身份哈希，但没有行为消费者

**问题与证据**

- Go2 binding 声明 `task_scales={horizontal:1.0, vertical:0.589087}`。
- 生产源码中 `task_horizontal_scale/task_vertical_scale` 只出现在
  `runtime_projection.py:117-118`（字段）、`:230`（数值校验）、`:452-460`（序列化/身份）、
  `:730-737`（解析）和 `:833-834`（构造）；没有 terrain、curriculum、reward input、success/failure、command
  或 evaluation geometry 读取这两个字段。
- 测试侧唯一行为是 `tests/unit/test_stair_phase5_contract.py:114-115` 对数值本身做断言。真正的 Go2 几何缩放结果
  另行固化在训练/评测协议数字中。

**非阻断事实**

- 本轮所有实际训练与评测几何均来自冻结协议且身份一致；`task_scales` 当前是身份元数据而非运行控制量，不改变
  Phase 6 结果。

## F8：`TaskRegistry` / `BackendRegistry` 是未进入产品路由的平行脚手架

**问题与证据**

- `src/true_rl_agent/tasks/registry.py:8` 的 `TaskRegistry` 与
  `src/true_rl_agent/backends/registry.py:8` 的 `BackendRegistry` 在 `src/`、`scripts/` 中没有构造、注册或查询
  调用；活引用仅见 `tests/unit/test_plugin_registries.py:125,137`，以及
  `tests/unit/test_phase14_extension_docs.py:80` 把两者写成 extension registration surface。
- 实际产品入口统一走 `src/true_rl_agent/composition/defaults.py:19-28` 的
  `build_application_composition_registry()` 与 `TaskCompositionRegistry`；`run_loop.py`、`run_round.py`、
  `materialize_job.py`、Stair live/cold-start/reference 入口均使用这一条路。
- 因此仓库同时暴露了“测试/文档中的插件 registry”和“真实 composition registry”两套扩展表面，前者不会影响
  生产组合解析。

**非阻断事实**

- Phase 6 的 Go2 row、Stair plugin、runner/backend 都经活跃 `TaskCompositionRegistry` 成功解析并完成三代，
  两个旧 registry 没有进入本轮执行路径。

## F9：no-selection 记录中的最近失败源码没有进入下一代 source catalog

**问题与证据**

- `src/true_rl_agent/prompting/source_catalog.py:1635-1662` 的
  `_resolve_latest_failed_record_source()` 从 generation record 顶层 `candidate` 读取 artifact hash，并由
  `_candidate_package_root(record)` 要求唯一 candidate artifact root。
- Phase 6 三条权威 ledger record 的顶层都没有 `candidate` 字段；每条均有两个非空
  `trials[*].candidate`。第二代是合法 `no_selection`，两条 trial 分别为
  `go2_stair_wb_prog_gait_v2` 与 `go2_stair_contact_penalty_v1`，并各自有 lane-local candidate artifact。
- 第三代两条真实 `prompt_payload.json` 中，logical source
  `attempt/latest_failed/training_code/components.py` 均为
  `status="pending"`、`pending_reason="latest_failed_record_has_no_sealed_source_artifact"`、
  `source_owner="prior_ledger_record"`；上一代最近失败候选源码没有成为可见 source payload。
- 该 resolver 的 CodeGraph blast radius 只有一个内部 caller，未发现覆盖这一 generation-record 形状的测试。

**非阻断事实**

- 第三代仍收到 selector comparison、公开评测反馈、avoidance cards 与 accepted current-best 完整源码；缺失的是上一代
  no-selection 候选的精确失败源码，不影响 runner 权威、预算或本轮判决。

## F10：当前仓库仍暴露一组没有内部消费者的符号

**问题与证据**

- 在当前 `src/`、`scripts/`、`tests/` 全集内，以下 8 个常量都只有“定义 + `__all__`”两处命中：
  `STAIR_UNIFIED_TRAINING_PROFILE_RELATIVE_PATH`、`STAIR_UNIFIED_EVALUATION_SUITE_RELATIVE_PATH`、
  `STAIR_G1_37DOF_ROBOT_TASK_CELL_ID`、`STAIR_GO2_12DOF_ROBOT_TASK_CELL_ID`、
  `STAIR_GENERATION_LAUNCH_READINESS_FILENAME`、`STAIR_UNIFIED_PROTOCOL_V1_RELATIVE_PATH`、
  `STAIR_UNIFIED_ACCEPTANCE_PROTOCOL_RELATIVE_PATH`、`STAIR_VNEXT_REWARD_AUTHORITY_RELATIVE_PATH`。
- `_is_required_feedback_entry`、`_GATE_FIELDS`、`COMPONENT_ALLOWED_IMPORT`、`CodeChangeManifest` 与
  `AuthoringCandidateProvider` 在同一范围均只有声明处 1 次命中，没有内部调用、构造或类型引用。
- 对照项 `STAIR_UNIFIED_PROTOCOL_RELATIVE_PATH` 有 6 次命中，并被 active composition 与 entrypoint contract
  消费，说明上述计数不是把所有导出常量都误判为无消费者。

**非阻断事实**

- 这些符号没有进入 Phase 6 的 Go2 generation、runner、evaluation、decision 或 ledger 路径；当前仓库内的零引用
  不能证明不存在仓库外消费者，因此本轮只记录其维护表面，不把删除变成终态代码改动。

## F11：派生统计量的 receipt identity 尚未跨 Python 版本稳定

**问题与证据**

- 同一最终提交和同一 P6 原始产物使用项目解释器
  `/data/yuyangliu/envs/isaaclab/bin/python` 复验为 `status=pass, phase6_go=true`；使用服务器
  `/usr/bin/python3` `3.10.12` 则在 pilot selector replay 报
  `Phase 6 selector receipt differs from task-owned replay`。
- pilot 第一个 screen 的 `anchor_score_stats` 中，stored/rebuilt 的 `standard_error` 为
  `0.0092455678158868 / 0.009245567815886802`，`std` 为
  `0.016013793201939554 / 0.016013793201939557`；seed macro scores、count、mean 与协议 hash 一致，差异为
  `statistics.stdev` 的 1 ULP，随后改变 `screen_evidence_hash`。
- 绕过该处作只读定位后，formal sealed summary 仍出现
  `standard_error 0.0021486745397333576 / 0.002148674539733358`、
  `std 0.005684858480550637 / 0.005684858480550638`，并改变 `task_evidence_hash`。因此 ordered-summary
  的 `1e-12` 语义容差已生效，但 identity hash 仍按原始浮点字节计算。

**非阻断事实**

- 训练、评测、判决与正式复验均使用声明的项目 IsaacLab 环境且完整通过；系统 Python 不是冻结执行环境，
  该差异不改变任何候选排序、接受决定或 current-best。它限制的是跨解释器 receipt hash 重建能力，不能把本轮
  GO 表述为任意 Python 版本下字节级重放。


## F12：容量投影先削经验证据、后削静态契约文本，导致三代中两代丢失诊断细节

**问题与证据**

- `src/true_rl_agent/tasks/stair_climbing/prompt_adapter.py:1695-1799` 的
  `_compact_public_feedback_to_fit_task_sections()` 共有五级：命中即返回、`_compact_required_feedback_entry`、
  `_compact_historical_comparison_entry`、逐条弹出历史 entry 的 `while` 循环，最后才是
  `_minimum_core_stair_prompt_sections()`。**第二至第四级只作用于
  `public_feedback_summary.feedback_entries`**，不触碰 `reward_inputs_contract`、`task`、
  `public_eval_contract`、`selection_guidance` 这些每轮不变的静态契约文本。
- `prompt_adapter.py:1803-1979` 的 `_minimum_core_stair_prompt_sections()` 是**原子**投影：一次性重建全部段，
  之后才判断是否需要再升级到 `minimum_core_single_entry`，没有“只削到刚好够”的行为。它同时省略
  `reward_field_descriptions`、`duplicated_eval_identity_detail`、`guidance_prose_detail`（静态）与
  `feedback_cell_grids`、`feedback_diagnostic_detail`、`runtime_activation_detail`、
  `historical_feedback_entries`（经验证据）。
- 三代真实 lane-a prompt 的 provider task-section 实测（目标
  `MAX_TASK_PROMPT_SECTIONS_BYTES - _TASK_PROMPT_RESERVED_HEADROOM_BYTES = 32,256 B`）：

  | generation | 投影前 | 实际送出 | projection_level |
  |---|---:|---:|---|
  | 1 | `31,875` | `31,875` | 无（余量仅 `381 B`） |
  | 2 | `36,060` | `25,618` | `minimum_core`（超限 `3,804 B`） |
  | 3 | `36,048` | `25,606` | `minimum_core`（超限 `3,792 B`） |

  第 1 代之所以不降级，只是因为它还没有 `latest_generation_trial_comparisons`；该段自第 2 代起恒定存在，
  因此**降级不是随轮数累积的偶发压力，而是从第 2 代起的稳态**。`feedback_entries` 三代都是 `1` 条，
  增长不来自历史堆积。
- 直接调用生产函数测量“只压静态契约文本”的收益（以 gen1 未投影内容为输入）：
  `_minimum_core_reward_inputs_contract` 省 `6,992 B`、`_minimum_core_task_section` 省 `814 B`、
  `_minimum_core_selection_guidance` 省 `0 B`，合计 **`7,806 B` > 超限 `3,804 B`**。即第 2、3 代
  只压缩静态契约文本即可入界，并保留全部 stratum 诊断、cell grid 与 runtime activation 细节，还余
  `4,002 B`。
- 实际后果可见：第 2、3 代 `runtime_feedback_summary` 均为
  `status="omitted_for_capacity"`、`reason="minimum_core_projection"`；`reward_inputs_contract`
  由 `14,846 B` 压到 `7,854 B`。而唯一被接受的候选出自未降级的第 1 代。
- 该投影不进入任何冻结身份：`tasks/base.py:110-134` 的 `resolved_task_protocol_hash()` 与
  `composition/base.py:283-320` 的 `genesis_identity_snapshot()` 都不含 prompt 内容；
  `generation_session.py:1355-1373` 的 `bound_contract_hashes`（authoring_policy / continuation_policy /
  evaluation_cost / evaluation_protocol / exploration_projector / final_confirmation_plan /
  provider_policy / resource_plan / retry_policy / selector / session_cost）同样不含 prompt 投影。

**非阻断事实**

- `capacity_projection_receipt` 完整记录 `original_bytes`、`projected_content_without_receipt_bytes`、
  `original_content_hash` 与 `omitted` 列表，降级是有 receipt 的显式事实，不是静默截断；三代 selector、
  guard、acceptance 与 current-best 均由 runner 从权威 receipt 判定，不依赖 prompt 内容。本项影响的是后续
  LOOP 的改进效率，不影响本轮任何判决的成立。按当前形状，`N` 代实验中会有 `N-1` 代运行在 `minimum_core`。

## F13：Genesis 身份含执行设备与绝对解释器路径，且不持久化身份快照体

**问题与证据**

- `src/true_rl_agent/composition/base.py:283-320` 的 `genesis_identity_snapshot()` 把
  `task_defaults` 整体纳入身份；Stair 的 `task_defaults` 含
  `execution_device`（`tasks/stair_climbing/composition.py:88`）与
  `isaac_python`（`composition.py:85`）。因此同一实验换 GPU 编号或换机器都会得到不同的
  `runtime_snapshot_hash`。当前 G1-V2 快照中该路径为 `/data/yuyangliu/envs/isaaclab/bin/python`。
- `src/true_rl_agent/agent_core/session_genesis.py:697-707` 的 `SessionGenesisArtifact` 只有
  `runtime_snapshot_hash` 字段，`to_dict()` 与 `_hash_payload()` 也只写该摘要；**产生它的快照体不落盘**。
  P5 两行 `session_genesis.json` 顶层键中与身份相关的只有 `runtime_snapshot_hash`。
- 实测可复现：对 P5 G1 genesis 目录运行生产工具
  `scripts/create_stair_reference_genesis.py --mode load-only`，使用参数默认的 `cuda:0` 报
  `ValueError: session Genesis identity drift`；加 `--execution-device cuda:1` 后
  `status=pass`、`session_genesis_hash=080154a88c0721c0a6a4ebe99d1ab32309f83f448d428dc5472ac3aa4be41fcc`
  与记录一致。P5 两行是并行资格化的，Go2 在 `cuda:0`、G1 在 `cuda:1`。
- 设备敏感本身是既定且有测试的设计（`tests/unit/test_reference_genesis.py:2155`
  `test_rejects_public_execution_device_drift_on_reload`）。问题在于：不匹配时错误只有
  `session Genesis identity drift` 一句，因为快照体不存在，**无法定位是哪个字段漂移**；GPU 编号属于放置
  决策而非科学身份，且已由 `gpu_process_binding.resource_binding_hash` 单独记录。

**非阻断事实**

- Go2 行的 `fd56b175...` 在 P5 genesis、P6 pilot、P6 validation 三处一致，导入链与 Phase 6 结论不受影响；
  G1-V2 的 P5 Genesis 在正确设备参数下于终态提交仍可复验通过，G1 行没有退化。本项限制的是身份漂移的
  可诊断性与跨机复现能力。

## F14：cost contract 靠 `isinstance` 在 20 处分派，且 V2 判定隐含“V2 或更新”

**问题与证据**

- `generation_session_cost.py:1541-1542` 与 `:1679-1680` 显示
  `StairGenerationSessionCostContractV3` 与 `V4` 均继承 `StairGenerationSessionCostContractV2`；
  `StairGenerationSessionCostContract`（pilot）是独立类。四者按
  `(session_mode, imports_reference_genesis)` 在 `generation_session.py:1175-1270` 一处四路分支选出。
- 版本判定散落在 **20 处**：`generation_session.py` 12 处、`generation_session_cost.py` 8 处，按版本计
  `V2=10`、`V3=6`、`V4=4`；另有 3 处针对 `StairGenerationSessionCostContractLike` 协议的类型校验。
- 由于继承关系，`isinstance(x, ...ContractV2)` 对 V3、V4 实例同样为真。因此
  `generation_session.py:1381,4839,5970` 与 `generation_session_cost.py:2147,2171,2177,2365,2401,2477`
  共 10 处的实际语义是“V2 或任何后继版本”，这一优先级没有在任何类型或文档中声明，只能靠通读全部分支还原。
- 同类风险已经发生过一次：`agent_core/session_genesis.py` 的 schema v5 需要写入 13 处成员判定，其中
  `backend_execution_environment` 回读一处被漏掉，由单独提交 `6508354` 补上。
- `480d999` 新建 V4 而非扩展 V2 本身是正确的：V2 的 `contract_hash` 已绑定既有 receipt，加字段会使其失效。
  问题不在新建版本，而在分派方式。

**非阻断事实**

- 本轮 pilot 与 validation 的成本合同选择、operation reserve、admission 与关账全部正确执行，
  `19/19` attempts 关闭、open reservations 为 `0`，独立验收器成本重放通过。这是新增 session mode 或新增
  任务列时的正确性风险面，不是已发生的错误。

## F15：会话编排的三分之二结构通用却绑在 Stair 类型上，任务轴尚不满足计划的扩展门

**问题与证据**

- 对 `src/true_rl_agent/tasks/stair_climbing/generation_session.py` 的 136 个顶层符号按函数体内容分类：

  | 类别 | 符号数 | 行数 | 占比 |
  |---|---:|---:|---:|
  | A 含真实 Stair 领域语义（`c0_id`/`c1_id`/`riser`/`tread`/`j_ood`/`stratum`/`selector`/`acceptance` 等） | `17` | `2,441` | `27.1%` |
  | B 结构通用，仅出现 `Stair*` 类型名或 `STAIR_*` 常量 | `91` | `6,000` | `66.6%` |
  | C 完全通用 | `28` | `574` | `6.4%` |

- B 类中最大的符号正是会话编排本身：`_run_bounded_generation_session`（`488` 行）、
  `_build_formal_report`（`278`）、`run_stair_generation_pilot`（`255`）、`_run_one_generation`（`246`）、
  `_assert_provider_request_budget`（`200`）、`collect_actual_cost_usage`（`177`）、
  `_preflight_reference_v3_round_trip`（`176`）、`_finalize_formal_generation_session`（`157`）。
- 对照共享层：`src/true_rl_agent/agent_core/generation_*.py` 合计 `4,346` 行，只提供契约、生命周期与
  receipt 类型；`src/true_rl_agent/tasks/push_recovery/` 全部 `6,361` 行且**没有 generation session 模块**。
  Stair 侧 `generation_session.py` 与 `generation_session_cost.py` 合计 `11,894` 行。
- 上位计划 `PLAN/Variant_construction/STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md:695` 规定“新增任务列只允许提交
  TaskPlugin、任务分布/评测/acceptance 和薄 bindings。任何要求复制 runner、ledger、LOOP 或按 terrain 建 cell
  的方案，默认视为架构回退”；同文件 `:609` 把“增加任务列不修改 RobotProfile/AgentCore”列为 Phase 7 GO 门。
  按当前形状，新增任务列需要另行实现上述 B 类编排。
- 需要注明的方法学边界：B 类是**静态词法分类**，只说明这些符号体内不出现 Stair 领域词汇，不等于抽取成本低；
  Stair 类型签名可能编码了真实耦合。该数字用于标记 Phase 7 任务轴门的风险量级，不构成重构方案。

**非阻断事实**

- 机器人轴的对应主张已被本轮完整证明：Go2 在产品源码中的全部足迹为 6 个文件
  （`robots/{isaaclab_locomotion,defaults,rsl_rl_runtime}`、
  `tasks/stair_climbing/{row_catalog,robot_binding,reference_programs/go2_...}`），
  共享层 97 个文件中机器人词汇命中 `0`，`robots/` 中任务词汇命中 `0`，仓库无 Go2 专用 LOOP/launcher。
  Phase 6 的结论对象是行轴，本项只影响后续任务列扩展的工作量与 Phase 7 的措辞边界。

## F16：仅供审计使用的 replay 加载器位于产品任务模块内

**问题与证据**

- `src/true_rl_agent/tasks/stair_climbing/generation_session.py:8970-9237` 的
  `load_completed_stair_generation_session()` 共 `268` 行，配套只读 dataclass
  `CompletedStairGenerationSession`（`:470-490`，`21` 行）。
- 该函数在 `src/` 与 `scripts/` 内**没有任何调用者**，仅有定义处与 `__all__:9323` 两处命中。实际消费者是
  `PLAN/Variant_construction/P6/verify_phase6.py:2269,2273`，以及
  `tests/integration/test_stair_generation_session.py:1526` 与
  `tests/unit/test_stair_generation_session.py:79,124`。
- 引入该函数的提交 `b590e57`（“Add independent Phase 6 session replay verifier”）向产品模块新增 `331` 行；
  `generation_session.py` 因此由 `8,771` 行增至 `9,326` 行。
- 其 docstring 已明确它不调用 live/resume 入口、不构造 `StairGenerationSessionFoundation`、
  只加载已发布契约并复验身份链，接受判定仍留在验收器 —— 设计意图是清楚的，问题只在放置位置。

**非阻断事实**

- 独立验收器在项目解释器下 `status=pass, phase6_go=true`，产品 LOOP 不经过该函数，本轮执行与判决不受影响；
  各阶段验收器复用 `true_rl_agent` 类型是既有惯例（P5 验收器 import 16 个产品模块，P6 为 25 个），
  本项记录的是产品模块承担审计职责带来的维护面增长。
