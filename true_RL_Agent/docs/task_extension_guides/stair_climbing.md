# Stair Climbing Task Extension Guide

This guide describes the calibrated unified V3 Stair cells. Frozen Phase 0 and legacy contracts are identified separately below; their historical geometry and outcome rules are not the current task contract.

## Current Cells and Authorities

The active rows are:

| Cell | Task protocol | Robot binding |
|---|---|---|
| `stair.g1_37dof.unified_v3` | [G1 protocol](../../configs/tasks/stair_unified_protocol_v3.json) | [G1 binding](../../configs/tasks/stair_g1_37dof_unified_binding_v3.json) |
| `stair.go2_12dof.unified_v3` | [Go2 protocol](../../configs/tasks/stair_go2_12dof_unified_protocol_v3.json) | [Go2 binding](../../configs/tasks/stair_go2_12dof_unified_binding_v3.json) |

The task-owned [row catalog](../../src/true_rl_agent/tasks/stair_climbing/row_catalog.py) identifies active rows and frozen predecessors. Each selected protocol's `component_refs` resolves its training profile, ordered evaluation suite, acceptance protocol, and reward authority. Use those resolved authorities rather than importing thresholds from another robot or a historical report.

The application registry retains the lazy Stair provider and selects Push as its default task; the Push provider now defaults to G1 force-evaluation V3. Importing `true_rl_agent.tasks.stair_climbing` is inert; it does not import Isaac Lab or register a Gym environment. Registry defaults do not establish Push reference qualification or authorize a generation session. Current Push rows and historical compatibility are described in the [Push guide](push_recovery.md). Formal Stair generation uses the launcher and readiness path in the [runbook](../stair_fresh_15round_runbook.md).

## Current Boundary

- `TaskPlugin` and task config own Stair protocol semantics.
- The task package owns the failure classifier, metrics, recorder, seed/cell resolver, and prompt/memory adapters.
- Robot assets, morphology, joints, bodies, and state/action adapters belong to the robot layer; composition binds them to task requirements.
- `BackendAdapter`, Runner, and AgentCore consume generic jobs and bound identities, not their own copies of Stair outcome definitions.
- A runner-owned approved extension is required before granting any new observation, terrain, curriculum, or action-space capability.

AgentCore, Runner, DecisionEngine, and ledger must not acquire robot-specific thresholds or task-specific failure rules. Stair and Push use the P4 shared generation host, which receives the selected task's phased evaluation adapter. This engineering integration does not establish physical reference qualification; the current Push qualification work remains open under P6.

## Training and Candidate Authority

The unified protocol freezes training budget, random initialization, checkpoint selection, runtime controls, and training identity. Reference and candidates follow the same fixed random initialization protocol; an accepted checkpoint is not the next candidate's training start.

Stair grants `complete_reward_program`: a candidate replaces the complete `training_code/components.py` reward program within its selected task-owned input ABI and reward authority. `RewardTerm` and `DiagnosticTerm` remain shared source structures; historical additive Push wrappers are not the Stair complete-reward implementation. Active Push rows also use their own complete reward contract, independently of the force evaluation protocol.

PPO, episode accumulation, logs, and runtime evidence consume the same authoritative reward result. Candidate source and runtime activation evidence establish what executed; rationale does not grant authority or prove a mechanism.

Candidates cannot modify PPO, budget, initialization, checkpoint selection, evaluation, outcome definitions, acceptance, or observation/action space. Terrain and curriculum permissions come from the selected protocol; the current Stair candidate has no LLM curriculum capability. Runtime controls must be read from their task-owned contract without semantic fallbacks.

## Active Outcome Semantics

For unified V2/V3, [StairClimbingTaskConfig](../../src/true_rl_agent/tasks/stair_climbing/config.py) constructs `StairEvaluationSemantics.from_task_outcomes(...)` from the selected task protocol's `episode_outcomes`. The standalone `stair_evaluation_semantics_v7.json` serves the legacy path; it is not the unified task authority.

Both V3 cells retain `first_valid_top_arrival_latched`. The first valid top arrival must satisfy the selected cell's platform bounds, stable orientation, and valid support requirements. Pre-arrival `forbidden_contact` and `off_course` latches block success. A valid arrival closes the outcome immediately: there is no subsequent 20-step stable-top success gate or post-success reclassification.

The current primary reasons are:

- `success`
- `forbidden_contact`
- `off_course`
- `base_orientation_instability`
- `slip`
- `insufficient_clearance`
- `timeout`

There is no independent `fall` primary reason in unified V2/V3. Failure priority, diagnostic versus terminal classification, robot-specific clearance/support tolerances, and sustained-condition thresholds remain in the protocol. Timeout comes from the ordered evaluation suite's cell horizon. Do not duplicate these definitions in the runner or an extension guide.

## Active Evaluation and Acceptance

[G1 suite](../../configs/tasks/stair_unified_evaluation_suite_v2.json) and [Go2 suite](../../configs/tasks/stair_go2_12dof_unified_evaluation_suite_v2.json) define ordered C0/C1/C2-related strata, cohort geometry, seed counts, per-cell horizons, and family qualification. Their `score_contract` defines:

- cell score: `0.5 × success_rate + 0.5 × mean_goal_progress_fraction`;
- `J_ID = 0.5 × S_C0-ID + 0.5 × S_C1-ID`;
- `J_OOD`: equal-weight aggregation over scalar, C1 frontier and C2 families; all must qualify;
- `J_primary = J_OOD`; ID and phase control are reported as capability guards.

These are not the old seven-bin or 35-cell macro-success objectives. Report success n/N and rate separately from J_primary and stratum/family scores. A higher aggregate does not establish improvement in every family or on another robot.

The selected component is [unified acceptance v7](../../configs/tasks/stair_unified_acceptance_protocol_v7.json). It compares paired J_primary seed samples using the headroom/noise rule, with separate `j_ood`, `c0_id`, `c1_id`, `joint_ood`, and `phase_control` guards. For candidates, the task also requires sealed J to meet the absolute floor and be no more than `0.05` below public-accept J; assembly and receipt replay enforce the same predicate. Reference sealed results are telemetry, not candidate acceptance. The acceptance protocol owns the thresholds; the task owns score construction and ordering. The shared evaluation layer consumes bound score samples and guards rather than implementing Stair geometry.

P3 retains raw episode facts in ordered summary v4 / unified summary v3. Completion time is diagnostic only. The frozen P3 qualification uses a measured 20–80% success transition in every objective stratum on public-accept. This criterion participates in its learnable-cell/family-mask and scoring contract; the current Stair suite, mask, scores and original qualification decisions remain unchanged. The 2026-09-22 [reference admission revision](../../PLAN/2new_world/P6/REFERENCE_ADMISSION_REVISION_ZH.md) changes only new Push force-reference admission, not the Stair scoring domain. The broader goal of demonstrated basic ability and improvement space can guide a separately declared future pool; it does not cancel existing frozen criteria. Selection uses the separate public calibration pool, never dev/accept/sealed results. Saturated ID and phase control are outside this study's primary objective. Robot facts set initial terrain search scales; that robot's fixed reference locates the actual difficulty transition. Training support and absolute course dimensions remain declared training conditions. See the [design and limits](../../PLAN/2new_world/P3/IMPLEMENTATION_PLAN_ZH.md).

For C2, dev/accept/sealed keep the selected calibration base, amplitude, frequency and mesh resolution. Distinct surface phases and seeds provide the formal instances. The exact paired ID flat control is deliberately reused and is outside J_primary; other cross-cohort geometry duplicates remain invalid. Study final retains its separately committed base-geometry extrapolation, so it tests a broader unseen combination. Revising a generator after auditing holdout results creates a new comparison domain and does not make reused geometry unseen again. See the [cohort audit and revision](../../PLAN/2new_world/P3/COHORT_ALIGNMENT_ZH.md).

Evaluation identities bind the selected protocol and components. Historical results keep their original bindings; changing a protocol does not reinterpret old ledger entries or qualify an old Genesis for the new contract.

## Public Memory Summary

The task-specific public summary adapter projects validated public Stair metrics and bounded recent feedback. It must preserve the score definition, available cell/stratum evidence, primary outcomes, progress, and validated reward diagnostics within the public-context budget.

It must not expose hidden evaluation results, private seeds, checkpoint selectors, or runner-owned terrain-generation internals. AgentCore uses the generic public context interface; Stair-specific interpretation belongs in the Stair TaskPlugin package. Missing evidence is not a zero-valued measurement.

V2 cells remain available for existing session compatibility and retain their original schemas, scores and bindings. The three new-run entrypoints default to G1 V3 through the task row catalog; select Go2 V3 explicitly. Resuming a historical session requires its original cell. P4 must still close the orchestration and session cost contract before a new formal search.

## Historical Compatibility Evidence

The historical baseline manifest [isaaclab_g1_rough_stair_v1.json](../../configs/baselines/isaaclab_g1_rough_stair_v1.json), [Phase 0 descriptor](../../configs/tasks/stair_extension_schema.json), [V1 protocol](../../configs/tasks/stair_protocol_v1.json), and [Phase 0 report](../../PLAN/stair/phase0_report.json) retain their original identities.

The byte-frozen `scripts/isaaclab_stair_phase0_probe.py` targets the historical RSL-RL 3 runtime. Do not run it on the current Sim 6 / Lab 3 / RSL-RL 5 backend: its `runner.load()` has the old checkpoint semantics. Keep its bytes unchanged for historical verification; current evaluation uses `scripts/isaaclab_stair_rsl_rl_eval.py` and the shared inference-checkpoint loader.

The official G1 rough-locomotion checkpoint was admitted only for compatibility and evaluation-chain smoke evidence. Its frozen probe had 0/56 Stair successes; it does not become current-best, Session Genesis, or a warm-start for complete-reward training. Checkpoint safety/structure and runtime compatibility remain separate from task viability.

The old seven-height bins and parameter-only stair-height classifications belong to that compatibility contract. They cannot authorize a unified V2 reward generation. The frozen descriptor must stay byte-stable because historical evidence binds its hash; extending a task requires its current TaskPlugin, task config, robot binding, and shared contracts, not a second authority in this descriptor.

## Video and Further Integration

The mainline uses one Sim 6 / Lab 3 / PhysX / RSL-RL 5 runtime for training and evaluation. Native recording is a presentation option of the same eval executor (`--video-dir`, with the `public_dev` phase), including the resolved robot, course, checkpoint and episode evidence. The old independent checkpoint-video exporter has been removed. [P2](../../PLAN/2new_world/P2/ACCEPTANCE_ZH.md) completed full-budget training and evaluation/native recording of its checkpoint. Numerical equivalence with the old backend was not established. After fixing the C2 calibration base across formal cohorts, [P3](../../PLAN/2new_world/P3/ACCEPTANCE_ZH.md) records both robots passing the original public-accept qualification rule. It also reports remaining differences in cohort difficulty; qualification does not establish identical discrimination on every split. Each cell needs valid reference qualification, and P4 must update measured session costs before a new formal search. Existing diagnostic videos retain their original protocol meaning.

The [new mainline plan](../../PLAN/2new_world/NEW_WORLD_PHASED_PLAN_ZH.md) covers backend alignment, metric discrimination/generalization, and Extension A: Go2 × Push Recovery. This guide documents the current boundary; it does not claim those phases are already implemented.


## 接入新机器人并标定 Stair

1. 在机器人层注册实际资产、关节/支撑角色、运行时投影与 RobotProfile 事实。不要从名字猜尺寸，不把足中心间距当作足底碰撞长度。当前 L 是标称 root 到支撑 body 中心的垂直距离，只用于初始难度尺度。
2. 在 Stair 侧确定该机器人的训练支持和任务容差，薄 binding 连接角色与执行来源。训练踏深、课程范围、平台、宽度、级数等属于明确的训练/任务条件；改变训练条件必须重新训练 reference。无需手写旧版评测几何。
3. 使用 `scripts/prepare_stair_evaluation.py generate-suite --training-profile <profile.json> --robot-profile-id <registered-id> --output <suite.json>` 生成公开粗题池。新 suite、acceptance 和 protocol 在同一配置目录冻结。对已有训练行的评测后继，可用 `prepare --source-cell-variant-id <source> --cell-variant-id <registered-successor> --suite-path <suite.json> --acceptance-protocol <acceptance.json>` 建立 draft；它从源训练合同生成 suite，不迁移某个机器人专用模板。
4. 用该机器人的固定、完整训练 checkpoint 执行 `scripts/calibrate_stair_reference.py --cell <draft-cell> --checkpoint <best.pt> --device cuda:0 --output <new-output>`。先冻结样本量、范围和停止规则；通过 `prepare_stair_evaluation.py revise --measurement <result.json> --purpose refine --decision <decision.json>` 做一次公开细化，再实测细化题池，最后以 `--purpose formal` 冻结正式 cohort。C2 固定标定基底、留出表面相位；配对平面控制复用，不能宣称所有几何均独立。
5. 核对每个目标 stratum 的成功转折、米制与相对尺度、C1 pattern 覆盖和 C2 对应平面控制。全满/全零时不能用新指标救资格；应记录 NO-GO 后论证新的地形范围。正式 suite 冻结并将可执行合同签为 active 后，重新执行生产 reference qualification；active 不等于 reference 资格通过，不能边看 accept/sealed 边改题。
6. reference 资格通过才建立新研究的 Genesis 并使用相应入口；同一研究的题池、seed 与门固定。旧数据保留旧身份，已消费的 final 探针不得充当新研究的未见留出。新机器人需要真实物理标定；合成 L 范围测试只能验证生成器。精确的当前规则和限制见 [P3 卷间构造修订](../../PLAN/2new_world/P3/COHORT_ALIGNMENT_ZH.md)。
