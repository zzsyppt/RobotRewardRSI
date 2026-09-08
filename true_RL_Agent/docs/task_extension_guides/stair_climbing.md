# Stair Climbing Task Extension Guide

This guide describes the active unified V2 Stair cells. Frozen Phase 0 and legacy contracts are identified separately below; their historical geometry and outcome rules are not the current task contract.

## Current Cells and Authorities

The active rows are:

| Cell | Task protocol | Robot binding |
|---|---|---|
| `stair.g1_37dof.unified_v2` | [G1 protocol](../../configs/tasks/stair_unified_protocol_v2.json) | [G1 binding](../../configs/tasks/stair_g1_37dof_unified_binding_v2.json) |
| `stair.go2_12dof.unified_v2` | [Go2 protocol](../../configs/tasks/stair_go2_12dof_unified_protocol_v2.json) | [Go2 binding](../../configs/tasks/stair_go2_12dof_unified_binding_v2.json) |

The task-owned [row catalog](../../src/true_rl_agent/tasks/stair_climbing/row_catalog.py) identifies active rows and frozen predecessors. Each selected protocol's `component_refs` resolves its training profile, ordered evaluation suite, acceptance protocol, and reward authority. Use those resolved authorities rather than importing thresholds from another robot or a historical report.

The application registry retains the lazy Stair provider and legacy Push default. Importing `true_rl_agent.tasks.stair_climbing` is inert; it does not import Isaac Lab or register a Gym environment. This compatibility registry behavior does not establish a Push reference-program generation session. Formal Stair generation uses the launcher and readiness path in the [runbook](../stair_fresh_15round_runbook.md).

## Current Boundary

- `TaskPlugin` and task config own Stair protocol semantics.
- The task package owns the failure classifier, metrics, recorder, seed/cell resolver, and prompt/memory adapters.
- Robot assets, morphology, joints, bodies, and state/action adapters belong to the robot layer; composition binds them to task requirements.
- `BackendAdapter`, Runner, and AgentCore consume generic jobs and bound identities, not their own copies of Stair outcome definitions.
- A runner-owned approved extension is required before granting any new observation, terrain, curriculum, or action-space capability.

AgentCore, Runner, DecisionEngine, and ledger must not acquire robot-specific thresholds or task-specific failure rules. The current formal generation assembly still has Stair coupling; extending it to Go2 Push requires explicit integration work, not only a new configuration name.

## Training and Candidate Authority

The unified protocol freezes training budget, random initialization, checkpoint selection, runtime controls, and training identity. Reference and candidates follow the same fixed random initialization protocol; an accepted checkpoint is not the next candidate's training start.

Stair grants `complete_reward_program`: a candidate replaces the complete `training_code/components.py` reward program within its selected task-owned input ABI and reward authority. `RewardTerm` and `DiagnosticTerm` remain shared source structures; additive Push wrappers are not the Stair complete-reward implementation.

PPO, episode accumulation, logs, and runtime evidence consume the same authoritative reward result. Candidate source and runtime activation evidence establish what executed; rationale does not grant authority or prove a mechanism.

Candidates cannot modify PPO, budget, initialization, checkpoint selection, evaluation, outcome definitions, acceptance, or observation/action space. Terrain and curriculum permissions come from the selected protocol; the current Stair candidate has no LLM curriculum capability. Runtime controls must be read from their task-owned contract without semantic fallbacks.

## Active Outcome Semantics

For unified V2, [StairClimbingTaskConfig](../../src/true_rl_agent/tasks/stair_climbing/config.py) constructs `StairEvaluationSemantics.from_task_outcomes(...)` from the selected task protocol's `episode_outcomes`. The standalone `stair_evaluation_semantics_v7.json` serves the legacy path; it is not the active unified V2 authority.

Both active cells use `first_valid_top_arrival_latched`. The first valid top arrival must satisfy the selected cell's platform bounds, stable orientation, and valid support requirements. Pre-arrival `forbidden_contact` and `off_course` latches block success. A valid arrival closes the outcome immediately: there is no subsequent 20-step stable-top success gate or post-success reclassification.

The current primary reasons are:

- `success`
- `forbidden_contact`
- `off_course`
- `base_orientation_instability`
- `slip`
- `insufficient_clearance`
- `timeout`

There is no independent `fall` primary reason in unified V2. Failure priority, diagnostic versus terminal classification, robot-specific clearance/support tolerances, and sustained-condition thresholds remain in the protocol. Timeout comes from the ordered evaluation suite's cell horizon. Do not duplicate these definitions in the runner or an extension guide.

## Active Evaluation and Acceptance

[G1 suite](../../configs/tasks/stair_unified_evaluation_suite_v1.json) and [Go2 suite](../../configs/tasks/stair_go2_12dof_unified_evaluation_suite_v1.json) define ordered C0/C1/C2-related strata, cohort geometry, seed counts, per-cell horizons, and family qualification. Their `score_contract` defines:

- cell score: `0.5 × success_rate + 0.5 × mean_goal_progress_fraction`;
- `J_ID = 0.5 × S_C0-ID + 0.5 × S_C1-ID`;
- `J_OOD`: equal-weight aggregation over qualified nonempty families;
- `J_primary = (2 × J_OOD + J_ID) / 3`.

These are not the old seven-bin or 35-cell macro-success objectives. Report success n/N and rate separately from J_primary and stratum/family scores. A higher aggregate does not establish improvement in every family or on another robot.

The selected component is [unified acceptance v6](../../configs/tasks/stair_unified_acceptance_protocol_v6.json). It compares paired J_primary seed samples using the headroom/noise rule, with separate `j_ood`, `c0_id`, and `c1_id` guards, bootstrap floors, and public-to-sealed consistency policy. The acceptance protocol owns these thresholds; the task owns score construction and ordering. The shared evaluation layer consumes bound score samples and guards rather than implementing Stair geometry.

Evaluation identities bind the selected protocol and components. Historical results keep their original bindings; changing a protocol does not reinterpret old ledger entries or qualify an old Genesis for the new contract.

## Public Memory Summary

The task-specific public summary adapter projects validated public Stair metrics and bounded recent feedback. It must preserve the score definition, available cell/stratum evidence, primary outcomes, progress, and validated reward diagnostics within the public-context budget.

It must not expose hidden evaluation results, private seeds, checkpoint selectors, or runner-owned terrain-generation internals. AgentCore uses the generic public context interface; Stair-specific interpretation belongs in the Stair TaskPlugin package. Missing evidence is not a zero-valued measurement.

## Historical Compatibility Evidence

The historical baseline manifest [isaaclab_g1_rough_stair_v1.json](../../configs/baselines/isaaclab_g1_rough_stair_v1.json), [Phase 0 descriptor](../../configs/tasks/stair_extension_schema.json), [V1 protocol](../../configs/tasks/stair_protocol_v1.json), and [Phase 0 report](../../PLAN/stair/phase0_report.json) retain their original identities.

The official G1 rough-locomotion checkpoint was admitted only for compatibility and evaluation-chain smoke evidence. Its frozen probe had 0/56 Stair successes; it does not become current-best, Session Genesis, or a warm-start for complete-reward training. Checkpoint safety/structure and runtime compatibility remain separate from task viability.

The old seven-height bins and parameter-only stair-height classifications belong to that compatibility contract. They cannot authorize a unified V2 reward generation. The frozen descriptor must stay byte-stable because historical evidence binds its hash; extending a task requires its current TaskPlugin, task config, robot binding, and shared contracts, not a second authority in this descriptor.

## Video and Further Integration

The mainline uses one Sim 6 / Lab 3 / PhysX / RSL-RL 5 runtime for training and evaluation. Native recording is a presentation option of the same eval executor (`--video-dir`, with the `public_dev` phase), including the resolved robot, course, checkpoint and episode evidence. The old independent checkpoint-video exporter has been removed. [P2](../../PLAN/2new_world/P2/ACCEPTANCE_ZH.md) completed full-budget training and evaluation/native recording of its checkpoint. Numerical equivalence with the old backend was not established; P3 must qualify the new reference, and P4 must update session costs before a new formal search. Existing diagnostic videos retain their original protocol meaning.

The [new mainline plan](../../PLAN/2new_world/NEW_WORLD_PHASED_PLAN_ZH.md) covers backend alignment, metric discrimination/generalization, and Extension A: Go2 × Push Recovery. This guide documents the current boundary; it does not claim those phases are already implemented.
