# Stair Climbing Task Extension Guide

Phase 0 provides immutable compatibility evidence for the official G1 rough-locomotion checkpoint. Phase A and Phase B now add the shared reference-program contracts and a task-owned staged production chain: `StairClimbingTaskConfig`, `TaskPlugin`, one composition provider, dedicated train/eval entrypoints, seven-bin evaluation, failure classification, public metrics, prompt memory, and runner-owned evidence projection.

Stair is deliberately absent from the default application registry. It can only be selected through the explicit staging registry with a real official-upstream `ReferenceProgramManifest`. Importing `true_rl_agent.tasks.stair_climbing` remains inert and does not import Isaac Lab or register a Gym environment.

The frozen assets are:

- `configs/tasks/stair_extension_schema.json`
- `configs/tasks/stair_protocol_v1.json`
- `configs/baselines/isaaclab_g1_rough_stair_v1.json`
- `scripts/isaaclab_stair_phase0_probe.py`
- `src/true_rl_agent/tasks/stair_climbing/isaaclab_phase0_probe.py`
- `PLAN/stair/phase0_report.json`

## Current Boundary

The production pieces remain task-owned:

- `TaskPlugin` and task config own Stair protocol semantics;
- the task package owns the failure classifier, metrics, recorder, seed/bin resolver, and prompt/memory adapters;
- `BackendAdapter`, Runner, and AgentCore see only generic jobs plus opaque profile/protocol hashes;
- a runner-owned approved extension is required for any future observation, terrain, curriculum, or action-space capability.

AgentCore, Runner, DecisionEngine, and ledger should not change because of stair task semantics. They already operate on TaskSpec, TrainJobSpec, EvalJobSpec, CandidateSurfacePolicy, runtime activation evidence, and public ledger records.

## Phase 0 Compatibility Checkpoint

The pinned checkpoint is represented by a historical baseline manifest, not hard-coded in AgentCore:

- Isaac Lab 2.3.0 official `Isaac-Velocity-Rough-G1-v0` RSL-RL checkpoint.
- The revised complete-reward loop does not warm-start from this checkpoint. Every R0/R1/... policy is trained from the same fixed random initialization protocol.
- Admission is split into checkpoint safety/structure, exact runtime robot/observation/action/termination schema, terrain binding, first-episode completeness, and task viability. No single self-reported compatibility flag is sufficient.
- Checkpoint loading goes through the shared `build_checkpoint_safety_report()` path with `weights_only=True`.
- The checkpoint never becomes current-best or Session Genesis. Its current role is runtime compatibility and evaluation-chain smoke testing.

The official pyramid-stair terrain is not the target task because its origin is on the central high platform. The task-owned Phase 0 overlay preserves the official G1 policy observation/action contract but uses a one-way staircase whose origin is on the bottom start platform.

The Phase 0 report tracks one complete first episode per environment, with a horizon of 600 control steps (12 seconds). A task-owned Isaac recorder captures the post-transition state and per-term termination masks before auto-reset; the frozen protocol maps those terms to primary reasons. All 56 episodes reached the first riser, then terminated through `base_contact` and triggered the fall diagnostic; stair-climbing success was 0/56. This admits the checkpoint only as compatibility evidence, not as a stair-capable policy, training start, or acceptance evidence.

## Frozen Eval Bins

The V1 stair protocol declares fixed bins:

- 0.10 m
- 0.15 m
- 0.20 m
- 0.25 m
- 0.30 m
- 0.35 m
- 0.40 m

These bins are task-owned eval protocol, not LLM-tunable prompt controls.

## Fixed Train Protocol

The frozen V1 protocol owns the train distribution, budget, episode length, evaluation layout, and acceptance gates. Draft V2 binds the `complete_reward_program` capability and fixed random initialization. Phase C will activate one pre-reset complete-reward dispatcher; additive Push reward wrappers are not reused.

Reset velocities, command resampling, standing/heading fractions, and force/push events are explicit `runtime_controls`. Runtime builders must index those fields directly; they must not supply semantic fallbacks. Terrain config-class defaults are intentionally invalid so geometry can only come from the frozen protocol.

- the official full reward is transcribed into canonical `training_code/components.py` as R0;
- each accepted generation replaces the complete reward program and becomes the next source parent;
- PPO, episode accumulation, logs, and runtime evidence must consume the same single reward result;
- `RewardTerm` and `DiagnosticTerm` remain shared source structures, but Stair candidate authority is the complete reward program, not independent additive tuning.

Height-map and terrain feature projection must be a runner-owned approved extension. A normal candidate must not change obs/action space, height-map projection, privileged terrain inputs, or backend terrain generation.

The stair task owns its `CandidateSurfacePolicy`, derived from `StairProtocolV1`. Push recovery must not contain stair-height ranges or any other stair-specific candidate semantics.

## Failure Taxonomy

The V1 primary failure classifier covers:

- base contact.
- fall.
- insufficient clearance.
- base pitch instability.
- slip.
- timeout.

The G1 minimal asset does not expose an independently validated toe body or contact-point signal through the current contact sensor. `toe_collision` is therefore reserved and not admitted to the effective ABI or primary taxonomy. It may be added only in a later protocol version after a real signal probe.

## Public Memory Summary

The staged task-specific public summary adapter compresses only validated public Stair metrics: seven-bin macro success, dominant primary failure, diagnostic rates, and progress trends. It is bounded to recent rounds.

The adapter must not expose hidden eval results, private seeds, checkpoint selectors, or runner-owned terrain generation internals. AgentCore should keep using the generic public context interface; stair-specific memory belongs in the stair TaskPlugin package.

## Extension Surfaces

The V2 ordinary candidate surface is one complete `training_code/components.py` reward program. It may use only the Phase C `StairRewardInputsV2` detached-clone ABI after that ABI is activated. It cannot modify training budget, PPO, fixed random initialization, terrain, evaluation, checkpoint selection, success/failure definitions, or acceptance.

Runner-owned approved extension surfaces:

- height-map observation projection.
- terrain metadata feature projection.
- changes to obs/action space.
- terrain generator progression.
- stair_height curriculum progression.

The old V1 classifications for stair-height range, scalar weight, and schedule-only edits remain parameter-only compatibility metadata. They do not satisfy the V2 `complete_reward_program` capability and cannot be accepted as a full reward generation.
