# Stair Climbing Task Extension Guide

Phase 0 provides immutable compatibility evidence for the official G1 rough-locomotion checkpoint. The implemented task chain adds shared reference-program contracts and task-owned production components: `StairClimbingTaskConfig`, `TaskPlugin`, one composition provider, dedicated train/eval entrypoints, seven-bin evaluation, failure classification, public metrics, prompt memory, and runner-owned evidence projection.

The production application registry exposes Stair only as a lazy provider and keeps Push Recovery as the default task. Building the registry or selecting Push does not import or resolve Stair; explicit Stair selection resolves and verifies the official-upstream `ReferenceProgramManifest`. Importing `true_rl_agent.tasks.stair_climbing` remains inert and does not import Isaac Lab or register a Gym environment.

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

The frozen V1 protocol owns the train distribution and episode semantics. The active task-owned evaluation profile extends its historical seven-height anchor into the frozen height-by-tread grid. Final V2 binds the fixed `4096 × 24 × 2000` training budget, `complete_reward_program` capability, fixed random initialization, and one pre-reset complete-reward dispatcher; additive Push reward wrappers are not reused.

Reset velocities, command resampling, standing/heading fractions, and force/push events are explicit `runtime_controls`. Runtime builders must index those fields directly; they must not supply semantic fallbacks. Terrain config-class defaults are intentionally invalid so geometry can only come from the frozen protocol.

- the official full reward is transcribed into canonical `training_code/components.py` as R0;
- each accepted generation replaces the complete reward program and becomes the next source parent;
- PPO, episode accumulation, logs, and runtime evidence must consume the same single reward result;
- `RewardTerm` and `DiagnosticTerm` remain shared source structures, but Stair candidate authority is the complete reward program, not independent additive tuning.

Height-map and terrain feature projection must be a runner-owned approved extension. A normal candidate must not change obs/action space, height-map projection, privileged terrain inputs, or backend terrain generation.

The stair task owns its `CandidateSurfacePolicy`, derived from `StairProtocolV1`. Push recovery must not contain stair-height ranges or any other stair-specific candidate semantics.

## Active Acceptance Protocol

Historical Phase 0 and Final V2 files remain immutable evidence. Production Stair loads
`configs/tasks/stair_acceptance_protocol_v5.json` as a separate trusted authority and
binds its hash into the effective task identity and `EvaluationBindingV1`. The active
rule compares candidate and current-best 35-cell generalization macro scores on the same
ordered five-seed public-accept cohort, while requiring the historical `0.40 m` tread
anchor to satisfy a bounded paired non-regression rule. Its effective tolerance is
`min(0.05, 0.02 + 1.5 * paired_guard_SE)`: `0.02` is the practical margin and `0.05`
is a hard regression cap. This admits the historical R4 edge case caused by paired cohort
noise while still rejecting the materially regressed R5. The objective materiality threshold is the maximum of `min_delta`,
20% of remaining score headroom, and 1.5 times the paired-delta standard error. Objective
and guard absolute floors remain separate bootstrap gates.

The task-owned evaluation profile gives each tread cell a distance-scaled timeout while
keeping the `0.40 m` anchor at 12 seconds. The `0.30/0.35/0.40/0.45/0.50 m` horizons are
`9.8/10.9/12.0/13.1/14.2 s`, rounded to the nearest control step. Isaac runs the shared
batch to the longest horizon; the Stair episode tracker closes each cell at its own
deadline. The active
[`stair_evaluation_semantics_v6.json`](../../configs/tasks/stair_evaluation_semantics_v6.json)
file is the sole authority for success and failure classification. A first valid
physical top arrival only latches arrival evidence; primary success requires the
subsequent stable top traversal declared by that file. Until stable success is reached,
post-arrival torso contact, fall, or off-course can still block success, and reaching the
cell deadline without stable success is a timeout.

The frozen eval-v5 semantics used first valid top arrival as primary success and kept
the subsequent stability traversal diagnostic-only. That executable-looking snapshot
is no longer shipped beside the active task configuration: completed v5 artifacts bind
its hash, and the repository history retains its exact bytes. The active loader accepts
only v6, and v5 results must not be interpreted or compared as eval-v6 outcomes.

The shared evaluation layer only understands bounded objective/guard scores, ordered
score samples, and a score ceiling. Height/tread cell aggregation, the seven-bin anchor,
and seed ordering remain task-owned. Push Recovery continues to use the historical
noise-aware V2 rule, and old ledger hashes and replay semantics are unchanged.

The public-to-sealed consistency tolerance is acceptance policy, not terrain geometry.
It is therefore stored in the acceptance protocol. Changing that tolerance changes the
acceptance and evaluation binding hashes, but does not change the evaluation profile,
the profile cohort commitment, or the deterministic seed values. Production artifacts
retain exact per-cell primary outcome counts; the prompt view uses one success-count
grid plus sparse nonzero failure-count grids over a shared height-by-tread axis. It does
not duplicate success as an outcome grid and omits only all-zero failure grids. When
multi-round feedback requires the bounded core projection, each failure grid is encoded
reversibly as `;`-separated height rows with `,`-separated tread columns. Projection v3
declares both separators and the shared row/column order; the task-level primary outcome
list remains the single reason universe. This keeps the reference and latest evaluated
failure maps inside the unchanged 32 KiB task-section budget.

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

The task-specific public summary adapter compresses only validated public Stair metrics: seven-bin macro success, dominant primary failure, diagnostic rates, progress trends, and bounded validated runtime reward contributions. It is bounded to recent rounds.

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

## Frozen Descriptor Versus Active Extension Contract

`configs/tasks/stair_extension_schema.json` is the hash-pinned Phase 0 descriptor. It
must remain byte-stable because the Phase 0 report and baseline compatibility evidence
bind its descriptor hash. It is intentionally not expanded with current guard,
evaluation-profile, or generalization fields.

Current and future task integration is documented by `TaskPlugin`, `TaskSpec`, the
task-owned config/profile modules, and the shared acceptance contract. Adding current
fields to the frozen descriptor would create a second authority and invalidate historical
evidence rather than improve extensibility.
