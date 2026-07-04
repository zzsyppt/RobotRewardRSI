# Stair Climbing Task Extension Guide

This guide is schema-level only. Phase 12 does not create a `stair_climbing` runtime plugin, does not create stair training commands, and does not run stair experiments.

## Required Future Pieces

Future stair support should add only task-owned extension pieces:

- TaskPlugin for stair semantics.
- task config for invariant stair bounds.
- baseline manifest for the starting policy.
- failure classifier for stair-specific failure modes.
- metrics projection for public stair eval summaries.
- BackendAdapter capability extension only if the backend needs a new materialization feature.

AgentCore, Runner, DecisionEngine, and ledger should not change because of stair task semantics. They already operate on TaskSpec, TrainJobSpec, EvalJobSpec, CandidateSurfacePolicy, runtime activation evidence, and public ledger records.

## Future Baseline

The baseline should be represented by a baseline manifest, not hard-coded in AgentCore:

- Isaac/Unitree G1 rough locomotion policy trained on stair_height 0.05-0.20 m.
- The baseline checkpoint is a runner-sealed warm_start checkpoint.
- Compatibility must state backend id, robot, observation schema hash, action schema hash, and policy architecture.

## Future Eval Bins

The first stair eval protocol should declare fixed bins:

- 0.10 m
- 0.15 m
- 0.20 m
- 0.25 m
- 0.30 m
- 0.35 m
- 0.40 m

These bins are task-owned eval protocol, not LLM-tunable prompt controls.

## Fixed Train Protocol

The future stair TaskPlugin should declare:

- baseline train distribution.
- terrain metadata projection.
- height map availability.
- foot clearance observation summary and DiagnosticTerm entry.
- terrain progress DiagnosticTerm entry.

Height-map and terrain feature projection must be a runner-owned approved extension. A normal candidate must not change obs/action space, height-map projection, privileged terrain inputs, or backend terrain generation.

## Failure Taxonomy

The future failure classifier should at least cover:

- toe collision.
- insufficient clearance.
- base pitch instability.
- slip.
- timeout.

These are public diagnostics unless the task explicitly promotes one into a fixed eval metric.

## Future Public Memory Summary

The future stair task may add a task-specific public summary adapter for next-round prompt context. That adapter should compress only public stair eval metrics such as fixed-bin success rates, toe collision counts, clearance histograms, pitch instability summaries, slip summaries, and terrain progress diagnostics.

The adapter must not expose hidden eval results, private seeds, checkpoint selectors, or runner-owned terrain generation internals. AgentCore should keep using the generic public context interface; stair-specific memory belongs in the stair TaskPlugin package.

## Extension Surfaces

Ordinary candidate surfaces:

- RewardTerm for public-input reward shaping such as foot clearance margin or pitch stability margin.
- DiagnosticTerm for toe collision counts, clearance histograms, terrain progress summaries, and slip summaries.

Runner-owned approved extension surfaces:

- height-map observation projection.
- terrain metadata feature projection.
- changes to obs/action space.
- terrain generator progression.
- stair_height curriculum progression.

Pure stair_height range changes, reward-weight changes, or schedule-only proposals are parameter-only and cannot be code-level accepted improvements.
