# Phase 6 live experiment summary

Date: 2026-06-21

Run id: `2026-06-21_phase6_live_deepseek_001`

Server artifact directory:

`/data/yuyangliu/runs/phase_6_multi_round_loop/2026-06-21_phase6_live_deepseek_001`

## 1. What control task this experiment tested

This experiment tested a humanoid push-recovery locomotion-control task on the
Unitree G1 29-DoF robot.

The policy is trained in simulation to keep a standing / near-zero velocity
command while external velocity-kick disturbances are applied to the robot base.
The target behavior is not walking forward. The target behavior is to remain
upright and stable after push disturbances.

Core evaluation setup:

- robot: Unitree G1 29-DoF
- environment profile: `phase2_public`
- command: standing / zero base velocity command
- disturbance mechanism: Isaac event `push_by_setting_velocity`
- push body: `torso_link`
- public evaluation seeds: 3 public seeds per round
- evaluation success criteria: survive the episode without violating protected
  fall / height / drift thresholds
- decision rule: accept a candidate only if public improvement over the current
  best exceeds the configured public threshold

The Phase 6 loop started from the Phase 2 baseline, not from the previous
Phase 5 best candidate. Hidden / IID evaluation information was not used as
LLM input.

## 2. Overall result

Final accepted candidate: round 3.

Final public success rate:

- mean success rate: `0.854166`
- per-seed success rates: `0.882812`, `0.859375`, `0.820312`
- approximate percentage: `85.42%`

Baseline public mean before accepted improvement was approximately `0.557292`,
so the accepted candidate improved public success by about `+0.296875`
absolute, or `+29.69 percentage points`.

Rounds 4 and 5 were evaluated after round 3, but both were rejected. Therefore
the final best configuration remained the round 3 candidate.

Final accepted best configs on server:

- `current_best/round_03_phase1_push_recovery.yaml`
- `current_best/round_03_phase2_baseline.yaml`

## 3. Round-by-round LLM proposal and outcome

| round | LLM improvement direction | Concrete config / reward change | Public result | Decision |
| --- | --- | --- | --- | --- |
| 1 | Increase posture stability by penalizing pitch/roll orientation error more strongly. | `phase2.reward_weights.flat_orientation_l2: -8.0 -> -10.0` | mean `0.554687`; values `0.656250`, `0.500000`, `0.507812`; delta `-0.002604`; threshold `0.114228` | rejected |
| 2 | Add broad push randomization in both lateral and fore-aft directions, plus randomized push timing, to train general push robustness. | `phase2_train.push.velocity_range_mps.x: [0.0, 0.0] -> [-0.5, 0.5]`; `phase2_train.push.velocity_range_mps.y: [0.0, 0.0] -> [-0.5, 0.5]`; `interval_range_s: [1.0, 1.0] -> [0.5, 1.5]` | mean `0.041667`; values `0.023438`, `0.015625`, `0.085938`; delta `-0.515625`; threshold `0.104686` | rejected |
| 3 | Add moderate forward push disturbances specifically to train backward-fall recovery, with slight push-timing randomization. | `phase2_train.push.velocity_range_mps.y: [0.0, 0.0] -> [0.0, 0.3]`; `interval_range_s: [1.0, 1.0] -> [0.8, 1.2]` | mean `0.854166`; values `0.882812`, `0.859375`, `0.820312`; delta `+0.296875`; threshold `0.103906` | accepted |
| 4 | Add a small lateral push range to improve resistance to roll / side disturbances on top of the round 3 best candidate. | `phase2_train.push.velocity_range_mps.x: [0.0, 0.0] -> [-0.1, 0.1]` | mean `0.828125`; values `0.890625`, `0.804688`, `0.789062`; delta `-0.026041`; threshold `0.050000` | rejected |
| 5 | Push harder on backward-fall recovery by increasing forward push magnitude, and strengthen orientation / height penalties. | `phase2_train.push.velocity_range_mps.y: [0.0, 0.3] -> [0.0, 0.5]`; `flat_orientation_l2: -8.0 -> -12.0`; `base_height: -12.0 -> -15.0` | mean `0.565104`; values `0.570312`, `0.539062`, `0.585938`; delta `-0.289062`; threshold `0.050000` | rejected |

## 4. Interpretation

The useful improvement was not simply "make pushes stronger" or "increase all
penalties." The best candidate was a moderate curriculum-style change:

- introduce fore-aft disturbance only in the positive `y` direction;
- keep the push magnitude moderate at `[0.0, 0.3]`;
- randomize push timing slightly with `[0.8, 1.2]`;
- avoid adding large lateral disturbance or overly strong reward penalties in
  the same round.

Round 2 shows that too much push randomization too early can collapse training.
Round 5 shows that increasing push magnitude and reward penalties after the
round 3 improvement can also regress performance.

The practical conclusion is that this task currently benefits from a narrow
and staged push-recovery curriculum rather than aggressive all-direction
domain randomization.

## 5. Safety and anti-cheating status

The run completed with `status: pass`.

Anti-cheating / leakage checks:

- the loop reported `start_from: phase2_baseline`;
- prompts were built from public-only summaries and operation-only feedback;
- hidden / IID evaluation diagnostics were not used as prompt input;
- raw accept/reject decision artifacts were not fed back to the LLM;
- a forbidden-token scan over all five live prompt files found no protected
  hidden/eval-seed fields;
- rejected rounds did not overwrite the best candidate.

This means the accepted round 3 result should be interpreted as a public-seed
Phase 6 result, not as hidden/IID validation.

## 6. What information was given to the LLM prompt

The full prompt text is not copied here. Conceptually, the LLM input was
structured into these parts:

1. Task objective and phase boundary

   The model was told to improve Phase 2 push-recovery performance during
   Phase 6, starting from the Phase 2 baseline. It was also constrained not to
   do Phase 7 extension work.

2. Public-only performance summary

   The prompt included public evaluation statistics such as success rate,
   failure label distribution, condition-flag rates, and secondary diagnostic
   metrics. Hidden/IID evaluation fields were excluded.

3. Current best candidate state

   The model saw the current best public result and the current best editable
   configuration context, so it could propose incremental changes rather than
   restarting from scratch each round.

4. Previous round feedback

   The prompt included public-only feedback from prior rejected or accepted
   candidates, summarized through operation-level diffs and public outcomes.
   This helped the model avoid repeating obviously bad changes.

5. Editable knobs and guardrails

   The prompt described which config/reward fields were allowed to change,
   including push velocity ranges, push timing, and selected reward weights.
   It also constrained values to finite, bounded, physically meaningful ranges.

6. Verifier and anti-cheating rules

   The model was told that protected verifier files, evaluation seeds,
   acceptance thresholds, hidden/IID diagnostics, and eval definitions were not
   editable or usable for proposal generation.

7. Required output format

   The model had to produce a structured candidate proposal that could be
   parsed into config changes and semantic operation diffs. This enabled the
   verifier and orchestrator to inspect changes before training.

In short, the prompt gave the LLM enough public training/evaluation feedback to
act as a reward/config tuning agent, but excluded protected hidden evaluation
information and required every proposal to pass static verification before it
could be trained.
