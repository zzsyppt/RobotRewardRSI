# Stair 15-round experiment follow-ups (`e24777b`)

## Run identity

- Status: running; do not change the frozen protocol or runtime for non-blocking items.
- Code commit: `e24777b6122e3ec8611e71900781c5cea7efd54c`
- Session: `stair-r0-formal-214a994-20260808`
- Genesis hash prefix: `c67c83fc...`（权威完整值只保存在 Genesis artifact）
- Ledger epoch: `stair-smoke-fix-e24777b-20260808t073320z`
- Server output: `/data/yuyangliu/true_rl_agent_phase_g/formal_15round_e24777b_smokefix_20260808T073320Z`

## Triage rule

A blocking issue invalidates experiment comparability, evaluation truthfulness, evidence
lineage, process safety, or the ability to complete the run. Blocking issues are fixed at
their owning source and require verification plus a fresh compatible epoch when needed.

Candidate quality, provider prose quality, bounded authoring retries, and possible prompt
or memory improvements are non-blocking while existing guards keep them out of training or
let fixed train/eval/acceptance decide their value. These items are handled after all 15
rounds so the experiment contract remains fixed.

## Non-blocking observations

### R1 authoring

- Attempt 1 (`goal_settle_v1`) materialized an invalid top-level expression and was rejected
  by the static verifier as `forbidden_top_level:training_code/components.py:Expr`.
  The repair loop behaved correctly; no training was started for this source.
- Attempt 2 (`repair_v1`) reproduced the exact R0 source checksum
  `a1255b58...` and was rejected by
  the anti-degenerate gate as `identical_or_format_only`.
- Attempt 3 (`stair_climb_repair_v1`) introduced a real mechanism delta, passed all 17 Stair
  smoke checks, and entered fixed-budget training.
- Prompt payload sizes were `57,440`, `60,039`, and `66,205` bytes. The increase is explained
  by bounded repair feedback and the latest failed source; it remains below the 256 KiB cap.
  Continue recording per-attempt size and section duplication across later rounds.
- R1 memory correctly contained zero prior experiments. Later rounds must be checked for
  bounded growth, identity-correct feedback, and omission accounting.

### Candidate and rationale semantics

- The executable R1 candidate uses `goal_progress_fraction`, not `base_height`, for progress.
  Its dense torso-contact term is clipped to `[0, 1]` before a `-1.0` weight. This is a valid
  bounded hypothesis even though earlier experiments suggest repeatedly adding contact-force
  penalties may be less promising than near-goal posture mechanisms. Fixed evaluation should
  decide this candidate without manual intervention.
- Provider rationale remains advisory. The prior R7 explanation misstated a maximum weighted
  contribution by 10x even though executable code and runtime evidence were correct. Prompt v6
  now states that executable source plus verifier/runtime evidence is authoritative and that
  quantitative claims must include conditioning, callable clamps, `RewardTerm` clips, and
  weights. After this run, consider whether a runner-generated contribution-bound summary can
  make this evidence easier to consume without parsing or trusting natural-language arithmetic.
- Prompt v6 now defines `base_height` from the canonical
  `base_pos_relative_to_terrain_origin[:, 2]`, identifies the bottom-platform terrain-origin
  frame, and declares `local_clearance=false`. Do not add a second local-height interpretation;
  continue checking later candidates for misuse of this field.

### R1 result and visual evaluation audit

- R1 public-accept mean was `0.7714285714` and sealed mean was `0.7704081633`.
  It was accepted over R0 because the paired delta `0.1928571429` strictly exceeded the
  frozen threshold `0.0842857143`, with every required gate passing.
- R1 showed a strong non-monotonic height profile: public-dev run 1 produced
  `stair_035=0/8` and `stair_040=8/8`. A fresh checkpoint-video rollout reproduced the exact
  same 56-env bin counts and formal state-machine outcomes; the exporter did not reclassify.
- In the 0.35 m video, envs 40/43/47 reached the top region and then terminated by torso
  contact at steps 382/420/422. In the 0.40 m video, envs 48/51/55 completed the stable hold
  and latched success at steps 409/417/417. The terminal frames and reset labels are visible.
  This supports a real gait resonance/height-sensitivity interpretation rather than a bin,
  aggregation, or success-state-machine defect. It is not a reason to alter evaluation.
- Local visual evidence:
  `outputs/stair_e24777b_diagnostic_videos/r1_stair035_env404347/` and
  `outputs/stair_e24777b_diagnostic_videos/r1_stair040_env485155/`.

### R2 authoring and training

- R2 used one `65,415` byte prompt. It retained exactly one experiment-memory entry, one
  accepted feedback entry, and one recent outcome. The prompt included R1 per-bin success,
  dominant torso-contact failure, reached-goal-without-success rate, runtime activation, and
  the accepted parent source. The context is currently sufficient and bounded.
- R2 (`stair_v2_contact_fix`) responds to the R1 feedback but changes several weights at once.
  Its `settle_near_top` comment claims a 0..1 near-goal scale while the code only applies a
  lower clamp; `goal_progress_fraction > 1` therefore makes the scale exceed one, and the
  RewardTerm has no `clip_max`. This is a candidate-quality and explanation mismatch, not a
  frozen-system defect. Monitor the actual runtime contribution and let fixed evaluation
  reject it if harmful.
- At training step 675 all observed scalars remained finite, but the last-100 window had lower
  terrain curriculum progress, higher base-contact termination, and larger policy noise than
  R1. Do not interrupt solely for weak candidate performance.
- The completed runtime evidence confirmed the predicted defect: `settle_near_top` reached a
  maximum absolute weighted contribution of `62.16549683`. Public-dev fell to `0.1071428571`,
  public-accept to `0.0214285714`, and sealed to `0.0510204082`; the sealed-validation gate
  failed and the candidate was rejected as `required_gate_failed`. The guard chain therefore
  contained this provider-authored unbounded mechanism and kept R1 as the accepted head.

### R3 result

- R3 (`stair_v3_targeted_fix`) returned to bounded terms and passed runtime activation,
  budget, Stair behavior, public evidence, and sealed validation gates. Its public-dev,
  public-accept, and sealed means were `0.2142857143`, `0.2071428571`, and `0.2040816327`.
- The dominant public-dev outcome was timeout (`73.21%`). `77.98%` of episodes crossed the
  goal-progress threshold without completing the stable top hold, so this is a policy-control
  failure rather than missing progress. The paired delta against R1 was `-0.5642857143`; the
  fixed acceptance rule correctly rejected it and retained R1.
- The prompt was `81,471` bytes, with three bounded experiment-memory entries and two recent
  outcomes. This is still below the cap, but the jump from R2 should be included in the final
  section-by-section byte audit.

### R4 authoring and training

- Attempt 1 (`stair_flexible_upper_body_v1`) repeated an evaluated mechanism with only
  parameter/rationale changes and was rejected as `historical_mechanism_repeat`.
- Attempt 2 (`stair_settle_and_contact_fix`) used a forbidden `torch` import and was rejected
  by the static verifier. Attempt 3 fixed only that verifier issue and selected
  `stair_settle_and_contact_fix_v2` for training.
- The selected candidate starts from accepted R1, progress-weights the bounded torso-contact
  term, and adds a `[0, 1]`-clipped, `+0.2` near-top low-velocity reward. Its executable terms
  are bounded; no R2-style runaway contribution is visible in source. Prompt sizes were
  `87,955`, `82,502`, and `82,304` bytes. The first-attempt increase reflects the third prior
  experiment; repair prompts replace the latest failed-source context instead of growing
  monotonically.
- Runtime evidence confirmed the new term was active and bounded: `settle_near_top` reached
  `0.17709015`, while `progress_reward` and `dense_base_contact` remained capped at `0.5` and
  `1.0` maximum absolute weighted contribution.
- R4 public-dev, public-accept, and sealed means were `0.8214285714`, `0.8464285714`, and
  `0.8545918367`. Its paired delta over accepted R1 was `0.075`, strictly above the adaptive
  `0.0457142857` threshold, and every required gate passed. R4 was therefore accepted as the
  new head. Public-dev success at 0.35 m rose to `45.83%`; 0.40 m was `33.33%`.

### R5 start

- R5 prompt lineage names R4 (`stair_settle_and_contact_fix_v2`) as current best and binds
  parent checksum `f3d1b26d...`, exactly matching the accepted R4 source. It contains four
  bounded experiment-memory entries, three recent outcomes, zero omitted experiments, and is
  `82,187` bytes.
- `stair_contact_penalty_boost_v1` changes R4's torso-contact mechanism using squared progress,
  a lower force normalization threshold, and weight `-2.0`. Although raw progress can exceed
  one after crossing the goal, the enclosing `RewardTerm` clips the callable output to `[0, 1]`,
  so the weighted contribution remains bounded. Fixed evaluation should determine whether this
  more aggressive contact shaping helps rather than treating it as a system defect.
- Training completed the exact `196,608,000` environment-step budget. Runtime maxima were
  `0.5` for progress, `2.0` for dense torso contact, and `0.17706065` for near-top settling.
- R5 public-dev, public-accept, and sealed means were `0.9583333333`, `0.9571428571`, and
  `0.9005102041`. Its paired delta over R4 was `0.1107142857`, above the reduced adaptive
  threshold `0.0307142857`, with all required gates passing. R5 was accepted as the new head.
  Public-dev success reached `91.67%` at 0.35 m and `79.17%` at 0.40 m; public-accept had
  268 successes in 280 episodes. The approximately 5.7-point public/sealed gap should remain
  visible in final reporting rather than presenting the public score as all-distribution skill.
- Exact public-dev run-1 video replay reproduced the formal 56-env summary byte-for-semantics:
  macro `0.9821428571`, 0.35 m `8/8`, and 0.40 m `7/8`, with no video reclassification.
  Tracked 0.35 m envs 40/43/47 all completed stable hold at steps 413/406/418. The renderer
  used exact task-owned pre-reset Isaac trajectories, preserved terminal/reset labels, and did
  not recompute physics.

### R6 start

- Attempt 1 (`stair_contact_penalty_double_v1`) changed only parameters of R5's evaluated
  mechanism and was rejected as `historical_mechanism_repeat`. Attempt 2 added a structurally
  new `base_contact` binary term and entered training from R5's source.
- The new binary term uses weight `-100` and only `clip_min=0`; it is likely redundant with the
  existing termination penalty because `base_contact` is itself a terminal event, and raw goal
  progress can exceed one. This is a candidate-quality risk rather than an authority or lineage
  defect. Record its observed maximum contribution and fixed evaluation outcome.
- Runtime evidence bounded the binary term at exactly `100.0` maximum absolute weighted
  contribution. Training remained finite and reached high terrain levels, but formal behavior
  regressed: public-dev `0.7321428571`, public-accept `0.7821428571`, and sealed
  `0.7882653061`. Public-accept had 51 off-course failures versus 8 torso contacts, supporting
  the interpretation that the sparse large penalty distorted control rather than improving the
  remaining failure mode. The paired delta was `-0.175`; R6 was rejected and R5 remained head.

### R7 authoring

- Attempt 1 (`stair_contact_sensitivity_v1`) again changed only parameters of R5's evaluated
  contact mechanism and was rejected as `historical_mechanism_repeat` before training.
- The first prompt was `94,718` bytes. Its bounded experiment memory held six experiments in
  about 6.2 KiB, three recent outcomes, and zero omissions; the prompt-size increase is not
  caused by unbounded memory growth, but still requires the final section-byte audit.
- Attempt 2 added `foot_reach_penalty` and trained from R5. Its observed maximum weighted
  contribution was only `0.24150926`, but the learned behavior still regressed sharply:
  public-dev `0.4642857143`, public-accept `0.5035714286`, and sealed `0.5102040816`.
  Public-accept contained 83 falls and 50 torso contacts, with near-zero 0.10/0.15 m success.
  The candidate was rejected by a `-0.4535714286` paired delta. This is useful evidence that a
  numerically small reward term can alter the optimization attractor; contribution magnitude
  is activation/risk evidence, not a bound on behavioral impact.

### Capability-gain audit: R1 -> R4 -> R5

- Comparability is intact. R1, R4, and R5 used the same initial policy-state hash
  `55154410...`, formal training binding, exact `196,608,000` environment-step budget,
  task/eval/evaluation-binding hashes, and paired acceptance seed cohort. Their final
  checkpoint hashes are distinct. Candidate source/package/path were absent from evaluation.
- R4 beat R1 on all five paired acceptance seeds. The paired mean delta was `+0.075`, with a
  descriptive paired t interval of approximately `[+0.018, +0.132]`; sealed mean also rose
  from `0.7704081633` to `0.8545918367`. This is a real seven-bin macro capability gain, but
  not per-bin dominance: public-accept changed by `+0.60` at 0.30 m, `+0.475` at 0.35 m, and
  `-0.55` at 0.40 m.
- R5 beat R4 on all five paired acceptance seeds. The paired mean delta was `+0.1107142857`,
  with a descriptive paired t interval of approximately `[+0.058, +0.164]`; sealed mean rose
  again to `0.9005102041`. Public-accept retained all lower-bin rates, improved 0.35 m by
  `+0.475`, and improved 0.40 m by `+0.30`. This is strong evidence of a broader R5-over-R4
  capability gain.
- R5 beat R1 by `+0.1857142857` on average with all five paired deltas positive, but it still
  trailed R1 by `0.25` at 0.40 m while gaining `0.60` at 0.30 m and `0.95` at 0.35 m. Report
  R5 as stronger over the frozen 0.10--0.40 m macro task, not as dominant at every height.
- Same-seed exact trajectory videos reproduced formal summaries with no reclassification.
  At 0.35 m, tracked R1 envs all failed, R4 had one success plus off-course/contact failures,
  and R5 had three stable-hold successes. At 0.40 m, the tracked R4 envs all failed while the
  same R5 envs succeeded. R4 env40 crossed progress fraction `1.29` but remained an off-course
  failure, directly showing that success is not merely thresholded forward progress.
- The audit establishes checkpoint capability under the frozen task distribution. It does not
  establish universal robustness or isolate one reward term causally: each reward program has
  one fixed training realization and no shadow ablation or training-seed replication.

### R8 result and R9 prompt-boundary repair

- R8 (`stair_horizontal_contact_penalty_v1`) completed the fixed budget and all evidence gates.
  Public-accept was `0.9` and sealed validation was `0.9107142857`, so the paired comparison
  correctly rejected it below R5. Public-dev 0.40 m success was `0.375`; R5 remained the head.
- Before authoring R9, the Stair task sections exceeded the shared `32 KiB` safety boundary.
  The live loop failed before provider invocation, candidate creation, training, evaluation, or
  ledger mutation; the eight completed rows therefore remain intact. The cause was a count-only
  detailed-feedback selector: adding the fourth full behavior record made the section exceed
  the byte limit even though the five-entry count limit had not been reached.
- Raising the shared limit was rejected. The R8 provider payload was already `91,012` bytes.
  Its largest sections were source payloads (`15,031`), public history (`12,704`), reward-input
  contract (`12,509`), required package contract (`11,661`), and detailed Stair feedback
  (`7,697`). Exact repeated text inside the detailed feedback was only a few hundred bytes;
  removing schemas or prose would not provide a durable bound.
- The task-owned repair keeps the full bounded experiment-memory index and compacts only the
  lower-priority detailed Stair behavior window against the canonical whole-section byte size.
  Current accepted/reference head and latest evaluated feedback are mandatory; the best rejected
  candidate outranks ordinary recent records. The prompt declares the omitted-record count and
  whether byte compaction occurred, while the shared sanitizer remains the final hard boundary.
- Exact replay of the eight-row formal ledger measured `34,339` bytes before compaction and
  `31,750` after it. The retained detailed records are R5 (accepted head), R7 (recent comparison),
  and R8 (latest and best rejected); R6's identity, score, reject reason, and eight salient runtime
  components remain in experiment memory. This closes the blocker without losing cross-round
  mechanism memory or changing Push Recovery.
- The resumed R9 first attempt used the expected `31,750` byte task sections and a `98,030` byte
  complete prompt. It bound the R5 parent checksum and R8 runtime evidence, then authored
  `stair_settle_ramp_v1`: one callable-only change that replaces R5's `progress > 0.95` binary
  settle gate with a bounded 0.5--1.0 linear ramp. It did not repeat R6/R7/R8 mechanisms or
  change unrelated coefficients, which is direct evidence that the compacted prompt remained
  sufficient for attributable authoring. The new term's maximum possible weighted contribution
  is `0.2`; its main risk is optimization conflict with the fixed forward command, not runtime
  unboundedness.

### Prompt-capacity decision after the R9 blocker

- Do not raise the shared `32 KiB` task-section limit merely because the ninth round crossed it.
  The original selector was bounded by record count but not by canonical serialized bytes, so a
  larger constant would only move the same failure to a later round. The source repair is the
  task-owned byte-budgeted retention policy, with the shared sanitizer still acting as the final
  fail-closed boundary.
- R10 repair attempt 2 provides a post-fix measurement: task sections were `31,776` bytes and the
  canonical provider payload was `71,000` bytes (`94,772` bytes after provider-message rendering).
  The largest payload sections were inline candidate sources (`15,094`), reward inputs (`9,380`),
  public history (`9,335`, including `9,052` of experiment memory), required package (`8,600`),
  and detailed Stair feedback (`5,781`). Exact repeated strings of at least 80 characters account
  for only about `1,417` avoidable bytes, so broad prose trimming is not a durable primary fix.
- Across all 18 R1--R10 authoring attempts, rendered payload size rose from `57,440` bytes to a
  maximum of `99,609`, still far below the `262,144`-byte overall prompt boundary. After byte-aware
  compaction, the rendered Stair task section stayed between `31,750` and `31,776` bytes for R9/R10.
  Normal exact duplicate text was `1.1--1.7 KiB`; the historical R1 attempt 3 reached about
  `6.4 KiB` because two logical sources carried identical content hashes. The completed R1--R14
  audit found this in only one of 24 attempts, so do not expand the source alias schema for that
  rare case unless the full run or a simpler implementation supplies stronger evidence.
- The current retention order is appropriate: preserve accepted/current-head behavior, latest
  evaluated behavior, then the best rejected behavior; keep identities, scores, rejection reasons,
  and salient runtime components for omitted rounds in experiment memory. R10 retained detailed
  feedback for R5/R8/R9 and omitted six older detailed records while retaining their compact index.
- The next shared cleanup should replace duplicate source summaries/hashes in
  `authoring_context_frame` with canonical references where consumers permit it. This is a small
  metadata reduction, not justification for dropping either the current-best source or a genuinely
  relevant failed source. Inline latest-failed source should be selected by evidence and frame kind,
  not included unconditionally forever.
- The declared `14 KiB` experiment-memory limit remains a post-run correctness fix: it currently
  bounds only the `experiments` array while naming itself as a whole-object serialized limit. Bound
  the complete canonical object (or rename the field explicitly) before changing its numerical cap.

### R9 and R10 results after prompt recovery

- R9 (`stair_settle_ramp_v1`) completed the exact fixed budget and all evidence gates, but scored
  public-dev `0.8452380952`, public-accept `0.8535714286`, and sealed `0.8290816327`. Its paired
  delta against R5 was `-0.1035714286`; it was correctly rejected and R5 remained the head.
- R10 attempt 1 was a parameter/rationale-only repeat of the accepted contact mechanism and the
  historical-mechanism guard rejected it before training. Repair attempt 2 used a `31,776`-byte
  Stair task section and authored `stair_orient_near_top_v1` from the R5 checksum. This demonstrates
  that compact feedback still exposed enough mechanism history for the guard and repair flow.
- R10 added one clipped near-top roll/pitch penalty. It overlaps conceptually with the existing
  full-episode flat-orientation term, but was bounded and safe: all 19 terms ran `48,001` times,
  `orient_near_top` had maximum absolute weighted contribution `2.0` and final mean `-0.00165669`.
  Training used exactly `196,608,000` environment steps, matched initial policy hash `55154410...`,
  and produced checkpoint `sha256:e4bafa7d...`.
- Fixed evaluation rejected the superficially healthy training run. R10 scored public-dev
  `0.9107142857`, public-accept `0.9142857143`, and sealed `0.8826530612`; its paired delta against
  R5 was `-0.0428571429` versus threshold `0.0275775807`. Public-accept retained strong 0.35 m
  behavior but averaged about `45%` at 0.40 m and had 21 torso-contact failures in 280 episodes.
  All evidence gates passed, so this is a clean capability rejection rather than an infrastructure
  failure. R5 remains current best.

### R11 near-threshold result

- R11 prompt remained bounded at `31,897` task-section bytes and `100,232` total rendered bytes.
  It retained detailed R5/R9/R10 feedback; experiment memory held nine entries, omitted one old
  round, and exposed the omission. The full memory object was `14,623` bytes while its experiments
  array was `12,926`, again demonstrating the declared `14,336`-byte scope mismatch without
  unbounded growth.
- `stair_contact_penalty_quartic_v1` changed only R5's progress-weighted torso-contact callable
  from squared to quartic progress. Its prose incorrectly described this as larger near-goal
  penalty: for `0 < progress < 1`, quartic magnitude is smaller, although the slope is steeper near
  one and values above one can grow faster before the enclosing `[0, 1]` clip. The executable code
  remained bounded and attributable, so fixed evaluation rather than manual intervention decided it.
- Training completed the exact budget with matched initial policy state. The mid-training drop at
  iteration 1000 recovered by iteration 1999; the contact term's observed maximum weighted
  contribution was `2.0` and the final checkpoint was `sha256:16cd01eb...`.
- R11 produced a promising but insufficiently paired gain: public-dev `0.9583333333`,
  public-accept `0.9714285714`, sealed `0.9591836735`, and public-accept 0.40 m success `82.5%`.
  The five paired deltas against R5 were mixed; mean delta was only `+0.0142857143` with paired
  standard error `0.0142857143`, below the frozen `0.0214285714` threshold. The evaluator correctly
  rejected it as `paired_headroom_delta_not_met`. After the 15-round run, compare R5 and R11 using
  exact-seed video evidence and, if deciding whether to promote the mechanism, independent training
  replications; do not retroactively relax this run's acceptance rule.

### R12 semantic-redundancy result

- R12 attempt 1 changed R11's quartic progress exponent to cubic. The historical-mechanism guard
  classified it as a parameter/rationale-only repeat of R11 and rejected it before training. This
  confirms that the compact memory still supports cross-round anti-degenerate enforcement.
- Repair attempt 2 added a bounded `-2` binary torso-contact term gated by progress above `0.95`.
  It was structurally distinct from the prior full-episode `-100` binary candidate, but semantically
  weak because torso contact already triggers the existing `-200` non-timeout termination penalty.
  Runtime evidence matched that concern: the new term activated with maximum contribution `2.0`
  but had zero final sampled mean. It nevertheless can alter optimization and was allowed to run
  because this is candidate quality, not an authority or numerical-safety defect.
- R12 scored public-dev `0.8035714286`, public-accept `0.8892857143`, and sealed `0.8801020408`;
  paired delta against R5 was `-0.0678571429`. Public-accept 0.40 m improved to `92.5%`, but 0.20 m
  fell to `47.5%`, 0.25 m to `92.5%`, and ten episodes fell. The fixed macro evaluation correctly
  rejected the cross-bin tradeoff. A post-run semantic redundancy improvement may warn about terms
  fully dominated by terminal penalties, but must not assume small runtime mean implies small
  behavioral impact.

### R13 multi-mechanism attribution failure

- R13 remained within the recovered prompt envelope: the Stair task section was `31,791` bytes
  and the complete rendered payload was `100,422` bytes. Detailed feedback retained R5/R11/R12
  while nine older detailed records were omitted but still represented in bounded experiment
  memory. There was no recurrence of the original `32 KiB` task-section failure.
- `stair_sqcontact_settle_v1` changed two mechanisms relative to R5: it replaced the accepted
  squared-progress contact term with a squared-force variant and also raised `settle_near_top`
  from `0.2` to `0.5`. This violates the provider-facing single-mechanism attribution guidance,
  but the current static contract does not reject it. The ledger correctly records
  `mechanism_attribution.status=not_attributed` and makes no causal claim, so allowing the bounded
  candidate to complete did not corrupt acceptance evidence.
- The fixed evaluator detected a large capability regression despite healthy, fully activated
  training: public-dev was `0.5297619048`, public-accept was `0.5785714286`, and sealed validation
  was `0.5867346939`. The paired delta against R5 was `-0.3785714286`; the candidate was correctly
  rejected. Public-accept produced 162 successes, 73 falls, and 45 torso contacts across 280
  episodes, confirming that normal training completion did not mask behavioral collapse.
- Post-run governance should make a provider-declared single experimental mechanism machine-checkable
  enough to stop obvious compound edits before expensive training. It should compare canonical
  candidate structure against the selected parent and report changed terms/weights/functions; it
  should not attempt to infer arbitrary reward semantics in the shared layer.

### R14 training/evaluation divergence

- R14 first tried another parameter-only contact-weight adjustment and was correctly stopped as a
  historical repeat before training. Its repair prompt remained bounded at `94,688` bytes and
  authored `stair_sigmoid_contact_penalty_v1`, a single structural change from R5's quadratic
  progress gate to a sigmoid centered at `goal_progress_fraction=0.9`; all other code was preserved.
- Training completed exactly `196,608,000` environment steps with the expected initial policy hash.
  It looked substantially better than R5/R11 on training curriculum: terrain level reached about
  `5.65`, final base-contact termination was about `2.5%`, and mean episode length was about `596`.
  All 18 terms ran `48,001` times; the sigmoid contact term was bounded with maximum weighted
  contribution `1.4621172`. Checkpoint: `sha256:798563bb...`.
- The frozen seven-bin evaluator correctly rejected that attractive training curve. R14 scored
  public-dev `0.8869047619`, public-accept `0.8821428571`, and sealed `0.8801020408`; paired delta
  against R5 was `-0.075` versus threshold `0.0214285714`. Public-accept was `100%` at every
  0.10--0.35 m bin but only `17.5%` at the held-out 0.40 m frontier, with 247 successes, 13 falls,
  and 20 torso contacts in 280 episodes. This is strong evidence that the evaluator is detecting
  a real distribution-frontier regression rather than accepting smoother training metrics.

### Prompt-capacity decision after R13

- Do not raise the `32 KiB` Stair task-section cap. It is an internal fail-closed semantic boundary,
  not the provider model's context limit. R9--R14 authoring has stayed near `31.8 KiB` for task data
  while complete rendered prompts remained below about `101 KiB`, so the task-owned byte-aware
  selector is controlling growth without starving current-head or recent-failure evidence.
- Preserve self-contained cross-round contracts because provider calls are stateless. Optimize only
  measured duplication: identical source payloads by content hash, repeated source identity metadata
  that can reference one canonical entry, and fixed feedback legends that can be hoisted once. Do
  not delete current-best source code merely because its checksum appears in several metadata views.
- Keep the latest failed source inline only when a repair attempt or evidence-relevance rule needs its
  exact code; otherwise retain its identity, score, rejection reason, and salient runtime evidence in
  compact experiment memory. This is a post-run selection refinement, not a reason to change the
  frozen experiment ABI during R14/R15.
- The prompt artifact is the exact provider-facing serialization: `canonical_prompt_json()` uses
  `indent=2`, so R14 repair really sends `94,688` bytes (the compact-JSON representation would be
  `70,952` bytes but is not the production wire format). Compact JSON is a possible lossless wire
  optimization, but changing the canonical serializer also changes prompt hashes and every byte
  contract that uses it. Do not couple that broad ABI change to the immediate Stair capacity repair.
- DeepSeek's default context cache matches identical input prefixes. Across the first attempts of
  R1--R14, the current section order provides only about `20.7 KiB` of mean common prompt prefix
  because dynamic `authoring_context_frame` appears before several stable contracts. Reordering the
  existing stable sections first would raise the measured common prefix to about `38.1 KiB` without
  deleting semantic content. Treat this as a cost/latency optimization and verify candidate quality;
  it does not replace the internal task-section bound.
- Provider response usage currently drops `prompt_cache_hit_tokens` and
  `prompt_cache_miss_tokens`, so cache effectiveness cannot be audited from run artifacts. Add bounded
  provider-usage telemetry before claiming the section reorder saves real tokens. Usage counts are
  shared operational evidence and must not contain task semantics or provider response content.
- Prompt efficiency is also reduced by a contract contradiction, not by size. Seven of the first 24
  R1--R14 attempts (`29%`) were rejected before training as historical
  `parameter_and_rationale_only_delta`. The shared mechanism gate and system documentation make
  parameter-only candidates ineligible for policy improvement, while Stair selection guidance says
  the provider may "tune one existing mechanism's coefficients." Remove that false option from the
  Stair guidance and add one task-neutral statement of the actual historical-source guard to shared
  guidance. Do not weaken the guard merely to make the contradictory prompt true.

### R15 provisional semantic-input finding

- R15 consumed three provider attempts before training. Attempt 1 was another historical
  parameter/rationale-only contact variant and was stopped before materialization; attempt 2 was
  the run's first provider JSON-parse failure; attempt 3 authored
  `stair_upright_near_top_v1` and passed the existing static and smoke gates.
- The accepted-for-training source is bounded and changes one declared term, but it is conceptually
  close to R10's rejected near-top orientation term: both gate posture shaping on
  `goal_progress_fraction > 0.95`. The historical guard sees a different function and therefore
  cannot identify this semantic overlap. That is a candidate-efficiency limitation, not an
  evaluation-authority defect; the frozen evaluator should still decide the completed run.
- More importantly, the rationale states that `projected_gravity[:, 2]` is `+1` when upright.
  The canonical runtime passes Isaac Lab's `robot_data.projected_gravity_b` directly, and the
  repository's trusted upright fixtures set its z component to `-1`. The new term returns this
  signed component and declares `[0, 1]` clipping, so it is zero for normal upright states and
  becomes positive only after tilt exceeds roughly 90 degrees toward an inverted base. It is
  therefore either inactive on healthy motion or directionally wrong on abnormal motion, rather
  than the claimed upright incentive. Runtime activation evidence after training must distinguish
  those cases.
- This mistake is not evidence for raising the 32 KiB task-section cap. The public Stair input
  contract describes `projected_gravity` as the world gravity direction in the base frame but does
  not make its upright sign and useful upright scalar explicit. After the frozen run, strengthen
  that one task-owned canonical field semantic (for example, upright is approximately
  `[0, 0, -1]` and upright alignment is `-projected_gravity[:, 2]`) and add prompt-contract/smoke
  tests. Do not add Stair posture semantics to shared prompt infrastructure.

### R15 final result and 15-round closure

- Training completed the exact `196,608,000` environment-step budget, matched the frozen initial
  policy-state hash, and produced checkpoint `sha256:7c569a41...`. All 19 terms were dispatched;
  `upright_near_top` was called `48,001` times and reached maximum weighted contribution
  `+0.49998313` while ending with zero sampled mean. Combined with the canonical upright value
  `projected_gravity[:, 2] ~= -1`, this confirms directionally wrong activation on severely
  inverted near-top states rather than the claimed upright reward.
- The evaluator contained the bad candidate correctly. R15 scored public-dev `0.9404761905`,
  public-accept `0.9571428571`, and sealed `0.9438775510`. Its paired delta against R5 was exactly
  zero versus threshold `0.0328056662`, so it was rejected as
  `paired_headroom_delta_not_met`; every evidence gate passed and no runner error occurred.
- R15's public-accept 0.40 m success was `72.5%`, exactly R5's value. It traded R5's one 0.35 m
  failure for one 0.10 m failure and produced 268 successes, one fall, and 11 torso contacts versus
  R5's 268 successes, three falls, and nine torso contacts. The equal macro score therefore does
  not hide a capability gain, and the wrong reward term has no promotion or causal authority.
- The complete run used 27 provider attempts across 15 rounds. Prompt payloads ranged from
  `57,440` to `100,422` bytes (mean `88,092`); eight attempts were rejected as historical mechanism
  repeats and one as identical/format-only. Only one prompt duplicated an entire source payload
  (`4,978` bytes, during early R1 repair). Exact repeated long-string content otherwise averaged
  about `2.3 KiB` per prompt and consisted mainly of lineage IDs, hashes, and bounded summaries.
  This is not evidence for broad contract deletion or a new deduplication subsystem.
- With the current order, adjacent calls share about `19.9 KiB` of mean exact byte prefix. Moving
  existing stable sections ahead of dynamic authoring/history sections raises the measured mean to
  about `37.3 KiB` without deleting content. Actual cache benefit remains unproven until provider
  usage telemetry is recorded.
- R15's full experiment-memory object is `14,622` bytes while its entries array is `12,931` bytes;
  the declared whole-object limit is `14,336` bytes. Keep the numerical limit and correct its scope,
  retaining the latest accepted and latest evaluated records under pressure.

## Post-run work queue

- Aggregate prompt bytes, stable section hashes, source payload bytes, repair count, and actual
  duplicated bytes for every authoring attempt.
- Consider deduplicating repeated source identity summaries across `authoring_context_frame`,
  `source_catalog`, `candidate_lineage`, and `source_payloads` only as a shared post-run change.
  It is not the cause of the Stair task-section overflow, and changing that ABI mid-run would
  have a wider blast radius than the task-owned repair.
- Correct the experiment-memory byte-limit contract. `EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES`
  currently bounds only the serialized `experiments` list while the public `limits` object calls
  it `max_serialized_bytes`; synthetic R8--R15 continuation reaches about `16.1 KiB` for the full
  memory object despite the declared `14 KiB`. The memory remains finitely bounded by 16 rounds
  and the list-byte loop, so this is not a live-run blocker, but the post-run repair should either
  bound the complete object or name the field explicitly as an entries-list limit.
- Do not repair the full-object bound with an unconditional oldest-entry pop. On the real R14
  prompt, reducing the nine-entry object from `14,610` bytes by that rule removes R5, the still-active
  accepted head, and leaves eight rejected entries. The shared memory policy should pin the latest
  accepted record and latest evaluated record, then evict the oldest unpinned entries. These are
  generic decision identities; Stair scores, terrain progress, and other task semantics stay out of
  the shared layer.
- Capture provider prompt/cache token counts in bounded authoring evidence, then reorder stable prompt
  contracts before dynamic sections and compare cache-hit ratios on a controlled replay. Consider
  compact provider JSON only as a separately versioned follow-up if prefix caching is insufficient.
- Align provider-visible authoring guidance with the existing anti-degenerate and acceptance policy:
  numeric/string/format-only variants of an evaluated source are not valid mechanism candidates.
  Verify both Stair and Push Recovery prompts, and keep task-specific reward advice in their adapters.
- Audit experiment-memory size, retained experiment identities, omission counts, and whether
  each new candidate responds to the correct immediately available evidence.
- Compare every round's public/sealed scores and per-height outcomes with runtime diagnostics
  and selected checkpoint videos before judging evaluation realism.
- Review whether task-owned machine-generated reward contribution bounds would improve authoring
  feedback without expanding the public ABI or duplicating semantics.
- Review provider response guidance for avoidable module-level prose expressions only if this
  failure recurs often enough to justify changing the contract; one safely rejected attempt is
  not sufficient evidence.
- Clarify the task-owned `projected_gravity` sign convention and test that a provider-visible
  upright example produces a positive upright-alignment scalar. Keep the raw runtime field as the
  canonical Isaac value; do not add a second derived runtime source merely to repair documentation.

## Post-run repair closure

- Keep the shared Stair task-section limit at `32 KiB`. The measured problem was not provider
  context exhaustion: the 27 frozen calls were at most about `100 KiB`, only one contained a full
  duplicate source, and long exact duplicate strings averaged about `2.3 KiB`. Broadly deleting
  contracts, current-best source, or exact evidence would remove authoring authority for little
  byte benefit.
- Prompt serialization is now explicitly `true_rl_agent.prompt_payload.v3`. Stable contracts and
  selection guidance precede source/history/repair sections, preserving semantic content while
  increasing the measured adjacent common prefix from about `19.9 KiB` to `37.3 KiB`. Provider
  responses now retain bounded canonical `request_units`, `response_units`, total units, and request
  cache hit/miss units in private authoring-attempt evidence. The raw DeepSeek usage names are mapped
  at the provider boundary; usage is not inserted into repair prompts and contains no response text.
- Experiment memory now applies the declared `14 KiB` limit to the complete object and protects the
  latest accepted and latest evaluated identities before evicting older unpinned entries. A frozen
  R15 reserialization reduced the full object from `14,622` to `13,071` bytes. That old prompt had
  already omitted R5, so final validation must rebuild from the complete ledger and prove that R5 is
  restored and pinned; the reserialization alone is not accepted as that proof.
- Stair guidance no longer advertises ineligible coefficient-only candidates, and shared guidance
  states the existing numeric/string/format-only historical-source rule. The Stair-owned reward
  input contract now declares upright `projected_gravity ~= [0, 0, -1]` and the canonical alignment
  expression `-projected_gravity[:, 2]`. Push Recovery retains its own task semantics and tests reject
  leakage of the Stair sign advice into its prompt.
- A no-provider/no-training frozen-R15 reserialization produced a `90,924` byte v3 prompt, a
  `32,235 / 32,768` byte Stair task section, and a `13,071 / 14,336` byte memory object. The Phase D
  R0 capacity export also passed at `46,985` bytes. A complete-ledger server export remains required
  before adopting the new baseline.
- Exact task-owned Isaac trajectories were exported for R5 and R11 on the same public-dev run-1
  0.40 m environments 48, 50, and 53. R5 classified env50 as torso-contact failure while R11
  classified all three as success; terminal states and reset markers are preserved and the video
  exporter reports no reclassification. This supports a real local R11 improvement, but R11 remains
  rejected because its formal paired delta did not reach the frozen acceptance threshold.
- The final complete-ledger no-provider/no-training reconstruction closed the remaining prompt proof.
  It replayed all 15 records and the original reference Genesis, resolved private candidate sources,
  and produced a `99,303 / 262,144` byte v3 prompt. Task-owned sections were
  `30,515 / 32,768` bytes and the complete experiment-memory object was
  `13,046 / 14,336` bytes. Memory retained R5 (`round_index=5`, `accepted=true`) and R15 as the
  latest evaluated candidate while explicitly reporting seven omitted records. Candidate lineage
  and the inline current-best source both resolve R5, so current-head identity has one replayed
  authority rather than a memory/source split.
- A controlled two-call provider probe then sent that exact `99,303` byte prompt twice without
  verifier, training, evaluation, or ledger mutation. The first call reported `26,865` request
  units, all cache miss. The second reported `26,752` cache-hit units and `113` miss units out of
  the same `26,865` request units, a `99.58%` hit ratio. This validates the v3 stable-prefix order
  and the new usage-field mapping against the real provider. It does not validate either probe
  candidate and gives those candidates no experimental or acceptance authority.
