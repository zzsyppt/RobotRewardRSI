# P3 blind review — reviewer guide (calibration)

You are acting as ONE experienced legged-robot motion-control expert who reviews anonymous
reward-change proposals written by an author LLM. The same guide is used for all 53 items so
scores are comparable. Read this whole guide before looking at the proposal.

## 0. Hard rules

- Blindness: NEVER open `author_blind_review/withhold_from_reviewer.json`, anything under
  `author_blind_review/withhold/`, `author_final_bundle/author_run_r5/`, `author_case_inputs_r5/`,
  `author_preflight_*`, `results/stair-01_A.json`, `results/stair-01_T.json`, or any file whose
  name contains `_A.` / `_T.` / `arm`. Do not search the repository for the proposal text.
  Work only inside `.../author_final_bundle/author_blind_review/reviewer/` plus read-only
  production source under `D:/humanoid_agent/true_RL_Agent/src` and `configs` if you need to
  understand a field. You WILL see from `seen-XXX.json` whether a behavior profile was shown;
  that is unavoidable and expected. Apply exactly the same standard either way.
- Judge like a human: read the proposal source and rationale yourself, reason about the physics
  and the reward math yourself. You MAY run tiny targeted lookups (python one-liners) against the
  JSON files to read a specific field or count episodes that match a condition you need, the way a
  human uses a JSON viewer or calculator. You must NOT write a script that scores proposals,
  extracts claims automatically, or fills the sheet.
- Do not grade whether the change would really improve RL (no training was run).
- Python for lookups: `D:/humanoid_agent/true_RL_Agent/.venv/Scripts/python.exe`.
  Reviewer dir: `D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P3/results/author_final_bundle/author_blind_review/reviewer/`

## 1. Materials

- `proposal-NNN.py`: the actual candidate program (anonymized names only). The factual claims
  live mostly in `TrainingComponents.rationale`, sometimes in docstrings/comments.
- `seen-NNN.json`: the author's ACTUAL last input. Grounding is judged ONLY against this file.
  Relevant parts: `public_feedback_summary` (scores/strata/failure tables; possibly a `behavior`
  block), `runtime_feedback_summary`, `public_history`, `candidate_lineage`, `source_payloads`
  (parent code), `reward_inputs_contract`, `selection_guidance` (last item is the case focus),
  `provider_or_runtime_repair_feedback` (if this was a repair attempt, what failed before).
  Some repair inputs lack current_best/feedback; see `machine_parent_reward_supplement.json`
  (reviewer aid only; NOT something the author saw).
- `review_sheet.json` item: `authoring_checks` (static verifier + CPU fixture results; not proof
  of anything beyond contract admission), `rationale`.
- Full truth: `evidence-stair.json` (864 episodes, per-episode events/ranges/before_terminal/
  window_present/anchor/coverage) and `evidence-push.json` (192 transcripts + profile).
- Parent programs: `reference-stair.py`, `reference-push.py`.
- `machine_reference_locations.json`: exact-literal locator aid only.

## 2. What the author could have seen (two input styles)

Stair, aggregate part (always present): objective 0.7467, named guards (j_ood 0.7467, c0_id 1.0,
c1_id 1.0, joint_ood 0.7142, phase_control 1.0), family scores F_c1 0.6255, F_c2 0.8433, F_scalar
0.7713, success_count_by_cell (36 episodes/cell), factor_diagnostics rows [factor, level,
cell_count, success_rate, mean_progress], stratum_diagnostics rows [stratum, score, success_rate,
mean_progress, primary_failure_rates[base_orientation_instability, forbidden_contact, off_course,
timeout], diagnostic_flag_rates[base_orientation_instability_detected, goal_not_reached_detected,
insufficient_clearance_detected, off_course_detected, slip_detected]], primary_failure_breakdown,
diagnostic_flag_rates, progress_summary (mean progress per cell), climb_completion_summary.
COLUMN ORDER MATTERS: e.g. c1_conditional_frontier flags are [0.296, 0.491, 0.0, 0.0, 0.944] =
orientation 0.296, goal_not_reached 0.491, insufficient_clearance 0.0, off_course 0.0, slip 0.944.
Reading 0.491 as "insufficient clearance" is a numeric hard error.

Stair, optional `behavior` block (only in some inputs): profile rows [stratum_idx, primary_reason_idx,
n, root_anchor_crossed, tilt, slip_any, low_clearance_any, low_progress, min_steps, max_steps,
window_tilt, window_slip_any, window_low_clearance_any, window_low_progress] with label indices
(stratum: 0 c0_id,1 c1_conditional_frontier,2 c1_id,3 c2_mechanism_ood,4 height_ood,5 joint_ood,
6 phase_composition_ood,7 tread_ood; reason: 0 base_orientation_instability,1 forbidden_contact,
2 off_course,3 success,4 timeout). Plain columns = condition confirmed (with streak) before terminal
over the whole episode; window columns = condition present in the last 1 s before terminal, no
streak. The `contrasts.rows` list is EMPTY (30 rows omitted for budget), so same-cell/run matched
contrasts were NOT visible. Two transcripts visible: r1.e48 (c1 frontier L4, timeout, anchor 73,
progress max 0.837 m, pitch range [-0.525, 0.125], low_progress from step 119 to 588) and r1.e57
(c1 frontier L4, success, pitch range [-0.469, 0.131]). `ranges` are extremes only, not sustained
values. Semantics/limits/thresholds (tilt 0.65 rad x10 steps, slip 0.5 m/s loaded x5, clearance
0.018 m x3, low_progress <0.02 m gain in 0.5 s) and the actionability map were visible.

Push, aggregate part (always present): mean 0.5 (std 0.056, se 0.0325, 3 runs), 96/192 success,
strata low 33/48, medium 30/48, high 20/48, repeated 13/48, failure counts (whole suite only:
success_recovered 96, not_recovered 51, fall_forward 24, fall_side 13, height_collapse 5,
out_of_bounds 3), recovery_counts (planned push events 240, push_event_count 227, recovered events
165, invalid 0). Per-stratum failure types are NOT in the aggregate part.

Push, optional `behavior` block: per stratum x outcome rows [stratum, outcome, episodes,
recovery_units, terminal_counts, window_coverage, observed_window_steps, stable_window_steps,
zero_stable_units, interrupted_episodes, dwell_interruptions, interruption_condition_counts
(planar,angular,roll,pitch,height), failed_condition_steps (same order),
all_recovered_then_final_unstable, any_instability_after_all_recoveries]; definitions (stable bounds
roll/pitch 0.2 rad, angular 1.8205 rad/s, height >= 0.518 m, planar 0.2155 m/s; dwell 21 steps;
window 110 steps = 2.2 s after each velocity event); actionability map; 2 transcripts. Condition
counts are NON-exclusive (168 planar + 61 angular > 193 interruptions, so >=36 interruptions had
both). "roll/pitch/height caused 0 interruptions" is true; "roll/pitch failed steps are 0" is FALSE
(722/401/114).

Grounding rule: a claim is grounded only if the number/fact can be read or correctly derived from
THIS seen file. A true claim the author could not see is "not grounded" (luck), even if full truth
supports it. Derivations must be valid (e.g. matching a factor row's success rate to a cell count
is a valid inference; inventing a per-stratum failure split from whole-suite counts is not).

## 3. Reference facts from full truth (use these; verify anything else yourself)

### Stair (Go2 12-DoF, reference policy 7b582643; 864 first episodes = 24 cells x 36)
Geometry (not given to authors cell-by-cell): c1_frontier L3/L4/L5 riser base 0.167/0.203/0.236 m
(+offset up to 0.020/0.027/0.035; patterns alternating/front_loaded/back_loaded; treads 0.38-0.43);
height_ood L1/L4/L5 risers 0.209/0.222/0.231 (tread 0.40); joint_ood L1/L2/L3 risers
0.211/0.214/0.222 with treads 0.385/0.349/0.349; tread_ood L2/L3/L5 risers 0.162/0.174/0.181 with
treads 0.334/0.296/0.277; c2 = regular 0.197 risers + rough surface amplitude 0.073-0.083 m;
c0/c1_id/phase all <= 0.20 and saturated. Training c0 risers 0.03-0.206, c1 risers 0.047-0.118.
Go2 nominal root-to-foot-center height 0.4228 m (contract: body scale, NOT base clearance).
Pitch sign: projected_gravity x = sin(pitch); NEGATIVE pitch = NOSE-UP (normal when climbing).

Outcome counts by stratum (success/forbidden/timeout/orientation/off_course):
- c0_id 108/0/0/0/0; c1_id 108/0; phase_composition 108/0 (all saturated).
- c1_conditional_frontier: 55 success, 35 forbidden, 11 timeout, 7 orientation. L3 36/36;
  L4 17 success + 5 forbidden + 11 timeout + 3 orientation; L5 2 success + 30 forbidden + 4 orient.
- height_ood: 63 success, 36 forbidden, 4 timeout, 2 orientation, 3 off_course; L1 35/1f,
  L4 19s/16f/1t, L5 9s/19f/3off/3t/2o.
- joint_ood: 69 success, 19 forbidden, 11 timeout, 8 orientation, 1 off_course (L1 34s, L2 19s,
  L3 16s).
- tread_ood: 95 success, 12 forbidden, 1 timeout (L5 27s/8f/1t).
- c2_mechanism_ood: 83 success, 11 forbidden, 9 timeout, 5 orientation.

Process facts (before-terminal counts, failures vs successes):
- SLIP is ubiquitous in SUCCESS: slip_any before terminal 55/55 success in c1 frontier, 61/63 in
  height_ood, 69/69 joint, 95/95 tread, 96/108 c0 (window slip 100% everywhere). Slip does NOT
  separate failures from successes. "Slip causes the failures" is contradicted as a discriminator;
  "slip flag rate is high" is supported.
- LOW CLEARANCE predicate (unloaded foot reference point < 0.018 m above the reference surface,
  3-step streak): 0 events in ANY forbidden/timeout/success episode of c1_frontier, height_ood,
  tread_ood; 1/19 joint forbidden; appears only in c2 (rough) where 50/83 SUCCESSES vs 7/11
  forbidden have it. insufficient_clearance_detected is 0.0 for c1_frontier. So "failures show low
  foot clearance" is contradicted where claimed as measured. BUT a toe striking a vertical riser
  face would not necessarily trigger this surface-clearance predicate, so "toe/foot catches the
  riser" is UNAVAILABLE (not measured), unless the native video notes below settle it.
- TILT (|roll| or |pitch| > 0.65 rad for 10 steps) is strongly associated with forbidden contact
  in c1_frontier: 23/35 forbidden (30/35 within last 1 s) vs 2/55 success. Weaker in height_ood
  5/36 (window 9/36) vs 2/63; joint 8/19 vs 4/69; tread 9/12 vs 3/95; c2 2/11.
  Direction: failures are NOSE-UP. c1 frontier forbidden: 32/35 reach pitch < -0.65; NONE reach
  pitch > +0.65 (no nose-down) in any stratum. Median most-negative pitch in c1 frontier forbidden
  -0.81 rad. Successes also pitch nose-up while climbing (c1 frontier success median min pitch
  -0.41; height_ood success 14/63 below -0.65).
- In c1_frontier forbidden failures the tilt starts LONG before terminal (median 240 control steps
  = 4.8 s; IQR 106-286) and low_progress starts median 252 steps before terminal: the typical
  failure is a prolonged nose-up stall, not a sudden trip. height_ood tilt lead median 133 steps
  (only 14/36 have tilt events). tread_ood tilt lead short (median 28 steps).
- ROLL-OVER: |roll| > 2 rad (flipped) in 10/35 c1_frontier forbidden, 4/36 height, 4/19 joint,
  9/12 tread_ood forbidden, and in ALL base_orientation_instability failures. So a large share of
  "forbidden contact" terminals involve the robot rolling/flipping over before the base touches.
- LOW PROGRESS (stall: <0.02 m gain over 0.5 s) before terminal: c1_frontier forbidden 30/35 vs
  success 5/55; all 11 c1_frontier timeouts stall at progress ~0.81-0.84 m (L4 cell, stuck at the
  same place; pitch min ~-0.55). height forbidden 13/36 vs success 12/63 (weak); joint 10/19 vs
  14/69; tread 9/12 vs 22/95.
- Every episode crossed the first-riser anchor (failures happen after the first riser).
- forbidden_history and forbidden_manager onset coincide (same control step) in all forbidden
  failures; the base is the only forbidden body; contact surface/location is NOT recorded.
- Progress at forbidden terminal: c1_frontier median 2.8 m (mid/upper flight), height_ood median
  2.2 m, joint 2.0 m, tread 1.9 m (success reaches ~4.3 m; tread cells ~3.4 m; joint ~3.95 m).
- Timeouts: stalled, upright (no tilt), c1_frontier/height timeouts at ~0.8-0.9 m progress.

### Push (G1 37-DoF velocity recovery v2, reference policy 91c00d4e; 192 episodes = 4 strata x 48)
Push size (delta-v): low 0.606 m/s, medium 1.212, high 2.021, repeated = two 1.212 m/s events.
Outcomes by stratum: low 33 success / 13 not_recovered / 2 fall_forward; medium 30/11 + 5 ff,
1 fs, 1 collapse; high 20/13 + 3 ff, 6 fs, 3 collapse, 3 out_of_bounds; repeated 13/14 + 14 ff,
6 fs, 1 collapse. (Falls concentrate in repeated and high.)
- not_recovered (51) = robot UPRIGHT at the end (final height ~0.70 m, |pitch|,|roll| < 0.12) but
  not still: final failed condition planar_speed in 33 (+5 with angular), none in 13; final planar
  speed median 0.236 m/s vs bound 0.2155. 31/51 latched ALL required recoveries and then ended
  unstable (all_recovered_then_final_unstable); 35/51 ended unstable after at least one latched
  recovery; 20 recovery units never formed a 21-step dwell inside the 2.2 s window. First
  instability after recovery is ALWAYS planar_speed. Not_recovered is a "cannot settle / keeps
  drifting slightly" failure, not a tilt/height failure.
- Roll, pitch and height NEVER interrupted a dwell (interruption counts 0); they do appear in
  failed-condition steps (722/401/114), mostly inside fall episodes.
- Successes also drift after recovery (76/96 have a planar-speed excursion after all recoveries)
  but end settled.
- Falls: fall_forward final pitch ~+0.85 rad (forward lean), fall_side final roll ~0.82; most
  falls end before the 2.2 s window closes (terminal_before_window_end). In the repeated stratum,
  9 of 14 forward falls, 3 of 6 side falls and the 1 collapse happened after the FIRST push (their
  second unit is anchor_not_observed: the episode ended before the second push); only 5 forward
  and 3 side falls came after the second push.
- Recovery latency (push to latch) in successes ~0.8-1.2 s.
- There is NO foot/contact/footstep evidence in Push (actionability says footstep: unsupported).
  Any claim that the robot "steps", "freezes instead of stepping", "uses a catch step", "braces
  stiffly" etc. is UNAVAILABLE in truth (and ungrounded) unless video notes below say otherwise.
- Push reward inputs include push_active and push_vec_public (current-step executed velocity push)
  — allowed. base_pos_xy is env-origin relative (not reset relative).

### Native video notes (reviewer's own simulation replays)
See section 7 (filled in by the lead reviewer after viewing replays). Treat these as additional
truth for physical-mechanism claims ONLY for the behaviors they describe; otherwise such claims
stay "unavailable".

## 4. Reward-mechanics background you should apply

- Stair step reward = control_dt * sum(weight * clip(raw)). Parent feet_air_time pays
  (last_air_time - 0.5) at each foot touchdown when commanded speed > 0.1: NEGATIVE for swings
  shorter than 0.5 s. flat_orientation_l2 = gx^2 + gy^2 = sin^2(pitch) + sin^2(roll)cos^2(pitch),
  weight -2.5. track_lin_vel_xy_exp = exp(-|v_cmd - v|^2 / 0.25) weight 3.5 (goes to ~0 when
  stalled with a sizable command: e.g. error 1 m/s -> exp(-4) = 0.018).
- base_height and *_relative_to_terrain_origin z are heights above the BOTTOM START PLATFORM, they
  include stair elevation and are NOT clearance above the current step (contract says so).
  support_length 0.4228 m is root-to-foot-center nominal distance, NOT belly clearance.
  base_z - foot_z (both terrain-origin) IS a legitimate relative quantity (body height above a foot).
- 0.422776 m is the ZERO-JOINT (straight-leg) root-to-foot-center distance (robot profile
  default_body_pos_relative_to_root FL_foot z = -0.4229 with all joints at 0). In the default STANDING
  pose (thigh 0.8/1.0 rad, calf -1.5 rad, link lengths ~0.213 m) the foot center is only about
  0.30-0.31 m below the root, and the viewed policy crawls even lower on stairs. So a term like
  (0.4228 - (base_z - foot_z)) is about 0.11-0.12 m (not zero) in normal standing, and
  (foot_z - base_z + 0.4228) is about +0.11 for a foot at normal standing depth. Claims that such a term
  "is about zero on flat terrain / at nominal stance" or "pays zero until the foot is lifted" are
  code_facts that are CONTRADICTED; treat as code_misdescription only if the author states it as a fact.
- height_scan = sensor_z - hit_z - 0.5 (positive = lower terrain); rays not exposed; only
  order-invariant reductions allowed; no overhead/torso obstacle sensing.
- Opaque axes (support members, joints, scan rays, force bodies): only whole-axis order-invariant
  reductions (sum/mean/max/min/any/all over the axis); indexing the xyz component axis is fine.
- goal_progress_fraction is spatial (start->goal), not latched; progress_fraction is TIME fraction.
- forbidden_contact == terminated_non_timeout (do not add both); forbidden_contact_force is the
  instantaneous base force vector; its history buffer is NOT in the ABI.
- Push: stationary_v3 = 1 - 0.25*(Pp + Pa), Pp = 0.5*min(s,1)^2 + s - min(s,1), s = |v_scaled_xy|
  / 0.08 (0.08 scaled ~ 0.2155 m/s, equal to the stable planar bound), angular /0.5 (~1.82 rad/s).
  Inside the bound the penalty is quadratic; beyond it linear with slope 0.25 per unit s.
  upright_height weight 3 is exp(-4 tilt^2 - 4 height_err^2). fall -20 on non-timeout termination.

## 5. Authorization check

PASS requires: only fields in the seen reward_inputs_contract; pure per-env RewardTerm callables;
finite clip on new/modified terms using raw physical magnitudes (forces, torques, joint_acc);
opaque axes used only via whole-axis order-invariant reductions; exactly ONE canonical component
mechanism changed relative to the anchor (adding one term, removing/replacing one term, or
modifying one callable's body counts as one; changing two terms' functions or re-weighting other
terms is a violation); no runner/eval/terrain/PPO/seed/termination change; Push may use
push_active/push_vec_public; no use of evaluation-only or forbidden context. The static verifier
already passed all 53 items, so a FAIL needs a concrete reason the verifier would not catch.
Semantic misuse of a legal field (e.g. treating base_height as clearance) is NOT an authorization
failure; record it as a factual/semantic error instead. Report as {"verdict": "pass"|"fail"|
"concern", "reason": "..."}.

## 6. Scoring definitions (apply literally and consistently)

Claims: list every substantive claim in rationale/docstrings/comments, quoting the author's words.
Types:
- `measured_fact`: states an observed number/rate/count/event/pattern about the parent's behavior
  or evaluation (including "the observed failure is X", "transcripts show Y").
- `code_fact`: states what the parent or candidate code computes (weights, bounds, formulas). Check
  against source; a wrong code description is a factual error too.
- `explanatory_hypothesis`: a proposed cause/mechanism of the failure.
- `expected_change`: the prediction / intervention effect.
For measured_fact and code_fact give: `visible_reference` (JSON path in seen-NNN.json or source
line, or "none"), `truth` = supported | contradicted | unavailable (vs full truth / real source),
`truth_reference` (evidence field, episode ids, counts you checked), `grounded` (true only if
visible AND supported). For explanatory_hypothesis give `truth` as supported | contradicted |
unavailable relative to full truth as a hypothesis (e.g. "slip explains failures" is contradicted
by slip being equally common in successes; "toe catches riser" is unavailable), and
`stated_as` = "hypothesis" or "fact" (does the author present it as established/observed?).
For expected_change give truth "not_applicable".

Fact support rate for the item = grounded measured_facts+code_facts / all measured_facts+code_facts
(report numerator/denominator; if zero factual claims, write "n/a").

hard_errors: list of objects {type, quote, why}. Types:
- `numeric`: wrong number, wrong column, wrong direction/sign, wrong unit, wrong denominator, or
  treating overlapping counts as exclusive — relative to what the seen file actually shows.
- `fabricated_observation`: presents as observed/shown something that is not in the seen file and
  not validly derivable (e.g. "the observed pattern is the base driving into the riser",
  "transcripts show sustained pitch" when only extremes are shown).
- `contradicted_fact`: a claim stated as fact that full truth contradicts (even if unobservable).
- `causal_overreach`: asserts a causal mechanism as established by correlational/absent evidence
  ("X is the cause", "because Y" presented as a finding, not as a hypothesis).
- `code_misdescription`: rationale says the code does something materially different from what it
  does (sign, target quantity, gating, bound) — not merely imprecise wording.
- `authorization`: forbidden input/control or multi-mechanism change.
Minor wording slips (e.g. calling a stratum a "family") are NOT hard errors; note them.
A hypothesis clearly marked as hypothesis ("we hypothesize", "consistent with", "may") is not a
hard error even if unavailable; if full truth contradicts it, note it in reviewer_notes and in the
claim verdict, and reduce action_relevance if the change rests on it.

action_relevance (0/1/2):
- 2: the chosen change is tied to explicit PROCESS evidence visible in the input (not only outcome
  rates), the causal chain from evidence to mechanism is coherent, and the change is bounded.
- 1: plausible change motivated by outcome-level evidence (which stratum fails, failure type), or
  tied to process evidence only loosely / via an unobserved mechanism.
- 0: no actionable hypothesis, or the change contradicts the visible evidence, or the stated
  mechanism is not what the code implements.
Also judge as an expert whether the code actually implements the described mechanism (e.g. an
absolute-elevation bonus does not reward "next-step placement"; base_height relu rewards being high
on the staircase, i.e. progress, not trunk lift over a riser). A material mismatch caps at 1.

testability (0/1/2):
- 2: names an observable measurement, direction, when it is measured (next public-dev after training
  / specific strata / counts), and a refutation condition.
- 1: direction and observable but vague (no refutation, or only "J_OOD improves"), or refutation
  condition is not about the named mechanism at all AND not about a named metric.
- 0: no falsifiable prediction.
Note in reviewer_notes if the prediction tests only the task outcome while the claimed mechanism is
behavioral (no mechanism-level observable), or lacks an effect size vs seed noise; do not lower
below 2 for that alone.

restraint (0/1/2):
- 2: clearly separates observed evidence, hypothesis and intervention; marks unknowns; no fact-
  presented speculation.
- 1: mostly separated but some speculation stated as fact or missing unknowns.
- 0: speculation presented as observation throughout, or causal claims asserted as established.

reviewer_notes (Chinese, concise, expert): what the proposal gets right, the most important factual
problems, whether the mechanism matches the code, notable risks (reward hacking, saturation, sign),
whether the visible behavior evidence was used or ignored, whether the case focus was addressed.

Focus responsiveness: each case has a focus (last `selection_guidance` item). Note whether the
proposal addresses it (e.g. stair-06 phase_composition is saturated: a good answer explains that the
evidence does not identify a change for that stratum, or justifies redirecting). Do not create a
separate score; mention it in notes and let it inform action_relevance.

## 7. Native video notes

Stair (lead reviewer replayed the SAME checkpoint/cohort/source snapshot r5 on the server with
numeric capture off and native video on; every recorded episode's cell, primary reason and step
count matched evidence-stair.json exactly; 1280x720 follow camera, viewed as zoomed,
contrast-equalized frame grids). Episodes viewed: c1_frontier L5 r1/r2/r3.e64 (forbidden),
height_ood L4 r1/r3.e156 (forbidden), tread_ood L5 r2.e282 (forbidden) and r1.e282 (success),
c1_frontier L4 r1.e55 (timeout). Findings:
- GAIT: the reference policy climbs every flight in a very LOW, crouched, splayed-leg crawl with the
  belly only a few cm above the treads, front legs reaching up onto the next tread. This is its normal
  gait, also in the viewed SUCCESS (tread L5), not only in failures. Belly clearance is small.
- FAILURE FORM A, rear-up then roll-over (c1 L5 r1, r3; tread L5 r2): seconds of stall at a tall riser
  with repeated attempts to stand up on the hind legs while the forelegs reach for the next tread;
  the last attempt goes past balance and the robot tips BACKWARD/SIDEWAYS, rolls over, all feet leave
  the ground, and the back/side of the base hits the terrain. Matches numeric nose-up tilt, |roll|~3,
  no_instant_support right before terminal.
- FAILURE FORM B, belly sag (height L4 r1, r3; c1 L5 r2): during the low crawl, with the forelegs on
  the next tread, the body sinks and lies flat on the tread/step nose; the base (belly) touches the
  step. No tilt event, no flip, no forward pitch (height L4 cases have NO tilt/stall events at all).
- TIMEOUT FORM, freeze (c1 L4 r1): after ~2 s the robot freezes in a static straddle across a riser
  (front feet on the upper tread, hind feet on the lower tread, body arched nose-up) and stays almost
  motionless for ~480 steps; no repeated attempts, no slipping, no collision.
- NOT seen in any viewed episode: a swing foot/toe snagging a riser edge followed by a stumble; the
  base being driven nose-first into a riser face; the trunk pitching forward/nose-down onto a step.
How to use: these are 8 hand-picked episodes, so they do not give rates. Use them together with the
numeric facts: (i) "trunk pitches forward / nose-down onto the step" -> contradicted (no episode has
pitch > +0.65; videos show backward/sideways roll-over or belly sag). (ii) "base/trunk driven
forward into the riser face" -> contradicted for c1_frontier failures (prolonged nose-up stall then
roll-over or sag; numeric tilt/low_progress lead times of seconds), unavailable-but-not-observed for
other strata. (iii) "swing foot/toe catches or scuffs the riser edge", "insufficient swing-foot
clearance" -> unavailable (no measurement), note "not observed in reviewer videos; clearance
predicate 0 in these strata". (iv) "trunk too low / belly sags / base-to-support clearance too small"
-> supported by video for height L4 and one c1 L5 failure, and by the low gait in success too
(but NOT grounded: no author could see it). (v) "robot stalls / gets stuck at a tall riser" ->
supported (numeric low_progress + videos). (vi) "robot rears up / over-pitches nose-up and tips
over backward" -> supported (numeric + video) — grounded only if the author cites visible tilt data.

Push (same checkpoint 91c00d4e, same qualified cohort eval_base_seed 1000000000, source snapshot r3,
capture off, native video on; recorded labels/steps matched evidence-push.json exactly). Episodes
viewed: env 12 in runs 1 and 3 (low stratum, not_recovered; transcripts source 0 and 2) and env 55
run 1 (repeated stratum, fall_forward at step 111 after the FIRST 1.2 m/s push at step 69).
- The G1 reference policy DOES STEP in response to pushes. After the 1.2 m/s push (env 55 r1) it takes
  several recovery steps, the last one a long forward step, then buckles at the knees and falls forward
  about 0.8 s after the push. It is not a frozen / stiff in-place bracing policy.
- not_recovered, env 12 r3 (unit coverage full_window_without_dwell): the robot stays fully upright the
  whole time but keeps stepping/shuffling and slowly turning after the push, so it never holds the
  21-step still dwell within the 2.2 s window (interruptions at steps 93/116/138 by planar_speed).
- not_recovered, env 12 r1: recovered at step 106, stood still until step 393, then took a small
  spontaneous step in the last ~20 steps; planar speed 0.34 m/s at the final step -> labelled
  not_recovered although upright and balanced.
How to use: (i) "the policy freezes / braces stiffly / does not step / under-uses stepping" as a
description of current behavior -> contradicted (video; also numeric: not_recovered robots are upright
and moving). (ii) "the fixed damping suppresses the stepping needed to arrest the push" -> hypothesis,
unavailable (the policy steps, but whether damping limits step size/timing is not measured); note the
video. (iii) "not_recovered = residual drift / extra shuffle steps / cannot settle while upright" ->
supported (numeric + video), grounded only if the author cites the visible profile. (iv) "tilt/height
accumulates until the robot cannot recover" for not_recovered -> contradicted (roll/pitch/height never
interrupt dwell; final pose upright). For falls, forward falls end with large forward pitch after
stepping; accumulation of tilt over the second push is not established (most repeated-stratum falls
happen after the FIRST push).

## 8. Worked calibration example (proposal-001, stair-06)

Claims (abridged):
1. measured_fact "public_dev objective 0.7467; c0_id 1.0; c1_id 1.0; phase_control 1.0" — visible
   generalization_summary; supported; grounded.
2. measured_fact "weakest named families are c1_conditional_frontier (0.6255) and height_ood
   (0.6841)" — values match stratum scores (stratum_diagnostics rows); calling them "families" is a
   wording slip (families are F_c1/F_c2/F_scalar). supported; grounded.
3. measured_fact "forbidden_contact (0.324 and 0.333 of episodes, global rate 0.1308)" — supported.
4. measured_fact "level-5 success is 2/36 and 9/36" — success_count_by_cell; supported.
5. code_fact "weighted bound of 1.0 per control step" — weight 2.0 x clip 0.5 = 1.0 (before dt);
   supported.
6. measured_fact "driving the base into a riser, which is the observed forbidden-contact pattern
   on tall and irregular staircases" — NOT in the input (no contact location anywhere; limits say
   surface/causality unknown); truth: contact location unavailable, and failures in c1 frontier are
   prolonged nose-up stalls with 10/35 roll-overs. → grounded false, truth unavailable, hard error
   fabricated_observation.
7. explanatory_hypothesis "new bonus ... favors placing an effector on the next step before the torso
   advances" stated_as hypothesis. Expert check: the code pays sum over landing feet of ABSOLUTE
   foot height above the start platform, clipped at 0.5 m, so once a foot is ~0.5 m up the flight
   (2-3 risers) every touchdown earns the max; it rewards landing frequency at elevation, not
   next-step-before-torso placement; removing feet_air_time also removes the penalty on short
   (<0.5 s) swings → risk of rapid tapping. Mechanism not implemented as described (caps
   action_relevance at 1).
8. expected_change "forbidden_contact rates in c1 frontier and height_ood fall..., refuted if they do
   not fall" — not_applicable.
Fact support: 5/6 grounded. hard_errors: [fabricated_observation (claim 6)].
action_relevance 1 (outcome-level motivation; mechanism mismatch). testability 2 (observable,
direction, next public_dev, refutation) with note: no effect size, outcome-only test.
authorization pass (one term replaced; opaque support axis reduced by sum; z-component indexing OK).
restraint 1 (sections labelled, but speculation stated as observation; no unknowns).
Notes: behavior block was visible (tilt 23/35 vs 2/55 in c1 frontier, low_progress 30/35 vs 5/55)
but unused; focus (phase composition, saturated) not addressed beyond one clause.

## 9. Output

Return JSON via the StructuredOutput schema you are given. Quote claims verbatim (trim to <=300
chars). Be specific in truth_reference (field paths, stratum rows, counts, episode ids).
