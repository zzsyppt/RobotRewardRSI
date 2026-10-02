# Independent audit: proposals 003 and 042

Saved on 2026-09-24 before opening any original review_sheet.json entries. The human_review_20260924 guide and report were not read. This is a bounded audit of the two proposals, not a suite regrade. Only audit documents are written; no API, simulator, training, replay generation, or candidate execution is used.

## Basis and evidence hierarchy

The original rubric says: action relevance 0=no actionable hypothesis, 1=plausible but weakly tied, 2=explicit process evidence supports a bounded change; testability 2 requires named observable, direction, window and failure condition. Hard errors include invented facts, wrong direction/unit/denominator, treating correlation as proof, or forbidden control/input. It explicitly excludes grading whether unrun training improves RL performance. Restaint and authorization have qualitative definitions, not a numeric rubric.

Files are under `PLAN/feedback_sys/P3/results/author_final_bundle/author_blind_review/reviewer/` unless stated otherwise. CodeGraph was queried before code exploration; it returned unrelated indexed source instead of the exact generated files and Stair runtime source, so those uncovered files were read directly.

`seen-003.json` is a static-repair frame. `authoring_context_frame.omitted_sections` explicitly omits public_feedback_summary, runtime_feedback_summary and public_history as nonrequestable; `source_payloads[0]` contains the previous failed candidate including its rationale. Repair feedback requires fixing only the opaque support-axis index and preserving the mechanism. Thus claims inherited from that rationale are visible to the final author but are not independent measured observations in this final frame. This distinction matters: it is neither an absent-from-input invention nor sufficient primary evidence merely because a prior model wrote it.

`seen-042.json` includes the reference source, public aggregate feedback and the full visible reward/input and evaluation contracts. Its runtime feedback is `not_available` with `no_latest_nonaccepted_trial`; no episode command/speed series or reward gradients are provided. Its repair is output-budget exhaustion, not a semantic rejection.

`evidence-stair.json` is reviewer-side full behavioral evidence, not something these two final seen files expose. It contains 864 first episodes, with reference-point/contact/event limits explicitly declared. It cannot alone identify a toe/riser collision surface or prove a reward intervention's counterfactual effect.

## Independent findings: 003

### What the code actually does

`proposal-003.py:49-54` computes `d=max(0, base_height-support_z)`, pays an airborne support `exp(-max(0,d-0.30)/0.08)`, and sums all four support members. Its added weight is 0.4 and final clip is [0,4] (`:69-75`). `base_height` and support z both reference the same terrain origin, so their difference is root-to-support **world vertical separation**. The subtraction correctly cancels accumulated staircase elevation. It is not measured clearance over the local surface, a toe/sole position, a root-to-foot Euclidean length, or a plane aligned with a pitched trunk. Its prose “trunk plane” is loose if interpreted literally.

All eight reference function bodies at lines 6-46 are identical to `reference-stair.py:6-46`, and all eight weights are unchanged. There is exactly one additional RewardTerm. The repaired `[:, :, 2]` selects the declared spatial z component and then uses a whole-axis sum; it no longer picks an opaque support member. The previous code in `seen-003.json:source_payloads[0].content` used `[:,2]` and verifier feedback explicitly names `opaque_input_axis_selection:support_pos_relative_to_terrain_origin:line_52`.

The analytic bounds are correct: each airborne term is in [0,1], raw sum in [0,4], weighted rate in [0,1.6], maximum per-control-step reward 0.032 when dt=0.02. The maximum reference linear tracking rate is 3.5, so the maximum-to-maximum comparison is valid. It does not establish small actual contribution or non-domination, because actual tracking can be much smaller than its maximum. The author's sentence only compares bounds, so do not convert that caveat into a proven numerical hard error.

### Facts, hypotheses and unknowns

- `:79-80` J_OOD 0.7467 is visible in candidate_lineage (exact 0.7466970595216049).
- `:98-102` dominant forbidden-contact and weakest score strata are inherited in the visible failed-source rationale, and confirmed by full truth. Their measured summaries are omitted from this repair frame. Do not call the inherited numbers newly invented; do not call the final frame an independent remeasurement either.
- The stall near 0.84 m and pitch near -0.525 in `:101-103` match `evidence-stair.json:episodes[episode_id=r1.e48]`: frontier level 4, timeout at step 588, progress range [-0.02056,0.83743] m, pitch range [-0.525434,0.125286] rad. `before_terminal.low_progress=true`. These are real measured extrema; extrema alone do not prove simultaneous pitch/progress or the asserted collision sequence.
- `:102-104` “consistent with a swing foot hanging near full extension so the toe catches a tall riser and the trunk pitches onto the step” is an unobserved candidate mechanism, not a measured fact. “consistent with” is hypothesis language and is followed by an explicit “Hypothesis:” label. The narrative does not demonstrate toe location, leg extension or causal order, but it also does not explicitly claim a proven cause. Under the original rubric, unsupported plausibility should reduce relevance/restraint; it is not automatically the same as treating correlation as proof.
- Full evidence has zero confirmed low_clearance_any episodes in the entire frontier (0/108), 3/108 in height_ood and 1/108 in joint_ood. This provides no positive measured low-clearance support for the claimed frontier mechanism. It does **not** causally disprove a toe hitting a vertical riser: that diagnostic measures an unloaded support reference point relative to a surface and declares it is not toe/sole center or contact surface. The reference-point predicate and toe collision hypothesis are different propositions.
- `:104-109` clearance improvement, contact reduction and timeout reduction are untested intervention predictions. No observed training outcome exists here. Paying airborne feet can also change swing duration/support occupancy; that is a possible side effect to test, not a demonstrated reward failure.

### Tentative independent assessment

- Action relevance: **1/2**. There is an actionable bounded lift hypothesis tied to real tall-riser failure/timeout evidence, but no explicit measured foot-clearance/extension/collision process supporting this exact intervention. Code repair is concretely relevant to the verifier failure; this alone is not behavioral AR=2.
- Testability: **2/2** under the supplied rubric. `:109-113` names the next public_dev run, three named strata, success increase and forbidden-contact decrease, plus parent-relative c0/c1/phase protection. Failure is any promised direction/guard not being met. It could be stronger with a direct clearance mediator and quantified minimum effect, but those extra requirements are not in the original testability definition.
- Authorization: **yes**, read-only inspection of code and visible contracts: declared inputs, pure stateless reward, one mechanism, preserved eight reference mechanisms. No forbidden runtime state/control is used.
- Restraint: **partial**. A mechanism is labeled as a hypothesis and the outcome is a future prediction. It should more clearly label the inherited foot-catch story as unobserved and avoid implying local clearance from vertical separation. It does not claim an already demonstrated training benefit.
- Hard error: **none conclusively established by this independent reading**. There is a weakly supported mechanical story and imprecise “trunk plane” wording. Calling it an invented measured fact requires a stricter reading than its “consistent with” and “Hypothesis” phrasing. This tentative result intentionally keeps uncertainty instead of grading an unrun counterfactual.

## Independent findings: 042

### What the code actually does

`proposal-042.py:49-51` returns spatial progress fraction times clipped **body-frame** x velocity, then applies weight 2.0 and final clip [0,1] (`:66-72`). The added weighted reward rate is [0,2], at most 0.04 per 0.02 s step, below the reference tracking maximum 3.5. The reference eight function bodies and weights are unchanged. One allowed mechanism is added.

The current runtime verifies the frame: `src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py:206-209` assigns world and body velocities separately; `:227-230` separately computes yaw-frame velocity; `:282` exposes body velocity as base_lin_vel. Spatial goal progress is independently computed from world stair x at `:256-265`.

Consequently the broad physical guarantee in `proposal-042.py:88-90` (“without paying for ... in-place hopping, or retreat”) is not secured by the code. A simple state counterexample, not a claim about an observed trajectory: yaw=roll=0, pitch=-0.5 rad, world velocity=(0,0,0.5) m/s, goal progress=0.5. Then body vx = -sin(-0.5)*0.5 = 0.23971277 m/s and added rate = 2*0.5*0.23971277 = 0.23971277, despite zero stair-x progress velocity. World velocity=(-0.1,0,0.5) gives positive body vx about 0.1519545 while retreating along stair x. The term is indeed zero when body vx<=0, not for every physically stalled/hopping/retreating state. This is a code-semantics overclaim, not evidence that such exploitation occurred.

### Grounded numeric claims and metric caveat

The author's `:82-84` numbers are grounded in `seen-042.json:public_feedback_summary.feedback_entries[0].feedback`: c1 score=0.6255176638425926; level_4 mean progress=0.5842113844444444 and success 17/36; level_5 progress=0.6411168208333333 and success 2/36; height level_5 progress=0.5894963886111111 and success 9/36; forbidden rate=0.13078703703703703. These are aggregate mean progress values, not demonstrated stopping locations (“stalls at progress 0.589” overstates a mean as a process location).

The test at `:93` changes the word to “success above 0.6255.” Actual frontier success is 55/108=0.5092592592592593. Score is (0.5092592592592593+0.7417760684259259)/2 = 0.6255176638425926. Thus 0.6255 must not be reported as the **current success baseline**. However, the literal text can validly specify success>0.6255 as a stricter future target; it does not explicitly say “current success is 0.6255.” Flag metric ambiguity and seek a precise metric label, rather than automatically counting an invented baseline hard error.

`c0_id,c1_id >=0.95` does not by itself show the named guard-min criterion holds. Seen public_eval_contract names j_ood, c0_id, c1_id, joint_ood, phase_control and a seed-min/mean aggregation. Proposal `:94-95` says this ensures the “0.05 paired non-regression cap” but omits other named guards and per-seed behavior. This is a genuine overstatement of an evaluation condition, not authorization to change that condition.

### Mechanism claim and mathematical check

`proposal-042.py:85-87` asserts the anchor exponential tracking “collapses toward zero when the commanded speed is unreachable ... so it stops supplying forward gradient in exactly the mid-climb region.” The visible author input contains no commands/speeds/reward-rate traces identifying this mechanism or “exactly” that region. The full event evidence does not contain such a gradient measurement either. This is an unsupported factual/causal assertion, not merely an uncertain future intervention claim.

Analytically, with lateral error=0, r=exp(-(c-v)^2/0.25) and d(3.5r)/dv = 28(c-v)exp(-(c-v)^2/0.25). At v=0, c=0.2/0.5/1.0/1.5 gives weighted derivatives 4.772005/5.150312/0.512838/0.005183. An unreachable commanded speed alone does not imply a near-zero gradient; magnitude depends on tracking error. These are reward derivatives with respect to a state feature, **not PPO policy gradients**. They establish only that the author's deterministic inference is too broad, not what the actual learning signal was.

### Tentative independent assessment

- Action relevance: **1/2**. A bounded forward drive is plausibly related to incomplete climbing, but aggregate failures/progress do not provide explicit process evidence for lost tracking incentive. If AR were meant to score only causal correctness, 0 is arguable, but original AR=0 says “no actionable hypothesis,” which this candidate does not meet.
- Testability: **1/2** under a strict reading. Observable direction and numeric targets exist, and an implicit next evaluation is obvious, but unlike 003 the rationale does not name the public_dev window/run explicitly; score/success wording and guard sufficiency need repair. Under a generous “next proposed evaluation” reading, **2/2 is defensible**. Lack of a mediator is a relevance/diagnosis weakness, not an original-rubric requirement for outcome testability.
- Authorization: **yes**. Allowed fields and one pure reward mechanism; flawed explanations do not turn it into forbidden control/input.
- Restraint: **poor/partial**. `:95` properly says no claim about unrun training benefit, but the present-tense gradient-collapse explanation and physical anti-hopping guarantee are overstated.
- Hard errors: at least one clear unsupported factual mechanism assertion (`:85-87`), with additional independently demonstrable code/guard overclaims (`:88-95`). Count at the claim level if needed, but do not multiply one mechanism story into several nominal independent failures. Score/success ambiguity alone is not a proven factual hard error given its future-target phrasing.

## Recomputed full-evidence arithmetic

| Quantity | Recomputed |
|---|---:|
| Episodes | 864 |
| Success | 689/864 = 0.7974537037037037 |
| Forbidden contact | 113/864 = 0.13078703703703703 |
| Base orientation primary | 22/864 = 0.02546296296296296 |
| Off course | 4/864 = 0.004629629629629629 |
| Timeout | 36/864 = 0.041666666666666664 |
| Frontier success / forbidden | 55/108 / 35/108 |
| Height success / forbidden | 63/108 / 36/108 |
| Joint success / forbidden | 69/108 / 19/108 |

The event counts exactly agree with the seen-042 aggregates. Event and prompt domain hashes are not identical representations, so this agreement does not prove identity of every upstream source/projection. The exact r1.e48 ID and ranges ground the cited episode, and source identities remain explicitly recorded.

## Files and hashes at independent-save time

- proposal-003.py SHA256 `fd13e8ac7263bd5389b4c59c4065fbe1d0334295ec0762390911a66543626241`
- proposal-042.py SHA256 `db009b1d85d93604ab33294ec33923e5502369d79b6e8c1b72c304aebf146278`
- seen-003.json SHA256 `39cd83c84af2b347ed9a03d0b440d471be5b41c4d58d882daffa65f0311b80fc`
- seen-042.json SHA256 `3289ca91608a855ea9d7e5dff1a2b4677263efa7822ee83019080298bda98d58`
- evidence-stair.json SHA256 `50e0c0e19415d5fab68bcbf0e28272a5a3782d7dade51781cc4c4f4ca813eb89`
- reference-stair.py SHA256 `10b1a34218d7fbb3ed124a6913e6d45d1462e5508d062510f31a3d1d7ac272cb`
- rubric.json SHA256 `86a87b750a9b40bc76b444e91d34507cc07a51c663e8521b2f2edc30d7eb3358`

Prior memory was used only for the established discipline of separating observed evidence, candidate hypotheses, acceptance and causal attribution (MEMORY.md:72-81, rollout 01a0c383-a7db-7461-b439-b67605a3a84a). All proposal-specific facts above were checked against current files.
