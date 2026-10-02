# Independent re-audit: proposals 008 and 014

Saved on 2026-09-24 before opening `review_sheet.json`, the human reviewer guide, or the prior report. This audit read the two proposal files, their exact `seen-*.json` inputs, `reference-push.py`, `evidence-push.json`, and `rubric.json`. This subagent received no prior findings about these proposals. Independence is bounded: proposal identities were assigned by the parent, the parent notes that prior conversation discussed them, and the same artifacts/rubric are reused. This is not a second human rater or a fresh experiment.

All relative evidence paths in this document are under `results/author_final_bundle/author_blind_review/reviewer/`. CodeGraph was queried first for the three Python artifacts; it returned unrelated indexed production symbols, not these generated artifacts. The artifact sources were therefore read directly. No simulator, training, paid provider, or candidate execution was run. Only audit outputs are written.

## Rules applied

The original `rubric.json:2-8` separates fact support, hard errors, action relevance (AR 0-2), testability (0-2), authorization, and restraint. It explicitly excludes grading whether unrun interventions improve RL. Observation, code algebra, mechanistic explanation, and future performance are assessed separately. A behavioral association does not prove or disprove a reward intervention's causal effect.

## Evidence and arithmetic, independently checked

Both inputs contain identical `public_feedback_summary` and `reward_inputs_contract` objects. Their requested focus differs: 008 requests the **medium** stratum (`seen-008.json:596`); 014 requests the **low** stratum (`seen-014.json:596`). Both provide the same reference source inline at line 734 and mark it `reference_current_best` at line 699.

| Quantity | Visible source | Recomputed from full evidence |
|---|---|---|
| Episodes / successful episodes | both seen files:750-757 | 192 / 96 |
| Terminal not_recovered | both seen files:788-797 | 51 |
| Recovery units / executed event windows | both seen files:930-945 | 240 planned units; 227 observed anchors; 165 recovered windows |
| Failed-condition steps: planar/angular/roll/pitch/height | both seen files:958-964 | 9466 / 3232 / 722 / 401 / 114 |
| Dwell interruptions total | both seen files:950 | 193 |
| Interruptions containing planar/angular/roll/pitch/height | both seen files:951-957 | 168 / 61 / 0 / 0 / 0 |
| Exact interruption condition sets | visible totals plus overlap warning; full transcript check | planar only 132; angular only 25; both 36 |

The overlap is calculable even from the visible aggregate: 168 + 61 - 193 = 36 both, hence 168 - 36 = 132 planar-only and 61 - 36 = 25 angular-only. Full `evidence-push.json.transcripts` confirms these exact counts over 192 episodes, 240 recovery objects and 193 interruption objects. Summing `recoveries[].failed_condition_steps` gives the five values above. Source counts are 64 each for source 0/1/2. Terminal labels sum to 96 success + 51 not_recovered + 24 forward falls + 13 side falls + 5 height collapses + 3 out_of_bounds = 192.

The input explicitly warns that condition counts overlap and are descriptive, not causal (`seen-008.json:873-895`, identical in 014; full evidence:84-85). Dwell interruption counting stops at the first verified latch; failed-condition steps cover the entire observed window. Zero posture-related **interruption** counts do not mean zero posture **violation** steps.

Useful strata, all already visible in the compact profile:

| Stratum | Success episodes | Planar failed steps / observed steps | Dwell interruptions | Planar-containing / angular-containing |
|---|---:|---:|---:|---:|
| Low | 33/48 | 1676/5276 | 66 | 63 / 10 |
| Medium | 30/48 | 1899/4852 | 21 | 18 / 3 |
| Repeated | 13/48 | 3594/7722 | 69 | 55 / 36 |

Medium not_recovered and success episodes have planar failed-step fractions 464/1210 = 38.35% and 1093/3300 = 33.12%. Repeated counterparts are 1419/3080 = 46.07% and 1010/2860 = 35.31%. These descriptive differences do not identify the effect of stronger damping. All 51 not_recovered episodes include 45 recovered windows, and 31/51 have all required recoveries latched before final instability; initial dwell failure is not the only route to terminal failure. Success cases also contain violations (3936 planar failed steps overall). Neither fact refutes a proposed reward change.

## Proposal 008

**Actual change and grounded statements.** `proposal-008.py:17-26` replaces `stationary_v3` with a higher-curvature velocity penalty. All four other callables and five numerical weights are preserved (`reference-push.py:11-47` versus proposal lines 11-49). For normalized speed r >= 0, the old per-axis penalty is 0.5*r^2 for r <= 1 and r - 0.5 for r >= 1. The new penalty is identically 0.5*r^2 everywhere, because 0.5 + (r-1) + 0.5*(r-1)^2 = 0.5*r^2 above the bound. Thus the unchanged in-bound term, unchanged normalizers, derivative 1 at the bound (before the outer 0.25 factor), and superlinear excess claim at lines 54-56 are correct code statements. New minus old reward is -0.125*((planar-1)_+^2 + (angular-1)_+^2).

The scaled bounds 0.08 and 0.5 agree with the visible physical bounds and scales: 0.08*sqrt(9.81*0.74) = 0.2155461899 m/s; 0.5/sqrt(0.74/9.81) = 1.8204914691 rad/s (`seen-008.json:24-29,880-885`). The 9466/3232 failed steps and zero posture-related interruption counts are supported.

**Hard error that stands independently.** Lines 58-59 say **“168 of 193 dwell interruptions caused by planar speed alone”**. The correct count containing planar speed is 168; the count whose recorded failed-condition set is planar-only is **132**. This is a wrong exclusivity claim, contradicted by both visible aggregate arithmetic and full transcripts. “Caused” is also stronger than the documented descriptive condition label; the mechanically failing predicate should not be upgraded to a causal claim about learned policy or reward adequacy.

**Claims requiring restraint.** “The recovery gate is decided by velocity” (line 57) is overbroad if read as the whole success/failure mechanism: falls, height collapse, final stability and repeated-event completion also matter. A narrower statement that recorded dwell interruptions here are velocity-labelled is supported. “Severity is not reflected in the learning signal” (60-61) needs the immediately preceding constant-marginal-slope qualification: old penalty magnitude does increase with severity, while its out-of-bound derivative stays constant. The curvature change is justified as a hypothesis, not as a demonstrated remedy.

**Testability.** Lines 64-67 name the next same public-dev plan, medium/repeated strata, lower planar failed-condition steps and dwell interruptions, higher recovered units and success episodes, score > 0.5, and a count-based failure condition. This merits testability **2** under the original rubric. No evaluation establishes these directions. “Hypothesis refuted” should mean failure of this prespecified next-run prediction; one noisy run cannot refute every causal version of the mechanism.

**Tentative judgments:** AR **2** (explicit process evidence supports a bounded velocity-curvature experiment; medium focus is only partially developed, since the rationale cites pooled numbers); testability **2**; authorization **yes** by source/contract inspection; restraint **partial / not a full pass**, because of exclusivity and overbroad gate/learning-signal wording. At least one definite hard factual error. The observed error does not establish that the intervention is useless.

## Proposal 014

**Actual change and authorized scope.** Lines 17-26 reconstruct world XY velocity using sqrt(g*L), subtract `push_active * push_vec_public`, normalize to the same planar bound, and replace the reference ramp with quadratic excess penalties. `clip_min=-8.0` at line 46 is an allowed RewardTerm output clip. The other four callables/weights are preserved. Both push inputs are expressly allowed and current-step-only (`seen-014.json:438-458,527-529,594`), and world-frame scaled velocity has matching units/frame (`494-514`). This is one reward callable replacement. The use of these signals is **not** forbidden observation access or a push-schedule change.

**Supported measured/code claims.** The 96/192 episode successes, 51 not_recovered, 9466 planar/3232 angular/114 height failed steps, 168 planar-containing/61 angular-containing interruptions, and repeated success baseline 13/48 are visible and correct. The current-step subtraction really is present in the code. Nominal coefficients 3.0 on upright_height and 0.25 before the old speed-penalty sum are visible.

**Definite hard numerical error.** Line 51 says **“failed-condition steps ... roll/pitch 0”**. The visible vector is ordered planar, angular, roll, pitch, height and gives **roll 722, pitch 401** (`seen-014.json:919-924,958-964`). The zeros belong to **interruption_condition_counts**, a different measurement. Full transcript summation independently confirms 722/401.

**Unsupported mechanism stated as verified.** Lines 3-4 and line 51 claim subtraction means **“only self-generated residual motion/velocity”** is penalized. The contract guarantees only the current step's injected increment. On subsequent steps `push_active=0` and `push_vec_public=0`, so the reward uses the entire actual velocity, which can still contain consequences of an earlier push. Even on the injection step, subtraction is not a causal decomposition of contact/body dynamics. A grounded claim is that the callable numerically removes the recorded current-step increment, not that it identifies self-generated motion. The numerical subtraction is real, but its physical/learning interpretation is unverified and the global “only” claim exceeds the input semantics.

**Weight interpretation and intervention remain hypotheses.** Larger nominal coefficients do not establish larger runtime contributions, deficient policy pressure, or the gradient delivered by PPO. The input has `runtime_feedback_summary.status = no_prior_runtime_feedback` (`seen-014.json:1707-1711`). Low posture violation counts could be consistent with the existing posture reward doing its job; no ablation decides. Velocity-focused reweighting is plausible, but observational frequencies alone establish neither its necessity nor its ineffectiveness.

**Code details omitted from the rationale.** The callable also removes the old +1 constant, doubles each effective inside-bound quadratic coefficient from 0.125 to 0.25, assigns different planar/angular outer curvature, and adds clipping at -8. Clipping means arbitrary high severity does not yield indefinitely increasing negative signal. The composite function is continuous at r=1 but has a derivative kink: left derivative 0.5, right derivative 0 for each new penalty; this does not contradict the literal narrow claim that the outside quadratic term itself is smooth, so I would not automatically count the word “smooth” as a separate hard error. No result says these choices harm or improve training.

**Task relevance and testability.** The assigned decision is the low-magnitude stratum (`seen-014.json:596`), but the rationale cites pooled counts and predicts repeated-stratum success >13/48. It never evaluates low success 33/48 or low process evidence, or explains insufficient low evidence. A testable overall push experiment is supplied, but the assigned low-stratum analysis is weak. Testability **2** still follows the literal rubric: next public-dev run, named repeated-success and planar-interruption directions, and an activation-conditioned unchanged-metric failure condition. The intervention bundles subtraction, curvature, effective scaling and clipping within the one authorized callable, so any causal interpretation would require a later controlled comparison.

**Tentative judgments:** AR **1 for the assigned low-stratum decision** (a broadly relevant velocity experiment could merit AR 2 if scope were ignored); testability **2**; authorization **yes** by source/contract inspection; restraint **not a pass**. At least one definite numerical hard error plus unsupported verified residual-origin language. Benefits, harm, and the precise policy learning effect remain unknown.

## Boundaries

These findings do not grade RL performance, claim either new proposal was evaluated, infer causal failure from successful episodes having violations, or infer falsehood simply because a claim is not measured. No video is needed for the arithmetic and input-contract conclusions. Live production code was not used to overrule the frozen author input, and no montage was interpreted without episode identity.
