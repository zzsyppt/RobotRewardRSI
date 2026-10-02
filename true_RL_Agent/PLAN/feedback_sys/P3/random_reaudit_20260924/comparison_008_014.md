# Comparison with original review: proposals 008 and 014

Date: 2026-09-24. `independent_008_014.md` was saved before opening the original review. Only the `proposal-008` and `proposal-014` objects of `results/author_final_bundle/author_blind_review/reviewer/review_sheet.json` were then displayed. No reviewer guide or prior report was opened. JSON-path references below refer to those objects. This is a bounded artifact re-audit, not a fresh human rating, training result, or suite-wide regrade.

## Result

The strongest original findings survive: **008 incorrectly makes 168 planar-containing interruptions exclusive**, and **014 misreads roll/pitch failed-step counts as zero**. Both proposals are authorized and testable. The original review is strongest on source/contract checking and weakest where static reward differences or behavioral associations become claims about how the policy will learn. Several qualitative code-description hard errors for 014 should be split and narrowed rather than counted wholesale as objective factual contradictions.

| Dimension | 008 original | 008 independent | 014 original | 014 independent |
|---|---|---|---|---|
| Action relevance | 1 | Tentative 2; explicit process evidence and one bounded curvature experiment satisfy literal rubric, with weak medium-focus analysis | 1 | 1 for the assigned low-stratum decision; low focus is missed |
| Testability | 2 | 2 | 2 | 2 |
| Authorization | pass | pass by source/contract inspection | pass | pass by source/contract inspection |
| Restraint | 1 | partial; original ordinal retained as compatible, not newly calibrated | 1 | partial/insufficient; original ordinal compatible |
| Hard factual findings | one numeric entry | same numeric error stands | four entries | numeric and current-step/global-residual misdescription stand; other entries need decomposition/qualification |

The original `rubric.json` gives a 0-2 scale only for action relevance and testability. It gives a qualitative restraint criterion, so this re-audit does not invent new numerical restraint thresholds. No suite aggregates or fact-support denominators are changed.

## 008: agreements and corrections

1. **Original hard error stands exactly.** `hard_errors[0]` correctly identifies overlap versus exclusivity. Full transcript recomputation gives planar-only **132**, planar+angular **36**, angular-only **25**, total **193**. The exclusive split was derivable from the exact author-visible row; no extra hidden evidence was necessary. The arithmetic is 168+61-193=36 and 168-36=132.

2. **Source and authorization assessment stands.** `mechanism_matches_code.verdict=yes` and authorization pass agree with direct comparison. Only `stationary_v3` is replaced; the four other callables and every weight remain. Unbounded quadratic output is a reward-scale property, not unauthorized control. Static/synthetic smoke receipts in the original review are historical engineering checks only; this re-audit did not execute them or promote them to RL success.

3. **Action relevance 1 is defensible as a conservative judgment, but not forced by the evidence.** The author cites actual velocity process evidence, changes one relevant callable, and predicts directional process/outcome changes. This matches the literal AR-2 wording, “explicit process evidence supports a bounded change.” The proposed causal explanation is unproven, but every unrun intervention remains unproven under `not_scored`. The author's failure to analyze medium rows justifies `focus_addressed=partial`; it does not by itself prove the mechanism irrelevant.

   The review correctly calculates a tiny reward difference at a near-bound snapshot. But `reviewer_notes` saying the change “几乎不作用于主导失败” goes beyond this: rewards earlier in the trajectory can affect the learned policy and later drift. A small static terminal-state difference establishes a local algebraic property, not the causal effect of retraining. The potential large-speed risk is appropriately described elsewhere as unmeasured; keep it as a hypothesis.

4. **The success comparison needs denominators.** The original note that successes contain 3936 planar failed steps proves that mere presence of a violation is not sufficient for failure. It does not prove speed-related behavior is nondiscriminating. From the same visible profile: not_recovered has 3291/7150 = **46.03%** planar failed observed-window steps, while success has 3936/11990 = **32.83%**. Medium is 38.35% versus 33.12%; repeated is 46.07% versus 35.31%. These comparisons remain descriptive and cannot establish the proposed curvature intervention's effect in either direction.

5. **Two restraint reasons are overstated.** `score_reasons.restraint` says the proposal “never labels a hypothesis as such,” but `proposal-008.py:67` explicitly says “the hypothesis is refuted,” immediately after “Falsifiable expected measurement.” Its “severity is not reflected” wording and wrong “alone” count still warrant restraint criticism. The same reason says “speed magnitudes and timing are not visible,” although the actual seen input includes a not_recovered transcript's final planar speed **0.247787 m/s** and first post-recovery instability **1.64 s** later. Not every magnitude/time is present, but wholesale absence is incorrect. This matters because the original action-relevance reason itself uses those visible values.

6. **Reward units need care.** At normalized planar speed r=1.1 with angular speed zero, reference rate=0.85 and 008 rate=0.84875; rate difference=-0.00125. With control dt=0.02, the resulting per-control-step reward difference is -0.000025. At r=5.6, rates are -0.275 versus -2.920; at r=9.4, -1.225 versus -10.045. The review's approximate magnitude comparison is largely correct, but its repeated “per step” wording denotes pre-integration reward rates; the visible task contract says dt is applied once. No PPO or value-learning result follows from this calculation alone.

## 014: each original hard-error entry reconsidered

### Numeric roll/pitch error: retain

`hard_errors[0]` is firm. Author-visible failed steps are [9466,3232,722,401,114] in the declared planar/angular/roll/pitch/height order. The zeros are interruption counts. The independent transcript sum exactly reproduces **roll 722 / pitch 401**. This is independent of all interpretive disagreements below.

### “Almost never violated”: important criticism, not a denominator-free hard contradiction

`hard_errors[1]` adds a separate contradicted-fact entry for the phrase “orientation and height errors that are almost never violated.” The supporting factual check is correct: all **42** fall_forward/fall_side/height_collapse episodes finish outside roll/pitch or height stability bounds. This is 42/192=**21.875% of episodes**, and 42/96=**43.75% of failures**. A low violation frequency therefore cannot justify dismissing posture failure severity.

However, “almost never” does not specify its denominator or threshold. The review switches to an episode-level terminal denominator, while the author was discussing failed-condition **steps**. A post-comparison recomputation unioned roll/pitch/height `condition_violation_spans` within each recovery unit: **869 / 22169 = 3.9199%** of observed recovery-window unit-steps have at least one of those violations. There are **59** episodes with such a violation somewhere in observed windows. These figures can coexist with 42 bad fall endpoints and zero posture interruptions.

Thus the phrase is imprecise and used in an unsupported explanatory argument. It is not an additional clean numerical contradiction separate from the already-counted zero error without a specified denominator. Retain the false zero claim and the caution that rare-but-severe events matter; qualify or merge the separate “almost never” hard-error entry. The old policy's low posture violation rate might even reflect successful posture shaping; no ablation here tests that explanation.

### “Only self-generated residual velocity”: retain the narrow error, remove causal overreach

`hard_errors[2]` correctly identifies that `push_active` and `push_vec_public` apply only to the current executed event step. Later reward evaluations still use the entire current velocity. Neither the contract nor this subtraction provides a causal decomposition between self-generated motion and consequences of previous pushes. The global **“only self-generated”** claim therefore fails.

But the review then says **“the candidate therefore penalizes the injected momentum more, not less.”** Algebra supports a stronger penalty for sufficiently high *total observed speed on later steps*, compared at identical state inputs. It does not measure which share of that speed is attributable to injected momentum or predict learned-policy behavior. Also, on the actual event step the subtraction changes the input, so a blanket direction over the whole trajectory is not valid. Retain “does not isolate the origin of later motion,” not a proved physical decomposition or a proved ineffective intervention.

The review's **“-8/step ... vs about -0.8 in the reference”** comparison has a concrete arithmetic problem. With angular speed zero, no current injection, and the frozen bound 0.21554618994545 m/s, 014 first clips at:

`v = (1 + sqrt(7.75)) * 0.21554618994545 = 0.8156013874 m/s`.

At that same v, the **reference full speed term is +0.1790294546**, whereas 014 is -8. The old negative penalty portion is -0.8209705454, but the reference includes +1. Comparing the full new term to only the old penalty loses that +1. At dt=0.02 the actual integrated speed-term contributions are **+0.0035805891 versus -0.16**. Additional angular penalty can change the reference value and cause earlier clipping; no fixed -0.8 full-term comparator is justified without stating that angular input. This correction does not weaken the large-negative-rate concern, but fixes its basis.

### “Exact difference / smooth / weak”: split true omissions from debatable wording

`hard_errors[3]` bundles three different issues:

- **Omissions are real:** the rationale's “exact” description omits removal of the +1 constant, doubled inside-bound coefficient, asymmetric outer curvature, and a new -8 output clip. These are direct source facts. The resulting change in per-step rates is measurable algebra. Whether it changes a learned survival/termination preference is an untested hypothesis, especially in an episodic reward/discount setting.
- **Derivative kink is real, but the quoted smooth claim is narrower:** 014's whole piecewise penalty is continuous but has derivative 0.5 just inside r=1 and 0 just outside. The actual phrase is “uses a smooth quadratic term **outside each bound**.” That outside quadratic term is smooth on r>1. Treating it as a claim of C1 continuity at the junction is an interpretation, not a direct contradiction of the words.
- **Doubling does not disprove “weak”:** the effective inside coefficient doubles from 0.125 to 0.25, but the author did not say unchanged. “Keeping a weak inside-bound preference” has no quantitative threshold and could still describe the new preference relative to the outer penalty. Record the doubled coefficient and lack of complete disclosure; do not count the adjective alone as hard fact error.

The original review's later correction about net settling incentive is useful and stands: moving from r=1.1 to r=0.5 gains **0.1975** in the 014 speed reward versus **0.11875** in the reference (with the other axis zero and no clipping/current push). A locally weaker outside slope at 1<r<1.125 does not imply lower net incentive to settle. This already demonstrates why local derivative comparisons require caution.

## What remains justified or unknown

The original low-focus verdict for 014 stands: the requested low stratum is not addressed, and the prediction changes to repeated success >13/48 without explaining why. Testability 2 and authorization pass stand for both. A stronger test would report observed-window exposure/termination alongside raw violation counts, because earlier falls can reduce counts, and distinguish pre-latch interruption measurements from post-latch final instability. These are limitations of the named prediction, not evidence that it will fail.

No causal benefit, harm, knee-buckling mechanism, reduced learning stability, or early-termination preference has been measured for either proposed program. The review's expressly hedged risk discussions can remain as hypotheses. Visual evidence supplied elsewhere in the parent audit is not used here to settle those causal questions, and this subaudit opened no montage. No claim that a robot remains upright throughout an episode follows merely from its final pose.

The original review has no apparent authorization false-positive for these two proposals. Its firm numerical error findings are reproducible. Its interpretive hard-error count for 014 is less robust than the numeric findings, and its 008 AR-1 cap is a judgment call rather than an experimentally established irrelevance result. The audit should preserve these distinctions instead of replacing the original review with a claim that the proposals are known to work.
