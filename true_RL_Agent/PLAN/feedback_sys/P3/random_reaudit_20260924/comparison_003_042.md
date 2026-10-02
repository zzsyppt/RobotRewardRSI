# Comparison with original review: 003 and 042

Comparison written only after `independent_003_042.md` was saved (2026-09-24 23:54:58 +08:00). Read only the original review_sheet.json entries whose blind_id is proposal-003 or proposal-042. Did not read the human-review guide or report. The original rows reference those materials; those references were treated as claims, not followed.

## Verdict

The original headline judgments are mostly reasonable for these two sampled proposals. 003 has no original hard error, AR=1, testability=2 and authorization pass; the independent audit agrees. 042's unsupported causal explanation and false anti-hopping code guarantee are real errors. The original third hard error, the “success above 0.6255” numeric prediction, is questionable because a future success target can legitimately exceed the present success rate. The original review also contains some overly certain mechanism/counterfactual statements that should be softened.

| Item | Original | Independent comparison |
|---|---|---|
| 003 hard errors | none | Agree; hedged toe-catch hypothesis is weak evidence, not an invented measured fact |
| 003 AR/testability | 1/2 | Agree with 1/2 |
| 003 authorization | pass | Agree from actual code/input contract |
| 003 restraint | 1 | Qualitatively partial is fair |
| 003 fact support | 9/9 | Conditional on treating inherited failed-source rationale as valid secondary provenance; no independent final-frame measurement |
| 042 causal overreach | hard error | Stands: author asserts unmeasured command reachability/gradient loss and exact stall region |
| 042 anti-hopping guarantee | hard error | Stands: body-frame vx rectification does not guarantee zero for physical hopping/retreat |
| 042 success >0.6255 | numeric hard error | Questionable; original conflates a possible future threshold with a claim about current success |
| 042 guard implication | contradicted claim; no hard error after H3 | Contradiction stands; original core rubric does not itself supply H3's exclusion, so report separately rather than silently erase |
| 042 AR/testability | 1/2 | AR1 agrees; T2 is defensible if next evaluation is implicit; strict original window requirement gives T1 |
| 042 authorization | pass | Agree |
| 042 restraint | 1 | Qualitatively poor/partial is fair |

## 003: what to retain, what to correct

Retain the accurate code review: vertical z is selected on the spatial axis, member ordering is not used, all eight reference functions/weights remain, one bounded mechanism is added, and 1.6 is the correct maximum weighted reward rate. The reference-origin subtraction correctly yields root-to-support vertical separation; this is not an illegal use of base_height. Preserve the observation that it does not measure toe-to-riser clearance and could be increased through other body/foot configurations.

Retain the real r1.e48 evidence: timeout at 588, max progress 0.837430 m, minimum pitch -0.525434 rad, low_progress before terminal. The final repair author sees this via its previous candidate's rationale, not primary evaluator feedback. The original marks this provenance and therefore does not invent final-frame visibility. Its 9/9 should be described as “claims traceable to final input, including secondary source assertions,” not 9/9 independently evidenced measured observations. Reconstructing the earlier author turn was outside this audit.

Soften original `claims[9].truth='contradicted'` and its sign-error wording. The author says “consistent with” and does not explicitly say positive pitch or nose-down. Negative pitch indicates nose-up, but a vague “trunk pitches onto the step” is not by itself a wrong-sign numerical statement. A timeout with no forbidden-manager contact prevents calling trunk contact an observed fact for that episode; it does not identify or exclude a toe touching a riser face. Split the claim: toe catch/full extension **unavailable**, observed terminal trunk-contact reading **unsupported/inconsistent with this timeout record**, resulting treatment **untested hypothesis**. The original appropriately gave no hard error; its claim-level contradiction label is stronger than needed.

Soften original expected-change notes that “trunk-lowering route aggravates” the observed failure and that the reward is “likely already near-saturated.” Original uncertain_items concedes no per-step separation was measured. These are intervention risks or geometric possibilities, not demonstrated consequences. Four airborne feet with root below them will earn the maximum; a synthetic inverted fixture validates one such code behavior, not that PPO will prefer inversion or that the complete reward makes it beneficial.

The zero frontier low-clearance event count is useful negative diagnostic context, but cannot disprove toe/riser collision because the retained diagnostic uses support reference points and local surface height rather than riser-face contact. The original itself says riser faces are unmeasured, which should control its conclusion.

The original focus-addressed=no is consistent with the proposal never mentioning c2. However, the final 003 frame omits selection_guidance and explicitly requires preserving the existing mechanism. Focus miss must not be recast as refusal of a visible instruction by this repair response. Whether an earlier author round ignored the focus is not established by this final-frame audit.

## 042: hard errors that stand

1. **Unsupported causal mechanism asserted as fact** (`proposal-042.py:85-87`, original hard_errors causal_overreach). Visible aggregate score/progress/failure data does not establish unreachable commands, vanished learning signal, or “exactly” a mid-climb region. The full evidence shows an early frontier-L4 failure population. Arithmetic from visible means is reproducible: failed-L4 mean fraction = (0.5842113844444444*36-17)/19 = **0.21218999157894727**. Height-L5 failed mean = (0.5894963886111111*36-9)/27 = **0.4526618514814814**. These are failed-episode mean fractions, not every episode's stopping location. They challenge “exactly mid-climb” without proving a cause.
2. **Code guarantee about no in-place hopping** (`proposal-042.py:88-90`, original code_misdescription). Current runtime exposes base_lin_vel as body-frame velocity (isaaclab_reward_runtime.py:208,282). At pitch=-0.5 rad, world v=(0,0,0.5), goal progress=0.5, added rate is **0.2397127693** although world stair-x velocity is zero. At world v=(-0.1,0,0.5), added rate is **0.1519545131** despite stair-x retreat. This deterministic counterexample validates the code-description error. It says nothing about occurrence in trained trajectories.

Correct an original review wording bug: “only exact standstill and net backward motion are unpaid” is too broad. Exact zero body velocity is unpaid; instantaneous body vx<=0 is unpaid. Net backward displacement, or instantaneous world-x retreat accompanied by suitable vertical motion/orientation, does not guarantee an unpaid trajectory. The original correctly identifies body-frame rectification, but its exception sentence should use the correct frame and instantaneous predicate.

## 042: questionable or incomplete original judgments

**The numeric hard error is not compelled by the literal quote.** Current frontier success is 55/108=0.5092592593; score is (0.5092592593+0.7417760684)/2=0.6255176638. The author correctly calls 0.6255 a score in its observed-data sentence, then predicts “success above 0.6255.” That is suspicious metric borrowing but remains a falsifiable future success threshold. A proposal could intentionally target success>0.6255, equivalent to at least 68/108 successes if counted over this same equally sized cohort. It does not say “current success is 0.6255” or “increase from 0.6255.” Recommend marking this **metric ambiguity / target needs clarification**, not a definite wrong-denominator hard error. This is the main sample-level overjudgment.

**The guard implication is a real factual error.** c0/c1>=0.95 alone does not imply the public_dev selector's mean of seed-wise minimum named guard stays within 0.05 of its parent. Other guards are j_ood/joint_ood/phase_control and seed-level aggregation matters. The original lead adjustment removed a hard error under H3 while retaining the contradicted claim. That exclusion is absent from the original rubric.json. Without importing a later scoring convention, report the rule error explicitly. Do not claim joint_ood is definitively the binding guard on every seed: the original uncertain_items admits that per-seed minima were not checked, and means alone cannot prove it. The invalid implication can be demonstrated without naming the exact binding guard.

**Do not turn a small raw tracking reward into a vanished gradient.** Original truth_reference calls “collapse mathematically real” from a 0.8 m/s eval command. Even granting that unverified-in-this-subtask config number, at v=0, lateral error=0: raw reward exp(-0.64/0.25)=**0.0773047404**, weighted reward=**0.2705665916**, and derivative of the weighted reward with respect to forward speed is **1.7316261859**. The raw reward being small does not establish zero/negligible policy-learning signal. These are state-feature reward derivatives, not PPO gradients. The original should support only the exponential's decreasing magnitude at large error, while keeping actual lost incentive/learning signal unknown.

**Progress gating is weaker early, not necessarily ineffective early.** At goal fraction q=0.2, the added rate is 0.4*clamp(v_body_x,0,1), and its slope on 0<v_body_x<1 is 0.4. Thus original mechanism_matches_code “so it cannot keep climb pressure there” is false as a deterministic statement. It is positive at positive speed and smaller early. Whether this is enough, beneficial or harmful requires training. Similarly “does not act on the true failure forms” and “largely duplicates tracking near the top” are unvalidated mechanism judgments. Rewrite them as plausible mismatch/side-effect risks, without predicting PPO's result.

**Testability=2 is reasonable but interpretive.** There are concrete targets and directions. The absence of a mediator or effect-size-versus-seed-noise is not by itself grounds to lower the score, since the original rubric does not require either. The literal rationale lacks an explicit evaluation window; if “next public_dev” is inferred from task context, T2 is defensible. Retain this as the same implicit-window judgment the original already discloses.

The original final lead adjustment treats “height_ood L5 stalls at 0.589” as a grounded mean with loose wording, rather than fabricated observation. That is reasonable for its accurate numbers, provided the diagnosis does not use 0.589 as a measured stall location. Failed-only mean is 0.4526618515 and episode extrema vary; neither mean is a terminal location.

## Bounded result and unresolved questions

For the original hard-error list in these two rows: **2 of the 3 listed 042 errors clearly stand; the third is ambiguous.** 003 remains free of a conclusive hard error. Separately, 042's guard-rule claim is certainly wrong, despite the original policy choice not to count it as hard. This audit does not recompute suite scores or alter the original sheet.

Unknown: actual improvement/harm under PPO, actual exploitation of body-frame rectification, toe/riser contact geometry, learned effect of lift reward, per-seed binding guards, and whether earlier author context justified the focus selections. Full-evidence outcomes cannot establish intervention causality. The parent is independently checking existing video identity and imagery; no video conclusion is needed for the two deterministic code/numeric findings above.
