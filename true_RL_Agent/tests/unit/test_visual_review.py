"""Semantic delivery contracts; synthetic verdicts are not model acceptance."""

from copy import deepcopy
from dataclasses import replace
import unittest

from true_rl_agent.evaluation.visual_observer import audit_observer_output, visual_process_sentences
from true_rl_agent.evaluation.visual_review import (
    REVIEW_CRITERION, SCHEMA_VERSION, attach_visual_review, visual_review_binding, visual_review_verdicts,
)


def synthetic_review(audit, verdicts=None):
    """Explicit test fixture only: never used to assess real observer accuracy."""
    verdicts = verdicts or ["supported"] * len(audit["observations"])
    review = {"schema_version": SCHEMA_VERSION, "source_sha256": visual_review_binding(audit),
              "criterion": REVIEW_CRITERION, "reviewer": {"id": "synthetic-fixture", "kind": "dev"},
              "decisions": [{"observation_index": i, "verdict": verdict,
                             "reason": "Synthetic fixture decision covering text and uncertainty."}
                            for i, verdict in enumerate(verdicts)]}
    return attach_visual_review(audit, review)


class VisualReviewTests(unittest.TestCase):
    def audit(self, numeric_status="consistent", count=1):
        from tests.unit.test_visual_observer import output, window
        media = replace(window(), measurement_claims=({"id": "axis_below_zero", "body": "root",
            "definition": "Fixture predicate, not a prose verification."},))
        raw = output(numeric_claim={"id": "axis_below_zero", "asserted": False})
        raw["episodes"][0]["observations"] *= count
        return audit_observer_output(raw, [media], numeric_check=lambda *_: {"status": numeric_status})

    def project(self, audit):
        return visual_process_sentences(audit, episode_labels={"e1": "r1.e91"}, cohort_episode_n=12,
                                        observer_version="synthetic.v1", evidence_ref="observer")

    def test_numeric_consistency_cannot_replace_semantic_review(self):
        audit = self.audit()
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["observations"][0]["numeric_status"], "consistent")
        self.assertEqual(visual_review_verdicts(audit), ["not_evaluated"])
        rows = self.project(audit)
        self.assertEqual([row["kind"] for row in rows], ["S5"])
        self.assertIn("1 unreviewed; 0 shown", rows[0]["text"])
        self.assertIn("0 mechanically invalid", rows[0]["text"])

    def test_verdicts_keep_denominator_and_original_audit(self):
        audit = self.audit(count=3)
        original = deepcopy(audit)
        reviewed = synthetic_review(audit, ["contradicted", "insufficient", "supported"])
        self.assertEqual(audit, original)
        self.assertEqual({key: value for key, value in reviewed.items() if key != "semantic_review"}, original)
        rows = self.project(reviewed)
        self.assertIn("observer/observations/2", rows[0]["evidence_refs"])
        self.assertIn(original["observations"][2]["raw_observation"]["text"], rows[0]["text"])
        self.assertEqual(rows[0]["semantic_review"]["reviewer"]["id"], "synthetic-fixture")
        for text in ("3 statements", "1 semantically supported", "1 contradicted", "1 insufficient", "1 shown"):
            self.assertIn(text, rows[1]["text"])
        for verdict in ("contradicted", "insufficient"):
            self.assertEqual([row["kind"] for row in self.project(synthetic_review(self.audit(), [verdict]))], ["S5"])

    def test_changed_text_uncertainty_identity_and_images_expire_review(self):
        reviewed = synthetic_review(self.audit())
        for field in ("text", "uncertainty", "identity", "image", "raw_output"):
            altered = deepcopy(reviewed)
            record = altered["observations"][0]
            if field in ("text", "uncertainty"):
                record["raw_observation"][field] += " changed"
            elif field == "identity":
                record["media_identity"]["episode_id"] = "different-case"
            elif field == "image":
                record["image_evidence"][0]["image_sha256"] = "different-image"
            else:
                altered["raw_output"]["episodes"][0]["not_visible"].append("new caveat")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "binding changed"):
                self.project(altered)

    def test_task_rejection_does_not_rebind_or_admit_reviewed_observation(self):
        reviewed = synthetic_review(self.audit())
        source_hash = visual_review_binding(reviewed)
        reviewed["observations"][0]["violations"].append("source_binding")
        self.assertEqual(visual_review_binding(reviewed), source_hash)
        rows = self.project(reviewed)
        self.assertEqual([row["kind"] for row in rows], ["S5"])
        self.assertIn("1 mechanically invalid", rows[0]["text"])

    def test_incomplete_duplicate_unexplained_and_cross_case_reviews_are_rejected(self):
        audit = self.audit(count=2)
        valid = synthetic_review(audit)["semantic_review"]
        for mutation in ("missing", "duplicate", "reason", "binding", "criterion", "provenance"):
            review = deepcopy(valid)
            if mutation == "missing":
                review["decisions"].pop()
            elif mutation == "duplicate":
                review["decisions"][1]["observation_index"] = 0
            elif mutation == "reason":
                review["decisions"][0]["reason"] = " "
            elif mutation == "binding":
                review["source_sha256"] = visual_review_binding(self.audit())
            elif mutation == "criterion":
                review["criterion"] = "one_clause_only"
            else:
                review["reviewer"]["kind"] = "unknown"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                attach_visual_review(audit, review)
        with self.assertRaisesRegex(ValueError, "already attached"):
            attach_visual_review(synthetic_review(audit), valid)

    def test_support_does_not_erase_numeric_conflict_or_container_failure(self):
        reviewed = synthetic_review(self.audit(numeric_status="conflict"))
        rows = self.project(reviewed)
        self.assertIn("axis_below_zero=false: conflict (predicate only)", rows[0]["text"])
        self.assertIn("1 numeric conflicts", rows[1]["text"])
        reviewed["violations"].append("output_schema")
        rows = self.project(reviewed)
        self.assertEqual([row["kind"] for row in rows], ["S5"])
        self.assertIn("Visual output not delivered", rows[0]["text"])


if __name__ == "__main__":
    unittest.main()
