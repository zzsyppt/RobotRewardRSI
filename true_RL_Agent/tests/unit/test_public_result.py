from __future__ import annotations

import math
import unittest

from true_rl_agent.evaluation.public_result import PublicEvalResult


class PublicEvalResultTests(unittest.TestCase):
    def test_public_score_requires_finite_non_boolean_number(self) -> None:
        for value in (None, "0.5", True, False, math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite non-boolean"):
                    PublicEvalResult(
                        status="pass",
                        public_score=value,  # type: ignore[arg-type]
                        metrics={},
                    )

    def test_finite_integer_and_float_scores_are_valid(self) -> None:
        self.assertEqual(PublicEvalResult("pass", 0, {}).public_score, 0.0)
        self.assertEqual(PublicEvalResult("pass", 0.5, {}).public_score, 0.5)

    def test_status_and_metrics_use_the_same_strict_field_contract(self) -> None:
        for status in (None, True, "ok"):
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, "status must be pass or fail"):
                    PublicEvalResult(status=status, public_score=0.5, metrics={})
        for metrics in (None, [], "metrics"):
            with self.subTest(metrics=metrics):
                with self.assertRaisesRegex(ValueError, "metrics must be an object"):
                    PublicEvalResult(status="pass", public_score=0.5, metrics=metrics)


if __name__ == "__main__":
    unittest.main()
