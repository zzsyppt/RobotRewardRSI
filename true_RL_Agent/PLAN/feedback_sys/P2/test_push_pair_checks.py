"""CPU checks for the independent Push pair reader (no simulator required)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("verify_push_pair", Path(__file__).with_name("verify_push_pair.py"))
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)


class PushPairChecks(unittest.TestCase):
    def test_discrete_exact_float_absolute_and_structure(self):
        baseline = {"episodes": 64, "success": True, "pose": [0.2, -0.1], "missing": None}
        within = {**baseline, "pose": [0.2000005, -0.1]}
        self.assertTrue(PAIR.compare_values(baseline, within)["equal"])
        for altered in ({**baseline, "episodes": 63}, {**baseline, "success": 1},
                        {**baseline, "pose": [0.200002, -0.1]}, {**baseline, "extra": 0},
                        {**baseline, "missing": 0}):
            with self.subTest(altered=altered):
                self.assertFalse(PAIR.compare_values(baseline, altered)["equal"])
        self.assertFalse(PAIR.compare_values(float("nan"), float("nan"))["equal"])

    def test_resource_receipt_observed_missing_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case.time"
            self.assertEqual(PAIR.resource_time(path)["status"], "MISSING")
            path.write_text("peak_rss_kib=123456 elapsed_s=12.34 exit=0\n")
            actual = PAIR.resource_time(path)
            self.assertEqual(actual["peak_rss_bytes"], 123456 * 1024)
            self.assertEqual(actual["elapsed_s"], 12.34)
            for text in ("peak_rss_kib=0 elapsed_s=1 exit=0", "peak_rss_kib=123 elapsed_s=12 exit=1", "not a time receipt"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    PAIR.resource_time(path)

    def test_failure_report_is_not_a_pass_when_pair_does_not_exist(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "verification.json"
            project = Path(__file__).resolve().parents[3]
            result = PAIR.main(["--project-root", str(project), "--on", str(Path(folder) / "missing_on"),
                                "--off", str(Path(folder) / "missing_off"), "--output", str(output)])
            self.assertEqual(result, 1)
            self.assertEqual(json.loads(output.read_text())["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
