"""Run the existing shared feedback boundary regressions on CPU, without API I/O."""
import json
import os
import sys
import time
import unittest
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
MODULES = [
    'tests.unit.test_prompt_capacity_projection',
    'tests.unit.test_prompt_projection_and_mechanism',
    'tests.unit.test_prompt_safety_boundaries',
    'tests.unit.test_prompt_source_catalog',
    'tests.unit.test_source_request_service',
    'tests.unit.test_source_request_feedback',
    'tests.unit.test_openai_compatible_provider',
    'tests.unit.test_native_video',
]


def main():
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    with (OUT / 'shared_regressions.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {'modules': MODULES, 'python': sys.executable, 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'skipped': result.skipped,
              'elapsed_s': time.monotonic() - started, 'actual_provider_calls': 0,
              'status': 'PASS' if result.wasSuccessful() and not result.skipped else 'FAIL'}
    (OUT / 'shared_regressions.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
