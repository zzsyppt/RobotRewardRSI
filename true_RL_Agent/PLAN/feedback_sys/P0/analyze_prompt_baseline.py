"""Reconcile text/vision probes and historical usage; does not send requests."""
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def fit_text_estimator(points):
    """Fit only comparable text request byte/token pairs, including fixed wrapper cost."""
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sum((x - mean_x) ** 2 for x, _ in points)
    return slope, mean_y - slope * mean_x


def text_estimate_analysis(calibration, frames):
    training = [(r['prompt_bytes'], r['usage']['prompt_tokens']) for r in calibration['frames'].values()]
    validation = [(r['payload_bytes'], r['usage']['prompt_tokens']) for r in frames]
    slope, intercept = fit_text_estimator(training)
    rows = []
    for frame, (size, actual) in zip(frames, validation):
        predicted = max(0, math.ceil(slope * size + intercept))
        rows.append({'name': frame['name'], 'actual_prompt_tokens': actual, 'predicted_prompt_tokens': predicted,
                     'relative_error': predicted / actual - 1})
    pooled_slope, pooled_intercept = fit_text_estimator(training + validation)
    calibration_path = ROOT / 'PLAN/2new_world/P5/audit_capacity/result.json'
    return {'method': 'affine least squares fitted on the four historical calibration requests only',
            'calibration_source': {'path': calibration_path.relative_to(ROOT).as_posix(),
                'sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest()},
            'calibration_validation_hashes_disjoint': not ({r['source_sha256'] for r in calibration['frames'].values()}
                & {r['payload_sha256'] for r in frames}),
            'tokens_per_user_utf8_byte': slope, 'intercept_tokens': intercept,
            'calibration_byte_range': [min(x for x, _ in training), max(x for x, _ in training)],
            'calibration_samples': len(training), 'validation_samples': len(validation), 'validation': rows,
            'max_validation_absolute_relative_error': max(abs(r['relative_error']) for r in rows),
            'pooled_nine_fit_training_error_only': {
                'max_absolute_relative_error': max(abs((pooled_slope * x + pooled_intercept) / y - 1)
                                                 for x, y in training + validation)},
            'usage': 'workload-local planning estimate; actual usage is the cost source; not a qualification gate',
            'limits': 'retrospective model choice; shared source families; fixed system wrapper; text only; do not extrapolate the negative intercept to short prompts or images'}


def main():
    source = json.loads((OUT / 'prompt_baseline.json').read_text(encoding='utf-8'))
    calibration = json.loads((ROOT / 'PLAN/2new_world/P5/audit_capacity/result.json').read_text())
    receipts = {p.stem: json.loads(p.read_text()) for p in (OUT / 'provider_receipts').glob('*.json')}
    rows = []
    totals = {'calls': len(receipts), 'prompt_tokens': 0, 'completion_tokens': 0,
              'cache_hit_tokens': 0, 'cache_miss_tokens': 0, 'peak_price_estimate_usd': 0, 'elapsed_s': 0}
    for name, receipt in receipts.items():
        path = OUT / 'provider_requests' / (name + '.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['request_sha256']
        assert receipt['status'] == 'received'
        usage = receipt['usage']
        assert usage['prompt_tokens'] == usage['prompt_cache_hit_tokens'] + usage['prompt_cache_miss_tokens']
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
        totals['prompt_tokens'] += usage['prompt_tokens']
        totals['completion_tokens'] += usage['completion_tokens']
        totals['cache_hit_tokens'] += usage['prompt_cache_hit_tokens']
        totals['cache_miss_tokens'] += usage['prompt_cache_miss_tokens']
        totals['elapsed_s'] += receipt['elapsed_s']
        totals['peak_price_estimate_usd'] += (usage['prompt_cache_hit_tokens'] * .006 +
                                             usage['prompt_cache_miss_tokens'] * .30 + usage['completion_tokens'] * 1.20) / 1e6
    for frame in source['frames']:
        receipt = receipts[frame['name']]
        tokens = receipt['usage']['prompt_tokens']
        rows.append({**frame, 'usage': receipt['usage'], 'response_model': receipt['response_model'],
                     'system_fingerprint': receipt['system_fingerprint'], 'payload_bytes_per_prompt_token': frame['payload_bytes'] / tokens,
                     'input_plus_candidate_output_reserved': tokens + frame['reserved_candidate_output_tokens']})
    before = receipts['long_history']['usage']
    repeat = receipts['long_history_repeat']['usage']
    media = receipts['media_native']['usage']
    control = receipts['media_text_control']['usage']
    report = {'schema_version': 'feedback_p0_prompt_measurement.v2', 'frames': rows, 'totals': totals,
              'text_estimator': text_estimate_analysis(calibration, rows),
              'cache_repeat': {'same_input_token_count': before['prompt_tokens'] == repeat['prompt_tokens'],
                'first_hit_tokens': before['prompt_cache_hit_tokens'], 'repeat_hit_tokens': repeat['prompt_cache_hit_tokens'],
                'repeat_miss_tokens': repeat['prompt_cache_miss_tokens'], 'claim': 'one measured best-effort prefix-cache hit, not a guarantee'},
              'media_probe': {'input_token_delta': media['prompt_tokens'] - control['prompt_tokens'],
                'images_sent': receipts['media_native']['images_sent'], 'native_response': receipts['media_native']['response_text'],
                'no_image_response': receipts['media_text_control']['response_text'],
                'transport_status': 'PASS', 'vision_diagnosis_profile': 'NOT_STARTED'},
              'pricing': {'source': 'https://api-docs.deepseek.com/quick_start/pricing/', 'checked_date': '2026-09-23',
                'usd_per_million_peak': {'cache_hit': .006, 'cache_miss': .30, 'output': 1.20},
                'billing_statement': False, 'explanation': 'conservative peak-rate usage estimate, no account billing endpoint read'}}
    (OUT / 'prompt_measurements.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'totals': totals, 'media_input_token_delta': report['media_probe']['input_token_delta'],
                      'text_estimator_max_validation_error': report['text_estimator']['max_validation_absolute_relative_error']}))


if __name__ == '__main__':
    main()
