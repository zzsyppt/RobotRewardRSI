"""One old completed probe comparison; only writes this control directory."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FIX = HERE / 'validation_fixtures'
OUT = HERE / 'validation'

def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'file': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def main():
    OUT.mkdir(exist_ok=False)
    original = FIX / 'old_count_prepared_text.py'
    adapted = HERE / 'count_prepared_text.py'
    old_text, new_text = original.read_text(), adapted.read_text()
    start = old_text.index('        parts, images = [], []')
    end = old_text.index('        require(len(images)', start)
    assert old_text[start:end] in new_text
    watched = {str(path): ref(path) for path in FIX.rglob('*') if path.is_file()}
    commands = [
        ('old_algorithm', [sys.executable, '-B', str(original), '--prepared-root', str(FIX/'old_push13'), '--output', str(OUT/'old_algorithm.json')]),
        ('r3_desktop', [sys.executable, '-B', str(adapted), '--prepared-root', str(FIX/'old_push13'), '--capture-card', str(FIX/'capture_card.json'),
                        '--desktop-input', str(FIX/'desktop_blind/input.json'), '--desktop-adapter', str(FIX/'prepare_desktop_review.py'),
                        '--allow-development-single', '--output', str(OUT/'r3_desktop.json')]),
    ]
    steps = []
    for name, argv in commands:
        process = subprocess.run(argv, cwd=HERE, capture_output=True)
        for kind, data in (('stdout', process.stdout), ('stderr', process.stderr)):
            (OUT/(name+'.'+kind+'.txt')).write_bytes(data)
        step = {'argv': argv, 'exit_code': process.returncode, 'stdout': ref(OUT/(name+'.stdout.txt')), 'stderr': ref(OUT/(name+'.stderr.txt'))}
        steps.append(step)
        assert process.returncode == 0, step
    old = json.loads((OUT/'old_algorithm.json').read_bytes())
    new = json.loads((OUT/'r3_desktop.json').read_bytes())
    a, b = old['records'][0], new['records'][0]
    assert a['text_blocks'] == b['text_blocks'] and a['images'] == b['images']
    assert a['serialized_text_tokens_deepseek_v4'] == b['serialized_text_tokens_deepseek_v4']
    assert a['text_utf8_bytes'] == b['text_utf8_bytes']
    p = new['reservation_rules']['model_capacity_parameters']
    assert p['gpt6.1-sol-a']['context_window'] == p['gpt6.1-sol-b']['context_window'] == 258400
    assert all(v['auto_compact_limit'] is None for v in p.values())
    assert p['deepseek']['context_window'] is None
    assert all(ref(path) == expected for path, expected in watched.items())
    desktop = b['desktop_transport']
    result = {'status': 'PASS_OLD_DEVELOPMENT_STATIC_ALGORITHM_EQUIVALENCE_NOT_RELEASED',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'script': ref(adapted), 'original_algorithm': ref(original),
        'validation_script': ref(__file__), 'source_fixture_registration': ref(FIX/'fixture_sources.json'),
        'executed_checks': steps, 'old_measurement': ref(OUT/'old_algorithm.json'), 'r3_measurement': ref(OUT/'r3_desktop.json'),
        'original_text_image_loop_verbatim_unchanged': True, 'all_text_block_records_identical': True,
        'original_message_text_proxy_tokens': b['serialized_text_tokens_deepseek_v4'],
        'complete_desktop_carrier_text_proxy_tokens': desktop['complete_serialized_carrier_text_tokens_deepseek_v4'],
        'carrier_additional_serialized_proxy_delta': desktop['carrier_tokens_minus_original_text_block_sum'],
        'same_messages_reconstructed': desktop['messages_reconstructed_exactly'],
        'carrier_and_original_not_added_together': True, 'fixture_inputs_unchanged': True,
        'model_capacity_parameters': p, 'formal_r3_raw_read': False, 'models': 0, 'api_calls': 0, 'gpu_runs': 0,
        'tokenizer_copies_created': 0, 'capacity_release': False,
        'limits': 'One old completed text-only Push13 prepared request and its final desktop carrier. Formal mixed-role request counts, image token cost, real SOL tokenizer/peak, visual consumption and capacity release are not demonstrated.'}
    (OUT/'validation_receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'original_message_text_proxy_tokens', 'complete_desktop_carrier_text_proxy_tokens', 'carrier_additional_serialized_proxy_delta')}))

if __name__ == '__main__':
    main()
