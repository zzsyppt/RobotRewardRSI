"""R4 transport-only adapter for the two actual, unmodified desktop receipts.

The frozen observer/prompt/answer/mechanical checks are unchanged. Absolute
Windows paths and recorded ordinal names are normalized only in a private copy.
The actual desktop trace audit is required before this metadata adapter runs.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath

B = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929')
C = B / 'n5_execution_20261001_r4'
ROOT = B / 'n5_confirmation_20261001_r4/observers/push'
DRIVER = C / 'postcapture_preparation/sol_observer_control.py'
CARD_SHA = '811d7f140a5b3015f66a5093ba516c60bdc4a05c9ac827461d19bb05b8be3041'
ALIASES = {
    '4008fd2fefc52acf792e5a306b76a3c3387f102d1b35335dd6240afb801e6c35': ('e1', 'index'),
    'e5e0399d58d931cf7a4a9c2d8b1aded59eaa0dbd1d8ba2378e5dce0a68103c4e': ('e2', 'ordinal'),
}
LOCAL = PureWindowsPath(r'D:\humanoid_agent\true_RL_Agent\PLAN\feedback_sys\P3NEW\results\n5_confirmation_execution_r4\push_observer_inputs')


def main():
    assert hashlib.sha256(DRIVER.read_bytes()).hexdigest() == '408ee0839df19f550d08b94a26997459d7f4fec56487f678cd9c87c2d100b919'
    proof = json.loads((ROOT / 'desktop_access_audit.json').read_bytes())
    assert proof['status'] == 'PASS_OBSERVABLE_ACCESS_AND_TRANSPORT_NOT_SEMANTIC'
    assert [row['alias'] for row in proof['checks']] == ['e1', 'e2']
    for row in proof['checks']:
        assert row['image_returns_match_original_png_bytes_and_dimensions'] is True
        assert row['current_outputs_match_hashes_returned_at_original_write'] is True
        raw = ROOT / row['alias'] / 'sol_response/access_receipt.json'
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == row['access_receipt']['sha256']
    spec = importlib.util.spec_from_file_location('r4_frozen_observer_control', DRIVER)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    original_check = control.check_access_metadata

    def actual_access_check(access, materials, materials_sha256):
        alias, ordinal = ALIASES[materials_sha256]
        base = LOCAL / alias / 'blind_inputs'
        normalized = deepcopy(access)
        for row in normalized['files_read'] + normalized['images_viewed']:
            path = PureWindowsPath(row['path'])
            assert path.is_absolute() and path.is_relative_to(base)
            row['path'] = path.relative_to(base).as_posix()
        for row in normalized['images_viewed']:
            assert 'view_order' not in row
            row['view_order'] = row.pop(ordinal)
        result = original_check(normalized, materials, materials_sha256)
        result['original_access_limitations'] = deepcopy(access['limitations'])
        result['transport_representation_normalization'] = {
            'absolute_path_root': str(base), 'original_order_key': ordinal,
            'normalized_order_key': 'view_order', 'order_base': 1,
            'original_receipt_bytes_unchanged': True,
            'semantic_or_model_input_changes': False,
            'actual_desktop_trace_audit_sha256': hashlib.sha256((ROOT / 'desktop_access_audit.json').read_bytes()).hexdigest(),
        }
        return result

    control.CARD_SHA = CARD_SHA
    control.check_access_metadata = actual_access_check
    return control.run('push', 'audit', ROOT / 'desktop_invocations.json')


if __name__ == '__main__':
    raise SystemExit(main())
