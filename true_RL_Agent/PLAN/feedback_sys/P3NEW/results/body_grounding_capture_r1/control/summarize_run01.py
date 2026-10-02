"""Compact read-only tables for review of the downloaded original evidence."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
verification_path = root / 'receipt/verification.json'
verification = json.loads(verification_path.read_text())
run, = [r for r in verification['runs'] if r['run_index'] == 1]
with (root / 'materials/run01_body_image_points.csv').open('x', newline='', encoding='utf-8') as stream:
    writer = csv.writer(stream)
    writer.writerow(['image_file', 'control_step', 'view_id', 'body_name', 'pixel_x', 'pixel_y', 'projection_status'])
    for image in run['images']:
        for point in image['body_image_points']:
            xy = point['image_xy_px'] or [None, None]
            writer.writerow([Path(image['file']).name, point['control_step'], point['view_id'],
                             point['body_name'], *xy, point['projection_status']])
snapshot_path = root / 'materials/push_run01/video/public_dev-01.snapshots.json'
snapshot = json.loads(snapshot_path.read_text())
rows = []
for sample in snapshot['samples']:
    for view in sample['views']:
        params = view['camera_params']
        inverse = np.linalg.inv(np.asarray(params['cameraViewTransform']))
        actual_eye = inverse[3, :3] * params['metersPerSceneUnit']
        requested_eye = np.asarray(view['camera_eye_w'])
        rows.append({'step': sample['control_step'], 'view': view['view_id'],
                     'camera_model': params['cameraModel'], 'metres_per_unit': params['metersPerSceneUnit'],
                     'actual_eye_from_view_matrix_m': actual_eye.tolist(), 'recorded_requested_eye_m': requested_eye.tolist(),
                     'max_abs_eye_difference_m': float(np.max(np.abs(actual_eye - requested_eye)))})
receipt = {'scope': 'numeric camera view-matrix origin check only; image landing still needs visual review',
           'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
           'verification_sha256': hashlib.sha256(verification_path.read_bytes()).hexdigest(),
           'camera_origin_checks': rows, 'max_abs_eye_difference_m': max(row['max_abs_eye_difference_m'] for row in rows)}
with (root / 'materials/run01_camera_origin_check.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({'rows': len(rows), 'models': sorted({row['camera_model'] for row in rows}),
                  'metres_per_unit': sorted({row['metres_per_unit'] for row in rows}),
                  'max_abs_eye_difference_m': receipt['max_abs_eye_difference_m']}))
