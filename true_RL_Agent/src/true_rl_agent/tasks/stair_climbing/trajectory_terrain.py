"""M10: recorded reference points relative to qualified nominal Stair surfaces.

This has no physical collision/sole interpretation and does not alter the legacy
terrain query. Historical source qualification belongs to the offline caller.
"""
from __future__ import annotations

import numpy as np

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.trajectory_translation.contracts import stable_id
from true_rl_agent.tasks.stair_climbing.rough_surface import normalize_stair_surface_profile, rough_surface_offset_m

CATALOG_VERSION = 'trajectory.p0.measurements.v5'
FRAME = 'world_z_up_translation_only'
CONFIG = {'boundary_guard_m': 1e-4, 'surface': 'reconstructed_nominal_exterior_top',
          'boundary_rule': 'closed_float_constructed_coordinate_intervals',
          'runtime_surface': 'unknown_not_recorded'}
STATUS = ('known', 'nonfinite_position', 'missing_origin', 'unqualified_geometry',
          'outside_patch', 'boundary_band')


def sample_reference_height(points, origin, geometry, *, geometry_digest, unit, frame):
    """Return world nominal heights, signed vertical differences, status codes.

Unknown samples stay NaN in numeric traces and become null in JSON evidence.
The only supported transform is the explicitly qualified world-axis translation.
"""
    if unit != 'm' or frame != FRAME:
        raise ValueError('terrain position unit/frame is not qualified')
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('terrain reference points require [sample, xyz]')
    height = np.full(len(points), np.nan)
    status = np.zeros(len(points), dtype=np.uint8)
    if geometry is None or geometry.get('surface_profile', {}).get('kind') not in ('flat', 'bounded_piecewise_linear_v1'):
        status[:] = STATUS.index('unqualified_geometry')
        return height, height.copy(), status
    if canonical_json_hash(geometry) != geometry_digest:
        raise ValueError('terrain geometry digest mismatch')
    risers, treads = geometry['riser_heights_m'], geometry['tread_depths_m']
    dimensions = [*geometry['size_m'], geometry['start_platform_length_m'], geometry['floor_depth_m'],
                  geometry['top_platform_length_m'], *risers, *treads]
    if (len(geometry['size_m']) != 2 or not risers or len(risers) != len(treads)
            or not np.isfinite(dimensions).all() or min(dimensions) <= 0
            or abs(geometry['start_platform_length_m']+sum(treads)+geometry['top_platform_length_m']
                   -geometry['size_m'][0]) > 1e-9):
        raise ValueError('malformed nominal terrain geometry')
    profile = normalize_stair_surface_profile(geometry['surface_profile'])
    if 2*profile.get('amplitude_m', 0) >= min(risers):
        raise ValueError('roughness does not preserve separate positive risers')
    if origin is None or np.shape(origin) != (3,) or not np.isfinite(origin).all():
        status[:] = STATUS.index('missing_origin')
        return height, height.copy(), status
    origin = np.asarray(origin, dtype=np.float64)
    local = points - origin
    local[:, :2] += [geometry['start_platform_length_m']/2, geometry['size_m'][1]/2]
    finite = np.isfinite(points).all(axis=1)
    status[~finite] = STATUS.index('nonfinite_position')
    sx, sy = geometry['size_m']
    x, y = local[:, 0], local[:, 1]
    outside = finite & ((x < 0) | (x > sx) | (y < 0) | (y > sy))
    status[outside] = STATUS.index('outside_patch')
    # Follow the generator's sequential float additions, not rounded cell helpers.
    edges = np.empty(len(treads)+1)
    edges[0] = geometry['start_platform_length_m']
    for i, depth in enumerate(geometry['tread_depths_m']):
        edges[i+1] = edges[i] + depth
    levels = np.r_[0., np.cumsum(geometry['riser_heights_m'], dtype=np.float64)]
    guard = CONFIG['boundary_guard_m']
    boundary = (x <= guard) | (x >= sx-guard) | (y <= guard) | (y >= sy-guard)
    for edge in edges:
        boundary |= (x >= edge-guard) & (x <= edge+guard)
    status[finite & ~outside & boundary] = STATUS.index('boundary_band')
    known = np.flatnonzero(status == 0)
    segment = np.searchsorted(edges, x[known], side='right')
    height[known] = levels[np.minimum(segment, len(levels)-1)]
    if profile['kind'] == 'bounded_piecewise_linear_v1':
        for row, index in zip(known, segment, strict=True):
            if 1 <= index <= len(geometry['tread_depths_m']):
                height[row] += rough_surface_offset_m(profile, tread_index=int(index-1),
                    longitudinal_fraction=float((x[row]-edges[index-1])/geometry['tread_depths_m'][index-1]),
                    lateral_fraction=float(y[row]/sy))
    height[known] += origin[2]
    return height, points[:, 2]-height, status


def measure_terrain(episode, geometry, *, geometry_digest, terrain_type, regime):
    """Measure each recorded base/support point at its own XY; no contact operand."""
    ep = episode
    if ep.source['coordinates']['positions'] != 'world_m':
        raise ValueError('terrain episode coordinates must be world_m')
    if (ep.source['runtime_projection']['kinematic_base_source'] != 'articulation_root'
            or ep.source['coordinates']['support_position'] != 'bound_rigid_body_reference_point'):
        raise ValueError('terrain reference-point mapping is not qualified')
    entities = (*ep.entities['root'], *ep.entities['support'])
    supports = tuple('robot_0.support.'+name for name in ep.source['runtime_projection']['resolved_roles']['support_effector'])
    if ep.entities['root'] != ('robot_0.root',) or ep.entities['support'] != supports or len(entities) != 5:
        raise ValueError('terrain reference-point entity mapping differs from projection')
    if ep.task['progress_direction_world'] != [1., 0., 0.]:
        raise ValueError('terrain nominal axes are not the qualified world axes')
    base = ep.arrays.get('base_pos_w', np.full((len(ep.steps), 3), np.nan))
    support = ep.arrays.get('support_pos_w', np.full((len(ep.steps), 4, 3), np.nan))
    positions = np.concatenate([base[:, None, :], support], axis=1)
    if positions.shape != (len(ep.steps), len(entities), 3):
        raise ValueError('terrain reference-point entity shape mismatch')
    origin = ep.task.get('origin_world')
    geometry_binding = {'cell_id': ep.task['cell_id'], 'terrain_type': terrain_type,
                        'geometry_digest': geometry_digest, 'geometry': geometry,
                        'regime': regime, 'origin_world_m': ([float(v) if np.isfinite(v) else None for v in origin]
                                                            if origin is not None else None),
                        'static_refs': {**ep.task['static_refs'], 'terrain_types': {
                            **ep.task['static_refs']['terrain_env_origins_w'], 'array': 'terrain_types'}}}
    records, heights, relatives, statuses = [], [], [], []
    for column, entity in enumerate(entities):
        channel = 'base_pos_w' if column == 0 else 'support_pos_w'
        h, relative, status = sample_reference_height(positions[:, column], origin, geometry,
            geometry_digest=geometry_digest, unit='m', frame=FRAME)
        counts = {'total': len(status), **{name: int(np.count_nonzero(status == i)) for i, name in enumerate(STATUS)}}
        counts['unknown'] = counts['total']-counts['known']
        indices = np.flatnonzero(status == 0)

        def witness(row):
            return {'step': int(ep.steps[row]), 'position_world_m': [float(v) if np.isfinite(v) else None
                                                                  for v in positions[row, column]],
                    'height_world_m': float(h[row]) if status[row] == 0 else None,
                    'relative_z_m': float(relative[row]) if status[row] == 0 else None,
                    'status': STATUS[int(status[row])], 'column': None if column == 0 else column-1,
                    'source_ref': ep.reference((channel,), row, row+1)}

        extrema = {key: witness(int(indices[op(values[indices])])) if len(indices) else None
                   for key, values, op in [('height_min', h, np.argmin), ('height_max', h, np.argmax),
                       ('relative_min', relative, np.argmin), ('relative_max', relative, np.argmax)]}
        stats = {key: {'min': float(values[indices].min()), 'max': float(values[indices].max()),
                       'mean': float(values[indices].mean())} if len(indices) else None
                 for key, values in [('height_world_m', h), ('relative_z_m', relative)]}
        value = {'counts': counts, 'sample_cell_seconds': {k: v*ep.dt_s for k, v in counts.items()},
                 'statistics': stats, 'witnesses': extrema,
                 'first_by_status': {name: witness(int(np.flatnonzero(status == i)[0]))
                                     for i, name in enumerate(STATUS) if (status == i).any()},
                 'geometry_binding': geometry_binding, 'runtime_surface': 'unknown_not_recorded'}
        record = ep.record('M10.nominal_reference_height', 'measurement', (entity,), (channel,), value,
                           unit='m', frame=FRAME, config=CONFIG)
        record['catalog_version'] = CATALOG_VERSION
        record.pop('record_id')
        record['record_id'] = stable_id('nominal-terrain', record)
        records.append(record)
        heights.append(h); relatives.append(relative); statuses.append(status)
    return records, tuple(np.stack(a, axis=1) for a in (heights, relatives, statuses))
