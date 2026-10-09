"""Recorded reference points relative to qualified nominal Stair surfaces.

Source bindings and geometry are checked before description computation;
these vertical differences are not physical sole clearance.
"""
from __future__ import annotations

import numpy as np

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.rough_surface import normalize_stair_surface_profile, rough_surface_offset_m

FRAME = 'world_z_up_translation_only'
BOUNDARY_GUARD_M = 1e-4
STATUS = ('known', 'nonfinite_position', 'missing_origin', 'unqualified_geometry',
          'outside_patch', 'boundary_band')


def sample_reference_height(points, origin, geometry, *, geometry_digest, unit, frame):
    """Return world nominal heights, signed vertical differences, status codes.

Unknown samples stay NaN in numeric traces.
The only supported transform is the explicitly qualified world-axis translation.
"""
    if unit != 'm' or frame != FRAME:
        raise ValueError('terrain position unit/frame is not qualified')
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('terrain reference points require [sample, xyz]')
    profile = _qualified_geometry(geometry, geometry_digest)
    return _sample_reference_height(points, origin, geometry, profile)


def _qualified_geometry(geometry, geometry_digest):
    if geometry is None or geometry.get('surface_profile', {}).get('kind') not in ('flat', 'bounded_piecewise_linear_v1'):
        return None
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
    return profile


def _sample_reference_height(points, origin, geometry, profile):
    """Compute only after coordinate/geometry qualification."""
    height = np.full(len(points), np.nan)
    status = np.zeros(len(points), dtype=np.uint8)
    if profile is None:
        status[:] = STATUS.index('unqualified_geometry')
        return height, height.copy(), status
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
    edges = np.empty(len(geometry['tread_depths_m'])+1)
    edges[0] = geometry['start_platform_length_m']
    for i, depth in enumerate(geometry['tread_depths_m']):
        edges[i+1] = edges[i] + depth
    levels = np.r_[0., np.cumsum(geometry['riser_heights_m'], dtype=np.float64)]
    guard = BOUNDARY_GUARD_M
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


def prepare_terrain_inputs(episode, geometry, *, geometry_digest):
    """Qualify source bindings before entering description computation."""
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
    profile = _qualified_geometry(geometry, geometry_digest)
    return entities, positions, profile


def measure_terrain(episode, geometry, inputs):
    """Return height, signed relative z and status arrays after prepare_terrain_inputs."""
    entities, positions, profile = inputs
    origin = episode.task.get('origin_world')
    heights, relatives, statuses = [], [], []
    for column in range(len(entities)):
        h, relative, status = _sample_reference_height(positions[:, column], origin, geometry, profile)
        heights.append(h); relatives.append(relative); statuses.append(status)
    return tuple(np.stack(a, axis=1) for a in (heights, relatives, statuses))
