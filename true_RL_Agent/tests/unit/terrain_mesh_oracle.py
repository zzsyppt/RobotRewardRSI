"""Independent nominal geometry oracle: triangle vertices/faces, never surface helpers."""
import numpy as np


def mesh_height(points, origin, cell):
    points = np.asarray(points, dtype=np.float64)
    heights = np.full(len(points), np.nan)
    status = np.zeros(len(points), dtype=np.uint8)
    if origin is None or np.shape(origin) != (3,) or not np.isfinite(origin).all():
        return heights, heights.copy(), np.full(len(points), 2, dtype=np.uint8)
    local = points - origin + np.asarray(cell['mesh_origin'])
    x, y = local[:, 0], local[:, 1]
    finite = np.isfinite(points).all(axis=1)
    status[~finite] = 1
    floor = np.asarray(cell['meshes'][0]['vertices'])
    lo, hi = floor[:, :2].min(axis=0), floor[:, :2].max(axis=0)
    outside = finite & ((x < lo[0]) | (x > hi[0]) | (y < lo[1]) | (y > hi[1]))
    status[outside] = 4
    guard = 0.0001
    edge = (x <= lo[0]+guard) | (x >= hi[0]-guard) | (y <= lo[1]+guard) | (y >= hi[1]-guard)
    for mesh in cell['meshes'][1:]:
        vertices = np.asarray(mesh['vertices'])
        # Each raised solid supplies its actual x boundaries; duplicates are harmless.
        for boundary in (vertices[:, 0].min(), vertices[:, 0].max()):
            edge |= (x >= boundary-guard) & (x <= boundary+guard)
    status[finite & ~outside & edge] = 5
    known = status == 0
    top = np.full(len(points), -np.inf)
    for mesh in cell['meshes']:
        v = np.asarray(mesh['vertices']); faces = np.asarray(mesh['faces'])
        for t in v[faces]:
            normal = np.cross(t[1]-t[0], t[2]-t[0])
            if normal[2] <= 0: continue
            low, high = t[:, :2].min(axis=0), t[:, :2].max(axis=0)
            candidate = np.flatnonzero(known & (x >= low[0]-1e-12) & (x <= high[0]+1e-12)
                                      & (y >= low[1]-1e-12) & (y <= high[1]+1e-12))
            if not len(candidate): continue
            uv = np.linalg.solve(np.stack([t[1, :2]-t[0, :2], t[2, :2]-t[0, :2]], axis=1),
                                 (local[candidate, :2]-t[0, :2]).T).T
            inside = (uv[:, 0] >= -1e-12) & (uv[:, 1] >= -1e-12) & (uv.sum(axis=1) <= 1+1e-12)
            selected, uv = candidate[inside], uv[inside]
            z = t[0, 2]+uv[:, 0]*(t[1, 2]-t[0, 2])+uv[:, 1]*(t[2, 2]-t[0, 2])
            top[selected] = np.maximum(top[selected], z)
    if not np.isfinite(top[known]).all():
        raise AssertionError('qualified nominal XY did not intersect an upward mesh face')
    heights[known] = top[known] + origin[2] - cell['mesh_origin'][2]
    return heights, points[:, 2]-heights, status
