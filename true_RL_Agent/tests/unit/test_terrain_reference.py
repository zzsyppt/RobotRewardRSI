"""Frozen geometry domain, semantic counterexamples, historical mesh checks."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.trajectory_terrain import sample_reference_height, measure_terrain, summarize, CONFIG, FRAME
from tests.unit.terrain_mesh_oracle import mesh_height, check_record, check_first_extrema
from tests.unit.test_trajectory_measurement import sample_episode

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT/'outputs/p2c3/historical_02/stdout.json'


def geometry():
    return {'size_m': [8., 4.], 'start_platform_length_m': 1.5, 'floor_depth_m': .5,
            'riser_heights_m': [.2, .3], 'tread_depths_m': [.5, .6], 'top_platform_length_m': 5.4,
            'surface_profile': {'kind': 'flat'}}


def sample(p, origin=(0., 0., 0.), g=None, **kwargs):
    g = geometry() if g is None else g
    return sample_reference_height(p, origin, g, geometry_digest=canonical_json_hash(g),
                                   unit='m', frame=FRAME, **kwargs)


def episode(n=2, dt=.02):
    ep = sample_episode(n, dt=dt)
    ep.source.update(coordinates={'positions': 'world_m', 'support_position': 'bound_rigid_body_reference_point'},
                     runtime_projection={'kinematic_base_source': 'articulation_root',
                                         'resolved_roles': {'support_effector': ['FL', 'FR', 'RL', 'RR']}})
    ep.task.update(cell_id='synthetic', static_refs={'terrain_env_origins_w': {
        'file': 'fixture.npz', 'sha256': 'c'*64, 'array': 'terrain_env_origins_w', 'array_index': [0]}})
    return ep


def measure(ep):
    g = geometry()
    return measure_terrain(ep, g, geometry_digest=canonical_json_hash(g), terrain_type=0, regime='flat')


class TerrainReferenceTests(unittest.TestCase):
    def test_flat_individual_xy_and_signed_difference(self):
        h, r, s = sample([[0, 0, .4], [.8, 0, .1], [1.4, 0, .8], [2, 0, .6]])
        np.testing.assert_allclose(h, [0, .2, .5, .5], atol=1e-15)
        np.testing.assert_allclose(r, [.4, -.1, .3, .1], atol=1e-15)
        self.assertEqual(s.tolist(), [0]*4)

    def test_nonzero_origin_and_translation(self):
        p = np.array([[0, .2, .4], [.9, -.3, .6], [2., .1, .9]])
        origin = np.array([12., -8., 2.5])
        a, b, s = sample(p)
        x, y, t = sample(p+origin, origin)
        np.testing.assert_allclose(x, a+2.5, atol=1e-14)
        np.testing.assert_allclose(y, b, atol=1e-14)
        np.testing.assert_array_equal(s, t)
        # Rising world z together with terrain origin is not extra relative elevation.
        np.testing.assert_allclose(y, b)

    def test_patch_edges_risers_and_outside_are_unknown(self):
        p = [[-.75, 0, 0], [7.25, 0, 0], [0, -2, 0], [0, 2, 0], [.75, 0, 0],
             [1.25, 0, 0], [1.85, 0, 0], [-.751, 0, 0], [7.251, 0, 0], [0, 2.01, 0]]
        h, r, s = sample(p)
        self.assertEqual(s.tolist(), [5]*7+[4]*3)
        self.assertTrue(np.isnan(h).all() and np.isnan(r).all())

    def test_boundary_band_closed_and_adjacent(self):
        g = geometry()
        edge, guard = 2., CONFIG['boundary_guard_m']
        values = [edge-guard, edge+guard, np.nextafter(edge-guard, -np.inf), np.nextafter(edge+guard, np.inf)]
        _, _, status = sample([[x, 0, 1] for x in values], origin=(.75, 0, 0), g=g)
        self.assertEqual(status.tolist(), [5, 5, 0, 0])

    def test_unknown_missing_and_nonfinite(self):
        h, _, status = sample([[np.nan, 0, 1], [0, 0, np.inf], [0, 0, 1]])
        self.assertEqual(status.tolist(), [1, 1, 0]); self.assertTrue(np.isnan(h[:2]).all())
        h, _, status = sample([[0, 0, 1]], origin=None)
        self.assertEqual(status.tolist(), [2]); self.assertTrue(np.isnan(h).all())
        h, _, status = sample_reference_height([[0, 0, 1]], [0, 0, 0], None,
            geometry_digest=None, unit='m', frame=FRAME)
        self.assertEqual(status.tolist(), [3]); self.assertTrue(np.isnan(h).all())
        g = geometry(); g['surface_profile'] = {'kind': 'unrecorded_heightfield'}
        self.assertEqual(sample([[0, 0, 1]], g=g)[2].tolist(), [3])

    def test_semantic_unit_frame_digest_rejection(self):
        g = geometry()
        for key, val in [('unit', 'cm'), ('frame', 'body'), ('frame', 'world_rotated'), ('geometry_digest', 'bad')]:
            args = dict(geometry_digest=canonical_json_hash(g), unit='m', frame=FRAME); args[key] = val
            with self.subTest(key=key, val=val), self.assertRaises(ValueError):
                sample_reference_height([[0, 0, 1]], [0, 0, 0], g, **args)

    def test_xyz_frame_swap_changes_height_not_silently_corrected(self):
        p = [[1.4, 0, 1], [0, 1.4, 1]]
        h, _, _ = sample(p)
        self.assertEqual(h.tolist(), [.5, 0.])

    def test_malformed_geometry_rejected_even_if_rehashed(self):
        for key, value in [('riser_heights_m', [-.2, .3]), ('tread_depths_m', [.5]),
                           ('top_platform_length_m', 10), ('size_m', [8, -4])]:
            g = geometry(); g[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): sample([[0, 0, 1]], g=g)

    def test_measurement_nan_inf_missing_origin_json_and_gaps(self):
        ep = episode(3); ep.steps = np.array([1, 3, 5])
        ep.arrays['support_pos_w'][0, 0, 0] = np.nan
        ep.arrays['support_pos_w'][1, 0, 0] = np.inf
        records, arrays = measure(ep)
        json.dumps(records, allow_nan=False)
        self.assertEqual(records[1]['value']['counts']['unknown'], 2)
        self.assertEqual(records[1]['value']['sample_cell_seconds']['total'], .06)
        self.assertIsNone(records[1]['value']['first_by_status']['nonfinite_position']['position_world_m'][0])
        for origin in (None, [np.nan, 0, 0]):
            ep.task['origin_world'] = origin
            r, a = measure(ep); json.dumps(r, allow_nan=False)
            self.assertEqual(r[0]['value']['counts']['unknown'], 3)
            self.assertIsNone(r[0]['value']['statistics']['relative_z_m'])

    def test_measurement_entity_unit_and_axis_mismatch(self):
        for kind in ('entity', 'unit', 'axis', 'base'):
            ep = episode()
            if kind == 'entity': ep.entities['support'] = tuple(reversed(ep.entities['support']))
            elif kind == 'unit': ep.source['coordinates']['positions'] = 'world_cm'
            elif kind == 'axis': ep.task['progress_direction_world'] = [0, 1, 0]
            else: ep.source['runtime_projection']['kinematic_base_source'] = 'body_role'
            with self.subTest(kind=kind), self.assertRaises(ValueError): measure(ep)

    def test_independent_record_oracle_rejects_resigned_semantics(self):
        ep = episode(); r, (h, z, s) = measure(ep)
        for key, value in [('entity_ids', ['other']), ('unit', 'cm'), ('frame', 'body')]:
            bad = copy.deepcopy(r[0]); bad[key] = value; bad['record_id'] = 'resigned'
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_record(bad, ep.arrays['base_pos_w'], h[:, 0], z[:, 0], s[:, 0], .02, ep.entities['root'][0])

    def test_first_extremum_witness_tamper_rejected_on_flat_tie(self):
        ep = episode(); records, (h, z, s) = measure(ep)
        check_first_extrema(records[0], h[:, 0], z[:, 0], ep.steps)
        bad = copy.deepcopy(records[0]); bad['record_id'] = 'resigned'
        bad['value']['witnesses']['height_min']['step'] = 2
        with self.assertRaises(AssertionError): check_first_extrema(bad, h[:, 0], z[:, 0], ep.steps)

    def test_episode_equal_and_pooled_denominators_differ(self):
        a, b = episode(1, .1), episode(3, .02)
        a.arrays['base_pos_w'][:, 2] = 1
        b.arrays['base_pos_w'][:, 2] = 3
        b.arrays['base_pos_w'][0, 0] = np.nan
        groups = summarize([{'records': measure(a)[0]}, {'records': measure(b)[0]}])['groups']
        g = groups['flat/robot_0.root']
        self.assertEqual(g['episode_equal']['mean']['mean'], 2)
        self.assertEqual(g['pooled_known_fraction'], .75)
        self.assertAlmostEqual(g['pooled_known_time_fraction'], .875)

    @unittest.skipUnless(HISTORY.is_file(), 'historical mesh artifact not installed; required by P2c3 acceptance runner')
    def test_historical_all_cells_triangles_and_boundaries(self):
        self.assertTrue(HISTORY.is_file(), 'P2c3 acceptance requires original historical mesh evidence, no skip')
        history = json.loads(HISTORY.read_text())
        for cell in history['sources'][0]['cells']:
            vertices, centers, seams = [], [], []
            for mesh in cell['meshes']:
                v = np.array(mesh['vertices']); t = v[np.array(mesh['faces'])]
                normals = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
                top = t[normals[:, 2] > 0]
                # Every actual top face centroid, vertex and edge midpoint.
                vertices.extend(top.reshape(-1, 3).tolist())
                centers.extend(top.mean(axis=1).tolist())
                seams.extend(((top[:, 0]+top[:, 2])/2).tolist())
            p = np.array(vertices+centers+seams)-cell['mesh_origin']
            p[:, 2] += .3
            g = cell['geometry']
            observed = sample(p, g=g)
            expected = mesh_height(p, [0, 0, 0], cell)
            for i in (0, 1): np.testing.assert_allclose(observed[i], expected[i], atol=1e-10, rtol=0, equal_nan=True)
            np.testing.assert_array_equal(observed[2], expected[2])


if __name__ == '__main__': unittest.main()
