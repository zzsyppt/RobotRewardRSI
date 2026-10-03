"""Offline admission tests: raw-only, frozen source/geometry/base, no free text."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('terrain_builder', ROOT/'scripts/build_stair_terrain_supplement.py')
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA


class OfflineTerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = json.loads((ROOT/'PLAN/feedback_sys/NEW_SYS/P2a/source_lock.json').read_text())['sources']
        cls.qualification = ROOT/'outputs/p2c3/historical_02/stdout.json'
        if not cls.qualification.is_file() or any(not (ROOT/s['manifest_relative_path']).is_file() for s in cls.sources):
            raise unittest.SkipTest('historical artifacts not installed; required by P2c3 acceptance runner')

    def reference(self, source=0):
        s = self.sources[source]
        return {'schema_version': NUMERIC_EVIDENCE_REF_SCHEMA, 'manifest_path': str(ROOT/s['manifest_relative_path']),
                'manifest_sha256': s['manifest_sha256']}

    def call(self, directory, **kwargs):
        args = dict(reference=self.reference(), source_id='k1', output=Path(directory)/'build',
                    qualification=self.qualification, qualification_sha256=builder.OFFLINE_QUALIFICATION_SHA,
                    base_public=None)
        args.update(kwargs)
        return builder.build(**args)

    def test_tampered_qualification_even_with_recomputed_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/p2c3') as d:
            path = Path(d)/'bad.json'
            history = json.loads(self.qualification.read_text())
            history['sources'][0]['files'] = []
            path.write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, 'qualification artifact hash'):
                self.call(d, qualification=path, qualification_sha256=builder.file_sha256(path))
            self.assertFalse((Path(d)/'build').exists())

    def test_relabelled_or_cross_source_archive_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/p2c3') as d:
            with self.assertRaisesRegex(ValueError, 'base/source mismatch'):
                self.call(d, reference=self.reference(1))
            self.assertFalse((Path(d)/'build').exists())

    def test_arbitrary_or_resigned_public_text_rejected(self):
        basepath = ROOT/'outputs/p2a/k1/public.json'
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/p2c3') as d:
            bad = Path(d)/'base.json'; value = json.loads(basepath.read_text())
            value['invented'] = 'FL sole clearance 42m'
            bad.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'base differs'):
                self.call(d, base_public=bad)
            real_hash = builder.file_sha256
            def corrupt(path): return real_hash(bad) if path == basepath else real_hash(path)
            with patch.object(builder, 'file_sha256', corrupt), self.assertRaisesRegex(ValueError, 'historical public hash'):
                self.call(d)

    def test_unqualified_archive_and_geometry_metadata_rejected(self):
        original = builder._load_contract(self.reference())
        for mode in ('unknown_archive', 'origin', 'type', 'cell'):
            archive = copy.copy(original)
            if mode == 'unknown_archive':
                reference = {**self.reference(), 'manifest_sha256': '0'*64}
            else:
                reference = self.reference()
                from dataclasses import replace
                cell = archive.cells[0]
                changes = {'origin': {'expected_origin_m': (0., 0., 1.)},
                           'type': {'terrain_type_index': 999},
                           'cell': {'spec': archive.cells[12].spec}}[mode]
                archive.cells = (replace(cell, **changes), *archive.cells[1:])
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=ROOT/'outputs/p2c3') as d:
                with patch.object(builder, '_load_contract', return_value=archive), self.assertRaises(ValueError):
                    self.call(d, reference=reference)
                self.assertFalse((Path(d)/'build').exists())


if __name__ == '__main__': unittest.main()
