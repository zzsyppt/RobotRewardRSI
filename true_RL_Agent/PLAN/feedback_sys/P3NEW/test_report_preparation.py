"""Temporary offline fixtures only: no real fact, seal, call or semantic verdict."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_facts_offline as facts
import prepare_confirmation_reports as reports
import run_confirmation_review as runner
import audit_confirmation_facts as structural


class ReportPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.index = {'units': [], 'facts': []}
        packets = {}
        for axis, count in (('episode', 24), ('focus', 12), ('media', 4)):
            for i in range(1, count + 1):
                packet = f'{axis}-{i:02d}'
                task = ('stair' if i <= count // 2 else 'push') if axis != 'focus' else ('push' if i <= 6 else 'stair')
                ds = axis == 'media' or i in (1, count // 2 + 1)
                packets[packet] = {'A': {'judge_id': 'gpt6.1-sol-a', 'model': 'gpt-6.1-sol', 'carrier': 'fresh_cli'},
                    'B': {'judge_id': 'deepseek' if ds else 'gpt6.1-sol-b',
                          'model': runner.MODEL if ds else 'gpt-6.1-sol', 'carrier': 'deepseek_api' if ds else 'fresh_cli'}}
                cls.index['units'].append({'packet_id': packet, 'axis': axis, 'task': task, 'scope': 'anonymous-01'})
        assignment = cls.root / 'assignment.json'
        runner.save(assignment, {'schema_version': runner.ASSIGNMENT_SCHEMA, 'status': 'FROZEN', 'packets': packets})
        cls.index['assignment'] = runner.ref(assignment)
        materials = []
        protocol = Path(runner.__file__).parent / 'N5_FACT_REVIEW_PROTOCOL_ZH.md'
        for name, role, data in (('protocol', 'review_protocol', runner.packet_protocol(protocol)),
                                 ('questions', 'focus_questions', {'questions': ['fixture']}),
                                 ('raw', 'raw_numeric', {'x': [1, 2, 3]})):
            path = cls.root / (name + '.json')
            runner.save(path, data)
            materials.append(reports.material(path, role, name))
        for unit in cls.index['units']:
            path = cls.root / (unit['packet_id'] + '.json')
            runner.save(path, {'schema_version': 'p3new.confirmation_review_materials.v1', 'packet_id': unit['packet_id'],
                'stage': 'facts', 'purpose': 'unit_test_fixture_only', 'reference_contract': runner.coverage.BLOCK_CONTRACT,
                'materials': materials, **runner.assignment_fields(assignment, unit['packet_id'])})
            unit['facts_manifest'] = runner.ref(path)
        prepared = facts.prepare(cls.index, cls.root / 'prepared-facts')
        for row in prepared['rows']:
            folder = Path(row['prepared']['file']).parent
            card = runner.read(folder / 'prepare.json')
            catalog = runner.read(folder / 'reference_catalog.json')
            answer = {'judge_id': row['judge_id'], 'packet_id': row['packet_id'], 'stage': 'facts', 'complete': True,
                'reference_catalog_sha256': card['reference_catalog']['sha256'],
                'findings': {'unit_id': row['packet_id'], 'materials_seen': ['protocol', 'questions', 'raw'],
                    'coverage': 'All temporary fixture fields; no media supplied', 'major_visible_forms': [],
                    'unknowns': [], 'missing_materials': [],
                    'necessary_facts': [{'fact_id': 'fixture-F1', 'unit_id': row['packet_id'], 'question_bundle': 'Q1',
                    'statement': 'Fixture only', 'body': None, 'interval': None, 'event_anchor': None,
                    'modality': 'numeric', 'predicate': None, 'polarity': 'positive', 'scope': 'fixture',
                    'numerator': None, 'denominator': None, 'necessary_reason': 'Q1 fixture binding only',
                    'evidence_refs': [catalog['items'][0]['anchor_id']]}]}}
            answer_path = folder / 'fixture-answer.json'
            runner.save(answer_path, answer)
            row.update(request=runner.ref(folder / 'request.http_body.json'),
                       reference_catalog=runner.ref(folder / 'reference_catalog.json'), answer=runner.ref(answer_path))
            structure = structural.audit_contract(folder / 'prepare.json', answer_path)
            assert structure['structural_contract_complete'], structure['errors']
            structure_path = folder / 'fixture-structure.json'
            runner.save(structure_path, structure)
            row['structure_audit'] = runner.ref(structure_path)
            if row['judge_id'] == 'deepseek':
                receipt = {'status': 'received', 'complete': True, 'finish_reason': 'stop', 'incomplete_reasons': [],
                    'packet_id': row['packet_id'], 'judge_id': row['judge_id'], 'stage': 'facts',
                    'request_sha256': card['request_sha256'], 'content': row['answer']}
            else:
                transport_path, registration_path = folder / 'fixture-transport.json', folder / 'fixture-registration.json'
                runner.save(transport_path, {'status': 'PREPARED_TRANSPORT_ONLY_NOT_RELEASED', 'development_only': False,
                    'packet_id': row['packet_id'], 'judge_id': row['judge_id'], 'stage': 'facts',
                    'source_prepare': row['prepared'], 'source_request': row['request'], 'source_manifest': row['facts_manifest']})
                runner.save(registration_path, {'status': 'REGISTERED_ONE_CALL', 'transport': runner.ref(transport_path)})
                row.update(transport_receipt=runner.ref(transport_path), registration=runner.ref(registration_path))
                receipt = {'status': 'COMPLETED_TRANSPORT_VERIFIED_NOT_RELEASED', 'exit_code': 0,
                           'raw_answer': row['answer'], 'registration': row['registration']}
            path = folder / 'fixture-driver.json'
            runner.save(path, receipt)
            row['driver_receipt'] = runner.ref(path)
            cls.index['facts'].append(row)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete80_engineering_only_and_mixed_caps(self):
        self.assertEqual(len(reports.audit_all(self.index)), 80)
        self.assertEqual(sum(r['max_output_tokens'] == 65536 for r in self.index['facts']), 8)
        self.assertEqual(sum(r['max_output_tokens'] == 32768 for r in self.index['facts']), 72)
        for unit in self.index['units']:
            self.assertNotIn('max_output_tokens', runner.read(Path(unit['facts_manifest']['file'])))

    def test_current_structure_generator_rejects_helper_or_protocol_mismatch(self):
        row = self.index['facts'][0]
        prepared, answer = Path(row['prepared']['file']), Path(row['answer']['file'])
        other = self.root / 'wrong-source.txt'
        other.write_text('wrong source fixture', encoding='utf-8')
        with patch.object(structural, 'HELPER', other), self.assertRaisesRegex(ValueError, 'coverage_helper_differs'):
            structural.audit_contract(prepared, answer)
        with patch.object(structural, 'PROTOCOL', other), self.assertRaisesRegex(ValueError, 'protocol_differs'):
            structural.audit_contract(prepared, answer)

    def test_missing_duplicate_or_wrong_catalog_fail_before_output(self):
        for mutate in ('missing', 'duplicate', 'catalog'):
            index = deepcopy(self.index)
            if mutate == 'missing': index['facts'].pop()
            elif mutate == 'duplicate': index['facts'][-1] = index['facts'][0]
            else: index['facts'][0]['reference_catalog']['sha256'] = '0' * 64
            out = self.root / ('bad-' + mutate)
            with self.assertRaises(ValueError): reports.prepare(index, out)
            self.assertFalse(out.exists())

    def test_incomplete_transport_rejected_and_original_not_changed(self):
        index = deepcopy(self.index)
        path = self.root / 'incomplete-driver.json'
        runner.save(path, {'status': 'TRANSPORT_NOT_VERIFIED'})
        index['facts'][0]['driver_receipt'] = runner.ref(path)
        with self.assertRaisesRegex(ValueError, 'transport incomplete'):
            reports.audit_all(index)

    def test_all80_sealed_before_any_real_prepare_and_own_catalogs_survive(self):
        index = deepcopy(self.index)
        numeric, media = [], []
        for unit in index['units']:
            if unit['axis'] != 'media':
                numeric.append({**{k: unit[k] for k in ('packet_id', 'task', 'axis', 'scope')},
                                'facts_manifest_sha256': unit['facts_manifest']['sha256']})
            else:
                number = int(unit['packet_id'].split('-')[1])
                unit.update(alias='e1' if number % 2 else 'e2', original_id=f'r1.e{number}')
                media.append({'unit_id': unit['packet_id'], 'task': unit['task'], 'alias': unit['alias'],
                              'source_run': 1, 'source_env': number, 'facts_manifest': unit['facts_manifest']})
        for name, value in {
            'numeric_packing': {'purpose': 'confirmation', 'packets': numeric,
                'source_sha256': {'capture_card': 'fixture-capture', 'raw_manifest': 'fixture-raw', 'private_source_manifest': 'fixture-private'}},
            'media_packing': {'units': media, 'capture_card': {'sha256': 'fixture-capture'}}
        }.items():
            path = self.root / (name + '.json'); runner.save(path, value); index[name] = runner.ref(path)
        index['tasks'] = {task: {'raw_manifest': {'sha256': 'fixture-raw'}, 'raw_sources': {'sha256': 'fixture-private'}} for task in ('stair', 'push')}
        # Task selection is tested independently. Here the real report manifest,
        # seal and runner serialization execute for every slot, with tiny text.
        def selected(unit, data, destination, originals):
            destination.mkdir(parents=True)
            path = destination / 'tiny.json'; runner.save(path, {'report': {'text': 'fixture only'}})
            if unit['axis'] == 'media': runner.save(destination / 'observer_inventory_mapping.json', {'fixture_only': True})
            return [reports.material(path, 'automatic_report', 'fixture-report')]
        out = self.root / 'complete-report-fixture'
        original_prepare = runner.prepare
        counts = []
        def prepare_checked(*args, **kwargs):
            counts.append(len(list((out / 'seals').glob('*/*.json'))))
            return original_prepare(*args, **kwargs)
        with patch.object(reports, 'task_materials', return_value={}), patch.object(reports, 'select_materials', side_effect=selected), \
             patch.object(runner, 'prepare', side_effect=prepare_checked):
            receipt = reports.prepare(index, out)
        self.assertEqual(counts, [80] * 80)
        self.assertEqual(len(receipt['reports']), 80)
        self.assertIsNone(receipt['semantic_acceptance'])
        self.assertFalse(receipt['capacity_release'])
        for result in receipt['reports']:
            original = next(r for r in index['facts'] if (r['packet_id'], r['judge_id']) == (result['packet_id'], result['judge_id']))
            seal = runner.read(Path(result['seal']['file']))
            card = runner.read(Path(result['prepared']['file']))
            self.assertEqual(seal['facts_manifest_sha256'], original['facts_manifest']['sha256'])
            self.assertEqual(seal['answer'], original['answer'])
            self.assertEqual(card['own_facts_reference_catalog'], original['reference_catalog'])
            self.assertEqual(result['max_output_tokens'], facts.output_cap(result['judge_id']))
        wrong = deepcopy(index); wrong['units'][0]['task'] = 'push'
        with self.assertRaisesRegex(ValueError, 'routing differs'):
            reports.verify_source_index(wrong)

    def test_own_seal_preserves_answer_catalog_and_failure_cleans_all(self):
        index = deepcopy(self.index)
        out = self.root / 'failed-materials'
        index['tasks'] = {}
        with patch.object(reports, 'verify_source_index'), self.assertRaises(KeyError):
            reports.prepare(index, out)
        self.assertFalse(out.exists())
        row = index['facts'][0]
        folder = self.root / 'own-seal-test'
        folder.mkdir()
        seal = folder / 'seal.json'
        runner.save(seal, {'schema_version': 'p3new.confirmation_facts_seal.v1', 'status': 'SEALED',
            'packet_id': row['packet_id'], 'judge_id': row['judge_id'], 'answer': row['answer'],
            'facts_manifest_sha256': row['facts_manifest']['sha256'], 'reference_catalog': row['reference_catalog'],
            'reference_contract': runner.coverage.BLOCK_CONTRACT})
        public = folder / 'public.json'
        runner.save(public, {'report': {'test': 'fixture'}})
        manifest = folder / 'report.json'
        reports.pack.prepare_report_manifest(Path(row['facts_manifest']['file']), [reports.material(public, 'automatic_report', 'report')], manifest)
        body, _, card = runner.build(manifest, row['judge_id'], seal)
        materials, _ = runner.coverage.request_materials(json.loads(body))
        self.assertEqual(materials['own_sealed_facts']['data'], runner.read(Path(row['answer']['file'])))
        self.assertEqual(card['own_facts_reference_catalog'], row['reference_catalog'])
        other = self.index['facts'][1]['judge_id']
        with self.assertRaisesRegex(ValueError, 'another judge'):
            runner.build(manifest, other, seal)

    def test_three_axis_selection_keeps_whole_scope_and_raw_observations(self):
        folder = self.root / 'axis-fixture'
        folder.mkdir()
        refs = {}
        for name in ('pool', 'examples', 'shared', 'locator', 'public'):
            path = folder / (name + '.json'); runner.save(path, {'fixture': name}); refs[name] = path
        data = {'aliases': {'anon': 'r1.e1'}, 'rows': {'r1.e1': (0, {'all': [1, 2, 3]})},
                'shared': {'definitions': 'complete'}, 'provenance': {'fixture': True}, 'refs': refs,
                'request': refs['public']}
        unit = {'task': 'push', 'scope': 'anon', 'axis': 'episode'}
        episode = reports.select_materials(unit, data, folder / 'episode')
        self.assertEqual(runner.read(Path(episode[0]['file']))['complete_episode_derivation'], {'all': [1, 2, 3]})
        unit['axis'] = 'focus'
        focus = reports.select_materials(unit, data, folder / 'focus')
        self.assertEqual({x['id'] for x in focus}, {'report-pool', 'report-examples', 'report-shared', 'report-locator', 'report-final-request'})
        raw = {'episodes': [{'alias': 'e1', 'observations': [
            {'text': 'original insufficient retained', 'uncertainty': 'sparse frames', 'body': 'FR_calf', 'relation': 'temporal_change'},
            {'text': 'other original', 'uncertainty': 'partial', 'body': 'root', 'relation': 'spatial'}], 'not_visible': ['occluded']}]}
        raw_path, audit_path = folder / 'raw.json', folder / 'audit.json'
        runner.save(raw_path, raw)
        runner.save(audit_path, {'raw_output': raw, 'observations': [{'alias': 'e1', 'media_identity': {'episode_id': 'r1.e1'},
                                                                   'image_evidence': [{'image_sha256': 'png-sha'}]}]})
        data['full'] = {'sentences': [{'id': 'own', 'media_identity': {'episode_id': 'r1.e1'}},
                                   {'id': 'other', 'media_identity': {'episode_id': 'r1.e2'}}]}
        data['public'] = {'report': {'visual': [{'id': 'own', 'text': 'own visual'}, {'id': 'other', 'text': 'other alias'},
                                              {'id': 'visual.coverage', 'text': 'recorded decision counts'}]}}
        runner.save(refs['public'], data['public'])
        unit.update(axis='media', alias='e1', original_id='r1.e1', observer_output=runner.ref(raw_path), observer_audit=runner.ref(audit_path))
        original = {'image': {'role': 'raw_image', 'sha256': 'png-sha'},
                    'metadata': {'role': 'raw_media_metadata', 'data': {'alias': 'e1', 'images': [{'sha256': 'png-sha'}]}}}
        media = reports.select_materials(unit, data, folder / 'media', original)
        self.assertEqual(runner.read(Path(media[0]['file'])), raw)
        self.assertEqual(media[-1]['review_text_roots'], [{'json_pointer': '/data/report/visual/0/text'}, {'json_pointer': '/data/report/visual/2/text'}])
        self.assertIn('not ground truth', media[-1]['label'])
        mapping = runner.read(folder / 'media/observer_inventory_mapping.json')
        self.assertEqual(len(mapping['observations']), 2)
        envelopes = {m['id']: {**m, 'data': runner.read(Path(m['file']))} for m in media}
        inventory = runner.coverage.build_inventory(envelopes)
        # Two original text+uncertainty pairs, one not-visible statement, own
        # projection and recorded count. No other alias, body or relation count.
        self.assertEqual(len(inventory['items']), 7)

    def test_task_selection_uses_all_report_examples_not_focus_or_positive_subset(self):
        folder = self.root / 'shared-selection'
        (folder / '_private').mkdir(parents=True)
        full = {'domain': {'archive_hash': 'source-sha'}, 'semantics': {}, 'semantic': {'contract': {}},
                'episodes': [{'episode_id': 'r1.e1', 'events': []}, {'episode_id': 'r2.e9', 'events': []}]}
        public = {'report': {'process': [{'text': 'r1.e1 measured'}], 'counterexample': [{'text': 'r2.e9 same process success'}]}}
        raw_sources = {'sources': [{'source_id': 'stair1', 'manifest_sha256': 'source-sha'}],
                       'anonymous_mapping': [{'task': 'stair', 'episode_id': 'anon', 'original_id': 'r1.e1'}]}
        values = {'full': full, 'raw_manifest': {'fixture': True}, 'raw_sources': raw_sources}
        refs = {}
        for key, value in values.items():
            path = folder / ('_private/sources.json' if key == 'raw_sources' else 'manifest.json' if key == 'raw_manifest' else key + '.json')
            runner.save(path, value); refs[key] = runner.ref(path)
        for key, value in {
            'phase': {'task_evidence': {'behavior_full': refs['full'], 'behavior': public}},
            'request': {'messages': [{'content': 'fixture'}, {'content': json.dumps({'public_feedback_summary': {'behavior': {'profile': public}}})}]}
        }.items():
            path = folder / (key + '.json'); path.write_bytes(runner.encoded(value)); refs[key] = runner.ref(path)
        refs['request_status'] = 'offline_serialized_not_sent'
        # Existing arithmetic helpers have their own tests. This checks selection
        # supplies the entire pool and BOTH report-referenced episodes, not a subset.
        with patch.object(reports.pack, 'report_pool_operands', return_value={'fixture': 'whole pool'}) as pool, \
             patch.object(reports.pack, 'report_examples', return_value={'fixture': 'all referenced episodes'}) as examples:
            result = reports.task_materials('stair', refs, folder / 'out')
        self.assertEqual(pool.call_args.args, (folder, 'stair', {'anon': 'r1.e1'}))
        self.assertEqual(examples.call_args.args[2], ['r1.e1', 'r2.e9'])
        self.assertEqual(result['public'], public)


if __name__ == '__main__':
    unittest.main()
