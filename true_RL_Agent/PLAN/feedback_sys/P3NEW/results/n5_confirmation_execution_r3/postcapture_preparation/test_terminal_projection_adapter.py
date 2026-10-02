"""Synthetic interface/mutation checks only; no R3 capture material is opened."""
from copy import deepcopy
from pathlib import Path
import unittest
from terminal_projection_check import load_contract, verify_episode_projection

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2]/'results/n5_confirmation_source_r3'

class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(SOURCE/'source_final',SOURCE/'source_manifest.final.json')

    def result(self,task,actual,source):
        return verify_episode_projection(actual,source,task=task,contract=self.contract)

    def stair(self):
        source = {'task':'stair','episode_id':'synthetic-stair','steps':4,'dt_s':0.02,
                  'original_success':False,'original_outcome':'forbidden_contact',
                  'original_terminal_record':{'episode_steps':4,'primary_reason':'forbidden_contact',
                     'success':False,'max_progress_m':0.23,'progress_fraction':0.1,
                     'diagnostic_flags':['excluded_production_label']},'direct_measurements':{'separate':123}}
        actual = deepcopy(source)
        del actual['direct_measurements']
        del actual['original_terminal_record']['diagnostic_flags']
        return actual,source

    def push(self):
        events = [{'event_index':i,'control_step':i+1,'observed_delta_v_world_mps':[0.0,0.2],
                   'injection_verified':True,'recovered':False,'recovery_control_step':None,
                   'recovery_censored':True,'process_diagnosis':{'excluded':True}} for i in (0,1)]
        source = {'task':'push','episode_id':'synthetic-push','steps':4,
                  'original_terminal_record':{'episode_steps':4,'success':False,'planned_push_count':2,
                  'push_events':events,'failure_label':'not_recovered','diagnostic_flags':['excluded']}}
        actual = deepcopy(source)
        del actual['original_terminal_record']['diagnostic_flags']
        for event in actual['original_terminal_record']['push_events']: del event['process_diagnosis']
        return actual,source

    def test_stair_legal_projection_preserves_source(self):
        actual,source = self.stair()
        before = deepcopy(source)
        result = self.result('stair',actual,source)
        self.assertEqual(result['status'],'PASS_EXACT_RETAINED_FIELDS')
        self.assertEqual(len(result['excluded_source_fields']),1)
        self.assertEqual(source,before)

    def test_push_legal_projection_lists_terminal_and_both_event_exclusions(self):
        actual,source = self.push()
        result = self.result('push',actual,source)
        self.assertEqual(result['status'],'PASS_EXACT_RETAINED_FIELDS')
        self.assertEqual(len(result['excluded_source_fields']),3)

    def test_changed_retained_value_fails(self):
        actual,source = self.stair()
        actual['original_terminal_record']['max_progress_m'] = 0.24
        self.assertEqual(self.result('stair',actual,source)['status'],'FAIL')

    def test_omitting_allowed_source_field_fails(self):
        actual,source = self.stair()
        del actual['original_terminal_record']['progress_fraction']
        self.assertEqual(self.result('stair',actual,source)['status'],'FAIL')

    def test_diagnostic_leak_is_not_permitted(self):
        actual,source = self.stair()
        actual['original_terminal_record']['diagnostic_flags'] = ['excluded_production_label']
        self.assertEqual(self.result('stair',actual,source)['status'],'FAIL')

    def test_event_diagnostic_leak_fails(self):
        actual,source = self.push()
        actual['original_terminal_record']['push_events'][0]['process_diagnosis'] = {'excluded':True}
        self.assertEqual(self.result('push',actual,source)['status'],'FAIL')

    def test_no_missing_source_field_default_is_invented(self):
        actual,source = self.push()
        actual['original_terminal_record']['mean_action_l2'] = 0.0
        self.assertEqual(self.result('push',actual,source)['status'],'FAIL')

    def test_bool_and_integer_are_distinct(self):
        actual,source = self.push()
        actual['original_terminal_record']['push_events'][0]['recovered'] = 0
        self.assertEqual(self.result('push',actual,source)['status'],'FAIL')

    def test_event_order_is_preserved(self):
        actual,source = self.push()
        actual['original_terminal_record']['push_events'].reverse()
        self.assertEqual(self.result('push',actual,source)['status'],'FAIL')

    def test_nonterminal_metadata_is_still_exact(self):
        actual,source = self.stair()
        actual['steps'] = 3
        self.assertEqual(self.result('stair',actual,source)['status'],'FAIL')

if __name__ == '__main__': unittest.main()
