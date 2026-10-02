"""Bounded nonblind checks of the original answer; no scoring or repair."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parents[2]
sys.path.insert(0, str(P))
import confirmation_review_coverage as coverage
raw_path = D.parent / 'focus_arithmetic_navigation_r1/original_focus/stair_c1_conditional_frontier.json'
assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == '129fc7f6949c10029c8389801195e25d6bfc9af2a6e25aa61da3ba753f0d7669'
source = json.loads(raw_path.read_bytes())
request = json.loads((D / 'prepared/request.http_body.json').read_bytes())
related = coverage.request_materials(request)[0]['related-raw']['data']
members = source['all_scope_rows']
by_id = {row['episode_id']: row for row in members}
dwell = {name: required for name, width, required in related['predicate_order']}
def first(row, predicate, component=0):
    spans = row['direct_measurements']['threshold_intervals_by_component'][predicate][component]
    return min((a+dwell[predicate]-1 for a, b in spans if b-a+1 >= dwell[predicate]), default=None)
def confirmed(row, predicate, component=0):
    step = first(row, predicate, component)
    return step is not None and step < row['steps']
def distribution(predicate, selector, width=1):
    group = [row for row in members if selector(row)]
    return [sum(confirmed(row, predicate, i) for row in group) for i in range(width)]
fail = lambda row: not row['original_success']
success = lambda row: row['original_success']
forbidden = lambda row: row['original_outcome'] == 'forbidden_contact'
pool = source['complete_task_pool_members']
assert len(pool) == len({row['episode_id'] for row in pool}) == 864
actual_pool = dict(Counter(row['original_outcome'] for row in pool))
claimed_pool = {'success': 682, 'forbidden_contact': 127, 'timeout': 37,
                'base_orientation_instability': 16, 'off_course': 1, 'slip': 1}
checks = []
def check(fact_ids, description, actual, expected):
    checks.append({'fact_ids': fact_ids, 'description': description, 'actual': actual,
                   'claimed_or_expected': expected, 'matches': actual == expected})
check(['focus07-f01'], 'whole-pool original outcome counts', actual_pool, claimed_pool)
check(['focus07-f02'], 'frontier original outcomes', dict(Counter(r['original_outcome'] for r in members)),
      {'success': 55, 'forbidden_contact': 36, 'timeout': 9, 'base_orientation_instability': 8})
for fact, name, fail_count, success_count in (
    ('f05', 'root_x_window_max_minus_start_lt_query_m', 46, 8),
    ('f06', 'abs_roll_or_pitch_gt_protocol_rad', 33, 2),
    ('f07', 'root_up_axis_world_z_lt_zero', 27, 0)):
    check(['focus07-'+fact], 'before-T confirmation by original failure/success',
          [distribution(name, fail)[0], distribution(name, success)[0]], [fail_count, success_count])
for fact, name, expected in (
    ('f09', 'loaded_planar_speed_gt_protocol_mps', [[24,24,31,19],[25,19,54,15],[19,17,21,14]]),
    ('f10', 'unloaded_speed3d_le_query_mps', [[6,4,2,12],[0,0,0,1],[0,0,0,0]]),
    ('f11', 'unloaded_clearance_lt_protocol_m', [[1,0,0,0],[0,0,0,0],[1,0,0,0]])):
    check(['focus07-'+fact], 'same-body confirmation counts: failure/success/forbidden-contact',
          [distribution(name, selector, 4) for selector in (fail, success, forbidden)], expected)
check(['focus07-f12'], 'per-body upward/downward confirmation totals',
      [distribution(name, lambda row: True, 4) for name in ('unloaded_vz_gt_query_mps','unloaded_vz_lt_negative_query_mps')],
      [[108]*4, [108,105,108,108]])
terminal_matches = all(first(row,'forbidden_buffer_force_gt_protocol_n') == row['steps'] and
    all(row['direct_measurements']['raw_termination_intervals'][name] == [[row['steps'],row['steps']]]
        for name in ('done_mask','wrapper_done_mask','base_contact')) for row in members if forbidden(row))
check(['focus07-f13'], '36 forbidden-contact force first-at-T and terminal masks; other72 no occurrence',
      [sum(forbidden(r) for r in members), terminal_matches,
       sum(first(r,'forbidden_buffer_force_gt_protocol_n') is not None for r in members if not forbidden(r))], [36,True,0])
expected_timeout_starts = {'stair-0049':116,'stair-0052':99,'stair-0053':102,'stair-0056':119,
    'stair-0339':110,'stair-0342':107,'stair-0343':97,'stair-0344':106,'stair-0635':104}
check(['focus07-f14'], 'all timeout low-progress single intervals',
      {r['episode_id']:r['direct_measurements']['threshold_intervals_by_component']['root_x_window_max_minus_start_lt_query_m'][0]
       for r in members if r['original_outcome']=='timeout'}, {name:[[start,588]] for name,start in expected_timeout_starts.items()})
r = by_id['stair-0061']
m = r['direct_measurements']
check(['focus07-f17'], 'located first-confirmation and original single intervals',
      [first(r,'root_x_window_max_minus_start_lt_query_m'),first(r,'abs_roll_or_pitch_gt_protocol_rad'),
       first(r,'root_up_axis_world_z_lt_zero'),m['threshold_intervals_by_component']['abs_roll_or_pitch_gt_protocol_rad'][0],
       m['threshold_intervals_by_component']['root_up_axis_world_z_lt_zero'][0],r['steps']],
      [119,156,156,[[147,637]],[[156,637]],637])
r = by_id['stair-0067']
check(['focus07-f18'], 'clearance/up-axis/force confirmation and sampled terminal differences',
      [first(r,'abs_roll_or_pitch_gt_protocol_rad'),first(r,'root_x_window_max_minus_start_lt_query_m'),
       first(r,'root_up_axis_world_z_lt_zero'),first(r,'unloaded_clearance_lt_protocol_m',0),
       first(r,'forbidden_buffer_force_gt_protocol_n'),r['steps']], [255,250,615,618,623,623])
for run, success_id, low_expected, rl_expected, denominator in (
    ('run_01','stair-0071',10,5,10),('run_02','stair-0353',8,5,9),('run_03','stair-0646',8,6,9)):
    selector = lambda row, run=run: forbidden(row) and row['run_group']==run and row['cell'].endswith('_level_5')
    matched = [row for row in members if selector(row)]
    check(['focus07-f20','focus07-f24'], 'same-run/cell matched '+run,
          [confirmed(by_id[success_id],'root_x_window_max_minus_start_lt_query_m'),
           confirmed(by_id[success_id],'loaded_planar_speed_gt_protocol_mps',2), len(matched),
           distribution('root_x_window_max_minus_start_lt_query_m',selector)[0],
           distribution('loaded_planar_speed_gt_protocol_mps',selector,4)[2]],
          [True,True,denominator,low_expected,rl_expected])
result = {'scope':'nonblind_development_arithmetic_readcheck_only','raw_answer_sha256':hashlib.sha256((D/'execution/raw_answer.json').read_bytes()).hexdigest(),
    'request_sha256':hashlib.sha256((D/'prepared/request.http_body.json').read_bytes()).hexdigest(),
    'original_focus_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),'checks':checks,
    'mismatches':[check for check in checks if not check['matches']],
    'limits':'Bounded selected claims checked against original supplied interval/membership operands, no production diagnostic function or new necessary-fact truth. Not all29 facts independently proven. Does not reconstruct complete per-step trajectories, visual motion, contact pairs, source execution or causation. Original answer unmodified; no formal scores or pass rate.'}
with (D/'arithmetic_recheck.json').open('xb') as stream:
    stream.write(json.dumps(result,ensure_ascii=False,indent=2).encode()+b'\n')
print(json.dumps({'checked_claim_groups':len(checks),'mismatches':result['mismatches']}))
