"""Render measured-condition timelines; these plots are not simulator frames."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(full_path: Path, episode_id: str, output: Path) -> dict:
    data = full_path.read_bytes()
    full = json.loads(data)
    is_stair = 'stair.behavior' in full['schema_version']
    if is_stair:
        row = next(e for e in full['episodes'] if e['episode_id'] == episode_id)
        dt, end, outcome = row['dt_s'], row['steps'], row['primary_reason']
        spans = {}
        for event in row['events']:
            key = event['body'] + ' / ' + event['code']
            spans.setdefault(key, []).append((event['start'], event['end']))
        anchors = [] if row['anchor']['step'] is None else [(row['anchor']['step'], 'root first-riser crossing')]
    else:
        source, slot = (int(x[1:]) for x in episode_id.split('.'))
        row = next(e for e in full['transcripts'] if e['source'] == source and e['env_id'] == slot)
        dt, end, outcome = full['definitions']['step_dt_s'], row['valid_steps'], row['terminal_label']
        spans, anchors = {}, []
        for unit in row['recoveries']:
            name = str(unit['identity'])
            anchor = unit['anchor_control_step']
            if anchor is not None:
                anchors.append((anchor, name + ' anchor'))
            if unit['recovery_control_step'] is not None:
                anchors.append((unit['recovery_control_step'], name + ' recovery'))
            for condition, intervals in unit['condition_violation_spans'].items():
                spans[name + ' / ' + condition] = [(i['start_step'], i['end_step_inclusive']) for i in intervals]
            spans[name + ' / positive-dwell interrupted'] = [(e['control_step'], e['control_step']) for e in unit['dwell_interruptions']]
        relapse = row.get('first_instability_after_all_recoveries')
        if relapse is not None:
            anchors.append((relapse['control_step'],
                            'first later instability: ' + ', '.join(relapse['failed_conditions'])))
    height = max(4.2, .27 * len(spans) + 2.5)
    fig, ax = plt.subplots(figsize=(12, height), layout='constrained')
    colors = plt.get_cmap('tab10')
    for index, (name, intervals) in enumerate(sorted(spans.items())):
        ax.broken_barh([((start-.5)*dt, (last-start+1)*dt) for start, last in intervals],
                       (index-.32, .64), facecolors=colors(index % 10))
    ax.set_yticks(list(range(len(spans))), sorted(spans))
    for index, (step, name) in enumerate(anchors):
        ax.axvline(step*dt, color='black', linestyle=':', linewidth=.9)
        ax.annotate(name, (step*dt, 1), xycoords=('data', 'axes fraction'),
                    xytext=(2, 6 + 12*(index % 2)), textcoords='offset points', fontsize=8)
    ax.set_xlim(0, (end+1)*dt)
    ax.set_ylim(-.7, max(len(spans)-.3, .7))
    ax.set_xlabel('Episode time (s); bars represent inclusive control samples, not substep timing')
    scope = ('All observed episode samples' if is_stair else
             'Bars: recovery-window conditions only; blank outside windows does not mean stable\n'
             'Final failed conditions: ' + (', '.join(row['final_failed_conditions']) or 'none'))
    ax.set_title(f'{episode_id} | {outcome}\nMeasured-condition timeline (derived; not native video)\n{scope}', pad=35, fontsize=11)
    ax.grid(axis='x', color='#dddddd')
    fig.savefig(output, dpi=135)
    plt.close(fig)
    receipt = {'schema_version': 'feedback_p3.derived_timeline.v1', 'episode_id': episode_id,
        'source_full_sha256': hashlib.sha256(data).hexdigest(), 'renderer': 'render_timeline.v2',
        'sample_dt_s': dt, 'last_observed_control_step': end, 'native_media': False,
        'plot_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'limits': 'Bars are derived observations, not proof of collision, footstep or causal mechanism.'}
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--episode', required=True, help='Stair r1.e0; Push s0.e0 (logical run and environment slot).')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(render(args.full, args.episode, args.output)))
