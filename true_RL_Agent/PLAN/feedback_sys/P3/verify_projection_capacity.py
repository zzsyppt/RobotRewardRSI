"""Check current bounded views against the already independently audited full views."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--derived', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.root / 'src'))
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior
    from true_rl_agent.tasks.push_recovery.behavior import project_push_behavior

    inventory = json.loads((args.derived / 'inventory.json').read_text())
    args.output.mkdir(exist_ok=False, parents=True)
    checks = {}
    for name, item in inventory.items():
        raw = (args.derived / name / 'full.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['full']['sha256'], name
        full = json.loads(raw)
        stair = item['task'] == 'stair'
        project = project_stair_behavior if stair else project_push_behavior
        view = project(full)
        assert project(view) == view, name
        assert_public_projection_safe(view)
        rows = view['transcripts' if stair else 'representative_transcripts']
        label = 'primary_reason' if stair else 'terminal_label'
        success = 'success' if stair else 'success_recovered'
        originals = full['episodes' if stair else 'transcripts']
        has_success = any(row[label] == success for row in originals)
        has_failure = any(row[label] not in {success, 'no_push_control'} for row in originals)
        assert not has_success or any(row[label] == success for row in rows), name
        assert not has_failure or any(row[label] not in {success, 'no_push_control'} for row in rows), name
        if stair:
            population = sum(row[2] for row in view['profile']['rows'])
        else:
            profile = view['profile']
            population = next(dict(zip(profile['columns'], row))['episodes']
                              for row in profile['rows'] if row[:2] == ['all', 'all'])
        assert population == item['episodes'], name
        data = json.dumps(view, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        assert len(data) <= (8192 if stair else 12288), name
        target = args.output / (name + '.json')
        target.write_bytes(data)
        checks[name] = {'status': 'PASS', 'bytes': len(data), 'population': population,
            'representatives': len(rows), 'labels': [row[label] for row in rows],
            'full_sha256': item['full']['sha256'], 'public_sha256': hashlib.sha256(data).hexdigest()}
    result = {'status': 'PASS', 'derived_at_unix': time.time(),
        'projection_execution_source_hash': canonical_json_hash(project_source_snapshot(args.root)),
        'scope': 'new projection of existing audited full views; no receipt rewritten or raw rollout rerun',
        'datasets': checks}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
