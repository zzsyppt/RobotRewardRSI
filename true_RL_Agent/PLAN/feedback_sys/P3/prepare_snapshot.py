"""Freeze a production-only P3 source copy for isolated remote validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / 'src'))
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    from true_rl_agent.evaluation.binding import canonical_json_hash
    snapshot = project_source_snapshot(root)
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / 'source.zip'
    entries = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative, expected_hash in snapshot['files'].items():
            path = root / relative
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected_hash:
                raise ValueError('source changed while freezing: ' + relative)
            bundle.writestr(relative, data)
            entries.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest(),
                            'bytes': len(data)})
    manifest = {'schema_version': 'feedback_p3.source_snapshot.v1',
                'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'purpose': 'immutable production source; no PLAN runtime dependency',
                'execution_source_hash': canonical_json_hash(snapshot),
                'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': entries}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(entries), 'archive_bytes': archive.stat().st_size,
                      'archive_sha256': manifest['archive_sha256']}))


if __name__ == '__main__':
    main()
