"""Upload an exclusive r4 snapshot and retrieve static-only CPU preflight receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REMOTE = '/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_preflight_r4_inputs'


def run(name, argv, **kwargs):
    result = subprocess.run(argv, capture_output=True, **kwargs)
    (HERE / (name + '.stdout.txt')).write_bytes(result.stdout)
    (HERE / (name + '.stderr.txt')).write_bytes(result.stderr)
    (HERE / (name + '.command.json')).write_text(json.dumps(dict(argv=argv, returncode=result.returncode), indent=2) + '\n', encoding='utf-8')
    if result.returncode:
        raise RuntimeError(name + ' failed: ' + result.stderr.decode(errors='replace')[:1000])
    return result


def main():
    bootstrap = "from pathlib import Path\np=Path('" + REMOTE + "')\nassert not p.exists()\nassert not p.with_name('source_n5_preflight_r4').exists()\nassert not p.with_name('n5_preflight_r4').exists()\np.mkdir()\nprint('exclusive r4 input directory created')\n"
    run('remote_stage', ['ssh', 'humanoid-server', 'python3 -'], input=bootstrap.encode(), timeout=40)
    names = ('source.tar.gz', 'source_manifest.json', 'preflight_card.json', 'remote_static_preflight.py', 'inherited_r3_evidence.json')
    run('upload', ['scp', *[str(HERE / n) for n in names], 'humanoid-server:' + REMOTE + '/'], timeout=120)
    executed = run('remote_execute', ['ssh', 'humanoid-server', 'python3 ' + REMOTE + '/remote_static_preflight.py'], timeout=480)
    small = json.loads(executed.stdout)
    run('download', ['scp', 'humanoid-server:' + REMOTE + '/remote_results.tar', str(HERE / 'remote_results.tar')], timeout=120)
    assert hashlib.sha256((HERE / 'remote_results.tar').read_bytes()).hexdigest() == small['results_tar']['sha256']
    target = HERE / 'remote_preflight'
    target.mkdir(exist_ok=False)
    with tarfile.open(HERE / 'remote_results.tar') as archive:
        for entry in archive.getmembers():
            assert entry.isfile() and not entry.name.startswith('/') and '..' not in Path(entry.name).parts
            # Materialized reference artifact paths exceed Windows MAX_PATH.
            root = Path('\\\\?\\' + str(target)) if os.name == 'nt' else target
            p = root / entry.name
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as stream:
                stream.write(archive.extractfile(entry).read())
    print(json.dumps(small))


if __name__ == '__main__':
    main()
