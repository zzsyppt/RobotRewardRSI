"""Preserve one read-only SSH stdout and its exact command/cost locally."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

script, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=False)
command = ['ssh', '-o', 'BatchMode=yes', 'humanoid-server',
           '/data/yuyangliu/envs/isaaclab6_validation/bin/python -B -']
started = time.perf_counter()
with (output/'stdout.json').open('wb') as out, (output/'stderr.txt').open('wb') as err:
    result = subprocess.run(command, input=script.read_bytes(), stdout=out, stderr=err)
receipt = {'argv': command, 'stdin_script': str(script),
           'stdin_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
           'exit_code': result.returncode, 'wall_seconds': time.perf_counter()-started,
           'files': {n: {'bytes': (output/n).stat().st_size,
                        'sha256': hashlib.sha256((output/n).read_bytes()).hexdigest()}
                     for n in ('stdout.json', 'stderr.txt')}}
(output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
raise SystemExit(result.returncode)
