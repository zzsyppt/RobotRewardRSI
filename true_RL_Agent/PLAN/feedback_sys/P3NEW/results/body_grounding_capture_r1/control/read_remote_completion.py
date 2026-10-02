"""Use one SSH connection for transfer of CPU analysis helpers and post-processing."""
from pathlib import Path
import base64
import json
import shlex
import subprocess

control = Path(__file__).resolve().parent
payload = {name: base64.b64encode((control / name).read_bytes()).decode('ascii')
           for name in ('measure_geometry.py', 'collect_completed_capture.py')}
remote = '''from pathlib import Path
import base64,json,subprocess
c=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/body_grounding_capture_r1_control')
payload=json.loads(base64.b64decode(PAYLOAD))
for name,data in payload.items():
 p=c/name;b=base64.b64decode(data)
 if p.exists():
  assert p.read_bytes()==b, 'analysis helper differs'
 else:
  with p.open('xb') as f:f.write(b)
raise SystemExit(subprocess.run(['/data/yuyangliu/envs/isaaclab6_validation/bin/python',str(c/'collect_completed_capture.py')]).returncode)
'''.replace('PAYLOAD', repr(base64.b64encode(json.dumps(payload).encode()).decode()))
command = '/data/yuyangliu/envs/isaaclab6_validation/bin/python -c ' + shlex.quote(remote)
raise SystemExit(subprocess.run(['ssh', '-o', 'ConnectTimeout=10', 'humanoid-server', command]).returncode)
