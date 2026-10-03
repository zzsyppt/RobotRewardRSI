"""Current vendor evidence only: no git metadata means no historical byte claim."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

start = time.perf_counter()
root = Path('/data/yuyangliu/vendor/IsaacLab')
base = root/'source/isaaclab/isaaclab'
names = ['terrains/terrain_generator.py', 'terrains/terrain_importer.py', 'terrains/utils.py',
         'sim/schemas/schemas.py', 'sim/schemas/schemas_cfg.py', 'sim/utils.py']
git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
print(json.dumps({'scope': 'current_vendor_only_historical_runtime_revision_not_verified',
    'git': {'exit_code': git.returncode, 'stdout': git.stdout, 'stderr': git.stderr},
    'files': [{'path': str(base/n), 'sha256': hashlib.sha256((base/n).read_bytes()).hexdigest(),
               'text': (base/n).read_text()} for n in names],
    'server_writes': 0, 'wall_seconds': time.perf_counter()-start}))
