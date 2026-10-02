"""Reviewer-only native video replay of the frozen P3 Stair public-dev job (capture off, one env index)."""
import json, sys, subprocess, os
from pathlib import Path
ROOT = Path("/data/yuyangliu/AGENT/feedback_p3_20260924")
SRC = ROOT / "source_r5"
job = json.loads((ROOT / "stair_v3_reference_r5_on/public_dev_job.json").read_text())
argv = list(job["metadata"]["cli_argv"])
env_index = int(sys.argv[1]); device = sys.argv[2]; out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)
res = []
skip = False
for i, a in enumerate(argv):
    if skip: skip = False; continue
    if a in ("--numeric-evidence-dir", "--numeric-evidence-max-bytes"):
        skip = True; continue
    res.append(a)
def setopt(flag, value):
    i = res.index(flag); res[i + 1] = value
setopt("--summary-path", str(out / "evaluation/public_dev/summary.json"))
setopt("--device", device)
setopt("--resolved-cell-manifest-path", str(ROOT / "stair_v3_reference_r5_on/composition/resolved_cell_manifest.json"))
res += ["--video-dir", str(out / "video"), "--video-env-index", str(env_index)]
(out / "argv.json").write_text(json.dumps(res, indent=1))
env = dict(os.environ); env.pop("CUDA_VISIBLE_DEVICES", None)
p = subprocess.run(res, cwd=SRC, env=env)
sys.exit(p.returncode)
