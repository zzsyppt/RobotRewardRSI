"""One-time conversion of the frozen G1-29 URDF to a runtime-independent USD.

Run with the original Isaac Lab importer. Production only reads the resulting
asset from the target vendor port; it does not invoke the old runtime.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert importlib.metadata.version("isaacsim") == "5.1.0.0"
app = AppLauncher(headless=True, device="cuda:0", multi_gpu=False).app
try:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

    spawn = UNITREE_G1_29DOF_CFG.spawn
    converter = UrdfConverterCfg(
        **{k: getattr(spawn, k) for k in UrdfConverterCfg.__dataclass_fields__}
    )
    converter.usd_dir = str(args.output.resolve())
    converter.force_usd_conversion = True
    result = UrdfConverter(converter)
    files = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    }
    urdf = Path(spawn.asset_path)
    report = {
        "source_urdf": str(urdf),
        "source_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "importer_isaacsim": importlib.metadata.version("isaacsim"),
        "importer_isaaclab": importlib.metadata.version("isaaclab"),
        "usd_path": str(result.usd_path),
        "files": files,
    }
    (args.output / "asset_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)
finally:
    app.close()
