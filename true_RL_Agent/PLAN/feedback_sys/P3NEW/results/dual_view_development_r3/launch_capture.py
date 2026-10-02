from pathlib import Path
import hashlib, json, os, subprocess, sys
card_path=Path(__file__).with_name("capture_card.json")
if hashlib.sha256(card_path.read_bytes()).hexdigest()!=sys.argv[3]:
    raise SystemExit("capture card hash mismatch")
card=json.loads(card_path.read_text())
env=dict(os.environ, PYTHONPATH=card["source_root"]+"/src")
raise SystemExit(subprocess.run(card["commands"][sys.argv[1]][sys.argv[2]], env=env).returncode)
