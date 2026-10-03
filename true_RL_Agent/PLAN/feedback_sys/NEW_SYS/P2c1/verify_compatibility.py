"""Read-only old artifacts plus normal v2 replay, no historical fact regrading."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"src"))
from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


if __name__ == "__main__":
    checked = []
    for source in read(Path(__file__).parents[1]/"P2a/verification.json")["sources"]:
        for info in source["files"]:
            path = ROOT/info["path"]
            assert sha(path) == info["sha256"] and path.stat().st_size == info["bytes"]
            checked.append(info)
    replay = []
    for sid in ("k1","k2","k3"):
        directory = ROOT/"outputs/p2a"/sid
        public = read(directory/"public.json")
        assert project_stair_behavior(read(directory/"behavior_full.json")) == public
        assert project_stair_behavior(public) == public
        replay.append({"source": sid,"full_and_public_replay_equal": True,"public_sha256":sha(directory/"public.json")})
    push = read(Path(__file__).parents[1]/"P2b1/RULES_CODE_FREEZE.json")
    assert len(push["files_sha256"]) == 22
    for path, digest in push["files_sha256"].items():
        assert sha(ROOT/path) == digest
    frozen = read(Path(__file__).parents[1]/"P1b/verification.json")["artifacts"]
    for name in ("evidence.jsonl","policy_card.json","policy_card_zh.md"):
        assert sha(ROOT/frozen[name]["path"]) == frozen[name]["sha256"]
    print(json.dumps({"status":"PASS","p2a_artifact_count":len(checked),"p2a_artifacts":checked,
        "v2_replays":replay,"push_frozen_files":22,"p1b_frozen_card_files":3,
        "scope":"v2 content SHA and actual full/public replay; NOT rerun of historical 247/309 fact gates"},indent=2))
