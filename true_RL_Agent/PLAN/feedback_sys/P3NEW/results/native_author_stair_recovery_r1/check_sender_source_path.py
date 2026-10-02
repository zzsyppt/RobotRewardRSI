"""CPU import/spawn check of the fixed sender copy, without constructing a provider."""
from pathlib import Path
import importlib.util
import json
import multiprocessing
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
BASE = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
SOURCE = BASE / "source_observer_v3_r8"
HELPERS = BASE / "text_author_smoke_r1/PLAN/feedback_sys/P3NEW"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE / "src"))
sys.path.append(str(HELPERS))
spec = importlib.util.spec_from_file_location("sender_sourcepath_fixed", OUT / "sender_sourcepath_fixed.py")
sender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sender)


def child(queue):
    import true_rl_agent
    from true_rl_agent.evaluation.visual_observer import read_native_snapshot_media
    queue.put({"source_root": str(Path(true_rl_agent.__file__).resolve().parents[2]),
               "first_path": sys.path[0], "reader_callable": callable(read_native_snapshot_media)})


if __name__ == "__main__":
    stage = sender.D.read(HELPERS / "results/native_author_smoke_r3/stage.released.json")
    usage = sender.D.read(OUT / "recovered_usage.json")
    sender.check_usage(usage, stage, expected_count=5)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=child, args=(queue,))
    process.start()
    result = queue.get(timeout=30)
    process.join(timeout=30)
    assert process.exitcode == 0
    assert result == {"source_root": str(SOURCE), "first_path": str(SOURCE / "src"),
                      "reader_callable": True}
    result.update(status="PASS", original_history_rows_unchanged=True,
                  historical_calls=4, recovered_calls=1, api_calls=0, credential_contents_read=False)
    sender.D.write(OUT / "sourcepath_spawn_check.json", result)
    print(json.dumps(result))
