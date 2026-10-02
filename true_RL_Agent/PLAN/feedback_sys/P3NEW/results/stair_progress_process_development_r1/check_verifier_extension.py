"""Bounded extension verification; old receipt bytes and source files are read-only."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
VERIFIER = ROOT / "PLAN/feedback_sys/P3NEW/verify_stair_confirmation_arithmetic.py"
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.tasks.stair_climbing.behavior import _progress_process


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("stair_verifier", VERIFIER)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    before = json.loads((HERE / "verifier_before_extension.json").read_bytes())
    receipt = HERE.parent / "n5_arithmetic_operand_link_development_r1/stair/source_operand_link_batch.json"
    receipt_sha = sha(receipt)
    frozen = json.loads(receipt.read_bytes())
    base = HERE.parent / "n5_review_capacity_development_r2"
    episodes, mutations, errors = [], [], []
    for row in before["episodes"]:
        path = base / "prepared_report_capacity" / row["packet"] / "gpt6-sol/request.http_body.json"
        assert sha(path) == row["request_sha256"]
        raw, full = verifier.materials_from_http(path)
        old = verifier.verify_stair_episode(raw, full)
        unchanged = verifier.digest(old) == row["old_result_canonical_sha256"]
        n = raw["clock"]["last_step"]
        pos = verifier.decode_column(raw["raw_columns"]["base_pos_w"], n)
        q = verifier.decode_column(raw["raw_columns"]["base_quat_w"], n)
        w, x, y, z = q.T
        samples = np.column_stack((
            pos[:, 0] - raw["static_values"]["terrain_env_origins_w"][0],
            np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            np.arcsin(np.clip(2*(w*y-z*x), -1, 1)), 1-2*(x*x+y*y))).tolist()
        changed = deepcopy(full)
        events = verifier.decode_events(changed["complete_episode_derivation"]["events"])
        changed["complete_episode_derivation"]["progress_process"] = _progress_process(samples, events)
        checked = verifier.verify_stair_episode(raw, changed)
        progress = next(c for c in checked["checks"] if c["field"] == "progress_process")
        state = {"packet": row["packet"], "raw_episode": raw["episode"]["episode_id"],
                 "old_full_result_canonical_bytes_unchanged": unchanged,
                 "new_progress_check": progress, "new_status": checked["status"],
                 "new_checked_n": checked["checked_n"], "old_checked_n": old["checked_n"],
                 "whole_recomputed_sha256": verifier.digest(checked["recomputed_fields"])}
        episodes.append(state)
        if not unchanged or checked["failed_n"] or checked["checked_n"] != old["checked_n"]+1:
            errors.append(row["packet"])
        if row["packet"] == "episode-02":
            for name in ("omit_excursion", "wrong_recovery_null", "wrong_signed_extreme", "wrong_event_index"):
                altered = deepcopy(changed)
                excursions = altered["complete_episode_derivation"]["progress_process"]["excursions"]
                if name == "omit_excursion": excursions.pop(1)
                elif name == "wrong_recovery_null": excursions[1]["recovery_step"] = None
                elif name == "wrong_signed_extreme": excursions[1]["attitude_extrema"]["pitch_rad"][0] *= -1
                else: excursions[1]["overlapping_root_event_indices"][0] += 1
                answer = verifier.verify_stair_episode(raw, altered)
                check = next(c for c in answer["checks"] if c["field"] == "progress_process")
                mutations.append({"mutation": name, "caught": check["status"] == "fail", "check": check})
                if check["status"] != "fail": errors.append(name)
        assert sha(path) == row["request_sha256"]
    assert sha(receipt) == receipt_sha
    result = {"status": "DEVELOPMENT_EXTENSION_PASS" if not errors else "DEVELOPMENT_EXTENSION_FAIL",
        "not_n5_qualification": True, "semantic_review_vote": False,
        "old_verifier_sha256": before["verifier_sha256"], "current_verifier_sha256": sha(VERIFIER),
        "frozen_source_operand_link": {"path": str(receipt.relative_to(ROOT)), "receipt_sha256": receipt_sha,
            "registered_verifier_sha256_unchanged": frozen["current_verifier_sha256"], "receipt_bytes_unchanged": True},
        "legacy_result_comparison": "Exact canonical serialized result bytes for verify_stair_episode; CLI provenance/invocation SHA necessarily records the current script in any newly created receipt.",
        "independent_oracle": "Extended acceptance verifier imports no production progress helper. This development driver invokes production only as the subject under test.",
        "episodes": episodes, "mutation_checks": mutations, "errors": errors,
        "new_api_calls": 0, "gpu_jobs": 0, "ssh_calls": 0}
    with (HERE / "verifier_extension_check.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "episodes": len(episodes), "mutations_caught": sum(m["caught"] for m in mutations), "errors": errors, "verifier_sha256": sha(VERIFIER), "receipt_sha256": sha(HERE / "verifier_extension_check.json")}))


if __name__ == "__main__":
    main()
