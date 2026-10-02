"""CPU capacity fixture over a real public shape; never a measured parent diff."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "src"))
    from true_rl_agent.tasks.push_recovery.behavior import compare_push_behavior, project_push_behavior
    from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes

    original = json.loads(args.input.read_text(encoding="utf-8"))
    assert project_push_behavior(original) == original
    parent = deepcopy(original)
    parent["binding"]["checkpoint_hash"] = "sha256:" + "1" * 64
    table = parent["profile"]
    row = next(row for row in table["rows"] if row[:2] == ["all", "all"])
    for key in ("dwell_interruptions", "zero_stable_units", "all_recovered_then_final_unstable", "any_instability_after_all_recoveries"):
        row[table["columns"].index(key)] = 123
    labels = row[table["columns"].index("terminal_counts")]
    labels["fall_backward"] = 20
    labels["success_recovered"] -= 20
    with_diff = deepcopy(original)
    with_diff["parent_diff"] = compare_push_behavior(original, parent,
        expected_parent_checkpoint_hash=parent["binding"]["checkpoint_hash"])
    assert with_diff["parent_diff"]["status"] == "available"
    assert canonical_prompt_json_bytes(with_diff) > 12288
    compact = project_push_behavior(with_diff)
    assert canonical_prompt_json_bytes(compact) <= 12288
    assert compact["profile"] == original["profile"]
    assert compact["parent_diff"] == with_diff["parent_diff"]
    assert compact["selection"] == original["selection"]
    assert len(compact["representative_transcripts"]) == 2
    omitted = []
    for before, after in zip(original["representative_transcripts"], compact["representative_transcripts"]):
        expected = deepcopy(before)
        if "final_pose_omitted" in after:
            expected.pop("final_pose")
            expected["final_pose_omitted"] = "public_byte_budget"
            omitted.append(before["terminal_label"])
        assert after == expected
    assert project_push_behavior(compact) == compact
    assert json.loads(args.input.read_text(encoding="utf-8")) == original
    report = {"status": "PASS", "scope": "synthetic parent counts over a real public shape; not an observed policy comparison",
        "source": str(args.input), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "original_bytes": canonical_prompt_json_bytes(original), "with_diff_before_bytes": canonical_prompt_json_bytes(with_diff),
        "with_diff_after_bytes": canonical_prompt_json_bytes(compact), "max_bytes": 12288,
        "optional_final_pose_omitted_for_labels": omitted, "representatives_retained": 2,
        "original_view_unchanged": True, "complete_profile_and_diff_unchanged": True,
        "all_recovery_windows_and_final_predicates_unchanged": True, "projection_idempotent": True}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
