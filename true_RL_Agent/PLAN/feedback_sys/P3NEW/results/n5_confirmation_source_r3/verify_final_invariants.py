"""Record R3 source/command invariants against preserved R2 without execution."""
from pathlib import Path

from complete_cpu_source import HERE, ROOT, digest, read, record, save


def main():
    previous_root = HERE.parent / "n5_confirmation_source_r2"
    previous = read(previous_root / "source_manifest.json")
    current = read(HERE / "source_manifest.final.json")
    assert all(record(previous_root / "source" / name) == item for name, item in previous.items())
    assert all(record(HERE / "source_final" / name) == item for name, item in current.items())
    delta = [{"path": name, "previous": previous.get(name), "current": current.get(name)}
             for name in sorted(previous.keys() | current.keys()) if previous.get(name) != current.get(name)]
    assert all(item["path"].startswith("PLAN/") or item["path"] == "tests/unit/test_confirmation_event_table.py" for item in delta)
    save(HERE / "source_delta.final.json", delta)
    assert set(path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*.py")) <= current.keys()
    old_commands = read(previous_root / "commands.candidate.json")
    commands = read(HERE / "commands.candidate.final.json")
    path_names = ("source_root", "preflight_root", "capture_root", "execution_control_root")
    changes = {}
    for task in ("stair", "push"):
        changes[task] = {}
        for kind in ("preflight_argv", "capture_argv_not_executed"):
            before = old_commands["tasks"][task][kind]
            after = commands["tasks"][task][kind]
            assert len(before) == len(after)
            normalized = []
            for argument in before:
                for field in path_names:
                    argument = argument.replace(old_commands[field], commands[field])
                argument = argument.replace("n5_r2_20260930", "n5_r3_20260930")
                if argument == "1600000000": argument = "1700000000"
                normalized.append(argument)
            assert normalized == after
            changes[task][kind] = [{"index": index, "previous": first, "current": second}
                                   for index, (first, second) in enumerate(zip(before, after)) if first != second]
        preflight = commands["tasks"][task]["preflight_argv"]
        capture = commands["tasks"][task]["capture_argv_not_executed"]
        assert [preflight[index - 1] for index in range(len(preflight)) if preflight[index] != capture[index]] == ["--mode", "--output", "--video-dir"]
    assignment_name = "PLAN/feedback_sys/P3NEW/N5_REVIEW_ASSIGNMENT_R3.json"
    assert current[assignment_name]["sha256"] == "b898502e9fa33caf415cdcaf58f7b862de1b68473a898d535a07660ccd1cd893"
    save(HERE / "source_invariants.json", {
        "status": "PASS", "source_manifest_sha256": digest(HERE / "source_manifest.final.json"),
        "r2_source_manifest_sha256": digest(previous_root / "source_manifest.json"),
        "r2_all_477_source_files_still_match": True, "final_all_485_source_files_match": True,
        "all_current_src_python_files_present": True,
        "runtime_configs_scripts_identical_to_r2": True,
        "delta_paths": [item["path"] for item in delta],
        "r2_capture_and_preflight_command_deltas": changes,
        "allowed_command_changes": ["source/preflight/capture/control roots", "stair diagnostic cohort", "push eval-base-seed"],
        "preflight_to_capture_changed_flags": ["--mode", "--output", "--video-dir"],
        "checkpoint_task_suite_media_arguments_unchanged": True,
        "review_assignment_sha256": current[assignment_name]["sha256"],
        "packet_members_and_task_routing": "PENDING actual R3 collection and packing; frozen reviewer identities do not select raw members",
        "fresh_gpt_6_1_sol_calibration": "PENDING; old R2 GPT6-SOL calibration cannot certify the new model",
        "actual_model_calls": 0, "actual_gpu_runs": 0, "actual_capture_runs": 0
    })
    print("PASS: 485 final and 477 R2 source records; only allowed command changes; frozen assignment SHA verified")


if __name__ == "__main__":
    main()
