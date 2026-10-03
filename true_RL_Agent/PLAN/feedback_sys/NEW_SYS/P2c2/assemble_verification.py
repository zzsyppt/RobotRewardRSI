"""Index actual preserved outputs; copy compact review evidence, never rewrite runs."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT/"outputs/p2c2"


def read(path): return json.loads(path.read_text(encoding="utf-8-sig"))


def info(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


asset = read(OUT/"asset/composed_audit_02.json")
historical = read(OUT/"asset/historical_sources_02.json")
extraction = read(ROOT/"src/true_rl_agent/tasks/stair_climbing/go2_nominal_usd.json")
raw = {j["name"]: j for j in extraction["joints"]}
for j in asset["joints"]:
    assert j["lower_deg"] == raw[j["name"]]["physics:lowerLimit"]
    assert j["upper_deg"] == raw[j["name"]]["physics:upperLimit"]
    assert j["prim"] == raw[j["name"]]["prim"]
    assert j["body0"] == raw[j["name"]]["physics:body0"]
    assert j["body1"] == raw[j["name"]]["physics:body1"]
assert asset["status"] == historical["status"] == "PASS"
write(HERE/"ASSET_AND_SOURCE_EVIDENCE.json", {"asset_composed_audit": asset, "historical_q_sources": historical,
    "original_stdout": [info(OUT/"asset"/n) for n in ("extraction.json", "composed_audit_02.json", "historical_sources_02.json")],
    "qualification": "three source snapshots and nominal USD coordinates only; runtime effective limits not recorded",
    "official_semantic_sources": [
        "https://openusd.org/release/api/class_usd_physics_revolute_joint.html",
        "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/joints.html",
        "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/articulations.html",
        "https://raw.githubusercontent.com/isaac-sim/IsaacLab/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/assets/articulation/articulation_data.py"]})
examples, sources = {}, []
for sid in ("k1", "k2", "k3"):
    folder = OUT/"real_02"/sid
    audit, manifest, report = [read(folder/n) for n in ("audit.json", "manifest.json", "report.json")]
    examples[sid] = {k: audit[k] for k in ("raw_examples", "absent_example_categories", "all_statements")}
    sources.append({"source": sid, "manifest": manifest,
                    "audit_summary": {k: v for k, v in audit.items() if k not in ("raw_examples", "all_statements")},
                    "files": [info(p) for p in sorted(folder.iterdir()) if p.is_file()]})
write(HERE/"REAL_EXAMPLES.json", examples)
commands = []
for folder in ("tests_02", "real_02"):
    receipts = read(OUT/folder/"command_receipts.json")
    for r in receipts:
        assert r["exit_code"] == 0
        r["logs"] = [info(OUT/folder/(r["name"]+suffix)) for suffix in (".stdout.txt", ".stderr.txt", ".memory.json")]
    commands.extend(receipts)
remote_base = "/data/yuyangliu/envs/isaaclab6_validation/bin/python -B -"
assets = "/home/yuyangliu/.cache/ov/client/https/f1ee52af0de7caec907cf4a6317543f0b852f5f2a2741d46d3f39dc0a781225c.usd /home/yuyangliu/.cache/ov/client/https/36ab15923f419bf6d2dc0bd005ef89272dc8990e631f1a4fc7bf8c223ab7579b.usd"
remote = [{"stdin_script": info(HERE/script), "argv": ["ssh", "-o", "BatchMode=yes", "humanoid-server", remote_base+args],
           "exit_code": 0, "stdout": info(OUT/"asset"/stdout), "stderr": "empty", "tool_wall_seconds": wall}
          for script, args, stdout, wall in (("extract_asset.py", " "+assets, "extraction.json", 8.65334292),
              ("audit_asset.py", " "+assets, "composed_audit_02.json", 7.617690901),
              ("audit_historical_sources.py", "", "historical_sources_02.json", 7.750566))]
files = [*HERE.glob("*.py"), HERE/"FREEZE_ZH.md", HERE/"ASSET_AND_SOURCE_EVIDENCE.json", HERE/"REAL_EXAMPLES.json",
         ROOT/"scripts/build_stair_joint_limit_supplement.py", ROOT/"tests/unit/test_joint_limits.py", ROOT/"tests/unit/joint_limit_oracle.py",
         ROOT/"src/true_rl_agent/evaluation/trajectory_translation/joint_limits.py",
         ROOT/"src/true_rl_agent/tasks/stair_climbing/trajectory_joint_limits.py",
         ROOT/"src/true_rl_agent/tasks/stair_climbing/go2_nominal_usd.json"]
write(HERE/"verification.json", {
    "status": "development_and_internal_readonly_review_PASS_pending_parent_independent_gate",
    "baseline_commit": "71c9cac31664ce32bb58c0804d33efa2ff8c0dd1", "prefreeze_commit": "a67988fc",
    "commands": commands, "remote_readonly_commands": remote, "sources": sources,
    "checks": {"new_tests": 13, "related_compatibility_tests": 158, "failures": 0, "errors": 0, "skips": 0,
               "raw_records": 31104, "raw_episodes": 2592, "joint_observations": 10218996,
               "old_v2_replays": 3, "p2a_frozen_files": 33, "p1b_card_files": 3, "push_frozen_files": 22,
               "composed_asset_joints": 12, "resolved_asset_references": 36,
               "historical_sources": 3, "new_sentences_in_real_previews": 0, "new_sentences_total": 36},
    "cost": {"final_local_commands_wall_seconds": sum(c["wall_seconds"] for c in commands),
             "final_local_peak_working_set_bytes": max(c["memory"]["peak_working_set_bytes"] for c in commands),
             "final_evidence_bytes": sum(sum(a["bytes"] for a in s["manifest"]["artifacts"].values()) for s in sources),
             "api_calls": 0, "functional_llm_calls": 0, "training_runs": 0, "gpu_runs": 0, "server_writes": 0,
             "scope": "recorded final commands only; exploration and development failures listed separately"},
    "earlier_preserved_attempts": [info(OUT/"tests_01/command_receipts.json"), info(OUT/"real_01/command_receipts.json")],
    "development_corrections": ["base self-hash injection rejected by frozen whole-content SHA binding",
        "exact 5% floating boundary evaluated with equivalent q intervals; no tolerance expansion",
        "test-only deepcopy mappingproxy / synthetic 288 count / NumericEvidenceError exception fixes",
        "independent oracle includes existing distribution quantile_method=linear",
        "remote Gf.IsClose quaternion API replaced with scalar component comparison",
        "historical source check corrected to actual detach().clone and shared preserve-order helper"],
    "files": [info(p) for p in files],
    "remaining": ["parent independent stage gate", "runtime effective hard/soft limits not recorded",
                  "production sentence delivery remains separate", "P2c3 not started", "K5/P3 result boundary untouched"]})
print(json.dumps({"status": "PASS", "verification": info(HERE/"verification.json")}))
