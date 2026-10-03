"""Persist actual command output/costs using the existing P2b1 CPU wrapper."""
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
spec = importlib.util.spec_from_file_location("p2b1_commands", ROOT / "PLAN/feedback_sys/NEW_SYS/P2b1/run_final.py")
commands = importlib.util.module_from_spec(spec)
spec.loader.exec_module(commands)
commands.OUT = ROOT / "outputs/p2b2/final"
commands.OUT.mkdir(parents=True, exist_ok=True)


def cli_fixture():
    """Persist labelled synthetic admission inputs; invoke the actual CLI next."""
    from tests.unit.test_push_checkpoint_admission import PushCheckpointAdmissionTests
    from true_rl_agent.runner.checkpoint import file_sha256
    fixture = PushCheckpointAdmissionTests()
    fixture.setUp()
    try:
        source = commands.OUT / "cli_fixture"
        source.mkdir(exist_ok=True)
        (source / "resolved_cell_manifest.json").write_bytes(
            (fixture.root / "source/resolved_cell_manifest.json").read_bytes())
        checkpoint = source / "synthetic.pt"
        checkpoint.write_bytes(b"synthetic CPU admission fixture; never trained or executed")
        fixture.trained["checkpoint"]["checksum"] = file_sha256(checkpoint)
        (source / "source_train.json").write_text(json.dumps(fixture.trained) + "\n", encoding="utf-8")
        return ["scripts/push_numeric_replay.py", "--cell", "push.go2_12dof.recovery_v3",
            "--source-manifest", str(source / "resolved_cell_manifest.json"),
            "--checkpoint-summary", str(source / "source_train.json"), "--checkpoint", str(checkpoint),
            "--device", "cuda:1", "--output", str(commands.OUT / "cli_preflight"),
            "--mode", "preflight", "--capture", "on", "--behavior-feedback-profile", "trajectory_numeric_v1",
            "--trajectory-report-max-bytes", "12288"]
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    receipts = []
    modules = sorted("tests.unit." + p.stem for p in (ROOT / "tests/unit").glob("test_push*.py"))
    modules += ["tests.unit." + name for name in (
        "test_stair_trajectory_pipeline", "test_stair_trajectory_report", "test_stair_numeric_dev_replay",
        "test_stair_phased_evaluation", "test_stair_trajectory_production", "test_gpu_binding_projection_contract",
        "test_phase_postprocessing", "test_composition")]
    modules += ["tests.integration.test_generation_session"]
    jobs = [("targeted", ["-m", "unittest", "tests.unit.test_push_trajectory_pipeline", "tests.unit.test_push_numeric_replay", "tests.unit.test_push_phased_evaluation", "-v"]),
        ("regression", ["-m", "unittest", *modules, "-v"]),
        ("lifecycle", ["PLAN/feedback_sys/NEW_SYS/P2b2/capture_lifecycle.py", "--output", "outputs/p2b2/final_run"]),
        ("protocols", ["PLAN/feedback_sys/NEW_SYS/P2b2/audit_protocols.py"]),
        ("stair_frozen", ["PLAN/feedback_sys/NEW_SYS/P2b1/check_stair_compat.py"]),
        ("cli_help", ["scripts/push_numeric_replay.py", "--help"]),
        ("cli_preflight", cli_fixture())]
    for name, argv in jobs:
        with (commands.OUT / (name + ".wrapper.txt")).open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            receipt = commands.run(name, argv)
        receipts.append(receipt)
        (commands.OUT / "command_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
        print(name, "exit", receipt["exit_code"], "seconds", round(receipt["wall_seconds"], 3), flush=True)
    raise SystemExit(int(any(receipt["exit_code"] for receipt in receipts)))
