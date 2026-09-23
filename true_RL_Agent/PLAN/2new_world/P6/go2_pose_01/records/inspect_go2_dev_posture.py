"""D-only posture replay; reuses production recovery and existing native helpers."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")

dev = module("matched_dev", "train_stationary_reference_dev.py")
native = module("native_dev", "push_native_dev_replay.py")
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.runner.eval_binding import bind_eval_spec_to_checkpoint
from true_rl_agent.runner.subprocess_runner import SubprocessRunner

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("preflight", "run"), required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
source = ROOT / "outputs/p6_go2_matched_dev_01"
output = a.output.resolve()
output.mkdir(parents=True, exist_ok=False)
started = time.time()
record = read(source / "result.json")
assert record["status"] == "PASS" and record["phases_executed"] == ["public_dev"]
provider = dev.PushRecoveryCompositionProvider()
composition = provider.resolve(values={**provider.cli_defaults(), **dev.BUDGET,
    "cell_variant_id": record["cell"], "execution_device": "cuda:0",
    "isaac_python": sys.executable, "reference_program_id": dev.REFERENCE_ID,
    "push_protocol_path": str(Path(__file__).parent / "calibration_protocols_01/scale_2p0.json")},
    context=TaskCompositionContext(project_root=ROOT, output_dir=source / "composition", read_only=True))
environment = read(source / "backend_execution_environment.json")
composition.backend_adapter.validate_session_execution_environment(execution_environment=environment)
jobs = dev.materialize(composition, composition.reference_program_artifact, output=output,
    session_id=record["session_id"], environment_hash=environment["manifest_hash"])
# This production method only reads completed training; no run_train invocation.
trained = composition.runner.recover_completed_train(jobs.train_spec)
plan = bind_eval_spec_to_checkpoint(jobs.eval_spec, trained.checkpoint)
original = read(source / "public_dev_job.json")
assert plan.to_public_dict() == original["spec"]["evaluation_plan"]
raw = read(source / "public_dev_result.json")
expected = {"phase_receipt": {"diagnostic_result_path": str(source / "public_dev_result.json"),
                             "sha256": dev.file_sha256(source / "public_dev_result.json")},
            "phase_result_hash": dev.canonical_json_hash(raw),
            "episodes": [s["per_episode"] for s in raw["task_evidence"]["seed_summaries"]]}
spec = native.video_job(composition, plan, output=output, env_index=32)
write(output / "native_job.json", dev.job_record(spec))
report = {"mode": a.mode, "purpose": "D posture inspection only; not qualification",
          "train_jobs": 0, "holdout_jobs": 0, "retries": 0, "provider_calls": 0,
          "selected_env_index": 32, "checkpoint_sha256": trained.checkpoint.checksum,
          "source_result_sha256": expected["phase_receipt"]["sha256"], "started_at_unix": started}
if a.mode == "run":
    result = SubprocessRunner(allowed_executables=(sys.executable,), cwd=ROOT, timeout_s=900).run_eval_phase(spec)
    write(output / "public_dev_result.json", result.to_private_dict())
    recordings = read(output / "video/recordings.json")
    comparison = native.compare_dev_replay(expected, result, recordings["recordings"], env_index=32, base_seed=42)
    write(output / "comparison.json", comparison)
    assert comparison["status"] == "PASS"
    report["status"] = "PASS"
else:
    report["status"] = "PREPARED"
report["ended_at_unix"] = time.time()
write(output / "result.json", report)
print(json.dumps(report))
