#!/usr/bin/env python3
"""Run a bounded training capture check from a normally materialized train argv.

The input is a JSON argv list, or an object with metadata.cli_argv.
Stair validates the original production invocation then uses its existing
128-transition smoke executor. Push must be materialized with a fresh diagnostic
16 x 8 x 3 = 384 transition contract. Neither result qualifies a policy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.argv_contract import option_occurrences, unique_option_value
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.gpu_lease import GpuLeaseManager, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.process_handle import run_owned_process
from true_rl_agent.runner.resource_shadow import discover_nvidia_gpu_inventory


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _set_option(argv: list[str], name: str, value: str) -> None:
    positions = option_occurrences(argv, name)
    if positions:
        unique_option_value(argv, name)  # Reject duplicate or inline spellings.
        argv[positions[0] + 1] = value
    else:
        argv.extend((name, value))


def prepare_argv(source, *, output: Path, capture: bool) -> tuple[str, list[str]]:
    argv = source if isinstance(source, list) else source.get("metadata", {}).get("cli_argv")
    if not isinstance(argv, list) or len(argv) < 3 or any(not isinstance(v, str) for v in argv):
        raise ValueError("input must contain a materialized train argv list")
    argv = list(argv)
    entries = {"stair": ROOT / "scripts/isaaclab_stair_rsl_rl_train.py",
               "push": ROOT / "scripts/isaaclab_rsl_rl_train.py"}
    selected = next((name for name, path in entries.items() if Path(argv[1]).resolve() == path.resolve()), None)
    if selected is None:
        raise ValueError("only the current production Stair/Push train entrypoints are supported")
    if unique_option_value(argv, "--subject-kind") not in {"candidate", "reference"}:
        raise ValueError("training smoke requires a candidate or reference")
    if unique_option_value(argv, "--init") != "random":
        raise ValueError("this bounded smoke validates random initialization only")
    if selected == "push":
        expected = {"--num-envs": "16", "--num-steps-per-env": "8", "--max-iterations": "3", "--budget-steps": "384"}
        if any(unique_option_value(argv, key) != value for key, value in expected.items()):
            raise ValueError("Push requires a newly materialized 16 x 8 x 3 diagnostic contract")
    _set_option(argv, "--summary-path", str(output / "training_summary.json"))
    _set_option(argv, "--checkpoint-output", str(output / "smoke_checkpoint.pt"))
    if selected == "push":
        _set_option(argv, "--run-dir", str(output / "run"))
    positions = option_occurrences(argv, "--capture-training-numeric")
    if len(positions) > 1 or any(argv[p] != "--capture-training-numeric" for p in positions):
        raise ValueError("invalid capture flag")
    if positions:
        argv.pop(positions[0])
    if capture:
        argv.append("--capture-training-numeric")
    return selected, argv


def _stair_child(argv: list[str]) -> int:
    """Reuse production identity/admission checks; no replacement train loop."""
    spec = importlib.util.spec_from_file_location("stair_train_entrypoint", ROOT / "scripts/isaaclab_stair_rsl_rl_train.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    args = entry._build_parser().parse_args(argv[2:])
    args.device = entry.require_execution_device(args.device, "smoke execution device")
    binding = entry.validate_gpu_process_binding_invocation(args, environment=os.environ)
    if binding is None or not args.headless:
        raise ValueError("training smoke child must use the parent's GPU binding and headless mode")
    args.resolved_cell_manifest = entry.load_optional_resolved_cell_manifest_from_args(
        args, expected_task_id=args.task_id, expected_base_task_protocol_hash=args.base_task_protocol_hash)
    context = entry.validate_stair_train_invocation(args, project_root=ROOT, resolved_cell_manifest=args.resolved_cell_manifest)
    preflight = entry.preflight_stair_execution_source_manifest(context.config)
    environment = entry.verify_current_isaac_lab_execution_environment(args.backend_execution_environment_manifest_hash)
    program = entry._load_and_qualify_program(context)
    from true_rl_agent.tasks.stair_climbing.training_diagnostics import load_stair_training_comparison
    from true_rl_agent.tasks.stair_climbing.isaaclab_execution import execute_stair_training_smoke
    from isaaclab.app import AppLauncher
    comparison = load_stair_training_comparison(args.runtime_comparison_json, args.runtime_comparison_hash,
                                               context=context, candidate_program=program)
    summary_path = Path(args.summary_path)
    entry.bind_isaac_log_path(args, summary_path)
    app = AppLauncher(args, multi_gpu=False).app
    result = execute_stair_training_smoke(
        config=context.config, source_manifest_preflight=preflight,
        agent_configuration=context.agent_configuration, reward_program=program,
        device=args.device, run_dir=summary_path.parent / "run", checkpoint_output=Path(args.checkpoint_output),
        run_name="training_numeric_smoke", resolved_cell_manifest=args.resolved_cell_manifest,
        runtime_comparison=comparison, capture_training_numeric=args.capture_training_numeric,
        numeric_evidence_identity={"job": context.identity.to_public_dict(),
            "training_binding_hash": context.training_binding.binding_hash,
            "backend_execution_environment_verification": environment,
            "diagnostic_smoke": True},
    )
    fields = result.to_summary_fields()
    fields["validation_training_executed"] = fields.pop("training_executed")
    fields.update(schema_version="true_rl_agent.training_numeric_smoke.stair.v1",
                  formal_training_executed=False, policy_qualification=False,
                  original_training_job_identity=context.identity.to_public_dict(),
                  reward_program=program.to_public_dict(), environment_source_preflight=preflight.to_summary_dict(),
                  backend_execution_environment_verification=environment)
    observation = entry.capture_gpu_compute_context(binding)
    fields = entry.attach_gpu_runtime_evidence(fields, binding, compute_context_observation=observation)
    _write(summary_path, fields)
    code = 0 if fields["status"] == "pass" else 1
    if code == 0:
        entry.require_physx_success(summary_path.with_suffix(".kit.log"))
    return entry.close_isaac_app_on_success(app, exit_code=code)


def expected_training_numeric_identity(summary: dict, *, task: str, argv: list[str]) -> dict:
    """Bind an archive to the executed invocation and its returned summary."""
    job = {name: unique_option_value(argv, "--" + name.replace("_", "-"))
           for name in ("job_id", "task_id", "subject_kind", "subject_id")}
    if option_occurrences(argv, "--subject-manifest-hash"):
        job["subject_manifest_hash"] = unique_option_value(argv, "--subject-manifest-hash")
    actual_job = summary["original_training_job_identity"] if task == "stair" else {
        key: summary.get(key) for key in job}
    if actual_job != job:
        raise ValueError("training summary job differs from executed invocation")
    checksum = unique_option_value(argv, "--program-source-checksum")
    initial = summary["training"]["initial_policy_state_hash"]
    if not isinstance(initial, str) or not initial:
        raise ValueError("training summary lacks its initial policy identity")
    expected = {"task": {"push": "push_recovery", "stair": "stair_climbing"}[task], "job": job,
                "training_binding_hash": unique_option_value(argv, "--training-binding-hash"),
                "initial_policy_state_hash": initial,
                "task_protocol_hash": unique_option_value(argv, "--task-protocol-hash"),
                "training_seed": int(unique_option_value(argv, "--seed")),
                "backend_execution_environment_verification": summary["backend_execution_environment_verification"]}
    if task == "push":
        if summary["program_source_checksum"] != checksum or summary["training_binding_hash"] != expected["training_binding_hash"]:
            raise ValueError("training summary program or binding differs from executed invocation")
        expected.update(program_source_checksum=checksum, train_attempt_id=summary["run"]["train_attempt_id"],
                        run_dir=str(Path(unique_option_value(argv, "--run-dir")) /
                                    ("attempt-" + summary["run"]["train_attempt_id"])))
    else:
        if summary["reward_program"]["program_source_checksum"] != checksum:
            raise ValueError("training summary program differs from executed invocation")
        expected.update(program=summary["reward_program"], diagnostic_smoke=True,
                        run_dir=str(Path(unique_option_value(argv, "--summary-path")).parent / "run"))
        if option_occurrences(argv, "--resolved-cell-manifest-path") or option_occurrences(argv, "--resolved-cell-manifest-hash"):
            from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
            manifest = load_resolved_cell_manifest(
                unique_option_value(argv, "--resolved-cell-manifest-path"),
                expected_manifest_hash=unique_option_value(argv, "--resolved-cell-manifest-hash"),
                expected_task_id=job["task_id"],
                expected_base_task_protocol_hash=unique_option_value(argv, "--base-task-protocol-hash"))
            # The train argv protocol belongs to TrainingBinding (possibly a
            # legal ancestor); numeric observations identify the current cell.
            # Keep the training binding check above and verify both identities.
            expected.update(task_protocol_hash=manifest.resolved_cell_task_protocol_hash,
                            resolved_cell_manifest_hash=manifest.manifest_hash,
                            resolved_cell_identity=manifest.summary_identity())
            preflight = summary["environment_source_preflight"]
            if preflight.get("task_protocol_hash") != manifest.resolved_cell_task_protocol_hash:
                raise ValueError("training source preflight differs from resolved execution protocol")
            expected["environment_source_preflight"] = preflight
    return expected


def check_sampled_rewards(evidence: dict, *, task: str, expected_identity: dict | None = None,
                          require_all_windows: bool = False) -> dict:
    """Offline arithmetic checks of saved observations, not reward re-execution."""
    import numpy as np
    from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
    manifest = load_numeric_evidence_manifest(evidence["reference"])
    coverage = manifest["metadata"]
    if evidence.get("status") != "complete" or evidence["coverage"] != coverage:
        raise ValueError("training coverage differs from bound numeric manifest")
    if expected_identity is not None and any(manifest["identity"].get(k) != v for k, v in expected_identity.items()):
        raise ValueError("training numeric identity differs from expected execution")
    terms = coverage["semantics"]["terms"]
    sampled_inputs = coverage["semantics"].get("sampled_inputs", {})
    if set(sampled_inputs) not in (set(), {"last_action", "prev_action", "joint_vel", "joint_torque"}):
        raise ValueError("training sampled input declaration differs")
    if [term["index"] for term in terms] != list(range(len(terms))):
        raise ValueError("training term indices are not consecutive")
    windows = coverage["windows"]
    schema = coverage["schema_version"]
    if schema not in {"true_rl_agent.training_numeric_capture.v1", "true_rl_agent.training_numeric_capture.v2"}:
        raise ValueError("unknown training sampling schema")
    if (schema == "true_rl_agent.training_numeric_capture.v2"
            and coverage.get("sampling_strategy") != "step_one_anchor_then_equal_bin_centers_within_disjoint_training_thirds"):
        raise ValueError("training sampling strategy differs from its schema")
    if [w["name"] for w in windows] != ["early", "middle", "late"]:
        raise ValueError("training capture must declare three ordered windows")
    env_ids = np.asarray(coverage["sampled_env_ids"], dtype=np.int64)
    if not 0 < len(env_ids) <= min(16, coverage["num_envs"]) or not np.array_equal(env_ids, np.arange(len(env_ids))):
        raise ValueError("training sampled environment denominator differs")
    observed, planned = coverage["observed_control_steps"], coverage["planned_control_steps"]
    if not 0 <= observed <= planned or coverage["steps_per_iteration"] <= 0:
        raise ValueError("training observed control-step denominator differs")
    actual_steps = {w["name"]: [] for w in windows}
    saved_index = None
    ordered_steps = []
    sample_count = chunks = 0
    for metadata, arrays in iter_numeric_evidence(evidence["reference"]):
        chunks += 1
        if metadata["kind"] == "sampling_index":
            if saved_index is not None or set(arrays) != {"sampled_control_step"}:
                raise ValueError("training sampling index is duplicated or malformed")
            saved_index = arrays["sampled_control_step"]
            continue
        if metadata["kind"] != "training_sample" or saved_index is not None:
            raise ValueError("training sample kind or index order differs")
        step = metadata["training_control_step"]
        window = next((w for w in windows if w["name"] == metadata["window"]), None)
        if (window is None or not window["start_step"] <= step <= window["end_step"]
                or not 1 <= step <= observed or (ordered_steps and step <= ordered_steps[-1])):
            raise ValueError("training sample step or window differs")
        local_iteration, within = divmod(step - 1, coverage["steps_per_iteration"])
        expected_clock = {"source_control_step": step + coverage["source_step_offset_excluded"],
                          "local_rollout_iteration": local_iteration, "step_in_rollout": within + 1,
                          "learning_iteration_from_control_steps": coverage["initial_learning_iteration"] + local_iteration}
        if any(metadata.get(k) != v for k, v in expected_clock.items()):
            raise ValueError("training sample clock differs from excluded preflight offset")
        if (not set(sampled_inputs).issubset(arrays)
                or not np.array_equal(arrays["env_id"], env_ids)
                or any(a.shape != (len(env_ids), *sampled_inputs.get(name, {}).get("trailing_shape", []))
                       for name, a in arrays.items())):
            raise ValueError("training sample environment axis differs")
        ordered_steps.append(step)
        actual_steps[window["name"]].append(step)
        total = None
        for term in terms:
            raw = arrays[f"raw_term_{term['index']:03d}"]
            weighted = arrays[f"weighted_term_{term['index']:03d}"]
            scale = metadata["task"]["reward_weight_scale"] if task == "push" else 1.0
            expected = raw.astype(weighted.dtype) * (term["weight"] * scale)
            if not np.allclose(expected, weighted, rtol=1e-6, atol=1e-6):
                raise ValueError("saved raw and weighted training terms disagree")
            total = weighted.copy() if total is None else total + weighted
        if total is not None:
            actual = arrays["total_rate"] if task == "stair" else arrays["candidate_total"]
            if not np.allclose(total, actual, rtol=1e-6, atol=1e-6):
                raise ValueError("saved weighted training terms do not sum to actual total")
        if task == "push":
            expected = arrays["input_reward"] if total is None else arrays["input_reward"] + arrays["candidate_total"]
            if not np.allclose(expected, arrays["returned_reward"], rtol=1e-6, atol=1e-6):
                raise ValueError("saved Push returned reward differs from input plus candidate")
        sample_count += 1
    if saved_index is None or not np.array_equal(saved_index, np.asarray(ordered_steps, dtype=np.int64)):
        raise ValueError("training sampling index differs from actual samples")
    previous_end = 0
    edges = (0, planned // 3, 2 * planned // 3, planned)
    for index, window in enumerate(windows):
        count, width = len(actual_steps[window["name"]]), window["planned_sample_steps"]
        if (not 0 <= width <= 8 or window["start_step"] <= previous_end or window["end_step"] > planned):
            raise ValueError("training planned windows overlap or differ from their denominator")
        if schema == "true_rl_agent.training_numeric_capture.v1":
            # Legacy archives retain their original contiguous window meaning.
            scheduled = list(range(window["start_step"], window["end_step"] + 1))
            if len(scheduled) != width:
                raise ValueError("legacy training window differs from its denominator")
        else:
            start, stop = edges[index] + 1, edges[index + 1]
            length = stop - start + 1
            if ((window["start_step"], window["end_step"]) != (start, stop)
                    or width > length or (length > 0 and width == 0)):
                raise ValueError("training sampling thirds differ from the declared plan")
            scheduled = [start + ((2 * j + 1) * length - 1) // (2 * width) for j in range(width)]
            if index == 0 and scheduled:
                scheduled[0] = 1
            if window.get("planned_steps") != scheduled:
                raise ValueError("training dispersed sample plan differs from its schedule")
        if any(step not in scheduled for step in actual_steps[window["name"]]):
            raise ValueError("training observed sample was not scheduled")
        previous_end = window["end_step"]
        status = ("not_scheduled" if not width else "sampled" if count == width else
                  "partial" if count else "not_reached" if observed < scheduled[0] else "not_sampled")
        expected = {"sampled_steps": actual_steps[window["name"]], "sampled_control_steps": count,
                    "sampled_env_transitions": count * len(env_ids),
                    "planned_sampled_env_transitions": width * len(env_ids), "status": status}
        if any(window.get(k) != v for k, v in expected.items()):
            raise ValueError("training window coverage differs from actual samples")
    if require_all_windows and (not sample_count or observed != planned or coverage["training_status"] != "pass"
                               or any(w["status"] != "sampled" for w in windows)):
        raise ValueError("training smoke capture did not cover all three windows")
    return {"verified_chunks": chunks, "verified_sampled_control_steps": sample_count,
            "verified_sampled_env_transitions": sample_count * len(env_ids),
            "observed_training_env_transitions": observed * coverage["num_envs"],
            "planned_training_env_transitions": planned * coverage["num_envs"],
            "identity_matches": True if expected_identity is not None else None,
            "coverage_matches": True, "raw_weighted_sum_arithmetic_matches": True if sample_count else None,
            "tolerance": {"rtol": 1e-6, "atol": 1e-6},
            "ppo_post_manager_reward_observed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--train-argv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture", choices=("on", "off"))
    parser.add_argument("--device", help="Unmasked host CUDA ordinal")
    parser.add_argument("--isaac-python", default=sys.executable)
    parser.add_argument("--lease-root", type=Path, default=Path("/tmp/true_rl_agent_gpu_leases"))
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--stair-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    source = json.loads(args.train_argv.read_text(encoding="utf-8"))
    if args.stair_child:
        return _stair_child(source)
    if args.output is None or args.capture is None or not args.device or not 1 <= args.timeout_s <= 900:
        parser.error("--output, --capture, --device and a timeout in [1,900] are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema_version": "true_rl_agent.training_numeric_smoke.v1", "status": "PREPARING",
              "mode": args.mode, "capture": args.capture, "policy_qualification": False,
              "formal_training_claim": False, "eval_jobs": 0, "provider_requests": 0, "train_process_starts": 0,
              "source_argv_sha256": file_sha256(args.train_argv), "device": args.device}
    try:
        task, child_argv = prepare_argv(source, output=output, capture=args.capture == "on")
        # A venv interpreter is commonly a symlink to the base Python. Its
        # invocation path selects pyvenv.cfg, so do not resolve that symlink.
        interpreter = Path(args.isaac_python).expanduser().absolute()
        if not interpreter.is_file():
            raise FileNotFoundError("Isaac Python must be an existing file: " + str(interpreter))
        child_argv[0] = str(interpreter)
        report.update(task=task, diagnostic_workload={"num_envs": 16, "num_steps_per_env": 8,
                      "max_iterations": 1 if task == "stair" else 3, "env_steps": 128 if task == "stair" else 384},
                      subject_kind=unique_option_value(child_argv, "--subject-kind"),
                      program_source_checksum=unique_option_value(child_argv, "--program-source-checksum"))
        if args.mode == "preflight":
            _write(output / "prepared_argv.json", child_argv)
            report["status"] = "PREPARED_NOT_EXECUTED"
            return 0
        if "CUDA_VISIBLE_DEVICES" in os.environ or "NVIDIA_VISIBLE_DEVICES" in os.environ:
            raise ValueError("parent must see the unmasked host GPU inventory")
        inventory = discover_nvidia_gpu_inventory()
        target = next(item for item in inventory if item.execution_device == args.device)
        identity = canonical_json_hash({"argv": child_argv, "output": str(output), "device": target.to_dict()})
        holder = PreGenesisGpuLeaseHolder(session_id="training-numeric-smoke", genesis_identity_hash=identity,
            session_contract_hash=identity, lane_id=args.device, invocation_segment_id=identity[:16], logical_job_id=task)
        manager = GpuLeaseManager(lease_root=args.lease_root, devices={args.device: target.identity})
        with manager.acquire(args.device, holder=holder) as lease:
            lease.assert_owned()
            for name, value in (("--device", "cuda:0"), ("--gpu-host-execution-device", args.device),
                                ("--gpu-process-execution-device", "cuda:0"),
                                ("--gpu-resource-binding-hash", target.identity_hash)):
                _set_option(child_argv, name, value)
            _write(output / "executed_train_argv.json", child_argv)
            command = child_argv if task == "push" else [child_argv[0], str(Path(__file__).resolve()),
                "--stair-child", "--train-argv", str(output / "executed_train_argv.json")]
            report.update(status="RUNNING", gpu_lease=dict(lease.receipt), child_argv=command)
            _write(output / "result.json", report)
            def on_started(pid: int) -> None:
                report.update(train_process_starts=1, child_pid=pid)
                _write(output / "result.json", report)
            process = run_owned_process(command, cwd=ROOT, timeout_s=args.timeout_s, log_dir=output / "process",
                termination_grace_s=5, environment_overrides={"CUDA_VISIBLE_DEVICES": target.uuid, "NVIDIA_VISIBLE_DEVICES": None},
                on_started=on_started)
            report.update(returncode=process.returncode, child_pid=process.pid)
            if process.returncode:
                raise RuntimeError("training smoke child failed; see owned process logs")
        summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "pass":
            raise ValueError("training smoke summary did not pass")
        evidence = summary.get("training", {}).get("numeric_evidence", {})
        report["numeric_evidence"] = evidence
        if args.capture == "on":
            expected_identity = expected_training_numeric_identity(summary, task=task, argv=child_argv)
            report["sampled_reward_checks"] = check_sampled_rewards(
                evidence, task=task, expected_identity=expected_identity, require_all_windows=True)
            coverage = evidence["coverage"]
            if (coverage["num_envs"] != 16 or coverage["steps_per_iteration"] != 8
                    or coverage["planned_control_steps"] != (8 if task == "stair" else 24)):
                raise ValueError("training numeric denominator differs from diagnostic workload")
        elif evidence.get("status") != "not_enabled":
            raise ValueError("disabled training smoke unexpectedly captured numeric data")
        report.update(status="PASS", training_summary_sha256=file_sha256(output / "training_summary.json"),
                      initial_policy_state_hash=summary.get("training", {}).get("initial_policy_state_hash"))
        return 0
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        report["wall_seconds"] = time.monotonic() - started
        _write(output / "result.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
