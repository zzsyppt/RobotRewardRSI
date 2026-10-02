"""R2 sequential facts control only; never edits a frozen prompt or seals answers."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent
PLAN = BASE / "sol_facts_execution_preparation.all40.json"
CONTROL = BASE / "sol_facts_control"
ORDER = ([f"episode-{i:02d}" for i in range(1, 25)]
         + [f"focus-{i:02d}" for i in range(1, 13)]
         + [f"media-{i:02d}" for i in range(1, 5)])
DRIVER_SHA = "ec6c2b00c2f1a082476c54f498ce7e644d1f39b4c9c803fe8e45b38ab78b4276"
BENIGN_ITEMS = {"agent_message", "reasoning", "todo_list", "plan"}
TOOL_ITEMS = {"command_execution", "file_change", "mcp_tool_call", "web_search",
              "tool_call", "custom_tool_call", "function_call", "image_generation",
              "collab_tool_call", "collaboration_tool_call"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"file": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def check_ref(item):
    actual = ref(item.get("file", item.get("path")))
    if actual["sha256"] != item["sha256"] or (
            "size_bytes" in item and actual["size_bytes"] != item["size_bytes"]):
        raise ValueError("reference differs: " + actual["file"])
    return actual


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def check_unit(unit, source_root):
    from PIL import Image
    for name in ("manifest", "prepared_card", "request"):
        check_ref(unit[name])
    prepared = Path(unit["prepared"])
    output = Path(unit["execution_output"])
    expected_output = BASE / "sol_facts_execution" / unit["packet_id"]
    if output.resolve() != expected_output.resolve() or output.exists():
        raise ValueError("execution output not new or outside fixed packet path")
    card = read(prepared / "prepare.json")
    if (card["packet_id"] != unit["packet_id"] or card["judge_id"] != "gpt6-sol"
            or card["stage"] != "facts" or card["own_facts_seal"] is not None
            or card.get("capacity_facts_fixture") is not None):
        raise ValueError("prepared identity/stage/isolation differs")
    for name in ("manifest", "runner", "transport", "coverage_helper"):
        check_ref(card[name])
    for material in card["materials"]:
        check_ref(material)
    # Frozen builder is imported only for offline equality; its run is never called here.
    import run_confirmation_review as review
    body, _instruction, rebuilt = review.build(card["manifest"]["file"], "gpt6-sol")
    if rebuilt != card or body != Path(unit["request"]["file"]).read_bytes():
        raise ValueError("frozen rebuilt request differs")
    if hashlib.sha256(body).hexdigest() != unit["request_sha256"]:
        raise ValueError("request differs from registered remote-equal SHA")
    if len(body) > card["body_limit_bytes"] or card["png_count"] > card["max_png_count"]:
        raise ValueError("frozen hard carrier limit exceeded")
    blocks = json.loads(body)["messages"][1]["content"]
    images = []
    for index, block in enumerate(blocks):
        if block["type"] == "text":
            continue
        if block["type"] != "image_url":
            raise ValueError("unknown block carrier")
        label = json.loads(blocks[index - 1]["text"])
        url = block["image_url"]["url"]
        if not url.startswith("data:image/png;base64,"):
            raise ValueError("unknown image carrier")
        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != label["sha256"]:
            raise ValueError("image label hash differs")
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            image.verify()
        images.append({"attachment_index": len(images), "material_id": label["material_id"],
                       "sha256": sha, "width": width, "height": height, "size_bytes": len(raw)})
    if len(images) != unit["image_count"] or len(images) != card["png_count"]:
        raise ValueError("image count differs")
    if images:
        check_ref(unit["image_order_check"])
        if images != read(unit["image_order_check"]["file"])["images"]:
            raise ValueError("original image bytes/order/dimensions differ")
    expected_argv = [sys.executable, "-B", str(source_root / "PLAN/feedback_sys/P3NEW/run_sol_review.py"),
                     "--prepared", str(prepared), "--output", str(output), "--execute-one-call"]
    if [os.path.normcase(v) for v in expected_argv] != [os.path.normcase(v) for v in unit["pending_execution_argv"]]:
        raise ValueError("registered argv differs")
    return {"packet_id": unit["packet_id"], "request_sha256": unit["request_sha256"],
            "manifest_sha256": unit["manifest"]["sha256"], "output_new": True,
            "request_bytes": len(body), "image_count": len(images),
            "image_dimensions": sorted({(i["width"], i["height"]) for i in images}),
            "input_text_tokens_deepseek_proxy": unit["input_text_tokens_deepseek_proxy"],
            "text_proxy_total_with_registered_reserves": unit["text_proxy_total_with_registered_reserves"]}


def preflight():
    plan = read(PLAN)
    if plan["execution_order"] != ORDER or [u["packet_id"] for u in plan["execution_plan"]] != ORDER:
        raise ValueError("fixed all40 order differs")
    if (BASE / "sol_facts_execution").exists():
        raise ValueError("formal output root already exists; no automatic resume or retry")
    for item in plan["sources"].values():
        check_ref(item)
    check_ref(plan["execution_python"]["executable"])
    if Path(sys.executable).resolve() != Path(plan["execution_python"]["executable"]["file"]).resolve():
        raise ValueError("use the registered bundled Python")
    if ref(plan["sources"]["driver"]["file"])["sha256"] != DRIVER_SHA:
        raise ValueError("frozen driver differs")
    cli = shutil.which("codex")
    if not cli:
        raise ValueError("Codex CLI unavailable")
    cli_differs = Path(cli).resolve() != Path(plan["cli_readonly_checks"]["path"]).resolve()
    environment_blockers = ["actual_CLI_differs_from_registered_CLI"] if cli_differs else []
    cli_version = subprocess.run([cli, "--version"], capture_output=True, text=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    cli_help = subprocess.run([cli, "exec", "--help"], capture_output=True, text=True,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    required_flags = ["--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--strict-config",
                      "--json", "--color", "--sandbox", "--model", "--config", "--cd", "--output-last-message", "--image"]
    missing_flags = [flag for flag in required_flags if flag not in cli_help.stdout]
    if cli_version.returncode != 0 or cli_help.returncode != 0 or missing_flags:
        environment_blockers.append("CLI_readonly_compatibility_check_failed")
    source_root = Path(plan["sources"]["driver"]["file"]).parents[3]
    sys.path[:0] = [str(source_root / "PLAN/feedback_sys/P3NEW"), str(source_root / "src"), str(source_root)]
    roots = []
    for task in ("stair", "push"):
        path = BASE / "observers" / task / "root_module_review_receipt.json"
        receipt = read(path)
        check_ref(receipt["review"])
        if (receipt["formal_facts_calls_before_seal"] != 0
                or receipt["formal_report_calls_before_seal"] != 0
                or receipt["original_raw_answers_preserved"] is not True):
            raise ValueError("root module pre-facts barrier differs")
        roots.append({"task": task, "receipt": ref(path), "recorded_at_utc": receipt["recorded_at_utc"],
                      "review_sha256": receipt["review_sha256"], "review_semantics_read": False})
    units = [check_unit(unit, source_root) for unit in plan["execution_plan"]]
    capture = read(plan["sources"]["capture_card"]["file"])
    proxy_limit = capture["materials"]["input_token_proxy_limit"]
    if any(u["input_text_tokens_deepseek_proxy"] > proxy_limit for u in units):
        raise ValueError("registered input text proxy limit exceeded")
    result = {"schema_version": "p3new.sol_facts_control_preflight.v1", "recorded_at_utc": now(),
              "status": "ALL40_INPUT_PASS_ENVIRONMENT_BLOCKED_NO_MODEL_CALL" if environment_blockers else "ALL40_PREFLIGHT_PASS_NO_MODEL_CALL", "control": ref(__file__),
              "plan": ref(PLAN), "python": ref(sys.executable), "cli": ref(cli),
              "frozen_driver": ref(plan["sources"]["driver"]["file"]), "root_module_barriers": roots,
              "fixed_order": ORDER, "units": units, "input_text_proxy_limit": proxy_limit,
              "configured_context_window": 512000, "configured_auto_compact_threshold": 450000,
              "actual_sol_input_tokens": None, "actual_sol_image_tokens": None,
              "actual_context_completion_proven": False, "model_calls": 0,
              "facts_seals_created": 0, "reports_created": 0, "environment_blockers": environment_blockers,
              "cli_readonly_checks": {"registered_path": plan["cli_readonly_checks"]["path"],
                  "registered_path_exists": Path(plan["cli_readonly_checks"]["path"]).exists(),
                  "actual_path": cli, "version_exit_code": cli_version.returncode,
                  "version_stdout": cli_version.stdout, "exec_help_exit_code": cli_help.returncode,
                  "exec_help_stdout": cli_help.stdout, "missing_named_flags": missing_flags,
                  "runtime_config_acceptance_tested": False, "model_called": False},
              "capacity_limit": "DeepSeek text tokenizer is a proxy only. No 1024-token image formula is applied to SOL.",
              "no_semantic_answer_or_original_observation_read": True}
    return plan, source_root, result


def postflight(unit, driver_exit):
    output = Path(unit["execution_output"])
    errors = []
    def load_control_json(path):
        if not path.exists():
            return {}
        try:
            value = read(path)
            if not isinstance(value, dict):
                raise ValueError("not an object")
            return value
        except (OSError, ValueError, TypeError):
            errors.append(path.name + "_invalid_or_unreadable")
            return {}
    event_types, item_types = Counter(), Counter()
    tools, unknown, compactions, failed = [], [], [], []
    events_path = output / "events.jsonl"
    if not events_path.exists():
        errors.append("events_missing")
    else:
        for line_number, line in enumerate(events_path.read_bytes().splitlines(), 1):
            if not line:
                continue
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                errors.append("invalid_event_json_at_line_" + str(line_number))
                continue
            if not isinstance(event, dict) or not isinstance(event.get("item", {}), dict):
                errors.append("invalid_event_shape_at_line_" + str(line_number))
                continue
            kind = event.get("type", "")
            item_kind = event.get("item", {}).get("type", "")
            event_types[kind] += 1
            if item_kind:
                item_types[item_kind] += 1
                descriptor = {"line": line_number, "event_type": kind, "item_type": item_kind}
                if item_kind in TOOL_ITEMS or "tool" in item_kind or item_kind.endswith("_call"):
                    tools.append(descriptor)
                elif item_kind not in BENIGN_ITEMS and "compact" not in item_kind.lower():
                    unknown.append(descriptor)
            if "compact" in kind.lower() or "compact" in item_kind.lower():
                compactions.append({"line": line_number, "event_type": kind, "item_type": item_kind})
            if kind in {"error", "turn.failed"}:
                failed.append({"line": line_number, "event_type": kind})
    if driver_exit != 0: errors.append("driver_exit_nonzero")
    if tools: errors.append("actual_tool_activity")
    if unknown: errors.append("unknown_item_type_requires_manual_control_review_not_called_tool")
    if compactions: errors.append("compaction_observed")
    if failed: errors.append("error_or_turn_failed_event")
    if event_types["thread.started"] != 1 or event_types["turn.completed"] != 1:
        errors.append("fresh_single_turn_completion_not_confirmed")
    receipt_path = output / "receipt.json"
    receipt = load_control_json(receipt_path)
    if receipt.get("exit_code") != 0: errors.append("cli_exit_not_zero_or_missing")
    if receipt.get("complete") is not True: errors.append("complete_not_true")
    if receipt.get("answer_error"): errors.append("frozen_answer_error")
    for key in ("registration", "events", "raw_answer", "answer_coverage"):
        if key not in receipt:
            errors.append(key + "_receipt_missing")
        else:
            try: check_ref(receipt[key])
            except (OSError, ValueError): errors.append(key + "_reference_mismatch")
    coverage_path = output / "answer_coverage.json"
    coverage = load_control_json(coverage_path)
    if coverage.get("mechanical_valid") is not True or coverage.get("review_complete") is not True:
        errors.append("frozen_mechanical_coverage_not_complete")
    return {"schema_version": "p3new.sol_facts_control_result.v1", "packet_id": unit["packet_id"],
            "finished_utc": now(), "status": "EXECUTION_MECHANICAL_PASS" if not errors else "STOPPED_UNFINISHED",
            "driver_exit_code": driver_exit, "cli_exit_code": receipt.get("exit_code"),
            "event_type_counts": dict(event_types), "item_type_counts": dict(item_types),
            "actual_tool_events": tools, "unknown_item_events": unknown, "compaction_events": compactions,
            "failure_events": failed, "usage": receipt.get("usage"), "complete": receipt.get("complete"),
            "coverage_mechanical_valid": coverage.get("mechanical_valid"),
            "coverage_review_complete": coverage.get("review_complete"),
            "coverage_reference_count": coverage.get("references_checked"),
            "coverage_error_count": len(coverage.get("errors", [])), "errors": errors,
            "artifacts": {p.name: ref(p) for p in (output.iterdir() if output.exists() else []) if p.is_file() and p.name != "prompt.txt"},
            "semantic_support_proven": False, "facts_seal_created": False,
            "reports_created": False, "actual_invoice_cost": None,
            "compaction_telemetry_limit": "CLI event stream only; hidden backend telemetry unavailable"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-facts", action="store_true")
    parser.add_argument("--root-release-note")
    args = parser.parse_args()
    if args.execute_facts and not args.root_release_note:
        raise ValueError("execute only after explicit root launch, recorded with --root-release-note")
    plan, source_root, preparation = preflight()
    if not args.execute_facts:
        save(CONTROL / "preflight.json", preparation)
        print(json.dumps({"status": preparation["status"], "units": len(preparation["units"]), "model_calls": 0}))
        return
    if preparation["environment_blockers"]:
        raise ValueError("environment blockers remain; no formal call released")
    save(CONTROL / "launch.json", {"created_at_utc": now(), "root_release_note": args.root_release_note,
                                  "preflight": preparation, "automatic_retries": 0, "scope": "facts_only_no_seals_no_reports"})
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(source_root / "src"), str(source_root), str(source_root / "PLAN/feedback_sys/P3NEW")])
    results = []
    for index, unit in enumerate(plan["execution_plan"]):
        check_unit(unit, source_root)
        check_ref(plan["sources"]["driver"])
        packet = unit["packet_id"]
        save(CONTROL / (packet + ".dispatch.json"), {"started_at_utc": now(), "packet_id": packet,
             "ordinal": index + 1, "argv": unit["pending_execution_argv"], "automatic_retry": False})
        print(json.dumps({"status": "DISPATCHING_FROZEN_ONE_CALL", "packet_id": packet, "ordinal": index + 1}), flush=True)
        with (CONTROL / (packet + ".stdout.txt")).open("xb") as stdout, (CONTROL / (packet + ".stderr.txt")).open("xb") as stderr:
            process = subprocess.run(unit["pending_execution_argv"], stdout=stdout, stderr=stderr,
                                     env=env, cwd=BASE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result = postflight(unit, process.returncode)
        save(CONTROL / (packet + ".result.json"), result)
        results.append({"packet_id": packet, "status": result["status"], "result": ref(CONTROL / (packet + ".result.json"))})
        print(json.dumps({k: result[k] for k in ("packet_id", "status", "driver_exit_code", "cli_exit_code", "complete", "coverage_review_complete", "errors", "usage")}), flush=True)
        if result["errors"]:
            break
    status = "ALL40_EXECUTION_MECHANICAL_PASS_NO_SEALS" if len(results) == 40 and all(r["status"] == "EXECUTION_MECHANICAL_PASS" for r in results) else "STOPPED_UNFINISHED_NO_RETRY"
    save(CONTROL / "batch_result.json", {"finished_at_utc": now(), "status": status, "fixed_denominator": 40,
         "attempted_units": len(results), "results": results, "not_attempted": ORDER[len(results):],
         "semantic_support_proven": False, "facts_seals_created": 0, "reports_created": 0})
    print(json.dumps({"status": status, "attempted_units": len(results), "fixed_denominator": 40}), flush=True)
    if status != "ALL40_EXECUTION_MECHANICAL_PASS_NO_SEALS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
