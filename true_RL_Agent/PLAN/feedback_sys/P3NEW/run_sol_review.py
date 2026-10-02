"""One new Codex CLI context using the lossless prepared-message carrier.

Preparation is offline. Execution needs explicit credentials and a new private
runtime home; no resume, application retry, semantic score or facts sealing.
"""
import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import prepare_desktop_review as adapter

WRAPPER = """Perform exactly the review encoded below. Its messages, roles and block_order
are the frozen review input, supplied once in full. Read all text blocks once in
original message/block order; do not reread input.json with tools. At each
image_file position call native view_image with that relative PNG path and
detail=original, in block order, and inspect its successful image result.
When tools require code-mode exec, use exactly this fixed wrapper, replacing
only the PNG path with the current image_file path:
const r = await tools.view_image({path:"images/image-0000.png",detail:"original"}); image(r.image_url,"original");
No other JavaScript or exec wrapper is allowed.
Use no other tools, files, history, memory, web or other judge's answer.
Return only the JSON required by the original system message, within its
32768 output-token reservation. If incomplete, report complete=false and exact
unreviewed scope. The carrier retains roles as JSON fields in one CLI user input.
BEGIN ORIGINAL CARRIER JSON\n""".encode("utf-8")


def save(path, value):
    with Path(path).open("xb") as stream:
        stream.write(adapter.encoded(value))


def child_environment(runtime_home):
    environment = os.environ.copy()
    removed = sorted(key for key in environment if key.startswith("CODEX_"))
    for key in removed:
        del environment[key]
    environment["CODEX_HOME"] = str(runtime_home)
    return environment, removed


def code_view_template(relative):
    return 'const r = await tools.view_image({path:' + json.dumps(relative) + ',detail:"original"}); image(r.image_url,"original");'


def selective_trace(rollout, expected, blind):
    """Export metadata and successful image results, never private reasoning."""
    trace = {"session_ids": [], "contexts": [], "usage": [], "compactions": [], "tools": []}
    pending = {}
    for line in Path(rollout).read_bytes().splitlines():
        row = json.loads(line)
        kind, payload = row.get("type"), row.get("payload", {})
        if kind == "session_meta":
            trace["session_ids"].append(payload.get("id"))
        elif kind == "turn_context":
            trace["contexts"].append({key: payload.get(key) for key in ("model", "effort", "cwd", "approval_policy")})
        elif kind == "compacted" or (kind == "event_msg" and "compact" in payload.get("type", "")):
            trace["compactions"].append({"timestamp": row.get("timestamp"), "type": kind, "event_type": payload.get("type")})
        elif kind == "event_msg" and payload.get("type") == "token_count":
            info = payload.get("info") or {}
            trace["usage"].append({key: info.get(key) for key in ("total_token_usage", "last_token_usage", "model_context_window")})
        elif kind == "response_item" and payload.get("type") in ("function_call", "custom_tool_call"):
            tool = {"name": payload.get("name"), "call_id": payload.get("call_id"),
                    "arguments_sha256": adapter.sha(str(payload.get("arguments", payload.get("input", ""))).encode()),
                    "result_success": False}
            if tool["name"] == "view_image":
                try:
                    arguments = json.loads(payload["arguments"])
                    path = Path(arguments["path"])
                    tool.update(path=str((path if path.is_absolute() else Path(blind) / path).resolve()),
                                requested_detail=arguments.get("detail"))
                except (ValueError, KeyError, TypeError):
                    tool["argument_error"] = True
            elif (tool["name"] == "exec" and payload.get("type") == "custom_tool_call"
                  and len(trace["tools"]) < len(expected) and isinstance(payload.get("input"), str)
                  and payload["input"].strip() == code_view_template(expected[len(trace["tools"])]["file"])):
                relative = expected[len(trace["tools"])]["file"]
                tool.update(view_transport="fixed_code_mode", tool_input=payload["input"],
                            path=str((Path(blind) / relative).resolve()), requested_detail="original")
            trace["tools"].append(tool)
            pending[tool["call_id"]] = tool
        elif kind == "response_item" and payload.get("type") in ("function_call_output", "custom_tool_call_output"):
            tool = pending.get(payload.get("call_id"))
            if tool is None or not (tool["name"] == "view_image" or tool.get("view_transport") == "fixed_code_mode"):
                continue
            body = payload.get("output")
            images = [item for item in body if isinstance(item, dict) and item.get("type") == "input_image"] if isinstance(body, list) else []
            tool["result_image_count"] = len(images)
            if len(images) == 1:
                image = images[0]
                tool["actual_detail"] = image.get("detail")
                url = image.get("image_url", "")
                if isinstance(url, str) and url.startswith(("data:image/png;base64,", "data:application/octet-stream;base64,")):
                    try:
                        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                        if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
                            tool["result_image_sha256"] = adapter.sha(raw)
                            tool["result_image_size"] = list(struct.unpack(">II", raw[16:24]))
                        else:
                            tool["result_decode_error"] = True
                    except ValueError:
                        tool["result_decode_error"] = True
                tool["result_success"] = tool["actual_detail"] == "original" and "result_decode_error" not in tool
        elif kind == "response_item" and payload.get("type", "").endswith("_call"):
            trace["tools"].append({"name": payload["type"], "call_id": payload.get("call_id"),
                                   "payload_sha256": adapter.sha(adapter.encoded(payload)), "result_success": False})
    expected_paths = [str((Path(blind) / item["file"]).resolve()) for item in expected]
    expected_sizes = [list(struct.unpack(">II", Path(path).read_bytes()[16:24])) for path in expected_paths]
    tools = trace["tools"]
    trace["access_verified"] = len(tools) == len(expected) and all(
        (tool["name"] == "view_image" or tool.get("view_transport") == "fixed_code_mode") and tool.get("path") == path
        and tool.get("requested_detail") == "original" and tool["result_success"]
        and tool.get("result_image_sha256") == item["sha256"]
        and tool.get("result_image_size") == size
        for tool, path, item, size in zip(tools, expected_paths, expected, expected_sizes))
    return trace


def cli_argv(cli, output, blind, policy):
    argv = [str(cli), "exec", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
            "--strict-config", "--json", "--color", "never", "-s", "read-only", "-m", policy["model"],
            "-C", str(blind), "-o", str(output / "raw_answer.json")]
    overrides = {"model_reasoning_effort": policy["reasoning_effort"],
                 "model_context_window": policy["model_context_window"],
                 "model_auto_compact_token_limit": policy["model_auto_compact_token_limit"],
                 "memories.use_memories": False, "memories.generate_memories": False,
                 "project_doc_max_bytes": 0, "skills.include_instructions": False,
                 "skills.bundled.enabled": False, "cloud.skills.enabled": False,
                 "orchestrator.mcp.enabled": False, "features.skip_host_skill_discovery": True,
                 "features.plugins": False, "features.apps": False}
    for key, value in overrides.items():
        argv.extend(["-c", key + "=" + json.dumps(value)])
    return argv + ["-"], overrides


def run(prepared, output, *, source_root, cli_policy, assignment_file=None,
        own_facts_seal=None, own_facts_answer=None, allow_development_single=False,
        execute=False, auth_source_home=None, runtime_home=None, cli=None):
    output = Path(output).resolve()
    policy = adapter.read(cli_policy)
    transport = adapter.prepare(prepared, output, source_root=source_root, cli_policy=cli_policy,
        assignment_file=assignment_file, own_facts_seal=own_facts_seal, own_facts_answer=own_facts_answer,
        allow_development_single=allow_development_single)
    blind = output / "blind"
    stdin = WRAPPER + (blind / "input.json").read_bytes() + b"\nEND ORIGINAL CARRIER JSON\n"
    (output / "stdin.txt").write_bytes(stdin)
    (output / "executed_driver.py").write_bytes(Path(__file__).read_bytes())
    cli = Path(cli or shutil.which("codex") or "")
    adapter.require(cli.is_file(), "Codex executable must be explicitly available")
    argv, overrides = cli_argv(cli.resolve(), output, blind, policy)
    registration = {"status": "REGISTERED_ONE_CALL" if execute else "PREPARED_NO_CALL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "transport": adapter.ref(output / "transport_receipt.json"),
        "driver": adapter.ref(output / "executed_driver.py"), "cli": adapter.ref(cli),
        "stdin": adapter.ref(output / "stdin.txt"), "policy": adapter.ref(cli_policy), "argv": argv,
        "requested_config": overrides, "resume": False, "ephemeral": False,
        "application_retries": 0, "hidden_transport_retries": "unobserved; no zero-retry claim",
        "output_reservation": 32768, "cli_output_cap_enforced": False,
        "native_roles": False, "original_message_roles_preserved_in_json": True,
        "backend_image_processing": "unobserved; local PNG bytes and tool-result SHA/detail are separate evidence",
        "os_filesystem_read_isolation": False, "semantic_score": None}
    if not execute:
        save(output / "registration.json", registration)
        return registration
    adapter.require(auth_source_home is not None and runtime_home is not None, "explicit auth source and new runtime home required")
    runtime_home, auth_source_home = Path(runtime_home).resolve(), Path(auth_source_home).resolve()
    adapter.require(not runtime_home.exists() and (auth_source_home / "auth.json").is_file(), "new home and existing explicit auth.json required")
    adapter.require(not runtime_home.is_relative_to(output) and not output.is_relative_to(runtime_home), "runtime home must be outside exported output")
    environment, removed = child_environment(runtime_home)
    registration.update(runtime_home=str(runtime_home), removed_inherited_environment_names=removed)
    save(output / "registration.json", registration)
    runtime_home.mkdir(parents=True, mode=0o700)
    stdout_path, stderr_path = runtime_home / "cli.stdout.jsonl", runtime_home / "cli.stderr.txt"
    receipt = {"status": "EXECUTION_FAILED", "registration": adapter.ref(output / "registration.json"),
               "semantic_score": None, "actual_invoice_cost": None, "fingerprint": None}
    try:
        shutil.copyfile(auth_source_home / "auth.json", runtime_home / "auth.json")
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            result = subprocess.run(argv, input=stdin, stdout=stdout, stderr=stderr, env=environment,
                timeout=policy.get("timeout_seconds", 1800), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        receipt["exit_code"] = result.returncode
    except (subprocess.TimeoutExpired, OSError) as error:
        receipt["execution_error"] = type(error).__name__
    finally:
        # Preserve this context's runtime evidence, remove the copied credential.
        (runtime_home / "auth.json").unlink(missing_ok=True)
    for name, path in (("private_stdout", stdout_path), ("private_stderr", stderr_path)):
        if path.exists():
            receipt[name] = adapter.ref(path)
    try:
        events = [json.loads(line) for line in stdout_path.read_bytes().splitlines() if line]
        thread_ids = [item["thread_id"] for item in events if item.get("type") == "thread.started"]
        receipt.update(thread_ids=thread_ids, usage=[item["usage"] for item in events if item.get("type") == "turn.completed"])
        rollouts = list((runtime_home / "sessions").rglob("*.jsonl"))
        adapter.require(len(thread_ids) == 1 and len(rollouts) == 1, "one new thread and one retained rollout required")
        trace = selective_trace(rollouts[0], transport["images"], blind)
        adapter.require(trace["session_ids"] == thread_ids, "rollout thread differs from actual CLI thread")
        receipt["images_unchanged_after_exec"] = all(adapter.ref(blind / item["file"])["sha256"] == item["sha256"] for item in transport["images"])
        receipt["input_unchanged_after_exec"] = adapter.ref(blind / "input.json") == transport["blind_input"]
        save(output / "access_trace.json", trace)
        receipt["access_trace"] = adapter.ref(output / "access_trace.json")
        receipt["actual_models"] = sorted({item["model"] for item in trace["contexts"] if item["model"]})
        verified = (trace["access_verified"] and receipt["images_unchanged_after_exec"]
                    and receipt["input_unchanged_after_exec"] and bool(trace["contexts"])
                    and all(item["model"] == policy["model"] and item["effort"] == policy["reasoning_effort"]
                            for item in trace["contexts"])
                    and any(type(item.get("input_tokens")) is int and item["input_tokens"] > 0
                            for item in receipt["usage"]) and not trace["compactions"])
        receipt["status"] = "COMPLETED_TRANSPORT_VERIFIED_NOT_RELEASED" if receipt.get("exit_code") == 0 and verified else "TRANSPORT_NOT_VERIFIED"
    except (OSError, ValueError, KeyError, TypeError) as error:
        receipt["trace_error"] = type(error).__name__ + ": " + str(error)
    answer = output / "raw_answer.json"
    if answer.exists():
        receipt["raw_answer"] = adapter.ref(answer)
    else:
        receipt["status"] = "EXECUTION_FAILED"
        receipt["answer_missing"] = True
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(output / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "output", "source-root", "cli-policy"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("review-assignment", "own-facts-seal", "own-facts-answer", "auth-source-home", "runtime-home", "cli"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--allow-development-single", action="store_true")
    parser.add_argument("--execute-one-call", action="store_true")
    args = parser.parse_args()
    result = run(args.prepared, args.output, source_root=args.source_root, cli_policy=args.cli_policy,
        assignment_file=args.review_assignment, own_facts_seal=args.own_facts_seal,
        own_facts_answer=args.own_facts_answer, allow_development_single=args.allow_development_single,
        execute=args.execute_one_call, auth_source_home=args.auth_source_home, runtime_home=args.runtime_home, cli=args.cli)
    print(json.dumps({key: result[key] for key in ("status", "exit_code", "usage", "trace_error", "execution_error") if key in result}))
