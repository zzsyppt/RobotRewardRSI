"""Fake regression for sender postcall failure persistence; no network/credentials.

Run directly; optionally --receipt NEW.json preserves the check result.
Host/release/context construction and author replies are fake. The sender run(),
check_usage(), filesystem terminals, replay guard and next-case guard are real.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import send_native_author_smoke as S


def verify():
    historical = S.D.read(S.P.HISTORY / "usage.json")
    provider = S.D.read(S.P.HISTORY / "execution_stage.released.json")["provider"]
    outcomes = []
    for scenario in ("fingerprint_drift", "model_drift", "unknown_usage", "budget_violation", "two_calls", "pass"):
        with tempfile.TemporaryDirectory(prefix="native-terminal-fake-") as temp, ExitStack() as stack:
            root = Path(temp)
            shared, output, material = root / "journal", root / "new_material", root / "material"
            for name in ("provenance.json", "behavior_full.json", "public_report.json"):
                S.D.write(material / name, {})
            rows = [{"case_id": task + "-native-dev", "task": task, "generation_id": S.GENERATION,
                     "lane_id": lane, "round_id": S.GENERATION + "-" + lane,
                     "material": S.D.ref(material / "provenance.json")}
                    for task, lane in (("stair", "lane-a"), ("push", "lane-b"))]
            stage = {"provider": provider, "episodes": rows,
                     "budget": {"shared_journal_directory": str(shared),
                                "prices_usd_per_million": {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2}}}
            stage_path = output / "stage.released.json"
            S.D.write(stage_path, stage)
            S.D.write(shared / "usage.json", historical)
            before_bytes = (shared / "usage.json").read_bytes()
            after = deepcopy(historical)
            new = deepcopy(historical["attempts_detail"][0])
            new["round_id"] = rows[0]["round_id"]
            after["attempts_detail"].append(new)
            after["attempts"] = 5
            after["usd_peak_actual_or_reserved"] += new["usd_peak"]
            if scenario == "fingerprint_drift":
                new["system_fingerprint"] = "fake-drift"
            elif scenario == "model_drift":
                new["response_model"] = "fake-model"
                after["response_models"].append("fake-model")
            elif scenario == "unknown_usage":
                new["usage_known"] = False
            elif scenario == "budget_violation":
                after["violations"] = ["stage_usd_exceeded"]
                after["usd_peak_actual_or_reserved"] = 4.01
            transport = SimpleNamespace(calls=2 if scenario == "two_calls" else 1,
                                        directory=output / "transport" / rows[0]["case_id"])
            result = SimpleNamespace(stop_reason="candidate_verified", candidate=None, attempt_records=[],
                prompt_payload_ref=SimpleNamespace(to_public_dict=lambda: {"fake": True}),
                authoring_evidence_ref=SimpleNamespace(to_private_dict=lambda: {"fake": True}),
                static_failure=None, provider_failure=None)
            loop = SimpleNamespace(provider=SimpleNamespace(delegate=SimpleNamespace(validate_credential=lambda: None)),
                                   run_round=lambda **kwargs: result)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")))
            stack.enter_context(patch("true_rl_agent.agent_core.openai_compatible_provider.load_env_file",
                                      side_effect=AssertionError("credential read forbidden")))
            stack.enter_context(patch.object(S.sys, "platform", "linux"))
            stack.enter_context(patch.object(S, "PYTHON", str(Path(sys.executable).absolute())))
            stack.enter_context(patch.object(S, "SHARED", shared))
            stack.enter_context(patch.object(S.P, "HISTORY", shared))
            stack.enter_context(patch.object(S, "validate"))
            stack.enter_context(patch.object(S, "acquire_process_file_lock", return_value=nullcontext()))
            constructor = stack.enter_context(patch.object(S, "construct", return_value=(loop, transport)))
            accounting = stack.enter_context(patch.object(S, "accounting", side_effect=[deepcopy(historical), after]))
            errors = []
            for _ in range(2):
                try:
                    returned = S.run(stage_path, output, rows[0]["case_id"])
                except ValueError as exc:
                    errors.append(str(exc))
                else:
                    assert scenario == "pass", "failed delivery replay returned success"
                    assert returned["delivery_status"] == "PASS"
            terminal_path = output / "results" / (rows[0]["round_id"] + ".json")
            terminal = S.D.read(terminal_path)
            assert terminal["stop_reason"] == "candidate_verified", "author stop_reason was rewritten"
            assert constructor.call_count == 1 and accounting.call_count == 2, "replay attempted another send"
            assert (shared / "usage.json").read_bytes() == before_bytes, "historical usage overwritten"
            if scenario != "pass":
                assert len(errors) == 2 and terminal["delivery_status"] == "FAILED"
                kind = "transport" if scenario == "two_calls" else "accounting"
                assert terminal["postcall_validation"][kind]["status"] == "FAILED"
                try:
                    S.run(stage_path, output, rows[1]["case_id"])
                except ValueError:
                    pass
                else:
                    raise AssertionError("later episode advanced past failed delivery")
                assert constructor.call_count == 1
            else:
                assert not errors
                terminal.pop("delivery_status")
                S.D.write(terminal_path, terminal)
                try:
                    S.run(stage_path, output, rows[0]["case_id"])
                except ValueError:
                    pass
                else:
                    raise AssertionError("legacy unvalidated terminal returned success")
                assert constructor.call_count == 1
            outcomes.append({"scenario": scenario, "status": "PASS", "construction_count": constructor.call_count,
                             "postcall_validation": terminal["postcall_validation"], "error_count": len(errors)})
    return {"schema_version": "p3new.native_author_terminal_regression.v1", "status": "PASS_OFFLINE_FAKE_REGRESSION",
            "created_utc": datetime.now(timezone.utc).isoformat(), "driver": S.D.ref(Path(S.__file__)),
            "verifier": S.D.ref(Path(__file__)), "scenarios": outcomes, "api_calls": 0, "credential_reads": 0,
            "training": "NOT_RUN", "formal_N5_review": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        raise ValueError("receipt already exists")
    result = verify()
    if args.receipt:
        S.D.write(args.receipt.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
