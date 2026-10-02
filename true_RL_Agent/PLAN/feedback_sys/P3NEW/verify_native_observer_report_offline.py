"""Synthetic CPU checks for two-call N5 preparation, reservations and audit fidelity.

Source export reconstruction is mocked; actual production request preparation,
budgets, submission with fake transport, receipts and observation audit run.
No source captures, API keys, network, simulator or formal N5 evidence are used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import struct
import tempfile
import zlib
from unittest.mock import patch

import prepare_native_observer as P
import send_native_observer as S
from true_rl_agent.evaluation.visual_observer import (
    IDENTITY_FIELDS, LAYOUT_VERSION, MediaImage, MediaWindow, prepare_observer_request,
)
from true_rl_agent.tasks.stair_climbing.visual_observation import attach_stair_visual_observations
from true_rl_agent.tasks.push_recovery.visual_observation import attach_push_visual_observations


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(S.json_text(value), encoding="utf-8")


def denied(fn, fragment):
    try:
        fn()
    except ValueError as exc:
        assert fragment in str(exc), (fragment, str(exc))
    else:
        raise AssertionError("expected rejection: " + fragment)


def output(alias, *, malformed=False, long_text=False):
    observation = {"body": "root", "spatial_visibility": "visible", "relation": "temporal_change",
        "uncertainty": "Sparse displayed samples only.",
        "text": "x" * 201 if long_text else "The body appears higher at step 6 than at step 1.",
        "evidence_refs": [{"step": 1, "view_id": "v0"}, {"step": 6, "view_id": "v0"}]}
    result = {"episodes": [{"alias": alias, "not_visible": [], "observations": [observation]}]}
    if malformed:
        result["extra"] = "This top-level violation must survive aggregation."
    return result


class Transport:
    def __init__(self, answer=None, error=False):
        self.answer, self.error, self.calls = answer, error, 0

    def post_json(self, **kwargs):
        self.calls += 1
        if self.error:
            raise TimeoutError("synthetic transport failure")
        return {"model": P.MODEL, "system_fingerprint": P.FINGERPRINT,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "choices": [{"message": {"content": self.answer if isinstance(self.answer, str)
                                     else json.dumps(self.answer)}}]}


def verify():
    scenarios = []
    # An actual PNG with each report consuming 24 original full-HD image inputs.
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1920, 1080, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress((b"\0" + bytes((40, 50, 60)) * 1920) * 1080)) + chunk(b"IEND", b""))
    image = MediaImage(png, 1920, 1080, (1,), "v0")
    for scenario in ("normal", "first_transport_failure", "malformed_container", "invalid_json", "invalid_observation",
                     "changed_saved_audit", "over_budget", "changed_release", "old_contract"):
        with tempfile.TemporaryDirectory(prefix="observer-report-cpu-") as temp, ExitStack() as stack:
            root = Path(temp)
            export, selection_dir, report = root / "export", root / "selection", root / "report"
            capture = root / "capture.json"
            write(capture, {"synthetic": True})
            write(export / "extraction.json", {"synthetic": True})
            selection = {"schema_version": "p3new.native_observer_selection.v2", "scope": "n5_confirmation",
                "purpose": "synthetic preparation only", "source": {"capture_card_sha256": P.digest(capture)},
                "episodes": [{"alias": alias, "run_index": 1, "env_index": env, "selected_steps": list(range(1, 7))}
                             for alias, env in (("e1", 2), ("e2", 3))]}
            write(selection_dir / "selection.json", selection)
            write(selection_dir / "card.json", {"observer_profile": "READY_FOR_PREPARATION",
                "pool_complete": True, "missing_classes": [],
                "selection_sha256": P.digest(selection_dir / "selection.json")})
            secret = root / "fake.env"
            secret.write_text("DEEPSEEK_API_KEY=synthetic-not-a-real-key\n", encoding="utf-8")

            def fake_build(source, selection_path, *, episode_alias=None):
                alias = episode_alias or "e1"
                chosen = P.read_json(selection_path)
                steps = range(1, 8 if scenario == "over_budget" else 7)
                identity = {key: "private-" + key for key in IDENTITY_FIELDS}
                identity.update(run_id="1", episode_id="r1.e" + ("2" if alias == "e1" else "3"))
                identity.update({"numeric_" + key: value for key, value in list(identity.items())})
                window = MediaWindow(alias, "camera.v1", LAYOUT_VERSION, "original_full_rgb.v1", ("root",),
                    tuple(replace(image, displayed_steps=(step,), view_id=view)
                          for step in steps for view in ("v0", "v1")), identity=identity)
                prepared = prepare_observer_request([window], model=P.MODEL, max_tokens=P.MAX_TOKENS,
                                                     generation_options=P.GENERATION_OPTIONS)
                card = {"status": "PREPARED_NOT_SENT", "scope": chosen["scope"], "model": P.MODEL,
                    "expected_fingerprint": P.FINGERPRINT, "task": "stair", "provider_requests": 0,
                    "request_sha256": hashlib.sha256(prepared["request_bytes"]).hexdigest(),
                    "request_bytes": len(prepared["request_bytes"]),
                    "extraction_sha256": P.digest(export / "extraction.json")}
                return {"prepared": prepared, "windows": (window,), "numeric_check": None,
                    "card": card, "private_sidecar": {"source_export": str(export)},
                    "mechanical_audit": {"passed": True, "source_reconstruction_mocked": True},
                    "alignment": {"not_checked": ["synthetic fixture has no simulator source"]},
                    "selection_bytes": selection_path.read_bytes()}

            stack.enter_context(patch.object(P, "build", fake_build))
            stack.enter_context(patch.object(S, "build", fake_build))
            stack.enter_context(patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")))
            if scenario == "over_budget":
                denied(lambda: P.prepare_report(export, selection_dir, capture, report), "budget exceeded")
                assert not report.exists()
                scenarios.append({"scenario": scenario, "passed": True, "fake_transport_calls": 0})
                continue
            if scenario == "old_contract":
                old = {"schema_version": "p3new.native_observer_selection.v1", "scope": "old_development_not_n5",
                       "episodes": [{"run_index": 1, "env_index": 2, "selected_steps": [1, 6]}]}
                assert P.selection_episode(old, None) == (old["episodes"][0], "e1", 12)
                denied(lambda: P.selection_episode(old, "e1"), "without an alias override")
                assert P.selection_episode(selection, "e2") == (selection["episodes"][1], "e2", 6)
                denied(lambda: P.selection_episode(selection, None), "explicit episode_alias")
                missing = deepcopy(selection); missing["episodes"].pop()
                denied(lambda: P.selection_episode(missing, "e1"), "both frozen episode aliases")
                write(selection_dir / "old.json", old)
                card = P.prepare(export, selection_dir / "old.json", report)
                release = root / "release.json"
                write(release, {"schema_version": "p3new.native_observer_release.v1", "status": "RELEASED_FOR_ONE_CALL",
                    "scope": "old_development_not_n5", "prepared_card_sha256": P.digest(report / "card.json"),
                    "request_sha256": card["request_sha256"], "model": P.MODEL,
                    "expected_fingerprint": P.FINGERPRINT, "reserved_cost_usd": P.RESERVATION_USD})
                transport = Transport(output("e1"))
                result = S.send(report, release, secret, transport=transport)
                assert result["mechanical_audit_passed"] and transport.calls == 1
                denied(lambda: S.send(report, release, secret, transport=transport), "cannot be submitted again")
                scenarios.append({"scenario": scenario, "passed": True, "fake_transport_calls": transport.calls})
                continue

            card = P.prepare_report(export, selection_dir, capture, report)
            assert card["cumulative_reservations"][-1]["images"] == 24
            assert card["cumulative_reservations"][-1]["pixels"] == 49_766_400
            release = root / "release.json"
            released = {"schema_version": "p3new.native_observer_report_release.v1", "scope": "n5_confirmation",
                "status": "RELEASED_FOR_TWO_CALLS", "report_card_sha256": P.digest(report / "report.json"),
                "packets": card["packets"], "selection_sha256": card["selection_sha256"],
                "extraction_sha256": card["extraction_sha256"], "capture_card_sha256": card["capture_card"]["sha256"],
                "report_limits": card["report_limits"], "model": P.MODEL, "expected_fingerprint": P.FINGERPRINT,
                "reserved_cost_usd": P.RESERVATION_USD}
            write(release, released)
            if scenario == "changed_release":
                released["packets"][1]["request_sha256"] = "0" * 64
                write(release, released)
                denied(lambda: S.rebuild_report(report, release), "report release hashes/limits differ")
                scenarios.append({"scenario": scenario, "passed": True, "fake_transport_calls": 0})
                continue
            one = Transport(output("e1", malformed=scenario == "malformed_container",
                                   long_text=scenario == "invalid_observation"),
                            error=scenario == "first_transport_failure")
            if scenario == "invalid_json":
                # Same missing-list-close shape as the real r9 answer; never repair it.
                one.answer = '{"episodes":[{"alias":"e1","observations":[],"not_visible":["hidden"}]}'
            two = Transport(output("e2"))
            denied(lambda: S.send_report(report, release, secret, episode_alias="e2", transport=two), "first unsubmitted")
            assert two.calls == 0
            try:
                S.send_report(report, release, secret, episode_alias="e1", transport=one)
            except TimeoutError:
                assert scenario == "first_transport_failure"
            denied(lambda: S.send_report(report, release, secret, episode_alias="e1", transport=one), "first unsubmitted")
            S.send_report(report, release, secret, episode_alias="e2", transport=two)
            assert (one.calls, two.calls) == (1, 1)
            denied(lambda: S.send_report(report, release, secret, episode_alias="e2", transport=two), "first unsubmitted")
            _, budget, items = S.rebuild_report(report, release)
            assert (budget.calls, budget.images, budget.pixels, budget.reserved_cost_usd) == (2, 24, 49_766_400, 0.1)
            assert items[0]["receipt"]["budget"]["calls"] == 1
            assert items[1]["receipt"]["budget"]["calls"] == 2
            if scenario == "changed_saved_audit":
                path = report / "e1/response/mechanical_audit.json"
                value = P.read_json(path); value["raw_output"] = {"episodes": []}
                write(path, value)
                denied(lambda: S.aggregate_report(report, release, root / "combined.json"), "per-call audit differs")
            else:
                result = S.aggregate_report(report, release, root / "combined.json")
                assert result["passed"] is (scenario == "normal")
                assert len(result["observations"]) == (1 if scenario in {"first_transport_failure", "invalid_json"} else 2)
                assert all(row["reference"].startswith("calls[") for row in result["observations"])
                if scenario == "malformed_container":
                    assert "calls[0]:output_schema" in result["violations"]
                    assert result["calls"][0]["audit"]["raw_output"]["extra"]
                if scenario == "invalid_observation":
                    assert "bounded_text" in result["observations"][0]["violations"]
                    assert len(result["observations"][0]["raw_observation"]["text"]) == 201
                numeric = {"id": "numeric-main", "kind": "S0", "section": "process",
                           "text": "Original numerical process is retained."}
                stair = {"domain": {"archive_hash": "private-source_id", "checkpoint_hash": "private-checkpoint_id",
                                    "task_protocol_hash": "private-protocol_id"},
                    "episodes": [{"episode_id": f"r1.e{env}", "run": 1, "slot": env} for env in (2, 3)],
                    "semantic": {"sampling": {}, "sentences": [numeric]}}
                push = {"binding": {"checkpoint_hash": "private-checkpoint_id", "resolved_protocol_hash": "private-protocol_id"},
                    "source_bindings": [{"source_index": 0, "logical_run": 1, "manifest_sha256": "private-source_id",
                                         "actual_run_index": 1, "actual_run_status": "verified_archive_identity"}],
                    "transcripts": [{"source": 0, "env_id": env} for env in (2, 3)], "sentences": [numeric], "evidence_catalog": {}}
                stair_attached = attach_stair_visual_observations(stair, result, observer_version="synthetic")
                push_attached = attach_push_visual_observations(push, result, observer_version="synthetic")
                for attached, rows in ((stair_attached, stair_attached["semantic"]["sentences"]),
                                       (push_attached, push_attached["sentences"])):
                    assert attached["observer_evidence"] == result
                    assert attached["observer_source_binding"]["rejected_observation_indices"] == []
                    assert rows[0] == numeric
                    # These transport fixtures contain no semantic review.
                    # Mechanical success alone must never admit free visual prose.
                    assert not any(row["kind"] == "S3" for row in rows)
                    if scenario in {"normal", "invalid_observation"}:
                        assert f"{len(result['observations'])} unreviewed" in rows[-1]["text"]
                    assert f"{len(result['observations'])} statements" in rows[-1]["text"]
                    if scenario in {"malformed_container", "invalid_json"}:
                        assert "Visual output not delivered" in rows[-1]["text"]
                assert stair_attached["episodes"] == stair["episodes"]
                assert push_attached["transcripts"] == push["transcripts"]
                if scenario == "invalid_json":
                    failed = result["calls"][0]["audit"]
                    assert failed["observations"] == [] and failed["raw_output"] is None
                    assert failed["error_type"] == "JSONDecodeError"
                    assert (report / "e1/response/raw_answer.txt").read_text() == one.answer
                    missing_stair = attach_stair_visual_observations(stair, failed, observer_version="synthetic")
                    missing_push = attach_push_visual_observations(push, failed, observer_version="synthetic")
                    for rows in (missing_stair["semantic"]["sentences"], missing_push["sentences"]):
                        assert rows[0] == numeric and [row["kind"] for row in rows] == ["S0", "S5"]
                        assert "Statement count unavailable" in rows[-1]["text"]
                        assert "JSONDecodeError" in rows[-1]["text"]
            scenarios.append({"scenario": scenario, "passed": True, "fake_transport_calls": one.calls + two.calls})
    return {"schema_version": "p3new.native_observer_report_cpu_check.v1", "passed": True,
            "source_reconstruction": "mocked; production source checks unchanged and not certified by this fixture",
            "source_sha256": {Path(module.__file__).name: P.digest(Path(module.__file__)) for module in (P, S)},
            "check_driver_sha256": P.digest(Path(__file__)),
            "provider_requests": 0, "gpu_jobs": 0, "scenarios": scenarios}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify()
    if args.receipt:
        if args.receipt.exists():
            raise ValueError("receipt must be fresh")
        write(args.receipt, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
