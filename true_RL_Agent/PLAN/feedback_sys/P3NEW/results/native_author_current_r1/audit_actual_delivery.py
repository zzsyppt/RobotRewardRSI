"""Audit saved, completed HTTP calls; never send, train, repair, or score value."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "remote_execution"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strings(value):
    if isinstance(value, dict):
        for part in value.values():
            yield from strings(part)
    elif isinstance(value, list):
        for part in value:
            yield from strings(part)
    elif isinstance(value, str):
        yield value
        if value[:1] in ("{", "["):
            try:
                decoded = json.loads(value)
            except ValueError:
                return
            yield from strings(decoded)


def main():
    stage = read(RUN / "stage.released.json")
    usage = read(RUN / "usage.json")
    assert usage["attempts"] == 8 and not usage["violations"]
    rows = []
    for task, lane in (("stair", "a"), ("push", "b")):
        case = task + "-native-dev"
        entry = next(r for r in stage["episodes"] if r["case_id"] == case)
        request_path = RUN / "transport" / case / "request.http_body.json"
        assert sha(request_path) == entry["initial_http_body"]["sha256"]
        request = read(request_path)
        public_path = RUN / "materials" / task / "public_report.json"
        public = read(public_path)
        actual_text = set(strings(request["messages"][-1]["content"]))
        expected_text = list(strings(public["report"]))
        missing = [text for text in expected_text if text not in actual_text]
        assert not missing, (task, missing)
        terminal = read(RUN / "results" / (entry["round_id"] + ".json"))
        assert terminal["delivery_status"] == "PASS" and terminal["transport_invocations"] == 1
        assert terminal["train_status"] == "not_run"
        response_path = RUN / "transport" / case / "raw_response.json"
        response = read(response_path)
        assert response["system_fingerprint"] == "aeb56401ca74e127821c4f9126dcb669"
        assert request["reasoning_effort"] == "high" and request["max_tokens"] == 65536
        candidate = json.loads(response["choices"][0]["message"]["content"])
        (HERE / (task + "_candidate_original.py")).write_bytes(candidate["components_py"].encode("utf-8"))
        charge = next(x for x in usage["attempts_detail"] if x["round_id"] == entry["round_id"])
        rows.append({"task": task, "actual_request_sha256": sha(request_path),
            "response_sha256": sha(response_path), "public_report_sha256": sha(public_path),
            "report_string_leaves_present": len(expected_text), "missing_string_leaves": 0,
            "stop_reason": terminal["stop_reason"], "delivery_status": terminal["delivery_status"],
            "usd_peak": charge["usd_peak"], "usage": response["usage"],
            "candidate_source_sha256": sha(HERE / (task + "_candidate_original.py"))})
    summary = {"status": "ACTUAL_DELIVERY_VERIFIED_NOT_SEMANTIC_OR_VALUE_PASS", "rows": rows,
        "historical_calls": 6, "new_calls": 2, "total_calls": 8, "no_retry": True,
        "new_usd_peak": sum(row["usd_peak"] for row in rows),
        "all_eight_usd_peak": usage["usd_peak_actual_or_reserved"], "actual_invoice_usd": None,
        "limits": ["These author calls use frozen r3 production source.",
            "The later 36-byte Push numeric-versus-visual clarification has offline projection verification only.",
            "Delivery and static candidate checking do not prove semantic correctness or trained policy improvement."]}
    (HERE / "actual_delivery_verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
