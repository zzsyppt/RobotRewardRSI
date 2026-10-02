"""Preserve a declared-complete answer for capacity sizing; never certify or send it."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
R2 = PHASE / "results/n5_review_capacity_development_r2"
R3 = PHASE / "results/capacity_event_table_r3"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    answer_path = HERE / "gpt6-sol_episode-02_facts.json"
    access_path = HERE / "access_receipt.json"
    assert sha(answer_path) == "312365777c99b3e7666a86d914da759ca1fc54e3de9c3529a2da3186f163c2f7"
    assert sha(access_path) == "fca598082643f5aed88d975bf0a6f61c115bb1876fb0d49889dad2af49a756d5"
    answer, access = read(answer_path), read(access_path)
    assert (answer["judge_id"], answer["packet_id"], answer["stage"], answer["complete"]) == ("gpt6-sol", "episode-02", "facts", True)
    assert access["output"]["sha256"] == sha(answer_path)
    assert len(access["read_files"]) == 5 and access["complete"]
    for row in access["read_files"]:
        path = Path(row["path"])
        assert sha(path) == row["sha256"] and path.stat().st_size == row["bytes"]
    facts_manifest = R2 / "packets/episode-02/facts.manifest.json"
    material_rows = read(facts_manifest)["materials"]
    for row in material_rows:
        path = (facts_manifest.parent / row["file"]).resolve()
        assert sha(path) == row["sha256"]
    raw_path = R2 / "materials/related/episode-02.json"
    raw = read(raw_path)
    assert raw["clock"] == {"source_field": "episode_length", "first_step": 1, "last_step": 525, "stride": 1}
    field_checks = []
    for group, source_group in (("raw_columns", "source_fields"), ("direct_columns", "source_direct_fields")):
        for name, field in raw[group].items():
            dtype = np.dtype(field["dtype"])
            if "by_c_order_component" in field:
                components = [np.fromstring(values, sep=",", dtype=dtype) for values in field["by_c_order_component"]]
            else:
                components = []
                for intervals in field["inclusive_constant_intervals_by_component"]:
                    values, next_step = [], 1
                    for first, last, value in intervals:
                        assert first == next_step and first <= last
                        values.extend([value] * (last - first + 1))
                        next_step = last + 1
                    assert next_step == 526
                    components.append(np.asarray(values, dtype=dtype))
            assert all(len(values) == 525 for values in components)
            data = np.stack(components, axis=1).reshape((525, *field["component_shape"]))
            expected = raw[source_group][name]
            digest = hashlib.sha256(data.tobytes(order="C")).hexdigest()
            assert list(data.shape) == expected["shape"] and digest == expected["c_order_bytes_sha256"]
            field_checks.append({"field": name, "shape": list(data.shape), "values": data.size, "sha256": digest})
    assert len(field_checks) == 11 and sum(row["values"] for row in field_checks) == 25200
    findings = answer["findings"]
    assert len(findings["necessary_facts"]) == 24 and len({row["fact_id"] for row in findings["necessary_facts"]}) == 24
    assert findings["major_visible_forms"] == [] and findings["coverage"]["episode_steps"] == [1, 525]
    material_hashes = {row["material_id"]: row["sha256"] for row in findings["materials_seen"]}
    assert material_hashes["facts_manifest"] == sha(facts_manifest)
    assert material_hashes["related-raw"] == sha(raw_path)
    for fact in findings["necessary_facts"]:
        for ref in fact["evidence_refs"]:
            assert ref["sha256"] == material_hashes[ref["material_id"]]
    seal_path = HERE / "gpt6-sol_episode-02_facts.seal.json"
    save(seal_path, {"schema_version": "p3new.confirmation_facts_seal.v1", "status": "SEALED",
        "judge_id": "gpt6-sol", "packet_id": "episode-02", "facts_manifest_sha256": sha(facts_manifest),
        "answer": {"file": answer_path.name, "sha256": sha(answer_path)},
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "old_development_not_n5",
        "seal_meaning": "complete original answer and source identity sealed; not semantic correctness certification",
        "access_receipt_sha256": sha(access_path), "agent_task": "/root/dev_episode02_facts"})
    old_manifest_path = R3 / "packets/episode-02/report.capacity.manifest.json"
    report_manifest = read(old_manifest_path)
    report_manifest.pop("capacity_only")
    for row in report_manifest["materials"]:
        row["file"] = str((old_manifest_path.parent / row["file"]).resolve())
    report_manifest["preparation_scope"] = "old_development_actual_facts_capacity_only_not_n5"
    manifest_path = HERE / "report.manifest.json"
    save(manifest_path, report_manifest)
    runner_path = PHASE / "run_confirmation_review.py"
    assert sha(runner_path) == "c09ef80569d1649a42c397285e12c91a9dabd8bdff9ab87f6c902f733bd5d038"
    spec = importlib.util.spec_from_file_location("existing_review_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    prepared = HERE / "prepared_report/episode-02/gpt6-sol"
    card = runner.prepare(manifest_path, "gpt6-sol", prepared, seal_path)
    current = read(prepared / "request.http_body.json")
    old = read(R3 / "prepared_report_capacity/episode-02/gpt6-sol/request.http_body.json")
    assert current["messages"][0] == old["messages"][0]
    assert current["messages"][1]["content"][:-1] == old["messages"][1]["content"][:-1]
    own = json.loads(current["messages"][1]["content"][-1]["text"])
    assert own["role"] == "own_sealed_facts" and own["data"] == answer and own["sha256"] == sha(answer_path)
    assert current["max_tokens"] == old["max_tokens"] == 32768
    assert sha(answer_path) == access["output"]["sha256"] and sha(access_path) == "fca598082643f5aed88d975bf0a6f61c115bb1876fb0d49889dad2af49a756d5"
    receipt = {"status": "ACTUAL_ANSWER_LENGTH_REPORT_PREPARED_SCOPE_UNCONFIRMED", "scope": "old_development_not_n5",
        "answer_sha256": sha(answer_path), "answer_bytes": answer_path.stat().st_size, "necessary_fact_count": 24,
        "access_receipt_sha256": sha(access_path), "facts_manifest_sha256": sha(facts_manifest),
        "raw_numeric_fields_reconstructed": field_checks, "all_25200_numeric_values_hash_verified": True,
        "facts_content_scope_review": "one successful numeric-only episode; no group/visual/causal claims promoted; not semantic scoring",
        "access_method": "CPU reconstruction of complete source arrays plus model inspection of scans and event windows; not proof every raw scalar entered model context",
        "declared_complete": answer["complete"], "cpu_source_arrays_traversed_and_verified": True,
        "model_visible_evidence": "selected 25-step trajectory scans and event windows per access receipt; actual full original-value delivery not demonstrated",
        "full_payload_context_capacity_demonstrated": False, "facts_protocol_compliance_established": False,
        "facts_semantic_completeness_established": False, "sol_actual_context_capacity_pass": False,
        "answer_size_is_upper_bound_for_other_packets": False,
        "actual_agent_input_peak_tokens": None, "agent_compaction_observed": None,
        "telemetry_reason": "parent received only canonical task name; no per-agent rollout/thread id; no wider thread search or extra turn",
        "original_answer_unchanged": True, "own_facts_data_complete_equal_after_serialization": True,
        "all_other_report_text_blocks_equal_to_r3": True, "report_manifest_sha256": sha(manifest_path),
        "facts_seal_sha256": sha(seal_path), "request_sha256": card["request_sha256"], "request_bytes": card["request_bytes"],
        "prepared_card_sha256": sha(prepared / "prepare.json"), "runner_sha256": sha(runner_path),
        "report_model_calls": 0, "gpu_runs": 0, "release_present": False, "tokenizer_measurement": "pending CPU only"}
    save(HERE / "verification.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "answer_bytes", "necessary_fact_count", "request_sha256", "request_bytes")}, indent=2))


if __name__ == "__main__":
    main()
