"""One old-development semantic execution check; no provider or scoring calls."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("confirmation_runner", BASE / "run_confirmation_review.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = OUT / "inputs"
    inputs.mkdir(exist_ok=False)
    prior = BASE / "results/capacity_actual_facts_r1"
    original_path = prior / "prepared_report/episode-02/gpt6-sol/request.http_body.json"
    original = json.loads(original_path.read_bytes())
    manifest_path = prior / "report.manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for item in manifest["materials"]:
        p = Path(item["file"])
        item["file"] = str(p if p.is_absolute() else (manifest_path.parent / p).resolve())

    protocol_path = BASE / "N5_FACT_REVIEW_PROTOCOL_ZH.md"
    text = protocol_path.read_text(encoding="utf-8")
    operative = "## 1." + text.split("## 1.", 1)[1].split("## root", 1)[0]
    operative = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", operative)
    protocol = inputs / "review_protocol.json"
    runner.save(protocol, {
        "method": "independent_arithmetic_and_dual_semantics_v1",
        "source_sha256": sha(protocol_path), "operative_draft": operative,
        "this_packet_modality": "numeric_only",
        "execution_scope": [
            "Review every actual interpretation, definition boundary and process relation in the supplied full derivation, and match all own sealed necessary facts. There is no bounded author report or visual observation in this episode packet.",
            "The supplied independent arithmetic audit records program checks and explicit unchecked scope, not a model answer or semantic truth. Do not manufacture hundreds of repeated model votes for individual formalized event scalars.",
            "Report all reviewed interpretation locators and all unreviewed interpretation locators. Keep program/provenance limits separate; complete applies only to the explicitly assigned semantic review, not whole N5 admission.",
            "All complete numeric trajectories and the complete derivation are still supplied. Read them in full. Do not replace full process inspection with production-selected positives or a program summary.",
            "Structure-object matching is allowed for the complete derivation only when that object and its stated definition/polarity/relation actually express the sealed necessary fact. Raw operands merely being present is not a match.",
            "No image is supplied; major_visible_forms is empty. Use only these material bytes and your own supplied sealed facts. Do not open material references or search the repository.",
        ],
    })
    for item in manifest["materials"]:
        if item["role"] == "review_protocol":
            item.update(file=str(protocol), sha256=sha(protocol))

    audit_path = BASE / "results/n5_arithmetic_split_development_r1/stair/verification.json"
    audit = json.loads(audit_path.read_bytes())
    fields = ("schema_version", "status", "scope", "production_oracle_calls",
              "complete_independent_l0_coverage", "episode_id", "steps", "checks", "unchecked",
              "checked_n", "failed_n", "unchecked_n", "rounding", "fixed_rule_details", "verifier_sha256")
    audit_view = {k: audit[k] for k in fields}
    audit_view["source_audit_sha256"] = sha(audit_path)
    audit_view["bound_materials"] = [{k: item[k] for k in ("id", "role", "sha256")}
                                    for item in manifest["materials"] if item["role"] in ("raw_numeric", "full_derivation")]
    audit_file = inputs / "arithmetic_audit.json"
    runner.save(audit_file, audit_view)
    manifest["materials"].append({"id": "independent-arithmetic", "role": "arithmetic_audit",
        "file": str(audit_file), "sha256": sha(audit_file), "mime_type": "application/json"})
    new_manifest = OUT / "report.manifest.json"
    runner.save(new_manifest, manifest, exclusive=True)
    seal = prior / "gpt6-sol_episode-02_facts.seal.json"
    body, instruction, card = runner.build(new_manifest, "gpt6-sol", seal)
    payload = json.loads(body)

    def blocks(request):
        return [json.loads(x["text"]) for x in request["messages"][1]["content"] if x["type"] == "text"]

    preserved = {b["role"]: b for b in blocks(original) if b["role"] != "review_protocol"}
    current = {b["role"]: b for b in blocks(payload)}
    assert all(current[role] == value for role, value in preserved.items())
    assert payload["messages"][0] == original["messages"][0]
    (inputs / "request.http_body.json").write_bytes(body)
    runner.save(OUT / "registration.json", {
        "status": "REGISTERED_ONE_OLD_DEVELOPMENT_EXECUTION_NOT_N5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_doc_sha256": sha(BASE / "SEMANTIC_EXECUTION_SCOPE_REGISTRATION_ZH.md"),
        "requested_model": "gpt-6-sol", "requested_reasoning": "high", "fork_turns": "none",
        "permitted_model_calls": 1, "automatic_retries": 0, "new_deepseek_calls": 0, "gpu_jobs": 0,
        "request_sha256": sha(inputs / "request.http_body.json"), "request_bytes": len(body),
        "protocol_sha256": sha(protocol), "arithmetic_audit_sha256": sha(audit_file),
        "runner_sha256": sha(BASE / "run_confirmation_review.py"), "helper_sha256": sha(Path(__file__)),
        "original_request_sha256": sha(original_path),
        "unchanged_material_roles": list(preserved), "original_system_unchanged": True,
        "exact_desktop_usage_fingerprint_cost": "unknown; HTTP wrapper settings do not configure the agent",
    }, exclusive=True)
    print(json.dumps({"request_sha256": sha(inputs / "request.http_body.json"),
                      "request_bytes": len(body), "preserved": list(preserved)}))


if __name__ == "__main__":
    main()
