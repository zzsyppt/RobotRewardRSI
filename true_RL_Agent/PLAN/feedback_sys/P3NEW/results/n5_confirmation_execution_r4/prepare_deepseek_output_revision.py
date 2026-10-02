"""CPU-only preparation of seven R4 DS facts at 65536; never send or resume.

Requires the root's actual revision-card file and SHA. The original capture,
source, prepared requests, receipts and answers remain unchanged. Episode-01
is excluded. This does not prepare report requests or release any API call.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import prepare_facts_offline as prep

PACKETS = ("episode-13", "focus-01", "focus-07", "media-01", "media-02", "media-03", "media-04")
FAILURE_RECEIPT_SHA = "c010182691bce2a93902450db302a446292f64c2579a1e73700234549478e65e"


def checked(reference):
    path = Path(reference.get("path", reference.get("file", "")))
    actual = prep.ref(path)
    prep.require(actual["sha256"] == reference["sha256"], "Original source SHA differs: " + str(path))
    if "size_bytes" in reference:
        prep.require(actual["size_bytes"] == reference["size_bytes"], "Original source size differs")
    return path


def prepare_revision(card_path, card_sha, output):
    card_path, output = card_path.resolve(), output.resolve()
    prep.require(len(card_sha) == 64 and all(c in "0123456789abcdef" for c in card_sha), "Actual revision-card SHA required")
    prep.require(prep.ref(card_path)["sha256"] == card_sha, "Revision-card SHA differs")
    prep.require(isinstance(prep.read(card_path), dict), "Revision card must be a JSON object")
    prep.require(output.is_relative_to(prep.C.resolve()) and output != prep.C.resolve() and not output.exists(),
                 "Output must be a new directory inside R4 execution root")
    prep.verify_source(prep.S, prep.C / "runtime_source_manifest.json")
    prep.require(prep.ref(prep.C / "capture_card.json")["sha256"] == prep.CARD_SHA, "Original capture card differs")
    p = prep.S / "PLAN/feedback_sys/P3NEW"
    sys.path[:0] = [str(p), str(prep.S / "src")]
    review = importlib.import_module("run_confirmation_review")
    prep.require(Path(review.__file__).resolve() == (p / "run_confirmation_review.py").resolve(), "Wrong frozen runner")
    export_path = prep.C / "facts_offline_preparation/export_receipt.json"
    export = prep.read(export_path)
    prep.require(export["assignment"]["sha256"] == prep.ASSIGNMENT_SHA, "Original assignment differs")
    selected = {r["packet_id"]: r for r in export["records"]
                if r["judge_id"] == "deepseek" and r["packet_id"] in PACKETS}
    prep.require(set(selected) == set(PACKETS), "Seven original DS slots required")
    originals, inputs = {}, []
    # Validate every original before creating any new directory.
    for packet in PACKETS:
        row = selected[packet]
        manifest_path = checked(row["remote_manifest"])
        prepare_path = checked(row["remote_prepare"])
        request_path = checked(row["remote_request"])
        catalog_path = prepare_path.parent / "reference_catalog.json"
        manifest, prepared = prep.read(manifest_path), prep.read(prepare_path)
        body = prep.read(request_path)
        prep.require(manifest["stage"] == prepared["stage"] == "facts" and manifest["purpose"] == "confirmation"
                     and prepared["judge_id"] == "deepseek" and prepared["packet_id"] == packet
                     and manifest["max_output_tokens"] == prepared["max_output_tokens"] == body["max_tokens"] == 32768
                     and prepared["capacity_facts_fixture"] is None and prepared["own_facts_seal"] is None,
                     "Not an original nonfixture DS facts request")
        prep.require(prepared["request_sha256"] == prep.ref(request_path)["sha256"]
                     and review.encoded(body) == request_path.read_bytes(), "Original request binding/serialization differs")
        reviewers = review.assigned_reviewers(manifest)
        prep.require(reviewers["B"]["judge_id"] == "deepseek", "Not the original B identity")
        old_catalog = prep.read(catalog_path)
        prep.require(hashlib.sha256(review.coverage.canonical(old_catalog)).hexdigest() == prepared["reference_catalog"]["sha256"],
                     "Original catalog canonical SHA differs")
        artifact_names = ("receipt.json", "raw_response.json", "raw_answer.txt", "raw_reasoning.txt")
        if packet == "episode-13":
            receipt_path = prepare_path.parent / "receipt.json"
            prep.require(prep.ref(receipt_path)["sha256"] == FAILURE_RECEIPT_SHA, "Original failed receipt differs")
            receipt = prep.read(receipt_path)
            prep.require(receipt["complete"] is False and receipt["finish_reason"] == "length"
                         and receipt["usage"]["completion_tokens"] == 32768, "Not the registered failed attempt")
        else:
            prep.require(not any((prepare_path.parent / n).exists() for n in artifact_names),
                         "A nominally unissued facts slot already has execution evidence: " + packet)
        new_manifest = deepcopy(manifest)
        new_manifest["max_output_tokens"] = 65536
        for item in new_manifest["materials"]:
            path, _ = review.referenced(manifest_path.parent, item)
            item["file"] = str(path)  # Preserve referent when moving the manifest.
        watched = [manifest_path, prepare_path, request_path, catalog_path]
        watched += [prepare_path.parent / n for n in artifact_names if (prepare_path.parent / n).exists()]
        originals.update({str(path): prep.ref(path) for path in watched})
        inputs.append((packet, new_manifest, prepared, request_path, catalog_path, manifest_path))
    output.mkdir()
    results = []
    for packet, manifest, old_card, old_request, old_catalog, old_manifest in inputs:
        unit = output / packet
        unit.mkdir()
        manifest_path = unit / "facts.manifest.json"
        review.save(manifest_path, manifest, exclusive=True)
        dest = unit / "prepared"
        card = review.prepare(manifest_path, "deepseek", dest)
        old_body = prep.read(old_request)
        expected = {**old_body, "max_tokens": 65536}
        new_request = dest / "request.http_body.json"
        prep.require(new_request.read_bytes() == review.encoded(expected),
                     "New request differs beyond max_tokens; preserve this failed preparation")
        new_catalog = dest / "reference_catalog.json"
        prep.require(new_catalog.read_bytes() == old_catalog.read_bytes()
                     and card["reference_catalog"] == old_card["reference_catalog"],
                     "Original reference catalog changed")
        prep.require(card["materials"] == old_card["materials"]
                     and card["review_assignment"] == old_card["review_assignment"]
                     and card["reviewer"] == old_card["reviewer"]
                     and card["png_count"] == old_card["png_count"], "Material/identity/image metadata changed")
        results.append({"packet_id": packet, "judge_id": "deepseek", "stage": "facts",
                        "original_manifest": prep.ref(old_manifest), "original_request": prep.ref(old_request),
                        "new_manifest": prep.ref(manifest_path), "new_prepare": prep.ref(dest / "prepare.json"),
                        "new_request": prep.ref(new_request), "reference_catalog": prep.ref(new_catalog),
                        "catalog_canonical_sha256": card["reference_catalog"]["sha256"],
                        "only_http_change": {"max_tokens": {"old": 32768, "new": 65536}},
                        "complete_request_except_max_tokens_exact": True,
                        "messages_images_order_and_metadata_unchanged": True,
                        "reference_catalog_file_and_canonical_hash_unchanged": True})
    for path, before in originals.items():
        prep.require(prep.ref(Path(path)) == before, "Original evidence changed: " + path)
    prep.verify_source(prep.S, prep.C / "runtime_source_manifest.json")
    prep.require(prep.ref(card_path)["sha256"] == card_sha, "Revision card changed")
    result = {"status": "CPU_PREPARED_NOT_RELEASED", "scope": "same_cohort_engineering_output_revision_not_original_R4_PASS",
              "revision_card": prep.ref(card_path), "script": prep.ref(Path(__file__)),
              "original_export_receipt": prep.ref(export_path), "source_manifest": prep.ref(prep.C / "runtime_source_manifest.json"),
              "original_evidence_checked_unchanged": list(originals.values()), "slots": results,
              "episode_01_excluded": True, "facts_prepared": 7, "reports_prepared": 0, "new_api_calls": 0,
              "facts_seals_created": 0, "executable_release_created": False, "release_drafts_only": True,
              "pending": "Root separately reviews the actual revision/card/budget and creates releases; no dispatch or retry here."}
    prep.save(output / "preparation_receipt.json", result)
    print(json.dumps({"status": result["status"], "facts_prepared": 7,
                      "receipt": prep.ref(output / "preparation_receipt.json")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision-card", required=True, type=Path)
    ap.add_argument("--revision-card-sha256", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    prepare_revision(args.revision_card, args.revision_card_sha256, args.output)
