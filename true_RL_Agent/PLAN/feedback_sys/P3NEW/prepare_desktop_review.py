"""Lossless prepare-only carrier adapter for one fresh SOL review context."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys



def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def encoded(value): return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"file": str(path), "sha256": sha(raw), "bytes": len(raw)}


def frozen_review(source_root):
    frozen = Path(source_root).resolve()
    review_file = frozen / "PLAN/feedback_sys/P3NEW/run_confirmation_review.py"
    # Reuse existing assignment, envelope and material reconstruction functions.
    # This never calls build/prepare/run or reads a raw-material file.
    sys.path[:0] = [str(review_file.parent), str(frozen / "src")]
    spec = importlib.util.spec_from_file_location("p3new_r3_desktop_frozen_review", review_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.coverage.__file__).resolve() == review_file.with_name("confirmation_review_coverage.py"),
            "coverage must come from the frozen source")
    return module


def blocks(messages):
    ordinal = 0
    for message_index, message in enumerate(messages):
        content = message["content"]
        rows = [{"type": "text", "text": content}] if isinstance(content, str) else content
        require(isinstance(rows, list), "unsupported message content")
        for block_index, block in enumerate(rows):
            yield ordinal, message_index, block_index, block
            ordinal += 1


def adapt_messages(messages):
    """Preserve every message/text; replace each inline PNG at its original position."""
    adapted, images, order, text_records = deepcopy(messages), [], [], []
    for ordinal, mi, bi, block in blocks(messages):
        position = {"ordinal": ordinal, "message_index": mi, "block_index": bi}
        if block.get("type") == "text":
            require(isinstance(block.get("text"), str), "text block is not a string")
            raw = block["text"].encode("utf-8")
            order.append({**position, "type": "text"})
            text_records.append({**position, "sha256": sha(raw), "utf8_bytes": len(raw)})
            continue
        require(set(block) == {"type", "image_url"} and block["type"] == "image_url", "unsupported content block")
        url = block["image_url"].get("url", "")
        prefix = "data:image/png;base64,"
        require(url.startswith(prefix), "only inline original PNG is supported")
        raw = base64.b64decode(url[len(prefix):], validate=True)
        require(raw.startswith(b"\x89PNG\r\n\x1a\n") and base64.b64encode(raw).decode("ascii") == url[len(prefix):],
                "invalid or noncanonical PNG carrier")
        require(bi > 0 and isinstance(messages[mi]["content"], list), "PNG lacks adjacent label")
        previous = messages[mi]["content"][bi - 1]
        require(previous.get("type") == "text", "PNG label must be the immediately preceding text")
        label = json.loads(previous["text"])
        require(label.get("role") == "raw_image" and isinstance(label.get("material_id"), str)
                and label.get("sha256") == sha(raw), "PNG adjacent label identity/hash differs")
        relative = f"images/image-{len(images):04d}.png"
        original_attributes = {key: value for key, value in block["image_url"].items() if key != "url"}
        replacement = {"type": "image_file", "image_file": {"file": relative, "sha256": sha(raw),
                       "original_image_url_attributes": original_attributes}}
        adapted[mi]["content"][bi] = replacement
        record = {**position, "image_index": len(images), "file": relative, "material_id": label["material_id"],
                  "sha256": sha(raw), "bytes": len(raw)}
        images.append((record, raw))
        order.append({**position, "type": "image_file"})
    return {"schema_version": "p3new.desktop_messages.v1", "messages": adapted, "block_order": order}, images, text_records


def reconstruct_messages(input_path):
    """Read only the adapted input and its PNG whitelist; rebuild original messages."""
    input_path = Path(input_path).resolve()
    carrier = read(input_path)
    require(carrier.get("schema_version") == "p3new.desktop_messages.v1", "unknown desktop schema")
    messages = deepcopy(carrier["messages"])
    order = []
    for ordinal, mi, bi, block in blocks(messages):
        order.append({"ordinal": ordinal, "message_index": mi, "block_index": bi, "type": block["type"]})
        if block["type"] == "text":
            continue
        require(block["type"] == "image_file", "unknown adapted block")
        record = block["image_file"]
        relative = PurePosixPath(record["file"])
        require(not relative.is_absolute() and ".." not in relative.parts and relative.parts[0] == "images",
                "image reference is outside the blind directory")
        path = input_path.parent.joinpath(*relative.parts).resolve()
        require(path.is_relative_to(input_path.parent), "image resolved outside the blind directory")
        raw = path.read_bytes()
        require(sha(raw) == record["sha256"], "adapted PNG changed")
        messages[mi]["content"][bi] = {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
            **record["original_image_url_attributes"]}}
    require(order == carrier["block_order"], "message/block sequence changed")
    return messages


def prepare(prepared, output, *, source_root, cli_policy, assignment_file=None, own_facts_seal=None, own_facts_answer=None, allow_development_single=False):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    require(not output.exists(), "output must be new; never overwrite a prior context")
    review = frozen_review(source_root)
    review_file = Path(review.__file__).resolve()
    policy = read(cli_policy)
    require(policy.get("schema_version") == "p3new.sol_cli_policy.v1" and policy.get("status") == "FROZEN"
            and policy.get("model") == "gpt-6.1-sol" and policy.get("reasoning_effort") == "high", "wrong CLI policy")
    window, compact = policy.get("model_context_window"), policy.get("model_auto_compact_token_limit")
    require(type(window) is int and type(compact) is int and 0 < compact <= window, "invalid CLI window policy")
    card_path, request_path = prepared / "prepare.json", prepared / "request.http_body.json"
    initial_card_ref = ref(card_path)
    card, body = read(card_path), request_path.read_bytes()
    require(card.get("schema_version") == "p3new.confirmation_review_prepared.v1"
            and card.get("status") == "PREPARED_NOT_RELEASED"
            and not card.get("capacity_facts_fixture") and not card.get("synthetic_only"), "genuine prepared input required")
    require(sha(body) == card["request_sha256"] and len(body) == card["request_bytes"], "request SHA/size differs")
    require(len(body) <= review.MAX_BODY == card["body_limit_bytes"] and card["max_png_count"] == review.MAX_IMAGES,
            "prepared byte/image ceiling differs")
    payload = json.loads(body)
    require(encoded(payload) == body, "request serialization differs from frozen format")
    require(card["max_output_tokens"] == payload["max_tokens"] == 32768, "R3 output reservation differs")
    judge, packet, stage = card["judge_id"], card["packet_id"], card["stage"]
    require(stage in ("facts", "report") and isinstance(packet, str) and packet, "invalid packet/stage")
    development = judge == "gpt6.1-sol" and allow_development_single
    require(judge in review.SOL_PAIR or development, "formal identity must be SOL-a or SOL-b")
    assignment_verified = None
    if not development:
        require(assignment_file is not None, "formal transport needs the local frozen assignment")
        actual = review.assignment_fields(assignment_file, packet)["review_assignment"]
        binding = card["review_assignment"]
        require(actual["source"]["sha256"] == binding["source"]["sha256"]
                and actual["packet_id"] == binding["packet_id"] and actual["reviewers"] == binding["reviewers"], "assignment binding differs")
        matches = [item for item in actual["reviewers"].values() if item["judge_id"] == judge]
        require(len(matches) == 1 and matches[0] == card["reviewer"] and matches[0]["carrier"] == "fresh_cli",
                "prepared reviewer is not the assigned fresh CLI identity")
        require(card["runner"]["sha256"] == ref(review_file)["sha256"]
                and card["coverage_helper"]["sha256"] == ref(review.coverage.__file__)["sha256"], "prepared source differs from explicit source root")
        assignment_verified = ref(assignment_file)
    messages = payload["messages"]
    require(len(messages) == 2 and messages[0]["role"] == "system" and isinstance(messages[0]["content"], str)
            and messages[1]["role"] == "user" and isinstance(messages[1]["content"], list), "unexpected frozen message structure")
    instruction = messages[0]["content"]
    require(instruction.startswith(f"You are independent judge {judge} for packet {packet}, stage {stage}. "), "message identity differs")
    sources = {item["id"]: item for item in card["materials"]}
    require(len(sources) == len(card["materials"]), "duplicate prepared source identity")
    material_envelopes, _ = review.coverage.request_materials(payload)
    require(set(material_envelopes) == set(sources), "message/source material list differs")
    for material_id, envelope in material_envelopes.items():
        require(envelope["sha256"] == sources[material_id]["sha256"] and envelope["role"] == sources[material_id]["role"], "message source hash/role differs")
    own = []
    for _, _, _, block in blocks(messages):
        if block["type"] == "text":
            try: item = json.loads(block["text"])
            except ValueError: continue
            if isinstance(item, dict) and item.get("role") == "own_sealed_facts": own.append(item)
    seal_verified = answer_verified = None
    if stage == "facts":
        require(card["own_facts_seal"] is None and own_facts_seal is None and own_facts_answer is None and not own, "facts cannot carry answers")
    else:
        require(own_facts_seal is not None and len(own) == 1, "report needs its own bound seal")
        seal_verified = ref(own_facts_seal)
        require(seal_verified["sha256"] == card["own_facts_seal"]["sha256"], "own facts seal SHA differs")
        seal = read(own_facts_seal)
        require(seal["schema_version"] == "p3new.confirmation_facts_seal.v1" and seal["status"] == "SEALED"
                and seal["judge_id"] == judge and seal["packet_id"] == packet, "seal belongs to another identity/packet")
        require(seal["answer"]["sha256"] == own[0]["sha256"] == sources["own_sealed_facts"]["sha256"]
                and own[0]["judge_id"] == judge and review.envelope(own[0]["data"], judge, packet, "facts")
                and own[0]["data"]["complete"], "inline facts do not match own seal")
        answer_path = Path(own_facts_answer) if own_facts_answer is not None else Path(seal["answer"]["file"])
        if not answer_path.is_absolute():
            answer_path = Path(own_facts_seal).resolve().parent / answer_path
        answer_verified = ref(answer_path)
        require(answer_verified["sha256"] == seal["answer"]["sha256"]
                and read(answer_path) == own[0]["data"], "sealed answer content differs from inline facts")
        require(seal.get("reference_contract") == card.get("reference_contract"), "seal reference contract differs")
        if card.get("reference_contract"):
            require(seal["reference_catalog"]["sha256"] == card["own_facts_reference_catalog"]["sha256"], "own facts catalog binding differs")
    adapted, images, text_records = adapt_messages(messages)
    require(len(images) == card["png_count"] <= review.MAX_IMAGES, "PNG count differs")
    for item, raw in images:
        source = sources[item["material_id"]]
        require(source["role"] == "raw_image" and source["sha256"] == item["sha256"]
                and source["size_bytes"] == len(raw), "PNG differs from prepared source")
    blind = output / "blind"
    blind.mkdir(parents=True)
    for item, raw in images:
        target = blind / item["file"]
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(raw)
    input_path = blind / "input.json"
    input_path.write_bytes(encoded(adapted))
    reconstructed = reconstruct_messages(input_path)
    require(encoded(reconstructed) == encoded(messages), "messages do not reconstruct exactly")
    restored_payload = {**payload, "messages": reconstructed}
    require(encoded(restored_payload) == body, "original request cannot be rebuilt exactly")
    require(ref(card_path) == initial_card_ref and ref(request_path)["sha256"] == sha(body), "prepared inputs changed while reading")
    receipt = {"schema_version": "p3new.desktop_review_preparation.v1", "status": "PREPARED_TRANSPORT_ONLY_NOT_RELEASED",
        "created_utc": datetime.now(timezone.utc).isoformat(), "development_only": development,
        "judge_id": judge, "packet_id": packet, "stage": stage, "configured_model": "gpt-6.1-sol", "reasoning_effort": "high", "carrier": "fresh_cli", "new_thread": True,
        "source_prepare": ref(card_path), "source_request": ref(request_path), "source_manifest": card["manifest"],
        "declared_material_sources": card["materials"], "prepared_runner": card["runner"], "prepared_transport": card["transport"],
        "frozen_runner": ref(review_file), "frozen_coverage": ref(review.coverage.__file__), "adapter": ref(__file__),
        "review_assignment": card.get("review_assignment"), "assignment_verified_local": assignment_verified,
        "own_facts_seal": card["own_facts_seal"], "own_facts_seal_verified_local": seal_verified,
        "own_facts_answer_verified_local": answer_verified,
        "blind_input": ref(input_path), "child_read_whitelist": [ref(input_path)] + [ref(blind / item["file"]) for item, _ in images],
        "text_blocks": text_records, "text_utf8_bytes": sum(item["utf8_bytes"] for item in text_records),
        "images": [item for item, _ in images], "messages_reconstructed_exactly": True, "request_reconstructed_exactly": True,
        "original_text_utf8_unchanged": True, "original_png_bytes_unchanged": True, "http_envelope_supplied_to_child": False,
        "actual_image_view_required": "Root must require view_image detail=original in message/block order and audit actual tool trace; this preparation does not claim vision consumption.",
        "capacity": {"body_limit_bytes": review.MAX_BODY, "png_limit": review.MAX_IMAGES, "output_reservation": 32768,
                     "cli_policy": ref(cli_policy), "configured_context_window": window, "configured_auto_compact_threshold": compact, "tokenizer_count_performed": False, "actual_capacity_release": "PENDING external actual-packet capacity checks; 360000 proxy is not a guarantee"},
        "actual_calls": 0, "other_answers_read": False, "answers_written_or_sealed": False, "semantic_score": None,
        "limits": "Transport only. Root owns fresh-context launch, whitelist, raw answer/access trace, independent release gates and all-80-facts barrier. Filesystem isolation is not claimed."}
    (output / "transport_receipt.json").write_bytes(encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cli-policy", type=Path, required=True)
    parser.add_argument("--review-assignment", type=Path)
    parser.add_argument("--own-facts-seal", type=Path)
    parser.add_argument("--own-facts-answer", type=Path, help="Local copy only when the sealed answer path is remote; SHA must match the own seal")
    parser.add_argument("--allow-development-single", action="store_true")
    args = parser.parse_args()
    result = prepare(args.prepared, args.output, source_root=args.source_root, cli_policy=args.cli_policy, assignment_file=args.review_assignment,
                     own_facts_seal=args.own_facts_seal, own_facts_answer=args.own_facts_answer,
                     allow_development_single=args.allow_development_single)
    print(json.dumps({key: result[key] for key in ("status", "judge_id", "packet_id", "stage", "actual_calls")}))


if __name__ == "__main__":
    main()
