"""CPU-only static token count; no model request, credentials, GPU, or release."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

TOKENIZER_SHA = "89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614"
ZIP_SHA = "e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495"
URL = "https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="Existing tokenizer JSON or ZIP; no network")
    parser.add_argument("--allow-static-download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    if args.tokenizer:
        blob = args.tokenizer.read_bytes()
        acquisition = "existing_local_static_file"
    elif args.allow_static_download:
        import urllib.request
        blob = urllib.request.urlopen(URL, timeout=60).read()
        acquisition = "public_static_tokenizer_download_only"
    else:
        parser.error("provide cached --tokenizer; explicit flag required for static download")
    if digest(blob) == TOKENIZER_SHA:
        tokenizer_bytes = blob
    else:
        assert digest(blob) == ZIP_SHA, "unexpected static tokenizer ZIP"
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            tokenizer_bytes = next(value for name in archive.namelist()
                                   if name.endswith("tokenizer.json")
                                   for value in [archive.read(name)] if digest(value) == TOKENIZER_SHA)
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    count = lambda value: len(tokenizer.encode(value, add_special_tokens=False).ids)
    with zipfile.ZipFile(args.materials) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        material_bytes = {name: archive.read(name) for name in manifest["files"]}
    for name, expected in manifest["files"].items():
        assert digest(material_bytes[name]) == expected["sha256"]
        assert len(material_bytes[name]) == expected["bytes"]
    rows = []
    for variant, filename in manifest["requests"].items():
        payload = json.loads(material_bytes[filename])
        assert payload["max_tokens"] == 32768
        parts = [{"role": "system", "tokens": count(payload["messages"][0]["content"])}]
        for block in payload["messages"][1]["content"]:
            assert block["type"] == "text"
            parts.append({"role": json.loads(block["text"])["role"], "tokens": count(block["text"])})
        total = sum(part["tokens"] for part in parts)
        rows.append({"variant": variant, "request_sha256": digest(material_bytes[filename]),
                     "request_bytes": len(material_bytes[filename]), "by_role": parts,
                     "input_proxy_tokens": total, "target_180000_pass": total <= 180000,
                     "conservative_reserved_total": total + 32768 + 32768 + 8192,
                     "fits_258400_with_conservative_reserves": total + 32768 + 32768 + 8192 <= 258400})
    assert next(row for row in rows if row["variant"] == "r3_synthetic_baseline")["input_proxy_tokens"] == 195968
    result = {"status": "OLD_DEVELOPMENT_STATIC_LENGTH_ONLY_NOT_N5_RELEASE", "actual_model_calls": 0,
              "gpu_runs": 0, "tokenizer_url": URL, "tokenizer_sha256": digest(tokenizer_bytes),
              "tokenizer_acquisition": acquisition, "materials_zip_sha256": digest(args.materials.read_bytes()),
              "method": "Complete serialized text blocks, add_special_tokens=False. DeepSeek tokenizer is only a proxy for SOL.",
              "reserves_unchanged": {"material_target": 180000, "system_tools": 32768,
                                     "output": 32768, "margin": 8192, "effective_context": 258400},
              "original_answer_tokens": count(material_bytes["actual_facts.json"].decode("utf-8")),
              "records": rows, "actual_agent_input_peak_tokens": None, "agent_compaction_observed": None,
              "full_payload_context_capacity_demonstrated": False,
              "facts_protocol_compliance_established": False, "facts_semantic_completeness_established": False,
              "answer_size_is_upper_bound_for_other_packets": False}
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
