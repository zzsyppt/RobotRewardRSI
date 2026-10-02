"""Reproduce one old-development report display change; no model call or GPU.

The remote CPU measurement reads the public static tokenizer ZIP in memory and
checks its frozen hashes. It never reads credentials or writes remote files.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
HELPERS = HERE.parents[1]
OLD = HERE.parent / "n5_review_capacity_development_r2"
sys.path.insert(0, str(HELPERS))
spec = importlib.util.spec_from_file_location("event_table_capacity_packer", HELPERS / "prepare_confirmation_materials.py")
packer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packer)
runner = packer.runner


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def main():
    old_packet = OLD / "packets/episode-02"
    manifest_path = old_packet / "report.capacity.manifest.json"
    manifest = read(manifest_path)
    old_slice_path = old_packet / "full_episode.json"
    old_slice = read(old_slice_path)
    source_path = Path(old_slice["source"]["full_file"])
    assert sha(source_path) == old_slice["source"]["full_sha256"]
    original_full = read(source_path)
    source_row = original_full
    for part in old_slice["source_pointer"].strip("/").split("/"):
        source_row = source_row[int(part)] if isinstance(source_row, list) else source_row[part]
    original_row = old_slice["complete_episode_derivation"]
    assert canonical(source_row) == canonical(original_row)
    assert len(original_row["events"]) == 620 and original_row["steps"] == 525
    watched = {source_path, manifest_path, old_slice_path, old_packet / "facts.manifest.json"}
    for item in manifest["materials"]:
        watched.add((old_packet / item["file"]).resolve())
    for judge in ("deepseek", "gpt6-sol"):
        watched.add(old_packet / f"{judge}.synthetic_capacity_facts.json")
        watched.add(OLD / "prepared_report_capacity/episode-02" / judge / "request.http_body.json")
    before = {str(path): sha(path) for path in watched}

    new_slice = deepcopy(old_slice)
    new_slice["complete_episode_derivation"] = packer.review_episode_derivation(
        original_row, task="stair", material_profile="related_v1")
    reconstructed = packer.expand_review_episode_derivation(new_slice["complete_episode_derivation"])
    assert canonical(reconstructed) == canonical(original_row)
    new_packet = HERE / "packets/episode-02"
    new_slice_path = new_packet / "full_episode.json"
    save(new_slice_path, new_slice)
    for item in manifest["materials"]:
        if item["id"] == "complete-episode-full":
            item["file"] = "full_episode.json"
            item["sha256"] = sha(new_slice_path)
        else:
            item["file"] = str((old_packet / item["file"]).resolve())
    new_manifest = new_packet / "report.capacity.manifest.json"
    save(new_manifest, manifest)

    requests = []
    prepared = []
    for judge in ("deepseek", "gpt6-sol"):
        fixture = old_packet / f"{judge}.synthetic_capacity_facts.json"
        destination = HERE / "prepared_report_capacity/episode-02" / judge
        card = runner.prepare(new_manifest, judge, destination, capacity_facts_fixture=fixture)
        old_request = read(OLD / "prepared_report_capacity/episode-02" / judge / "request.http_body.json")
        new_request = read(destination / "request.http_body.json")
        old_blocks, new_blocks = old_request["messages"][1]["content"], new_request["messages"][1]["content"]
        assert len(old_blocks) == len(new_blocks)
        changed = []
        for index, (left, right) in enumerate(zip(old_blocks, new_blocks)):
            if left != right:
                changed.append(index)
                left_data, right_data = json.loads(left["text"]), json.loads(right["text"])
                assert left_data["role"] == right_data["role"] == "full_derivation"
                right_data["data"]["complete_episode_derivation"] = packer.expand_review_episode_derivation(
                    right_data["data"]["complete_episode_derivation"])
                assert right_data["data"] == left_data["data"]
        assert len(changed) == 1
        assert {k: v for k, v in old_request.items() if k != "messages"} == {
            k: v for k, v in new_request.items() if k != "messages"}
        assert old_request["messages"][0] == new_request["messages"][0]
        requests.extend([{"variant": "old_r2", "judge": judge, "payload": old_request},
                         {"variant": "event_table_r3", "judge": judge, "payload": new_request}])
        prepared.append({"judge": judge, "request_sha256": card["request_sha256"],
                         "http_body_bytes": card["request_bytes"], "status": card["status"]})

    remote = r'''
import hashlib, io, json, sys, urllib.request, zipfile
from tokenizers import Tokenizer
records = json.load(sys.stdin)
url = "https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip"
blob = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(blob).hexdigest() == "e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495"
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    candidates = [archive.read(name) for name in archive.namelist() if name.endswith("tokenizer.json")]
    text = next(value for value in candidates if hashlib.sha256(value).hexdigest() == "89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614")
tokenizer = Tokenizer.from_str(text.decode())
rows = []
for record in records:
    payload = record["payload"]
    parts = [{"role": "system", "tokens": len(tokenizer.encode(payload["messages"][0]["content"], add_special_tokens=False).ids)}]
    for block in payload["messages"][1]["content"]:
        assert block["type"] == "text"
        parts.append({"role": json.loads(block["text"])["role"], "tokens": len(tokenizer.encode(block["text"], add_special_tokens=False).ids)})
    total = sum(part["tokens"] for part in parts)
    rows.append({"variant": record["variant"], "judge": record["judge"], "input_proxy_tokens": total,
                 "by_role": parts, "target_180000_pass": total <= 180000,
                 "conservative_reserved_total": total + 32768 + 32768 + 8192,
                 "fits_258400_with_conservative_reserves": total + 32768 + 32768 + 8192 <= 258400})
print(json.dumps({"tokenizer_url": url, "tokenizer_sha256": hashlib.sha256(text).hexdigest(), "records": rows}))
'''
    import base64
    # The code is shell-quoted independently; only JSON request data goes to stdin.
    code = "import base64;exec(base64.b64decode('" + base64.b64encode(remote.encode()).decode() + "'))"
    import shlex
    command = "PYTHONDONTWRITEBYTECODE=1 /data/yuyangliu/envs/isaaclab6_validation/bin/python -c " + shlex.quote(code)
    completed = subprocess.run(["ssh", "humanoid-server", command], input=json.dumps(requests).encode(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=180)
    measurement = json.loads(completed.stdout)
    baseline = {row["judge"]: row["input_proxy_tokens"] for row in measurement["records"] if row["variant"] == "old_r2"}
    assert baseline == {"deepseek": 205153, "gpt6-sol": 205160}
    save(HERE / "tokenizer_check.json", {"status": "OLD_DEVELOPMENT_CPU_ONLY_NOT_N5_RELEASE", "actual_model_calls": 0,
         "method": "Same frozen official DeepSeek v4 tokenizer, every actual text block, add_special_tokens=False; proxy only for SOL, not actual usage.",
         "reserves_unchanged": {"material_target": 180000, "system_tools": 32768, "output": 32768, "margin": 8192, "effective_context": 258400},
         **measurement})
    assert before == {str(path): sha(path) for path in watched}
    verification = {"status": "LOSSLESS_EVENT_DISPLAY_VERIFIED_CAPACITY_NOT_RELEASED", "actual_model_calls": 0,
        "gpu_runs": 0, "confirmation_data_read": False, "original_episode_steps": 525, "event_count": 620,
        "complete_episode_canonical_bytes_equal_after_expansion": True,
        "original_events_canonical_sha256": hashlib.sha256(canonical(original_row["events"])).hexdigest(),
        "original_full_row_canonical_sha256": hashlib.sha256(canonical(original_row)).hexdigest(),
        "old_files_unchanged": before, "only_changed_request_material_role": "full_derivation",
        "all_other_text_blocks_byte_equal": True, "same_output_settings": True,
        "original_compact_event_characters": len(canonical(original_row["events"])),
        "new_compact_event_table_characters": len(canonical(new_slice["complete_episode_derivation"]["events"])),
        "prepared": prepared, "preparer_sha256": sha(HELPERS / "prepare_confirmation_materials.py"),
        "runner_sha256": sha(HELPERS / "run_confirmation_review.py"), "verification_script_sha256": sha(Path(__file__))}
    save(HERE / "verification.json", verification)
    print(json.dumps({"prepared": prepared, "tokens": measurement["records"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
