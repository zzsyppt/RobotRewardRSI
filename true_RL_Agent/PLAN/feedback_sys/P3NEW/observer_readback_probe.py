"""One frozen synthetic image readback; no robot or semantic qualification."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(root, output):
    sys.path.insert(0, str(root / "src"))
    from true_rl_agent.evaluation.visual_observer import render_frame_grid, audit_request_text
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for step, angle in zip((17, 203, 411, 809), (0, 30, 60, 90)):
        width, height = 256, 160
        pixels = bytearray(b"\xff" * (width * height * 3))
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        for y in range(height):
            for x in range(width):
                dx, dy = x - width / 2, y - height / 2
                if abs(dx * c + dy * s) <= 80 and abs(-dx * s + dy * c) <= 15:
                    i = (y * width + x) * 3
                    pixels[i:i+3] = bytes((20, 75, 220))
        frames.append((step, width, height, bytes(pixels)))
    image = render_frame_grid(frames, columns=2)
    (output / "readback.png").write_bytes(image.data)
    payload = {"model": "deepseek-flash", "max_tokens": 2048, "temperature": 0,
        "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Read this image. Return JSON with panels in row-major order. For each panel, give the printed integer label, the shape color, and its orientation (horizontal, slants down-right, or vertical). Read labels from pixels; do not invent missing text. Add unreadable_labels if needed."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image.data).decode(), "detail": "high"}}]}]}
    request = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    (output / "request.json").write_bytes(request)
    audit = audit_request_text(request)
    assert audit["passed"]
    source_paths = [root / "src/true_rl_agent/evaluation/visual_observer.py",
                    root / "src/true_rl_agent/agent_core/openai_compatible_provider.py"]
    card = {"schema_version": "feedback.p3new.observer_readback.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ONE_DEVELOPMENT_CALL_RELEASED", "purpose": "synthetic renderer/remote label and shape readback only",
        "not_robot_evidence": True, "not_confirmation": True, "max_actual_calls": 1, "automatic_retries": 0,
        "endpoint": "https://api.deepseek.com/chat/completions", "model": payload["model"],
        "generation": {"thinking": "disabled", "max_tokens": 2048},
        "reserved_usd": 0.01, "parent_observer_budget_usd": 3.5, "phase_budget_usd": 15,
        "reservation_assumption": {"input_tokens_ceiling": 10000, "peak_input_miss_usd_per_million": .3,
            "peak_output_usd_per_million": 1.2, "computed_peak_ceiling_usd": .0054576},
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/",
        "thinking_source": "https://api-docs.deepseek.com/guides/thinking_mode/",
        "request_sha256": sha(request), "request_bytes": len(request), "blind_text_audit": audit,
        "image_sha256": sha(image.data), "width": image.width, "height": image.height,
        "image_count": 1, "raw_subframes": 4, "pixels": image.width * image.height,
        "expected_not_sent": {"labels": [17, 203, 411, 809], "color": "blue",
                              "orientation": ["horizontal", "slants down-right", "slants down-right", "vertical"]},
        "source_hashes": {str(p.relative_to(root)).replace('\\', '/'): sha(p.read_bytes()) for p in source_paths}}
    dump(output / "card.json", card)
    print(json.dumps({"prepared": str(output), "request_sha256": card["request_sha256"], "reserved_usd": .01}))


def send(root, packet, secret):
    sys.path.insert(0, str(root / "src"))
    from true_rl_agent.agent_core.openai_compatible_provider import UrllibJsonTransport
    card = json.loads((packet / "card.json").read_text(encoding="utf-8"))
    request = (packet / "request.json").read_bytes()
    assert sha(request) == card["request_sha256"]
    assert all(sha((root / name).read_bytes()) == value for name, value in card["source_hashes"].items())
    receipt_path = packet / "receipt.json"
    if receipt_path.exists():
        raise RuntimeError("one-call probe already reserved; inspect its receipt instead of resubmitting")
    key = None
    for line in secret.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() == "DEEPSEEK_API_KEY":
            key = value.strip().strip('\"').strip("'")
    if not key:
        raise RuntimeError("configured DeepSeek key missing")
    receipt = {"request_sha256": sha(request), "card_sha256": sha((packet / "card.json").read_bytes()),
               "status": "reserved", "actual_call_n": 1, "reserved_usd": card["reserved_usd"],
               "started_utc": datetime.now(timezone.utc).isoformat()}
    dump(receipt_path, receipt)
    try:
        result = UrllibJsonTransport().post_json(url=card["endpoint"], payload=json.loads(request), api_key=key, timeout_s=60)
    except Exception as error:
        receipt.update(status="error", error_type=type(error).__name__, cost_status="reserved_unknown")
        dump(receipt_path, receipt)
        raise RuntimeError(receipt["error_type"]) from None
    receipt.update(status="received", ended_utc=datetime.now(timezone.utc).isoformat(), response=result,
                   response_model=result.get("model"), fingerprint=result.get("system_fingerprint"), usage=result.get("usage"))
    usage = result.get("usage", {})
    hit = usage.get("prompt_cache_hit_tokens", usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        receipt["peak_cost_ceiling_usd"] = (hit * .006 + miss * .3 + usage["completion_tokens"] * 1.2) / 1_000_000
        receipt["off_peak_cost_usd"] = receipt["peak_cost_ceiling_usd"] / 2
    dump(receipt_path, receipt)
    print(json.dumps({k: receipt.get(k) for k in ("status", "response_model", "fingerprint", "usage", "peak_cost_ceiling_usd")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "send"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--secret", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root.resolve(), args.packet.resolve())
    else:
        if args.secret is None:
            parser.error("send requires the configured secret file path")
        send(args.root.resolve(), args.packet.resolve(), args.secret)
