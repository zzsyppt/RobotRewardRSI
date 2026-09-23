"""Measure provider tokenizer usage for bounded P5 frames, with one output token."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig
    from true_rl_agent.prompt_serialization import canonical_prompt_json, canonical_prompt_json_bytes
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--call", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = {path.parent.name + "_" + path.stem: json.loads(path.read_text()) for path in args.source}
    source_digests = {path.parent.name + "_" + path.stem: hashlib.sha256(path.read_bytes()).hexdigest() for path in args.source}
    declared = json.loads(args.policy.read_text())
    live = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(model=declared["model"], output_dir=args.output / "unused_candidates",
        base_url="https://api.deepseek.com", env_file=Path("/data/yuyangliu/secrets/deepseek.env"), api_key_env="DEEPSEEK_API_KEY",
        max_tokens=1, timeout_s=120, reasoning_mode="disabled", model_context=declared["model_context"]))
    report = {"purpose": "Measure actual historical reconstructions and explicitly declared branch frames; no preflight stand-in for late history",
        "measurement": "Same serialized message objects; thinking disabled, one output token. Actual candidate generation is a separate probe.",
        "reserved_candidate_output_tokens": declared["max_tokens"], "frames": {}}
    for name, frame in frames.items():
        (args.output / (name + ".json")).write_text(canonical_prompt_json(frame))
        item = {"source_sha256": source_digests[name], "prompt_bytes": canonical_prompt_json_bytes(frame), "frame_kind": frame["authoring_context_frame"]["frame_kind"]}
        if args.call:
            request = live._chat_completion_request(frame)
            response = live.transport.post_json(url="https://api.deepseek.com/chat/completions", payload=request,
                api_key=live._api_key(), timeout_s=120)
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            assert isinstance(prompt_tokens, int) and prompt_tokens > 0
            item.update(usage=usage, response_model=response.get("model"), finish_reason=response["choices"][0]["finish_reason"],
                input_plus_reserved_output=prompt_tokens + declared["max_tokens"],
                within_declared_window=prompt_tokens + declared["max_tokens"] <= declared["model_context"]["context_window_tokens"])
            assert item["within_declared_window"]
        report["frames"][name] = item
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, item, flush=True)


if __name__ == "__main__":
    main()
