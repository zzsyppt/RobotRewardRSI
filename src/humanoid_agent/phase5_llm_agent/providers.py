"""LLM providers for Phase 5 proposals."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a response."""


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    model: str
    content: str
    raw_response: dict[str, Any]
    request_summary: dict[str, Any]


def call_provider(
    *,
    provider_name: str,
    messages: list[dict[str, str]],
    phase5_config: dict[str, Any],
    replay_response: str | Path | None = None,
    allow_api_spend: bool = False,
) -> ProviderResult:
    provider_cfg = phase5_config.get("provider", {}).get(provider_name, {})
    provider_type = str(provider_cfg.get("type", provider_name))
    if provider_type == "dry_run":
        return _dry_run_response()
    if provider_type == "replay":
        if replay_response is None:
            raise ProviderError("replay provider requires --replay-response")
        return _replay_response(Path(replay_response), provider_name=provider_name)
    if provider_type == "openai_compatible":
        if not allow_api_spend:
            raise ProviderError(f"{provider_name} provider requires --allow-api-spend to prevent accidental charges")
        return _openai_compatible_response(provider_name=provider_name, messages=messages, config=provider_cfg)
    raise ProviderError(f"Unknown provider type for {provider_name}: {provider_type}")


def _dry_run_response() -> ProviderResult:
    response = {
        "analysis": {
            "hypothesis": "Public failures suggest coupled backward/height/drift instability; propose a small training-only push and existing reward-weight adjustment.",
            "evidence": {
                "source": "dry_run",
                "note": "Deterministic local response for parser/verifier testing; not an LLM result.",
            },
            "patch_summary": [
                "increase training-only phase2 push exposure",
                "increase existing upright/base-height/drift shaping conservatively",
            ],
            "expected_effect": "reduce early collapse without changing evaluation difficulty",
            "risk": "may make the policy stiffer and reduce recovery stepping freedom",
            "rollback_condition": "reject if public mean does not improve beyond noise-aware threshold",
        },
        "patch": {
            "operations": [
                {
                    "op": "set",
                    "document": "phase1",
                    "path": "profiles.phase2_train.push.velocity_range_mps.x",
                    "value": [-0.08, 0.08],
                    "reason": "add mild train-only sagittal push exposure",
                },
                {
                    "op": "set",
                    "document": "phase1",
                    "path": "profiles.phase2_train.push.velocity_range_mps.y",
                    "value": [-0.08, 0.08],
                    "reason": "add mild train-only lateral push exposure",
                },
                {
                    "op": "set",
                    "document": "phase2",
                    "path": "reward_weights.flat_orientation_l2",
                    "value": -9.0,
                    "reason": "slightly increase upright shaping using an existing reward term",
                },
                {
                    "op": "set",
                    "document": "phase2",
                    "path": "reward_weights.base_height",
                    "value": -13.0,
                    "reason": "slightly increase height-collapse penalty using an existing reward term",
                },
                {
                    "op": "set",
                    "document": "phase2",
                    "path": "reward_weights.base_linear_velocity",
                    "value": -2.25,
                    "reason": "reduce excessive drift with an existing velocity penalty",
                },
            ]
        },
    }
    content = json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True)
    return ProviderResult(
        provider="dry_run",
        model="deterministic-local-phase5-dry-run",
        content=content,
        raw_response={"choices": [{"message": {"content": content}}], "dry_run": True},
        request_summary={"provider": "dry_run", "network": False},
    )


def _replay_response(path: Path, *, provider_name: str) -> ProviderResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "choices" in data:
        content = data["choices"][0]["message"]["content"]
        raw = data
    elif isinstance(data, dict):
        content = json.dumps(data, ensure_ascii=False)
        raw = {"choices": [{"message": {"content": content}}], "replay_json_object": True}
    else:
        raise ProviderError(f"Replay file must contain a JSON object: {path}")
    return ProviderResult(
        provider=provider_name,
        model="replayed-response",
        content=content,
        raw_response=raw,
        request_summary={"provider": "replay", "path": str(path), "network": False},
    )


def _openai_compatible_response(
    *,
    provider_name: str,
    messages: list[dict[str, str]],
    config: dict[str, Any],
) -> ProviderResult:
    key_env = str(config.get("api_key_env", "PHASE5_LLM_API_KEY"))
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ProviderError(f"Missing API key environment variable for {provider_name}: {key_env}")

    base_url = _config_value(config, "base_url", default="").rstrip("/")
    if not base_url:
        raise ProviderError(f"Missing base_url for provider: {provider_name}")
    endpoint = str(config.get("endpoint", "/chat/completions"))
    url = base_url + endpoint
    model = _config_value(config, "model", default="")
    if not model:
        raise ProviderError(f"Missing model for provider: {provider_name}")
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": float(config.get("temperature", 0.2)),
        "max_tokens": int(config.get("max_tokens", 8192)),
    }
    if config.get("reasoning_effort") is not None:
        body["reasoning_effort"] = config["reasoning_effort"]
    if config.get("thinking") is not None:
        body["thinking"] = config["thinking"]
    if config.get("response_format") is not None:
        body["response_format"] = config["response_format"]

    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    timeout = int(config.get("timeout_seconds", 120))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured API endpoint.
            raw_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"{provider_name} API HTTP {exc.code}: {error_text[:500]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"{provider_name} API connection failed: {exc}") from exc
    raw = json.loads(raw_text)
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"{provider_name} response did not contain choices[0].message.content") from exc
    return ProviderResult(
        provider=provider_name,
        model=model,
        content=content,
        raw_response=raw,
        request_summary={
            "provider": provider_name,
            "provider_type": "openai_compatible",
            "base_url": base_url,
            "endpoint": endpoint,
            "model": model,
            "api_key_env": key_env,
            "api_key_present": True,
            "network": True,
            "temperature": body.get("temperature"),
            "max_tokens": body.get("max_tokens"),
            "reasoning_effort": body.get("reasoning_effort"),
            "response_format": body.get("response_format"),
            "thinking": body.get("thinking"),
        },
    )


def _config_value(config: dict[str, Any], name: str, *, default: str) -> str:
    env_name = config.get(f"{name}_env")
    if env_name:
        return str(os.environ.get(str(env_name), default))
    return str(config.get(name, default))
