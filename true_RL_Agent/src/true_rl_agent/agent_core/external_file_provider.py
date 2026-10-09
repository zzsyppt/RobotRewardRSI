"""File handoff to an external author, using the normal typed authoring loop."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from true_rl_agent.agent_core.authoring_payload import AuthoringPayloadMaterializer, authoring_system_prompt
from true_rl_agent.agent_core.authoring_response import SourceRequestResponse, build_source_request_denied_error
from true_rl_agent.agent_core.provider_errors import EXTERNAL_RECEIPT_SCHEMA, ProviderCandidateError
from true_rl_agent.agent_core.provider_limits import DEFAULT_PROVIDER_MAX_RESPONSE_BYTES
from true_rl_agent.fs_contract import create_once_bytes
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object
from true_rl_agent.memory import PublicPromptBuild
from true_rl_agent.prompt_artifacts import verify_prompt_payload
from true_rl_agent.prompting import assert_public_only_prompt_payload, serialize_prompt_payload

EXTERNAL_PROVIDER_BACKEND = "external_file"


class ExternalAuthorWaiting(TimeoutError):
    """A published, journal-bound request is still awaiting its external response."""

    def __init__(self, request_dir: Path):
        self.request_dir = request_dir
        super().__init__("external author response is pending for the original request")


@dataclass(frozen=True)
class ExternalFileProviderConfig:
    output_dir: Path
    model: str
    max_tokens: int
    timeout_s: int
    max_response_bytes: int = DEFAULT_PROVIDER_MAX_RESPONSE_BYTES
    resume_pending: bool = False

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("external author requires an explicit requested model")
        for name in ("max_tokens", "timeout_s", "max_response_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"external author {name} must be a positive integer")
        if type(self.resume_pending) is not bool:
            raise TypeError("external pending resume policy must be boolean")


class ExternalFileCandidateProvider:
    """Publish system/user files; the executor atomically delivers response.txt.

    The optional exchange callback is an IO seam for CPU fixtures or a host
    executor. This provider never launches a model, subprocess or network call.
    """

    def __init__(self, config: ExternalFileProviderConfig, *, exchange: Callable[[Path], None] | None = None):
        self.config = config
        self.exchange = exchange
        self._materializer = AuthoringPayloadMaterializer(config.output_dir, backend=EXTERNAL_PROVIDER_BACKEND)
        self._attempt_handle = None
        self._prompt_build = None
        self._resumed_request = None

    def bind_authoring_attempt(self, handle, prompt_build):
        if self.config.resume_pending:
            self._attempt_handle, self._prompt_build = handle, prompt_build
            self._resumed_request = None

    def resume_authoring_attempt(self, handle):
        if not self.config.resume_pending:
            return None
        attempt_dir = Path(handle.result_path).parent
        binding = load_strict_json_object(attempt_dir / "external_request_binding.json")
        request_dir = (self.config.output_dir / "requests" / f"{handle.sequence:03d}").resolve()
        expected_attempt = {"round_id":handle.round_id, "attempt_id":handle.attempt_id,
                            "sequence":handle.sequence, "prompt_payload":handle.prompt_payload_ref.to_public_dict()}
        if binding["authoring_attempt"] != expected_attempt or binding["request_dir"] != str(request_dir):
            raise ValueError("external request/journal binding mismatch")
        request = load_strict_json_object(request_dir / "request.json")
        if request != binding["request"] or request["authoring_attempt"] != expected_attempt:
            raise ValueError("external published request differs from journal binding")
        waiting = load_strict_json_object(attempt_dir / "external_author_waiting.json")
        if waiting != expected_attempt:
            raise ValueError("external request has no matching timeout handoff")
        payload = verify_prompt_payload(handle.prompt_payload_ref)
        user = serialize_prompt_payload(payload).encode("utf-8")
        if ((request_dir / "user.txt").read_bytes() != user or request["user_sha256"] != _hash(user)
                or request["system_sha256"] != _hash((request_dir / "system.txt").read_bytes())
                or request["requested_model"] != self.config.model
                or request["requested_output_units"] != self.config.max_tokens
                or request["response_max_bytes"] != self.config.max_response_bytes):
            raise ValueError("external request prompt/config binding mismatch")
        # A materialized candidate without an attempt result is a different crash
        # boundary. Do not parse again or allocate a replacement candidate number.
        if any(self.config.output_dir.glob(f"{handle.sequence:03d}_*")):
            raise RuntimeError("external candidate already materialized without completed journal; resume unsupported")
        build = PublicPromptBuild(round_public_context=binding["round_public_context"],
            provider_payload=payload,
            parent_program_source_checksum=payload["candidate_lineage"].get("parent_program_source_checksum"))
        self._attempt_handle, self._prompt_build = handle, build
        self._resumed_request = (request_dir, request)
        return build

    def next_candidate(self, *, prompt_payload):
        response = self.next_authoring_response(prompt_payload=prompt_payload)
        if isinstance(response, SourceRequestResponse):
            raise build_source_request_denied_error(response)
        return response.candidate

    def next_authoring_response(self, *, prompt_payload):
        assert_public_only_prompt_payload(prompt_payload)
        if self._resumed_request is not None:
            request_dir, request = self._resumed_request
            self._materializer.begin_attempt(sequence=self._attempt_handle.sequence)
            return self._read_response(request_dir, request, prompt_payload)
        handle = self._attempt_handle
        self._materializer.begin_attempt(sequence=handle.sequence if handle is not None else None)
        root = self.config.output_dir / "requests"
        root.mkdir(parents=True, exist_ok=True)
        number = (handle.sequence if handle is not None else
                  max((int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdecimal()), default=0) + 1)
        request_dir = root / f"{number:03d}"
        request_dir.mkdir()
        system = authoring_system_prompt(backend=EXTERNAL_PROVIDER_BACKEND).encode("utf-8")
        user = serialize_prompt_payload(prompt_payload).encode("utf-8")
        (request_dir / "system.txt").write_bytes(system)
        (request_dir / "user.txt").write_bytes(user)
        request = {
            "schema_version": "true_rl_agent.external_author_request.v1",
            "backend": EXTERNAL_PROVIDER_BACKEND, "requested_model": self.config.model,
            "requested_output_units": self.config.max_tokens,
            "context_policy": "fresh_author_actual_system_user_only",
            "system_sha256": _hash(system), "user_sha256": _hash(user),
            "response_file": "response.txt", "response_max_bytes": self.config.max_response_bytes,
            "delivery": "write temporary file then atomically rename to response.txt",
            "usage_status": "unknown_full_reserve",
        }
        if handle is not None:
            if verify_prompt_payload(handle.prompt_payload_ref) != prompt_payload:
                raise ValueError("external request prompt differs from reserved prompt")
            request["authoring_attempt"] = {"round_id":handle.round_id, "attempt_id":handle.attempt_id,
                "sequence":handle.sequence, "prompt_payload":handle.prompt_payload_ref.to_public_dict()}
            binding = {"authoring_attempt":request["authoring_attempt"], "request_dir":str(request_dir.resolve()),
                       "request":request, "round_public_context":dict(self._prompt_build.round_public_context)}
            create_once_bytes(Path(handle.result_path).with_name("external_request_binding.json"),
                              canonical_json_bytes(binding, trailing_newline=True))
        pending_request = request_dir / "request.pending"
        pending_request.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        pending_request.replace(request_dir / "request.json")
        if self.exchange is not None:
            self.exchange(request_dir)
        return self._read_response(request_dir, request, prompt_payload)

    def _read_response(self, request_dir, request, prompt_payload):
        deadline = time.monotonic() + self.config.timeout_s
        response_path = request_dir / "response.txt"
        while not response_path.exists():
            if time.monotonic() >= deadline:
                if self._attempt_handle is not None:
                    waiting_path = Path(self._attempt_handle.result_path).with_name("external_author_waiting.json")
                    if not waiting_path.exists():
                        create_once_bytes(waiting_path, canonical_json_bytes(request["authoring_attempt"], trailing_newline=True))
                    elif load_strict_json_object(waiting_path) != request["authoring_attempt"]:
                        raise ValueError("external timeout handoff differs from the original request")
                    raise ExternalAuthorWaiting(request_dir.resolve())
                raise TimeoutError("external author response was not delivered before timeout")
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        with response_path.open("rb") as stream:
            raw = stream.read(self.config.max_response_bytes + 1)
        receipt = {
            "schema_version": EXTERNAL_RECEIPT_SCHEMA, "backend": EXTERNAL_PROVIDER_BACKEND,
            "requested_model": self.config.model, "usage_status": "unknown_full_reserve",
            "system_sha256": request["system_sha256"], "user_sha256": request["user_sha256"],
            "response_text_sha256": _hash(raw), "response_text_utf8_bytes": len(raw),
            "response_capture": ("complete" if len(raw) <= self.config.max_response_bytes else "byte_limit_prefix"),
            "requested_output_units": self.config.max_tokens,
        }
        if len(raw) > self.config.max_response_bytes:
            raise ProviderCandidateError("response_truncated", "external response exceeds the byte limit",
                                         provider_receipt=receipt)
        try:
            response_text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProviderCandidateError("json_parse_error", "external response is not UTF-8",
                                         provider_receipt=receipt) from exc
        return self._materializer.parse(response_text, prompt_payload=prompt_payload,
                                        provider_usage={}, provider_receipt=receipt)


def _hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()
