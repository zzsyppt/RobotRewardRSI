"""Bounded, blind visual observations; task semantics remain task supplied.

This module does not schedule calls, choose episodes, or promote observations to
rules. Callers own their journal and must persist receipts before another call.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.openai_compatible_provider import (
    JsonTransport,
    UrllibJsonTransport,
)


SCHEMA_VERSION = "true_rl_agent.visual_observer.v3"
LAYOUT_VERSION = "rgb_step_view_grid.v2"
MAX_OBSERVATION_CHARS = 200
MAX_UNCERTAINTY_CHARS = 120
MAX_OBSERVATIONS_PER_EPISODE = 3
MAX_NOT_VISIBLE_ITEMS = 3
MAX_NOT_VISIBLE_CHARS = 240
MAX_MEASURED_FACTS = 5
MAX_MEASURED_FACT_CHARS = 1000
IDENTITY_FIELDS = ("session_id", "task_id", "robot_id", "protocol_id", "source_id",
                   "checkpoint_id", "episode_id", "run_id")
BLIND_TOKENS = (
    "score", "success", "fail", "terminated", "termination", "forbidden",
    "timeout", "recovered", "primary_reason", "成功", "失败", "分数", "得分",
    "终止", "禁触", "超时", "恢复成功", "未恢复", "结局", "评阅结论",
)
OBSERVER_INSTRUCTION = (
    "Describe visible morphology that adds to the supplied measurements. measured_facts "
    "are task-generated arithmetic statements about S0, not things seen in pixels; "
    "S0 definitions specify bodies, units and frames. Use them to resolve direction, "
    "but do not substitute one measured property for another or repeat the numeric table. "
    "body_image_points locate named link origins using same-frame simulator camera calibration; "
    "xy_px refers to the original PNG's image_size_px [width, height], regardless of display resizing; "
    "pixel coordinates start at the image top-left. They identify geometry, not visibility, "
    "surface contact or the centre of an anatomical part. A projected point may be occluded. "
    "Each observation should make one independently checkable shape or spatial claim "
    "about a named body, possibly comparing its states at cited samples. Keep distinct "
    "parts and measurement dimensions separate. Combine complementary views, not "
    "unrelated claims. Do not force an action name or phases onto subtle differences. "
    "When detail is unclear, keep the coarser supported shape and state what is unresolved. "
    "Zero observations is permitted when there is no supported visual addition. "
    "Describe sample states or endpoint differences; sparse images cannot establish "
    "continuous, repeated or monotonic motion between them. Camera-relative position "
    "is not world displacement. Do not infer causes, force, friction, torque, collision "
    "or surface contact from appearance.\n"
    "Reply with JSON whose only top-level key is episodes. Each episode has only "
    "alias, observations, not_visible. type/json_object are API settings, not answer fields. "
    f"Use at most {MAX_OBSERVATIONS_PER_EPISODE} observations per episode. Each has only body, "
    "evidence_refs, text, uncertainty, spatial_visibility (visible, partial, occluded), "
    "relation (state, temporal_change), and optionally numeric_claim. "
    f"text and uncertainty must be nonempty and at most {MAX_OBSERVATION_CHARS} and "
    f"{MAX_UNCERTAINTY_CHARS} characters respectively. All factual wording, including "
    "uncertainty, needs evidence. spatial_visibility concerns spatial detail, not "
    "time coverage. Use supplied body names. evidence_refs is a list of {step, view_id} "
    "pairs identifying displayed samples. Cite all samples used, both comparison "
    "endpoints, and every view used. temporal_change requires different steps in at "
    "least one shared view; two views of one step are the same state. Do not return "
    "start/end: code computes bounds from references. "
    f"not_visible is a list of at most {MAX_NOT_VISIBLE_ITEMS} strings, each at most "
    f"{MAX_NOT_VISIBLE_CHARS} characters; [] is allowed. "
    "Optional numeric_claim is exactly {id, asserted}, using a supplied measurement_claims "
    "id and a boolean for whether that exact predicate holds at the cited steps. "
    "Omit it if irrelevant or uncertain; it never certifies the rest of your sentence. "
    "Check JSON closure, text lengths and reference pairs before returning."
)


def _json_bytes(value: Any) -> bytes:
    # Match UrllibJsonTransport's actual request serialization.
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class MediaImage:
    data: bytes
    width: int
    height: int
    displayed_steps: tuple[int, ...]
    view_id: str

    def __post_init__(self) -> None:
        if self.data[:8] != b"\x89PNG\r\n\x1a\n" or len(self.data) < 24:
            raise ValueError("media image must be PNG")
        if struct.unpack(">II", self.data[16:24]) != (self.width, self.height):
            raise ValueError("image dimensions disagree with PNG header")
        if self.width <= 0 or self.height <= 0 or not self.displayed_steps:
            raise ValueError("image dimensions and displayed steps must be nonempty")
        if tuple(sorted(set(self.displayed_steps))) != self.displayed_steps:
            raise ValueError("displayed steps must be distinct and increasing")
        if any(type(step) is not int or step < 0 for step in self.displayed_steps):
            raise ValueError("displayed steps must be nonnegative integers")
        if not isinstance(self.view_id, str) or re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", self.view_id) is None:
            raise ValueError("image view_id must be a stable view identifier")


@dataclass(frozen=True)
class MediaWindow:
    alias: str
    camera_version: str
    layout_version: str
    roi_version: str
    body_names: tuple[str, ...]
    images: tuple[MediaImage, ...]
    s0: tuple[Mapping[str, Any], ...] = ()
    visibility: str = "sampled frames"
    # Only the private receipt receives this sidecar, never the model payload.
    identity: Mapping[str, Any] = field(default_factory=dict)
    # Task-owned measurement definitions, never task outcomes or behavior names.
    measurement_claims: tuple[Mapping[str, str], ...] = ()
    # The task freezes these definitions; this module checks matching metadata,
    # not whether the chosen measurement is semantically appropriate for a task.
    s0_definitions: tuple[Mapping[str, str], ...] = ()
    # Task-generated readable measurements, not model observations. Their scalar
    # references point into this window's S0; anatomy and wording stay task-owned.
    measured_facts: tuple[Mapping[str, Any], ...] = ()
    # Same-snapshot calibrated link-origin projections, selected by the task.
    # No world coordinates or camera matrices need enter the model context.
    body_image_points: tuple[Mapping[str, Any], ...] = ()


def read_native_snapshot_media(
    path: Path, *, manifest_sha256: str, selected_steps: Sequence[int] | None = None,
) -> tuple[dict[str, Any], tuple[MediaImage, ...]]:
    """Read actual samples from a sealed native capture, without relabeling them.

    The returned manifest stays private. The task must bind its run, seed and env
    to the numerical archive and verify sample timing before creating a window.
    Requested but uncaptured steps are not images. No resizing is performed.
    When selected_steps is supplied, only those image bytes are loaded and hash
    checked; the sealed manifest's complete sample metadata is still checked.
    Selecting evidence for a question remains the task caller's responsibility.
    """
    raw = path.read_bytes()
    if _sha(raw) != manifest_sha256:
        raise ValueError("native snapshot manifest hash differs")
    manifest = json.loads(raw)
    if manifest.get("schema_version") not in {
        "native_rgb_snapshots.v1", "native_rgb_snapshots.v2", "native_rgb_snapshots.v3",
        "native_rgb_snapshots.v4",
    }:
        raise ValueError("unsupported native snapshot schema")
    if selected_steps is not None:
        selected_steps = tuple(selected_steps)
        if (not selected_steps or any(type(step) is not int or step <= 0 for step in selected_steps)
                or tuple(sorted(set(selected_steps))) != selected_steps):
            raise ValueError("selected snapshot steps must be increasing positive integers")
    width, height = manifest["resolution"]
    images: list[MediaImage] = []
    pairs: set[tuple[int, str]] = set()
    previous_step = 0
    for sample in manifest["samples"]:
        step = sample["control_step"]
        if (type(step) is not int or not previous_step < step <= manifest["episode_control_step_count"]
                or sample["state_before"] != sample["state_after"]):
            raise ValueError("native snapshot interval or state proof differs")
        previous_step = step
        for view in sample["views"]:
            pair = (step, view["view_id"])
            if pair in pairs or Path(view["file"]).name != view["file"]:
                raise ValueError("native snapshot view or adjacent filename differs")
            pairs.add(pair)
            if selected_steps is not None and step not in selected_steps:
                continue
            data = (path.parent / view["file"]).read_bytes()
            if _sha(data) != view["sha256"]:
                raise ValueError("native snapshot image hash differs")
            images.append(MediaImage(data, width, height, (step,), view["view_id"]))
    if len(pairs) != manifest["snapshot_count"]:
        raise ValueError("native snapshot image count differs")
    if selected_steps is not None and not set(selected_steps) <= {step for step, _ in pairs}:
        raise ValueError("selected snapshot step was not captured")
    return manifest, tuple(images)


@dataclass
class ObserverBudget:
    """One report's cumulative reservations, including repeated submissions.

    Reconstruct this state from the existing journal on recovery. A timeout or
    response rejection never refunds an image or its reserved dollar amount.
    """

    max_pixels: int
    max_image_bytes: int
    max_cost_usd: float
    max_calls: int
    aliases: set[str] = field(default_factory=set)
    identity_keys: dict[str, str] = field(default_factory=dict)
    images: int = 0
    subframes: int = 0
    pixels: int = 0
    image_bytes: int = 0
    calls: int = 0
    reserved_cost_usd: float = 0.0

    def reserve(self, windows: Sequence[MediaWindow], cost_usd: float) -> dict[str, Any]:
        if not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("call cost reservation must be finite and nonnegative")
        aliases = self.aliases | {window.alias for window in windows}
        identity_keys = dict(self.identity_keys)
        for window in windows:
            key = _sha(json.dumps(dict(window.identity), sort_keys=True).encode())
            if window.alias in identity_keys and identity_keys[window.alias] != key:
                raise ValueError("observer alias changed identity within one report")
            identity_keys[window.alias] = key
        assets = [asset for window in windows for asset in window.images]
        totals = {
            "images": self.images + len(assets),
            "subframes": self.subframes + sum(len(asset.displayed_steps) for asset in assets),
            "pixels": self.pixels + sum(asset.width * asset.height for asset in assets),
            "image_bytes": self.image_bytes + sum(len(asset.data) for asset in assets),
            "calls": self.calls + 1,
            "reserved_cost_usd": self.reserved_cost_usd + cost_usd,
        }
        limits = {"images": 24, "subframes": 96, "pixels": self.max_pixels,
                  "image_bytes": self.max_image_bytes, "calls": self.max_calls,
                  "reserved_cost_usd": self.max_cost_usd}
        if len(aliases) > 4 or any(totals[key] > limit for key, limit in limits.items()):
            raise ValueError("observer report budget exceeded")
        self.aliases = aliases
        self.identity_keys = identity_keys
        for key, value in totals.items():
            setattr(self, key, value)
        return {"episodes": len(aliases), **totals}


def _public_window(window: MediaWindow) -> dict[str, Any]:
    if re.fullmatch(r"e[0-9]{1,4}", window.alias) is None:
        raise ValueError("observer alias must be an opaque e-number")
    if not window.images or not window.body_names:
        raise ValueError("each media window needs images and body names")
    if not all(isinstance(body, str) and body for body in window.body_names):
        raise ValueError("body names must be nonempty strings")
    steps = sorted({step for image in window.images for step in image.displayed_steps})
    pairs = [(step, image.view_id) for image in window.images for step in image.displayed_steps]
    if len(pairs) != len(set(pairs)):
        raise ValueError("each displayed step/view_id pair must identify one image")
    definition_fields = {"name", "unit", "body", "reference_frame", "source"}
    definitions = {}
    for definition in window.s0_definitions:
        if (not isinstance(definition, Mapping) or set(definition) != definition_fields
                or not all(isinstance(value, str) and value for value in definition.values())
                or definition["body"] not in window.body_names
                or definition["name"] in definitions):
            raise ValueError("S0 definition needs a unique name, known body, unit, reference_frame and source")
        definitions[definition["name"]] = dict(definition)
    scalars = []
    scalar_keys = set()
    for scalar in window.s0:
        if not isinstance(scalar, Mapping) or set(scalar) != definition_fields | {"step", "value"}:
            raise ValueError("S0 scalar needs its definition metadata, step and value")
        if (type(scalar["value"]) not in (int, float)
                or not math.isfinite(scalar["value"]) or type(scalar["step"]) is not int
                or scalar["step"] not in steps):
            raise ValueError("S0 must be a finite displayed-step scalar")
        metadata = {key: scalar[key] for key in definition_fields}
        if not isinstance(scalar["name"], str) or definitions.get(scalar["name"]) != metadata:
            raise ValueError("S0 scalar differs from its task-provided definition whitelist")
        scalar_key = (scalar["name"], scalar["step"])
        if scalar_key in scalar_keys:
            raise ValueError("S0 scalar name/step must be unique")
        scalar_keys.add(scalar_key)
        scalars.append(dict(scalar))
    facts, fact_ids = [], set()
    if len(window.measured_facts) > MAX_MEASURED_FACTS:
        raise ValueError("too many measured facts for one media window")
    for fact in window.measured_facts:
        if (not isinstance(fact, Mapping) or set(fact) != {"id", "body", "text", "scalar_refs"}
                or not isinstance(fact["id"], str)
                or re.fullmatch(r"[a-z0-9_.]+", fact["id"]) is None
                or fact["id"] in fact_ids or fact["body"] not in window.body_names
                or not isinstance(fact["text"], str) or not fact["text"].strip()
                or len(fact["text"]) > MAX_MEASURED_FACT_CHARS):
            raise ValueError("measured fact requires a unique id, known body and bounded text")
        refs = fact["scalar_refs"]
        if not isinstance(refs, (list, tuple)) or not refs:
            raise ValueError("measured fact needs S0 references")
        seen_refs = set()
        for ref in refs:
            if (not isinstance(ref, Mapping) or set(ref) != {"name", "step"}
                    or not isinstance(ref["name"], str) or type(ref["step"]) is not int):
                raise ValueError("invalid measured fact scalar reference")
            key = (ref["name"], ref["step"])
            if (key not in scalar_keys or key in seen_refs
                    or definitions[ref["name"]]["body"] != fact["body"]):
                raise ValueError("measured fact must reference its body's actual S0 samples")
            seen_refs.add(key)
        fact_ids.add(fact["id"])
        facts.append({**dict(fact), "scalar_refs": [dict(ref) for ref in refs]})
    claims, identifiers = [], set()
    for claim in window.measurement_claims:
        if (set(claim) != {"id", "body", "definition"}
                or not all(isinstance(value, str) and value for value in claim.values())
                or re.fullmatch(r"[a-z0-9_.]+", claim["id"]) is None
                or claim["body"] not in window.body_names or claim["id"] in identifiers):
            raise ValueError("measurement claim definition must have unique id, known body and definition")
        identifiers.add(claim["id"])
        claims.append(dict(claim))
    points, point_keys, per_image = [], set(), {}
    for point in window.body_image_points:
        if set(point) != {"body_name", "control_step", "view_id", "image_xy_px", "projection_status"}:
            raise ValueError("body projection fields differ")
        pair = (point["control_step"], point["view_id"])
        key = (*pair, point["body_name"])
        if pair not in pairs or point["body_name"] not in window.body_names or key in point_keys:
            raise ValueError("body projection needs a distinct known body and displayed step/view")
        point_keys.add(key)
        per_image[pair] = per_image.get(pair, 0) + 1
        if per_image[pair] > 6:
            raise ValueError("body projection workset exceeds six points per image")
        xy, status = point["image_xy_px"], point["projection_status"]
        if status in {"in_frame", "outside_frame"}:
            if (not isinstance(xy, (list, tuple)) or len(xy) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)):
                raise ValueError("front-of-camera projection needs finite image coordinates")
        elif status not in {"behind_camera", "on_camera_plane"} or xy is not None:
            raise ValueError("unprojectable body needs an explicit camera status and no pixel")
        points.append({"body": point["body_name"], "step": point["control_step"],
                       "view_id": point["view_id"], "xy_px": xy, "status": status})
    return {
        "alias": window.alias, "camera_version": window.camera_version,
        "layout_version": window.layout_version, "roi_version": window.roi_version,
        "displayed_steps": steps, "body_names": list(window.body_names),
        # Keep definitions once; the complete S0 remains in the private window.
        "s0": [{key: scalar[key] for key in ("name", "step", "value")} for scalar in scalars],
        "s0_definitions": list(definitions.values()), "measured_facts": facts,
        "body_image_points": points,
        "visibility": window.visibility, "measurement_claims": claims,
    }


def audit_request_text(request_bytes: bytes, extra_tokens: Sequence[str] = ()) -> dict[str, Any]:
    """Scan the serialized JSON text layer, not random base64 image characters."""
    decoded = json.loads(request_bytes)
    image_proofs: list[dict[str, Any]] = []

    def text_layer(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: text_layer(item) for key, item in value.items()}
        if isinstance(value, list):
            return [text_layer(item) for item in value]
        if isinstance(value, str) and value.startswith("data:image/png;base64,"):
            data = base64.b64decode(value.split(",", 1)[1], validate=True)
            image_proofs.append({"sha256": _sha(data), "bytes": len(data)})
            return "<image bytes>"
        return value

    text_bytes = _json_bytes(text_layer(decoded))
    text = text_bytes.decode("utf-8").casefold()
    violations = [token for token in (*BLIND_TOKENS, *extra_tokens) if token.casefold() in text]
    return {"passed": not violations, "violations": violations,
            "request_sha256": _sha(request_bytes), "request_bytes": len(request_bytes),
            "text_sha256": _sha(text_bytes), "image_proofs": image_proofs}


def prepare_observer_request(
    windows: Sequence[MediaWindow], *, model: str, max_tokens: int = 2048,
    extra_blind_tokens: Sequence[str] = (),
    generation_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Chat Completions image request without invoking any provider."""
    if not windows or max_tokens <= 0:
        raise ValueError("observer request needs media and an output bound")
    content: list[dict[str, Any]] = []
    private = []
    identities: dict[str, dict[str, Any]] = {}
    for window in windows:
        public = _public_window(window)
        identity = dict(window.identity)
        if not all(isinstance(identity.get(key), str) and identity[key] for key in IDENTITY_FIELDS):
            raise ValueError("media identity sidecar is incomplete")
        if window.s0 and not all(identity.get(key) == identity.get("numeric_" + key)
                                 for key in IDENTITY_FIELDS):
            raise ValueError("S0 numeric identity differs from its media identity")
        if window.alias in identities:
            raise ValueError("one alias needs one MediaWindow containing all of its views")
        identities[window.alias] = identity
        content.append({"type": "text", "text": _json_bytes(public).decode("utf-8")})
        for asset in window.images:
            content.append({"type": "text", "text": _json_bytes({
                "alias": window.alias, "view_id": asset.view_id,
                "displayed_steps": list(asset.displayed_steps),
                "image_size_px": [asset.width, asset.height],
            }).decode("utf-8")})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(asset.data).decode("ascii"),
                "detail": "high",
            }})
        private.append({"alias": window.alias, "identity": identity,
                        "images": [{"view_id": asset.view_id,
                                    "displayed_steps": list(asset.displayed_steps),
                                    "sha256": _sha(asset.data)} for asset in window.images]})
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": OBSERVER_INSTRUCTION},
                            {"role": "user", "content": content}]}
    options = dict(generation_options or {})
    if set(options) - {"thinking", "reasoning_effort"}:
        raise ValueError("observer generation options only control thinking/effort")
    payload.update(options)
    request_bytes = _json_bytes(payload)
    audit = audit_request_text(request_bytes, extra_blind_tokens)
    if not audit["passed"]:
        raise ValueError("blind observer input contains outcome or review tokens: "
                         + ", ".join(audit["violations"]))
    return {"payload": payload, "request_bytes": request_bytes, "generation_options": options,
            "audit": audit, "private_sidecar": private}


def audit_observer_output(
    output: Mapping[str, Any], windows: Sequence[MediaWindow], *,
    numeric_check: Callable[[Mapping[str, Any], MediaWindow], Mapping[str, Any] | None] | None = None,
    inference_check: Callable[[Mapping[str, Any]], Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Preserve every raw observation; checks annotate rather than filter it.

    Task callbacks return {status: consistent|conflict, numeric_evidence: ...}.
    Their result is stored beside the original observation and cannot replace it.
    A mapping is usable only against the media run's own numeric identity.
    """
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    episodes = output.get("episodes") if isinstance(output, Mapping) else None
    if not isinstance(output, Mapping) or set(output) != {"episodes"} or not isinstance(episodes, list):
        issues.append("output_schema")
        if not isinstance(episodes, list):
            return {"schema_version": SCHEMA_VERSION, "raw_output": output, "observations": records,
                    "violations": issues, "passed": False}
        # Readable observations still count before filtering. The container issue
        # remains fatal for S3 delivery; this is an audit, not a schema repair.
    by_alias: dict[str, list[MediaWindow]] = {}
    for window in windows:
        by_alias.setdefault(window.alias, []).append(window)
    for alias, grouped in by_alias.items():
        if len(grouped) != 1:
            issues.append(f"{alias}:one_window_per_episode")
    seen: set[str] = set()
    required = {"body", "spatial_visibility", "uncertainty", "text",
                "relation", "evidence_refs"}
    for episode_index, episode in enumerate(episodes):
        prefix = f"episodes[{episode_index}]"
        if not isinstance(episode, dict):
            issues.append(prefix + ":episode_schema")
            continue
        alias = episode.get("alias")
        known = by_alias.get(alias, []) if isinstance(alias, str) else []
        if not known or alias in seen:
            issues.append(prefix + ":episode_identity")
        if isinstance(alias, str):
            seen.add(alias)
        if (set(episode) != {"alias", "observations", "not_visible"}
                or not isinstance(episode.get("not_visible"), list)
                or len(episode.get("not_visible", [])) > MAX_NOT_VISIBLE_ITEMS
                or not all(isinstance(value, str) and len(value) <= MAX_NOT_VISIBLE_CHARS
                           for value in episode.get("not_visible", []))):
            issues.append(prefix + ":episode_schema")
        observations = episode.get("observations")
        if not isinstance(observations, list):
            issues.append(prefix + ":observations_schema")
            continue
        if len(observations) > MAX_OBSERVATIONS_PER_EPISODE:
            issues.append(prefix + ":observation_count")
        for observation_index, observation in enumerate(observations):
            violations: list[str] = []
            record: dict[str, Any] = {"alias": alias, "raw_observation": observation,
                                     "reference": f"{prefix}.observations[{observation_index}]",
                                     "violations": violations, "numeric_status": "visual_only"}
            records.append(record)
            if (not isinstance(observation, dict)
                    or not required <= set(observation)
                    or set(observation) - required - {"numeric_claim"}):
                violations.append("observation_schema")
                continue
            if len(known) != 1:
                violations.append("displayed_window")
                selected = None
            else:
                selected = known[0]
                available = {}
                for image_index, asset in enumerate(selected.images):
                    for step in asset.displayed_steps:
                        available.setdefault((step, asset.view_id), []).append({
                            "step": step, "view_id": asset.view_id,
                            "image_index": image_index, "image_sha256": _sha(asset.data)})
                refs = observation["evidence_refs"]
                valid_refs = (isinstance(refs, list) and bool(refs)
                    and all(isinstance(ref, Mapping) and set(ref) == {"step", "view_id"}
                        and type(ref["step"]) is int and isinstance(ref["view_id"], str)
                        for ref in refs))
                pairs = [(ref["step"], ref["view_id"]) for ref in refs] if valid_refs else []
                valid_refs = (valid_refs and len(set(pairs)) == len(pairs)
                    and all(len(available.get(pair, [])) == 1 for pair in pairs))
                if not valid_refs:
                    violations.append("view_step_reference")
                    cited_steps = []
                else:
                    record["image_evidence"] = [available[pair][0] for pair in pairs]
                    cited_steps = sorted({step for step, view_id in pairs})
                    record["derived_window"] = {"start": cited_steps[0], "end": cited_steps[-1]}
                    # Gaps concern this observation's citations, not all input images.
                    record["cited_step_gaps"] = [[a + 1, b - 1]
                        for a, b in zip(cited_steps, cited_steps[1:]) if b > a + 1]
                    record["temporal_coverage"] = (
                        "single_cited_step" if len(cited_steps) == 1 else
                        "sparse_cited_steps" if record["cited_step_gaps"] else "consecutive_cited_steps")
                record["cited_steps"] = cited_steps
                relation = observation["relation"]
                if relation not in ("state", "temporal_change"):
                    violations.append("relation")
                elif relation == "temporal_change":
                    per_view = {}
                    for step, view_id in pairs:
                        per_view.setdefault(view_id, set()).add(step)
                    if not any(len(steps) >= 2 for steps in per_view.values()):
                        violations.append("temporal_change_requires_same_view_steps")
                if observation["body"] not in selected.body_names:
                    violations.append("unknown_body")
                record["media_identity"] = dict(selected.identity)
                record["replay_observation"] = bool(selected.identity.get("is_replay", False))
            if "numeric_claim" in observation:
                claim = observation["numeric_claim"]
                definitions = {row["id"]: row for row in selected.measurement_claims} if selected else {}
                if (not isinstance(claim, Mapping) or set(claim) != {"id", "asserted"}
                        or not isinstance(claim.get("id"), str) or type(claim.get("asserted")) is not bool
                        or claim["id"] not in definitions
                        or definitions[claim["id"]]["body"] != observation["body"]):
                    violations.append("numeric_claim_definition")
            if observation["spatial_visibility"] not in ("visible", "partial", "occluded"):
                violations.append("spatial_visibility")
            if (not isinstance(observation["text"], str) or not observation["text"]
                    or len(observation["text"]) > MAX_OBSERVATION_CHARS
                    or not isinstance(observation["uncertainty"], str)
                    or not observation["uncertainty"] or len(observation["uncertainty"]) > MAX_UNCERTAINTY_CHARS):
                violations.append("bounded_text")
            if isinstance(observation["text"], str):
                from true_rl_agent.evaluation.process_report import UNSUPPORTED_NORMALITY_PHRASES
                if any(term in observation["text"].casefold() for term in UNSUPPORTED_NORMALITY_PHRASES):
                    violations.append("unqualified_normality")
            if inference_check is not None:
                violations.extend(inference_check(observation))
            if (selected is not None and not violations and "numeric_claim" in observation
                    and numeric_check is not None):
                check = numeric_check(observation, selected)
                if check is not None:
                    identity = selected.identity
                    same_binding = all(
                        identity.get(key) is not None and identity.get(key) == identity.get("numeric_" + key)
                        for key in IDENTITY_FIELDS)
                    if not same_binding:
                        violations.append("numeric_identity_mismatch")
                    elif check.get("status") not in ("consistent", "conflict"):
                        violations.append("numeric_check_schema")
                    else:
                        record["numeric_status"] = check["status"]
                        record["numeric_check"] = dict(check)
    for alias in by_alias.keys() - seen:
        issues.append(f"{alias}:missing_episode")
    return {"schema_version": SCHEMA_VERSION, "raw_output": output, "observations": records, "violations": issues,
            "passed": not issues and not any(record["violations"] for record in records),
            "semantic_support": "not_evaluated"}


def visual_process_sentences(
    audited: Mapping[str, Any], *, episode_labels: Mapping[str, str],
    cohort_episode_n: int, observer_version: str, evidence_ref: str,
    episode_priority: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """One bounded verbatim observation plus whole-output accounting.

    Tasks supply approved episode labels and bind the returned slots to their
    full artifact. Selection never changes the all-output evaluation denominator.
    A numeric check supports its declared measurement only, not the free prose.
    Mechanical validity permits review; only a bound semantic review admits S3.
    Optional task-owned ranks use approved aliases; lower ranks are preferred.
    """
    if episode_priority is not None and (not isinstance(episode_priority, Mapping)
            or any(alias not in episode_labels or type(rank) is not int or rank < 0
                   for alias, rank in episode_priority.items())):
        raise ValueError("episode_priority requires approved aliases and nonnegative integer ranks")
    priority = dict(episode_priority or {})
    default_priority = max(priority.values(), default=-1) + 1
    records = list(audited["observations"])
    from true_rl_agent.evaluation.visual_review import visual_review_verdicts
    verdicts = visual_review_verdicts(audited)
    mechanically_usable = [(i, record) for i, record in enumerate(records)
              if not record["violations"] and record.get("media_identity")
              and record.get("alias") in episode_labels]
    usable = [(i, record) for i, record in mechanically_usable if verdicts[i] == "supported"]
    if audited["violations"]:
        usable = []  # malformed episode container is not a renderable observation
    # Preserve conflicts first, then the task's episode rank and a process
    # comparison. A consistent scalar predicate does not certify richer prose.
    usable.sort(key=lambda pair: (
        pair[1]["numeric_status"] != "conflict",
        priority.get(pair[1]["alias"], default_priority),
        pair[1]["raw_observation"]["relation"] != "temporal_change", pair[0]))
    selected = usable[:1]
    rows = []
    for index, record in selected:
        raw = record["raw_observation"]
        replay = record.get("replay_observation", False)
        identity = record["media_identity"]
        same_run = all(identity.get(key) is not None and identity.get(key) == identity.get("numeric_" + key)
                       for key in IDENTITY_FIELDS)
        alignment = "replay_observation" if replay else "same_run" if same_run else "media_only"
        temporal = {"single_cited_step": "one cited step",
                    "consecutive_cited_steps": "consecutive cited steps",
                    "sparse_cited_steps": "sparse cited steps"}[record["temporal_coverage"]]
        bounds = record["derived_window"]
        claim = raw.get("numeric_claim")
        if claim:
            status = record["numeric_status"] if record["numeric_status"] != "visual_only" else "not checked"
            check = f" {claim['id']}={str(claim['asserted']).lower()}: {status} (predicate only)."
        else:
            check = " Visual only."
        image_evidence = record["image_evidence"]
        refs = [f"{evidence_ref}/observations/{index}"] + [
            f"{evidence_ref}/observations/{index}/image_evidence/{ref_index}"
            for ref_index in range(len(image_evidence))]
        refs.append(f"{evidence_ref}/semantic_review")
        rows.append({
            "id": "visual.observation", "kind": "S3", "section": "visual",
            "text": (f"{episode_labels[record['alias']]}/{raw['body']} {bounds['start']}-{bounds['end']} "
                     f"(spatial: {raw['spatial_visibility']}; {temporal}; {alignment}): "
                     f"{raw['text']} Uncertainty: {raw['uncertainty']}.{check}"),
            "subject": raw["body"], "predicate": claim["id"] if claim else "visual_shape",
            "rule_version": observer_version,
            "scope": {"label": episode_labels[record["alias"]], "numerator": 1, "denominator": cohort_episode_n},
            "window": dict(bounds),
            "media_identity": identity, "alignment_status": alignment,
            "relation": raw["relation"], "image_evidence": image_evidence,
            "evidence_category": record["numeric_status"], "visibility": raw["spatial_visibility"],
            "temporal_coverage": record["temporal_coverage"], "cited_steps": record["cited_steps"],
            "cited_step_gaps": record["cited_step_gaps"],
            "semantic_review": {"verdict": "supported", "reviewer": dict(audited["semantic_review"]["reviewer"])},
            "evidence_refs": refs, "required": False,
        })
    received = len({record.get("alias") for record in records
                    if isinstance(record.get("alias"), str) and record["alias"] in episode_labels})
    invalid = len(records) - len(mechanically_usable)
    conflicts = sum(record.get("numeric_status") == "conflict" for record in records)
    if audited["violations"]:
        received = 0
        reason = ", ".join(audited["violations"])
        if audited.get("error_type"):
            reason += f" ({audited['error_type']})"
        count = (f"{len(records)} statements enumerated" if records else
                 "Statement count unavailable (0 entries enumerated)")
        coverage = (f"Visual output not delivered: {reason}. {count}; "
                    f"0/{cohort_episode_n} episodes delivered; 0 shown. Original output retained for inspection.")
    else:
        coverage = (f"{len(records)} statements; {received}/{cohort_episode_n} episodes; "
                    f"{invalid} mechanically invalid; "
                    f"{verdicts.count('supported')} semantically supported; "
                    f"{verdicts.count('contradicted')} contradicted; "
                    f"{verdicts.count('insufficient')} insufficient; "
                    f"{verdicts.count('not_evaluated')} unreviewed; {len(selected)} shown. "
                    "Numeric checks verify predicates, not prose."
                    + (f" {conflicts} numeric conflicts; see full evidence." if conflicts else ""))
    rows.append({
        "id": "visual.coverage", "kind": "S5", "section": "visual",
        "text": coverage,
        "subject": "observed bodies", "predicate": "visual_coverage", "rule_version": observer_version,
        "scope": {"label": "observer output", "numerator": received, "denominator": cohort_episode_n},
        "gap_type": "visual_sampling_and_output_omissions", "evidence_refs": [evidence_ref], "required": True,
    })
    from true_rl_agent.evaluation.process_report import validate_sentence
    for row in rows:
        validate_sentence(row)
    return rows


def update_visual_projection_count(report: dict[str, Any]) -> None:
    """The shared visual template counts actually delivered whole statements."""
    rows = report["report"]["visual"]
    shown = sum(row["id"] == "visual.observation" for row in rows)
    for row in rows:
        if row["id"] == "visual.coverage":
            row["text"] = re.sub(r"\b\d+ shown\.", f"{shown} shown.", row["text"], count=1)


def call_observer(
    prepared: Mapping[str, Any], windows: Sequence[MediaWindow], *, budget: ObserverBudget,
    reserved_cost_usd: float, url: str, api_key: str, receipt_sink: Callable[[dict[str, Any]], None],
    transport: JsonTransport | None = None, timeout_s: int = 180,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """One explicit submission; no retries, scheduling, credential discovery or pricing."""
    if _json_bytes(prepared["payload"]) != prepared["request_bytes"]:
        raise ValueError("prepared observer request was modified")
    rebuilt = prepare_observer_request(
        windows, model=prepared["payload"]["model"], max_tokens=prepared["payload"]["max_tokens"],
        generation_options=prepared.get("generation_options"))
    if (rebuilt["request_bytes"] != prepared["request_bytes"]
            or rebuilt["private_sidecar"] != prepared["private_sidecar"]):
        raise ValueError("budget media do not match the prepared request")
    if not audit_request_text(prepared["request_bytes"])["passed"]:
        raise ValueError("observer blind input audit did not pass")
    totals = budget.reserve(windows, reserved_cost_usd)
    receipt = {"schema_version": SCHEMA_VERSION, "status": "reserved",
               "request": prepared["payload"], "request_sha256": _sha(prepared["request_bytes"]),
               "request_bytes": len(prepared["request_bytes"]), "budget": totals,
               "private_sidecar": prepared["private_sidecar"]}
    receipt_sink(dict(receipt))
    try:
        response = (transport or UrllibJsonTransport()).post_json(
            url=url, payload=prepared["payload"], api_key=api_key, timeout_s=timeout_s)
    except Exception as exc:
        receipt.update(status="transport_error", error_type=type(exc).__name__, usage=None,
                       fingerprint=None, cost_status="reserved_unknown")
        receipt_sink(receipt)
        raise
    fingerprint = response.get("system_fingerprint")
    receipt.update(status="received", response=response, usage=response.get("usage"),
                   model=response.get("model"), fingerprint=fingerprint,
                   fingerprint_matches=(bool(fingerprint) and
                       (expected_fingerprint is None or fingerprint == expected_fingerprint)))
    receipt["eligible_for_confirmation"] = (
        expected_fingerprint is not None and receipt["fingerprint_matches"]
        and isinstance(receipt["usage"], dict)
        and all(type(receipt["usage"].get(key)) is int and receipt["usage"][key] >= 0
                for key in ("prompt_tokens", "completion_tokens")))
    receipt_sink(receipt)
    return receipt


def candidate_name_entry(*, raw_name: str, alias: str, start: int, end: int,
                         observer_version: str, canonical_name: str | None = None,
                         merge_history: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """A proposal ledger row only; no operation in this module installs a rule."""
    if canonical_name is not None and not merge_history:
        raise ValueError("a canonical name requires a recorded human merge decision")
    return {"schema_version": "true_rl_agent.visual_candidate_name.v1",
            "raw_name": raw_name, "canonical_name": canonical_name,
            "source": {"alias": alias, "start": start, "end": end},
            "observer_version": observer_version, "occurrences": 1,
            "merge_history": [dict(item) for item in merge_history], "status": "candidate"}


_DIGITS = ("111101101101111", "010110010010111", "111001111100111",
           "111001111001111", "101101111001001", "111100111001111",
           "111100111101111", "111001001001001", "111101111101111",
           "111101111001111")


def render_frame_grid(frames: Sequence[tuple[int, int, int, bytes]], *, view_id: str,
                      columns: int = 4, roi: tuple[int, int, int, int] | None = None) -> MediaImage:
    """Render RGB frames with literal control-step labels, without image dependencies.

    Frames are (step, width, height, RGB bytes); ROI is (left, top, right, bottom).
    One grid contains one view; v2 binds its id beside its pixels in the request.
    The 24-pixel label band, 9x15 step glyphs and unscaled crops are unchanged.
    """
    if not frames or columns <= 0:
        raise ValueError("grid requires frames and a positive column count")
    steps = tuple(frame[0] for frame in frames)
    source_width, source_height = frames[0][1:3]
    left, top, right, bottom = roi or (0, 0, source_width, source_height)
    if not (0 <= left < right <= source_width and 0 <= top < bottom <= source_height):
        raise ValueError("ROI falls outside the source frame")
    crop_width, crop_height = right - left, bottom - top
    cell_width = max(crop_width, max(len(str(step)) for step in steps) * 12 + 4)
    cell_height = crop_height + 24
    width, height = min(columns, len(frames)) * cell_width, math.ceil(len(frames) / columns) * cell_height
    pixels = bytearray(b"\xff" * (width * height * 3))
    for index, (step, frame_width, frame_height, rgb) in enumerate(frames):
        if (frame_width, frame_height) != (source_width, source_height) or len(rgb) != frame_width * frame_height * 3:
            raise ValueError("grid source dimensions or RGB length disagree")
        if type(step) is not int or step < 0:
            raise ValueError("grid step must be a nonnegative integer")
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        for row in range(crop_height):
            source = ((top + row) * source_width + left) * 3
            dest = ((y + 24 + row) * width + x) * 3
            pixels[dest:dest + crop_width * 3] = rgb[source:source + crop_width * 3]
        for offset, digit in enumerate(str(step)):
            for bit, marked in enumerate(_DIGITS[int(digit)]):
                if marked == "1":
                    for label_row in range(3):
                        dest = ((y + 4 + (bit // 3) * 3 + label_row) * width
                                + x + 2 + offset * 12 + (bit % 3) * 3) * 3
                        pixels[dest:dest + 9] = b"\0" * 9
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    scanlines = b"".join(b"\0" + pixels[row * width * 3:(row + 1) * width * 3] for row in range(height))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))
    return MediaImage(png, width, height, steps, view_id)
