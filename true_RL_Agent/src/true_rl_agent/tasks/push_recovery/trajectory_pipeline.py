"""One source-bound root card per public-dev seed group, after GPU release."""
from __future__ import annotations

from copy import deepcopy

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import ReportBudgetExceeded, SENTENCE_SECTIONS, report_bytes
from true_rl_agent.runner.numeric_evidence import NumericEvidenceError, load_numeric_evidence_manifest
from .numeric_capture import validate_push_numeric_reference, PUSH_NUMERIC_SCHEMA, PUSH_NUMERIC_V1_SCHEMA
from .trajectory_adapter import iter_push_episodes
from .trajectory_card import _build_measured_card, project_push_card, MAX_PUBLIC_BYTES, TEMPLATE_VERSION
from .trajectory_measurement import measure_push_episode
from .trajectory_report import FULL_SCHEMA, PUBLIC_SCHEMA, VERSION, phase_binding, render_verified_report, require_phase_binding

LEGACY_PROFILE = "legacy_process_v1"
TRAJECTORY_PROFILE = "trajectory_numeric_v1"
PROFILES = (LEGACY_PROFILE, TRAJECTORY_PROFILE)
DEFAULT_PROFILE = LEGACY_PROFILE
SUPPORTED_CELLS = tuple(f"push.{robot}.recovery_{version}"
                       for robot in ("g1_37dof", "go2_12dof") for version in ("v2", "v3"))


class DescriptionComputationError(RuntimeError):
    """Pure Push description computation failed after complete source admission."""

    def __init__(self, stage):
        super().__init__(f"Push description {stage} failed")
        self.stage = stage


def validate_feedback_profile(profile: str, max_bytes: int, cell_variant_id: str) -> None:
    if profile not in PROFILES:
        raise ValueError("unsupported Push behavior feedback profile")
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PUBLIC_BYTES:
        raise ValueError("Push trajectory report byte budget must be in 1..12288")
    if profile == LEGACY_PROFILE and max_bytes != MAX_PUBLIC_BYTES:
        raise ValueError("trajectory report budget requires trajectory_numeric_v1")
    if profile == TRAJECTORY_PROFILE and cell_variant_id not in SUPPORTED_CELLS:
        raise ValueError("Push trajectory_numeric_v1 supports physical G1/Go2 recovery v2/v3 only")


def _source_id(sources: list[dict]) -> str:
    ordered = sorted(sources, key=lambda s: s["identity"]["run_index"])
    return "p" + canonical_json_hash([s["reference"]["manifest_sha256"] for s in ordered])[:12]


def build_phase_trajectory(seed_summaries, *, expected_identity: dict, context: dict,
                           expected_seed_count: int,
                           max_bytes: int = MAX_PUBLIC_BYTES) -> tuple[dict, dict, dict]:
    """Read admitted archives and measure once; retries rebuild from current code."""
    validate_feedback_profile(TRAJECTORY_PROFILE, max_bytes, context["cell_variant_id"])
    if len(seed_summaries) != expected_seed_count or expected_seed_count <= 0:
        raise ValueError("Push trajectory phase seed count mismatch")
    sources = []
    for seed in seed_summaries:
        ref = seed.get("numeric_evidence_ref")
        if not ref:
            raise ValueError("Push trajectory requires every public-dev numeric manifest")
        validate_push_numeric_reference(ref, seed, expected_identity=expected_identity)
        identity = load_numeric_evidence_manifest(ref)["identity"]
        if (identity["task_numeric_schema"], identity["mechanism"]) not in {
            (PUSH_NUMERIC_V1_SCHEMA, "world_planar_external_force"),
            (PUSH_NUMERIC_SCHEMA, "world_planar_external_force"),
            (PUSH_NUMERIC_SCHEMA, "world_planar_velocity_increment"),
        }:
            raise ValueError("Push trajectory supports force numeric v1/v2 and velocity numeric v2")
        if (identity.get("resolved_cell_manifest_hash") != context["resolved_cell_manifest_hash"]
                or identity.get("resolved_protocol_hash") != context["resolved_protocol_hash"]):
            raise ValueError("Push trajectory archive/cell/protocol context mismatch")
        sources.append({"reference": dict(ref), "identity": identity})
    sources.sort(key=lambda s: s["identity"]["run_index"])
    if [s["identity"]["run_index"] for s in sources] != list(range(1, expected_seed_count + 1)):
        raise ValueError("Push trajectory requires each planned seed run exactly once")
    source_id = _source_id(sources)
    inputs = {"profile": TRAJECTORY_PROFILE, "sources": sources, "identity": deepcopy(expected_identity),
              "context": deepcopy(context), "source_id": source_id, "semantic_versions": {"delivery":VERSION,"template":TEMPLATE_VERSION},
              "max_report_bytes": max_bytes, "expected_seed_count": expected_seed_count}
    episodes = [e for s in sources for e in iter_push_episodes(s["reference"], expected_identity=s["identity"])]
    # Read every archive and check cross-run source compatibility before any
    # fallible descriptor runs. Admission and I/O errors remain evaluation errors.
    if not episodes:
        raise ValueError("Push card requires episodes")
    first = episodes[0]
    if any(ep.source["group_identity"] != first.source["group_identity"]
           or ep.task["parameters"] != first.task["parameters"]
           or ep.task["time"] != first.task["time"] for ep in episodes):
        raise ValueError("Push cohort policy/protocol/capability mismatch; no cross-condition pooling")
    stage = "measurement"
    try:
        records = [measure_push_episode(e) for e in episodes]
        stage = "reporting"
        card = _build_measured_card(records, source_id)
        full = {"schema_version": FULL_SCHEMA, "card": card, "sources": sources,
                "identity": deepcopy(expected_identity), "context": deepcopy(context),
                "max_report_bytes": max_bytes, "delivery_version": VERSION}
        public = render_verified_report(full)
        included = {r["id"] for s in SENTENCE_SECTIONS for r in public["report"][s]}
        required = [c["id"] for c in card["claims"] if c["mandatory"]]
        # Retain the receipt field name for existing phase readers; no cache is read or written.
        receipt = {"schema_version": "push.trajectory_build_receipt.v1", "complete": True,
                   "cache_key": canonical_json_hash(inputs), "inputs": inputs, "source_id": source_id,
                   "episodes": len(episodes), "card_id": card["card_id"],
                   "phase_binding": phase_binding(expected_identity, sources),
                   "required_claim_ids": required,
                   "omitted_claims": [{"id": c["id"], "reason": "public_byte_budget"}
                                      for c in card["claims"] if c["id"] not in included],
                   "public_content_hash": public["report_projection"]["content_hash"],
                   "full_derivation_sha256": canonical_json_hash(full), "report_bytes": report_bytes(public)}
        validate_build_receipt(receipt, public)
    except (ReportBudgetExceeded, NumericEvidenceError, OSError):
        raise
    except Exception as exc:
        raise DescriptionComputationError(stage) from exc
    return full, public, receipt


def validate_build_receipt(receipt: dict, public: dict) -> None:
    """Replay a sealed phase receipt and public object; never open raw paths."""
    if (receipt.get("schema_version") != "push.trajectory_build_receipt.v1" or receipt.get("complete") is not True
            or public.get("schema_version") != PUBLIC_SCHEMA):
        raise ValueError("Push trajectory build receipt version/completion mismatch")
    inputs = receipt["inputs"]
    required = receipt["required_claim_ids"]
    included = {r["id"] for s in SENTENCE_SECTIONS for r in public["report"][s]}
    require_phase_binding(public, receipt["phase_binding"])
    if (receipt["cache_key"] != canonical_json_hash(inputs) or inputs["profile"] != TRAJECTORY_PROFILE
            or receipt["phase_binding"] != phase_binding(inputs["identity"], inputs["sources"])
            or inputs["source_id"] != _source_id(inputs["sources"])
            or receipt["source_id"] != inputs["source_id"] or receipt["source_id"] != public["domain"]["source_id"]
            or receipt["card_id"] != public["card_id"] or not required or len(set(required)) != len(required)
            or not set(required) <= included
            or canonical_json_hash(required) != public["report_projection"]["required_claim_ids_hash"]
            or canonical_json_hash([c["id"] for c in receipt["omitted_claims"]]) != public["report_projection"]["omitted_claim_ids_hash"]
            or receipt["public_content_hash"] != public["report_projection"]["content_hash"]
            or receipt["full_derivation_sha256"] != public["full_derivation_sha256"]
            or receipt["report_bytes"] != report_bytes(public)
            or public["report_projection"]["max_bytes"] != inputs["max_report_bytes"]):
        raise ValueError("Push trajectory build receipt source/card/projection mismatch")
    validate_feedback_profile(inputs["profile"], inputs["max_report_bytes"], inputs["context"]["cell_variant_id"])


def replay_full_report(full: dict, *, expected_sources: list[dict], expected_identity: dict) -> dict:
    """Explicit CPU audit of full artifacts, requiring external producer bindings."""
    if (full.get("schema_version") != FULL_SCHEMA or full["sources"] != expected_sources
            or full["identity"] != expected_identity or full["delivery_version"] != VERSION):
        raise ValueError("Push full report external source/identity mismatch")
    for source in expected_sources:
        if any(source["identity"].get(k) != v for k, v in expected_identity.items()):
            raise ValueError("Push full report archive identity mismatch")
    episodes = [e for s in expected_sources for e in iter_push_episodes(s["reference"], expected_identity=s["identity"])]
    project_push_card(full["card"], episodes=episodes, source_id=_source_id(expected_sources))
    return render_verified_report(full)
