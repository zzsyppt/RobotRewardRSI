"""One Stair description profile, bound directly to the phase's raw archive."""
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest, iter_numeric_evidence
from .trajectory_adapter import ARCHIVE_CAPABILITIES
from .trajectory_production import PROFILE, DEFAULT_PUBLIC_BYTES, MAX_PUBLIC_BYTES, build_summary, source_domain

PROFILES = (PROFILE,)
SUPPORTED_CELLS = ("stair.go2_12dof.unified_v2", "stair.go2_12dof.unified_v3")


class DescriptionCapabilityUnavailable(ValueError):
    """An intact archive belongs to a cell/capture contract without a translator."""


def validate_feedback_profile(profile, max_bytes, cell_variant_id):
    if profile != PROFILE:
        raise ValueError("unsupported Stair behavior feedback profile")
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PUBLIC_BYTES:
        raise ValueError(f"Stair trajectory report byte budget must be in 1..{MAX_PUBLIC_BYTES}")


def build_phase_description(reference, *, expected_identity):
    archive = load_numeric_evidence_manifest(reference)
    required = {"job_id","task_id","checkpoint_hash","evaluation_plan_hash","evaluation_binding","seed_namespace","splits"}
    if (not required <= expected_identity.keys() or expected_identity["splits"] != ["public_dev"]
            or any(archive["identity"].get(k) != v for k,v in expected_identity.items())):
        raise ValueError("trajectory evaluation/source identity mismatch")
    context = archive["metadata"]
    if (context["profile"] not in ARCHIVE_CAPABILITIES
            or context["resolved_cell_manifest"]["cell_variant_id"] not in SUPPORTED_CELLS):
        # A capability gap never excuses corrupt bytes, array headers, or identity.
        for _ in iter_numeric_evidence(reference):
            pass
        raise DescriptionCapabilityUnavailable("current translator does not cover this cell/capture contract")
    summary = build_summary(reference)
    domain = source_domain(reference, archive)
    comparison = {"domain":domain, **summary.pop("comparison")}
    return domain, summary, comparison
