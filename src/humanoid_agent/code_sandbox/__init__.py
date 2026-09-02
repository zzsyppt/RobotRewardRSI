"""Code-candidate sandbox interfaces.

Phase 8A provides schema and direct human-authored no-op loading.  Phase 8C
adds an isolated subprocess loader backed by server-supported namespace and
Landlock restrictions.
"""

from .components import DiagnosticTerm, LossTerm, RewardInputs, RewardTerm, ScheduleTerm, TrainingComponents
from .loader import (
    CandidateLoadError,
    CandidateManifest,
    candidate_package_fingerprint,
    load_candidate_manifest,
    load_training_components,
    resolve_entrypoint_file,
)
from .isolated_loader import run_isolated_component_load
from .verifier import VerifierIssue, verify_candidate_package

__all__ = [
    "CandidateLoadError",
    "CandidateManifest",
    "DiagnosticTerm",
    "LossTerm",
    "RewardInputs",
    "RewardTerm",
    "ScheduleTerm",
    "TrainingComponents",
    "candidate_package_fingerprint",
    "load_candidate_manifest",
    "load_training_components",
    "resolve_entrypoint_file",
    "run_isolated_component_load",
    "VerifierIssue",
    "verify_candidate_package",
]
