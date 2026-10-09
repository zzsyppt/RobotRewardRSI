"""Task-neutral ownership contract for runner-projected public eval metrics."""


SEALED_VALIDATION_EVIDENCE_KEY = "sealed_validation"
CANDIDATE_CONTAMINATION_FIELDS = (
    "candidate_code_loaded_in_eval",
    "candidate_package_loaded",
    "candidate_package_loaded_in_eval",
    "candidate_path_argument_present",
)
RUNNER_OWNED_PUBLIC_EVAL_METRIC_KEYS = frozenset(
    {
        "evaluation_binding",
        "evaluation_subject",
        "subject_manifest_hash",
        "program_source_checksum",
        "training_binding_hash",
        "capability_profile_hash",
        "execution_device",
        "checkpoint_safety_status",
        "checkpoint_safety_report",
        "evidence_manifest",
        SEALED_VALIDATION_EVIDENCE_KEY,
        "sealed_validation_gate",
        *CANDIDATE_CONTAMINATION_FIELDS,
    }
)
