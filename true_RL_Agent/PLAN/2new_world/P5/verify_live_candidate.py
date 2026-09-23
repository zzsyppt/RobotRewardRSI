"""Finish the real provider probe with current production execution admission."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from true_rl_agent.candidates.loader import load_candidate_manifest
    from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
    from true_rl_agent.candidates.verifier import seal_and_verify_candidate
    from true_rl_agent.composition.base import TaskCompositionContext
    from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
    base = Path(__file__).parent / "provider_live_01"
    report = json.loads((base / "result.json").read_text())
    public = report["candidate"]
    paths = [p.parent for p in (base / "candidates").rglob("candidate_manifest.yaml")
             if load_candidate_manifest(p.parent).candidate_id == public["candidate_id"]]
    assert len(paths) == 1
    candidate = CandidatePackageRef(candidate_id=public["candidate_id"], path=str(paths[0]),
        change_set=CandidateChangeSet(frozenset(public["changed_components"])))
    provider = StairClimbingCompositionProvider()
    composition = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": "stair.go2_12dof.unified_v3"},
        context=TaskCompositionContext(project_root=ROOT, output_dir=base / "final_composition"))
    checked = seal_and_verify_candidate(candidate, artifact_root=base / "final_sealed",
                                        verifier=composition.candidate_admission_policy)
    result = checked.result.to_public_dict()
    result["verified_artifact_present"] = checked.verified_artifact is not None
    if checked.verified_artifact is not None:
        result["execution_authorization"] = checked.verified_artifact.execution_authorization.to_public_dict()
    (base / "execution_admission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": checked.result.status, "reason": checked.result.reason,
                      "verified_artifact_present": checked.verified_artifact is not None}))
    assert checked.result.status == "pass" and checked.verified_artifact is not None


if __name__ == "__main__":
    main()
