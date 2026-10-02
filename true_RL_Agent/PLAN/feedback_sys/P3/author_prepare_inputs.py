"""Write path-portable, hash-bound references to already qualified author parents."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from author_experiment import digest, read, reference_context, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stair-qualification", type=Path, required=True)
    p.add_argument("--stair-training-summary", type=Path, required=True)
    p.add_argument("--push-session", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.root / "src"))
    from true_rl_agent.composition.cell_manifest import load_resolved_cell_manifest
    from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
    def ref(path):
        return {"path": str(path.resolve()), "sha256": digest(path)}
    q = a.stair_qualification
    stair = {"task": "stair", "mode": "existing_stair_qualification",
        "context_authority": "new empty diagnostic-author namespace; no persisted Genesis, search admission or new qualification",
        "composition_values": {"cell_variant_id": "stair.go2_12dof.unified_v3"},
        "reference_program_source_checksum": "sha256:10b1a34218d7fbb3ed124a6913e6d45d1462e5508d062510f31a3d1d7ac272cb",
        "checkpoint_hash": "sha256:7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c",
        "qualification_result": ref(q / "result.json"), "qualification_assembly": ref(q / "assembly.json"),
        "training_summary": ref(a.stair_training_summary)}
    genesis_path = a.push_session / "session_genesis.json"
    g = read(genesis_path)
    anchor = g["anchor"]
    config = PushRecoveryTaskConfig.from_manifest(load_resolved_cell_manifest((a.push_session / "resolved_cell_manifest.json").resolve()))
    push = {"task": "push", "mode": "existing_reference_genesis", "context_authority": "empty history anchored in genuine recovery_v2 velocity qualification; not force-v3 qualification",
        "composition_values": {"cell_variant_id": "push.g1_37dof.recovery_v2", "push_protocol": config.push_protocol.to_dict(),
            "reference_program_id": anchor["reference_manifest"]["reference_id"], "reference_admission_policy": None,
            "eval_base_seed": 1000000000},
        "cohort_seed_provenance": "P6/g1_qualified_01/records/g1_fresh_qualification_declaration_01.json command; verified against full Genesis evaluation binding",
        "reference_program_source_checksum": anchor["reference_manifest"]["source_checksum"], "checkpoint_hash": anchor["checkpoint_hash"],
        "genesis": ref(genesis_path)}
    a.output.mkdir(parents=True, exist_ok=True)
    for task, spec in (("stair", stair), ("push", push)):
        comp, best, _, _, namespace, _ = reference_context(spec, root=a.root, base=a.output, output=a.output / "validation" / task)
        spec["validated_evaluation_binding"] = best.evaluation_binding().to_public_dict()
        spec["diagnostic_replay_namespace"] = namespace
        write(a.output / f"author_context_{task}.json", spec)
    print("PASS: two existing reference contexts restored; no GPU/API")


if __name__ == "__main__":
    main()
