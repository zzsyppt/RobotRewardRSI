"""Replay the unmodified rejected A01 candidate through production smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.candidates.component_smoke_runtime import run_component_smoke_check
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.tasks.push_recovery.component_smoke import PushRecoveryComponentSmokeAdapter
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig


def main():
    evidence = Path(__file__).resolve().parent / "attempt_01"
    source = next((evidence / "session/provider_candidates").glob(
        "*/lane-a/001*/training_code/components.py"))
    payload = source.read_bytes()
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory)
        (candidate / "training_code").mkdir()
        (candidate / "training_code/components.py").write_bytes(payload)
        (candidate / "candidate_manifest.yaml").write_text(
            "schema_version: true_rl_agent.candidate.v1\n"
            "candidate_id: push_standing_recovery_push_compensated_v4\n"
            "entrypoint: training_code.components:build_training_components\n",
            encoding="utf-8",
        )
        ref = CandidatePackageRef(
            candidate_id="push_standing_recovery_push_compensated_v4",
            path=str(candidate),
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
        )
        for cell in ("push.go2_12dof.recovery_v3", "push.g1_37dof.recovery_v3"):
            adapter = PushRecoveryComponentSmokeAdapter(PushRecoveryTaskConfig.for_cell(cell))
            results[cell] = run_component_smoke_check(ref, adapter)
            assert results[cell]["status"] == "pass", results[cell]
            assert results[cell]["checked_reward_terms"] == 5
    receipt = {
        "status": "PASS",
        "source": source.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes_unchanged": source.read_bytes() == payload,
        "results": results,
    }
    (evidence / "first_candidate_production_smoke_audit.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "source_sha256": receipt["source_sha256"]}))


if __name__ == "__main__":
    main()
