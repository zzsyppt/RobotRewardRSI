"""Read-only replay of completed Stair v6 reports from the retired session host.

Use run_generation_session.py --terminal-audit-only for the current shared host.
This reader requires a compatible task/backend checkout and never runs work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from true_rl_agent.tasks.stair_climbing.generation_history import (
    HistoricalStairGenerationConfig, audit_completed_stair_formal_session,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    load_stair_generation_session_cost_contract,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--cell-variant-id", required=True)
    parser.add_argument("--expected-source-commit", required=True,
                        help="Expected commit of the compatible audit checkout; original provenance is replayed separately.")
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--gpu0-uuid", required=True)
    parser.add_argument("--gpu1-uuid", required=True)
    parser.add_argument("--source-reference-genesis-dir", type=Path)
    parser.add_argument("--source-reference-session-id")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve(strict=True)
    cost = load_stair_generation_session_cost_contract(output / "generation_session_cost.json")
    config = HistoricalStairGenerationConfig(
        project_root=args.project_root.resolve(strict=True), output_dir=output,
        lease_root=args.lease_root.resolve(), session_id=args.session_id,
        cell_variant_id=args.cell_variant_id, expected_source_commit=args.expected_source_commit,
        machine_id=args.machine_id, gpu0_uuid=args.gpu0_uuid, gpu1_uuid=args.gpu1_uuid,
        max_allocated_active_gpu_hours=cost.max_allocated_active_gpu_hours,
        max_simulated_control_steps=cost.max_simulated_control_steps,
        max_output_bytes=cost.max_output_bytes, resume_existing=True,
        terminal_audit_only=True, session_mode="formal_generation_session",
        source_reference_genesis_dir=(args.source_reference_genesis_dir.resolve(strict=True)
                                     if args.source_reference_genesis_dir else None),
        source_reference_session_id=args.source_reference_session_id,
    )
    report = audit_completed_stair_formal_session(config)
    print(json.dumps({"status": "PASS", "session_id": report["session_id"],
                      "schema_version": report["schema_version"], "acceptance_authority": False},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
