"""Generate, refine and freeze Stair evaluation declarations for any robot row."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.robots.defaults import build_default_robot_registry
from true_rl_agent.tasks.stair_climbing.calibration import (
    build_stair_calibration_suite, freeze_stair_formal_suite, refine_stair_calibration_suite,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row
from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
from true_rl_agent.tasks.stair_climbing.unified_protocol import StairUnifiedProtocol
from true_rl_agent.tasks.stair_climbing.unified_scoring import validate_unified_stair_summary
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_protocol(path, source, payload, suite, acceptance):
    payload["component_hashes"].update(evaluation_suite_hash=suite.suite_hash,
                                        acceptance_protocol_hash=acceptance.protocol_hash)
    protocol = replace(source, _payload=payload, evaluation_suite=suite, acceptance_protocol=acceptance)
    payload["task_protocol_hash"] = protocol.task_protocol_hash
    payload["protocol_hash"] = protocol.protocol_hash
    write(path.with_name(payload["component_refs"]["evaluation_suite"]), suite.to_dict())
    write(path, payload)
    return StairUnifiedProtocol.load(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-suite", help="no predecessor evaluation suite required")
    generate.add_argument("--training-profile", type=Path, required=True)
    generate.add_argument("--robot-profile-id", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--control-dt-s", type=float, default=0.02)
    generate.add_argument("--horizon-control-steps", type=int, default=600)
    prepare = commands.add_parser("prepare", help="evaluation-only successor of a registered training row")
    prepare.add_argument("--source-cell-variant-id", required=True)
    prepare.add_argument("--cell-variant-id", required=True)
    prepare.add_argument("--suite-path", type=Path, required=True)
    prepare.add_argument("--acceptance-protocol", type=Path, required=True)
    revise = commands.add_parser("revise")
    revise.add_argument("--measurement", type=Path, required=True)
    revise.add_argument("--purpose", choices=("refine", "formal"), required=True)
    revise.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()
    registry = build_default_robot_registry()
    if args.command == "generate-suite":
        suite = build_stair_calibration_suite(
            training_profile=StairUnifiedTrainingProfile.load(args.training_profile),
            robot_profile=registry.profile(args.robot_profile_id),
            geometry_policy={"control_dt_s": args.control_dt_s,
                             "anchor_horizon_control_steps": args.horizon_control_steps})
        write(args.output, suite.to_dict())
        print(suite.suite_hash)
        return
    if args.command == "prepare":
        old_row, row = stair_robot_row(args.source_cell_variant_id), stair_robot_row(args.cell_variant_id)
        config = StairClimbingTaskConfig.defaults(ROOT / old_row.protocol_relative_path)
        binding = row.load_binding(project_root=ROOT)
        suite = build_stair_calibration_suite(
            training_profile=config.unified_training_profile,
            robot_profile=registry.profile(binding.robot_profile_id),
            geometry_policy=config.ordered_evaluation_suite.to_dict()["geometry_policy"])
        acceptance = AcceptanceProtocol.from_mapping(json.loads(args.acceptance_protocol.read_text(encoding="utf-8")))
        source_manifest = resolve_stair_cell_manifest(
            cell_variant_id=old_row.cell_variant_id, config=config, plugin=StairClimbingTaskPlugin(config))
        payload = config.unified_protocol.to_dict()
        payload["training_lineage"] = {
            "mode": "evaluation_only_successor", "source_cell_variant_id": old_row.cell_variant_id,
            "source_base_task_protocol_hash": config.base_task_protocol_hash,
            "source_resolved_cell_task_protocol_hash": source_manifest.resolved_cell_task_protocol_hash,
            "source_resolved_cell_manifest_hash": source_manifest.manifest_hash,
            "source_training_profile_hash": config.unified_training_profile.profile_hash,
        }
        version = row.cell_variant_id.rsplit("_v", 1)[1]
        payload.update(protocol_id=payload["protocol_id"].rsplit("_v", 1)[0] + "_v" + version,
                       cell_variant_id=row.cell_variant_id, protocol_state="draft")
        path = ROOT / row.protocol_relative_path
        if args.suite_path.resolve().parent != path.parent or args.acceptance_protocol.resolve().parent != path.parent:
            raise ValueError("Stair protocol components belong beside the protocol file")
        payload["component_refs"].update(evaluation_suite=args.suite_path.name,
                                          acceptance_protocol=args.acceptance_protocol.name)
        frozen = write_protocol(path, config.unified_protocol, payload, suite, acceptance)
        print(row.cell_variant_id, frozen.protocol_hash)
        return
    measurement = json.loads(args.measurement.read_text(encoding="utf-8"))
    if measurement["status"] != "PASS":
        raise ValueError("Calibration execution failed")
    row = stair_robot_row(measurement["cell"]["cell_variant_id"])
    path = ROOT / row.protocol_relative_path
    config = StairClimbingTaskConfig.defaults(path)
    if config.protocol_state != "draft":
        raise ValueError("Cannot revise a signed active protocol")
    source = StairUnifiedEvaluationSuite.from_mapping(
        measurement["suite"], training_profile=config.unified_training_profile)
    if source.suite_hash != config.ordered_evaluation_suite.suite_hash:
        raise ValueError("Calibration evidence is not for the current draft suite")
    summary = validate_unified_stair_summary(suite=source, semantics=config.evaluation_semantics,
                                            split="calibration", summary=measurement["summary"])
    if args.purpose == "refine":
        suite, decision = refine_stair_calibration_suite(
            suite=source, summary=summary, training_profile=config.unified_training_profile)
    else:
        suite = freeze_stair_formal_suite(calibration_suite=source, calibration_summary=summary,
                                         training_profile=config.unified_training_profile)
        decision = {"source_suite_hash": source.suite_hash, "formal_suite_hash": suite.suite_hash}
    write_protocol(path, config.unified_protocol, config.unified_protocol.to_dict(),
                   suite, config.acceptance_protocol)
    write(args.decision, decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
