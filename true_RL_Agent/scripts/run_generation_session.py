"""Prepare, run or independently audit one task × robot generation session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from true_rl_agent.agent_core.generation_application import REPORT_FILENAME, run_generation_session
from true_rl_agent.agent_core.generation_cost import FrozenGenerationCostContract
from true_rl_agent.agent_core.generation_preparation import GenerationLaunchConfig, prepare_generation_foundation
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from true_rl_agent.composition.cli import (
    add_task_runtime_arguments, add_task_selection_argument, select_task_composition_provider,
)
from true_rl_agent.composition.defaults import build_application_composition_registry
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.runner.source_checkout import require_source_checkout
from true_rl_agent.runner.gpu_process_binding import discover_gpu_inventory
from true_rl_agent.runner.process_handle import isolate_orchestrator_process_group


def parse_args(argv=None, *, registry=None):
    registry = registry or build_application_composition_registry()
    provider = select_task_composition_provider(argv, registry)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_task_selection_argument(parser, registry=registry, provider=provider)
    add_task_runtime_arguments(parser, provider.cli_defaults())
    provider.add_cli_arguments(parser)
    # This formal entrypoint requires an intentional matrix selection. Other
    # application entrypoints may still use their task-owned defaults.
    parser.set_defaults(task_id=None, cell_variant_id=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--lease-root", type=Path, required=True)
    parser.add_argument("--provider-env-file", type=Path, required=True)
    parser.add_argument("--provider-policy", type=Path, required=True)
    parser.add_argument("--cost-contract", type=Path, required=True,
                        help="Frozen per-operation cost contract for this exact cell and workload.")
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--gpu0-uuid", required=True)
    parser.add_argument("--gpu1-uuid", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--max-allocated-active-gpu-hours", type=float, required=True)
    parser.add_argument("--max-simulated-control-steps", type=int, required=True)
    parser.add_argument("--max-output-bytes", type=int, required=True)
    parser.add_argument("--source-reference-genesis-dir", type=Path)
    parser.add_argument("--source-reference-session-id")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--terminal-audit-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pause-after-generation", type=int, choices=(3, 6, 9, 12))
    parser.add_argument("--nvidia-smi-executable", default="nvidia-smi")
    parser.add_argument("--final-replication-source", type=Path)
    args = parser.parse_args(argv)
    if args.task_id is None or args.cell_variant_id is None:
        parser.error("formal sessions require explicit --task-id and --cell-variant-id")
    args._composition_provider = provider
    return args


def prepare_from_args(args):
    output = args.output_dir.resolve()
    if output.exists() != args.resume_existing:
        raise ValueError("new sessions require a new directory; resume requires an existing directory")
    source = require_source_checkout(PROJECT_ROOT, expected_commit=args.expected_source_commit)
    provider = args._composition_provider
    context = TaskCompositionContext(project_root=PROJECT_ROOT, output_dir=output,
                                     read_only=args.resume_existing)
    values = {**provider.cli_defaults(), **vars(args)}
    values["max_generations"] = 15
    # This pass freezes science. The second pass adds the final execution plan;
    # task providers must keep that operational addition outside scientific IDs.
    composition = resolve_task_composition(provider=provider, values=values, context=context)
    plan = composition.task_plugin.prepare_final_confirmation(output_dir=output,
        session_id=args.session_id, required_closed_generations=15,
        resume_existing=args.resume_existing, replication_source=args.final_replication_source)
    values["mode"] = "formal_generation_session"
    values["final_confirmation_plan"] = output / "final_confirmation" / "plan.json"
    composition = resolve_task_composition(provider=provider, values=values, context=context)
    cost = FrozenGenerationCostContract.from_mapping(load_strict_json_object(args.cost_contract))
    policy = load_strict_json_object(args.provider_policy)
    expected = (args.gpu0_uuid, args.gpu1_uuid)
    if not args.terminal_audit_only:
        inventory = discover_gpu_inventory(args.nvidia_smi_executable)
        if set(inventory) != {0, 1}:
            raise ValueError("formal sessions require exactly host GPU indices 0 and 1")
        if tuple(inventory[index] for index in range(2)) != expected:
            raise ValueError("selected GPU indices do not match their frozen UUIDs")
    launch = GenerationLaunchConfig(project_root=PROJECT_ROOT, output_dir=output,
        lease_root=args.lease_root.resolve(), provider_env_file=args.provider_env_file.resolve(),
        session_id=args.session_id, cell_variant_id=composition.resolved_cell_manifest.cell_variant_id,
        machine_id=args.machine_id, gpu0_uuid=args.gpu0_uuid, gpu1_uuid=args.gpu1_uuid,
        max_allocated_active_gpu_hours=args.max_allocated_active_gpu_hours,
        max_simulated_control_steps=args.max_simulated_control_steps, max_output_bytes=args.max_output_bytes,
        source_reference_genesis_dir=(args.source_reference_genesis_dir.resolve()
            if args.source_reference_genesis_dir else None),
        source_reference_session_id=args.source_reference_session_id,
        resume_existing=args.resume_existing, terminal_audit_only=args.terminal_audit_only,
        pause_after_generation=args.pause_after_generation, nvidia_smi_executable=args.nvidia_smi_executable)
    return prepare_generation_foundation(launch, composition=composition, source_commit=source.head_commit,
        inventory=expected, cost=cost, final_plan=plan, provider_policy=policy)


def main(argv=None, *, registry=None):
    args = parse_args(argv, registry=registry)
    if not args.prepare_only and not args.terminal_audit_only:
        isolate_orchestrator_process_group()
    foundation = prepare_from_args(args)
    if args.prepare_only:
        result = {"status": "prepared", "session_contract_hash": foundation.session_contract.contract_hash,
                  "output_dir": str(foundation.config.output_dir)}
    else:
        if not args.terminal_audit_only and not (foundation.config.output_dir / REPORT_FILENAME).exists():
            from true_rl_agent.agent_core.generation_runtime import generation_provider_config
            from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider

            provider = OpenAICompatibleCandidateProvider(generation_provider_config(
                output_dir=foundation.config.output_dir / "provider_candidates",
                env_file=foundation.config.provider_env_file, policy=foundation.provider_policy))
            provider.validate_credential()
        result = run_generation_session(foundation)
    print(json.dumps(result.get("checks", result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
