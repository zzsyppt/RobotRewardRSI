"""Preflight or create the formal Stair R0 reference Genesis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.ledger import (
    PROVIDER_AUTHORING_MODE,
    load_ledger,
    primary_ledger_epoch_hash,
)
from true_rl_agent.agent_core.session_genesis import (
    SESSION_GENESIS_FILENAME,
    prepare_session_genesis,
)
from true_rl_agent.composition.base import (
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.composition.cli import add_execution_device_argument
from true_rl_agent.composition.defaults import build_application_composition_registry
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.runner.base import Runner
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.job_argv import (
    validate_eval_job_argv,
    validate_train_job_argv,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
)
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import (
    STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
    active_unified_stair_cell_variant_ids,
)


class _LoadOnlyRunner(Runner):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_train(self, train_spec):
        _ = train_spec
        self.calls.append("train")
        raise RuntimeError("load-only Genesis validation attempted training")

    def run_eval(self, eval_spec):
        _ = eval_spec
        self.calls.append("eval")
        raise RuntimeError("load-only Genesis validation attempted evaluation")

    def run_baseline_eval(self, eval_spec):
        _ = eval_spec
        self.calls.append("baseline_eval")
        raise RuntimeError("load-only Genesis validation attempted baseline evaluation")


def _build_parser(*, execution_device: object) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "qualify", "create", "load-only"),
        required=True,
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cell-variant-id",
        choices=active_unified_stair_cell_variant_ids(),
        default=STAIR_G1_37DOF_UNIFIED_CELL_VARIANT_ID,
    )
    parser.add_argument("--final-confirmation-plan", type=Path, default=None)
    add_execution_device_argument(parser, default=execution_device)
    return parser


def main(argv: list[str] | None = None) -> int:
    registry = build_application_composition_registry()
    provider = registry.get(STAIR_TASK_ID)
    task_values = dict(provider.cli_defaults())
    args = _build_parser(
        execution_device=task_values["execution_device"]
    ).parse_args(argv)
    output_dir = args.output_dir.resolve()
    task_values["execution_device"] = args.execution_device
    task_values["cell_variant_id"] = args.cell_variant_id
    if args.mode == "create" and args.final_confirmation_plan is None:
        raise ValueError(
            "Stair reference Genesis create requires --final-confirmation-plan"
        )
    if args.mode == "qualify" and args.final_confirmation_plan is not None:
        raise ValueError(
            "Stair reference qualification does not use a final-confirmation plan"
        )
    if (
        args.mode in {"create", "load-only"}
        and args.final_confirmation_plan is not None
    ):
        task_values.update(
            {
                "mode": "live",
                "rounds": STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
                "session_id": args.session_id,
                "final_confirmation_plan": args.final_confirmation_plan,
            }
        )
    composition = resolve_task_composition(
        provider=provider,
        values=task_values,
        context=TaskCompositionContext(
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
        ),
    )
    reference_artifact = composition.reference_program_artifact
    if reference_artifact is None:
        raise ValueError("Stair composition has no reference program artifact")

    if args.mode == "preflight":
        binding = composition.backend_adapter.resolve_session_evaluation_binding(
            session_id=args.session_id,
            session_dir=output_dir,
            task_spec=composition.task_plugin.task_spec,
            allow_create=True,
        )
        jobs = composition.backend_adapter.materialize_reference_jobs(
            session_id=args.session_id,
            task_spec=composition.task_plugin.task_spec,
            reference=reference_artifact,
            artifact_root=output_dir / "reference_program_artifacts",
            policy_initialization=composition.policy_initialization,
            evaluation_binding=binding,
        )
        validate_train_job_argv(
            jobs.train_spec,
            jobs.train_spec.metadata["cli_argv"],
        )
        validate_eval_job_argv(
            jobs.eval_spec,
            jobs.eval_spec.metadata["cli_argv"],
        )
        payload = {
            "status": "pass",
            "mode": "preflight",
            "session_id": args.session_id,
            "task_id": composition.task_id,
            "reference_manifest_hash": reference_artifact.manifest.manifest_hash,
            "program_source_checksum": reference_artifact.manifest.source_checksum,
            "training_binding_hash": jobs.train_spec.training_binding.binding_hash,
            "execution_device": jobs.train_spec.execution_device,
            "budget_steps": jobs.train_spec.budget_steps,
            "max_iterations": composition.task_defaults["train_max_iterations"],
            "train_subject": jobs.train_spec.subject.to_public_dict(),
            "eval_subject": jobs.eval_spec.subject.to_public_dict(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    genesis_path = output_dir / SESSION_GENESIS_FILENAME
    ledger_path = output_dir / "ledger.jsonl"
    load_only_runner = _LoadOnlyRunner()
    if args.mode == "load-only" and not genesis_path.is_file():
        raise ValueError("load-only mode requires an existing session Genesis")
    if args.mode in {"create", "qualify"} and ledger_path.exists():
        raise ValueError("reference Genesis creation requires an absent ledger")
    before_checksum = file_sha256(genesis_path) if genesis_path.is_file() else None
    before_mtime_ns = genesis_path.stat().st_mtime_ns if genesis_path.is_file() else None
    before_ledger_checksum = (
        file_sha256(ledger_path) if ledger_path.is_file() else None
    )
    before_ledger_mtime_ns = (
        ledger_path.stat().st_mtime_ns if ledger_path.is_file() else None
    )
    genesis = prepare_session_genesis(
        output_dir=output_dir,
        session_id=args.session_id,
        task_plugin=composition.task_plugin,
        backend_adapter=composition.backend_adapter,
        runner=(
            load_only_runner
            if args.mode == "load-only"
            else composition.runner
        ),
        policy_initialization=composition.policy_initialization,
        genesis_identity_snapshot=composition.genesis_identity_snapshot(),
        reference_program_artifact=reference_artifact,
        bind_backend_execution_environment=(args.mode in {"create", "qualify"}),
        reference_phased_evaluation_adapter=(
            composition.task_plugin.phased_evaluation_adapter()
        ),
    )
    if args.mode in {"create", "qualify"} and ledger_path.exists():
        raise ValueError("reference Genesis creation unexpectedly created a ledger")
    anchor = genesis.reference_anchor_public()
    if anchor is None:
        raise ValueError("created Stair Genesis has no reference anchor")
    ledger_records = load_ledger(ledger_path) if args.mode == "load-only" else []
    replay = replay_acceptance_ledger(
        ledger_records,
        initial_current_best=genesis.initial_current_best,
        expected_session_genesis_hash=genesis.session_genesis_hash,
        acceptance_protocol=composition.task_plugin.task_spec.acceptance_protocol,
        expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_initial_policy_state_hash=(
            genesis.expected_initial_policy_state_hash
        ),
        expected_ledger_epoch_hash=(
            primary_ledger_epoch_hash(genesis.session_genesis_hash)
            if args.final_confirmation_plan is not None
            else None
        ),
    )
    replay.raise_for_continuation()
    payload = {
        "status": "pass",
        "mode": args.mode,
        "session_id": args.session_id,
        "task_id": composition.task_id,
        "session_genesis_hash": genesis.session_genesis_hash,
        "initial_current_best": genesis.initial_current_best.to_public_dict(),
        "reference_anchor": anchor,
        "reference_source_checksum": reference_artifact.manifest.source_checksum,
        "cell_variant_id": args.cell_variant_id,
        "reference_evaluation_mode": "phased",
        "provider_call_count": 0,
        "ledger_state": "present" if ledger_path.is_file() else "absent",
        "ledger_record_count": len(ledger_records),
        "replay_current_best": replay.current_best.to_public_dict(),
    }
    if args.mode == "load-only":
        after_checksum = file_sha256(genesis_path)
        after_mtime_ns = genesis_path.stat().st_mtime_ns
        if load_only_runner.calls:
            raise RuntimeError("load-only mode invoked runner work")
        if before_checksum != after_checksum or before_mtime_ns != after_mtime_ns:
            raise RuntimeError("load-only mode changed the session Genesis")
        if not ledger_records and replay.current_best != genesis.initial_current_best:
            raise RuntimeError("empty-ledger replay changed the Genesis current-best")
        if before_ledger_checksum is None:
            if ledger_path.exists():
                raise RuntimeError("load-only mode created a session ledger")
            after_ledger_checksum = None
            after_ledger_mtime_ns = None
        else:
            if not ledger_path.is_file():
                raise RuntimeError("load-only mode removed the session ledger")
            after_ledger_checksum = file_sha256(ledger_path)
            after_ledger_mtime_ns = ledger_path.stat().st_mtime_ns
            if (
                before_ledger_checksum != after_ledger_checksum
                or before_ledger_mtime_ns != after_ledger_mtime_ns
            ):
                raise RuntimeError("load-only mode changed the session ledger")
        payload["load_only_evidence"] = {
            "status": "pass",
            "genesis_checksum_before": before_checksum,
            "genesis_checksum_after": after_checksum,
            "genesis_mtime_ns_before": before_mtime_ns,
            "genesis_mtime_ns_after": after_mtime_ns,
            "runner_call_count": 0,
            "provider_call_count": 0,
            "replay_status": "pass",
            "replay_ledger_record_count": len(ledger_records),
            "ledger_checksum_before": before_ledger_checksum,
            "ledger_checksum_after": after_ledger_checksum,
            "ledger_mtime_ns_before": before_ledger_mtime_ns,
            "ledger_mtime_ns_after": after_ledger_mtime_ns,
            "reference_source_status": (
                "pass"
                if genesis.reference_program_source()
                == reference_artifact.source_text
                else "fail"
            ),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
