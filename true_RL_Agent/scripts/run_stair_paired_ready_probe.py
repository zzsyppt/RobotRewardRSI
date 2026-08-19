#!/usr/bin/env python3
"""Prepare and run the Stair P6-A2 author-only paired-ready probe."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.composition.base import TaskCompositionContext  # noqa: E402
from true_rl_agent.fs_contract import read_bounded_regular_file  # noqa: E402
from true_rl_agent.json_contract import strict_json_object_loads  # noqa: E402
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.paired_ready_probe import (  # noqa: E402
    create_stair_paired_ready_probe_plan,
    load_stair_paired_ready_probe_plan,
)
from true_rl_agent.tasks.stair_climbing.paired_ready_probe_runtime import (  # noqa: E402
    STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_CHANGED_PATHS,
    prepare_stair_paired_ready_source_cases,
    run_stair_paired_ready_probe_cases,
)


PLAN_FILENAME = "probe_plan.json"
_MAX_SOURCE_LEDGER_BYTES = 64 * 1024 * 1024
_RUNTIME_SOURCE_PATHS = (
    Path("scripts/run_stair_paired_ready_probe.py"),
    Path("src/true_rl_agent/tasks/stair_climbing/paired_ready_probe.py"),
    Path("src/true_rl_agent/tasks/stair_climbing/paired_ready_probe_runtime.py"),
)
_RUNTIME_CORRECTION_SOURCE_PATHS = tuple(
    Path(path)
    for path in STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_CHANGED_PATHS
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or execute the zero-GPU, zero-eval Stair paired-ready probe."
        )
    )
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider-env-file", type=Path, default=None)
    parser.add_argument(
        "--resume-receipt-projection-fix",
        action="store_true",
        help=(
            "Resume only the plan-bound R01 AuthoringEvidenceRef projection "
            "failure through a create-once runtime correction receipt."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_root = _existing_directory(args.source_root, "source_root")
    source_archive = _existing_file(args.source_archive, "source_archive")
    output_dir = _absolute(args.output_dir)
    if args.mode == "prepare":
        if args.provider_env_file is not None:
            raise ValueError("prepare mode does not consume provider credentials")
        if args.resume_receipt_projection_fix:
            raise ValueError("prepare mode cannot resume a runtime correction")
        _prepare(
            source_root=source_root,
            source_archive=source_archive,
            output_dir=output_dir,
        )
        return 0

    env_file = _existing_file(args.provider_env_file, "provider_env_file")
    result = _run(
        source_root=source_root,
        source_archive=source_archive,
        output_dir=_existing_directory(output_dir, "output_dir"),
        env_file=env_file,
        resume_receipt_projection_fix=args.resume_receipt_projection_fix,
    )
    print(
        "P6-A2 paired-ready result: "
        f"go={result['go']} valid={result['valid_structurally_distinct_count']}/15 "
        f"leaks={result['equivalent_leak_count']}"
    )
    return 0 if result["go"] is True else 2


def _prepare(*, source_root: Path, source_archive: Path, output_dir: Path) -> None:
    _require_reproducible_source_commit()
    if output_dir.exists():
        raise FileExistsError("paired-ready output_dir must be fresh")
    output_dir.mkdir(parents=True, exist_ok=False)
    composition, prompt_readiness = _composition(output_dir)
    task_config = composition.task_plugin.config
    prepared = prepare_stair_paired_ready_source_cases(
        source_root=source_root,
        task_config=task_config,
        admission_policy=composition.candidate_admission_policy,
        candidate_capability_profile=(
            composition.task_plugin.task_spec.candidate_capability_profile
        ),
        authoring_capability_readiness=prompt_readiness,
    )
    identity = _source_identity(
        source_root=source_root,
        source_archive=source_archive,
    )
    plan = create_stair_paired_ready_probe_plan(
        output_dir / PLAN_FILENAME,
        cases=[item.plan_input for item in prepared],
        source_commit=_source_commit(),
        source_archive_checksum=identity["source_archive_checksum"],
        source_ledger_checksum=identity["source_ledger_checksum"],
        source_session_genesis_hash=identity["source_session_genesis_hash"],
    )
    print(
        "P6-A2 plan created: "
        f"{output_dir / PLAN_FILENAME} hash={plan['plan_hash']} "
        f"cases={plan['case_count']} requests={plan['budget']['max_provider_requests']}"
    )


def _run(
    *,
    source_root: Path,
    source_archive: Path,
    output_dir: Path,
    env_file: Path,
    resume_receipt_projection_fix: bool,
) -> dict[str, Any]:
    _require_reproducible_source_commit()
    plan_path = output_dir / PLAN_FILENAME
    plan = load_stair_paired_ready_probe_plan(plan_path)
    identity = _source_identity(
        source_root=source_root,
        source_archive=source_archive,
    )
    for field, expected in identity.items():
        if plan.get(field) != expected:
            raise ValueError(f"paired-ready plan {field} no longer matches runtime")
    runtime_source_commit = _source_commit()
    runtime_correction_identity = _runtime_correction_identity(
        plan_source_commit=str(plan["source_commit"]),
        runtime_source_commit=runtime_source_commit,
        allowed=resume_receipt_projection_fix,
    )

    composition, prompt_readiness = _composition(output_dir)
    task_config = composition.task_plugin.config
    prepared = prepare_stair_paired_ready_source_cases(
        source_root=source_root,
        task_config=task_config,
        admission_policy=composition.candidate_admission_policy,
        candidate_capability_profile=(
            composition.task_plugin.task_spec.candidate_capability_profile
        ),
        authoring_capability_readiness=prompt_readiness,
    )
    smoke_adapter = composition.task_plugin.component_smoke_adapter()
    return run_stair_paired_ready_probe_cases(
        output_dir=output_dir,
        plan_path=plan_path,
        prepared_cases=prepared,
        admission_policy=composition.candidate_admission_policy,
        component_smoke_adapter=smoke_adapter,
        authoring_capability_readiness=prompt_readiness,
        env_file=env_file,
        runtime_source_commit=runtime_source_commit,
        runtime_correction_identity=runtime_correction_identity,
    )


def _composition(output_dir: Path):
    provider = StairClimbingCompositionProvider()
    values = dict(provider.cli_defaults())
    values.update(
        {
            "mode": "p6a2-author-only",
            "rounds": 15,
            "session_id": "p6a2-author-only",
            "final_confirmation_plan": None,
        }
    )
    composition = provider.resolve(
        values=values,
        context=TaskCompositionContext(
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
        ),
    )
    readiness = composition.authoring_capability_readiness
    readiness.require_selected_capabilities_ready()
    return composition, readiness.to_prompt_mapping()


def _source_identity(*, source_root: Path, source_archive: Path) -> dict[str, str]:
    ledger_path = source_root / "ledger.jsonl"
    ledger = read_bounded_regular_file(
        ledger_path,
        max_bytes=_MAX_SOURCE_LEDGER_BYTES,
        label="paired-ready source ledger",
    )
    records = [
        strict_json_object_loads(
            line,
            context=f"paired-ready source ledger line {index}",
        )
        for index, line in enumerate(ledger.splitlines(), start=1)
        if line.strip()
    ]
    if len(records) != 15:
        raise ValueError("paired-ready source ledger must contain exactly 15 records")
    genesis_hashes = {record.get("session_genesis_hash") for record in records}
    if len(genesis_hashes) != 1:
        raise ValueError("paired-ready source ledger Genesis identity is not unique")
    genesis_hash = next(iter(genesis_hashes))
    if not isinstance(genesis_hash, str) or len(genesis_hash) != 64:
        raise ValueError("paired-ready source Genesis hash is invalid")
    return {
        "source_archive_checksum": _sha256_file(source_archive),
        "source_ledger_checksum": f"sha256:{hashlib.sha256(ledger).hexdigest()}",
        "source_session_genesis_hash": genesis_hash,
    }


def _source_commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = process.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("paired-ready source commit is invalid")
    return value


def _require_reproducible_source_commit() -> None:
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if tracked.stdout.strip():
        raise ValueError("paired-ready runtime requires a clean tracked worktree")
    relative_paths = [path.as_posix() for path in _RUNTIME_SOURCE_PATHS]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative_paths],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _runtime_correction_identity(
    *,
    plan_source_commit: str,
    runtime_source_commit: str,
    allowed: bool,
) -> dict[str, Any] | None:
    if runtime_source_commit == plan_source_commit:
        if allowed:
            raise ValueError(
                "paired-ready runtime correction was requested without commit drift"
            )
        return None
    if not allowed:
        raise ValueError("paired-ready plan source_commit no longer matches runtime")

    parent = subprocess.run(
        ["git", "rev-parse", f"{runtime_source_commit}^"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if parent != plan_source_commit:
        raise ValueError(
            "paired-ready runtime correction must directly follow the plan commit"
        )
    prefix = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_root_paths = sorted(
        f"{prefix}{path.as_posix()}" for path in _RUNTIME_CORRECTION_SOURCE_PATHS
    )
    all_changed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            plan_source_commit,
            runtime_source_commit,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if sorted(
        line.strip() for line in all_changed.stdout.splitlines() if line.strip()
    ) != expected_root_paths:
        raise ValueError("paired-ready runtime correction includes unrelated changes")
    return {
        "plan_source_commit": plan_source_commit,
        "runtime_source_commit": runtime_source_commit,
        "runtime_parent_commit": parent,
        "changed_paths": [
            path.as_posix()
            for path in sorted(
                _RUNTIME_CORRECTION_SOURCE_PATHS,
                key=lambda item: item.as_posix(),
            )
        ],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _absolute(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("paired-ready paths must be absolute")
    return candidate.resolve(strict=False)


def _existing_file(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required")
    candidate = _absolute(path).resolve(strict=True)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular file")
    return candidate


def _existing_directory(path: Path, label: str) -> Path:
    candidate = _absolute(path).resolve(strict=True)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"{label} must be a directory")
    return candidate


if __name__ == "__main__":
    raise SystemExit(main())
