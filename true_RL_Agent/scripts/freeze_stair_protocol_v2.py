#!/usr/bin/env python3
"""Materialize StairProtocolV2 from historical Phase D and operator revisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_G1_ROUGH_AGENT_CONFIGURATION,
)
from true_rl_agent.tasks.stair_climbing.local_paths import (  # noqa: E402
    resolve_existing_directory,
    resolve_existing_file,
    resolve_fresh_path,
)
from true_rl_agent.tasks.stair_climbing.phase_d_run import (  # noqa: E402
    load_validated_stair_phase_d_run,
)
from true_rl_agent.tasks.stair_climbing.pilot_artifact_store import (  # noqa: E402
    validate_stair_phase_d_pilot_artifact,
)
from true_rl_agent.tasks.stair_climbing.pilot_artifacts import (  # noqa: E402
    load_canonical_pilot_json,
)
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (  # noqa: E402
    validate_stair_protocol_v2_freeze_candidate,
)
from true_rl_agent.tasks.stair_climbing.protocol_v2_freeze import (  # noqa: E402
    build_stair_protocol_v2_payload,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-v2-draft", required=True)
    parser.add_argument("--pilot-artifact-manifest", required=True)
    parser.add_argument("--pilot-artifact-store-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        draft_path = _existing_file(args.protocol_v2_draft, "protocol_v2_draft")
        artifact_manifest_path = _existing_file(
            args.pilot_artifact_manifest,
            "pilot_artifact_manifest",
        )
        artifact_store_root = _existing_directory(
            args.pilot_artifact_store_root,
            "pilot_artifact_store_root",
        )
        artifact = validate_stair_phase_d_pilot_artifact(
            manifest=load_canonical_pilot_json(
                artifact_manifest_path,
                context="stair Phase D pilot artifact manifest",
            ),
            artifact_store_root=artifact_store_root,
        )
        output = _fresh_output(args.output)
        report = load_validated_stair_phase_d_run(
            protocol_path=draft_path,
            output_root=artifact["payload_root"],
            project_root=PROJECT_ROOT,
        )
        payload = build_stair_protocol_v2_payload(
            draft_path=draft_path,
            phase_d_report=report,
            pilot_artifact_manifest=artifact["manifest"],
            agent_configuration=STAIR_G1_ROUGH_AGENT_CONFIGURATION,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_json_bytes(payload, trailing_newline=True))
        validated_hash = validate_stair_protocol_v2_freeze_candidate(
            output,
            base_v1_path=draft_path.with_name("stair_protocol_v1.json"),
        )
        if validated_hash != payload["protocol_hash"]:
            raise ValueError("generated Stair V2 candidate hash drifted")
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed
        print(str(exc), file=sys.stderr)
        return 1
    print(
        "stair protocol V2 materialized "
        f"budget_steps={payload['production_training']['budget_steps']} "
        f"protocol_hash={payload['protocol_hash']}",
        flush=True,
    )
    return 0


def _existing_file(value: object, field_name: str) -> Path:
    return resolve_existing_file(Path(str(value)), field_name=field_name)


def _existing_directory(value: object, field_name: str) -> Path:
    return resolve_existing_directory(Path(str(value)), field_name=field_name)


def _fresh_output(value: object) -> Path:
    return resolve_fresh_path(Path(str(value)), field_name="Stair V2 output")


if __name__ == "__main__":
    raise SystemExit(main())
