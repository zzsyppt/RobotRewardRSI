"""Export a sealed journal user payload without rebuilding a prompt or calling a provider."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from true_rl_agent.prompt_artifacts import RoundAuthoringJournal, verify_prompt_payload
from true_rl_agent.prompting.payload import serialize_prompt_payload


def export_payload(round_dir: Path, round_id: str, sequence: int, output: Path,
                   external_request_dir: Path | None = None) -> dict:
    snapshot = RoundAuthoringJournal(round_dir, round_id=round_id).snapshot()
    attempt = next(a for a in snapshot.attempts if a["sequence"] == sequence)
    if attempt["status"] != "completed" and sequence != 1 and external_request_dir is None:
        raise ValueError("pending later attempts require the original external request binding")
    prompt_path = round_dir / "authoring_attempts" / attempt["attempt_id"] / "prompt_payload.json"
    ref = next(r for r in snapshot.prompt_payload_refs if Path(r.path).resolve() == prompt_path.resolve())
    user = serialize_prompt_payload(verify_prompt_payload(ref)).encode("utf-8")
    info = {"scope": "sealed_journal_user_payload_only", "round_id": round_id,
            "attempt_id": attempt["attempt_id"], "sequence": sequence,
            "journal_status": attempt["status"], "prompt_payload_ref": ref.to_public_dict(),
            "historical_replay": "original_saved_input; not_rebuilt_with_current_code"}
    system = None
    if external_request_dir is not None:
        binding = json.loads(prompt_path.with_name("external_request_binding.json").read_bytes())
        request = json.loads((external_request_dir / "request.json").read_bytes())
        expected = {"round_id": round_id, "attempt_id": attempt["attempt_id"],
                    "sequence": sequence, "prompt_payload": ref.to_public_dict()}
        if (binding["authoring_attempt"] != expected or request["authoring_attempt"] != expected
                or binding["request"] != request
                or Path(binding["request_dir"]).resolve() != external_request_dir.resolve()):
            raise ValueError("external request differs from journal binding")
        system = (external_request_dir / "system.txt").read_bytes()
        digest = lambda value: "sha256:" + hashlib.sha256(value).hexdigest()
        if ((external_request_dir / "user.txt").read_bytes() != user
                or digest(user) != request["user_sha256"] or digest(system) != request["system_sha256"]):
            raise ValueError("external system/user bytes differ from sealed request")
        info.update(scope="verified_external_system_and_user", external_request_dir=str(external_request_dir.resolve()),
                    system_sha256=request["system_sha256"], user_sha256=request["user_sha256"])
    # Create a separate export only after validating all requested evidence.
    output.mkdir(parents=True, exist_ok=False)
    (output / "user_payload.json").write_bytes(user)
    if system is not None:
        (output / "system.txt").write_bytes(system)
    (output / "export_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New export directory")
    parser.add_argument("--external-request-dir", type=Path,
                        help="Also verify the persisted external system/user binding")
    args = parser.parse_args()
    info = export_payload(args.round_dir, args.round_id, args.sequence, args.output, args.external_request_dir)
    print(json.dumps({"scope": info["scope"], "bytes": info["prompt_payload_ref"]["size_bytes"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
