"""Bind review task definitions to numeric archives and their source contracts.

This is a PLAN-only evidence helper. It does not infer behavior from a report,
grant author permissions, or prove that today's source bytes ran in an old job.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "p3new.review_task_contract.v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def pointer(value: object, ref: str) -> object:
    result = value
    for part in ref.lstrip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        result = result[int(part)] if isinstance(result, list) else result[part]
    return result


def push_field_constants() -> dict[str, list[str]]:
    path = ROOT / "src/true_rl_agent/tasks/push_recovery/numeric_capture.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"POSE_FIELDS", "PLANAR_MOTION_FIELDS", "CONDITIONS"}
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                result[name] = list(ast.literal_eval(node.value))
    if set(result) != wanted:
        raise ValueError("Push capture field constants are absent")
    return result


def excerpt(relative: str, first: str, last: str) -> dict:
    path = ROOT / relative
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if first in line]
    if len(starts) != 1:
        raise ValueError(f"source anchor is not unique: {relative}: {first}")
    start = starts[0]
    ends = [i for i in range(start, len(lines)) if last in lines[i]]
    if not ends:
        raise ValueError(f"source end anchor absent: {relative}: {last}")
    end = ends[0]
    return {"file": relative, "sha256": digest(path), "first_line": start + 1,
            "last_line": end + 1, "verbatim": "\n".join(lines[start:end + 1])}


def add(fields: list[dict], manifest: dict, name: str, ref: str) -> None:
    fields.append({"name": name, "numeric_manifest_pointer": ref, "value": pointer(manifest, ref)})


def reward_input_contract(task: str, manifest: dict, resolved_cell_path: Path | None) -> dict:
    from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
    if task == "stair":
        from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
        resolved = ResolvedCellManifestV1.from_mapping(manifest["metadata"]["resolved_cell_manifest"])
        projection = stair_runtime_projection_from_manifest(resolved)
        archived = manifest["metadata"]["runtime_projection"]
        if projection.to_summary_dict() != archived:
            raise ValueError("Stair reward-input projection differs from the captured projection")
        return {"status": "bound_to_archived_schema", "input_schema": projection.reward_input_schema_vnext(),
                "schema_hash": projection.reward_input_schema_hash_vnext,
                "archive_schema_pointer": "/metadata/runtime_projection/reward_input_schema_hash_vnext",
                "kinematic_base_source": archived["kinematic_base_source"]}
    if resolved_cell_path is None:
        return {"status": "not_supplied", "reason": "Push requires its independently bound resolved-cell manifest to identify the reward ABI"}
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryV2RewardInputs
    resolved = ResolvedCellManifestV1.from_mapping(read(resolved_cell_path))
    if resolved.manifest_hash != manifest["identity"]["resolved_cell_manifest_hash"]:
        raise ValueError("Push reward ABI resolved-cell manifest differs from the numeric identity")
    schema = {"schema_version": "true_rl_agent.push_reward_inputs.v2", "fields": list(PushRecoveryV2RewardInputs.field_names())}
    schema_hash = canonical_json_hash(schema)
    if resolved.contract_hashes.get("reward_input_schema_hash") != schema_hash:
        raise ValueError("Push reward-input schema differs from its bound resolved-cell manifest")
    return {"status": "bound_to_archived_schema", "input_schema": schema, "schema_hash": schema_hash,
            "resolved_cell_source_sha256": digest(resolved_cell_path),
            "archive_identity_pointer": "/identity/resolved_cell_manifest_hash"}


def build(task: str, numeric_path: Path, sources_path: Path, resolved_cell_path: Path | None = None) -> dict:
    numeric_path, sources_path = numeric_path.resolve(), sources_path.resolve()
    manifest, sources = read(numeric_path), read(sources_path)
    numeric_sha = digest(numeric_path)
    matches = [row for row in sources.get("sources", [])
               if row.get("manifest_sha256") == numeric_sha
               and Path(row.get("manifest_path", "")).resolve() == numeric_path]
    if not matches:
        raise ValueError("source manifest does not bind this numeric manifest by path and SHA256")
    expected_task = "unitree_stair_climbing" if task == "stair" else "unitree_push_recovery"
    if manifest["identity"]["task_id"] != expected_task or manifest.get("complete") is not True:
        raise ValueError("wrong task or incomplete numeric archive")
    metadata = manifest["metadata"]
    fields: list[dict] = []
    for name, ref in (("task_identity", "/identity/task_id"),
                      ("capture_profile", "/metadata/profile"),
                      ("media_status", "/metadata/media_status")):
        add(fields, manifest, name, ref)
    contract: dict = {"schema_version": SCHEMA, "task": task,
        "scope": "Definitions and provenance only; not a necessary-fact label or report verdict",
        "generator": {"file": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
                      "sha256": digest(Path(__file__))},
        "numeric_manifest": {"sha256": numeric_sha},
        "source_manifest": {"sha256": digest(sources_path), "matching_manifest_entries": len(matches)},
        "manifest_fields": fields, "source_contracts": [], "binding_checks": {}, "limitations": []}
    if task == "stair":
        candidates = []
        for path in (ROOT / "configs/tasks").glob("stair*.json"):
            value = read(path)
            if value.get("task_protocol_hash") == metadata["semantics"]["task_protocol_hash"]:
                candidates.append((path, value))
        if len(candidates) != 1:
            raise ValueError(f"expected one Stair protocol matching archived semantics, found {len(candidates)}")
        protocol_path, protocol = candidates[0]
        if protocol["episode_outcomes"] != metadata["semantics"]["episode_outcomes"]:
            raise ValueError("Stair protocol outcomes differ from archived numeric semantics")
        if protocol["task_id"] != expected_task:
            raise ValueError("Stair protocol task differs from archive")
        contract["task_protocol"] = {"file": str(protocol_path.resolve().relative_to(ROOT)).replace("\\", "/"),
            "sha256": digest(protocol_path),
            "task_protocol_hash": protocol["task_protocol_hash"],
            "definitions": [{"name": name, "source_pointer": ref, "numeric_manifest_pointer": other,
                             "value": pointer(protocol, ref)}
                            for name, ref, other in (
                                ("outcome_abi", "/episode_outcomes", "/metadata/semantics/episode_outcomes"),
                                ("task_id", "/task_id", "/identity/task_id"))],
            "author_permission_definition": {"source_pointer": "/candidate_authority",
                                             "value": protocol["candidate_authority"]}}
        for name, ref in (("clock_and_coverage", "/metadata/coverage"),
                          ("coordinates", "/metadata/coordinates"),
                          ("body_bindings", "/metadata/runtime_projection/resolved_roles"),
                          ("invalid_padding", "/metadata/invalid_padding"),
                          ("contact_history", "/metadata/contact_force_history")):
            add(fields, manifest, name, ref)
        contract["source_contracts"] = [
            excerpt("src/true_rl_agent/tasks/stair_climbing/numeric_capture.py",
                    '"semantics": config.evaluation_semantics.to_dict()', '"media_status": "not_collected"'),
            excerpt("src/true_rl_agent/tasks/stair_climbing/numeric_capture.py",
                    'raw_semantics = context["semantics"]', 'raise NumericEvidenceError("Stair archived semantics mismatch")'),
            excerpt("src/true_rl_agent/tasks/stair_climbing/candidate_surface.py",
                    'boundary = protocol.to_dict()["candidate_boundary"]', 'code_level_components=frozenset(boundary["code_level_components"])')]
        contract["source_contracts"].append(excerpt(
            "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
            '"resolved_roles": {', '"forbidden_contact": list(self.forbidden_contact_body_names)'))
        contract["source_contracts"].append(excerpt(
            "src/true_rl_agent/tasks/stair_climbing/runtime_projection.py",
            'def reward_input_fields_vnext(', 'def reward_input_schema_hash_vnext('))
        contract["binding_checks"] = {"source_manifest_matches_numeric_sha": True,
            "protocol_hash_matches_archived_semantics": True, "outcome_abi_matches_archived_semantics": True,
            "protocol_task_matches_archive": True}
        contract["limitations"] = [
            "The archived evaluation_binding.task_protocol_hash is a resolved-cell hash, not the base Stair task_protocol_hash; these are deliberately not equated.",
            "Current source-file SHA records review-time source bytes. The old numeric archive does not by itself prove those exact bytes executed; use the frozen execution source manifest for that stronger claim."]
    else:
        if manifest["identity"].get("resolved_protocol_hash") != metadata["parameters"]["resolved_protocol_hash"]:
            raise ValueError("Push resolved protocol identity differs from captured parameters")
        if manifest["identity"].get("task_numeric_schema") != metadata["schema_version"]:
            raise ValueError("Push numeric schema identity differs from metadata")
        constants = push_field_constants()
        for source_name, metadata_name in (("POSE_FIELDS", "pose_fields"),
                                           ("PLANAR_MOTION_FIELDS", "planar_motion_fields"),
                                           ("CONDITIONS", "condition_fields")):
            if constants[source_name] != metadata[metadata_name]:
                raise ValueError(f"Push archived {metadata_name} differs from source capture declaration")
        for name, ref in (("capture_point", "/metadata/capture_point"),
                          ("disturbance_mechanism", "/metadata/parameters/mechanism"),
                          ("clock", "/metadata/time"),
                          ("pose_field_order", "/metadata/pose_fields"),
                          ("condition_field_order", "/metadata/condition_fields"),
                          ("planar_motion_field_order", "/metadata/planar_motion_fields"),
                          ("coordinates", "/metadata/coordinates"),
                          ("threshold_and_window_parameters", "/metadata/parameters"),
                          ("contact_status", "/metadata/contact_status"),
                          ("episode_coverage", "/metadata/episode_coverage")):
            add(fields, manifest, name, ref)
        mechanism = metadata["parameters"]["mechanism"]
        if mechanism == "world_planar_velocity_increment":
            runtime_excerpt = excerpt("src/true_rl_agent/tasks/push_recovery/runtime.py",
                                      'failure = (height < p["fall_height_m"])', 'return failure')
        elif mechanism == "world_planar_external_force":
            runtime_excerpt = excerpt("src/true_rl_agent/tasks/push_recovery/force_runtime.py",
                                      "def observe(self) -> Any:", 'return failure')
        else:
            raise ValueError(f"unsupported Push mechanism for source contract: {mechanism}")
        contract["source_contracts"] = [
            excerpt("src/true_rl_agent/tasks/push_recovery/numeric_capture.py",
                    'POSE_FIELDS = (', 'CONDITIONS = ('),
            excerpt("src/true_rl_agent/tasks/push_recovery/numeric_capture.py",
                    '"capture_point": "after_physics_after_task_observe_before_termination_and_reset"',
                    '"media_status": "not_collected_by_numeric_profile", "contact_status": "not_collected"'),
            runtime_excerpt,
            excerpt("src/true_rl_agent/tasks/push_recovery/candidate_surface.py",
                    'PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY = CandidateSurfacePolicy(',
                    '"reward_callable", "control_logic", "recovery_logic"'),
            excerpt("src/true_rl_agent/tasks/push_recovery/runtime_adapter.py",
                    '"candidate_receives": list(push_recovery_reward_input_fields())',
                    '    return result')]
        contract["source_contracts"].append(excerpt(
            "src/true_rl_agent/tasks/push_recovery/reward_inputs.py",
            'class PushRecoveryV2RewardInputs(', '    terminated_non_timeout: Any'))
        contract["source_contracts"].append(excerpt(
            "src/true_rl_agent/tasks/push_recovery/robot_binding.py",
            'binding = RobotTaskBinding(', 'role_bindings={"base": profile.semantic_roles["base"].targets}'))
        if mechanism == "world_planar_velocity_increment":
            contract["source_contracts"].append(excerpt(
                "src/true_rl_agent/tasks/push_recovery/failure_classifier.py",
                "def classify_recovery_episode(", '"invalid_injection_count": invalid}'))
        elif mechanism == "world_planar_external_force":
            contract["source_contracts"].append(excerpt(
                "src/true_rl_agent/tasks/push_recovery/failure_classifier.py",
                "def classify_force_recovery_episode(", '"verified_push_event_count": sum(event["injection_verified"] for event in events)'))
        contract["author_permission_definition"] = {
            "source_contract_indices": [3, 4],
            "meaning": "Verbatim task-owned candidate component classification and runtime input boundary; no permission inferred from report text"}
        contract["binding_checks"] = {"source_manifest_matches_numeric_sha": True,
            "numeric_schema_matches_identity": True, "resolved_protocol_hash_matches_parameters": True,
            "capture_field_orders_match_source": True}
        contract["limitations"] = [
            "Push terminal labels and recovery are recorded by runtime_episode_records; the source excerpts define capture and failure boundary but do not prove a broader physical mechanism.",
            "CandidateSurfacePolicy classifies proposed changes; it does not by itself prove every author prompt or public delivery permission.",
            "Current source-file SHA records review-time source bytes. The old numeric archive does not by itself prove those exact bytes executed; use the frozen execution source manifest for that stronger claim."]
    contract["reward_input_contract"] = reward_input_contract(task, manifest, resolved_cell_path)
    return contract


def verify(contract: dict, numeric_path: Path, source_path: Path, resolved_cell_path: Path | None = None) -> None:
    if contract.get("schema_version") != SCHEMA:
        raise ValueError("wrong contract schema")
    if digest(ROOT / contract["generator"]["file"]) != contract["generator"]["sha256"]:
        raise ValueError("generator drift")
    numeric_ref, source_ref = contract["numeric_manifest"], contract["source_manifest"]
    numeric_path, source_path = numeric_path.resolve(), source_path.resolve()
    if digest(numeric_path) != numeric_ref["sha256"] or digest(source_path) != source_ref["sha256"]:
        raise ValueError("numeric/source manifest drift")
    manifest, sources = read(numeric_path), read(source_path)
    if contract["reward_input_contract"] != reward_input_contract(contract["task"], manifest, resolved_cell_path):
        raise ValueError("bound reward input contract differs")
    matching = [row for row in sources.get("sources", [])
                if row.get("manifest_sha256") == numeric_ref["sha256"]
                and Path(row.get("manifest_path", "")).resolve() == numeric_path.resolve()]
    if len(matching) != source_ref["matching_manifest_entries"] or not matching:
        raise ValueError("source manifest no longer binds numeric manifest")
    for field in contract["manifest_fields"]:
        if pointer(manifest, field["numeric_manifest_pointer"]) != field["value"]:
            raise ValueError(f"manifest field drift: {field['name']}")
    for source in contract["source_contracts"]:
        path = ROOT / source["file"]
        if digest(path) != source["sha256"]:
            raise ValueError(f"source file drift: {source['file']}")
        lines = path.read_text(encoding="utf-8").splitlines()
        actual = "\n".join(lines[source["first_line"] - 1:source["last_line"]])
        if actual != source["verbatim"]:
            raise ValueError(f"source excerpt drift: {source['file']}")
    if contract["task"] == "stair":
        ref = contract["task_protocol"]
        path = ROOT / ref["file"]
        if digest(path) != ref["sha256"]:
            raise ValueError("Stair task protocol file drift")
        protocol = read(path)
        if (protocol["task_protocol_hash"] != ref["task_protocol_hash"]
                or protocol["task_protocol_hash"] != manifest["metadata"]["semantics"]["task_protocol_hash"]):
            raise ValueError("Stair task protocol hash binding drift")
        for item in ref["definitions"]:
            actual = pointer(protocol, item["source_pointer"])
            if actual != item["value"] or actual != pointer(manifest, item["numeric_manifest_pointer"]):
                raise ValueError(f"Stair task protocol binding drift: {item['name']}")
        if pointer(protocol, ref["author_permission_definition"]["source_pointer"]) != ref["author_permission_definition"]["value"]:
            raise ValueError("Stair author authority drift")
    else:
        metadata = manifest["metadata"]
        if manifest["identity"].get("resolved_protocol_hash") != metadata["parameters"]["resolved_protocol_hash"]:
            raise ValueError("Push resolved protocol binding drift")
        if manifest["identity"].get("task_numeric_schema") != metadata["schema_version"]:
            raise ValueError("Push numeric schema binding drift")
        constants = push_field_constants()
        if (constants["POSE_FIELDS"] != metadata["pose_fields"]
                or constants["PLANAR_MOTION_FIELDS"] != metadata["planar_motion_fields"]
                or constants["CONDITIONS"] != metadata["condition_fields"]):
            raise ValueError("Push capture field order binding drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stair", "push"))
    parser.add_argument("--numeric-manifest", type=Path)
    parser.add_argument("--source-manifest", type=Path,
                        help="raw export _private/sources.json or equivalent manifest with numeric path and SHA")
    parser.add_argument("--resolved-cell-manifest", type=Path,
                        help="Push resolved-cell manifest bound by the numeric archive identity; Stair carries this inline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="verify an existing generated contract instead of creating one")
    args = parser.parse_args()
    if args.verify:
        if not all((args.numeric_manifest, args.source_manifest)):
            parser.error("verification requires --numeric-manifest and --source-manifest")
        verify(read(args.verify), args.numeric_manifest, args.source_manifest, args.resolved_cell_manifest)
        print(json.dumps({"status": "PASS", "contract_sha256": digest(args.verify)}))
        return
    if not all((args.task, args.numeric_manifest, args.source_manifest, args.output)):
        parser.error("creation requires --task, --numeric-manifest, --source-manifest, and --output")
    contract = build(args.task, args.numeric_manifest, args.source_manifest, args.resolved_cell_manifest)
    verify(contract, args.numeric_manifest, args.source_manifest, args.resolved_cell_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(contract, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    print(json.dumps({"status": "PASS", "task": args.task, "bytes": args.output.stat().st_size,
                      "sha256": digest(args.output), "file": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
