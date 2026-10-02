"""One offline verification of six existing Stair focuses; no model execution."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
P = HERE.parents[2]
sys.path.insert(0, str(P))
import confirmation_review_coverage as coverage
import prepare_confirmation_materials as packing

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode() + b"\n")

expected = {
    "stair_c1_conditional_frontier.json": "129fc7f6949c10029c8389801195e25d6bfc9af2a6e25aa61da3ba753f0d7669",
    "stair_c2_mechanism_ood.json": "861d0a4e11b26b5ec11cd5f7128066cf013d6c547759a5ff95374c8164764986",
    "stair_height_ood.json": "36f28adf225b8cc9324330f58e2710ed4076e7ff41be77d7f9b3d909c7c43677",
    "stair_joint_ood.json": "e7378b7605c6a9984c0af90e68fa3dbac4f2ea53f28d000b34894cb4b9cadb48",
    "stair_phase_composition_ood.json": "339bf45a281449b67e02cb73651e73309d039c1feff5241bc4cebd2e91a6690e",
    "stair_tread_ood.json": "e50e367e513fa2e120f88bef79b871ae4a46fd43bbc5f5393828c856901870c3",
}
originals = {}
for name, digest in expected.items():
    raw = (HERE / "original_focus" / name).read_bytes()
    assert sha(raw) == digest
    value = json.loads(raw)
    originals[value["focus"]] = (name, value, digest)

rows = []
archive_path = HERE.parent / "facts_offline_preparation/sol_original_prepared.zip"
with zipfile.ZipFile(archive_path) as archive:
    for number in range(7, 13):
        packet = f"focus-{number:02}"
        request_bytes = archive.read(f"prepared/{packet}/gpt6.1-sol-a/request.http_body.json")
        request = json.loads(request_bytes)
        materials, _ = coverage.request_materials(request)
        previous = materials["related-raw"]["data"]
        name, source, source_sha = originals[previous["focus"]]
        rebuilt = packing.related_focus(source)
        navigation = rebuilt.pop("arithmetic_navigation")
        # The existing postcapture packet adapter preserves raw related_v1 and
        # changes only the packet's declared schema/profile plus source links.
        for key in ("schema_version", "profile", "source_materials"):
            if key in previous:
                rebuilt[key] = previous[key]
        assert rebuilt == previous, [key for key in set(rebuilt) | set(previous) if rebuilt.get(key) != previous.get(key)]
        rebuilt["arithmetic_navigation"] = navigation
        assert len(navigation["predicate_components"]) == 37
        # Independent recomputation from original inclusive intervals, without
        # interval_stats, the new helper, or the serialized summary matrix.
        for group in navigation["groups"]:
            scope, run, cell, success, outcome, denominator, counts = group
            members = [r for r in source["all_scope_rows"]
                if (scope == "all_scope" or (r["original_success"] == success and r["original_outcome"] == outcome))
                and (scope != "run_cell_by_original_outcome" or (r["run_group"] == run and r["cell"] == cell))]
            assert len(members) == denominator
            for component, count in zip(navigation["predicate_components"], counts):
                predicate, component_index, _, _, dwell, _, _ = component
                supplied = before = at_T = never = 0
                for member in members:
                    values = member["direct_measurements"]["threshold_intervals_by_component"].get(predicate)
                    if values is None or component_index >= len(values) or values[component_index] is None:
                        continue
                    supplied += 1
                    ends = [a+dwell-1 for a, b in values[component_index] if b-a+1 >= dwell]
                    confirmation = min(ends, default=None)
                    before += confirmation is not None and confirmation < member["steps"]
                    at_T += confirmation == member["steps"]
                    never += confirmation is None
                assert count == [supplied, denominator-supplied, before, at_T, never]
        save(HERE / "rebuilt" / f"{packet}.json", rebuilt)
        # Measure complete existing text carrier with only this material and
        # its catalog changed. Protocol/questions/system wording stay fixed;
        # system/catalog hash text is updated, with no executable prepared gate.
        revised_materials = dict(materials)
        related = dict(materials["related-raw"])
        related["data"] = rebuilt
        related["sha256"] = sha((json.dumps(rebuilt, ensure_ascii=False, allow_nan=False, indent=2)+"\n").encode())
        revised_materials["related-raw"] = related
        catalog = coverage.build_reference_catalog(revised_materials)
        future = json.loads(request_bytes)
        content = []
        for block in future["messages"][1]["content"]:
            item = json.loads(block["text"])
            if item.get("role") == "reference_material_header" and item["envelope"]["material_id"] == "related-raw":
                content.extend(coverage.render_material_blocks(related, catalog))
            elif item.get("role") == "reference_material_block" and item["locator"]["material_id"] == "related-raw":
                continue
            elif item.get("role") == "reference_catalog":
                item["data"] = catalog
                item["sha256"] = sha(coverage.canonical(catalog))
                content.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})
            else:
                content.append(block)
        future["messages"][1]["content"] = content
        previous_catalog = coverage.request_reference_catalog(request)
        future["messages"][0]["content"] = future["messages"][0]["content"].replace(
            sha(coverage.canonical(previous_catalog)), sha(coverage.canonical(catalog)))
        reconstructed, _ = coverage.request_materials(future)
        assert reconstructed["related-raw"]["data"] == rebuilt
        assert coverage.request_reference_catalog(future) == catalog
        save(HERE / "capacity_inputs" / f"{packet}.old.json", request)
        save(HERE / "capacity_inputs" / f"{packet}.new.json", future)
        rows.append({"packet_id": packet, "source_file": name, "source_sha256": source_sha,
            "old_request_sha256": sha(request_bytes), "members": len(source["all_scope_rows"]),
            "components": 37, "groups": len(navigation["groups"]),
            "all_original_values_preserved": True, "independent_interval_recount": True,
            "navigation_compact_utf8_bytes": len(coverage.canonical(navigation)),
            "related_compact_utf8_bytes_before": len(coverage.canonical(previous)),
            "related_compact_utf8_bytes_after": len(coverage.canonical(rebuilt)),
            "rebuilt_sha256": sha((HERE / "rebuilt" / f"{packet}.json").read_bytes())})

save(HERE / "rebuild_verification.json", {"status": "PASS_OFFLINE_DEVELOPMENT_ONLY", "rows": rows,
    "current_packer_sha256": sha(Path(packing.__file__).read_bytes()), "verifier_sha256": sha(Path(__file__).read_bytes()),
    "source_remote_root": "/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r4/numeric_raw/focus",
    "transport_deltas": "same full original request messages; related material/navigation + its catalog/source hashes only. Measurement fixtures are not executable prepared requests.",
    "no_api_or_capture": True, "not_formal_acceptance": True})
print(json.dumps({"status": "PASS", "packets": len(rows), "navigation_bytes": [r["navigation_compact_utf8_bytes"] for r in rows]}))
