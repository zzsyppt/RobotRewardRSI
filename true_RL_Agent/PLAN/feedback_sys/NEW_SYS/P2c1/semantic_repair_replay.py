"""Bounded public-source replay of the parent semantic counterexamples."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from tests.unit.cycle_oracle import check_records
from tests.unit.test_stair_trajectory_report import StairTrajectoryReportTests as Fixture
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.trajectory_translation.cycles import CYCLE_CONFIG
from true_rl_agent.tasks.stair_climbing.trajectory_cycles import preview_with_cycles


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def rejected(name,operation):
    try:
        operation()
    except (ValueError, AssertionError) as exc:
        return {"case":name,"accepted":False,"error_type":type(exc).__name__,"reason":str(exc)}
    raise AssertionError("counterexample accepted: "+name)


def replay(output):
    lock = read(Path(__file__).parents[1]/"P2a/source_lock.json")
    spec = importlib.util.spec_from_file_location("independent_cycle_raw",Path(__file__).with_name("audit_raw.py"))
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    result = {"status":"PASS","sources":[],"counterexamples":[],"metadata_records_checked":0,
              "raw_episode_checks":[],"external_model_calls":0,"gpu_runs":0,"training_runs":0}
    for source in lock["sources"]:
        sid = source["source_id"]
        directory = ROOT/"outputs/p2c1/real_01"/sid
        manifest, report = read(directory/"manifest.json"), read(directory/"report.json")
        assert manifest["source_manifest_ref"]["manifest_sha256"] == source["manifest_sha256"]
        for name, info in manifest["artifacts"].items():
            assert sha(directory/name) == info["sha256"] and (directory/name).stat().st_size == info["bytes"]
        with (directory/"cycles.jsonl").open(encoding="utf-8") as stream:
            episodes = [json.loads(line) for line in stream]
        # Independent literals, not a call to production schema validation.
        for episode in episodes:
            for record in episode["records"]:
                for key, value in {"unit":"s; phase turns","frame":"recorder_readout","record_type":"measurement",
                    "catalog_version":"trajectory.p0.measurements.v3","base_catalog_version":"trajectory.p0.measurements.v2",
                    "extractor_version":"trajectory.contact_cycles.v1"}.items():
                    assert record[key] == value, (sid,episode["episode_id"],key)
                result["metadata_records_checked"] += 1
        base = read(ROOT/"outputs/p2a"/sid/"public.json")
        preview, receipt = preview_with_cycles(base,report,episodes)
        encoded = json.dumps(preview,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        assert encoded == (directory/"preview.json").read_bytes()
        assert receipt == read(directory/"preview_receipt.json")
        (output/(sid+"_preview.json")).write_bytes(encoded)
        write(output/(sid+"_preview_receipt.json"),receipt)
        result["sources"].append({"source":sid,"manifest_sha256":source["manifest_sha256"],
            "episodes":len(episodes),"records":sum(len(e["records"]) for e in episodes),
            "artifact_hashes":manifest["artifacts"],"preview_bytes":len(encoded),"preview_byte_identical":True,
            "receipt_unchanged":True,"statements_validated":report["optional_statements"]})
        if sid not in ("k1","k3"):
            continue
        # One full real first episode per source, independent NPZ extraction/formulas.
        _, run, env, steps, forces, dt, refs = next(raw.raw_episodes(source))
        first = episodes[0]
        assert (first["run_index"],first["env_index"]) == (run,env)
        check_records(first["records"],steps,forces,dt)
        example = {"source":sid,"manifest_sha256":source["manifest_sha256"],"run_index":run,"env_index":env,
            "dt_s":dt,"raw_steps":steps.tolist(),"raw_current_force_xyz_n":forces.tolist(),"records":first["records"],
            "complete_statement":report["optional_statements"][0],"independent_numeric_and_semantic_check":"PASS"}
        example_path = output/(sid+"_real_example.json")
        write(example_path,example)
        result["raw_episode_checks"].append({"source":sid,"run_index":run,"env_index":env,"records":len(first["records"]),
            "path":str(example_path.relative_to(ROOT)),"sha256":sha(example_path),"status":"PASS"})
        if sid != "k1":
            continue
        Fixture.setUpClass()
        try:
            foreign_base = Fixture.project(Fixture.full["audit"]["required_report_bytes"])["public_report"]
            result["counterexamples"].append(rejected("foreign_source",
                lambda: preview_with_cycles(foreign_base,report,episodes)))
        finally:
            Fixture.tearDownClass()
        for case in ("foot_swap_same_id","unit_swap_same_id","foot_swap_recomputed_id","coherent_group_key_id_swap"):
            forged = deepcopy(report)
            statement = forged["optional_statements"][0]
            if case == "unit_swap_same_id":
                assert "0.585s" in statement["text"]
                statement["text"] = statement["text"].replace("0.585s","0.585ms")
            else:
                statement["text"] = statement["text"].replace("FL_foot","RR_foot")
            if case in ("foot_swap_recomputed_id","coherent_group_key_id_swap"):
                key = "contact_cycles:robot_0.support.FL_foot"
                group = deepcopy(forged["summary"]["groups"][key])
                group["entity_ids"] = ["robot_0.support.RR_foot"]
                new_key = key.replace("FL_foot","RR_foot")
                statement["id"] = "cycle:"+canonical_json_hash([forged["binding"],CYCLE_CONFIG,new_key,group])
                if case == "coherent_group_key_id_swap":
                    del forged["summary"]["groups"][key]
                    forged["summary"]["groups"][new_key] = group
            result["counterexamples"].append(rejected(case,lambda: preview_with_cycles(base,forged,episodes)))
        for field, value in (("unit","milliseconds; phase degrees"),("frame","physical_contact_time")):
            for resign in (False,True):
                bad = deepcopy(first)
                for record in bad["records"]:
                    record[field] = value
                    if resign:
                        record["record_id"] = "cycle-evidence:"+canonical_json_hash(record)
                name = field+("_resigned" if resign else "_same_id")
                result["counterexamples"].append(rejected("preview_"+name,
                    lambda: preview_with_cycles(base,report,[bad,*episodes[1:]])))
                result["counterexamples"].append(rejected("oracle_"+name,
                    lambda: check_records(bad["records"],steps,forces,dt)))
    # One fresh build exercises the builder's new preview path; numerical records
    # and all user-facing files must be byte-identical to the already audited build.
    fresh = output/"k1_rebuild"
    result["fresh_build"] = {"manifest_sha256":sha(fresh/"manifest.json"),"unchanged_artifacts":{}}
    for name in ("cycles.jsonl","report.json","preview.json","preview_receipt.json"):
        assert sha(fresh/name) == sha(ROOT/"outputs/p2c1/real_01/k1"/name)
        result["fresh_build"]["unchanged_artifacts"][name] = sha(fresh/name)
    write(output/"semantic_replay.json",result)
    print(json.dumps({"status":result["status"],"source_previews":len(result["sources"]),
        "metadata_records_checked":result["metadata_records_checked"],"raw_episodes":len(result["raw_episode_checks"]),
        "rejected_counterexamples":len(result["counterexamples"]),"fresh_build_artifacts_unchanged":4},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    replay(parser.parse_args().output.resolve())
