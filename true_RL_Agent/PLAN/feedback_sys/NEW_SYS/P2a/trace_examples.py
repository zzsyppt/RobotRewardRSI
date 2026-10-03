"""Small reviewable raw-array -> independent fact -> actual public sentence trace."""
import json
from pathlib import Path
import audit_replay as audit


def main():
    targets = {"k1": (2, 66, [228], "episode_base_runs"),
               "k2": (2, 198, [271], "episode_terminal"),
               "k3": (3, 64, [154, 508], "inverted_before")}
    result = []
    for source in audit.read(audit.LOCK)["sources"]:
        name = source["source_id"]; run_id, env_id, steps, template = targets[name]
        replay = audit.read(audit.OUT / name / "replay.json")
        directory = Path(replay["card_reference"]["manifest_path"]).parent
        card = audit.read(directory / "policy_card.json")
        public = audit.read(audit.OUT / name / "public.json")
        public_ids = {row["id"] for rows in public["report"].values() if isinstance(rows, list) for row in rows}
        needle = f".r{run_id}.e{env_id}"
        claim = next(c for c in card["claims"] if c["template_id"] == template
                     and needle in c["scope"] and c["claim_id"] in public_ids)
        record_ids = {v["record_id"] for v in claim["operands"].values()}
        compact = {}
        manifest = audit.read(directory / "manifest.json")
        for line in (directory / "evidence.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("record_id") in record_ids:
                record_id = r["record_id"]
                if r["record_type"] == "evidence_link":
                    with Path(manifest["source_files"][r["file_id"]]["evidence_path"]).open("rb") as stream:
                        stream.seek(r["byte_offset"])
                        r = json.loads(stream.read(r["byte_length"]))
                compact[record_id] = r
        operands = {key: audit.old.follow(compact[v["record_id"]], v["path"]) for key, v in claim["operands"].items()}
        for m, run, env, dt, origin, outcome, arrays in audit.old.raw_episodes(source):
            if (run, env) != (run_id, env_id): continue
            facts, tracks, up, vxy, _ = audit.raw_facts(m, arrays, dt, origin, outcome)
            anchors = []
            for step in steps:
                chunk = next(c for c in m["chunks"] if c["metadata"]["run_index"] == run
                             and c["metadata"]["step_start"] <= step < c["metadata"]["step_start"] + c["metadata"]["steps"])
                keys = ["base_quat_w", "forbidden_contact_forces_w", "forbidden_contact_forces_w_history",
                        "support_contact_forces_w", "support_lin_vel_w", "done_mask", "wrapper_done_mask", "termination_0", "termination_1"]
                anchors.append({"step": step, "chunk": chunk["file"], "chunk_sha256": chunk["sha256"],
                    "zero_based_row": step - chunk["metadata"]["step_start"], "env_index": env,
                    "raw_values": {k: arrays[k][step - 1].tolist() for k in keys if k in arrays},
                    "missing_channels": [k for k in keys if k not in arrays],
                    "independent_up_z": float(up[step - 1]), "independent_FL_speed_xy_mps": float(vxy[step - 1, 0])})
            fact_keys = (["robot_0.body.base.current_marker", "robot_0.body.base.history_diagnostic"] if name == "k1" else
                         ["task_endpoint", "robot_0.body.base.history_diagnostic"] if name == "k2" else
                         ["body_up", "robot_0.support.FL_foot.low_speed_off"])
            result.append({"source": name, "source_manifest_sha256": source["manifest_sha256"],
                "checkpoint_hash": source["identity"]["checkpoint_hash"], "run": run, "env": env,
                "raw_anchors": anchors, "independent_facts": {k: facts[k] for k in fact_keys},
                "emitted_operands": operands, "claim": claim,
                "scope_note": "Public-selected delivery example, additional to predeclared necessary samples; not an independent random experiment."})
            break
    audit.write(Path(__file__).with_name("raw_to_sentence_examples.json"), result)
    print("three raw-to-sentence traces saved")


if __name__ == "__main__":
    main()
