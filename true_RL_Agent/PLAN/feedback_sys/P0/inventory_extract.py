"""P0 policy/parent/evaluation-source catalog. Stdlib only; no experiments.

Writes inventory_manifest.json. Episode facts remain in their original summaries.
"""
from __future__ import annotations
import collections
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
SOURCES, POLICIES, PAIRS, SESSIONS, CHECKS = {}, {}, [], [], []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    path = Path(path).resolve()
    key = path.relative_to(REPO).as_posix()
    if key not in SOURCES:
        sid = "src_" + hashlib.sha256(key.encode()).hexdigest()[:16]
        SOURCES[key] = dict(source_id=sid, path=key, sha256=sha(path), size_bytes=path.stat().st_size)
    return SOURCES[key]["source_id"]


def read(path):
    source(path)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def get(obj, *keys):
    for key in keys:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def check(name, passed, **details):
    CHECKS.append(dict(name=name, passed=bool(passed), **details))


def policy(cp, task, role):
    if cp not in POLICIES:
        POLICIES[cp] = dict(policy_id=cp, task_id=task, roles=[], candidate_ids=[], checkpoint_locations=[], training_runs=[], evaluations=[])
    row = POLICIES[cp]
    if role not in row["roles"]:
        row["roles"].append(role)
    return row


REMOTE = {}
for name in ("remote_baseline.json", "inventory_push_checkpoint_verification.json", "reference_remote_verification.json", "inventory_parent_remote_verification.json"):
    path = OUT / name
    if path.is_file():
        data = read(path)
        for row in data.get("checkpoints", data.get("reference", [])):
            REMOTE[row["path"]] = dict(exists=row.get("exists"), sha256=row.get("sha256"), matches_recorded=row.get("matches_historical", row.get("matches_summary", row.get("matches_expected"))), collected_at=data.get("collected_at", data.get("checked_at")), source_id=source(path))


def location(row, path, evidence_source):
    if path and not any(x["path"] == path for x in row["checkpoint_locations"]):
        row["checkpoint_locations"].append(dict(path=path, recorded_sha256=row["policy_id"], record_source_id=evidence_source, local_bytes=False, remote_check=REMOTE.get(path)))


def train(row, path, binding=None, seed=None):
    if not path.is_file():
        return None
    obj = read(path)
    b, s = binding or {}, seed or {}
    t, cp = obj.get("training", {}), obj.get("checkpoint", {})
    if not s and get(t, "effective_agent_execution", "seed") is not None:
        s = dict(seed=t["effective_agent_execution"]["seed"], source_ids=[source(path)])
    item = dict(source_id=source(path), job_id=obj.get("job_id"), attempt_id=get(obj, "run", "train_attempt_id"), status=obj.get("status"), environment_seed=s.get("seed"), seed_source_ids=s.get("source_ids", []), seed_policy=get(obj, "run", "seed_policy"), initialization_mode=b.get("initialization_mode"), initialization_checkpoint_hash=b.get("initialization_checkpoint_hash"), initial_policy_state_hash=t.get("initial_policy_state_hash"), seed_schedule_hash=obj.get("initialization_seed_schedule_hash"), training_binding_hash=obj.get("training_binding_hash"), resolved_cell_manifest_hash=obj.get("resolved_cell_manifest_hash"), resolved_cell_identity=obj.get("resolved_cell_identity"), budget_steps=t.get("budget_steps"), final_iteration=t.get("latest_checkpoint_iteration"), program_source_checksum=obj.get("program_source_checksum"), checkpoint_hash=cp.get("checksum"))
    if item not in row["training_runs"]:
        row["training_runs"].append(item)
    location(row, cp.get("uri"), source(path))
    return item


def evaluate(row, path, label):
    if not path.is_file():
        return None
    obj = read(path)
    result = obj.get("phase_result", obj)
    evidence = result.get("task_evidence", {})
    summary = evidence.get("summary", {})
    ordered = summary.get("ordered_summary", {})
    prefix = "/phase_result" if "phase_result" in obj else ""
    phase_claims = {prefix + key: value for key, value in (("/evaluation_phase", result.get("evaluation_phase")), ("/phase", result.get("phase")), ("/task_evidence/phase", evidence.get("phase")), ("/task_evidence/summary/split", summary.get("split")), ("/task_evidence/summary/ordered_summary/split", ordered.get("split"))) if value is not None}
    for index, seed in enumerate(evidence.get("seed_summaries", [])):
        split = get(seed, "run", "split")
        if split is not None:
            phase_claims[prefix + f"/task_evidence/seed_summaries/{index}/run/split"] = split
    public_dev = bool(phase_claims) and set(phase_claims.values()) == {"public_dev"}
    check("evaluation_declares_public_dev", public_dev, source_id=source(path))
    if not public_dev:
        raise ValueError(f"Expected actual public_dev split evidence: {path}: {phase_claims}")
    facts = [fact for seed in evidence.get("seed_summaries", []) for fact in seed.get("per_episode", [])]
    if facts:
        counts = dict(collections.Counter(f.get("failure_label") for f in facts))
        successes, total = sum(f.get("success") is True for f in facts), len(facts)
        pointer = prefix + "/task_evidence/seed_summaries/*/per_episode"
    else:
        counts, total = ordered.get("primary_outcome_counts", {}), ordered.get("total_episode_count")
        successes = counts.get("success")
        saved = any("episode_facts_by_cell" in s for s in ordered.get("seed_summaries", []))
        pointer = prefix + "/task_evidence/summary/ordered_summary/seed_summaries/*/episode_facts_by_cell" if saved else None
    binding = result.get("evaluation_binding", {})
    cp = result.get("checkpoint_hash") or get(result, "checkpoint_safety_report", "checkpoint_hash")
    item = dict(source_id=source(path), label=label, checkpoint_hash=cp, phase="public_dev", evaluation_binding_hash=result.get("evaluation_binding_hash") or binding.get("evaluation_binding_hash"), eval_protocol_hash=result.get("eval_protocol_hash") or binding.get("eval_protocol_hash"), task_protocol_hash=binding.get("task_protocol_hash"), seed_cohort_hash=binding.get("seed_cohort_hash"), resolved_cell_manifest_hash=result.get("resolved_cell_manifest_hash"), episodes=total, successes=successes, success_rate=successes/total if successes is not None and total else None, j_primary=summary.get("j_primary"), primary_counts=counts, facts_pointer=pointer, cost=result.get("evaluation_execution_cost"))
    item["phase"] = next(iter(phase_claims.values()))
    item["phase_evidence"] = phase_claims
    item["facts_pointer"] = pointer.split("/*")[0] if pointer else None
    item["episode_collection_field"] = "per_episode" if facts else "episode_facts_by_cell" if pointer else None
    identity = evidence.get("execution_identity", {})
    preflight = identity.get("environment_source_preflight") or result.get("environment_source_preflight", {})
    backend = identity.get("backend_execution_environment_verification") or result.get("backend_execution_environment_verification", {})
    item["execution_source_manifest_hash"] = preflight.get("source_manifest_hash")
    item["runtime_dynamics_hash"] = preflight.get("runtime_dynamics_hash")
    item["execution_source_evidence_pointer"] = prefix + ("/task_evidence/execution_identity" if identity.get("environment_source_preflight") else "") + "/environment_source_preflight"
    item["execution_environment_manifest_hash"] = backend.get("actual_manifest_hash")
    item["execution_environment_evidence_pointer"] = prefix + ("/task_evidence/execution_identity" if identity.get("backend_execution_environment_verification") else "") + "/backend_execution_environment_verification"
    for key in ("environment_source_preflight", "backend_execution_environment_verification"):
        if identity.get(key) and result.get(key):
            check("execution_identity_copies_agree", identity[key] == result[key], source_id=source(path), field=key)
    check("execution_build_identity_recorded", (item["execution_source_manifest_hash"] or item["runtime_dynamics_hash"]) and item["execution_environment_manifest_hash"], source_id=source(path))
    row["evaluations"].append(item)
    check("evaluation_checkpoint_matches_policy", cp == row["policy_id"], source_id=source(path))
    if total is not None:
        check("primary_counts_close", sum(counts.values()) == total, source_id=source(path))
    return item


def train_seeds(root):
    result = {}
    for path in sorted(root.glob("job_attempts/*/*.reservation.json")):
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if obj.get("job_kind") != "train":
            continue
        log = path.with_name(path.name.replace(".reservation.json", ".logs")) / "stdout.log"
        if log.is_file():
            match = re.search(r"Environment seed\s*:\s*(\d+)", log.read_text(encoding="utf-8", errors="replace"))
            if match:
                result[obj["logical_job_id"]] = dict(seed=int(match.group(1)), source_ids=[source(path), source(log)])
    return result


def session(root, label, rationale_root, expected):
    genesis = read(root / "session_genesis.json")
    ledger_path = root / "ledger.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ledger_id = source(ledger_path)
    sid, task = genesis["session_id"], genesis["task_id"]
    anchor, binding = genesis["anchor"], genesis["anchor"]["training_binding"]
    reference = policy(anchor["checkpoint_hash"], task, label + ":reference")
    location(reference, get(anchor, "checkpoint_artifact", "uri"), source(root / "session_genesis.json"))
    seeds, attempts = train_seeds(root), {}
    for path in sorted(root.glob("authoring/*/*/authoring_attempts/*/attempt_result.json")):
        obj = read(path)
        prompt = path.with_name("prompt_payload.json")
        attempts.setdefault(obj["round_id"], []).append(dict(attempt_result_source_id=source(path), prompt_source_id=source(prompt), prompt_sha256="sha256:" + sha(prompt), candidate_id=get(obj, "provider_attempt", "candidate_id"), sequence=obj.get("sequence")))
    components = list(sorted(rationale_root.glob("*/*/*/training_code/components.py")))
    component_sources = [source(path) for path in components]
    session_pairs, untrained = [], []
    for index, rec in enumerate(records):
        for lane_index, trial in enumerate(rec["trials"]):
            cp, trial_id = (trial.get("train") or {}).get("checkpoint_hash"), trial["trial_id"]
            if not cp:
                path = root / "runner" / trial_id / "train/summary.json"
                failed = read(path) if path.is_file() else {}
                untrained.append(dict(trial_id=trial_id, status=trial["trial_status"], train_summary_source_id=source(path) if failed else None, train_attempt_id=get(failed, "run", "train_attempt_id"), train_status=failed.get("status")))
                continue
            row = policy(cp, task, label + ":candidate")
            candidate = trial.get("candidate") or {}
            cid = candidate.get("candidate_id")
            if cid not in row["candidate_ids"]:
                row["candidate_ids"].append(cid)
            run = train(row, root / "runner" / trial_id / "train/summary.json", binding, seeds.get(trial_id + "-train"))
            evaluate(row, root / "runner" / trial_id / "eval/public_dev/summary.json", label)
            lane_attempts = attempts.get(trial_id, [])
            selected = [a for a in lane_attempts if a["prompt_sha256"] == get(trial, "authoring", "prompt_payload_checksum")]
            paths = [p for p in components if rec["generation_id"] in p.parts and trial["lane_id"] in p.parts and p.parent.parent.name.endswith("_" + str(cid))]
            rationale_ids = [source(p) for p in paths]
            for path in paths:
                md = path.parent.parent / "rationale.md"
                if md.is_file():
                    rationale_ids.append(source(md))
            pair = dict(session_id=sid, generation=rec["generation_index"], lane=trial["lane_id"], trial_id=trial_id, parent_policy_id=rec["parent_current_best"]["checkpoint_hash"], candidate_policy_id=cp, candidate_id=cid, trial_status=trial["trial_status"], parent_program_source_checksum=rec["parent_current_best"].get("program_source_checksum"), candidate_program_source_checksum=candidate.get("program_source_checksum"), train_source_id=run["source_id"] if run else None, selected_prompt_source_ids=[a["prompt_source_id"] for a in selected], authoring_attempts=lane_attempts, rationale_source_ids=rationale_ids, ledger_source_id=ledger_id, ledger_pointer=f"/{index}/trials/{lane_index}", relationship="authoring_parent; training_init_is_recorded_separately")
            PAIRS.append(pair)
            session_pairs.append(pair)
            check("ledger_train_checkpoint_matches_summary", run and run["checkpoint_hash"] == cp and run["status"] == "pass", trial_id=trial_id)
            check("selected_authoring_prompt_resolved", len(selected) == 1, trial_id=trial_id)
    row = dict(session_id=sid, label=label, root=root.relative_to(REPO).as_posix(), genesis_source_id=source(root / "session_genesis.json"), ledger_source_id=ledger_id, reference_policy_id=reference["policy_id"], training_binding=binding, initial_policy_state_hash=anchor.get("initial_policy_state_hash"), generation_count=len(records), trained_candidate_count=len(session_pairs), unique_candidate_policy_count=len({p["candidate_policy_id"] for p in session_pairs}), accepted_trial_ids=[p["trial_id"] for p in session_pairs if p["trial_status"] == "accepted"], authoring_attempt_count=sum(len(v) for v in attempts.values()), rationale_components_source_ids=component_sources, failed_or_untrained_trials=untrained, historical_gate_status=get(anchor, "qualification_gate", "status"))
    SESSIONS.append(row)
    observed = [row["trained_candidate_count"], row["unique_candidate_policy_count"], len(row["accepted_trial_ids"])]
    check("session_counts", observed == expected, session_id=sid, observed=observed, expected=expected)
    return row, reference


def main():
    formal_root = OUT / "inventory_remote_raw/outputs/formal_go2_stair_15round_20260901_c6ab8ba"
    push_root = REPO / "PLAN/2new_world/Go2_push/attempt_08/session"
    formal, formal_ref = session(formal_root, "formal_c6_stair_v2", OUT / "author_evidence_raw/formal_c6/provider_candidates", [23, 22, 1])
    parent_root = OUT / "inventory_remote_raw/outputs/p5_go2_derived_reference_ef6826e"
    parent_manifest_path = parent_root / "resolved_cell_manifest.json"
    parent_manifest = read(parent_manifest_path)
    parent_train = train(formal_ref, parent_root / "runner/p5-go2-derived-reference-ef6826e/train/summary.json", formal["training_binding"])
    formal_ref["resolved_cell_manifest_source_id"] = source(parent_manifest_path)
    check("formal_parent_training_checkpoint_bound", parent_train["checkpoint_hash"] == formal_ref["policy_id"])
    check("formal_parent_training_manifest_and_binding_bound", parent_train["status"] == "pass"
          and parent_train["training_binding_hash"] == formal["training_binding"]["binding_hash"]
          and parent_train["resolved_cell_manifest_hash"] == parent_manifest["manifest_hash"]
          and parent_train["initial_policy_state_hash"] == formal["initial_policy_state_hash"])
    push, push_ref = session(push_root, "go2_push_attempt_08_force_v3", push_root / "provider_candidates", [10, 10, 2])
    evaluate(push_ref, push_root / f"runner/{push['session_id']}/eval/public_dev/summary.json", "go2_push_attempt_08_force_v3_reference")
    gate_path = push_root / "reference_public_dev_gate.json"
    gate = read(gate_path)
    push["public_dev_reference_gate"] = dict(source_id=source(gate_path), status=get(gate, "task_gate", "status"), score=get(gate, "task_gate", "public_dev_score"))
    train(push_ref, REPO / "PLAN/2new_world/P6/go2_dev_D1/dev/composition/runner/p6-go2-matched2-dev-20260912/train/summary.json", dict(initialization_mode="random", initialization_checkpoint_hash=None))
    for robot in ("go2_12dof", "g1_37dof"):
        base = REPO / "PLAN/2new_world/P3/results/cohort_alignment_20260909" / f"{robot}_qualification"
        path = base / "evaluation/public_dev/summary.json"
        obj = read(path)
        cp = get(obj, "checkpoint_safety_report", "checkpoint_hash")
        row = policy(cp, obj["task_id"], f"stair_v3_{robot}_reference")
        evaluate(row, path, f"stair_v3_{robot}_reference")
        q = read(base / "result.json")
        row["qualification"] = dict(source_id=source(base / "result.json"), status=q.get("status"), reference_gate=q.get("reference_gate"))
        remote_train = q.get("training_summary")
        local_train = REPO / remote_train.split("true_RL_Agent/")[-1]
        row["training_source_recorded"] = dict(remote=remote_train, local=local_train.relative_to(REPO).as_posix(), local_exists=local_train.is_file())
        train(row, local_train)
        location(row, get(q, "checkpoint", "uri") or get(q, "checkpoint", "path"), source(base / "result.json"))
    g1 = REPO / "PLAN/2new_world/P6/g1_qualified_01/formal"
    g = read(g1 / "session_genesis.json")
    row = policy(g["anchor"]["checkpoint_hash"], g["task_id"], "p6_g1_push_reference")
    train(row, g1 / f"runner/{g['session_id']}/train/summary.json", g["anchor"]["training_binding"])
    evaluate(row, g1 / f"runner/{g['session_id']}/eval/public_dev/summary.json", "p6_g1_push_reference")
    replay = OUT.parent / "audit4_replay/gen008a_v3/public_dev_summary.json"
    if replay.is_file():
        obj = read(replay)
        cp = get(obj, "checkpoint_safety_report", "checkpoint_hash") or obj.get("checkpoint_hash")
        if cp in POLICIES:
            evaluate(POLICIES[cp], replay, "audit4_formal_gen008a_stair_v3_replay")
    cost_path = formal_root / "generation_actual_cost/0018-final-confirmation-settlement.json"
    cost = read(cost_path)
    formal["historical_cost"] = dict(source_id=source(cost_path), gpu_capacity_hours=get(cost, "actual", "allocated_gpu_capacity_hours_charged"), train_jobs=get(cost, "actual", "train_jobs"), job_attempts=get(cost, "actual", "job_attempts"), eval_seed_units=get(cost, "actual", "eval_seed_units"), provider_attempts=get(cost, "actual", "provider_attempts"), reference_training_imported=True)
    check("parent_and_candidate_policy_ids_resolve", all(p["parent_policy_id"] in POLICIES and p["candidate_policy_id"] in POLICIES for p in PAIRS), pair_count=len(PAIRS))
    check("rationale_source_resolved_for_trained_pairs", all(p["rationale_source_ids"] for p in PAIRS))
    check("referenced_sources_exist", all((REPO / s["path"]).is_file() for s in SOURCES.values()), source_count=len(SOURCES))
    grouped = collections.defaultdict(list)
    for item in CHECKS:
        grouped[item["name"]].append(item)
    checks = [dict(name=name, passed=all(x["passed"] for x in rows), checked=len(rows), failures=[x for x in rows if not x["passed"]]) for name, rows in grouped.items()]
    result = dict(schema_version="feedback_p0_policy_catalog.v2", generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), extractor_sha256=sha(Path(__file__)), scope="saved policies, authoring-parent relations and public-dev evaluation sources; no episode duplication", sources=sorted(SOURCES.values(), key=lambda s: s["source_id"]), policies=sorted(POLICIES.values(), key=lambda p: p["policy_id"]), parent_candidate_pairs=PAIRS, sessions=SESSIONS, checks=checks, boundaries="Parent edges describe authoring history. Train initialization and one search session are recorded separately. Public-dev facts stay in original summaries; no independent-session effect is inferred.")
    (OUT / "inventory_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = [x for x in CHECKS if not x["passed"]]
    print(json.dumps(dict(policies=len(POLICIES), parent_candidate_pairs=len(PAIRS), sources=len(SOURCES), check_groups=len(checks), failed_checks=failures), ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
