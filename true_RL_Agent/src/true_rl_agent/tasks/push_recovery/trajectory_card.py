"""Task-owned, offline Push cards with bounded, deterministic public projection."""
from __future__ import annotations
from true_rl_agent.evaluation.process_report import format_number, select_whole_claims

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from true_rl_agent.evaluation.trajectory_translation.cohort import distribution
from true_rl_agent.evaluation.trajectory_translation.contracts import NumericEpisode, stable_id
from .trajectory_adapter import ADAPTER_VERSION, iter_push_episodes
from .trajectory_measurement import CATALOG_VERSION, CONFIG, DISTURBANCES, ROOT_METRICS, measure_push_episode

CARD_VERSION = "push.root_behavior.card.v1"
PUBLIC_VERSION = "push.root_behavior.public.v1"
TEMPLATE_VERSION = "push.root_zh.v1"
MAX_PUBLIC_BYTES = 12288
WITNESS_ORDER = ("recovered", "full_window_without_dwell", "terminal_before_window_end", "final_unstable_after_recovery")
STATUSES_ZH = {"recovered": "已锁存恢复", "full_window_without_dwell": "完整窗未达dwell",
    "terminal_before_window_end": "任务提前终止截尾", "rollout_truncated": "采样截尾",
    "anchor_not_observed": "恢复锚点未观测", "invalid_injection": "扰动验证无效", "unknown_samples": "窗口有未知记录"}


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _tri(values: list) -> dict:
    known = [v for v in values if v is not None]
    return {"true": sum(v is True for v in known), "valid": len(known), "unknown": len(values) - len(known), "total": len(values)}


def aggregate_push(records: list[dict]) -> dict:
    """One explicit policy/protocol group; all episodes contribute before facets."""
    if not records:
        raise ValueError("Push card requires episodes")
    values = [r["value"] for r in records]
    first = values[0]
    binding = first["source"]["group_identity"]
    if (any(v["source"]["group_identity"] != binding or v["parameters"] != first["parameters"]
            or v["timing"] != first["timing"] for v in values)
            or any(r["catalog_version"] != CATALOG_VERSION or r["measurement_config"] != CONFIG for r in records)):
        raise ValueError("Push cohort policy/protocol/capability/config mismatch; no cross-condition pooling")
    if len({v["episode_id"] for v in values}) != len(values):
        raise ValueError("duplicate Push episodes")
    runs = defaultdict(set)
    for v in values:
        runs[v["run_index"]].add(v["source"]["manifest_sha256"])
    if any(len(hashes) != 1 for hashes in runs.values()):
        raise ValueError("multiple archives assigned to the same Push run")

    def cohort(selected: list[dict]) -> dict:
        n = len(selected)
        selected_ids = {v["episode_id"] for v in selected}
        scalars = {}
        for key in first["pose"]:
            measured = [v["pose"][key]["mean"] for v in selected if v["pose"][key]["mean"] is not None]
            scalars[key] = {"label": first["pose"][key]["label"], "unit": first["pose"][key]["unit"],
                            "episode_means": distribution(measured), "valid_episodes": len(measured),
                            "unknown_episodes": n - len(measured), "total_episodes": n,
                            "valid_samples": sum(v["pose"][key]["valid_samples"] for v in selected),
                            "unknown_samples": sum(v["pose"][key]["unknown_samples"] for v in selected)}
        fractions = [v["stability"]["true_samples"] / v["stability"]["known_samples"]
                     for v in selected if v["stability"]["known_samples"]]
        numerator = sum(v["stability"]["true_samples"] for v in selected)
        denominator = sum(v["stability"]["known_samples"] for v in selected)
        recoveries = [r for v in selected for r in v["recoveries"]]
        return {"total_episodes": n, "record_ids": [r["record_id"] for r in records if r["value"]["episode_id"] in selected_ids],
                "outcomes": dict(sorted(Counter(v["outcome_class"] for v in selected).items())),
                "pose": scalars, "stability": {"episode_equal": distribution(fractions),
                    "valid_episodes": len(fractions), "unknown_episodes": n - len(fractions),
                    "pooled_true_samples": numerator, "pooled_known_samples": denominator,
                    "pooled_fraction": numerator / denominator if denominator else None,
                    "unknown_samples": sum(v["stability"]["unknown_samples"] for v in selected)},
                "any_recovered": _tri([v["any_recovered"] for v in selected]),
                "final_stable": _tri([v["final"]["stable"] for v in selected]),
                "final_unstable_after_recovery": _tri([v["final_unstable_after_recovery"] for v in selected]),
                "recovery_units": {"unit": "sequence" if binding["mechanism"] == "world_planar_external_force" else "event",
                    "total": len(recoveries), "status_counts": dict(sorted(Counter(r["status"] for r in recoveries).items())),
                    "latency_s_recovered_only": distribution([r["latency_s"] for r in recoveries if r["latency_s"] is not None])}}
    facets = {"all": cohort(values)}
    for outcome in ("success", "failure", "unknown"):
        facets["outcome:" + outcome] = cohort([v for v in values if v["outcome_class"] == outcome])
    for stratum in sorted({v["stratum"] for v in values}):
        facets["stratum:" + stratum] = cohort([v for v in values if v["stratum"] == stratum])
    return facets


def select_examples(records: list[dict]) -> dict:
    bins = defaultdict(list)
    for r in records:
        v = r["value"]
        bins[(v["stratum"], v["outcome_class"])].append(r)
    def key(r, center):
        v = r["value"]
        speed = v["pose"]["planar_speed_mps"]["mean"]
        return (speed is None, abs(speed-center) if speed is not None and center is not None else 0,
                v["run_index"], v["env_index"])
    selected = []
    for _, group in sorted(bins.items()):
        center = distribution([r["value"]["pose"]["planar_speed_mps"]["mean"] for r in group
                               if r["value"]["pose"]["planar_speed_mps"]["mean"] is not None])["median"]
        selected.append(min(group, key=lambda r:key(r,center))["record_id"])
    witnesses = {}
    ordered = sorted(records, key=lambda r: (r["value"]["run_index"], r["value"]["env_index"]))
    for status in ("recovered", "full_window_without_dwell", "terminal_before_window_end"):
        witnesses[status] = next(({"record_id": r["record_id"], "slot": recovery["slot"]}
            for r in ordered for recovery in r["value"]["recoveries"] if recovery["status"] == status), None)
    witnesses["final_unstable_after_recovery"] = next(({"record_id": r["record_id"]}
        for r in ordered if r["value"]["final_unstable_after_recovery"] is True), None)
    return {"rule": "nearest_median_root_speed_per_stratum_outcome_then_run_env",
            "representatives": selected, "witnesses": witnesses}


def _fmt(value) -> str:
    return format_number(value, 6)


def _render_claims(domain: dict, facets: dict, examples: list[dict], selection: dict, parameters: dict, timing: dict) -> list[dict]:
    claims = []
    def claim(key, text, refs, mandatory=True):
        claims.append({"id": stable_id("push-claim", TEMPLATE_VERSION, domain, key, text, refs),
                       "key": key, "text": text, "record_ids": refs, "mandatory": mandatory,
                       "kind": "measured_relation" if refs else "scope_limitation",
                       "status": "supported_measurement" if refs else "unavailable_or_contract_boundary"})
    all_ = facets["all"]
    n, refs = all_["total_episodes"], all_["record_ids"]
    counts = all_["outcomes"]
    claim("scope", f"本卡仅描述同一权重、{domain['mechanism']}协议的{n}个公开历史回合，来自{len(domain['manifest_sha256'])}份档案；任务成功{counts.get('success', 0)}、失败{counts.get('failure', 0)}、结果未知{counts.get('unknown', 0)}回合。", refs)
    anchor = "最后一个计划脉冲结束后，完整且验证的序列" if domain["mechanism"] == "world_planar_external_force" else "每次已验证速度写入事件后"
    claim("protocol", f"控制记录dt={_fmt(timing['step_dt_s'])}s；{anchor}的恢复窗为其后{timing['recovery_window_steps']}格，连续≥{timing['required_dwell_steps']}格满足五条件才锁存恢复：速度模≤{_fmt(parameters['recovery_velocity_mps'])}m/s、角速度模≤{_fmt(parameters['recovery_angular_velocity_radps'])}rad/s、|roll|与|pitch|≤{_fmt(parameters['recovery_tilt_rad'])}rad、高度≥{_fmt(parameters['recovery_height_m'])}m。终点稳定与任务结局另判。", refs)
    for key in ("planar_speed_mps", "xy_drift_m", "height_m"):
        m = all_["pose"][key]
        label, unit = m["label"], m["unit"]
        claim(key, f"全程{label}：先逐回合取有效记录均值，再回合等权平均为{_fmt(m['episode_means']['mean'])}{unit}；回合均值中位数{_fmt(m['episode_means']['median'])}{unit}，有效{m['valid_episodes']}/{n}回合，未知{m['unknown_episodes']}回合；有效/未知记录{m['valid_samples']}/{m['unknown_samples']}。", refs)
    s = all_["stability"]
    claim("stable", f"五条件稳定记录占比：回合等权平均{_fmt(s['episode_equal']['mean'])}（有效{s['valid_episodes']}/{n}回合，未知{s['unknown_episodes']}）；另按记录格池化为{s['pooled_true_samples']}/{s['pooled_known_samples']}={_fmt(s['pooled_fraction'])}，未知{s['unknown_samples']}格。两种加权分别报告，稳定不等于任务成功。", refs)
    recovery = all_["recovery_units"]
    status_text = "、".join(f"{STATUSES_ZH[k]}{v}" for k, v in recovery["status_counts"].items()) or "无适用单位"
    claim("recovery", f"恢复按{recovery['unit']}计，共{recovery['total']}单位：{status_text}；截尾或未知不算完整窗未恢复，velocity重复扰动可每回合贡献两个单位。", refs)
    final = all_["final_stable"]
    unstable = all_["final_unstable_after_recovery"]
    claim("final", f"真实任务终点满足五条件为{final['true']}/{final['valid']}回合，终点未知{final['unknown']}/{n}；已观测锁存恢复且终点可判断者中，终点再次不稳定{unstable['true']}/{unstable['valid']}回合，其余{unstable['unknown']}/{n}回合不适用或未知。锁存恢复不保证其后一直稳定。", refs)
    facet_order = ["outcome:success", "outcome:failure", "outcome:unknown", *sorted(k for k in facets if k.startswith("stratum:"))]
    for name in facet_order:
        facet = facets[name]
        r = facet["recovery_units"]
        m = facet["pose"]["planar_speed_mps"]
        claim(name, f"分面{name}：总{facet['total_episodes']}回合；根部速度模回合均值等权平均{_fmt(m['episode_means']['mean'])}m/s，有效{m['valid_episodes']}、未知{m['unknown_episodes']}回合；观测锁存恢复{r['status_counts'].get('recovered', 0)}/{r['total']}{r['unit']}单位（其余保留各自截尾/未恢复状态）。", facet["record_ids"], name != "outcome:unknown")
    capability = domain["root_signed_xy"]
    claim("capability", f"根部方向能力={capability}；扰动向量与根部运动是不同证据，世界XY不是机器人前后左右。pre是事件前固定窗，可能含此前扰动；零分母不构成阴性结论。速度×dt仅是控制记录格积分代理，不是精确路径；无足接触、支撑、步态、动作力矩和逐步奖励证据。", [])
    claim("reward_parent", "父代对照不可用：未提供合格父代档案。评测reward=0来自评测stub，不能代表训练奖励；本卡未重算训练分项，也不主张奖励、根部运动或外扰造成某足部机制。未调用视觉或模型，尚未接入正常runner和最终作者payload。", [])
    by_id = {r["record_id"]: r for r in examples}
    for status in WITNESS_ORDER:
        witness = selection["witnesses"][status]
        label = STATUSES_ZH.get(status, "恢复后终点不稳定")
        if witness is None:
            claim("witness:" + status, f"全队列{n}回合扫描：{label}的预声明见证为no_witness（仅表示本批未找到符合该观测定义的例子）。", refs)
            continue
        v = by_id[witness["record_id"]]["value"]
        if "slot" in witness:
            r = next(r for r in v["recoveries"] if r["slot"] == witness["slot"])
            detail = f"锚点{r['anchor_step']}，窗观测{r['observed_window_samples']}/{r['expected_window_samples']}格、未观测{r['unobserved_window_samples']}格；已观测中稳定{r['stable_window_samples']}格、未知稳定{r['unknown_stability_samples']}格；锁存前最长dwell={r['max_dwell_before_latch_steps']}/{r['required_dwell_steps']}格，恢复步={r['recovery_step']}"
        else:
            detail = "已有恢复锁存，真实终点五条件不稳定"
        claim("witness:" + status, f"{label}见证r{v['run_index']}.e{v['env_index']}：{detail}；任务结局{v['task_outcome']}。", [witness["record_id"]])
    for record in examples:
        if record["record_id"] not in selection["representatives"]:
            continue
        v = record["value"]
        statuses = "、".join(STATUSES_ZH[r["status"]] for r in v["recoveries"]) or "无适用扰动"
        label = v["task_outcome"] or "未知"
        claim("example:" + v["episode_id"], f"预选r{v['run_index']}.e{v['env_index']}（{v['stratum']}）：任务记录{label}；{statuses}；观测{v['coverage']['observed_samples']}/{v['coverage']['expected_samples']}格，根部速度模均值{_fmt(v['pose']['planar_speed_mps']['mean'])}m/s；终点stable={v['final']['stable']}。", [record["record_id"]], False)
        for event in v["events"]:
            if event["windows"] is None:
                claim(f"event:{v['episode_id']}:{event['slot']}", f"r{v['run_index']}.e{v['env_index']}计划扰动{event['slot']}未观测，前/中/后窗口未知。", [record["record_id"]], False)
                continue
            parts = []
            for name in ("pre", "during", "post"):
                window = event["windows"][name]
                if window is None:
                    parts.append(f"{name}=瞬时写入无持续控制格")
                else:
                    metric = window["pose"]["planar_speed_mps"]
                    parts.append(f"{name}[{window['steps_inclusive'][0]},{window['steps_inclusive'][1]}]速度模均值{_fmt(metric['mean'])}m/s（有效{metric['valid_samples']}/{window['expected_samples']}格，未知{metric['unknown_samples']}）")
            along = event["root_along_disturbance"]
            vector = ",".join(_fmt(x) for x in event["vector_world"])
            vector_unit = DISTURBANCES[domain["mechanism"]][1]
            claim(f"event:{v['episode_id']}:{event['slot']}", f"r{v['run_index']}.e{v['env_index']}扰动{event['slot']}：记录世界向量[{vector}]{vector_unit}，验证={event['injection_verified']}；" + "；".join(parts)
                  + f"；post根部速度在记录扰动世界XY方向的投影均值{_fmt(along['mean_velocity_mps'])}m/s（{along['status']}，有效{along['valid_samples']}、未知{along['unknown_samples']}格），不归因为足部补偿。", [record["record_id"]], False)
    return claims


def project_push_card(card: dict, max_bytes: int = MAX_PUBLIC_BYTES, *,
                      episodes: list[NumericEpisode], source_id: str) -> dict:
    """Bind every operand to qualified raw episodes before public projection.

    IDs establish byte identity only. The caller supplies the same explicit
    adapter source context as the builder; a card cannot authorize its operands.
    """
    if card["schema_version"] != CARD_VERSION:
        raise ValueError("not an offline Push root card")
    core = {k: v for k, v in card.items() if k not in {"card_id", "public_report", "projection_audit"}}
    if card["card_id"] != stable_id("push-card", core):
        raise ValueError("Push card content/identity hash mismatch")
    records = [measure_push_episode(e) for e in episodes]
    expected = _card_core(records, source_id)
    for name, value in expected.items():
        if card[name] != value:
            raise ValueError(f"Push card {name} semantic operands differ from raw episodes")
    return _project_bound_card(card, max_bytes)


def _project_bound_card(card: dict, max_bytes: int = MAX_PUBLIC_BYTES) -> dict:
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PUBLIC_BYTES:
        raise ValueError("Push public byte budget must be in 1..12288")
    claims = card["claims"]
    def output(ids):
        selected = set(ids)
        return {"schema_version": PUBLIC_VERSION, "domain": card["domain"],
            "card_id": card["card_id"], "template_version": TEMPLATE_VERSION,
            "claims": [{"id":c["id"],"text":c["text"]} for c in claims if c["id"] in selected],
            "omitted_claim_count":len(claims)-len(selected), "delivery_status":"offline_projection_only"}
    public, _, _ = select_whole_claims([c["id"] for c in claims if c["mandatory"]],
        [c["id"] for c in claims if not c["mandatory"]],output,max_bytes=max_bytes)
    return public


def _card_core(records: list[dict], source_id: str) -> dict:
    cohorts = aggregate_push(records)
    selection = select_examples(records)
    by_id = {r["record_id"]: r for r in records}
    example_ids = list(dict.fromkeys([*selection["representatives"],
        *(selection["witnesses"][key]["record_id"] for key in WITNESS_ORDER if selection["witnesses"][key] is not None)]))
    examples = [by_id[key] for key in example_ids]
    v = records[0]["value"]
    domain = {"source_id": source_id, "acceptance_split": "public_dev",
        "checkpoint_hash": v["source"]["checkpoint_hash"],
        "resolved_protocol_hash": v["parameters"]["resolved_protocol_hash"],
        "mechanism": v["parameters"]["mechanism"],
        "root_signed_xy": v["source"]["capabilities"]["root_signed_xy"],
        "manifest_sha256": sorted({r["value"]["source"]["manifest_sha256"] for r in records})}
    claims = _render_claims(domain, cohorts, examples, selection, v["parameters"], v["timing"])
    return {"schema_version": CARD_VERSION, "domain": domain, "config": dict(CONFIG),
            "adapter_version": ADAPTER_VERSION, "template_version": TEMPLATE_VERSION,
            "parameters": v["parameters"], "timing": v["timing"],
            "cohorts": cohorts, "selection": selection, "examples": examples, "claims": claims,
            "parent_comparison": {"status": "unavailable", "reason": "no_qualified_parent_archive"}}


def build_push_card(records: list[dict], source_id: str, *, episodes: list[NumericEpisode]) -> dict:
    """Measured fields retain their raw channel/column, episode and time role."""
    if len(records) != len(episodes):
        raise ValueError("Push measurement/raw episode count mismatch")
    for record, episode in zip(records, episodes):
        expected = measure_push_episode(episode)
        if record != expected:
            raise ValueError(f"Push measurement operands differ from raw episode {episode.episode_id}")
    card = _card_core(records, source_id)
    card["card_id"] = stable_id("push-card", card)
    card["public_report"] = _project_bound_card(card)
    selected = {c["id"] for c in card["public_report"]["claims"]}
    card["projection_audit"] = {"max_bytes": MAX_PUBLIC_BYTES, "bytes": len(json_bytes(card["public_report"])),
        "omitted": [{"id": c["id"], "reason": "public_byte_budget"} for c in card["claims"] if c["id"] not in selected]}
    return card


def build_push_artifacts(sources: list[dict], *, source_id: str, output: Path) -> dict:
    """Local CPU entrypoint. Fresh output avoids silently mixing implementations."""
    started = time.perf_counter()
    episodes = [e for s in sources for e in iter_push_episodes(s["reference"], expected_identity=s["identity"])]
    records = [measure_push_episode(e) for e in episodes]
    card = build_push_card(records, source_id, episodes=episodes)
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output / name).write_bytes(json_bytes(value) + b"\n")
    (output / "evidence.jsonl").write_bytes(b"".join(json_bytes(r) + b"\n" for r in records))
    write("policy_card.json", card)
    write("public.json", card["public_report"])
    (output / "policy_card_zh.md").write_text("# Push 根部行为卡（离线）\n\n" + "\n\n".join(
        f"{c['text']}\n\n`{c['id']}`" for c in card["claims"]) + "\n", encoding="utf-8", newline="\n")
    paths = [Path(__file__), Path(__file__).with_name("trajectory_adapter.py"),
             Path(__file__).with_name("trajectory_measurement.py")]
    manifest = {"schema_version": "push.root_artifacts.v1", "complete": True, "source_id": source_id,
        "sources": sources, "config": CONFIG, "card_id": card["card_id"], "episodes": len(records),
        "processing_scope": "all_valid_control_rows_root_only", "integration": "offline_only_P2b2_pending",
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())},
        "wall_seconds": time.perf_counter() - started, "model_calls": 0, "gpu_runs": 0}
    write("manifest.json", manifest)
    return manifest


def load_push_artifacts(manifest_path: Path, *, manifest_sha256: str, expected_sources: list[dict]) -> dict:
    """Verify an offline artifact before reuse; source admission remains explicit.

    P2b2 must additionally bind these sources to its running phase and policy.
    No local locator is itself a public permission or source-identity grant.
    """
    data = manifest_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest_sha256:
        raise ValueError("Push artifact manifest hash mismatch")
    manifest = json.loads(data)
    if (manifest["schema_version"] != "push.root_artifacts.v1" or manifest["complete"] is not True
            or manifest["sources"] != expected_sources):
        raise ValueError("Push artifact source binding mismatch")
    root = manifest_path.parent
    for name in ("evidence.jsonl", "policy_card.json", "public.json", "policy_card_zh.md"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest["artifacts"][name]:
            raise ValueError("Push artifact bytes changed")
    records = [json.loads(line) for line in (root / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    allowed = {s["reference"]["manifest_sha256"]: s["identity"] for s in expected_sources}
    if any(r["value"]["source"]["manifest_sha256"] not in allowed or
           r["value"]["source"]["identity"] != allowed[r["value"]["source"]["manifest_sha256"]] for r in records):
        raise ValueError("Push evidence source binding mismatch")
    card = json.loads((root / "policy_card.json").read_bytes())
    episodes = [e for s in expected_sources for e in iter_push_episodes(s["reference"], expected_identity=s["identity"])]
    rebuilt = build_push_card(records, manifest["source_id"], episodes=episodes)
    if rebuilt != card:
        raise ValueError("Push card differs from complete bound episode evidence")
    if rebuilt["public_report"] != json.loads((root / "public.json").read_bytes()):
        raise ValueError("Push public artifact differs from its card")
    return card
