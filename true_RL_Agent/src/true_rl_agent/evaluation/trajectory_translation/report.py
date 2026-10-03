"""Deterministic offline cards and resolvable claims; no author ABI integration."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .cohort import (ACTIVITY_VERSION, COHORT_VERSION, aggregate_cohort, cohort_binding,
                     compose_activity, run_summary, select_examples, summarize_episode)
from .contracts import CATALOG_VERSION, EXTRACTOR_VERSION, NumericEpisode, stable_id
from .measurement import NUMERIC_CONFIG

CARD_VERSION = "trajectory.offline_card.v1"
TEMPLATE_VERSION = "trajectory.zh_templates.v1"
SELECTION_VERSION = "trajectory.card_selection.v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_groups(path: Path, manifest: dict):
    """Keep only one episode's measurements, not the million event dictionaries."""
    digest, counts = hashlib.sha256(), Counter()
    header, measurements, refs, offset, first = None, {}, {}, 0, 0
    file_id = manifest["evidence"]["sha256"]
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            record = json.loads(raw)
            counts[record["record_type"]] += 1
            locator = {"file_id": file_id, "byte_offset": offset, "byte_length": len(raw)}
            if record["record_type"] == "episode":
                if header is not None:
                    yield header, measurements, refs, {"file_id": file_id, "byte_offset": first, "byte_length": offset - first}
                header, measurements, refs, first = record, {}, {}, offset
                refs["header"] = {**locator, "record_id": stable_id("header", record["episode_id"])}
            else:
                if header is None or record["episode_id"] != header["episode_id"]:
                    raise ValueError("record outside its episode namespace")
                if record["catalog_version"] != CATALOG_VERSION or record["extractor_version"] != EXTRACTOR_VERSION:
                    raise ValueError("incompatible measurement version")
                if record["evidence_ref"]["manifest_sha256"] != header["source"]["manifest_sha256"]:
                    raise ValueError("record source differs from episode")
                if record["record_type"] == "measurement":
                    if record["key"] in measurements:
                        raise ValueError("duplicate measurement key")
                    measurements[record["key"]] = record
                    refs[record["key"]] = {**locator, "record_id": record["record_id"]}
            offset += len(raw)
    if header is not None:
        yield header, measurements, refs, {"file_id": file_id, "byte_offset": first, "byte_length": offset - first}
    if digest.hexdigest() != file_id or offset != manifest["evidence"]["bytes"]:
        raise ValueError("P1a evidence byte/hash mismatch")
    if dict(counts) != manifest["record_counts"]:
        raise ValueError("P1a record counts mismatch")


def load_bundle(manifest_path: Path, expected_sha256: str, raw_episodes: Iterable[NumericEpisode]) -> dict:
    """Bind P1a bytes to newly validated source episodes supplied by a task adapter."""
    if file_sha256(manifest_path) != expected_sha256:
        raise ValueError("P1a manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["schema_version"] != "trajectory.numeric_manifest.v1" or manifest["complete"] is not True
            or manifest["acceptance_split"] != "public_dev" or manifest["catalog_version"] != CATALOG_VERSION
            or manifest["numeric_config"] != NUMERIC_CONFIG or manifest["extractor_version"] != EXTRACTOR_VERSION):
        raise ValueError("incompatible or non-public P1a input")
    path = manifest_path.parent / manifest["evidence"]["path"]
    stream = iter(_episode_groups(path, manifest))
    selection = None if manifest["selection"] == "all_episodes" else {tuple(pair) for pair in manifest["selection"]}
    summaries, binding, source = [], None, None
    scanned = scanned_rows = 0
    for episode in raw_episodes:
        scanned += 1
        scanned_rows += len(episode.steps)
        if selection is not None and (episode.run_index, episode.env_index) not in selection:
            continue
        try:
            header, measures, refs, timeline = next(stream)
        except StopIteration as exc:
            raise ValueError("P1a evidence omitted a selected source episode") from exc
        if (header["episode_id"] != episode.episode_id or header["source"] != episode.source
                or header["task"] != episode.task or header["entities"] != {k: list(v) for k, v in episode.entities.items()}
                or header["coverage"]["valid_rows"] != len(episode.steps)
                or header["coverage"]["first_step"] != int(episode.steps[0])
                or header["coverage"]["last_step"] != int(episode.steps[-1])
                or header["coverage"]["dt_s"] != episode.dt_s):
            raise ValueError("P1a header differs from validated raw source/task/identity/coverage")
        if header["source"]["manifest_sha256"] != manifest["source_manifest_ref"]["manifest_sha256"]:
            raise ValueError("manifest/source binding mismatch")
        expected_keys = {"root_progress", "low_progress", "body_up", "contact_combinations", "force_summary", "task_endpoint"}
        for entity in episode.entities["support"]:
            expected_keys.update(entity + suffix for suffix in (".marker", ".low_speed_off", ".relative_position", ".history_diagnostic"))
        for entity in episode.entities["forbidden"]:
            expected_keys.update((entity + ".current_marker", entity + ".history_diagnostic"))
        expected_keys.update(entity + ".pose" for entity in episode.entities["joints"])
        if set(measures) != expected_keys:
            raise ValueError("incomplete enabled measurement catalog")
        current_binding = cohort_binding(header, manifest["catalog_version"])
        if episode.task["contact_threshold_n"] != 1.0:
            raise ValueError("P0 v2 card lexicon requires the frozen 1 N marker threshold")
        if binding is not None and binding != current_binding:
            raise ValueError("mixed source/trial/checkpoint/protocol/robot cohort")
        binding, source = current_binding, header["source"]
        activity = compose_activity(episode)
        refs["low_progress_activity"] = {"record_id": activity["record_id"], "local": True}
        summary = summarize_episode(header, measures, refs, activity)
        summary["timeline_ref"] = timeline
        summaries.append(summary)
    try:
        next(stream)  # Runs the whole-file hash/count check after the last yield.
    except StopIteration:
        pass
    else:
        raise ValueError("P1a contains episodes outside validated selection")
    if (scanned != manifest["scanned_episodes"] or scanned_rows != manifest["scanned_valid_rows"]
            or len(summaries) != manifest["emitted_episodes"]
            or sum(e["coverage"]["valid_rows"] for e in summaries) != manifest["emitted_valid_rows"]):
        raise ValueError("source/selection processing counts differ")
    if not summaries:
        raise ValueError("empty selected source")
    return {"manifest": manifest, "binding": binding, "source": source, "summaries": summaries,
            "input": {"manifest_path": str(manifest_path.resolve()), "manifest_sha256": expected_sha256,
                      "evidence_path": str(path.resolve()), **manifest["evidence"]}}


def read_link(link: dict, files: dict[str, dict]) -> dict:
    path = Path(files[link["file_id"]]["evidence_path"])
    with path.open("rb") as stream:
        stream.seek(link["byte_offset"])
        raw = stream.read(link["byte_length"])
    record = json.loads(raw)
    actual = record.get("record_id") or stable_id("header", record["episode_id"])
    if actual != link["record_id"]:
        raise ValueError("reference byte locator/record ID mismatch")
    return record


def resolve_path(record: Any, path: list) -> Any:
    for key in path:
        record = record[key]
    return record


def _n(value: Any) -> str:
    if value is None:
        return "未知"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _vector(values: list | None) -> str:
    return "未知" if values is None else "[" + ", ".join(_vector(v) if isinstance(v, list) else _n(v) for v in values) + "]"


def _run(run: dict | None) -> str:
    if run is None:
        return "没有已观察到的该状态区间"
    a, b = run["steps_half_open"]
    return (f"步{a}–{b-1}，{run['sample_count']}条，格时长代理{_n(run['sample_cell_duration_proxy_s'])}s、"
            f"观测跨度{_n(run['observed_span_s'])}s，左/右截尾={_n(run['left_censored'])}/{_n(run['right_censored'])}")


def _den(m: dict) -> str:
    label = "相邻记录边" if m["count_unit"] == "adjacent_record_edge" else "记录"
    return (f"有效/未知/未处理/总回合={m['valid_episode_count']}/{m['unknown_episode_count']}/"
            f"{m['unprocessed_episode_count']}/{m['total_episode_count']}，"
            f"有效/未知{label}={m['valid_samples']}/{m['unknown_samples']}，缺步={m['unobserved_control_steps']}")


def cohort_metric_literals(template: str, key: str, metrics: dict) -> dict:
    """Derive the enabled cohort wording from its actual metric key and unit."""
    metric = metrics[key]
    def checked(literals, unit):
        if metric["unit"] != unit:
            raise ValueError("cohort metric unit differs from descriptor")
        return literals

    scalar_labels = {"net_progress_m": ("首末记录任务净进度", "m"),
                     "complete_path_xy_m": ("完整有效相邻边水平采样路径", "m"),
                     "terminal_time_s": ("有效终点时刻", "s")}
    activity_labels = {"low_progress_any_foot_moving": "低进度确认且至少一足XY速度≥0.05",
                       "low_progress_all_feet_low_speed": "低进度确认且全部足XY速度<0.05"}
    if template == "scalar_distribution" and key in scalar_labels:
        label, unit = scalar_labels[key]
        return checked({"label": label, "unit": unit}, unit)
    if template in ("ratio", "presence") and key in activity_labels:
        label = activity_labels[key]
        return checked({"label": "全回合可判定记录中的“" + label + "”合取占比" if template == "ratio" else label}, "1")
    if template == "presence" and key.endswith(".low_speed_off.long_observed"):
        entity = key.removesuffix(".low_speed_off.long_observed")
        return checked({"label": f"{entity}低XY速度off已观察连续记录格时长≥0.5s"}, "episode")
    entity, _, suffix = key.rpartition(".")
    if template == "off_runs" and suffix == "marker":
        return checked({"label": "", "entity": entity}, "1")
    if template == "ratio":
        labels = {"marker": "current marker采样占比", "norm_share": "净法向力模份额",
                  "signed_fz_share": "有符号世界Fz份额"}
        if suffix in labels:
            return checked({"label": f"{entity} {labels[suffix]}"}, "1" if suffix == "marker" else "force_share")
        if entity == "combination":
            code = int(suffix)
            support_count = sum(k.endswith(".marker") for k in metrics)
            vector = [bool(code & (1 << i)) for i in range(support_count)]
            return checked({"label": f"接触组合code={code}（角色顺序布尔向量{vector}）采样占比；全off不等于全身腾空"}, "1")
    raise ValueError("unsupported cohort template/metric semantics")


def render_statement(template: str, s: dict) -> str:
    """Closed vocabulary: no arbitrary model text or physical upgrades."""
    if template == "cohort_scope":
        return f"本队列共{s['episodes']}回合、{s['rows']}条有效记录；存档任务结果：{_n(s['outcomes'])}。"
    if template == "scalar_distribution":
        m = s["metric"]
        return f"{s['label']}：回合等权中位数{_n(m['episode_equal']['median'])}{s['unit']}，四分位[{_n(m['episode_equal']['q25'])},{_n(m['episode_equal']['q75'])}]；{_den(m)}。"
    if template == "ratio":
        m = s["metric"]
        p = m.get("pooled", {})
        return (f"{s['label']}：回合等权均值{_n(m['episode_equal']['mean'])}，池化比值{_n(p.get('value'))}"
                f"（分子{_n(p.get('numerator'))}/分母{_n(p.get('denominator'))} {p.get('operand_unit','')}）；{_den(m)}。")
    if template == "presence":
        m = s["metric"]
        p = m["observed_presence"]
        return (f"{s['label']}：已观察阳性{p['numerator']}/可判定{p['valid_denominator']}回合，"
                f"未知{p['unknown_episode_count']}、未处理{p['unprocessed_episode_count']}、总数{p['total_episode_count']}；"
                "仅限现有记录，未知不作阴性，多个模式可重叠。")
    if template == "off_runs":
        r = s["metric"]["off_runs"]
        return (f"{s['entity']}的 marker-off 区间共{r['count']}段，完整{r['complete_count']}段，"
                f"左/右截尾{r['left_censored_count']}/{r['right_censored_count']}段（可重叠）；"
                f"只在完整段内，按区间等权的观测跨度中位数{_n(r['complete_observed_span_s']['median'])}s；{_den(s['metric'])}。")
    if template == "episode_scope":
        c, t = s["coverage"], s["task"]
        return (f"{s['name']}：条件{t['cell_id']}，任务存档结果{t['outcome']['primary_reason']}，"
                f"记录步{c['first_step']}–{c['last_step']}共{c['valid_rows']}条，dt={_n(c['dt_s'])}s；"
                "使用控制步末/重置前记录，没有t=0或终局后观察。")
    if template == "episode_root":
        v = s["value"]
        d = v["drawdown"]
        return (f"根部净位移xyz={_vector(v['net_world_xyz_m'])}m，任务净进度{_n(v['net_progress_m'])}m；"
                f"采样折线路径XY/XYZ={_n(v['sampled_path_xy_m'])}/{_n(v['sampled_path_xyz_m'])}m，完整路径={_n(v['path_complete'])}；"
                f"正/负进度变化量={_n(v['positive_progress_variation_m'])}/{_n(v['negative_progress_variation_m'])}m；"
                f"最大已观察回撤{_n(d['max_drawdown_m'])}m，峰/谷/首次返回步={_n(d['peak_step'])}/{_n(d['trough_step'])}/{_n(d['return_step'])}，"
                f"返回观察右截尾={_n(d.get('return_observation_right_censored'))}。")
    if template == "episode_attitude":
        v = s["value"]
        runs = [r for r in v["negative"]["intervals"] if r["state"] == "true"]
        return (f"归一化wxyz的身体上轴竖直分量首个/末个有效记录值={_n(v['up_z']['first'])}/{_n(v['up_z']['last'])}"
                f"（步{_n(v['up_z']['first_valid_step'])}/{_n(v['up_z']['last_valid_step'])}），最小值{_n(v['up_z']['minimum'])}，"
                f"最小值步{_n(v['up_z']['minimum_step'])}；负值记录{v['negative']['true_count']}条，"
                f"首次负值步{_n(runs[0]['steps_half_open'][0] if runs else None)}，姿态未知{v['up_z']['unknown_count']}条；不解释失败原因。")
    if template == "episode_marker":
        v = s["value"]
        r = run_summary(v["intervals"], "false")
        return (f"{s['entity']} current marker：on/off/unknown={v['true_count']}/{v['false_count']}/{v['unknown_count']}，"
                f"有效{v['valid_count']}条、占比{_n(v['fraction'])}；onset/offset={v['observed_onset_count']}/{v['observed_offset_count']}；"
                f"完整off段{r['complete_count']}、最短off段{_n(r['shortest_samples'])}条；最长off：{_run(r['longest'])}。")
    if template == "episode_low_speed":
        v = s["value"]
        longest = v["longest"]
        text = f"{s['entity']}速度通道XY<0.05m/s且current marker off："
        if longest is None:
            return text + f"未观察到满足记录，谓词未知{v['unknown_count']}条。"
        return (text + _run(longest["interval"]) + f"；该段XY最大/中位速度={_n(longest['max_speed_xy_mps'])}/{_n(longest['median_speed_xy_mps'])}m/s，"
                f"XYZ诊断最大速度{_n(longest['max_speed_xyz_mps'])}m/s，力模最大{_n(longest['max_force_norm_n'])}N，"
                f"上轴竖直分量范围[{_n(longest['minimum_up_z'])},{_n(longest['maximum_up_z'])}]（姿态有效{longest['orientation_valid_count']}条）；不据此断言直立举足。")
    if template == "episode_force":
        v = s["value"]
        return (f"按{'/'.join(s['entities'])}顺序，记录净法向力模份额={_vector(v['norm_share'])}，有符号世界Fz份额={_vector(v['signed_fz_share'])}；"
                f"各自联合有效记录{v['joint_valid_count']}/{v['signed_fz_joint_valid_count']}，未知{v['unknown_joint_count']}/{v['signed_fz_unknown_joint_count']}；"
                "不是重量份额、完整外力或物理冲量。")
    if template == "episode_combinations":
        v = s["value"]
        return (f"按{'/'.join(s['entities'])}顺序、权重{_vector(v['bit_weights'])}编码，各组合0–{len(v['counts'])-1}记录数={_vector(v['counts'])}；"
                f"联合有效/未知={v['valid_count']}/{v['unknown_count']}；全off不证明全身腾空，组合不命名步态。")
    if template == "episode_relative":
        v = s["value"]
        ranges = [[r["minimum"], r["maximum"]] for r in v["body_xyz_m"]]
        return (f"{s['entity']}足参考点相对根部的首帧xyz={_vector(v['first_body_xyz_m'])}m，身体坐标xyz范围={_vector(ranges)}m；"
                f"另列速度通道世界XY范围[{_n(v['world_xy_speed_mps']['minimum'])},{_n(v['world_xy_speed_mps']['maximum'])}]m/s；"
                "二者参考点映射未验证，不把速度当该位置精确导数。")
    if template == "episode_history":
        v = s["value"]
        if any(name.endswith("_history") for name in v["missing_channels"]):
            return (f"{s['entity']} history未采集，有效/unknown={v['valid_count']}/{v['unknown_count']}；"
                    "history ANY及其与current的差异不可判定，不把缺失算作off或无事件。")
        return (f"{s['entity']} history ANY超过1N：on/off/unknown={v['true_count']}/{v['false_count']}/{v['unknown_count']}；"
                f"history-on且current-off={v['history_any_on_current_off_count']}条，反向差异={v['current_on_history_all_off_count']}条，"
                f"联合有效{v['joint_valid_count']}条；无逐槽时间/valid，不替代current。")
    if template == "episode_terminal":
        v = s["value"]
        if "forbidden_contact_forces_w_history" in v["missing_channels"]:
            return (f"终点步{v['terminal_step']}、时刻{_n(v['terminal_time_s'])}s，manager/wrapper标记={_n(v['terminal_flags'])}；"
                    f"termination索引依次对应{_vector(s['term_order'])}；"
                    f"base终点current力模={_vector(v['tail_forbidden_norms_n'][-1])}N；"
                    "history未采集、事件不可判定；manager终止信号独立保留，不将其推定为history或current力事件，任务结果保持存档评价器权威。")
        return (f"终点步{v['terminal_step']}、时刻{_n(v['terminal_time_s'])}s，manager/wrapper标记={_n(v['terminal_flags'])}；"
                f"termination索引依次对应{_vector(s['term_order'])}；"
                f"末{len(v['tail_steps'])}条步号={_vector(v['tail_steps'])}，base current力模={_vector(v['tail_forbidden_norms_n'])}N，"
                f"另列history各槽力模={_vector(v['tail_forbidden_history_norms_n'])}N；任务结果保持存档评价器权威。")
    if template == "episode_base_runs":
        current, history = s["current"], s["history"]
        cr = [r for r in current["intervals"] if r["state"] == "true"]
        hr = [r for r in history["intervals"] if r["state"] == "true"]
        if any(name.endswith("_history") for name in history["missing_channels"]):
            return (f"{s['entity']} current越阈{current['true_count']}条，区间：{'；'.join(_run(r) for r in cr) or '无已观察阳性'}；"
                    f"history未采集，有效/unknown={history['valid_count']}/{history['unknown_count']}，不提供history事件结论。")
        return (f"{s['entity']} current越阈{current['true_count']}条，区间：{'；'.join(_run(r) for r in cr) or '无已观察阳性'}；"
                f"history ANY越阈{history['true_count']}条，区间：{'；'.join(_run(r) for r in hr) or '无已观察阳性'}；单条短事件保留。")
    if template == "episode_activity":
        a, b = s["value"]["low_progress_any_foot_moving"], s["value"]["low_progress_all_feet_low_speed"]
        ad, bd = a["longest_true_diagnostics"], b["longest_true_diagnostics"]
        details = f"最长至少一足达阈值段：{_run(ad['interval'] if ad else None)}；最长全部足低速段：{_run(bd['interval'] if bd else None)}。"
        if ad:
            details += (f"前一段按{'/'.join(ad['entity_ids'])}顺序，分足达阈值记录数={_vector(ad['xy_ge_threshold_counts'])}，"
                        f"分足有效数={_vector(ad['valid_counts'])}，同一足覆盖该段每条记录={_vector(ad['same_entity_ge_threshold_every_record'])}。")
        return (f"低进度窗口确认与足速度组合：至少一足XY≥0.05的true/false/unknown={a['true_count']}/{a['false_count']}/{a['unknown_count']}；"
                f"全部足XY<0.05的true/false/unknown={b['true_count']}/{b['false_count']}/{b['unknown_count']}；"
                + details + "合取可由不同足交替贡献，不断言迈步次数或整体静止。")
    if template == "inverted_before":
        if not s["negative_step"] < s["start_step"]:
            raise ValueError("before claim lacks observed ordering")
        return (f"身体上轴已在步{s['negative_step']}出现负值；{s['entity']}所选低速off段首次记录在步{s['start_step']}，"
                "前者早于后者；该顺序不解释机制，低速off本身不能证明直立举足。")
    if template == "successful_return":
        d = s["drawdown"]
        if s["success"] is not True or not (d["max_drawdown_m"] > 0 and d["return_step"] is not None):
            raise ValueError("successful return claim lacks its premises")
        return (f"该成功回合仍有已观察回撤{_n(d['max_drawdown_m'])}m（峰步{d['peak_step']}、谷步{d['trough_step']}、返回步{d['return_step']}）；"
                "回撤过程与成功可同时存在，返回进度不等于已证明恢复平衡。")
    if template == "cell_comparison":
        a, b = s["success"], s["failure"]
        return (f"同一公开cell={s['cell']}的成功/失败子集各{a['episode_count']}/{b['episode_count']}回合；"
                f"首末记录净进度中位数{_n(a['metrics']['net_progress_m']['episode_equal']['median'])}/{_n(b['metrics']['net_progress_m']['episode_equal']['median'])}m；"
                f"成功：{_den(a['metrics']['net_progress_m'])}；失败：{_den(b['metrics']['net_progress_m'])}；"
                f"终点时间中位数{_n(a['metrics']['terminal_time_s']['episode_equal']['median'])}/{_n(b['metrics']['terminal_time_s']['episode_equal']['median'])}s；"
                "仅同cell整回合描述，时长/成功组成不同，未逐实例匹配随机化或命令，不作因果或改善判断。")
    if template == "definition":
        return s["text"]
    raise ValueError(f"unsupported statement template: {template}")


def make_claim(template: str, section: str, scope: str, bindings: dict[str, tuple[str, list]],
               literals: dict, records: dict[str, dict], scope_record_id: str) -> dict:
    slots = dict(literals)
    for name, (record_id, path) in bindings.items():
        slots[name] = resolve_path(records[record_id], path)
    claim = {"record_type": "claim", "kind": "measured_relation", "status": "supported_measurement",
             "section": section, "scope": scope, "template_id": template, "template_version": TEMPLATE_VERSION,
             "scope_record_id": scope_record_id,
             "operands": {k: {"record_id": v[0], "path": v[1]} for k, v in bindings.items()},
             "literals": literals, "text": render_statement(template, slots),
             "applicability": "enabled catalog; stated sampled/readout scope only",
             "review_state": "independent_acceptance_required"}
    if template == "definition":
        claim["kind"], claim["status"] = "capability_statement", "declared_scope"
    claim["claim_id"] = stable_id("claim", CARD_VERSION, claim)
    return claim


def validate_claims(claims: list[dict], records: dict[str, dict]) -> None:
    ids = set()
    for claim in claims:
        if claim["claim_id"] in ids or not claim["operands"]:
            raise ValueError("duplicate or ungrounded claim")
        ids.add(claim["claim_id"])
        if claim["claim_id"] != stable_id("claim", CARD_VERSION, {k: v for k, v in claim.items() if k != "claim_id"}):
            raise ValueError("claim content/identity mismatch")
        slots = dict(claim["literals"])
        scope = records[claim["scope_record_id"]]
        if scope["record_type"] == "episode_summary":
            if claim["scope"] != scope["display_name"]:
                raise ValueError("claim display scope mismatch")
        elif scope["record_type"] == "cohort_summary":
            if claim["scope"] != scope["group"]:
                raise ValueError("claim cohort scope mismatch")
        elif scope["record_type"] != "capabilities" or claim["template_id"] != "definition":
            raise ValueError("invalid claim scope type")
        for name, operand in claim["operands"].items():
            if operand["record_id"] not in records:
                raise ValueError("unresolved claim operand")
            record = records[operand["record_id"]]
            if scope["record_type"] == "episode_summary":
                if record.get("episode_id") != scope["episode_id"]:
                    raise ValueError("claim operand outside episode/source scope")
                source_hash = record.get("source_manifest_sha256") or record.get("evidence_ref", {}).get("manifest_sha256") or record.get("source", {}).get("manifest_sha256")
                if source_hash != scope["source_manifest_sha256"]:
                    raise ValueError("claim source identity mismatch")
            elif scope["record_type"] == "cohort_summary":
                allowed_groups = {scope["group"]}
                if claim["template_id"] == "cell_comparison":
                    allowed_groups |= {scope["group"] + "/success", scope["group"] + "/failure"}
                if record.get("group") not in allowed_groups or not set(record["member_ids"]) <= set(scope["member_ids"]):
                    raise ValueError("claim operand outside cohort/condition scope")
            elif operand["record_id"] != claim["scope_record_id"]:
                raise ValueError("capability claim outside declared scope")
            slots[name] = resolve_path(record, operand["path"])
        if claim["template_id"] in ("scalar_distribution", "ratio", "presence", "off_runs"):
            if scope["record_type"] != "cohort_summary" or set(claim["operands"]) != {"metric"}:
                raise ValueError("cohort metric operand role mismatch")
            path = claim["operands"]["metric"]["path"]
            if len(path) != 2 or path[0] != "metrics":
                raise ValueError("cohort metric path mismatch")
            expected = cohort_metric_literals(claim["template_id"], path[1], scope["metrics"])
            if claim["literals"] != expected:
                raise ValueError("cohort metric label/entity/unit/role mismatch")
        expected_key = {"episode_root": "root_progress", "episode_attitude": "body_up",
                        "episode_marker": ".marker", "episode_low_speed": ".low_speed_off",
                        "episode_relative": ".relative_position", "episode_history": ".history_diagnostic",
                        "episode_force": "force_summary", "episode_combinations": "contact_combinations",
                        "episode_terminal": "task_endpoint", "episode_activity": "low_progress_activity"}.get(claim["template_id"])
        if expected_key is not None:
            source_record = records[claim["operands"]["value"]["record_id"]]
            actual_key = source_record["key"]
            if not (actual_key.endswith(expected_key) if expected_key.startswith(".") else actual_key == expected_key):
                raise ValueError("template measurement meaning mismatch")
            if "entity" in slots and slots["entity"] != source_record["entity_ids"][0]:
                raise ValueError("template entity differs from measurement")
            if "entities" in slots and slots["entities"] != source_record["entity_ids"]:
                raise ValueError("template entity order differs from measurement")
        if claim["template_id"] == "episode_base_runs":
            for role, suffix in (("current", ".current_marker"), ("history", ".history_diagnostic")):
                r = records[claim["operands"][role]["record_id"]]
                if r.get("key") != slots["entity"] + suffix or r["entity_ids"] != [slots["entity"]]:
                    raise ValueError("current/history role or entity mismatch")
        if claim["template_id"] == "cell_comparison":
            for role in ("success", "failure"):
                if slots[role]["group"] != scope["group"] + "/" + role:
                    raise ValueError("comparison outcome role mismatch")
            if slots["cell"] != scope["group"].removeprefix("cell_id:"):
                raise ValueError("comparison cell label mismatch")
        if claim["template_id"] == "inverted_before":
            r = records[claim["operands"]["start_step"]["record_id"]]
            if r.get("key") != slots["entity"] + ".low_speed_off" or r["entity_ids"] != [slots["entity"]]:
                raise ValueError("inverted-before entity mismatch")
        if claim["text"] != render_statement(claim["template_id"], slots):
            raise ValueError("claim text differs from resolved operands")
        expected_kind = ("capability_statement", "declared_scope") if claim["template_id"] == "definition" else ("measured_relation", "supported_measurement")
        if (claim["kind"], claim["status"]) != expected_kind:
            raise ValueError("claim kind/status mismatch")


def select_policy_summary(claims: list[dict], records: dict[str, dict]) -> list[str]:
    """A small policy view references existing audited claims; no new wording."""
    picked = []

    def add(c, unavailable=None):
        if c is None and unavailable is not None:
            c = next((row for row in claims if row["template_id"] == "definition"
                      and row["operands"]["text"]["path"] == ["text", "unavailable_" + unavailable]), None)
        if c is not None and c["claim_id"] not in picked:
            picked.append(c["claim_id"])

    def first(template, scope=None, metric=None):
        return next((c for c in claims if c["template_id"] == template
                     and (scope is None or c["scope"] == scope)
                     and (metric is None or c["operands"].get("metric", {}).get("path") == ["metrics", metric])), None)

    def values(c, name):
        o = c["operands"][name]
        return resolve_path(records[o["record_id"]], o["path"])

    add(first("cohort_scope", "all"))
    add(first("scalar_distribution", "all", "net_progress_m"))
    for key in ("low_progress_any_foot_moving", "low_progress_all_feet_low_speed"):
        add(first("ratio", "all", key))
    for scope in ("all", "success"):
        modes = [c for c in claims if c["scope"] == scope and c["template_id"] == "presence"
                 and c["operands"]["metric"]["path"][-1].endswith(".long_observed")
                 and values(c, "metric")["observed_presence"]["numerator"] > 0]
        if modes:
            chosen = min(modes, key=lambda c: (-values(c, "metric")["observed_presence"]["numerator"], c["claim_id"]))
            add(chosen)
            if scope == "success":
                add(first("presence", "failure", chosen["operands"]["metric"]["path"][-1]), "failure_mode")
        else:
            add(None, scope + "_mode")
            if scope == "success":
                add(None, "failure_mode")
    returns = [c for c in claims if c["template_id"] == "successful_return"]
    if returns:
        add(min(returns, key=lambda c: (-values(c, "drawdown")["max_drawdown_m"], c["claim_id"])))
    else:
        add(None, "successful_return")
    add(first("inverted_before"), "inverted_before")
    short_history = [c for c in claims if c["template_id"] == "episode_base_runs"
                     and values(c, "current")["true_count"] == 0 and values(c, "history")["true_count"] > 0]
    if short_history:
        add(min(short_history, key=lambda c: (values(c, "history")["true_count"], c["claim_id"])))
    else:
        # Current-only archives cannot witness history. Preserve an independent
        # recorded manager endpoint in this version's short-event slot instead.
        missing_history_endpoints = [c for c in claims if c["template_id"] == "episode_terminal"
            and "forbidden_contact_forces_w_history" in values(c, "value")["missing_channels"]]
        if missing_history_endpoints:
            add(min(missing_history_endpoints, key=lambda c: (
                not any(v is True for k, v in values(c, "value")["terminal_flags"].items() if k.startswith("termination_")),
                c["claim_id"])))
        else:
            add(None, "short_history")
    for key in ("activity_definition", "contact_definition", "missing", "deferred", "parent"):
        add(next(c for c in claims if c["template_id"] == "definition" and c["operands"]["text"]["path"] == ["text", key]))
    return picked


def render_markdown(card: dict) -> str:
    lines = ["# 离线策略行为卡片", "", "状态：P1b 开发交付；最终作者协议与投递在 P1c，尚未执行。", "",
             "数字与过程仅解释启用目录和已有记录。展开选择不改变主队列分母；另源反例不进入主队列频率。本页不是盲测、泛化或因果证明。", ""]
    primary = set(card["policy_summary_claim_ids"])
    by_id = {c["claim_id"]: c for c in card["claims"]}
    lines += ["## 策略主视图", "", "以下为离线阅读摘要；完整证据保留在展开审计明细中。作者紧凑卡片、字节预算和最终请求留待P1c。", ""]
    for claim_id in card["policy_summary_claim_ids"]:
        c = by_id[claim_id]
        lines += [f"- **{c['scope']}** {c['text']} `[{c['claim_id']}]`", ""]
    lines += ["## 展开审计明细", ""]
    sections = list(dict.fromkeys(c["section"] for c in card["claims"] if c["claim_id"] not in primary))
    for section in sections:
        lines += [f"<details><summary>{section}</summary>", ""]
        for c in card["claims"]:
            if c["section"] == section and c["claim_id"] not in primary:
                lines += [f"- **{c['scope']}** {c['text']} `[{c['claim_id']}]`", ""]
        lines += ["</details>", ""]
    return "\n".join(lines) + "\n"


def build_card(primary: dict, examples: list[dict], frozen_selection: dict) -> tuple[dict, list[dict]]:
    """Select facts before wording; return card and its compact evidence records."""
    all_bundles = [primary, *examples]
    files = {b["manifest"]["evidence"]["sha256"]: b["input"] for b in all_bundles}
    if len({b["binding"]["manifest_sha256"] for b in all_bundles}) != len(all_bundles):
        raise ValueError("duplicate source in primary/examples")
    summaries = [s for b in all_bundles for s in b["summaries"]]
    by_id = {s["episode_id"]: s for s in summaries}
    if len(by_id) != len(summaries):
        raise ValueError("duplicate episode across sources")
    required = set()
    for selected in frozen_selection["episodes"]:
        source_hash = frozen_selection["source_manifest_hashes"][selected["source_id"]]
        matches = [s for s in summaries if s["source_manifest_sha256"] == source_hash
                   and (s["run_index"], s["env_index"]) == (selected["run_index"], selected["env_index"])]
        if len(matches) != 1:
            raise ValueError("frozen selected example missing or ambiguous")
        required.add(matches[0]["episode_id"])
    cohort = aggregate_cohort(primary["summaries"], primary["binding"])
    groups = cohort["groups"]
    evidence = [*summaries, *groups.values()]
    records = {r["record_id"]: r for r in evidence}
    claims = []

    def claim(template, section, scope, bindings, **literals):
        if scope in groups:
            scope_id = groups[scope]["record_id"]
        elif scope == "本卡范围":
            scope_id = cap["record_id"]
        else:
            scope_id = next(s["record_id"] for s in summaries if s["display_name"] == scope)
        c = make_claim(template, section, scope, bindings, literals, records, scope_id)
        claims.append(c)
        return c["claim_id"]

    def group_claim(template, group, key, section="总体行为"):
        g = groups[group]
        return claim(template, section, group, {"metric": (g["record_id"], ["metrics", key])},
                     **cohort_metric_literals(template, key, g["metrics"]))

    capability_text = {
        "measurement_scope": "仅启用P0 v2已验证量：根部进度/采样路径、归一化姿态、分足current marker与原始转换、低XY速度off、接触组合、记录力、相对位置、关节范围和任务终点；完整底账仍保留全部短事件和nearest配对。",
        "contact_definition": "contact marker表示所记录世界系净法向接触力模严格大于1N；记录间物理接触连续性未知，未触发不等于完全未接触，力模份额不等于重量或承载份额。",
        "clock_history": "控制时刻标记采集读取；current传感更新延迟和history逐槽时间/valid均未记录，history ANY与current分列，不补成物理子步或真实冲量。",
        "activity_definition": "低进度确认使用完整至少0.5s窗口内最大新增进度<0.02m；初始不足窗为unknown。足速度组合沿用XY阈值0.05m/s，描述同条读取记录中的合取，不命名迈步、拒绝落脚或整体静止。",
        "coverage": "默认回合等权，池化数值另用采样格时长代理加权；各量独立分母。格时长N×dt与观测跨度(N−1)×dt分列，左右截尾不进入完整区间时长分布；unknown和缺步不是阴性。",
        "comparison_scope": "先描述全队列，再展示结果、stratum和精确cell分面；不同终点时长/成功组成会影响整回合量。同cell比较只表示同公开条件格，未逐实例匹配实际随机化和命令；不是显著性、改善或因果结论。",
        "parent": "父代比较不可用：没有显式绑定的父代档案。另源开发反例不充当父代，也不进入主队列分母。",
        "missing": "缺少实际动作、力矩/饱和、逐步奖励及门控/air-time、实际质量/全身动量、切向力和接触对象、全部非足端接触力；精确奖励响应和物理对账不可用。已记录非足端力范围仅为绑定的base。",
        "deferred": "周期资格、有效/名义限位绑定、逐类地形几何资格、历史link位置与COM速度参考点映射尚未启用，保留后续验证义务；默认站姿和高位异常阈值未冻结，不生成举腿或拖地标签。",
        "delivery": "当前为纯数值离线开发卡片；媒体未检查、模型调用为零；最终作者公开schema/lexicon/comparison迁移及serialize_prompt_payload投递验收留到P1c。未验证新策略/新任务/新形态或闭环收益。",
        "selection": "展开例保留冻结E1样本、各终局净进度中位数邻近例、最大成功回撤返回极值例，并为每种长低速off保留成功见证及可用的同cell失败见证；稳定episode ID破同分。成功见证不代表典型或模式因果，其它事实与原短事件可按引用展开。",
    }
    if not frozen_selection["episodes"]:
        capability_text["selection"] = capability_text["selection"].replace("展开例保留冻结E1样本、", "展开例仅从本次档案选择")
    history_missing = primary["source"]["contact_force_history"].get("status") == "not_recorded"
    if history_missing:
        capability_text["clock_history"] = ("本源v1仅采current读数，history未采集；history有效分母为0、全部unknown，"
            "不补槽、不推定history事件。current更新延迟未知；manager终止独立保留，不等同current或history力阈值。")
    if not groups["success"]["episode_count"]:
        capability_text["no_success"] = "本队列没有成功回合，成功参考不可用；仍报告全部现有行为，不补零或借用其它来源。"
    cap = {"record_type": "capabilities", "record_id": stable_id("capabilities", CARD_VERSION, primary["binding"], capability_text),
           "text": capability_text, "source_inventory": primary["source"]["input_inventory"],
           "parent_comparison": cohort["parent_comparison"], "author_projection": "not_implemented_P1c"}
    evidence.append(cap)
    records[cap["record_id"]] = cap
    for key, text in capability_text.items():
        claim("definition", "定义、范围与能力缺口", "本卡范围", {"text": (cap["record_id"], ["text", key])})
    for group in ("all", "success", "failure"):
        g = groups[group]
        claim("cohort_scope", "结果与队列", group,
              {"episodes": (g["record_id"], ["episode_count"]), "rows": (g["record_id"], ["recorded_rows"]),
               "outcomes": (g["record_id"], ["outcome_counts"])})
        if not g["episode_count"]:
            continue
        for key in ("net_progress_m", "complete_path_xy_m", "terminal_time_s"):
            group_claim("scalar_distribution", group, key)
        for entity in primary["binding"]["entities"]["support"]:
            for suffix in ("marker", "norm_share", "signed_fz_share"):
                group_claim("ratio", group, entity + "." + suffix)
            group_claim("presence", group, entity + ".low_speed_off.long_observed", section="持续状态与失败过程")
            if group == "all":
                group_claim("off_runs", group, entity + ".marker")
        for code in (0, 11):
            group_claim("ratio", group, "combination." + str(code))
        for key in ("low_progress_any_foot_moving", "low_progress_all_feet_low_speed"):
            group_claim("presence", group, key, section="持续状态与失败过程")
            group_claim("ratio", group, key, section="持续状态与失败过程")
    for name, g in groups.items():
        if name.startswith("stratum_id:") and "/" not in name:
            claim("cohort_scope", "条件分面", name,
                  {"episodes": (g["record_id"], ["episode_count"]), "rows": (g["record_id"], ["recorded_rows"]),
                   "outcomes": (g["record_id"], ["outcome_counts"])})
            for key in ("low_progress_any_foot_moving", "low_progress_all_feet_low_speed"):
                group_claim("presence", name, key, section="条件分面")
        if name.startswith("cell_id:") and "/" not in name:
            success, failure = groups[name + "/success"], groups[name + "/failure"]
            if success["episode_count"] and failure["episode_count"]:
                claim("cell_comparison", "同cell成功对照", name,
                      {"success": (success["record_id"], []), "failure": (failure["record_id"], [])}, cell=name.split(":", 1)[1])
    selected = []
    for bundle in all_bundles:
        ids = {s["episode_id"] for s in bundle["summaries"]}
        selection = select_examples(bundle["summaries"], required & ids) if bundle is primary else [
            {"episode_id": eid, "reasons": ["separate_source_frozen_counterexample"]} for eid in sorted(required & ids)]
        selected.extend(selection)
    example_claims = {}
    for selected_item in selected:
        summary = by_id[selected_item["episode_id"]]
        section = "展开例：" + summary["display_name"]
        scope = summary["display_name"]
        start = len(claims)
        header = read_link(summary["references"]["header"], files)
        records[summary["references"]["header"]["record_id"]] = header
        measure_records = {}
        for key, ref in summary["references"].items():
            if key == "header":
                continue
            if ref.get("local"):
                record = summary["activity"]
            else:
                record = read_link(ref, files)
                evidence.append({"record_type": "evidence_link", **ref})
            records[ref["record_id"]] = record
            measure_records[key] = record
        records[summary["activity"]["record_id"]] = summary["activity"]
        evidence.append({"record_type": "evidence_link", **summary["references"]["header"]})

        def mclaim(template, key, **literals):
            return claim(template, section, scope, {"value": (measure_records[key]["record_id"], ["value"])}, **literals)

        claim("episode_scope", section, scope, {"coverage": (summary["record_id"], ["coverage"]),
              "task": (summary["record_id"], ["task"]), "name": (summary["record_id"], ["display_name"])})
        mclaim("episode_root", "root_progress")
        mclaim("episode_attitude", "body_up")
        for entity in header["entities"]["support"]:
            mclaim("episode_marker", entity + ".marker", entity=entity)
            mclaim("episode_low_speed", entity + ".low_speed_off", entity=entity)
            mclaim("episode_relative", entity + ".relative_position", entity=entity)
            mclaim("episode_history", entity + ".history_diagnostic", entity=entity)
        for template, key in (("episode_force", "force_summary"), ("episode_combinations", "contact_combinations")):
            claim(template, section, scope, {"value": (measure_records[key]["record_id"], ["value"]),
                  "entities": (measure_records[key]["record_id"], ["entity_ids"])})
        claim("episode_terminal", section, scope, {"value": (measure_records["task_endpoint"]["record_id"], ["value"]),
              "term_order": (summary["references"]["header"]["record_id"], ["task", "termination_term_order"])})
        for entity in header["entities"]["forbidden"]:
            claim("episode_base_runs", section, scope, {
                "current": (measure_records[entity + ".current_marker"]["record_id"], ["value"]),
                "history": (measure_records[entity + ".history_diagnostic"]["record_id"], ["value"])}, entity=entity)
        mclaim("episode_activity", "low_progress_activity")
        first_negative = summary["highlights"].get("first_negative")
        for entity in header["entities"]["support"]:
            longest = summary["highlights"].get(entity + ".low_speed_off")
            if first_negative is not None and longest and first_negative < longest["interval"]["steps_half_open"][0]:
                claim("inverted_before", section, scope, {
                    "negative_step": (summary["record_id"], ["highlights", "first_negative"]),
                    "start_step": (measure_records[entity + ".low_speed_off"]["record_id"], ["value", "longest", "interval", "steps_half_open", 0])}, entity=entity)
        d = summary["highlights"].get("drawdown")
        if summary["task"]["outcome"]["success"] and d and (d["max_drawdown_m"] or 0) > 0 and d["return_step"] is not None:
            claim("successful_return", section, scope, {"drawdown": (measure_records["root_progress"]["record_id"], ["value", "drawdown"]),
                  "success": (summary["record_id"], ["task", "outcome", "success"])})
        example_claims[summary["episode_id"]] = [c["claim_id"] for c in claims[start:]]
    if not frozen_selection["episodes"]:
        # The production profile preserves the six conditional summary slots.
        # Missing selected evidence is explicit; never import another archive's
        # witness or claim that absence in this view proves physical absence.
        selected_claims = [c for c in claims if c["claim_id"] in select_policy_summary(claims, records)]
        slots = {"all_mode": ("presence", "all", "全队列长低速off模式"),
                 "success_mode": ("presence", "success", "成功回合长低速off见证"),
                 "failure_mode": ("presence", "failure", "该成功模式的失败对照"),
                 "successful_return": ("successful_return", None, "成功回撤后返回见证"),
                 "inverted_before": ("inverted_before", None, "先翻转后低速off反例"),
                 "short_history": ("episode_base_runs", None, "current未触发而history触发的短事件见证")}
        if history_missing:
            slots["short_history"] = ("episode_terminal", None, "history未采集；独立manager终点见证")
        missing = {"unavailable_" + key: label + "：本次卡片未选出符合目录条件的证据；该槽不可提供事实，不代表物理上未发生。"
                   for key, (template, scope, label) in slots.items()
                   if not any(c["template_id"] == template and (scope is None or c["scope"] == scope) for c in selected_claims)}
        if missing:
            unavailable = {"record_type": "capabilities", "record_id": stable_id("capabilities", CARD_VERSION, primary["binding"], missing),
                           "text": missing}
            evidence.append(unavailable)
            records[unavailable["record_id"]] = unavailable
            for key in missing:
                claims.append(make_claim("definition", "定义、范围与能力缺口", "本卡范围",
                    {"text": (unavailable["record_id"], ["text", key])}, {}, records, unavailable["record_id"]))
    # All episode summaries contain measurement links; the reading selection does
    # not delete anything from the underlying event ledger.
    selected_ids = {e["episode_id"] for e in selected}
    omissions = [{"episode_summary_id": s["record_id"], "reason": "not selected for reading; full measurement links and event byte span retained"}
                 for s in summaries if s["episode_id"] not in selected_ids]
    card = {"schema_version": CARD_VERSION, "catalog_version": CATALOG_VERSION,
            "template_version": TEMPLATE_VERSION, "selection_version": SELECTION_VERSION,
            "cohort_binding": primary["binding"], "claims": claims, "selected_examples": selected,
            "policy_summary_claim_ids": select_policy_summary(claims, records),
            "reading_layers": "small policy view plus expandable complete audit; neither is the final author payload",
            "example_claims": example_claims, "omissions": omissions,
            "detail_omissions": {"joint_ranges": "numeric per-joint aggregates and P1a measure refs retained; not ranked into reading view",
                                 "nearest_onset_pairs": "all original directional nearest search references retained in P1a; no periodic/gait label",
                                 "raw_quaternion_diagnostics": "raw-vs-normalized diagnostics retained in P1a; all stated orientation uses normalized wxyz"},
            "author_projection": {"status": "not_implemented", "phase": "P1c", "actual_payload_checked": False},
            "review_status": "development_self_test_pending_independent_parent_acceptance",
            "scope": "primary cohort only; additional source examples never pooled"}
    validate_claims(claims, records)
    evidence.extend(claims)
    return card, evidence
