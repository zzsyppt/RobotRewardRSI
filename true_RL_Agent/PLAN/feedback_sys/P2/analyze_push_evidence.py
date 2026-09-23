"""One CPU analysis of hash-verified P2 Push archives and existing recomputations.

Input copies preserve remote manifest bytes; this reader resolves chunks through
the local copy inventory without editing or relabeling any original reference.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "results/push_evidence_sources"
CONDITIONS = ("planar_speed", "angular_speed", "roll", "pitch", "height")


def read_verified(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("source hash mismatch: " + str(path))
    return json.loads(data)


def describe(values):
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)} if values else {
        "count": 0, "min": None, "median": None, "max": None}


def summarize(episodes):
    windows = [w for e in episodes for w in e["recoveries"]]
    interruptions = [i for w in windows for i in w["dwell_interruptions"]]
    return {"episodes": len(episodes), "recovery_units": len(windows),
        "episodes_with_pre_latch_dwell_interruption": sum(any(w["dwell_interruptions"] for w in e["recoveries"]) for e in episodes),
        "interruption_count": len(interruptions),
        "interruption_condition_counts": dict(Counter(c for i in interruptions for c in i["failed_conditions"])),
        "episodes_with_interruption_condition": {c: sum(any(c in i["failed_conditions"] for w in e["recoveries"]
            for i in w["dwell_interruptions"]) for e in episodes) for c in CONDITIONS},
        "interruption_after_anchor_seconds": describe([i["after_anchor_s"] for i in interruptions]),
        "coverage_units": dict(Counter(w["coverage"] for w in windows)),
        "all_required_recoveries_latched": sum(e["all_required_recoveries_latched"] for e in episodes),
        "all_latched_but_final_unstable": sum(e["all_required_recoveries_latched"] and bool(e["final_failed_conditions"]) for e in episodes),
        "all_latched_then_any_unstable_step": sum(e["post_all_recovery_unstable_steps"] > 0 for e in episodes),
        "any_latched_but_final_unstable": sum(e["final_unstable_after_recovery"] for e in episodes),
        "zero_stable_step_recovery_units": sum(w["stable_window_steps"] == 0 for w in windows),
        "final_failed_condition_episodes": dict(Counter(c for e in episodes for c in e["final_failed_conditions"])),
        "strata": dict(Counter(e["stratum"] for e in episodes))}


def analyze_case(report_path):
    report = json.loads(report_path.read_text())
    if report["status"] != "PASS" or report["episodes"] != 192:
        raise ValueError("pair was not accepted: " + str(report_path))
    prefix = report_path.stem.removesuffix("_verification")
    episodes, sources, definition = [], [], None
    for archive, derived in zip(report["archives"], report["numeric_recompute"], strict=True):
        seed = archive["run_index"]
        if seed != derived["run_index"]:
            raise ValueError("source seed mismatch")
        manifest_path = SOURCES / f"{prefix}_seed_{seed}_manifest.json"
        manifest = read_verified(manifest_path, archive["reference"]["manifest_sha256"])
        recompute_path = SOURCES / Path(derived["path"]).name
        recompute = read_verified(recompute_path, derived["sha256"])
        if not recompute["matches_runtime"] or recompute["identity"] != manifest["identity"]:
            raise ValueError("derived evidence identity mismatch")
        d = manifest["metadata"]
        current_definition = {key: d[key] for key in ("time", "parameters", "condition_fields", "pose_fields")}
        current_definition["force"] = d.get("force")
        # Mass is measured per environment; do not treat its values as protocol.
        if current_definition["force"]:
            current_definition["force"] = {k: v for k, v in current_definition["force"].items() if k != "total_mass_kg"}
        if definition is not None and definition != current_definition:
            raise ValueError("seed definitions differ")
        definition = current_definition
        arrays = {key: [] for key in ("valid", "control_step", "conditions", "pose")}
        for chunk in manifest["chunks"]:
            path = SOURCES / f"{prefix}_seed_{seed}" / chunk["file"]
            data = path.read_bytes()
            if len(data) != chunk["stored_bytes"] or hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                raise ValueError("raw chunk binding mismatch")
            with np.load(path, allow_pickle=False) as values:
                for key in arrays:
                    value = values[key]
                    if value.dtype.str != chunk["arrays"][key]["dtype"] or list(value.shape) != chunk["arrays"][key]["shape"]:
                        raise ValueError("raw chunk dtype/shape mismatch")
                    arrays[key].append(value.copy())
        arrays = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
        for original in recompute["per_episode"]:
            e = dict(original, seed=seed)
            env = e["env_id"]
            mask = arrays["valid"][:, env]
            steps, conditions = arrays["control_step"][mask, env], arrays["conditions"][mask, env]
            if len(steps) != e["valid_steps"] or int(steps[-1]) != e["snapshot_metrics"]["episode_steps"]:
                raise ValueError("episode sample denominator differs")
            e["final_failed_conditions"] = [name for name, passed in zip(CONDITIONS, conditions[-1]) if not passed]
            e["final_pose"] = dict(zip(d["pose_fields"], map(float, arrays["pose"][mask, env][-1])))
            if bool(e["final_failed_conditions"]) == e["snapshot_metrics"]["final_recovery_condition"]:
                raise ValueError("final stability differs from recomputation")
            latched = [w["recovery_control_step"] for w in e["recoveries"]]
            e["all_required_recoveries_latched"] = bool(latched) and all(s is not None for s in latched)
            later_unstable = np.zeros(len(steps), dtype=bool)
            if e["all_required_recoveries_latched"]:
                later_unstable = (steps > max(latched)) & ~conditions.all(axis=1)
            e["post_all_recovery_unstable_steps"] = int(later_unstable.sum())
            e["first_unstable_after_all_recovery"] = None
            if later_unstable.any():
                position = int(np.flatnonzero(later_unstable)[0])
                e["first_unstable_after_all_recovery"] = {"control_step": int(steps[position]),
                    "seconds_after_last_recovery": (int(steps[position]) - max(latched)) * d["time"]["step_dt_s"],
                    "failed_conditions": [name for name, passed in zip(CONDITIONS, conditions[position]) if not passed]}
            for w in e["recoveries"]:
                if w["anchor_control_step"] is None or not w["anchor_verified"]:
                    continue
                anchor = w["anchor_control_step"]
                in_window = (steps > anchor) & (steps <= anchor + d["time"]["recovery_window_steps"])
                if int(in_window.sum()) != w["observed_window_steps"]:
                    raise ValueError("window sample denominator differs")
                w["condition_failure_spans"] = {}
                for column, name in enumerate(CONDITIONS):
                    selected = steps[in_window & ~conditions[:, column]].astype(int).tolist()
                    if len(selected) != w["failed_condition_steps"][name]:
                        raise ValueError("window condition count differs")
                    spans = []
                    for step in selected:
                        if spans and step == spans[-1][1] + 1:
                            spans[-1][1] = step
                        else:
                            spans.append([step, step])
                    w["condition_failure_spans"][name] = [{"start_step": start, "end_step_inclusive": end,
                        "start_after_anchor_s": (start - anchor) * d["time"]["step_dt_s"],
                        "end_after_anchor_s": (end - anchor) * d["time"]["step_dt_s"], "observed_steps": end - start + 1}
                        for start, end in spans]
            episodes.append(e)
        sources.append({"run_index": seed, "remote_manifest_reference": archive["reference"],
            "local_manifest": str(manifest_path.relative_to(HERE)), "remote_recompute": derived,
            "local_recompute": str(recompute_path.relative_to(HERE)), "chunks_verified": len(manifest["chunks"])})
    counts = dict(Counter(e["terminal_label"] for e in episodes))
    if counts != report["failure_counts"]:
        raise ValueError("episode outcome counts differ from pair report")
    groups = {label: summarize([e for e in episodes if e["terminal_label"] == label])
              for label in ("not_recovered", "success_recovered")}
    return {"cell": report["cell"], "pair_report": str(report_path.relative_to(HERE)),
        "pair_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "numeric_reducer_sha256": report["numeric_reducer_sha256"], "sources": sources, "definition": definition,
        "episodes": len(episodes), "outcome_counts": counts, "groups": groups,
        "by_stratum": {stratum: {label: summarize([e for e in episodes if e["stratum"] == stratum and e["terminal_label"] == label])
            for label in ("not_recovered", "success_recovered")} for stratum in ("low", "medium", "high", "repeated")},
        "short_terminal_episodes": sum(e["valid_steps"] < definition["time"]["max_episode_steps"] for e in episodes),
        "episode_coverage": dict(Counter(e["coverage"] for e in episodes)),
        "all_recovery_unit_coverage": dict(Counter(w["coverage"] for e in episodes for w in e["recoveries"])),
        "per_episode": episodes}


def write_markdown(result):
    cases = result["cases"]
    names = {"push_g1_v2": "G1 velocity v2", "push_g1_v3": "G1 force v3",
             "push_go2_v2": "Go2 velocity v2", "push_go2_v3": "Go2 force v3"}
    lines = ["# P2 Push 真实过程证据：首问题核对", "", "2026-09-23；这是固定参考权重的 public-dev 过程分析，不是新训练或新策略资格结论。",
        "", "**目前已能把首问题拆成可核对的三种观测：正 dwell 被打断、窗口内从未形成正 dwell、已达标后终局再次不稳定。成功组也会出现子条件短暂不达标或恢复后越界；因此不能把一个诊断计数直接写成根因。**",
        "", "## 证据与分母", "", "读取 4 份配对验收报告、12 份独立 CPU 复算、12 份原 manifest 和 42 个 NPZ 块。逐项核对保存的 SHA-256、数组 dtype/shape、每 episode 有效步数、窗口步数、子条件计数与终局稳定状态；分析脚本执行返回 PASS。原件的 manifest 路径和字节没有改写。",
        "", "每个 cell 是 3 seeds × 64 个首 episode。这里仅计 capture-on 的 192 条；off 是同场景配对不变性对照，不能再算一批独立案例。不同协议也不是独立新策略。每个强度层各 48 条；JSON 保留按强度分层的结果，下面总体计数不作因果或显著性检验。",
        "", "| cell | success | not_recovered | 其他终局失败 | 早于时限终止 |",
        "|---|---:|---:|---:|---:|"]
    for key, c in cases.items():
        n, s = c["groups"]["not_recovered"]["episodes"], c["groups"]["success_recovered"]["episodes"]
        lines.append(f"| {names[key]} | {s}/192 | {n}/192 | {192-n-s}/192 | {c['short_terminal_episodes']}/192 |")
    lines += ["", "所有 768 条首 episode 均有完整终止帧；没有 rollout 数据截断。早终止是实际任务失败，不是采集丢失。Go2 velocity v2 这批没有真实短回合，因此不能声称该 cell 的真实短回合覆盖已经得到本批案例验证。",
        "", "补充实现闭合证据：G1 velocity 与 Go2 velocity 都使用同一 `PushRuntimeState`，选择状态类只由 force 协议开关决定，不按机器人分支（`runtime.py:127–134`）。两形态共用 overlay 中的 reset/termination 回调；采集数组按环境轴保存 root 姿态和事件，不依赖 12/37 维关节轴。共享 `_observe_pose` 生成 `active=~closed`；velocity 与 force 都先记录这个有效掩码，再将终局写入 closed，reset 均过滤已闭合 episode（`runtime.py:215–223,308,337–340`；`force_runtime.py:70–79,175–177`）。force 覆盖了自己的 observe，尾部闭合公式与 velocity 相同，并非完全共用同一个 observe 方法。",
        "", "真实 G1 velocity 的 35 个短回合覆盖 velocity 的终止帧、闭合及档案路径；真实 Go2 force 的 50 个短回合覆盖 Go2 环境与共用基础掩码；真实 Go2 velocity 的 192 条均验证了时限闭合和持久复算。CPU velocity 提前坍塌后 reset 保持原 episode、force 持久档案排除闭合后 padding、velocity 持久步序断档拒绝三个既有测试本次重跑通过（`test_push_physical_protocol.py:177`、`test_push_numeric_capture.py:103,179`）。相关四个生产文件 SHA 均与实际 `source_snapshot_r2.json` 一致。",
        "", "这些是跨 profile 的互补路径证据；未发现 Go2 velocity 另有独立的 valid/closed/reset 路径漏验，不为凑终局标签另训或制造终止。声明仍分别保留“每 profile 的真实完整 cohort 已验”和“跨 profile 的早终止路径已覆盖”，不写成四个 cell 各自都观测到所有终局类型。",
        "", "## 恢复计时不能混用", "", "五条件是平面速度、角速度范数、绝对 roll、绝对 pitch、高度同时达标。控制步长均为 0.02 s；下面用实际量化步数，不能拿连续协议秒数直接取整。",
        "", "| cell | 扰动起始步 | 恢复计时锚点 | eligible 窗口 | 连续 dwell |",
        "|---|---|---|---|---|"]
    for key, c in cases.items():
        d = c["definition"]; t = d["time"]; force = d["force"]
        starts = t["event_scheduled_steps"]
        anchor = ("各 event 注入步 " + "/".join(map(str, starts))) if not force else ("最后脉冲结束；单次/重复为 " + "/".join(str(s + force["pulse_steps"]) for s in starts))
        lines.append(f"| {names[key]} | {'/'.join(map(str, starts))} | {anchor} | {t['recovery_window_steps']} 步 / {t['recovery_window_steps']*t['step_dt_s']:.2f} s | {t['required_dwell_steps']} 步 / {t['required_dwell_steps']*t['step_dt_s']:.2f} s |")
    lines += ["", "窗口严格为 `0 < step - anchor <= window_steps`。velocity 每 event 独立恢复，重复扰动产生两个恢复单元；force 整个序列完整且验证通过后，以最后脉冲结束为唯一锚点，产生一个单元。因此四组恢复单元分母依次是 240、192、240、192，不能把 force 两次推力拆成两次恢复。",
        "", "G1 阈值：平面速度 ≤ 0.215546 m/s，角速度 ≤ 1.820491 rad/s，高度 ≥ 0.518 m；Go2 对应 ≤ 0.158473 m/s、≤ 2.476136 rad/s、≥ 0.28 m。两者 |roll|、|pitch| ≤ 0.2 rad。比较按原 float32 判据；这里的小数是便于阅读的显示值。",
        "", "## 哪些条件在多久后打断 dwell，成功组是否也有", "",
        "“中断”仅指首次恢复锁存前，已大于零的 dwell 遇到不稳定步而清零。起始就一直不稳定不会生成中断；恢复锁存后的越界也不属于这里的中断。以下时刻是相对各自锚点，计数按中断事件，条件可同时不通过而重叠。",
        "", "| cell / 终局组 | 有中断 episode / 组分母 | 中断数 | 条件计数：速度/角速度/roll/pitch/高度 | 相对锚点秒：min / median / max |",
        "|---|---:|---:|---|---|"]
    for key, c in cases.items():
        for label, g in c["groups"].items():
            times = g["interruption_after_anchor_seconds"]
            time_text = "—（没有中断）" if not times["count"] else " / ".join(f"{times[k]:.2f}" for k in ("min", "median", "max"))
            counts = "/".join(str(g["interruption_condition_counts"].get(k, 0)) for k in CONDITIONS)
            lines.append(f"| {names[key]} / {label} | {g['episodes_with_pre_latch_dwell_interruption']}/{g['episodes']} | {g['interruption_count']} | {counts} | {time_text} |")
    lines += ["", "G1 force 的同一 low 层内，not_recovered 中 7/11 有中断，成功中 7/29 也有；这已经给出负证据，不能说有中断就会失败。Go2 每种协议仅一个 not_recovered，不能据此估计普遍失败机制。",
        "", "具体可回查例子（seed 是 public-dev run_index，env_id 为 0 起始；均为 episode_index=1）：",
        "", "- **G1 force，seed 1/env 1，low，not_recovered：** 锚点 83；平面速度分别在锚点后 0.68、1.44、1.88 s 把已有 2、16、1 步 dwell 清零。整个 110 步窗口有 32 个稳定步，但最长连续计数仅 16，未达到 21。终局反而已稳定，仍因截止窗口内没有达成 dwell 而失败。",
        "- **同 seed 的 G1 force env 2，low，success：** 锚点后 0.76 s 也出现一次平面速度中断，清掉 1 步 dwell；之后在 step 143 达标，终局稳定。两例不同方向/状态，不是干预对照。",
        "- **Go2 velocity，seed 1/env 46，high，唯一 not_recovered：** 81 步窗口内没有任何五条件同时成立的步，max_dwell=0，没有“清零中断”。pitch 从锚点后 0.22 s 至 1.62 s 连续不达标（71 个采样步），平面速度也有 29 个不达标步。最终姿态稳定，不能把最终快照当作窗口内恢复成功。",
        "- **Go2 force，seed 1/env 57，repeated，唯一 not_recovered：** 序列锚点 84；同样 stable_window_steps=0。高度在锚点后 0.12–1.62 s 连续不达标（76/81 步）；平面速度 19 步、roll 10 步、角速度 3 步不达标，计数重叠。终局高度 0.279205 m 仍低于 0.28 m。",
        "", "这些连续区间从已校验的原始 `conditions` 和 `control_step` 数组提取，逐条件总步数与原独立复算完全一致。离散区间端点表示采样时刻；不要用末时刻减首时刻替代包含两端的样本数。",
        "", "## 已恢复后再次不稳定", "",
        "| cell | not_recovered 中所有必需恢复已达标但终局不稳 | success 中达标后曾再出现不稳定步 |",
        "|---|---:|---:|"]
    for key, c in cases.items():
        n, s = c["groups"]["not_recovered"], c["groups"]["success_recovered"]
        lines.append(f"| {names[key]} | {n['all_latched_but_final_unstable']}/{n['episodes']} | {s['all_latched_then_any_unstable_step']}/{s['episodes']} |")
    lines += ["", "G1 velocity 的 50 个 not_recovered 中，25 个所有必需 event 已恢复；另 25 个至少一个完整窗口未达 dwell。若只看“任何一次恢复后终局不稳”，会得到 28，额外 3 个是重复扰动只恢复了一部分，不能合并成全部恢复。G1 force 的 14 个恰分为 7 个已恢复后终局不稳、7 个窗口内未恢复。",
        "", "例如 G1 velocity seed 1/env 1：step 105 首次恢复，step 155（恢复后 1.00 s）第一次再次由平面速度越界，终局 step 412 的平面速度为 0.350363 m/s，超过 0.215546 m/s；其他四条件终局均达标。成功组也有恢复后越界，只是终局稳定。因此“曾再次不稳”和“终局不稳”必须分开。",
        "", "## 短回合与删失", "", "| cell | 已恢复单元 | 全窗口仍未达 dwell | 终局先于窗口结束 | 尚未观测到锚点 |",
        "|---|---:|---:|---:|---:|"]
    for key, c in cases.items():
        v = c["all_recovery_unit_coverage"]
        lines.append(f"| {names[key]} | " + " | ".join(str(v.get(k, 0)) for k in ("recovered", "full_window_without_dwell", "terminal_before_window_end", "anchor_not_observed")) + " |")
    lines += ["", "这些列以恢复单元为分母，不能当作终局标签计数；例如 Go2 force 有 2 个全窗口未恢复单元，但只有 1 个最终标签 not_recovered，另一条后来触发了优先级更高的跌落终局。",
        "", "- G1 force seed 1/env 32 在 step 80 侧倒，早于单次脉冲结束锚点 83；没有恢复窗口观测，不能记为观察了 110 步却 dwell=0。",
        "- G1 force seed 1/env 4 在 step 91 侧倒，只观察锚点后的 8 步；Go2 force seed 1/env 53 在 step 83 后向跌落，早于重复序列锚点 84。",
        "- G1 velocity seed 2/env 48 在 step 125 前倒，第二 event 计划 step 234 尚未注入，仍留在对应分母中。",
        "", "`recovery_censored=true` 在现有复算表示未观察到恢复事件，既可能是完整窗口内未恢复，也可能是提前终止/锚点未到。统计分析必须使用 coverage 细分，不能一律称为数据丢失或同一种右删失；本批没有 `rollout_truncated`。",
        "", "## 能支持与不能支持的结论", "",
        "证据已支持定位判据失败的时刻、对应子条件、已有 dwell 长度、恢复锁存与终局关系，并可在同 cell、同强度层中检索成功反例。它未识别动作/奖励修改的因果效果，没有脚接触或逐脚动作证据，不能推出滑步、支撑切换错误、接触冲量根因或应如何调 reward。force 注入完整性仍使用已有 physics-hook 测量，不是独立重新仿真。",
        "", "资源口径补充：原 Push 配对报告把 GNU time RSS 简写成 process-tree max，应按训练验收中的精确定义理解为进程高水位及 waited children 的代理观测，不是同时进程树 RSS 的总和。成对 RSS 差不能冒充隔离后的采集分配峰值；不为这处文字说明重跑 GPU。",
        "", "这是离线过程证据可用性的验证；尚未证明作者 LLM 实际读取了这些新数组/派生比较，更没有模型修订收益实验。",
        "", "复现：`python PLAN/feedback_sys/P2/analyze_push_evidence.py`。输入为 `results/push_evidence_sources/` 中按原 hash 保留的副本与四份 `results/push_*verification.json`；输出 [机器可核 JSON](PUSH_EVIDENCE_FINDINGS.json)，包括每个 episode 的来源、条件区间、恢复后首次越界、分层计数。此脚本不运行策略、不修改原 receipt。"]
    (HERE / "PUSH_EVIDENCE_FINDINGS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cases = {p.stem.removesuffix("_verification"): analyze_case(p)
             for p in sorted((HERE / "results").glob("push_*verification.json"))}
    if len(cases) != 4:
        raise ValueError("expected four Push on/off pairs")
    result = {"schema_version": "true_rl_agent.p2.push_evidence_findings.v1",
        "scope": "fixed reference checkpoints, public_dev, 3 seeds x 64 first episodes per cell; capture-on only, not doubled by paired off",
        "source_kind": "actual source_r2 Isaac evaluation plus hash-bound CPU recomputation; no policy execution in this analysis",
        "interruption_definition": "before first recovery latch, an eligible unstable step resets an already positive dwell counter",
        "condition_counts_scope": "all observed eligible window steps, including after latch; not the same as interruption counts",
        "cases": cases}
    target = HERE / "PUSH_EVIDENCE_FINDINGS.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(result)
    print(json.dumps({"status": "PASS", "cases": len(cases), "episodes": sum(c["episodes"] for c in cases.values()), "output": str(target)}))


if __name__ == "__main__":
    main()
