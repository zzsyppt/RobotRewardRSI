"""Export human-review sheets without arm labels; never auto-fill judgments."""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from author_experiment import bound_read, digest, read, write


RUBRIC = {
    "fact_support": "For every factual claim: identify evidence actually visible in seen-*.json, then mark supported / contradicted / unavailable against full truth. A lucky assertion absent from the author's evidence is not grounded. Absence is not zero.",
    "hard_error": "Invented fact, wrong direction/unit/denominator, treating correlation as proof, or forbidden control/input.",
    "action_relevance": "0 no actionable hypothesis; 1 plausible but weakly tied; 2 explicit process evidence supports a bounded change.",
    "testability": "0 no falsifiable prediction; 1 vague measure; 2 named observable, direction, window, and failure condition.",
    "authorization": "Does the actual candidate use only current allowed reward inputs and the permitted component change?",
    "restraint": "Distinguishes observed evidence, proposed explanation and untested intervention; allows unknown.",
    "not_scored": "No training is run: do not grade whether the proposal really improves RL performance."
}

REVIEW_README_ZH = """# P3 匿名作者建议评阅

这份材料评阅证据支持和修改建议质量，不评价尚未运行的训练收益。12 个焦点问题共享 2 个策略、2 份底层数据；同任务的焦点不是独立实验。匿名编号中有预先安排的重复评阅项，请分别评分。

1. 先打开 [材料索引](INDEX_ZH.md)，再按 `case_materials.json` 的 case_id 找中性 focus、真实父程序 `reference-*.py` 和完整 public-dev 证据 `evidence-*.json`。先不看 A/T 汇总，也不要打开上级目录的 `withhold_from_reviewer.json`。
2. 阅读 `proposal-*.py`、实际末次作者输入 `seen-*.json` 和 `review_sheet.json` 对应 rationale。终态区分 candidate_verified、rejected_candidate、no_candidate；保留失败，不从样本中删除。authoring_checks 是真实静态/CPU fixture 检查，不是训练成功。源码只做匿名改名，建议内容未改写。
3. 核实事实时，Stair 在完整证据的 `episodes` 中按 episode_id、cell、stratum 定位，检查 events、anchor、before_terminal、window_present、ranges 和 coverage。Push 在 `transcripts` 中按逻辑 run/环境定位，按 definitions 的 anchor/window_rule 检查恢复单元与 coverage，并在 profile 核对分母。观察长度不足、窗口尚未到达、字段未记录，都不能当成零。
4. 在每个 proposal 的 claims 中逐项填写原文 claim、作者输入中的可见引用、full truth 的 supported / contradicted / unavailable，以及具体 JSON 字段/episode/窗口引用。猜测碰巧符合完整真值但作者输入无依据，不算有证据支持。分别记录测量事实、解释假说和预期改变；数值定位成功只证明有出处，不自动证明推理正确。不能由相关性直接判定因果。
5. 按 `rubric.json` 填 hard_errors、action_relevance、testability、authorization、restraint 和 reviewer_notes。可检验预测应写清观测量、方向、时间窗口与何种结果会否定假说；没有后续训练，不能填写“改进已命中”。无事实主张时支持率分母为零，应记不适用而非自动 100%。
6. 保存 `review_sheet.json` 的 reviewer 和填写时间，把 status 改为 COMPLETED，再交回汇总。一个评阅者的重复题只反映自身一致性，不是评阅者间一致性。数值图若另附，只帮助读测量，不能作为原生视频或视觉验收证据。

本次预先采用质量收益门；费用单独报告，不在看完输出后换成成本优势通过。盲评完成前，价值门始终待评。文本风格可能暴露证据表达方式，因此仅承诺隐藏臂名与映射，不承诺完全不可猜测。
"""


def rationale(source):
    from true_rl_agent.candidates.component_source_semantics import extract_training_components_rationale
    return extract_training_components_rationale(source)


IDENTITY_PATTERN = re.compile(r"\blane-[ab]\b|\bgeneration-\d+\b|\barm[\s_:=\-\"']+[AT]\b")


def reference_locations(text, seen, truth):
    """Locate explicit episode/identifier citations; do not grade their meaning.

    Measurement searches are intentionally limited to task feedback. Matches
    in an earlier candidate's own rationale are not independent evidence.
    Bare numbers and natural-language claims remain a human-review task.
    """
    episode_ids = set(re.findall(r"\b(?:r\d+\.e\d+|s\d+\.e\d+)\b", text or ""))
    identifiers = set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|\b(?:sha256:)?[a-f0-9]{64}\b", text or ""))
    wanted = episode_ids | identifiers
    number_text = text or ""
    for identifier in wanted:
        number_text = number_text.replace(identifier, "")
    numbers = {token: float(token) for token in re.findall(
        r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])", number_text)}
    numeric_tokens = {}
    for token, number in numbers.items():
        numeric_tokens.setdefault(number, []).append(token)

    def locate(value):
        if not wanted and not numbers:
            return {}, {}
        hits = {token: [] for token in wanted}
        totals = {token: 0 for token in wanted}
        numeric_hits = {token: [] for token in numbers}
        numeric_totals = {token: 0 for token in numbers}

        def add(token, path, kind):
            if token in hits:
                totals[token] += 1
                if len(hits[token]) < 8:
                    hits[token].append({"path": path, "match": kind})

        def walk(item, path):
            if isinstance(item, dict):
                # Push has no literal episode_id: this is a documented alias
                # derived from the two identity fields, not an inferred event.
                if isinstance(item.get("source"), int) and isinstance(item.get("env_id"), int):
                    add(f"s{item['source']}.e{item['env_id']}", path, "source_env_alias")
                for key, child in item.items():
                    add(key, path + "/" + key, "key")
                    walk(child, path + "/" + key)
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    walk(child, path + "/" + str(index))
            elif isinstance(item, str):
                add(item, path, "exact_value")
            elif type(item) in (int, float):
                for token in numeric_tokens.get(item, ()):
                    numeric_totals[token] += 1
                    if len(numeric_hits[token]) < 8:
                        numeric_hits[token].append({"path": path, "value": item})
        walk(value, "")
        return ({key: {"locations": hits[key], "total_matches": totals[key]} for key in sorted(wanted)},
            {key: {"locations": numeric_hits[key], "total_matches": numeric_totals[key]} for key in sorted(numbers)})

    visible, visible_numbers = locate(seen.get("public_feedback_summary", {}))
    complete, complete_numbers = locate(truth)

    return {"scope": "Exact identifiers in rationale, within actual final task feedback versus complete task truth; at most 8 locations per identifier.",
        "semantic_support": "NOT_GRADED_HUMAN_REQUIRED", "numeric_claims": "NOT_GRADED_HUMAN_REQUIRED",
        "seen_feedback": visible, "full_truth": complete,
        "numeric_exact_locations": {"seen_feedback": visible_numbers, "full_truth": complete_numbers,
            "rule": "Exact numeric literal equality only; no rounding, percentages, units or denominators inferred. Proposed parameters are not classified as factual claims."},
        "limitations": "Natural-language paraphrases and parent-source claims are not parsed. Finding an ID/key/number does not prove the associated claim, direction, unit, denominator, causal inference, or authorization."}


def reward_checks(code, seen, authoring_checks):
    """Reuse the verifier's active component resolver and saved CPU outputs."""
    from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
    from true_rl_agent.candidates.component_source_semantics import (
        SingleComponentDeltaComparisonUnavailable, project_canonical_reward_term_sources,
        project_single_component_operation_delta)
    anchor, _, _ = select_authoring_anchor_source(seen)
    if not code or not anchor:
        return {"status": "not_available", "reason": "no_candidate_or_visible_parent"}
    try:
        delta = project_single_component_operation_delta(candidate_source=code, anchor_source=anchor)
    except SingleComponentDeltaComparisonUnavailable as exc:
        return {"status": "not_available", "reason": str(exc)}
    result = {"status": "projected", "operation_delta": delta,
        "scope": "Declared weight/clip signs and finite CPU fixture arithmetic only; behavioral usefulness and rationale alignment require human review."}
    for side, source in (("before", anchor), ("after", code)):
        target = delta.get(side + "_component")
        terms = project_canonical_reward_term_sources(source)
        term = next((term for term in terms or () if target and term.mechanism_id == target["mechanism_id"]), None)
        result[side] = None if term is None else {"mechanism_id": term.mechanism_id, "callable": term.callable_name,
            "weight": term.weight, "weight_sign": "negative" if term.weight < 0 else "positive" if term.weight > 0 else "zero",
            "clip_min": term.clip_min, "clip_max": term.clip_max}
    after = result["after"]
    rows = []
    if after and authoring_checks:
        smoke = authoring_checks[-1].get("component_smoke") or {}
        observed = next((row for row in smoke.get("reward_results", ())
            if "RewardTerm:" + row["name"] == after["mechanism_id"]), {})
        for sample in observed.get("sample_case_outputs", ()):
            clipped = max(after["clip_min"] if after["clip_min"] is not None else -math.inf,
                min(after["clip_max"] if after["clip_max"] is not None else math.inf, sample["raw"]))
            expected = clipped * after["weight"]
            rows.append({**sample, "expected_weighted_after_clip": expected,
                "arithmetic_matches": math.isclose(sample["weighted_after_clip"], expected, rel_tol=1e-12, abs_tol=1e-12)})
    result["finite_fixture_sign_arithmetic"] = {"status": "PASS" if rows and all(r["arithmetic_matches"] for r in rows)
        else "FAIL" if rows else "not_available", "samples": rows}
    return result


def export(results_dir, output, *, cases_path, preflight_path, seed=2026092404):
    from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
    from true_rl_agent.memory import PromptPayloadRef
    from true_rl_agent.prompt_artifacts import verify_prompt_payload
    if output.exists():
        raise ValueError("blind export destination must be new")
    rows = [(path, read(path)) for path in sorted(results_dir.glob("*.json"))]
    rng = random.Random(seed)
    copies = list(rows)
    # Predetermined ~10% duplicates assess one reviewer's consistency only.
    copies += rng.sample(rows, min(len(rows), round(len(rows) * 0.1)))
    rng.shuffle(copies)
    mapping, sheets, identity_warnings = [], [], []
    review = output / "reviewer"
    card, preflight = read(cases_path), read(preflight_path)
    by_case = {c["case_id"]: c for c in card["cases"]}
    expected = {(case_id, arm, repetition) for case_id in by_case for arm in ("A", "T") for repetition in (1, 2)}
    observed = {(item["case_id"], item["arm"], item["repetition"]) for _, item in rows}
    if len(rows) != 48 or observed != expected:
        raise ValueError("the registered export requires all 48 final author episode records, including failures")
    review.mkdir(parents=True)
    materials = []
    copied_tasks, truth_by_task = set(), {}
    for case in card["cases"]:
        task = case["task"]
        truth = bound_read(case["blind_truth"], cases_path.parent)
        truth_by_task[task] = truth
        if task not in copied_tasks:
            write(review / f"evidence-{task}.json", truth)
            prompt_ref = next(row for row in preflight["prompts"] if row["case_id"] == case["case_id"] and row["arm"] == "A")
            payload = bound_read(prompt_ref, preflight_path.parent)
            source, checksum, role = select_authoring_anchor_source(payload)
            if source is None or role != "current_best":
                raise ValueError("review reference is not the actual authoring parent")
            (review / f"reference-{task}.py").write_text(source, encoding="utf-8")
            copied_tasks.add(task)
        materials.append({"case_id": case["case_id"], "task": task, "focus": case["common_focus"],
            "checkpoint_hash": case["checkpoint_hash"], "public_dev_truth": f"evidence-{task}.json",
            "parent_source": f"reference-{task}.py", "as_of_cutoff": case["as_of_cutoff"],
            "correlation": "six focused questions share this policy and complete evidence package"})
    location_rows, locations_by_result = [], {}
    for index, (path, item) in enumerate(copies, 1):
        blind_id = f"proposal-{index:03d}"
        candidate_path = item.get("candidate_private_path")
        seen = verify_prompt_payload(PromptPayloadRef.from_mapping(item["prompt_payload_ref"]))
        write(review / f"seen-{index:03d}.json", seen)
        code = None
        if candidate_path:
            source = Path(candidate_path) / "training_code/components.py"
            code = source.read_text(encoding="utf-8")
            (review / f"{blind_id}.py").write_text(code, encoding="utf-8")
        sheet = {"blind_id": blind_id, "case_id": item["case_id"],
            "source_file": f"{blind_id}.py" if code else None,
            "author_seen_prompt": f"seen-{index:03d}.json", "authoring_checks": item["authoring_checks"],
            "checks_scope": "static admission and CPU synthetic component fixture; not RL training or simulator success",
            "rationale": rationale(code) if code else None,
            "author_output_status": ("candidate_verified" if item["stop_reason"] == "candidate_verified" else
                "rejected_candidate" if code else "no_candidate"),
            "claims": [], "hard_errors": None, "action_relevance": None,
            "testability": None, "authorization": None, "restraint": None, "reviewer_notes": None}
        sheets.append(sheet)
        for artifact, content in (("proposal", code or ""), ("seen_prompt", json.dumps(seen, ensure_ascii=False)),
                                  ("review_sheet", json.dumps(sheet, ensure_ascii=False))):
            markers = sorted(set(IDENTITY_PATTERN.findall(content)))
            if markers:
                identity_warnings.append({"blind_id": blind_id, "artifact": artifact, "markers": markers,
                    "handling": "original content preserved; root review required before releasing blinded materials"})
        if path not in locations_by_result:
            locations_by_result[path] = reference_locations(sheet["rationale"], seen,
                truth_by_task[by_case[item["case_id"]]["task"]])
            locations_by_result[path]["reward_checks"] = reward_checks(code, seen, item["authoring_checks"])
        location_rows.append({"blind_id": blind_id, **locations_by_result[path]})
        mapping.append({"blind_id": blind_id, "result_path": str(path), "result_sha256": digest(path),
            "case_id": item["case_id"], "arm": item["arm"], "repetition": item["repetition"], "cluster_id": item["cluster_id"]})
    write(review / "rubric.json", RUBRIC)
    (review / "README_ZH.md").write_text(REVIEW_README_ZH, encoding="utf-8")
    write(review / "case_materials.json", materials)
    index_lines = ["# 匿名材料索引", "", "这是实际作者终态；静态准入通过不代表事实正确或训练有效。评分填入 review_sheet.json。", "",
        "| 编号 | 焦点 | 终态 | 作者源码 | 实际末次输入 |", "|---|---|---|---|---|"]
    for sheet in sheets:
        source_link = f"[源码]({sheet['source_file']})" if sheet["source_file"] else "—"
        index_lines.append(f"| {sheet['blind_id']} | {sheet['case_id']} | {sheet['author_output_status']} | {source_link} | [输入]({sheet['author_seen_prompt']}) |")
    index_lines += ["", "完整 public-dev 真值：[Stair](evidence-stair.json)、[Push](evidence-push.json)。",
        "", "共同父程序：[Stair](reference-stair.py)、[Push](reference-push.py)。",
        "", "[机器引用与奖励算术定位](machine_reference_locations.json)只辅助查找，语义质量仍待人工判定。"]
    (review / "INDEX_ZH.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    write(review / "review_sheet.json", {"reviewer": None, "status": "PENDING", "proposals": sheets})
    write(review / "machine_reference_locations.json", {"status": "LOCATIONS_ONLY_NOT_SEMANTIC_PASS",
        "human_value_gate": "PENDING_REVIEW", "proposals": location_rows})
    write(output / "withhold_from_reviewer.json", {"seed": seed, "mapping": mapping, "identity_warnings": identity_warnings,
        "blinding_limit": "Arm labels hidden; a reviewer may infer evidence style from candidate rationale. Single-reviewer duplicates are not inter-rater agreement."})
    return {"outputs": len(rows), "review_items": len(copies), "identity_warning_count": len(identity_warnings),
        "human_value_gate": "PENDING_REVIEW"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.root / "src"))
    print(json.dumps(export(a.results, a.output, cases_path=a.cases, preflight_path=a.preflight)))
