# R.0-b 工具与输入约定

`author_ceiling_probe.py` 复用旧 P3 的资格/上下文恢复、任务 adapter 和回执解析；实际调用仍由当前生产 `CandidateAuthoringLoop`、静态准入、CPU smoke、源码服务、journal 和预算守卫处理。新 P3-R 的输出上限与未知调用预留取获选 R.0-a 运行卡，历史 P3 工具及旧账不变。此脚本不包含人工诊断事实，不调用仿真或候选训练。

## 外部数据

`--data` 指向下述 JSON。路径相对该数据文件解析；文件 SHA-256 均为实际原文件 hash。四个 case 必须为 `stair-01`、`stair-03`、`push-02`、`push-04`，每个都有四个变体。旧 P3 案例文件、资格与 source refs 沿原路径读取并验 hash；服务器执行时可直接引用原封存目录。

```json
{
  "schema_version": "feedback_p3r.ceiling_inputs.v1",
  "p3_cases": {
    "path": "/data/yuyangliu/AGENT/feedback_p3_20260924/author_case_inputs_r5/case_inputs.json",
    "sha256": "填写实际文件hash"
  },
  "cases": [
    {
      "case_id": "stair-01",
      "variants": {
        "A": {"base": "A"},
        "T0": {"base": "T"},
        "Tc-num": {"base": "A", "behavior": {"path": "stair-01_Tc-num.json", "sha256": "实际hash"}},
        "Tc-vis": {"base": "A", "behavior": {"path": "stair-01_Tc-vis.json", "sha256": "实际hash"}}
      },
      "review_reference": {
        "registered_failure_shape": "只描述预先登记的可观察失败形态，不给机制答案",
        "source_refs": [],
        "unknowns": []
      }
    }
  ]
}
```

示例只展示一案，实际文件需填四案。`behavior` 接受带事实、限制及来源的普通字典，不要求套生产 behavior schema。它是开发参照，不伪装成正常生产投影。Stair 注入 `public_feedback_summary.behavior`，Push 注入 `public_feedback_summary.public_dev.behavior`；其他聚合、源码权限、父本、焦点和任务合同保持共同底座。Tc-vis 应在文件中自述哪些条目来自视觉，不能宣称已发送图像。

`review_reference` **仅进入盲评附件**，不进入作者输入；它必须先于作者输出冻结，不得包含奖励答案。文件自身不用 `variant`、`arm` 或 Tc 名称提示作者所处条件，具体来源和模态仍应明确。

Tc-vis 可使用本轮新录制、重新核对的开发 replay 观察。它们是 R.0 的新诊断参照，不能宣称这些文字在旧 P3 的 as-of 时点已经存在，也不回填或修改旧 P3 证据。旧 as-of 约束的是被恢复的共同历史底座；新增整理材料另按本轮文件和请求 hash 冻结。R.4 的正式对照仍须统一证据来源与可用时点后冻结，不能把本轮开发参照直接当作历史可用信息。

## 使用顺序

1. 准备四案、八份整理证据文件和实际 hash，先完成来源/数值核查。
2. `preflight --root <当前源码> --p3-tools <旧P3工具目录> --data <数据文件> --runtime-stage <获选R0a运行卡> --budget <阶段预算> --output <新的准备目录>`。无 API。生成 16 个真实首包，逐包要求反馈完整保留、共同焦点未丢失，检查剥离 behavior 后公平性，保存 system/user/request hash、完整生产源码快照、工具 hash、实际字节、task-section 容量回执及固定 32 次调度。相同案例第二次生成的变体顺序反转。输入预留与输出上限取该运行卡，而不取阶段总预算中初始设置的旧副本。即使当前费用分配不足，也可完成离线准备，明确返回 `INSUFFICIENT_NO_API`。
3. R.0-a PASS 后，将其实际 `summary.json` 的路径/hash 写入新卡的 `runtime_summary_ref`。根代理审核完整卡后设为 `FROZEN_API_RELEASED`。没有该门、实际资源或同底座冻结，不进入运行。
4. `run --root ... --p3-tools ... --stage <准备目录/stage.json> --output <运行目录>`。最多 32 个 episode、每个最多 5 次 provider attempt、2 次源码请求；源码请求、修复与失败均占原机会。运行前要求当前分配足够覆盖全部 160 次最坏请求：40,000 输入、32,768 输出时为 US$8.211456；若获选输出改为 65,536，则为 US$14.502912，超过当前 US$12 分配，必须由根代理重新登记项目内分配并重新冻结后才能运行。本工具不会自行加额度或减机会。未知尝试依生产日志停止审查，不盲重发。
5. `inspect --p3-tools ... --stage ... --output <运行目录>` 从原 journal 重新计算峰价/预留；已知 usage 保留完整实际值与已知缓存拆分，未知 usage 按本次运行卡完整输入及输出上限预留。报告标记 `p3r_selected_runtime_cap.v1`，与历史 P3 旧默认预留明确区分。`export --root ... --p3-tools ... --stage ... --results <运行目录> --output <新的盲评目录>` 只在 32 个终态齐全后执行，包含失败和无候选，不只导出成功回答。

为让评阅与后续作者运行并行，导出另支持 `--case-id push-04`（也可选另三案），要求该案的全部 8 个终态已经齐全，失败照常保留。编号始终先按冻结的完整 32-slot schedule、原种子打乱后分配，再取案例子集；之后完整导出时，每份 proposal 的编号与内容一致。输出明确标注 `CASE_8_ONLY_FULL_32_NOT_EXPORTED`，不作全局结论。只将当前案例的评阅参照交给评阅者；映射仍在 reviewer 目录之外。评分规则和其余案例输入保持冻结，不根据先完成的评阅改变后续实验。

本轮远端运行仍使用原已冻结脚本；新增导出功能的文件须上传为独立 `author_ceiling_export_r2.py` 后只执行 `export`，不得覆盖运行副本或重写它的 hash。示例：`python author_ceiling_export_r2.py export --root <源码目录> --p3-tools <P3工具> --stage <冻结stage.json> --results <运行目录> --case-id push-04 --output <新的案例盲评目录>`。不传 `--case-id` 才导出完整 32 份。离线子集/完整一致性验证见 [验收记录](results/ceiling_case_export_offline_check.json)。

运行入口会重新核源码/工具快照、所有输入 hash 和实际首包 request hash。新的公共修复提示统一作用四变体，旧 P3 提案不参与对照。若开发期间源码继续修改，旧 preflight 不能直接调用，须在新目录冻结当前最终底座。

## 盲导出与机器证据边界

每份材料包含原候选源码、初次与最终实际输入、父源码、已有完整数值档案、机器附件和待填评分表。变体映射只在同级 `withhold_mapping.json`；评阅者仅使用 `reviewer` 目录。按固定种子打乱，显式身份扫描照实报告，输入格式可让模型猜到条件这一局限保留。

评阅为本轮用户明确选择的 `model_review`，只填机制对齐（是/否/不可判）、行动相关性 0–2，以及可复核的数字/代码错误记录。不自动写语义评分。机器附件的数值/标识位置匹配不是事实正确性；奖励算术只覆盖真实保存的 CPU 样本与可解析规则。没有核完的条目不能被空列表伪装成“零错误”。完整初稿和末稿分别可查，旧 rationale 不是观测真值。

这套工具产生开发瓶颈证据，不产生 P3-R 价值 PASS、真人评阅或训练收益结论。

## 完成后的实际请求验收

`audit_ceiling_requests.py --root <冻结生产源码> --stage <冻结stage.json> --results <ceiling_run> --output <新的审计JSON>` 只读原 journal、所有 attempt 的 prompt ref、用量/响应回执与终态。它逐项核对初稿、修复、源码续答的完整反馈和共同焦点，重算已保存 payload 的 task-section 容量回执及 provider 序列化，验证真实 5-attempt / 2-source 合同与用量。按 case、variant、frame、attempt 输出；失败始终留在分母，未知 usage 单列，不补 token 估算。

该审计不要求每个修复帧都有失败源码；任务可按既有公开边界排除它。重现序列化不是补造历史 HTTP 抓包，也不证明语义或训练效果。正式入口须在原服务器路径、冻结源码和完整 32 终态到齐后执行；本地有限核验只读取已有 r2 同 hash 镜像、已保存修复样本和历史丢反馈反例，不创建新 journal。见 [离线检查](results/actual_request_audit_offline_check.json)。
