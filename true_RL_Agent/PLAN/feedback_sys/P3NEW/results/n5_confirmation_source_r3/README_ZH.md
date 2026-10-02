# N5 R3 源码冻结与 CPU preflight

2026-09-30。状态：**SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE**。本目录仅完成源码、来源、候选命令与 CPU 验证；模型/API 调用、GPU 运行、正式采集均为 0。未创建最终 capture card，未放行正式评阅。

## 本轮权威入口

- `source_freeze_receipt.json`：完成回执；`source_invariants.json`：与 R2 的源码/命令不变量。
- 最终源：`source_final/`，485 个文件；清单 `source_manifest.final.json`，SHA-256 `5cb1b36c5ca2e7af40d62c51e7cf1fe9aae4790e080353b8947cc6e14e5902d5`。
- 远端最终源：`/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r3_final`。
- 候选命令：`commands.candidate.final.json`。其中 capture 参数仅冻结，尚未执行。preflight 与 capture 仅 `--mode`、`--output`、`--video-dir` 不同。
- 实际 CPU 证据：`downloaded_final/control/verification.json` 与同目录执行日志；任务产物在 `downloaded_final/preflight/`。原始归档 `verification_results.final.tar` SHA-256 为 `f4317c15a8bc6fbfad7163284da71f33200cf69b94b4abecb348c4acb6b32b9d`。

## 验证结果

84 项测试通过（27 项任务/采集契约测试，48 项 reference 测试，5 项语义输入测试，4 项事实/outcome 边界测试）。Stair 与 Push 实际 CPU preflight 均退出 0，checkpoint SHA 与 R2 相同；实际模块来源都位于最终冻结树，执行前后 485 个源文件 SHA 不变。

| 任务 | 新绑定 | 实际派生 runner seeds | 新 seed cohort hash |
|---|---|---|---|
| Stair | `n5_r3_20260930` / `public_dev_diagnostic_n5_r3_20260930` | `1162956430, 500595133, 1830177040` | `be0735de21d34e492edcdff218eb5ac3f690fd74c2f4eabcec921ba4506fd55c` |
| Push | `eval_base_seed=1700000000` | `1700010000, 1700010001, 1700010002` | `b0bf84088fb3e50c0a6bc95f5155bcd07e104a56cf9fccd33be661170bc7edeb` |

各任务新 runner seeds 与已采 R1/R2 均无交集；新 hash 未出现在核对到的 32 份 `public_dev_job.json` 承诺中（Stair 15、Push 17）。`history_cohorts.complete.json` 包含此前 R2 核过的 28 份与新增 R2 记录；这不是服务器所有实验的穷尽清单。同一 checkpoint/地形/任务套件的新 reset 轨迹不构成新地形泛化证据。

`source_delta.final.json` 给出相对 R2 的完整差异。运行时 `src/`、配置和采集脚本与 R2 字节相同；PLAN 差异包括 related_v2/blocks_v2、scope 句准确性修正、独立 SOL-a/SOL-b 身份与冻结名单绑定、混合评阅协议/合同及测试。既有媒体准备差异是 Push runner seed 按实际 `eval_base_seed` 派生的绑定修正；本轮未改变媒体选择规则。checkpoint、任务/地形 suite、媒体环境/双视角/分辨率/逐步采样参数和正式阈值均沿用 R2。

评阅名单 `N5_REVIEW_ASSIGNMENT_R3.json` 已收入冻结树，SHA-256 `b898502e9fa33caf415cdcaf58f7b862de1b68473a898d535a07660ccd1cd893`。它只冻结 40 单位的 A/B 身份：40 SOL-a；B 为 8 DeepSeek + 32 SOL-b。实际 R3 raw 成员及各任务首 episode/focus 路由须在新材料 packing 后核验。GPT-6.1-SOL 新载体校准仍 **PENDING**；R2 的旧 GPT6-SOL 校准不能认证新模型。

## 保留的失败与修复

初始 484 文件快照遗漏了测试列表中已有的 `tests/unit/test_confirmation_event_table.py`，首次 CPU 验证因此出现 1 个导入错误，23 项实际测试通过，尚未进入任务 preflight。初始 `source/`、清单和 `failed_cpu_initial/` 均保留。`complete_cpu_source.py` 仅补入这一现有测试，生成独立 `source_final/`；另 484 文件完全一致，见 `source_correction.json`。远端最终测试/preflight 一次通过；本地下载后解包遇 Windows 长路径限制，由 `finish_local_receipt.py` 仅恢复经 SHA 核验的本地解包，未重跑远端任务。最终完成状态以 `source_freeze_receipt.json` 为准；准备阶段 JSON 保留当时状态。

R2 原冻结树 477 文件仍全部匹配其旧清单，旧 results、旧 capture card、旧请求和原答未修改。后续须由新的执行卡明确放行；本目录的 CPU PASS 不代替正式采集、模型校准或语义验收。
