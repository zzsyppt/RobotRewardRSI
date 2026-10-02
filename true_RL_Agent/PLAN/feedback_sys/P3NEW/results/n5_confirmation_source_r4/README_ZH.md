# R4 最终源码与 CPU 预检

最终状态为 `SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE`。以 `source_freeze_receipt.json` 显式引用的 `source_manifest.verified.json`（492 文件）、`source.verified.tar.gz` 和 `downloaded_verified/control/verification.json` 为准，不能取名为 `.final` 的中间失败件。API、模型、GPU、训练及采集调用均为 0；本回执不放行采集或正式评阅。

最终清单 SHA256：`1b7cb40e5b1d1780011e1748a9e6adae88517dd880b74c8b0b5f23df66a1a645`。最终封包 SHA256：`1c8e3a1b9578cf2a44c80ac9f9aa35f3b344586a0005cf5c907e8364a2f1b0aa`。CPU verification SHA256：`2ab9654f06a8070301777e16c55aabaea0a5b87dc73b8039ee39aaa37e207ed7`。最终 492 文件在远端 CPU 执行前后完整核验一致；关键导入均来自新 source。

保留了真实失败过程：首次 491 文件封包缺 `tests/unit/test_body_projection.py`，导致新增 Stair 测试 import 失败，既有任务视觉 26 项通过。仅补该既有 fixture，生成 `.final` 的 492 文件清单/封包；新增 Stair 2 项通过，但 reference runner 15 项中 2 项因错误文案缺少 `model/carrier` 子串而失败（错误分配已正确拒绝）。root 仅修正文案及换行，随后仅续跑这 2 项、CLI 载体 3 项和两任务 preflight，全部通过。按唯一检查计 46 项：26 + 2 + 13 + 2 + 3，不叠加本机或重复通过数。原 491 / 492 清单、封包和失败日志未覆盖，补齐与文案差异分别见 `source_completion.json` / `source_wording_correction.json`。当前 source 比最初 491 目录新增一个 fixture，并改变一处 runner 文案，不能称整个初始目录不变。

Stair 实际 preflight：`n5_r4_20261001`；runner seeds 为 `[648978242, 13688135, 1557805454]`，seed cohort hash 为 `d41ac09594b4a4b4b48c2df075088328301a68d293bf9310882030574115ac32`，evaluation binding hash 为 `1db7a52e1cec0769caf34926267b41981c8aa066341ca26354962ee758b5905b`。

Push 实际 preflight：`eval_base_seed=1800000000`；runner seeds 为 `[1800010000, 1800010001, 1800010002]`，seed cohort hash 为 `13ada9059f86784a1bdfe2793c3c1bd29cc42f7bb58c59d99216145332ab570d`，evaluation binding hash 为 `99261b3e10d2cf707d241ca6d38bfc5a238cc4750d1fc45bcc67fe61e45edf02`。

两任务各一次 `--mode preflight` 均 exit 0；checkpoint 与 R3 相同。历史检查覆盖 36 个已记录 public-dev jobs（Stair 17、Push 19）及 R1/R2/R3 派生 runner seeds，新 cohort hash 和 seed 无重叠；这不是服务器所有实验的穷尽清单。未改任务/地形 suite、媒体规则或阈值。`commands.candidate.json` SHA256 为 `b6c3676c18c2eea26ec0eb5189ef4ec21f3a8294a2414367053e8eab66b8d57b`；其中 capture argv 未执行，与 preflight argv 只允许 mode/output/video-dir 差异。

远端 source 为 `/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r4_final`；最终控制回执位于 `n5_confirmation_source_r4_final_control/verified`，实际预检输出位于 `n5_confirmation_preflight_r4_final`。下一次采集机器卡必须显式引用本次最终 receipt，并另取最新资源及账本记录；本次未新建资源/账本凭据或采集卡。实际新 packing 后仍需核任务首项与角色表路由；40 单位、80 facts barrier、本人 seal 后 report 等原规则不变。
