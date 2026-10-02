# N4 四案例文本烟测：部署交接

远端独立仓库根目录：`/data/yuyangliu/AGENT/feedback_p3new_20260929/text_author_smoke_r1`。连接别名 `humanoid-server`，回执主机名 `deyugptserver`。Python `/data/yuyangliu/envs/isaaclab6_validation/bin/python`，实测版本 3.12.14。

最终离线检查为 `PASS_REMOTE_OFFLINE_PREPARATION`：391 个生产源码文件、59 个冻结输入均匹配；四个 HTTP 首包在远端重新构建，与本地冻结字节完全一致。真实 socket 连接在检查过程中被阻止，API / GPU / 训练均为 0。既有 `/data/yuyangliu/secrets/deepseek.env` 仅检查常规文件及可读性，未打开内容。

当前卡仍为 `PREPARED_NOT_RELEASED`：

- 相对路径：`PLAN/feedback_sys/P3NEW/text_author_smoke_r3/stage.draft.json`。
- SHA-256：`8bfe937e79781ab73c1ba21290c8bead92e5bf527b6832ed28114e5764e39824`。
- 源码快照 SHA-256：`a8d0791622f5383dd455e45010029faa49bf5c2639b7f65bb13d494bda050be2`。
- 最新回执：同目录 `remote_preparation_check_current_materials.json`。之前两份 remote_preparation_check 回执均保留，但它们使用旧 Push phase/full，不构成当前完整派生的一致性证明。

首次部署 448 个文件；源码单位字段增量为 6 文件，随后当前作者材料修正增量为 29 文件（材料、准备脚本、请求和旧字节备份，无生产源码改动）。`deployment_manifest.json`、`deployment_units_delta.json`、`deployment_material_delta.json` 顺次记录精确 old/new 哈希；当前源码身份以 stage 引用的快照为准。未复制密钥、NPZ chunks、checkpoint、视频或新增校准准备脚本。

当前 Push 输入来自 `text_author_smoke_r3/current_push_materials`：full SHA 为 `9299a937d63251679b2a2c00b38de18e7dd7b237b6c9cc09af0e668ef415de12`，phase SHA 为 `9af5fb6baecee53342a2d197b6fe71f427a506f485858d30341d1cbcbc8783dd`。当前生产投影函数、外置 full 引用与 task_evidence_hash 已在本地/远端同时核对。原 `delivery_offline_r1` 未覆盖。

Stair 两包不变；Push-02 新 HTTP SHA 为 `acf68aaf59790eb60e04cc9e84a51f115b5c961acf812cd8da0dc02c9c9a81bb`，Push-04 为 `66b96f46f1c688058b24e942978b036efc15e9f9eb912953cbec9f8b6f17e25b`。每个 Push payload 仅 full 派生摘要、diagnostic phase 摘要、任务区段摘要三处变化；数值/报告句子未变。旧 release 由主代理在任何作者调用前撤销，远端 `stage.released.json` 仍为 `REVOKED_BEFORE_ANY_AUTHOR_CALL`，本次部署没有修改它。

## 主代理放行字段

核对具体作者模型当前价格、别名映射与预期响应身份后，必须从上列最新 draft 重新生成同目录 `stage.released.json`；不得继续使用已撤销卡的旧 frozen_inputs。仅在新副本填写：

| 字段 | 要求 |
| --- | --- |
| `status` | `FROZEN_API_RELEASED` |
| `pricing_status` | `VERIFIED_FOR_AUTHOR_MODEL` |
| `budget.prices_usd_per_million` | 实际核实的 `cache_hit`、`cache_miss`、`output`，单位 US$/百万 token |
| `provider.verified_response_model` | 已核实的实际响应 `model` 字段 |
| `provider.verified_system_fingerprint` | 已核实的指纹；入口在逐 episode 终态及下一 episode 前检查 |

保存核价/映射来源及核对时间。既有价格仅历史预算快照，不能因旧数字小于 US$4 就自动放行。输入请求模型与 high / 65536 / 40k 均已冻结；若改请求模型或作者设置，须重新准备请求与哈希。此次不改这些字段。运行仍为 4 个 episode，每个最多 5 个实际请求，总最多 20 个、US$4，source followup 与修复也计入；不做 A/T 或训练门。

## 执行命令

在 `humanoid-server` 上进入上述独立仓库根目录。完成 released 卡复核后，逐条执行并检查每条终态和 usage，再开始下一条：

```sh
/data/yuyangliu/envs/isaaclab6_validation/bin/python PLAN/feedback_sys/P3NEW/prepare_text_author_smoke.py --stage PLAN/feedback_sys/P3NEW/text_author_smoke_r3/stage.released.json --run-case stair-01 --output PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1
/data/yuyangliu/envs/isaaclab6_validation/bin/python PLAN/feedback_sys/P3NEW/prepare_text_author_smoke.py --stage PLAN/feedback_sys/P3NEW/text_author_smoke_r3/stage.released.json --run-case stair-03 --output PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1
/data/yuyangliu/envs/isaaclab6_validation/bin/python PLAN/feedback_sys/P3NEW/prepare_text_author_smoke.py --stage PLAN/feedback_sys/P3NEW/text_author_smoke_r3/stage.released.json --run-case push-02 --output PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1
/data/yuyangliu/envs/isaaclab6_validation/bin/python PLAN/feedback_sys/P3NEW/prepare_text_author_smoke.py --stage PLAN/feedback_sys/P3NEW/text_author_smoke_r3/stage.released.json --run-case push-04 --output PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1
```

不要对已终止案例换目录重跑。四条命令共用唯一日志目录，沿用现有 `CandidateAuthoringLoop` / `CostGuardedProvider` / P3R usage reader。错误或中断保持原 journal；发现费用、usage 或模型/指纹问题时停止后续案例。最新离线回执已经存在，无需重跑验证脚本覆盖它。

实际执行后，按 `review_contract.json` 和 `review_sheet_template.json` 核对作者原句及最终真实可见 prompt，保留全部失败。此次纯文本不覆盖真实视觉读法与真实 S4 因果误读；不能把这些维度记作零错误或正式验收通过。
