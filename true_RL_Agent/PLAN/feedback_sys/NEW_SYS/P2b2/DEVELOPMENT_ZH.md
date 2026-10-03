# P2b2：Push 正常自动轨迹反馈接线

2026-10-03 阶段状态补记：本次 P2c 委派明确确认 P2b2 独立 CPU 接线及 root 亲核已经 PASS，
因此 CPU 阶段已放行。下文原开发自测记录保留，结尾“待独立验收”描述的是原交付时点。
此更新不表示 K5、真实模型理解、新 GPU rollout 或闭环收益通过；P2c1 另设独立门。

起点为统一分支 `codex/trajectory-translator-upgrade` 的 `7c1c811177ead0a920546d5ed8e23d1d05eacdc5`。本轮只完成 Push 的正常 composition → public-dev 数值组 → CPU finalizer → phase/full receipt → generation → 下一作者 HTTP 序列化接线。未新建分支；没有使用恢复引用 `codex/push-trajectory-p2b1` 写入。唯一写入者为主代理，两位子代理只读复核。

P2b1 冻结文件 SHA256 仍为 `ec8158315d79fba1ff2f9122016de5f1eb5af5f7a0de96cc6f09dc5cda8ff5e8`，其中 22 个文件逐一重新核对。`trajectory_adapter.py`、`trajectory_measurement.py`、`trajectory_card.py` 及其冻结测量、模板、单位、signed direction、事件 pre/post、稳定 latch 和 unknown 规则没有修改。未查看 K5 行为、summary 或结果。

## 正常入口与支持范围

| 配置/能力 | 行为 |
|---|---|
| 默认 `legacy_process_v1` | 保持旧报告和旧 Genesis/task-defaults 形状 |
| 显式 `trajectory_numeric_v1` | composition 绑定正常 CPU finalizer；每个完整 public-dev 数值组自动建自身卡片 |
| `trajectory_report_max_bytes` | 默认及上限 12288；约束整个公开 behavior 对象，包含身份、版本和投影收据，不是整个作者请求 |
| 支持 cell | Push G1-37DoF / Go2-12DoF，recovery v2/v3；其他 cell 显式拒绝 |
| force numeric v1 | 可做原有模长/根部证据，`root_signed_xy=not_recorded`，不伪造 signed XY |
| force numeric v2 / velocity numeric v2 | 各按自己的协议解释，signed 能力为 `recorded_world_xy`，世界 XY 不解释为机器人前后左右 |
| 缺 archive、缺 seed/phase、协议/身份混合、必保句超预算 | 明确失败，不降级旧卡，不把不完整组说成完整 192 回合 |

正常 generation 入口（保留目标环境原有 session/provider/cost/GPU/source 参数）：

```text
python scripts/run_generation_session.py --task-id unitree_push_recovery --cell-variant-id push.go2_12dof.recovery_v3 --behavior-feedback-profile trajectory_numeric_v1 --trajectory-report-max-bytes 12288 <已有 session/provider/cost/GPU/source 参数>
```

独立 replay CLI 同样支持这两个参数；不再遗漏配置或 host GPU → child `cuda:0` 的 `GpuProcessBinding`。示例：

```text
python scripts/push_numeric_replay.py --cell push.go2_12dof.recovery_v3 --source-manifest <已训练策略的resolved_cell_manifest.json> --checkpoint-summary <其训练summary.json> --checkpoint <其checkpoint.pt> --device cuda:1 --output <全新输出目录> --mode preflight --capture on --behavior-feedback-profile trajectory_numeric_v1 --trajectory-report-max-bytes 12288
```

`--mode run` 是目标环境实际 rollout 入口，本轮未运行。新 profile 不允许 `--capture off`。实际 CPU CLI preflight 命令及 `PREPARED` 输出已保存于 verification，输入明确标为 synthetic admission fixture，不能解释为 checkpoint 装载或 GPU 验收。profile 与预算进入既有 Genesis scientific parameters；恢复时改变任一配置均在额外执行前拒绝。

## 接线与校验边界

新增 task-owned `trajectory_pipeline.py` 和 `trajectory_report.py`，没有新 DSL 或通用执行框架。原 runner 已有“子进程结束 → GPU lease 释放 → CPU finalizer → seal receipt”顺序继续使用。

1. `composition.py` 用 `partial(finalize_push_eval_phase_summary, ...)` 绑定显式 profile、预算、计划 public seed 数。replay CLI 透传相同配置。
2. `phased_evaluation.py` 在 CPU 阶段验证整组 manifest、run index、job/task/subject/source/training/checkpoint/plan/eval binding 和 cell/protocol，再交给冻结 adapter、measurement、card。正常三 seed 聚合三份 archive。合成单测是 12 回合，K4 三 archive 是 192 回合，各自按真实分母报告。
3. frozen `project_push_card(card, episodes=episodes, source_id=source_id)` 仍必须拿到原 episodes，重算语义。改 metric/unit/pre/post 后重签自 hash 也会拒绝。source ID 由外部、排序后的完整 manifest SHA 列表派生，不采用待验证卡自己声明的 source ID。
4. CPU 阶段产生 full.v4 / public.v4。全卡和全部证据保持私有；公开对象是有限合法 JSON，20 必保槽先保留，可选句按整句加入，省略 ID 与原因进入私有 receipt。原卡句子和 claim ID 不改。
5. `phased_adapter.py` 将 producer identity、全部 source SHA、card/full/public hash 和 build receipt 交叉绑定到既有 phase/full receipt。冻结 capture 没有记录 `capability_profile_hash`，所以不伪称 raw 记录了它；该字段继续由外层 phase/plan 契约校验。capture 中确有的 training binding 加入新 profile 原档检查，legacy 路径不变。
6. `behavior.py` 对已封存 public.v4 做无原档 IO 的校验。private full.v4 不能进入普通 prompt 投影；独立 CPU full replay API 需显式传入外部来源与身份。正常 prompt 不重复加载大 archive。
7. `prompt_adapter.py` 保留 producer phase binding，并逐 sibling 验证 trial/candidate/checkpoint/plan/program 来源。selected feedback、siblings、parent 和 current accepted 分开标明。跨代选择回到真正包含该 trial 的 generation；minimum-core 投影保留来源映射。父代不仅比较 checkpoint，也比较角色、subject、manifest、program/training/eval binding，不能将最后 lane 冒充 best。

每个 public-dev phase 的产物为：

```text
trajectory/<cache_key>/evidence.jsonl
trajectory/<cache_key>/policy_card.json
trajectory/<cache_key>/public.json
trajectory/<cache_key>/complete.json
behavior_full.json
summary.json
```

cache key 包含原来源、身份、profile、context、预算和相关生产代码哈希。CPU 重试可复用文件，但仍用原 episodes 完整验证；完整 runner receipt replay 原样复用，测试禁止它重新加载 raw。缺完成 manifest 的构建可重做，已有完整缓存字节变化则拒绝。CPU finalizer 失败不会伪造 sealed summary，也不会自动重跑 GPU。

冻结 `reward_parent` 槽含 P2b1 的“尚未接入”历史交付描述；本轮保留原句与 ID，在新的 delivery 句中明确其历史范围。未以接线任务为由改冻结模板。独立 parent source 可见不等于数值配对父代比较已资格化；报告仍明确该比较不可用。

## 最终 CPU 生命周期和真实请求

复用已有 Push integration factory、正常 generation application、composition、runner、receipt、Genesis 和 P1c/P1d HTTP 捕获；只替换模拟器进程边界、训练结果和非 public-dev outcome 的 fixture。没有离线手工 card 注入。

`outputs/p2b2/final_run/verification.json` 为最终生命周期原始记录。K4 force-v1 的同一组三个 archive 被显式 fixture rebind 到 reference 和两位 sibling，NPZ 字节保持不变，原档不修改。每份原/fixture identity、变化字段、manifest SHA 和 chunk SHA 均在本目录 verification。此项证明接线，不证明三个新 checkpoint 的科学表现。

| 角色 | source ID | card ID | 公开字节 | 原卡句/总句 | 省略 |
|---|---|---|---:|---:|---:|
| reference parent | `p22ecc6544d79` | `c0370e7acdc0c7a3e596aad0a1ebc26b3a8f78e94361bc5c64a7c916e0515d74` | 12041 | 29/37 | 8 |
| lane A | `pec12bbbaca7f` | `f4cc93fb298a3259491848855d82ccef96ee7c9c69e04ea0cf37c601595605d1` | 12280 | 30/37 | 7 |
| lane B | `p4fbcaecf4381` | `44f975a6aee9deb6da2739520d5c4c5f3e9107d221636c9c34132a1045880138` | 12279 | 30/37 | 7 |

表内 card ID 统一加 `push-card:` 前缀。每卡各 3 archive、192 episodes、20 必保槽。reference 三 phase 形成正常 full receipt；候选 A 在 public-accept 早拒绝，结果为 `formal_rejected`，不伪称该候选拥有 sealed phase/full result。下一作者选择 A 的反馈，B 为 `qualified_not_selected`，current best 仍明确为 reference parent。accepted-A / nonaccepted-B、跨代回退和角色错配另由单元测试覆盖，不能说本生命周期实际接受了 A。

三个作者请求均来自正式 provider serializer，在 urllib `Request.data` 形成后、网络之前截获：

| 请求 | HTTP bytes | 原始 user bytes |
|---|---:|---:|
| g1-lane-a | 66213 | 60801 |
| g1-lane-b | 66920 | 61474 |
| g2-lane-a | 81259 | 74843 |

下一请求的四个允许 frame 均用同一真实 generation/history 构造，各保留同一 A 卡和完整来源映射：

| frame | HTTP bytes | 原始 user bytes |
|---|---:|---:|
| draft | 81259 | 74843 |
| source_followup | 78864 | 72586 |
| static_repair | 72375 | 66490 |
| reasoning_exhaustion_repair | 82879 | 76389 |

每框验证 33 条实际发送句（30 条原卡 claim + 3 条 delivery 句）的原始 UTF-8 / HTTP 转义后字节位置；20 必保槽、counterexample、unknown/no_witness 均保留。provider section receipt 和 actual payload bytes 一致。所有请求扫描无 `manifest_path`、`numeric_evidence_dir`、`trajectory_build` 或 `evidence.jsonl`。原文在 `outputs/p2b2/final_run/{requests,frames}/<名称>/{request.http_body.json,user_payload.json}`，完整 SHA、字节位置、claim 与省略映射见 [verification.json](verification.json)。

## 验收结果、成本和保留失败

- 最终专项：22 项通过，覆盖新 pipeline、正常 CLI/binding、完整 phase/receipt；旧 Push recovery task 模块 59 项通过。
- 宽回归第一次：386 项，380 PASS / 1 FAIL / 1 ERROR / 4 SKIP。唯一 FAIL 是起点 commit 已存在的 static-repair 测试旧断言：要求删除当前 frame 契约明确要求保留的 feedback/selection guidance。在 `git archive` 起点源码副本复现后，只修正断言，未改变 frame 产品行为；59 项模块复跑全过。
- 按测试 ID 去重、采用最终结果：**387 项 = 382 PASS / 0 FAIL / 1 ERROR / 4 SKIP**。不是全仓通过。ERROR 为 `test_two_native_snapshot_streams_close_at_each_env_terminal`，本地无 `omni`（经 `omni.replicator.core` 导入）；4 SKIP 为需要目标 Isaac source closure 的 Stair generation 集成测试。
- 最终生命周期 PASS：37.316 秒（外部 Python 命令 38.245 秒），实际 Python 进程峰值 working set 293,478,400 B。6 个完成 phase replay 禁止 raw loader，输出字节和时间戳不变；9 次 lease 检查覆盖 rollout 持有、三个 public-dev CPU finalizer 之前可重新获取；2 次 Genesis 配置漂移拒绝。
- 三卡构建 wall 约 1.149 / 3.092 / 2.824 秒（parent/A/B），full 文件 481,678 / 476,134 / 480,545 B。并发 lane 的 `process_time()` 是整个进程计时，不能相加称独占 CPU 成本。fixture 的 rollout wall=0.01 和原档 simulated counters 不是本次 GPU 成本。
- K4 原始 source-lock 的 force-v1 / velocity-v2 另经 CPU pipeline + raw semantic replay 通过：各 192 episodes，报告 12276 / 12035 B，均保留 20 必保槽；分别省略 7 / 3 条可选句。混 force/velocity 明确拒绝。force 的 11/81 观测、缺70格，以及 velocity 完整窗未达 dwell 但 terminal stable 的区别仍保留；世界 signed 方向和能力未知边界未变。这是原始协议投影审计，不是 velocity 正常 GPU generation 验收。
- Stair 冻结三卡文件与 P2a 33 个产物重新做 SHA/字节检查一致。复用检查器带出的 247/309 是历史 receipt 数字，本轮没有重新执行其原数组事实审计，未改 P0/P2a 产物。

开发 fixture 初期的 summary/context/source 格式修正保留于 `outputs/p2b2/lifecycle_01..04`。第一次最终 lifecycle 使用更深目录，使 sealed manifest 路径长达262字符：枚举可见而 stat 报缺失，正常权限读取也被拒绝；失败日志未删除，未更改生产权限逻辑。缩短至 `outputs/p2b2/final_run` 后同一生产链路通过。仓库已有长路径的沙箱假 D 则经正常权限 `git -c core.longpaths=true status` 核实，并非实际删除。

所有实际命令、stdout/stderr SHA、峰值内存、原始失败与后续复跑在 verification。测试源码和必要 audit metadata 入 Git，原始 archive、HTTP 大文件和 outputs 不入 Git。关键 diff 原文保存在本地 `outputs/p2b2/final/key_diff.patch`；也可直接审阅本提交。

复跑入口（输出目录需全新；短目录避免本机长路径限制）：

```text
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2b2/run_validation.py
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2b2/capture_lifecycle.py --output outputs/p2b2/recheck
```

本轮实际 GPU / 外部模型 / HTTP 网络调用均为 0，没有新训练。未验证目标 Isaac checkpoint 装载、真实 rollout 到 finalizer 的成本或并发运行、真实作者模型理解、闭环收益及留出资格。完成后仅提交并推送已授权统一升级分支，停写等待独立验收，不进入 P3 或 DeepSeek 调用。
