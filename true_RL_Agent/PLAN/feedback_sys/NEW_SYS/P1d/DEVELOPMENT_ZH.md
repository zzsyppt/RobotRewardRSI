# P1d：正常任务自动数值反馈接线

本阶段从 `41f5a44bf6dc63e1fd8aa8efed291d13f44f7511` 开始。用户确认 P0/P1a/P1b/P1c 的相应受限资格已经独立及父线程亲核放行；P1d 是父线程指出的工程补充，完成后停写等待独立验收，不进入 P2。

## 实现与正常入口

P1c 的 full.v4/public.v5 和生产 HTTP 捕获已证明显式卡片能投递，但 composition 中的 finalizer 没有卡片参数，普通新 trial 仍走旧过程报告。P1d 在正常 `StairClimbingCompositionProvider` 中引入显式版本选择：

| 配置 | 行为 |
|---|---|
| 未提供参数，或 `--behavior-feedback-profile legacy_process_v1` | 保持旧行为；旧 task defaults、Genesis 科学身份形状不增加新字段 |
| `--behavior-feedback-profile trajectory_numeric_v1 --trajectory-report-max-bytes 8192` | 每个 public-dev numeric v2 archive 自动生成本 trial 的 P1a 证据、P1b 卡，再进入 P1c finalizer |
| 新 profile + Stair G1，或未知 profile / archive 版本 | 明确拒绝；不借用 R4、不静默返回旧报告 |
| Push | 未增加新卡配置或适配接口 |

新 profile 支持范围限定 `stair.go2_12dof.unified_v2` / `stair.go2_12dof.unified_v3`；本次真实归档及完整 CPU 生命周期验收均为 Go2 unified v3。默认继续为 legacy：现有默认 cell 是 G1，不能以 Go2 卡片能力冒充默认全任务支持。选择 profile 和字节预算会进入既有 Genesis scientific parameters，同目录恢复不能改配。训练协议、奖励、胜负规则、GPU/模拟步数预算均未改变。

正常应用参数示例（沿用目标环境已有的其余运行参数、credential 与冻结 cost contract）：

```text
python scripts/run_generation_session.py --task-id unitree_stair_climbing --cell-variant-id stair.go2_12dof.unified_v3 --behavior-feedback-profile trajectory_numeric_v1 --trajectory-report-max-bytes 8192 <既有session/provider/cost/GPU/source参数>
```

该命令是后续目标环境入口说明，本次没有运行训练、GPU、模型或网络请求。目标服务器运行环境为 `isaaclab6_validation`。本地 CPU 验收命令在下节，可直接复跑，不需要新 key 或安装。

## 接线、来源和恢复

`runner` 的子进程结束并释放原 GPU lease 后，已有 CPU finalizer 执行 `build_trial_trajectory_card`。原离线脚本的构建函数移至 task-owned `trajectory_artifacts.py`；两脚本成为薄 CLI 包装。生产直接调用函数，不启动外部建卡脚本。

产物集中在该 phase 的目录内：

```text
eval/public_dev/
  numeric_evidence/manifest.json + *.npz
  trajectory/<cache_key>/numeric/{manifest.json,evidence.jsonl}
  trajectory/<cache_key>/card/{manifest.json,evidence.jsonl,policy_card.json,policy_card_zh.md}
  trajectory/<cache_key>/complete.json
  behavior_full.json
  summary.json
```

cache key 绑定完整 archive SHA、job/checkpoint/plan/evaluation identity、source ID、测量配置及相关生产代码哈希。预算独立于数值/卡片缓存，投影重新按预算计算。source ID 由 archive SHA 派生，整套缓存仍用完整 SHA。没有完成 manifest 的本 key 子目录才可重新构建；已发布 manifest 不覆盖；完成 phase 由既有 runner receipt 原样回放，不打开离线 archive、不重新建卡。失败遵守原 SUMMARY_INVALID / nonretryable 规则，不额外发起 GPU 重跑。

private `trajectory_build` receipt 同时保存到 phase task evidence 与 full audit；phase hash、full 文件 SHA、attempt/trial journal 覆盖封存结果。恢复验证来源、cache key、public card/content hash、18 个必保 ID 及字节预算。正常 reference Genesis 的 producer phase binding 从验证过的 public-dev 生成，随 task metrics 进入 qualified public result；下一作者从正常 Genesis 或 generation history 取卡。筛选器复用同一个 phase 附件校验器，仍只从原 evaluator summary 算分。

自动卡仅使用当前 archive 全队列和确定性例子选择，无 P0 必要事实导入、无 frozen E1 示例、无 legacy 反例入队。保留 P1c 的 18 个必保结构；真实档案缺合格模式/反例时，对应槽明确“本次卡片未选出符合目录条件的证据……不代表物理上未发生”。不捏造 R4 特征来填 K1。必保句超预算则失败，额外句按整句优先级添加。

## CPU 验收边界

```text
.venv/Scripts/python.exe -u PLAN/feedback_sys/NEW_SYS/P1d/capture_lifecycle.py --output outputs/p1d_final --r4 outputs/trajectory_p0/raw/numeric_evidence/manifest.json --k1 outputs/trajectory_p1d/raw/k1_numeric_evidence/manifest.json
```

`capture_lifecycle.py` 用正常 composition 构建真实 SubprocessRunner，正常 generation application/Genesis/phase/筛选/接收/trial journal/ledger/context 全链路运行。只在进程边界注入既有 archive handoff；finalizer 从 composition 获取 profile，没有手工传卡。第一代两位作者由正式序列化器在 urllib `Request.data` 后、网络发送前捕获；第一代封存、恢复后捕获第二代 lane-a 请求并停止。最后一次截停按现有生命周期记录为 exception，不能表述为 15 代 session 全部完成。

真实原始来源：

| 来源 | 原始 manifest SHA256 | 原始 checkpoint |
|---|---|---|
| R4 | `8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83` | `sha256:7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c` |
| K1 | `0389a10d7cb9dbe14fff063f43c56482fcc6fbd918281a37beacb59c043db8d3` | `sha256:fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729` |

K1 由已授权 SSH 别名只读复制，服务器路径 `/data/yuyangliu/AGENT/feedback_p2_audit_20260924/stair_parent_on/numeric_evidence`，27 NPZ，74,694,953 字节。原始文件不修改，不在服务器运行 GPU。

生命周期 fixture 将原 archive 的 job/checkpoint/plan/evaluation/namespace/backend/source-preflight identity 显式绑定到测试任务；逐一记录 changed_identity_fields，并逐个核对 NPZ 字节 SHA 未变。R4/K1 原 source-preflight 不同，fixture 统一使用 K1 preflight 使正常比较协议能运行；这不证明两次真实运行的环境可比较。归档 metadata 没有保存 active_terminal_terms；构造 handoff outcome 时仅补空 tuple，env_index 按原数组序号恢复。训练、非 public-dev 阶段和作者草稿复用既有测试 fixture，未执行新策略、Isaac rollout、真实训练或模型理解。

原合成 fixture 每 operation 磁盘预留只够小型 JSON，不够真实 P1a JSONL，因此在 cost contract 冻结之前，仅增加本验收 fixture 的 session output reserve 至 6 GiB，初次和 resume 一致；不改变科学预算或选择规则。生产部署应按实际 archive 数量冻结足够磁盘预留，不能照搬这个合成测试预算。

## 验证记录

最终 CPU 生命周期 `PASS_CPU_COMPOSITION_ARCHIVE_GENERATION_HTTP`，439.83 秒，0 GPU / 0 外部模型 / 0 HTTP 网络调用。第一代接受 lane A；下一作者收到最后测量的 lane B，并明确 `accepted_best_relationship=not_established_for_selected_trial`，没有把 B 的卡冒充 accepted policy。第一代两个原请求均收到父代卡。

| 角色 | 原 archive | 本 phase source ID | card ID（前缀；完整值见 verification） | public 字节 / 必保数 | 全卡 claims |
|---|---|---|---|---:|---:|
| parent | K1 | `s43c01e90fef0` | `fb62b90af7053f17…` | 8056 / 18 | 250 |
| lane A | K1 | `s20afce8efb95` | `7c176b3629b1194c…` | 8078 / 18 | 250 |
| lane B | R4 | `sef4c0ea08fa1` | `7123f9f7a8ef213c…` | 8118 / 18 | 400 |

原始 HTTP：`outputs/p1d_final/requests/{g1-lane-a,g1-lane-b,g2-lane-a}/request.http_body.json`，同目录 `user_payload.json` 是请求中的原始 user 文本。第二代 HTTP 为 98,119 B，SHA256 `fb6a6106dcf80b41b9a635edec896b922dc40ae7fb96a5831bbc1248a0305f90`；所有原请求 hash 均在 verification。8192 限制针对整个 public behavior 对象，非整个作者请求。

回归：P1c 相关 75（原74＋fresh capture拒绝跳过CPU的新测）、P1b 65、P1a 107、旧能力201全部通过；另有新profile11项、正常composition/context70项及runner/lifecycle检查。全部实际测试 ID 去重为 **634 PASS / 8 SKIP / 0 FAIL**。8 skip 是3个Linux专属进程契约、1个Windows不可用的文件symlink检查与4个需目标Isaac源码闭包的Stair集成测试。补充命令曾误写两个不存在的模块名，原日志记2个collection ERROR；随后运行正确的composition/context模块70项通过，不把调用错误计作真实测试通过。最终生产改动后受影响的report/phase/profile47项复跑通过，最后新增“删除unavailable槽后重算一致hash仍拒绝”的11项profile集通过。

冻结 E1 仍为必要247 / 完整309；重建旧卡三文件逐字节相同，原数组审计866回合、69,280指标、8,400聚合、32,042引用、801 claims通过。此项仍是历史冻结材料的回归，不为K1宣称独立E2。

性能/存储：parent单次建证据＋卡107.79秒；并行兄弟lane为226.03/239.89秒（同一Python进程，不能视为独占核心吞吐）。P1a各1.125/1.125/1.290 GB，P1b各56.30/56.30/61.54 MB，整个session为3,978,823,189 B。卡manifest记录的进程峰值最高979,865,600 B，是进程累计峰值、包含并行lane，不是每卡独占内存。full各5.67/5.57/9.81 MB。含原档复制、原档探针、六次开发重跑和回归文件的P1d保留输出总计约13.32 GB，未把开发重跑计为生产单次成本。已有阶段逐回合流式处理，P1a/P1b/P1c保留各自完整性核查；仅去掉P1c对同一primary archive的重复校验，没有删除事实审计换取性能。

最终命令、原始日志 SHA、每个 trial 原来源/fixture identity/source ID/card ID、HTTP 请求路径和性能以 [verification.json](./verification.json) 为准。原始大文件保存在 `outputs/trajectory_p1d` 和指定 session 输出目录，不入 Git。

开发期间发现并修复：archive JSON 的训练 profile 映射顺序需保留；合成 fixture 的存储不足；normal public-dev selector 拒绝 numeric 附件；新配置未进入 Genesis identity；private build receipt 未交叉验证；缺合格例的自动卡需显式保留18槽。失败原始日志保留在 verification 的 development failures 项中，不删去失败记录。

历史 P1b card 的严格代码哈希门禁因构建器提取而拒绝旧 manifest，属于正确拒绝。用同一冻结 P1a/E1 输入重建后，`policy_card.json`、`evidence.jsonl`、`policy_card_zh.md` 与旧卡逐字节相同，再用当前代码 manifest 通过独立原数组审计。P0/P1a 冻结材料未改；legacy 反例只在这项历史兼容回归中出现，不进入自动卡队列。

## 尚未资格化

本次仅证明受支持档案的 CPU 接线及真实序列化投递。目标环境仍需单列短 eval 验证：实际 checkpoint 装载、Isaac rollout 产出新 archive、GPU lease 与 CPU 后处理的实际运行成本。未运行外部作者模型，DeepSeek 理解仍未测。未证明新策略科学资格、跨 lineage 描述迁移、Push/G1 新卡、视觉增益或闭环收益。父代卡可作为来源进入作者上下文，但数值父代比较仍明确 unavailable。周期、关节限位、逐几何地形、历史 link/COM 映射仍是后续义务。

P2a文档勘误：多行docstring的existing-summary测试被旧单行计数漏掉，实际642项=634 PASS+8 SKIP；原日志不改。证据文件82份=81个唯一引用文件+verification索引自身，索引不作自引用hash。P1d已由用户本轮确认独立及父线程亲核放行。
