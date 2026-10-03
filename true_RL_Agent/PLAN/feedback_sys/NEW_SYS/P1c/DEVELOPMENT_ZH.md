# P1c：版本化卡片进入最终作者请求

2026-10-03 开发交付，等待父线程独立验收。起点为已放行的 P1b B1 `00e63feb76123292984071cd4a6bc5e9d362bd1a`，专用分支 `codex/trajectory-translator-upgrade`。本轮单写，只读子线程审查；不进入 P2/P3。计划依据为相邻 v1.1、P0/P1a/P1b 原文及既有开发验收流程。

## 交付结果与可复核入口

真实 `outputs/trajectory_p1b/repair_b1/final_card/` 经生产 CPU finalizer、任务版本分发、公开过滤、Stair prompt adapter、既有 diagnostic context adapter、PublicContextBuilder、预算与 `serialize_prompt_payload`，进入生产 provider 的 HTTP `Request.data`。四帧均在 urllib opener 外发前中止，保存原始字节。本轮没有模型回复、外部网络调用、训练、GPU/Isaac 执行或新凭证。

- 最终原文：`outputs/trajectory_p1c/final_capture/{draft,source_followup,static_repair,reasoning_exhaustion_repair}/request.http_body.json`；对应 `user_payload.json` 是嵌套 user content 的精确 UTF-8 字节。
- `sentence_delivery.json`：每个公开句子的 section、完整文字、user/HTTP 字节偏移、原 claim。
- `projection_audit.json` / `behavior_full.json`：全部801条 claim、18必保ID、783省略ID及理由、P1a文件SHA/byte locator、源绑定及保序操作数记录。
- `claim_to_evidence.json`：801条逐项 ID→原句→operand→内部记录或 P1a byte locator→原数组引用链。
- `e1_delivery_coverage.json`：18冻结例、247必要事实的 claim 映射、309完整事实的 record 映射，明确作者层与完整审计层。
- [REVIEW_DELIVERY_ZH.md](./REVIEW_DELIVERY_ZH.md)：从实际产物摘录的18句、请求hash、E1逐例表。
- [verification.json](./verification.json)：本轮命令、原始日志/产物SHA、代码SHA、边界和审查发现。原始 outputs 均不入 Git。

| 帧 | 公开report字节 | task sections字节 / 上限 | user JSON字节 | HTTP body字节 |
|---|---:|---:|---:|---:|
| draft | 8094 | 41691 / 57344 | 65792 | 71512 |
| source-followup | 8094 | 41691 / 57344 | 67069 | 72873 |
| static-repair | 8094 | 41506 / 57344 | 66107 | 71911 |
| reasoning-exhaustion-repair | 8094 | 41691 / 57344 | 67338 | 73132 |

8192B约束针对完整公开行为report，不是整个prompt。Push既有12288B接口和实现未改，本轮不声称新卡已跨任务验证。followup使用既有source request service中的真实白名单代码请求；两种repair输入是明确的合成错误，候选未验证/执行，未伪装模型输出。源代码目录仍是代码身份，不是轨迹档案检索接口。

## 最小生产改动与版本迁移

新增任务内 `trajectory_report.py`，不改共享测量层、P1b模板或奖励协议。新版本 `true_rl_agent.stair.behavior.full.v4` / `true_rl_agent.stair.behavior.public.v5`，派生版本 `stair.trajectory_report.v1`，词表 `stair.trajectory_terms.v1`。共享六个句子区加 actionability 保持不变；新报告不经过旧 Stair 四词校验。`project_stair_behavior`显式分发新版本，public v1/v2/v3/v4与旧full仍按原分支回放。

finalizer只增加可选 `trajectory_card_ref`；默认 `None` 继续旧report。现有 composition 向runner注入finalizer的机制不变；CPU回放脚本以同一 `partial(finalize_stair_eval_phase_summary, config=..., trajectory_card_ref=...)` 接入。不自动给未来trial套当前离线卡，不宣称在线默认已经切换。P1c在已验证Stair输入上证明新路径可用。

关键实际代码：

```python
validate_claims(card["claims"], records)  # Includes P1b metric/label/entity/unit binding.
if card["policy_summary_claim_ids"] != select_policy_summary(card["claims"], records):
    raise ValueError("trajectory mandatory summary selection mismatch")
```

```python
if required_bytes > max_bytes:
    raise ValueError(f"trajectory mandatory facts exceed byte budget: {required_bytes} > {max_bytes}")
for cid in optional:
    candidate = render([*included, cid])
    if report_bytes(candidate) <= max_bytes:
        included.append(cid)
        public = candidate
```

```python
if anchor.get("public_result_hash") != canonical_json_hash(result):
    raise ValueError("Stair trajectory reference public result hash mismatch")
binding = anchor.get("trajectory_phase_binding")
if not isinstance(binding, Mapping):
    raise ValueError("Stair trajectory reference requires explicit phase binding")
require_trajectory_phase_binding(profile, **binding)
```

`PLAN/feedback_sys/P3R/confirmation_context.py`只为新schema增加六字段phase输入绑定，取自已经过原回放检查的实际phase；旧Stair/Push分支不变。该绑定在Stair原始输入上核对，随后只投递已验证公开profile；内部receipt和路径不进prompt。self-hash是完整性检查而非签名，public-only旧档回放仍依赖外层封存phase/hash与身份；full回放额外用原操作数重做全部语义校验、选择及公开视图，核对原主源绑定和包含/省略审计，不能靠改句后重签蒙混。

## 来源和trial关系

当前主源是 R4 public-dev diagnostic job：

```text
job: stair-dev-replay-7b582643510cc9dd-diagnostic-eval-diagnostic-public-dev
checkpoint: sha256:7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c
plan: c10b9bbcd5191909f9ec925e042bd7cb33e5516b0e1d5ff1cb415968f4e89f19
namespace: public_dev_diagnostic_n5_r4_20261001
archive: 8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83
card: d83ddf0b8ee28e8d00705de81029930474b47ba230532596d50aabd2e9b0d9a2
catalog: trajectory.p0.measurements.v2
```

P1b的 `cohort_binding.trial_id` 实为 archive evaluation job_id；新domain改用明确的 `evaluation_job_id`。当前卡属于同一已资格化policy的额外诊断观察，不是generation trial，也不产生新的accepted-best决定。generation路径对每个带新report的trial核 `trial_id + '-eval-public-dev'`、phase checkpoint及plan，包括最终未选中的siblings；保留既有最后一条已测lane选择，但明确 independent of acceptance，仅正式accepted且selected trial相同才标记 accepted_in_this_generation。

父代缺档为unavailable；即使两个新报告版本/域相同，本轮也不实现父代科学比较。旧新混合版本、父checkpoint或比较域不同均不可比。legacy两例仍独立，不成为父代、不混入864回合分母。公开profile包含任务/协议、eval binding、seed namespace、projection、archive与catalog；绑定错配先失败，不能靠公开scrub掩盖。

## 预算、保留与有据调整

保留P1b原15条主句，再强制加入时钟/history、coverage、comparison_scope三条定义，共18条；原句和稳定ID不变，只增加scope及单回合cell/outcome限定。必保内容覆盖864分母、unknown、两类低进度合取、成功/失败对照、成功回撤、先翻转后lowoff反例、单条history事件与截尾、缺测、后续义务及父代不可用。

其余整句按有限模板优先级再按claim ID确定性尝试；不截句、不缩数字、不删除限定词。本档剩余98B放不下任意可选句，省略782条可选claim及1条已过时的P1b“投递未实现”状态句。公开只给计数/原因/ID清单摘要hash；完整ID、原文和理由在audit。没有“801句全塞”或“作者能凭ID自动读取内部文件”的假设。

与P0英文作者表面预设的调整：P1b接受卡为中文，当前任务要求逐句保真与中文UTF-8预算边界，因此P1c直接保留中文原文，协议包壳用现有英文；没有自动翻译或第二套自由语义。实际8094B可容纳全部必保句，测试另覆盖精确8192、8193超限、缩预算到必保字节及少1B显式失败。语言效果仍待P3。

full采用保序操作数JSON字符串而非普通嵌套dict：P1b少数模板用映射插入顺序输出JSON文字，full封存canonical排序会改变它。该内部字符串保留原操作数顺序，再用现有P1b validator重导出；没有改P1b渲染、度量或数值，亦没有加入通用对象顺序框架。

## 验证与修复过程

最终本轮原始日志在 `outputs/trajectory_p1c/final_checks/`：74项P1c相关、65项P1b、107项P1a、201项旧专项全部OK、零skip。这些组有重复模块，不把447次执行称为447个互不重复测试。必要247/247、完整309/309全部pass。原数组独审866例、69280 metric、8400 aggregate metric、32042引用、801句通过；P0 18选择与原必要账不改。

`audit_delivery.py`逐字节验证四帧、逐claim与源卡对账、15个真实产物负例及E1分层。新增单测还覆盖旧版本回放、兄弟trial、父代、任务/catalog、非公开内容与绝对路径、重新签名的句子/必保删除/负计数/坏digest、minimum-core压缩后的完整report及最终serializer。已知广域1failure/8errors/19skip不在本轮修复范围，没有重称全仓通过。

只读复审发现并在最终检查前修复：reference输入缺少独立phase绑定；full仅检查公开self-hash；seed域未交叉核对；负计数/坏digest可重签通过；full公开domain未对照已有主源。开发早期还保留了 sealed字符串误判、source-followup缺必需反馈和映射排序的失败日志，未覆盖或删去。最终证据仅指 `final_capture` / `final_checks`。

E1覆盖必须分层读：18个冻结单回合的细节claim均没有进入当前18条主句；它们共246必要/308完整事实，再加1条全队列事实，形成247/309门，仍在P1a和完整卡片审计中核验。只有该队列事实关联作者中的3条聚合claim，且不代表其全部字段已重述。主卡聚合/能力句不能算成所有单回合事实已投递。包括 e129、e35 code11、e60 XY区别以及两legacy例的详细数值均为audit-only。完整覆盖表逐项列出，不更换E1抽样来抬高作者覆盖。

## 未覆盖与停止点

CPU finalizer实测wall 8.505s、CPU 8.453s；不是新runner/GPU rollout。本轮只验证最终请求编码、来源/预算和语义保留，不证明作者读懂、生成候选可行、DeepSeek可用、收益或泛化。未进行干净6.1SOL作者理解试验，故没有作者模型receipt/input-output或伪造DeepSeek记录。

公开协议原有 `sealed_validation_seed_count=7` 是固定预算合同；最终prompt不含sealed结果/seed明细、内部绝对路径或原数组locator。周期资格、有效/名义限位、逐类地形、历史link位置/COM速度映射、新增采集、视觉、跨策略与Push新卡验证继续后续义务。完整audit可在工程验收中按ID检索，作者请求本身没有新增检索权限。

提交并推送专用分支后停写，等待父线程逐源码、原请求、原数组/反例独立验收；本记录不自行放行P1c。
