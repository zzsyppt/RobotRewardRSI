# R1a：统一生产选择与 CPU 作者输入验收

起点 `82828fd66d0cd687a63559f2fc90c834a4d35b43`，分支 `codex/trajectory-translator-upgrade`。本阶段只实现 R1a；P2c3 已独立限定离线 PASS，旧文档的待验收字样不作为阻塞。真实 collision 仍未知。一个 writer，Astra ultra 只读开发审查与独立数值复算；无功能模型、网络请求、GPU、训练、新采集或 server 修改。

## 生产行为与版本

显式 CLI `--behavior-feedback-profile trajectory_numeric_v2` 经 composition → 正常 Stair Go2 CPU finalizer → phase report → generation/context → 既有 provider 的 system/user 与 HTTP JSON serializer。统一选择基础、cycle、名义 joint limits、名义 terrain，先保必要语义再选整句可选事实。不是在旧 8KB 卡片末尾 append；新路径直接加载已验证的基础 operands，不受旧 author view 的 8KB gate 限制。

独立生产 catalog 为 `stair.trajectory.production_catalog.v1`；public/full schema 为 `true_rl_agent.stair.behavior.public.v6` / `full.v5`；selection 为 `stair.trajectory.unified_selection.v1`，lexicon 为 `stair.trajectory_terms.v2`。这与原 cycle/limits/terrain 的 v3/v4/v5 补充目录不同。archive/profile/catalog/report/selection/lexicon/预算/代码与资产哈希进入构建、缓存或 session Genesis 身份；completed marker 同时固定 numeric、card 与 measurements 引用。

历史 `trajectory_numeric_v1`、旧 full.v4/public.v5 与三个独立 supplement 的回放入口保留。默认配置不切换已有 session，旧 profile 默认 8192B；新 profile 未指定预算时为 10240B，显式上限 12288B。Push root 路径和实现未改。

## 资格边界与下一阶段接口

cycle 基于已验证的 current-marker 采集契约，支持合法 v1/v2/v3 源，不依赖固定历史 manifest 列表。名义能力的历史列表仍恰好是已验收的三个 hash；新增 `trajectory_historical_nominal.json` 是对 P2c3 已跟踪来源证据的固定压缩引用，含 288 个 env binding 与七个 geometry source hash，不扩白名单。其 qualification SHA 为 `84aecf9e412be7578ee567ca65cc1c6ff48351f8f45c1fb789a1a61913e8a48f`。

新 `stair.unified_first_episode.v3` 接口要求 `stair.nominal_measurement_inputs.v1`：pre-reset control-step-end；q 的实际来源表达式、rad、USD local-frame/no-added-offset、articulation names/resolved ids、projection hash、recorder source hash；terrain 的生成源版本、world-z-up/translation-only、名义布局、逐 env cell/type/origin/digest。缺少某能力字段则明确 unavailable；字段矛盾拒绝。合法但资产不在已可验证 nominal 模型中只降级 limits；不是吞掉错误契约。

`numeric_capture` 仅增加可选契约入口，默认仍 v2，未更改 runtime recorder 的行为。CPU fixture 新增元数据标记 `CPU_fixture_metadata_not_new_rollout`，数组保持原历史字节。结构验收不能证明字段已经 runtime 实录，结果明确 `not_verified_by_CPU_validation`。effective per-env limits 与真实 runtime terrain/collision 一直 unknown。R1b 仍须记录这两类证据并执行一次真实新 GPU 档案，本阶段未做。

## 预声明、选择和预算

先写 [PREDECLARATION_ZH.md](PREDECLARATION_ZH.md) 再做预算实验。18 个基础必要槽逐项保留旧 claim → 新 claim 映射，包括有效分母/缺失、短终点 history 与 manager event、成功返回、翻转反例、censoring、clock 与比较边界。无适用事件时保留明确 unavailable；有成功反例时必选。统一 definitions 去掉已启用 cycle 却仍称未启用的旧状态。

新增必选：qualified unique cycles 最大的足；全回合最小 signed margin 最小的关节；C0/C1/C2 各选 known fraction 最低的参考点。并列由实体名确定。每条保留 pooled 有效分母、unknown、episode-equal 统计与其有效回合数。可选句按稳定 ID 整句加入；内部 receipt 保留 omitted IDs、原因、operands，原18槽映射不能缺项。作者卡明确 IDs 是追溯索引，不代表模型可自动读内部审计文件。

以下为固定 K1/K2/K3 原历史档案 CPU 回放，单位全为 UTF-8 bytes。三个来源均需 23 个必要槽；8192B 无一满足，显式拒绝，不生成残缺卡。

| 来源 | 必要字节 | 10240B 实际 / 选句 | 12288B 实际 / 选句 | 10240B 测量句/千B | gaps 字节 |
|---|---:|---:|---:|---:|---:|
| K1 | 9469 | 10161 / 25 | 12140 / 30 | 1.378 | 3204 |
| K2 | 9674 | 10175 / 25 | 12229 / 31 | 1.376 | 3201 |
| K3 | 9889 | 10167 / 24 | 12191 / 29 | 1.574 | 2514 |

选择满足覆盖的最小预声明档 **10240B**。12288B 只增加可选句，密度分别为 1.565/1.635/1.723 测量句/千B。这里“测量句”是 scoped claims，不是相互独立的原子事实；所有窗口也不被计为独立样本。完全相同句子的重复数为零，这不等于没有语义重叠。gaps 占 10240 档实际卡 31.53%/31.46%/24.73%，保留下来的大部分是分母和推断边界；没有为压缩删除必要限定。

额外检查五个固定短语的重复（cycle物理限定、limits限定、terrain统计定义、Valid episodes、基础unavailable限定）：首次以外的相同短语占 K1/K2 各362B、K3 212B，两档相同。这是封闭短语表的字节计数，不是假装全面语义去重；不同实体的同统计名称也计入。cycle/limits限定各只出现一次。

历史档案预算原始输出、完整选择/省略 receipt：`outputs/r1a/budgets_final/artifacts/`。实际最终请求及同源替代预算尺寸由 `inspect_delivery.py` 核验；替代预算只做离线 serializer sizing，不称第二次 generation 执行。真实 tokens 全部 **unknown**。旧 frame 中已有 byte-derived `context_units` 不是实测 token，本阶段不把它用于 token 结论。

## 实际 provider 载荷与来源切换

完整正常 CPU lifecycle 已 PASS，三个 finalizer 产出不同 source/card，三个 completed replay 不修改产物；已关闭 generation 重启和 profile/预算 Genesis drift 均被拒绝。首代两请求绑定 reference K1，第二代 lane-a 请求精确绑定首代最后 lane-b 的 K3 fixture 卡，与 parent/reference、lane-a 卡不同。第二代 current_best 是 accepted lane-a，behavior 明确来自最后 measured sibling lane-b；不把行为来源伪称为接受关系。fixture 的 generation_outcome=accepted 只是预置模拟评估结果，不是新策略实测。

| 实际请求 | report B | system B | user B | HTTP body B | 12288B 同源离线 user / HTTP B |
|---|---:|---:|---:|---:|---:|
| g1-lane-a | 10239 | 1326 | 72598 | 78846 | 74608 / 80896 |
| g1-lane-b | 10239 | 1326 | 73270 | 79552 | 75280 / 81602 |
| g2-lane-a | 10170 | 1326 | 92495 | 100219 | 94610 / 102394 |

同源8192B替代分别需9465/9465/9880B，均明确不足；10240B重投影的载荷尺寸与实际请求一致。系统段保留既有 public-only 规则，新增事实真正位于 user。每个实际请求23个必要ID全部存在，逐句同时验证 user 和 HTTP 的转义后字节位置、完整 public 对象一致及 provider task-section receipt。卡字节不能替代整个请求字节。

实际 g1-lane-a user 的原样片段（完整五条及逐句 trace 见 [FINAL_USER_EXCERPTS.json](FINAL_USER_EXCERPTS.json)）：

```text
RR_calf_joint: 25548/276259 finite recorded q samples lie inside the nominal USD 5% edge band; 11335/276259 lie outside nominal bounds. Episode-equal median minimum signed margin -0.00101691rad; median range occupancy 0.83349. Unknown samples 0/276259. Effective runtime limits unknown; no saturation or cause claim. Valid episodes 864/864.
c2_rough / robot_0.support.FR_foot: nominal terrain known 40837/40905 reference-point samples; 68 unknown. Episode-equal median mean (point z - nominal height) 0.0673287m. Valid episodes 108/108.
```

来源切换后 g2-lane-a 的 cycle 改为 FL 足：3554/8052 qualified windows、5548/10619 unique cycles、640/864 qualified episodes、0.56s、154071/297511 unique qualified edges、224/864 unknown rhythm；关节改为 FL_calf_joint。不是复用 K1 旧事实。计数、限定、原18槽映射、source hash 和 raw/measurement/full/user/HTTP 引用均存入 [EVIDENCE.json](EVIDENCE.json) 与上述片段文件。

## 独立复算与反例

独立审核不调用新 production measurement/aggregate 函数，只使用原 NPZ、独立 USDA bounds parser 与 triangle-mesh oracle，复算所选 K1 两整句，未重新 qualify 全部历史。

- `RR_calf_joint`：25548/276259 finite q 在名义内侧 5% band；11335/276259 在名义 bounds 外；unknown 0，864/864 有效回合。逐回合 minimum margin 的中位数 `-0.0010169080005465458 rad`，range occupancy 中位数 `0.8334895653490983`。边界由 USDA 独立解析为 `[-2.7227000509895944,-0.8377599051700412] rad`。
- 越界反例 run1/seed194951408/env277/step406，`chunk_000006.npz` 零基 row21，`joint_pos` col11，q `-2.808295726776123 rad`，signed margin `-0.0855956757865286 rad`。仅是名义范围外；effective runtime limits 未记录。
- C2 `robot_0.support.FR_foot`：40837/40905 known，68 unknown＝3 boundary＋65 outside；108/108 有效回合。逐回合 mean(point-z − nominal-height) 的中位数 `0.06732868987251237 m`。
- terrain 正例 run1/env122/step1，origin `[-3.25,-6,0] m`，参考点 `[-3.116915702819824,-6.107839107513428,0.08737929165363312] m`，nominal height `0 m`。
- terrain 反例同 env/step333，`chunk_000005.npz` 零基 row12，`support_pos_w` col1，点 `[0.6999581456184387,-6.361866474151611,1.6008573770523071] m` 为 boundary_band，height unknown，不能填0。这不是足底间隙或真实接触距离。

独立代理另算小例：q=[0,1,10,19,20,-1,21,NaN]，bounds=[0,20]，finite7/8、edge4/7、outside2/7、minmargin−1、occupancy1.1。cycle onsets=[4,24,44,64,84,104]、dt=.02，两个 qualified windows、5 unique cycles、100/108 known edges、period=.4；截短到 n=84 虽有4 complete cycles，缺边界上下文仍0 qualified。terrain 平移 `[12,-8,2.5]` 后 relative height/status 不变；边界与 patch 外都是 unknown。

可复算脚本 [recompute_selected.py](recompute_selected.py) 同时验证所有27个原NPZ SHA、两条句子的聚合数值和真实 witness 的 manifest/run/env/step/channel/chunk SHA/row；压缩原数据 74,694,953B。代理原执行 wall4.679s、CPU4.531s，结果与唯一 writer 再执行输出一并留档。

## 验证与开发中发现的问题

受影响单测共 **119 项通过**（unittest 93.924s，封装命令97.146s），涵盖新 profile、正常 CLI 默认/Genesis、source/profile/budget/cache 切换、completed marker、短/不等长/missing、资格 unavailable 与矛盾拒绝、current/parent/sibling、防止旧最新 lane 缺卡时回退陈旧卡、tamper/replay，以及旧 adapter/report/operator 和 Push pipeline 回归。

独立审查发现并已修复：旧8KB projection 先挡住新路径；新统计缺有效 episode denominator；completed cache 未固定 measurement ref；新 public validator 未复用旧强约束；measurement 重哈希后单位/实体/phase operand 可改；base source domain 可伪换 checkpoint；新 manifest 非原子发布。上述修复各有对应断言/回归。真实 lifecycle 首次又发现 `public_projection` 少了 public.v6 分支，修复后 reference finalizer 至两个首代实际 HTTP payload 成功。

保留失败记录：`budgets_02` 因 reuse 目录父级未建而失败，脚本修复后 `budgets_final` PASS；`lifecycle_01` 因上述 v6 分支失败；`lifecycle_02` 两个 candidate 最终 manifest 路径268字符，Windows sandbox 的 `lstat` 假报 missing，原 candidate 五文件与 preflight 均通过。只将完整 lifecycle fixture 输出根缩短至 `D:/humanoid_agent/.tmp/r1a_l3`，未改生产 sealer/权限/候选内容。沙箱深路径假删除另以正常权限只读 Git status 核对，未真实删除，也不加入暂存。

命令（工作目录 `D:/humanoid_agent/true_RL_Agent`，需现有历史只读档案，不下载模型）：

```powershell
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/R1a/run_validation.py tests --output outputs/r1a/tests_01
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/R1a/run_validation.py budgets --output outputs/r1a/budgets_final
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/R1a/run_validation.py lifecycle --output D:/humanoid_agent/.tmp/r1a_l3
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/R1a/inspect_delivery.py --lifecycle D:/humanoid_agent/.tmp/r1a_l3/artifacts --output outputs/r1a/delivery_final
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/R1a/recompute_selected.py
```

这些输出目录应新建，复跑需换一个不存在的目录；budget wrapper 复用已 SHA 绑定的 `outputs/r1a/budgets_01` measurement，若首次建立则运行 `compare_budgets.py --output outputs/r1a/budgets_01`。所有模型/network/GPU/训练/新rollout计数均为0；CPU历史fixture运行的模拟设备/非公开phase数字不能当作真实硬件或新实验结果。

最终命令成本：单测97.146s/22,303B；冻结预算41.814s/3,000,685B；完整lifecycle710.293s/4,405,920,282B（原历史数组副本和中间证据，不入Git）；完整载荷检查与三档替代37.656s；唯一writer独立原数组复算6.826s CPU6.750s。最终三主命令均 exit0，执行期间记录的源码哈希未变。开发失败与早期小回归的额外成本未混称上述最终成本，保留原日志可查。

## 停止线

代码与测试提交：`ae146e167438f43e4fcee6419185a5b6ea456a8c`（15文件，950 additions / 58 deletions）。真实完整代码 diff 为 `outputs/r1a/CODE.diff`，SHA256 `df4dc98e93cd171eedbd2f2300e82bd4d4700255e87ff146d562e3adf93823e7`，也可用 `git show ae146e16` 查看；下一小提交只含本阶段脚本与证据。

Astra ultra 最终只读开发审查无阻塞：三源的27/27/30个原NPZ SHA、三份 HTTP/user/phase/full、各23必要句逐项一致；独立 stream measurement 重聚合 K1/R4 五新句均吻合（1.279s/1.551s），与原数组 oracle 验证互补。119项测试记录的18个源码hash与提交工作树逐个一致。该审查不代替父级独立验收。

R1a 仅交付 CPU 生产接线、契约、确定性选择、真实 serializer 证据及历史反例。最终状态和成本见本目录验收证据；commit/push 后停在 R1a 等父级独立验收。没有自动启动 R1b，也没有完成实际 per-env limits/runtime terrain、新GPU档案、作者理解、训练收益、完整 G1 肢体覆盖或 E2/E3。
