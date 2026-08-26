# Phase 0 独立审计（第三方复核）

> - 审计日期：2026-08-26
> - 审计对象：`PLAN/Variant_construction/P0/` 全部交付物 + 主计划 Phase 0 章节
> - 审计基线：`f3593567b3901453f7dbf80dd6c450554b957fb2`
> - 方式：重跑验收器、独立交叉比对、独立硬编码穷举扫描、设计并运行**篡改测试**、重跑测试子集
> - 立场：本文不替代 P0 自己的验收报告，只做独立复核

---

## 0. 结论

**Phase 0 的 GO 成立，执行质量高于此前多数阶段，且没有过度复杂化——反而偏精简。**

但发现 **1 个可演示的实质缺陷**、**1 个范围性遗漏（会比 Go2 更早阻塞）**、**1 组备料不足（会逼 Phase 1 重跑服务器探针）**，外加若干不对称与工程脆弱点。全部修补估计 **半天以内**。

| 编号 | 问题 | 严重度 | 何时爆炸 |
|---|---|---|---|
| **F1** | `execution_profiles` 是自指验证，可篡改而不被发现（已实测） | **高** | Phase 1 从错误事实建 RobotProfile |
| **F2** | 债务清单只覆盖"机器人轴"，漏了"考卷网格形状轴"（110 处 / 11 模块） | **高** | Phase 3A / Phase 4，比 Go2 更早 |
| **F3** | Phase 1 所需 RobotProfile 六项事实只齐两项，但服务器探针已经跑过了 | 中 | Phase 1 必须再上一次服务器 |
| F4 | 三个 profile 字段集不一致；Go2 资产哈希只覆盖 8 KB 包装文件；Push schema 哈希是占位标签 | 中 | Phase 1/5 |
| F5 | 验收器依赖测试私有 API、无测试覆盖、且含被设计成会变假的断言 | 中 | Phase 3 起验收器永久失败 |

---

## 1. 我独立复核**通过**的部分（这些不用再查）

| 复核项 | 我做了什么 | 结果 |
|---|---|---|
| 验收器是真复算还是抄旧报告 | 通读 `verify_phase0_snapshot.py` 全部 409 行 | **真复算**：`local_contracts`、reward ABI、source hashes、runner argv 全部现场从代码解析/materialize |
| 验收器当前状态 | 实跑 | `status=pass`，1.4 s |
| 回归声明（子集） | 独立跑高信号子集（composition / genesis / evaluation binding / stair entrypoint·ABI·protocol·eval·semantics·evidence·runtime_env / push task·runtime） | **196 tests OK，6 skipped，0 失败** |
| 回归声明（全量） | 独立跑 `unittest discover -s tests` 全量套件（830 s） | **1483 tests OK，29 skipped，0 失败**——与 P6-D 记录的 1483 逐数对上 |
| 未跟踪产物计数 | `git status --porcelain -uall` | **1526 全量 / 1520 排除 P0**，与报告逐字对上 |
| 未改动他人文档 | 重算我自己那份审计文档的 sha256 | `b0b6b6bc7cab6910…` 与快照记录**一致** |
| 硬编码清单是否穷举 | 独立扫描：`grep -rln "foot\|feet\|ankle" src/true_rl_agent` | **11 个产品模块，全部已在 H1–H4 中**，无遗漏 |
| 快照事实是否与仓库权威一致 | 逐字段比对 G1-37 payload ↔ `protocol_v1.policy_compatibility` | joint_names(37)、body_names(44)、action dim/scale/selector、sim_dt/decimation/control_dt、asset id **全部一致** |
| 是否过度复杂化 | 清点交付物 | 6 个文件 / 2 人日 / 一个只读脚本；无新服务、数据库、框架、调度器 |

**特别值得肯定的两点**：

1. 把 G1-37 与 G1-29 拆成两个执行身份，并明确"只共享 family 标签、不共享执行身份"——这解决了我上一轮指出的 M2（37 vs 29 DoF），而且解决得比我建议的更彻底。
2. `evidence_boundary` 把"当前产品字节"和"冻结正式实验字节"分成两条独立身份，并写进事实否定表。这类区分正是这个项目最容易出错的地方。

---

## 2. F1【高】`execution_profiles` 是自指验证——已实测可篡改

### 现象

验收器对 `execution_profiles` 只做两件事：

```python
digest = canonical_json_hash(payload)
_assert_equal(digest, profile["fingerprint_hash"], profile["profile_key"])  # 自己对自己
...
if len(fingerprints) != len(set(fingerprints)):                              # 唯一性
```

**没有任何一句把 payload 与外部权威比对。** 只要 payload 内部自洽，任何内容都能通过。

### 实测证据

我把 G1-37 的 payload 改成明显错误的值，按验收器自己的规则一致地重算 fingerprint 和 snapshot hash，然后调用 `verify()`：

```
篡改：joint_names[0]  = "FL_hip_joint"   （改成四足关节名）
      action.dimension = 12               （改成 Go2 的动作维度）
      control.decimation = 8              （改掉控制频率）

验收器返回：status = pass
```

测试脚本：`<scratchpad>/tamper_test.py`（只读原快照，写临时副本，未触碰仓库文件）。

### 为什么这是最该修的一条

- **这块恰好是 Phase 0 唯一的"新"信息。** `local_contracts`、reward ABI、source hashes、runner argv 在仓库里都有源，验收器也确实重算了；只有 execution profile 是这次新采集的，而它是唯一没有独立验证的。
- **Phase 1 正是从它建 RobotProfile 的。** 一个没人能发现错误的事实源，会把错误直接带进矩阵的地基。
- **而且它本地就能校验。** G1-37 的 joint/body/action/control 在 `configs/tasks/stair_protocol_v1.json: policy_compatibility` 里全都有，而且那份才是**运行时真正使用的权威**（`isaaclab_reward_runtime.py` 就是从 `robot_schema.joint_names` 解析关节 id 的）。两者若不一致，运行时听 protocol 的，快照会静默地错。

### 修法（约 10 行）

```
verify() 增加：
  1. G1-37 payload.joint_names / body_names / action / control
     ↔ protocol_v1.policy_compatibility 逐字段断言
  2. 三个 payload 的必需字段集一致性校验（见 F4）
  3. 远端事实（asset sha256、Go2 probe、G1-29 schema）本地确实无法复算 ——
     这部分不要沉默，另写一个服务器侧 re-probe 脚本，标注"需在服务器执行"
```

第 3 条要说明：验收报告"不每次 SSH/启动 Isaac"的设计决策是**对的**，不该改。问题不在于不复算，而在于**没有把"本地可校验"和"只能在服务器校验"这两类事实分开标注**，导致读者会以为整份快照都被验证过。

---

## 3. F2【高】债务清单漏了"考卷网格形状"这根轴

### 现象

`OWNERSHIP_SCOPE_AND_HARDCODES_ZH.md` 的 H1–H4 在**机器人/双足**这条轴上是穷举完整的（我独立验证过）。但当前 Stair 还有一条同等规模的耦合完全没进清单：**35-cell 评测是 `7 阶高 × 5 踏深` 的笛卡尔网格，整条投影链都按网格形状写死。**

独立扫描结果（`by_tread` / `tread_depths_m` / `anchor_cells` / `*_grid` / `eval_bins`）：

| 模块 | 引用数 |
|---|---:|
| `prompt_adapter.py` | 50 |
| `generalization_contract.py` | 21 |
| `evaluation_profile.py` | 12 |
| `public_projection.py` | 11 |
| `plugin.py` | 5 |
| 其余 6 个模块 | 11 |
| **合计** | **110 处 / 11 个模块** |

其中两处是**结构性硬不变量**，不是命名问题：

```python
# evaluation_profile.py:222  —— 强制笛卡尔积
expected_envs = len(base_protocol.eval_bins) * len(treads) * episodes
if _positive_int(payload.get("eval_num_envs"), "eval_num_envs") != expected_envs:

# evaluation_profile.py:136-139 —— 假设 cells 按网格顺序排列
for cell in self._cells[:: len(self.anchor_cells)]
```

### 为什么比 Go2 债务更紧急

主计划的关键路径是 **3A → 4**：

- Phase 3A 的 C1 套卷是 **≤18 题的非网格集合**（含 hard corners）→ 直接撞上 `evaluation_profile.py:222` 的笛卡尔不变量；
- Phase 4 是第一次真实 LOOP，反馈质量决定候选质量 → 撞上 `prompt_adapter.py` 的 50 处网格投影。

**Go2 债务在 Phase 5 才爆炸，网格债务在 Phase 3A 就爆炸。** Phase 0 唯一没画出来的那条债务，恰好是最先要还的。

### 这是门的范围问题，不是执行问题

Phase 0 的验收门原文是"明确列出所有 **G1-37/双足** hardcode"。按字面，他们完成了。
**是门写窄了**：Phase 0 的实际目的是"把迁移债务定牢"，而迁移债务不止机器人轴一条。

### 修法

补一节 **H5：评测/反馈的网格形状耦合**，与 H1–H4 平级，列出上面 11 个模块和两条硬不变量，并注明根治位置在 Phase 3A（evaluation suite 改为 cell 列表）与 Phase 3A 末（反馈投影）。约 1 小时。

---

## 4. F3【中】Phase 1 要的六项事实只齐两项——而探针已经跑过了

主计划 §3.2 规定 `RobotProfile` 拥有：**robot identity、物理尺度、执行器/控制能力、默认姿态、显式 semantic body/joint roles、可用传感器与挂载能力**。

对照快照：

| RobotProfile 需要 | 快照有吗 | 说明 |
|---|---|---|
| robot identity | ✅ | family、asset identity + sha256、上游 revision，三个 profile 都全 |
| 控制时序 | ✅ | sim_dt / decimation / control_dt，三个都全 |
| **物理尺度**（站立/髋高、腿长、足尺寸） | ❌ **完全没有** | 快照全文无 `hip_height` / `standing` / `base_height` |
| **执行器**（PD 增益、力矩/速度上限） | ⚠️ **只有 Go2** | Go2 有 `DCMotorCfg / effort 23.5 / velocity 30`；**G1-37 与 G1-29 完全没有**；且 Go2 也缺 stiffness/damping |
| **默认姿态** `default_joint_pos` | ❌ | 快照全文无 `default_joint` / `init_state` |
| **传感器与挂载** | ❌ | 无 `height_scan` / `scanner` / `prim_path` / `contact_sensor` / `resolution` |
| semantic roles | ❌（合理） | 这是 Phase 1 的工作，Phase 0 不做是对的 |

### 为什么这条值得单独提

快照自己写着 `probe_mode: "headless single-scene articulation and read-only asset read"` ——
**他们已经在服务器上把三个 articulation 都实例化了。** 那意味着：

- `robot.data.default_joint_pos` → 默认姿态
- `robot.actuators` → 执行器分组、stiffness、damping、effort/velocity limit
- `robot.data.default_root_state[:, 2]` + 默认姿态下的 body 位置 → 站立高度、腿长
- scene 里的 `height_scanner` / `contact_forces` cfg → 挂载点、pattern 分辨率、history length

**这些全都是同一次探针里一次属性读取的事。** 现在 Phase 1 必须为此再上一次服务器、再排一次 GPU。

### 三条最关键的缺失，及其下游后果

1. **物理尺度** —— 这是"按机器人缩放几何"的唯一输入。主计划 §4.1 自己写了"RobotProfile 提供 standing/base height 等尺度事实用于生成初始范围"，但 Phase 0 一个数都没采。没有它，Phase 5 定不出 Go2 的合格阶高范围。
2. **执行器参数** —— 决定机器人**物理上能不能爬某个阶高**。缺了它，Go2 的难度资格化只能靠试错训练（每次约 1 GPU 小时）。
3. **height scanner 的 pattern**（分辨率 0.1 m、size 1.6×1.0）与**观测噪声幅度** —— 这两个数决定 C2 roughness 属于"可感知"还是"盲态"。当前仓库里 `observation_corruption_enabled` 只是一个开关，**具体噪声幅度继承上游默认值、从未冻结**。上游一改版本，训练条件就变而所有指纹不变。Phase 0 是冻结它的自然时机。

---

## 5. F4【中】三处不对称与占位

### 5.1 同一 schema 下三个 payload 字段集不同，且无校验

三个 profile 都声明 `schema_version = "…p0_execution_fingerprint.v1"`，但：

```
G1-37 keys: … asset, joint_names, body_names,            action, control
G1-29 keys: … asset, joint_names, body_names,            action, control
Go2   keys: … asset, joint_names, body_names, actuator,  action, control
                                              ^^^^^^^^ 只有 Go2 有
```

一个 `v1` schema 却没有必需字段集，验收器也不校验字段集。结果是"G1 没有执行器信息"这件事**在结构上看不出来**——它看起来像是一个合法的 v1 payload。

修法：在验收器里加一句必需键集断言；把 actuator 列为必需字段（G1 两个 profile 需补采）。

### 5.2 Go2 的资产哈希只覆盖一个 8 KB 包装文件

| 资产 | size_bytes | 含义 |
|---|---:|---|
| `g1_minimal.usd` | 22,195,512 | 完整资产，sha256 覆盖真实几何/物理 |
| `go2.usd` | **8,142** | 这是 payload 引用型 USD，**sha256 不覆盖被引用的几何与物理** |

也就是说：**G1 的资产身份是强的，Go2 的是弱的。** 如果被引用的 payload 变了，`go2.usd` 的 sha256 不会变。

修法：Phase 5 前对 Go2 补一个"解析后 articulation 指纹"（关节/刚体/质量/惯量/限位的规范化哈希），而不是只哈希入口文件。这也正好和 F3 的执行器补采一起做。

### 5.3 Push 的 schema hash 是占位标签

```python
# push_recovery/composition.py:33-34
PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH = "push_recovery_obs_v1"
PUSH_RECOVERY_ACTION_SCHEMA_HASH = "unitree_actions_v1"
```

这不是内容哈希，是字符串标签（快照里对应 `<str:20>` / `<str:18>`，而 Stair 侧是 `<str:64>` 的真 sha256）。

后果：**厂商包 `unitree_rl_lab` 的 29-DoF 观测/动作若发生变化，Push 侧的"契约身份"不会变。** 而 Phase 1 的核心验收正是"两个 legacy G1 execution profile 都真实接入同一角色词汇"——其中一个的身份是假的。

修法：Phase 1 迁移 Push 时必须先用真实内容哈希替换；在此之前，快照里应显式标注这两个值为 `placeholder_not_content_bound`，避免被当成已冻结契约。

---

## 6. F5【中】验收器自身的三个工程脆弱点

### 6.1 依赖测试的私有 API

```python
from tests.unit.test_stair_entrypoint_materialization import StairEntrypointMaterializationTests
case = StairEntrypointMaterializationTests(methodName="test_materialized_jobs_use_stair_entrypoints_and_pass_pure_preflight")
_, jobs = case._materialize(root, "phase0-snapshot-candidate", execution_device="cuda:0")
```

Phase 0 的权威验收脚本，通过**一个测试类的私有方法**来获得 runner argv。该测试一旦重构（改方法名、改 `_materialize` 签名），验收器就会坏掉——而且是静默坏掉。

正确做法：argv 应从产品 API（`IsaacLabRslRlBackendAdapter` + job materialization）直接取，测试和验收器各自消费同一个产品入口。

### 6.2 没有任何测试覆盖这个脚本

`grep -rn "verify_phase0_snapshot\|Variant_construction" tests/ src/` → **零命中**。

一个 409 行、被指定为"常规漂移检查"的脚本，完全在测试体系之外。它会安静地腐烂。
（顺带：我第一次用 stdin 方式运行它时，因为多进程 spawn 重入直接失败——说明它对执行方式也敏感。）

### 6.3 三条断言被设计成会变假

```python
STAIR_ROUGHNESS_TOKENS = ("roughness", "surface_noise", "height_noise", "random_rough")
return {
    "production_go2_integration": "go2" in production_text,
    "production_stair_surface_roughness": any(token in stair_text ...),
    "production_hexapod_integration": "hexapod" in production_text,
}
```

这三条断言的内容是 **"当前产品里还没有 Go2 / roughness / 六足"**。

- Phase 3 一落地 roughness → 第二条翻转
- Phase 5 一接入 Go2 → 第一条翻转

也就是说，这个"常规漂移检查"**从 Phase 3 开始会永久失败**。而且它是纯文本扫描（283 个文件 / 5 MB），一句注释里出现 `go2` 就会误报。

修法（二选一）：

- 把这三条标注为 **"Phase 0 一次性基线事实"**，从常规验收器里移出，或
- 改成"相对 Phase 0 基线的差异报告"：允许出现，但必须说明是哪个 Phase 引入的。

---

## 7. 过度复杂化？没有

明确说一句：**这个 Phase 没有过度设计。**

- 交付物 6 个文件、2 人日、一个只读脚本；无新服务、数据库、调度器、安全框架 —— 与自己写的"本次明确不做"清单完全一致；
- 9 个验收门看着多，但每个门都对应一次真跑过的检查，不是纸面条款；
- 从 codex 只采纳了 4 条思想、明确列出 4 个不采纳的子系统 —— 这种"显式拒绝"比默默不用更有价值。

唯一"仪式感略大于价值"的是 `pre_edit_untracked_inventory` 里的四个哈希：其中 `path_size_mtime_sha256` 包含 mtime，跨机器/跨检出不可复算，而验收器根本不检查它。留作一次性证据没问题，但不要把它当成可复验项。

---

## 8. 建议的最小修补（合计半天以内）

按优先级：

| # | 动作 | 估时 | 对应 |
|---|---|---|---|
| 1 | 验收器加"G1-37 payload ↔ protocol_v1 逐字段"断言 + 必需字段集校验 | 30 min | F1 / F4.1 |
| 2 | 快照里把事实分成 `locally_verifiable` / `server_only` 两类并标注 | 30 min | F1 |
| 3 | 补 **H5：评测网格形状耦合**清单（11 模块 / 110 处 / 2 条硬不变量） | 1 h | F2 |
| 4 | 服务器补一次只读探针：默认姿态、执行器（三个 profile）、尺度事实、传感器挂载与 pattern、观测噪声幅度、接触传感器 history | 2 h | F3 |
| 5 | Push 两个占位 schema hash 标注为 `placeholder_not_content_bound`，Phase 1 必须替换 | 15 min | F4.3 |
| 6 | negative scan 三条断言改标为"Phase 0 一次性事实"或差异报告 | 30 min | F5.3 |
| 7 | runner argv 改从产品 API 取，不再依赖测试私有方法；并给验收器加一个 smoke test | 1 h | F5.1 / F5.2 |

第 4 条最值钱：它决定 Phase 1 需不需要再排一次 GPU。

---

## 9. 一句话

> Phase 0 做得扎实，GO 站得住。
> 但它**验证了所有本来就能验证的东西，唯独没验证自己新采的那部分**（F1）；
> 它**穷举了机器人轴的债务，却漏了会更早爆炸的考卷形状债务**（F2）；
> 它**已经把服务器探针跑起来了，却没顺手把 Phase 1 要的料带回来**（F3）。
>
> 三条都不需要返工，只需要一次半天的补采与补断言。

---

## 附：本次审计运行过的命令

```bash
# 验收器复跑
./.venv/Scripts/python.exe PLAN/Variant_construction/P0/verify_phase0_snapshot.py     # pass, 1.4s

# 高信号回归子集
python -m unittest -q tests.unit.test_composition tests.unit.test_session_genesis \
  tests.unit.test_evaluation_binding tests.unit.test_stair_entrypoint_materialization \
  tests.unit.test_stair_complete_reward_contract tests.unit.test_stair_protocol_v2 \
  tests.unit.test_stair_evaluation_profile tests.unit.test_stair_evaluation_semantics \
  tests.unit.test_stair_episode_evidence tests.unit.test_stair_isaaclab_runtime_env \
  tests.unit.test_push_recovery_task tests.unit.test_push_recovery_runtime_adapter
# → Ran 196 tests, OK (skipped=6)

# 全量回归
python -m unittest discover -s tests -q
# → Ran 1483 tests in 829.973s, OK (skipped=29)

# 未跟踪产物核对
git status --porcelain -uall | grep -c '^??'                       # 1526
git status --porcelain -uall | grep '^??' | grep -vc 'P0/'         # 1520

# 独立硬编码扫描
grep -rln "foot\|feet\|ankle" src/true_rl_agent --include=*.py     # 11 模块，全在 H1-H4
grep -rn "by_tread\|anchor_cells\|_grid\|eval_bins" src/true_rl_agent  # 110 处，不在清单

# 交叉比对与篡改测试
<scratchpad>/tamper_test.py                                        # 篡改后仍 status=pass
```
