# P2b1 验收前语义绑定修复

父线程对 `3801a20c1502af013b700e95b5862e3b657353d9` 的首次独立验收为 **FAIL**：
原数组数值、69句、预算和兼容性子范围通过，但六项重新渲染、重新计算ID后的语义互换仍可公开。
本文件记录这一失败后的修复，不将初稿或首次验收改称通过。K5一直未读取、未运行。

修复代码提交：`25d68a69d8e733a2549af7caeac5863c543d0b42`，直接继承3801a20c；
仍在 `codex/push-trajectory-p2b1`，主升级分支没有快进。
状态：**开发复验与两路只读复核通过，等待父线程独立复验放行**。

## 改动与信任边界

仅修改两个Push生产文件，净增40行。没有修改Stair、通用DSL、capture、旧report/runner或数值阈值。

- `trajectory_measurement.py` 固定六量的原pose列、中文标签与单位；扰动向量字段/单位由force或velocity机制定义。
  测量输出复制source、parameters、timing、CONFIG与impulse列表，避免修改待验卡片时连带改变原始依据。
- `build_push_card(records, source_id, *, episodes)` 校验完整记录数，并将每条完整record与原episode重测值比较，
  包括数值、原数组引用、episode/event/window/协议身份；之后才聚合。原有聚合器继续拒绝重复episode及混协议。
- `project_push_card(card, ..., *, episodes, source_id)` 从外给episodes重建固定核心操作数，逐域比较后才公开。
  不以待验card中的参数、协议、窗口或来源作为期待值；source_id亦由调用方明确给出。
- loader根据外部锁定的`expected_sources`重新经过adapter读取原档案，并保留原manifest/chunk SHA及身份准入。
  无缺档fallback、隐式路径查找、全局信任缓存或额外签名机制。

关键原文：

```python
for record, episode in zip(records, episodes):
    expected = measure_push_episode(episode)
    if record != expected:
        raise ValueError(f"Push measurement operands differ from raw episode {episode.episode_id}")
```

```python
records = [measure_push_episode(e) for e in episodes]
expected = _card_core(records, source_id)
for name, value in expected.items():
    if card[name] != value:
        raise ValueError(f"Push card {name} semantic operands differ from raw episodes")
```

该生产重测属于来源/操作数绑定校验，不是独立数值oracle。API要求调用方提供合格原episodes；
任意手造或任意修改`NumericEpisode`本身不构成档案认证。生产CLI与loader负责经adapter获得它们。
私有render仅形成候选文本，不能代替公开投影验证。

## 原六项真实反例复验

`audit_semantic_binding.py`读取同一六份K4锁定档案，为全部反例传入真实原episodes；
所有反例都重新render、重新算ID。拒绝均为语义不符的`ValueError`，没有靠缺少新参数的`TypeError`过门。

| 原反例 | 修复结果 |
|---|---|
| force聚合height/speed交换 | 拒绝：`Push card cohorts semantic operands differ from raw episodes` |
| velocity r1.e46向量单位m/s改N | 文案从协议仍派生m/s，错误event字段因examples不匹配被拒绝 |
| 同事件pre/post互换 | 拒绝：examples与原episode重建窗口不同 |
| e46方向引用改到e34 | 拒绝：examples原数组引用不匹配 |
| velocity卡换绑force协议hash | 拒绝：domain与原来源协议不同 |
| 192份force evidence交换height/speed并重新record ID | 建卡前拒绝：measurement operands与原episode不同 |

第一例仍可在私有render形成原错误候选句，但它无法进入公开投影：

> 全程根部平面速度模：先逐回合取有效记录均值，再回合等权平均为0.380818m/s；回合均值中位数0.38335m/s，有效192/192回合，未知0回合；有效/未知记录46942/0。

正确原速度均值仍为0.254063m/s。全部候选句、异常与真raw上下文标记保存在
`outputs/p2b/semantic_binding_fix/semantic_swaps.json`。
合法两协议原卡均通过；新增测试另验证合法signed轴变换配套新raw后通过，而旧卡不能套用新raw。
force-v2实际capture→adapter→card测试仍通过，未将signed能力写死为velocity。

## 原内容与数据保持

正式CLI在临时目录重新构建两协议各192回合，逐字节比较后清理临时副本；两路只读复核之一又只在内存独立重建。
`evidence.jsonl/policy_card.json/policy_card_zh.md/public.json`各两份，共 **8/8字节一致**。
原69句、所有record/card/claim ID、预选代表及见证均保持；公开仍为12265/11447B、34/37与32/32句，20个必保槽保持。

原始6manifest+18NPZ共24文件再次核SHA一致，NPZ总3,486,938B。
数值规则、dt、dwell、窗口、阈值、目录版本和模板版本均未改变。
原`FREEZE_ZH.md` SHA仍为`f65251e91b493dc13d3c00e9dad06813363ceeecf10dea78e6b669b88c92e9a5`。

最终产物仍只有`outputs/p2b/{k4_force_v1,k4_velocity_v2}`一套。
两份manifest仅更新`code_sha256`与`wall_seconds`，内容产物hash均未变；旧manifest小文件保存在修复证据目录。
原`verification.json`及旧日志是3801a20c历史开发证据；当前修复以
[SEMANTIC_BINDING_VERIFICATION.json](./SEMANTIC_BINDING_VERIFICATION.json)为准，不把旧freeze/hash解释为当前版本。

## 最终验证范围

完整argv、exit、stdout/stderr SHA及实际Python峰值内存见
`outputs/p2b/semantic_binding_fix/command_receipts.json`。日志原文：

```text
Push新专项及语义测试：Ran 37 tests in 5.136s — OK
旧Push八模块：Ran 77 tests in 43.304s — FAILED (errors=1)
Stair/shared五模块：Ran 111 tests in 28.613s — OK
```

本次真实执行225个唯一测试：224 PASS、1 ERROR。
唯一ERROR仍是既有`test_two_native_snapshot_streams_close_at_each_env_terminal`缺少`omni.replicator`，未改旧路径。
按测试ID合并历史结果为718唯一项＝709 PASS＋8 SKIP＋1环境ERROR；未重新执行历史其余部分，不能称全广域最终版本全绿。
最终测试日志在修复目录；初次37项快速运行亦通过。本次新增audit脚本首次执行漏传numeric reference schema，
在进入反例前失败；修正审计脚本后最终真实六反例通过，初次traceback保留于会话工具记录。

直接复验命令（项目目录）：

```powershell
.venv/Scripts/python.exe -B -m unittest tests.unit.test_push_trajectory tests.unit.test_push_trajectory_semantics -v
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2b1/audit_semantic_binding.py --output <new_receipt.json>
```

父线程独立raw oracle原脚本按原字节复制后重新运行，未修改科学检查：384回合、六量/signed/时序/引用/分母等PASS，
不导入项目代码。Stair frozen247/309再次PASS（4168/45157 scalar）；801句卡三内容hash保持，
原数组核866ep、69280 metric、8400 aggregate、32042 refs通过。P2a三来源33产物SHA不变。
`git diff --check`通过；未重跑三来源GB全构建。

当前192ep建卡函数时间force0.831s、velocity0.892s；完整命令1.161/1.216s；Python峰值约67.5/69.4MB。
这是本批CPU成本；显式原episodes重测增加一次测量，未增加缓存或泛化框架。

## 冻结更新与后续门

旧规则代码冻结SHA `c9e8301c6e6e72575d42f9c0267aca9c4582ceaf12b8c0afed1ae42750c75e78` 已由新版本取代。
新`RULES_CODE_FREEZE.json` SHA：
`ec8158315d79fba1ff2f9122016de5f1eb5af5f7a0de96cc6f09dc5cda8ff5e8`。
新记录绑定修复commit/tree、22个文件hash，并显式记载首次父验收FAIL、验收前修复、K5从未读取/运行。

K5将来仍只能独立首次采192ep并盲核一次；本次修复发生在它之前，不存在K5首次通过记录。
P2b2正常runner/profile、版本分发、receipt、最终payload与legacy接线仍未实施；未启动K5/P3。
