# P2b1 Push 根部离线迁移交接

更新：3801a20c的首次父线程验收因语义互换反例FAIL；修复代码25d68a69已完成开发复验，
当前结果与新冻结版本见[验收前语义绑定修复](./SEMANTIC_BINDING_FIX_ZH.md)。下文保留原开发记录，
不将原自测/只读审查PASS视为父线程放行。

基线 `70f0c51176134b63e637414641dcf981456d8da8`；分支 `codex/push-trajectory-p2b1`。
先冻结提交 `e6350588`，代码提交 `6fa1206ac6a26a5c6dcc40d8c1cd0fa55f3065db`。
状态：**开发自测及两路只读审查完成，待委托父线程独立验收放行**。
P2b1a 的规则/来源冻结、P2b1b 的实现、P2b1c 的验证分别留证；不进入 K5/P3。

## 本轮实际完成

正式 CPU CLI `scripts/build_push_behavior_card.py`：六份锁定的 public manifest →
任务旧 reducer 验证 → Push adapter → 根部测量/完整稳定时序/扰动窗口/恢复队列 →
中文卡片 → 有限 public projection。每协议各192回合，所有有效行参与，未换预选样本。

新代码只在 Push 增加三个模块及一个 CLI；通用层、Stair、旧 Push capture/decoder/report/runner
生产文件均未改。生产新增683行（adapter113、measurement243、card296、CLI31）；
测试352行，独立 raw oracle304行，验收编排脚本113行。两轮顺序 NPZ 读取用于旧任务契约准入和
新测量；K4小档案无需新增缓存/DSL/框架，未做无关重构。

复用 NumericEpisode 身份/原数组定位、PredicateTrack/interval_rows、distribution。
Push 自有 `push.root_measurements.v1`、`push.root_behavior.card.v1`、
`push.root_behavior.public.v1`、`push.root_zh.v1`；没有冒用 Stair P0 四足 catalog。

输出唯一最终副本在 ignored `outputs/p2b/{k4_force_v1,k4_velocity_v2}`：
`manifest.json`、`evidence.jsonl`、`policy_card.json`、`policy_card_zh.md`、`public.json`。
`public.json`是离线投影审计件，不是已接通的作者输入。每卡20个必保句，整句装入12288B预算。
force保留34/37句、12265B；velocity保留32/32句、11447B（UTF-8规范JSON，不含文件尾换行）。
force省略r3.e53第二事件、r3.e58两事件的3条可选细节；完整卡/证据仍保留，省略ID与原因可查。

**P2b2 尚未实施**：正常 composition/profile 与 session 身份、runner释放GPU后的CPU finalizer、
任务 full/public 版本分发、phase/generation receipt、最终作者 payload 本地捕获、legacy正常回放。
本轮没有改这些入口，也没有以 sidecar 宣称接线完成。

## K4 来源和能力

共同 checkpoint：`1495390b192b913432b8472cb150e2db07fdc2eb811eeabe4ab87c577b24e0ff`。
这是同一权重的两个协议，不能算两个lineage，不能合池。完整身份和chunk SHA见
[source_lock.json](./source_lock.json)，均由已授权 SSH 只读下载，未修改远端文件/环境/secrets。

| 来源 | 三份manifest SHA256（run 1→3） | 可用量 |
|---|---|---|
| K4 force numeric v1 | `df5c5df7c34b464d337d030ac3bdcc29652c5dbc3bc54e14d42b5cfb762a5acb`；`d046a312e7513c05cb10250b7ca7f3c7aa0f949c65b758c1f3d261ac3e6d738e`；`88cd2c660d822853f416b19f056a34f7bdd5af0b9f82443d77bd4d45ce31d9f7` | 192ep、46942有效格、9NPZ、1342648B |
| K4 velocity numeric v2 | `07b09b76ac2127d430a8458f0fda4ee3970ba23ea47cf8cb8aeba1817e1469f6`；`6e37742005beb9af385d11ec54b7b67f349889144e2f0a8d714bd4f48bc901be`；`1882c39533447701852e22645a3f8482f0183ab3efa2ce271749934f166f1188` | 192ep、58176有效格、9NPZ、2144290B |

force协议 `8404a906a0c19ba762d7e0e416cf920c92fa3bf3cd957ef6945c45688f6fa7f7`；
velocity协议 `39f0d453a04e0c1a059395597b6a54aa3715dd739edd394b78f90641217f76e1`。
v1有根部模量、姿态、扰动事件方向，根部signedXY缺失；v2有根部世界signedXY。
schema能力和mechanism独立；新增实际调用capture的CPU force-v2合成夹具全链测试通过，未写死v1=force。
两者均无足接触/步态/支撑/关节/逐步动作力矩奖励。评测reward=0是旧stub，训练reward不在此重算。

## 关键语义和源码原文

完整生产差异保存在 `outputs/p2b/key_diff.patch`，可对照代码提交。

原生dtype恢复条件保留 inclusive 边界；没有用缺测填零：

```python
threshold = np.asarray(p[name], dtype=pose.dtype)
passed = value >= threshold if column == 0 else value <= threshold
tracks.append(np.where(np.isfinite(value), passed.astype(np.int8), -1))
```

force完成、验证和恢复资格分开，位移不会变成force失败条件：

```python
eligible = (complete & a["event_verified"])[:, :planned].all(axis=1)
verified = ~a["unplanned_wrench"] & a["event_verified"][:, :planned].all(axis=1)
anchor_seen = bool(np.any(complete[:, :planned].all(axis=1) & (episode.steps >= anchor)))
```

恢复窗 `0<step-anchor<=window`；dwell≥required首次锁存；终点stable、任务失败、恢复锁存分别输出。
velocity事件步是写入前读数；force during=(start,end]，post从end+1；pre可能含此前扰动。
`N×dt`仅记录格时长，`(N−1)×dt`是观测跨度；速度格积分不等于精确路径或接触机制。
未观测格和已观测但unknown格分列；全回合有效均值分布与稳定格池化分别报告。

绑定投影不是只重渲染一遍句子：

```python
core = {k: v for k, v in card.items() if k not in {"card_id", "public_report", "projection_audit"}}
if card["card_id"] != stable_id("push-card", core):
    raise ValueError("Push card content/identity hash mismatch")
```

`load_push_artifacts`再核外部manifest SHA、expected_sources、四产物SHA，按完整episode evidence
重建card和public。P2b2仍须把expected_sources绑定正在运行的phase/policy，不把路径当授权。

## 原数组到测量到中文

`outputs/p2b/raw_to_sentence.json`包含全部预选和补充见证的chunk SHA/row/env、原始pose/signedXY、
完整测量引用、句子ID和是否进入public。预选force7、velocity5；加冻结规则见证后各10/6回合。
没有事后换成好看例子。两路只读审查亲核全69条中文、其中19个例证事件metadata pointer与向量。
root亲读两套中文及全部16个样例终点/恢复原数组表示；oracle对全384回合直接np.load复算。

| 真实例 | 原数组/测量 | public句子含义与引用 |
|---|---|---|
| force r1.e0 | anchor62；step63速度0.168657超过阈值；step64..79连续16格达标，step79锁存 | 窗81/81格、稳定80、dwell16/16、success_recovered；claim `f2e6e18dcaa469a776494805b219c3126ad99ecef6de23281d6796e219e65a19` |
| force r1.e37 | 窗63..143完整81格、稳定0、dwell0；终点160 roll=-0.868107，fall_side | 完整窗未达dwell见证，未伪装截尾；claim `078f0645f6d61a4acc6487dc99e19bc3e0e32ee445cad73261a15fdb03e61c43` |
| force r1.e32 | 终点73高度0.173851<0.18；恢复窗只观测11/81格 | 明示未观测70格，已观测中unknown0；claim `7f7f556b7dca8c2e6ba2b752eae67a17c9ac7933ff440688770b2cc8e938acdd` |
| velocity r1.e46 | 52..132窗稳定0/81；终点303速度0.002463906、高度0.347651809等满足stable | 终点stable=True但任务not_recovered；事件后方向投影0.247262m/s，world XY而非机器人方向；见raw trace中的4条public关联句 |

全队列force：141成功/51失败；192序列=141恢复+2完整窗无dwell+38提前终止截尾+11锚点未观测。
velocity：191成功/1失败；240事件=239恢复+1完整窗无dwell，终点稳定192/192。
真实K4未出现“恢复后终点不稳定”，两卡明确no_witness；合成反例单独覆盖，不伪称真实见证。
force稳定回合等权均值0.8287855034881773，格池化40717/46942=0.8673895445443313；
velocity对应0.9441522277227722、54927/58176。分母不同是预期，不混成单个比例。

## 验证结果与原始日志

完整argv/exit/stdout/stderr SHA见 `outputs/p2b/command_receipts.json`；主核验结果及文件SHA见
[verification.json](./verification.json)。常用可复验命令（项目目录）如下：

```powershell
.venv/Scripts/python.exe -B scripts/build_push_behavior_card.py --source-lock PLAN/feedback_sys/NEW_SYS/P2b1/source_lock.json --group k4_force_v1 --output <fresh_output>
.venv/Scripts/python.exe -B scripts/build_push_behavior_card.py --source-lock PLAN/feedback_sys/NEW_SYS/P2b1/source_lock.json --group k4_velocity_v2 --output <fresh_output>
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2b1/audit_replay.py
.venv/Scripts/python.exe -B -m unittest tests.unit.test_push_trajectory -v
```

- 独立oracle不导入translator/reducer，直接验证6manifest/18NPZ与384回合：六量、signedXY、stable runs、
  窗口边界、dwell/anchor/latency/中断/截尾/终点、分面均值分位数、两类权重、引用解析与预选规则。
- 最终新增专项：`Ran 30 tests in 4.942s — OK`（直接/测量包装运行时间略有差异，原文以最终日志为准）。
  signed swap、v1方向缺失、force+signed、dwell−1/=边界、gap/unknown、终点缺失、condition错配、
  source/协议/sealed错配、足机制句拒绝、预算、0成功、父代缺失、落盘加载/篡改均覆盖。
- 广域回归在最后文本/loader修复前已启动：`Ran 710 tests in 780.803s — FAILED (errors=2, skipped=8)`。
  第一项是已修的新增loader句序问题，最终30项重新跑过；第二项是旧
  `test_two_native_snapshot_streams_close_at_each_env_terminal` 因缺`omni.replicator`失败。
  单项再次复现同一环境错误；相关3文件与基线只可能有CRLF差异，规范化后源码相同，未修旧路径。
  **按测试ID去重、以最终专项覆盖旧结果：711项=702 PASS+8 SKIP+1环境ERROR。不是一次广域全绿。**
- 8 SKIP：4个需要目标Isaac源码闭包的Stair会话测试、3个Linux进程组/parent-death测试、1个文件symlink测试。
- frozen247/309全部PASS，4168/45157 scalar；冻结card三文件逐字节SHA一致；801条claim raw核查PASS。
  P2a K1/K2/K3的33个既有生产产物SHA一致。未重建/复制三份GB numeric产物；未改P2a测量语义。
- `git diff --check`通过。项目venv无ruff，本轮未安装lint工具；Python导入/执行和上述测试已验证语法。

开发中修复记录：初版velocity向量维数、force验证分类、窗边缘缺样、numpy布尔序列化、固定见证入卡、
阈值说明/必保分面、card身份绑定、JSON键序导致重渲染句序。这些均使用K4开发材料修复，未改数值阈值、
窗边界或预选规则；最终模板代码才是后续K5冻结版本，不冒称初稿一次PASS。
测量包装一度误测Windows venv launcher的约5.8MB，最终改为实际Python进程峰值；包装模块导入也曾因
缺ROOT sys.path失败（0实际用例），修复后专项30/30。早期包装stderr被最终路径覆盖，原错误在会话工具记录，
不伪称本地仍有那份原stderr；广域710项原始错误日志完整保留。

## 性能、冻结与下一道门

最终建卡每协议约0.6s函数时间；含Python启动的完整命令约1.04/1.08s。
实际Python峰值工作集58,970,112B/58,691,584B（不是venv launcher）。
独立384回合oracle含启动约1.38s、峰值81,973,248B；最终命令收据含精确值。
每协议evidence约3.8MB，card约0.47/0.40MB；原NPZ两组总3,486,938B。
这些是本机本批档案成本，不外推大型新轨迹吞吐。

[FREEZE_ZH.md](./FREEZE_ZH.md)的先行规则SHA：
`f65251e91b493dc13d3c00e9dad06813363ceeecf10dea78e6b669b88c92e9a5`。
[RULES_CODE_FREEZE.json](./RULES_CODE_FREEZE.json)绑定最终代码commit/tree、规则/模板、预选、输入SHA与依赖文件hash；
其最终文件SHA记录在verification.json，供独立K5任务在采集/解封前核验。

K5 Gen4B未读行为/summary、未运行，后续须独立采192ep force并盲核一次；定义仅为
“本次规则冻结后未用于translator调规则案例”，不是全历史完全unseen。若失败先保留失败，改进后属于
修复回归，不能改称首次holdout通过。完整E2、真实模型理解、normal runner/payload均未覆盖。
本轮server写入/模型API/GPU/训练调用均0。委托父线程放行前不启动K5或P3。
