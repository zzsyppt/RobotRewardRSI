# R1b：一次真实 public-dev 采集与 CPU 闭环

真实运行通过：固定代码 `ce25a9db7317bde4e0954fedc8de33ededc62542`，正常 `scripts/stair_numeric_dev_replay.py` → composition → 单卡租约 → Isaac → 新 numeric v3 archive → lease释放后的CPU summary。只有一次完整评测，没有训练、模型API、accept/sealed评测或新参考资格化；未改环境、凭证、其他作业和旧raw/ckpt。

服务器目录为 `/data/yuyangliu/isolated/r1b_20261004_real01`，下文路径相对该目录。`source/` 是合法远端独立sparse checkout，运行时固定上述SHA且干净。命令原文在 `run01.sh` / `command.json`，不是第二套runner：

```sh
/data/yuyangliu/envs/isaaclab6_validation/bin/python -B -u scripts/stair_numeric_dev_replay.py \
  --checkpoint /data/yuyangliu/isolated/new_world_p2_audit_final/true_RL_Agent/PLAN/2new_world/P2/results/audit3_go2_full_train/runner/audit3_go2_full_train/checkpoints/best.pt \
  --train-summary /data/yuyangliu/isolated/new_world_p2_audit_final/true_RL_Agent/PLAN/2new_world/P2/results/audit3_go2_full_train/runner/audit3_go2_full_train/train/summary.json \
  --protocol configs/tasks/stair_go2_12dof_unified_protocol_v3.json \
  --output /data/yuyangliu/isolated/r1b_20261004_real01/run01 \
  --device cuda:0 --isaac-python /data/yuyangliu/envs/isaaclab6_validation/bin/python \
  --mode run --cohort formal --capture-numeric on \
  --behavior-feedback-profile trajectory_numeric_v2 --trajectory-report-max-bytes 10240 \
  --numeric-evidence-max-bytes 268435456 --timeout-s 900
```

工作目录为 `source/true_RL_Agent`。脚本仅以 `/usr/bin/time -v` 记录成本；复用既有过程级 `UNITREE_ROS_DIR/LD_LIBRARY_PATH/XDG_CACHE_HOME/PIP_CACHE_DIR`，精确值见 `command.json`，没有修改全局环境。GPU前再次查无计算作业、磁盘可用71GB；固定新版本的CPU preflight通过。官方suite未缩减：24条件×12环境×3seed＝864回合。checkpoint SHA256为 `7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c`，保留原unified_v2训练身份，经现有合法祖先接当前v3评测。

## 实际契约及边界

本轮只增加自然可读的起止限位快照和一次地形配置记录，没有新profile/资格框架或逐步额外采集。

- recorder实录12个关节的原生名称、顺序与 `resolved_joint_ids=[0..11]`；q源仍是 `robot.data.joint_pos[:, resolved.joint_ids]`，rad、USD joint local frames，无额外零位偏置。
- 实际hard/soft position-limit buffers均为288×12×2，轴为env/joint/lower-upper；按同一ids经 `as_torch` 固化一次。环境构造后、首次显式seed reset前读取，最后评分回合结束时比值，结果相同。只证明起止快照一致，未监控中途变化，也不证明不可越界。soft范围不是simulation约束。
- 例calf的hard范围为 `[-2.722700119, -0.837759912]rad`，soft为 `[-2.628453016, -0.932007015]rad`。名义USD限位仍另用原闭包，不偷换为这两个buffer或物理有效限位。
- 当前Go2配置未加入限位随机化；安装源码 `envs/mdp/events.py:1924–1962` 的reset只读取soft范围裁剪重置状态，`Articulation.reset`只重置actuator/wrench。源码依据支持本次配置理解，但未据此扩大起止快照结论。
- 实际 importer为 `isaaclab.terrains.terrain_importer.TerrainImporter`，generator声明为 `isaaclab.terrains.terrain_generator:TerrainGenerator`；`/World/ground`、8×4m、1行24列、完整subterrain顺序均保存。所有seed首chunk的origins/types与当次nominal contract相同。
- generator源码单块居中、网格放置、整幅居中均是平移（`terrain_generator.py:384–388/333–340/181–187`）；importer `:89–95/251–253` 导入mesh/origins。没有读取实际mesh world matrix或cooked碰撞面，地形句仍为名义参考点高度差。

当前安装IsaacLab6.1.14、PhysX扩展1.1.3、IsaacSim6.0.1.0、Torch2.11.0+cu128、RSL-RL5.0.1。新环境manifest为 `749e5aa6679fb54d9d72c4e4d77ea1b7e71c38745d6ab1903e351f1db709ea61`，不是复用旧环境验收身份。两份现有USD缓存字节本轮重读，与新archive所声明的名义闭包 `c7fe9da20ae348ea53d34c266b7b6cea7746c94d2a4593450183dd464f783faf` 相符；没有下载或替换资产。

## 新原数组与作者文字

新source为 `run01/numeric_evidence/manifest.json`，SHA256 `830878007049c07ac2cbfc2e767ca7a9f964bfca1f3a269e6090c6766900f6ec`，未加入历史白名单。30个NPZ、864回合、300105有效控制行；独立读取valid/episode_length核全队列有效前缀与所有终点，117个终点runtime done为真，不能把其他存档结局当runtime触发器。

存档计数：689成功、113禁用接触、36超时、22姿态失稳、4偏离路线。独立只读核验者在看新卡前，以固定seed20261004按主要结局分层抽四回合，读取7个chunk，SHA均匹配；不调用生产测量器：

| 原数组锚点（env从0起） | 独立末0.5s事实 |
|---|---|
| r1/e230，成功，步300 | 净前进4.304770589m，末vx+0.760144353m/s；四足体相对活动各25/25条边。 |
| r1/e67，接触，步455 | 末vx+0.308296561m/s、上轴z始终正，末倾角24.833392°；终点base力230.672850N。 |
| r2/e177，超时，步600 | 根部末窗仅前进0.000144720m，但体相对足活动19/25条边；不能把世界点位近静止说成身体相对也完全静止。 |
| r2/e128，姿态标签，步600 | 末vx+0.000002146m/s、末倾角145.589304°；足体相对活动0/25。 |

对应首/末原向量、chunk/row定位和限定在 `independent_samples.json`。四例不代表全队列发生率，未用个例否定组内中位数。随后核最终文字相容：接触保留29曾向下/72未向下且仍前进/12其余，超时保留33近零/3仍前进。

最终正文代表句（摘录，定义仍统一给出场景原点/身体坐标及分位统计）：

> 超时组……末窗分布：|有向速度|<0.01m/s的33回合（速度中位-4.77e-06m/s）；有向速度>0.2m/s的3回合（速度中位0.735m/s）。低进度14620点/36回合：上轴z≥0活动/低活动1607/13013点。
>
> 例r1/e160步283，base当前0N，history槽力模0/0/1.54N；同拍记录base_contact为真；槽时标未记录，不能推定物理接触起点或持续时长。

后一例另直接读 `chunk_000004.npz` row26/env160：history末槽原向量 `[-0.433526278,-3.16286908e-8,1.475395322]N`，模为1.537769940N，与1.54N文字相符。

原CLI完整输出保留，公开报告10118B、full120231B。本轮实录buffer后发现旧gaps句仍说“运行时限位/表面未记录”，故仅在共用文字源头改为“动作力矩、奖励分项未记录；有效限位/碰撞面未验证”，词汇版本v4。由原已保存CPU摘要重投影，公开报告10127B、full120240B；只变这句，所有数值、摘要和入选事实不变。旧五档也只重投影，全部≤10240且选择不变。新source已用于此措辞修正与开发核验，不能称严格unseen；没有调整测量、选择规则或重跑GPU。

## 投递、成本与交付

原真实phase经同次配置的任务公共投影校验；reference/job/checkpoint/archive身份未重绑定。正常任务prompt合约、`candidate_draft` payload builder和生产 `UrllibJsonTransport` 在发送前离线捕获通过。原请求在 `author_capture/`；措辞修正后的最终user50580B、HTTP55416B在 `final_author_request/`，report与当前CPU投影逐对象相同。

**投递边界：这不是完整generation的“正式下一轮”。** 单个public-dev phase无accepted-best/qualified-anchor权限；正常context builder不能据此建立已接受父代。本轮明确reference观测接入正常draft合约，空候选历史/待定编辑基准，没有伪造accepted锚点、candidate trial、Genesis或父代差分。要证明完整generation下一轮，仍需合法已资格化锚点/会话输入；本轮没有为此额外评测或训练。

| 实际成本/产物 | 数值与口径 |
|---|---|
| CLI墙时 / 外层time墙时 | 349.906s / 351.06s |
| eval lease / 环境probe lease | 299.362s / 0.216s，总299.578s |
| rollout+聚合 | 280.016s |
| lease释放后CPU后处理 | 墙时48.703s，CPU48.665s |
| RSS峰值 | `/usr/bin/time -v` MaxRSS 4,937,616KiB，整命令口径 |
| GPU显存 | 外部周期采样最大4004MiB，不宣称连续峰值 |
| raw | 压缩83,531,904B，数组解压233,434,080B |
| 原run01目录 | 完成时85,082,363B；不含源码checkout和后加审核副本 |

31次外部采样仅出现child891102和目标GPU UUID `GPU-06661cdd-1a0e-904e-a2e3-705eda5fb65b`。五个样本记录phase=postprocessing时GPU无进程、lease锁已释放；最后PASS仍无GPU进程/锁。本轮没有capture-off对照，因此不报告“采集开销差”。

验证：本地capture/replay/production三模块29测试通过；最终采集模块本地5测试、目标环境5测试通过；文字修正后production10测试通过。只新增一条两环境、乱序关节、不同hard/soft值的快照映射测试。测试组有重叠，不相加成独立总数。

服务器保留 `run01/`（原raw、真实CLI/phase/attempt/环境结果）、`resource_samples.jsonl`、`runtime_source_evidence.json`、`raw_boundary_check.json`、`nominal_asset_check.json`、`selected_sentence_check.json`、`independent_samples.json`、`author_capture/`、`cpu_text_fix/`、`final_author_request/`。本地小型副本在 `D:/humanoid_agent/.tmp/r1b_real01/`，未复制或修改完整raw/ckpt。

本轮至此停止，交同一独立审计01a10282。实际mesh变换/cooked碰撞面、动作/力矩、1–2个真实reward profile，以及合法锚点下完整generation下一请求、作者理解和训练收益仍待办，未隐去或标记完成。

## 独立验收后的两处文字修正

独立验收确认真实运行与数值通过，但原离线请求有两处状态/文字矛盾。以下只用已保存摘要和原真实phase做CPU修正，原 `run01/`、报告、请求和审计均保留；未再读全量NPZ或启动GPU、模型API。

1. 一次性观测装配脚本 `.tmp/r1b_real01/capture_author.py` 原来展开空历史对象再覆盖status，继承了无效的 `reason=no_validated_public_stair_evaluation`。现在只复用schema，直接构造available对象，behavior中的reference/job/phase身份不变：

   ```python
   missing_feedback = sections['public_feedback_summary']
   sections['public_feedback_summary'] = {
       'schema_version': missing_feedback['schema_version'],
       'status': 'available',
       'behavior': {...},  # 原真实reference观测字段，不变
   }
   ```

   用实际builder核两个有限例：空历史仍为 `not_available`、正确reason且无behavior；接入观测后为 `available`，键仅 `schema_version/status/behavior`，没有reason。正常生产available分支原本即无缺失reason，无需新增兼容层。相邻minimum-core压缩分支未在本次请求触发，本轮未扩展为压缩框架改造。

2. `trajectory_production.py` 删除固定句中的“父代匹配与数值差分见本次反馈；”，只留“本卡描述当前队列。”，词汇版本v5。共用报告约定各section非空，故保留中立范围说明，不改schema。卡片不再断言比较存在，真实父代状态继续由调用边界提供。测量、选择规则均未变。

六份已有摘要重投影、replay通过；每份只改 `claim:parent`，短45B，入选/必选事实未变，其他作者文字逐句相同，数值摘要及source逐对象相同：

| 队列 | 修正前公开报告B | 修正后公开报告B |
|---|---:|---:|
| 本次真实R1b | 10127 | 10082 |
| R4 | 10179 | 10134 |
| K1 | 10150 | 10105 |
| K2 | 10096 | 10051 |
| m50 | 6101 | 6056 |
| m300 | 7360 | 7315 |

因此原行为验收事实保留，包括R1b接触29/72/12、超时33/3、history的1.54N例及相对z保持例；旧五档的分组过程、历史缺失和m50/m300差异文字也不变。生产模块现有10测试通过。本次没有新增永久测试脚手架。

原真实phase在冻结的ce25运行checkout重新经任务公共投影校验；修正装配脚本以新文件 `capture_author_acceptance_fix.py` 运行，输出 `author_capture_acceptance_fix/`，未覆盖旧脚本/输出。再将当前CPU重投影正文放入draft，重算预算并经实际 `UrllibJsonTransport` 离线捕获：最终user **50487B**、HTTP **55319B**；公开报告10082B，其中正文8615B。HTTP中的user与payload序列化字节相同，正文与当前CPU投影逐对象相同，预算再次计算不变。

最终 `public_feedback_summary.status=available`、无reason；`parent_diff=[{"id":"claim:parent","text":"本卡描述当前队列。"}]`。真实subject仍为reference `go2-stair-official-flat-derived-v1`，job仍为 `stair-dev-replay-7b582643510cc9dd-eval-public-dev`，phase hash未变。lineage仍为none、current_best/parent checksum均null、source_payloads为空，投递边界仍是上文的离线reference观测装配，并非完整generation下一轮。

新证据在本地 `.tmp/r1b_real01/acceptance_fix/`：六档 `before.json/full.json/public.json/author.txt`、`projection_checks.json`、修正前脚本副本、真实phase新捕获和 `final_author_request/`；同名新证据目录同步至原服务器R1b目录。旧 `final_author_request/` 保留为验收历史，当前结果以上述新目录为准。到此停止，交同一独立验收，不推进后续训练或采集。
