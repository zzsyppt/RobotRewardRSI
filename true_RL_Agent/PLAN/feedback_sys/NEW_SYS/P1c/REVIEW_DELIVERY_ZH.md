# P1c 最终投递原文与分层证据

本页从本轮 `final_capture` 的真实序列化产物摘录；完整原始HTTP/user字节与逐句偏移保存在本地 outputs，hash见下表及verification.json。没有模型回复或理解声明。

## 请求原文字节与SHA-256

| 帧 | HTTP字节 | HTTP SHA-256 | user字节 | user SHA-256 |
|---|---:|---|---:|---|
| draft | 71512 | `6363af2a473d023c8076043e87f096d18eb76db15567cea1fc5f47fd8c33e8b0` | 65792 | `c052242823e226498527ef8ecab884f76d8a619f360c414d1c9c0cbb26efb5cb` |
| source_followup | 72873 | `81ab8af8b304f6c6c5b911a4e9451502137437f83a2f45bbe94f438ce92f9d31` | 67069 | `9e708f6ddbae42628dae5d4f83c1bd4bb2d80ee62f3d8c532587b11be614092a` |
| static_repair | 71911 | `af63d5b6284f8de5d9bbb6a7257876ff945f3fe6372ec0b6ef5a8e91244049b2` | 66107 | `b0827c96bd1eb8d1c2ec77163f8fb25bbcac57f7bc83b855c36f56d560b6353e` |
| reasoning_exhaustion_repair | 73132 | `a31980c50b7882ebf8f48323758859b57a7402e54a152dfd371af24742bf6caa` | 67338 | `5c62b4e9010e519df2cda6dae443f6548ad1f34927c6fc23aacf1a66546ad384` |

实际profile位于每个user JSON的 `public_feedback_summary.behavior.profile`；receipt用同一内容核算8094B，距8192B剩98B。卡ID：`card:d83ddf0b8ee28e8d00705de81029930474b47ba230532596d50aabd2e9b0d9a2`。

## 作者实际收到的18条原claim

下列文字直接取最终public report，没有二次概括；新增的visual/omissions说明不计作P1b原claim。

- **counterexample** `claim:d48b600b73ceaf7d1a1777ca2eb11389af8180abe5369f0f1779b175248d68a0`

  r4.r3.e281 | public_dev.tread_ood_level_5 | success: 该成功回合仍有已观察回撤0.343054m（峰步270、谷步324、返回步396）；回撤过程与成功可同时存在，返回进度不等于已证明恢复平衡。

- **counterexample** `claim:a83eeb32f03402c04a0359b8fca2aba909ac7e917ffd58b73cf0e38395474c51`

  r4.r3.e64 | public_dev.c1_conditional_frontier_level_5 | base_orientation_instability: 身体上轴已在步154出现负值；robot_0.support.FL_foot所选低速off段首次记录在步508，前者早于后者；该顺序不解释机制，低速off本身不能证明直立举足。

- **gaps** `claim:65423f52b25115303e5f8f3e5c4994153b2452892087ef7ee6ea899e2d462537`

  本卡范围: 低进度确认使用完整至少0.5s窗口内最大新增进度<0.02m；初始不足窗为unknown。足速度组合沿用XY阈值0.05m/s，描述同条读取记录中的合取，不命名迈步、拒绝落脚或整体静止。

- **gaps** `claim:38c1fa46d5df98f7998f4471e545b071b6024bcdc0c1b96150e34900cc1b2aa4`

  本卡范围: contact marker表示所记录世界系净法向接触力模严格大于1N；记录间物理接触连续性未知，未触发不等于完全未接触，力模份额不等于重量或承载份额。

- **gaps** `claim:529d689bcc27d1ba71e62d708064427386463ccccbe039bd7b89bf2c0ec6fd9c`

  本卡范围: 缺少实际动作、力矩/饱和、逐步奖励及门控/air-time、实际质量/全身动量、切向力和接触对象、全部非足端接触力；精确奖励响应和物理对账不可用。已记录非足端力范围仅为绑定的base。

- **gaps** `claim:4dfd0ff9f90c4891992efb283f685ce8d4998c18fd36d136c16de63b99d2d0f1`

  本卡范围: 周期资格、有效/名义限位绑定、逐类地形几何资格、历史link位置与COM速度参考点映射尚未启用，保留后续验证义务；默认站姿和高位异常阈值未冻结，不生成举腿或拖地标签。

- **gaps** `claim:4905c04791940368fe92d9dd25d9b4de74a357e9442e63248217977ce30424e8`

  本卡范围: 控制时刻标记采集读取；current传感更新延迟和history逐槽时间/valid均未记录，history ANY与current分列，不补成物理子步或真实冲量。

- **gaps** `claim:71a42ce90f64b1d4bcce49c15596357288f2b7f3ffa7bd5dca8e09ec934ddd30`

  本卡范围: 默认回合等权，池化数值另用采样格时长代理加权；各量独立分母。格时长N×dt与观测跨度(N−1)×dt分列，左右截尾不进入完整区间时长分布；unknown和缺步不是阴性。

- **gaps** `claim:0f4053c747faa5aa00648efc760d62ccd8d1e7437eb2fae71bf632eee9c00cab`

  本卡范围: 先描述全队列，再展示结果、stratum和精确cell分面；不同终点时长/成功组成会影响整回合量。同cell比较只表示同公开条件格，未逐实例匹配实际随机化和命令；不是显著性、改善或因果结论。

- **parent_diff** `claim:d2cea6e2ede85bf39ec5c1600eabddcb0f761518e6e1c26044e26222d114c612`

  本卡范围: 父代比较不可用：没有显式绑定的父代档案。另源开发反例不充当父代，也不进入主队列分母。

- **process** `claim:5d6a6c02cbdd2a49de978387a2a26465ef5ece0367e3c40e427fe34d1a5954ed`

  all: 首末记录任务净进度：回合等权中位数4.254518m，四分位[3.535696,4.3658]；有效/未知/未处理/总回合=864/0/0/864，有效/未知记录=298375/0，缺步=0。

- **process** `claim:bf48483ed9d0232f2dca19c48bbe08f0d5cc4cc2755e6b5ea79fc97cbeef6489`

  all: 全回合可判定记录中的“低进度确认且至少一足XY速度≥0.05”合取占比：回合等权均值0.04381，池化比值0.063282（分子351.1/分母5548.14 sample_cell_s）；有效/未知/未处理/总回合=864/0/0/864，有效/未知记录=277407/20968，缺步=0。

- **process** `claim:a91beab2ca7ac052ba2a71a7167b99460a08f5cc07be8887cb6350d9c851b28d`

  all: 全回合可判定记录中的“低进度确认且全部足XY速度<0.05”合取占比：回合等权均值0.020526，池化比值0.033915（分子201.96/分母5954.86 sample_cell_s）；有效/未知/未处理/总回合=864/0/0/864，有效/未知记录=297743/632，缺步=0。

- **process** `claim:29e737ccc199df4c247f606c152c79412d4c3300a8351292586db6f5ef283b7f`

  all: robot_0.support.RR_foot低XY速度off已观察连续记录格时长≥0.5s：已观察阳性28/可判定864回合，未知0、未处理0、总数864；仅限现有记录，未知不作阴性，多个模式可重叠。

- **process** `claim:75f680f5d7a322e8e7972f76c722324f98f7c9040cfc4b1097d1a943af0198a1`

  success: robot_0.support.RL_foot低XY速度off已观察连续记录格时长≥0.5s：已观察阳性3/可判定682回合，未知0、未处理0、总数682；仅限现有记录，未知不作阴性，多个模式可重叠。

- **process** `claim:89f3ef38fc7738d22543bd48de09bdcd2918558b6bab971918c26b37179e15f9`

  failure: robot_0.support.RL_foot低XY速度off已观察连续记录格时长≥0.5s：已观察阳性19/可判定182回合，未知0、未处理0、总数182；仅限现有记录，未知不作阴性，多个模式可重叠。

- **process** `claim:2761bf72c077c904ee46b9150ca2c469346cf02f07189cc06d6bd32ecc55ae5e`

  r4.r1.e127 | public_dev.c2_mechanism_ood_level_4 | forbidden_contact: robot_0.body.base current越阈0条，区间：无已观察阳性；history ANY越阈1条，区间：步326–326，1条，格时长代理0.02s、观测跨度0s，左/右截尾=false/true；单条短事件保留。

- **results** `claim:c808e671c3bbfd0daac8a07ac7ff8e8fa1fab3b50d345404e42739fac5d0ede4`

  all: 本队列共864回合、298375条有效记录；存档任务结果：{"base_orientation_instability":16,"forbidden_contact":125,"off_course":1,"slip":1,"success":682,"timeout":39}。

## 省略与E1分层

783条省略：782条整句预算省略，1条过时P1b投递状态；完整清单、原文和理由在projection_audit.json的omitted_claims中。18必保ID和801逐项映射在同目录。每个作者ID可回溯内部audit，并不提供自动作者检索权限。

| 冻结单回合 | 必要事实通过数 | 完整事实通过数 | 作者直接细节claim |
|---|---:|---:|---:|
| r4.r1.e35 | 13 | 17 | 0 |
| r4.r2.e44 | 13 | 17 | 0 |
| r4.r3.e82 | 13 | 17 | 0 |
| r4.r1.e139 | 13 | 17 | 0 |
| r4.r3.e148 | 13 | 17 | 0 |
| r4.r3.e184 | 13 | 17 | 0 |
| r4.r3.e218 | 13 | 17 | 0 |
| r4.r3.e264 | 13 | 17 | 0 |
| r4.r1.e129 | 13 | 17 | 0 |
| r4.r1.e175 | 13 | 17 | 0 |
| r4.r1.e206 | 13 | 17 | 0 |
| r4.r1.e60 | 17 | 17 | 0 |
| r4.r1.e48 | 14 | 17 | 0 |
| r4.r1.e123 | 13 | 17 | 0 |
| r4.r2.e164 | 13 | 17 | 0 |
| r4.r3.e190 | 14 | 17 | 0 |
| legacy_dev.r1.e54 | 18 | 18 | 0 |
| legacy_dev.r1.e60 | 14 | 18 | 0 |

18例合计246必要/308完整事实；另1条r4.all_episodes_basic_counts_and_forces队列事实组成247/309门。只有该队列事实关联3条作者聚合claim，映射状态为related_claim_only_not_all_fact_fields。E1单回合细节均在完整审计层，不能声称18例所有事实或309事实进入作者请求。

## 关键语义与实际反例

- all分母864、success682、failure182不混入legacy2。两种合取有效记录分母分别277407、297743，unknown分别20968、632，文字保持原卡。
- r4.r3.e281成功仍回撤0.343054m；r4.r3.e64在步154翻转早于步508低速off，不命名直立举足。
- r4.r1.e127只在history有1条步326事件，current为0，保留右截尾。r4.r1.e129更详细的步356原数组反例仍在audit，未替换成已投递claim。
- 缺口、current/history、采样代理、条件/同源比较范围、父代不可用均在最终请求。周期/限位/地形/历史link-COM仍未启用。

15项真实来源/预算失败日志在negative_cases.json；逐项来源映射在claim_to_evidence.json，原始数值复算与冻结事实验证日志在final_checks。
