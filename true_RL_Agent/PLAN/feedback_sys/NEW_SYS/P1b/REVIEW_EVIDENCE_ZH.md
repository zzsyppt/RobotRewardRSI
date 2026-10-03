# P1b 交付阅读与原数组核查摘录

这是开发线程交付证据，不表示父线程已验收。完整801句和四产物位于 `outputs/trajectory_p1b/final_card/`；15句主视图不是作者最终输入。

## 策略主视图全文

- **all** 本队列共864回合、298375条有效记录；存档任务结果：{"base_orientation_instability":16,"forbidden_contact":125,"off_course":1,"slip":1,"success":682,"timeout":39}。 `[claim:c808e671c3bbfd0daac8a07ac7ff8e8fa1fab3b50d345404e42739fac5d0ede4]`

- **all** 首末记录任务净进度：回合等权中位数4.254518m，四分位[3.535696,4.3658]；有效/未知/未处理/总回合=864/0/0/864，有效/未知记录=298375/0，缺步=0。 `[claim:5d6a6c02cbdd2a49de978387a2a26465ef5ece0367e3c40e427fe34d1a5954ed]`

- **all** 全回合可判定记录中的“低进度确认且至少一足XY速度≥0.05”合取占比：回合等权均值0.04381，池化比值0.063282（分子351.1/分母5548.14 sample_cell_s）；有效/未知/未处理/总回合=864/0/0/864，有效/未知记录=277407/20968，缺步=0。 `[claim:bf48483ed9d0232f2dca19c48bbe08f0d5cc4cc2755e6b5ea79fc97cbeef6489]`

- **all** 全回合可判定记录中的“低进度确认且全部足XY速度<0.05”合取占比：回合等权均值0.020526，池化比值0.033915（分子201.96/分母5954.86 sample_cell_s）；有效/未知/未处理/总回合=864/0/0/864，有效/未知记录=297743/632，缺步=0。 `[claim:a91beab2ca7ac052ba2a71a7167b99460a08f5cc07be8887cb6350d9c851b28d]`

- **all** robot_0.support.RR_foot低XY速度off已观察连续记录格时长≥0.5s：已观察阳性28/可判定864回合，未知0、未处理0、总数864；仅限现有记录，未知不作阴性，多个模式可重叠。 `[claim:29e737ccc199df4c247f606c152c79412d4c3300a8351292586db6f5ef283b7f]`

- **success** robot_0.support.RL_foot低XY速度off已观察连续记录格时长≥0.5s：已观察阳性3/可判定682回合，未知0、未处理0、总数682；仅限现有记录，未知不作阴性，多个模式可重叠。 `[claim:75f680f5d7a322e8e7972f76c722324f98f7c9040cfc4b1097d1a943af0198a1]`

- **failure** robot_0.support.RL_foot低XY速度off已观察连续记录格时长≥0.5s：已观察阳性19/可判定182回合，未知0、未处理0、总数182；仅限现有记录，未知不作阴性，多个模式可重叠。 `[claim:89f3ef38fc7738d22543bd48de09bdcd2918558b6bab971918c26b37179e15f9]`

- **r4.r3.e281** 该成功回合仍有已观察回撤0.343054m（峰步270、谷步324、返回步396）；回撤过程与成功可同时存在，返回进度不等于已证明恢复平衡。 `[claim:d48b600b73ceaf7d1a1777ca2eb11389af8180abe5369f0f1779b175248d68a0]`

- **r4.r3.e64** 身体上轴已在步154出现负值；robot_0.support.FL_foot所选低速off段首次记录在步508，前者早于后者；该顺序不解释机制，低速off本身不能证明直立举足。 `[claim:a83eeb32f03402c04a0359b8fca2aba909ac7e917ffd58b73cf0e38395474c51]`

- **r4.r1.e127** robot_0.body.base current越阈0条，区间：无已观察阳性；history ANY越阈1条，区间：步326–326，1条，格时长代理0.02s、观测跨度0s，左/右截尾=false/true；单条短事件保留。 `[claim:2761bf72c077c904ee46b9150ca2c469346cf02f07189cc06d6bd32ecc55ae5e]`

- **本卡范围** 低进度确认使用完整至少0.5s窗口内最大新增进度<0.02m；初始不足窗为unknown。足速度组合沿用XY阈值0.05m/s，描述同条读取记录中的合取，不命名迈步、拒绝落脚或整体静止。 `[claim:65423f52b25115303e5f8f3e5c4994153b2452892087ef7ee6ea899e2d462537]`

- **本卡范围** contact marker表示所记录世界系净法向接触力模严格大于1N；记录间物理接触连续性未知，未触发不等于完全未接触，力模份额不等于重量或承载份额。 `[claim:38c1fa46d5df98f7998f4471e545b071b6024bcdc0c1b96150e34900cc1b2aa4]`

- **本卡范围** 缺少实际动作、力矩/饱和、逐步奖励及门控/air-time、实际质量/全身动量、切向力和接触对象、全部非足端接触力；精确奖励响应和物理对账不可用。已记录非足端力范围仅为绑定的base。 `[claim:529d689bcc27d1ba71e62d708064427386463ccccbe039bd7b89bf2c0ec6fd9c]`

- **本卡范围** 周期资格、有效/名义限位绑定、逐类地形几何资格、历史link位置与COM速度参考点映射尚未启用，保留后续验证义务；默认站姿和高位异常阈值未冻结，不生成举腿或拖地标签。 `[claim:4dfd0ff9f90c4891992efb283f685ce8d4998c18fd36d136c16de63b99d2d0f1]`

- **本卡范围** 父代比较不可用：没有显式绑定的父代档案。另源开发反例不充当父代，也不进入主队列分母。 `[claim:d2cea6e2ede85bf39ec5c1600eabddcb0f761518e6e1c26044e26222d114c612]`

## 展开样例卡片全文：r4.r1.e129

- r4.r1.e129：条件public_dev.c2_mechanism_ood_level_4，任务存档结果forbidden_contact，记录步1–356共356条，dt=0.02s；使用控制步末/重置前记录，没有t=0或终局后观察。 `[claim:af72a911a72daf419fa10f9f4807ba8974f1fb46792aff3cb0fdf9d8717ee575]`

- 根部净位移xyz=[3.8591, -0.563359, 1.334461]m，任务净进度3.8591m；采样折线路径XY/XYZ=4.102134/5.108119m，完整路径=true；正/负进度变化量=3.891744/0.032644m；最大已观察回撤0.013854m，峰/谷/首次返回步=351/356/未知，返回观察右截尾=true。 `[claim:b86c792f494df1281a0bb9bba1f8aac208a8b5b2d637cf00e6431946f1c6dcf3]`

- 归一化wxyz的身体上轴竖直分量首个/末个有效记录值=0.999552/0.981684（步1/356），最小值0.885578，最小值步135；负值记录0条，首次负值步未知，姿态未知0条；不解释失败原因。 `[claim:56af22d1f301755e213a007103f215c5a839de9c62a0cce17801afde32716578]`

- robot_0.support.FL_foot current marker：on/off/unknown=230/126/0，有效356条、占比0.646067；onset/offset=17/17；完整off段16、最短off段1条；最长off：步140–159，20条，格时长代理0.4s、观测跨度0.38s，左/右截尾=false/false。 `[claim:6e9802011da09dabbeae6764b2c5dedf19f3f2ac2434c31d571256fbadddfce6]`

- robot_0.support.FL_foot速度通道XY<0.05m/s且current marker off：步174–174，1条，格时长代理0.02s、观测跨度0s，左/右截尾=false/false；该段XY最大/中位速度=0.004858/0.004858m/s，XYZ诊断最大速度0.131458m/s，力模最大0N，上轴竖直分量范围[0.978815,0.978815]（姿态有效1条）；不据此断言直立举足。 `[claim:cd55b51658c5eba77851e13f9acc058dfa093a9571168eb2ce69f6078c645d40]`

- robot_0.support.FL_foot足参考点相对根部的首帧xyz=[0.218542, 0.177075, -0.284655]m，身体坐标xyz范围=[[-0.040628, 0.389821], [0.113327, 0.345727], [-0.346568, 0.061455]]m；另列速度通道世界XY范围[0.004858,5.931572]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:c20ae479ef75d18583e7ab785879eb67b056b3178a534046dc0ce0413a398f49]`

- robot_0.support.FL_foot history ANY超过1N：on/off/unknown=239/117/0；history-on且current-off=9条，反向差异=0条，联合有效356条；无逐槽时间/valid，不替代current。 `[claim:2ad6a18efdf73b2ed76813ccd12718e230662925b9af7774fe4a322ab1a28157]`

- robot_0.support.FR_foot current marker：on/off/unknown=242/114/0，有效356条、占比0.679775；onset/offset=20/20；完整off段19、最短off段1条；最长off：步37–50，14条，格时长代理0.28s、观测跨度0.26s，左/右截尾=false/false。 `[claim:f2606d93dc07e30ea4eca48bed662dca35746f8ef331144f49e2fc051ea1af22]`

- robot_0.support.FR_foot速度通道XY<0.05m/s且current marker off：步46–46，1条，格时长代理0.02s、观测跨度0s，左/右截尾=false/false；该段XY最大/中位速度=0.042269/0.042269m/s，XYZ诊断最大速度0.094162m/s，力模最大0N，上轴竖直分量范围[0.99913,0.99913]（姿态有效1条）；不据此断言直立举足。 `[claim:ce319f1f89e896cc669f6d69fc6302ce9e410c79394b304f506f526e9509800e]`

- robot_0.support.FR_foot足参考点相对根部的首帧xyz=[0.203459, -0.171224, -0.319131]m，身体坐标xyz范围=[[-0.041432, 0.402448], [-0.328845, -0.125692], [-0.360464, -0.015733]]m；另列速度通道世界XY范围[0.001508,5.880048]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:70c059b516d6acb42fe29661dd71cdb519891a5c1bfe42d05a58beba0f9c9a3f]`

- robot_0.support.FR_foot history ANY超过1N：on/off/unknown=250/106/0；history-on且current-off=8条，反向差异=0条，联合有效356条；无逐槽时间/valid，不替代current。 `[claim:e7faa5355829b6d7e42eb350b1bd01fa754fbbb49fba2710930b8d459796076b]`

- robot_0.support.RL_foot current marker：on/off/unknown=125/231/0，有效356条、占比0.351124；onset/offset=32/32；完整off段31、最短off段1条；最长off：步72–89，18条，格时长代理0.36s、观测跨度0.34s，左/右截尾=false/false。 `[claim:389b3812704cf70069926c1a8b5d7cf11fcc617c29ad55369fc89dcfbbdc3a79]`

- robot_0.support.RL_foot速度通道XY<0.05m/s且current marker off：步4–4，1条，格时长代理0.02s、观测跨度0s，左/右截尾=false/false；该段XY最大/中位速度=0.013/0.013m/s，XYZ诊断最大速度0.727402m/s，力模最大0N，上轴竖直分量范围[0.99382,0.99382]（姿态有效1条）；不据此断言直立举足。 `[claim:3391446574633025a7700e2cfa9a6e6d978a8d710145c51b3f0a8e04677b1075]`

- robot_0.support.RL_foot足参考点相对根部的首帧xyz=[-0.29444, 0.137342, -0.339792]m，身体坐标xyz范围=[[-0.565712, -0.262868], [-0.010775, 0.214663], [-0.38573, -0.124219]]m；另列速度通道世界XY范围[0.007417,5.150718]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:f22a2e0de3bd3bb018b1a49f7e6b2040d190f873803bd4df784b3aa62a9b580f]`

- robot_0.support.RL_foot history ANY超过1N：on/off/unknown=138/218/0；history-on且current-off=13条，反向差异=0条，联合有效356条；无逐槽时间/valid，不替代current。 `[claim:d3329d2c6fab58b165ae916f2b7537f8dda4042386463748dd3ed842b0925c6b]`

- robot_0.support.RR_foot current marker：on/off/unknown=205/151/0，有效356条、占比0.575843；onset/offset=23/22；完整off段22、最短off段1条；最长off：步197–208，12条，格时长代理0.24s、观测跨度0.22s，左/右截尾=false/false。 `[claim:663d449634fdad942beddae474c035de1e0164d558a05a33a3edb5f4b7d4f09c]`

- robot_0.support.RR_foot速度通道XY<0.05m/s且current marker off：步153–153，1条，格时长代理0.02s、观测跨度0s，左/右截尾=false/false；该段XY最大/中位速度=0.02106/0.02106m/s，XYZ诊断最大速度0.364576m/s，力模最大0N，上轴竖直分量范围[0.941575,0.941575]（姿态有效1条）；不据此断言直立举足。 `[claim:290c057bd10f050f89a7ed4e42a13c4340dc705b808bf2c6f9bde2346d3e9aa1]`

- robot_0.support.RR_foot足参考点相对根部的首帧xyz=[-0.23472, -0.121532, -0.295435]m，身体坐标xyz范围=[[-0.456598, -0.086216], [-0.173328, 0.113651], [-0.364331, -0.08821]]m；另列速度通道世界XY范围[0.008795,3.926609]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:2809ae414ca2bfcf84243ba4b5bfc773ae39e93c3451dd7f0746832972087163]`

- robot_0.support.RR_foot history ANY超过1N：on/off/unknown=216/140/0；history-on且current-off=11条，反向差异=0条，联合有效356条；无逐槽时间/valid，不替代current。 `[claim:40ad851738a30979cc616678167e25944b65620f10b43f8414d791ec245cecef]`

- 按robot_0.support.FL_foot/robot_0.support.FR_foot/robot_0.support.RL_foot/robot_0.support.RR_foot顺序，记录净法向力模份额=[0.278444, 0.256399, 0.172311, 0.292846]，有符号世界Fz份额=[0.295617, 0.262911, 0.161473, 0.28]；各自联合有效记录356/356，未知0/0；不是重量份额、完整外力或物理冲量。 `[claim:eacbc3f5e26ccf75db4f74ab3549d64cb28551d47f5789fbb3fdcb860e05f69e]`

- 按robot_0.support.FL_foot/robot_0.support.FR_foot/robot_0.support.RL_foot/robot_0.support.RR_foot顺序、权重[1, 2, 4, 8]编码，各组合0–15记录数=[46, 4, 17, 18, 23, 18, 9, 16, 8, 6, 12, 120, 4, 5, 7, 43]；联合有效/未知=356/0；全off不证明全身腾空，组合不命名步态。 `[claim:39297376afefd245ca450dd1ccf26123d02bb989d2b060a3ec3c51ffcdeee999]`

- 终点步356、时刻7.12s，manager/wrapper标记={"done_mask":true,"wrapper_done_mask":true,"termination_0":false,"termination_1":true}；termination索引依次对应[time_out, base_contact]；末5条步号=[352, 353, 354, 355, 356]，base current力模=[[0], [0], [0], [0], [0]]N，另列history各槽力模=[[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [635.403135]]]N；任务结果保持存档评价器权威。 `[claim:19bf67881374784643c06b85b41ce7efa8c5a7ed3ef72c29621d8c64556bfaa8]`

- robot_0.body.base current越阈0条，区间：无已观察阳性；history ANY越阈1条，区间：步356–356，1条，格时长代理0.02s、观测跨度0s，左/右截尾=false/true；单条短事件保留。 `[claim:db3b7c5d462b0a3ca1c3d4c8844d83d8eaeec756a72cd3d1fb23006515fc1737]`

- 低进度窗口确认与足速度组合：至少一足XY≥0.05的true/false/unknown=0/331/25；全部足XY<0.05的true/false/unknown=0/356/0；最长至少一足达阈值段：没有已观察到的该状态区间；最长全部足低速段：没有已观察到的该状态区间。合取可由不同足交替贡献，不断言迈步次数或整体静止。 `[claim:b11a7085db54219cf247307ae7fdd5ebf69f6beff30ad2257393825d5d932fcb]`

## 展开样例卡片全文：legacy_dev.r1.e54

- legacy_dev.r1.e54：条件public_dev.c1_conditional_frontier_level_4，任务存档结果base_orientation_instability，记录步1–588共588条，dt=0.02s；使用控制步末/重置前记录，没有t=0或终局后观察。 `[claim:675e3d2a8fb03154222ed66c50531e4be695b5c28368e57b8ef7250f5fc7d9f1]`

- 根部净位移xyz=[1.175649, 0.262674, 0.174423]m，任务净进度1.175649m；采样折线路径XY/XYZ=1.958915/3.043304m，完整路径=true；正/负进度变化量=1.473177/0.297528m；最大已观察回撤0.213423m，峰/谷/首次返回步=112/185/221，返回观察右截尾=false。 `[claim:d1ae85956a17657aba03979cf184f6e5691d22bfbd8aeb3bcc6e73b81c909e0a]`

- 归一化wxyz的身体上轴竖直分量首个/末个有效记录值=0.999768/-0.977291（步1/588），最小值-0.985646，最小值步309；负值记录311条，首次负值步278，姿态未知0条；不解释失败原因。 `[claim:268699711170d8603ad1221892a35cc2e64238c60ad0f17794a2e17b2f051aa6]`

- robot_0.support.FL_foot current marker：on/off/unknown=194/394/0，有效588条、占比0.329932；onset/offset=13/13；完整off段12、最短off段2条；最长off：步281–588，308条，格时长代理6.16s、观测跨度6.14s，左/右截尾=false/true。 `[claim:5c9d417214b7b0e972913d4e38264153f04e6f78a45f8e5d60ceb135560ca447]`

- robot_0.support.FL_foot速度通道XY<0.05m/s且current marker off：步326–588，263条，格时长代理5.26s、观测跨度5.24s，左/右截尾=false/true；该段XY最大/中位速度=0.047335/0.009557m/s，XYZ诊断最大速度0.047362m/s，力模最大0N，上轴竖直分量范围[-0.977311,-0.976188]（姿态有效263条）；不据此断言直立举足。 `[claim:8ffc90097133def0339b8fc69d31b817e46f7b1684d69d5b0a3ed2f36ad5d0c1]`

- robot_0.support.FL_foot足参考点相对根部的首帧xyz=[0.219183, 0.191495, -0.2813]m，身体坐标xyz范围=[[-0.13064, 0.530666], [0.029194, 0.382857], [-0.35168, 0.059315]]m；另列速度通道世界XY范围[0.000351,5.842118]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:1d1ad540f593e9b97af32083a9b5863354f43a754a6df58a2da53700242e610c]`

- robot_0.support.FL_foot history ANY超过1N：on/off/unknown=200/388/0；history-on且current-off=6条，反向差异=0条，联合有效588条；无逐槽时间/valid，不替代current。 `[claim:8fe8892907eb381548edf92f31a303959fdbe5ef38c3327955fa9820998275a3]`

- robot_0.support.FR_foot current marker：on/off/unknown=135/453/0，有效588条、占比0.229592；onset/offset=12/12；完整off段11、最短off段1条；最长off：步245–588，344条，格时长代理6.88s、观测跨度6.86s，左/右截尾=false/true。 `[claim:325aa33aa2351e1dfcf559f4b61bbd191123fd2d943a93fe0df2fbe109745285]`

- robot_0.support.FR_foot速度通道XY<0.05m/s且current marker off：步325–588，264条，格时长代理5.28s、观测跨度5.26s，左/右截尾=false/true；该段XY最大/中位速度=0.048357/0.026855m/s，XYZ诊断最大速度0.048872m/s，力模最大0N，上轴竖直分量范围[-0.977311,-0.976186]（姿态有效264条）；不据此断言直立举足。 `[claim:4d2ba12cc5a31f542a0bb4617cc6df218e8b400cef6cbe1b26a26a324c4bcfe8]`

- robot_0.support.FR_foot足参考点相对根部的首帧xyz=[0.20424, -0.185564, -0.307707]m，身体坐标xyz范围=[[-0.122189, 0.446055], [-0.431102, -0.00612], [-0.381609, 0.00405]]m；另列速度通道世界XY范围[0.002903,5.896373]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:c062fd32c7892779656fae056f21cef7e53c696cf29865ff8e729fb94a2eb1d5]`

- robot_0.support.FR_foot history ANY超过1N：on/off/unknown=135/453/0；history-on且current-off=0条，反向差异=0条，联合有效588条；无逐槽时间/valid，不替代current。 `[claim:6f844e75737410525bd3b322b309ffa95a6076b6931d3e9ab45f9a14eed67914]`

- robot_0.support.RL_foot current marker：on/off/unknown=141/447/0，有效588条、占比0.239796；onset/offset=12/12；完整off段11、最短off段1条；最长off：步279–588，310条，格时长代理6.2s、观测跨度6.18s，左/右截尾=false/true。 `[claim:4274d4b8697e6e648cca63ac347960c49a70b936ff759b950b80bbfe58f7946f]`

- robot_0.support.RL_foot速度通道XY<0.05m/s且current marker off：步325–588，264条，格时长代理5.28s、观测跨度5.26s，左/右截尾=false/true；该段XY最大/中位速度=0.046102/0.004191m/s，XYZ诊断最大速度0.046583m/s，力模最大0N，上轴竖直分量范围[-0.977311,-0.976186]（姿态有效264条）；不据此断言直立举足。 `[claim:a200d57bfb56b17fa7aeb1685efab2bf30fdce9032ac5153d3ed17b937f8c321]`

- robot_0.support.RL_foot足参考点相对根部的首帧xyz=[-0.297822, 0.143769, -0.336978]m，身体坐标xyz范围=[[-0.582003, -0.155724], [-0.052978, 0.431165], [-0.386298, -0.038452]]m；另列速度通道世界XY范围[0.000257,4.693659]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:e90e148ee8dff0a0ff5bb6e5d858e4551999f3806a33418ab5bbc31bcbeae057]`

- robot_0.support.RL_foot history ANY超过1N：on/off/unknown=144/444/0；history-on且current-off=3条，反向差异=0条，联合有效588条；无逐槽时间/valid，不替代current。 `[claim:e29d65a53fa8bc11b1a1891cfa091c9e33497cfa2ee0f757706fee66c8a0ff2a]`

- robot_0.support.RR_foot current marker：on/off/unknown=133/455/0，有效588条、占比0.22619；onset/offset=17/17；完整off段16、最短off段1条；最长off：步234–588，355条，格时长代理7.1s、观测跨度7.08s，左/右截尾=false/true。 `[claim:da1bba5896dbc712b551b7467048d986e373c6797df30d3a61d862e6cee9e384]`

- robot_0.support.RR_foot速度通道XY<0.05m/s且current marker off：步324–588，265条，格时长代理5.3s、观测跨度5.28s，左/右截尾=false/true；该段XY最大/中位速度=0.039462/0.010333m/s，XYZ诊断最大速度0.042825m/s，力模最大0N，上轴竖直分量范围[-0.977311,-0.976186]（姿态有效265条）；不据此断言直立举足。 `[claim:06d61130e91716546cc8ff8ca5474522e1be06e4109ce28536a6b3f618dc0bd7]`

- robot_0.support.RR_foot足参考点相对根部的首帧xyz=[-0.252769, -0.131435, -0.276432]m，身体坐标xyz范围=[[-0.543135, -0.079446], [-0.391699, 0.08065], [-0.363042, -0.049778]]m；另列速度通道世界XY范围[0.001954,6.071648]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:21e3fcce9904fe473f1041766181f1a63db787fd68d61392266003d02b4265fd]`

- robot_0.support.RR_foot history ANY超过1N：on/off/unknown=139/449/0；history-on且current-off=6条，反向差异=0条，联合有效588条；无逐槽时间/valid，不替代current。 `[claim:4830d21ce6ce6910e944123fead40858d551463bea557e27a9c751381a0dc56e]`

- 按robot_0.support.FL_foot/robot_0.support.FR_foot/robot_0.support.RL_foot/robot_0.support.RR_foot顺序，记录净法向力模份额=[0.363097, 0.183725, 0.250887, 0.202291]，有符号世界Fz份额=[0.367056, 0.1831, 0.256293, 0.193552]；各自联合有效记录588/588，未知0/0；不是重量份额、完整外力或物理冲量。 `[claim:7847d6ab563f8252929f0be46b69616f5699674d76749fc9f2a1a5d29a445738]`

- 按robot_0.support.FL_foot/robot_0.support.FR_foot/robot_0.support.RL_foot/robot_0.support.RR_foot顺序、权重[1, 2, 4, 8]编码，各组合0–15记录数=[322, 42, 0, 27, 25, 8, 12, 19, 7, 27, 0, 22, 13, 9, 15, 40]；联合有效/未知=588/0；全off不证明全身腾空，组合不命名步态。 `[claim:2eae9722d0d3d6b673371dd62be32a999f2eb0d26a59181d31301e3e28ea320e]`

- 终点步588、时刻11.76s，manager/wrapper标记={"done_mask":false,"wrapper_done_mask":false,"termination_0":false,"termination_1":false}；termination索引依次对应[time_out, base_contact]；末5条步号=[584, 585, 586, 587, 588]，base current力模=[[0], [0], [0], [0], [0]]N，另列history各槽力模=[[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]]N；任务结果保持存档评价器权威。 `[claim:b96528f9dbff6f4958051af857f7e7537168267bf09e2bfa9271e1e632bd402c]`

- robot_0.body.base current越阈0条，区间：无已观察阳性；history ANY越阈0条，区间：无已观察阳性；单条短事件保留。 `[claim:9a0e378ae4f7f8939ee90dc6559184feadb2c3fe657c33190ee57de1573b4684]`

- 低进度窗口确认与足速度组合：至少一足XY≥0.05的true/false/unknown=89/474/25；全部足XY<0.05的true/false/unknown=264/324/0；最长至少一足达阈值段：步134–202，69条，格时长代理1.38s、观测跨度1.36s，左/右截尾=false/false；最长全部足低速段：步326–588，263条，格时长代理5.26s、观测跨度5.24s，左/右截尾=false/true。前一段按robot_0.support.FL_foot/robot_0.support.FR_foot/robot_0.support.RL_foot/robot_0.support.RR_foot顺序，分足达阈值记录数=[68, 54, 30, 47]，分足有效数=[69, 69, 69, 69]，同一足覆盖该段每条记录=[false, false, false, false]。合取可由不同足交替贡献，不断言迈步次数或整体静止。 `[claim:1fcbd2bd8c3e9cfdd46ec95ec2153f9ca264362206cc5e18353b2f29a758f477]`

- 身体上轴已在步278出现负值；robot_0.support.FL_foot所选低速off段首次记录在步326，前者早于后者；该顺序不解释机制，低速off本身不能证明直立举足。 `[claim:89cd90a793264b81c20248a6324745307195386b19433cd48bf1c20e4aac8946]`

- 身体上轴已在步278出现负值；robot_0.support.FR_foot所选低速off段首次记录在步325，前者早于后者；该顺序不解释机制，低速off本身不能证明直立举足。 `[claim:9fa699e25d711a5219f3f2ea5ec8df18ef5aa12267fe67f1fa3d82c3b1be82e4]`

- 身体上轴已在步278出现负值；robot_0.support.RL_foot所选低速off段首次记录在步325，前者早于后者；该顺序不解释机制，低速off本身不能证明直立举足。 `[claim:b20cbf15b6e6784785511e02436ce1a6b11459b384f2dafcab6ca493b8103cbb]`

- 身体上轴已在步278出现负值；robot_0.support.RR_foot所选低速off段首次记录在步324，前者早于后者；该顺序不解释机制，低速off本身不能证明直立举足。 `[claim:5181765905906ef5401cca059fbd4e594ef7d445fb45348adddc4a14afc0e498]`

## 展开样例卡片全文：legacy_dev.r1.e60

- legacy_dev.r1.e60：条件public_dev.c1_conditional_frontier_level_5，任务存档结果success，记录步1–525共525条，dt=0.02s；使用控制步末/重置前记录，没有t=0或终局后观察。 `[claim:ea3ffe4fd6e2348f1480beb15adbff8820622978b7b2aff0c76fcf3c763fb726]`

- 根部净位移xyz=[4.629802, -0.503832, 1.722455]m，任务净进度4.629802m；采样折线路径XY/XYZ=5.378943/7.194784m，完整路径=true；正/负进度变化量=4.877664/0.247862m；最大已观察回撤0.125879m，峰/谷/首次返回步=84/116/141，返回观察右截尾=false。 `[claim:d70bfd8602a49db50b9e815ca41a3dce39d8485db43ecc5b2cf211d466ba3d8d]`

- 归一化wxyz的身体上轴竖直分量首个/末个有效记录值=0.999566/0.999015（步1/525），最小值0.632799，最小值步431；负值记录0条，首次负值步未知，姿态未知0条；不解释失败原因。 `[claim:3f10aac9b34a7013320823dd23026f7d5693cdecb5f6a9d1e5a5132f31651fa5]`

- robot_0.support.FL_foot current marker：on/off/unknown=321/204/0，有效525条、占比0.611429；onset/offset=22/21；完整off段21、最短off段2条；最长off：步319–343，25条，格时长代理0.5s、观测跨度0.48s，左/右截尾=false/false。 `[claim:6f838d0f94377b050c14741fdd0255707d7807f03a10b7e92fa35a4922286630]`

- robot_0.support.FL_foot速度通道XY<0.05m/s且current marker off：步241–241，1条，格时长代理0.02s、观测跨度0s，左/右截尾=false/false；该段XY最大/中位速度=0.042078/0.042078m/s，XYZ诊断最大速度0.087968m/s，力模最大0N，上轴竖直分量范围[0.952006,0.952006]（姿态有效1条）；不据此断言直立举足。 `[claim:bc1af568598124fc74d56dbf27681f0fd5ddfe85691a5218ef5d343490ca1bea]`

- robot_0.support.FL_foot足参考点相对根部的首帧xyz=[0.218411, 0.177119, -0.284742]m，身体坐标xyz范围=[[-0.141036, 0.423579], [0.073468, 0.36772], [-0.35334, 0.096121]]m；另列速度通道世界XY范围[0.006568,6.41973]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:bb563add0041b6dde42b30062feda63ca14286d80d5abe4a3501f5b4cf5082f1]`

- robot_0.support.FL_foot history ANY超过1N：on/off/unknown=325/200/0；history-on且current-off=4条，反向差异=0条，联合有效525条；无逐槽时间/valid，不替代current。 `[claim:efea48328068f802b27a650d61ddbc462613594944fe6cec76197462dd89d8f5]`

- robot_0.support.FR_foot current marker：on/off/unknown=344/181/0，有效525条、占比0.655238；onset/offset=28/27；完整off段27、最短off段1条；最长off：步449–460，12条，格时长代理0.24s、观测跨度0.22s，左/右截尾=false/false。 `[claim:001c78a1cf65589626208075c0ba2d82dffa52830c9733cc95e339636a6f4338]`

- robot_0.support.FR_foot速度通道XY<0.05m/s且current marker off：未观察到满足记录，谓词未知0条。 `[claim:8e977a63e0ac24c1b22a4007c9d2d65d9e8110ce26c2c6e9917fa55a7e3a81e8]`

- robot_0.support.FR_foot足参考点相对根部的首帧xyz=[0.203376, -0.171317, -0.319153]m，身体坐标xyz范围=[[-0.128237, 0.461236], [-0.339312, -0.050522], [-0.36726, 0.004363]]m；另列速度通道世界XY范围[0.0049,6.441611]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:db49abd1be75297d1e793aaabb66e85461b148163a5d691e40c5781b9cc3440e]`

- robot_0.support.FR_foot history ANY超过1N：on/off/unknown=353/172/0；history-on且current-off=9条，反向差异=0条，联合有效525条；无逐槽时间/valid，不替代current。 `[claim:72460c44b454d332844a3f39366e88e0069bbdada9aba1944fa6a8017fbb8eaf]`

- robot_0.support.RL_foot current marker：on/off/unknown=195/330/0，有效525条、占比0.371429；onset/offset=29/29；完整off段28、最短off段1条；最长off：步244–272，29条，格时长代理0.58s、观测跨度0.56s，左/右截尾=false/false。 `[claim:181eca595d85f3bf23486e2e314a7d088f501316c4a7918f76f2ea47ddb3b02c]`

- robot_0.support.RL_foot速度通道XY<0.05m/s且current marker off：步4–4，1条，格时长代理0.02s、观测跨度0s，左/右截尾=false/false；该段XY最大/中位速度=0.013528/0.013528m/s，XYZ诊断最大速度0.728331m/s，力模最大0N，上轴竖直分量范围[0.99392,0.99392]（姿态有效1条）；不据此断言直立举足。 `[claim:7d5b0a74fda876a346e003b603e9bdca636752ed63522675fcbd17f3e8a7f018]`

- robot_0.support.RL_foot足参考点相对根部的首帧xyz=[-0.294731, 0.137524, -0.339657]m，身体坐标xyz范围=[[-0.571174, -0.162799], [-0.009281, 0.137524], [-0.39268, -0.106883]]m；另列速度通道世界XY范围[0.002483,6.105902]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:6c204f95fa2ceb0e3b6f82383d9898c3440c105ebbf52124348217817bf664e9]`

- robot_0.support.RL_foot history ANY超过1N：on/off/unknown=207/318/0；history-on且current-off=12条，反向差异=0条，联合有效525条；无逐槽时间/valid，不替代current。 `[claim:87ee6a69567959c09777b275422256abcc3cc114b63b03f75a719d74473a70d1]`

- robot_0.support.RR_foot current marker：on/off/unknown=317/208/0，有效525条、占比0.60381；onset/offset=29/28；完整off段28、最短off段1条；最长off：步470–484，15条，格时长代理0.3s、观测跨度0.28s，左/右截尾=false/false。 `[claim:ed84f77c7e371d7b11be124f6501ffbd7cfcaf722dd587fe081580b247f09a52]`

- robot_0.support.RR_foot速度通道XY<0.05m/s且current marker off：步102–103，2条，格时长代理0.04s、观测跨度0.02s，左/右截尾=false/false；该段XY最大/中位速度=0.025142/0.01571m/s，XYZ诊断最大速度0.048352m/s，力模最大0N，上轴竖直分量范围[0.769743,0.771788]（姿态有效2条）；不据此断言直立举足。 `[claim:4e47154c038be88693ec2f8a86c3eeeadbfdf1996774c1137f53e8a78b258428]`

- robot_0.support.RR_foot足参考点相对根部的首帧xyz=[-0.234788, -0.121709, -0.295177]m，身体坐标xyz范围=[[-0.559581, -0.08564], [-0.132029, 0.079675], [-0.372596, -0.111086]]m；另列速度通道世界XY范围[0.006278,4.214084]m/s；二者参考点映射未验证，不把速度当该位置精确导数。 `[claim:3829bef270b2e660dae3312fa18a3fdc6472246c4aa04db1307a2db2ad34dc61]`

- robot_0.support.RR_foot history ANY超过1N：on/off/unknown=342/183/0；history-on且current-off=25条，反向差异=0条，联合有效525条；无逐槽时间/valid，不替代current。 `[claim:7d538dac92dcceacb3b6ee883f1469f3d75a6b31a9cb0685dd07544665763adb]`

- 按robot_0.support.FL_foot/robot_0.support.FR_foot/robot_0.support.RL_foot/robot_0.support.RR_foot顺序，记录净法向力模份额=[0.263065, 0.262533, 0.174739, 0.299663]，有符号世界Fz份额=[0.273573, 0.271268, 0.176731, 0.278428]；各自联合有效记录525/525，未知0/0；不是重量份额、完整外力或物理冲量。 `[claim:7e9ab1804c37d8f053790d8680991f7754364991f66d36972f7f3b37716d89a8]`

- 按robot_0.support.FL_foot/robot_0.support.FR_foot/robot_0.support.RL_foot/robot_0.support.RR_foot顺序、权重[1, 2, 4, 8]编码，各组合0–15记录数=[62, 6, 8, 47, 38, 2, 18, 27, 21, 15, 7, 164, 28, 9, 22, 51]；联合有效/未知=525/0；全off不证明全身腾空，组合不命名步态。 `[claim:79c873e8c6a5e43445a5a51374fa8d9b7e043aedd4bb001a22978f1ec01ab7f9]`

- 终点步525、时刻10.5s，manager/wrapper标记={"done_mask":false,"wrapper_done_mask":false,"termination_0":false,"termination_1":false}；termination索引依次对应[time_out, base_contact]；末5条步号=[521, 522, 523, 524, 525]，base current力模=[[0], [0], [0], [0], [0]]N，另列history各槽力模=[[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]]N；任务结果保持存档评价器权威。 `[claim:f9f56111cf67cba74b5dceac3a5c511e996e7bbe26597b6da99d447f37605519]`

- robot_0.body.base current越阈0条，区间：无已观察阳性；history ANY越阈0条，区间：无已观察阳性；单条短事件保留。 `[claim:6b9631951bf57baaa0aac7f72271621a72a9d01d1e9994bb826340cf3dde83e5]`

- 低进度窗口确认与足速度组合：至少一足XY≥0.05的true/false/unknown=39/462/24；全部足XY<0.05的true/false/unknown=0/524/1；最长至少一足达阈值段：步105–131，27条，格时长代理0.54s、观测跨度0.52s，左/右截尾=false/false；最长全部足低速段：没有已观察到的该状态区间。前一段按robot_0.support.FL_foot/robot_0.support.FR_foot/robot_0.support.RL_foot/robot_0.support.RR_foot顺序，分足达阈值记录数=[25, 25, 17, 20]，分足有效数=[27, 27, 27, 27]，同一足覆盖该段每条记录=[false, false, false, false]。合取可由不同足交替贡献，不断言迈步次数或整体静止。 `[claim:b71b462a45faa3185b86eb5d2e2f30c3fc4d590c1c9ab08a6987c7cc1f385646]`

- 该成功回合仍有已观察回撤0.125879m（峰步84、谷步116、返回步141）；回撤过程与成功可同时存在，返回进度不等于已证明恢复平衡。 `[claim:8e2a961c9d043c2741bb8c081773f2f0d497a57dd47eeaa193b5f6d5251766f3]`

## 可回查的原数组与引用

完整逐字段值、原NPZ chunk SHA/row/run/env/step和P1a byte offset保存在 `outputs/trajectory_p1b/raw_spotchecks.json`。下面列父线程复核的关键锚：

| 锚 | 原数组/重算与解释 |
|---|---|
| r4 e129 step356 | current base=[0,0,0]；history末槽力模635.403135N；一条history越阈、右截尾。不能改为current接触。 |
| r4 e35 step14 | 按FL/FR/RL/RR为[true,true,false,true]，权重1/2/4/8，code11。 |
| r4 e60 step626 FL | XY=0.044785992941m/s <.05；XYZ=0.055060159744m/s >.05；M04只能用XY。 |
| r4 e60 119–220 | 102条每条有至少一足XY≥.05；逐足89/70/87/72；各足有效102；无单足覆盖全段。不能据此命名步态。 |
| legacy e54 | 归一化wxyz的up-z自278步为负；FL低速off326–588，263条、5.26格时长、5.24跨度、右截尾，非直立举足证据。 |
| legacy e60 | 存档success；峰84、谷116、返回141；回撤0.125878811m。成功与回撤同时存在。 |
| 成功长lowoff | r4 e197 RL88–112、r3 e52 RR109–133，25条、0.50格时长、0.48跨度；各有同cell失败见证。 |

每句均可从claim.operands走到本地summary/activity/cohort记录或P1a byte locator，再到原数组evidence_ref；无需复制P1a的1.29GB进卡。

## 分母亲核

| 描述 | true/false/unknown记录 | 有效记录分母 | 阳性/可判定回合；unknown |
|---|---:|---:|---:|
| low_progress | 27653/249122/21600 | 276775 | 186/186；678 |
| low_progress且any足XY≥.05 | 17555/259852/20968 | 277407 | 183/183；681 |
| low_progress且all足XY<.05 | 10098/287645/632 | 297743 | 67/368；496 |

这三项是各自的三值合取，不以27653条low_progress阳性记录充当共用分母。回合presence的阴性必须全覆盖；有阳性见证即可确定阳性。回合等权与sample_cell加权池化分别列示。路径按相邻边计数297511，不混作298375条记录。

| 组合 | all（298375） | success（229344） | failure（69031） |
|---|---:|---:|---:|
| code0 | 30733 | 21720 | 9013 |
| code11 | 107361 | 91018 | 16343 |

力模与signed-Fz分别有有效分母；单回合份额资格保持P1a未乘dt的sum_N阈值，池化分开声明N·s阈值。缺步、unknown、未处理、完整run与截尾run各列；故障不能通过把unknown当阴性隐藏。

## 必要事实覆盖与未展开理由

原18预选及输入hash完全不变。P1a严格门必要247/247和完整309/309均pass、无unsupported；P1b独立复算每个紧凑指标及分面，并将246个样例必要事实与1个队列必要事实映射到实际claim。映射位于 `final_independent_audit.json.necessary_coverage`。
这里的覆盖是核心事实进入主视图或展开卡且完整细节可追溯，不是每个原始标量都写入中文：原始/归一化四元数诊断、全部转换行、力累加值、关节逐项范围与nearest全搜索岛保留P1a及底账；不把保留在底账的内容冒称主视图已逐字表达。周期/限位/地形几何资格、历史link/COM映射继续未启用；缺父代、动作/力矩/逐步奖励/all_nonfoot等明确不可用。

## 验证边界

无作者serialize_prompt_payload、实际请求或字节预算验收；无P2/P3、视觉、闭环收益、新policy/robot泛化；训练reward/结果规则/旧report ABI未改。父线程独立验收仍待执行。
