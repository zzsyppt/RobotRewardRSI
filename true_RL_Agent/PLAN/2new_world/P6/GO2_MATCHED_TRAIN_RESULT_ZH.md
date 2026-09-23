# D：Go2 匹配训练对照结束，余量门仍 NO-GO

2026-09-12。执行前的[计划](GO2_MATCHED_TRAIN_DEV_ZH.md)和[声明](go2_matched_dev_declaration_01.json)不变。本轮工程执行及离线核对 PASS，开发资格 NO-GO；没有正式资格、Genesis 或新的 holdout 测量。

## 测量与解释

| 条件 | train 扰动 | eval 扰动 | 原 dev 三个 seed | 合计 |
|---|---:|---:|---|---:|
| C：固定 B checkpoint | ×1 | ×2 | 60/64、59/64、57/64 | 176/192（91.67%） |
| D：新随机初始化训练 | ×2 | ×2 | 64/64、64/64、64/64 | 192/192（100%） |

D 的 low/medium/high/repeated 均为48/48；240次计划扰动全部发生并验证，全部记录恢复。C 的16个失败场景在D成功，原176个成功仍成功，描述性差值为+8.33个百分点。它没有进入原先冻结的 [0.2,0.9] 开发区间。

两次使用相同 Go2 PPO、v3奖励源码、初始模型hash、所查完整推理ABI，以及相同eval协议、原三个dev seed与场景分配。B/D执行源码快照不同；本次没有保存完整初始原状态或逐步dwell轨迹。因此只报告这一个训练seed下的观察，不宣称严格单因素因果或多seed复现。

当前证据支持：在这批条件下，Go2具备恢复能力，匹配强度训练后的reference仍饱和。G1新reference已具备资格，不能把Go2的余量问题与G1旧reference的站立自旋问题混称为“两个机器人训练都崩了”。

## 执行与验收

- 随机初始化seed42，500×4096×24＝49,152,000 env-steps；最终iteration499。训练209,350次施扰attempt/verified相等，reward执行12,000个control steps、60,000次term调用。
- 只执行1次训练、1次public-dev（10042/10043/10044，各64episode），accept/sealed/final/provider/重试均为0。
- [42个下载文件](go2_dev_D1/download_manifest.json)全部核对SHA及大小。[离线审计](go2_matched_dev_audit_01.json)重算C/D计数、分层、方向、写入速度增量、时序及保存的恢复窗口，工程PASS、开发NO-GO分开记录。保存的恢复时间不等于独立证明连续dwell；该限制保留。
- [目标收尾](go2_matched_dev_closure_01.json)重新计算实际checkpoint字节hash：1495390b192b913432b8472cb150e2db07fdc2eb811eeabe4ab87c577b24e0ff。总wall679.348秒，约0.18871 GPU分配小时，低于3600秒上限。
- 373份产品源码在运行前后不变；本轮仅将既有开发入口的协议、reference及session参数显式接到canonical composition，没有修改公共宿主、PPO、物理执行或资格阈值。旧B入口原字节已保存，SHA仍与B声明一致。
- 原B两格的summary/manifest/checkpoint、G1新资格49文件及c6完整1271文件（含71份checkpoint）均未改动；本任务进程已退出，GPU compute为空。

本轮仅做必要CPU物化与Ruff检查，使用目标实际环境preflight后启动。没有修改产品源码，不重复跑整套回归；此前产品验收的版本与范围继续见[当前验收](ACCEPTANCE_ZH.md)。离线脚本的两个输入结构假设按实物纠正：训练协议在physical_protocol内，episode_index随同一runtime的三次reset递增；实验输出和原声明未改。

## 停止与剩余问题

C有限网格与D匹配训练假设均结束，两个NO-GO都保留。不追加档位、换seed、改奖励系数或自动重训，也不消费后两相寻找通过理由。

P6剩余问题是Go2公开任务域如何同时表达有意义的扰动泛化和可测的改进空间。继续工作应先审查任务域及训练/评测关系，形成独立且有物理依据的研究声明；新的任务定义不能用已消费holdout挑选，也不能将这次停止条件改写为通过。此处没有预声明下一轮扫描或资格训练。Go2的新资格及对应成功/失败原生证据尚缺，P6整体仍NO-GO，未进入P7。
