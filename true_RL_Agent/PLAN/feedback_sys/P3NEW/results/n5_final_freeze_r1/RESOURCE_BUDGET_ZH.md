# 采集资源与费用登记

2026-09-30。本页整理两项只读独立核查，不是新采集或模型执行回执。来源核对者：`/root/n5_readiness_gap_audit`（资源）、`/root/n5_final_source_preflight_r4`（费用）；root复核公式与既有账本，未停止vLLM，未移动或删除资料。

## 存储与内存

Stair预选env 2/63的最大步数600/637，Push最多412步。三seed双视角，最终原图最多7422＋4944＝12366张。Stair每seed结束才裁尾，其最后seed裁尾前最多7496张；Stair→Push串行后，总量上限仍是两任务最终总数。

服务器旧r3全密采、r4双env及Push部位密采的原图实测：Stair 1171542–1713838 B/张；Push 1647548–2698005 B/张。最大实测大小乘最大帧数为26059042356 B＝24.27 GiB；图像加50%余量得36.40 GiB，数值/元数据/报告/日志留3.60 GiB，**本次按40 GiB工作估算预留**。这是实测推算，非压缩率保证。原RGB的PNG极端上界约6223848 B/张，全部71.68 GiB，单列参考，不作为无条件阻塞。

2026-09-30 01:49–01:55 UTC只读核查：`/data`可写，空72571379712 B＝67.59 GiB；没有更大的可写持久挂载。D盘空2269268418560 B＝2.064 TiB。沿用`/data`，不新增存储平台；启动前与Stair完成后检查余量，下一任务需自己的剩余估算＋8 GiB操作余量。若不足，先将本次已完成Stair原件直接下载D并逐文件SHA核对，才可释放**仅本次已保全的远端PNG副本**；不删除NPZ/manifest或历史资料，不在同盘先造全量tar副本。此备选尚未执行。

两张RGB数组约11.87 MiB；PNG编码有额外临时副本，逐步落盘。元数据仍随时刻累积：旧Push v4的412步manifest文件15.23 MiB，Python深尺寸约21.11 MiB，六回合约126.63 MiB。登记1 GiB采集附加主机内存余量，不称整个Isaac进程RSS上限。Stair全部原始数值数组上界233923680 B，64步块约7.48 MiB；Push全部9017856 B，128步设备缓冲＋host副本约1.78 MiB。主机MemAvailable约265.83 GiB；GPU峰值及Isaac总RSS未经本轮实测。

源码依据：`runner/native_video.py`逐张PNG写入、元数据保留及zlib编码；两任务`numeric_capture.py`分块和实际字段；`runner/numeric_evidence.py`压缩NPZ写入。具体源码均由r4的400文件清单绑定。时限依据为r4继承的任务配置/映射及旧Push原始412步manifest；旧实测目录为服务器`feedback_p3new_20260929/{stair_dual_view_dev_r3,stair_multi_env_dev_r4,push_dual_view_dev_r3,push_multi_env_dev_r4,push_body_grounding_dev_r2}`。

两GPU各29024/32607 MiB由既有VLLM占用，0%利用率不代表可用；生产lease与资源检查仍需通过，不停止这些进程。采集计划只使用一张可用32GB GPU，Stair→Push串行。

## 历史API与正式预留

| 历史类别 | DeepSeek调用数 | 峰价usage折算USD |
|---|---:|---:|
| 图像读回 | 1 | 0.000227400 |
| 旧可恢复观察 | 2 | 0.003589068 |
| 校准r1/r2 | 2 | 0.080592900 |
| N.4全部作者 | 8 | 0.243092124 |
| 原生观察r5–r11 | 12 | 0.117887928 |
| 合计 | 25 | 0.445389420 |

核对依据：r11 `control/usage.json`的23次/$0.402907032已含校准；最后N.4只增加2次/$0.042482388，不再加整份8次费用。容量facts的DeepSeek材料是synthetic且actual_calls=0；实际facts/报告执行为桌面SOL，费用未知。历史峰价折算不是发票实付。

正式DeepSeek评阅80次×$0.15，Stair观察2次×$0.05，新增预留$12.10；加历史$0.44538942＝$12.54538942，小于原总cap $15，余$2.45461058。N.4的$4是子cap，不另加额度。单次360000输入＋32768输出按[2026-09-30核实的官方峰价](https://api-docs.deepseek.com/quick_start/pricing/)为$0.1473216；每次仍按完整实际请求核预留，失败计账、无自动重试。桌面模型费用单列unknown，不记0，不包含在此API估算中。

以上只有规则冻结和资源安排；尚无正式N.5结果。
