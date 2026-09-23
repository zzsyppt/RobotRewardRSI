# Go2 同训练/评测强度的独立公开开发对照

2026-09-12。属于 P6，未进入 P7。C 的固定 B checkpoint 有限网格已经结束：×2 为176/192，NO-GO，不能修改该判决。本研究不延长其强度网格、不改 seed 来刷分。

## 问题与边界

C 测量的是按原强度训练的 checkpoint 在更强扰动上的表现。当前 G1 在 ×1.5 重新训练并换新 cohort 后的 dev 为96/192，而固定 B checkpoint 在原 dev 上为167/192；两个因素都改变过，不能把差值唯一归因于训练。本对照专门保持 Go2 的原三个 dev seed 不变，在 C 已测过的 ×2 条件下重新随机初始化训练，区分训练分布与固定策略评测。只使用公开开发信息，不依赖 G1 或 Go2 的 accept/sealed 结果选参数。

只允许这一次匹配训练，不增加强度档、方向/时序变化、reward 系数、训练 seed 或额外评测 seed。Go2 原 PPO、物理、资产、动作/观测、恢复标准与 stationary_v3 源码不变；训练和评测共同使用 calibration_protocols_01/scale_2p0.json 的完整协议（实际文件名及 SHA 以执行前 JSON 声明为准）。Push composition 冻结完整数据，公共宿主不新增机器人特判。

## 固定执行流程

1. 复用既有 dev-only 训练入口和 owned-process/GPU lease；将协议路径、session ID 和 v3 reference 显式传入 canonical composition，避免隐式默认漂移。保留 B 原执行入口的已冻结源码证据，不修改 B 结果或 checkpoint。
2. 本地物化检查与目标实际环境 preflight 通过后，另写不可变 JSON 声明，冻结源码/协议/预算/初始条件/实际 argv，再启动。
3. 仅一次 random-init seed42、4096×24×500=49,152,000 env-steps；随后一次原 public-dev 3×64，实际派生 seed10042/10043/10044；0 accept/sealed/final/provider/重试。训练加 dev 总 wall 上限3600秒（另沿用进程清理宽限），最多1 GPU分配小时。
4. 用生产 Runner 完成训练、绑定实际 checkpoint 后执行 dev；独立复算计数、注入、声明条件和 B/C 锚不变。结果与 C×2 分开记录；成功率差只说明同一训练 seed 下该分布干预的观察，不宣称多 seed 效应。

## 判决与停止

所有物理/身份/预算证据有效且 dev 落在原 [0.2,0.9] 时，本次仅判开发可用；之后才允许在完整新条件下另冻一次正式资格与未消费 cohort，不复用本次训练来伪造新 cohort 的训练身份。若 dev 饱和、低于下限或证据无效，结束此假设，不加档、换 seed、改 reward 或自动重训。技术失败保留原回执并停止，不重试。

本文件不是资格声明。正式资格、成功/失败原生录像与2×2总体收尾仍是 P6 完成条件；不得把开发 PASS 或旧 C 阴性回放代替它们。
