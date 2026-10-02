# 旧开发案例：完整时序可读视图

仅用于人工检查工作集是否更容易看懂。**不是新的诊断器、N5 评阅输入或验收结果，也不说明加图能提升模型准确率。** 不改变 N5 的事实输入，不增加正式视觉评阅分母，不据此重新评分。

来源是旧 `r1.e60 / stair-0061` 的同一份 `related-raw`：

`../n5_review_capacity_development_r2/materials/related/episode-02.json`

SHA-256：`1eb9a45bd69bcea985074875b99fcf3d34d2673a6b569c600678958af79965f1`。

共 525 个连续控制步，间隔 0.02 秒。所有面板使用相同的 1–525 横轴，宽度 2400 px，没有选择阳性窗口、重采样、平滑或路径简化。文件实际供应的 16 个连续/离散字段均已解码并与原有 C-order 字节哈希核对；未供应字段没有补零或推测。没有读取自动报告、模型答案或调用生产分类器。

| 图 | 查看内容 | 解释边界 |
|---|---|---|
| [01_root_motion.png](01_root_motion.png) | 根部位置 xyz；相邻位置差分 | 静态 terrain origin 平移后仍为世界轴；没有测量的根部速度列，差分明确单独标注，第一步缺定义断线 |
| [02_root_attitude.png](02_root_attitude.png) | 有符号 roll/pitch、up-axis z、原始 wxyz 四元数 | 保留符号、单位和完整曲线，不翻译成高层动作/失败标签 |
| [03_foot_reference_positions.png](03_foot_reference_positions.png) | 四个具名足部参考点的 xyz | 刚体参考点，不等于脚趾皮肤或已经证实的接触点 |
| [04_foot_velocities.png](04_foot_velocities.png) | 四足世界轴速度分量与平面模长 | 平面模长公式直接写出，不给出滑动、接触或动作意图分类 |
| [05_foot_surface_relation.png](05_foot_surface_relation.png) | 参考点 z、原档案的投影局部表面高度、两者之差 | 几何投影关系，不是 contact-pair 证据或解剖间隙 |
| [06_force_norms.png](06_force_norms.png) | 当前力模长与传感器缓冲最大模长 | 缓冲槽没有逐槽观测时间/有效标志，不能说成完整 physics 时序的峰值 |
| [07_recorded_flags.png](07_recorded_flags.png) | 原始 done/termination/valid 五列 | 只画原记录，不重写任务终止或成功判据 |

`manifest.json` 对每个面板登记原字段、C-order 列号、绘图算术、单位、525 个输入点与缺定义点；绑定原输入和脚本 SHA。`delivery_receipt.json` 核对回传 PNG 的完整 SHA 和实际像素尺寸。

图像是**有损的可读辅助视图**：线条只是连接离散采样点，不代表未记录的连续运动。精确极值、阈值关系、持续时间和独立算术仍应回到同一原 JSON/NPZ 核验。不能凭图片的视觉分辨率确认逐点误差或将所有曲线转成新的必要事实。

本机没有现成 Matplotlib。按 root 的追加授权，使用服务器现有 `/data/yuyangliu/envs/isaaclab6_validation/bin/python`、Matplotlib 的 CPU Agg 后端生成；显式清空 `CUDA_VISIBLE_DEVICES`。没有安装/修改环境，没有模型/API/GPU 调用。远端路径仅为本次开发绘图：`/data/yuyangliu/AGENT/feedback_p3new_20260929/trajectory_readable_view_r1/`。

已逐张查看 7 张 PNG，核对图例、坐标/单位、具名部位与文字布局；这一检查只确认可读性，不裁决过程语义。查看工具提供的是缩放预览，例如 01/02/05 分别显示为 1856×1336、2048×1024、1632×1533，不能称为以原始 2400 px 宽度逐像素阅读。原始输入保持不变。
