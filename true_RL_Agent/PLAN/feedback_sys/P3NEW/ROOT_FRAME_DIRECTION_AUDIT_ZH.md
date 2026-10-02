# Go2 根姿态方向开发核查

日期：2026-09-30。范围：Stair 双视角开发材料 r3，run 1 / env 282，以及 observer r7 的一条根姿态表述。本记录是来源与方向计算核查，不是 N5、腿部/接触语义评分，也不构成 observer 整体资格结论。

## 结论与触发原句

r7 原答写道：

> At steps 99 and 196 the body is pitched nose-down over the stair edge, with the front lower legs reaching down and the rear legs trailing behind.

固定 Go2 资产的实际关节与头部几何确认：`root = base`，机体 `+X` 为前向，`+Z` 为机体上方。对齐数值中，第 99、196 步的机体前向轴在世界 Z 方向分别为 **+0.341824、+0.482101**，相对世界水平面的仰角分别约 **+19.988°、+28.823°**。因此，按已核实的机体系与世界水平定义，这两个采样点的前向轴向上倾斜，原句中的 `nose-down` 与根姿态数值不符。

这一结论不判定原句后半部分的腿部形态或接触状态，不把稀疏点外推为连续动作，也不从根姿态推出头部位移、运动原因或任务成败。

## 材料绑定与对齐

以下路径相对 server `/data/yuyangliu/AGENT/feedback_p3new_20260929/`，冻结源码为 `source_dual_view_r3/`。历史文件未改写。

| 材料 | 路径与 SHA-256 |
|---|---|
| r3 导出数值 | `stair_dual_view_dev_r3_export/run01_env0282.npz`<br>`bbf9e675b1325346b59ad571d45321e37159e8afaf6f2f67570a277166465505` |
| 数值来源 manifest | 原件 `stair_dual_view_dev_r3/numeric_evidence/manifest.json`；本轮核验副本 `stair_dual_view_dev_r3_export/numeric_sources/archive_00.manifest.json`，同 SHA<br>`fc30504a7d0f74a6b8fbecddc780e7d216bcb5704808174cc486350c665131ae` |
| 原生采样快照 | `stair_dual_view_dev_r3_export/video/public_dev_seed_1.snapshots.json`；与 r3 源目录同名文件同 SHA<br>`6de77e7d7cc745b99f3082bc765e95081adc7dd619994ab622b62663ef005c93` |
| r7 原答 | `stair_observer_native_v3_dev_r7/response/raw_answer.txt`<br>`feacaa19159051a1ad255355d46e3fc3f2782af51283e529dbe60c2224a1a740` |

本轮上游数值核查确认：四个已选采样点的 native `xyzw` 重排为 `wxyz` 后，与 NPZ 的 float32 值逐元素相同；相应 `state_before = state_after`。既有 r7 alignment 记录覆盖 343 步、4 个字段，均为 aligned。这些是材料一致性证据，不是额外的视觉语义评分。本记录作者另以 CPU 标量公式复算了下表四点。

## 四元数与方向复算

输入按 `q = (w, x, y, z)`，表示根机体系至世界系的旋转；令

```text
forward_world = R(q) · (1, 0, 0)
up_world      = R(q) · (0, 0, 1)
forward_world_z = 2(xz − wy)
elevation_deg  = atan2(forward_world_z, hypot(forward_world_x, forward_world_y)) × 180/π
```

保留原 float32 输入，没有额外归一化；四点四元数范数与 1 的偏差不超过 `1.11e-7`。方向向量下表显示 6 位小数，角度显示 6 位小数。

```text
step   1: [0.9999427795410156,  0.0007748744101263583,  0.010667585767805576, 0.00037731757038272917]
step  99: [0.9819588661193848, -0.022247588261961937,  -0.17556169629096985,  0.06663438677787781]
step 196: [0.9632313251495361,  0.019036561250686646,  -0.248256653547287,    0.1009528785943985]
step 343: [0.9927985668182373,  0.0017850795993581414,  0.02586422674357891,  0.11695686727762222]
```

| Step | `forward_world` | `up_world` | 前向轴仰角 |
|---:|---|---|---:|
| 1 | `(0.999772, 0.000771, -0.021333)` | `(0.021335, -0.001542, 0.999771)` | −1.222405° |
| 99 | `(0.929476, 0.138676, 0.341824)` | `(-0.347754, 0.020296, 0.937366)` | +19.988030° |
| 196 | `(0.856354, 0.185030, 0.482101)` | `(-0.474414, -0.086798, 0.876012)` | +28.822697° |
| 343 | `(0.971304, 0.232322, -0.050938)` | `(0.051773, 0.002506, 0.998656)` | −2.919818° |

## 机体系的实际资产证据

导出 metadata 绑定 `robot_profile_id = unitree_go2_12dof_isaaclab_v1`、`kinematic_base_source = articulation_root`、`resolved_roles.base = [base]`。冻结与当前 `configs/robots/unitree_go2_12dof_isaaclab_v1.json` 文件完全相同：

- 文件 SHA：`970783e60cee799890028e0303ad2470cb9a257e27890bdf87564dadab55b804`。
- 重算 profile hash：`3e9c110449be639fceb48d6792fc8f683f43bd34978286325bba99547d20f1c0`，与导出 metadata 相同。
- metadata runtime adapter hash：`14f03caebe810885d5681d73812987b2d31706cf432c4de5597b14a9dbfcd3f0`。

profile JSON 为单行；关键路径为 `semantic_roles.base.targets`、`runtime_facts.asset_closure.root_identity`、`runtime_facts.asset_closure.entries`、`runtime_facts.morphology.default_body_pos_relative_to_root_m`。冻结源码 `src/true_rl_agent/robots/isaaclab_locomotion.py:40–47` 将实际 `config.scene.robot.spawn.usd_path` 强制设为该 profile 的 `root_identity`：

```text
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd
```

本轮直接读取 server 缓存，未启动 Isaac Sim：

```text
/home/yuyangliu/.cache/ov/client/https/f1ee52af0de7caec907cf4a6317543f0b852f5f2a2741d46d3f39dc0a781225c.usd
```

其 SHA `ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d` 与 profile 固定资产相同；引用 mesh 缓存 SHA `2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5` 也与 closure 相同。使用 `/data/yuyangliu/envs/isaaclab6_validation/bin/python` 导入 `pxr.Sdf`，通过 `Sdf.Layer.FindOrOpen(path).ExportToString()` 解码获得下列证据。**行号属于本次 USDA 导出文本，不是二进制 USD 的物理行号。**

| USDA 导出行 | 实际内容 | 方向依据 |
|---|---|---|
| 5；15–28 | `upAxis = "Z"`；`base` 带 `PhysicsArticulationRootAPI`，零平移、identity 旋转 | 资产原始根与 base 轴一致 |
| 74、97 | FL/FR hip joint `localPos0 = (+0.1934, ±0.0465, 0)` | 前腿在 +X |
| 132、155 | RL/RR hip joint `localPos0 = (-0.1934, ±0.0465, 0)` | 后腿在 −X |
| 113 | Head_upper fixed joint `localPos0 = (0.285, 0, 0.01)` | 头部在 +X |
| 257、309 | thigh→calf、calf→foot 的 local offset 均为 `(0, 0, -0.213)` | 腿脚沿 −Z 延伸 |

冻结 `src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py:77–86` 的 `articulation_root` 分支直接读取 root quaternion；`isaaclab_runtime_env.py:108–115,149–150` 将其转换为 recorder 的 `wxyz`。因此这里的方向判断来自绑定资产及数值旋转，不依赖 Euler 正负号惯例或摄像机画面朝向。

## 后续建议与边界

建议由任务使用外置、具名的 frame specification，绑定既有 profile hash 与实际 root USD SHA，声明 `root = base`、机体前向 `+X`、机体上方 `+Z`；在任务数值提示中提供 `root_forward_world_z` 等可复算字段。保留既有 profile hash 和采集绑定，公共 observer 层不硬编码 Go2 轴定义。

上述为实施建议，本记录不宣称代码已交付。此次仅开发审计与 CPU 计算：没有 API 调用、GPU 仿真、N5 读取或历史结果改写，也没有新增腿部、接触或因果语义结论。
