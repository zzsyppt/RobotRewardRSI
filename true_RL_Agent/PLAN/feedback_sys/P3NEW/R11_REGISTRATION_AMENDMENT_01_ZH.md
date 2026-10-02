# r11调用前补充登记：像素坐标的原图尺寸

2026-09-30。**尚未生成或发送r11请求，0次API。** 本补充与[原登记](./R11_REGISTRATION_ZH.md)共同生效；原登记原文和SHA保持。只修输入坐标参照，不变案例、显示步、机位、测量、评分口径、模型、输出上限或一次调用US$0.05预留。

实际输入审阅发现，`xy_px`是原PNG像素坐标，但逐图文字标签没有尺寸。现将已有的原图宽高明确为`image_size_px: [width, height]`，并注明坐标以该尺寸为准。原PNG和坐标值不修改；不增加另一套归一化坐标。HTTP请求文字因此变化，必须另行准备和封存，不能复用旧请求SHA。

采集继续使用原401文件源`source_body_grounding_r1`，manifest SHA `b406b38f0e5befa647115a41b505474af3c93d207f84951e5b16b31bad448645`。r1/r2的原始资料、采集源码与回执不改，也不因此重新采集。

观察准备使用独立的`source_body_grounding_observer_r11`，从上述冻结源复制401文件，只替换`src/true_rl_agent/evaluation/visual_observer.py`；其余400个文件逐SHA相同。它不包含正在并行修订的Push短报告逻辑。

- 新观察源manifest SHA：`0e8668f839831a4ab86d99c99ad258fb03cc36f7e5ec821d0d014fa71adf0258`。
- 新观察源bundle SHA：`9977ea8bbf69eb133448c00aa03c1516cc8389c25b83ceb87448768c50516630`。
- 唯一生产文件新SHA：`b02ed4817e06efb6cf73786a9c18f511284239d7d0a4bc0d85b8b7d0dbb6e030`。
- 来源、差异及51项相关测试结果见[冻结回执](./results/body_grounding_observer_r11/control/observer_source_freeze_receipt.json)。扩大到既有媒体测试时发现一个v4 mock尚未迁移，已修测试fixture；没有为测试放宽生产采集边界。

完整r2档案核验和导出通过后，使用新观察源准备唯一请求。根代理仍须审核实际文字、12张原图、尺寸、42个S0、5条测量事实、72个点、来源及实际HTTP字节，之后才单独封存release。传输失败或原答错误均按原规则保留，不追加调用。本补充不代表观察者语义或N.5通过。
