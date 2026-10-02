## 当前状态（2026-10-01）：开发资料，正式确认 INCONCLUSIVE

两任务采集已结束：Stair UTC 15:41:03、Push UTC 16:10:19，均退出 0 且原 wrapper/binding 通过。两任务各两个全新 SOL/high 观察上下文已完成，共 48 张原图实际读回；没有正式 facts/report、作者或训练调用。

Stair 六条原句的 root 非盲模块检查为 4 supported、1 contradicted、1 insufficient。错误包括 RR/RL 身体错置以及远侧前腿被遮挡却引用该视角作细节判断，见 [原句复核](./stair_observer_inputs/root_module_review_receipt.json)。输入中 `body_image_points=[]`，同帧具名几何已保存却未接给观察者。因此按既定修正纪律[整轮转开发](./cohort_disposition.json)，保留原输入、原答和全部分母，不在本轮重试取得确认通过。

固定 CPU 链已闭合：numeric、media、validate、audit 均成功，九项独立审计均退出 0，详见[短索引](./control_cpu/cpu_closure_summary.json)。原始算术、来源、范围与未核项详细记录保留；这些结果不等于模型语义验收通过。

Push 原六句的 root 非盲画面检查为 6 supported，机械审计零违规，见 [模块评阅](./push_observer_inputs/root_module_review_receipt.json)。R3 原包的[离线作者投递检查](./offline_delivery_original_r1/control/execution_receipt.json)已一次完成：Stair 两条未获准原句均未进入短报告及最终作者 payload，非视觉内容不变；报告分别 7,130/8,192 B、11,000/12,288 B，各只展示 1 条视觉观察。没有实际作者调用。

修正后的 Stair 两包开发观察共五句均获 root 模块支持，见[开发验收](../stair_body_identity_dev_r1/README_ZH.md)，不回判本轮。独立评阅全轨迹包的[静态容量检查](./control_cpu/capacity/capacity_summary.json)发现七个 SOL 单位按 DeepSeek 代理计数加预留超过当前 258,400 窗口；这不是 GPT 的实际 token 数，也不是生产主作者的输入。生产 DeepSeek 的[完整作者请求检查](../production_author_capacity.md)与[图文观察者检查](../production_observer_capacity_20261001.json)单列，不能用 GPT 的窗口替生产模型作保证。

随后只做了一次[最大旧包 CLI 容量探针](../cli_capacity_dev_r1/README_ZH.md)，实际输入246,277 token，生成32 token；加未来32,768输出与8,192余量后为287,237。该探针不是正式盲评，未生成facts或评分。下一轮评委需事前冻结独立进程窗口及干净上下文；完整原始曲线保持。生产DeepSeek本次作者输入约20.4k/15.8k，含最大输出约86k/81k；四个图文观察者含图像上界及最大输出约50.5k–53.3k，与验收大包分别说明。

源头修正只在 Stair 任务层与其资产配置中补四个 calf link 的同帧图像位置，沿用共享投影、原图、提示及每图六点上限。定位点不证明可见性、接触或运动过程；旧缺失保持空。修正后的真实输入与一次性开发观察另存 `../stair_body_identity_dev_r1/`，不回填原轮结果。后续须冻结真正未使用 cohort，阶段仍 IN_PROGRESS。

## 历史采集执行更新（2026-09-30）

UTC 15:41:03 Stair 实际退出 0，wrapper 结果 PASS、phase binding 与 R3 卡一致；Push 经原一次性队列于 15:41:09 启动（driver 3778560、wrapper 3778563）。Stair 导出、固定媒体选择和原生观察者准备三步均退出 0，原图盲包已下载，43 份运输文件逐一核 SHA/size；见 [运输回执](./stair_observer_inputs/download_receipt.json)。root 已实际逐张查看全部 24 张 1920×1080 原图，机器人全身及台阶关系可辨，保留浅色对比、远侧肢体和腹部自然遮挡限制；见 [输入画质记录](./stair_observer_inputs/root_input_quality.json)。

两名全新 `gpt-6.1-sol/high` 观察者分别于 UTC 15:52:05/15:52:30 启动，`fork_turns=none`，各自仅获本包匿名文字与12原图，平台记录的实际模型/effort 已核；见 [实际调用身份](./stair_observer_inputs/desktop_invocations.json)。尚未生成正式 facts/report；观察者回答、工具访问及逐句模块核对仍待完成。这是固定策略上的 N5 语义验收，没有新作者调用或训练。以下旧时点记录保留其历史含义。

清理并验证冗余存储后，Stair 于 UTC 14:23:06 启动，实际驱动 PID 3749979、runner wrapper PID 3749984。启动时数据盘空闲46,170,984,448 B，485源文件及检查点字节重新核验，使用冻结R3卡。见 `stair.driver.dispatch.json` 与 `stair.started.json`。此时为运行中，不是采集完成；Push、观察者和正式双评均未启动。未新增API调用。后处理控制入口为 `postcapture_preparation/postcapture_commands.md` 及 `preparation_manifest.final.json`；其真实新材料检查尚未执行。以下准备记录描述冻结时点，不覆盖此执行更新。

UTC 14:49:44 再次只读确认两个实际进程均存活，尚无退出回执，第二个 seed 已在写入双视角原图。采集中的 PNG 数会受各回合按自身终点裁尾影响，不能当作完成百分比或最终帧数。

已完成 [起始两机位原图读回](./early_render_readback/README_ZH.md)：全身和台阶可辨，自然遮挡及浅色对比限制保留，不据一对起始图认证整段。采集后 CPU 控制已部署，且获独立工程复核；见 [控制复核](./independent_control_review.json)。fresh SOL [材料搬运](./desktop_review_preparation/README_ZH.md)的五项针对检查及[独立复核](./desktop_review_preparation/independent_review.json)也已完成，验证原文、inline 图片顺序、原图字节和本人事实封存绑定。这些均未生成正式观察者/facts/report 原答；实际新材料容量、来源算术、模块与独立语义验收仍待执行。

UTC 15:03:18 启动本轮一次性串行衔接，实际 PID 3765194；见 `push.queue.dispatch.json`、`push.queue.started.json` 与 `dispatch_push_after_stair.py`。它等待上述 Stair 驱动真实退出、退出码为 0、结果 PASS 且 phase 身份符合冻结卡，再调用原样的 `capture_driver.py push` 一次。原驱动仍重新核源/权重、实时磁盘/GPU及 fresh output，失败不重试。队列启动不是 Push 已开始；后续以 `push.driver.dispatch.json`、`push.started.json`、`push.exit.json` 为准。队列不触发后处理、观察者或正式评阅。

[R3 容量工具](./capacity_preparation/README_ZH.md)已完成旧 Push13 的 CPU 对照：原文字块代理计量旧/新均为 61475，实际 desktop 完整序列化为 68834，两种口径不相加。角色窗口读取真实卡的 258400；未知自动压缩线、图片 token 和实际 SOL token 保留未知。此为旧文字样本的计量接线验证，不放行新材料或图文请求。调用时若在服务器使用 desktop 输入，需显式指向已部署且哈希一致的 `capacity_preparation/validation_fixtures/prepare_desktop_review.py`；正式材料搬运在本地同级 `desktop_review_preparation/` 执行。

[N1–N4 工程继承复核](./n1_n4_carry_forward_audit.json)未发现需要在 N5 之外补做的实质遗漏：词条反例、三种旧 Stair 形态回溯、候选台账/规程、开发媒体工程、四类投递与八槽读法已有证据。R2/R3 清单中 396 个 `src/configs/scripts` 文件 SHA 相同，本地最终冻结字节也逐一相符。历史早期四次文本调用缺完整 HTTP/推理存档，以及历史实际送达不覆盖后续源修订的限制仍保留；不能补造或追加作者槽。本项只支持工程继承，不认证新 R3 数据、图像或独立语义资格。

后续实际 fresh 上下文的留档复用 [可观察记录提取工具](./desktop_review_preparation/trace_control/README_ZH.md)。一次旧校准 CPU 对照中，10条工具输入与原 trace 逐字节相同，身份、模型/effort及可见用量相符。只读明确指定的会话文件，输出有限身份、工具输入、可见计量和压缩事件类型；不复制完整会话或私有推理。访问白名单、完整读图、重试与压缩仍由 root 对真实调用记录检查，不由工具自动宣称通过。

# R3 最终机器卡与采集驱动

2026-09-30，**FROZEN / READY_NOT_STARTED**。仅完成本地最终冻结和静态核查，未部署、未启动采集、未调用 API/模型。`pre_resource/` 中保留存储处理前的真实阻断草稿。

- `capture_card.json` SHA-256：`5c97844ff53027b80991a7308f991960e013df52fbc268046721e9771d42abc4`
- `capture_driver.py` SHA-256：`f812429b033b32c538818a668d55fec6d72963c7c254fcf9c47ade1e23591255`
- `runtime_source_manifest.json` 绑定最终 485 文件，SHA-256：`5cb1b36c5ca2e7af40d62c51e7cf1fe9aae4790e080353b8947cc6e14e5902d5`
- `resource_after_cleanup.json`：14:17:23 UTC 真实只读快照，可用 **46,171,140,096 bytes**；超过 Stair 31,162,953,808 / Push 28,276,851,520 原门槛。GPU0/1仅7/6 MiB，无GPU compute进程及指定capture/eval进程，新capture/control根均不存在。
- `budget_ledger.inherited.json` 与 `budget_ledger.working.json` 字节相同，6条历史记录、累计0.723248796美元，SHA `40250ddaa8f9b68d350b8c208a31a4dfb7289f39017713bb0742e01c3321aa91`。新增API预留2.40美元，原总账15美元不变。
- `final_card_readonly_audit.json`：PASS。checkpoint、媒体/选择规则、正式阈值不变；新的cohort/source/output、related_v2/blocks_v2、SOL-a/SOL-b和16次DeepSeek均已核。新载体校准达到原agreement门；未冒称语义验收。

原驱动仅替换最终卡SHA，继续单任务运行、复核源字节/checkpoint、检查fresh output/磁盘/GPU、记录进程与退出，不自动重试或启动评阅。Push仍要求本轮Stair退出0。任何后续部署/启动由root核查后执行，运行时资源检查和原runner lease仍生效。

构建命令已实际完成：

```text
python -B PLAN/feedback_sys/P3NEW/results/n5_confirmation_execution_r3/freeze_capture.py PLAN/feedback_sys/P3NEW/results/n5_confirmation_execution_r3/resource_after_cleanup.json
```

远端control目标：`/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930_r3`。至少需要原样部署card、driver、runtime_source_manifest；工作账本供后续独立模型release使用。实际capture argv已完整封在card，与CPU预检仅mode/output/video路径不同。更详细的阶段边界见 `../n5_confirmation_source_r3/PREPARATION_REVIEW_ZH.md`，其中存储不足段是较早时点。
