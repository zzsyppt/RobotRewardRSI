# P3-R：作者修复帧证据保留

日期：2026-09-29。范围：R.0/R.2 的无条件工程项；不代表作者价值门通过。

## 已修问题与边界

以前，存在本回合失败源码的 `static_repair` 会整段删除任务反馈和选题提示。现在初稿、修复、源码续答均将 `public_feedback_summary`、`selection_guidance` 和已有 `exploration_directive` 作为必留输入。任务已经在自己的 adapter 中做过预算投影；通用 frame 不解释脚、台阶、推力或行为类别，也不从旧 rationale 重建观测。

普通修复仍省略可选历史、示例和运行细节。若任务已生成带 `capacity_projection_receipt` 的 minimum-core，回执绑定整组 task sections，此时保留这组已经裁好的视图：不能删其中一部分后继续沿用原 hash，也没有必要在 frame 层另造一条回执链。该组仍受 56 KiB 限制，完整请求仍按现有上限计量。

共有作者提示增加一条规则：多个过程观察不构成修改多个机制的授权；观察与假说分开，修复时保留当前焦点，旧 rationale 只是提案。此规则同等作用于 A/T，属于 R.0-b 及以后共有底座；R.0-a 直接重放冻结旧首包，不回写这条提示。

没有新增 schema、服务、任务 hook 或 repair 专用事实副本。两任务的 8/12 KiB 行为投影和来源边界保持；生产源码不依赖 PLAN。参考本地 Codex `codex-rs/core/src/context_manager/history.rs` 的 `for_prompt` / `raw_items` 区分：保留原始记录，改变发送视图。

## 真实历史回归

脚本：[rebuild_repair_frames.py](rebuild_repair_frames.py)。结果：[summary.json](results/repair_frame_rebuild/summary.json)。

- 固定 P3 48 个 episode 中实际失去反馈的 8 个末帧：A 2 个，T 6 个。
- 同 episode 首帧提供原聚合/行为反馈和焦点；末帧提供确切失败源码、修复原因与谱系；经生产 frame 投影和真实 provider request serializer 重建。
- A 保持原聚合反馈和原 missing 标记；不向 A 补 T 的行为数据。
- 重建后反馈、当前焦点逐字等于首帧；失败源码与修复原因逐字等于末帧。旧文件 hash 保持不变。
- 最大完整 payload **73,177 B**；最大 task sections **40,521 / 57,344 B**。没有实际超限证据需要再加 repair 压缩层。
- 没有 API 调用，没有重评旧输出，没有重读原始 NPZ；这验证实际请求投递，不证明作者会正确使用这些证据。

## 验证

两任务测试从真实生产 `PublicContextBuilder` 走到 `_chat_completion_request`，再解析真正序列化消息；覆盖初稿、带失败源码的验证器/单机制修复、无源码的推理耗尽、有源码的截断修复、源码续答。检查聚合与行为证据、焦点、整请求字节及 task receipt。另有两任务 minimum-core 原子回执回归。

命令（仓库根目录）：

```powershell
.venv/Scripts/python.exe PLAN/feedback_sys/P3R/rebuild_repair_frames.py
.venv/Scripts/python.exe -m unittest tests.unit.test_candidate_authoring_loop tests.unit.test_prompt_projection_and_mechanism tests.unit.test_task_prompt_adapter tests.unit.test_prompt_capacity_projection tests.unit.test_stair_phase_d_prompt_preflight tests.unit.test_push_behavior tests.unit.test_stair_behavior -q
.venv/Scripts/python.exe -m unittest tests.unit.test_memory_context_phase13 tests.unit.test_prompt_source_catalog tests.unit.test_prompt_artifacts -q
```

主回归 **123 项通过**，结果见 [unit_tests.log](results/repair_frame_rebuild/unit_tests.log)；最后一组历史容量、来源与回执测试 **106 项通过**。历史 8 例记录 PASS；单独核实了所有历史/重建文件 hash 与报告一致。

Stair 的既有实际投递测试同时接入同阶段的 CPU 后处理 finalizer，检查 `postprocessing → pass`、派生封存以及 CPU/墙钟耗时存在；GPU 租约释放顺序由对应 runner 接线测试负责。
