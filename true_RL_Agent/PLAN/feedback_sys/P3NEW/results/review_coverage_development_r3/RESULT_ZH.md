# N5 评阅留痕工具开发验收

**机械工具验收通过；没有模型调用、GPU、SSH 或正式 N5 资格结论。** 旧 r2 原答、材料与评分均未修改。验收中构造的答案、FROZEN 卡均明确为 synthetic，只检验工具流程；没有把它们当成评委票、事实封存资格或正式 cohort。

## 实际改动

1. `run_confirmation_review.py` 在 report 阶段自动添加 `review_text_inventory`，并保存同源 JSON。inventory 含稳定 ID、精确材料 SHA/JSON Pointer、原始字符串或完整字符串列表；没有把事件表的数值叶子转为模型主张。当前 full 切片枚举 shared 中解释条款，身份绑定字段另属绑定检查；bounded request 通过已绑定的报告 locator 解开原 JSON 字符串，定位实际报告。观察者输出或其它布局必须明确提供 `review_text_roots`，不猜测根路径。原始材料完整保留。
2. 新 `confirmation_review_coverage.py` 检查每个 inventory 项恰有一条 reviewed/unreviewed 留痕。reviewed 必须关联指向该项精确位置的 claim；unreviewed 保持未完成。claims/fact_matches 的报告定位与所有 `evidence_refs` 必须能在实际投递材料内解析，SHA 必须匹配；联合路径、错误路径无法蒙混过去。每个 own sealed fact ID 仍须在 fact_matches 出现一次，missing/disputed 不得删掉。显式其它 `unreviewed_locators` 也阻止完整状态。
3. API runner 在保留原答后写 `answer_coverage.json`，覆盖或引用不合格使回执 incomplete。新增离线 `audit-answer` 子命令，桌面独立评委可走同一机械检查。它不调用模型、不修 JSON、不改票、不把引用可解析或 inventory 填满判成语义支持。
4. `prepare_confirmation_materials.pack` 同时接受 development 与 confirmation。confirmation 要求 source manifest 的 `frozen_inputs.capture_card.sha256`、`frozen_inputs.review_protocol.sha256` 与实际文件一致，采集卡 status 为 FROZEN；正式正文取同文件冻结内容，标 `operative_protocol` / `FROZEN_FOR_CONFIRMATION_NOT_RELEASED`。development 保持原草案方式。正式输入禁止进入 synthetic report-capacity 分支。小函数 `prepare_report_manifest` 保留全部 facts 材料，再追加明确的报告、算术记录及独立 task_definition；不制造 seal，也不自动放行。

## 验收证据

[`verification.json`](verification.json) 的 20 项检查全部通过，SHA256 `29714cc4fbb37459827b173fcde351bf2adfc27660f6b44c6ad1e2618da1aa6e`。

- 24 个回合及 12 个焦点包的 inventory 原文定位全部可解析。Stair 回合 39 项，Push 回合 20 项；对应焦点 61/54 项。数量是**留痕单元数，不是支持率或召回率分母**，复合语句仍由评委拆命题。
- 原 r2 完整材料逐对象保留；新的 report HTTP body 为 410,807 bytes，仅离线准备，未发送。request SHA256 `0e4aee61b51d88604f6a4ec1fcfbff134dcb9863e5b479b2f0e658dff4751533`。
- 原 r2 `complete=true` 无法通过新机械检查：旧请求本来没有新 inventory，另有 11 处报告引用不可解析（7 处 fact_matches、4 处 claims）。这属于新开发审计，不反向修改旧评分，也不能说 39 项旧原文全未被人或模型读过；只能说没有逐项留痕证明。
- 缺项、重复项、明确未核、伪联合 pointer、证据 hash 错误、少一条必要事实匹配记录、其它明确未核项，均无法取得完整状态；明确 unreviewed 是有效记录，但不是完成。
- facts 请求没有 report、inventory 或 arithmetic_audit。旧真实原量复制为临时 synthetic confirmation fixture，完整执行 36 单位 pack；冻结状态/正文、hash 不一致拒绝、正式禁止 synthetic capacity、报告保留所有 facts 均通过。临时大文件已清理。

实际冻结时须同时封存三个脚本：runner SHA `3e1f7fee8ed25585a720fd9d66acac19a3fac64deb5862f60bd62b0f223ceca0`；packer SHA `defde89cd2d171187378efafae99f61b2f774ca184073425a51c3f09b95afd11`；coverage helper SHA `80de74ca19f0bea8e9c9fc0e32ebaaf45b0ab06360d5805692b4193e938872b4`。新 prepared card 已绑定 helper SHA，重建会发现变化。

## 协议需要同步的最小改动

- 修改协议原第 88 行“runner 只查外壳”的旧描述：现在还核实际文本留痕、引用解析和 own facts ID 覆盖；全包双方 facts→report 屏障仍由 root 检查，工具不自动完成。
- report 输出增加 `findings.inventory_review[{inventory_id,status,claim_ids,reason}]`。inventory 只追踪原文覆盖，绝不充当语义评分分母，也不替代全部独立可证伪命题的拆分。
- 统一新引用根：`{material_id,sha256,json_pointer}` 从投递 envelope 开始，内容通常从 `/data` 起；JSON 字符串另列 `decode_json_string_at` 后再解析内层 pointer。多个位置列数组。保留旧单材料无歧义 pointer 的机械读取能力，但不修复伪引用。
- ABI/输入可用性/权限条款只能由独立投递的 task_definition 支持。自动报告自身、inventory 或算术 PASS 均不能背书；无法独立支持时记录 insufficient/unverifiable 及原因。该种判断可以已完成评阅，但不能计入“supported”。

工具仍**不能证明**评委真的逐像素/逐轨迹阅读、复合主张拆分完整、理由正确或语义判断公平；这些仍属评阅与审计。当前 inventory 的自动根对应现有切片与报告结构，新增布局需显式登记 text roots 后再冻结；不能仅凭机械通过声称覆盖任意未来格式。没有运行正式双评、现场 provider 调用或新容量实验。

桌面原答机械审计例：`python -B PLAN/feedback_sys/P3NEW/run_confirmation_review.py audit-answer --request <实际 request.http_body.json> --answer <未改原答.json> --stage report --output <新的机械审计.json>`。只创建新输出；旧答案不重写。
