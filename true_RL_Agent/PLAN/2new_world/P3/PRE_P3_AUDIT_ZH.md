# P3 前置审计：必要修复与边界

日期：2026-09-08。起点为本地与服务器共同的 `6f17a4218f3039657147f5bbf61ce034d95b396d`，本地工作区干净，服务器仅残留 `__P0_SENTINEL_OUTPUT__/`。P2 工程验收与其冻结执行副本保留，不把后续代码变更冒充原 P2 源码。

## 逐项判断

| 项 | 判断与处置 | 验收 |
|---|---|---|
| 1 重复解析协议 | 成立。历史 timeout 改为 TaskConfig 从自己已加载的协议派生的只读字段；composition 不再为它构造第二份完整配置。不新增缓存或配置来源。 | 配置、composition、entrypoint 回归；旧协议 hash 与字段值不变。 |
| 2 import 顺序 | 成立。native_video import 归回项目 import 区域。 | 静态差异核对。 |
| 3 learner 证据 | 成立。Stair 保存构造核验结果到评测摘要及录像旁车；Push checkpoint eval 也在 load 前核验，并保存相同结构。构造证据不进入任务 outcome、评分或 task_evidence 身份。 | 摘要归属断言；目标环境实际评测。 |
| 4 初始 σ 前置条件 | 当前调用顺序没有错误。保留“构造后、load/learning 前”核验，明确其证据描述初始化而非 checkpoint 的已学习参数；不新增 learner 生命周期状态机。 | 核对所有生产调用点；新增 Push 调用也位于 load 前。 |
| 5 Push 7200 s | 不能仅凭 Stair 36000 s 判定活缺陷：已注册的是 G1-29 legacy Push，完整预算为 500 iterations，Stair 为 2000。Go2 Push 尚未注册，不能套用旧成本作准入承诺。保持旧合同；P4/P6 必须在实际新格子资格化时签发 placement deadline 与成本，不能静默改历史字段。 | 代码与现行预算核对；P3 的 Push 方法设计不宣称已完成 Go2 Push 完整预算。 |
| 6 Phase-0 probe | 成立。文件受字节冻结约束，不改。README 与任务指南明确它只适用于旧 RSL-RL 3，不能在当前后端执行。 | 冻结 probe SHA256 不变；历史合同测试。 |
| 7 sentinel 目录 | 成立，服务器只有一份 42419 B 占位 manifest。生成源为旧 Phase-0 verifier；改用 TemporaryDirectory 后，按确认的精确路径清除残留。 | 无活运行使用；删除前后目录与文件清单。 |
| 8 Kit 日志体积 | 已提交的 P2 取证日志保留，避免改写已签回执或增加压缩兼容 reader；删除新提交中的文件也不能缩小既有 Git 历史。本 Phase 原始日志留磁盘，Git 保存精简结果、来源与校验和，不再版本化重复启动日志。 | P3 ignore 规则与证据引用可解析；不重写 Git 历史。 |
| 9 recorder 测试 | 成立且已复现：目标 Python 执行该模块显示 ERROR，却 exit 0、没有测试汇总。假数据按真实 ARTICULATION_ROOT 合同补全，增加 WXYZ 输出断言；两项 Isaac 用例放到子进程，先由 unittest 打印结果，再把真实退出码交给 Kit。主回归进程不加载 Kit。 | 完整模块实跑通过；删去 fixture 字段的临时反例输出原 AttributeError、子/父 FAILED 汇总且最终 exit 1。关闭 fast_shutdown 的尝试虽打印 OK，却在底层析构时 SIGSEGV，已淘汰，未算通过。 |

## 实施顺序

先完成上述局部修复、定向回归与真实 recorder/评测核验，再开始 P3 协议实现。使用独立的 `/data/yuyangliu/isolated/new_world_p3/true_RL_Agent` 执行副本，保留旧 15 rounds 与 P2 checkpoint。验证只使用固定运行时和已有公开诊断产物；不调用 LLM、不消费旧 sealed 或新 final holdout。

本地定向 52 项、全量 1612 项通过（19 项按环境跳过）；目标 Python 专项 28 项全部执行通过，包含真实动态 recorder、469 帧编码和 checkpoint 张量用例。Stair 与 Push 实际评测均保存了 learner 构造证据与推理加载格式。最终部署、资源与历史核对见本目录 `PRE_P3_ACCEPTANCE_ZH.md`，这些结果不代表 P3 的新指标已经实现。
