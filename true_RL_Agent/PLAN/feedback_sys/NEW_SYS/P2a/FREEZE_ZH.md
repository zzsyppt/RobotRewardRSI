# P2a 冻结边界与独立核查预声明

基线 `aaf67aa23bd7e4aa8aae7a9ec9539f3aa45c6c3e`，专用分支
`codex/trajectory-translator-upgrade`。用户确认 P1d 已独立及父线程亲核放行。
本文件先于 P2a 生产修改及三源新卡片输出保存。只有主开发线程写本地；审查线程只读。
正常权限 `git -c core.longpaths=true status --short` 为空；沙箱长路径 D 非真实删除。
已读用户级 AGENTS，CodeGraph 对新增 trajectory 符号未提供有效索引，直接读取源码。
已核相关 skills；本任务是现有仓库 CPU 实现，不需文档产物/模型/视觉插件。

## 本轮范围

- Go2 Stair K1/K2/K3 各自完整公开队列、各自证据/卡片及正常 8192B public projection。
- 冻结 `trajectory.p0.measurements.v2`、NUMERIC_CONFIG、已有 v2 模板和选择规则。
  新增有严格版本资格的 v1 缺 history 分支；不伪造数组、不升级 current 为 history。
- 当前力和 manager 终止独立；缺 history 的区间/分母为 unknown，不是阴性。
  v1 文句明确 missing；终止原始证据及终止见证不因缺 history 被丢弃。
- 不运行 LLM、GPU、训练，不实现 Push/P3/P4、不修改采集器或接受协议。
- 周期、有效/名义限位、逐类地形、历史 link/COM 映射均保留后续资格义务。

## 来源与用途

| 源 | manifest SHA256 | 版本 | checkpoint SHA256 |
|---|---|---|---|
| K1 | 0389a10d7cb9dbe14fff063f43c56482fcc6fbd918281a37beacb59c043db8d3 | v2 | fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729 |
| K2 | fe488cb848dbe61454c0b9cc4fa6d8f8fc278deb1157065170577d9e91be5f20 | v1 | bcd8de3a9d0e6e6a6c84b8c7833dc4a1b0ed697758ddee6f9c9eb31416a2f93c |
| K3/R4 | 8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83 | v2 | 7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c |

三实际不同权重。K2 是完整训练 Gen8A，不是 K1 中途 checkpoint，但与 K1 共享初始化/seed；
reward 不同。K3 与 K1 同 reward、不同初始化，是既有调规则材料。
均按 development_regression/descriptive_audit，非三个独立随机实验；未见资格未建立。
K1/K2 binding ff0b754f…，K3 binding 1db7a52e…，不跨源逐 episode 配对。
来源关系以 `PLAN/feedback_sys/P0` inventory（不是 NEW_SYS/P0）为准。

K1/K2 metadata 已显示执行源码/后端环境不同；相同 binding/seed 不保证实际随机化、
命令和时间覆盖相同。本轮不新增生产父代算子；仅并列共同 current 定义及逐源分母。
按父线程补充确认，严格父代比较后续接入“同一当前环境、同协议条件、明确父子身份”的
档案，核对实际随机化/命令，冻结共同描述量/整回合或共同时间窗及组成，另行实现绑定
双方来源的描述差分。代码版本变化须解释其相关性，不机械要求所有 source hash 永久
相同。本轮不实现跨环境 bridge 框架，也不把父代比较永久设为 unavailable。

## 预声明核查

1. 每源按八 stratum × success/failure，从非空格按
   SHA256(`p2a-stair-v1|source_sha|run|env`) 最小值选一例；补每个尚未覆盖终局的同样最小值。
   选择仅用 manifest 结果/条件元数据，在新生产文字前落盘，不从好看的输出挑例。
2. 独立原数组扫描补关键反例：current 为零而 manager 终点触发；v2 history-only 单步；
   翻转先于长低速 off；成功回撤返回；XY/XYZ 阈值分歧；长 off 中单步重触发。
   每类选最小 run/env/step 合格例，空类明示无见证；不为填槽调阈值。
3. 每个预选例独立算根部进度/路径、归一姿态、逐足 current、on/off/截尾/短转换、
   低 XY off、16 组合、受力模/Fz、相对位置、关节范围、history 能力/终止。
4. 全队列独立 NumPy 复算全部现有 compact metrics 和各分面聚合；逐句检查数字、实体、
   单位、范围、引用、unknown、限定词。预选必要事实与公共摘要省略分开统计。
5. v2 旧测量 JSONL 与旧卡内容逐字节回归；manifest 仅允许代码 hash/实测成本变化。
   v1 最小夹具检查来源换绑拒绝、缺history、零current仍保manager、unknown分母、
   原始短事件、18保留槽及不足预算拒绝；重跑全部相关回归和冻结247/309门。
6. 原始日志保留，性能/峰值/存储按源记录。每源只保留一份新完整最终证据，短 outputs
   路径，不改 Windows 设置，不把 raw/GB证据入 Git。若发现通用 bug，记录前后并重验所有受影响源。

小阶段：冻结→版本适配与夹具/只读审查→三源重放/独算→根线程复核/提交。
开发自测与内部评审不替代委托父线程最终放行。交付 commit/push 后停止，不进入 Push/P3。

## 2026-10-03 审查后资格澄清（不属于原预声明）

父线程独立验收指出，第 2 项的原实现只选取完整单条 current on，没有检查前侧长 off。
原 K1 r1.e0、K2 r1.e2、K3 r1.e0 只具备短 on 资格；“长 off 重触发”分类撤回。
原文、原选例账、原审计日志及其哈希作为历史记录保留；原预声明并未预先精确定义下述条件。

此次按用户审查后澄清，资格改为：**前侧同足 off 已观察格时长 ≥0.5s，紧随 1 条完整 on，
后侧至少 1 条 off，且这段见证无缺步/unknown**。current 仍用冻结的有限力模 `>1N`；
off 时长使用 `N×dt`，不把 `(N−1)×dt` 观察跨度混同为格时长。前侧允许左截尾，
后侧不要求 ≥0.5s；缺步/unknown 分段，不能跨越它们累计资格。按
`(run, env, on_step, stable entity ID)` 排序取全队列最小合格项。

这只修正审计选例，不修改生产测量、阈值、模板、卡片/public 或来源冻结。
新增正例属于看过材料后的定向补正，不能称为原预声明兑现、未见测试或完整 E2 通过。
旧三例保留为 `isolated_current_on`；细节见 [RETRIGGER_CORRECTION_ZH.md](./RETRIGGER_CORRECTION_ZH.md)。
