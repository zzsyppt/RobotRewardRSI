# Phase 0审计修订验收

日期：2026-09-23，依据主计划v1.2.3及P0审计逐条核对。**P0_BASELINE：PASS**，机器结果见 [acceptance_checks.json](./acceptance_checks.json)。本次修复的是P0取证、分析与后续实施边界；本报告不验收Phase 1及以后的生产能力。

## 实际交付

| 目标 | 可复核结果 |
|---|---|
| 源码与既有修复基线 | 保留初次P0的534文件历史manifest及其dirty工作树身份；校验清单digest自一致，当前源码差异仅作信息记录；初次执行的Stair78＋Push70＋共享161＝309项CPU回归、0跳过记录保留 |
| 完整策略和关系目录 | 37个策略身份、33条真实已训练父子边、38个评估来源、302个仓库相对路径来源引用；原facts留在summary |
| 补齐遗漏的真实实验 | formal23训练条目/22权重；最新Push10候选/10边、2 accepted、实际训练seed42；服务器11份Push权重只读重hash均一致；另核formal原parent权重/hash/train/binding/manifest |
| Stair行为结构 | 两策略1728条v3事实通过当前summary validator并逐cell闭合；primary、全部diagnostic、cell/stratum、进度/closure步数及对照覆盖均可重算 |
| 实际作者A基线 | 10条Push详细数值引用核对、36条formal literal rationale及实际输入/结局绑定；formal23完成dev、12无对应完成训练、1训练失败 |
| 输入与token估计 | 原8次API和60条历史输入绑定保持；改为4点校准/5点检查，最大实测误差7.60%，实际usage费用约$0.030562 |
| 媒体可行性 | 原单帧传输净增578 tokens、覆盖/尺度与缓冲分析保留；生产视觉及新采集器仍待对应阶段 |
| 减少脚手架 | 删除seal机制、两项仪式性negative controls、9.88MB重复JSONL、重复原件压缩包、已完成的一次性取回/prepare/run脚本、旧token参数/派生字段与孤儿JSON；人工标注移到数据文件；预算改为GPUh/$ |
| 计划纠偏 | 首批真实parent–candidate同版采集，A/T各自新生成，单人语义审核＋机器核引用；task-owned比较规则和unknown保持明确 |

## 验证方法与结果

本次实际执行四个离线分析脚本：`inventory_extract.py`、`stair_behavior_baseline.py`、`author_evidence_baseline.py`、`analyze_prompt_baseline.py`。每项从保留的原件重算；Stair使用现行validator，作者只用AST读取候选，不执行候选代码。然后执行 `verify_p0.py`，当前28组检查通过，涵盖原始来源、真实ledger/train/prompt关系、行为计数、作者来源、全部API回执、媒体及已有回归结果。13个P0顶层脚本内存编译通过；本轮另修复用户原审计3个错层相对链接，125个本地链接按实际文件存在性复核通过。

309项业务CPU回归属于初次P0执行结果，适用当时的源码快照。P0校验不要求工作树永久不变，也不以历史测试证明正在实施的Phase 1；Phase 1须单独登记当前改动与检查。本次未为文档/离线分析重复整套业务测试，也未新增安全测试框架。

复核入口（仓库根目录）：

```powershell
.\.venv\Scripts\python.exe PLAN/feedback_sys/P0/inventory_extract.py
.\.venv\Scripts\python.exe PLAN/feedback_sys/P0/stair_behavior_baseline.py
.\.venv\Scripts\python.exe PLAN/feedback_sys/P0/author_evidence_baseline.py
.\.venv\Scripts\python.exe PLAN/feedback_sys/P0/analyze_prompt_baseline.py
.\.venv\Scripts\python.exe PLAN/feedback_sys/P0/verify_p0.py
```

这些命令重新生成派生JSON并核查原件，不访问API或启动GPU。无需seal或另建基线目录；原始历史请求/回执保留，修订由本文与开发日志记录。

## 结论与边界

当前已有足够材料启动真实策略的过程采集，原“没有可比candidate”结论撤回。P3 reference与Gen8-A的v3成绩可作描述比较，但不是历史父子且执行版本不同；下一步用真实parent→candidate同版重评。进度不是楼层、ever不是先后、不同seed/cell不自动等价。

作者确实使用了部分现有标量证据，也存在未获过程支持的机制解释及奖励符号误读。后续T应帮助修正这些问题，再检验真实修改收益；历史rationale不自动构成因果或前瞻预测证据。

本次新增训练、模拟器评估和provider调用均为0。用户审计提供的Gen8-A回放作为既有证据复用；约290秒外层耗时单独登记。标准时序、生产行为投影、记忆/查询与视觉仍按后续Phase完成，不能以本次基线PASS代替。
