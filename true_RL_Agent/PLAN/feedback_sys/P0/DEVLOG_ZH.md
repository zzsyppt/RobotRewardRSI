# P0开发与核验日志

日期：2026-09-23。当前为P0审计修订；原用户审计保留。本轮P0修改限定在证据和离线分析；初次534文件基线作为历史快照保留，后续生产改动另行验证。

## 初次P0（历史）

依据总计划v1.2.1先写实施卡，再核源码/协议、取回177份原始JSON、检查71份旧checkpoint及2份当前reference权重；用生产请求构建器准备五类输入，沿用已有DeepSeek配置做8次短输出/媒体计量。Stair78、Push70、共享161项CPU回归通过、0跳过。

期间只在P0脚本修过Python3.10分块hash、导入路径和导出格式；SSH取件失败只重试取件，没有重发provider。原请求、receipt和失败日志保留。初次没有训练或simulator调用。

初版的不足是策略/作者库存不全，过多重复文档，最终seal/negative controls和预算矩阵增加了基线维护负担；这些设计在本次撤销，不能再按旧验收说明执行。

## 本次审计修复

1. 按原审计§1–§5逐项核对，区分真实遗漏和解释过度。先确定只修P0证据及下一阶段范围，不提前建立诊断服务。
2. 重写库存提取器：直接从session genesis/ledger/train/eval/prompt建立policy和真实父子边，独立记录训练初始化。补formal全训练池、最近Push及审计Gen8-A回放；只读获取formal36份候选源码并核Push11份远端checkpoint。
3. 新增Stair双策略分析，重算1728条facts及分层/共现/对照覆盖，调用当前summary validator。修复审读中发现的trial路径取错层级，并将“全部诊断均ever”收敛为症状ever与结局派生标志分开。
4. 新增实际作者A基线：AST提取46份rationale，绑定实际输入/回执/结局；详细核10条Push的数值引用。识别G4-B文字“惩罚”与实际负×负贡献不一致，作为作者证据使用反例保留，不回写历史实验。
5. 改写输入分析：历史4点拟合、另5点检查误差，最大7.60%；旧冻结参数和冗余派生估计现已删除，原receipt中的reservation不改，移除strict-token资格。9点一起拟合只报样本内误差。
6. 精简资源为GPUh/$、采集容量放到对应实现；主计划改为首批真实parent→candidate同版采集，不全量重放22个策略；A/T都新生成，机器核事实＋1名人工评语义。
7. 删除P0 seal、全目录封存及两项自造破坏用例、重复episode JSONL、原件压缩副本和两个一次性取回脚本。原始数据不删除，取证结果通过源hash/ledger/真实回执核查。
8. 同类审读补核真实split与执行source/runtime版本、区分作者parent和训练init、修正Phase 5统一离散度门及三类强制归因。详见 [逐条处理](./AUDIT_RESPONSE_ZH.md)。

## 当前验证与范围

四个离线分析脚本和汇总验收均实际重跑，来源、33条父子边、38份评估、行为分母和46份作者材料闭合；详情见 [验收报告](./ACCEPTANCE_ZH.md)。本P0修改不触碰生产代码，复核已有309项CPU记录；它们不能验证并行实施的Phase 1。

递归compileall首次误扫取回的历史候选目录，写pyc时遇Windows长路径限制；没有执行候选。当时改为只对15个P0顶层脚本内存编译，全部通过；历史36候选仍由AST提取核语法和身份。清理生成的P0缓存目录被自动审批策略拒绝（工具仅返回blocked by policy），因此保留缓存，不影响代码/数据验收。未修改系统长路径设置或历史源码。当时10份文档123个本地链接通过；本轮清理后13个脚本编译、125个本地链接检查通过。

无新增API、训练或GPU回放。用户审计已执行的Gen8-A结果从 `../audit4_replay/gen008a_v3/` 读取；不会把它计成本轮新运行。remote hash存在性有采样时点，后续启动前按原runner流程检查即可。

后续第一步仍是Phase 1离线接入边界，Phase 2在task-owned层保存必要时序，Phase 3检验作者使用价值。本次没有新增生产框架、数据库、通用任务语义或开放agent服务。

## 七项遗留的实修（2026-09-23）

1. 删除已完成的一次性输入准备/API运行脚本、旧token参数文件和两份派生frames中的旧估计字段；原payload/request/receipt字节不变，保留唯一局部affine分析。
2. 删除无消费者的stair_document_validation.json。
3. 原审计仅修正3处相对链接到上一级真实文件/目录。历史审计脚本不再读取已删除参数，改为回查原receipt的reservation；原审计JSON不覆盖。
4. collect_remote_baseline改为按来源相对路径复制原JSON，不再创建tar副本。本次未重新执行整个远端采集。
5. inventory的来源、会话本地根目录和已记录本地训练路径改成仓库相对POSIX路径，稳定source_id不变；消费者使用仓库根目录解析。
6. 十个人工Push判断及源码符号反例移至author_review_annotations.json；脚本按数据给定候选、hash、JSON pointer和AST条件核查，删除候选/代数特例分支。该数据仍是人工回顾标注，不是自动语义判断。
7. 只读核formal fca原parent权重hash，取回两份原始训练/manifest JSON并绑定formal genesis，登记Phase 2卡。未复制权重或启动GPU。

四分析与verify重新执行：37策略、33父子边、302来源、38评估、28组汇总检查通过。历史source_baseline.json原件未改，校验其manifest自一致和捕获身份，当前文件差异只作说明。新增GPU/API用量均为0。

第7项追加只读核查：两份训练manifest逐字段比较，确认v1→v2的评估规则差异和相同训练/机器人合同。追踪P6入口与runner发现“binding一致”不等于v1可直接进入v3；原parent合法祖先准入仍待P2，详见范围卡。本次仅补P0文档和链接，未执行网络/模拟器/API，未改任何准入代码。Phase 1旧离线范围明确由P1完整实施卡接替。
