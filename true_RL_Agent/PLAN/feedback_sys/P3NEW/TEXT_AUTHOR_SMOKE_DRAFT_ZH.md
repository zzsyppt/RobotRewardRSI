# N4 四个文本作者读法烟测：待放行启动卡

当前交付为 `text_author_smoke_r3/stage.draft.json`，状态 `PREPARED_NOT_RELEASED`，准备过程 API / GPU / 训练调用均为 0。原 release 已由主代理保留并标为 `REVOKED_BEFORE_ANY_AUTHOR_CALL`。此前首包不变仅证明当前代码能投递旧 Push phase/full，不证明旧材料由当前 units 定义派生；该解释已纠正。新作者输入改用 `text_author_smoke_r3/current_push_materials` 的 current full/public/phase，保留 `delivery_offline_r1` 历史证据不改。Stair 两包未变；Push 两包因 full、phase 与任务区段摘要更新而变化，数值与报告正文不变。具体 old/new 请求哈希与三个变化位置见 `preparation_check.json`。正式执行须由主代理从当前 draft 另存并重新放行 released 卡。

本轮只检查真实作者能否按当前新过程报告作出有依据的解释和修改建议。使用旧 P3R 四个开发案例，各运行一个文本 episode；不新增种子、不训练、不比较 A/T，不以作者“价值”设门。四个案例来自两个已合格策略及两份历史 public-dev 测量，不得记为四个独立策略。`observer_readback_r1` 的真实图像调用不构成 N3 语义通过；本轮不接入其输出，也不手编 Tc-vis。

## 输入与四个固定槽

保持 P3 四案例原来的 `common_focus`、合格父策略源码和历史资格证据。通过现有 `confirmation_context.build_context`、`PublicContextBuilder`、`author_ceiling_probe.first_prompt` 装入当前真实过程报告，保留“原资格 + 独立诊断反馈”的明确边界。旧 helper 的 `T` 仅用于保留 behavior 字段，本轮没有 A arm 或 A/T 比较。

| 顺序 | 案例 / 原焦点 | 固定 generation / lane | HTTP bytes | 输入估算单位 |
| --- | --- | --- | ---: | ---: |
| 1 | stair-01 / c1_conditional_frontier | 001 / lane-a | 77293 | 17753 |
| 2 | stair-03 / height_ood | 001 / lane-b | 77288 | 17752 |
| 3 | push-02 / medium | 002 / lane-a | 65163 | 14923 |
| 4 | push-04 / repeated | 002 / lane-b | 65161 | 14922 |

generation 前缀固定为 `p3new-text-smoke-20260929-generation-`；lane 只是现有日志存储槽。输入单位为本地预算预检估算，实际 token 只认 provider usage。

当前原样生成的过程报告已冻结：

| 报告 | bytes | SHA-256 |
| --- | ---: | --- |
| Stair `results/delivery_offline_r1/stair/public_report.json` | 7229 | `395fbde36cb7f2b39801d49ae4cad42ea2885707b75bbc0bec720f7b8b622db4` |
| Push `text_author_smoke_r3/current_push_materials/public_report.json` | 10520 | `88465e53b1868e8b7669057736465a219d648ca2adc1082b44f4e09ac59f6ae3` |

四个完整 HTTP 请求在 `text_author_smoke_r3/prompts/<case>/request.http_body.json`，用户 payload 单独保存。每个请求及上下文、原资格资料、当前 public/full/phase、旧 runner、准备脚本与生产源码快照均有 SHA-256；共核对 59 个冻结文件。完整哈希见机器卡和 `preparation_check.json`。首包含完整当前过程文本，无 image_url，无 Windows 绝对运行路径。外置 `behavior_full.json` 仅供核对，未内联给作者；phase 的原始数据路径字符串保留为历史证据元数据，实际构建依赖按哈希映射到仓库内冻结副本。

当前真实 phase 没有绑定的 parent/candidate S4。不得把此前 synthetic S4、self 比较或视觉容量 fixture 放入本轮输入，也不得把它们算作真实作者读法覆盖。

## 作者与预算

沿用既定 `deepseek-v4-flash` 配置，`reasoning_mode=adaptive` / `reasoning_effort=high`、输出上限 65536、输入储备/实际上限 40000、无空最终回复自动重试。生产序列化器将 adaptive 解析为实际 HTTP `thinking.type=enabled` 并保留 `reasoning_effort=high`。实际请求 JSON 已经通过真实 Urllib 序列化器送到假 opener，未联网。

严格限定 4 个 episode，每个最多 5 次实际请求（draft、source followup、静态修复、reasoning exhaustion repair 均计入）；每个最多 2 次 source request，包含在该 5 次内。总请求最多 20、阶段支出上限 US$4。现有 `CostGuardedProvider(max_generations=2)` 每次发出前核对共享的两个 generation × 两个 lane 日志和单位储备，作者循环每个 lane 最多 5 次。不更换 episode 身份补跑，不删除失败/中断日志以重置预算。

旧 runtime_r2 价格快照为输入 cache miss 0.3、hit 0.006、输出 1.2 US$/百万；按全 miss 的每请求 40k+65536 计算为 US$0.0906432，20 次为 US$1.812864。**这是历史快照预算演算，不是当前具体作者模型的定价保证。** 放行前必须核对该请求名当前模型映射/价格，并填入 `budget.prices_usd_per_million`；本卡的 `pricing_status` 仍为待核价。实际输出超请求上限仍全额入账，不能裁成 65536；下一次调用前仍受现有共享单位上限约束。

## 最薄现有入口与远端运行

新增脚本只提供固定四案例的准备和单案例入口；调用链为：

`prepare_text_author_smoke.run_case` → `episode_loop` → 生产 `CandidateAuthoringLoop.run_round`，预算复用 `CostGuardedProvider`，费用读取复用 `P3R/author_ceiling_probe.usage_report` 和原 `P3/author_experiment.usage_report`。真实终态、源码、attempt records、最终 prompt ref、authoring evidence ref、usage 和模型/指纹回执全部保留。继续使用现有静态验证与有界 CPU component fixture，不启动仿真或训练。不要运行旧 `author_ceiling_probe.py` 顶层 CLI；其固定排程会产生 32 个 episode。

运行主机固定 `humanoid-server`，Python 固定 `/data/yuyangliu/envs/isaaclab6_validation/bin/python`（主代理已核对 3.12.14）。凭据仅在该主机读取 `/data/yuyangliu/secrets/deepseek.env` 中的 `DEEPSEEK_API_KEY`；不复制、打印或上传密钥。仓库根路径由脚本位置推导，不写死 Windows 工作目录。唯一运行输出为仓库内 `PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1`，四次入口共享该目录。

主代理先完成如下检查后另存 `text_author_smoke_r3/stage.released.json`：

1. 核对具体作者模型当前价格/映射及批准预算，填 `budget.prices_usd_per_million`、`provider.verified_response_model`、`provider.verified_system_fingerprint`；保留价格/模型核对依据。将 `pricing_status` 设为 `VERIFIED_FOR_AUTHOR_MODEL`。现有准备脚本不会自动采用旧值放行。
2. 部署 `frozen_inputs` 全部相对路径文件，以及源码快照列出的 `src/true_rl_agent`、`scripts`、`configs`；不需要为本轮读取全部 NPZ、模型 checkpoint 或视频。远端重建四个首包并比对既有 HTTP 哈希；Windows 上已验证改输出目录仍字节一致，Linux 重建尚待核查。检查远端依赖与凭据文件可读，不能把未检查记为通过。
3. 核对四案例、原样报告、未执行状态与上述界限后，由主代理明确将副本状态设为 `FROZEN_API_RELEASED`。`stage.draft.json` 保留不改。准备检查已确认 draft 在读取凭据/联网前拒绝运行。

在远端已部署仓库根目录，单案例命令如下；每次结束核对保存的终态、usage、模型与指纹后再执行下一项。只有 released 卡能进入作者循环。

```sh
/data/yuyangliu/envs/isaaclab6_validation/bin/python PLAN/feedback_sys/P3NEW/prepare_text_author_smoke.py --stage PLAN/feedback_sys/P3NEW/text_author_smoke_r3/stage.released.json --run-case stair-01 --output PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1
/data/yuyangliu/envs/isaaclab6_validation/bin/python PLAN/feedback_sys/P3NEW/prepare_text_author_smoke.py --stage PLAN/feedback_sys/P3NEW/text_author_smoke_r3/stage.released.json --run-case stair-03 --output PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1
/data/yuyangliu/envs/isaaclab6_validation/bin/python PLAN/feedback_sys/P3NEW/prepare_text_author_smoke.py --stage PLAN/feedback_sys/P3NEW/text_author_smoke_r3/stage.released.json --run-case push-02 --output PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1
/data/yuyangliu/envs/isaaclab6_validation/bin/python PLAN/feedback_sys/P3NEW/prepare_text_author_smoke.py --stage PLAN/feedback_sys/P3NEW/text_author_smoke_r3/stage.released.json --run-case push-04 --output PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1
```

未知 usage、实际输入超储备、费用违规或模型/指纹漂移必须保留回执并停止后续 episode。异常中断时先检查真实 journal，不换新输出路径重新开始。入口逐终态检查模型/指纹；每次真实请求的预算检查由生产 guard 执行。若实际输出超过请求上限，记录协议偏差并按真实值计费。任何失败也占用原案例的位置，不能换案例凑满成功数。

## 读法与句子引用核查

沿用现有 `TrainingComponents.rationale` 与源码/docstring 的事实陈述作为作者产物，不增设答案格式。五类检查仅是评阅清单，不把参考答案追加进作者输入：分母/结果、常见与少见过程及其区分力、同域成功反例、局部过程的身体/窗口/时间关系、未覆盖观察与采样缺口。每项记录使用、未使用、错误或材料不足，不能把“未使用”直接等同硬错误。

`review_contract.json` 冻结 14 类误读口径；每项必须有作者原句、实际最终可见 prompt 的 JSON 位置、对应报告句 ID/full 数值来源及理由。直接引用 ID、短原文引用、明确释义分别统计；匹配到 ID/数字不能当作理解正确。核查分母、单位、域、身体、区间、终帧与严格之前，并区分观察、假设、干预、待验证预测。无依据的事实、把 score 当成功率、把整段数值当尾窗、把子组当总域、源码行为伪称不变等，只在证据可直接确立时记硬错误；争议保持 pending。

以每个 episode 实际最终 journal 的 prompt 为准，不能默认作者最终看到的是准备首包。`P3/blind_export.py` 的 rationale/reference_locations/reward_checks 仅作已有提取和定位辅助；自然句 ID、释义及语义判定须留下人工/独立评阅记录。评阅用 `reviewer/<case>.json` 保存当前 full 的全部句子及来源；不得复用旧 Tc-vis 答案或旧评分给新输出贴标签。

本轮没有真实视觉内容、没有真实 S4 差分；无实际冲突句的案例也不能验证“冲突被误当多机制”读法。相应维度明确写 `NOT_EXERCISED`，不记零错误或通过。初始 `review_sheet_template.json` 均为 `NOT_RUN` / null。烟测产物不能替代 N3/N5 正式语义验收，旧开发检查也不算正式双评。
