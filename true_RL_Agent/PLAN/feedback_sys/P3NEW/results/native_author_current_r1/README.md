# N4 最后两槽：仅准备，NOT_RELEASED

本地 `prepare_current.py` 已完成当前报告构建与8帧序列化，`sender.validate` 通过。`preparation_receipt.json` 保留新增过程文字、full/public SHA、实际请求字节 SHA 与四帧投递证据。这里没有实际作者调用，不能关闭“最后一次实际请求”验收。

Stair 使用 r10 自己的 fc305… 原始 manifest 与30个 NPZ，当前生产重算 progress 后附原观察和原核对。Push 使用 r12 同源 r2 v3 reviewed projection，当前生产投影相等，明确不是 L0 重算；SOL 原答没有伪造 provider HTTP 回执。两任务原 raw_answer 与 audit.raw_output 精确相等。

原共享 journal 只读复核为6次、$0.200609736，55个历史JSON SHA未变，gen004不存在。两次各预留40k输入和65536输出，按峰值$0.3/$1.2每百万计共$0.1812864，连历史最多$0.381896136，低于原$4；总episode最多8，仍每槽1call、0followup、无retry。价格来源：https://api-docs.deepseek.com/quick_start/pricing/ （2026-09-30核）。

`launch_card.draft.json` 给出完整输入SHA、现有helper的必要SHA、同原材料仓库和frozen r3源的环境、prepare/check-only/run argv。`remote_prepare_inputs.tar.gz` 只供root审阅后部署：不带本地生成的contexts/prompts/full，不移账本；新的driver保存在新结果目录，不覆盖旧driver。远端重新prepare会生成实际主机路径绑定的字节，先各执行一次check-only，再由root冻结released卡并唯一发送；本地卡不能直接当远端执行卡。若既有PLAN helpers的SHA不匹配，停止并报告具体差异，不覆盖旧冻结文件。

本轮未执行远端写入、API、GPU或训练。正式N5采集不在此范围。
