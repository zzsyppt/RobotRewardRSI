我们当前对动作进行比较精细的结构化之后，需要这些字段：
verb 动作的动词
directObject  直接交互的物体
directObjectLocation 交互物体的位置
indirectObject 间接交互的物体
hand 交互的手部
description 动作的自然语言描述
startTimestamp / endTimestamp 起止时间戳

后续还会加每个动作的关键帧。所以在我们这个设定下任务还是比较复杂的，得识别物体 手部 时序的动作，动作的起止时间，还有视频里一般都是左右手是并行和物体交互的。直接把视频给大模型 目前只有 Qwen3.8-Max 和 Gemini 3 + 系列测试一下还算不错，我觉得是可用的程度
一些比较小的物体和比较小的动作上识别效果还是有问题，尤其是在长上下文环境下，qwen3.8-max 还是有能力衰退的，就比如有些 case 让 qwen3.8 单独识别没问题，但是放在 agent 里 100k 上下文就失败了
他里面那个 GT 参考一下就行，因为GT 给的粒度太粗了