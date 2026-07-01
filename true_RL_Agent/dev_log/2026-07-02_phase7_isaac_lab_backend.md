# Phase 7 IsaacLabRslRlBackendAdapter dry-run

## 目标

按照计划阶段 7，实现第一版真实后端目标的 dry-run materialization：把 task、candidate、policy initialization 映射为 Isaac Lab + RSL-RL train/eval job specs。当前阶段只生成 CLI job spec 字符串，不启动真实训练。

## 本次实现

- 新增 `backends/isaac_lab_rsl_rl.py`。
- 定义 `IsaacLabRslRlBackendConfig`，包含 python、train/eval entrypoint、默认预算、public/trusted seed 数、robot、obs/action schema、policy architecture。
- 定义 `IsaacLabRslRlBackendAdapter`，声明 backend id、支持 task、支持 random/warm_start。
- materialize random train job spec，metadata 中包含 dry-run train command。
- materialize warm-start train job spec，metadata 中显式包含 `--init-checkpoint`。
- materialize checkpoint eval job spec，metadata 中包含 checkpoint policy 和 dry-run eval command。
- 定义 `BackendCompatibilityReport` 和 `check_compatibility()`，对 unsupported task / initialization 返回结构化 failure。
- 声明 Isaac Lab、RSL-RL、Unitree assets、CUDA runtime 等外部依赖。
- 声明 future curriculum capabilities，但 `effective_now=false`，避免第一阶段把 curriculum tuning 接入训练。
- 新增 `tests/unit/test_isaac_lab_backend.py`。

## 验证

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.unit.test_isaac_lab_backend -v
```

结果：4 个测试通过。

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v
```

结果：40 个测试通过。

```powershell
rg -n "humanoid_agent|D:\\humanoid_agent|/data/yuyangliu/AGENT|\\.\\.[/\\]src[/\\]humanoid_agent" true_RL_Agent\src true_RL_Agent\scripts true_RL_Agent\configs
```

结果：无匹配。

```powershell
rg -n "subprocess\\.(run|Popen|call)|os\\.system|Start-Process|torchrun" true_RL_Agent\src true_RL_Agent\tests
```

结果：无匹配。

## Review 结论

- BackendAdapter 不 import 父项目脚本。
- 当前只生成 CLI job spec 字符串，不执行训练命令。
- random 与 warm_start 的训练起点参数不同。
- eval job 明确 checkpoint policy：需要 runner sealed train output。
- 后端能力不足会返回结构化 compatibility failure。
- future curriculum capability 只声明支持潜力，不作为当前 effective capability。

## 下一步

进入阶段 8：Candidate 与 Prompt Projection 基础能力。阶段 8 应定义 candidate package schema、loader/verifier、controlled code hooks、prompt-facing projection、source-pack policy 和 public-only prompt payload check。

