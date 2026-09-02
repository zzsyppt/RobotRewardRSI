# Phase 8D：code-level verifier v1

日期：2026-06-25

## 设计判断

Phase 8D 的目标是给 code-level candidate 增加第二层 verifier。第一层主防线仍然是 Phase 8C 的 OS 隔离：

- 独立候选进程；
- network / PID namespace；
- Landlock 文件系统限制；
- 环境变量白名单；
- candidate output 与 runner artifacts 分离。

8D 不应把 8E 的训练闭环提前做进去。当前 candidate schema 还没有正式开放可执行 reward / obs / loss callable 的长训接入，因此此阶段不伪造“reward finite / reset-step / PPO-loop hook 已验证”。这些检查只有在 8E 真正把 candidate callable 接入训练循环后才有意义。

## 已实现检查

### 静态检查

候选包级：

- manifest 必须可解析；
- entrypoint 必须位于 `training_code/`；
- 禁止 symlink；
- 忽略 `__pycache__` / `.pyc` 缓存文件；
- 限制文件数量、单文件大小、总大小；
- 限制 top-level entries；
- 限制文件 suffix；
- manifest 不得声明 forbidden capabilities，例如：
  - runner hook；
  - PPO loop hook；
  - optimizer hook；
  - eval wrapper；
  - checkpoint selector；
  - filesystem / network / secrets。

Python AST 级：

- 禁止明显危险 import：
  - `subprocess`；
  - `socket`；
  - `requests` / `urllib` / `http`；
  - `os` / `pathlib` / `shutil` / `glob`；
  - `importlib` / `ctypes`；
  - `multiprocessing` / `threading`。
- 禁止危险 builtin call：
  - `eval`；
  - `exec`；
  - `compile`；
  - `open`；
  - `__import__`；
  - `input`；
  - `breakpoint`。
- 禁止 protected token：
  - `iid_hidden`；
  - `phase2_iid_hidden`；
  - `iid_hidden_eval`；
  - `hidden_audit`；
  - API key / secret 相关 token。
- 禁止 AST 数字常量中的 protected eval seed：`1001`、`2001`。

注意：数字 seed 检查是 AST / token-aware，不使用裸 substring，避免把普通浮点指标误判为 eval seed。

### 动态检查

如果服务器支持 Phase 8C hard isolation，则：

- 在 8C 隔离子进程中加载 candidate components；
- 父进程不 import 候选 entrypoint；
- 扫描 isolated child 的 public candidate/components summary；
- 如果 candidate 运行时拼接出 `iid_hidden` 等 forbidden public output，拒绝。

如果本地机器没有 Linux `unshare` + Landlock，则本地 selftest 只跑静态检查并标记 hard isolation unavailable；服务器强验收必须启用 hard isolation。

## 明确不加入的检查

本阶段暂不加入：

- Isaac/RSL-RL full train-start；
- reward callable finite check；
- observation wrapper finite check；
- optimizer / gradient / forward-backward runtime hook；
- eval observation/action wrapper 检查。

理由：这些检查必须在 candidate callable 真正接入训练执行路径后才有意义。现在硬加会制造“看起来验证了、实际上没验证”的伪安全。

## 实现文件

```text
configs/cold_start/phase8d_verifier.yaml
src/humanoid_agent/code_sandbox/verifier.py
scripts/phase8d_verify_candidate.py
scripts/phase8d_verifier_selftest.py
```

## 验收

本地：

```powershell
python scripts/phase8d_verifier_selftest.py
```

服务器强验收：

```bash
python scripts/phase8d_verifier_selftest.py
python scripts/phase8d_verify_candidate.py \
  --candidate-dir examples/cold_start_candidates/noop_candidate \
  --config configs/cold_start/phase8d_verifier.yaml \
  --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/phase8d_noop_verify \
  --require-server-enforcement
```

必须覆盖：

- no-op candidate 通过；
- forbidden import 被拒；
- hidden token literal 被拒；
- protected eval seed 常量被拒；
- forbidden manifest capability 被拒；
- 不允许的 top-level file 被拒；
- 运行时拼接出的 forbidden public output 被拒。

## 追加硬化：parent 静态扫描安全边界

日期：2026-06-25

### 核实结论

BUG-1 成立。旧实现虽然在 file inventory 阶段把 symlink 记为 `symlink_not_allowed`，但后续 Python AST 扫描仍会通过 `rglob("*.py")` 和普通文本读取触达候选 symlink 目标。这会让 parent 进程读取 candidate 目录外的内容；若目标是特殊文件，还可能造成 verifier 卡死或资源耗尽。

BUG-2 成立。旧实现主要覆盖 `.py` AST、manifest 和动态 public summary；`rationale.md`、`operation_diff.json`、`diagnostics.json`、`.txt` 等文本产物没有统一 forbidden-token 扫描。这些文件后续可能进入 ledger/prompt，因此不能只依赖 8C 的运行隔离。

次要项也成立且低成本：`getattr` / `setattr` / `globals` / `vars` / `dir` 这类动态访问入口应先列入静态 denylist；`base64` / `builtins` / `pickle` 应列入 forbidden import roots；非 UTF-8、特殊文件、异常解析不应让 verifier 崩溃，而应作为 candidate reject。

### 修改内容

- parent-side 内容读取统一走安全入口：
  - `candidate_manifest.yaml` 也先通过安全读取门槛，再交给 manifest loader 解析；
  - `lstat` 后拒绝 symlink；
  - 只读取 regular file；
  - POSIX 上使用 `O_NOFOLLOW`；
  - 读取前后受 `max_file_bytes` 限制；
  - 非 UTF-8、stat/read/parse 异常转为 verifier issue。
- 全量扫描 candidate 包内允许后缀文本：
  - `.py`；
  - `.yaml` / `.yml`；
  - `.json`；
  - `.md`；
  - `.txt`。
- `rationale.md`、`operation_diff.json`、`diagnostics.json` 中出现 protected token 或 protected eval seed 会被拒绝。
- 扩展静态 denylist：
  - forbidden call：`getattr`、`setattr`、`globals`、`locals`、`vars`、`dir`；
  - forbidden import root：`base64`、`builtins`、`pickle`。

### 新增验收样例

- symlink 到 candidate 目录外的 `.py`：必须因为 `symlink_not_allowed` 被拒，且不能出现“读取目标后发现 forbidden import/literal”的副作用；
- `rationale.md` 泄露 `iid_hidden` / `2001`：必须拒绝；
- `operation_diff.json` 泄露 `hidden_audit` / `2001`：必须拒绝；
- Python 中调用 `getattr`：必须拒绝。

### 边界说明

这次修改没有把 8E 的训练执行验证提前放进 8D。它只收紧 parent 静态 verifier 的输入安全与 prompt-leak 防线。真正的 reward/obs/loss callable 长训隔离、runtime budget hook、梯度/optimizer 计数仍属于 8E 接入真实训练路径后的验收项。
