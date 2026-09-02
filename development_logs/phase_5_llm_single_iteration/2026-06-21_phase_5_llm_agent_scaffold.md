# Phase 5 LLM-agent single-iteration scaffold log

Date: 2026-06-21

## Scope

Phase 5 requires one LLM-agent proposal loop:

```text
build prompt -> LLM outputs analysis + config patch -> validate -> verifier -> smoke/train/eval
```

This implementation completes the safe proposal-generation and verifier part of
that loop. It does not store API keys, does not expose iid-hidden results to the
Agent, and does not let the LLM write files directly.

Live DeepSeek execution is intentionally gated behind a user-provided environment
variable and an explicit `--allow-api-spend` flag.

## Design plan

1. Use a provider-agnostic interface:
   - `dry_run`: deterministic local response for CI and parser/verifier tests;
   - `replay`: replay a saved JSON response;
   - any configured `openai_compatible` provider profile;
   - `deepseek` is only one profile, not hard-coded into the runner.
2. Make the LLM output structured JSON operations, not raw file edits.
3. Apply those operations locally under a strict allowlist:
   - `configs/phase1_push_recovery.yaml`: training-only `phase2_train.push`;
   - `configs/phase2_baseline.yaml`: baseline metadata, reward weights,
     reward params, and limited PPO scalar fields.
4. Reject illegal edits before verifier when possible.
5. Always run Phase 3 static verifier after applying a candidate.
6. Write reproducibility artifacts without secrets:
   - `prompt.txt`;
   - `prompt_messages.json`;
   - `provider_request_summary.json`;
   - `raw_response.json`;
   - `parsed_response.json`;
   - `analysis.json`;
   - candidate config pair;
   - `candidate.diff`;
   - `verifier_summary.json`;
   - `run_summary.json`.

## Implemented files

- `configs/phase5_llm_agent.yaml`
  - provider profiles;
  - DeepSeek V4 Pro defaults;
  - custom OpenAI-compatible profile;
  - prompt settings;
  - patch contract and bounds.
- `src/humanoid_agent/phase5_llm_agent/prompting.py`
  - builds a constrained prompt from public summary and current config context.
- `src/humanoid_agent/phase5_llm_agent/providers.py`
  - `dry_run`, `replay`, and generic `openai_compatible` providers;
  - API keys are read only from environment variables.
- `src/humanoid_agent/phase5_llm_agent/patching.py`
  - validates and applies JSON operations;
  - rejects protected paths such as eval seeds before verifier;
  - enforces finite values, reward bounds, and push range bounds.
- `src/humanoid_agent/phase5_llm_agent/runner.py`
  - end-to-end proposal generation;
  - candidate file writing;
  - Phase 3 verifier integration.
- `scripts/phase5_llm_single_iteration.py`
  - CLI entry point.
- `scripts/phase5_llm_agent_selftest.py`
  - no-network regression test.

## API extensibility

`deepseek` is configured as:

```yaml
provider:
  deepseek:
    type: openai_compatible
    api_key_env: DEEPSEEK_API_KEY
    base_url: https://api.deepseek.com
    endpoint: /chat/completions
    model: deepseek-v4-pro
```

Future OpenAI-compatible providers can use the built-in
`custom_openai_compatible` profile:

```bash
export PHASE5_LLM_API_KEY=...
export PHASE5_LLM_BASE_URL=https://example.com
export PHASE5_LLM_MODEL=some-model
python3 scripts/phase5_llm_single_iteration.py \
  --provider custom_openai_compatible \
  --allow-api-spend \
  --summary-file ...
```

For providers with a non-OpenAI protocol, add a new provider type in
`providers.py`; the prompt, patch contract, verifier, output layout, and
training handoff remain unchanged.

## Validation

Local validation:

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  -m py_compile `
  scripts\phase5_llm_single_iteration.py `
  scripts\phase5_llm_agent_selftest.py `
  src\humanoid_agent\phase5_llm_agent\__init__.py `
  src\humanoid_agent\phase5_llm_agent\patching.py `
  src\humanoid_agent\phase5_llm_agent\prompting.py `
  src\humanoid_agent\phase5_llm_agent\providers.py `
  src\humanoid_agent\phase5_llm_agent\runner.py

C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  scripts\phase5_llm_agent_selftest.py
```

Result: pass.

Server validation without API key:

```bash
cd /data/yuyangliu/AGENT
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_llm_agent_selftest.py
```

Result: pass.

Dry-run on the real Phase 2 five-seed summary:

```bash
cd /data/yuyangliu/AGENT
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_llm_single_iteration.py \
  --provider dry_run \
  --summary-file /data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json \
  --output-dir /data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_dry_run_001
```

Result:

```json
{
  "status": "pass",
  "provider": "dry_run",
  "verifier_status": "pass",
  "output_dir": "/data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_dry_run_001"
}
```

DeepSeek missing-key safety probe:

```bash
unset DEEPSEEK_API_KEY
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_llm_single_iteration.py \
  --provider deepseek \
  --allow-api-spend \
  --summary-file /data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json \
  --output-dir /data/yuyangliu/runs/phase_5_llm_single_iteration/2026-06-21_deepseek_missing_key_probe
```

Result:

```json
{
  "status": "fail",
  "errors": [
    {
      "type": "ProviderError",
      "message": "Missing API key environment variable for deepseek: DEEPSEEK_API_KEY"
    }
  ],
  "verifier_status": "skipped"
}
```

No key value is printed or written.

## Live DeepSeek command

When the user is ready, set the key only in the server shell:

```bash
export DEEPSEEK_API_KEY='...'
```

Then run:

```bash
cd /data/yuyangliu/AGENT
PYTHONPATH=/data/yuyangliu/AGENT/src python3 scripts/phase5_llm_single_iteration.py \
  --provider deepseek \
  --allow-api-spend \
  --summary-file /data/yuyangliu/runs/phase_2_baseline_variance_5seed/2026-06-20_5seed_formal/summary.json \
  --output-dir /data/yuyangliu/runs/phase_5_llm_single_iteration/$(date +%Y-%m-%d_deepseek_v4_pro_001)
```

## Decision

Phase 5 proposal scaffold is ready.

Go criteria met:

- prompt builder works;
- provider interface is extensible;
- DeepSeek profile is configured but requires explicit spend authorization;
- API key values are never written by the code;
- LLM output is constrained to JSON operations;
- protected eval/budget edits are rejected before verifier;
- generated dry-run candidate passes Phase 3 verifier.

Remaining work before a formal Phase 5 result:

1. user sets `DEEPSEEK_API_KEY` in the server shell;
2. run one live `deepseek-v4-pro` proposal;
3. run train-start smoke on the resulting candidate;
4. run the agreed fixed-budget public evaluation;
5. decide accept/reject without exposing iid-hidden to the next proposal.
