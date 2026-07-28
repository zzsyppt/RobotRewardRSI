# Authoring Memory and Audit Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve bounded, actionable source-request feedback across rounds, decouple internal round state from provider compatibility payloads, and move the push-recovery dry-run scenario into task ownership.

**Architecture:** `PublicContextBuilder` produces one frozen two-value assembly from the same sanitized ledger records: durable `round_public_context` for orchestration and a canonical `provider_payload` for providers/artifacts. Source resolution produces a separate allowlisted round summary that is persisted on every ledger outcome and projected through the existing bounded public-history channel. The legacy alias becomes a fixed scalar summary, while the synthetic push audit moves intact below the push-recovery task package.

**Tech Stack:** Python 3.11+, frozen dataclasses, stdlib `unittest`, AST-based package-boundary checks, existing canonical JSON/public-safety helpers.

## Global Constraints

- Preserve current training budgets, eval seeds/protocol, checkpoint selection, PPO, acceptance thresholds, candidate permissions, and task ABI.
- Keep task-specific semantics under `true_rl_agent.tasks.<task_id>`; shared layers may carry only generic opaque task summaries/metadata.
- Keep public history at most 8 recent rounds and at most 8192 serialized bytes.
- Keep each `source_request_feedback` at or below 2048 serialized bytes.
- Retain at most `MAX_SOURCE_REQUESTS_PER_ROUND * MAX_SOURCE_IDS_PER_REQUEST` combined source IDs; current shared values are `2 * 2 = 4`.
- Keep the deprecated `public_context` alias below 1024 serialized bytes and exclude recent outcomes, history limits/omissions, task summaries, source data, and full repair feedback.
- Keep the complete provider attempt records only in existing run-summary/failure-audit locations; never add them to successful ledger history.
- Preserve all unrelated dirty-worktree changes. Because target files overlap those changes, do not stage or commit implementation files automatically; use focused tests, `git diff --check`, and explicit diff review as checkpoints.
- Do not migrate the broader eval score ABI, redesign `RewardInputs`, remove `candidates.prompt_projection`, or add a generic audit plugin abstraction.

## File Responsibility Map

- `src/true_rl_agent/memory/context_builder.py`: build durable round context and canonical provider payload together.
- `src/true_rl_agent/prompting/payload.py`: canonical section assembly and the bounded legacy alias only.
- `src/true_rl_agent/agent_core/authoring_loop.py`: consume the two-value build and aggregate source resolutions for one authoring round.
- `src/true_rl_agent/agent_core/source_request_service.py`: resolve requests and emit minimal public attempt results with real budget values.
- `src/true_rl_agent/agent_core/source_request_feedback.py`: own the generic round-feedback schema, aggregation, projection, and byte/id bounds.
- `src/true_rl_agent/agent_core/source_request_limits.py`: single-source request and feedback limits/reason codes.
- `src/true_rl_agent/prompting/source_catalog.py`: expose read-only membership in the closed logical-ID catalog for defensive feedback projection.
- `src/true_rl_agent/agent_core/round.py`: persist optional generic source feedback on normal runner paths.
- `src/true_rl_agent/agent_core/live_loop.py`: forward the feedback on normal, provider-failure, and static-failure paths.
- `src/true_rl_agent/agent_core/context.py`: fixed allowlist projection into bounded history and idempotent omission accounting.
- `src/true_rl_agent/agent_core/loop.py`: replay path consumes explicit round context.
- `src/true_rl_agent/tasks/push_recovery/loop_audit.py`: owns the push-specific synthetic scenario.
- `scripts/run_loop.py`, `scripts/run_round.py`: explicit push-audit composition roots.
- `tests/unit/test_project_boundaries.py`: prevents concrete task imports/literals from returning to generic packages.

---

### Task 1: Separate Durable Round Context from Provider Payload

**Files:**
- Modify: `src/true_rl_agent/memory/context_builder.py`
- Modify: `src/true_rl_agent/memory/__init__.py`
- Modify: `src/true_rl_agent/prompting/payload.py`
- Modify: `src/true_rl_agent/agent_core/authoring_loop.py`
- Modify: `src/true_rl_agent/agent_core/live_loop.py`
- Modify: `src/true_rl_agent/agent_core/loop.py`
- Modify: `src/true_rl_agent/tasks/prompt_adapter.py`
- Test: `tests/unit/test_memory_context_phase13.py`
- Test: `tests/unit/test_prompt_projection_and_mechanism.py`
- Test: `tests/unit/test_candidate_authoring_loop.py`
- Test: `tests/unit/test_task_prompt_adapter.py`
- Test: `tests/integration/test_agent_core_loop.py`

**Interfaces:**
- Produces: frozen `PublicPromptBuild(round_public_context: Mapping[str, Any], provider_payload: Mapping[str, Any])`.
- Produces: `PublicContextBuilder.build_prompt(...) -> PublicPromptBuild`.
- Preserves: `PublicContextBuilder.build_prompt_payload(...) -> dict[str, Any]` as a provider-only convenience wrapper.
- Preserves: `build_authoring_prompt_payload(...) -> dict[str, Any]` for existing external/tests; live authoring uses the two-value API internally.
- Renames: `CandidateAuthoringResult.public_context` to the unambiguous `round_public_context`; no internal compatibility property is retained.

- [ ] **Step 1: Write failing API and separation tests**

Add tests equivalent to:

```python
build = PublicContextBuilder(task_memory_adapter=CustomTaskMemoryAdapter()).build_prompt([record])

self.assertEqual(build.round_public_context["previous_rounds"], 1)
self.assertIn("recent_outcomes", build.round_public_context)
self.assertEqual(
    build.provider_payload["public_feedback_summary"]["task_public_summary"]["custom_metric"],
    7,
)
self.assertNotIn("recent_outcomes", build.provider_payload["public_context"])
self.assertNotIn("task_public_summary", build.provider_payload["public_context"])
self.assertNotIn("source_catalog", build.round_public_context)
```

Add a call-site regression that patches/removes the legacy alias after provider assembly and proves `CandidateAuthoringResult.round_public_context` and replay ledger `public_context` still contain the durable round state.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m unittest tests.unit.test_memory_context_phase13 tests.unit.test_prompt_projection_and_mechanism tests.unit.test_candidate_authoring_loop tests.unit.test_task_prompt_adapter tests.integration.test_agent_core_loop -v
```

Expected: failures because `PublicPromptBuild`/`build_prompt` do not exist and internal call sites still read `payload["public_context"]`.

- [ ] **Step 3: Implement the two-value build at the single construction boundary**

Add the frozen result and make the builder sanitize records once:

```python
@dataclass(frozen=True)
class PublicPromptBuild:
    round_public_context: Mapping[str, Any]
    provider_payload: Mapping[str, Any]
```

`build()` remains the durable-context convenience method. `build_prompt()` must:

1. Convert input mappings to `ExperimentRecord` once.
2. Apply task-owned record scrub policy once.
3. Build the durable context with `build_public_context` and optional sanitized `TaskMemoryAdapter` summary.
4. Build task prompt sections from the sanitized records plus durable context without inserting them into durable context.
5. Build source catalog/lineage/payload sections only in a provider-context copy.
6. Add only explicit within-attempt `candidate_repair_feedback` and `source_payloads` to that provider-context copy.
7. Assemble and public-safety-check the provider payload.
8. Return both sanitized values without recovering either from the legacy alias.

Use the exact signature `build_prompt(self, records: Sequence[ExperimentRecord | Mapping[str, Any]], *, latest_failed_candidate: Any = None, candidate_repair_feedback: Mapping[str, Any] | None = None, source_payloads: Sequence[Mapping[str, Any]] | None = None, frame_kind: str | None = None, runner_capability_readiness: Mapping[str, Mapping[str, Any]] | None = None) -> PublicPromptBuild`.

Export `PublicPromptBuild` from `true_rl_agent.memory`.

- [ ] **Step 4: Replace internal alias reads**

- In `CandidateAuthoringLoop`, call a new `build_authoring_prompt(...) -> PublicPromptBuild`, send/save `provider_payload`, and set `selected_round_public_context = prompt_build.round_public_context`.
- Rename `CandidateAuthoringResult.public_context` and all three live-loop consumers to `round_public_context`.
- In replay `AgentLoop`, send `provider_payload` and pass `round_public_context` to `RoundInputs`.
- Keep `build_authoring_prompt_payload` as a thin compatibility wrapper returning `dict(build.provider_payload)`.
- Confirm no orchestration path indexes a provider payload to reconstruct internal context.

- [ ] **Step 5: Replace the legacy near-copy with a fixed summary**

Implement this exact alias base shape in `payload.py`:

```python
{
    "schema_version": "true_rl_agent.public_context_legacy_summary.v1",
    "deprecated": True,
    "public_only": True,
    "previous_rounds": public_context.get("previous_rounds", 0),
    "best_public_score": public_context.get("best_public_score"),
    "last_outcome": public_context.get("last_outcome"),
    "pure_knob_streak": public_context.get("pure_knob_streak", 0),
    "canonical_sections": {
        "history": "public_history",
        "public_feedback": "public_feedback_summary",
        "runtime_feedback": "runtime_feedback_summary",
    },
}
```

When repair feedback exists, add only `candidate_repair_feedback_ref`. Measure with `canonical_prompt_json_bytes`; raise if the result is not below `MAX_LEGACY_PUBLIC_CONTEXT_ALIAS_BYTES = 1024`.

Extend `_generic_public_feedback_summary` so a mapping-valued durable `task_public_summary` appears only at `public_feedback_summary.task_public_summary` when no task prompt adapter supplies a custom section.

Also update `NeutralTaskPromptAdapter`: its neutral `public_feedback_summary` transparently carries the sanitized opaque `public_context.task_public_summary` when present. A custom prompt adapter remains authoritative for its own feedback section and is not silently merged by shared code.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all listed modules pass, and alias-specific expectations assert the new fixed shape and byte limit.

- [ ] **Step 7: Checkpoint the task without staging user changes**

Run:

```powershell
git diff --check
rg -n 'prompt_payload\["public_context"\]|provider_payload\["public_context"\]' src/true_rl_agent/agent_core src/true_rl_agent/tasks
```

Expected: diff check exits 0; no internal orchestration read remains.

---

### Task 2: Aggregate Bounded Source-Request Round Feedback

**Files:**
- Create: `src/true_rl_agent/agent_core/source_request_feedback.py`
- Modify: `src/true_rl_agent/agent_core/source_request_limits.py`
- Modify: `src/true_rl_agent/agent_core/source_request_service.py`
- Modify: `src/true_rl_agent/agent_core/authoring_loop.py`
- Modify: `src/true_rl_agent/prompting/source_catalog.py`
- Create: `tests/unit/test_source_request_feedback.py`
- Test: `tests/unit/test_source_request_service.py`
- Test: `tests/unit/test_candidate_authoring_loop.py`

**Interfaces:**
- Produces: `build_source_request_round_feedback(source_request_results: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None`.
- Produces: `project_source_request_round_feedback(value: Any) -> dict[str, Any] | None` for idempotent history defense.
- Produces: `serialized_source_request_feedback_bytes(value: Mapping[str, Any]) -> int` using canonical UTF-8 serialization.
- Produces: `CandidateAuthoringResult.source_request_feedback: Mapping[str, Any] | None`.
- Produces: `is_known_prompt_source_logical_id(logical_id: Any) -> bool` from the closed source catalog.
- Constants: `SOURCE_REQUEST_ROUND_FEEDBACK_SCHEMA_VERSION`, `MAX_SOURCE_REQUEST_FEEDBACK_BYTES = 2048`, `MAX_SOURCE_REQUEST_FEEDBACK_ENTRIES = MAX_SOURCE_REQUESTS_PER_ROUND * MAX_SOURCE_IDS_PER_REQUEST`, and the closed `SOURCE_REQUEST_DENIAL_REASONS`.

- [ ] **Step 1: Write failing aggregation tests**

Cover returned-only, denied-only, mixed partial, repeated IDs, unavailable source, turn limit, ID limit, and byte budget. Assert the full fixed shape, including real cumulative usage and custom `max_source_requests_per_round`.

Add a malicious-input regression proving the round summary contains none of these producer feedback fields: `source_request`, `rationale`, `instructions`, `content`, `path`, raw provider text, or unknown keys.

Add overflow assertions:

```python
self.assertLessEqual(canonical_prompt_json_bytes(feedback), 2048)
self.assertLessEqual(
    len(feedback["returned_logical_ids"]) + len(feedback["denied_logical_ids"]),
    MAX_SOURCE_REQUESTS_PER_ROUND * MAX_SOURCE_IDS_PER_REQUEST,
)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest tests.unit.test_source_request_feedback tests.unit.test_source_request_service tests.unit.test_candidate_authoring_loop -v
```

Expected: missing aggregation API/field failures.

- [ ] **Step 3: Implement the source-feedback contract module**

Extend `SourceRequestResolution.attempt_record()` with only selected budget integers. The round builder consumes those service-produced attempt results - not provider failure details - and computes status from all outcomes before truncation. Deduplicate returned IDs in encounter order; for repeated denials, retain the latest reason for each ID. Apply the shared combined-ID ceiling by retaining the most recent denials first and then the most recent returned IDs while preserving order inside each retained subset.

Emit only:

```python
{
    "schema_version": "true_rl_agent.source_request_round_feedback.v1",
    "status": status,
    "request_count": len(source_request_results),
    "returned_logical_ids": retained_returned,
    "denied_logical_ids": retained_denied,
    "budget": {
        "source_request_count": last_budget_count,
        "max_source_requests_per_round": last_budget_limit,
        "max_source_ids_per_request": last_id_limit,
        "cumulative_returned_bytes": last_cumulative_bytes,
        "max_source_request_bytes_per_round": last_byte_limit,
    },
    "omitted_returned_logical_id_count": omitted_returned,
    "omitted_denied_logical_id_count": omitted_denied,
}
```

For the byte ceiling, repeatedly drop the oldest returned ID first, then oldest denial, incrementing the matching omission count. Never drop schema/status/count/budget. Public-safety-check the result and assert the final canonical byte length is at most 2048. The projector accepts only closed-catalog logical IDs and service denial reason codes, preserves prior omission counts, counts newly invalid ID entries, and produces the same value when called repeatedly.

- [ ] **Step 4: Attach the summary to every authoring result**

At the single `_result` convergence point, extract `source_request_result` mappings from the existing attempt records and call the round builder. This covers every return branch without a parallel event stream; the result remains `None` when no request occurred. Do not derive it from repair feedback or provider failure details.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Step 2 command. Expected: all source-service and authoring-loop tests pass.

- [ ] **Step 6: Checkpoint**

Run `git diff --check` and inspect the diff for any source content/rationale/instruction copied into the new summary.

---

### Task 3: Persist and Project Source Feedback Across Every Ledger Path

**Files:**
- Modify: `src/true_rl_agent/agent_core/round.py`
- Modify: `src/true_rl_agent/agent_core/live_loop.py`
- Modify: `src/true_rl_agent/agent_core/context.py`
- Test: `tests/integration/test_live_loop.py`
- Test: `tests/unit/test_memory_context_phase13.py`

**Interfaces:**
- Consumes: `CandidateAuthoringResult.source_request_feedback` from Task 2.
- Produces: optional generic root ledger field `source_request_feedback` on normal, provider-failure, and static-authoring-failure records.
- Produces: fixed `recent_outcomes[*].source_request_feedback` history projection.

- [ ] **Step 1: Write failing end-to-end persistence tests**

Add one live-loop test where an unavailable source is denied and a candidate succeeds in the same round; assert the second-round provider payload contains prior logical ID, `source_not_requestable`, and real request/ID/byte limits.

Add a final-attempt source-request failure test and assert the next-round payload exposes the identical bounded schema rather than generic error details.

Assert successful ledger records do not gain `provider_attempt_records`, while existing failure records retain them.

- [ ] **Step 2: Write failing history-safety/idempotence tests**

Build a ledger record containing valid source feedback plus tainted unknown fields and verify only the fixed shape reaches history. Verify both limits:

```python
self.assertLessEqual(len(json.dumps(entry["source_request_feedback"], sort_keys=True).encode("utf-8")), 2048)
self.assertLessEqual(len(json.dumps(history["recent_outcomes"], sort_keys=True).encode("utf-8")), 8192)
```

For generic details, compact a mapping twice and assert the second result equals the first and preserves the original `omitted_detail_count` plus only newly omitted fields.

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
python -m unittest tests.integration.test_live_loop tests.unit.test_memory_context_phase13 -v
```

Expected: feedback is absent from the success ledger/history and omission count is not idempotent.

- [ ] **Step 4: Wire persistence at the ledger source**

- Add optional `source_request_feedback` to `RoundInputs`.
- Store its sanitized mapping as a root ledger field in `run_candidate_round`.
- Pass it from `run_live_loop` to `RoundInputs`.
- Add it to `_append_provider_failure_record` and `_append_static_authoring_failure_record` arguments/records.
- Keep `None` when there was no source request.

- [ ] **Step 5: Add the dedicated history projector**

Add `source_request_feedback` to `_PUBLIC_HISTORY_ENTRY_KEYS` and route it to `project_source_request_round_feedback`, which explicitly selects schema/status/count, logical IDs/reason codes, five budget fields, and two omission counts. Reject malformed types and count omitted invalid entries. Do not reuse `_compact_detail_mapping`.

Make omission accounting idempotent by treating an existing nonnegative `omitted_detail_count` as prior state, excluding that key from newly omitted fields, and returning `prior + newly_omitted` only when nonzero.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Step 3 command. Expected: both denial paths are actionable next round, all safety assertions pass, and compaction is stable.

- [ ] **Step 7: Checkpoint**

Run `git diff --check` and confirm the normal-success ledger contains the bounded summary but not full attempt records.

---

### Task 4: Move the Push Dry-Run Scenario into Task Ownership

**Files:**
- Create: `src/true_rl_agent/tasks/push_recovery/loop_audit.py`
- Delete: `src/true_rl_agent/audit/loop_audit.py`
- Delete: `src/true_rl_agent/audit/__init__.py`
- Modify: `scripts/run_loop.py`
- Modify: `scripts/run_round.py`
- Modify: `tests/integration/test_dry_run_audit_loop.py`
- Create: `tests/unit/test_project_boundaries.py`

**Interfaces:**
- Produces: `run_push_recovery_loop_audit(...)`.
- Produces: artifact schema `true_rl_agent.push_recovery.loop_audit.v1`.
- Keeps shared: `DryRunAuditRunner`, `build_eval_run_audit`, opaque `task_eval_metadata`, and generic runner/eval/ledger contracts.

- [ ] **Step 1: Write failing ownership and boundary tests**

Update integration imports to the task-owned entrypoint and assert the new schema.

Create an AST boundary test that scans `agent_core`, `prompting`, `memory`, `evaluation`, `runner`, and any existing generic `audit` directory. Derive concrete task packages dynamically from child directories below `src/true_rl_agent/tasks` that contain `__init__.py`. Reject both direct imports below `true_rl_agent.tasks.<concrete_task>` and `from true_rl_agent.tasks import <concrete_task>`, while allowing shared task contracts such as `tasks.base`, `tasks.prompt_adapter`, and `tasks.component_smoke`.

Also scan those generic packages for the scenario literals `push_recovery`, `push_drift`, `recovery_logic`, `mean_base_height`, and `mean_tilt_rad`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest tests.integration.test_dry_run_audit_loop tests.unit.test_project_boundaries -v
```

Expected: task-owned import is missing and the current generic audit violates the boundary.

- [ ] **Step 3: Relocate without inventing a generic abstraction**

Move the complete synthetic scenario to `tasks/push_recovery/loop_audit.py`, rename the entrypoint, update its schema, and use `PublicContextBuilder.build_prompt(...)` so it passes `round_public_context` to `RoundInputs` and provider payload to the replay provider/artifact writer.

Delete the generic audit module/re-export. Update only the two scripts and integration test to import the task-owned entrypoint explicitly.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: 15-round audit and CLI tests pass, new schema is present, and the boundary test reports no concrete task ownership leak.

- [ ] **Step 5: Checkpoint**

Run:

```powershell
git diff --check
rg -n -i "push_recovery|push_drift|recovery_logic|mean_base_height|mean_tilt_rad" src/true_rl_agent/agent_core src/true_rl_agent/prompting src/true_rl_agent/memory src/true_rl_agent/evaluation src/true_rl_agent/runner src/true_rl_agent/audit
```

Expected: no live generic-package hit; an absent `src/true_rl_agent/audit` path is acceptable.

---

### Task 5: Whole-System Verification and Payload/Memory Audit

**Files:**
- Verify only; modify production/tests only if a newly reproduced defect first receives a failing regression test.

**Interfaces:**
- Validates all acceptance criteria from the approved design.

- [ ] **Step 1: Run the focused regression suite**

```powershell
python -m unittest tests.unit.test_source_request_feedback tests.unit.test_source_request_service tests.unit.test_candidate_authoring_loop tests.unit.test_memory_context_phase13 tests.unit.test_prompt_projection_and_mechanism tests.unit.test_task_prompt_adapter tests.unit.test_project_boundaries tests.integration.test_agent_core_loop tests.integration.test_live_loop tests.integration.test_dry_run_audit_loop -v
```

Expected: zero failures/errors.

- [ ] **Step 2: Run the full suite and compilation**

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

Expected: all tests pass, compilation exits 0, diff check emits no output.

- [ ] **Step 3: Run a fresh 30-round task-owned dry audit**

Use a new temporary/output directory and `run_push_recovery_loop_audit(rounds=30)`. Parse every saved provider payload with the existing canonical serialization helper and report:

- maximum total payload bytes, strictly below 262144;
- maximum single canonical section bytes, at most 98304;
- maximum serialized `public_history.recent_outcomes`, at most 8192;
- maximum serialized source feedback, at most 2048;
- maximum serialized legacy alias, strictly below 1024;
- zero payloads where `public_context.recent_outcomes` exists or equals canonical history;
- all task-owned audit checks true.

- [ ] **Step 4: Re-audit shared ownership and payload/state coupling**

Run AST/text scans over live source (not historical plans) and verify:

- no generic package directly imports a concrete task;
- no push-specific audit literal remains in generic packages;
- no internal orchestration indexes provider `public_context`;
- task-memory-only summaries have a canonical provider section;
- successful ledgers do not store complete attempt records;
- source contents/rationales/instructions/paths do not occur in cross-round feedback.

- [ ] **Step 5: Review the final diff against the approved spec**

Compare every changed hunk to `docs/superpowers/specs/2026-07-10-authoring-memory-and-audit-boundaries-design.md`. Confirm unrelated dirty edits remain present and were neither reset nor silently folded into an autonomous implementation commit.
