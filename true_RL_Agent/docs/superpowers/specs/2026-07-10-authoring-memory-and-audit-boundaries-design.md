# Authoring Memory and Audit Ownership Design

## Status

Approved in conversation on 2026-07-10. This document defines the implementation boundary; it does not authorize unrelated eval, runner, task ABI, or training changes.

## Goal

Make cross-round authoring memory actionable and bounded while separating internal round state from provider compatibility payloads and moving the push-recovery dry-run scenario into task ownership.

The resulting system must preserve enough public information to avoid repeated invalid source requests, remain within existing history and prompt budgets, and keep all push-recovery semantics out of shared packages.

## Project Boundaries

- The LLM proposes candidate packages and may request only prompt-catalog logical source IDs.
- AgentCore owns bounded authoring attempts, public history projection, round sequencing, and ledger handoff.
- The runner owns training/eval execution, budgets, checkpoints, seeds, decisions, and ledger append.
- Task-specific prompt, memory, eval, and synthetic audit semantics belong under `true_rl_agent.tasks.<task_id>`.
- Provider-facing payloads contain public data only and must not become an internal state transport.
- Full source contents, provider rationales, instructions, filesystem paths, private artifacts, raw provider output, and complete attempt records must not enter cross-round prompt history.
- Existing unrelated working-tree changes must be preserved; this work must not reset, rewrite, or absorb them accidentally.

## Confirmed Problems

### Source-request feedback is lost before history compression

`CandidateAuthoringLoop` retains public-safe attempt records, but the normal candidate-success ledger path does not receive them. A source request denied before a same-round successful draft therefore leaves no denial evidence in the next round. A final-attempt denial does persist full attempts through the provider-failure path, but generic history compaction keeps only the failure category/status and loses the logical ID, denial reason, and governing limits.

This occurs before the existing eight-round and 8192-byte history bounds. Expanding generic error details or putting complete attempt records into the prompt would treat the symptom, not the missing cross-round contract.

### Internal state is coupled to a lossy provider alias

Canonical payload sections are assembled first, then a legacy `public_context` alias is appended. Live authoring, replay `AgentLoop`, and the dry-run audit read that provider-facing alias back as internal round state. The current alias projection removes full repair feedback, so changing a compatibility view changes ledger and memory behavior.

The alias also repeats recent history. A current 30-round probe measured a maximum payload near 82 KB and roughly 12.5 KB of removable alias serialization. This is not a payload explosion, but it is avoidable protocol complexity and a correctness hazard.

### The generic audit namespace owns a push-specific scenario

`true_rl_agent.audit.loop_audit` constructs the push-recovery plugin, memory adapter, dry-run eval builders, push-specific candidate fixtures, and motion metrics. It is a synthetic push-recovery replay scenario, not the shared production eval audit. The shared `build_eval_run_audit` and injected `DryRunAuditRunner` seams are already task-neutral.

## Design

### 1. Separate internal round context from provider payload

Add an internal immutable assembly result, conceptually:

```python
PublicPromptBuild(
    round_public_context=...,
    provider_payload=...,
)
```

`PublicContextBuilder` will build both values from the same sanitized public records without deriving either from the other after serialization.

`round_public_context` contains only durable round state:

- `public_only`
- `previous_rounds`
- `best_public_score`
- `last_outcome`
- `pure_knob_streak`
- bounded recent public outcomes and omission metadata
- an optional task-memory summary produced by `TaskMemoryAdapter`

It excludes provider-only task sections, source catalog entries, source payload contents, compatibility aliases, and within-attempt source contents.

`provider_payload` contains the canonical ordered prompt sections, within-attempt repair feedback, source catalog, and allowed source payloads. All provider calls and saved prompt artifacts receive this exact object.

Internal consumers must use `round_public_context` directly:

- `CandidateAuthoringLoop`
- replay-only `AgentLoop`
- the task-owned dry-run audit

No production code may index `provider_payload["public_context"]` to recover internal state.

`PublicContextBuilder.build_prompt_payload(...)` remains a convenience wrapper for callers that need only the provider payload. Internal orchestration uses the two-value assembly API.

### 2. Persist a bounded source-request round summary

`CandidateAuthoringResult` gains an optional `source_request_feedback` value derived at the source-request service boundary. It is present only when at least one source request occurred.

The public shape is:

```json
{
  "schema_version": "true_rl_agent.source_request_round_feedback.v1",
  "status": "source_request_denied",
  "request_count": 1,
  "returned_logical_ids": [],
  "denied_logical_ids": [
    {
      "logical_id": "current_best/training_code/components.py",
      "reason": "source_not_requestable"
    }
  ],
  "budget": {
    "source_request_count": 1,
    "max_source_requests_per_round": 2,
    "max_source_ids_per_request": 2,
    "cumulative_returned_bytes": 0,
    "max_source_request_bytes_per_round": 49152
  },
  "omitted_returned_logical_id_count": 0,
  "omitted_denied_logical_id_count": 0
}
```

Rules:

- Status is deterministic: use `source_request_partial` when the round both returned and denied IDs, `source_request_denied` when it denied IDs and returned none, and `source_payload_returned` when all retained resolution outcomes returned source payloads without a denial.
- Logical IDs are drawn only from the closed prompt catalog.
- Denial reasons are service-defined enum-like codes.
- Returned and denied IDs are deduplicated in encounter order.
- The combined maximum retained returned/denied ID entries is derived from `MAX_SOURCE_REQUESTS_PER_ROUND * MAX_SOURCE_IDS_PER_REQUEST`. If a custom loop permits more requests, the summary still uses this shared prompt-memory ceiling and reports omissions.
- When the ID ceiling is reached, retain the most recent denial entries first, then the most recent returned IDs; preserve encounter order within the retained subset.
- The serialized summary has a hard 2048-byte ceiling; any future overflow is represented by omission counts rather than unbounded text.
- Exact limits and cumulative usage come from the real resolution records, including custom per-round request limits.
- The summary never includes source contents, provider rationale, instructions, candidate code, absolute paths, private artifact references, or raw error text.

The summary is persisted on every ledger path:

- successful candidate rounds through an optional generic `RoundInputs` field
- provider-failure rounds
- static-authoring-failure rounds

The field is `None` when no source request occurred. Complete attempt records remain in the run summary and existing failure-ledger audit records; they are not added to successful ledgers or prompt history.

### 3. Project source feedback into bounded public history

`context.py` adds `source_request_feedback` to the fixed recent-outcome field set and uses a dedicated defense-in-depth projector. The projector accepts only:

- schema/status/count
- returned catalog IDs
- denied catalog IDs and reason codes
- selected budget usage/limits
- omission counts

It does not reuse generic provider-error detail projection.

Existing history limits remain authoritative:

- at most 8 recent rounds
- at most 8192 serialized bytes for recent outcomes

The 2048-byte source-feedback ceiling ensures the latest actionable source state cannot dominate the history budget. The current source catalog in the new round remains authoritative; historical feedback tells the provider why a prior request failed, while the current catalog tells it whether availability has changed.

Generic transient authoring repairs that succeeded within the same round are not promoted into a new event stream. Final provider/static failures already have bounded summaries, and adding every resolved parse/verifier/smoke event would add prompt noise unrelated to future source availability.

### 4. Downgrade the legacy `public_context` alias

After internal consumers are decoupled, keep a single staged compatibility summary instead of the near-copy:

```json
{
  "schema_version": "true_rl_agent.public_context_legacy_summary.v1",
  "deprecated": true,
  "public_only": true,
  "previous_rounds": 30,
  "best_public_score": 0.46,
  "last_outcome": "reject",
  "pure_knob_streak": 0,
  "canonical_sections": {
    "history": "public_history",
    "public_feedback": "public_feedback_summary",
    "runtime_feedback": "runtime_feedback_summary"
  }
}
```

When within-attempt repair feedback exists, the alias may contain only the existing reference to `provider_or_runtime_repair_feedback`.

The alias must not contain:

- `recent_outcomes`
- history limits or omission state
- `task_public_summary`
- source catalog/payload data
- full repair feedback

Its serialized size must stay below 1024 bytes.

For a `TaskMemoryAdapter` used without a custom `TaskPromptAdapter`, the generic canonical `public_feedback_summary` carries the opaque sanitized value under `task_public_summary`. This prevents the alias downgrade from silently removing an advertised extension seam. A custom task prompt adapter may continue to project the task summary into its own canonical public-feedback shape.

### 5. Move the push dry-run scenario into task ownership

Move the complete scenario from:

```text
src/true_rl_agent/audit/loop_audit.py
```

to:

```text
src/true_rl_agent/tasks/push_recovery/loop_audit.py
```

Rename the public entrypoint to `run_push_recovery_loop_audit` and update the push-recovery CLI composition roots and integration tests to import it explicitly.

The task-owned artifact schema becomes:

```text
true_rl_agent.push_recovery.loop_audit.v1
```

Delete the generic `true_rl_agent.audit` re-export. Do not introduce an `AuditScenario` abstraction or a `TaskPlugin.audit_*` hook until a second real task demonstrates the same orchestration need.

Keep these shared:

- `build_eval_run_audit`
- `DryRunAuditRunner`
- opaque `task_eval_metadata`
- generic ledger, decision, runner, prompt, and memory contracts

Add a boundary test that rejects concrete task-package imports from:

- `agent_core`
- `prompting`
- `memory`
- `evaluation`
- `runner`
- any future generic `audit` package

Historical plans and development logs remain historical records; only live code, active documentation, tests, and CLI imports are updated.

### 6. Make omission bookkeeping idempotent

History values may pass through defensive compaction more than once. Existing `omitted_detail_count` can therefore be replaced by a count of the already-compacted mapping rather than preserving the original omission count.

Detail compaction will:

- preserve a valid prior omission count
- add only newly omitted fields
- produce the same result when applied repeatedly

The same idempotence rule applies to the new source-feedback omission counts.

## Data Flow

```text
ledger public records
  -> PublicContextBuilder
       -> round_public_context ---------------------> RoundInputs / ledger
       -> canonical provider context
            -> provider_payload -------------------> saved prompt + provider
                    -> source_request resolution
                         -> bounded round feedback -> CandidateAuthoringResult
                                                   -> every ledger outcome path

next round ledger
  -> fixed public-history projector
       -> bounded source_request_feedback
       -> public_history
       -> provider_payload
```

The only repeated values left in the legacy alias are four compatibility scalars. Recent outcomes have one canonical provider-facing location.

## Failure Handling

- Malformed source-feedback mappings fail closed to an absent or partially projected summary; they cannot expand the prompt surface.
- Unknown logical IDs and denial fields are omitted and counted.
- Forbidden public markers are redacted by the existing public-safety layer.
- A summary that would exceed 2048 bytes drops oldest returned IDs first and oldest denied IDs second, increments the matching omission counts, and never drops fixed status/count/budget fields.
- Current-round source resolution continues to use the live source catalog and existing request/id/byte limits.
- Moving the dry-run audit does not change production eval, acceptance, or runner behavior.

## Testing Strategy

Implementation follows test-driven development. Required red tests cover:

1. A denied source request followed by same-round candidate success is available in the next round with logical ID, reason, and limits.
2. A final-attempt denial/provider-failure path exposes the same bounded next-round shape.
3. Returned, partial, repeated, turn-limit, ID-limit, unavailable-source, and byte-budget outcomes aggregate deterministically.
4. Source feedback rejects or redacts source content, rationales, paths, secrets, and unknown fields.
5. Source feedback stays at or below 2048 bytes and public history stays within 8 rounds/8192 bytes.
6. Internal round context remains correct when the provider alias is shortened or absent.
7. Authoring, replay, and dry-run paths no longer read `provider_payload["public_context"]`.
8. The alias contains only the fixed short shape, remains below 1024 bytes, and no longer duplicates recent outcomes.
9. A task-memory-only adapter remains visible through canonical `public_feedback_summary.task_public_summary`.
10. Omission counts are stable under repeated compaction.
11. The push audit CLI and 15/30-round audit still pass after relocation.
12. Generic packages contain no direct concrete task imports or push-recovery literals.

Verification after focused tests:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

Use the bundled Python runtime if the default `python` is unreliable. Re-run a 30-round dry audit and report:

- maximum provider payload bytes, below 262144
- maximum canonical section bytes, below 98304
- maximum public-history bytes, at or below 8192
- maximum source-feedback bytes, at or below 2048
- maximum legacy-alias bytes, below 1024
- zero exact duplication of recent outcomes outside `public_history`
- zero task-specific imports or push literals in generic packages

## Non-Goals

- Do not change training budgets, eval seeds, checkpoint selection, PPO, acceptance thresholds, or candidate permissions.
- Do not migrate the broader eval score ABI from task-owned `success_rate_stats`; shared acceptance already has canonical `acceptance.score_stats`, and that independent migration overlaps active eval-hardening work.
- Do not redesign `RewardInputs` or introduce a new task ABI.
- Do not persist full provider attempt records on successful ledgers.
- Do not expose every resolved authoring error to future prompts.
- Do not create a generic audit plugin abstraction for one task.
- Do not remove the thin `candidates.prompt_projection` re-export facade in this change; it does not duplicate provider data or control internal state.

## Acceptance Criteria

The design is complete when:

- both source-denial paths retain actionable but bounded next-round feedback
- prompt history contains no source contents, raw attempt records, private values, or task-specific semantics
- all internal consumers use explicit round state instead of a provider compatibility alias
- the alias is a sub-1024-byte deprecated summary with no recent-history duplication
- task-memory output has a canonical provider-facing path
- the push audit scenario and artifact schema are task-owned
- omission accounting is idempotent
- focused and full test suites, compilation, diff checks, boundary scans, and the 30-round payload probe pass
- unrelated existing working-tree changes remain intact
