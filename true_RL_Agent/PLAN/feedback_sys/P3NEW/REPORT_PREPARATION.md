# N5 offline preparation control

These current-source scripts are for a future source freeze. Historical R4
snapshots, answers, receipts and `results/*/prepare_facts_offline.py` are unchanged.
No API, model, author, capture, GPU, retry, semantic scoring or release is built in.
The tests use temporary fixtures; they are not actual facts or review evidence.

## Entry points

```text
python -B prepare_facts_offline.py --index <40-unit-input-index.json> --output <new-facts-directory>
python -B audit_confirmation_facts.py --prepared <original-prepare.json> --answer <original-answer.json> --output <new-structure-audit.json>
python -B prepare_confirmation_reports.py --index <80-original-facts-index.json> --output <new-engineering-audit.json>
python -B prepare_confirmation_reports.py --index <80-original-facts-index.json> --output <new-report-directory> --prepare-reports
```

The first command copies only small manifests and resolves original material
paths. DeepSeek gets 65,536 output tokens; SOL gets the original 32,768 reservation
(no CLI hard output cap). It does not replace original packed manifests or change
their materials. Request/carrier capacity must be measured before any dispatch.

The structure command uses the helper/protocol from the same frozen P3NEW source,
checks the prepared helper SHA and packet protocol's original source SHA, and
retains the existing facts-field checks and semantic limitations. The old R4
script and its receipts remain unchanged; no manual rebinding is needed.

The first report command checks all 80 original answers and writes an engineering
audit only. The `--prepare-reports` command is an explicit operator action, after review of the existing
semantic-scope obligations. It rechecks the full barrier, seals all 80 originals
before preparing any report, and writes only a new directory. Any handled failure
removes that exclusively created output, leaving original evidence untouched.
`preparation.json` is written last; absent receipt is never a completed batch.
An abrupt process termination may leave a partial directory: preserve it as
incomplete, never use it or resume into it. No output is released automatically.

## Input table (real references only)

Each reference is `{file: absolute_path, sha256: actual_file_sha}` (the existing
control form `{path, sha256, size_bytes}` is also accepted). Local mirrors must
preserve bytes; originals embedded in existing source manifests must remain
resolvable on the preparation host. The runner/helper on that host must match
the original prepared source hashes. No old answers are reinterpreted under a
new helper. Source freezing and transport mirroring remain operator work.

Both commands take `assignment` and `units[40]`. Each unit has `packet_id`,
`task` (`stair`/`push`), `axis` (`episode`/`focus`/`media`), `scope`, and original
packed `facts_manifest` reference. The fixed 24/12/4 units, per-task 12/6/2, A
all SOL-a, and B first episode/focus in each task plus all four media as DeepSeek
are checked against the frozen assignment. Other B slots are SOL-b.

The reports index adds:

- `facts[80]`: `packet_id`, `judge_id`, and references `facts_manifest` (the exact
  adopted per-judge manifest, including any registered output-cap version),
  `prepared`, `request`, `reference_catalog`, `answer`, `driver_receipt`,
  `structure_audit`. SOL rows also give original `registration` and
  `transport_receipt`. A relocated SOL prepare must be the original prepare
  whose SHA the transport receipt binds; do not manufacture an equivalent card.
- `structure_audit` is the existing facts contract check, run using the same
  original request, protocol and coverage helper. Its complete flag, errors,
  identity and source hashes are checked, along with a fresh mechanical reference
  audit. This script does not invent a passing structural or root semantic vote.
- `numeric_packing` and `media_packing`: original packing receipts. They bind
  task/axis/scope/alias, original manifest SHA, numeric source hashes and common
  capture. This prevents a report from a different cohort being substituted.
- `tasks.stair` and `tasks.push`: references `full`, `phase`, `request`,
  `raw_manifest`, `raw_sources` (the corresponding `_private/sources.json`),
  and literal `request_status: offline_serialized_not_sent`. Full/phase/public,
  original HTTP serialization and raw archive source hashes must agree.
- Media units additionally have `alias`, `original_id` (the actual logical
  episode ID), `observer_output` and `observer_audit` references. The latter is
  the original per-alias mechanical audit with its unchanged `raw_output`, not
  the combined root semantic review. Original PNG SHA/alias and all cited image
  hashes are checked against the facts request. Each unit may list existing
  `arithmetic_audits` refs; these are program checks, never semantic truth votes.

The table is control-side only. It is not sent to either evaluator. No table of
future completion evidence is generated here.

## Material scope and remaining review

Episode units append the complete selected episode and shared definitions using
the existing lossless Stair representation. Focus units append the entire actual
task report/final request, full-pool operands and every example referenced anywhere
in that report, including counterexamples; no focus or positive-result filtering
is applied. Original facts materials are retained on all axes. Each report binds
that same judge's exact answer bytes, original catalog and adopted manifest SHA.

Media units retain every raw observation, including statements not admitted to
the author report. Inventory roots point to each original `text`/`uncertainty`
and each `not_visible` natural-language statement. The control-side
`observer_inventory_mapping.json` maps every such root back to its complete
original observation; no observation may disappear. Body, relation, modality,
time and reference fields remain complete in that same observation and constrain
whole-statement support. They are not separate extra text counts, and are not
exempt from review. Existing observation support/overreach rules are unchanged.

The entire actual public report is provided, but this media unit's inventory
includes only its own alias's projected observation plus `visual.coverage` as
recorded admission text. Those recorded counts are not truth labels for judging
the raw observations. Another alias's images are not implicitly available.
Root per-statement reasons/verdicts are not supplied as visual ground truth.

The engineering barrier proves identity, original-byte bindings, complete
transport and the existing structural/reference contract. It does **not** prove
that all necessary facts were found, that every supplied operand was read, or
that facts, necessity, localization or report claims are true. The explicit
operator review, actual own-facts request/carrier token and PNG capacity, release,
raw report capture, disagreements and original task-specific scores remain
outside this preparation script. A completed fixture run or structural PASS is
not an N5 semantic PASS.
