# Memory and native session-storage ownership assessment

## Decision

At `5b50132aefd95def802faa92a4756f89e990e94c`, retain memory's shared state,
lock, event/checkpoint prepare-and-apply primitives, and complete
run/publication/lifecycle operations. The review found no reason to split these
operations further just to reduce file size. The extracted owners implement
complete operations over explicit state and capabilities; their tests exercise
them without constructing a session-store implementation.

**The combined assessment satisfies the structural ownership requirements of
plan sections 5.2.5 and 5.2.** Creation/fork ownership and its diagnostic
correction are merged. No further production extraction is justified by this
review. The [assessment issue](https://github.com/cayu-tech/cayu/issues/2430) and
[umbrella issue](https://github.com/cayu-tech/cayu/issues/1111) track acceptance
of this record.

### Creation diagnostics and final integration

Post-merge CI exposed three fork-redaction regressions outside the initial
storage selection: generated creation-state representations printed borrowed
transcript/checkpoint content when exception locals were formatted.
[The correction](https://github.com/cayu-tech/cayu/pull/2433) makes all four
creation state/capability representations opaque and adds four regression cases.
Creation/fork algorithms, fields, callback identity and locking are unchanged.
The algorithm AST differs only in representation settings and class docstrings.

The expanded qualification ran at
`965db2501486dc09713627114c751ce49c523436`. Its **entire tree and detailed commit
message match merged `5b50132aefd95def802faa92a4756f89e990e94c`**. This verifies
integration without claiming that the long tests or package checks were rerun
after merge. The final selection includes all three workload-secret suites and
the four diagnostic regression cases; the earlier 934-case selection remains
historical evidence in [verification.json](verification.json).

The scope is structural ownership under
[plan section 5.2](https://github.com/cayu-tech/cayu/blob/96f5f5fd55d9a87900d6ede1963c216af19d5afc/SIMPLIFICATION_PLAN.md#52-split-backend-contract-and-schema-ownership).
Schema rebaselining, PostgreSQL row-shape changes, universal storage
orchestration and lifecycle feature work retain their separate decisions.
This assessment changes no production code, schema, API or test timing.

## Complete memory operation owners

Paths in this table are relative to `src/cayu/sessions/`. Counts include
comments, blank lines and declarations; they are not complexity scores.

| Owner | Responsibility | Lines |
|---|---|---:|
| `_memory_event_reads.py` | Event queries, accounting and terminal evidence reads | 796 |
| `_memory_session_queries.py` | Discovery, lineage, lifecycle and pending-action queries | 717 |
| `_memory_transcripts.py` | Transcript reads/writes, search and attribution indexes | 752 |
| `_memory_evidence.py` | Recall receipt and context-exposure persistence | 449 |
| `_memory_event_delivery.py` | Persisted-event outbox claims and settlement | 461 |
| `_memory_tool_grants.py` | Targeted grants, use binding and reconstruction | 931 |
| `_memory_queued_messages.py` | Queue admission, inspection, delivery and replay | 1,011 |
| `_memory_peer_content.py` | Peer-content admission, receipts, attempts and retry | 616 |
| `_memory_context_views.py` | Source capture, publication, selection and retention ownership | 813 |
| `_memory_session_creation.py` | Ordinary/fork creation, participant replay, bindings and receipts | 880 |

Each owner receives a callable that acquires its operation's lock and then
constructs a typed state view. Owners resolve current indexes when entering
that view, including dictionaries replaced during rollback. Most main operation
views have 3–20 direct fields. Creation is the largest at **37 fields**, spanning
session records, lineage, transcript indexes, selection receipts, participant
receipts, creation authority and paired publication capabilities. That interface
is a real cost of independent composition; its fields support atomic
creation and fork operations. Splitting them into separately locking owners would not
remove that coupling.

Intentional dependencies preserve existing atomic work: peer admission uses the
queue's ordering record; queue delivery and creation use transcript index
functions; creation uses the shared creation-fence rules. These helpers do not
reacquire the store lock. Pure validation and record contracts retain their
shared domain modules. Independent import/composition tests cover all ten owners.
Participant creation separately serializes participant ownership before awaiting
the nested store operation, as described below.

## Why the remaining boundary stays together

| Retained responsibility | Ownership reason |
|---|---|
| State, locks, capability versions and state factories | The composition root supplies one coherent set of current indexes and narrow callbacks. |
| Event/checkpoint preparation and application | Publication, grants, transcripts and queue operations share the same event, pending-action, accounting and checkpoint projections. Moving each projection behind a separately locking API would break the atomic boundary. |
| Status/interaction transitions, run fences and invocation release | Each operation checks current authority and then publishes the corresponding session, checkpoint, event and receipt changes under one lock. |
| Model-stage settlement and runtime publication | Promotion calls the same publication primitive that publishes transcript rows, events, checkpoint mutations, winner state and exact replay receipts together. |
| Closure, deletion, detach and export snapshots | Their admission and updates span lineage, active work, retention pins and multiple evidence indexes. The existing operation owns that consistent view. |
| Store interface methods | Delegates preserve the contract and operator-access guards while composing the complete operation owners. |

The largest retained operations are status/checkpoint transition
(347 lines), interaction transition (331), deletion (262), runtime publication
(234) and event application (202). These are full operations, not new forwarding
layers. The 229-line constructor mainly composes mutable indexes. Creation and fork
operations now have their own independently usable owner; the retained adapter
composes their state and capabilities.

Existing external-wait, execution-owner and creation-fence mixins remain. This
assessment adds no mixins and does not claim that every historical component is
independently constructible. Further changes to those components need an actual
ownership, correctness or performance finding.

### Authority, cancellation and replay

- Resource-access checks and run/closure admission remain inside the operation's
  locked read or mutation. Moving implementation does not remove the public
  store's operator-only guards.
- Event and checkpoint preparation precede their shared apply path. Transcript
  indexes, pending-action projections and durable receipt material remain in the
  same publication boundary. Replay validates exact receipt and material
  identity rather than publishing the operation a second time.
- Ordinary/fork creation and the other extracted mutations do not suspend while
  publishing under the store lock. Cancellation during lock acquisition cannot
  partly publish them. Participant creation holds the participant lock while
  awaiting nested creation or a fence read; this preserves participant-before-
  store lock ordering shared with deletion. Cancellation releases that lock.
  Receipt lookup after the await resolves the current receipt dictionary.
- The retained session-operation path has another explicit cancellation boundary:
  its owned off-thread commit guard is awaited while ownership is retained, then
  a captured cancellation propagates after the guard and publication settle.
- Context selection restores receipt, commitment and lifecycle-event indexes
  together after failed publication. Creation resolves those current indexes
  after acquiring the store lock and validates borrowed selection identity,
  provenance, adoption/transfer state and creation target before publication.
  Opaque checkpoint/event preparation records return to their paired apply
  callbacks; failed preparation leaves no partial creation.
- Native transaction, advisory-lock and connection ownership remain as recorded
  in the [native assessment](../native-session-storage-assessment-2026-10-11/README.md).
  SQLite retains its connection until an off-thread worker finishes even when
  the caller is cancelled; PostgreSQL operations keep their transaction scopes.

## Combined storage context and source cost

The discussion plan already records completed knowledge, task, shared-session,
SQLite schema/initialization and native backend/operation ownership. This record
consolidates the remaining memory boundary with the native session evidence;
it does not silently requalify every earlier delivery at today's revision.
The earlier [shared-session](https://github.com/cayu-tech/cayu/blob/96f5f5fd55d9a87900d6ede1963c216af19d5afc/docs/plans/session-contract-ownership-assessment-2026-10-11/README.md),
[SQLite initialization](https://github.com/cayu-tech/cayu/blob/96f5f5fd55d9a87900d6ede1963c216af19d5afc/docs/plans/sqlite-initialization-assessment-2026-10-08/README.md),
[knowledge inventory](https://github.com/cayu-tech/cayu/blob/96f5f5fd55d9a87900d6ede1963c216af19d5afc/docs/plans/knowledge-ownership-assessment-2026-10-05/assessment.json)
and native records retain their own revisions and qualification limits.

| Measured population | Before first memory extraction | Assessed revision |
|---|---:|---:|
| Memory session adapter | 12,090 | 6,969 |
| Memory adapter plus extracted owners | 12,090 | 14,395 |
| Extracted memory operation modules | 0 | 10 |
| SQLite / PostgreSQL session adapters | — | 8,222 / 7,473 |

Across thirteen memory extraction commits, comparing **each commit to its own
parent**, the net cost is **+2,296 production lines** (19.0%) and ten modules.
The diagnostic correction adds nine production documentation lines, for a
combined increase of 2,305 lines. Its cost is recorded separately from extraction
costs, with no interleaved growth in these files. The adapter shrinks by
5,121 lines (42.4%). [inventory.json](inventory.json) records every delivery,
owner, retained large method and baseline revision. This cost is an explicit
tradeoff for the independent operation boundaries; the original source
non-increase forecast was not achieved. These figures are not a total for the
whole storage campaign, and earlier delivery percentages must not be added.

All ten memory owners and the three adapters are below 10,000 lines. The
ownership decision also depends on the complete operation boundaries and the
integrated qualification below. File size alone does not establish it.

## Final integrated qualification

The 1,132-case selection is in [test-selection.json](test-selection.json), with
results and exact merge provenance in [verification.json](verification.json). It combines all memory
ownership tests with shared lifecycle, publication, model-stage, fork and
compaction cases, invocation lifecycle contracts/commands, checkpoint
publication, operation fault injection, creation fences, native
creation/publication ownership, context-selection/recipient consumers and all
three workload-secret suites.

The selection covers authority, replay, competing operations, failed admission,
cancellation, protected checkpoint roots and independent composition. It uses
scripted providers and a disposable UTF-8 PostgreSQL database; no live model API
or production service is involved. PostgreSQL is required, so missing database
coverage fails rather than silently skipping.

Source and wheel counts, Python 3.11 coverage, import-origin checks, package
parity and static/collection results are recorded separately. The final results
were executed at the reviewed correction, whose entire tree matches merged
`5b50132ae`. Hosted CI is separate from this local selection.

| Fresh check | Result |
|---|---:|
| Python 3.14 source selection | 1,132 passed, including 226 PostgreSQL cases |
| Python 3.14 installed-wheel selection | Same 1,132 cases passed, including 226 PostgreSQL |
| Python 3.11 memory/fork-redaction selection | 332 passed; 800 deselected |
| Strict full collection with current timing entries | 47,891 collected |
| Source/wheel/sdist/installed Python and stub parity | 1,385 files identical |

There were no skips in either combined run. The import guard checked 919 loaded
Cayu modules per combined run. Lint, formatting, production type checks, source
size policy, generated sidecar/dashboard checks and release-archive validation
passed. All 46,498 timing entries are byte-for-byte unchanged by this assessment.
The correction preserved 46,494 earlier values and added four measured entries.

The twenty-one creation ownership cases include diagnostic opacity, direct
composition, ordinary and
fork preparation failures, cancelled lock waits, current state/clock lookup,
participant replay, receipt replacement and creation/deletion serialization.
Context-view cases cover restored receipt/commitment/lifecycle state and the
next operation's fresh lookup. Additional independent creation probes check
opaque prepare/apply pairing and publication order for success, preparation
failure and cancellation.

## Bounded performance

These measurements remain at `1c0e9bfb0`, before the diagnostic correction;
that correction changes representation settings and class docstrings only.
The timed workloads do not format state representations. No new benchmark run
is claimed for the merge.

[benchmark.py](benchmark.py) loads the pre-extraction memory adapter from
`7fbcd127e8ffecef6fefeb6add32826962ddf040` alongside the current adapter, using
the **same current shared dependencies**. It measures the adapter composition
change, not two historical releases. Setup, imports, result serialization and
evidence hashing are outside the timed region. Five warmup pairs precede thirty alternating pairs.
Each pair checks complete workload event payloads, transcript rows/attribution,
checkpoint values and read results. Their hashes match in every pair.

| Workload per sample | Before / current median CPU | Change |
|---|---:|---:|
| 50 bounded event/transcript query pairs, 100 seeded rows each | 131.0195 / 132.0060 ms | +0.75% |
| 50 transcript/checkpoint updates plus checkpoint/event publications | 40.2135 / 40.6270 ms | +1.03% |

Median paired differences are +0.864 ms for the read batch and +0.2545 ms for
the write batch. This small measured CPU overhead is accepted for the independently
usable operation boundaries. Raw samples are retained in
[performance.json](performance.json). This probe does not qualify every
extracted owner, large-history scaling, sustained memory, contention, cold imports, native disk
throughput, agent latency or coding-assistant productivity. Earlier native and
per-delivery measurements keep their original scope.

Creation and fork setup are outside these two timed workloads. Their separate
[delivery evidence](https://github.com/cayu-tech/cayu/pull/2431) measures eight
workloads with 100 alternating pairs: seven small operations add 8.5–27 µs
(1.30–6.44%); the 512-row fork changes by −6 µs (−0.21%). That evidence belongs
to reviewed `31f5df5a3`, whose entire tree matches merged `1c0e9bfb0`; it was not
rerun as part of these two probes. Do not add the percentages across workloads.

## Reproduce

Run from the repository root with development and PostgreSQL dependencies:

```sh
python docs/plans/memory-storage-assessment-2026-10-12/inventory.py
CAYU_REQUIRE_POSTGRES_TESTS=1 CAYU_TEST_POSTGRES_DSN="$DISPOSABLE_TEST_DSN" \
  python docs/plans/memory-storage-assessment-2026-10-12/run_selection.py
python docs/plans/memory-storage-assessment-2026-10-12/benchmark.py
```

The test database must be disposable; fixtures create/drop their own databases.
For the wheel run, use an interpreter with the built wheel installed and pass
`--installed-package /path/to/site-packages/cayu` to `run_selection.py`.
The guard checks loaded Cayu module origins at session start and finish.

The recorded selection already includes all workload-secret suites and diagnostic
regressions. To reproduce the minimum-Python selection, add
`-k 'memory or (workload_secret_redaction and fork)'` to the command.

The inventory is pinned to the final assessed revision. The benchmark rejects
dirty production source and records its production commit and source-tree hash.
Its committed samples retain the pre-correction revision shown above. Subsequent
source changes require their own qualification.
