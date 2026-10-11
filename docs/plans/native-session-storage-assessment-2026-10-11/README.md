# Native session-storage ownership assessment

This assessment covers SQLite and PostgreSQL session storage after native
publication ownership. The current inventory compares with parent `7fbcd127e`.
Behavioral and performance evidence below identifies its measured revisions.

## Scope and stopping point

The result supports completing the native session-backend portion of the initial
structural storage work. Complete query/evidence, transcript, queued/peer-message,
event/outbox, creation/fork, grant/view, recall, public-authority and model-stage
operations have native owners. Runtime publication and checkpoint/event
publication now share a complete publication owner in each backend.

The native adapters retain bounded run/checkpoint/lifecycle responsibilities and
compose those operation owners. Their remaining methods are inventoried in
[inventory.json](inventory.json). No further native extraction is required by
this assessment. This conclusion does not complete memory-backend decomposition,
the combined storage assessment, schema rebaselining, lifecycle-policy changes,
or pending-action and retention features.

| Retained responsibility | Reason to retain the current boundary |
| --- | --- |
| Store construction, resource access and protocol dispatch | This is the native composition root and supported store boundary. |
| Run/interaction transitions, fencing and invocation release | Each complete operation owns its authority checks, locks and mutations. Shared policy already has separate domain owners. |
| Ordinary checkpoint reads, transforms and snapshot capture | Readers and writers retain the same authoritative snapshot and protected-root validation. |
| Pending-action and terminal-receipt projections | These are bounded views over native run/checkpoint evidence; admission-policy changes remain separate. |
| Closure, erasure, event pruning and export snapshots | Transaction-local evidence, deletion fences and native retention protections stay together. Shared closure rules and record builders are already separate. |
| Native row readers and transaction-local writers supplied to owners | These provide explicit capabilities without passing a whole session store into an operation owner. |

The largest retained methods are the pending-action queries (922 SQLite / 748
PostgreSQL lines), status/checkpoint transitions (587 / 563), and interaction
transitions (522 / 504). Each remains one complete native operation. This
assessment sets no requirement to split a transaction or create a module for
every method.

## Publication ownership

Each native publication owner supplies:

- Runtime publication and exact replay, including the model-stage promotion
  transaction.
- Receipt reads and validation against durable transcript/event material.
- Checkpoint/event publication, including operation transforms, protected
  checkpoint roots, budget identities, terminal receipts and outbox work.
- Native commit guards, rollback, cancellation and driver-error handling.

Required store hooks retain their signatures and compose these functions.
Private adapter decoder and material-validator methods are removed; retained
native consumers call their canonical owner directly. Shared publication rules,
runtime execution, checkpoint schemas and connection lifetime retain their
existing owners.

The owners import without concrete adapters or optional PostgreSQL drivers.
Direct tests close and discard the seeding store, then compose real native
connections, readers and writers. They exercise the actual publication
transaction, including stage promotion, rather than replacing it with a
recording callback.

## Size accounting

| Backend | Adapter before | Adapter after | Publication owner | Net production lines |
| --- | ---: | ---: | ---: | ---: |
| SQLite | 9,188 | 8,222 | 1,090 | +124 |
| PostgreSQL | 8,515 | 7,473 | 1,176 | +134 |

This batch removes 2,008 lines from the adapters and adds 258 production lines
across two modules. The new signatures and composition boundaries account for
the growth. Both adapters and receiving owners are below 10,000 lines. The
native module inventory includes existing schema and support owners; it does
not count them as additions in this batch. The largest existing native modules
are schema-history owners at 4,516 SQLite and 5,126 PostgreSQL lines.

These figures compare this delivery with its own parent. They do not estimate
net deletion or productivity improvement across the broader campaign.

## Verification

Original qualification at `5bc3c4e89`, against parent `eef15d848`, used Python
3.14 with real SQLite and a disposable PostgreSQL 17 server. The source selection
passed 318 distinct cases;
the parent passed all 304 existing cases. The installed wheel passed the same
318 cases, with import guards confirming that Cayu modules came from the wheel.
The 14 added cases cover independent composition, rollback, cancellation,
receipt replay and stage promotion through real native publication.

All 109 distinct recovery cases pass: the PostgreSQL and SIGKILL recovery lanes,
plus the offline generated coding-loss case. These exercise durable
acknowledgement loss, restart, cancellation, staged model publication and
worker fencing. [verification.json](verification.json) records the tested
implementation and result counts.

The first three implementation commits also passed their backend's direct
ownership cases independently. The fourth passed the complete 14-case
publication selection; the fixture cleanup passed all 22 stage/publication
ownership cases. An AST comparison against the parent verified eight moved
operation bodies, both decoders, six protocol signatures/forwarders, 425
retained methods and 27 retained module helpers, allowing only the explicit
capability and canonical-owner rebindings.

Ruff, formatting, full configured source/example/maintenance type checks,
generated sidecar/dashboard checks and duration-guarded collection pass.
Collection finds 47,696 tests. All parent test timings are preserved, with 14
measured additions. Wheel/source-archive release checks and installed-layout
smoke checks pass; all 1,372 Python/stub files match source, wheel, source
archive and installation byte for byte.

Initial local runs encountered exhausted disk space and SQLite disk I/O errors.
Those interrupted runs are excluded from the passing totals; affected groups
were rerun after completed delivery artifacts were removed.

### Rebase qualification

Against parent `7fbcd127e`, source revision `d63987d47` passes 332 cases: the
318-case selection plus the newly merged manual-model-recovery composition
tests. A guarded installed-wheel selection passes 68 cases, including native
publication/stage composition, cancellation, replay, concurrency, material
validation, documentation integrity and manual recovery. No cases are skipped.

The rebase uses canonical session imports and preserves all moved bodies,
native SQL, protocol signatures and retained adapter operations. Static,
formatting, type, generated-file and duration-guarded collection checks pass;
47,725 tests collect. All 46,318 parent timing values and lines are retained,
with the same 14 additions. Package checks pass, with all 1,375 Python/stub
files matching source, wheel, source archive and installation. The earlier
parent, recovery and benchmark results retain their original revision scope.

## Bounded performance comparison

The comparison uses six balanced parent/head pairs per backend, with a fresh
process for each sample and ten warmups per workload. Each sample measures 100
receipt reads and replays, and 30 new runtime and checkpoint/event publications.
Session setup is outside the measured interval. SQLite runs against a local
file; PostgreSQL runs through a local Unix socket. Both revisions use the same
schema and equivalent seed data. Result summaries and normalized SQL statement
counts match across all samples.

This comparison is pinned to parent `eef15d848` and implementation `5bc3c4e89`.
The subsequent rebase changes native imports to the canonical session owners;
the benchmark does not measure the intervening shared-store and recovery work.

PostgreSQL counting covers `execute` calls and each parameter set consumed by
`executemany`, with batch-call totals recorded separately. Driver-managed
transaction commands and server-internal work are outside that count. SQLite
uses trace callbacks; SQL counts are compared within each backend. Before each
PostgreSQL sample, an untimed temporary-table probe checks list, generator and
empty batches: three calls consume five parameter sets and persist five rows.

Runtime publication records 16 PostgreSQL statements, including one transcript
insert and one event insert through two batch calls. Checkpoint/event publication
records 12 statements, including one event insert through a batch call. The
stored results include normalized statements, batch calls and parameter-set
counts. This rerun replaces the earlier comparison that omitted batched writes
from PostgreSQL SQL counts.

| Workload | SQLite elapsed / client CPU | PostgreSQL elapsed / client CPU |
| --- | ---: | ---: |
| Receipt read | +0.02% / −0.20% | −5.57% / −4.42% |
| Runtime replay | −5.44% / −5.54% | −0.43% / +0.12% |
| Runtime write | −17.00% / −15.23% | −0.03% / +0.29% |
| Checkpoint/events | −2.66% / −3.52% | −1.10% / −0.25% |

Elapsed medians range from −17.00% to +0.02%. The largest increase is SQLite
receipt reading: 0.79644 ms to 0.79660 ms, approximately 0.17 microseconds.
Individual paired samples vary in both directions; SQLite runtime-write pairs
range from −23.92% to +4.31%. These shared-host, instrumented measurements
establish neither a speedup nor zero overhead. They do not qualify contention,
large records, tail latency or PostgreSQL server CPU. The sample medians and
paired changes are recorded in [performance-summary.json](performance-summary.json).

## Reproduce the inventory

From the repository root:

```sh
python docs/plans/native-session-storage-assessment-2026-10-11/inventory.py
```

The script reads source and Git history without importing adapters or opening a
database. The publication and stage ownership tests live in
`tests/core/test_native_publication_ownership.py`,
`tests/core/test_postgres_publication_ownership.py`,
`tests/core/test_native_model_stage_ownership.py` and
`tests/core/test_postgres_model_stage_ownership.py`. Shared test fixtures live in
support modules, with no imports between these collected test modules.

[test-selection.json](test-selection.json) lists the existing behavioral
selection; add the two publication ownership modules for the 318-case head
selection. [recovery-selection.json](recovery-selection.json) records the
process-loss selection. Run PostgreSQL selections with
`CAYU_REQUIRE_POSTGRES_TESTS=1` and `CAYU_TEST_POSTGRES_DSN` pointing to a
disposable test server.

The paired benchmark requires the project's test dependencies and a PostgreSQL
test account allowed to create databases. It creates and drops only its own
uniquely named database and uses a temporary SQLite directory. From the changed
checkout, with `CAYU_TEST_POSTGRES_DSN` set to that test server:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  docs/plans/native-session-storage-assessment-2026-10-11/benchmark.py \
  --base /path/to/parent-checkout --head /path/to/changed-checkout \
  --output /path/to/new-results-directory
```

Use parent `eef15d848` and implementation `5bc3c4e89` to reproduce the recorded
comparison with this assessment's corrected harness. The production code and
workloads are unchanged; PostgreSQL instrumentation now includes batched writes.
Harness hashes are recorded in `verification.json`. Results depend on the machine
and its load.
