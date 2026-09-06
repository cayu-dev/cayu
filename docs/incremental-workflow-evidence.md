# Incremental evidence for saved workflow evaluations

Use `capture_incremental_workflow_eval_attempt` to verify a large, completed
workflow attempt without building an in-memory `Trajectory`. SQLite and PostgreSQL
copy bounded pages into a private disk-backed snapshot. Deterministic scoring uses
bounded summaries of the complete verified scope. No workflow, factory, projector,
provider, tool, probe, or model judge runs during capture.

This is an explicit saved-attempt API. Live workflow targets still use
`SessionTrajectoryBounds`; their defaults and eager hard ceilings are unchanged.
The original failed or unavailable trial remains intact. Legacy reports first use
[`import_workflow_eval_attempt`](workflow-eval-recovery.md#import-reports-written-before-attempt-anchors)
to establish an explicit saved-store anchor.

## Capture and score

Reconnect the original target to the original saved session store. Preserve the
original target configuration and supply the original input and projected output.
Create one admission controller for the worker and share it across every capture.

```python
from pathlib import Path
from cayu import (
    IncrementalEvidenceAdmission,
    IncrementalEvidenceLimits,
    WorkflowEvalResult,
    capture_incremental_workflow_eval_attempt,
    score_incremental_workflow_eval_capture,
)
from cayu.evals.corpus import FinalOutputEqualsAssertionSpec

admission = IncrementalEvidenceAdmission(
    max_captures=4,
    max_buffer_bytes=512 * 1024 * 1024,
    max_spill_bytes=2 * 1024 * 1024 * 1024,
)
limits = IncrementalEvidenceLimits(max_events=1_000_000)
output = WorkflowEvalResult(
    final_output=original_projected_text,
    structured_output=original_projected_structured_output,
)
capture = await capture_incremental_workflow_eval_attempt(
    original_target,
    source_trial,
    messages=original_messages,
    output=output,
    limits=limits,
    admission=admission,
)
with Path("incremental-capture.json").open("x") as stream:
    stream.write(capture.model_dump_json())

score = await score_incremental_workflow_eval_capture(
    original_target,
    source_trial,
    capture,
    (FinalOutputEqualsAssertionSpec(id="answer", expected=expected_answer),),
    messages=original_messages,
    output=output,
    admission=admission,
)
with Path("incremental-score.json").open("x") as stream:
    stream.write(score.model_dump_json())
```

`SavedIncrementalWorkflowCapture` and `SavedIncrementalWorkflowScore` are separate
versioned JSON documents, with independent IDs and timestamps. They are not
`EvalRun` or `Trajectory` documents and are not accepted by their HTML/server
renderers. The score includes the complete compact revalidation receipt, the source
capture ID, original target and evidence-policy revisions, assertion revisions,
and `scoring_semantics="cayu.incremental-assertions.v1"`. Both documents record
`model_calls=0`; execution usage remains in the original trial.

`evidence_complete=true` means the selected scope passed full evidence validation.
`retained_evidence="summary_only"` means payloads have been removed: the document
retains session seals and bounded counters, not a trace preview masquerading as
complete assertion input. Preserve the original store for another consumer or
future scoring. No output text or event payload is included in these summaries.

## Supported consumers

| Assertion | Evidence and limits |
| --- | --- |
| `FinalOutputEqualsAssertionSpec`, `FinalOutputContainsAssertionSpec` | Original hash-bound projected output, existing application redaction and 65,536-character admission limit. Oversized output is unavailable, not truncated assertion input. |
| `RootStatusAssertionSpec` | Exact saved workflow completion. |
| `ChildStatusAssertionSpec` | All direct children admitted before the root terminal boundary, preserving origin order. |
| `ProcessEventAssertionSpec` | Constant-size counters over every selected event, in root then descendant preorder. No payload or event-name list accumulates. |
| Other portable or arbitrary assertions | `assertion_requires_unsupported_evidence`, before source reads. |
| Model-backed judges | Unsupported by this scoring API; capture never dispatches a judge. |

One scoring action accepts 1–100 specs. Supported assertions agree with eager
scoring on manageable evidence. Process-event counting deliberately has separately
versioned semantics: it counts the full admitted scope beyond the eager process
fact retention limit. Assertion spec identities remain their existing content
revisions; the score records the incremental execution semantics too.

Full-trace, random-access, workspace, artifact, memory-attribution, cost/usage and
model-judge consumers must explicitly use an applicable existing bounded path.
The private backing is not an invitation to pass an unlimited trace to a model.
There is no implicit materialization, probe execution, or provider fallback.

## Resource contract

`IncrementalEvidenceLimits` rejects unknown fields, booleans as integers, invalid
ranges and unlimited values. A capture records the effective limits and their
`capture_policy_revision`. A recapture may change limits while retaining the
original execution target revision; provide `expected_evidence_sha256` to require
an earlier evidence seal.

| Limit | Default | Hard maximum |
| --- | ---: | ---: |
| Child events per complete source pass | 1,000,000 | 100,000,000 |
| Child transcript records per pass | 100,000 | 10,000,000 |
| Canonical individual record | 1 MiB | 8 MiB |
| Canonical child evidence per pass | 256 MiB | 4 GiB |
| Private backing allocation | 512 MiB | 8 GiB |
| Records per source page | 4 | 256 |
| Retained child identities | 100 | 500 |
| Depth including root | 32 | 32 |
| Lifecycle bookkeeping per session | 10,000 records / 2 MiB | 100,000 records / 16 MiB |
| Whole capture deadline | 300 seconds | 3,600 seconds |
| Deliberately materialized root journal and session | 10,000 events / 4 MiB | 10,000 events / 16 MiB |

Child event/transcript/evidence limits are aggregate across the entire tree, not
reset allowances for each child. The event bound is a **work budget**, independent
of the page size and eager 100,000-event materialization ceiling. Two complete
source passes are required per capture. Scoring performs a new capture against
the prior seal. Local validation makes several bounded passes over the backing;
`spill_records_read` accounts for these reads separately from source records.

The root remains a deliberate bounded materialization. Root event reads use the
store's byte-before-hydration capability, one record at a time, with an independent
3:2 transport allowance plus the query API's 256-byte envelope guard. Canonical
root and individual-record limits still apply after transport. Native PostgreSQL
child transport has its existing independent 3:2 JSONB expansion guard. Increasing
canonical bounds does not disable these transport checks.

There is one active child backing and source reader per capture. Source pages are
pulled synchronously with the copy; no background producer queue accumulates.
Consumed payloads are released before the next child. Bookkeeping is bounded by
500 lineage candidates, selected session limits, and explicit lifecycle count and
byte limits. The private SQLite cache is 256 KiB. Its record identity index grows
on disk and is charged to the spill budget. SQLite floors that budget to 4,096-byte
pages; the minimum is 12,288 bytes. Authenticated JSON envelopes also have a finite
`3 * max_record_bytes + 4096` encoded-record cap.

`IncrementalEvidenceAdmission` defaults to at most eight active captures, a
512 MiB buffer reservation budget, and 4 GiB of spill reservations. All three
limits must admit a capture. Reservations account for transport pages, current
serialization buffers, bounded root materialization, cache and bookkeeping.
`buffer_envelope(limits)` exposes the per-capture reservation. This is a serialized
buffer/resource reservation, **not a process RSS promise**: Python objects,
allocator retention, database/OS caches, imports and caller-owned application
state have separate costs. No universal process-wide controller is created
implicitly. Multiple controllers/workers require an explicitly partitioned outer
resource budget.

Excess arrivals fail immediately with `aggregate_admission_rejected`; there is no
hidden unbounded wait queue, retry loop or execution scheduler. Callers retain
control of later admission. A reservation is released only after owned copy and
validation work settles and its backing is removed, including cancellation.

## Snapshot, identity and integrity

`SessionStore.supports_incremental_terminal_evidence` declares the capability.
`load_bounded` preflights a complete session, including invocation metadata and
labels. `export_terminal_session_evidence(session_id, spool=...)` resolves terminal
status, publication marker, epoch, terminal event, counts, record bounds and the
complete attributed transcript under one native snapshot:

- SQLite uses `BEGIN`, a bounded cursor and `fetchmany`. Cancellation/deadline
  interrupts the source query; rollback releases the snapshot.
- PostgreSQL uses `REPEATABLE READ READ ONLY`, a statement timeout, and keyset
  pages: event sequence strictly after the cursor and at or before the fixed
  terminal cutoff; transcript `session_order` strictly after the cursor. No
  offset pagination or driver-wide `fetchall` of the entire trace is used.
- The database transaction and connection are released before local integrity
  validation or scoring. A slow consumer cannot keep that source snapshot open.

The backing contains bounded JSON records, native runtime authority proofs and
position-bound authentication tags. Its key exists only in the owning object.
Reads verify integrity before decoding, reject oversized encoded records before
hydration, enforce event identity uniqueness, and detect missing/repeated or
out-of-order delivery. It is never reopened from a caller-supplied path and is
not a portable saved-evidence format. Records returned to a reader are detached.

The same backend-neutral terminal-content and exact-byte validators used by
strict eager capture operate on the indexed sequence. Lifecycle bookkeeping has
its own bounds. PostgreSQL also checks that the event document agrees with its
indexed identity/payload columns. Runtime authority proofs are included in the
versioned evidence seal rather than lost through JSON projection.

Recovery checks the original target/projector/input/output identity, deterministic
root ID, exact attempt and completion, root digest, and fresh lineage. Children
must originate after the attempt marker and finish within their parent boundary
and before workflow completion. Traversal orders siblings by origin sequence and
session ID. A second complete traversal repeats source snapshots and checks every
session seal and closure, followed by root revalidation. A saved expected digest
rejects subsequent mutation/deletion, changed authority flags, different attempts,
and replacement sessions.

This is per-session snapshot plus closure revalidation, not a single database
transaction across all sessions or an immutable historical database. Unrelated
sessions' appends cannot enter the selected scope. Changes to a selected session's
header invalidate its seal, even if they accompany an append beyond its selected
terminal prefix. Changes before the first seal cannot be compared with nonexistent
historical hashes; preserve original store backups for that proof. No API can
promise that an external writer never changes the source after revalidation.

Ordinary incremental terminal export accepts completed/failed sessions. Interrupted
or unsettled descendants, multi-attempt workflow journals and stores lacking the
capability fail explicitly. Choose the separate bounded eager API when applicable;
there is no silent fallback or relaxation of terminal checks.

## Cancellation and diagnostics

`IncrementalWorkflowCaptureError` retains a payload-free code, effective limits
and `IncrementalCaptureProgress`. Progress records the last known stage/session,
source records/bytes copied across passes, local spill reads, largest canonical
record, largest source page, and measured completed-spill size. Failure counters
are observations, not claims that unread records were processed. No post-expiry
source I/O is performed to manufacture a more precise diagnosis.

Cancellation stops the copy or local validator. A SQLite worker is settled before
its backing can be closed or removed. Database snapshots, cursors, backing files
and admission reservations are released on failure. Low-level callers must await
export before leaving `with EvidenceSpool(limits)` and close any reader they keep.

The live eager workflow path also records post-execution timeout phase and retained
capture counts. `case_timeout` remains the authoritative outer deadline;
`capture_diagnostic.code="deadline_exceeded"` does not relabel completed execution
as still running or infer a phase from elapsed time. Phases include terminal load,
result projection, child capture, probe capture, revalidation and assertions. Target
quiescence retains its existing independent cleanup contract.

## Reproducible measurements

Run `uv run python benchmarks/incremental_evidence.py --output /tmp/evidence.json`.
It seeds only synthetic records, then uses a fresh process for each measurement.
The matrix covers 10,000 / 100,000 / 250,000 records, large records at two trace
lengths, and 100 simultaneous arrivals under shared admission. It reports peak
RSS, import baseline, source evidence bytes, observed page sizes, spill allocation,
local read amplification, elapsed time and rejected admissions separately.

The conformance suite exercises both SQLite and PostgreSQL, including 250,000-record
traces, eager equivalence, exact boundaries, independent limits, snapshot mutation
and deletion, corrupted delivery, cancellation, slow/partial consumers and cleanup.
These synthetic results do not establish scores for retained private GAIA runs.

### Measured synthetic baseline (2026-09-06)

macOS, Python 3.14.2; fresh measurement processes, SQLite, four-record pages.
[Raw results](../benchmarks/incremental-evidence-results-2026-09-06.json) retain
budgets, page counters, local read amplification and timings. Other local tests
were running concurrently, so timings are observations rather than throughput
qualification. Sizes below are MiB. Evidence and spill columns are per capture;
RSS columns are the whole measurement process (including imports).

| Synthetic records | Payload bytes | Arrivals / admitted | Evidence | Spill | Peak RSS | RSS growth over import baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 128 | 1 / 1 | 3.85 | 6.81 | 223.52 | 4.78 |
| 100,000 | 128 | 1 / 1 | 38.70 | 68.21 | 222.88 | 4.44 |
| 250,000 | 128 | 1 / 1 | 97.07 | 170.77 | 216.47 | 4.52 |
| 64 | 800,000 | 1 / 1 | 48.85 | 49.05 | 272.00 | 52.64 |
| 256 | 800,000 | 1 / 1 | 195.38 | 196.16 | 283.27 | 64.88 |
| 1,000 | 128 | 100 / 4 | 0.39 | 0.69 | 227.95 | 9.50 |

The 100-arrival case rejected 96 immediately under a four-capture controller,
512 MiB aggregate buffer reservation budget and 4 GiB aggregate spill budget.
Each admitted capture reserved 64.67 MiB of buffer resources and 512 MiB of spill.
Every case observed at most four source records per page; large-record pages
reached 3.05 MiB. All reservations returned to zero and backing files were removed.

Small-record RSS growth stayed approximately flat while evidence and disk grew
with trace length. Large records increased RSS substantially, including allocator
retention; the two lengths do not establish a universal RSS plateau. The enforced
bounds concern serialized pages, individual records, root/lifecycle bookkeeping
and spill allocation. These results do not convert that reservation into an RSS
limit or establish 100 simultaneously active captures. Admission explicitly
limits active work; applications must size worker/process memory separately.
