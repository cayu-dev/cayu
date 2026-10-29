# Storage retention

Cayu keeps durable state until something removes it. Retention is the explicit
way to remove old sessions, eval runs, artifacts and leftover workspaces while
keeping everything that other runtime state still references. It is off by
default. Nothing is compacted, deleted or disposed unless an operator runs
`cayu storage prune --apply`, application code applies a policy, or the
application starts the optional retention worker.

Retention is built from parts that work on their own:

| Part | Where | What it prunes |
| --- | --- | --- |
| `SessionStore.apply_retention_policy` | `SQLiteSessionStore`, `PostgresSessionStore` | Session events (compact) or whole sessions (delete) |
| `EvalStore.apply_retention_policy` | `SQLiteEvalStore`, `PostgresEvalStore` | Terminal eval runs with their results and trial checkpoints |
| `apply_artifact_retention_policy` | `cayu.artifacts` | Artifacts in one artifact store |
| `apply_workspace_retention` | `cayu.runtime` | Leftover runner and coding workspaces of terminal sessions |
| `StorageRetentionPolicy`, `CayuApp.apply_storage_retention` | `cayu.storage`, `CayuApp` | All of the above with references collected from every configured store |
| `run_storage_retention_worker` | `cayu.runtime` | The same policy on an interval |

`InMemorySessionStore` and `InMemoryEvalStore` report
`supports_storage_retention = False` and raise `NotImplementedError`.

## Policies

Every policy (`RetentionPolicy` in `cayu.storage`) has the same base fields:

| Field | Meaning |
| --- | --- |
| `older_than` | Required. Select items older than this duration. |
| `mode` | `compact` or `delete`. Only sessions can be compacted; other targets always delete. |
| `max_items` | Act on at most this many items in one run. Default 1,000, maximum 10,000. |
| `max_bytes` | Optional. Stop after the selected items reach this many stored bytes. |
| `dry_run` | Defaults to `True`. A dry run changes nothing. |

The per-target policies add:

| Policy | Extra fields |
| --- | --- |
| `SessionRetentionPolicy` | `statuses` (`completed` and `failed` by default; only terminal statuses are accepted), `compact_tool_output_min_bytes` (default 16 KiB) and `compaction_level` (`events` by default, or `transcript`; see [Sessions](#sessions)). Age is the session's last update. |
| `EvalRetentionPolicy` | `statuses` (`completed`, `failed` and `cancelled` by default; queued, running and cancelling runs are never selected). Age is the run's `finished_at`. |
| `ArtifactRetentionPolicy` | `scopes` (`session` and `environment`), `min_size_bytes`, and `target_total_bytes`, which stops a run once the store's total size is at or below that size. Age is the artifact's `created_at`. |
| `WorkspaceRetentionPolicy` | `statuses` (terminal session statuses). Age is the session's last update. |

Items are taken oldest first. A report (`RetentionReport`) lists the items a
dry run would act on or an apply acted on, with counts and logical bytes, the
protected items with the protection that kept each, and the number deferred by
the budget.

```python
from datetime import timedelta

from cayu.storage import RetentionMode, SessionRetentionPolicy

plan = await session_store.apply_retention_policy(
    SessionRetentionPolicy(older_than=timedelta(days=30), mode=RetentionMode.COMPACT)
)
report = await session_store.apply_retention_policy(
    SessionRetentionPolicy(
        older_than=timedelta(days=30), mode=RetentionMode.COMPACT, dry_run=False
    )
)
```

## Sessions

`compact` keeps the session. It removes `model.text.delta` and
`model.thinking.delta` events, and replaces the body of a stored
`tool.call.completed` or `tool.call.failed` result whose `content` is at least
`compact_tool_output_min_bytes` with a marker naming its size and SHA-256. It
keeps the session record and status, the checkpoint, terminal events,
`model.completed` events and their usage, structured tool results, artifact
references and every other event. Usage and cost accounting read the same
totals before and after. Compaction never touches an event that an
interaction pointer or an undelivered event side effect still references, and
running it again finds nothing to do. The removed payloads cannot be
recovered: replaying a compacted session's token stream, or reading the full
tool output from its stored event, is no longer possible.

`compaction_level` chooses how far `compact` goes:

| Level | What it also changes |
| --- | --- |
| `events` (default) | Nothing else. The transcript, including the tool results the model saw, is unchanged. |
| `transcript` | Replaces each `tool_result` body of at least `compact_tool_output_min_bytes` in the transcript's tool messages with the same marker. The tool name, call id, `structured` result, artifacts and error flag stay. |

From storage revision 134, a tool result of at least 1 KiB that the runtime
publishes is stored once, in its `tool.call.completed` or `tool.call.failed`
event. The transcript's tool message keeps the tool name, call id, `structured`
result, artifacts and error flag, plus a reference that names the event and
records the result's size and SHA-256; every transcript read restores the body
from the event and fails if it no longer matches. The `events` level leaves
such an event alone, because the transcript reads its result from it, and the
`transcript` level replaces the body in both places with the same marker.
Sessions stored before revision 134, and smaller results, keep a copy in each
place: there the `events` level halves a large tool output's footprint and
`transcript` removes the rest. The transcript level rewrites only tool
messages that a runtime-publication receipt covers, which is every tool result
the runtime published; messages the caller supplied, such as initial input
that a session's input contract digests, are left alone. After compaction at
this level the console and dashboard show the marker where the output was, a
continued session's model sees the marker instead of the original output, and
an eval scenario captured from the session afterwards sees the marker too
(sessions that an eval already references are protected). Report counts add
`transcript_tool_results_compacted`.

Runtime-publication receipts bind the exact transcript segment and events they
published by SHA-256, and recovery, replay, the inbox, producer outputs and
collaboration outputs re-check that binding whenever they load a receipt.
Before compaction rewrites a body that a receipt binds, it verifies the
receipt against the stored material; if the receipt no longer verifies, that
receipt's material is left untouched. After the rewrite it stores a
retention attestation (a runtime-owned session operation under
`__cayu_retention_compaction_v1__:`) with the receipt's original digests and
the digests of the compacted material, in the same transaction. A receipt
load accepts compacted material only when an attestation bound to that exact
receipt names its digests; any other change still fails. Events a receipt
appends or references are never deleted. Databases compacted by an earlier
release, which rewrote tool events without an attestation, keep failing the
check for those receipts.

Transcript retention by message count remains the separate
`compact_transcript(keep_last=...)` operation.

### What the session tables hold

| Table | What it holds | What shrinks it |
| --- | --- | --- |
| `cayu_events` | Every event of every session. A tool's terminal event carries its result body. | `compact` removes text and thinking deltas and replaces large tool-result bodies (see above); `delete`. |
| `cayu_transcript_messages` | The messages the model sees. A tool result stored once in its event is a reference here. | `compact` at the `transcript` level replaces large tool-result bodies; `delete`. |
| `cayu_session_operations` | Exactly-once records, described below. | `delete` only. |
| `cayu_checkpoints` | One row per session: the latest runtime checkpoint. | `delete`. |
| `cayu_sessions`, `cayu_session_labels` | The session record, metadata and labels. | `delete`. |

### What `cayu_session_operations` holds

Session operations are durable, keyed records that make the runtime's writes
exactly-once. They are often the largest table after transcripts and events,
and `compact` keeps all of them:

| Key | Record | Why it stays |
| --- | --- | --- |
| `__cayu_runtime_publication_v1__:` | Receipt of one atomic publication (model step, tool round, approval, user input): request, transcript and event digests | A retried or late publication replays against it instead of publishing twice; recovery, the inbox and output readers verify it. |
| `__cayu_model_completion_stage_v1__:` | Preparation, dispatch, completed response and winner of one model step | Crash recovery finishes or replays the step from these instead of calling the provider again. |
| `__cayu_interaction_transition_v1__:`, `__cayu_invocation_terminal_event_v1__:` | The session and event at an interaction transition or an invocation's terminal event | A retried transition or terminal publication must match the original exactly. |
| `__cayu_child_session_notification_v1__:`, `__cayu_zero_work_interruption_v1__` | Child-session notification delivery and zero-work interruption receipts | Delivery and interruption stay exactly-once across restarts. |
| `tool-effect:`, `producer-output:`, `session-continuation:`, `session-export:` | Act-once tool effects, producer outputs, continuations and session exports | Dropping a tool-effect record lets a retried act-once tool run its effect again; the others back public reads. |
| `__cayu_retention_compaction_v1__:` | Retention attestations | Keep compacted receipts verifiable. |

No record type is dropped on its own: each one is the evidence a later retry,
recovery or read compares against, and Cayu cannot prove that none will come
(a completed session can be resumed, and a stale worker can retry a
publication after the session finished). They are removed with their session
by `delete`, which goes through the store's deletion guards.

`delete` removes the whole session through the store's own `delete_session`
transaction and guards.

A parent/fork lineage is one retention unit: it is pruned whole or not at
all, and children are removed before their parents, so a budget that stops
part-way never leaves a child whose parent was deleted.

The session store keeps a session, and reports why, when any of these hold:

| Protection | Reference that keeps the session |
| --- | --- |
| `live_task` | A task that is not completed, failed, cancelled or dependency-skipped names the session. |
| `execution_lease` | An unreleased, unexpired execution lease is held on the session's current run. |
| `pending_action` | The checkpoint has a pending approval, user input or delegated action, or its pending-action metadata has not been backfilled. |
| `pending_clarification` | A collaboration participant bound to the session has an open request or clarification question, or an unsettled clarification delivery or service. |
| `checkpoint_dependency` | Another session holds an active context-view selection on this session's transcript and checkpoint. |
| `snapshot_pin` | An unreleased agent-snapshot pin or protection mentions the session. |
| `eval_reference` | A stored eval run, result, captured result or trial checkpoint names the session. |
| `knowledge_evidence` | Knowledge evidence that is not detached points at the session or one of its events. |
| `product_operation` | A pending product operation is bound to the session. |
| `event_delivery_backlog` | An event watcher has not consumed the session's events, a watcher dead letter is unresolved, or a persisted event side effect is not delivered. |
| `session_export` | Compaction only: the session has session-export records whose commitments cover its stored events. |
| `closure_in_progress` | A session or task closure owns the session. |
| `erasure_guard` | The store's own deletion guard refuses the session, for example an incomplete terminal publication, an active model-completion stage, an active recovery claim or a pending budget settlement. Compaction applies the same guard. |
| `lineage` | Another session in the same parent/fork lineage must be kept. |
| `caller_protected` | The caller named the session in `protected_session_ids`. |
| `changed` | The session changed between planning and its write transaction. |

### Batches and concurrency

Planning reads the session graph and the large reference tables from a
non-blocking snapshot (a SQLite reader connection, or an ordinary Postgres
read), then evaluates protections in bounded batches, releasing the store
between them. An apply then changes one lineage per write transaction. Inside
that transaction it locks the lineage (Postgres takes the same advisory and
row locks as `delete_session`), re-reads it, re-checks every protection and
writes the lineage's audit entries before it commits. Reference documents are
read again for every apply batch; row counts and byte lengths are not used as
mutation revisions. Postgres also holds share locks on the reference tables
through the batch, so a publisher cannot change them between validation and
deletion. The store's write lock is
released between lineages, so other writers proceed during a long prune, and a
session that becomes protected after planning is kept and reported. Pass
`progress=` to observe each planned and committed batch.

## Evals

An eval item is one terminal run. Deleting it removes the run, its result, its
fresh result record and its trial checkpoints in one transaction. Captured
result records, corpora, suites, cases, scenarios, authored suites, baselines,
baseline history and judge calibrations are never removed. A run is kept while:

| Protection | Reference that keeps the run |
| --- | --- |
| `baseline` | An eval baseline or baseline history selects the run's result. |
| `campaign_evidence` | The run retains trial checkpoints for a benchmark campaign; resume, retry, rescore and inspection read them. |
| `eval_reference` | Another run's retry lineage names the run. |
| `snapshot_pin` | An agent-snapshot pin names the run, its result revision or one of its trial result digests. |
| `caller_protected` | `protected_ids` names the run id or result revision. |
| `changed` | The run changed after planning. |

Each run is re-read, re-checked and deleted in its own transaction with its
audit entry; the eval store's write lock is released between runs.

## Artifacts

Artifact stores keep no reference index, so retention decides what is still
needed from the stores that can name an artifact:

| Protection | Reference that keeps the artifact |
| --- | --- |
| `pinned` | A durable pin retains it: public pins, resource-retention pins and workspace-checkpoint pins. `LocalArtifactStore` and `S3ArtifactStore` report pins through `has_retention_pins`; a store that cannot report them keeps every artifact, and `delete` enforces pins again. |
| `session_reference` | Its owning session still exists or a session-closure claim fences it, or a session transcript or session operation names it. Session-scoped artifacts are therefore removed only after their session is deleted. |
| `eval_reference` | An eval run, result or trial checkpoint names it. |
| `knowledge_evidence` | Knowledge evidence names it as a source. |
| `snapshot_pin` | An agent-snapshot pin names it. |
| `caller_protected` | `protected_artifact_ids` names it. |
| `erasure_guard` | A configured reference source cannot fence publication during deletion, or no reference database was supplied. |

Each artifact is rechecked against the configured reference databases before
apply. SQLite holds `BEGIN IMMEDIATE`; Postgres holds share locks on the
reference tables. The locks cover reference validation and the artifact's
actual deletion, and are released between artifacts. Stores sharing a database
use one transaction. Supported reference sources are SQLite session, eval and
snapshot stores and Postgres session and eval stores, including separate
databases. Unknown or in-memory reference sources keep artifacts rather than
allow deletion based on an unguarded list.

Direct callers of `apply_artifact_retention_policy` pass `session_store`,
`eval_store` and `snapshot_stores` for the sources that can publish references.
The application-level API supplies these automatically. Explicit `references`
and protected ids add exclusions; they do not replace a live reference source.
Cancellation waits for the active artifact batch and its audit entry to settle.

An artifact store has no transaction to share with an audit record, so an
artifact apply writes its audit through a `RetentionAuditSink`, normally the
session store, one entry after each deletion.

## Workspaces

Cayu keeps no global inventory of the resources its environment factories
allocate; each session's checkpoint records its own allocations. Workspace
retention therefore selects terminal sessions last updated before the cutoff
that still own unsettled environment allocations, pending disposals or a
pending completion finalization, and disposes them through the runtime's
incomplete-session recovery. Recovery reaps each allocation through its
environment factory's own reaping fence, so Docker coding containers and other
factory resources are removed by the code that created them. A session is kept
while:

| Protection | Reason |
| --- | --- |
| session protections | The session store reports a live task, execution lease, pending action, snapshot pin, closure or other protection for it. |
| `workspace_checkpoint` | A workspace checkpoint is mutating or checkpointing. |
| `branch_retention` | The environment's workspace keeps durable branch state (`WorkspaceBranchRetentionStrength.DURABLE`). |
| `erasure_guard` | The session's environment is no longer registered, its allocation records are inconsistent, or recovery refused the cleanup. |
| `caller_protected` | `protected_session_ids` names it. |

Running, interrupted and other non-terminal sessions are never selected.
Resources without a durable owner record (for example a container whose
session row was removed by hand) are not touched.

## Application-level policy

`StorageRetentionPolicy` combines the per-target policies. Only the targets
given are pruned, and its `dry_run` overrides the sub-policies' own flags.
`StorageRetentionPolicy.for_targets(...)` builds one policy with the same age
and budget for several targets.

```python
from datetime import timedelta

from cayu.storage import StorageRetentionPolicy, StorageRetentionTarget

policy = StorageRetentionPolicy.for_targets(
    [StorageRetentionTarget.SESSIONS, StorageRetentionTarget.EVALS],
    older_than=timedelta(days=30),
    dry_run=False,
)
report = await app.apply_storage_retention(
    policy, eval_store=eval_store, snapshot_stores=[snapshot_store]
)
```

`CayuApp.apply_storage_retention` uses the application's session store, task
store and every registered artifact store; pass the eval store and agent
snapshot stores the application uses. Before pruning it collects references
from all of them, so references that live in another database still protect
what they name:

- live tasks in the task store protect their sessions;
- eval runs and results protect the sessions and artifacts they name;
- held agent snapshots protect the sessions, eval runs and artifacts they name;
- session transcripts, session operations and knowledge evidence protect the
  artifacts they name.

Targets run in dependency order: workspaces (which need the session's
allocation records), evals (which free the sessions they name), sessions, then
artifacts (which belong to sessions). A target the application cannot serve is
listed in `skipped`; a target that fails is listed in `errors` and the others
still run. `apply_storage_retention(policy, session_store=..., ...)` in
`cayu.runtime` takes the stores explicitly for applications assembled
differently. Use `protected_session_ids`, `protected_eval_ids` and
`protected_artifact_ids` for references outside the application.

## Background worker

`run_storage_retention_worker(app, stop, policy=..., interval_seconds=...)`
applies a policy every interval until `stop` is set. It follows the project
worker contract and is never started implicitly. A failed pass is logged and
retried on the next interval.

```python
from datetime import timedelta

from cayu import CayuApp
from cayu.runtime import run_storage_retention_worker
from cayu.storage import StorageRetentionPolicy, StorageRetentionTarget

POLICY = StorageRetentionPolicy.for_targets(
    [StorageRetentionTarget.SESSIONS, StorageRetentionTarget.ARTIFACTS],
    older_than=timedelta(days=30),
    dry_run=False,
)


async def run_retention(app: CayuApp, stop) -> None:
    await run_storage_retention_worker(app, stop, policy=POLICY, interval_seconds=3600)
```

```toml
[tool.cayu.workers]
retention = "workers:run_retention"
```

Run it with `cayu worker retention`.

## Growth warning

Because retention is opt-in, a service can fill its disk without noticing.
`SQLiteSessionStore.inspect_storage_growth()` and
`PostgresSessionStore.inspect_storage_growth()` return a read-only
`StorageGrowthReport`: the database's allocated bytes (and, for SQLite, the
free pages only `VACUUM` returns), the session and terminal-session counts, the
last update of the oldest terminal session, and the start of the newest
session retention apply recorded in the audit tables.
`storage_growth_warning(report)` turns it into an operator hint when the store
holds at least 1 GiB or a terminal session older than 30 days, and no session
retention apply is recorded within the last 7 days; the thresholds are
keyword arguments.

`cayu serve` runs this check once in the background at startup and logs the
hint as a warning. `report_storage_growth(session_store)` in `cayu.runtime`
does the same for workers and other entry points. Nothing is pruned: the
warning only says that retention exists.

## Audit records

Every apply writes a durable audit record, including an apply that changes
nothing. The run row records the store kind, the policy and, when the run
finishes, a summary with totals and protection counts. Each changed item gets
an entry with its counts and bytes. Session and eval entries are written in the
same transaction as the change, so an interrupted run still records exactly
what it removed and stays in the `started` state. Artifact and workspace
entries are written right after each change.

```python
records = await session_store.list_retention_audits(limit=20)
for_session = await session_store.list_retention_audits(item_id="session-id")
artifact_runs = await session_store.list_retention_audits(store_kind="artifacts")
record = await session_store.load_retention_audit(records[0].audit_id)
```

Eval stores list their own runs the same way. The audit tables arrive in
storage revision 118, an additive migration. Dry runs work on older databases;
an apply asks you to run `cayu storage migrate` first.

Counts and bytes are logical stored sizes (the JSON text or content Cayu
stored), not file sizes.

## Reclaiming disk space

Pruning frees space inside the database; it does not shrink the files.

**SQLite** reuses freed pages for new writes but keeps the file at its largest
size. `cayu storage prune --apply --vacuum` runs `VACUUM` after the apply and
reports the file size before and after (`vacuum` in JSON). `VACUUM` rewrites
the whole database: it copies the remaining data into a temporary database in
SQLite's temporary directory (`SQLITE_TMPDIR`, `TMPDIR`, then `/var/tmp` or
`/tmp`) and writes it back through the write-ahead log next to the database.
The command refuses to start unless each place has free space for the data,
or twice that when both are on one file system, and it holds the write lock
while it runs, so run it in a quiet period. `inspect_storage_growth()`
reports the reclaimable free pages as `reclaimable_bytes`. Cayu does not
enable `auto_vacuum`; `PRAGMA incremental_vacuum` only works on a database
created with `auto_vacuum = INCREMENTAL`.

**PostgreSQL** autovacuum marks the rows that pruning deleted or rewrote as
reusable, so the tables stop growing without any action. Cayu does not run
`VACUUM FULL`: it rewrites each table under an `ACCESS EXCLUSIVE` lock that
blocks every reader and writer. To return space to the operating system after
a large prune, run `VACUUM (FULL, ANALYZE)` on the largest tables
(`cayu_transcript_messages`, `cayu_events`, `cayu_session_operations`) in a
maintenance window, or use an online tool such as `pg_repack`. A plain
`VACUUM (ANALYZE)` right after a large prune is safe at any time and refreshes
planner statistics.

**Migration backups** are full copies; see
[Session-store targets](session-store-targets.md#backup-authority) for
`--backup-dir` and `--keep-backups`.

## CLI

```bash
cayu storage prune --dry-run --older-than 30d --json
cayu storage prune --apply --older-than 30d --mode compact
cayu storage prune --apply --older-than 7d --mode compact --compaction-level transcript
cayu storage prune --apply --older-than 90d --target sessions --target evals --mode delete
cayu storage prune --dry-run --older-than 30d --target all --json
```

`--dry-run` or `--apply` is required. `--older-than` takes `90s`, `30m`, `12h`,
`30d` or `2w`. `--target` is `sessions` (the default), `evals`, `artifacts`,
`workspaces` or `all`, and can be repeated.

- `sessions` and `evals` resolve the store like the other storage commands
  (`--sqlite`, `--postgres`, `CAYU_DATABASE_URL`, then
  `[tool.cayu.session_store]`); the eval store is the one that shares the
  session store's database.
- `artifacts`, `workspaces` and `all` build the project's application from
  `[tool.cayu] factory` and use its configured stores; `--sqlite` and
  `--postgres` are rejected with them.

Other options: `--mode compact|delete` (sessions),
`--compaction-level events|transcript` (sessions with `--mode compact`),
`--status completed|failed` (sessions and workspaces), `--max-items`,
`--max-bytes`, `--compact-tool-output-min-bytes`, `--vacuum` (SQLite with
`--apply`), `--artifact-min-bytes`,
`--artifact-target-total-bytes`, and the repeatable `--protect-session`,
`--protect-eval` and `--protect-artifact`. JSON output has `schema_version`,
`backend`, `target_source`, `dry_run`, per-target `totals`, one entry in
`reports` per pruned store, `skipped` and `errors`; `--table` prints a summary.
The command exits `1` when a target fails.
