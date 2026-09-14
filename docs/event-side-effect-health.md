# Operating durable event side effects

`CayuApp.get_persisted_event_side_effect_health()` is a read-only, typed metric
source. It works without the optional server. It never emits a runtime event,
forwards a budget update, or invokes sinks/watchers. A failed read raises an error;
it does not alter delivery state or stop independent recovery.

```python
from cayu import PersistedEventSideEffectQuery

health = await app.get_persisted_event_side_effect_health()
page = await app.query_persisted_event_side_effect_deliveries(
    PersistedEventSideEffectQuery(claimable_only=True, limit=100)
)
while page.next_cursor is not None:
    page = await app.query_persisted_event_side_effect_deliveries(
        PersistedEventSideEffectQuery(
            claimable_only=True, limit=100, cursor=page.next_cursor
        )
    )
```

The protected optional-server routes are `GET /api/event-side-effects/health`
and `GET /api/event-side-effects/deliveries`. They use the configured API auth.
The open `GET /api/health` remains only `{"ok": true}`. The deliveries route
accepts `status`, `claimable_only`, `outstanding_only`, `limit` (1–200), and
`cursor`. SDK pages allow 1–1,000 rows and a set of `statuses`. Invalid cursors
return HTTP 400; invalid parameter types return 422; unavailable stores return
503 with `event_side_effect_store_unavailable`. Standalone routers have no owned
recovery task and return a null `recovery_loop`.

## Reading the snapshot

All ages and lease classifications use the same timezone-aware UTC `observed_at`.
Counts are exact for the store, including delivered rows. SQLite and PostgreSQL
represent every worker sharing that database; the in-memory store represents only
its process. These are point-in-time observations; a claim or acknowledgement can
change state immediately after a read.

| Field | Meaning |
| --- | --- |
| `pending` | Committed but never claimed (or explicitly deferred) |
| `leased_live` | A worker owns a claim expiring after `observed_at` |
| `leased_expired` | Persisted as leased, but the expiry is at or before observation |
| `failed_retryable` | Failed, not dead-lettered; includes deferred retries |
| `failed_deferred` | Failed with a durable retry deadline still in the future |
| `delivered` | All configured side effects acknowledged |
| `dead_lettered` | Automatic delivery stopped after the attempt budget |
| `claimable_total` | Pending + failed retryable − failed deferred + expired leases |
| `outstanding_total` | All rows except delivered, including dead letters |
| `repeatedly_failing` | Retryable failures with more than one attempt |
| `final_attempt_boundary` | Nonterminal work whose next claim would be the final attempt, or a live lease already executing that attempt |
| `max_outstanding_attempts` | Largest persisted attempt count outside delivered rows |

`max_automatic_attempts` comes from the same three-attempt constant as recovery.
A retry deadline matters: a failed record can be retryable without being claimable
**now**. Health does not change the existing retry-spacing policy.

`oldest_claimable_at` is the earliest current eligibility timestamp: update time
for pending, the later of update time and retry deadline for failed, lease expiry
for expired claims. `oldest_pending_at`, `oldest_failed_at`, and
`oldest_dead_letter_at` refer to update times. Each has a corresponding
`*_age_seconds`; absent work has null timestamps/ages, future timestamps clamp to
zero. `earliest_live_lease_expires_at` identifies the next lease expiry.

The server's separate `recovery_loop` is explicitly
`process_local_reset_on_restart`. It records configured/running/stopped state,
start time, last sweep start/completion/success, delivered count, monotonic duration,
interval, batch limit, whether the last successful batch was full, consecutive
failures, and bounded error/time. Attempts, successes, failures, and delivered-row
counters reset on restart. `saturated_batches` counts successful full batches and
survives the final empty batch and subsequent idle sweeps. The `last_*` fields
describe only the most recent batch, which is normally empty after a drain.
Use increases in `saturated_batches` between polls to observe repeated saturation.
Cancellation is not a sweep failure. A subsequent
success clears consecutive failures but retains the last error timestamp as history.
Different workers may report different loop state against the same durable totals.

## Inspection and redaction

Pages default to outstanding work. Ordering is stable by `(session_id, event_id)`;
sequence is returned for correlation and is not used as a global ordering key.
The opaque versioned cursor binds the filters, supports more than 1,000 rows,
and never reserves ownership. Keep filters unchanged when following it. Concurrent
acknowledgements can remove rows from later outstanding pages. Inserts before the
cursor appear on the next scan; pagination is not a frozen database snapshot.

Inspection returns defensive projections: identities, sequence, status,
claimability, attempts, lease/retry timestamps, update time, and `last_error`.
No source event, claim credential, private checkpoint, or sink configuration is
loaded into these responses. Errors cross an explicit sanitization boundary:
this surface replaces arbitrary application exception text with a fixed bounded,
single-line summary. Detailed diagnostics remain in access-controlled application
logs. Applications must redact sink errors at their boundary because raw sink
exceptions can contain deployment data, including secrets unknown to Runtime.

## Metrics and starting alerts

Poll the SDK or protected endpoint and export low-cardinality gauges for the
lifecycle counts, repeated failures, totals, ages, loop last-success age,
consecutive failures, and saturation. The loop counters above may be exported as
process counters. No Prometheus or OpenTelemetry dependency is required.
Never use session/event/claim IDs, application sink names, or exception strings
as metric labels; put correlation IDs in protected drill-down instead.

Suggested starting policies, owned by the deployment:

- `dead_lettered > 0`: page/high-priority alert; automatic delivery has stopped.
- Oldest claimable age above at least two recovery intervals: backlog warning.
- `repeatedly_failing > 0`: warning before retry exhaustion.
- Loop last-success age above at least two intervals, or consecutive failures:
  process-level recovery warning. A loop with no success yet also needs attention.
- Repeated increases in `saturated_batches`: capacity warning. A full batch alone
  is not failure; treat a counter decrease as a process restart.
- Live leases alone are normal; expired leases belong to actionable backlog.

Do not route these signals through RuntimeEventWriter, event sinks, budget
forwarding, or an EventWatcher. Such a path can fail with the dependency being
monitored or recursively create handoffs. The canonical `runtime.sink.failed`
exclusion remains unchanged. Framework recovery logging uses fixed messages and
suppresses repeated identical periodic failures until a successful sweep.

## Recovery runbook

1. Separate shared durable backlog from this worker's recovery-loop status.
2. If claimable work grows, check loop success and sink/budget dependencies.
3. Inspect bounded records by identity; use authorized durable event replay for
   context instead of exporting payloads through health.
4. Fix the downstream dependency or idempotency conflict before retrying.
5. SDK-only workers must schedule bounded
   `await app.recover_persisted_event_side_effects(limit=1000)` sweeps. The optional
   server owns startup draining and periodic recovery. Delivery remains at least
   once: sinks deduplicate by `(session_id, event_id)`; built-in budget stores do so.
6. Do not rerun providers, tools, compactions, or originating session mutations to
   repair committed side effects. Notification/delivery acknowledgement grants
   no execution authority.
7. For dead letters, preserve evidence and reconcile the external effect before
   any application-owned manual repair. This API does not requeue dead letters.
   Do not delete a session simply to clear an alert.

Non-delivered handoffs, including dead letters, remain protected from pruning.
Explicit session deletion removes the session, events, and handoffs together.

## Storage and rollout

Revision 86 adds payload-free covering health and outstanding-page indexes and
retains revision 85's compatibility floor. Run the normal explicit schema migration
before deployment; validation mode does not run DDL. Reads also remain correct
against a compatible revision-85 database, with less efficient aggregate access.
SQLite uses one storage-side aggregate statement; PostgreSQL does the same with
one database-clock observation. Neither deserializes events or loads delivery rows
into Python to compute aggregates. Exact global counts (especially delivered)
require scanning delivery metadata/index entries; they are O(number of retained
handoffs), not constant time. Poll at a deployment-appropriate interval. Pagination
uses indexed identity ordering and bounded results. Index construction consumes
storage and may temporarily contend with writers during the explicit migration;
schedule it as part of normal database deployment planning.
