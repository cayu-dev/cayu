# Building an application UI over Cayu sessions

This guide is for a custom browser UI (a "bring your own UI" app) that shows
Cayu sessions: a list of runs, a live timeline, a progress panel, token usage.
Most of the work in such a UI is watching sessions change, and the easy way to
write that (a short fixed timer that never stops) is what makes these apps
expensive. One idle tab written that way made about 1,600 requests an hour and
used a third of a 0.5 vCPU web task.

Follow the rules below. The first one removes most of the others.

## Follow live sessions with client.js

`mount_cayu(..., path="/cayu")` serves a dependency-free ES module at
`/cayu/client.js`, with TypeScript declarations at `/cayu/client.d.ts`. It is
versioned with the server: `GET /cayu/api/contract` advertises it under
`client` (`module_url`, `types_url`, `version`, and `guide_topic: "app-ui"`).
`create_server` serves it next to its API when the API path ends in `/api`
(`/client.js` for the default `/api`). Import it from the served URL; there is
no build step and nothing to install.

```js
import { connect } from "/cayu/client.js"

const cayu = await connect() // reads /api/contract and checks contract_version
```

The module is served behind the same access dependency as the API, and requests
use same-origin credentials, so cookie sessions work unchanged. For header-based
auth, pass `connect({ headers: { Authorization: ... } })`, and load the module
from a URL your app serves if the import itself cannot carry the header (the
file ships in the package as `cayu/server/browser_client/client.js`).

It provides:

- `followSession(sessionId, { afterSequence, excludeEventTypes, onEvent, onStatus, onError, signal })`
  returns `{ done, stop, lastSequence, status, transport, lastError }`.
- `watchSessions(filter, { onChange, onError, onStop, intervalMs, signal })`
  returns a function that stops the watch.
- `getSessionUsage(sessionId)` reads `GET /api/sessions/{id}/usage`.

What the client guarantees, so your code does not have to:

- **Transport.** `followSession` uses the session follow stream when the
  contract advertises `sse.session_follow`, and otherwise polls
  `GET /api/sessions/{id}/events?after_sequence=...`, backing off while nothing
  new arrives. A stream that fails repeatedly falls back to polling.
- **Order.** `onEvent` gets each event once, in `sequence` order, including
  across reconnects.
- **Hidden tabs.** While `document.hidden` is true the client sends no requests
  and closes any open stream. When the tab is visible again it resumes after
  the last sequence it delivered, with no gap and no full refetch.
- **Session end.** `done` resolves once the session is `completed`, `failed`,
  or `interrupted` and its remaining events are delivered. After that the
  client sends nothing for that session.
- **Overview cadence.** `watchSessions` refreshes at most every 15 seconds
  (lower `intervalMs` values are raised to 15 s), doubles the interval while
  the list is unchanged, and stops (`onStop("idle")`) once no listed session
  is `pending`, `running`, or `interrupting`.
- **Errors.** Transient failures (network, 408, 429, 5xx) are retried with
  exponential backoff and jitter, and never sooner than `Retry-After`. Other
  4xx responses (401, 403, 404, 422, ...) and responses that do not match the
  contract stop the operation and call `onError` with a `CayuClientError`
  whose `kind` says why (`auth`, `not_found`, `contract`, `http`). A stopped
  `followSession` also rejects `done` with that error.

`interrupted` includes pauses for tool approval or user input. After the UI
answers one, call `followSession` again with
`afterSequence: follow.lastSequence`. Likewise, call `watchSessions` again after
starting new work if the previous watch stopped as idle.

Do not hand-write polling loops for session events or session lists. If
`client.js` does not fit, use the same endpoints with the rules below.

## If you must poll

- Poll only while the tab is visible (`document.visibilityState === "visible"`),
  and pause the timer on `visibilitychange`.
- Poll only while something is active. Stop when every session you show is
  terminal, and start again when the user starts new work.
- Refresh overview data (lists, counts, dashboards) no more often than every
  15 seconds.
- Back off while responses do not change, for example by doubling the interval
  up to a ceiling of a minute or two.
- Stop following a session once it is terminal and you have its terminal event.
- Read events incrementally with `after_sequence`, advancing to
  `scan_through_sequence` when the page has one. Never re-read a session's whole
  history to find what changed.
- On errors, back off exponentially with jitter, honor `Retry-After`, and stop
  on 401, 403, and 404 instead of retrying them.

## Split overview from detail

Give each screen its own bounded read. The overview reads
`GET /api/sessions` with a status filter and a `limit`; a detail view reads one
session, its events after a cursor, and its usage. Do not build one
"everything" endpoint (for example an app-level `/api/state` that rebuilds the
whole workspace) that loads every job, its pending actions, and its full event
history on every call. Its
cost grows with every run the app has ever made, and every open tab pays it on
every refresh.

## Ask Cayu whether a session is executing

Use `await app.inspect_session_execution(session_id)` instead of a local worker
dictionary or an age threshold on `last_activity_at`. A console answer may resume
the session in the Cayu server process while an application worker has no local
task for it. A model stream or tool may also run silently for minutes.

```python
from cayu import SessionExecutionConfig

# Optional when constructing CayuApp; these are the defaults.
execution_config = SessionExecutionConfig(
    heartbeat_interval_seconds=15,
    lease_seconds=60,
    owner_label="review-worker",  # optional application label; never include secrets
)

execution = await app.inspect_session_execution(session_id)
if execution.state == "executing":
    label = "Running here" if execution.local_owner else "Running in another worker"
elif execution.state == "owner_lost":
    label = "Execution owner lost; inspect recovery"
else:
    label = execution.state
```

Pass `session_execution=execution_config` to `CayuApp` to change the interval or
label. The lease must cover at least three heartbeat intervals. The default
heartbeat writes one bounded owner row every 15 seconds per active session epoch,
independently of model deltas and tool progress. Claim and release each add one
small write. It does not update `last_activity_at`, rewrite the checkpoint, or
load the transcript. Coarse model/tool/publishing boundaries update
`last_progress_at` and `last_progress_kind` on the next heartbeat.

`SessionExecutionState` includes an opaque process-instance `owner_id`,
`owner_kind`, `run_epoch`, a hashed `operation_id` reference, `claimed_at`,
`heartbeat_at`, and `lease_expires_at`. Owner kinds cover application runners,
Cayu server streams, task workers, recovery, and foreground child delivery.
No hostname, PID, prompt, tool argument, or provider payload is included.
SQLite and PostgreSQL make the same projection visible to other processes;
in-memory stores work only inside their process.

`GET /api/sessions/{id}/state` exposes it under `execution`. Its ETag changes
on heartbeat renewal and when the lease expires. Polls themselves never renew
the lease or change the epoch:

```bash
curl --fail --silent --show-error "$CAYU_URL/api/sessions/$SESSION_ID/state"
curl --silent --show-error -H 'If-None-Match: W/"previous-etag"' \
  "$CAYU_URL/api/sessions/$SESSION_ID/state"
```

`waiting` means a retained human input, approval, or foreground child wait; it exposes no live owner.
`idle` means work has not started, and `terminal` means it has settled.
`owner_lost` means nonterminal work has no living owner for the current epoch,
including a lease that expired after a process died. PostgreSQL uses the database
clock; SQLite and memory use the host clock, so shared SQLite deployments require
synchronized host clocks. `unknown` covers
older writers and custom stores that cannot attest execution presence. Treat it
as uncertain rather than declaring a run dead. Maintained durable stores need
the additive schema migration 113 before a new binary starts writing leases.

This projection grants no execution authority. A live owner blocks stale-run
fencing and appears as `active_execution_owner` in recovery planning. The
inactivity-gated recovery calls honor it too: a `recover_incomplete_session(...)`
or `recover_incomplete_sessions(...)` request that sets `inactive_for_seconds`
and runs before expiry returns `skipped_execution_owner` with
`execution_lease_expires_at`, and batch recovery includes that result too. A
request that leaves `inactive_for_seconds` at its default `None` is an explicit
recover-now call: it does not consult the execution lease and is not skipped.
Only send it when you know the owner is gone.

Your own startup recovery loop must retry skipped entries after the reported
time rather than run a single sweep and discard them. The Cayu server's
`startup_recovery_statuses` sweep does this already: sessions it skipped for a
live lease, typically left by a process that crashed just before the restart,
are re-planned in the background after the lease expires, with the same
`recovery_inactive_after_seconds` threshold. It retries up to three times, then
logs the sessions whose owners are still alive and leaves them to explicit
recovery. A process crash can leave up to 60 seconds of the default lease.
Explicit operator interruption retains its existing cancellation contract;
execution presence does not authorize forced takeover.

Transient renewal errors do not release the row, and the heartbeat keeps retrying
for as long as the run is active, even after a store outage outlasts the lease.
Once the store answers again, the expired lease cannot be renewed, so the same
process reclaims observation for its run. That reclaim succeeds only while the
transaction still matches that process's run epoch; a successor fence wins and
the stale heartbeat stops.
Retiring a local recovery owner stops its heartbeat even when the durable
invocation must remain for a later recovery attempt. A live but hung operation
continues to heartbeat: lack of progress alone is not proof of death. Use the
application's operation deadline or explicit interruption to stop it.

After an owner is lost, use the existing recovery plan and fenced claim; inspecting or
showing the state never reserves recovery. An expired token cannot renew or
replace a successor's owner row. During orderly completion, retained cleanup
keeps its heartbeat until physical cleanup finishes.

## Keep GET handlers read-only

A GET handler in your app must not reconcile decisions, migrate data, backfill
projections, or write anything. Reads happen far more often than writes, from
every tab, and a read that can write makes idle tabs contend with real work. Do
that work on the write path that changes the data, or in a durable event sink
that reacts to the event once (`cayu guide human-attention` describes durable
sinks and how to repair them).

## Use usage and cost, not event folds

Do not compute tokens, cost, or other aggregates by folding a session's full
event history on each request. Use `GET /api/sessions/{id}/usage` (or
`getSessionUsage`) for tokens and `POST /api/sessions/{id}/cost` with a price
book for cost. For an aggregate Cayu does not provide, keep a projection that
you update from new events with an `after_sequence` cursor when they are
written, and read the projection.

## Idle budget

An idle open tab should cost under 1% of a 0.5 vCPU web process. In practice an
idle tab sends nothing while it is hidden or while nothing is running, and only
occasional cheap, bounded requests otherwise. When your installed Cayu provides
`cayu diagnostics requests`, check an idle tab against the budget with:

```bash
cayu diagnostics requests --budget-idle-cpu 0.01 --vcpu 0.5
```

## Example: a progress panel

A panel that shows a running session's tool activity and final token usage.
It streams or polls through `client.js`, pauses while hidden, and stops by
itself when the session ends.

```js
import { connect } from "/cayu/client.js"

const cayu = await connect()

export function showProgress(sessionId, panel) {
  const steps = panel.querySelector("ol")
  const status = panel.querySelector("[data-status]")
  const follow = cayu.followSession(sessionId, {
    excludeEventTypes: ["model.text.delta"],
    onStatus: (value) => {
      status.textContent = value
    },
    onEvent: (event) => {
      if (event.type.startsWith("tool.call.")) {
        const item = document.createElement("li")
        item.textContent = `${event.tool_name ?? "tool"}: ${event.type.slice(10)}`
        steps.append(item)
      }
    },
  })
  follow.done
    .then(async ({ reason, status: final }) => {
      if (reason !== "terminal") return // stopped by the caller
      const usage = await cayu.getSessionUsage(sessionId)
      status.textContent = `${final}, ${usage.usage.total_tokens} tokens`
    })
    .catch((error) => {
      status.textContent = `Stopped: ${error.message}`
    })
  return () => follow.stop() // call when the panel unmounts
}
```

For a list of active runs, use `watchSessions` instead of a timer:

```js
const stop = cayu.watchSessions(
  { status: ["pending", "running"] },
  { onChange: (sessions) => renderRuns(sessions) },
)
```
