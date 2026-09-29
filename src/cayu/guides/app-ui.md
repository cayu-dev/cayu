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
