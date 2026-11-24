# The web app and browser UIs over Cayu sessions

`cayu new` gives an Agent on every preset but service a web app. The first part of
this guide covers that app: its parts, its areas, the `@kit` alias, and who signs
in to it.

The rest is for any browser UI that shows Cayu sessions, the generated app or one
of your own: a list of runs, a live timeline, a progress panel, token usage.
Most of the work in such a UI is watching sessions change, and the easy way to
write that (a short fixed timer that never stops) is what makes these apps
expensive. One idle tab written that way made about 1,600 requests an hour and
used a third of a 0.5 vCPU web task. Follow the rules from
[Follow live sessions with client.js](#follow-live-sessions-with-clientjs) on. The
first one removes most of the others.

## The generated web app

The web app is the `web-app` capability, on by default in every preset but
service, which has none; `cayu new NAME --without web-app` leaves it out. `cayu serve` serves its committed build at `/`
(`[tool.cayu.serve] ui = "web/static"`), the Agent's routes under `/app`
(`routers = ["integrations.http:build_router"]`) and the Developer dashboard at
`/cayu/`. `uv run --no-sync cayu serve --dev` serves all three at
`http://127.0.0.1:8000/`.

Each file says in its first doc comment whether it is `GENERATED.` (the generator
owns it and may rewrite it) or `THE AGENT'S OWN.` (yours).

| Path | What it is |
| --- | --- |
| `web/src/main.tsx`, `web/src/app/` | The shell: browser routes, the area type, the layout with its navigation and profile menu, the status pages, and the requester for `/app`. Generated. |
| `web/src/generated/`, `web/src/api/` | The Overview and Settings areas, the Knowledge area when the Agent has knowledge, and typed clients for the generated routes. Generated. |
| `web/src/blocks/task-runs.tsx` | TaskRuns, a building block: the Agent's Cayu tasks, one row per run (`GET /app/task-runs`). |
| `web/src/agent/` | The Agent's own areas, and `index.tsx`, which lists them. Yours. |
| `web/src/kit/shadcn/` | The component kit every page imports through `@kit`. Generated. |
| `web/e2e/` | The smoke test: `cayu serve --dev` serves the app, and every area opens in a browser. Yours. |
| `integrations/http/__init__.py` | The routes under `/app`. The Agent's own go on its `router`, behind the same sign-in. Yours. |
| `integrations/http/generated/` | The routes the generated pages read: `/app/whoami`, `/app/settings`, `/app/overview`, `/app/task-runs` and `/app/inbox`, and `/app/knowledge` when the Agent has knowledge. Generated. |

Every app has the generated Overview and Settings, and its navigation links to
Cayu's Inbox (`/cayu/inbox`), where approvals and questions are decided, and to the
Developer dashboard. What `web/src/agent/` adds depends on the preset: the agent
preset has Conversations, which start and continue the agent's sessions through
Cayu's API at `/api`, and Task runs; coding and long-running have Task runs; with
organizations it has Conversations and Members, scoped to the person's
organizations (`cayu guide b2b`). An Agent with the `knowledge` capability (the agent and
coding presets, and the agent preset with organizations) also has the generated Knowledge
area ([The Knowledge area](#the-knowledge-area)).

After changing `web/src`, run `npm --prefix web ci` and `npm --prefix web run build`,
and commit `web/static`, so running and deploying need no Node.
`npm --prefix web run check:build` fails when `web/static` isn't the build of
`web/src`, and `npm --prefix web run dev` reloads the app live against
`cayu serve --dev`. The app builds, lints and formats with Vite+ (`vp`, settings in
`web/vite.config.ts`): `npm --prefix web run lint`, and `npm --prefix web run format`
(or `format:check`).

### Areas

An area is a part of the app with its own navigation entry and routes: an `AppArea`
(`web/src/app/area.ts`) with an `id`, a `label`, an `icon`, the `path` its entry
opens, and `routes`, its React Router routes under the app's layout.
`web/src/agent/index.tsx` exports what the shell shows for this Agent:

- `identity`: its name and what it does;
- `areas`: its areas, in navigation order;
- `shell`: an optional `SignIn` page for when nobody is signed in (without one, the
  app asks a person whose sign-in ended to reload and sign in again), and
  `ProfileItems` for the end of the profile menu.

### Add an area

Add the route under `/app` in `integrations/http/__init__.py`, on `router` and
behind the same sign-in. A route that changes the Agent's state takes
`Depends(may_change_dependency(person))` instead, which refuses a role that may not
change things.

```python
    @router.get("/findings")
    async def findings(signed_in: Person = Depends(person)) -> list[dict[str, str]]:
        return [{"id": "disk", "title": "The disk is almost full"}]
```

Write the page as an area of its own, in `web/src/agent/findings.tsx`:

```tsx
import { queryOptions, useQuery } from "@tanstack/react-query";
import { z } from "zod";
import { type Requester, watchInterval } from "@kit/data";
import { Alert02Icon } from "@kit/icons";
import { AppPageHeader, AppPageTitle } from "@kit/layouts/app";
import type { AppArea } from "../app/area";
import { useAreaApi } from "../app/services";

const finding = z.object({ id: z.string(), title: z.string() });
type Finding = z.infer<typeof finding>;

function findingsApi(request: Requester) {
  return { list: (signal?: AbortSignal) => request("findings", { signal, schema: z.array(finding) }) };
}

/** THE AGENT'S OWN. Findings: what the Agent found, from `GET /app/findings`. */
export const findingsArea: AppArea = {
  id: "findings",
  label: "Findings",
  icon: Alert02Icon,
  path: "/findings",
  routes: () => [{ path: "findings", Component: FindingsPage }],
};

function FindingsPage() {
  const api = useAreaApi(findingsApi);
  const findings = useQuery(
    queryOptions({
      queryKey: ["findings"],
      queryFn: ({ signal }) => api.list(signal),
      refetchInterval: watchInterval<Finding[]>(),
    }),
  );
  return (
    <div className="flex flex-col gap-page-section" data-origin="own">
      <AppPageHeader description="What the Agent found.">
        <AppPageTitle>Findings</AppPageTitle>
      </AppPageHeader>
      <ul>
        {findings.data?.map((item) => (
          <li key={item.id}>{item.title}</li>
        ))}
      </ul>
    </div>
  );
}
```

Then list it in `web/src/agent/index.tsx`, where it should appear in the navigation,
and rebuild:

```tsx
export const areas: AppArea[] = [overviewArea({ description: "..." }), findingsArea, taskRunsArea, settingsArea];
```

The page reads its route with the kit's `watchInterval`, the cadence `client.js`
keeps (below), instead of a timer of its own; answer the route with `tagged_json`
([Answer an unchanged read with a 304](#answer-an-unchanged-read-with-a-304)) so a
read of an unchanged answer costs a `304`. Add the area's page to the smoke test in
`web/e2e/` too.

### The `@kit` alias

Pages import components from `@kit/...` (`@kit/ui`, `@kit/data`, `@kit/icons`,
`@kit/patterns`, `@kit/layouts/app`), never from `web/src/kit/shadcn/` by path.
`const kit` in `web/vite.config.ts` and `@kit/*` in `web/tsconfig.json` point the
alias at `src/kit/shadcn`: stock shadcn components on Base UI, and the parts built
from them. Another kit is a folder next to it with the same entries (`ui`, `data`,
`lib`, `hooks`, `icons`, `patterns`, `layouts/app`, `layouts/app/router` and
`styles.css`); point both at it, and no page changes.

### Who signs in: operators or members

Without organizations, the app's people are the Agent's operators, who sign in as
they do to the Developer dashboard (`[tool.cayu.serve].auth` is
`OPERATOR_BASIC_AUTH`). Deployed on Cayu Cloud, they are Cloud's operators with
their Cloud roles: every role reads and decides what waits in the inbox, and
`admin` and `developer` change things. Under `cayu serve --dev` it is you, on a
loopback address only. The routes get the person from
`cayu.server.operator_sign_in`
([Know who is signed in to your own routes](#know-who-is-signed-in-to-your-own-routes)).

With organizations (the b2b preset, or `--with organizations` on the agent preset),
the app's people are the organizations' members instead. `integrations/sign_in.py`
signs them in: under `cayu serve --dev` as a demonstration person from
`operations/demo_seed.py`, on the app's own sign-in page; deployed, through Cayu
Cloud's sign-in (`APP_SIGN_IN=cloud`, the default). `APP_SIGN_IN=oidc` instead
accepts an OIDC bearer token from API clients that send their own; the web app has no
OIDC sign-in in the browser. `[tool.cayu.serve].auth` is `OPERATOR_ONLY_AUTH`, which keeps
the Developer dashboard and Cayu's `/api` for operators, so members reach their data
only through `/app` (`cayu guide b2b`).

### The Knowledge area

The Knowledge area shows the app's people what the Agent knows, written for them rather
than for its builders. `cayu new` adds it when the project has the `knowledge` capability:
the page in `web/src/generated/knowledge/`, listed just before Settings, and its routes in
`integrations/http/generated/knowledge.py`, which the Agent's router includes.

- **What the Agent knows.** Its kept knowledge as plain statements, latest first, with a
  search (`GET /app/knowledge?q=`, Cayu's keyword search, which matches whole words and
  their other forms). Each says where it came from (a conversation, linked when the app has
  Conversations; a document; a person) and when it was learned, and who corrected it.
- **Waiting to be remembered.** What the Agent asked to remember that nobody has kept or
  dropped yet (`GET /app/knowledge/suggestions`). These are the knowledge proposals in Cayu's
  Inbox; each links to its record there, where it is kept or dropped.
- **Correct and forget.** `POST /app/knowledge/correct` replaces a statement's text, which
  stays kept and records who corrected it; where it was learned stays as it was.
  `POST /app/knowledge/forget` archives it with who forgot it and why, as the Developer
  dashboard's archive does, and the Agent stops finding and recalling it. Both name the
  statement's `version`, so a statement that changed since the person read it answers 409.

Who reads and changes what follows the app's people. In an operator app every operator
reads, and Cayu Cloud's `admin` and `developer` (you, under `cayu serve --dev`) correct and
forget (`operator_knowledge`). With organizations each member reads only their
organizations' knowledge, through the same resource-access policy recall uses
(`MembershipAccessPolicy(..., knowledge=True)`); an organization's admins correct and forget
its statements, and nobody sees, corrects or forgets another organization's
(`member_knowledge`). Members can't open Cayu's Inbox, so their suggestions don't link
there: the Agent's operators keep or drop them. `tests/test_knowledge_area.py` checks each
of these.

The page shows no revisions, chunks, scores or recall evidence. The Developer dashboard
(`/cayu/`) stays the full view of knowledge, for builders and operators. Both reads answer
with `tagged_json` and the page follows them with `watchInterval`.

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
- `watchTasks(filter, onChange, { onError, onStop, intervalMs, signal })` (or
  `watchTasks(filter, { onChange, ... })`) watches the tasks that match a
  `filter` of `type` (one or several), `status` (one or several), `input`
  values, `run` (the task's `metadata.run`) and other `metadata` values, and
  returns a function that stops the watch; see [Show tasks](#show-tasks). It
  throws `unavailable` for an `input`, `metadata` or `run` filter on a server
  that doesn't advertise `task_value_filters`, which would ignore them.
- `watchInbox({ onChange, onError, onStop, intervalMs, signal })`,
  `listInbox`, `getInboxItem` and `settleInboxItem` read and settle the agent's
  inbox; see [Show the agent's inbox](#show-the-agents-inbox).
- `getAppInfo()` reads `GET /api/app`: the Agent's name, the person's access, and
  whether deciding needs protected human review (`human_review_required`) with which
  `human_review_purpose`.
- `getHumanReview(sessionId, { purpose })` and
  `readHumanReviewLongField(sessionId, reference, key)` read the protected review a
  person decides with; settle with its `reference` as `review_reference`. Keep the
  view out of shared caches and storage.

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
- **Task cadence.** `watchTasks` keeps the same 15-second floor and backoff,
  and revalidates each read with the list's `ETag`, so an unchanged list costs a
  `304` with no body. It never stops as idle, because new tasks can be queued
  at any time.
- **Errors.** Transient failures (network, 408, 429, 5xx) are retried with
  exponential backoff and jitter, and never sooner than `Retry-After`. Other
  4xx responses (401, 403, 404, 422, ...) and responses that do not match the
  contract stop the operation and call `onError` with a `CayuClientError`
  whose `kind` says why (`auth`, `not_found`, `contract`, `http`). A stopped
  `followSession` also rejects `done` with that error.

`interrupted` includes pauses for tool approval or user input. After the UI
answers one with `settleInboxItem`, call `followSession` again with
`afterSequence: follow.lastSequence`. Likewise, call `watchSessions` again after
starting new work if the previous watch stopped as idle.

Do not hand-write polling loops for session events or session lists. If
`client.js` does not fit, use the same endpoints with the rules below.

`client.js` calls Cayu's operator API, so it is for operators and for applications
whose users may use that API. An interface for an application's own users, who
reach only what a resource-access policy grants them, calls the application's
scoped routes instead; `cayu guide b2b` shows one, whose pages follow those routes
with the rules below (its kit's `watchInterval`).

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
from cayu.sessions import SessionExecutionConfig

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

An application that uses `CayuApp` directly has no such sweep. The next
`run` or `resume` of a conversation recovers its own session when it takes it
over, but a session the application never continues stays `running` with an
`owner_lost` execution, for example when the application finished a crashed
write from its own records instead of resuming the conversation. Call
`await app.recover_abandoned_sessions()` at startup or on a schedule. It uses
paginated incomplete-session recovery for running and interrupting sessions
inactive for `inactive_for_seconds` (default 3600), skips sessions with a live
owner, and returns the per-session results. Keep the window longer than the gap
after which a conversation's next message would still take its session over:
that takeover replays interrupted `NONE` and `IDEMPOTENT` calls, while the sweep
closes them with an unknown outcome. The sweep also settles ordinary model calls
lost after dispatch, charging their reserved budgets as unknown usage without
calling the provider again. Background provider operations and contracted tasks
retain their existing recovery paths; inspect the per-session results for work
that needs an explicit decision.

A session that died during an `EXTERNAL` tool call ends `interrupted` with
`pending_tool_effect` and keeps its environment allocation, because only an
explicit receipt or lookup may continue it. When the application has already
given up on that work, call `await app.abandon_session(session_id)`. It keeps
the unknown outcome without dispatching the call again, fails the session, and
releases its environment through the factory. The session can no longer be
resumed. Pass `abandon_sessions_of_terminal_tasks=True` to the sweep to do this
for every such session whose tasks are all terminal; later sweeps also pick up
sessions whose tasks finished after an earlier sweep interrupted them.

A session waiting on a question or an approval that nobody will answer, or one
interrupted long ago, is ended with `await app.close_session(session_id,
reason="...", closed_by=actor)`; pass `waiting_on=` the approval or input ID to
close it only while it still waits on that pause. It fails the session with the reason and actor
recorded, removes its pending approval or question from the inbox, keeps its
transcript and events, and releases its environment. `cayu session close` and
`POST /api/sessions/{id}/close` do the same. If a close stops after the session
failed but before its child sessions closed, repeating it or the next
`recover_abandoned_sessions()` sweep closes the rest after the root meets its
inactivity window. Retained cleanup stays retryable after all children close.

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

## Show the agent's inbox

An agent's app can show what the agent is waiting on a person for, and let the
person decide there: approve or deny a tool call, answer a question, keep or drop a
knowledge proposal. These are the same records Cayu Cloud and the Developer
dashboard show, so a decision made in any of them shows as settled in the others,
with the person who made it. [The agent's inbox](../../../docs/inbox.md) describes
the routes, item states and conflicts.

```js
import { connect } from "/cayu/client.js"

const cayu = await connect({ headers: { Authorization: `Bearer ${await signInToken()}` } })

const stop = cayu.watchInbox({
  onChange: ({ items }) => renderInbox(items),
})

async function decide(item, decision) {
  // decision is the settle body, for example { decision: "approve", reason: "Checked." }
  const result = await cayu.settleInboxItem(item.id, decision)
  if (result.settled) return renderSettled(result.item)
  // Someone else decided first, or the item was replaced or expired.
  renderSettled(result.item, result.message)
}
```

`signInToken`, `renderInbox` and `renderSettled` are the app's own. The rules the
client follows for you:

- **Cadence.** `watchInbox` reads the whole inbox (following `next_cursor`) no more
  often than every 15 seconds, doubles the interval while nothing changes, and pauses
  while the tab is hidden. Something can start waiting at any time, so it never stops
  as idle; call the function it returns when the inbox unmounts. If the server
  refuses a later page's cursor (409 or 422, for example after it restarted), that
  refresh drops what it read and starts over from the first page once. If that is
  refused too (workers without a shared `inbox_cursor_key`), the watch stops and
  calls `onError` with the server's answer. With `listInbox`, restart from no cursor
  after such a refusal.
- **Identity.** The person is whoever the request authenticates as. Send the person's
  own credential, never a shared one, or every decision records the shared identity.
- **Conflicts are answers.** `settleInboxItem` resolves with `settled: false`, a
  `code` and the item's current state when the item was already settled, replaced or
  expired, so the UI can show who decided instead of an error.
- **Retries are safe.** Repeating the same decision resolves with `replayed: true`;
  nothing runs twice.

Render one card per item with its `decisions` as the actions: `approve` and `deny`
(with an optional reason, and approve-with-changes when the app can edit the tool's
arguments), an answer field for a `question` with its `options` as choices, and
`keep` and `drop` for `knowledge`. A `recovery` item has no decisions; link it to the
session in the Developer dashboard. A settled item shows its
`settlement.decision`, `decided_by.subject` and `decided_at`. Render every text field
as escaped text.

## Show tasks

A page built on tasks (the latest results of a scheduled job, a list of runs, a
job someone just queued) follows them with `watchTasks` instead of its own timer:

```js
const stop = cayu.watchTasks(
  { type: "nightly_report", status: ["running", "completed"], input: { repository: "acme/api" } },
  (tasks) => renderReports(tasks), // newest update first
)
```

- **Filter.** `type` and `status` take one value or a list; several types are read
  with one request each. `input` and `metadata` match top-level values as the task's
  detail shows them, and `run` is the task's `metadata.run`. `limit` (default 100)
  is the page size per type.
- **What arrives.** The rows of `GET /api/tasks`, merged and newest update first.
  They leave out each task's input and result: read the one you show with
  `GET /api/tasks/{task_id}` when `onChange` reports it finished.
- **Cost.** The same 15-second floor and backoff as `watchSessions`, nothing while
  the tab is hidden, and each read revalidated with the list's `ETag`, so a quiet
  watch costs a `304` with no body. It never stops as idle; call the returned
  function when the page goes away.

## Know who is signed in to your own routes

When an Agent's people are its own team (Cayu Cloud's operators when deployed, you under
`cayu serve --dev`), its own routes sign them in with the same auth as the Developer
dashboard and get the person from `cayu.server.operator_sign_in`, instead of copying a
sign-in file into every Agent:

```python
from fastapi import APIRouter, Depends

from cayu.server.operator_sign_in import (
    OperatorPerson,
    may_change_dependency,
    operator_person_dependency,
)


def build_router(app, *, development, auth=None):  # [tool.cayu.serve] routers
    person = operator_person_dependency(auth, development=development)
    router = APIRouter(prefix="/app")

    @router.get("/me")
    async def me(signed_in: OperatorPerson = Depends(person)):
        return {"name": signed_in.name or signed_in.subject, "may_change": signed_in.may_change}

    @router.post("/recheck")
    async def recheck(signed_in: OperatorPerson = Depends(may_change_dependency(person))):
        ...

    return router
```

- **Who.** `cayu serve` passes `auth` (`--auth` over `[tool.cayu.serve].auth`) and
  `development`. Cayu Cloud's people come with their `subject`, `role`, `name` and
  `email`; the operator login without Cloud's identity header is `admin`; under `--dev`
  it is you, as `dev:<user>` and `admin`, on a loopback address only. Without `auth`
  outside `--dev` it refuses to build, so the routes never serve open.
- **Roles.** By default every Cloud role (`admin`, `developer`, `member`) may read and
  decide (`may_decide`), and `admin` and `developer` may change things (`may_change`);
  any other role gets 403, as does a `member` under `OPERATOR_ONLY_AUTH`, where members
  are the app's own users. Pass `rule=operator_roles(auth, decide=(), person=Person)` to
  keep that and change only who decides (`decide=()` when nobody decides in this app) or
  build the generated app's `Person`. `rule=OperatorRoles(...)`, or any function from
  `AuthContext` to your person, replaces the rule whole, without the auth's roles.
- **Writes.** A `POST` (or any method but `GET`, `HEAD` and `OPTIONS`) sent from another
  site gets 403, so a page elsewhere can't use the person's saved sign-in.
- **Elsewhere.** With `mount_cayu`, pass the dependency you give `AuthenticatedAccess`,
  or `None` with `development=True` beside `OpenAccess(trusted_local_development=True)`.
  `operator_context_dependency` gives the verified `AuthContext` alone, without a role
  rule. An app whose people are its customers signs them in with
  `ProxyIdentityAuth.product_dependency()` instead (`cayu guide b2b`).

## Answer exceptions as API errors

Build your `/app` router with `AppErrorRoute` and its handlers (and their dependencies)
answer exceptions the way Cayu's own routes do, with `{"detail": "..."}` and a status,
instead of a 500 or a `try` in every handler:

```python
from fastapi import APIRouter

from cayu.server.app_errors import AppErrorRoute, NotFound

router = APIRouter(prefix="/app", route_class=AppErrorRoute)


@router.get("/invoices/{invoice_id}")
async def invoice(invoice_id: str) -> dict:
    raise NotFound("Invoice not found.")  # 404 {"detail": "Invoice not found."}
```

Raise `NotFound` (404), `Forbidden` (403), `Conflict` (409) or `Invalid` (422) when the
message is written for the caller; it goes out as the detail. Other exceptions answer with
a fixed text, since a built-in exception's message may name a server path or a stored
value: Cayu's access refusals 404 `Resource unavailable.` (so a refusal doesn't say the
resource exists), any other `PermissionError` 403, Cayu's conflicts (`CAYU_CONFLICTS`: a
stale knowledge revision, a reused task or request id, a session in the wrong state) 409,
artifact storage being down 503, `LookupError` and `FileNotFoundError` 404, and any other
`ValueError` 422. `HTTPException` and request validation pass through, and an exception no
rule names is still a 500.

Add your own exceptions with `extend`; its rules are checked first and, without a
`detail`, send the exception's message:

```python
from cayu.server.app_errors import DEFAULT_APP_ERRORS, ErrorStatus

errors = DEFAULT_APP_ERRORS.extend(ErrorStatus((InvoiceLocked,), 409))
router = APIRouter(prefix="/app", route_class=errors.route_class())
```

`ErrorStatus(..., detail=...)` sets a fixed detail or a function of the exception, and
`errors.http_error(exc)` gives the mapping on its own. The routers `cayu new` generates
under `/app` use `AppErrorRoute`.

## Keep GET handlers read-only

A GET handler in your app must not reconcile decisions, migrate data, backfill
projections, or write anything. Reads happen far more often than writes, from
every tab, and a read that can write makes idle tabs contend with real work. Do
that work on the write path that changes the data, or in a durable event sink
that reacts to the event once (`cayu guide human-attention` describes durable
sinks and how to repair them).

## Answer an unchanged read with a 304

A route of your own that a page reads again and again should cost little while its
answer doesn't change. Return the answer through `tagged_json`:

```python
from fastapi import APIRouter, Request, Response

from cayu.server.etag import tagged_json

router = APIRouter()


@router.get("/app/reports")
async def list_reports(request: Request, response: Response) -> list[dict[str, str]]:
    reports = [{"id": "march", "status": "done"}]
    return tagged_json(request, response, reports)
```

It sets a weak `ETag` of the answer and `Cache-Control: private, no-cache`. A request
whose `If-None-Match` names that tag gets `304 Not Modified` with no body. A browser
keeps the answer and revalidates it on each `fetch` by itself, so the page needs no code
for it, and a watch that compares answers sees the same one.

Keep out of the tag what changes on every read, such as an `as_of` time: pass the rest
as `tagged` (for a Pydantic answer, `view.model_dump(mode="json", exclude={"as_of"})`),
or the tag never matches. A client that gets the `304` keeps the answer it had, with its
old `as_of`, so a page shows when it last read rather than the answer's `as_of`.

A route that builds its own response uses the parts `tagged_json` is made of:

```python
from fastapi.responses import JSONResponse

from cayu.server.etag import if_none_match_matches, json_body_etag


@router.get("/app/summary")
async def summary(request: Request) -> Response:
    body = {"open": 3, "done": 12}
    headers = {"ETag": json_body_etag(body), "Cache-Control": "private, no-cache"}
    if if_none_match_matches(request.headers.get("if-none-match"), headers["ETag"]):
        return Response(status_code=304, headers=headers)
    return JSONResponse(body, headers=headers)
```

The routes every generated web app reads under `/app` (`/app/settings`,
`/app/overview` and `/app/task-runs`) answer this way.

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
