# Developer dashboard

Cayu's bundled Developer dashboard is open-source React and TypeScript. The
compiled dashboard is convenient package data, not an opaque or mandatory product UI. An
application can choose among three supported modes:

| Mode | Owner | Use it when |
| --- | --- | --- |
| Bundled dashboard | Cayu | You want the stock zero-configuration Developer dashboard. |
| Editable Cayu dashboard | Your application after extraction | You want to customize the same source and generated API client used by Cayu maintainers. |
| Bring your own UI | Your application | You want a completely independent frontend over the versioned control-plane API. |

All three modes expose a Developer dashboard surface. Public deployment requires explicit authentication
and authorization; unauthenticated `ServerConfig.local_development()` access is for trusted
loopback development only.

## Local development reload

Use `cayu serve --dev --reload` to restart the server after Python source changes.
Reload is opt-in and prints a warning at every worker startup: in-memory state is
lost and active runs can be interrupted. Persistent stores retain saved data,
but do not keep active runs uninterrupted. Authentication and the loopback-only
`--dev` restriction remain unchanged. See [project server reload](project-server.md#reload-python-code-during-development)
for watched files, startup failures and shutdown behavior.

## Agent identity and viewer access

`GET /api/app` returns `name`, `runtime_version`, `role` and `full_access`.
The name comes from the project's normalized public identity (`[project].name`),
then the server deployment name, then `Cayu`; names outside the existing 128-character diagnostic bound are
omitted before choosing the next fallback. The runtime version is `unknown`
when installed package metadata is unavailable.

With proxy identity, the response carries the authenticated role (`admin`,
`developer`, `member`, or a configured custom role). `full_access` follows the
configured `ControlPlaneRoles`, so clients must use it instead of inferring
permission from the role's spelling. The proxy's default Basic-auth fallback
reports `admin`. Without a configured role policy, callers admitted by the
server's access dependency receive `role: null` and `full_access: true`.
Authentication claims on their own do not enable a role policy.

The read is available to members, advertises `capabilities.surfaces.app_info`,
and uses `Cache-Control: private, no-store`. This is presentation information:
each API route still enforces its own authorization and runtime preconditions.
Server contract 50 adds this read; independently hosted dashboards must
regenerate their client and upgrade their supported contract version together.

## Dashboard navigation and member access

The dashboard sidebar groups Home, Inbox, Sessions and Tasks, with Evals and
Usage under Improve and Specialists, Tools, Knowledge, Environments, Artifacts and
System under Agent. Items depend on the server capabilities and viewer access.
Existing URLs remain valid: `/pending-actions` redirects to Inbox at `/inbox`,
`/run` opens Home with the shared Start form, and Specialists uses `/specialists`; `/agents?q=...` redirects there with its search intact. Tools at `/tools` retains each specialist declaration, including differing schemas or policies with the same public name. Search sessions from the sidebar; `Ctrl+K` or
`Cmd+K` focuses search, and Enter opens the session index with its `q` filter.
The Sessions page uses `GET /api/sessions?include=overview`, displays authored titles
and completion summaries, and preserves the existing filter links. Eval trials and
forks are excluded by default and can be included explicitly. Its footer shows loaded
rows, not a matching population count; sidebar and Home counts remain separate reads.
The sidebar can be collapsed on desktop or opened as a drawer on mobile.
Escape closes the mobile drawer and returns focus to its opener.

Session detail opens a bounded Conversation view. Authored titles and summaries use
the same overview as the session index; tool calls start collapsed and structured
outputs retain their native result shape. Older/Newer controls and role/thinking
filters preserve transcript history. Live updates follow the tail only while it is
pinned; reading an older window does not move it. Tool links use exact transcript
message and part positions to open Trace. Trace contains event filters, runtime
and transport diagnostics, the session tree, and causal usage. Existing Workflow
URLs redirect there while retaining their filters and expansion state. Workflow
specialist/environment filters use `workflow_agent_name` and `workflow_environment_name`
in Session URLs, separately from the event log's exact filters. Hiding thinking
changes its presentation without changing native tool positions used by Trace.

The details panel starts closed and contains usage, labels, user metadata,
artifacts and browser controls when applicable. Runtime-owned metadata stays
read-only. Browser takeover, private input and handback use the existing browser
control authority. Evaluation promotion continues to use captured session evidence.

Full-access operators can follow up, stop with confirmation, resolve provider
operations, and recover pending actions using their public exact source references.
A running session accepts queued messages only when the deployment has both an
authenticated identity and a `SessionMessageAccessPolicy` granting the operation;
otherwise the composer explains its unavailability. Queue withdrawal uses the
record's current incarnation and revision. Unconfirmed writes retain their exact
request for observation or receipt reconciliation instead of enabling another write.
While a Session action or specialist Start is unconfirmed, route navigation and
form closure ask the operator to stay and recover it. Conversation and Trace share
the same observation recovery and result handling. Explicitly leaving discards the
mounted recovery handle; it does not cancel accepted server work. Inspect the
resulting session or task before submitting new work. Reloading the tab also loses
this local handle. These guards do not identify account switches; each recovery
request is authorized again by the server.
Run again creates a new direct session from a supported first text prompt; task-origin
sessions instead use declared task replay and navigate to the returned task, whose
worker may attach a session later. Member decisions retain protected human review
when configured and do not acquire execution permission.

Use the theme button to cycle System, Light and Dark. The choice is saved in
this browser; System follows the operating system appearance. The external
initialization script applies the choice before styles load under the existing
content security policy, including when the dashboard is mounted below a path.

To check the member experience, sign in through the configured identity proxy
with a role in `ControlPlaneRoles.read_and_decide`, then open the dashboard.
Locally, start `cayu serve` with `CAYU_PROXY_IDENTITY_ISSUER`,
`CAYU_PROXY_IDENTITY_AUDIENCE` and `CAYU_PROXY_IDENTITY_JWKS_URL` set (see
[Trust a sign-in proxy](project-server.md#trust-a-sign-in-proxy)), serve a JWKS
for a key you control, and send a token signed with that key whose `role` claim is
`member` in the identity header. `tests/server/test_proxy_identity.py` shows how
to build the key, the JWKS and the token.
The authenticated `/api/app` response drives the shell's access presentation.
Members can browse sessions and pending work, approve or deny a pending tool
approval, answer a pending question, and approve or reject knowledge proposals
when the corresponding server capability is available. Open Inbox and follow
settle permitted decisions inline with an optional reason. Recovery links to its
session for the existing evidence-based workflow. Arguments remain withheld;
protected human review is used when configured. Members cannot replace arguments.
For an interrupted task, explicitly select task continuation before deciding;
this requests a new owned claim and never borrows an active worker's authority.

System is omitted for members; opening it directly shows an access explanation
before its page requests start. Members can read eval suites, runs and
comparisons; run details, report downloads, execution configuration and the
advanced tools stay with full access. Home and Sessions
hide execution controls for members, including after an old `/run` redirect. Full-access
mutation controls elsewhere are hidden or disabled with an explanation.
Browser-control and evaluation-authoring panels do not mount for members.
These presentation rules also follow custom role policies; they do not grant
permission. The server independently authenticates and authorizes every read
and decision. The server-contract gate still precedes page reads, and an
unavailable app identity does not grant full access.

## Usage, environments, artifacts and setup

Usage opens the last 30 days by default. The Last 24 hours, Last 7 days and
Last 30 days choices start at the first displayed hour or day and end now;
the headline totals and chart use the same bounds. Custom links retain exact
timezone and microsecond boundaries. Existing specialist, environment, provider,
model, parent, budget, label and label-selector URL filters remain valid, and
model/provider links carry those session filters into Sessions. Chart counts
currently show all ordinary sessions started in the window, excluding forks and
eval trials, even when the usage totals are filtered. Models show tokens; top
sessions are ranked by tokens. Pricing coverage, exact windows and raw billing
breakdowns are available in the collapsed Debug section.

Environments links to `/environments/:name` for runner, workspace, branch,
file-storage and knowledge information, plus recent sessions and saved files.
Workspace branches are optional private copies created by agent code, not Git
branches or automatic per-session workspaces. Branch lifecycle rows describe bounded
attached-handle observations, including completed handles and incomplete inspection;
they are not an exact count of all open or durable branches. Existing
`/environments?q=...` links filter the list, while Session links open the exact
environment detail. Basic recent-session reads remain available without optional
overview enrichment. Without a runner, Shell, Git and
text-search tools are unavailable. MCP currently shows a count, and file storage
says whether files are kept without inferring its storage backend.

Artifacts keeps filters and paging in the URL. Search currently filters only
the loaded page, ordered newest first. A preview link uses both `artifact` and
`artifact_store_id` because an artifact ID alone is not globally unique. The
drawer offers Download and Open raw where supported, with internal identities
in Debug. Session and environment links remain within the configured dashboard
mount. Usage and Artifacts respect their server capability gates; optional
session-title/count reads are omitted when overview queries are unavailable.

System is for full-access viewers. It shows agent setup, disabled-feature reasons
and Evals readiness; capability matrices and raw diagnostics stay in Debug.
It reports deployment name and sign-in requirements without guessing deployment
kind. Member reads on the other pages still require server authorization.

## Project-owned Evals assembly

`cayu serve` assembles the durable, non-executable Evals foundation without
requiring an application to construct `EvalsConfig`: normalized project
identity from `[project].name`, release identity from the
[Release the application records](runtime-contracts.md#release-identity) or the
public application-manifest fingerprint, and a matching SQLite or PostgreSQL
`EvalStore` from the project's session-store declaration. Trusted loopback
`cayu serve --dev` may create `data/cayu.db` as the local default; production
does not invent a database.

After the application is constructed, Cayu generates one bounded `default` eval
target for every registered specialist. Each target keeps the agent's normal provider,
tools, environment defaults, approvals, and runtime policy. Its stable key is a
domain-separated SHA-256 of the project, agent, and profile identities, so the
key does not change when the application release changes. The current release
and exact application-manifest fingerprint remain separate result identity.

`GET /api/evals/targets` publishes the safe project/agent/profile mapping. Corpus
and run catalog requests select only one of those keys; an omitted selector uses
the server-published deterministic default. Imports and run admission resolve the
key carried by the immutable corpus and reject unknown keys. The browser never
manufactures a live application, credentials, tools, environments, request
templates, or other execution authority.

A generated project can publish one optional default judge by declaring
`[tool.cayu.evals.default_judge]` in `pyproject.toml`. The declaration names an
already-registered provider, exact model, `public-only` or
`public-and-transcript` privacy, an explicit same-model decision, and bounded
time/token ceilings. `[tool.cayu.evals].price_book = "bundled-public"` may
explicitly select Cayu's dated packaged public-rate snapshot; with that
selection, the judge may also declare a cost threshold and author-first
launches may narrow candidate spend per trial. Cayu gives the judge a separate
tool-free application and never infers authority, credentials, or pricing.
Same-model use remains a separate, labeled operator choice in the suite editor.
Run `cayu guide evals-ai-quality` for the complete declaration and calibration
workflow.

New maintained-service factories accept Cayu's opaque project context and pass
it unchanged to `create_agent_service(...)`. Existing factories remain
compatible but do not receive automatic assembly until migrated. Use:

```bash
cayu check --json
cayu generate service-context --dry-run
cayu generate service-context
```

The generator edits only the previous generated shape it can prove safe;
customized or conflicting code requires manual review. Explicit `EvalsConfig`
continues to win as one complete singleton registry and is never partially
combined with framework-assembled state. Direct embedded servers continue to
wire their trusted objects explicitly. The registry identity already includes a
profile dimension; this slice publishes only the normal-authority `default`
profile. Additional application-owned profiles are reserved for cases that
deliberately substitute fixtures, environments, or authority.

## Controlled scenario execution

The bundled Evals workflow can save and run a scenario without constructing an
Evals-specific Python object in the browser or project. After **Check
readiness** succeeds, **Run scenario** admits the exact saved scenario revision
and reviewed binding to the durable eval worker. The Runs view shows each
trial's current phase and presents approve/deny controls only when that exact
trial reaches an authored fresh-approval checkpoint.

Initial input, queued follow-ups, ordinary session resumes, typed `ask_user`
answers, portable JSON, and file references execute through the registered
application's normal runtime. Approval and resume checkpoints retain their
session and event cursor across process restart. Cancellation, result JSON/HTML,
baseline comparison, `cayu eval report`, and `cayu eval compare` use the same
contracts as ordinary corpus runs.

This availability is not limited to `--dev`. Trusted loopback development may
assemble local storage automatically; production must supply its normal durable
store, authentication, and executable application target explicitly or through
the maintained project-service assembly. Readiness remains a factual gate for
missing current providers, tools, policies, environments, secrets, fixtures,
pricing, or execution limits—it is not a requirement to write Evals-specific
runtime configuration.

For the shortest user paths, run `cayu guide evals-first` or `cayu guide
evals-production` from the same installed Cayu version as the server.

## Eject and customize the matching source

Every wheel and source distribution carries one deterministic dashboard-source bundle tied to
the Cayu package version, server contract, generated API client, and bundled production assets.
Extraction is local and performs no Git, GitHub, or network operation:

```bash
cayu dashboard eject ./control-plane
cd control-plane
npm ci
npm run dev
```

The Vite server proxies `/api` to `http://localhost:8000` by default. Build application-owned
production assets with:

```bash
npm run lint
npm run test
npm run typecheck
npm run check:api
npm run build
```

`check:api` compares the extracted OpenAPI baseline and generated client with an installed Cayu
Python package that includes the `server` extra (`cayu[server]`). Set `CAYU_PYTHON` to the
intended environment's interpreter when it is not available as `python`.

Serve the resulting `dist/` directory through the normal Cayu configuration seam:

```python
from pathlib import Path

from cayu.server import DashboardConfig, ServerConfig, create_server

server = create_server(
    cayu_app,
    config=ServerConfig.local_development(
        dashboard=DashboardConfig(directory=Path("control-plane/dist")),
    ),
)
```

Embedded applications can instead pass the same path to
`mount_cayu(..., dashboard_dir=Path("control-plane/dist"))` or
`mount_dashboard(..., dashboard_dir=Path("control-plane/dist"))`. Non-root mount paths and
React-router deep links use the base path and API URL injected by the server.

## Ownership, upgrades, and compatibility

The destination must be explicit and empty. Cayu never rewrites an ejected directory during
installation, upgrade, or later CLI commands. Eject a new copy elsewhere when you want to inspect
a newer release, then merge application changes deliberately.

The dashboard checks the server's exact control-plane contract before loading data. A Cayu
upgrade that changes that contract produces a visible compatibility failure, and `npm run
check:api` reports OpenAPI/client drift; neither mechanism silently modifies source.

Evals appears in navigation when its read capability is enabled and the viewer has
full access. If the read capability is disabled, opening `/evals` shows a capability
explanation before mounting the page or requesting Evals endpoints. When the page
is available, independent readiness describes captured evaluation, catalog reads
and writes, captured-result persistence, scenario conversion, fresh execution,
cancellation, comparison, and reports. Readiness is explanatory metadata only;
authenticated API routes continue to enforce operator identity, mutation policy,
and execution preconditions.

Server contract version 25 adds durable scenario launch, per-trial progress,
and fresh scenario-approval mutations. Independently deployed dashboards,
servers, and generated clients must be upgraded together.

The extracted project includes Cayu's `LICENSE` and `NOTICE`, third-party license inputs, a
reviewed license baseline, and production-build license finalization. Its normal build retains
`LICENSE`, `NOTICE`, `REDISTRIBUTION.md`, and `THIRD_PARTY_LICENSES.md` in `dist/`. Downstream
distributors must retain applicable notices and mark modified Cayu-derived files as changed.

## Bring your own UI

A completely custom frontend can consume the authenticated API without using Cayu's React
source. Begin with `GET /api/contract`, honor its capability projection, use the bounded list and
detail APIs, and fail closed on an unsupported `contract_version`. Authentication,
authorization, redaction, and capability contracts are identical regardless of frontend.

To show live sessions, import the browser client the server ships instead of writing polling
loops. `mount_cayu(..., path="/cayu")` serves a dependency-free ES module at `/cayu/client.js`
(types at `/cayu/client.d.ts`), behind the same access dependency as the API, and the contract
advertises it under `client`. Its `followSession` streams when the contract advertises
`sse.session_follow` and otherwise polls `after_sequence` with backoff; it pauses while the tab
is hidden and stops when the session ends. `watchSessions` refreshes overview lists no faster
than every 15 seconds and stops when nothing is active. Read
[`cayu guide app-ui`](../src/cayu/guides/app-ui.md) before building the UI: it also covers
splitting overview from detail, keeping GET handlers read-only, using usage and cost instead of
folding events, and the idle-tab budget.

For direct SSE integration, open the read-only follow stream that the contract advertises as `sse.session_follow` (`GET /api/sessions/{session_id}/events/stream`) with a
browser `EventSource`. Pass `after_sequence` for the last event you already rendered; the
browser's automatic reconnect sends `Last-Event-ID` and resumes without gaps or duplicates.
Filter out chatty events such as `exclude_event_type=model.text.delta` when you do not render
them. Close the `EventSource` when the `end` event arrives: the session is finished and the
server has already closed the stream. The stream never starts or changes work, so opening it is
safe from any page.

Do not poll `/events` or `/state` on a fixed timer while a page is open. If a client cannot hold
a stream open, poll `/events` with `after_sequence` and `wait_seconds` only while the page is
visible and the session is pending, running, or interrupting, and stop at a terminal status.
`GET /api/sessions/{session_id}/state` returns an `ETag`, so a browser revalidates unchanged
state with a `304` and no body. Concurrent follow streams are capped per caller and per
session; a `429` carries `Retry-After`.

To show what the agent is waiting on a person for, and let the person decide in the
app, use [the agent's inbox](inbox.md) through `client.js`: `watchInbox` lists
approvals, questions and knowledge proposals, and `settleInboxItem` settles one as the
signed-in person.

Show tokens and cost with the accounting endpoints that the contract lists under
`accounting.usage` and `accounting.cost`, each with its `path` and a `supported` flag:

- `GET /api/sessions/{session_id}/usage` returns the session's `SessionUsageSummary`. The
  session store carries the summary forward and reads only events appended since the
  previous read, so polling does not get slower as the session grows. Each response has an
  `ETag`; send it back as `If-None-Match` and the server answers `304 Not Modified` until the
  usage changes.
- `GET /api/sessions?include=usage` adds a `usage` array with one summary per listed session,
  in the same order as `sessions`, so an overview page needs one request instead of one per
  session.
- `POST /api/sessions/{session_id}/cost` estimates cost from a `PriceBook` in the request body.

Do not compute tokens or cost by folding raw `model.completed` events in the UI or in
application code. That skips cache read and write counters, auxiliary provider attempts,
hosted-tool usage, and price-book rules, and its cost grows with the whole event history.

Provider-operation resolution is an explicit mutation capability. When session
state reports `provider_operation_unavailable` or `ambiguous_submission`, the
UI must display the exact `stage_id` and `run_epoch`, recovery reason,
`duplicate_request_risk`, and server-provided allowed resolutions. Submit only
`fallback_retry` or `fail` to `POST /api/provider-operations/resolve`. Treat a
409 as a stale or conflicting durable decision, never as permission to resubmit
with a guessed provider operation id. The bundled dashboard requires an
operator reason and warns whenever `duplicate_request_risk=true` that fallback
can duplicate provider work and cost.

## Naming and compatibility

The `/cayu/` UI is the **Developer dashboard**, both locally and on Cloud.
The app registers **specialists**: the Specialists page lists them, and sessions,
tasks and "Start a session" identify the specialist they use.
A **subagent** remains a child session started with `SubagentTool`.

Python names (`CayuApp`, `register_agent`, `AgentSpec`), API fields, the `/api/agents`
read and existing authentication configuration remain unchanged. The old dashboard
`/agents` URL redirects to `/specialists`, preserving its search. In particular,
`CAYU_OPERATOR_USERNAME`, `CAYU_OPERATOR_PASSWORD` and
`cayu.server.environment_auth:OPERATOR_BASIC_AUTH` still configure the Developer
dashboard login. This naming change does not remove authentication outside Cloud.

### Session overview

Server contract 51 adds `GET /api/sessions?include=overview`. Its `overviews`
array aligns with `sessions` and uses the same seven merged states as Cloud:
`running`, `waiting_approval`, `waiting_input`, `needs_recovery`, `completed`,
`failed`, and `interrupted`. Waiting states include current delegated child
work. Rows include the stable `run_id`, timestamps, bounded redacted
`cayu:title`/`cayu:summary` metadata, `cayu:summary_tone` (`neutral` or
`attention`), own-session usage, browser/subagent/artifact use, and trusted
`eval`/`forked` flags. Missing or invalid display metadata is null. Cost is null
without configured dashboard pricing; otherwise `complete` identifies whether
all recorded usage was priced. Aggregate counts use decimal strings on the wire.

Server contract 54 adds overview filtering to the session list. Pass repeated
`overview_state` values to match any of the seven states, and use
`include_eval=false` and/or `include_forks=false` to exclude evaluation trials
and fork-lineage sessions. Both inclusion flags default to true on this list;
ordinary child sessions are not forks. These filters require `include=overview`
and cannot be combined with `ids` or a nonzero `offset`. Existing raw `status`,
debug, identity, label, ordering, and text filters intersect with the overview
selection. Public text search `q` matches the currently redacted list fields,
labels, and the same validated/redacted authored title shown by the overview;
summaries and arbitrary metadata are not searched. Hidden title substrings
cannot affect membership, totals, or continuation positions. Redaction is
applied when searching, including when stored titles predate the current policy.

Text search uses bounded candidate paging on both `GET /api/sessions` (with or
without `include=overview`) and `POST /api/sessions/summary`. It requires cursor
paging rather than a nonzero offset, returns `total_count: null`, and can return
empty pages with a continuation. Summary usage and costs cover only the selected
rows on that page. The shipped Sessions page automatically follows empty
continuation pages while showing a searching state. It stops at a match or
confirmed exhaustion; after 25 candidate pages it pauses with an explicit
Continue searching action. Query changes cancel the previous traversal, and
repeated cursors fail rather than looping. Native SDK store queries retain their store-data search
semantics; they are not the public control-plane exposure boundary.

A filtered request reads at most one native candidate page of `limit` rows,
then selects rows using the public text projection and/or shared overview facts. It can return fewer than
`limit` matches, including an empty `sessions` array with a non-null
`next_cursor`. Continue until `next_cursor` is null to traverse all matching
candidates. Filtered `total_count` is always null: display a loaded-result count,
not an exact matching total. The separate overview counts endpoint retains its
exact-count contract. This bounded list design does not promise snapshot
pagination when sessions change during traversal.

Filtered cursors are opaque, portable across server restarts, and bound to the
normalized query, inclusion flags, requested representations, order, and page
limit. Keep those parameters unchanged when continuing; changing them requires
restarting without a cursor. Native unfiltered cursors and filtered cursors are
not interchangeable. Invalid or mismatched continuations return 422 without
reflecting cursor contents. Check
`capabilities.surfaces.session_overview_queries.read.enabled` before using these
filters. The capability follows the shared overview store support; unavailable
evidence still fails the read rather than silently dropping an ambiguous state.

Repeated `ids` parameters select up to 200 public session IDs, preserving first
occurrence order and omitting missing sessions. IDs cannot be combined with
listing filters or nondefault pagination. Include `overview` to fetch the same
facts for detail views without making separate status and usage requests.

`GET /api/sessions/overview/counts` returns all-scope `running_now`,
`total_count`, `failed`, and `counts_by_state`. The creation window is
`[since, until)`, defaulting to the last 24 hours. `running_now` includes older
running sessions before `until`. Evaluation trials and fork-lineage sessions
are excluded unless `include_eval=true` or `include_forks=true`. Counts traverse
all pages before the fixed upper creation boundary, stopping before enriching
newer rows; they never use the old capped `/sessions/summary` result. Both reads
are available to members and use `Cache-Control: private, no-store`.
The response also includes `tokens` (`input`, `output`, `cached`),
`tokens_complete`, and `cost` using the ordinary overview amount/currency/complete
shape. Usage counts events in `[since, until)` from eligible sessions, including
sessions created before the window. Default eval/fork exclusions apply to both
usage and session counts. Without a price book `cost` is null; token counts remain
available. The shared accounting owner combines exact priced amounts before
presentation, and incomplete pricing is explicitly marked. Reads are live,
not a cross-store transaction; a changed session incarnation or unavailable
accounting source fails the request rather than publishing a partial total.

These are observational reads of current retained evidence, not an atomic
cross-store snapshot. Sessions replaced while joining list evidence are omitted;
refresh to observe the replacement. Activity can change while pages are read.
Subagent counts cover direct retained children with matching parent-run
lineage; artifact counts use the existing session-scoped store listing and
deduplicate shared registrations by store identity. No configured artifact store
means zero; a configured store without a complete count returns 503 rather than
a partial total. Unsupported store operations return 501; unavailable evidence
returns 503, including incomplete pending-action evidence on an eligible session
or its delegated child. Cloud v1 retains its existing best-available summaries
and advances past pending-action issues; control-plane overviews and exact counts
require complete pending evidence, including for terminal sessions whose pending
state cannot be materialized. Check
`capabilities.surfaces.session_overview.read.enabled` before using the overview include. The capability describes session accounting support;
individual reads can still fail when a backing service is unavailable.

Generate typed clients from the current OpenAPI schema;
existing session listing and Cloud v1 fields remain available. Display metadata
writes are a separate API; this read does not infer titles from messages or
change the session metadata mutation contract.



### Usage windows and series

`POST /api/usage/rollup` accepts an optional `series` object with `unit`
(`hour`, `day` or `week`) and an IANA `timezone` (default `UTC`). Check
`capabilities.surfaces.usage_series.read.enabled` before requesting it. The
response includes ordered buckets with explicit UTC `start_at`, `end_at` and
usage `totals`; omitted `series` returns null and performs no series query.
Buckets contain activity counters, not estimated costs. Existing headline
pricing remains independent.

Event windows are half-open. A session created earlier still contributes its
recent activity. Tokens, model steps and tool calls sum across buckets; distinct
activity sessions do not, because one session can appear in several buckets.
Each series has at most 512 buckets within the existing 366-day window; overly
fine resolutions are rejected rather than silently widened. Hours advance in
elapsed time, including both repeated hours at a daylight-saving transition.
Days follow local midnight and weeks are seven calendar days anchored to the
first local date. First and last buckets are clipped to the requested window.
A repeated midnight uses its first occurrence; a nonexistent midnight uses its
corresponding next real instant. Fully skipped calendar days have no bucket.

`GET /api/sessions/overview/counts` also accepts exact `agent_name`,
`environment_name`, `provider_name` and `model` filters against current session
attributes. Optional `bucket_unit` and `timezone` request a batched `series` of
`total_count` and `failed` per creation-time bucket. Check
`capabilities.surfaces.session_overview_series.read.enabled` for these options.
Failures describe the current state of those session starts, not failure events
occurring in the bucket. Existing eval/fork exclusions and older `running_now`
sessions remain unchanged. This live overview observation is separate from the
store snapshot used by the event-usage rollup.

`cayu usage rollup` and `cayu usage counts` expose the same data through a running
local server (`--server-url`) or Cloud Agent (`--application`). See
[Session inspection](session-inspection.md#event-window-usage-and-start-counts).

### Home and Inbox

Home reads its last-day sessions, failures, tokens and cost from the authoritative
server counts/accounting endpoint. Eval trials and forks are excluded. Cost is
shown when pricing is configured, with an explicit partial marker when incomplete;
otherwise the summary shows input plus output tokens. Needs attention, recent
sessions and task samples are lists, never population totals. Failure links open
Session Trace with its event log filtered to `session.failed`.
Starting a session uses its own execution permission and remains available to
full-access users even when a custom store cannot supply Home overview reads.

Inbox search, kind, session and specialist filters are server-side. Count chips
and the sidebar use exact server counts; access-limited or incomplete sources show
unavailable counts rather than guessed totals. Empty cursor pages are followed;
a long scan pauses with a continuation control, stale cursors restart, and source
issues remain visible, with an inspection link when the source identifies a
session. Child delegated actions have a separate read-only panel.
A delegated action whose chain of subagents ends in a `sleep_until` wake waits on
no person, so `GET /api/pending-actions` leaves it out, and `total_count` is `null`
while any delegated action is pending. Server contract 75 adds `wake` to
`DelegatedActionReference.action_kind`; independently hosted dashboards should
regenerate their client and update their supported contract version together.
Knowledge proposals retain their source fields. Optional reasons are durably bound
to decisions; runtime audit privacy may withhold them from the public response.

### Declared task creation and replay

Hosts opt in by registering `TaskTypeSpec` values on `CayuApp` before serving:

```python
from cayu.tasks import TaskTypeSpec

app.register_task_type(TaskTypeSpec(
    name="report",
    description="Build a report",
    agent_name="researcher",
    input_schema={
        "type": "object",
        "properties": {"topic": {"type": "string"}},
        "required": ["topic"],
        "additionalProperties": False,
    },
))
```

The specialist must be currently registered when work is admitted. A declaration
with `agent_name=None` leaves execution to the configured task worker. Schemas
use Draft 2020-12 with local references only. Registration copies declarations;
mutating the caller's schema does not change the catalog. Known secret-bearing
declarations are rejected, rather than redacted into a different input contract.

### Task pages, relations, and cancellation

The Tasks page groups exact server counts into Active, Waiting, Held, Failed and
Done. Search, type and specialist filters apply on the server; Cayu's internal
tasks are hidden unless requested. Load more follows the server cursor, while
custom stores without native counts show counts as unavailable. Existing
`/tasks?assigned_agent_name=...` links keep their filter; old `task_id` links open
the canonical `/tasks/{id}` detail page.

Each task shows its authorized session links, retry attempts, prerequisites,
dependents, parent and subtasks, with explicit continuation controls. Schedule
history, input, result/error and collapsed Debug retain their public task fields.
Members can read these sections, including schedule history, but cannot create,
replay, hold, resume or cancel tasks. Cancellation of a scheduled task uses its
current schedule revision; cancelling a running task does not prove previously
dispatched effects stopped.

Create a task uses a currently declared type and its input schema. A lost response
retains the exact operation and body in the current tab; reopen the form and
recover that operation before creating another. A rejected recovery request does
not establish that the original mutation failed; its exact identity stays fenced.
Specialist sessions remain independently available while a task is unresolved,
and a known task result can be dismissed. Run again similarly uses the
server's freshly read exact replay source for a new operation, while uncertain
retries retain their original source. Reloading the
browser loses this in-memory observation handle: inspect the task list before
submitting new work. Missing task declarations or replay authority are explained
in the UI. Handler deadline failures say "Task handler ran out of time".

Task list and detail records include `graph_id` and `prerequisite_task_ids`.
`GET /api/tasks?include=counts` returns `{items, total_count, counts_by_status,
next_cursor, as_of}`. The `task_index.read` capability reports native snapshot
support. Counts cover all eligible records in one store read snapshot, without
a page cap; only `items` is bounded by `limit` (maximum 1,000). `total_count`
includes the selected `status` and repeated `statuses` filters, while the twelve
`counts_by_status` entries ignore status selection so all chips remain useful.
Counts use the existing exact aggregate decimal-string representation.
Both honor type, specialist (`assigned_agent_name`), session, parent, public text
search and `include_internal`. Pass `include_internal=false` to exclude `cayu.*`
types before counting and paging. Search matches public redacted text, never
private input, result, metadata or a secret removed from the displayed fields.

Follow `next_cursor` with the same filters; do not send `offset` with
`include=counts`. These are keyset pages, not offsets disguised as cursors.
Each response has fresh counts and an observation time; pages do not retain a
historical multi-request snapshot, so updated sort fields may move a task.
Cursors are sealed, bound to the authenticated subject/tenant and filters, and
expire after 15 minutes. The deployment's existing `inbox_cursor_key` also seals
task cursors with a distinct purpose. Share that key across server workers; a
server-generated key lasts only for that server instance. Invalid, expired or
wrong-filter cursors return 422 and require restarting the listing. Unsupported
stores return 501; an incomplete/failed read returns 503, never a subtotal.
Without `include=counts`, the existing bare list response remains available,
including offset paging. Text and scalar filters on either shape match the
current displayed public values, including redacted scalar values; hidden raw
strings do not grant membership. Native SDK list filters retain their native
semantics. Resource access bounds are always applied independently. A custom
store without native index support still serves filtered bare lists by scanning
bounded native list pages and applying public matching before offset/limit;
these pages do not claim snapshot counts. A request examines at most 4,096
records across at most 16 native pages. Exhausting that budget or seeing repeated
identities while pages shift returns 503 without partial items. Exact counts
remain unavailable for these custom stores; native exact-count scans are uncapped.

`GET /api/tasks/{task_id}/dependents` returns the same bounded envelope for tasks
whose authoritative prerequisites contain that task. It includes internal task
types because they can be real dependencies. Unknown source tasks return 404.
Members may read tasks, dependents and `GET /api/tasks/{task_id}/schedule/events`;
schedule history retains the same public redaction as other task reads.

`POST /api/tasks/{task_id}/cancel` requires full access. Idle work becomes
cancelled. Live worker work instead retains its worker/lease and a durable
cancellation request until the owner drains and commits terminal cancellation;
a 200 response does not promise that dispatched effects have already stopped.
The returned task distinguishes these outcomes. Lifecycle conflicts return 409.
Managed schedules retain their revision fence: use the existing
`POST /api/tasks/schedule/cancel` with `operation_id` and `expected_revision`;
the general endpoint refuses them. Holds and resume retain their existing rules.

Native extensions may implement `TaskStore.query_task_index` using
`TaskIndexQuery` / `TaskIndexResult` from `cayu.tasks`. The operation must visit
all eligible records in one snapshot, apply the supplied public search predicate
before counting and pagination, preserve resource bounds, and fail without a
successful result on incomplete traversal. It is a trusted operator surface,
not a new model-tool access grant. The default implementation is unsupported.

#### Task CLI

The CLI can create and control tasks on a running server:

```bash
cayu task create report --input-file report.json --operation-id report-2026-10-29 --json
cayu task hold TASK_ID --reason "Waiting for review"
cayu task block TASK_ID --reason "Missing source data"
cayu task list --server-url http://127.0.0.1:8000 --json
```

`hold` uses the task pause transition; it does not interrupt work already owned
by a worker. The server applies the same lifecycle and full-access checks as
the dashboard. `--payload-file` accepts a JSON object of hold details. The
input defaults to `{}` when `--input-file` is omitted.

`cayu task graph-create graph.json --operation-id daily-report --json` admits
a whole dependency graph atomically. For example, `graph.json` can contain:

```json
{
  "nodes": [
    {"key": "collect", "type": "report", "input": {"topic": "source data"}},
    {
      "key": "summarize",
      "type": "report",
      "input": {"topic": "summary"},
      "prerequisite_keys": ["collect"]
    }
  ]
}
```

Every type must be declared by the agent. This is `POST /api/task-graphs` with
`operation_id` and `nodes`; the response contains `graph_id` and a `task_ids`
map keyed by the submitted node keys. The `task_graph_creation` capability
reports availability. Nodes are bounded by the existing task-graph limits;
cycles, duplicate keys, missing prerequisites and invalid inputs reject the
whole request. Specialists and verified invocation origins come from server
configuration and authentication, never from the JSON file. Workers still
own execution. Graph admission uses the existing atomic task-store operation;
no task is published if a later node is invalid.

Keep the operation ID and the original input file. Retrying an identical
creation returns the same result, including after a restart. Changing a node,
its prerequisites or its declaration under the same operation ID is rejected.
Ordering nodes or prerequisite lists differently does not change the graph.
A different authenticated actor has a separate operation namespace.

Task and knowledge mutations accept `--server-url` (default
`http://127.0.0.1:8000`, including a mount prefix when present) or
`--application NAME` for a Cayu Cloud Agent using the existing Cloud login.
For a protected server, set the complete Authorization header in
`CAYU_API_AUTHORIZATION`, or name another variable with `--authorization-env`.
Do not pass credentials in the URL. HTTPS is required outside loopback.
The CLI discovers the API with a read before sending a mutation once. It does
not automatically retry mutations on timeouts, lost responses or server
errors: inspect the state, and explicitly retry an idempotent creation or Keep
with the original operation ID and unchanged input. Block, Hold and Drop have
no operation receipt; inspect their current state before deciding what to do
following an uncertain response.

`GET /api/task-types` returns `{task_types: [...]}`. An empty list means the host
has declared no dashboard-createable types. The `task_creation` capability reports
whether the configured task store supports exact creation and source selection.
Members may read the catalog and replay availability; creation and replay require
full access under the existing control-plane role policy.

`POST /api/tasks` accepts `{operation_id, type, input}` and returns `{task_id}`.
Workers execute the pending task separately; this response does not promise a
session. Reuse the same nonblank `operation_id` after a lost response. Its identity
is scoped to the authenticated subject and tenant; incidental refreshed-token
claims do not change it. Exact duplicate requests return the same task identity;
changed input or declaration conflicts. A deleted result stays fenced and returns
`task_creation_result_deleted`, rather than silently creating new work.

`GET /api/tasks/{task_id}/replay` returns `{available, unavailable_reason, source}`.
The source binds task ID, creation time, root invocation, restart count and exact
session attachment (ID and instance). `POST /api/tasks/{task_id}/replay` accepts
`{operation_id, type, source}` with that exact source. The native transaction
revalidates the source and copies its original stored input; the browser neither
resubmits redacted input nor overrides it. The new task has a fresh root and no
copied session, lease, schedule, retry policy or graph relationships. Source changes
conflict before creation. Already committed exact retries reconcile their retained
receipt before looking at the source, so deleting the source does not defeat retry.

For Session detail, `GET /api/sessions/{session_id}/task-replay` resolves a unique
task from the session's authoritative task invocation and exact current instance.
Missing or ambiguous bindings are unavailable; display metadata never grants replay
authority. Unavailable reasons also distinguish unsupported stores, undeclared
types, missing specialists, invalid input and identities that cannot cross the
configured redaction boundary. Task creation records the `http_task` source;
authenticated origin is server-derived, never a body field.

SDK callers use `CayuApp.create_task_operation(TaskCreate(...),
TaskCreationOperation(...))` with a stable task ID and the current declaration.
Native Memory, SQLite and PostgreSQL stores implement atomic creation/receipt
publication and retain deletion fences. Custom stores must explicitly implement
`create_task_operation` and `task_replay_sources_for_session` before advertising
support; falling back to ordinary `create_task` would lose these guarantees.

### Kept knowledge

`GET /api/knowledge` lists active, unexpired entries within the application's
configured knowledge namespace, labels and access scope. Members can read this
route. Pending proposals still use `/api/knowledge/pending` and the existing
Keep/Drop decisions.

`namespace`, `kind` and `label` (`key=value`) narrow the list; `kind` and `label`
repeat, and every label must match. They stay inside the configured scope: a
`namespace` or `label` value that conflicts with it answers 422 saying which.
`GET /api/knowledge/kept/{entry_id}` returns one kept entry with its text, metadata and
chunks (`max_chunks` up to 50, `max_bytes` up to 128,000), shaped like
`GET /api/knowledge/pending/{entry_id}`. An entry that isn't active, is outside the
scope, or is hidden from the caller answers 404 the same way.

Each entry includes its existing provenance fields and a public preview. A
registered model actor supplies `specialist`; a matching retained source session
supplies `source_session_id`. Missing or conflicting evidence leaves these fields
null. `kept_by`, `kept_at` and `kept_by_kind` come from the native activation
receipt for the exact entry revision, never from entry metadata. The kind is
`reviewer` for a reviewed approval or `policy` for another governed activation.
Entries without a receipt report `attribution_unavailable_reason: not_recorded`;
stores without receipt discovery report `unsupported`.

`limit` is 1–100 and `max_bytes` bounds the native listing page (default 100,000,
maximum 1,048,576 bytes). Follow the opaque `next_cursor` until it is null. Cursors
are bound to the authenticated subject, tenant and configured knowledge scope.
An access-filtered page can be empty and still have a continuation. `total_count`
is always null; a page is not a total or a stable cross-request snapshot. An entry
that cannot fit or a response exceeding the public metadata bound returns 413
instead of an incomplete successful page. An invalid query parameter answers 422
naming it and its rule, for example `Invalid control-plane request: query parameter
limit must be at most 100.`, as every bounded control-plane route does for its
query, path and header parameters (body failures still don't say which field).

`lookups` is a decimal string counting each namespace/entry identity once per
retained `knowledge.lookup` event. `last_used_at` is the latest such timestamp.
The native reader holds one stable retained-event snapshot while traversing
byte-bounded event pages completely. `usage_through_sequence` describes that
snapshot's maximum sequence; it is not a globally committed insertion frontier. Counts describe retained history, not events pruned from storage.
They are null with `usage_unavailable_reason` when access is limited, the event
reader is unsupported, or a source cannot be read completely. No capped subtotal
is presented as an exact count.

Full-access callers can retire an entry with
`POST /api/knowledge/{entry_id}/retire` and
`{"expected_revision": <the displayed revision>}`. The native lifecycle owner
atomically transitions that exact active revision to archived, preserving its
content and durable evidence. A stale revision or incompatible lifecycle returns
409. The response contains the entry ID, new revision and archived status. This
operation does not promise idempotent acknowledgement replay: after a lost
response, inspect canonical state; repeating the old revision can return 409.
Members cannot retire entries. Retirement does not replace Keep or Drop for
pending proposals.

`cayu knowledge list` and `cayu knowledge show ENTRY_ID` read the same entries
from the command line, from a local store (`--sqlite`, `--postgres`, or the
project's store, opened read-only with every namespace visible) or from a Cayu
Cloud Agent through these routes (`--application NAME`, see
[reading an Agent's state](cayu-cloud.md#reading-an-agents-state-from-the-cli)):

```bash
cayu knowledge list --kind denial --label agent=reviewer
cayu knowledge list --status pending --namespace reviews --json
cayu knowledge show ENTRY_ID
```

To review proposals against a running server:

```bash
cayu knowledge list --status pending --server-url http://127.0.0.1:8000 --json
cayu knowledge keep ENTRY_ID --operation-id review-2026-10-29 --json
cayu knowledge drop ENTRY_ID --json
```

Keep calls the existing approval workflow with an explicit idempotency key;
repeating that key returns the approved-entry projection using the durable
activation receipt. Keep records authenticated reviewer attribution. Drop archives
a pending proposal through the existing rejection workflow; it does not record
reviewer attribution. Both preserve the server's review scope and permissions. Local store options
remain read-only; mutations always go through the running server.

`--status` is `active` (kept, the default) or `pending`, and `archived` from a
local store. `--kind` repeats for any of several kinds and `--label KEY=VALUE`
repeats to require several labels. A page with more after it prints the
`--cursor` to continue with. `show` prints an entry of any status with its text,
and `--json` adds its chunks. The CLI applies the filters to what an Agent
returns too, so an Agent on a Cayu without these filters still answers
correctly.

### Inspecting specialists and tools

Specialist cards open their model/provider configuration, system-prompt presence,
metadata, provider options and thinking settings. Recent sessions are bounded
per-specialist reads, with links to the full filtered Session and Tasks lists.
Prompt contents are not exposed by this inventory read. Members can inspect
these pages; starting a session still requires full access and runtime support.

Tools retains declaration variants and reads usage from the control plane. Counts
are retained-history aggregates by public tool name, shared across declarations
with that name, rather than per-specialist attribution. Unknown or unavailable
usage is shown explicitly rather than as zero. Expand a tool for its full schema,
source, approval policy and cursor-paged recent calls. Calls link to their exact
Trace event with a session incarnation; a replaced session cannot silently
substitute a different call.

Specialist activity and tool usage refresh while the page is visible. Recent tool
calls refresh a bounded latest window; explicitly loaded older calls retain their
snapshot instead of being replayed by each background refresh. Start from latest
calls to read one continuous fresh cursor chain when newer calls may fall between
the latest window and loaded history.

### Knowledge page

Knowledge shows proposals first, with the same Keep/Drop decisions and optional
reason as Inbox. Revision-matched proposal titles and full text are visible before deciding. Proposal evidence retains source metadata,
chunks and recorded metadata. Kept entries show public previews, recorded
activation attribution, scoped usage and source-session links when available.
Missing attribution or usage is explicitly unavailable, never inferred or zero.

Knowledge Reviews, Inbox and Session action lists share one active cursor window: decisions and session progress refresh that window only. Refresh actions/proposals returns to newest; loading more replaces the displayed window. Exact Inbox count chips still come from the separate server count owner.

Lists follow opaque cursors, including empty scoped pages. Automatic traversal
pauses every 25 pages; pages with source issues also stop for inspection before explicit continuation. Source diagnostics and session links remain visible on that page. Load more continues. Only the current page is displayed
and searched; loading another page replaces it. Counts label loaded entries,
not totals. Only the newest page polls automatically; older pages show that
refresh is paused. An empty older page does not imply that the queue is empty; source failures also prevent an all-clear message. Refresh returns to the newest page after history, an expired
cursor or an error. Members can review and read. Remove
is full-access and confirms the displayed revision; after an unconfirmed or
conflicting removal, refresh canonical state before starting a new intent.

### Tool display labels

`GET /api/agents` includes optional `display` metadata on each tool declaration:
`{verb, noun, plural}` or null. Built-in tools declare their own labels, including
renamed built-ins. Custom tools can supply `ToolSpec(display=ToolDisplay(...))`;
`ToolDisplay` is imported from `cayu.tools`.
For class-level declarations checked by `cayu check`, use a literal dictionary
(`display={"verb": "Read", "noun": "file"}`); the static declaration checker does
not evaluate nested `ToolDisplay(...)` constructors.
MCP and provider-hosted tools need not supply labels. Use the tool name when no
label is declared rather than guessing from its name. The read-only
`tool_display` capability advertises this metadata. Server contract 64 adds this
nullable field; independently hosted dashboards should regenerate their client
and update their supported contract version together.

A noun uses singular for one call and the declared plural (or noun plus `s`) for
other counts: `{verb: "Read", noun: "file"}` describes “Read 9 files”. Without a
noun, use the verb and call count: `{verb: "Searched"}` describes “Searched 10
times”. A missing label falls back to “Used query_ledger 2 times”. These strings
are public presentation metadata, redacted by the same owner as other tool
summaries. They do not change approval, execution identity or model tool schemas.

```python
from cayu import ToolSpec
from cayu.tools import ToolDisplay

spec = ToolSpec(
    name="query_ledger",
    display=ToolDisplay(verb="Queried", noun="ledger"),
)
```

## Environment inventory

`GET /api/environments` and `GET /api/environments/{environment_name}` expose
registered environments without starting their runners or factories, connecting
to MCP servers, or reading files from artifact storage. Encode the environment
name in the detail URL. Existing authentication and read-only role permissions
apply; responses use `Cache-Control: private, no-store`.

Each environment includes `mcp_server_names` from its registered MCP declarations
alongside `mcp_server_count`. The names pass through the agent's secret redaction;
commands, connection URLs, environment variables and headers are not included.
`artifact_store_kind` is `local_disk` or `s3` when the store declares that kind,
`unknown` for a configured store without recognized metadata, and `null` when no
artifact store is configured. It is a storage description, not an availability
check. The server does not infer the kind from a class name, store ID or path.

`session_count` is the exact number of currently retained sessions with that
stored environment name, serialized as a decimal string. It includes all
statuses, eval trials and forks. Deleted sessions are absent. This inventory
count is distinct from the Home start-window count and its default exclusions.
`last_activity_at` is the maximum durable session activity timestamp; a metadata
edit's `updated_at` alone does not advance it. Empty environments have count
`"0"` and no last activity. `activity_as_of` identifies the store snapshot.

Native stores aggregate at most 500 requested environment names per read without
loading session histories. Larger catalogs use several batches; each row keeps
its own batch's timestamp. These are consistent per-batch observations, not a
claim of one atomic snapshot across an arbitrarily large catalog. If a configured
store lacks activity support, all three activity fields are `null`; missing data
is never shown as zero. A failed supported read fails the whole response, rather
than publishing a partial catalog with incomplete counts.

`capabilities.surfaces.environment_details.read.enabled` advertises native
activity support. MCP and storage declarations remain available independently.
Use `cayu environment list` or `cayu environment show NAME` with `--server-url`
or `--application` for the same data, in JSON or table form.
