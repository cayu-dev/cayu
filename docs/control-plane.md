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
Usage under Improve and Specialists, Knowledge, Environments, Artifacts and
System under Agent. Items depend on the server capabilities and viewer access.
Existing URLs remain valid: Inbox uses `/pending-actions`
and Specialists uses `/agents`. Search sessions from the sidebar; `Ctrl+K` or
`Cmd+K` focuses search, and Enter opens the session index with its `q` filter.
The sidebar can be collapsed on desktop or opened as a drawer on mobile.
Escape closes the mobile drawer and returns focus to its opener.

Use the theme button to cycle System, Light and Dark. The choice is saved in
this browser; System follows the operating system appearance. The external
initialization script applies the choice before styles load under the existing
content security policy, including when the dashboard is mounted below a path.

To check the member experience, sign in through the configured identity proxy
with a role in `ControlPlaneRoles.read_and_decide`, then open the dashboard.
The authenticated `/api/app` response drives the shell's access presentation.
Members can browse sessions and pending work, approve or deny a pending tool
approval, answer a pending question, and approve or reject knowledge proposals
when the corresponding server capability is available. Open Inbox and follow
a pending item's session link to make its decision.

Evals, System and Run are omitted for members; opening those URLs directly
shows an access explanation before their page requests start. Full-access
mutation controls elsewhere are hidden or disabled with an explanation.
Browser-control and evaluation-authoring panels do not mount for members.
These presentation rules also follow custom role policies; they do not grant
permission. The server independently authenticates and authorizes every read
and decision. The server-contract gate still precedes page reads, and an
unavailable app identity does not grant full access.

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
tasks and New Run identify the specialist they use. A **subagent** remains a child
session started with `SubagentTool`.

Python names (`CayuApp`, `register_agent`, `AgentSpec`), API fields, `/agents`
routes and existing authentication configuration remain unchanged. In particular,
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
