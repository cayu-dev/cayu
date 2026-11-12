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
and completion summaries, and preserves the existing filter links. A row with no
title or labels is named from its first message, which comes with the overview
(`include=first_message`) when the server advertises it, or from one bounded
transcript read per such row against an older server. Eval trials and
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
model/provider links carry those session filters into Sessions. The chart shows
session starts and their current failures, excluding forks and eval trials, with
the specialist, environment, provider and model filters applied. Parent, budget
and label filters still apply to the headline; the chart is unavailable for those
filters. Models show usage and estimated costs when pricing is configured. Top
sessions are ranked by cost when estimates are complete and comparable, otherwise
by tokens with an explanation. Pricing coverage, exact windows and raw billing
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
text-search tools are unavailable. MCP shows declared server names, and file
storage shows its declared kind. Native activity support adds retained-session
counts and last activity, including eval trials and forks, with the observation
time available in the detail page. Unsupported activity is shown as unavailable;
an empty supported environment shows zero sessions. These declarations do not
imply live MCP or storage health checks.

Artifacts keeps search, sort, filters and paging in the URL. Search matches public
file metadata and producing-session titles across the selected stores before paging.
Sort by newest, oldest, filename or size. A preview link uses `artifact` and
`preview_store_id`; `artifact_store_id` independently filters the list. Existing
store-scoped preview links remain supported. The drawer offers Download and Open raw
where supported, with internal identities in Debug. Session and environment links
remain within the configured dashboard mount. Usage and Artifacts respect their
server capability gates. Artifact titles come from a bounded store projection;
missing or untitled producers display “Untitled session” without an overview read.

System is for full-access viewers. It shows agent setup, disabled-feature reasons
and Evals readiness; capability matrices and raw diagnostics stay in Debug.
It reports the explicit local, Cloud or embedded deployment kind alongside the
name and sign-in requirements. The `deployment_kind` capability advertises the
field; kind comes from startup configuration, never names or credentials.
`cayu diagnostics system --server-url URL --json` and
`cayu diagnostics system --application AGENT --json` expose the same snapshot.
See [deployment declarations](server-configuration.md#deployment-kind).
Member reads on the other pages still require server authorization.

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
newer rows; they never use the old capped `/sessions/summary` result. Each store
page of 200 sessions is enriched with the overview's page reads, so counts cost a
fixed number of store reads per page, not per session. Both reads
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

Add `include=first_message` (together with `include=overview`) to fill each
overview's `first_message`: the text parts of the session's first user message,
redacted exactly as `GET /api/sessions/{id}/transcript?role=user&limit=1`
redacts them and then as one joined string, trimmed, and cut to 256 characters
with a trailing ellipsis. It is null when that message has no text (images,
files and withheld peer content are skipped) or the session has no user
message, and always null without the include. The server reads every listed
session's first user message in one store read
(`SessionStore.query_first_transcript_records`), which returns what the
transcript route's store query returns and falls back to that query, with its
data bounds, while scoped access bounds apply. It drops a row whose session is
replaced meanwhile. Without `overview` the include returns 422. Check
`capabilities.surfaces.session_overview_first_message.read.enabled` before
sending it; an older server rejects the unknown include value with 422.
If a preview read fails, that row returns null with `first_message_unavailable: true`;
other rows remain available: when the page read fails, the server reads each row on
its own, at most four at a time, so only the failing rows lose their preview.
Empty or unrequested previews keep the flag false. `cayu session list --overview`
shows the same overviews and previews from the CLI (see
[Session inspection](session-inspection.md#session-overviews)).

Generate typed clients from the current OpenAPI schema;
existing session listing and Cloud v1 fields remain available. Display metadata
writes are a separate API; this read does not infer titles from messages or
change the session metadata mutation contract.



### Usage costs and session order

`POST /api/usage/rollup` accepts `session_order: "cost"` together with
`session_group_limit` (1–100). The default order is `tokens`. Check
`capabilities.surfaces.usage_cost_details.read.enabled` before requesting cost
order. Pricing uses the optional request `pricing` price book; reported provider
charges remain separate from these estimates.

The server collects at most 1,000 current matching sessions in one native
snapshot and prices every candidate before selecting the returned top N. The
snapshot also supplies each row's `agent_name` and `title`, avoiding additional
session reads. These descriptive values are nullable when unavailable or outside
the bounded display contract (256 characters); configured redaction still applies.
Usage is attributed by event time and actual normalized provider/model identity,
including terminal hosted tools and auxiliary attempts, rather than by a
session's configured model. Price schedules use each event's UTC date.

`session_ranking` explicitly reports `requested_order`, `effective_order`,
`currency`, and `unavailable_reason`. Complete single-currency costs permit cost
order; ties use tokens, model steps, then session ID. Known zero cost remains
zero. Missing prices or usage, incomplete pricing projections, mixed currencies,
more than 1,000 candidates, or an oversized candidate projection return token
order with an explicit reason. Clients must label this **Top sessions by tokens**,
not cost. An oversized candidate projection may cause one fresh token-only read;
the first snapshot is discarded entirely. Missing pricing is
`pricing_not_configured`; incomplete prices are `prices_incomplete`; projection
bounds use `pricing_projection_incomplete`, `candidate_limit`, or
`candidate_byte_limit`; currency disagreement is `mixed_currencies`.

With pricing, `provider_cost_breakdown` and `model_cost_breakdown` align with the
existing usage groups and include their explicit omitted-group cost remainder
when the native projection has one. A sampled native group list remains marked
as sampled; it does not invent an exact omitted-group count.
Each group retains pricing coverage, separate currencies, and unpriced reasons;
unknown cost must not be displayed as zero. If rejected pricing metrics have
lost their provider/model attribution, both dimension cost sections explicitly
report unavailable/truncated estimates for all groups and their remainder;
headline and session unpriced accounting remain available. The existing session cost breakdown
aligns with the selected session order and retains its aggregate remainder.
Neither this report nor its ordering changes budget admission.

The Usage dashboard requests cost order when this capability is enabled, including
for member viewers. It shows provider/model estimates and the returned session
titles and specialists without per-session reads. Currency codes, partial coverage
and omitted-group estimates remain visible; missing estimates are unavailable,
not zero. When the capability is disabled, the page requests token order and
explains that cost ranking is unsupported. Authorization denials discard cached
Usage reports and histograms until a fresh successful read; ordinary transient
refresh failures retain the last authorized snapshot with a stale-data warning.

The same data is available through `cayu usage rollup --server-url URL` or
`--application NAME`. Add `--price-book prices.json --top-sessions 20
--session-order cost`; JSON preserves the HTTP result and table output labels
cost versus token order and explains fallback. Without `--price-book`, the CLI
sends Cayu's bundled price book, so costs appear as estimates at public list
rates. The dashboard does the same unless `dashboard_config` sets `priceBook`;
set it to `None` to show tokens only. A request to the API that names no price
book still returns a null `cost`. Use your own price book for negotiated or
gateway rates.

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

The same endpoint accepts `parent_session_id`, `causal_budget_id`, repeated
`label=key=value` exact labels and repeated `label_selector` expressions (`key`,
`!key`, `key in (a,b)`, `key notin (a,b)`, `key=v`, `key!=v`, comma-separated to
require several). Labels and selectors use the `GET /api/sessions` syntax and
bounds. They build the same current-session filter that `POST /api/usage/rollup`
takes as `session_filter`, and every store applies it with the same query, so the
start counts and the rollup select the same sessions. Parent and budget IDs are
matched exactly, as the rollup matches them. Invalid filters return 422 without
echoing the input. Check
`capabilities.surfaces.session_overview_count_filters.read.enabled` before sending
these; an older server ignores unknown parameters and would return unfiltered
counts.
Failures describe the current state of those session starts, not failure events
occurring in the bucket. Existing eval/fork exclusions and older `running_now`
sessions remain unchanged. This live overview observation is separate from the
store snapshot used by the event-usage rollup.

The Developer dashboard uses one batched start-count request for its Sessions
chart and requests event activity in the headline rollup. Both use the same exact
window, the viewer's IANA time zone, and the specialist, environment, provider,
model, label, label-selector, parent-session and causal-budget filters. Bars show
session starts and their current failed state; hover and
accessible table details show **activity tokens**, including older sessions active
in that period. These populations intentionally differ. There is no per-bucket
cost estimate. Unsupported series capabilities leave the headline usable with an
explicit chart/detail unavailable state. Against a server without
`session_overview_count_filters`, label, parent-session and causal-budget filters
still apply to the headline and the start chart says it is unavailable for them,
rather than drawing unfiltered bars. A failed refresh keeps the last successful
numbers with a visible warning.

`cayu usage rollup` and `cayu usage counts` expose the same data through a running
local server (`--server-url`) or Cloud Agent (`--application`). Both take
`--agent`, `--environment`, `--provider`, `--model`, `--parent-session`,
`--causal-budget`, repeated `--label KEY=VALUE` and repeated `--label-selector EXPR`,
parsed with the server's rules before any request. `counts` with the last four
requires `session_overview_count_filters` and fails against an older server
instead of printing unfiltered counts. See
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
List rows leave out each task's error but carry its `error_code` and an
`error_summary`: the error's type and the first line of its message, such as
`GitError: couldn't find remote ref refs/pull/217/head`, redacted the way the
detail's error is and cut to 240 characters. Home and Tasks show it on failed
rows as "Failed: …"; the task's detail has the full error.
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
An optional body `{"reason": "..."}` (at most 4,096 characters) is recorded as the
task's error, `{"code": "task_cancelled", "message": reason}`.
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
cayu task resume TASK_ID
cayu task cancel TASK_ID --reason "Wrong input"
cayu task replay TASK_ID --operation-id report-2026-10-29-again --json
cayu task list --server-url http://127.0.0.1:8000 --json
cayu task wait TASK_ID --server-url http://127.0.0.1:8000 --json
```

`wait` reads the task until it ends and exits with how it ended
([exit codes](project-workers.md#inspect-tasks)). `hold` uses the task pause
transition; it does not interrupt work already owned by a worker. The server
applies the same lifecycle and full-access checks as the dashboard.
`--payload-file` accepts a JSON object of hold details. `resume` returns a held,
blocked or attention-needed task to the queue (`POST /api/tasks/{task_id}/resume`).
`cancel` uses `POST /api/tasks/{task_id}/cancel`, so a task a worker is running shows
`cancellation_requested` until its worker stops. `replay` queues a new task of the same
declared type and input, as the dashboard's replay does: it reads
`GET /api/tasks/{task_id}/replay` and refuses with its `unavailable_reason` (such as
`task_type_not_declared`) when the agent can't replay the task. Keep `--operation-id`
for an exact retry: the CLI keeps each replay request it sends (under
`$XDG_STATE_HOME/cayu/task-replays`, by default `~/.local/state`) and a retry with the same
operation ID sends it again unchanged, so a replay whose answer was lost returns the task it
created even after the source task's proof has changed. The kept request is dropped only when
the route refuses it (HTTP 400, 409, 413 or 422); after any other failure it stays for the
retry. `resume` accepts a task that goes back
to wait for its prerequisites or group (`waiting_dependencies`, `waiting_group`). With `--application NAME` each command reaches a Cayu Cloud Agent. The input defaults to
`{}` when `--input-file` is omitted.

#### Scrubbing stored failure text

A task's `error` is redacted when it is written, with the redaction of that time. After a
redaction fix, such as a credential shape that exception diagnostics now mask, records
written before it still hold the text. Two full-access routes apply the current
redaction again: the application's registered secrets first, then credential shapes
(`Authorization` values, bearer tokens, Git `x-access-token` credentials and prefixed tokens
such as GitHub's), the same parts exception diagnostics use.

- `POST /api/tasks/{task_id}/scrub-error` with an optional `{"dry_run": true}` scrubs one
  task's `error`, the `error` a pending cancellation keeps in its `status_payload`, and
  every terminalization, interrupted-handoff and retry-settlement receipt that keeps a copy
  of the task, in one store transaction, so a worker's retried terminalization still replays
  against the scrubbed row. It answers `task_id`, `applied`, `dry_run`, `fields` (`error`,
  `status_payload.error`), `masked_values`, `receipts` and `skipped_reason`; never the text.
  A task is left unchanged, with a `skipped_reason`, when a worker holds its lease
  (`leased`), when its error is a session's recorded runtime failure (`session_failure`:
  the session's append-only events and checkpoint bind that payload exactly, so a rewrite
  would make the session's failure replay conflict), or when a cancellation reconciliation
  receipt binds a digest of it (`cancellation_reconciliation`). An
  unknown task answers 404, a store without support 501.
- `POST /api/tasks/scrub-errors` with `{"dry_run", "after_task_id", "limit"}` (`limit` 1-100,
  default 100) does the same for one page of the tasks that keep failure text, in task-id
  order. It answers `examined`, the `tasks` it changed, would change or skipped, and
  `next_after_task_id` to continue with, null once the sweep is done.

The scrub doesn't change `updated_at`. Each field it changed records the scrub under
`cayu_scrub`: `{"scrubs": [{"scrubbed_at", "redaction", "masked_values"}]}`, the latest 20.
Each applied scrub also writes one JSON line to standard error, `{"event":
"cayu.task.failure_scrubbed", "task_id", "fields", "masked_values", "receipts", "subject",
"at"}`; Cayu Cloud keeps it with the Agent's logs. Neither holds the masked text. Members
can't use these routes. Task events (`task.failed` and the others) carry no failure text, so
there is nothing to scrub there; a session's own events are append-only and aren't changed.

From the command line:

```bash
cayu task scrub-error TASK_ID --dry-run --application my-agent
cayu task scrub-error TASK_ID --application my-agent
cayu task scrub-error --all --dry-run --server-url http://127.0.0.1:8000 --json
cayu task scrub-error --all --sqlite .cayu/cayu.db
```

With `--server-url` or `--application` the command calls these routes, and `--all` follows
the pages to the end (`--after TASK_ID` continues an interrupted sweep). With a local store
(`--sqlite`, `--postgres`, or the project's store) it opens the store for writing and scrubs
it directly; a bare store knows no registered secrets, so only credential shapes are masked
there. Output names tasks, fields and counts, never the text; `cayu task show TASK_ID` shows
the scrubbed error.

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
Keep/Drop decisions. The pending list pages the same way: follow its opaque
`next_cursor` with `cursor` until it is null. A pending cursor is bound to the
caller, the list's filters and the review scope, and expires after 15 minutes.

`namespace`, `kind` and `label` (`key=value`) narrow the list; `kind` and `label`
repeat, and every label must match. They stay inside the configured scope: a
`namespace` or `label` value that conflicts with it answers 422 saying which.
`GET /api/knowledge/kept/{entry_id}` returns one kept entry with its text, metadata and
chunks (`max_chunks` up to 50, `max_bytes` up to 128,000), shaped like
`GET /api/knowledge/pending/{entry_id}`. An entry that isn't active, is outside the
scope, or is hidden from the caller answers 404 the same way.

`GET /api/knowledge/archived` lists archived entries (kept entries archived and
proposals dropped), with the same filters, `limit`, `max_bytes` and sealed `cursor`;
each row adds the reviewed `decision` that published its archived revision (`archive`
or `drop`, null for one archived by revision alone), `decided_by` and the redacted
`reason`. `GET /api/knowledge/archived/{entry_id}` returns one archived entry shaped
like the kept detail, its `metadata` holding the decision. Both need full access, as
retiring does, so members get 403, and both read through the review scope: the
application's knowledge scope plus the archived revisions of the entries it admits
(`knowledge_review_scope`). A server whose knowledge scope admits only active and
pending entries, as a scaffolded project's does, still shows what it archived, while
search, recall and the agent's knowledge tools still see no archived entry.
The detail routes apply the application's redactor to complete content before
bounding chunks and return `content_redacted: true`. This flag confirms redaction
was applied, including when no secrets matched; it does not assert that content is
secret-free. Pending-list previews likewise redact before shortening the text.
Clients that display pending content can require the flag to distinguish older
Runtime versions that returned pending content without redaction.

Each entry includes its existing provenance fields and a public preview. Both
lists, kept and pending, carry the entry's `title` and, in `text_preview`, the
start of its text, whether or not it has a title: at most 200 characters, cut on a
word boundary and ending with `…` when cut, so two entries with similar titles can
be told apart from the list. The text is redacted whole before it is cut, so the
preview never shows more than the entry's detail route would. Older Cayu servers
put the title in `text_preview` when the entry had one. A
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
The native reader counts inside one stable retained-event snapshot: PostgreSQL
counts only the page's entries in one query, and other stores traverse
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
form does not promise idempotent acknowledgement replay: after a lost
response, inspect canonical state; repeating the old revision can return 409.

Give a `reason` (up to 4,096 characters), an `Idempotency-Key` header, or both, and
the retirement is recorded: the archived revision carries the reviewer (the verified
`AuthContext.subject`, as for Keep) and the redacted reason in its
`cayu:review_archive` metadata, bound to the publication receipt of that key. The
response then also returns the recorded `reason`. `expected_revision` becomes optional;
without it the server fences against the revision it reads. A retry under the same key
and the same reason replays the retirement, returning the same revision even though
the entry is now archived; another reason under that key, or a different revision,
returns 409. Members cannot retire entries. Retirement does not replace Keep or Drop
for pending proposals: a pending entry returns 409.

`POST /api/knowledge/{entry_id}/approve` and `/reject` take an optional JSON body
`{"reason": "..."}`, and both take `Idempotency-Key`. Keep records the redacted reason
in its activation receipt's reviewer annotations, where the inbox's Keep records it.
A Drop with a reason or a key is a reviewed rejection, the same as the inbox's Drop:
the archived revision records the reviewer and reason in its `cayu:review_rejection`
metadata, and a retry under the same key replays it. The inbox then shows that proposal
as dropped, with its reviewer and reason, once it verifies the drop's publication
receipt. A Drop with neither archives the
proposal as before, without attribution. Both responses carry the recorded `reason`,
or null.

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

To review proposals, and archive kept entries, against a running server or a Cloud
Agent (`--application NAME`):

```bash
cayu knowledge list --status pending --server-url http://127.0.0.1:8000 --json
cayu knowledge keep ENTRY_ID --operation-id review-2026-10-29 --reason "Matches the runbook" --json
cayu knowledge drop ENTRY_ID --operation-id drop-2026-10-29 --reason "Duplicates ENTRY_ID2" --json
cayu knowledge archive ENTRY_ID --reason "The worker now runs three processes" --json
```

Keep calls the existing approval workflow with an explicit idempotency key;
repeating that key returns the approved-entry projection using the durable
activation receipt. Keep records authenticated reviewer attribution. Drop archives
a pending proposal through the rejection workflow. Archive retires a kept (active)
entry, so `search_knowledge` no longer finds it, and requires `--reason`; `--revision N`
archives only if the entry is still at revision N. All three take `--operation-id`
(required for keep) and `--reason`; with either, drop and archive record the reviewer
and reason on the archived revision, and repeating the same command with the same
operation ID replays it instead of reporting a conflict. Archive needs full access; on
a Cloud Agent the CLI uses the Agent's operator login, which has it. An Agent on a Cayu
from before reasons makes the keep or drop without one, and the CLI warns that it
wasn't recorded. All preserve the server's review scope and permissions. The server's
knowledge scope needn't admit archived entries: with a scaffolded project's (active and
pending only) drop and archive still record the decision, read it back through the
review scope, and replay it when repeated with the same operation ID. The inbox answers
a drop from the decision it just recorded, and a repeated Drop replays the settlement
(`replayed: true`, decision `drop` and when) to anyone the inbox access rule lets see the
proposal. Who dropped it and why go only to that person and to full-access callers; other
members get the settlement without them. The item itself stays out of the inbox and
`GET /api/inbox/{item_id}`, since the scope can't read it. Local store options remain
read-only;
mutations always go through the running server.
`cayu knowledge show` prints who archived or dropped an entry and why (`archived_by`,
`archived_reason`, `dropped_by`, `dropped_reason`), from a local store or, with full
access, from a server or Cloud Agent; an Agent on a Cayu from before the archived routes
shows none.

`--status` is `active` (kept, the default), `pending` or `archived`; from a server or
Cloud Agent, `archived` needs full access. `--kind` repeats for any of several kinds and `--label KEY=VALUE`
repeats to require several labels. The table's preview column shows an entry's
title, or the start of its text when it has none; `--json` and `--jsonl` give
each entry's `title` and `text_preview`, the start of its text, as the routes
return them (a local store's text is redacted the same way before it is cut).
A page with more after it prints the
`--cursor` to continue with, for every status. An Agent on a Cayu whose pending list
has no cursor says only that more proposals exist; deploy it with a newer Cayu to
page past them. `show` prints an entry of any status with its text,
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

An environment registered with `register_environment_factory` has `is_factory: true`
and `factory_type`, the factory's class name. Its runner, workspace, binding, vault,
proxy, knowledge store and MCP servers come from the environment the factory creates
for each session, so the inventory reports them as `null` (or zero MCP servers)
rather than starting the factory. Only `artifact_store_id` and `artifact_store_kind`
can be set, from the store passed at registration. A session's
`environment.factory.completed` and `environment.binding.completed` events record
what that session got. `factory_type` is `null` for an environment registered
directly.

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

The dashboard's Environments list and detail page use these inventory fields
for retained-session counts, last activity, declared MCP names and file storage
kind. Counts include eval trials and forks; the All sessions link includes them
as well. The detail page shows the activity observation time, with exact timestamps
in Debug. Recent sessions and files remain separate, bounded lists. Missing native
activity support is shown as unavailable, and an empty supported environment shows
zero sessions. Refresh failures retain the last inventory with a visible warning;
authorization failures discard the shared cached inventory until a successful read replaces it. MCP names and storage descriptions remain
visible independently of activity support and do not imply a live connection or
storage health check.

Environment links preserve exact registered names. The special names `.` and `..`
use `/environments?name=.` and `/environments?name=..` so browser path normalization
cannot discard them; other names keep the existing named detail URLs.

### Shared environment allocations

A workflow run whose steps share one sandbox records it in the run's checkpoint
(see "Crashes and resumed runs" in
[environment factories](environment-factories.md)). A run that crashed and is
never resumed keeps that sandbox. Two routes let an operator find and release it:

- `GET /api/shared-allocations` lists the records of up to `limit` workflow runs
  (1 to 200, default 100), newest run first, with `next_cursor` for the next
  page. A page can be short or empty and still have a cursor. `workflow_run_id`
  reads one run and `environment_name` keeps one environment's. Each allocation
  has `allocation_id`, `workflow_run_id`, `workflow_name`, `environment_name`,
  `agent_name`, `scope`, `scope_id`, `state` (`creating`, `live`, `orphan` or
  `reaping`), `reconnectable`, `attempt_id`, `claim_generation`, `created_at`
  (null for older records), `run_state` (`running`, `stopped` or `ended`) with
  `run_state_reason`, and `metadata`: the factory result's non-secret metadata
  with the application's secrets redacted. Reconnect metadata and binding state
  are never returned. `inactive_for_seconds` (default 900) sets how long a run
  must have journaled nothing to count as stopped.
- `POST /api/shared-allocations/release` takes `{"allocation_id": ...,
  "mode": "discard" | "copy_back", "force": false, "inactive_for_seconds": null}`
  and returns the allocation as it was, `mode`, `forced` and a `message`.
  `discard` disposes of the sandbox without copying back, as recovery does;
  `copy_back` reconnects, copies the work to the source workspace and then
  disposes of it. It answers 404 for an allocation that isn't recorded,
  including one already released, and 409 when it refuses: the server process
  runs the workflow run itself, the run looks like it is running elsewhere
  (unless `force`), the environment isn't registered, the record changed while
  it was read or another release holds it, or a copy-back can't apply. A failed copy-back also answers 409
  and keeps the record fenced for another try.

Neither route is in the read-and-decide role list, so both need full access
(developer or admin), like the other session mutations. Responses use
`Cache-Control: private, no-store`. `capabilities.surfaces.shared_allocations`
advertises both; an older server has no such surface. The CLI is
`cayu environment allocations list` and `cayu environment allocations release
ALLOCATION_ID --discard` (or `--copy-back`, `--force`), with `--application`
for a Cloud Agent, `--server-url` for a running server, or neither to use the
project's application. The dashboard has no view of these yet.


### Global artifact queries

`GET /api/artifacts` accepts optional `q` (at most 256 characters) and `order_by`:
`created_at_desc` (default), `created_at_asc`, `filename_asc`, `filename_desc`,
`size_bytes_desc`, or `size_bytes_asc`. Search compares a literal substring after locale-independent Unicode lowercasing
of the query and public filename, kind, MIME type, specialist, environment, producing-session title
or artifact ID. File contents and private metadata are never searched. Redaction
precedes matching. Sort ties use store ID and artifact ID in the same direction.
Existing store, owner, specialist, environment and session filters apply before
search, sorting and paging. `session_title` is nullable on each returned row.

The `artifact_search.read.enabled` capability requires a native batched title
projection, at least one artifact store, and scan support from every registered
artifact store. A query selecting any
unsupported store returns 501 before scanning; select a supported store explicitly
with `artifact_store_id` in a mixed deployment. Basic newest-first listing without
`q` remains available for extensions that only implement `list`, including unknown
counts. It uses each store’s bounded listing and resolves titles only for the returned
page. Unavailable or incomplete inventories never return a partial global result.

Local and S3 stores implement the optional operator-only
`ArtifactStore.scan_metadata(consume, ...)` contract. Consumers receive awaited
batches of at most 500 immutable metadata records; the returned integer is exactly
the number delivered. Implementations must stop on callback failure/cancellation,
close their readers, and wait for dispatched threads before closing an iterator.
A scan counts records before its optional filters and may inspect one extra record
to detect overflow. Each identity occurs once; unreadable or incomplete inventories
raise instead of returning a successful count. Custom stores opt in with
`supports_metadata_scan = True` only when they meet that contract. Model-facing
artifact access does not gain this operator method.

For text search or a nondefault sort, the HTTP query caps scanned metadata at
100,000 records across selected stores,
uses batches of 100 and retains at most `offset + limit` matches. Its title cache
holds at most 5,000 IDs; native Memory, SQLite and PostgreSQL title reads return
only the requested title fields, at most 500 IDs per call. Query bound exhaustion
returns 503; selecting one store can narrow the inventory. These are observational
reads, not a frozen snapshot across stores: concurrent saves or removals can change
later pages. The supported pagination bounds remain `limit` 1–500 and `offset`
0–10,000.

The CLI uses the same protected API, filters and pagination:

```sh
cayu artifact list --server-url http://localhost:8000 --query report --order-by filename_asc
cayu artifact list --application my-agent --artifact-store-id files --limit 25 --offset 25 --json
```

Use `--session`, `--agent`, `--environment` and `--scope` for owner filters. JSON
retains exact store/file identities, nullable titles, unknown counts and continuation
fields; table output shows titles and the next offset.

Tool declarations on `GET /api/agents` optionally include `policy_conditions` and
`policy_conditions_complete`. These are redacted explanations of maintained Runtime
policy classes, not authorization results. Private allowlist values and deny patterns
are never published; custom policies, rules and guards are explicitly incomplete.
Provider controls, grants, mandatory controls, hooks, argument validation and execution
admission still apply. The explanations are not added to Cloud registry uploads.

`GET /api/sessions/display?ids=<public-session-id>` returns only validated, redacted
Runtime-authored titles with public session IDs, stable run IDs and the session-store
generation. Repeat `ids` for the requested sessions. It reads no prompt or transcript,
uses the existing control-plane read authorization and returns `private, no-store`.
Missing sessions are omitted; null titles mean no valid public title is available.
A store without stable identity returns 501; an identity change during the read returns
409. Consumers must match both generation and run identity before attaching a title
to previously reported activity.
