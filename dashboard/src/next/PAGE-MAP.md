# Developer dashboard page map

The production Developer dashboard uses `src/main.tsx`, `src/app/router.tsx` and
`src/app/layout.tsx`. The shell supplies `api/real.ts`, checks the server contract
before page reads, and respects the configured dashboard base path. Home, Inbox,
Sessions, Session/Trace, Tasks, Specialists, Tools, Knowledge and Evals use the
pages in this directory. Usage, Environments, Artifacts and System retain their
existing implementations under `src/routes/` inside the production shell.

This map describes the implemented surfaces, their owners and current runtime
contracts.

## Development and demo prototype

The prototype is deliberately retained for development, demonstrations and design
comparison. It is not the production server integration or a substitute for its
permissions, persistence and recovery checks. The published
[prototype](https://claude.ai/artifact/JXhabogyvF8bLbt9YqBCrS) and
[original design map](https://claude.ai/artifact/HxQypvmhJmouPdgKYWpP8X) are design
references; current runtime contracts take precedence over their simulated data.

```bash
npm run dev                       # production development entry; /api proxies to localhost:8000
# Open /prototype.html for the separate seeded prototype.
CAYU_PROTOTYPE=1 npx vite build   # prototype-only output in dist/prototype/
```

`prototype.html` starts `src/next/main.tsx` with hash routing, `src/next/mock/api.ts` and the
fake fetch adapter in `src/next/mock/legacy/`. Its bar switches seeded agents and resets
simulated state. `src/next/routes/legacy-page.tsx` and `src/next/legacy-config.ts` adapt the four
retained pages to that harness; production mounts those pages directly.
The normal production build starts `src/main.tsx`, not this mock entry point.
The prototype sources remain in the editable source archive; its separate build
is not the dashboard served by the Python package.

`CAYU_PROTOTYPE_SEEDS` may point to a JSON array of `AppSeed` values kept outside
the repository. Use made-up data for shared demos. Never put deployment snapshots,
credentials or private content under `dashboard/`: the source archive includes
that source tree. A prototype build containing supplied snapshots is also a copy
of that data.

## Product principles

- **One agent in depth.** Developers inspect this agent; Cloud covers multiple
  agents and the organization. End users use the agent's own UI.
- **Generic information.** Titles, summaries, labels and structured output come
  from session fields, not assumptions about a particular agent.
- **Conversation and Trace.** Conversation shows the work and decisions; Trace
  exposes exact server spans, events, workflow topology and execution diagnostics.
- **Human attention first.** Home, Inbox and inline session decisions reflect the
  same runtime records. Counts come from authoritative server reads, not samples.
- **Roles and capabilities.** Actions follow the server's effective permissions
  and configured capabilities. A custom role string is not itself a grant.
- **Navigation stays useful.** Existing supported route aliases and query mappings
  lead to the corresponding current surfaces.

## Pages

Page visibility and actions depend on the deployment and viewer permissions.
Baseline, comparison and execution controls apply to native Evals sources;
reported series expose their supported read-only evidence and downloads.

| Page | Previous surface | URL |
|---|---|---|
| Sidebar: ⌘K search (opens Sessions filtered), Home, Inbox with a count, Sessions with the running count when available, Tasks; Improve: Evals, Usage; Agent: Specialists, Tools, Knowledge (with the review count), Environments, Artifacts, System; theme | Four nav groups | – |
| Home: one line on the last 24 hours (sessions, failed, cost or tokens), Needs attention (sessions waiting on a person, and failures from the last day, which open their trace), and Recent sessions. "Start a session" is a small form for testing. It opens on Run a specialist, which works for every agent. Create a task (the agent's own worker runs it) sits next to it and is disabled, with a tooltip saying why, until the agent declares task types. | Dashboard, New Run | `/` (`/run` redirects; `?start=1` opens the form) |
| Session: conversation, collapsed tool work, structured output, inline approval / question / recovery cards, a compact "Message this session…" bar with Stop, and Run again in the menu | Session detail | `/sessions/:id` |
| Trace: a waterfall of the session's spans (turns, model steps, tool rounds, tool calls, commands, waits) built from its events, the selected span's detail (what went into a model request, tools offered, arguments, results, exit codes, raw events), and the event log with filters and a live tail, where a selected event opens in its own panel. Values the server withholds are named as hidden, never shown as raw markers. Conversation steps link to their span and back. | Event timeline, event detail, Workflow page, execution snapshots and other debug blocks | `?view=trace&span=…`, `&log=1` for the event log; `/sessions/:id/workflow` opens it |
| Details panel: Overview, Usage, Artifacts, Browser (only if used), and Open trace | Header stats, annotations | `?panel=…` |
| Sessions: "Start a session" (the same form as Home), title first, one state column, duration, tool calls, cost or tokens; server filters and cursor paging | Sessions | `/sessions` |
| Inbox: approvals, questions, recovery, knowledge, settled inline | Pending | `/inbox` (`/pending-actions` redirects) |
| Tools: every tool across specialists with its source (Cayu built-in, agent code, package, MCP server, provider), declared effect, its approval (never, always, sometimes, blocked), which specialists use it, calls, failures and last use; a tool opens to its description, input schema, approvals and recent calls (each opening its trace) | Tool lists inside Agents | `/tools` |
| Knowledge: entries waiting for review (also in the Inbox), then the kept entries with their kind, specialist, who kept them and when, lookups, the session they came from, and Remove | Knowledge (review only) | `/knowledge` |
| Specialists | Agents | `/specialists` (`/agents` redirects) |
| Tasks: chips with counts for Active, Waiting, Held, Failed and Done; each row says in plain words what the task is doing or waiting for ("Waiting on 3 tasks", "Retry 2 of 3 in 6m", "Scheduled for Nov 1, 6:00 AM", "Blocked: …"); filters for type and specialist; Cayu's own `cayu.*` tasks hidden behind a switch; Create a task | Tasks (queue table with worker, lease and availability columns, 12-status select, side panel) | `/tasks` |
| Task: status line, then Session (or Attempts with each attempt's session when it retries), Waiting on / Needed by, Started by / Subtasks, Input, Result or Error, Schedule with its history, and a collapsed Debug (worker, lease, concurrency key, invocation, payloads, raw task); Pause, Block, Flag for attention and Resume, with an optional reason | Tasks side panel | `/tasks/:id` |
| Evals: one list of suites with their specialist, case count, last run (passing count, when, from where) and verdict against the approved baseline; New suite; readiness only as a line when something isn't available | Evals (readiness tiles, execution profile card, Catalog / Results / Runs tabs, hash-first tables) | `/evals` |
| Eval suite: baseline (who approved it, when), runs with origin (CI branch, commit and PR; the dashboard; a session) and verdict, and cases with their latest trial dots; Run suite | Catalog and Runs tabs | `/evals/suites/:id` |
| Eval editor: shared full-page suite settings and case rail; Input/Checks tabs, all portable checks and typed judges; exact parent-aware Save, then separate reviewed Run; private in-memory drafts and explicit leave confirmation | Suite authoring sheet | `/evals/suites/new?target=…`, `/evals/suites/:id/edit?case=…` |
| Eval run: verdict (Improvement, Regression, Tradeoff, No change) and counts, passing / flaky / judge score / cost against the compared run, filter chips (Regressed, Improved, Failing, Flaky), compare with the baseline or any other run, Approve as baseline; a case opens a drawer with each trial's checks and judge reasons, both outputs side by side and the trial's trace | Results and Runs tabs, their inspectors and Compare | `/evals/runs/:id` |
| Eval settings: what trials run with per specialist, the judge, limits, what this deployment supports, and the advanced tools (corpus import and export, judge calibration, memory reports, production scenarios) | Readiness and execution profile cards, header actions | `/evals/settings` |
| Usage, Environments, Artifacts, System | unchanged, inside the new shell (the prototype harness marks them "Today's page, unchanged") | same URLs |

## Current runtime contracts

These numbers retain the original plan's backend-item references. They describe
current behavior rather than outstanding proposals.

| Item | Implemented contract and authoritative documentation |
| --- | --- |
| 1 | Session title through `RunRequest.title` and the HTTP run body; summary and explicit tone use reserved session metadata. Specialist descriptions use registered metadata. See [session display metadata](https://github.com/cayu-tech/cayu/blob/main/docs/control-plane.md). |
| 2 | `GET /api/sessions?include=overview` and overview counts share session-state rules with Cloud. Dashboard counts and the default Sessions view exclude eval trials/forks; the HTTP list exposes inclusion flags. Filtered lists show loaded rows, not an invented exact total. See [session overview](https://github.com/cayu-tech/cayu/blob/main/docs/control-plane.md). |
| 3 | Inbox filters, public-text search and exact counts; approval arguments remain withheld. Configured human review owns disclosure. Recovery links to the session. See [Inbox](https://github.com/cayu-tech/cayu/blob/main/docs/inbox.md). |
| 4–5 | Structured-output transcript parts and follow-ups use the session/runtime contracts. Queued messages additionally require authenticated identity and the deployment's message access policy. See [session operations](https://github.com/cayu-tech/cayu/blob/main/docs/control-plane.md). |
| 6 | `cayu serve --reload` watches development source and restarts the server; see [serving and reload](https://github.com/cayu-tech/cayu/blob/main/docs/control-plane.md). |
| 7, 13 | Declared task types, creation/replay, relations, lifecycle operations, exact status counts and internal-task filtering. See [tasks](https://github.com/cayu-tech/cayu/blob/main/docs/control-plane.md). |
| 8 | `GET /api/app` provides the configured project name, then deployment name, then `Cayu`; runtime version; actual authenticated role string or null; and effective `full_access`. See [dashboard identity](https://github.com/cayu-tech/cayu/blob/main/docs/control-plane.md). |
| 9, 11 | Registered tool display labels, source/approval classification and bounded usage/call reads. Unknown or unavailable values remain explicit; display labels never grant authority or change call identity. See [Specialists and Tools](https://github.com/cayu-tech/cayu/blob/main/docs/control-plane.md). |
| 10 | Server-owned Trace spans and exact transcript positions; public event-order inference is confined to the prototype. See [Trace](https://github.com/cayu-tech/cayu/blob/main/docs/control-plane.md). |
| 12 | Kept-knowledge reads and revision-fenced retirement. Reviews share Inbox decisions and source evidence. Members may read and Keep/Drop; Remove requires full access. See [Knowledge](https://github.com/cayu-tech/cayu/blob/main/docs/control-plane.md). |
| 14 | Suite-centred native reads and reported CI series. Reported evidence does not grant native execution, comparison or baseline authority. See [Evals](https://github.com/cayu-tech/cayu/blob/main/docs/evals.md) and [CI reporting](https://github.com/cayu-tech/cayu/blob/main/docs/recipes/ci-eval-reporting.md). |

## Production ownership and navigation

- `src/app/` owns production routing, layout, identity and the contract gate.
  `src/next/api/real.ts` implements the shared page API; generated HTTP types and
  the maintained clients remain the runtime boundary.
- `src/next/routes/` owns the ported pages. `src/routes/usage.tsx`,
  `environments.tsx`, `artifacts.tsx` and `system.tsx` remain live production
  implementations. The old `src/Layout.tsx` and ported route modules are removed.
  See [editable-dashboard migration](../../README.md#dashboard-structure-and-migration).
- `/agents` maps to `/specialists`; `/pending-actions` maps to `/inbox`; `/run`
  opens Home's Start form. The former `/sessions/:id/workflow` URL opens the
  embedded workflow in Trace and retains its supported filters/expansion state.
  Supported old Evals run/result query links map to the suite/run surfaces.
- `src/next/app/trace-runtime.ts` uses exact server span IDs and transcript positions.
  `src/next/app/trace-history.ts` keeps bounded historical/live cursors.
  `src/next/app/session-progress.ts` shares one incremental sequence follower between
  Conversation and Trace; paused and historical windows do not jump to the tail.
  `src/next/app/trace-model.ts` retains event-order construction only for mock fixtures.
- Session details retain annotations, usage, artifacts and Browser tools when
  used, including takeover and recordings. Start, task replay, queued messages
  and evaluation actions retain their own permission and recovery boundaries.
- Home failures open Trace's event log filtered to `session.failed`. Inbox and
  Session use protected review and explicit task continuation where required;
  a settled decision can still have execution running in the background.
- Tasks use the runtime index and exact detail reads, retain explicit continuation
  for retry attempts, and keep operation identity after an unconfirmed response.
- Specialists retain expandable configuration and bounded recent work. Tools
  retain declaration variants and exact call links; unavailable reads are not
  zero counts. Knowledge lists label loaded counts/search scope. Historical
  knowledge and Inbox cursor windows do not poll; explicit refresh returns to
  newest, and proposal detail stays bound to its revision.
- Evals uses suite/run read models, a full-page suite/case editor and Settings
  tools for profiles, calibration, scenarios, memory reports and import/export.
  Native and reported sources remain distinct. The guides are
  [first evaluation](https://github.com/cayu-tech/cayu/blob/main/src/cayu/guides/evals-first.md),
  [AI quality](https://github.com/cayu-tech/cayu/blob/main/src/cayu/guides/evals-ai-quality.md) and
  [production evaluations](https://github.com/cayu-tech/cayu/blob/main/src/cayu/guides/evals-production.md).

## Calibration recovery boundary

Calibration retries return an already-stored report for the same run ID and
reviewed definition without another judge call. That is not a durable claim on
work before report publication: if paid work ran but no report was stored,
retrying may repeat it. The open workbench retains private evidence and exact
request identity across catalog refreshes and offers confirmed local abandonment;
it does not persist that private draft across reload or cancel server work.
Stronger protection for uncertain paid work before publication is separate future
runtime work. See [calibration semantics](https://github.com/cayu-tech/cayu/blob/main/docs/evals.md#developer-dashboard-calibrate-an-ai-judge).

## Maintenance

Keep page behavior, the runtime contract, these instructions and guides together.
Rebuild the editable source archive when dashboard source or documentation changes;
keep the packaged notice and compiled/source digests synchronized. Production
verification uses the real server and installed/ejected assets. The retained
prototype is an additional visual/development harness, not production acceptance
evidence.
