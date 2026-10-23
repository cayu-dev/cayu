# Developer dashboard rework

The reworked Developer dashboard (`/cayu/`), built in this folder as the real app on a
fake API. It uses the dashboard's own Tailwind tokens and shadcn components and the
server's generated types, and replaces `src/routes` and `src/Layout.tsx` area by area
(see Porting), each area shipping against today's server where it can.

- **The plan:** backend prerequisites and the area-by-area porting sequence are listed below.
- **Clickable prototype:** https://claude.ai/artifact/JXhabogyvF8bLbt9YqBCrS
- **This page map, published (takes comments):** https://claude.ai/artifact/HxQypvmhJmouPdgKYWpP8X

## Run it

```bash
npm run dev                        # then open /prototype.html
CAYU_PROTOTYPE=1 npx vite build    # dist/prototype/, relative URLs, works from any folder
```

To add snapshots of real deployments, point `CAYU_PROTOTYPE_SEEDS` at a JSON array of
seeds (`AppSeed` in `mock/model.ts`) kept **outside the repository**. The dashboard
source archive packs every file under `dashboard/`, so real data must never live
in this folder.

The prototype bar at the top switches between agents and resets the fake data.

## Why

The current dashboard is an audit console for the stores with an evals workbench attached.

- Home is a page of store metrics.
- Session rows lead with random IDs and repeat the same values in several columns.
- The session page is one 3,855-line component, and the browser box renders on every session.
- Approvals are split between Pending and Knowledge, and neither page can act on them. The server's unified `/api/inbox` is never called.

Cayu Cloud now links to `/cayu/sessions/{id}` for a run's full conversation, so this dashboard is the detailed view of one agent, locally and behind Cloud.

It is the **developer's** tool for looking deeply into the whole agent. Cayu agents are autonomous workers, not chat assistants. The people the agent works for (an analyst, an accountant, a lawyer) use the agent's own UI, which each agent builds; it doesn't ship with Cayu.

## Principles

- **One agent in depth.** Cloud covers many agents and the organization.
- **Nothing agent-specific.** Titles, summaries, labels, metadata and structured output come from generic session fields. The prototype switches between unrelated agents to keep that honest.
- **Everything is inspectable.** Each session has a Trace: every model step, tool call, command and wait, with its exact events.
- **The session reads as a conversation.** A run of tool calls collapses into one muted line ("Read 9 files, searched 10 times"), as in Claude desktop.
- **What needs a person is the loudest thing on screen.** It shows as the Inbox count in the sidebar, Needs attention on Home, inline cards in the session, and one inbox.
- **The sidebar is navigation.** Every area of the dashboard stays one click away, grouped as Home, Inbox, Sessions, Tasks; Improve (Evals, Usage); Agent (Specialists, Tools, Knowledge, Environments, Artifacts, System). Sessions are listed on the Sessions page, not in the sidebar.
- **Existing links keep working.**

## Pages

| New | Replaces | URL |
|---|---|---|
| Sidebar: ⌘K search (opens Sessions filtered), Home, Inbox with a count, Sessions with the running count, Tasks; Improve: Evals, Usage; Agent: Specialists, Tools, Knowledge (with the review count), Environments, Artifacts, System; theme | Four nav groups | – |
| Home: one line on the last 24 hours (sessions, failed, cost or tokens), Needs attention (sessions waiting on a person, and failures from the last day, which open their trace), and Recent sessions. "Start a session" is a small form for testing. It opens on Run a specialist, which works for every agent. Create a task (the agent's own worker runs it) sits next to it and is disabled, with a tooltip saying why, until the agent declares task types. | Dashboard, New Run | `/` (`/run` redirects; `?start=1` opens the form) |
| Session: conversation, collapsed tool work, structured output, inline approval / question / recovery cards, a compact "Message this session…" bar with Stop, and Run again in the menu | Session detail | `/sessions/:id` |
| Trace: a waterfall of the session's spans (turns, model steps, tool rounds, tool calls, commands, waits) built from its events, the selected span's detail (what went into a model request, tools offered, arguments, results, exit codes, raw events), and the event log with filters and a live tail, where a selected event opens in its own panel. Values the server withholds are named as hidden, never shown as raw markers. Conversation steps link to their span and back. | Event timeline, event detail, Workflow page, execution snapshots and other debug blocks | `?view=trace&span=…`, `&log=1` for the event log; `/sessions/:id/workflow` opens it |
| Details panel: Overview, Usage, Artifacts, Browser (only if used), and Open trace | Header stats, annotations | `?panel=…` |
| Sessions: "Start a session" (the same form as Home), title first, one state column, duration, tool calls, cost or tokens; columns that never vary are hidden and named once | Sessions | `/sessions` |
| Inbox: approvals, questions, recovery, knowledge, settled inline | Pending | `/inbox` (`/pending-actions` redirects) |
| Tools: every tool across specialists with its source (Cayu built-in, agent code, package, MCP server, provider), declared effect, its approval (never, always, sometimes, blocked), which specialists use it, calls, failures and last use; a tool opens to its description, input schema, approvals and recent calls (each opening its trace) | Tool lists inside Agents | `/tools` |
| Knowledge: entries waiting for review (also in the Inbox), then the kept entries with their kind, specialist, who kept them and when, lookups, the session they came from, and Remove | Knowledge (review only) | `/knowledge` |
| Specialists | Agents | `/specialists` (`/agents` redirects) |
| Tasks: chips with counts for Active, Waiting, Held, Failed and Done; each row says in plain words what the task is doing or waiting for ("Waiting on 3 tasks", "Retry 2 of 3 in 6m", "Scheduled for Nov 1, 6:00 AM", "Blocked: …"); filters for type and specialist; Cayu's own `cayu.*` tasks hidden behind a switch; Create a task | Tasks (queue table with worker, lease and availability columns, 12-status select, side panel) | `/tasks` |
| Task: status line, then Session (or Attempts with each attempt's session when it retries), Waiting on / Needed by, Started by / Subtasks, Input, Result or Error, Schedule with its history, and a collapsed Debug (worker, lease, concurrency key, invocation, payloads, raw task); Pause, Block, Flag for attention and Resume, with an optional reason | Tasks side panel | `/tasks/:id` |
| Evals: one list of suites with their specialist, case count, last run (passing count, when, from where) and verdict against the approved baseline; New suite; readiness only as a line when something isn't available | Evals (readiness tiles, execution profile card, Catalog / Results / Runs tabs, hash-first tables) | `/evals` |
| Eval suite: baseline (who approved it, when), runs with origin (CI branch, commit and PR; the dashboard; a session) and verdict, and cases with their latest trial dots; Run suite | Catalog and Runs tabs | `/evals/suites/:id` |
| Eval run: verdict (Improvement, Regression, Tradeoff, No change) and counts, passing / flaky / judge score / cost against the compared run, filter chips (Regressed, Improved, Failing, Flaky), compare with the baseline or any other run, Approve as baseline; a case opens a drawer with each trial's checks and judge reasons, both outputs side by side and the trial's trace | Results and Runs tabs, their inspectors and Compare | `/evals/runs/:id` |
| Eval settings: what trials run with per specialist, the judge, limits, what this deployment supports, and the advanced tools (corpus import and export, judge calibration, memory reports, production scenarios) | Readiness and execution profile cards, header actions | `/evals/settings` |
| Usage, Environments, Artifacts, System | unchanged, in the sidebar; the prototype shows today's pages as they are, marked "Today's page, unchanged" | same URLs |

## Decisions

- **The dashboard is for developers to dive into the whole agent.** End users have the agent's own UI. Home is an overview of the agent, not a chat composer; starting a session is a small test action.
- **Trace is the deep view.** Every session can be read as a conversation or as a trace. Until backend item 10 lands, spans are linked by event order and the UI says so.

- **Audience:** developers. Customer operators use Cloud and the agent's own UI.
- **The word "Agent":** the UI calls the whole thing an agent ("this agent", "Agent code"), never an app. "App" is the code's internal name (`app.run`, `AppInfo`). The agents it registers are Specialists.
- **The word "Session":** it matches Anthropic Managed Agents, Google ADK, AWS AgentCore and the API. Cloud's rebuild should rename its "runs" to match.
- **Specialists and the Developer dashboard:** registered agents are Specialists, and the product is the Developer dashboard.
- **Look:** keep the current look; add dark mode (`.dark` tokens in `src/index.css`).
- **Evals, answered on 22 October:** suites are the main list; results and runs are one thing, a run; a run is always read against the approved baseline (or a run you pick); "Add to evals" sits in a finished session's menu, either scoring the session as it is or keeping it as a case that reruns; the readiness and execution-profile details, judge calibration, memory reports and corpus import move to Evals settings.

- **Tools and Knowledge sit under Agent.** The sidebar group is called Agent, not App: it holds what the agent is made of (specialists, tools, knowledge) and where it runs. Knowledge reviews show both in the Inbox and on the Knowledge page.

- **Answered on 22 October:**
  - "Start a session" offers Create a task next to Run a specialist, and every session and task gets Run again (backend item 7).
  - Any session can take a follow-up (backend item 5).
  - The composer becomes a compact one-line bar that grows when focused, with Stop while running.
  - Execution snapshots, retry cascade and mutation transport status live in the trace's session span, collapsed. No Debug tab and no developer-mode switch.
  - Built-in tools ship display labels; other tools show their name unless their definition sets one.
  - The details panel starts closed.
- **Tasks, answered on 22 October:** a task opens its own page; the list groups the twelve statuses into five plain groups and says why per row; Cayu's own tasks are hidden behind a switch; worker, lease and concurrency key move to the task's Debug section.

## Code

- **`api/types.ts`** holds every call the pages make. Existing endpoints use the generated server types.
- **`mock/api.ts`** answers those calls from an in-memory seed. Writes play out a short simulated run.
- **`api/real.ts`** makes the same calls against the server. Switching to it is the port.
- **`app/`** holds the query hooks, routes, services, theme and `session-model.ts`, which contains the pure helpers that `tests/next-session-model.test.mjs` covers.
- **`app/trace-model.ts`** builds the trace from events, including linking by order when the server hides IDs; `tests/next-trace-model.test.mjs` covers it.
- **`routes/`** holds the pages. `legacy-page.tsx` mounts today's unchanged pages from `src/routes/` in a second router that shares the history, so the new router picks up their links.
- **`mock/legacy/`** is a fake server for those pages: it answers their `/api` calls through `fetch`, from the same seed. `contract.ts` is a real 0.13.0 capture. `legacy-config.ts` points their links into the hash.
- **`mock/seeds/`** holds the made-up agents: Month-end close (three specialists, every state) and New agent (empty).
- **Real snapshots** come in through `CAYU_PROTOTYPE_SEEDS` (see Run it) and are listed after Month-end close. They never live in the repository.
- **`next.css`** loads Tailwind plus the shared tokens in `src/theme.css`, and adds status colors and dark mode. `src/index.css` excludes `next/` from its scan, so the packaged dashboard's CSS and bundle stay exactly as they are.

## Backend items (calls marked "Proposed" in `api/types.ts`)

Each page area has a separate port, with shared runtime prerequisites implemented independently.

1. **Agent-set title and summary.** Store them under the reserved metadata keys `cayu:title` and `cayu:summary`, set from `app.run(..., title=)` and at completion. An optional `cayu:summary_tone` (`neutral` or `attention`) lets the agent say a result needs a person to look. The dashboard shows an attention summary in the warning color and never guesses tone from the summary text, because "1 finding" is bad news for a reviewer and the expected result for others. Sessions have no title today. Display falls back to the labels, then the first user message, then the ID; the prototype's "Agent-set titles" switch shows both. Real data showed that the first message can be the same instruction preamble on every session. Specialists get the same treatment: `AgentSpec` has no description today, so an app-set `cayu:description` in agent metadata gives the Specialists page its one-line summary.
2. **`GET /api/sessions?include=overview`**. It returns the merged state, title, summary, start and end times, last activity, steps, tool calls, tokens, cost (when a price book is configured) and `uses`, which covers browser, subagents and artifacts. Move `CloudRunStatus` and `_RunReader.runs()` out of `cayu/cloud_reporting.py` into a provider-neutral `cayu.sessions.overview` that both use. It also flags eval trials (`cayu:eval_trial`) and forks, which the dashboard leaves out of counts as Cloud does.
3. **Inbox.**
   - Add `session_id`, `agent_name` and `kind` filters.
   - Return tool arguments on approvals; `_action_from_record` drops them today.
   - Add counts per kind for the chips and the sidebar badge.
   - Recovery stays out of the inbox's settle call, as `main` intends ("settled in the Developer dashboard, with the evidence of what happened"): the Inbox lists it and links to the session, which settles it through the existing recovery path.
4. **A `structured_output` transcript part**, so the UI doesn't depend on the internal `__cayu_submit_structured_output` tool name.
5. **Any session can take a follow-up**, completed ones included, as a new turn in the same session: `/resume` with a prompt, or `POST /api/sessions/{id}/messages`.
6. **`cayu serve --reload`.**
7. **Task types and Run again.** Cayu agents have no single entrypoint; they start sessions from their own code, usually by creating a durable task. Apps declare the task types they handle with an input schema (`GET /api/task-types`), and `POST /api/tasks` creates one so the app's own worker runs it the real way. Run again replays a task's input the same way. Today the server only lists tasks (`GET /api/tasks`).
8. **A small `GET /api/app`** returning the agent's name, runtime version and the viewer's role (`admin`, `developer`, `member`; `src/cayu/server/proxy_identity.py`), so pages hide what the viewer can't do.
9. **A display label on tools**, so a run of tool calls reads as one line ("Read 9 files, searched 10 times"), the way Claude shows "Ran 2 commands". Add an optional `display` to the tool spec: a verb, an optional noun and an optional plural. Return it on `/api/agents` tool summaries. Built-in tools ship with labels; other tools fall back to their name ("used query_ledger 2 times"). The prototype labels every built-in in `src/cayu/tools` (files, search, commands, git, browser, web, subagents, artifacts, knowledge), not only the ones the reviewer uses.
10. **`GET /api/sessions/{id}/trace`**: the span tree, built on the server. **Required**, not just faster: the public event projection (`runtime/_event_projection.py`) hides `model_step_id` and `execution_id` (`[PRIVATE_EVENT_AUTHORITY]`) and gives `tool_call_id`/`tool_round_id` per-event aliases that never match across events, so the browser can only link spans by event order. The endpoint links them with the private IDs and returns public span IDs, plus the tool call ID the transcript uses.
11. **Tool registration and usage.** Each tool summary on `/api/agents` gains `source` and `approval`, with the classification built for Cloud in `src/cayu/cloud_inventory.py` (sources `agent`, `cayu`, `package`, `mcp`, `provider`, `unknown`; approval `never`, `always`, `sometimes`, `blocked`, `unknown`), and a usage read returns calls, failures, approval decisions and recent calls per tool, reusing the usage counting in the same module. This is the local side of tool registration with Cloud.
12. **`GET /api/knowledge`** for kept entries (kind, specialist, source session, who kept it and when, lookups and last use from the `knowledge.lookup` event) and a way to retire one. Today only the review side exists (`/api/knowledge/pending`, approve, reject).
13. **Task relations and lifecycle.** Serialize `graph_id` and `prerequisite_task_ids` on task list and detail (the store has them; the Tasks page shows Waiting on and Needed by from them), add a general cancel (today only scheduled tasks can be cancelled), return status counts with the list so the chips don't depend on the page size, exclude `cayu.*` tasks on the server, and add a dependents read for Needed by. The retry series' attempts are found by walking predecessor and successor links; a series read would save the extra requests.
14. **Suite-centred eval reads** and **CI runs in the eval store** (needs a design: where results go, with what credential, against which target). One read for the suite list with each suite's last run, verdict and baseline, a suite read with its runs and per-case latest outcome, and a run read that compares case by case with the baseline or a named run (the server already decides compatibility and regressions in `/api/evals/result-comparisons`). Authored suites and corpus suites become one list. Authoring beyond "final output contains" needs a separate design.

## Porting

Every area follows the shared rules for roles, the contract check, the transition, scale, CSP, accessibility, docs and tests, and must preserve the actions the current dashboard offers.

1. Mount `createDashboardRouter()` from `src/main.tsx` with `realApi` and the dashboard's base path, keep the server-contract check and capability gating, and register its router type so `NavLink` and `useGo` become typed links. Keep every old route mounted with its old search params until its area lands.
2. Port area by area, each against today's server with the fallbacks the prototype already has, and switch each to the runtime change it needs as it lands. Evals is the exception: it needs suite-centred eval reads first.
3. Move Usage, Environments, Artifacts and System in unchanged, wrapped in `TopBar`/`PageBody`. The browser operator and recordings go into the session's Browser tab, and execution snapshots into the trace's session span.
4. Remove the old routes, `Layout.tsx` and their tests once their area has landed. Rewrite each area's part of `examples/dashboard_behavior_live.py`, and update the chunk-name assertions in `tests/server/test_server.py`.
