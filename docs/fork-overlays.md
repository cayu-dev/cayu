# Session fork overlays

A fork overlay lets an application outside the host, such as a supervisor,
fork a host session with its own instructions, its own tools, and its own
memory context. The host app ships no controller code. It enrolls the
controller once, and the controller sends overlays through an authenticated
entrance.

Every part of an overlay is optional, and the parts are independent:

| Part | What it does | Where it lives |
| --- | --- | --- |
| `instructions` | Text the model sees after the copied source history. | Child metadata, rendered per request. |
| `memory` | `ForkOverlayMemoryItem`s visible to this fork only. | Child metadata, rendered per request. |
| `tools` | `ForkOverlayToolGrant`s naming tools the controller serves. | Child ceiling and fork-controller body. |

Budget, stop conditions and the expected result shape are not new overlay
fields. Use the fork's `initial_invocation` or a later `ResumeRequest`:
`budget_limits`, `max_steps`, `limits` and `structured_output`. They are part
of the fork request identity when sent as `initial_invocation`. These are the
controller's own limits; the host's per-fork cap (see
[Running forks](#running-forks)) applies on top of them and the controller
cannot change it.

## Enrolling a controller

```python
from cayu.budgets import BudgetLimit, BudgetReservation
from cayu.fork_controllers import ForkController
from cayu.mcp import connect_mcp_toolset

compass_tools = await connect_mcp_toolset(
    McpServerSpec(
        name="compass",
        connection_id="compass",
        url="https://compass.example/mcp",
    )
)
app.enroll_fork_controller(
    ForkController(
        name="compass",
        subjects=("svc:compass",),
        agents=("finance-monthend",),
        host_tools=("lookup_ledger",),
        mcp_toolsets=(compass_tools,),
        allow_model_runs=True,
        fork_budget_limit=BudgetLimit.usd(
            2, reservation=BudgetReservation(max_input_tokens=200_000, max_output_tokens=16_000)
        ),
    )
)
```

Everything after `agents` is keyword-only.

- `subjects` are identities the host has authenticated. One subject acts for
  at most one controller.
- `agents` are the host agents the controller may fork. They must already be
  registered.
- `host_tools` are the only host tools any of the controller's forks may hold.
  The default is none. Each name must be a tool that at least one enrolled
  agent registers, so a misspelled name fails at enrollment. See
  [Host tool allowance](#host-tool-allowance).
- `mode` is the controller's declared operating mode. Only
  `ForkControllerMode.OBSERVE` (`"observe"`, the default) exists; any other
  value is rejected. See [Operating mode](#operating-mode).
- `mcp_toolsets` is optional. A controller toolset must use a URL transport,
  so its tools run on the controller's side, and may not use `secret_env` or
  `secret_headers`, so no host secret is resolved for it.
- `allow_model_runs` is False by default: the controller can create forks,
  but not run them. Turning it on requires `fork_budget_limit`, the host's
  spend cap for each fork. See [Running forks](#running-forks).

When a controller serves tools, enrollment registers one extra body per agent,
named `<agent>@fork-controller:<controller>`. It is the host agent's exact
registration (prompt, policies, hooks, context policy) plus the controller's
tools. The body runs only as an overlay fork from that controller: ordinary
runs, ordinary forks that name it, and host forks of its sessions are refused.

### Enrolling from the environment

A host can enroll one controller without code. When `CAYU_FORK_CONTROLLER_NAME`
is set, `create_server` and `mount_cayu` (and so `cayu serve`) enroll that
controller on the app after its agents are registered, and serve the
fork-controller routes at `<api path>/fork-controller`: `/api/fork-controller`
for `create_server` and `cayu serve`, `/cayu/api/fork-controller` for
`mount_cayu` at its default path. The routes authenticate with the
controller's own bearer token, never the operator login.

| Variable | Value |
| --- | --- |
| `CAYU_FORK_CONTROLLER_NAME` | The controller name. Unset means no controller is enrolled from the environment. |
| `CAYU_FORK_CONTROLLER_AGENTS` | Required. Comma-separated host agents it may fork. |
| `CAYU_FORK_CONTROLLER_TOKEN_SHA256` | Required. Comma-separated lowercase hex SHA-256 digests of its bearer tokens. Give two during rotation. |
| `CAYU_FORK_CONTROLLER_MODE` | Optional. Only `observe`. |
| `CAYU_FORK_CONTROLLER_HOST_TOOLS` | Optional. Comma-separated host-tool allowance; the default is none. |
| `CAYU_FORK_CONTROLLER_ALLOW_MODEL_RUNS` | Optional. `true` or `false`; the default is `false`. `true` needs the three cap variables below. |
| `CAYU_FORK_CONTROLLER_FORK_BUDGET_USD` | The per-fork cap in US dollars, priced with Cayu's bundled price book. See [Running forks](#running-forks). |
| `CAYU_FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS` | The cap's reservation: the most input tokens one model step may use. |
| `CAYU_FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS` | The cap's reservation: the most output tokens one model step may use. |
| `CAYU_FORK_CONTROLLER_TOTAL_BUDGET_USD` | Optional. The [controller total](#a-total-across-the-controllers-forks) in US dollars. Needs the three cap variables and shares their reservation. |

```bash
token=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
export CAYU_FORK_CONTROLLER_NAME=compass
export CAYU_FORK_CONTROLLER_AGENTS=finance-monthend
export CAYU_FORK_CONTROLLER_TOKEN_SHA256=$(printf %s "$token" | shasum -a 256 | cut -d' ' -f1)
cayu serve
curl -H "Authorization: Bearer $token" http://127.0.0.1:8000/api/fork-controller/enrollment
```

- The controller acts as subject `fork-controller:<name>`, which
  `GET .../enrollment` reports. A fork's `profile_adoption.requested_by` names
  that subject.
- The host holds only digests, so the token never reaches its process. Cayu
  can't check a token's length from its digest; use at least 32 random
  characters.
- Malformed configuration stops the server from starting: a missing required
  variable, any of these variables set without a name, an empty list entry, a
  repeated or malformed digest, an unknown mode or `ALLOW_MODEL_RUNS` value,
  `ALLOW_MODEL_RUNS=true` without all three cap variables, a malformed cap or
  total, `TOTAL_BUDGET_USD` without the cap variables, a
  durable session store without a durable budget ledger when model runs are on, an
  unregistered agent, an unknown host tool, or a name or subject already
  enrolled differently in code. Errors never include digests.
- Mounting the same app twice reuses the one enrollment.
- Budget reservations, such as each fork's spend, go through the app's budget
  ledger. An app whose session store comes from `open_application_stores`
  already uses the durable ledger built beside it (`SQLiteBudgetLedger` locally,
  `PostgresBudgetLedger` with a PostgreSQL `CAYU_DATABASE_URL`), so enrolling
  from the environment needs no ledger code. A `budget_ledger` the app passes to
  `CayuApp` still wins.
- A `create_server` configuration with the API disabled serves no fork-controller
  routes and enrolls nothing.
- The parts are usable on their own:
  `cayu.server.fork_controller.fork_controller_from_environment()` returns the
  `ForkController` and its `ForkControllerBearerAuth`;
  `ForkControllerBearerAuth.from_token_digests({...})` authenticates from
  digests; and `environment_fork_controller_router(app)` enrolls and returns
  the router for a host that mounts it itself.
- A controller enrolled this way serves no tools; enroll in code to attach
  `mcp_toolsets`.

## Forking with an overlay

```python
controller = app.fork_controller_access("svc:compass")  # after host authentication
events = controller.fork(
    ForkSessionRequest(
        source_session_id=trunk_id,
        session_id="compass-accruals",
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
        execution_profile_selection="current_child",
        profile_adoption=ExecutionProfileAdoptionIntent(
            idempotency_key="compass-accruals",
            reason="Investigate the accrual variance.",
            requested_by=ResolutionActor(subject="svc:compass", source="request"),
        ),
        overlay=ForkOverlay(
            controller="compass",
            instructions="Investigate the September accrual variance only.",
            tools=(ForkOverlayToolGrant(tool_name="mcp__compass__lookup"),),
            memory=(ForkOverlayMemoryItem(id="aug", content="August closed at 2%."),),
        ),
    )
)
```

`controller.resume(ResumeRequest(...))` continues a fork this controller
created, while the host still enrolls the controller for that session's agent
and allows it to run forks (see [Running forks](#running-forks)). `controller.snapshot_source(session_id)` returns the exact
`ForkSourceSnapshot` an `initial_invocation` fork needs; the child's first
invocation then runs through the host's ordinary durable dispatch.

`controller.enrollment()` returns a frozen `ForkControllerEnrollment`: the
controller name, the authenticated subject, `mode`, the enrolled `agents`,
`host_tools`, `served_tools` (the names of the tools the controller
serves), `allow_model_runs`, `fork_budget_limit` (the cap's
`max_estimated_cost` and `currency`, without the price book, or `None`) and
`total_budget_limit` (the controller total in the same form, or `None`). It
describes only the caller's own enrollment. Other controllers,
other subjects of the same controller and agents outside the enrollment never
appear.

Hosts that expose the entrance over HTTP mount
`cayu.server.fork_controller.create_fork_controller_router(app,
authenticate=...)`, which serves `GET /fork-controller/enrollment`,
`GET /fork-controller/sources/{id}/snapshot`, `POST /fork-controller/fork`,
`POST /fork-controller/resume` and the read routes in
[Reading sessions and events](#reading-sessions-and-events). The host's
`authenticate` callback names the subject; request bodies never do. A body's
`profile_adoption.requested_by` must name that same subject, and the recorded
actor is rebuilt from it with `source="http_auth"`; body `tenant` and `claims`
are dropped. A missing or empty subject returns 401. Denials, unknown sessions
and foreign sessions all return the same 403; a conflicting fork, a rejected
profile, or a source that changed after its exact snapshot returns 409.

### Running forks

A fork runs the host's model, and possibly the controller's tools, at the
host's expense. Running a controller's forks is the host's explicit choice,
and it always comes with a hard spend cap:

```python
ForkController(
    name="compass",
    subjects=("svc:compass",),
    agents=("finance-monthend",),
    allow_model_runs=True,
    # At most 2 USD per fork; each model step holds up to this many tokens'
    # cost until its real usage is known.
    fork_budget_limit=BudgetLimit.usd(
        2, reservation=BudgetReservation(max_input_tokens=200_000, max_output_tokens=16_000)
    ),
)
```

`allow_model_runs=True` without `fork_budget_limit` fails at construction, so
model runs are never unbounded. The cap is an ordinary session-scoped
`BudgetLimit`, so it can use the bundled price book or the host's own
(`pricing=my_prices`). It must keep the defaults that make it a hard cap:
`scope="session"`, `action="interrupt"`, an all-time window and
`allow_unpriced=False`. A model the price book does not price, or a completion
without usage, stops the fork instead of running uncounted.

It also needs a `reservation`: the most tokens one model step may use. Before
each model step, compaction or auxiliary call, the budget ledger reserves that
step's worst-case cost against the fork's cap and refuses the step when the
reservation would not fit. The reservation is settled to the real cost when
usage arrives. A step that ends without usage, such as a stream the controller
abandons, stays charged at its reservation, so dropping requests cannot spend
past the cap. Choose a reservation the cap holds a few times over: one larger
than the cap stops every fork before its first step.

The runtime also holds every request of a capped fork inside the reservation,
so no single step can cost more than it reserved:

- Output is capped at the reservation's `max_output_tokens` through the
  provider adapter's own output-token option
  (`ModelProvider.output_token_option_path()`); a lower cap the request already
  sets is kept. The bundled adapters declare it. A provider that declares none
  cannot run a capped fork: the step fails before any request.
- Input is bounded by the request's serialized size in bytes, which is at least
  its token count for text. A request larger than the reservation's
  `max_input_tokens` fails before dispatch, as does one with files or other
  non-text input that bytes cannot bound. Reserve input for the fork's whole
  context, measured in bytes.
- A provider-backed compactor's requests are checked the same way but are not
  rewritten, so a capped fork's compactor must set its own output cap within
  the reservation, for example `ModelCompactor(options={"anthropic":
  {"max_tokens": 4000}})`. An auxiliary call must
also fit in it (`InferenceLimits.max_input_tokens` and `max_output_tokens` at
most the reservation's). Reservations must outlive a restart as long as the
sessions do, so with a durable session store, enrolling a controller with
`allow_model_runs=True` fails unless the app has a durable `budget_ledger`
(`SQLiteBudgetLedger` or `PostgresBudgetLedger`). An in-memory session store
can keep the default in-memory ledger.

With the default, `allow_model_runs=False`:

- `controller.fork(...)` with an `initial_invocation` is refused with
  `ForkControllerDenied` (403 over HTTP);
- `controller.resume(...)` is always refused the same way.

The refusal comes before any child is created or dispatched, so no model,
hosted tool or controller-served MCP tool is called. A fork without an
`initial_invocation` is still created.

#### How the cap and the gate are enforced

The host enforces both through Cayu's budget machinery, for every session in
the controller's [fork lineage](#fork-lineage): the controller's forks and
every session created under them, such as a subagent. It does not matter what
starts the run:

- the controller's own `fork(initial_invocation=...)` and `resume(...)`;
- the host's `app.resume(...)`, `app.dispatch(...)` and `app.dispatch_inline(...)`;
- a new run the host starts under a fork, a subagent, or a recovery.

Before each model step, compaction or auxiliary model call, the runtime looks
up the controller the session's lineage names in the current enrollment. If
that controller is not enrolled, or does not allow model runs, the step is
refused with `ForkControllerDenied`: resumes and dispatches are refused before
any work, and a new run under the fork fails before its first model request.
Otherwise the cap is evaluated with the app's other budget limits, against
the session's durable cost events and its ledger reservations. A fork that
reaches its cap stops with `budget.limit_reached` before the next model step;
the step that crosses the cap finishes. Every later resume stops at once.

- The cap counts each session in the lineage on its own: a subagent under a
  fork has its own cap, with its own ledger entry. That entry's identity
  names the controller, the currency and the session's id and instance id, so
  a new fork that reuses a deleted fork's id starts fresh, but not the amount, price
  book or reservation, so a host that changes the cap keeps charging the
  fork's earlier reservations against the new one. Budget inspection reports
  such a session's cost as `partial`, since its checks name two different caps. A
  host that also wants a bound on what all the forks spend together sets a
  [controller total](#a-total-across-the-controllers-forks). App and causal
  budgets still apply to forks as usual.
- The controller's own `budget_limits` can only tighten what a fork may spend.
- Spend is read from durable events, and the cap and the gate from the current
  enrollment. A host that restarts with a lower cap, or with
  `allow_model_runs=False`, stops forks created earlier.
- The cap is not part of a fork's execution profile, so changing it never makes
  an existing fork fail profile checks.
- Provider-hosted tools such as OpenAI web search have no hard per-response
  ceiling, so a cap cannot bound them. A controller allowed to run models
  cannot name a hosted tool type in `host_tools`; enrollment fails. A
  hosted-tool agent the host runs under a fork fails at its first model step.
- A session without a lineage marker, such as one written before markers
  existed, settles its owner through its parents (see
  [Sessions without a marker](#sessions-without-a-marker)). One whose lineage
  does not settle cannot run.

#### A total across the controller's forks

The per-fork cap bounds each session, but a controller can open as many forks
as it likes. A host that wants to bound what the controller spends in all of
them sets `total_budget_limit`. It is optional, and nothing needs it unless the
host asks for a total:

```python
step = BudgetReservation(max_input_tokens=200_000, max_output_tokens=16_000)
ForkController(
    name="compass",
    subjects=("svc:compass",),
    agents=("finance-monthend",),
    allow_model_runs=True,
    fork_budget_limit=BudgetLimit.usd(2, reservation=step),
    # At most 20 USD across every fork, resume and subagent of this controller.
    total_budget_limit=BudgetLimit.usd(20, reservation=step),
)
```

The total is a `BudgetLimit` with the same hard shape as the per-fork cap:
the default `scope="session"`, `action="interrupt"`, an all-time window, no
`allow_unpriced` and a `reservation`. Construction refuses anything looser.
It applies on top of the per-fork cap, through the same entrances and checks:

- Every session in the controller's lineage gets the same ledger identity for
  the total, keyed by the controller and currency. Each model step,
  compaction and auxiliary call in any of them reserves against that one entry,
  so concurrent forks cannot together spend past the total, and a request a
  fork abandons stays charged at its reservation. Like the per-fork cap, the
  identity does not depend on the amount, so changing the total keeps charging
  earlier reservations.
- The check before each step counts the spend of every session the
  controller's lineage owns, not only the session about to run. A session
  without a lineage marker counts through its parents. The host's own sessions
  and other controllers' lineages never count. A fork that brings the lineage
  to its total stops with `budget.limit_reached`, and every other fork of the
  controller then stops before its next model request.
- With a durable session store and durable budget ledger, spend before a
  restart keeps counting after it.
- The ledger entry can only hold spend that reserved against the total. Set
  the total before the controller's forks run models: if any session in the
  lineage has a model attempt that did not reserve against it, for example
  because the host added the total later, the controller's forks fail closed
  with `ForkControllerDenied` instead of running under a total they could
  jointly pass. To start a new total over such forks, enroll the controller
  under a new name.
- Choose its reservation as for the per-fork cap. A total smaller than one
  reservation stops every fork before its first step. When the total's
  reservation is tighter than the per-fork cap's, each request is held to the
  tighter of the two.

`SessionStore.read_cost_accounting(..., fork_lineage_owner="compass")` gives
the same lineage-wide reading the check uses, on every built-in store, and
`without_reservation_for=<budget_limit_id>` narrows a read to model attempts
that never reserved against that limit.

## Reading sessions and events

A controller reads host sessions through the same entrance:

| SDK (`ForkControllerAccess`) | HTTP |
| --- | --- |
| `sessions(SessionQuery)` | `GET /fork-controller/sessions` |
| `session(id)` | `GET /fork-controller/sessions/{id}` |
| `events(id, after_sequence=, before_sequence=, order_by=, limit=, max_bytes=)` | `GET /fork-controller/sessions/{id}/events` |
| `query_events(EventQuery, exclude_forked_sessions=, max_bytes=)` | `POST /fork-controller/events/query` |
| `supervision_events(SupervisionEventQuery)` | `POST /fork-controller/supervision-events/query` |

The host enforces what a controller can read. A session is readable only when
all of these hold:

- its `agent_name` is one of the controller's enrolled `agents`, or the
  controller's own body for one (`<agent>@fork-controller:<controller>`);
- if it carries an overlay record, that record names this controller, so
  another controller's forks, including instructions-only forks, stay private
  to it;
- if it carries a fork-lineage marker, the marker names this controller or no
  controller, so everything created under another controller's fork stays
  private to that controller too;
- if it carries neither, its lineage settles through its parents (see
  [Sessions without a marker](#sessions-without-a-marker)).

A subagent child of an enrolled session is readable only when its own agent is
enrolled.

### Fork lineage

Every fork, and every session created under a fork or under any of its
descendants (a subagent, a delegated run, a grandchild), carries the
runtime-owned metadata entry `cayu:fork_lineage`:
`{"controller": <name or null>, "forked": true}`. A controller fork names its
controller; a host fork, and everything under it, names no controller. A child
copies the marker from its parent, or takes its controller from the parent's
overlay record, inside the same store write that creates it, so the marker is
durable on every store and survives restarts and the deletion of its parent.
Ordinary children whose parent row settles an unforked lineage carry
`{"controller": null, "forked": false}`. They inherit that marker through
further descendants, so deleting or detaching a parent keeps their controller
visibility and unforked supervision status unchanged.

- A session whose marker names another controller is unreadable, like that
  controller's fork itself. A marker that names no controller adds no limit.
- A marker that is present but malformed makes the session unreadable to every
  controller. The SQL clause and the memory predicate treat it the same way.
- Callers cannot set, replace or remove the marker. Run and fork requests that
  carry it are refused, metadata updates keep it, and subagent tool arguments
  drop it. The HTTP run body has no metadata field.

#### Sessions without a marker

Sessions written before markers existed have none: forks and descendants
already in a store when it is upgraded, and anything a writer from that
release still creates. No migration adds markers. Instead, a session that
carries neither an overlay record nor a marker settles its lineage through its
parents, using facts every release stored: the parent link, each overlay
record, and the `fork` invocation source every fork records. The walk climbs
parents until one carries an overlay record or a marker, which then decides as
above, or until it reaches a true root (no parent, and the root of its own
session tree), which is unforked unless the walk passed a fork. A host fork
passed on the way to a root puts the session in a host fork lineage.

- A session whose parents do not settle its lineage is unreadable to every
  controller and counts as forked. That covers a session whose parent was
  deleted, since only a marker would have kept its owner, and a chain longer
  than 256 parents.
- A child copies its parent's marker. An unmarked ordinary root also establishes
  an unforked marker for its children; other unmarked parents leave their children
  unmarked and settling through the same parent walk. A new fork resolves its source's
  lineage when it is created: a controller fork of a source that does not
  settle is refused, and a host fork of one gets no marker.
- SQLite and PostgreSQL walk the parents in the same SQL statement as the read,
  with a recursive query; the memory store walks them under its lock.

The session store evaluates this limit inside each read, in SQL on SQLite and
PostgreSQL and as a predicate in the memory store. Labels play no part, and no
read is filtered afterwards. Paging and counts are therefore exact. The limit
is rebuilt from the current enrollment on every read, so narrowing `agents` or
removing the controller when the host restarts cuts its reads at once. A store
that does not declare `supports_fork_controller_reads` refuses every
controller read with `NotImplementedError` (501 over HTTP).

- Absent sessions, sessions of other agents and other controllers' forks look
  identical: single-session reads raise `ForkControllerDenied` (403 over HTTP),
  and lists, event queries and supervision pages leave them out. A
  `session_ids` entry the controller cannot read matches nothing.
- `SessionQuery(created_after=...)` (`created_after` over HTTP, timezone-aware)
  lists only sessions created strictly after that instant, so a reader can
  poll for new sessions.
- Controller queries cannot set fields that carry host authority. Session and
  supervision queries refuse `causal_budget_id`. An event query accepts only
  `session_id`, `session_ids`, `event_type`, `event_types`,
  `exclude_event_types`, `agent_name`, `environment_name`, `tool_name`, `since`,
  `until`, `after_sequence`, `before_sequence`, `limit` and `order_by`; budget,
  workflow and private linkage fields are refused with `ValueError` (422 over
  HTTP).
- `since` may be omitted from an event query, so a ledger can read events
  older than its window, such as `session.execution_profile.decided`. Each page
  is bounded by `after_sequence`, `limit` (at most 1,000 over HTTP) and
  `max_bytes` (default 1 MiB, at most 4 MiB). A page over `max_bytes` raises
  `EventQueryResultTooLarge` (413 over HTTP). HTTP pages report
  `next_sequence` and `has_more`. `max_bytes` bounds only the records a page
  returns: `has_more` is a separate existence check past the last one, so a
  page whose records fit is returned even when the next record would not fit
  with them.
- An event query that is not for exactly one session reads in ascending
  sequence order only; a descending one is refused with `ValueError` (422 over
  HTTP). It starts after `after_sequence`, 0 when omitted, so on PostgreSQL
  every page, the first one included, stops below any event a still-open
  transaction holds, and continuing from the last returned sequence never
  skips an event that commits late with a lower sequence. Single-session reads
  keep both orders.
- `exclude_forked_sessions` on event and supervision queries drops every
  session in a forked lineage: a session with a `session.forked` event, an
  overlay record or a forked lineage marker, one whose parents lead to such a
  session or through a fork, and one whose lineage does not settle. That
  includes the controller's own forks
  and everything created under them, so a supervisor's investigations and
  their subagents are not counted as the supervised agent's work. A
  supervision record's `forked` is true for exactly those sessions.
- Supervision pages never include content, and controllers cannot ask for it.
  Every other filter the control plane accepts, `release_id` included, applies
  over HTTP exactly as through the SDK. Cursors are bound to the controller and
  to the query's filters: a cursor issued to another controller, by the control
  plane, or for a differently filtered query is rejected
  (`SupervisionEventCursorInvalid`, 422 over HTTP).
- HTTP responses use the control plane's serializers, so session ids and
  linkage are the same public values the control plane publishes. The routes
  do not mount the control plane.

### Correlation ids

Public linkage aliases such as `cayu_event_12:tool_call_id` name one record, so
they cannot pair a tool call's start with its terminal. Controller reads add
correlation ids that can:

| Event | Correlation fields in `payload` |
| --- | --- |
| `tool.call.started` | `tool_call_correlation_id`, `tool_round_correlation_id`, and the proposing attempt's `model_step_correlation_id` and `model_attempt_correlation_id` |
| `tool.call.completed`, `.failed`, `.blocked`, `.approval_denied`, `.approval_expired` | `tool_call_correlation_id`, `tool_round_correlation_id` |
| `model.started`, `model.completed`, `model.error` | `model_step_correlation_id`, `model_attempt_correlation_id`; a completion that proposed tools also has `tool_round_correlation_id` |
| `model.auxiliary.attempt_started`, `model.auxiliary.attempt_settled` | `auxiliary_operation_correlation_id`, `model_step_correlation_id`, `model_attempt_correlation_id` |

Supervision records from `POST /fork-controller/supervision-events/query` also
carry `tool_call_correlation_ids`, `tool_round_correlation_id`,
`model_step_correlation_id` and `model_attempt_correlation_id`, which are null
or empty when the touch has no such linkage.

A ledger joins records with these values:

- Every `tool.call.started` has exactly one terminal with the same
  `tool_call_correlation_id`.
- Every `model.started` has exactly one `model.completed` or `model.error` with
  the same `model_attempt_correlation_id`. A retried step keeps its
  `model_step_correlation_id` and gets a new attempt value.
- Auxiliary attempts pair on `auxiliary_operation_correlation_id` plus
  `attempt`.
- A supervision record's `tool_call_correlation_ids` follow the touched calls
  in order and equal those calls' `tool_call_correlation_id`. Its round, step
  and attempt values equal the same fields on the session's tool and model
  events.

The rest of each record is the control plane's projection. `purpose` (for
example `context_compaction`), `usage_metrics`, `timestamp` and `sequence`
survive. So do the auxiliary attempt's `auxiliary_inference.purpose`,
`auxiliary_outcome` and `usage_metrics`, and a
`session.execution_profile.decided` event's `decision` and
`candidate_profile.fingerprint`. The private tool call, round, model step,
model attempt and auxiliary operation ids never appear in controller-read
output.

SDK reads return the store's own records. `event_correlation_ids(record)` and
`supervision_correlation_ids(event)` on `ForkControllerAccess` return the same
values the HTTP routes add.

Each value is an HMAC of the session id and the private id under the host's
public authority alias keyring. Each kind has its own field name inside the
value, so values of different kinds never match. The values are the same on
every read and after a restart with the same key material. Rotating the active
key changes them, so a reader that joins records rebuilds its joins after a
rotation instead of mixing values from both keys.

- Correlation ids carry no authority. No session, interaction, approval, input
  or linkage resolver accepts one, and approval and user-input resolution
  refuse them.
- Only derived values are published. A payload field with one of these names
  that the host did not derive is dropped.
- `max_bytes` bounds each published page with its correlation values. A larger
  event or supervision page returns 413.
- Correlation values need a configured alias keyring. Without one they could
  only come from a process-local key that changes on restart, so a SQLite or
  PostgreSQL store with no keyring publishes none: controller reads still
  work, the correlation fields above are absent from every event and
  supervision record, and `event_correlation_ids` and
  `supervision_correlation_ids` return empty dicts. That includes the
  `tool_call_correlation_id` the control plane derives from its process-local
  key. An in-memory store always has a random key for as long as its data
  lives, so its reads carry the fields.

### Bearer-token authentication

`ForkControllerBearerAuth` is a ready-made `authenticate` callback that maps
bearer tokens to subjects:

```python
from cayu.server.fork_controller import ForkControllerBearerAuth

# CAYU_FORK_CONTROLLER_TOKENS='{"<token>": "svc:compass"}'
authenticate = ForkControllerBearerAuth.from_environment()
# or ForkControllerBearerAuth.from_file("/run/secrets/fork-controller-tokens.json")
# or ForkControllerBearerAuth({"<token>": "svc:compass"})
app_server.include_router(create_fork_controller_router(app, authenticate=authenticate))
```

- The configuration is a JSON object from token to subject. Two tokens may
  name the same subject, so a token can be rotated without downtime.
- Each token must be an RFC 6750 bearer credential of at least 32 characters.
  At most 64 tokens are accepted.
- The callback keeps only SHA-256 digests of the tokens. It compares the
  presented token with every entry using `hmac.compare_digest`, without
  stopping at a match.
- A missing, repeated, malformed or unknown `Authorization` header
  authenticates no one, and the router returns 401.
- Configuration errors raise `AuthConfigurationError` and name only the entry's
  position. Tokens never appear in errors, logs or `repr`.

An overlay that grants tools changes the child's body, so it needs
`execution_profile_selection="current_child"` with a `profile_adoption` whose
`requested_by.subject` is the authenticated controller. An overlay with only
instructions and memory keeps the inherited profile and needs neither. Send it
with the default selection: built-in adoption covers only forks that grant
controller tools, so an instructions-only fork that asks for `current_child` is
rejected unless a configured `ExecutionProfilePolicy` adopts it.

## Authority rules

- `CayuApp.fork_session`, scoped resource access, runtime hooks and message
  text cannot attach an overlay. Only `ForkControllerAccess` can, and only for
  its own controller name and enrolled agents. The source session's agent must
  be enrolled, and the child keeps that agent: a fork cannot retarget a source
  outside the enrollment to an enrolled agent.
- An overlay fork belongs to the controller that created it, with or without
  granted tools: only that controller can fork it or snapshot it. Other
  controllers and ordinary host forks are refused, because the fork's durable
  overlay record names its owner. The same holds for every session created
  under the fork, whose lineage marker names the same owner.
- Agent names containing `@fork-controller:` are reserved for the bodies that
  enrollment registers.
- The host ceiling is still inherited or narrowed, and then limited to the
  controller's `host_tools` (see [Host tool allowance](#host-tool-allowance)).
  Controller tools never come
  from `tool_capability_ceiling`; only the overlay's grants add them, and each
  grant must name a tool the enrolled controller serves. The store accepts a
  child ceiling wider than its source only by tools the overlay grants, on a
  body named for that controller; re-granting a tool the source already had is
  not a widening.
- Controller tools are never inherited. A fork of an overlay fork keeps only
  the tools its own overlay grants.
- Host tool policy, approvals and runtime hooks apply to controller tools as to
  any tool. Tool policies see the host agent's name.
- Without an `ExecutionProfilePolicy`, Cayu adopts an overlay fork's profile
  only when the change is limited to the tool set (`direct_tools`,
  `tool_implementations`, `tool_view_grants`, `effect_authority`), the first
  invocation's budget, result shape and stop semantics
  (`invocation_budget_policy`, `structured_output`, `finalization`), and
  `runtime`, and only when the source's own registration is unchanged since it
  was created. If a host or controller registration changed in between, the
  change is not the overlay's, and adoption needs a configured policy. An
  unchanged profile, such as a re-fork of an overlay fork with the same tools,
  is adopted the same way. The decision is recorded with policy identity
  `cayu:fork-controller-overlay-adoption:v1`. With a configured policy, the
  policy decides.
- Agent-scoped budgets are counted by exact agent name, so they would not see a
  body's spend. While the host agent has an agent-scoped budget, runs of its
  fork-controller bodies fail closed. App and causal budgets apply as usual;
  forks share the source's causal budget.
- Scoped application subjects cannot resume or recover an overlay fork; the
  controller continues it through its own entrance.
- Tools in an overlay fork cannot publish to the host's knowledge store:
  `remember_knowledge` returns `publication_owner_closed`, so overlay memory
  never becomes host knowledge through an inherited tool.

## Host tool allowance

The host, not the controller, decides which host tools a controller's forks may
use. The controller's `tool_capability_ceiling` can only narrow that further.

- When a fork omits `tool_capability_ceiling`, the inherited host tools are
  narrowed to the enrollment's `host_tools`. With the default allowance, the
  fork has no host tools.
- When a fork's `tool_capability_ceiling`, its `initial_invocation`'s ceiling
  or a resume's ceiling names a host tool outside `host_tools`, the request is
  denied with `ForkControllerDenied` (403 over HTTP). This happens before any
  child is created and before any model or controller tool call. Cayu rejects the request instead of dropping the tool, so a
  controller never runs without a tool it asked for.
- Controller-served tools that the overlay grants are not part of the
  allowance and are unaffected by it.
- Provider-hosted tools such as OpenAI web search come from the agent's
  registration, not from the ceiling, so a fork cannot drop them. A controller
  can fork, replay or resume a fork of an agent registered with a hosted tool
  only when `host_tools` names that tool's type (for example
  `"openai_web_search"`). Otherwise the request is denied before any child or
  model request. Enrollment rejects a hosted type no enrolled agent registers,
  and any hosted type when the controller may run models, since the per-fork
  cap cannot bound hosted tools.
- Every replay of an existing fork and every resume through the controller
  entrance checks the child's durable ceiling against the current allowance.
  If the host has since removed a tool from `host_tools` that the child's
  ceiling still names, the replay or resume is denied and the child's ceiling is not
  rewritten. Restoring the allowance makes the same fork replay again.

## Operating mode

`observe` is the only operating mode. An observing controller forks host
sessions, continues its own forks when the host allows model runs, reads the
sessions of its enrolled agents, and describes its enrollment. The
controller entrance has no way to write to the session it forked from: it
cannot add messages to it, steer it, resume it or resolve its approvals, and a
fork cannot use it as the destination. Modes that write to the trunk are not
available.

## Instructions and memory

The overlay layer is one user-role message after the selected history:

```text
<cayu_fork_overlay version="1">
{"controller":"compass","instructions":"...","memory":[...],"scope":"this_fork_only"}
</cayu_fork_overlay>
```

It is rebuilt from the child's durable record for every model request. It is
not appended to the transcript, not written to any knowledge store, and does
not change the source session, its transcript or its registered body. Because
the source history and system prompt are unchanged, the shared prompt prefix is
preserved. The layer is added after context selection, so `instructions` and
`memory` together are capped at 32,000 characters
(`FORK_OVERLAY_CONTEXT_MAX_CHARS`). Overlay text containing a value from the
app's `SecretRedactor` is rejected; vault secrets that the redactor does not
know about are not checked.

## Identity, restart and results

The overlay is part of the fork request, so it is inside the fork request
digest that the child's lineage records. A generated child id is derived from
it. `session.forked` also carries `fork_overlay_controller`,
`fork_overlay_fingerprint` and `fork_overlay_tool_names`. The child stores the
exact overlay with its fingerprint under runtime-owned metadata.

- Retrying the same fork after a lost acknowledgement or restart returns the
  same child and the same `session.forked` event, while the controller is still
  enrolled for that child's agent; the source session need not still exist. A different overlay for the
  same `session_id` conflicts.
- A changed overlay record fails closed before any model request, and so does a
  fork-controller body session whose overlay record is missing.
- A changed controller tool contract changes the child's execution profile, so
  resume fails closed until the change is adopted. A missing enrollment makes
  the body unregistered and resume fails.
- MCP adapters with an explicit `connection_id` now have a portable
  execution-profile identity based on the server's tool contract, so controller
  tools keep their identity across processes.

Results come back as an ordinary child outcome. To deliver a finding to the
trunk, use the durable session-message path (`enqueue_session_message`), not
the overlay.
