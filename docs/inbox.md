# The agent's inbox

The inbox is what an agent is waiting on a person for: tool approvals, questions
the agent asked with `ask_user`, recovery items after an interrupted tool call, and
knowledge proposals waiting for review. Each item is the agent's own record, so a
decision made in one place is settled everywhere: Cayu Cloud, the agent's own app
and its Developer dashboard (`/cayu/`) all read and settle the same records, and the
settled record names the person who decided.

The optional server serves it next to the rest of the control plane:

| Route | Does |
| --- | --- |
| `GET /api/inbox` | Lists what is waiting: approvals, questions and recovery items, then knowledge proposals |
| `GET /api/inbox/{item_id}` | Reads one item's current state, wherever it was settled |
| `POST /api/inbox/{item_id}/settle` | Approves, denies, answers, keeps or drops one item as the caller |

`GET /api/contract` advertises the routes under `capabilities.surfaces.inbox`. A
browser app uses them through the served `client.js`; see
[Application UIs](../src/cayu/guides/app-ui.md#show-the-agents-inbox). Every response
is sent with `Cache-Control: private, no-store`.

## Items

An item has an `id`, a `kind` (`approval`, `question`, `recovery` or `knowledge`),
`title` and `detail`, `created_at`, and the `decisions` it accepts:

| Kind | Comes from | Decisions |
| --- | --- | --- |
| `approval` | A tool call a `ToolPolicy` sent for approval. `detail` is the policy's `reason`; `tool_name` names the tool. | `approve`, `deny` |
| `question` | An `ask_user` call. `question` and `options` hold what the agent asked. | `answer` |
| `recovery` | A tool call whose outcome is unknown after an interruption. | none: settle it in the Developer dashboard, with the evidence of what happened |
| `knowledge` | A knowledge entry waiting for review. `knowledge` names its `entry_id`, `revision` and `kind`. | `keep`, `drop` |

Approvals, questions and recovery items also carry `session_id` (the public session
alias), `agent_name` and `attention_id`, the same human-attention identity that
`GET /api/pending-actions`, supervision events and the
[attention handoff](runtime-contracts.md#human-attention-handoff-receiver-protocol)
use. A delegated parent waiting on its child is not listed; the child's own item is.

They also carry `execution_deadline`, the paused session's stored execution
deadline, or `null` when it has none. `expires_at` and `source` say when and who
set it; `origin` says how the session got it: `explicit` (the request set
`execution_deadline`), `run_default` (the default for runs started inside a task
handler, from `run_task_worker(execution_timeout_s=...)`) or `enclosing` (an
enclosing `execution_deadline_scope`), and `null` for a session started before
Cayu recorded origins. `waiting_counts` is `false` when settling the item gives
the session back the time it had left when it paused, which is the case for a
top-level session whose deadline is `run_default`. Otherwise the deadline keeps
running while the item waits, and a settle after it passes is refused with 409
`conflict`, naming the deadline's expiry and origin. To continue such a session,
settle it from code with a new deadline, for example
`app.resolve_tool_approval(ToolApprovalRequest(..., execution_deadline=ExecutionDeadline.after(3600)))`.
`GET /api/pending-actions` returns the same `execution_deadline` on each action.

An `id` names one exact pause or proposal revision. A new question, or a revised
proposal, is a new item with a new id, so an id never changes meaning under a
reader. It is a locator, not a credential: every read and settle re-derives the item
from durable state.

Text fields are untrusted and already redacted for known secrets; render them as
escaped text. The Runtime does not show a tool's proposed arguments before it runs
the call, so an approval shows the policy's reason, not the arguments. To show a
person more, configure a [human-review policy](human-review.md), which decides what
each reviewer may see.

`GET /api/inbox` accepts `session_id`, `agent_name`, `kind` (`approval`,
`question`, `recovery`, `knowledge`), and `q`. Filters combine with AND.
Session, specialist and kind filters constrain the source query; public-text
search filters each bounded page after access checks and redaction. Knowledge proposals have no session or
specialist identity, so session/specialist filters exclude them. Blank filters
and unknown kinds return 422. A cursor belongs to its filters; changing them
requires starting without a cursor. The browser client's `listInbox` and
`watchInbox` accept these same filters.

`GET /api/inbox/counts` accepts `session_id`, `agent_name` and `q` and returns
`counts` for all four kinds, `total_count`, `complete`, and `unavailable_reason`.
It follows the same bounded source pages to exhaustion, independently of the
list's page size, deduplicating exact item identities. It never returns a capped
subtotal. With an inbox access rule, `counts` and `total_count` are null,
`complete` is false and `unavailable_reason` is `access_limited`; no hidden
store totals are exposed. Source issues or conflicting representations of an
item similarly return null counts with `source_incomplete`. Read failures and
cancellation do not publish partial totals. A repeated cursor returns 409.
The queue is live, not a transactional snapshot across sources; refresh after
settling an item. `client.getInboxCounts` reads this endpoint. The read-only
`inbox_queries` capability advertises filters, counts and argument availability.

Approval and recovery items explicitly return `arguments: null` with a reason.
`arguments_unavailable_reason: "no_review_policy"` means the app configured no
human review policy, so it shows no arguments for approvals; configure one such
as `ToolArgumentReview` ([human review](human-review.md)) to show selected
arguments. `"quarantined"` means the app has a policy: the inbox still lists no
arguments, and the policy decides what protected human review shows. Other kinds
return a null reason.
The public API never reconstructs executable arguments from private checkpoints
for display. The separately authorized human-review policy remains the route
for reviewer-specific disclosure; this field does not bypass it.

The list is paged. Follow `next_cursor` until it is `null`, including when a page
has no items. Knowledge proposals come after the session items, starting pages of
their own. Each request scans at most ten knowledge store pages of 200 proposals,
applying the inbox access rule before returning up to `limit` items. The cursor
continues after the last scanned proposal, including hidden proposals, so those
proposals cannot block later permitted ones. Settling or deleting preceding
proposals does not shift this keyset position. The queue is live: a proposal whose
priority or update time changes can move before the position; refresh to see it.
`issues` names sessions too large or invalid to list and reports an incomplete
source scan when the source cannot supply a continuation.

Paging needs no configuration: `cayu serve`, `create_server`, `mount_cayu` and
`create_router` page every inbox to its end. A caller no
[inbox access rule](#who-decides-and-who-may) limits may see every item, so its
cursor holds a plain keyset position. A caller the rule limits gets pages of its own
session items only: each session page continues after the last item the caller can
see, so a cursor never names other people's sessions, and the store's issues, which
name those sessions, are left out. A page can therefore hold fewer than `limit`
items, or finish a session's items past it. Each request reads at most ten store
pages of `limit` actions; when they hold only items the caller may not see, the page
is empty and its cursor seals the position the scan reached, so the caller keeps
paging to its own items behind them. A limited caller's knowledge cursor is always
sealed, since it can name a hidden proposal. `GET /api/pending-actions` pages a
limited caller the same way: a sealed cursor starts with `sealed_`, and `has_more`
always comes with a cursor.

A sealed cursor is encrypted and authenticated with AES-256-GCM, bound to the
verified subject and tenant and to the listing (the knowledge review scope and
query, and the inbox or pending-action filters), and expires after 15 minutes. Decoding
permits an issuing worker's clock to be up to 60 seconds ahead; it does not extend
the 15-minute expiry measured from the authenticated issuance time. Every page
rechecks current authorization.

The key is `inbox_cursor_key`, an optional 32-byte key passed to `create_server`,
`mount_cayu` or `create_router`. Without one, each server generates a key of its
own when it starts, so its sealed cursors stop working when it restarts and don't
work on another worker. When several workers serve an inbox with an access rule,
give them the same key, kept in the install's secret store, so a cursor works on
any of them; keys are specific to one inbox deployment, so don't share them between
applications. With a configured key, restarting preserves unexpired cursors and
rotation invalidates them. Either way, a malformed, expired or wrongly scoped
cursor, or a sealed cursor the key can't open, returns 422; restart the list from
no cursor. `client.watchInbox` does this once within the same refresh, dropping
the pages it read, and stops with the 422 as its error if the restarted listing is
refused too.

The key is a trusted server construction argument, not an HTTP field or a
serialized `ServerConfig` value. The host must load it from its secret source.
For example, a deployment can provision a base64-encoded secret named
`CAYU_INBOX_CURSOR_KEY_B64` and wire it explicitly in its server entry point:

```python
import base64
import os

cursor_key = base64.b64decode(os.environ["CAYU_INBOX_CURSOR_KEY_B64"], validate=True)
server = create_server(
    app,
    config=ServerConfig.protected(auth),
    inbox_access=inbox_access,
    inbox_cursor_key=cursor_key,
)
```

This environment name is an example host convention; Runtime does not read it
automatically. Hosts using `mount_cayu` or `create_router` pass the same decoded
bytes to that function instead. Neither this code nor Runtime prints a key.

## Item state

`GET /api/inbox/{item_id}` answers with the item's `status`:

| Status | Means |
| --- | --- |
| `pending` | Still waiting. `item` holds the item. |
| `settled` | Decided. `settlement` holds the `decision` (`approve`, `deny`, `expire`, `answer`, `recover`, `keep` or `drop`), `decided_by`, `decided_at` and the optional `reason`. |
| `superseded` | Replaced before anyone decided it: an operator interrupt replaced the question, or the proposal was revised. |
| `unavailable` | The state can't be read right now, for example while the Runtime is between recording a decision and publishing it. Read it again. |

`decided_by` is the person the Runtime recorded, with `subject`, `tenant` and
`source`. Knowledge decisions made through the inbox retain authenticated reviewer attribution.
A decision made outside a reviewed workflow can have no recorded reviewer.
Under open local access, session decisions have no authenticated actor and
reviewed knowledge decisions use the trusted-local identity. An approval that expired before anyone decided it is
settled with the decision `expire` and the system actor `cayu:approval-expiry`.

Search `q` is a case-insensitive substring match against the item's public,
redacted text and identities. Knowledge search covers its visible title and
preview, not undisclosed source text. Arguments are never searchable. Search
runs after access and redaction and may produce an empty page with a continuation;
counts traverse those same pages to exhaustion. Changing `q` invalidates a cursor.

## Settle an item

```http
POST /api/inbox/{item_id}/settle
Content-Type: application/json

{"decision": "approve", "reason": "Checked against the vendor master."}
```

| Field | For | Meaning |
| --- | --- | --- |
| `decision` | all | `approve`, `deny`, `answer`, `keep` or `drop`; it must be one of the item's `decisions` (422 otherwise) |
| `reason` | all | Optional rationale, durably bound to the decision; returned only when its public audit policy permits |
| `arguments` | `approve` | The changed arguments the tool runs with: approve with changes. The tool's schema and policy check them again; a refused edit leaves the approval pending (409 `conflict`). |
| `approval_revision` | `approve` | `{"proposed_digest", "approved_digest"}` naming an edit the application stored itself |
| `answer` | `answer` | The answer to the question |
| `review_reference` | `approve`, `deny`, `answer` | Required when the application configures a human-review policy; read it from `GET /api/sessions/{id}/human-review`. It can't be combined with `arguments`. |
| `task_worker_id`, `task_handoff_id` | `approve`, `deny`, `answer` | Exact task continuation authority, as on the ordinary resolve routes. `task_handoff_id` requires `task_worker_id`. |
| `task_continuation` | `approve`, `deny`, `answer` | Optional `"claim"` explicitly delegates a workerless interrupted task continuation to the runtime. Cannot be combined with worker/handoff IDs. |
| `profile_adoption` | `approve`, `deny` | `{"idempotency_key", "reason"}`: settle an approval after an upgrade changed the application's execution profile (see below). The caller is the requester. Without authentication, `requested_by` names them, otherwise the request is recorded as `unauthenticated-local-development`. |

With a configured `ControlPlaneRoles` policy, changed arguments and approval
revisions require a role in that policy's `full_access` set. The maintained server
wrapper applies this check to proxy authentication and custom auth dependencies
that expose the policy; a role spelling or unrelated claim does not grant access.

Without `task_continuation`, task-linked decisions retain the ordinary exact
worker/handoff authority requirements. With `"claim"`, the runtime claims its own
receipt-backed continuation for the selected task/session incarnation and pause;
it never borrows another worker's IDs or takes over an active lease. A competing
worker or changed handoff returns 409. Stores without bound, cancellation-quiescent
claims return 501. Decisions without an attached task use ordinary resolution.
The runtime revalidates the pause and protected review before dispatch, and the
authenticated person remains the recorded decider. Generated worker authority is
not returned to the browser. Observing an item never claims work.

A successful decision response means the decision was durably accepted; execution
may still be running. Closing the HTTP observer does not stop that continuation.
Cancelling its execution owner fences the task and retains ownership until already
dispatched work has drained; it does not prove a tool has stopped.

Answer reasons use the documented `UserInputResponse.metadata` key
`cayu.user_input.decision_reason`. It is included in the existing exact answer
and recovery digests. Approval and answer operator text is published in events
and settlement views only when the pause has positive `static` secret-scope
evidence. Dynamic or unknown scope withholds that text, including after manual
recovery; `settlement.reason` is null. This does not remove the private exact
request binding: a different reason still conflicts on replay. Recovery must
retain the answer metadata; its separate `reason` still describes the recovery.
Knowledge keep reasons are part of the exact reviewed-approval annotations.
Reviewed drops atomically publish an archived revision and its exact publication
receipt, preserving chunks and source evidence. They use the same bounded source
material limits as reviewed approval and refuse incomplete material before any
write. Repeating the same actor and reason replays the decision; changing either
returns a conflict. Stored annotations without matching publication evidence are
not a reviewed rejection.

The call returns once the Runtime durably accepted the decision. The agent carries on
in the background, as it does after a dashboard client disconnects, so a settle never
waits for the rest of the run. The response is the settled item:

```json
{
  "replayed": false,
  "item": {
    "id": "inbox_...",
    "kind": "approval",
    "status": "settled",
    "session_id": "close-september",
    "agent_name": "close-agent",
    "item": null,
    "settlement": {
      "decision": "approve",
      "decided_by": {"subject": "user_01J...", "tenant": null, "source": "http_auth"},
      "decided_at": "2026-10-21T09:14:03Z"
    }
  }
}
```

A refused settle answers 409 with the item's current state in `detail.item`, so the
caller can show who decided instead:

| `detail.code` | Means |
| --- | --- |
| `already_settled` | Someone settled it first, or this caller settled it differently |
| `superseded` | It was replaced before it was settled |
| `expired` | The approval expired before anyone decided it |
| `unavailable` | Its state can't be read right now; try again |
| `not_settleable` | A recovery item; settle it in the Developer dashboard |
| `conflict` | The Runtime refused this decision and the item is still pending, for example an edit its schema rejects |
| `profile_changed` | The application changed while the approval waited and the decision didn't adopt the change; settle again with `profile_adoption` |

An unknown id and an item the inbox access rule hides both answer 404, so the routes
don't reveal what exists.

### Approvals across an upgrade

A pending approval keeps the execution profile its run started with. When only
Cayu changed and the application has no `ExecutionProfilePolicy`, approving or
denying adopts the new Cayu version on its own. Any other change, such as a new
tool version, answers 409 `profile_changed` until the settle carries
`profile_adoption` and the application's policy adopts it. A settle that already
carried `profile_adoption` and was refused answers 409 `conflict`. Before anything runs,
the paused call is checked against the current tool and policy; if they would no
longer authorize it the same way, the settle answers 409 `conflict` and the
approval stays pending, so deny it and let the agent propose the call again. A
change to the agent's tool set (a tool added, removed or redescribed) can't be
adopted: approving or denying answers 409 `conflict`, and only the application
version that proposed the call can settle it. The
Developer dashboard offers the same decision under the update after a
`profile_changed` refusal. Use a new `idempotency_key` for each adoption.

### Repeating a settle

Settling is idempotent per person and decision. Sending the exact same decision
again, after a lost response or from a second tab, answers 200 with
`"replayed": true` and the recorded settlement; nothing runs twice. "Exact" is the
Runtime's own rule for retrying a resolution: the same person, decision, reason,
arguments and answer. Any other settle of a settled item gets 409 `already_settled`.
A person's identity is `subject`, `tenant` and the verified source, not the token's
other claims, so a refreshed token repeats the same request.

For a knowledge proposal, both Keep and Drop bind the exact proposal revision,
reviewer and optional reason to a recorded decision. Repeating that decision as
the same reviewer with the same reason replays its receipt; changing the reviewer
or reason conflicts. A decision made outside the reviewed workflow may have no
recorded reviewer; that is not a reviewed Drop receipt.

## Who decides, and who may

The person who decides is always the verified caller: the server's authentication
produces the `AuthContext`, and the settlement records its `subject` and `tenant`
with source `http_auth`. A request can't name anyone else. A service acting for a
person, such as Cayu Cloud, calls with a token that identifies that person, so the
Runtime records the person and not the service.

By default every caller the server's authentication admits may see and settle every
item, as with the rest of the control plane. A deployment that decides per person
passes an inbox access rule to `create_server(..., inbox_access=...)` or
`mount_cayu(..., inbox_access=...)`:

```python
from cayu.server import AuthContext, ServerConfig, create_server
from cayu.server.inbox import InboxItemScope


async def inbox_access(person: AuthContext, item: InboxItemScope) -> bool:
    # Each tenant sees its own sessions; recovery needs an operator.
    if person.tenant is not None and item.origin_tenant != person.tenant:
        return False
    return item.kind != "recovery" or "operator" in person.claims.get("roles", [])


server = create_server(app, config=ServerConfig.protected(auth), inbox_access=inbox_access)
```

The rule sees an `InboxItemScope`: the item's `kind`, `agent_name`, `session_id`
(public alias), `attention_id` and `origin_tenant` (the tenant its session was
started under), or `knowledge` (`entry_id`, `revision`, `kind`) for a proposal.
Every route describes a session item with that same complete scope, including
its exact `attention_id` during recovery and when retrying a settled decision.
Returning anything but `True` hides the item from that caller. The rule needs
authenticated access; with open access, `create_server` and `mount_cayu` refuse it.
The claims it reads are whatever the authentication dependency put in
`AuthContext.claims` (for `OidcBearerAuth`, its `claims_mapper`).

The rule is one decision point: the server applies it on every route that lists or
settles an item, not only on the inbox routes.

| Route | A hidden item |
| --- | --- |
| `GET /api/inbox`, `GET /api/pending-actions`, `GET /api/knowledge/pending` | Isn't listed |
| `GET /api/inbox/{item_id}`, `GET /api/knowledge/pending/{entry_id}` | 404 |
| `POST /api/inbox/{item_id}/settle`, `POST /api/tool-approvals/resolve`, `POST /api/tool-approvals/recover`, `POST /api/tool-rounds/recover`, `POST /api/user-input/resolve`, `POST /api/user-input/recover`, `POST /api/tool-authorization-grants/approve`, `POST /api/knowledge/{entry_id}/approve` and `/reject` | 404, before anything is replayed or changed |
| `GET /api/sessions/{id}/human-review` | 403 |
| `GET /api/human-attention/{attention_id}` | `unavailable` |
| `POST /api/supervision-events/query` | The touch that settled it isn't returned; interrupts and steering settle no item and are unaffected |

Because the store pages and counts across every session, a caller the rule limits
gets pages of its own items: `GET /api/pending-actions` and `GET /api/inbox` leave
out delegated parent rows and the store's issues, which name other sessions,
continue session pages after the caller's last visible item (or from a sealed
position past other people's items, as [paging](#items) describes), and report no
store-wide count (`total_count` is null, and `total_entries_known` counts only what
was returned).

The rule governs the items. Session reads and session operations (events,
transcripts, state, the follow stream, messages and interrupts) stay the operator
surface; don't give a person the rule hides items from access to those routes.

A verified tenant is provenance here, as in the rest of the control plane: it
doesn't filter the inbox by itself. To keep each tenant to its own sessions, check
`item.origin_tenant` in the rule, as above. The human-attention and supervision
reads keep their own tenant filter as well.

When the application configures a [human-review policy](human-review.md), approving,
denying and answering also need its `review_reference`, issued to the same person,
exactly as `POST /api/tool-approvals/resolve` and `POST /api/user-input/resolve` do.

## One record, three places

The inbox doesn't keep a copy. Settling an item claims the same pending approval or
question that `POST /api/tool-approvals/resolve` and `POST /api/user-input/resolve`
claim, and keeps or drops the same entry `POST /api/knowledge/{id}/approve` and
`/reject` do. So a decision made anywhere shows everywhere:

- `GET /api/inbox/{item_id}` reports it settled, with the person, wherever it was
  decided.
- `GET /api/pending-actions` stops listing it, and the Developer dashboard's
  Inbox and session page show it resolved.
- `POST /api/supervision-events/query` returns the touch (`approve`, `edit`, `deny`,
  `answer`) with the same `attention_id` and actor.
- `GET /api/human-attention/{attention_id}` and the attention handoff report it
  `resolved` or `cancelled` once the Runtime closes it.

## From the command line

`cayu inbox` lists, shows and decides the same items from a terminal:

```console
$ cayu inbox list
$ cayu inbox show inbox_...
$ cayu inbox approve inbox_... --reason "Checked against the vendor master."
$ cayu inbox deny inbox_... --reason "Wrong vendor."
```

Without `--application`, the commands open the project's application (`[tool.cayu]
factory`, or `--target module:factory`) and its store in this process, and serve
these same routes to themselves. A project with a public service (`[tool.cayu]
service`) is opened through it, so its sessions continue with the policies
`cayu serve` gives them. Decisions are recorded as `--reviewer`, by default
`local:` followed by your user name. After an approval, the approved call and the
rest of the turn run in this process, so `approve` returns when the session pauses
again or ends and the work it continued (such as a parent an approved child resumes)
has finished, or after `--wait-seconds` (default an hour), when what is left is
stopped and can be resumed later.

With `--application NAME`, they read a Cayu Cloud Agent through Cloud's inbox routes
with your `cayu cloud login` (or `--context`). Cloud carries a decision to the Agent
as you, so it needs Cloud sign-in on the Agent, and the Agent continues on its own.

`show` prints the item and, for a pending approval or question, the
[human review](human-review.md) the application's policy builds for you: the
arguments it shows, the calls the decision covers, and its guidance. It never shows
more than that policy allows, and with no policy it says that no arguments can be
shown. `--full` reads values the review shows only in part, whole. `--purpose`
selects the review purpose (the application's own by default, or `approvals` for a
Cloud Agent). When the application has a policy, `approve` and `deny` ask for a
fresh review and send its `review_reference` with the decision; a review that
withholds the call's content can only be denied. For a session whose task worker let
it go while it waits, `--claim-task` sends `task_continuation: "claim"`.

The output is a table by default and JSON with `--json`, with a `schema_version`.
An item someone else settled first is reported with who decided and when.
