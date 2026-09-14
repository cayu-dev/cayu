# Human-attention notifications

Use this integration when a person must answer a question, approve a tool, or
reconcile a manual-recovery gate after leaving the live stream. Runtime owns the
pending execution state and validates resolution. Your application owns recipient
scope, channel credentials, durable delivery, reminders, and routing. The agent
only calls `ask_user` or encounters its configured approval/recovery gate; it does
not call a separate notification tool.

The runnable, credential-free companion is
[`examples/human_attention/`](https://github.com/cayu-dev/cayu/tree/main/examples/human_attention/).
It uses SQLite for Runtime state and a separate SQLite database as a durable test
destination. Replace that destination with a service that durably commits before
acknowledging. Keep vendor adapters outside Runtime core. The broader operational
lifecycle remains `cayu guide durable-operations`.

## Supported entry points

```python
from cayu import HumanAttentionRequest, PendingActionQuery

page = await app.session_store.query_pending_actions(
    PendingActionQuery(limit=100)
)
for action in page.actions:
    notification = HumanAttentionRequest.from_pending_action(action)
    if notification is None:
        # A delegated parent is navigation only. Inspect its child within your
        # authorized scope. A legacy/custom row without incarnation is unavailable.
        continue
    current = await app.get_human_attention_state(notification.reference)
    if current.state == "active":
        await destination.accept(notification)  # return only after durable commit
```

`destination` is application-owned. The example's `Inbox.accept` makes one atomic,
unique insert, and `DurableAttentionSink.emit` commits event hints before returning.
`get_human_attention_state` is a read-only SDK surface that returns bounded state
and reason fields; it never claims delivery, retries tools, answers, or grants an
approval. Direct SDK callers must authorize access to the referenced session.

These are supported discovery contracts:

| Entry point | Contract |
| --- | --- |
| `SessionStore.query_pending_actions(PendingActionQuery(...))` | Bounded authoritative current actions, issues, and pagination |
| `PendingActionRecord.attention_id` | Stable logical attention identity, or null when no safe actionable identity exists |
| `PendingActionSession.instance_id` | Session incarnation, read in the same store snapshot as the action |
| `HumanAttentionRequest.from_pending_action(action)` | Fixed content-free summary plus exact SDK correlation reference |
| `CayuApp.get_human_attention_state(reference)` | Current action or positive terminal evidence; absence alone is unavailable |
| `EventSink.emit(event)` | At-least-once refresh hint delivery, acknowledged only after durable destination acceptance |
| `CayuApp.recover_persisted_event_side_effects(limit=...)` | Bounded recovery of committed event fan-out |

The optional server's protected `GET /api/pending-actions` includes `attention_id`
and the existing public resolution/navigation references. HTTP clients use those
public `input_id`, `approval_id`, `round_id`, and `tool_call_id` fields with the
corresponding protected resolver. They must not decode public event IDs into
execution authority. SDK `HumanAttentionReference` values belong inside the
trusted consumer/resolver boundary; HTTP responses use the server's normal
public-ID projection instead. Do not reconstruct an SDK reference from arbitrary
callback data or a public event payload.

## One underlying action

Deduplicate within a configured Runtime-store namespace by `attention_id`, not by
the display row `id`, event sequence, question text, or session ID alone. The
attention ID binds session ID, durable session incarnation, kind, and exact pause
identity. Manual recovery also binds its round/call identity. Deleting and creating
a session with the same name creates a different attention ID. Store namespaces
are deployment-owned: never collapse unrelated Runtime databases into one dedup
namespace, and keep identity keying stable across consumers/restarts.

For user input, both committed `session.awaiting_user_input` and
`session.interrupted` prompt refresh. The first can arrive before the interrupted
boundary is discoverable; the latter is the durable paused boundary. Both lead to
one attention ID. A missed early observation is repaired by later hints and
periodic queries, not by manufacturing an action from the event.

A `delegated_action` row represents a parent waiting on a child. It has no
attention ID and does not duplicate child question/approval content. Follow
`delegated_action.child_session_id` within authorized scope, query that child's
current pending action, and use the child's attention ID and exact resolver.
Nested delegation follows the same rule. An all-session authorized consumer will
discover the child directly. A parent-only enrollment must explicitly include an
authorized child scope or show the navigation request to an operator; it cannot
assume permission to inspect or answer the child.

## Safe notification content

The default summaries are exactly “User input required.”, “Tool approval
required.”, and “Manual recovery required.” The projection contains identities,
not questions, choices, arguments, checkpoints, transcripts, prompts, or arbitrary
exception messages. The demo's inspection output shows only the attention ID,
kind, summary, and notification state.

Keep detailed inspection access-controlled. Use `app.inspect_human_review(...)`
with a configured `HumanReviewPolicy` and verified `HumanReviewContext` when a
human needs permitted question/approval content. Recipient authorization and
policy-controlled disclosure are distinct from sink delivery. Do not copy private
`HumanReviewSource`, quarantined arguments, or generic raw event payloads into a
notification. Known-secret redaction alone cannot establish safe disclosure of
unknown deployment data. Application-owned IDs and links also need appropriate
scope; use the server's public session link projection for external navigation.

## Durable delivery and repair

1. Configure a persistent SessionStore and the same destination sink in every
   producer/recovery process. The durable handoff protects configured fan-out
   after commit. An `emit` return means the destination has committed durable
   work, not that a network request was launched or a background coroutine exists.
2. Let the sink enqueue only a bounded refresh hint. The example deduplicates
   immutable callbacks by `(session_id, event_id)` in a deployment-specific inbox.
   Notifications deduplicate separately by `attention_id`. Delivery can repeat
   after acceptance but before acknowledgement; callbacks may arrive out of order.
3. On consumer startup and periodically, enumerate authoritative pending actions.
   The example explicitly enrolls **all current actions**, including those that
   predate consumer enrollment. An application that intentionally excludes older
   work must persist an enrollment boundary and document that policy; filtering
   only future events otherwise silently misses existing pauses.
4. Follow `next_cursor` with unchanged filters. `limit` is bounded at 200.
   `has_more`, `issues`, byte-limit errors, repeated/malformed cursors, store errors,
   or a consumer page budget mean incomplete inspection. Never mark unseen work
   resolved because it was absent from one page. Session updates can move rows
   during pagination; repeat full scans for eventual repair.
5. Recheck `get_human_attention_state(reference)` before dispatching delayed work.
   A late opening event cannot recreate a settled request. The destination keeps
   terminal states monotonic with a conditional update, so a concurrent stale
   consumer cannot reopen one. A state can change immediately after a read; a
   notification is an invitation to inspect, never permission to execute.
6. Reconcile already-enrolled references independently of the global scan. A
   deleted session, a different incarnation, truncated query, store outage, or
   missing retained terminal evidence is `unavailable`, not resolution. Keep the
   notification/evidence and retry later. Do not pretend the canonical store is
   available by treating the notification database as pause authority.

The example caps work per invocation with `max_pages` and leaves event hints
unprocessed after an incomplete scan. Each destination write is independently
committed and idempotent. A consumer crash after a notification insert but before
acknowledgement is safe: another consumer repeats the same logical acceptance.
Notification failure never clears the pending action or rolls back a committed
pause. Polling complements durable fan-out and repairs enrollment or configuration
gaps; it does not grant exactly-once external delivery.

Refresh on approval request/approved/denied/expired events, input awaiting and
interrupted events, `session.delegated_action.updated`, checkpointed/resumed/
completed/failed session events, and completed/blocked/failed tool events. This is
a wake-up set, not a second lifecycle reducer. A `session.resumed` event does not
prove an answer completed; execution may still be uncertain. Do not subscribe to
or emit health changes through the same sink to repair its own failures.

## Interpreting state

| State | Evidence and consumer behavior |
| --- | --- |
| `active` | The exact incarnation-bound pending action is currently queryable; inspect again before deciding |
| `resolved` | Exact user-input/approval closure or matching definite manual-recovery evidence (including an operator-confirmed failed outcome); suppress future reminders |
| `cancelled` | Confirmed denied approval or blocked manual recovery; no execution grant is implied |
| `superseded` | Runtime's exact user-input supersession evidence, including an operator interruption; it is not an answer |
| `expired` | Runtime committed approval expiry and closure; the consumer did not invent a deadline |
| `unavailable` | Read failed, incarnation/session changed, query incomplete, or terminal evidence is absent; preserve and retry |

The inspector bounds terminal-event inspection at 100 records and returns
`unavailable` when that bound prevents proof. It does not expose receipt bodies or
source events. User-input closure and supersession use the existing exact durable
receipts/evidence. A committed manual-recovery failure resolves the attention
request even though the tool failed. Synthetic failures marked outcome-unknown or requiring manual
reconciliation remain unavailable. Manual recovery without a matching definite
result remains unknown. No user-input expiry is introduced. Approval expiry still
gates only the first grant: recovery of an approval granted in-window can continue after its
wall-clock expiry. An expired timestamp alone does not let a notification service
cancel authorized recovery; the resolver owns that decision.

## Answer through Runtime

A delivery receipt acknowledges transport only. It cannot resume a session.
After authenticating and authorizing the person, refresh the current action and
use the existing exact-action API:

- Question: `app.resolve_user_input(UserInputResponse(...))` or
  `POST /api/user-input/resolve`.
- Approval: `app.resolve_tool_approval(ToolApprovalRequest(...))` or
  `POST /api/tool-approvals/resolve`.
- Manual recovery: inspect its gate, obtain the independently verified outcome,
  and use `recover_user_input`, `recover_tool_approval`, or `recover_tool_round`
  with the corresponding typed recovery request. Never blindly retry the tool.

When human review is configured, pass the current review reference; notifications
cannot replace that authorization/content binding. The CLI demo treats its local
operator as trusted. It is not an internet-facing authentication handler.

## SDK and optional-server recovery

SDK-only deployments must own a scheduled loop that invokes bounded
`recover_persisted_event_side_effects(limit=...)` sweeps, waits between sweeps, and
keeps failures observable. The CLI `consume` command performs one bounded sweep
and one bounded reconciliation invocation. Run it on a deployment-owned schedule.
Respect existing durable retry deadlines; do not spin on failing deliveries.

For the optional server, construct `create_server(app, config=ServerConfig.protected(...))`
with the persistent store and sink. It owns startup and periodic handoff recovery.
Run the consumer independently against the same authorized store. The example's
`server.create_demo_server` factory demonstrates this setup and closes its store
on shutdown. Protected pending-action/resolution routes use the configured auth;
the open liveness route is not an inbox or notification channel.

Recovery stops automatically at dead-letter exhaustion. Inspect bounded delivery
rows with `SessionStore.list_persisted_event_side_effect_deliveries(...)` and fix
the destination before operator reconciliation. On Runtime versions with the
operational-health surface, use `get_persisted_event_side_effect_health()` and the
protected `/api/event-side-effects/health` endpoint for aggregate backlog/retry
pressure; this integration does not require it. Preserve dead-letter evidence and
reconcile external acceptance before any manual repair. Do not delete the session
or rerun providers/tools to clear an alert.

## Runnable proof

From a checkout with development dependencies installed:

```sh
attention_state=$(mktemp -d)
uv run python -m examples.human_attention.app pause "$attention_state"
uv run python -m examples.human_attention.app consume "$attention_state"
uv run python -m examples.human_attention.app inspect "$attention_state"
uv run python -m examples.human_attention.app answer "$attention_state"
uv run python -m examples.human_attention.app consume "$attention_state"
uv run python -m examples.human_attention.app late-hint "$attention_state"
```

Every command is a fresh process. There is one active notification before the
answer and one resolved notification afterward; the late opening hint leaves it
resolved. Use a fresh directory with `pause --kind tool_approval`, then `approve`
or `deny` instead of `answer`, to exercise approval handling.

Fault injection is intentional and limited to this fixture:

- `pause --crash-before-delivery` exits 17 after the pause boundary commits but
  before that sink delivery. A fresh `consume` discovers the pending action even
  while the crashed producer's delivery lease remains live.
- `pause --fail-after-accept` commits the destination hint and loses its
  acknowledgement. The original action remains available.
- `consume --crash-after-accept` exits 18 after the notification insert. Repeating
  `consume` or running concurrent consumers leaves one logical notification.
- `interrupt` supersedes a question through Runtime; the next `consume` records
  `superseded`, not `resolved`.

Focused contract tests and the actual fresh-process proof are separate:

```sh
uv run pytest tests/core/test_human_attention.py tests/core/test_foreground_child_pending_actions.py
uv run pytest tests/examples/test_human_attention_example.py
```
