# Connector toolkits: Slack and Snowflake

`cayu.connectors` has ready tools for an application's connections. Each toolkit calls its
connector through a connection slot ([Connections in the Runtime](adr/0003-connections-runtime-contract.md)),
so the Agent never holds the credential: on Cayu Cloud every call goes through the connection
broker, which attaches the Organization's credential and applies its rules.

```python
from cayu.approvals import ToolArgumentReview
from cayu.connectors import slack, snowflake

app.register_agent(
    spec,
    tools=[
        *slack.tools(slot="slack", include_writes=True),
        *snowflake.tools(slot="warehouse", write_slot="warehouse-write", include_writes=True),
    ],
    tool_policy=slack.approval_policy(snowflake.approval_policy()),
)
review = ToolArgumentReview({**slack.review_fields(), **snowflake.review_fields()})
```

`tools()` answers only the read tools unless it is given `include_writes=True`; give the write
tools with the toolkits' `approval_policy()`, as above, so a person approves each write.
`snowflake_query` uses `slot` and `snowflake_execute` its own `write_slot` (default
`snowflake-write`): bind `slot` to a connection whose Snowflake role can only read, and
`write_slot` to one whose role can write. A team may name the same slot for both. Slack keeps one
slot: the broker already checks each method against the binding's scopes.

## The client

`tools(client=None, slot=...)` calls through `client`, a `cayu.connections.ConnectionClient`.
Without one it uses `CloudConnectionBroker.from_environment()`, the broker Cayu Cloud binds the
Agent to. Outside Cayu Cloud there is none, and each tool answers the error
`no_connection_client`. For a laptop or a test, give each slot its own host and credential:

```python
import os

from cayu.connections import LocalConnections, LocalSlot
from cayu.connectors import slack

local = LocalConnections(
    {"slack": LocalSlot("https://slack.com", {"authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"})}
)
read_tools = slack.tools(local)
```

## Tools

| Tool | Does | Effect | Needs approval |
| --- | --- | --- | --- |
| `slack_post_message` | Post to a public channel (its ID or `#name`), or reply in a thread (`thread_ts`) | external | Yes |
| `slack_update_message` | Replace the text of a message the app posted (channel ID) | idempotent | Yes |
| `slack_list_channels` | Public channels, paged | none | No |
| `slack_read_channel` | A channel's messages (channel ID), newest first, paged; `oldest` and `latest` bound them | none | No |
| `slack_read_thread` | A thread's parent and replies (channel ID), paged | none | No |
| `slack_lookup_user` | A user by `user_id` or by `email` | none | No |
| `snowflake_query` | One read statement (`SELECT`, `SHOW`, `DESCRIBE`, or `WITH ... SELECT`), with its rows | none | No |
| `snowflake_execute` | One statement of any kind | external | Yes |

The write tools (`slack_post_message`, `slack_update_message`, `snowflake_execute`) come only
with `include_writes=True`.

Slack's posts and edits are JSON bodies (`application/json; charset=utf-8`); its reads are query
parameters. A paged read answers up to 50 items by default (`limit`, at most 200) and the next
page's `next_cursor`, which is null on the last page. A message's text is shortened to 4,000
characters (`text_shortened: true`), and further when the page wouldn't fit otherwise. A page
of channels drops their topics and purposes when it wouldn't fit, and if it still doesn't, asks
Slack again for fewer channels from the same cursor, so no channel is skipped. The Slack
connection holds only public-channel scopes, so private channels and direct messages aren't
reachable. Looking a user up by email needs the binding to hold `users:read.email`.

## Approval

A tool can't ask for approval itself, so each toolkit has `approval_policy(inner)`. It asks for
approval before its write tools run, with a reason naming the channel (`Post a Slack message to
C0123.`) or the kind of statement (`Run a DELETE statement in Snowflake.`), and leaves every other
decision to `inner` (default: allow). A call `inner` denies stays denied, and one `inner` already
sends for approval keeps `inner`'s result. Wrap one toolkit's policy in the other's to use both.

`review_fields()` names the arguments an approver sees, for
[`ToolArgumentReview`](human-review.md): the channel, thread and text of a Slack post or edit, and
the SQL with its database and schema for `snowflake_execute`.

## Snowflake statements

- **One statement per call.** A statement with anything but comments after its first `;` is
  refused before it is sent, and each submission sets `MULTI_STATEMENT_COUNT=1`.
- **Reads and writes.** `snowflake_query` reads the statement's words outside comments,
  strings and `$$` bodies. It runs a statement that starts with `SELECT`, `SHOW`, `DESCRIBE` or
  `DESC`, or with `WITH` when its common table expressions are followed by `SELECT` and it
  defines no procedure or function. It refuses any statement that names a `SYSTEM$` function
  (several change state), quoted or not, uses `IDENTIFIER(...)`, or calls in `TABLE(...)`
  anything but `RESULT_SCAN`, `FLATTEN`, `GENERATOR`, `SPLIT_TO_TABLE`, `INFER_SCHEMA` or an
  `INFORMATION_SCHEMA` function, since a procedure there can write, or chains statements with
  the flow operator `->>` (`SHOW TABLES ->> DROP TABLE t` is one statement). A `WITH` that ends in
  `INSERT`, `MERGE`, `UPDATE` or `DELETE`, an anonymous procedure, and anything else is refused
  with the error `not_a_read`, which points at `snowflake_execute`.
- **The role is the real limit.** What keeps `snowflake_query` to reads is the read-only
  Snowflake role of the connection bound to its `slot`, not the check: the check reads the
  statement's text, guards against accidents and sends writes to `snowflake_execute` and its
  approval, but can't see what a read does when it runs. A read can still have effects: a
  sequence's `NEXTVAL`, a user-defined or external function with network access, or anything
  that spends credits. Where `slot` and `write_slot` name the same slot, its role is what either
  tool can do.
- **Cancelled with the call.** When the tool call itself is cancelled (a tool timeout, or a
  person stopping the run) while the statement runs, the tool sends Snowflake its cancel, waiting
  at most 5 seconds, before the call ends.
- **Run to the end within the tool's time.** The statement is submitted with `async=true`,
  `retry=true` and a request ID derived from the call's idempotency key, so a retried call
  can collect the first submission's result instead of running the statement again (a call
  without an idempotency key gets a random request ID). `snowflake_execute` is still declared an
  external effect: the request ID is Snowflake's to honor, not a guarantee. The tool reads the
  statement's handle until it finishes, for at most `timeout_seconds` (default 120) and never
  past the run's deadline less 10 seconds. A rate-limited read waits its `Retry-After` and a
  temporary failure is read again, both within that time. A statement still running then is
  cancelled and the call answers `statement_timed_out`; a final failure while reading also
  cancels it. Either answer names the `statement_handle` and whether the cancel was confirmed.
  Snowflake's own statement timeout is set just past the tool's, in case the cancel never
  arrives.
- **Small answers.** An answer carries the statement handle, the columns (name and type), up to
  `max_rows` rows (default 200, at most 1,000), reading further result partitions only as
  needed, the total `row_count`, `returned_rows`, `truncated` and `truncation_reasons`
  (`max_rows`, `cells_shortened`, `answer_size`, `partition_unreadable`, `partition_too_large`,
  `deadline`, `rows_missing`). A cell is shortened to 4,000 characters, and rows are dropped from the end until
  the answer fits in about 60 KiB. Put a `LIMIT` on a query: the first result partition comes
  back whole, and one larger than the broker's 5 MiB fails the call.
- `warehouse`, `database`, `schema` and `role` passed to `tools()` set every statement's
  context; a call may name its own `database` and `schema`.

Each answer is compact JSON for the model and the same object in `ToolResult.structured`.

## Errors

A call that fails answers a tool error (`is_error`) whose `structured` holds `error` (a code),
`message`, and `advice` where there is something to do:

| Code | Advice |
| --- | --- |
| `connection_unbound`, `connection_not_ready`, `connection_unavailable`, `credential_invalid` | An admin must bind the slot or reconnect its connection |
| `scope_not_allowed`, `path_not_allowed`, `method_not_allowed`, `connector_not_brokered` | Not allowed for this Agent; don't retry |
| `too_many_calls` | Retry after `retry_after_seconds` (the broker's or the connector's `Retry-After`) |
| `upstream_failed`, `connection_failed`, `upstream_status` (5xx) | A temporary failure; retry later (a running statement is read again on its own) |
| `upstream_response_too_large` | Ask for less (with the statement handle when the statement finished) |
| `result_too_wide` | The columns alone don't fit: select fewer, or give them shorter names |
| `slack_error` | Slack's own `{"ok": false}` answer, its error in `slack_error` |
| `sql_error` | Snowflake refused the statement: its message, `snowflake_code`, `sql_state` and handle |
| `statement_timed_out` | The statement was cancelled at the tool's time |
| `invalid_arguments`, `not_a_read`, `deadline_too_close` | Fix the call |

An error in the tool's own code still raises, as in any tool.
