# Connector toolkits: Slack, Snowflake, SAP and GitHub

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
- **Suppliers' banks.** A supplier's number and its business partner's often differ, so
  `include_bank` finds each supplier's partner in `A_BusinessPartner` by its `Supplier` field and
  reads the partner's expanded `to_BusinessPartnerBank`. A supplier whose partner isn't found, or
  whose bank accounts SAP pages, gets `banks: null` and the reason `banks_incomplete`, never an
  empty list. In SAP's public sandbox each supplier's partner has the same number, so the
  recorded answers don't show a mapping. The toolkit's field list is a convenience, not a
  boundary: `sap_read` on `API_BUSINESS_PARTNER` reads any bank field the model asks for. The
  binding's service scopes, and the SAP role behind the connection, are what limit it.
- **Small answers.** An answer carries the statement handle, the columns (name and type), up to
  `max_rows` rows (default 200, at most 1,000), reading further result partitions only as
  needed, the total `row_count`, `returned_rows`, `truncated` and `truncation_reasons`
  (`max_rows`, `cells_shortened`, `answer_size`, `partition_unreadable`, `partition_too_large`,
  `deadline`, `rows_missing`). A cell is shortened to 4,000 characters, and rows are dropped from the end until
  the answer fits in about 60 KiB. Put a `LIMIT` on a query: the first result partition comes
  back whole, and one larger than the broker's 5 MiB fails the call.
- `warehouse`, `database`, `schema` and `role` passed to `tools()` set every statement's
  context; a call may name its own `database` and `schema`.

Each answer is compact JSON for the model and the same object in `ToolResult.structured`. The
answer the model reads is at most about 60 KiB; each tool declares a terminal payload of 160
KiB (`max_terminal_payload_bytes`), which holds that answer and its structured copy.

## SAP reads

`sap.tools(client=None, slot="sap")` reads SAP S/4HANA (Cloud Public Edition, Private Edition or
on-premise) over its released OData V2 APIs. It has no write tools, so there is nothing to
approve and no `approval_policy()` or `review_fields()`. The broker allows only `GET` under
`/sap/opu/odata/sap/<SERVICE>/` and sets the SAP client itself; each scope of the binding is one
service the Agent may read (`API_BUSINESS_PARTNER.read`), and a service it doesn't hold answers
`scope_not_allowed`.

```python
from cayu.connectors import sap

app.register_agent(spec, tools=sap.tools(slot="sap"))
```

| Tool | Reads |
| --- | --- |
| `sap_read` | Any entity set of a granted service: `service`, `entity_set`, `select` (required), and an OData `filter`, `orderby` and `expand` |
| `sap_supplier_invoices` | `A_SupplierInvoice` for 1 to 20 `company_codes` posted from `posting_date_from` to `posting_date_to` (both included, at most 93 days), most recent first, each with its purchase-order references |
| `sap_purchase_order_items` | `A_PurchaseOrderItem` of the given `purchase_orders` |
| `sap_goods_receipts` | `A_MaterialDocumentItem` with movement type 101 or 102 of the given `purchase_orders` |
| `sap_suppliers` | `A_Supplier` by `suppliers` ID; with `include_bank: true` also each one's `BusinessPartner` and its bank accounts: the bank, its country and validity, never the account number, IBAN, bank key or account holder |

- **Filters from typed arguments.** The named reads write `$filter` themselves: string values
  are quoted with their quotes doubled, dates are `datetime'YYYY-MM-DDT00:00:00'`, and a list
  of up to 100 values is compared in `or` chains of 20, one request each, read in turn as one
  list. The model writes OData only for `sap_read`.
- **Pages.** Each tool answers up to `top` rows (default 50, at most 500) from `skip` on, with
  `count` (SAP's `__count`; null when a named read stopped before its last chain of values),
  `returned` and `next_skip`, which is null on the last page. `skip` counts the rows SAP
  counts, chain after chain, and following `next_skip` answers each row once. SAP can count
  rows that the connection's authorisations then hide. A chain that answers none of the rows it
  counts ends a page that already holds rows; with none yet, the read moves on past those
  positions with the truncation reason `count_mismatch`, so `next_skip` never repeats `skip`
  and no row comes twice.
- **Plain rows.** SAP's `d.results` are unwrapped, `__metadata` and links of navigation
  properties that weren't expanded are dropped, and an expanded list is a plain list. When SAP
  pages an expanded list itself, only its first part arrives: the row names the property in
  `incomplete_expansions` and the answer has the truncation reason `expansion_incomplete`.
  Read the rest with `sap_read` on the navigation's own entity set.
  `/Date(ms)/` becomes `YYYY-MM-DD` (with its time when it has one) and `/Date(ms+mmmm)/` an ISO
  timestamp in its offset. Decimals stay the strings SAP sends.
- **Small answers.** Rows are dropped from the end until the answer fits in about 60 KiB, with
  `truncated: true`, `truncation_reasons` (`answer_size`; or `banks_incomplete`, `count_mismatch`,
  `expansion_incomplete` as above) and `next_skip` at the first row left out. An answer
  over the broker's 5 MiB fails with `upstream_response_too_large`: ask for a smaller `top`, or
  fewer fields or expansions.

## GitHub client

`cayu.connectors.github` is a client for code, such as workflows and task handlers, rather than
a set of tools. `GitHubClient(client=None, slot="github")` calls GitHub's REST and GraphQL APIs
through the slot; the same code runs on Cayu Cloud, through the broker, and on a laptop:

```python
from cayu.connectors import github

gh = github.GitHubClient(github.connections_from_environment(permissions={"issues": "write"}))
pulls = await gh.list(f"/repos/{repo}/pulls", query={"state": "open"})
await gh.post(f"/repos/{repo}/issues/{pulls.items[0]['number']}/comments", json={"body": text})
```

`connections_from_environment(permissions=...)` answers the broker on Cayu Cloud. Off it, it
answers `GITHUB_TOKEN` as `token_connections(token)`, or a GitHub App from `GITHUB_APP_ID` and
`GITHUB_APP_PRIVATE_KEY` as `app_connections(app_id, private_key, permissions=...)`; both set is
an error, as they sign in different names. The App signs an RS256 token with its key, and mints
an installation token for each call's one repository with exactly `permissions`, as a binding
fixes a slot's scopes on Cayu Cloud: no call asks for more. It keeps each repository's
installation and token, for at most 256 repositories, mints once however many calls miss at
once, replaces a token 5 minutes before it expires, and mints again once when GitHub answers
401. It calls only a repository's own API (`/repos/{owner}/{name}/...`), `/graphql` and
`/rate_limit` (with a repository, whose token it uses), the calls Cayu Cloud's GitHub connector
is to allow. `GitHubClient` doesn't name a repository for `/rate_limit` yet, so it can't read
that path through an App slot; a token slot answers it. `login()` is the App's `<slug>[bot]`, the
author of what it writes.

- **Calls.** `request(method, path)` answers GitHub's response; `get`, `post`, `put`, `patch`
  and `delete` its JSON; `text(path)` a raw file or a job's log. Paths start with `/` and are
  percent-encoded where needed (a branch keeps its `/`; a `%` is literal unless it starts an
  escape such as `%2F`). A call under `/repos/{owner}/{name}/`, or `graphql(...,
  repository=)`, names its repository in the `x-cayu-github-repository` header. The slot's
  host uses it to pick the credential and drops it, so GitHub never sees it:
  `token_connections` and `app_connections` do, and Cayu Cloud's GitHub connector has to.
- **Pages.** `list(path)` follows GitHub's next-page links, 100 items a page, and answers
  `Page(items, complete)`: `complete` is false when `max_pages` (default 50) or
  `keep_going(page_items)` stopped it early. `paginate(path)` yields the items one by one.
  `item_key` reads listings wrapped in an object (`check_runs`). A link is followed only to the
  same listing on the slot's API (GitHub's links may name its owner, repository or team by
  id), keeping the path asked for, and never to a page already read.
- **Errors.** Each failure is a `GitHubError` with `method`, `path`, `status`, GitHub's
  `message` and `errors[]` text, `codes` and `request_id`: `GitHubNotFound`,
  `GitHubPermissionDenied` (with `accepted_permissions`), `GitHubConflict` (409, 422),
  `GitHubRateLimited` (with `retry_after`, `secondary`), `GitHubUnavailable`,
  `GitHubConnectionRefused` (the broker's `code`), `GitHubAppNotInstalled`,
  `GitHubWriteRefused` and `GitHubGraphQLFailed`.
- **Rate limits and retries.** A rate limit (a 429, or a 403 with no requests left, a
  `Retry-After` or a secondary limit) and the broker's `too_many_calls` are waited out and the
  call sent once more when the wait is at most `max_rate_limit_wait_seconds` (60), as is a
  GraphQL answer whose errors are `RATE_LIMITED`; GitHub
  didn't act on a limited call, so a write is resent too. A 5xx or a lost answer is retried
  (`read_retries`, 2) only for reads and GraphQL queries; a write raises with `ambiguous` set.
  One client has at most 6 calls in flight.
- **GraphQL.** `graphql(query, variables, repository=...)` fails on any error in the answer,
  with GitHub's `data` and `errors`, unless `accept` takes it: accepted errors come back in
  `GraphQLAnswer.accepted` beside the `data`. `forbidden_under("statusCheckRollup")` reads a
  pull whose one check run the credential can't see; `errors_under(*path, types=...)` accepts
  by path. A null `repository` raises `GitHubNotFound`, unless `accept` took its
  `NOT_FOUND`. GitHub Enterprise Server's GraphQL API (`/api/graphql`) isn't reachable through
  a slot rooted at `/api/v3` yet.
- **Read-only mode.** `GitHubClient(..., read_only=True)` or `read_only_view()` refuses every
  method but `GET` and `HEAD`, and any GraphQL document with a `mutation`, before it leaves,
  and lists each in `refused` for a dry run's report. For a hard guarantee, bind a second slot
  to read scopes.
- **Redirects.** A redirect fails the call (a moved repository says so). `text()` follows a
  job log's redirect to GitHub's log storage (`*.actions.githubusercontent.com` and
  `productionresultssa<N>.blob.core.windows.net`) directly, with no credential, and
  keeps the log's last 5 MiB. On Cayu Cloud that download needs the Agent to reach those hosts,
  as the broker doesn't follow redirects.

## Errors

A call that fails answers a tool error (`is_error`) whose `structured` holds `error` (a code),
`message`, and `advice` where there is something to do:

| Code | Advice |
| --- | --- |
| `connection_unbound`, `connection_not_ready`, `connection_unavailable`, `credential_invalid` | An admin must bind the slot or reconnect its connection |
| `scope_not_allowed`, `path_not_allowed`, `method_not_allowed`, `connector_not_brokered` | Not allowed for this Agent; don't retry |
| `too_many_calls` | Retry after `retry_after_seconds` (the broker's or the connector's `Retry-After`) |
| `upstream_failed`, `connection_failed`, `upstream_status` (5xx), `upstream_invalid_response` | A temporary failure; retry later (a running statement is read again on its own) |
| `upstream_response_too_large` | Ask for less (with the statement handle when the statement finished) |
| `result_too_wide` | The columns alone don't fit: select fewer, or give them shorter names |
| `slack_error` | Slack's own `{"ok": false}` answer, its error in `slack_error` |
| `sql_error` | Snowflake refused the statement: its message, `snowflake_code`, `sql_state` and handle |
| `statement_timed_out` | The statement was cancelled at the tool's time |
| `sap_not_authorized` | SAP answered 401: the connection's credential was rejected, and an admin should check or rotate it; or 403: the communication arrangement or role for this service is missing in SAP |
| `sap_not_found` | SAP answered 404: check the service, entity set and field names |
| `sap_error` | SAP refused the read (4xx): its message, `sap_code` and `transaction_id` from a JSON error, or the `status` alone from an HTML or XML page |
| `query_not_allowed` | The broker refused a query option SAP reads don't take |
| `row_too_large` | One SAP row alone doesn't fit: select fewer fields or expand less |
| `invalid_arguments`, `not_a_read`, `deadline_too_close` | Fix the call |

An error in the tool's own code still raises, as in any tool.
