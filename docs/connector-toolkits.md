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
Each is an `ApprovalFloorToolPolicy`, which `cayu check` reads: a toolkit's write tools are
`approval_required` wherever `inner` allows them, so `--deploy` accepts them as guarded.

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

### Pull request facts

`read_pull(gh, repository, number)` answers one pull request as a `PullRequestFacts` record,
through REST `GET`s and one GraphQL query, so a read-only client reads it: state (`open`,
`closed`, `merged`), draft, labels, author, base, head branch and commit and when it was
committed, its people and times, `behind_by` and `ahead_by` (GraphQL's compare of the base with
`refs/pull/N/head`, which carries no files and works for a fork; left out with a gap if that
ref has moved off the record's head), `mergeable` and GitHub's
`mergeable_state`, `checks`, `reviews` (every one, oldest first) and `comments`
(`comments=False` skips them). `latest_reviews` is each reviewer's standing review as GitHub
decides it: their latest approval, request for changes or dismissal; a later comment-only
review leaves it standing. `list_pulls(gh, repository, states=..., updated_since=...)` answers
many through GraphQL, 20 a page, as the same record without `behind_by`, `ahead_by`, comments,
review bodies, check summaries, workflow names or run attempts; `PullRequestList.complete` is
false when `max_pages` (10) stopped it.

- **Mergeability.** GitHub works `mergeable` out after the first read. An open pull request
  whose `mergeable` is unknown is read again once after `mergeable_wait_seconds` (2); a listing
  reads its unknown ones again together, 50 a query, and takes each answer only for the head
  its record describes: one whose head moved meanwhile stays unknown, with a `mergeable` gap.
  Still unknown, it is `None`. A `dirty` merge state is `False`.
- **Checks.** `PullChecks.ci` holds the head's check runs and commit statuses, each a
  `PullCheck` with its `conclusion` and `workflow_run` (a `CheckWorkflowRun`: its id and, from
  `read_pull`, the `attempt` the check ran in; None for a job a rerun of the failed jobs kept
  from an earlier attempt). `reviewer_app="cayu-reviewer"` puts the check runs that App made in
  `PullChecks.reviewer`, whatever they are named: a workflow job with the same name stays CI.
  `PullChecks.state`, over CI only, is `failed` when a finished check concluded other than
  `success`, `neutral` or `skipped`, even if some weren't read; else `unknown` when some weren't
  read; else `none` with no checks (green or a wait is the caller's rule); else `running` while
  one hasn't finished; else `green`. `action_required` (waiting on a person) and `stale` (a
  check GitHub gave up on) are failures too: read `conclusion` before rerunning one. `failed`,
  `running` and `failed_names` list them.
- **Partial answers.** A fact this credential can't read (403) or GitHub can't find (404),
  among the compare, the head commit, check runs, workflow runs and statuses, is left out with
  a `FactGap(fact, reason)` in `gaps`, and the rest stands; so is one read in part, and any
  fact, reviews and comments included, whose answer is over the slot's 5 MiB. In a listing, a
  check the credential can't see (GraphQL's `FORBIDDEN` on the check or the rollup) makes only
  that pull request's checks incomplete; one on a check's workflow run (no `actions: read`)
  leaves the check read and adds a `workflow_runs` gap. Only `NOT_FOUND` and `FORBIDDEN` are
  taken as a missing fact: rate limits, outages and any other GraphQL error still fail.
- **Logins.** Every login is as REST writes it: GraphQL's bot logins get their `[bot]`, and a
  deleted account is `ghost`.

### Write once

`comment_once(gh, repository, number, body, marker, login=...)` comments on an issue or pull
request once however often a task restarts, and `file_issue_once(gh, repository, title, body,
marker, login=...)` files an issue once. Each looks first for an earlier write with the same
`Marker(key, data)` and answers a `WrittenOnce` whose `outcome` is `created`, `found`,
`updated` or `refused`:

```python
app = github.app_connections(app_id, private_key, permissions={"issues": "write"})
gh = github.GitHubClient(app)
marker = github.Marker(f"digests/{digest_id}")
written = await github.comment_once(gh, repo, 42, text, marker, login=await app.login())
```

- **Format.** The marker is the body's last line, a hidden HTML comment:
  `<!-- cayu-marker v1 KEY {"sha":"..."} -->`. A key is letters, digits and `. _ : / = + -`,
  never `--`; data is a JSON object whose strings spell `-`, `<`, `>`, `!` and `&` as `\u00XX`,
  so no value can end the comment early. `with_marker(body, marker)` appends it and keeps the
  whole within `MAX_BODY_BYTES` (262,000 bytes of UTF-8, under the 262,144 GitHub takes,
  whatever its refusal says about 65,536 characters), cutting the body between characters,
  never the marker.
- **Trust.** Only writes whose author is `login` are read, and only their trailing marker:
  `marker_of(body)` reads it, and `trusted_markers(comments, login=...)` reads a pull request's
  `PullComment`s. Marker text anyone else wrote, or quoted earlier in a trusted body, counts
  for nothing. Take `login` from the connection, never by hand: off Cayu Cloud an App's is
  `GitHubAppConnections.login()`, `<slug>[bot]`, which no user can be named, while a user can
  be named the bare slug, so an App connection refuses a login without `[bot]`. With a token,
  `login` is a person's, and their own hand-written comments are trusted too. Trust is the
  author's: a maintainer who edits the App's comment can change or remove its marker, and
  what they leave is read as written. `legacy=` reads an application's earlier format while
  it moves to this one.
- **Lookups.** Comments come from the thread's listing, and issues from the repository's
  issues listing filtered to those `login` created, open and closed, the author checked again
  on each. As the listing may not show a new write yet, a call also reads directly the write
  named by `known=` (the `id` of a comment, or `number` of an issue, an earlier call
  answered) and the one this client last made for the marker; each counts when it is still
  there (not deleted or moved to another repository), `login`'s, with the key. Of all the
  matches, the earliest is taken, whatever its labels, and the rest are `duplicates`. `labels` only tag a new
  issue; `labelled` says whether the issue found or filed has them all (GitHub drops labels
  the writer can't set). `since=` reads only comments updated since then, so use it only when
  the marked comment can't be older, as for a key naming its own day. A listing that reaches
  `max_pages` (50) without finding the marker raises `GitHubLookupIncomplete` and writes
  nothing. A found comment is left, or with `update=True` edited in place when its body
  differs.
- **Listings lag.** GitHub's listings show a just-written item seconds late: an issue has
  taken 3 seconds to appear in its creator's listing and 15 in the whole repository's. A
  repeat through the same client finds its write directly, and `known=` covers a new process
  that kept the answer, so an application that may run the same write again within seconds
  of a restart stores the `id` or `number` it was answered and passes it. A new process
  without it, within about 15 seconds of another's write, can write again; the duplicate is
  then reported in `duplicates` by every later call.
- **Races and retries.** One client's writes of the same marker take turns. Writers in other
  processes can both write; every later call then answers the earliest, and the others in
  `duplicates` (comments' ids, issues' numbers), which are reported and never deleted. A write
  whose answer was lost is never resent: the listing is read again after 1, 2, 4, 8 and 5
  seconds, and not found by then, its `ambiguous` error is raised. A read-only client looks
  up, and its refused write answers `refused`. Inline review comments and reviews aren't
  covered.
- **Model text.** `plain_text(text, limit=...)`, for any text a model wrote into a body, puts
  a zero-width space after each `@`, so it mentions no one (a mention can start a bot's
  command) while an address still reads as written, and removes the HTML comment delimiters
  `<!--` and `-->`, so a comment's text shows instead of hiding. Other HTML is left as it is.

### Testing with a fake GitHub

`cayu.connectors.github.testing.FakeGitHub` is a connection client that answers from data, so an
application's tests run the real `GitHubClient`, with its paging, errors, rate limits and
read-only mode, without GitHub or the network:

```python
from cayu.connectors.github import GitHubClient
from cayu.connectors.github.testing import (
    Call, CheckRun, Comment, CommitStatus, FakeGitHub, Issue, PullRequest, Release, Repository,
    Review, WorkflowRun, reply,
)

fake = FakeGitHub(login="my-agent[bot]")
pull = fake.add_pull("acme/widgets", 7, labels=["ready"], mergeable_unknown_reads=1)
ci = fake.add_workflow_run("acme/widgets", pull.head_sha, conclusion="failure")
fake.add_check_run("acme/widgets", pull.head_sha, "tests", run=ci, conclusion="failure")
gh = GitHubClient(fake)  # or fake.client()
await gh.post(f"/repos/acme/widgets/actions/runs/{ci.id}/rerun-failed-jobs")
assert ci.run_attempt == 2
assert not fake.unanswered
```

- **Data.** `add_repository`, `add_pull`, `add_issue`, `add_comment`, `add_review`,
  `add_workflow_run`, `add_check_run`, `add_status`, `add_release` and `set_log` answer plain
  objects (`Repository`, `PullRequest`, `Issue`, ...) a test may change, such as a pull's `head_sha`
  after a push. Pulls and issues share their numbers; owner and name match ignoring case. A pull
  answers `mergeable: null` on its first `mergeable_unknown_reads` reads, and
  `mergeable_state: "behind"` only with `requires_up_to_date=True`, as under branch protection.
  Logins ending in `[bot]` are bots.
- **Routes.** Pulls (list, read, merge, update-branch, reviews), compare, a commit, a commit's
  check runs and combined status, workflow runs and their reruns, a job's log (a 302 to a download the fake serves
  through `download_transport`, which `GitHubClient` takes from it), issues (list, create, read,
  edit), comments (list with `since`, create, read, edit), labels, releases and a
  collaborator's permission. Listings page with GitHub's next-page links, `page_size` items a
  page at most. `listing_lag` leaves a new issue or comment out of listings until it is that
  old by `clock`, as GitHub's listings show one seconds late; a direct read finds it at once.
- **Nothing guessed.** A route, query filter, write body field or GraphQL operation the fake
  doesn't answer raises `NotImplementedError` and is kept in `unanswered`: assert it is empty, as
  an application may catch the error. `route(method, pattern, handler)` adds a repository route,
  answered with `reply(status, body)` from the `Call`.
- **Writes take effect** as GitHub's do, by `login`: a comment or issue appears with its id,
  `html_url` and time; labels change; an issue closes; a merge closes the pull; an
  update-branch is accepted with 202, head unchanged; a rerun bumps `run_attempt` and queues new
  check runs for the jobs it reruns (those it replaced are listed only with `filter=all`), which
  `finish_run` completes. Refusals answer GitHub's status, text and order, as checked against
  GitHub, with its `documentation_url` and `status` in the body: a merge of a draft, a closed
  pull or one with conflicts (405), and one behind under `requires_up_to_date` (405; its text
  is unverified), then at a stale `sha` (409); an update-branch at a stale
  `expected_head_sha`, then of a closed or conflicting pull, then of one that isn't behind (422);
  a rerun of a run still going, or of failed jobs when none failed (403). A collaborator's
  permission is set on the repository; anyone else's is `read` on a public repository and
  `none` on a private one, and a login that isn't a user (in `users`, a collaborator or an
  author) is 404. A closed pull's `mergeable` reads null.
- **Failures.** `fail(method, path_glob, status=, body=, headers=, times=)`,
  `rate_limit(..., retry_after=, secondary=)` (reset in wall-clock time, as GitHub's) and
  `refuse(code, ...)` (the slot's host: `upstream_failed` for no answer, or a refusal such as
  `scope_not_allowed`) answer the next matching calls; the first fault set up that matches
  answers.
- **Recording.** `requests` holds every `Call`, with its query, headers (the repository header
  included) and body; `writes()` the ones that would change GitHub. A call whose repository
  header doesn't name its repository is refused, as Cayu Cloud's connector is to refuse it.
- **GraphQL.** There is no GraphQL engine: `on_graphql(name, answer)` answers the operation
  named `name` (the document's first definition, so not one that starts with a `fragment`) with
  `answer(variables, repository)`, the whole answer with its `data` and `errors`. `read_pull` reads
  the rest from the fake's routes once a test answers its `PullRequestCompare` query, and
  `list_pulls` reads its `PullRequestFacts` query's answer.

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
