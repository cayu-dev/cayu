# Session inspection

`cayu session` is the supported interface for diagnosing durable sessions and
submitting one explicit provider-operation disposition. Inspection uses
`SessionStore` contracts, so SQLite and PostgreSQL expose the same fields
without requiring operators to know Cayu table names or JSON paths. The
resolution command talks to the authenticated Cayu server; it never mutates the
store directly.


## Event-window usage and start counts

Use `cayu usage` to read the Developer dashboard's aggregate data from a running
local server or a Cloud Agent. This is separate from `cayu session rollup`,
which selects sessions by creation time and summarizes their retained activity.
The usage rollup selects events by time, including recent work in older sessions.

```bash
cayu usage rollup --server-url http://localhost:8000 \
  --since 2026-10-01T00:00:00Z --until 2026-10-08T00:00:00Z \
  --bucket day --timezone Asia/Bishkek --agent writer --json

cayu usage counts --application accounting-agent \
  --since 2026-10-01T00:00:00Z --until 2026-10-08T00:00:00Z \
  --bucket day --agent writer --table
```

Both commands accept `--agent`, `--environment`, `--provider` and `--model` as
exact current-session filters, plus `--parent-session ID`, `--causal-budget ID`,
repeated `--label KEY=VALUE` and repeated `--label-selector EXPR` (for example
`'team in (a,b)'` or `'!team'`). They select the same sessions in both commands,
so `counts --bucket` matches the dashboard chart for any headline filter. `counts`
with the last four needs a server that advertises
`session_overview_count_filters`. `--server-url` supports mounted APIs; authenticated
servers read the complete Authorization header from `CAYU_API_AUTHORIZATION`
or the variable named by `--authorization-env`. Cloud uses the existing login
and optional `--context` file. These commands do not open database files directly.

`rollup` optionally accepts `--price-book FILE` for estimated headline, provider,
model and session costs, and `--top-sessions N` for session detail ranked by tokens
by default. Add `--session-order cost` to rank by estimated cost; incomplete or
incomparable estimates fall back to tokens with an explicit reason.
Without a price book, costs are not requested. Counter buckets never claim to
be cost buckets; partial or unevaluated headline pricing remains explicit in
JSON. `counts` groups sessions by start time and reports their current failures,
excluding evaluation trials and forks unless `--include-eval` or
`--include-forks` is supplied. Older running sessions remain in `running_now`.
Activity-session counts cannot be summed across usage buckets. Explicit windows
and the optional hour/day/week series follow the
[control-plane contract](control-plane.md#usage-windows-and-series).

The Usage dashboard and rollup API also support parent-session, causal-budget,
exact-label and label-selector filters. These advanced filters do not yet have
equivalent options in `cayu usage rollup`; the CLI currently exposes the four
current-session filters listed above.

## Developer dashboard list

Open `/cayu/sessions` to browse canonical session state and agent-authored titles and
summaries. Missing titles fall back to labels, then the first user message, then the
session ID. Completed summaries use the warning colour only when their stored tone is
`attention`. Tool calls and token/cost figures come from the overview; partial costs
are marked and retain their currency.

Evaluation trials and forks are excluded initially. Use their separate inclusion
switches to show them with row markers. Waiting for you combines approval, input,
and recovery states. Advanced filters preserve specialist, provider, model,
environment, parent session, causal budget, exact labels and label selectors. Raw
status, debug state, sort and existing deep-link parameters remain available.
The footer reports rows loaded on the current page. Next page follows opaque server
cursors; an empty candidate page does not prove exhaustion. Members can browse the
list but are not offered Start a session.

## Titles, completion summaries, and specialist descriptions

Set a session title on its typed request. The `app.run(request)` signature is
unchanged; HTTP `POST /api/run` accepts the same optional `title` field.

```python
from cayu import Message, RunRequest

request = RunRequest(
    agent_name="assistant",
    session_id="quarterly-review",
    title="Review quarterly accounts",
    messages=[Message.text("user", "Review the quarterly accounts.")],
)
async for event in app.run(request):
    pass

# Set this from agent code once its outcome is known. It does not end a run.
await app.session_store.set_summary(
    "quarterly-review", "Two entries need review", tone="attention"
)
```

`set_summary` atomically replaces only the summary and tone; concurrent user
annotations are retained. It uses the same run-epoch and closure-owner fences
as other annotation writes. A scoped `access.sessions.set_summary(...)` also
requires current visibility and `modify` permission. The tone defaults to
`neutral`; only `neutral` and `attention` are accepted. Readers never infer tone
from text. Like other annotation writes, it updates `updated_at`, not runtime
`last_activity_at`. An adapter without this optional method raises
`NotImplementedError`; the memory, SQLite and PostgreSQL stores implement it.

The reserved conventions are defined once in `cayu.session_metadata`:

| Key | Owner and meaning |
| --- | --- |
| `cayu:title` | Session title from `RunRequest.title`, at most 256 characters. |
| `cayu:summary` | Agent-authored outcome from `set_summary`, at most 4096 characters. |
| `cayu:summary_tone` | Explicit `neutral` or `attention`, written with the summary. |
| `cayu:description` | One-line specialist description in `AgentSpec.metadata`, at most 1024 characters. |
| `cayu:eval_trial` | Runtime-owned eval-trial marker, inherited by descendants. |
| `cayu:attention_handoff` | Caller-set `False` opts out of human-attention handoff, inherited by descendants. |
| `cayu:run_lineage` | Runtime-owned original run ancestry. |
| `cayu:task_id` | Runtime-owned: the task the session runs under, inherited by descendants. See [Sessions run under a task](#sessions-run-under-a-task). |

Display text must be an exact string, nonblank, without control characters or
surrogates. Validation rejects rather than truncates. Titles pass through the
run's secret redactor before persistence. Direct store annotation APIs retain
the supplied text, as with existing user metadata; supply publication-safe
summaries. Session requests and metadata replacement cannot write title,
summary, or tone directly; use their typed APIs. Existing runtime authority
keys remain protected. Ordinary metadata replacement preserves these fields.

Titles and summaries survive storage reopen and session continuation. Forks
inherit the source title, summary and tone; a fork's agent may replace its own
summary when its outcome changes. Independent child runs receive their own
optional title. Display metadata does not change execution or session status.
The `session_display_metadata` surface capability advertises these conventions;
the existing session reads expose their metadata, and this adds no summary
HTTP mutation route.

## Validated structured output in transcripts

Successful native and tool-based structured answers add a `structured_output`
part to the existing transcript message:

```json
{"type": "structured_output", "name": "Answer", "output": {"answer": "ready"}}
```

`name` is the specification's name, or `null` for an unnamed specification;
`output` is its validated JSON value after the runtime's secret redaction.
The part is stored atomically with its original assistant text or tool result,
so retries and recovery do not append duplicate answers. Failed validation adds
no structured-output part. The original tool calls and results remain in the
transcript and event trace. Transcript pagination and message indexes are unchanged.

`GET /api/sessions/{id}/transcript` and `SessionStore.load_transcript` expose the
same part. Check `capabilities.surfaces.structured_output_transcript.read.enabled`
before rendering it. Custom transcript readers should recognize the new part
instead of identifying the internal submission tool by name. The runtime omits
this display evidence from subsequent model requests because the original text
or tool result already carries the answer. Caller-authored session input cannot
claim runtime validation by supplying a `StructuredOutputPart`.

Visible-text exports omit this display evidence; whole-record exports preserve
it for the authorized projector. Completed context views retain it under the
same exact publication receipt as the original message. A fork from an
authenticated context view imports the original text and tool results without
copying the source runtime's validation claim into caller input. Raw caller
messages containing the display part remain rejected.
The JSON value keeps the existing payload depth limit; typed transcript and
publication envelopes have bounded nesting headroom for their enclosing fields.
Byte and node limits remain unchanged.

## Let agents read earlier sessions

`ListSessionsTool` (`list_sessions`) and `ReadSessionTool` (`read_session`)
give a model read-only access to durable session records. Neither tool is part
of any preset; register them on the agents that need them, each with an access
policy:

```python
from cayu.tools import ListSessionsTool, MatchingSessionAccess, ReadSessionTool

verifier_access = MatchingSessionAccess(
    labels={"app": "scout"},
    exclude_labels={"role": "hunter"},
)
tools = [
    ListSessionsTool(app.session_store, access=verifier_access),
    ReadSessionTool(app.session_store, access=verifier_access),
]
```

The policy is required and is evaluated against the calling session on every
call. `OwnTreeSessionAccess()` admits the caller's root session tree: the root
named by its invocation provenance and every descendant. `MatchingSessionAccess`
admits sessions run by any of `agent_names` and carrying every pair in
`labels`, minus sessions carrying any pair in `exclude_labels`; it needs
`agent_names`, `labels` or `same_labels`, so it never admits the whole store.
Both hide the calling session unless `include_caller=True`.

Tools are registered once per application, so fixed label values can't say
"the same project as the caller". `same_labels` names label keys whose value
must equal the calling session's own value, and `exclude_same_labels` removes
sessions that share the caller's value for any of its keys:

```python
agents = ("scout-hunter", "scout-verifier")
# A hunter reads earlier hunter and verifier sessions on its repository and module.
hunter_access = MatchingSessionAccess(
    agent_names=agents,
    same_labels=("scout_repo", "scout_module"),
)
# A verifier reads the same, but never a session of its own run.
verifier_access = MatchingSessionAccess(
    agent_names=agents,
    same_labels=("scout_repo", "scout_module"),
    exclude_same_labels=("scout_run",),
)
```

A caller that lacks one of those labels may read nothing. As with
`exclude_labels`, a session without the excluded key is not removed, so label
every session the rule should keep out. A key may appear in only one of
`labels` and `same_labels`, and not in both `same_labels` and
`exclude_same_labels`. The rule is evaluated against the caller's labels at the
time of each call. Other rules implement the
`SessionReadAccess` protocol: `narrow(caller, query)` restricts the listing's
`SessionQuery` (or returns `None` to admit nothing) and `allows(caller, session)`
decides one session. The tools check every listed session with `allows`, and a
session it rejects reads exactly like a missing one. A `narrow` broader than
`allows` costs extra store reads but exposes nothing: when rows are filtered out,
`list_sessions` reads up to five store pages to fill the page, and its
`next_cursor` is built from the last session it returned. If those pages hold no
more readable sessions, the result sets `scan_limited` and asks the model to
narrow its filters.

### Tenant isolation

Both tools refuse to run unless the calling run has a resource-access scope
(an execution binding from a `ResourceAccessPolicy`). Without one they return a
`resource_scope_required` tool error that names the fix. An application with a
single-tenant session store opts out explicitly with
`ListSessionsTool(..., unscoped=True)` and `ReadSessionTool(..., unscoped=True)`;
use the option only for single-tenant stores.

With a scope active, every read applies it: listings, session loads, the
transcript window and the structured-output search, and the tools check each
returned session against it again. A session outside the scope reads exactly
like a missing one, and not-found results do not repeat the requested id.
Tenancy comes from the scope, never from the model: the `labels` filter rejects
keys the scope protects, and a cursor or session alias the tools did not issue
to this caller is rejected. Labels in a `MatchingSessionAccess` (trusted application
configuration) are intersected with the scope, so naming another tenant there
admits nothing. `same_labels` reads the caller's own labels, which the scope
already bounds, so it can't reach another tenant either.

Multi-tenant applications must put the tenant label key, such as
`organization`, in the scope's `protected_label_keys`. Keys used in the scope's
rules are protected as well. A protected key is what stops a run from creating a
session for another tenant or relabeling a session out of its own; the resource
access tests for these tools cover both for the tenant key.

`list_sessions` returns pages of at most 50 sessions, newest first, with
`next_cursor` for the next page. Filters are agent name, labels, status,
`created_after`, `last_activity_before` and case-insensitive text over the
redacted title, agent name and labels; hidden text and raw session ids never
select a row. Exact agent-name or label filters containing registered secrets
match no rows. Each row has the session's alias, its parent's alias, agent,
title, status, labels, timestamps, summary and a short outcome: the final
assistant text, or the final structured output when there is no text, cut to
`max_outcome_chars`.

### Session aliases and parents

The tools never show raw session ids. `session_id`, `parent_session_id` and
`next_cursor` carry opaque aliases from the store's public authority alias
codec (field `session_ref`), so the tools require a store with an alias keyring
(see [the alias keyring](#the-alias-keyring) below). An alias is an HMAC of
the session id, keyed and scoped to the calling session: it is stable for that
caller, so the model can list, read and cite it, but it reveals nothing about
the raw id, and another session gets a different alias it cannot resolve.
The tools require a writable alias registry even though they do not modify
session content. Construction rejects read-only and diagnostic stores before
any tool call. `ToolEffect.NONE` describes the model's access to session content;
Cayu still persists opaque alias mappings as internal bookkeeping.

`read_session` accepts only these aliases; the tools record each alias they show
so a later call, worker or restart can resolve it. Transcript content still
shows whatever ids the original messages contained.

A row's `parent_session_id` is the parent's alias only when the same scope and
policy would let the caller read the parent; otherwise it is `null`, exactly as
for a root session. The caller's own session counts as hidden unless the policy
sets `include_caller=True`.

`read_session` returns one session's details, its outcome including the final
structured output, and up to 50 transcript messages from `offset`, with
`next_offset` to continue. The window is read with
`SessionStore.load_bounded_transcript_window`, which checks each message's
stored size (`message_bytes`, recorded at write time since storage revision 129)
before loading it, so the size check reads no message bodies: a message over 1 MiB is listed with its index, role and
`stored_bytes` but not loaded, and the window stops before 4 MiB of loaded
messages, so `next_offset` starts at the first message not returned. The structured-output search reads at most the last
eight messages and 512 KiB of stored message bytes, newest first, through the
same bounded window read, so it applies the calling run's resource-access scope.
It reports `structured_output_status` as `found`, `none`,
`not_found_within_limit`, `too_large` or `unavailable` (including when the scope
denies the read) instead of reading further. Each message keeps at most `max_message_chars`
characters across its parts. Text, tool calls, tool results, structured output,
peer content, files and citations appear in the durable transcript's stored form,
so tool arguments the runtime withheld stay unavailable. Reasoning and provider
protocol state are omitted. Both tools redact the calling invocation's
registered secrets before JSON encoding or cutting text and fit their results to the declared
`max_terminal_payload_bytes`. Tool descriptions ask the model to cite the
session aliases it relies on and to treat their content as possibly stale.

`SessionQuery` supports the same scoping directly: `root_session_id` selects one
root session tree, served by the revision-129 root-session index on SQLite and
PostgreSQL, and `agent_names` selects sessions run by any listed agent.

### The alias keyring

A generated project builds its stores with
`public_authority_alias_codec_from_environment()`, which reads two variables:

- `CAYU_PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID`: the key ID that signs new aliases,
  for example `primary` (lowercase letters, digits, `-` and `_`, starting with a
  letter).
- `CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS`: a JSON object mapping each key ID to the
  unpadded base64url encoding of 32 random bytes, for example
  `{"primary": "<43 characters>"}`.

`cayu new` doesn't set them, because a store that has used a keyring keeps
requiring it (see below). To make a keyring, run `cayu keys generate`:

```bash
uv run --no-sync cayu keys generate --env-file .env
uv run --no-sync --env-file .env python run.py --message "YOUR REQUEST"
```

`--env-file` adds both variables to the file, creating it with owner-only
permissions if it doesn't exist, and prints only a confirmation. It refuses
when the file already sets either variable, because a store that has used those
keys needs them to open; `--replace` overwrites them, which is safe only when
no database depends on the old keys. A generated project doesn't load `.env` by
itself, so pass `--env-file .env` to `uv run` as above. Without `--env-file`
the command prints the two `.env` lines, and `--keys-only` prints only the
`CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS` JSON value, for a secret manager such as
`cayu cloud env set CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS --secret --value-file -`.
`--key-id` picks the key ID (default `primary`).

Keep the keys secret, and give every process that opens the same database the
same keys. A deployed app gets its own keyring from its secret manager rather
than a copy of the local one. To rotate, add a new key under a new ID to
`CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS`, keep the old ones (at most four keys in
all) so aliases they signed still resolve, and then set
`CAYU_PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID` to the new ID. Never give an
existing key ID new material. The
[runtime contract](runtime-contracts.md#conversation-interactions) has the
details.

Outside a generated project, pass the codec to the store and its keyring to the
app:

```python
from cayu.runtime import public_authority_alias_codec_from_environment

codec = public_authority_alias_codec_from_environment()
app = CayuApp(
    session_store=SQLiteSessionStore(path, public_authority_alias_codec=codec),
    public_authority_alias_keyring=codec.keyring,
)
```

`InMemorySessionStore`, and `CayuApp()` without a `session_store`, make their
own random key, so unit tests on them pass with no variables set while the real
app on SQLite or Postgres does not start without them.

Once a SQLite or Postgres store has opened with a keyring, it keeps requiring
the same keyring for every writable open. Sessions written without it would get
no aliases, so the keyed store could not resolve them later, and the store
refuses that open with an error that names both variables. A read-only store,
such as the one `cayu session` opens, still opens without the keyring; it reads
sessions normally, but alias lookups find nothing and it can't issue aliases.
Tests that build the app without the variables should use their own database
file or an `InMemorySessionStore` instead of the project's local database.

## Configure the session store

New local projects use one SQLite database at `data/cayu.db`:

```toml
[tool.cayu.session_store]
backend = "sqlite"
path = "data/cayu.db"
```

The file contains broader Cayu runtime state—sessions, events, transcripts,
checkpoints, tasks, knowledge, and budgets. The product-level `cayu.db` name
reflects that broader scope.

Relative paths are resolved from the `pyproject.toml` that declares them. A
PostgreSQL project names an environment variable rather than committing a DSN:

```toml
[tool.cayu.session_store]
backend = "postgres"
env = "CAYU_POSTGRES_DSN"
```

Target selection is deterministic. Explicit `--sqlite PATH` or `--postgres DSN`
wins, followed by `CAYU_DATABASE_URL`, then `[tool.cayu.session_store]`. In an
otherwise configured Cayu project, the CLI recognizes only the canonical
`data/cayu.db` convention. Other filenames are used only when selected explicitly.

The resolver reads `pyproject.toml`; it never imports or boots the application
factory. Inspection opens an existing SQLite database read-only with schema
validation and never creates or migrates it. Apply migrations separately with
`cayu storage migrate` before inspection when a deployment requires them.
SQLite WAL readers may need to create or update `cayu.db-wal` and `cayu.db-shm`
sidecars even though the database connection itself is query-only. Inspecting a
live database therefore requires a writable containing directory. Copy the
database and any non-empty WAL sidecar to writable media before inspecting a
stopped snapshot; Cayu does not silently use SQLite's `immutable=1` mode because
that would ignore live WAL changes. PostgreSQL inspection starts every acquired
store operation with `SET TRANSACTION READ ONLY` and uses validate-only schema
mode. The transaction-scoped guard remains effective behind transaction-pooling
PgBouncer, where backend session defaults are not stable across operations.

## Commands

All commands default to tables and support `--output table|json|jsonl` where a
list of records is returned. Limits are finite; JSON field names form a stable
CLI schema and include `schema_version` in JSON envelopes. Every `session usage`
JSONL row also carries the schema version so independently consumed
`model_call`, `unmatched_ledger`, and `aggregate` records cannot be mistaken for
the version 1 representation.

Session-inspection CLI schema version `9` adds estimated cost to `session usage`: each
model call's `cost`, `cost_currency` and `cost_state` (`priced` or `unpriced`), the
aggregate's `estimated_cost` (priced calls summed per currency) and `unpriced_calls`, and a
top-level `price_book` naming the book that priced them (see [Cost](#cost)). Version `8`
added the exact bounded
`runtime_build_provenance` manifest to `session show`. Operators can distinguish
same-version builds, inspect typed OCI or wheel origin, and see explicit weak or
unavailable legacy evidence without reading runtime-owned metadata. Version `8`
retains version `7`'s bounded
`provider_operation` object with `provider_operation_unavailable`,
`ambiguous_submission`, and `fallback_retry`; a recovery reason; duplicate
request risk; allowed resolutions; and the exact `stage_id` and `run_epoch`
needed for a fenced decision. Version `7` retains version `6`'s
`reconnect_scheduled`,
`reconnect_in_progress`, and `provider_operation_reconciled` states plus
bounded cancellation status, accounting status, and reservation count. These
and `provider_operation_in_progress` include only the provider, operation id,
stream protocol, and bounded control state; private recovery metadata is never
exposed. Version `6` retains version `5`'s reconnect states, version `4`'s
`synchronous` state and provider-operation object, version
`3`'s durable `interaction_id` correlation to event and transcript records, an
`interactions` collection, and response-scoped filters. It retains version
`2`'s identity-free aggregate usage
counters as canonical nonnegative decimal strings in `show.usage`,
`usage.aggregate`, and the JSONL aggregate record. Per-model-call counters
remain signed-64-bit JSON integers. Consumers must branch on `schema_version`
before accepting the additive response-correlation fields.

```console
# Newest activity first; filters may be combined.
cayu session list --status completed --agent reviewer --label tenant=acme

# Sessions created in a window: --since is inclusive, --until exclusive. A bare date
# means UTC midnight.
cayu session list --since 2026-10-27 --until 2026-10-28T12:00:00-07:00

# The sessions run under one task, such as those its handler started.
cayu session list --task TASK_ID

# Each session's overview and first user message, as the Sessions page shows them.
cayu session list --overview
cayu session list --overview --table --application bug-hunter

# Content-free operational summary, including runtime name/version, usage,
# pending-action counts, session timing, and the worker's event-loop lag.
cayu session show SESSION_ID

# A workflow run as a whole: steps, child sessions, usage and custom events.
cayu session workflow SESSION_ID --table

# Submit one explicit disposition through the authenticated server. The full
# Authorization value is read from CAYU_API_AUTHORIZATION by default.
export CAYU_API_AUTHORIZATION='Bearer ...'
cayu session resolve-provider-operation SESSION_ID \
  --stage-id STAGE_ID --run-epoch RUN_EPOCH \
  --action fallback_retry --reason 'Duplicate-request risk accepted.' \
  --server-url https://cayu.example.com

# Newest response-scoped summaries first, using a stable sequence cursor.
cayu session interactions SESSION_ID --limit 50

# Per-model-call and aggregate token/cache usage, priced from the bundled price book.
cayu session usage SESSION_ID --output json
cayu session usage SESSION_ID --price-table prices.json --table
cayu session usage SESSION_ID --interaction-id INTERACTION_ID --output json

# Paired parallel tool calls, timing, status, and result sizes, without results.
cayu session tools SESSION_ID --interaction-id INTERACTION_ID

# Stable sequence pagination. Payloads appear only when explicitly bounded.
cayu session events SESSION_ID --type tool.call.completed --output jsonl
cayu session events SESSION_ID --after-sequence 120 --include-payload 2048
cayu session events SESSION_ID --interaction-id INTERACTION_ID

# Bounded previews by default; --sizes identifies context-amplifying records.
cayu session transcript SESSION_ID --offset 0 --limit 100 --sizes
cayu session transcript SESSION_ID --include-content 4096 --output json
cayu session transcript SESSION_ID --interaction-id INTERACTION_ID

# The structured output a finished session answered with.
cayu session output SESSION_ID
cayu session output SESSION_ID --interaction-id INTERACTION_ID --table
```

### Runner commands of a tool call

Each `cayu session tools` row lists the runner commands its call ran, in
`runner_commands`, from the call's `runner.exec.completed` events. `runtime_ms` is how
long the Runtime waited for the runner. When the runner's backend keeps its own record
of the command, as the sandbox broker does, `id` is the backend's command id, the one
`cayu cloud sandboxes show` lists, and `accepted_at`, `started_at` and `finished_at` are
the backend's times: when it took the request, when it handed the command to the
sandbox, and when the command ended. `backend_ms` (accepted to finished), `queue_ms`
(accepted to started) and `run_ms` (started to finished) split them, so a slow call
reads as Runtime time, broker time and command time. A backend that doesn't report a
time leaves it `null`. `--table` shows each command as its id followed by those times.

### Session overviews

`cayu session list --overview` adds an `overview` object to each row: the projection
`GET /api/sessions?include=overview&include=first_message` returns, the same one the
Developer dashboard's Sessions page shows. It has the session's `state` (such as
`waiting_approval` or `failed`) and `reason`, `title`, `summary`, `first_message` (the
session's first user text, redacted, trimmed and cut to 256 characters) with
`first_message_unavailable`, `started_at`, `ended_at`, `model_steps`, `tool_calls`,
`tokens`, `cost`, `uses` (`browser`, `subagents`, `artifacts`), `eval` and `forked`. Counts
are decimal strings, as in the HTTP API. Filters and paging work as without the option.
`--table` prints one line per session with the state, title, first message, steps,
tokens and cost; `--jsonl` prints one row per line.

With `--application NAME` the Agent builds the overviews in the list request itself,
with its own redaction and price book. An Agent whose Cayu predates first-message
previews still lists overviews, with `first_message` null and
`first_message_unavailable: true`. With a local store (`--sqlite`, `--postgres` or the
project's store) the CLI builds the same projection from the store in this process.
A bare store knows neither the application's registered secrets nor its artifact
stores, so text is redacted with the CLI's secret patterns, as in the other session
commands, and `uses.artifacts` is `null`. Costs are priced with Cayu's bundled price
book, or `--price-book PATH`. Overviews need the `server` extra installed locally
(`pip install "cayu[server]"`).

### Where a session's time went

`cayu session show` ends with a `timing` object that splits the session's elapsed time,
from its first recorded event to its last, into parts that add up to it. Each moment
counts once, in the first part it belongs to, in this order:

| Field | Meaning |
| --- | --- |
| `waiting_ms` | Waiting for a person: an approval or input request until it was answered. |
| `setup_ms` | Environment setup: factory allocation or reconnect, binding (copy-in, Git baseline, setup commands) and deferred materialization. |
| `teardown_ms` | Environment teardown: binding finalization (copy-back and releasing the sandbox), which runs after the turn's answer. |
| `model_ms` | Model calls running. |
| `tool_ms` | Tool calls running; parallel calls count once. |
| `other_ms` | The rest of the time inside turns: the runtime's own work between steps. |
| `idle_ms` | Time outside any turn and its teardown, such as between two turns. |

A model call, tool call, environment operation or turn that is still running (in a live
session, or one whose process died) counts until the last recorded event, so a session
read mid-turn shows its work as it stands rather than as idle time.

`elapsed_ms`, `started_at` and `ended_at` give the whole span, `model_calls` and
`tool_calls` the counts, and `first_model_call_ms` how long after the start the first
model call began. `gap_ms` is the time after that not counted in `model_ms` or `tool_ms`:
the runtime's work between steps, environment setup and teardown, waits for a person and
idle time, including setup and waits that happen inside a call (a deferred environment
starting during a tool call, an approval wait, a call still running when the events were
read); in a session without a model call it is all the time not counted in `tool_ms`. So
these add up to the elapsed time:

| View | Fields |
| --- | --- |
| Total | `elapsed_ms` |
| Before the first model call | `first_model_call_ms` |
| Model calls running | `model_ms` |
| Tool calls running, parallel calls once | `tool_ms` |
| Neither, after the first model call | `gap_ms` |

The one exception is a tool call that ran before the first model call: its time counts in
both `first_model_call_ms` and `tool_ms`.

`per_model_call` says how long single model calls took: `calls` (those with a recorded
start and end), `total_ms`, `average_ms` and `max_ms`. `per_tool` lists the same for each
tool, the longest total first, with its `tool` name; parallel calls each count in full
here, and a call's time waiting for an approval is left out. It lists at most 100 tools
(`per_tool_truncated` says more were left out). One breakdown reads at most 100,000 of
the session's timed events (model, tool, environment, turn and wait events); `truncated`
says it stopped there, and the breakdown then covers the time up to the last event read.

`environment` lists each environment operation in the order it
started (at most 100; `environment_truncated` says more were left out): its `kind`
(`factory`, `binding`, `materialization` or `finalize`), `environment`, `started_at`,
`duration_ms`, `outcome` (`completed`, `failed` or `running`) and `phases`, the steps the
operation timed (see [Environment step timings](runtime-contracts.md#environment-step-timings)).
Each step has its `phase`, `started_at`, `duration_ms`, `outcome` (`completed`, `failed`,
or `unfinished` for a failed operation's step that was still running), `depth` (how many
steps it ran inside: `archive_preparation` runs inside `transfer`) and, when the step
reported them, `items` and `bytes`. A session recorded before Cayu timed the steps, or a
binding that reports none, has empty `phases`; the operation and the parts still show.

The table prints the parts with their share of the elapsed time, the time before the
first model call and the gaps, each tool's calls, then one row per operation and step:

```console
Time (53m48s elapsed)
part      time    share  notes
setup     7m42s   14%    environment setup: allocate, copy in, setup commands
model     17m02s  32%    22 calls, first after 8m42s, 46.4s average, longest 2m10s
tool      11m24s  21%    62 calls
teardown  12m12s  23%    environment teardown: copy back, release
...
Before the first model call: 8m42s. After it, 16m40s went to neither model nor tool calls (runtime work between steps, setup, teardown, waits and idle time).

Tool calls by tool
tool         calls  total  average  longest
run_tests    9      6m12s  41.3s    1m58s
read_file    31     3m40s  7.0s     35.0s
...

Environment operations
step                          time   outcome    work
binding (sandbox)             7m42s  completed  -
  transfer                    2m54s  completed  4877 items, 129048576 bytes
    archive_preparation       1m18s  completed  4877 items, 131072000 bytes
  final_source_observation    4m12s  completed  4877 items
finalize (sandbox)            12m12s completed  -
  copy_back_publication       11m58s completed  45 items, 138412032 bytes
  sandbox_release             2.1s   completed  -
```

For a Cloud Agent the CLI reads `GET /api/sessions/{id}/timing`, which the Agent computes
with the same code as a local read. An Agent whose Cayu predates it reports `timing` as
`null`, and the table says to redeploy; one whose Cayu predates `gap_ms`, `per_model_call`
and `per_tool` reports them as zero or empty.

### Sessions run under a task

Runtime marks each session with the task it runs under, in the runtime-owned session
metadata `cayu:task_id`: the task a session is attached to as its execution
(`RunRequest.task_id`), else the task whose handler is running when the session is
created, else the task of the session it was created under (a subagent or a later child).
Run, fork and metadata requests that set `cayu:task_id` are refused. A session a handler
starts some other way, such as through another application's API, is not marked.

`cayu session list --task TASK_ID`, `SessionQuery(task_id=...)` and
`GET /api/sessions?task_id=...` list them. `cayu task show` adds them up (see
[Inspect tasks](project-workers.md#inspect-tasks)). Sessions
created before Cayu recorded the mark have none, so they aren't listed; a task's own
`session_id` is still read. A Cloud Agent whose Cayu predates the mark ignores the filter;
the CLI notices and says to redeploy.

### Structured output

`cayu session output SESSION_ID` prints the structured output the session's latest
interaction validated: the `output` of the last `structured_output.validated` event of
that interaction. An earlier answer of a continued session is read with `--interaction-id`
(see `cayu session interactions`). JSON (the default) carries `output` with its `name`,
`attempt`, `interaction_id`, the event's `sequence` and `validated_at`, and the session's
`session_status`; `--table` prints those fields and the output indented. Values under secret-looking keys are redacted as in the
other session commands.

| Exit | `error.code` | Meaning |
| ---: | --- | --- |
| `0` | | The output was printed. |
| `1` | `SESSION_INSPECTION_FAILED` | The session or its store couldn't be read. |
| `3` | `NO_STRUCTURED_OUTPUT` | The interaction ended without a validated structured output: the run asked for none, or validation never succeeded. |
| `4` | `SESSION_NOT_FINISHED` | The session is pending, running or interrupting, or its latest interaction is paused for an approval or an answer. Read it again once it ends. |

### Cost

`session usage` and `session rollup` price every model call from Cayu's bundled, dated
price book (`cayu.budgets.default_price_book()`) on the date the call completed. Pass
`--price-table PATH` (or `--price-book PATH`) to use your own price book instead: a JSON
file written by `cayu.budgets.dump_price_book`. A call the book has no price for, or without
usage evidence, is `unpriced` and counted, never guessed. Costs are estimates from token
counts; the budget ledger's reconciled amounts (`pricing_state` and the unmatched ledger)
are separate evidence and are unchanged. Totals are kept per currency. Calls are priced
as Cayu's cost accounting prices them, hosted web searches included, and the totals also
cover auxiliary inference attempts, so `estimated_cost` matches
`POST /api/sessions/{id}/cost` for the same book. Tables show a group or session whose
calls all lacked a price as `unknown` and a partly priced one as `... + unpriced`, never
as free.

For a Cloud Agent the CLI sends the price book with the request, so the Agent prices with
the same book; an Agent whose Cayu predates priced usage returns its calls without cost.

### Why a call ended

Each `session usage` call row carries `finish_reason`, the normalized reason the model
call ended (`stop`, `tool_calls`, `length`, `content_filter`, `error` or `unknown`), and
`raw_finish_reason`, the provider's own value (for example `max_tokens`). The table shows
`finish_reason`; JSON and JSONL carry both. A `length` call was cut at the output cap. Both
are `null` for a call recorded without completion details. A Cloud Agent whose Cayu
predates these fields doesn't send them, so its JSON and JSONL rows have no
`finish_reason` or `raw_finish_reason` keys and the table shows `-`.

### Call timing

Each `session usage` call row also says how long the call took: `started_at`, when its
request started (its `model.started` event), `duration_ms`, from then to its completion,
and `first_output_ms`, from then to its first streamed text or thinking delta. A slow
provider shows a long `first_output_ms`; a call that streamed quickly but finished late
spent its time generating. `first_output_ms` is `null` for a call that streamed no text or
thinking, such as one that only returned tool calls, and all three are `null` for a call
whose start is outside the selected window or that a Cloud Agent on an older Cayu
reported. The table shows `duration` and `first_output`.

### Request controls

`session usage` also shows the generation controls each model call sent and, where the
provider's API echoes them back, what the provider reported. Each call in JSON and JSONL
carries:

| Field | Meaning |
| --- | --- |
| `request_controls` | The `reasoning_effort`, `thinking_budget_tokens`, `temperature`, `top_p`, `top_k` and `max_output_tokens` the call sent, after Cayu's mapping of `ThinkingConfig` and provider options and the adapter's defaults. A control that is absent wasn't sent; `{}` means none was. Null when the provider adapter doesn't report its controls, and for calls recorded before Cayu did. |
| `echoed_request_controls` | The controls the provider echoed on its response, with `null` where it echoed a control as null. Null when the response echoed nothing; OpenAI Responses echoes effort, cap, temperature and top_p, the other built-in APIs don't. |
| `request_controls_mismatch` | Controls that were sent and echoed with a different value, such as an output cap a proxy echoed as `null`. An empty list means the echo matched; null means there was nothing to compare: no echo, or no sent controls (an adapter that doesn't report them, or a `model.started` outside the selected window). Compaction calls are included. |

The table output adds a `Request controls` section, one row per call that has either
record:

```console
Request controls
sequence  sent                                                      echoed                                                                mismatch
14        reasoning_effort=high temperature=0.5 max_output_tokens=128000  reasoning_effort=high temperature=0.5 top_p=0.98 max_output_tokens=null  max_output_tokens
```

The sent controls come from the call's `model.started` event and the echo and mismatch
from its `model.completed` event (see [Request controls](runtime-contracts.md#request-controls)).
`cayu session events --type model.started --include-payload 2048` shows the same record
for every attempt, retries included. A Cloud Agent whose Cayu predates these fields returns
its calls without them.

### Event-loop lag

A worker records on each session it runs when its process's event loop fell behind while
the session ran (`runtime.event_loop.lagged` events; see
[Watch for saturation](project-workers.md#watch-for-saturation)). `cayu session show`
sums them as `event_loop_lag`: `reports`, `lagging_ms` (about how long the loop was late),
`max_lag_ms`, `first_at`, `last_at`, `processes` and `truncated` (more than 1,440 reports,
of which the first were read). The table adds a sentence when there were any. `cayu task
show` gives the same over the sessions run under the task, with `sessions` (how many were
read) and `sessions_lagged` (how many lagged); sessions that ran at the same time each
count the lag they saw. A session
slowed by its process shows lag here; one slowed by its own work doesn't. A Cloud Agent
whose Cayu predates these events reports `event_loop_lag` as `null`.

### Workflow runs

`cayu session workflow SESSION_ID` shows one workflow run as a whole. `SESSION_ID` is the
run's own session (the id passed to the workflow's `run`), whose `workflow.*` journal names
the child session each `step()` ran in:

```console
cayu session workflow team-solve-d489aeb3b57f --sqlite runs/team.db --table
cayu session workflow team-solve-d489aeb3b57f --application team-solver --json
```

JSON carries:

- `workflow`: the run's name (from `workflow.started`), agent, causal budget, when the
  journal says it started and completed, and `status`, read from the journal: `completed`
  once it recorded `workflow.completed` for its latest attempt (a run started again under
  the same session id counts from its new `workflow.started`), `running` while it hasn't and a session in the run
  is still live, and `stopped` when it hasn't and none is (a step raised, the run was cut
  off, or its process is between steps or running a `command()` check, which has no
  session; that command's row has `journal` `started` until it ends, and the journal can't
  tell a command still running from a process that died running it). `session_status` is the status of the run's own
  session, which the shipped journal marks completed before the run starts, so it doesn't
  say whether the run finished. A session without a workflow journal reports its own
  status in both.
- `steps`: one row per journaled step and its child session, in journal order, with the
  step id, agent, child session id and its status, `journal` (`started` or `completed`),
  `started_at` and `ended_at` (the journal's times, or the child session's creation and,
  once it finished, last activity), and model calls, tool calls, token usage (reasoning
  tokens included) and estimated cost. A step's numbers cover its child session and every
  session under it, which `sessions` counts. A `gated_loop` item row names its loop, item
  key and `outcome` (`pass` or `fail`) and the child of the step it ran; that step's own row
  carries the usage, unless no step names that child, in which case the item row does. A
  `command` step row (a `command()` check, which runs in no child session) has no child or
  usage and carries its `exit_code` and `outcome` (`passed`, `failed`, `timed_out` or
  `cancelled`); each run of it, a rerun included, is its own row. Direct children of the
  run that no step names are listed with `kind` `child`.
- `custom_events`: the `custom.*` events the workflow emitted, in sequence order, with a
  payload preview bounded by `--payload-bytes` (default 2048). Cayu's own `custom.cayu.*`
  markers are left out.
- `totals`: the run's sessions by status, model calls, tool calls, usage and estimated cost
  across the workflow session and every session under it, children of children included.

Usage comes from the same per-session aggregate as `cayu session usage`, priced as
[Cost](#cost) describes (`--price-book PATH`). Child sessions are found through their
`parent_session_id`, breadth first; `--max-sessions` (default 1000) and `--max-events`
(default 100,000, the run's own events) bound the read, and `complete: false` says one of
them stopped it. `--jsonl` prints a `step` record per row, a `custom_event` record per
event, and a closing `summary` record.

With `--application` the Agent's public event API names each step and its child session
(see [Workflow journal replay](runtime-contracts.md#workflow-journal-replay)), so the
output matches the Agent's store. A child session id that redaction would change comes
back as its public session alias, which the other `cayu session` commands accept. An
Agent on a Cayu that predates this withholds both ids; the output then has
`step_ids_withheld: true` and lists the run's child sessions (`kind` `child`) in place of
named steps, with the same gated-loop outcomes, custom events and totals.

### Rollups across many sessions

`cayu session rollup` sums usage, time by phase and tool failures across the sessions
created in a window and groups them by agent (default), UTC creation day, Release, or a
label:

```console
cayu session rollup --since 2026-10-01 --by day --table
cayu session rollup --since 2026-10-01 --by label:repository --agent hunter --json
cayu session rollup --by release --application bug-hunter --table
cayu session rollup --causal-budget CAUSAL_BUDGET_ID --by agent --table
cayu session rollup --root RUN_SESSION_ID --json
cayu session rollup --task TASK_ID --by agent --table
```

`--causal-budget ID` keeps the sessions charged to one causal budget, which a run shares
with the child sessions it spawns (`cayu session show` prints it as `causal_budget_id`; by
default it is the root session's id, or its task's). `--root SESSION_ID`
keeps a root session and every session its invocation started. Either one sums a single
workflow run; combined with `--label` or `--by label:KEY` they compare runs. `--task
TASK_ID` keeps the sessions run under one task (see
[Sessions run under a task](#sessions-run-under-a-task)), so `--by agent` shows which of a
task's agents used its time and tokens.

Each group reports its sessions and their statuses, model calls (with usage evidence and
errors), tool calls and tool failures, aggregate token usage (auxiliary inference and
hosted tool usage included, as the store's usage totals count them), estimated cost per
currency with the count of model calls the price book couldn't price (see [Cost](#cost)),
and time in milliseconds:

| Field | Meaning |
| --- | --- |
| `active_ms` | Active time of finished turns, without time spent waiting for a person. |
| `wall_ms` | The same turns' wall time, waits included. |
| `model_ms` | Union of model-call intervals (start to completion or error). |
| `tool_ms` | Union of tool-call intervals; parallel calls count once. |
| `setup_ms` | Union of environment factory, binding and materialization intervals. |
| `teardown_ms` | Union of binding finalization intervals (copy-back and release). Teardown runs after a turn's answer, so it is not part of `active_ms`. An Agent whose Cayu predates it reports `0`. |
| `gap_ms` | Active time no model call, tool call or setup covered: the runtime's own work between steps. |

Time spent waiting for a person (from an approval request to its decision, block or resume,
or from an input request to the answer, and at the latest until the model runs again) is
cut out of every phase, as it is out of `active_ms`, so a tool call that waited an hour for
approval counts only the time it ran. `gap_ms` is
approximate: an interval in an unfinished turn still counts as busy, and an unfinished turn
adds no active time. `release` groups a session by the Release it started
on (`session.started`); sessions without one group as `(none)`, as do sessions without
the label `--by label:KEY` names.

`failure_kinds` counts failed and blocked tool calls by tool and error kind. The kind comes
from structured evidence only: `blocked:<by>` for a blocked call, the first exception type
the failure evidence names (such as `TimeoutError`), its classification (`timeout`,
`deadline`, ...), an isolated tool's failure code, `tool_error` when the tool returned an
error result, else `unknown`. Groups keep exact counts for every kind while pages are
merged, so the answer doesn't depend on page size; the output lists each group's 200 most
common kinds and counts the rest as one `*`/`other` row, and tables show the top
`--failure-kinds` (default 10). A rollup page served by an Agent carries every kind.

The rollup reads one page of sessions at a time, oldest first, and each session's events in
byte-bounded pages of only the event types it needs, so memory stays bounded however large
the store is. `--max-sessions` (default 10,000) stops early; the output then has
`complete: false` and a `next_cursor` to pass back with `--cursor` (in JSONL, the last
record has `record_type: summary` and carries both). A Cloud Agent serves the
same computation a page at a time at `POST /api/sessions/rollup`; an Agent whose Cayu
predates it answers that it doesn't serve rollups yet.

### Closing a session nobody will continue

`cayu session close` ends a pending or interrupted session that won't continue, such
as one waiting on a question or an approval it can no longer resume from:

```console
cayu session close SESSION_ID --reason "Stranded by the 27 Oct release."
cayu session close SESSION_ID --application aifinance --reason "Stranded by the 27 Oct release."
```

It calls `CayuApp.close_session` (`POST /api/sessions/{id}/close`). The session fails
with a `session.failed` event that has `error_type: "SessionAbandoned"`, `closed:
true`, the reason and who closed it under `metadata`, and the `approval_id` or
`input_id` it waited on. Its pending approval or question leaves the inbox, a call
whose outcome is uncertain stays recorded as unknown and never runs again, and its
transcript and events are kept. Recovery then releases its environment. Its pending
or interrupted child sessions are closed with it; a subagent whose parent waits on
it is refused, so close the parent. A running session is closed only when the worker
that ran it is provably gone: its execution lease lapsed or its process died, or every
task of it finished, as for one a deploy cut off. It closes even where the session's
environment isn't registered (a per-run environment only that worker registered);
the calls it was running are listed as unknown and the environment binding stays
recorded. A run that stopped inside the agent's model call closes too: the call is
listed in `model_call_outcome_unknown` on `session.failed`, its budget is charged
at the reserved amount, and it is never made again. Model work that can't be settled
without the environment (a background provider operation, or an answer the provider
gave that isn't published yet) is refused with a message that names it and says
how to settle it. A session that is running or interrupting with a live owner or one not
known to be gone, already completed or failed, asleep (interrupt it first, which
cancels its wake), or has a task or a child session still running is refused. Repeating a close returns
the closed session, and closes any child sessions a close that stopped part way
left open, after checking every remaining child can close. Independent forks of a
closed session are left alone by a repeated close. A retry also traverses children
that already recorded their close, so an interruption deeper in the cascade does
not strand their descendants. You don't have to repeat it yourself: every
`recover_abandoned_sessions` sweep finishes a close that stopped part way once its
root meets `inactive_for_seconds`, with the reason and person the close recorded.
It also retries retained cleanup after every child is failed; an explicit zero
window permits immediate recovery. While one of the remaining children is
running again, the sweep closes none of them and reports `skipped_active` for the
closed session; interrupt that child and the next sweep, or a repeated close,
closes the rest.

You often don't have to close such a session yourself. `cayu serve` reconciles,
every 10 minutes, each running session whose owner is provably gone and that has
been quiet for an hour (`CayuApp.reconcile_orphaned_sessions`): it is left
interrupted, with a recovery item in the inbox when a call it was running has an
unknown outcome, or, where its environment isn't registered, it fails with reason
`orphaned_session_reconciled`. `ServerLifecycleConfig` sets both times
(`orphaned_session_reconcile_interval_seconds`, `None` to turn it off, and
`orphaned_session_inactive_after_seconds`).

With `--application NAME` it goes through the Agent's own Cayu API as you, like the
reads below, which needs a developer or admin. Without it, it opens the project's
application in this process (`[tool.cayu] factory`, or `--target module:factory`) and
records the close as `--as SUBJECT`, by default `local:` and your user name. Closing
settles the session's environment through that application, so it doesn't take
`--sqlite` or `--postgres` the way the read commands do. JSON output reports
`closed`, `status`, `actions` and `message`; `--table` prints one line.

### Sessions of a Cloud Agent

Add `--application NAME` to `list`, `rollup`, `show`, `workflow`, `interactions`, `usage`,
`tools`, `events`, `transcript` or `output` to read a Cayu Cloud Agent's sessions instead of a local
store. `NAME` is the Agent's id, name or slug, as in `cayu cloud` commands. The CLI signs
in the same way: `CAYU_CLOUD_API_KEY` or an API-key file, the selected private context
(`--context PATH`, `CAYU_CLOUD_CONTEXT`, or `cayu cloud context use`), or `cayu cloud login`.

```console
cayu session list --application bug-hunter --status failed
cayu session tools SESSION_ID --application bug-hunter --table
cayu session events SESSION_ID --application bug-hunter --type tool.call.failed --include-payload
cayu session output SESSION_ID --application bug-hunter
```

The CLI asks Cayu Cloud where the Agent serves Cayu and for its
[Developer dashboard login](cayu-cloud.md#agent-operator-credentials), which only
developers and admins may read, then reads the Agent's own `/api/sessions/...`
endpoints. The login stays in memory and is never printed. A request to a sleeping Agent
wakes it; the CLI waits up to 90 seconds while it starts. The Agent's Cayu API is found at
`/cayu/api` (an application that mounts Cayu under `/cayu`) or `/api` (`cayu serve`).

The commands print the same rows as for a local store, with what the Agent's public API
withholds left out:

- `list` and `show` report `run_epoch` as `null`; `list` also reports `last_activity_at`
  as `null`, and `total_count` as `null` with `--since` or `--until`, because an Agent on
  an older Cayu ignores the window and the CLI filters its rows itself. `show` reports record byte sizes, queue counts, `operation` and `budget` as
  `null`.
- `events`, `interactions` and `output` show the Agent's public view of each record: private
  ids are record-scoped aliases such as `cayu_event_7:tool_call_id`, so payload sizes differ
  from the store's.
- `usage` and `tools` pair events by their private identity on the Agent
  (`GET /api/sessions/{id}/model-calls` and `GET /api/sessions/{id}/tool-calls`). A
  tool call's `tool_call_id` is its public `tool_call_correlation_id`, the value its
  tool events carry; model step, attempt and round ids are `null`, and unclaimed budget
  reservations have a `null` `reservation_id`. Arguments recorded before a call ran show
  as `[quarantined]`, as the Agent's events endpoint shows them. An Agent whose Cayu predates these
  endpoints answers that it doesn't serve them yet.

`snapshots` and `resolve-provider-operation` read a local store or a server URL only.

`fallback_retry` starts a new model attempt from the last durable Cayu
boundary; it is not continuation of the unavailable provider operation. When
`duplicate_request_risk=true`, the provider may already have accepted the old
request, so the operator owns the decision to accept possible duplicate work
and cost. Existing budget and run limits can still stop the fallback before
provider dispatch; that typed interruption completes the accepted disposition.
An explicit operator interruption before replacement dispatch does the same.
`fail` closes the original attempt and session through the ordinary
durable failure path. Both actions are immutable, auditable, run-epoch fenced,
and reject a stale or conflicting retry. Because actual usage of unavailable
provider work cannot be proven, both actions conservatively charge any active
source reservation to its full reserved amount before replacement admission or
failure terminalization. If a CLI connection is ambiguous,
reuse the printed `mutation_id` only to observe the same mutation; do not issue
a second decision. Use HTTPS outside loopback and supply authentication required
by the deployment.

`events --include-payload` and `transcript --include-content` accept a per-record
UTF-8 byte ceiling. Transcript content also has a 1 MiB total-output ceiling per
invocation. `show` returns at most 200 sorted session labels and reports the
exact `label_count` plus `labels_truncated`. Sensitive key/value shapes, bearer
credentials, common API-token
forms, PostgreSQL URL passwords, provider signatures, and encrypted provider
state are redacted before rendering; opaque provider-state payloads are omitted
structurally rather than relying on provider-specific secret key names. Commands
that aggregate event histories retain only purpose-specific projections and stop
at 64 MiB or 100,000 retained event records. Large raw tool results remain
inspectable through exact size metadata because their content is not retained by
the `tools` projection. When denial or expiry evidence conflicts with completed
or failed execution for the same exact approval-scoped call, `tools` reports both
the status and approval state as unavailable and omits terminal result metrics.
For longer histories, `usage` and `tools` accept
`--after-sequence` and `--before-sequence` event windows so every portion remains
inspectable without disabling the safety ceiling. Totals in those outputs apply
to the selected window. Content flags are still an operator trust
decision: transcript prose and tool output may contain application data that is
not recognizable as a credential.

Usage output leaves ledger reservations unassociated when durable events do not
identify a model call; it never assigns them to the most recent call by timing
alone. Default table output includes a separate `Unmatched ledger` section, and
JSONL identifies rows with
`record_type=model_call|unmatched_ledger|aggregate`. The same
`--offset` and `--limit` page the model-call and unmatched-ledger collections
independently; JSON reports each collection's total, continuation offset, and
`has_more` state.
`model_calls_with_usage` distinguishes model steps with valid usage evidence from
all durable model completions. Summary cost states distinguish `unknown`,
`unpriced`, `partial`, `mixed_currency`, and fully `priced` evidence. Tool
`artifact_bytes` measures the serialized artifact metadata carried by the
durable result, not bytes in an external artifact store. Tool rounds resumed
from approval or user-input pauses are grouped only by their complete durable
model-step, model-attempt, and tool-round identity. They never derive synthetic
identifiers from approval or user-input IDs; missing, malformed, or contradictory
identity evidence is isolated and reported as unavailable.
Interaction listing is store-native and bounded; it does not group the complete
event history in the CLI. `--interaction-id` applies at the store query, so
adjacent responses cannot leak into scoped usage, tool, event, or transcript
output. Transcript results retain their absolute session indices. Interaction
`model_step_count` includes failed provider attempts and retries. Wall duration
is unavailable until the interaction is terminal because a paused interaction's
human wait is still increasing; active duration excludes that wait.

## Diagnostic examples

- Healthy completion: `cayu session list --status completed`, then `show` to
  verify terminal state, model/tool counts, and usage totals.
- Failed run: `cayu session list --status failed`, then `show` for terminal
  failure state and `events --type session.failed --include-payload 2048` for a
  bounded, redacted failure record.
- Awaiting input or approval: `show` reports pending-action counts and kinds;
  `events` can narrow the durable approval or user-input timeline. These commands
  inspect only—they never approve, answer, resume, or cancel.
- Parallel tools: `tools` groups calls by tool round and pairs each terminal event
  by `tool_call_id`, preserving missing terminals as running rather than
  multiplying calls through a join.
- Oversized result: `transcript --sizes` reports exact serialized message
  and part sizes without printing the 500 KB result. Use an explicitly bounded
  `--include-content` only when the content itself is needed.

### Inspect environments

Read an agent's registered environments and retained-session activity from its
running server or Cayu Cloud:

```bash
cayu environment list --server-url http://127.0.0.1:8000 --table
cayu environment show local --server-url http://127.0.0.1:8000 --json
cayu environment list --application my-agent --context ./cloud-context.json --json
```

The output includes declared MCP server names, the artifact store kind, exact
retained-session counts, last activity and each activity snapshot's time.
Unsupported activity reads are unavailable, not zero. The count includes eval
trials and forks and has no time-window restriction. See
[environment inventory](control-plane.md#environment-inventory) for the details.

Environment declarations belong to the running agent, so these commands require
`--server-url` or `--application`. `cayu inspect --environment NAME` remains the
static source-manifest inspection command. File output uses the shared CLI
protection for explicit, environment-selected and saved Cloud context files.

The sandboxes workflow runs share have their own commands. A run that crashed
and is never resumed keeps its sandbox; list the recorded ones and release one,
discarding it or copying its work back first:

```bash
cayu environment allocations list --application my-agent --table
cayu environment allocations list --workflow-run RUN_ID --server-url http://127.0.0.1:8000 --json
cayu environment allocations release ALLOCATION_ID --discard --application my-agent
cayu environment allocations release ALLOCATION_ID --copy-back --target myapp.app:build_app
```

Without `--application` or `--server-url` they open the project's application,
as `cayu session close` does. A release is refused while the run looks like it
is still running; `--force` overrides that when you know its process is gone.
See "Releasing a crashed run's allocation by hand" in
[environment factories](environment-factories.md) for the rules.
