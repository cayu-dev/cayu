# Session inspection

`cayu session` is the supported interface for diagnosing durable sessions and
submitting one explicit provider-operation disposition. Inspection uses
`SessionStore` contracts, so SQLite and PostgreSQL expose the same fields
without requiring operators to know Cayu table names or JSON paths. The
resolution command talks to the authenticated Cayu server; it never mutates the
store directly.

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

One way to make a key:

```bash
python -c "import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('='))"
```

Keep the keys secret, and give every process that opens the same database the
same keys. Outside a generated project, pass the codec to the store and its
keyring to the app:

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

Session-inspection CLI schema version `8` adds the exact bounded
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

# Content-free operational summary, including runtime name/version, usage, and
# pending-action counts.
cayu session show SESSION_ID

# Submit one explicit disposition through the authenticated server. The full
# Authorization value is read from CAYU_API_AUTHORIZATION by default.
export CAYU_API_AUTHORIZATION='Bearer ...'
cayu session resolve-provider-operation SESSION_ID \
  --stage-id STAGE_ID --run-epoch RUN_EPOCH \
  --action fallback_retry --reason 'Duplicate-request risk accepted.' \
  --server-url https://cayu.example.com

# Newest response-scoped summaries first, using a stable sequence cursor.
cayu session interactions SESSION_ID --limit 50

# Per-model-call and aggregate token/cache usage. Missing prices stay unknown.
cayu session usage SESSION_ID --output json
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
```

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
