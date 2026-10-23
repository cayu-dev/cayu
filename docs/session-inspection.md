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
