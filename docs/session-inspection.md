# Session inspection

`cayu session` is the supported read-only interface for diagnosing durable
sessions. It uses `SessionStore` contracts, so SQLite and PostgreSQL expose the
same fields without requiring operators to know Cayu table names or JSON paths.

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

Session-inspection CLI schema version `5` extends the bounded
`provider_operation` object with `reconnect_scheduled`,
`reconnect_in_progress`, and `provider_operation_reconciled` states. These and
`provider_operation_in_progress` include only the provider, operation id, and
stream protocol; private recovery metadata is never exposed. Version `5`
retains version `4`'s `synchronous` state and provider-operation object, version
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
