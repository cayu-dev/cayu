# SQLite store performance

File-backed `SQLiteSessionStore` instances use four read-only connections alongside
the writer connection. Each operation leases one reader until its worker has
physically finished, including after cancellation. Closing the store waits for
active connection owners. In-memory stores retain one shared connection because
separate `:memory:` connections represent different databases.

Reads started on the bulk read lane lease two further read-only connections,
run on a two-thread executor of their own, and never take one of the four.
The control-plane server puts its aggregate dashboard reads on that lane:
overview counts, session lists that include overviews or usage, usage rollups,
and tool usage. It admits two reads of each kind at a time, after
authentication, so a slow kind never queues a quick one, and a single-session
read, transcript or state read keeps answering while they run.
When the client of such a read disconnects before its response starts, the
server cancels the request, which interrupts the running SQLite statement.
Request diagnostics record it as status 499 (client closed request), counted
with 4xx, not as a server error.
Postgres stores let bulk-lane reads hold at most a quarter of the connection
pool.

SQLite session, task, and checkpoint readers cache successful validation by
exact stored content, including column names, scalar types, and session labels.
Updates invalidate the cached projection even when timestamps do not change.
Malformed data still fails validation on a miss. Every returned value is detached
from the cache, including the first return. These are process-local caches, each
limited to 64 entries and an 8 MiB conservative source-size estimate; decoded
Python objects add memory beyond that estimate. Oversized entries bypass caching.

Event rows are decoded and validated on each read. Copying an Event's payload and
validation stamp into a cache adds work to sequential history scans, especially
when the history exceeds the cache capacity. Fresh event projections already
belong to the caller and do not need another copy for cache ownership.

Checkpoint and event-list reads materialize their values in worker threads after
releasing the read connection. Runtime checkpoint transforms reuse decoded state
and copy only callback-visible roots before calling application code. Callback
output, complete-document limits, schema compatibility, and private authority
projection remain validated. Transaction-dependent transforms and writes retain
connection ownership so their authority checks and mutations stay atomic.

To measure checkpoint and event-row materialization without a model or service:

```sh
uv run python scripts/benchmark_sqlite_store_reads.py
```

This compares validation on every read with warm cached copies of a synthetic
checkpoint, then compares direct event decoding with the former cache strategy
across 100 nested event rows. It is not an end-to-end application benchmark.
Re-profile the actual multi-session workload to determine remaining SQL, copying,
or model bottlenecks.

## Developer dashboard reads at deployment size

`scripts/benchmark_session_overview_reads.py` builds a SQLite store shaped like a
busy deployment, 8,000 sessions over 60 days with five model steps, four tool
calls, a transcript and a checkpoint each, and times Home's 24-hour
`GET /api/sessions/overview/counts` and a 100-row
`GET /api/sessions?include=overview` page through the real server:

```sh
uv run python scripts/benchmark_session_overview_reads.py
uv run python scripts/benchmark_session_overview_reads.py --keep /tmp/sessions.db
uv run python scripts/benchmark_session_overview_reads.py --store /tmp/sessions.db
```

A few template sessions run through the real runtime; their rows are then copied
with new IDs, incarnations, sequences and times, so the store builds in seconds.
`tests/server/test_session_overview_scale.py` runs the same build in CI and fails
if either read slows past its bound. Session listings sort and cut a page by ID
before reading its rows, so a page reads 100 wide rows rather than sorting every
session's invocation and metadata.
