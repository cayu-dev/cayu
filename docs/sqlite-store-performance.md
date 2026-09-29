# SQLite store performance

File-backed `SQLiteSessionStore` instances use four read-only connections alongside
the writer connection. Each operation leases one reader until its worker has
physically finished, including after cancellation. Closing the store waits for
active connection owners. In-memory stores retain one shared connection because
separate `:memory:` connections represent different databases.

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
