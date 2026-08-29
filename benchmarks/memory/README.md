# Memory retrieval baseline

This directory freezes Cayu's pre-v5.1 retrieval behavior before revision-aware,
cross-source recall changes land. The current runner measures the existing
knowledge source only; it does not claim that knowledge and memory are the same
thing. Memory is the future recall system, while knowledge is one durable,
canonical source inside it.

Run the public corpus from the repository root:

```bash
PYTHONPATH=src python scripts/run_memory_retrieval_baseline.py
```

Use `--output path.json` to retain a new report. The checked report records the
observed in-memory and SQLite keyword baseline, including recall@5, MRR, nDCG@5,
false injection, stale results, authorization leaks (search and typed ID/chunk
reads), citation/source identity, candidate/truncation counts, model-facing byte
and estimated-token overhead, multilingual slices, and p50/p95 search latency.
Latency is environment-sensitive; correctness metrics and selected identities
are the reproducible CI contract.

## Private production-shaped imports

The same bounded `cayu.memory_retrieval_corpus.v1` schema accepts an
`origin` of `external_private`, trajectory identifiers, and turn indexes. Point
`--corpus` at an external JSON file to evaluate long production-shaped
trajectories. Do not copy private text or the resulting report into this public
repository. Loading is local, bounded to 4 MiB and 10,000 entries/cases, and
does not make provider or network calls.

Every result records the corpus revision, backend, search mode, embedding and
reranker identities, and configuration. Add those identities when evaluating a
semantic or reranked implementation; `null` means the current keyword-only
baseline did not use one.

## Cross-source recall baseline

`recall-corpus-v2.json` and `recall-baseline-results-v2.json` exercise the
bounded `RecallEngine` and calibrated automatic-recall admission across canonical
knowledge and explicitly selected transcripts. Run the checked in-memory/SQLite
matrix with:

```bash
PYTHONPATH=src python scripts/run_recall_baseline.py
```

The corpus covers knowledge-only and transcript-only matches, duplicate
cross-source content, current revision selection, authorization isolation,
malicious recalled text, Spanish and Russian text, and short follow-ups resolved
from bounded recent context. It checks recall and exact locators alongside
injected/offer/silent precision, false/stale/unauthorized injection, source
diversity, deterministic clipping, serialized contribution overhead, admission
latency, required-source failure, incomplete coverage, and backend identity
parity. Built-in stores in this corpus do not provide semantic embeddings, so the
semantic lane is deliberately recorded as partial instead of pretending that
empty results are complete.

`--corpus` accepts the same bounded public/private origin contract and
`--output` retains a report. Keep private production-shaped corpora and reports
outside this repository.

## Memory evidence overhead

`memory-evidence-performance-v1.json` is the hermetic 50-pair baseline for the durable
receipt/exposure and public attribution path. Regenerate it, or check the documented
regression ceilings without making provider calls, with:

```bash
PYTHONPATH=src python scripts/run_memory_evidence_performance.py \
  --output benchmarks/memory/memory-evidence-performance-v1.json \
  --check
```

The runner measures evidence-document preparation, in-memory/SQLite persistence,
steady-state SQLite bytes after close, a current-runtime zero-record control,
memory-bearing projection, and serialized report bytes. The zero-record control includes
the always-on attribution path and is explicitly not labeled as historical pre-feature
code. Run the hermetic command above to remeasure current code before accepting
performance-sensitive changes. Provider latency is deliberately absent. See
[Memory foundation contracts](../../docs/memory-foundation.md#bounded-public-memory-attribution)
for the fixed ceilings and interpretation. SQLite persistence enforces a sustained
median ceiling separately from a wider emergency p95 cap because hosted-runner fsync
scheduling is not a stable product-tail signal.

## Checkpoint-aware recall overhead

`checkpoint-recall-performance-v1.json` is the provider-free 50-sample baseline
for full-index, one-revision delta, maximum 250-reference delta (249 changes
plus one ready index projection), and explicit no-work processing over 500
existing records. Regenerate it, or check its fixed ceilings, with:

```bash
PYTHONPATH=src python scripts/run_checkpoint_recall_performance.py \
  --output benchmarks/memory/checkpoint-recall-performance-v1.json \
  --check
```

The benchmark measures the complete processor contract, including accessible
frontier reads, exact-revision restriction, recall diagnostics, fusion, and
checkpoint proposal construction. It excludes entry creation and provider
latency. In-memory and SQLite provide the credential-free timing matrix;
PostgreSQL and pgvector run behavioral parity in the integration suite.

## Agent work-context and checkpoint overhead

`agent-work-context-performance-v1.json` is the hermetic 50-sample baseline for
the independent durable agent work-context store. Regenerate it, or check its
fixed regression ceilings without provider calls, with:

```bash
PYTHONPATH=src python scripts/run_agent_work_context_performance.py \
  --output benchmarks/memory/agent-work-context-performance-v1.json \
  --check
```

The zero-record lane repeatedly constructs and closes a current-schema empty
store; it is not a historical pre-feature binary comparison. The populated
lanes measure indexed current reads, semantic revision appends through
compare-and-swap, checkpoint advances through compare-and-swap, and incremental
SQLite bytes per durable revision/receipt/checkpoint record. In-memory and
SQLite provide the credential-free timing matrix; PostgreSQL runs the same
behavioral conformance contract in its integration suite.

## Staged-recall delivery overhead

`recall-delivery-performance-v1.json` is the hermetic 50-stage baseline for the
atomic processing-result/checkpoint handoff and its durable worker lifecycle.
Regenerate it, or check every fixed p50/p95 ceiling without provider calls, with:

```bash
PYTHONPATH=src python scripts/run_recall_delivery_performance.py \
  --output benchmarks/memory/recall-delivery-performance-v1.json \
  --check
```

The populated lanes measure atomic stage/checkpoint commitment, oldest-pending
claim, and acknowledgement. The empty lane runs after all 50 deliveries are
acknowledged, so it measures the production-shaped indexed no-pending lookup
without treating pre-feature code as a control. In-memory and SQLite provide the
credential-free timing matrix; PostgreSQL runs shared conformance, concurrency,
reopen, migration, and cancellation rollback tests.

## Revision-bound knowledge-relation overhead

`knowledge-relation-performance-v1.json` is the hermetic baseline for 50 matching
relations surrounded by 5,000 unrelated relations. Regenerate it, or check the
exact-revision lineage regression ceilings without provider calls, with:

```bash
PYTHONPATH=src python scripts/run_knowledge_relation_performance.py \
  --output benchmarks/memory/knowledge-relation-performance-v1.json \
  --check
```

The current-runtime zero-relation control publishes the same 5,053 entries without
relations; it is not a historical pre-feature comparison. The populated lane
measures canonical preparation, atomic ten-relation batches, bounded 50-result
queries, isolated-endpoint lookup across the unrelated background, and steady-state
SQLite bytes after close. Run the command above to remeasure the workload.

## Reviewed knowledge-maintenance overhead

`knowledge-maintenance-performance-v1.json` is the provider-free baseline for 20
approved decisions with 20 exact predecessor revisions each. Regenerate it, or
check the same fixed regression ceilings, with:

```bash
PYTHONPATH=src python scripts/run_knowledge_maintenance_performance.py \
  --output benchmarks/memory/knowledge-maintenance-performance-v1.json \
  --check
```

The zero-decision control creates the same 420 current-runtime entries without
applying maintenance; it is not a historical binary comparison. The applied lane
measures canonical proposal/decision preparation, atomic activation plus archival
and relation publication, exact replay, receipt loading, and steady-state SQLite
bytes. It makes no provider calls.

## Reviewed knowledge-maintenance reference evaluation

`knowledge-maintenance-corpus-v1.json` and
`knowledge-maintenance-evaluation-results-v1.json` exercise the full reviewed
maintenance path rather than another isolated component benchmark. Run the checked
in-memory/SQLite matrix with:

```bash
PYTHONPATH=src python scripts/run_knowledge_maintenance_evaluation.py
```

The six workflows cover duplicate merge, authoritative supersession, an unresolved
contradiction that must fail before publication, a proposal made stale before review,
explicit reviewer rejection, and normal recall plus historical lineage after approval.
The report records routing precision/recall, exact claim-to-source retention, durable
source-revision evidence retention after the terminal review attempt, unsafe acceptance,
exact lifecycle preservation, lineage correctness, model-call count, and end-to-end
latency. PostgreSQL runs the same corpus in its store parity test. A case may contain up
to 100 entries, of which at most 50 can be maintenance sources; recall and historical
inspection use that same full source bound. Each recall query is limited to the recall
primitive's executable 8,192-byte boundary, so every accepted corpus can run without a
later input-validation failure.

This is deliberately an orchestration and safety evaluation. Its fixture planner and
independent fixture evaluator are deterministic and configured with a zero-model-call
budget, so a green result proves that Cayu preserves and enforces a known plan across
its boundaries. It does not claim that an arbitrary model will discover duplicates,
resolve truth, or write a high-quality replacement. Evaluate provider-backed components
separately with a private production-shaped corpus, and do not commit that corpus or its
report. The loader accepts the same schema with `origin="external_private"`, bounded to
4 MiB of canonical corpus input, 64 KiB of recorded configuration, and a 16 MiB result.
Lifecycle, evidence, and lineage scores bind returned store results to the exact requested
entry revisions and relation identities rather than accepting matching logical IDs alone.
