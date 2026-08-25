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
code. A dedicated required CI lane reruns the workload without coverage instrumentation,
so an unchanged artifact cannot conceal a regression and timing is not distorted by the
ordinary test harness. Provider latency is deliberately absent. See
[Memory foundation contracts](../../docs/memory-foundation.md#bounded-public-memory-attribution)
for the fixed ceilings and interpretation. SQLite persistence enforces a sustained
median ceiling separately from a wider emergency p95 cap because hosted-runner fsync
scheduling is not a stable product-tail signal.
