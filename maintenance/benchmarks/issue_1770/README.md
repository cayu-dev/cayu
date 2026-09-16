# Native tool-round checkpoint cost (#1770)

This experiment compares the standalone scripted-provider native app at baseline
`48447ecebd32de274f1777cd665cdf8671a7d293` and implementation commit
`4c61b5aae`. Both versions are installed wheels, with CPython 3.14.2 and identical
runtime dependencies exported from the repository's frozen `uv.lock`. The manifest
records wheel and workload hashes, platform, and installed dependency versions.

## Results

Mean of two unprofiled samples per cell, seconds per complete round:

| Store | Calls | Baseline wall | Candidate wall | Reduction | Baseline CPU | Candidate CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| memory | 1 | 1.425 | 0.864 | 39.4% | 1.394 | 0.855 |
| memory | 10 | 10.150 | 5.990 | 41.0% | 10.093 | 5.868 |
| memory | 20 | 31.211 | 18.712 | 40.0% | 30.994 | 18.608 |
| memory | 30 | 63.402 | 38.806 | 38.8% | 63.137 | 38.641 |
| sqlite | 1 | 1.233 | 0.924 | 25.1% | 1.201 | 0.903 |
| sqlite | 10 | 7.056 | 5.169 | 26.7% | 6.972 | 5.093 |
| sqlite | 20 | 20.713 | 15.145 | 26.9% | 20.504 | 14.990 |
| sqlite | 30 | 42.916 | 30.902 | 28.0% | 42.295 | 30.583 |

All 32 rounds passed the exact result and completion assertions. Raw samples are
in `timings.jsonl`. Setup means (baseline/candidate) were memory: 0.042/0.038s; sqlite: 0.144/0.145s.

## Workload and measurement

`scripts/benchmark_native_tool_round.py` creates a fresh app and session store for
each sample. One provider response requests N parallel-safe echo tools; a second
returns `done`. Each tool returns `Small retained observation`. The harness checks
exact execution and terminal-event counts, one final session completion, two
provider requests, and exactly one correctly identified, successful tool-result
message with the original content per requested call. SQLite samples use a fresh
temporary database and close the store after measurement.

The timed comparison runs 1, 10, 20, and 30 calls, twice on each of memory and
SQLite. Baseline and candidate samples run sequentially. The first seven baseline
samples (memory 1/10/20 twice and 30 once) were retained from the first comparison
pass because the baseline wheel and workload were unchanged. All candidate samples
use the final implementation, including traceback cleanup; remaining pairs reverse
their order on the second repeat. No regression tests or other experiment processes run
concurrently with these comparison samples. Ordinary workstation background
activity is uncontrolled. These are development measurements, not a production
capacity qualification or evidence of live-provider latency.

Setup time covers constructing the store, app, provider, and agent. Round time
covers consuming the entire app event stream. Process CPU time is measured over
the same round. Imports and post-run assertions/SQLite close are outside the
round timer. Profiling and allocation measurements are separate from timings.

Reproduce an individual version using its installed-wheel environment:

```sh
/path/to/version-env/bin/python scripts/benchmark_native_tool_round.py \
  --calls 1 10 20 30 --backends memory sqlite --repeats 2
/path/to/version-env/bin/python scripts/benchmark_native_tool_round.py \
  --calls 10 --backends memory --repeats 1 --profile /tmp/native-round.prof
/path/to/version-env/bin/python scripts/benchmark_native_tool_round.py \
  --calls 10 --backends memory sqlite --repeats 2 --payload-bytes 8192
```

## Attribution and change

The original 10-call memory profile placed the bounded durable JSON walker on
the dominant synchronous CPU path. Full-checkpoint copies occurred in
`pending_action_evidence_round_from_checkpoint`, called by checkpoint preparation
and terminal publication, and in the paired user-input authority reader.

* Pending-action classification previously detached and validated the entire
  checkpoint four times: approval, pending user input, answer intent, and pending
  tool round. It now admits one call-local snapshot and parses all four records
  from it. The newly constructed tool-round model also needs no second deep copy.
* User-input lifecycle reads previously detached and validated the full checkpoint
  twice. They now admit one snapshot and check both records against it.
* Each durable walker node previously rebuilt and sanitized its entire diagnostic
  path character by character. The walker starts at `$` and appends only numeric
  indices; it never puts caller keys into those paths. The new path constructor
  preserves the same 512-character cap and truncation without re-sanitizing its
  already safe input.

Public reader entry points still validate complete caller documents, including
unrelated retained fields, and detach their snapshots. No cross-call cache,
identity shortcut, or public trust switch is introduced. Private parsers are
called only with the immediate caller's owned validated snapshot. Typed model
validation, pause/intent and round identity checks, secret rejection, and terminal
publication/recovery rules remain in place.

The optimization reduces a constant factor; it does not establish linear total
round cost. Checkpoint preparation/publication still reads and validates retained
round data as tool results accumulate. Model reconstruction and other store/event
boundaries remain measurable costs.

## CPU, retained payload, and allocation evidence

The separate 10-call memory cProfile runs recorded the following. Cumulative
times overlap through nested calls and must not be added. Instrumentation changes
relative costs, so use the unprofiled table above for improvement estimates.

| Path | Baseline calls | Candidate calls | Baseline cumulative seconds | Candidate cumulative seconds |
| --- | ---: | ---: | ---: | ---: |
| `copy_durable_record` | 64,883 | 64,140 | 31.347 | 10.790 |
| `_walk_bounded_durable_json` | 74,565 | 73,822 | 35.256 | 13.384 |
| `_durable_child_path` | 2,738,375 | 1,961,517 | 17.039 | 0.722 |
| `_bounded_ascii_label` | 2,738,375 | 0 | 16.038 | 0.000 |
| `pending_action_evidence_round_from_checkpoint` | 191 | 191 | 16.262 | 5.234 |
| `Pydantic SchemaValidator.validate_python` | 32,596 | 32,596 | 14.435 | 12.292 |

`profiles.json` retains the selected function counts/times and the most expensive
callers of full durable JSON copies. Checkpoint store preparation invokes
pending-action classification as checkpoints and tool terminals are published.
Sharing its snapshot reduces traversal of accumulated round data; generated-path
construction removes character-by-character work at every visited node. Typed
Pydantic validation still runs the same number of times (32,596 in both 10-call
profiles). The 1-call profiles retain the fixed invocation/control-plane component
for comparison; they are not an estimate of setup alone.

Process CPU closely tracks wall time in the unprofiled runs. Setup averages are
small relative to total-round costs. Total time per tool, including fixed round
work, still rises between 10 and 30 calls: memory baseline 1.015 to 2.113 seconds,
candidate 0.599 to 1.294 seconds. The store continues to validate/reconstruct
retained state; these samples do not establish an asymptotic complexity class.

To vary retained payload independently of call count, each of ten tool results
received 8,192 additional ASCII bytes. No result truncation was allowed: the
harness verified the entire content in the final provider request. Means of two
unprofiled samples:

| Store | Baseline, +8 KiB/result | Candidate, +8 KiB/result |
| --- | ---: | ---: |
| memory | 11.235s | 6.892s |
| sqlite | 8.282s | 6.326s |

The independent one-call memory tracemalloc runs peaked at 5,784,794 bytes
(baseline) and 5,785,444 bytes (candidate): effectively unchanged. The benefit is
less repeated traversal and transient allocation, not a demonstrated reduction
in peak live memory. Allocation-instrumented elapsed times are excluded from the
main timing comparison. `supplemental.jsonl` retains these samples.

## Regression evidence

The final targeted run passed **651 tests** in 140.34 seconds. `validation.json`
records the exact test command. Ruff lint/format and ty checks passed on changed
Python files. Full repository tests and hosted CI were not run.

The new operation-count tests require exactly one full-checkpoint admission for
each compound reader, independent of unrelated retained-history size. They also
check malformed unrelated input rejection, output/source mutation isolation and
fresh reads after mutation, secret rejection with both ownership modes, and safe
bounded index-only diagnostic paths. Existing suites exercise checkpoint schema,
identity, user input, workload secrets, unknown tool effects, duplicate terminal
publication, cancellation before/after commit, and recovery.

A separate absent-authority probe (`reader_operations.py`) supplies
`{"retained": [{"text": "history"}, ...]}` and counts walker admissions, frame
constructions, and child paths. With 1,000 retained records, each full traversal
allocates 1,002 container frames and 1,002 detached containers. Pending-action
classification reduces these allocations from 4,008 of each to 1,002; lifecycle
reading reduces them from 2,004 to 1,002. These are cumulative operation counts,
not peak resident memory. Run the probe under each installed wheel:

```sh
/path/to/version-env/bin/python maintenance/benchmarks/issue_1770/reader_operations.py version-label
```
