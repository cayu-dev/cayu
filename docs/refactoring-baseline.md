# Structural measurements

Use `scripts/size_report.py` to inspect one committed source revision. It reports
file sizes, public-surface counts, static dependencies and selected call sites.
These measurements support code review; they do not establish complexity,
correctness or runtime performance on their own.

## Generate and reproduce reports

Run from the repository root with Python 3.11 or newer and Git:

```sh
python3 scripts/size_report.py --revision HEAD > /tmp/cayu-size-report.json
python3 scripts/size_report.py --check scripts/size_baseline.json
```

The committed snapshot records public revision
[`4e51711938591f99afa102d984b4133ff2e8b56c`](https://github.com/cayu-dev/cayu/commit/4e51711938591f99afa102d984b4133ff2e8b56c).
The checkout must contain that commit. If it is missing, including in a shallow
clone, fetch it explicitly before checking the snapshot:

```sh
git fetch --no-tags https://github.com/cayu-dev/cayu.git 4e51711938591f99afa102d984b4133ff2e8b56c
```

`--check` remeasures the saved revision and compares the complete report. It
checks reproducibility; it does not enforce limits against today's source.
Use `--repo /path/to/checkout` to measure another Git repository. The script
does not fetch commits or contact a network service.

The script reads regular file blobs from the resolved commit without importing
the measured source. Local edits, untracked files, symlinks and bytecode are
excluded. Git archive export rules and checkout line-ending conversion do not
affect the result. Output contains no timestamps or machine-specific paths.

## Compare revisions

Use the same version of the measurement script and the same metric definitions
for both revisions. Schema version 2 records every source Python file in
`source_module_lines` and every test file at or above 10,000 lines in
`test_modules_at_least_10000_lines`. Smaller replacement source modules remain
visible in the inventory.

Source totals include all Python files under `src/cayu/`, including packaged
examples. Compare matching file populations. Record the measured commit with
each result and review the relevant owners and callers when identifiers change.
Counts do not distinguish deleted code from code moved to another file.

An intentional snapshot refresh records a new revision. Changing metric
definitions requires a report schema-version change so unlike measurements are
not silently compared.

## Metric definitions

| Metric | Definition |
| --- | --- |
| Source/test lines | Physical lines, including blanks and comments, in tracked `.py` files under `src/cayu/` or `tests/`. Line endings are LF, CRLF or CR; Unicode separators inside strings do not end a line. A final line without a newline counts. |
| Export declaration lines | `_exports.py` and `__init__.pyi` lines under `src/cayu/`. These overlap source totals where the file is Python; do not add them to total LOC. |
| Root public names | Distinct strings in one standalone literal root `PUBLIC_NAMES` list or tuple. Other references, aliases, rebinding, dynamic declarations, or duplicate names fail measurement. |
| SessionStore methods | Unique directly declared public method names, including properties; overloads count once. |
| Recovery constructor parameters | Named positional and keyword-only arguments, excluding `self`; defaults do not affect the count. |
| Runtime dependencies | Distinct directly imported runtime modules, separately for all `storage/` and all `sessions/`. Includes local and type-checking imports; resolves relative and submodule imports. Details identify each importer. |
| Private test imports | Import statements targeting a `cayu` module with an underscore-prefixed path component. Importing a private symbol from a public module does not count. |
| Provider subclasses | Direct subclasses whose syntactic base resolves to a supported public `ModelProvider` import, outside `tests/support/`. Indirect inheritance is not counted. |
| App forwarding candidates | Private `CayuApp` methods containing exactly one syntactic call, to `self._session_engine.*`. This is a reproducible candidate set, not proof that the whole method can be deleted. |
| Round ownership calls | Source AST call sites with the recorded callee names, including aliases. Locations are an inspection inventory, not proof that similarly named methods have identical behavior. |
| Reader text occurrences | All textual occurrences of `pending_tool_round_from_checkpoint` in source Python, including imports, definitions, and comments. This is distinct from call count. |

Import and class inspection is static: it does not resolve dynamic imports,
indirect re-exports, or scope-dependent rebinding. Owner and callee definitions
must be reviewed when names change. The tool does not assign a duplication score;
the publication caller inventory exposes where protocol ownership must be checked.

## Test durations and performance

`committed_duration_snapshot` records the entry count and sum of the selected
revision's `.test_durations`. Entries can be stale and span multiple CI lanes.
The sum is neither the current collected-test count, CPU time, PR wall time nor
a runtime-speed measurement. Invalid, negative, non-finite, boolean and duplicate
timing values fail measurement.

Use the selectors in `scripts/run_ci.py` and collected test markers when
measuring a specific CI lane. Marker assignment can occur during collection;
file paths alone cannot establish lane membership.

Measure runtime performance separately with fixed workloads, environment and
dependency versions. Report observed operation counts, memory use and latency
alongside structural changes.

### Scripted tool-round workload

Run the same [benchmark script](../scripts/benchmark_tool_round.py) and dependency
versions from each source checkout:

```sh
PYTHONPATH=src uv run python scripts/benchmark_tool_round.py --output /tmp/tool-round.json
```

The default matrix uses 2, 16 and 64 tool calls; 0, 128 and 512 history messages;
and 1 or 8 concurrent sessions. Each batch uses a fresh in-memory store and one
tool round followed by a final model response per session. Results contain the
source revision, script digest, environment, three uninstrumented latency samples
and their median. Setup and one import-warmup batch are outside the timing window.
The larger cases can take substantially longer than the smaller ones.
`--output` saves the report atomically after each case. An interrupted run retains
its completed cases with `finished: false`. Optional `--case-timeout 120` runs
each case in an owned worker process with a 120-second budget covering startup,
warmup, all samples and observation. The memory-only worker is terminated when
that budget expires. Unfinished cases retain any completed
samples with `status: timed_out`, and the report has `complete: false`. They do not
produce a median or operation-count result. Without that option, cases have no time
budget. Use identical budgets when comparing revisions.

A separate pass counts public async store calls, including calls within backend
methods, and checkpoint admissions through the durable-JSON walker. It records
peak staged payload bytes from the public publication metrics. This measures
staged payloads, not total retained memory. Optional `--trace-python-allocations`
also records peak traced Python allocations during that pass; tracing can be
expensive. Provider and database I/O are outside this workload.

Use `--calls-per-round 2 --history-messages 0 --sessions 1 --samples 1` for a
small smoke run. Compare matching cases under comparable machine load; the script
reports measurements and does not enforce a performance threshold.

## Leased adapter workload

Use the development environment and existing characterization fixtures to
compare fresh public verification and result-resolution operations:

```sh
uv run python scripts/benchmark_leased_adapters.py --repo . --samples 15 --output /tmp/leased-after.json
uv run python scripts/benchmark_leased_adapters.py --repo /path/to/base-checkout --samples 15 --output /tmp/leased-before.json
```

Run the same script and Python environment in a separate process for each
revision. The matrix uses 1, 8, and 32 concurrent operations with fresh in-memory
stores and application instances. Setup and a warmup batch precede each timing
window. A separate pass counts public async store calls, including internal
calls; fixture setup is excluded from those counts. Temporary class wrappers
await the exact original store implementations and are restored after the pass.
Reports retain raw latency samples, medians, the target revision, runtime source
digest, script digest, Python version, and call counts. This workload measures
fresh in-memory completion dispatch and publication.

Use `--isolate-gc` for an additional diagnostic comparison. It collects cyclic
garbage before each batch, disables cyclic collection during the timing window,
and restores its prior enabled state afterward. Reports record this option and
the initial collector state. Default measurements include ambient garbage
collection; isolated measurements help assess sensitivity to its scheduling
and exclude cyclic collection cost from the measured window.
