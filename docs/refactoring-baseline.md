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
