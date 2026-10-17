# Distribution size

This page measures how big the Cayu wheel is and how much of it a minimal
application loads. It is input to a pending maintainer decision
about shrinking the core distribution. Nothing has moved yet, and there is no size
budget. The inventory and options below describe the choices, not a plan.

## Run the report

`scripts/report_distribution_size.py` prints the report. It never fails because
of size; it exits non-zero only when it cannot measure.

```sh
# Source tree, closure measured with the current interpreter
uv run python scripts/report_distribution_size.py

# A built wheel, closure measured in an environment where that wheel is installed
uv build --wheel --out-dir /tmp/cayu-dist
uv venv /tmp/cayu-wheel-env
uv pip install --python /tmp/cayu-wheel-env/bin/python /tmp/cayu-dist/cayu-*.whl
/tmp/cayu-wheel-env/bin/python -I scripts/report_distribution_size.py \
  --wheel /tmp/cayu-dist/cayu-*.whl

# Machine-readable output, or skip the closure probe
uv run python scripts/report_distribution_size.py --json
uv run python scripts/report_distribution_size.py --no-closure
```

`--python PATH` measures the closure with another interpreter. The closure is
always measured from wherever that interpreter imports `cayu`; the report prints
that location. With `--wheel`, the inventory is read from the archive, so install
the same wheel if the closure should match it.

Package CI runs the report on the built wheel as a report-only step in the
`core` package group ("Report distribution size and core import closure" in
`scripts/package_ci_steps.yml`). That lane runs on release tags, manual runs and
pull requests that touch packaging inputs, including the script itself. It adds
a few seconds to that job.

## What it measures

- **Wheel size:** the size of the `.whl` file (`--wheel` only).
- **Inventory:** every file under `cayu/` in the wheel or source package, grouped
  by top-level subpackage. `<root modules>` holds modules directly under `cayu/`,
  such as `applications.py`. Python lines are physical lines of `.py` files, the
  same count `wc -l` gives. Bytes are uncompressed. The source-tree inventory skips
  `__pycache__`, but it includes any untracked files in the package directory, so
  use a wheel for published numbers.
- **Core import closure:** a separate `python -I` process with `CAYU_*`
  environment variables removed records which `cayu` modules are in
  `sys.modules` after each stage:
  1. `import cayu`;
  2. **minimal app**: import `CayuApp`, `AgentSpec` and friends, construct
     `CayuApp(session_store=SQLiteSessionStore(<temp file>))`, register a stub
     `ModelProvider` subclass as the default provider and register one agent;
  3. **one stub turn**: run one turn of that app to completion with
     `run_to_completion`.

  The stub provider yields a text delta and a completion, so the probe needs no
  network access, credentials or provider SDK. Lines are counted from the loaded
  module files. The "minimal app" stage is the closure number described in the
  issue; the third stage shows what a turn adds.

## Current numbers

Measured on `main` at `968713221` (version 0.11.4) from a wheel built with
`uv build --wheel` and installed into a fresh Python 3.11 environment.

| Measure | Value |
| --- | --- |
| Wheel file | 9.6 MB (9,566,396 bytes) |
| Unpacked `cayu/` files | 1,455 files, 49.2 MB |
| Python source | 1,363 files, 1,036,779 lines |
| `import cayu` | 3 modules, 122 lines |
| Minimal app closure | 784 modules, 587,402 lines (57% of shipped Python) |
| After one stub turn | 809 modules, 609,869 lines (59%) |
| Third-party packages loaded by the minimal app | 15 top-level modules (pydantic, httpx, jsonschema, cryptography and their dependencies) |

Python is 8.1 MB of the 9.4 MB of compressed wheel content. The next largest
item is the editable dashboard source archive in `cayu/data` (3.9 MB unpacked,
0.6 MB compressed). The issue's 1,049,190-line figure came from `wc -l` over
every installed file; the counts here cover `.py` files only (the wheel also has
about 7,000 lines of `.pyi` stubs and 6,500 lines of Markdown guides).

Per subpackage, with the lines loaded after one stub turn:

| Subpackage | Python files | Python lines | Share | Loaded lines |
| --- | ---: | ---: | ---: | ---: |
| `runtime` | 299 | 211,226 | 20.4% | 197,107 |
| `storage` | 143 | 139,386 | 13.4% | 50,310 |
| `evals` | 84 | 82,326 | 7.9% | 3,190 |
| `sessions` | 120 | 72,147 | 7.0% | 68,675 |
| `tools` | 61 | 68,829 | 6.6% | 44,840 |
| `cli` | 45 | 54,458 | 5.3% | 0 |
| `<root modules>` | 61 | 48,003 | 4.6% | 33,447 |
| `collaboration` | 154 | 43,360 | 4.2% | 25,854 |
| `tasks` | 42 | 33,963 | 3.3% | 26,766 |
| `server` | 38 | 29,691 | 2.9% | 0 |
| `providers` | 36 | 26,793 | 2.6% | 24,640 |
| `workspaces` | 23 | 25,353 | 2.4% | 12,229 |
| `knowledge` | 30 | 22,381 | 2.2% | 6,582 |
| `egress` | 25 | 21,044 | 2.0% | 13,692 |
| `memory` | 12 | 20,938 | 2.0% | 12,998 |
| `runners` | 23 | 18,210 | 1.8% | 9,334 |
| `environments` | 18 | 17,248 | 1.7% | 16,239 |
| `mcp` | 15 | 14,595 | 1.4% | 14,595 |
| `budgets` | 14 | 13,214 | 1.3% | 11,573 |
| `verification` | 16 | 13,015 | 1.3% | 10,800 |
| `snapshots` | 5 | 12,181 | 1.2% | 0 |
| `artifacts` | 25 | 11,714 | 1.1% | 8,467 |
| `context` | 7 | 11,058 | 1.1% | 11,058 |
| everything else | 67 | 25,646 | 2.5% | 7,473 |

### What the closure shows

- `import cayu` is lazy. The cost arrives with the first real import such as
  `from cayu import CayuApp`, which pulls in most of `runtime`, `sessions`,
  `tasks`, `mcp`, `context` and `providers`.
- `cli`, `server`, `snapshots` and `delivery` are not loaded by a minimal app.
  Only three `evals` modules load (the package, its export table and
  `evals/testing.py`).
- A minimal SQLite app with no browser, Docker, collaboration or PostgreSQL
  configured still loads about 20,000 lines of browser-control modules, 8,400
  lines of Docker runner and coding-environment modules, 26,000 lines of
  `collaboration`, and three PostgreSQL support modules (2,300 lines). These are
  eager imports from core modules rather than research code.
- None of the research runners listed below is in the closure. Moving them out
  would shrink the wheel and what an agent sees in `site-packages`, but would not
  change the closure. Shrinking the closure means making optional subsystems
  load only when an application configures them.

## Research and benchmark runner inventory

Searched with `grep` across `src/cayu` (including relative imports, the lazy
export tables in `_exports.py`, `__init__.pyi` stubs and string module targets),
`src/cayu/guides`, the scaffold modules, `README.md`, `examples/`, `scripts/` and
`docs/`. No core runtime, server or scaffold module imports any of these, and no
installed guide or generated project mentions them. The question for each group
is how much public surface moving it would change.

### Group A: internal runners with no public import path (3,502 lines)

All under `cayu/evals/internal/`. They are not exported from `cayu.evals` and no
CLI command registers them.

| Module | Lines | Used by |
| --- | ---: | --- |
| `internal/browser_acceptance.py` | 2,131 | `scripts/run_browser_acceptance.py`; `docs/evals.md` |
| `internal/runtime_acceptance.py` | 664 | `scripts/nightly_verification.py`; `docs/runtime-qualification.md` runs it from an installed wheel with `cayu eval run cayu.evals.internal.runtime_acceptance:build` |
| `internal/browser_acceptance_operator.py` | 376 | `examples/aws/lambda_microvm_browser_control_live.py`, `scripts/run_browser_acceptance.py` |
| `internal/browser_acceptance_operator_server.py` | 225 | `examples/browser_acceptance/local_authenticated.py` |
| `internal/_browser_page_fault_guest.py` | 105 | read as text by `internal/browser_acceptance.py` |
| `internal/__init__.py` | 1 | |

The documented runtime-qualification procedure depends on
`runtime_acceptance` being installed, so moving it means changing that procedure
too.

### Group B1: research runners exported from `cayu.evals` (17,815 lines)

Each module's names are in `cayu.evals` `PUBLIC_NAMES` and `__init__.pyi`.
Nothing outside the export table, tests, docs, scripts and other research
modules imports them. Moving them removes public names from `cayu.evals`, which
needs a deprecation or migration decision.

| Module | Lines | Public names | Notes |
| --- | ---: | ---: | --- |
| `evals/browser_acceptance.py` | 5,210 | 48 | Documented in `docs/evals.md` "Browser acceptance profiles" and `examples/browser_acceptance/` |
| `evals/browser_acceptance_manifests.py` | 1,254 | 6 | |
| `evals/browser_acceptance_fixture.py` | 613 | 2 | `examples/browser_fetch/visual_live.py` uses a private helper from it |
| `evals/browser_acceptance_authentication.py` | 133 | 1 | |
| `evals/browser_acceptance_authenticated.py` | 119 | 1 | |
| `evals/external_private_memory_ablation.py` | 4,655 | 36 | Named in the issue. `docs/evals.md` describes it as something applications can run. Imports the CLI helper `cayu.cli._guarded_tree_publication` |
| `evals/knowledge_maintenance.py` | 2,628 | 13 | Provider-free knowledge-maintenance evaluation. No tests, docs or examples; classification is the least certain in this group |
| `evals/causal_memory_campaign.py` | 1,650 | 8 | No docs |
| `evals/recall_baseline.py` | 959 | 13 | No tests, docs or examples |
| `evals/memory_baseline.py` | 594 | 13 | Used only by `recall_baseline.py`. No tests, docs or examples |

`scripts/smoke_built_wheel_basic_imports.py` imports
`BrowserAcceptanceFixtureV1`, `deterministic_browser_acceptance_manifest` and
`run_causal_memory_reference_campaign` from the built wheel, so it would change
with this group.

### Group B2: benchmark campaigns behind `cayu eval` (3,041 lines)

`cayu eval run --package`, `cayu eval resume|retry|cancel|rescore`, `cayu eval
package ...` and campaign-directory `report`/`compare` are registered through
`cli/_benchmark_campaigns.py`. Building the `cayu eval` parser imports it, and it
imports `benchmark_campaign`, `benchmark_inspection` and `benchmark_package` at
module level, so this stack loads on every `cayu eval` command. Documented in
[Benchmark packages](benchmark-packages.md) and
[Benchmark campaigns](benchmark-campaigns.md).

| Module | Lines | Notes |
| --- | ---: | --- |
| `evals/benchmark_inspection.py` | 735 | |
| `evals/benchmark_campaign.py` | 699 | Lazily imports `cayu.server.evals_registry` and `cayu.server.evals_worker` |
| `cli/_benchmark_campaigns.py` | 520 | Named in the issue |
| `evals/benchmark_package.py` | 359 | Also exports 9 public names from `cayu.evals` |
| `evals/benchmark_synthetic.py` | 359 | Backs `cayu eval package init` |
| `evals/benchmark_rescore.py` | 151 | |
| `evals/benchmark_retry.py` | 143 | |
| `evals/benchmark_comparison.py` | 75 | |

Moving this group changes a CLI surface, not only Python imports.

### Total, and what stays

Groups A, B1 and B2 come to **24,358 lines, about 2.3% of shipped Python**. The
eval framework that applications use stays in core under any option: `runner`,
`execution`, `scenario*`, `suite_*`, `corpus`, `assertions`, `judges`, `models`,
`store`, `results`, `reporting`, `trajectory`, `evidence`, `testing`,
`workflow_*`, `runtime_replay`, `calibration` and `external`. Some modules with
research-sounding names are also core: `capacity.py` (used by the eval store,
runner and server), `memory_attribution.py` (used by `memory/execution.py`),
`memory_reporting.py` (backs `cayu eval memory-report` and server routes) and
`external_container.py` (an exported reference adapter for external eval
targets).

## Options

1. **Move research and benchmark runners out of the wheel**, into repo-only
   tooling or a separate `cayu-evals-lab` distribution, keeping the eval
   framework applications use in core. The inventory above is the candidate
   list: about 24,000 lines, with group A the cheapest to move.
2. **Split optional subsystems into separate distributions** in the `cayu`
   namespace (for example `cayu-server`, `cayu-cloud`, `cayu-collaboration`,
   `cayu-browser`), installed through the existing extras so
   `pip install "cayu[server]"` keeps working. This is where the large numbers
   are, but the closure findings above show the core already imports browser,
   Docker and collaboration modules eagerly, so a split needs those imports made
   lazy first.
3. **Measure and publish the core import closure.** The script in this page
   produces that number (587,402 lines across 784 modules today).
4. **Budget it in CI.** The report already runs in package CI. A budget would add
   a threshold on the wheel size, total lines or closure lines and fail on
   unreviewed growth.

## Decisions needed

- [ ] Which subsystems leave the core wheel, and in what form: repo-only or a
  separate distribution. For the runners: whether groups B1 and B2 may drop
  public `cayu.evals` names and `cayu eval` subcommands, and what replaces the
  installed `runtime_acceptance` qualification command.
- [ ] A size target for the core distribution, which measure it applies to
  (wheel bytes, total Python lines or closure lines), and when the CI step starts
  enforcing it.
- [ ] A migration plan that keeps existing extras-based installs working.
