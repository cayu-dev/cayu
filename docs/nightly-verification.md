# Nightly verification

**Status:** capability-report runner exists; scheduled GitHub Actions automation
is still future work and is tracked in #174.

Run the current report with:

```bash
uv run python scripts/nightly_verification.py \
  --json nightly-verification.json \
  --markdown nightly-verification.md
```

The report is the product. A pass count without a capability map is not enough.

## Ground Rules

1. **Every verified claim names a check.** A capability is verified only if a
   command or test exercised the relevant path and passed.
2. **Skips are first-class output.** A missing daemon, API key, extension, CLI,
   optional package, or disposable database must appear in the report with the
   exact reason.
3. **Hermetic and live coverage are different.** Fake-provider tests can verify
   Cayu runtime contracts. They cannot verify vendor APIs, sandbox CLIs, or
   database extensions.
4. **Smoke is not verified behavior.** A live example that exits 0 is useful,
   but it is reported as `smoke` until it checks expected output, files,
   artifacts, events, or usage.
5. **Costs and prerequisites are separate.** "No LLM spend" does not mean "no
   external service." E2B needs `E2B_API_KEY` even when no model is called.

## Runner

`scripts/nightly_verification.py` emits both machine-readable JSON and
human-readable Markdown. It can run all known checks, a selected subset, or only
list the check IDs:

```bash
uv run python scripts/nightly_verification.py --list
uv run python scripts/nightly_verification.py --check core-pytest --strict
uv run python scripts/nightly_verification.py --check internal-evals-hermetic --strict
uv run python scripts/nightly_verification.py --check sigkill-recovery --strict
uv run python scripts/nightly_verification.py --check docker-runner --strict
```

`--strict` exits nonzero when any selected check reports `failed`, `skipped`,
`smoke`, or `unclaimed`. Omit `--strict` for exploratory capability maps where
missing credentials or known holes are expected.

While checks run, the script logs per-check start, status, and duration to
stderr. Child command output stays captured for failure reasons and reports.
Each child check has a default 30-minute timeout; a timeout is reported as
`failed` with `timed_out: true` in the evidence.

Each result has this JSON shape:

```json
{
  "capability": "DockerRunner real container exec and timeout cleanup",
  "check_id": "docker-runner",
  "lane": "docker",
  "status": "verified",
  "command": "uv run pytest tests/runners/test_docker_live.py -q",
  "prerequisites": ["Docker daemon"],
  "reason": null,
  "evidence": {
    "returncode": 0,
    "passed": 1
  }
}
```

Allowed statuses:

| status | meaning |
| --- | --- |
| `hermetic` | verified with deterministic local dependencies only: no live provider credentials, network access, external sandbox quota, or LLM judge |
| `verified` | exercised against the target dependency and asserted behavior |
| `smoke` | exercised a live dependency, but expected behavior is not fully asserted |
| `skipped` | did not run because a prerequisite was missing |
| `failed` | ran and failed |
| `unclaimed` | no current check covers the capability |

## Verification Lanes

These lanes are split by physical prerequisites, not by whether they run locally
or in CI.

| Lane | Needs | Spend | Current check |
| --- | --- | ---: | --- |
| Python baseline | Python dev deps; provider keys are unset by the runner | $0 | `core-pytest`, `internal-evals-hermetic` |
| Process-death recovery | POSIX `SIGKILL`; deterministic SQLite stores and scripted providers | $0 | `sigkill-recovery` |
| Postgres integration | Docker/testcontainers or `CAYU_TEST_POSTGRES_DSN` | $0 | `postgres-required` |
| Docker runner live | Docker daemon | $0 | `docker-runner`, `docker-live-*` |
| `sbx` runner live | `sbx` CLI/runtime | $0 | `sbx-live-*` |
| microsandbox live | `cayu[microsandbox]` runtime support | $0 | `microsandbox-live-*` |
| E2B live | `cayu[e2b]`, `E2B_API_KEY` | E2B quota | `e2b-live-*` |
| Chat Completions live | `GEMINI_API_KEY` | provider-dependent | `gemini-eval`, `chat-completions-contract` |
| OpenAI/Anthropic contracts | provider API key; file readers for artifact files | provider-dependent | `context-counting-live`, `artifact-file-live` |
| OpenAI/Anthropic smoke | provider API key | provider-dependent | `*-live` provider-smoke checks |
| Dashboard browser | `cayu[browser]` and installed Chromium | $0 | `dashboard-behavior` |

The CI workflow also runs dashboard lint/typecheck, generated-client drift,
package build, and packaged-asset status checks. It still does not run dashboard
browser behavior tests.

## Current Coverage Map

The runner's `--list` output is the source of truth for exact check IDs. At a
high level:

| Capability | Status class | Check |
| --- | --- | --- |
| runtime loop, model steps, tool rounds, approvals, interrupts, subagents, evals, stores, server, local runner | verified baseline | `core-pytest` |
| first-party tool, workspace, context, knowledge, subagent, usage, and budget eval workflows | hermetic | `internal-evals-hermetic` |
| Postgres stores, migrations, pgvector, real dispatch claim path | verified when Postgres is available | `postgres-required` |
| real Docker container exec, timeout cleanup, and sync binding | verified when Docker is available | `docker-runner`, `docker-live-exec`, `docker-live-sync` |
| real `sbx` command cleanup and sync binding | verified when `sbx` is available | `sbx-live-exec`, `sbx-live-sync` |
| real microsandbox runner/workspace/runtime/sync binding | verified when microsandbox is available | `microsandbox-live-*` |
| real E2B runner/workspace/sync binding | verified when E2B is available | `e2b-live-*` |
| Gemini Chat Completions eval path | verified when `GEMINI_API_KEY` is present | `gemini-eval` |
| Chat Completions tool-call and structured-output contract | verified when `GEMINI_API_KEY` is present | `chat-completions-contract` |
| OpenAI/Anthropic artifact-file and context-counting contracts | verified when the selected provider key is present | `artifact-file-live`, `context-counting-live` |
| OpenAI/Anthropic knowledge, subagent, context-pressure, and structured-output demos | smoke | remaining provider-smoke checks |
| real `SIGKILL` recovery for tool rounds, approvals, background-child linkage, and SQLite task claims | verified on POSIX | `sigkill-recovery` |
| real `SIGKILL` recovery for Postgres task claim/attachment | verified when Postgres is available | `postgres-required` |
| packaged dashboard sessions list, session detail, and event detail | verified when Playwright Chromium is installed | `dashboard-behavior` |
| budgets under real provider spend | verified when `OPENAI_API_KEY` is present | `real-spend-budgets` |

Do not update this document with exact pass counts. Counts move as tests are
added and dependencies change; the generated report records current counts.

## Live Examples

There are 23 `examples/*_live.py` files:

| prerequisite | examples |
| --- | --- |
| Docker | `docker_interrupt_live.py`, `docker_sync_binding_live.py` |
| `sbx` | `sbx_interrupt_live.py`, `sbx_sync_binding_live.py` |
| microsandbox | `microsandbox_runner_live.py`, `microsandbox_runtime_live.py`, `microsandbox_workspace_live.py`, `microsandbox_sync_binding_live.py` |
| E2B key | `e2b_runner_live.py`, `e2b_workspace_live.py`, `e2b_sync_binding_live.py` |
| Gemini key | `chat_completions_contract_live.py` |
| Playwright Chromium | `dashboard_behavior_live.py` |
| OpenAI or Anthropic key | `structured_output_live.py`, `subagent_live.py`, `subagent_parallel_live.py`, `artifact_file_live.py`, `context_counting_live.py`, `context_pressure_calibration_live.py`, `knowledge_recall_live.py`, `knowledge_recall_many_live.py` |
| OpenAI key | `knowledge_embedding_live.py`, `real_spend_budget_live.py` |

The deterministic runner examples use `_live_checks.py` and raise on wrong
outputs, missing cleanup artifacts, missing files, or missing model/tool rounds.
`artifact_file_live.py` and `context_counting_live.py` also assert structural
provider/runtime behavior and report `verified`. The remaining model-backed
OpenAI/Anthropic demos are explicitly marked demo-only and report `smoke`.
All OpenAI/Anthropic live checks respect `CAYU_PROVIDER`; when it is set, the
matching API key must be present.

`examples/chat_completions_local_tools.py` remains a manual Gemini demo outside
the `*_live.py` glob. The asserted Gemini contract check is
`examples/chat_completions_contract_live.py`.

## Manual Runbook

Install the browser and server optional dependencies plus Chromium before
running the packaged dashboard contract. Chromium is intentionally not
installed by CI:

```bash
uv sync --extra browser --extra server
uv run playwright install chromium
uv run python scripts/nightly_verification.py --check dashboard-behavior --strict
```

Run the full visible map:

```bash
uv run python scripts/nightly_verification.py \
  --json /tmp/cayu-nightly-verification.json \
  --markdown /tmp/cayu-nightly-verification.md
```

Run only the local required lanes:

```bash
uv run python scripts/nightly_verification.py \
  --check core-pytest \
  --check internal-evals-hermetic \
  --check sigkill-recovery \
  --check postgres-required \
  --check docker-runner \
  --strict
```

`core-pytest`, `internal-evals-hermetic`, and `postgres-required` unset
live-provider credentials before running. `internal-evals-hermetic` executes:

```bash
uv run cayu eval run cayu.evals.internal.runtime_acceptance:build \
  --case-timeout-seconds 30 \
  --output .cayu-internal-runtime-acceptance.json
```

The ignored, repo-local output path avoids collisions with files owned by other
users in a shared system temporary directory. The check reports `hermetic` only
when all seven structural cases pass. That status
does not claim multi-phase approval resume, live-provider promotion, browser
behavior, `SIGKILL` recovery, provider billing reconciliation, LLM-judged
quality, or baseline release gating. Use the dedicated live lanes when model or
sandbox spend is intended.

`sigkill-recovery` runs the credential-free SQLite scenarios in a dedicated
POSIX process-death lane:

```bash
uv run python scripts/nightly_verification.py \
  --check sigkill-recovery \
  --strict
```

Each scenario launches a real worker process, waits for a committed-state
killpoint, sends `SIGKILL` to its process group, and recovers from a fresh
process using the same durable stores. The assertions cover automatic unknown
tool outcomes, manual tool reconciliation, partially finalized approval
interrupts, background-child reattachment, and both sides of the task
claim/attachment seam. The `postgres-required` lane additionally runs the
Postgres-marked claim cases. This proves deterministic process-boundary
recovery; it does not claim operating-system supervision, arbitrary external
exactly-once behavior, live-provider behavior, remote sandbox restart, machine
reboot, or cross-region failover.

Run credential-gated lanes only when the credential and quota are intentionally
available:

```bash
GEMINI_API_KEY=... uv run python scripts/nightly_verification.py \
  --check gemini-eval \
  --check chat-completions-contract \
  --strict

E2B_API_KEY=... uv run python scripts/nightly_verification.py \
  --check e2b-live-runner \
  --check e2b-live-workspace \
  --check e2b-live-sync \
  --strict
```

`chat-completions-contract` uses `CAYU_CHAT_COMPLETIONS_CONTRACT_MODEL`
defaulting to `gemini-3.1-flash-lite`. Keep it separate from `CAYU_GEMINI_MODEL` so
the structural contract is not accidentally moved to a model whose compatible
endpoint has different tool-continuation requirements.

For provider-output tests, do not assert exact prose. Assert structural facts:
session completed, expected tool calls occurred, files changed as expected,
usage exists, structured output validated, and finish reasons normalize to known
values.

## Known Holes

No capability in the current runner is classified as `unclaimed`. Issue #200
tracks the remaining controlled fault-injection additions.

Scheduled automation in #174 should decide which skipped or unclaimed statuses
are accepted for the nightly environment and which should fail the workflow.
