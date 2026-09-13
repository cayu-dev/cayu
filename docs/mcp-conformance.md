# MCP client coverage and conformance evidence

Cayu is an MCP **client**, not an MCP server. Modern `2026-07-28` support is
explicitly pinned and opt-in; 2025-era interoperability remains a separate wire
strategy. Passing the subset below is **not full MCP conformance** and does not
qualify Cayu against the upstream release's complete SDK requirements.

## Implemented surface

| Area | Current support | Verification boundary |
| --- | --- | --- |
| Modern HTTP and stdio | Discovery, stateless request metadata, tool listing/calls, resource listing/reading, cancellation and cleanup | Deterministic transport tests; official Python SDK tests for stdio and HTTP subscriptions |
| HTTP routing and parameter headers | Method/name/version cross-checks and bounded schema-declared parameter mirroring | Deterministic HTTP tests; official header scenarios blocked as described below |
| Catalogue metadata | Validate required cache hints, preserve server ordering and JSON Schema; retain Cayu admission and dispatch authority | Deterministic catalogue tests and selected official schema checks |
| Tool subscriptions | Explicit `subscriptions/listen`, acknowledgement/correlation validation, refresh fencing and owned cancellation on both transports | Deterministic fixtures and official Python SDK interoperability; no upstream client subscription scenario |
| Version selection | Explicit modern or 2025-era selection | No automatic negotiation or fallback; the upstream combined negotiation/request-metadata scenario is not supported |
| Caching | Metadata validation only | No authority-partitioned response cache yet |
| Extensions | No claim for MRTR, tasks, OAuth/CIMD, elicitation, sampling, prompts, logging, skills, or resource/prompt subscriptions | Not included in this conformance gate |

The implementation and deterministic regression tests live in `src/cayu/mcp/`
and `tests/core/test_mcp*.py`. The official SDK is pinned to `mcp==2.1.1` in the
development dependencies and `uv.lock`; neither it nor the conformance runner is
a Cayu production dependency.

## Official scenario subset

The referee is the unmodified
[official conformance repository at `7169291ec0b68eb370fddcd9947313ab0d5e4156`](https://github.com/modelcontextprotocol/conformance/tree/7169291ec0b68eb370fddcd9947313ab0d5e4156),
package `0.2.0-alpha.11`, installed with its own lockfile. Cayu's adapter only
invokes public client operations. It does not generate wire messages, retry a
rejected negotiation request, filter tool definitions, or patch the referee.

| Protocol | Scenario | Required evidence |
| --- | --- | --- |
| 2025-06-18 | `initialize` | Successful initialization check |
| 2025-06-18 | `tools_call` | Addition call and wire-schema validation |
| 2026-07-28 | `tools_call` | Addition call and wire-schema validation |
| 2026-07-28 | `json-schema-ref-no-deref` | Catalogue was listed without fetching the external `$ref` canary |
| 2026-07-28 | `json-schema-2020-12-preservation` | Tool discovery, schema echo, preservation of required keywords, and wire-schema validation |

The last scenario was introduced after the upstream frozen release requirements
and is `not_scored` there. It is still a required regression check in our subset;
we do not present it as satisfying a scored release requirement.

Three additional modern scenarios are **upstream-blocked**, not passed:
`http-standard-headers`, `http-custom-headers`, and `http-invalid-tool-headers`.
Their shared
[`BaseHttpScenario.sendDiscover` calls `sendJson`](https://github.com/modelcontextprotocol/conformance/blob/7169291ec0b68eb370fddcd9947313ab0d5e4156/src/scenarios/client/http-base.ts#L151),
which unconditionally returns `mcp-session-id`. Cayu rejects that discovery
response with `MCP 2026 HTTP responses must not mint an Mcp-Session-Id.` before
the header checks can be exercised. Do not remove this validation or patch the
fixture locally to manufacture a passing official result. A future upstream pin
update must reevaluate these scenarios and their expected checks.

Other upstream client scenarios remain outside this gate: combined automatic
negotiation/request metadata, authentication, MRTR, elicitation defaults, skills,
and legacy SSE retry. Some require unsupported features; others have not been
qualified against Cayu. The upstream client runner is HTTP-only and currently
has no subscription scenario. Server-side conformance scenarios test a different
product role and cannot establish Cayu client conformance.

## Reproduce in Docker

From the repository root, with Docker running:

```bash
docker build -f scripts/mcp-conformance.Dockerfile -t cayu-mcp-conformance .
evidence_dir=$(mktemp -d)
docker run --rm --init --network none -v "$evidence_dir:/results" cayu-mcp-conformance
```

Image construction needs network access. Verification uses container loopback
only (`--network none`); no providers, credentials, host Docker socket, or
privileged mode are needed. Python dependencies are frozen by `uv.lock`, the
upstream source by commit and npm lockfile. Base OS image tags remain updateable;
this is a pinned protocol/dependency baseline, not a bit-for-bit image promise.
The upstream npm lockfile currently has reported dependency advisories. This is
isolated verification tooling, not a deployable Cayu image. Do not inject secrets
or use `npm audit fix` to silently replace the referee's dependency baseline.

`summary.json`, raw official `checks.json`/client logs, runner logs, and SDK JUnit
evidence are written under `$evidence_dir/run`. An existing output directory is
rejected to prevent stale checks from satisfying a new run. Each official run
has a 30-second inner timeout and 45-second outer process-group deadline. SDK
verification has a 180-second deadline. Failures retain evidence. A missing
required check, any failure/warning/skip, a nonzero exit, or skipped SDK test
fails the gate. `full_conformance` is always false; `covered_subset_passed`
describes only the five rows above plus the separate SDK test group.

The SDK group requires four concrete test cases: modern stdio operations, HTTP
subscriptions, and stdio subscriptions in both graceful and Linux-contained
modes. These are reported as SDK interoperability, not official conformance
scenarios. Docker supplies the Linux environment needed for containment.

To reproduce the three blocked fixtures as well, use a fresh evidence directory:

```bash
diagnostic_dir=$(mktemp -d)
docker run --rm --init --network none -v "$diagnostic_dir:/results" cayu-mcp-conformance \
  --output /results/run --include-blocked
```

This diagnostic command intentionally exits nonzero. It never converts expected
failures into passes, even if an upstream change makes a previously blocked
scenario pass; coverage changes require review. The normal CI job runs only the
covered subset and uploads evidence even after failure. Run it manually through
the **MCP client conformance subset** workflow or on relevant pull requests.

For direct development on Linux, build the pinned upstream checkout with
`npm ci --ignore-scripts && npm run build`, then run:

```bash
uv run --frozen --extra dev --extra server python scripts/run_mcp_conformance.py \
  --upstream /absolute/path/to/conformance --output /absolute/new/evidence
```

Keep the coverage table, required check IDs, Docker source pin, and gate tests in
sync when changing the baseline. Inspect raw checks rather than relying on the
official runner's exit status: inapplicable scenarios can exit successfully
without producing meaningful checks.
