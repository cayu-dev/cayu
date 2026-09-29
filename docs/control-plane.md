# Open and replaceable control plane

Cayu's bundled developer/operator control plane is open-source React and TypeScript. The
compiled dashboard is convenient package data, not an opaque or mandatory product UI. An
application can choose among three supported modes:

| Mode | Owner | Use it when |
| --- | --- | --- |
| Bundled dashboard | Cayu | You want the stock zero-configuration operator UI. |
| Editable Cayu dashboard | Your application after extraction | You want to customize the same source and generated API client used by Cayu maintainers. |
| Bring your own UI | Your application | You want a completely independent frontend over the versioned control-plane API. |

All three modes expose an operator surface. Public deployment requires explicit authentication
and authorization; unauthenticated `ServerConfig.local_development()` access is for trusted
loopback development only.

## Project-owned Evals assembly

`cayu serve` assembles the durable, non-executable Evals foundation without
requiring an application to construct `EvalsConfig`: normalized project
identity from `[project].name`, release identity from `CAYU_RELEASE_ID` or the
public application-manifest fingerprint, and a matching SQLite or PostgreSQL
`EvalStore` from the project's session-store declaration. Trusted loopback
`cayu serve --dev` may create `data/cayu.db` as the local default; production
does not invent a database.

After the application is constructed, Cayu generates one bounded `default` eval
target for every registered agent. Each target keeps the agent's normal provider,
tools, environment defaults, approvals, and runtime policy. Its stable key is a
domain-separated SHA-256 of the project, agent, and profile identities, so the
key does not change when the application release changes. The current release
and exact application-manifest fingerprint remain separate result identity.

`GET /api/evals/targets` publishes the safe project/agent/profile mapping. Corpus
and run catalog requests select only one of those keys; an omitted selector uses
the server-published deterministic default. Imports and run admission resolve the
key carried by the immutable corpus and reject unknown keys. The browser never
manufactures a live application, credentials, tools, environments, request
templates, or other execution authority.

A generated project can publish one optional default judge by declaring
`[tool.cayu.evals.default_judge]` in `pyproject.toml`. The declaration names an
already-registered provider, exact model, `public-only` or
`public-and-transcript` privacy, an explicit same-model decision, and bounded
time/token ceilings. `[tool.cayu.evals].price_book = "bundled-public"` may
explicitly select Cayu's dated packaged public-rate snapshot; with that
selection, the judge may also declare a cost threshold and author-first
launches may narrow candidate spend per trial. Cayu gives the judge a separate
tool-free application and never infers authority, credentials, or pricing.
Same-model use remains a separate, labeled operator choice in the suite editor.
Run `cayu guide evals-ai-quality` for the complete declaration and calibration
workflow.

New maintained-service factories accept Cayu's opaque project context and pass
it unchanged to `create_agent_service(...)`. Existing factories remain
compatible but do not receive automatic assembly until migrated. Use:

```bash
cayu check --json
cayu generate service-context --dry-run
cayu generate service-context
```

The generator edits only the previous generated shape it can prove safe;
customized or conflicting code requires manual review. Explicit `EvalsConfig`
continues to win as one complete singleton registry and is never partially
combined with framework-assembled state. Direct embedded servers continue to
wire their trusted objects explicitly. The registry identity already includes a
profile dimension; this slice publishes only the normal-authority `default`
profile. Additional application-owned profiles are reserved for cases that
deliberately substitute fixtures, environments, or authority.

## Controlled scenario execution

The bundled Evals workflow can save and run a scenario without constructing an
Evals-specific Python object in the browser or project. After **Check
readiness** succeeds, **Run scenario** admits the exact saved scenario revision
and reviewed binding to the durable eval worker. The Runs view shows each
trial's current phase and presents approve/deny controls only when that exact
trial reaches an authored fresh-approval checkpoint.

Initial input, queued follow-ups, ordinary session resumes, typed `ask_user`
answers, portable JSON, and file references execute through the registered
application's normal runtime. Approval and resume checkpoints retain their
session and event cursor across process restart. Cancellation, result JSON/HTML,
baseline comparison, `cayu eval report`, and `cayu eval compare` use the same
contracts as ordinary corpus runs.

This availability is not limited to `--dev`. Trusted loopback development may
assemble local storage automatically; production must supply its normal durable
store, authentication, and executable application target explicitly or through
the maintained project-service assembly. Readiness remains a factual gate for
missing current providers, tools, policies, environments, secrets, fixtures,
pricing, or execution limits—it is not a requirement to write Evals-specific
runtime configuration.

For the shortest user paths, run `cayu guide evals-first` or `cayu guide
evals-production` from the same installed Cayu version as the server.

## Eject and customize the matching source

Every wheel and source distribution carries one deterministic dashboard-source bundle tied to
the Cayu package version, server contract, generated API client, and bundled production assets.
Extraction is local and performs no Git, GitHub, or network operation:

```bash
cayu dashboard eject ./control-plane
cd control-plane
npm ci
npm run dev
```

The Vite server proxies `/api` to `http://localhost:8000` by default. Build application-owned
production assets with:

```bash
npm run lint
npm run test
npm run typecheck
npm run check:api
npm run build
```

`check:api` compares the extracted OpenAPI baseline and generated client with an installed Cayu
Python package that includes the `server` extra (`cayu[server]`). Set `CAYU_PYTHON` to the
intended environment's interpreter when it is not available as `python`.

Serve the resulting `dist/` directory through the normal Cayu configuration seam:

```python
from pathlib import Path

from cayu.server import DashboardConfig, ServerConfig, create_server

server = create_server(
    cayu_app,
    config=ServerConfig.local_development(
        dashboard=DashboardConfig(directory=Path("control-plane/dist")),
    ),
)
```

Embedded applications can instead pass the same path to
`mount_cayu(..., dashboard_dir=Path("control-plane/dist"))` or
`mount_dashboard(..., dashboard_dir=Path("control-plane/dist"))`. Non-root mount paths and
React-router deep links use the base path and API URL injected by the server.

## Ownership, upgrades, and compatibility

The destination must be explicit and empty. Cayu never rewrites an ejected directory during
installation, upgrade, or later CLI commands. Eject a new copy elsewhere when you want to inspect
a newer release, then merge application changes deliberately.

The dashboard checks the server's exact control-plane contract before loading data. A Cayu
upgrade that changes that contract produces a visible compatibility failure, and `npm run
check:api` reports OpenAPI/client drift; neither mechanism silently modifies source.

Evals remains present in navigation even when its catalog has not been assembled. Server
contract version 19 introduced independent readiness for captured evaluation, catalog reads and writes,
captured-result persistence, scenario conversion, fresh execution, cancellation, comparison,
and reports. An unready Evals page renders those states without requesting unavailable Evals
endpoints. Readiness is explanatory metadata only; authenticated API routes continue to enforce
operator identity, mutation policy, and execution preconditions.

Server contract version 25 adds durable scenario launch, per-trial progress,
and fresh scenario-approval mutations. Independently deployed dashboards,
servers, and generated clients must be upgraded together.

The extracted project includes Cayu's `LICENSE` and `NOTICE`, third-party license inputs, a
reviewed license baseline, and production-build license finalization. Its normal build retains
`LICENSE`, `NOTICE`, `REDISTRIBUTION.md`, and `THIRD_PARTY_LICENSES.md` in `dist/`. Downstream
distributors must retain applicable notices and mark modified Cayu-derived files as changed.

## Bring your own UI

A completely custom frontend can consume the authenticated API without using Cayu's React
source. Begin with `GET /api/contract`, honor its capability projection, use the bounded list and
detail APIs, and fail closed on an unsupported `contract_version`. Authentication,
authorization, redaction, and capability contracts are identical regardless of frontend.

To show a running session's live progress, open the read-only follow stream that the contract
advertises as `sse.session_follow` (`GET /api/sessions/{session_id}/events/stream`) with a
browser `EventSource`. Pass `after_sequence` for the last event you already rendered; the
browser's automatic reconnect sends `Last-Event-ID` and resumes without gaps or duplicates.
Filter out chatty events such as `exclude_event_type=model.text.delta` when you do not render
them. Close the `EventSource` when the `end` event arrives: the session is finished and the
server has already closed the stream. The stream never starts or changes work, so opening it is
safe from any page.

Do not poll `/events` or `/state` on a fixed timer while a page is open. If a client cannot hold
a stream open, poll `/events` with `after_sequence` and `wait_seconds` only while the page is
visible and the session is pending, running, or interrupting, and stop at a terminal status.
`GET /api/sessions/{session_id}/state` returns an `ETag`, so a browser revalidates unchanged
state with a `304` and no body. Concurrent follow streams are capped per caller and per
session; a `429` carries `Retry-After`.

Show tokens and cost with the accounting endpoints that the contract lists under
`accounting.usage` and `accounting.cost`, each with its `path` and a `supported` flag:

- `GET /api/sessions/{session_id}/usage` returns the session's `SessionUsageSummary`. The
  session store carries the summary forward and reads only events appended since the
  previous read, so polling does not get slower as the session grows. Each response has an
  `ETag`; send it back as `If-None-Match` and the server answers `304 Not Modified` until the
  usage changes.
- `GET /api/sessions?include=usage` adds a `usage` array with one summary per listed session,
  in the same order as `sessions`, so an overview page needs one request instead of one per
  session.
- `POST /api/sessions/{session_id}/cost` estimates cost from a `PriceBook` in the request body.

Do not compute tokens or cost by folding raw `model.completed` events in the UI or in
application code. That skips cache read and write counters, auxiliary provider attempts,
hosted-tool usage, and price-book rules, and its cost grows with the whole event history.

Provider-operation resolution is an explicit mutation capability. When session
state reports `provider_operation_unavailable` or `ambiguous_submission`, the
UI must display the exact `stage_id` and `run_epoch`, recovery reason,
`duplicate_request_risk`, and server-provided allowed resolutions. Submit only
`fallback_retry` or `fail` to `POST /api/provider-operations/resolve`. Treat a
409 as a stale or conflicting durable decision, never as permission to resubmit
with a guessed provider operation id. The bundled dashboard requires an
operator reason and warns whenever `duplicate_request_risk=true` that fallback
can duplicate provider work and cost.
