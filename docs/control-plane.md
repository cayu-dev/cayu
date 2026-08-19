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

The extracted project includes Cayu's `LICENSE` and `NOTICE`, third-party license inputs, a
reviewed license baseline, and production-build license finalization. Its normal build retains
`LICENSE`, `NOTICE`, `REDISTRIBUTION.md`, and `THIRD_PARTY_LICENSES.md` in `dist/`. Downstream
distributors must retain applicable notices and mark modified Cayu-derived files as changed.

## Bring your own UI

A completely custom frontend can consume the authenticated API without using Cayu's React
source. Begin with `GET /api/contract`, honor its capability projection, use the bounded list and
detail APIs, and fail closed on an unsupported `contract_version`. Authentication,
authorization, redaction, and capability contracts are identical regardless of frontend.

Provider-operation resolution is an explicit mutation capability. When session
state reports `provider_operation_unavailable` or `ambiguous_submission`, the
UI must display the exact `stage_id` and `run_epoch`, recovery reason,
`duplicate_request_risk`, and server-provided allowed resolutions. Submit only
`fallback_retry` or `fail` to `POST /api/provider-operations/resolve`. Treat a
409 as a stale or conflicting durable decision, never as permission to resubmit
with a guessed provider operation id. The bundled dashboard requires an
operator reason and warns whenever `duplicate_request_risk=true` that fallback
can duplicate provider work and cost.
