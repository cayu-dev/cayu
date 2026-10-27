# Cayu editable dashboard

This is the version-matched source for Cayu's open-source Developer dashboard.
It was extracted from an installed Cayu release and is now owned by your application. Cayu
will not rewrite this directory during upgrades.

## Dashboard structure and migration

The production entry is `src/main.tsx`. `src/app/router.tsx` owns the routes and
`src/app/layout.tsx` supplies the real API, server-contract check, identity,
capability gates and sidebar. The ported pages live under `src/next/routes/`;
`next` is now shared production code, not a prototype-only directory.

If you customised the replaced dashboard, move those changes to their current
owners before rebuilding:

| Removed source | Current owner |
| --- | --- |
| `src/Layout.tsx` | `src/app/layout.tsx` and `src/app/router.tsx` |
| `src/routes/dashboard.tsx`, `run.tsx` | `src/next/routes/home-page.tsx` and the shared Start form |
| `src/routes/sessions.tsx`, `session-detail.tsx` | `src/next/routes/sessions-page.tsx` and `src/next/routes/session/` |
| `src/routes/pending-actions.tsx` | `src/next/routes/inbox-page.tsx` and shared session decision controls |
| `src/routes/agents.tsx`, `knowledge.tsx` | Specialists, Tools and Knowledge pages in `src/next/routes/` |
| `src/routes/tasks.tsx` | `src/next/routes/tasks-page.tsx` and `task-page.tsx` |
| `src/routes/evals.tsx` | `src/next/routes/evals/` and shared editors in `src/components/dashboard/` |

Usage, Environments, Artifacts and System still use their `src/routes/` modules.
Do not remove that directory wholesale. The full workflow inspector is embedded
in Session Trace. The `/sessions/:id/workflow` alias opens it; `/agents`,
`/pending-actions` and `/run` lead to Specialists, Inbox and Home's Start form.
Supported old Evals run/result query links map to the current Evals pages.
See [the page map](src/next/PAGE-MAP.md) for page ownership and runtime references.

Keep this source copy, generated HTTP types and the server contract matched.
Ejecting or upgrading Cayu does not automatically port application-owned UI
changes or migrate older prototype/browser state.

## Retained prototype

`prototype.html`, `src/next/mock/`, its `mock/legacy/` fetch adapter,
`src/next/routes/legacy-page.tsx`, `src/next/legacy-config.ts` and the
`CAYU_PROTOTYPE` build remain deliberately as a development/demo harness.
They supply simulated data and hash navigation; they are not the production
integration or proof of runtime permissions and durability.

With the development server running, open `/prototype.html`. To build only that
harness, run `CAYU_PROTOTYPE=1 npx vite build`; it writes `dist/prototype/`.
The ordinary build uses `src/main.tsx` and the real API. The prototype sources
are included in the editable archive, but its separate output is not the stock
dashboard served by Cayu.

Optional `CAYU_PROTOTYPE_SEEDS` snapshots must stay outside this source tree.
Use synthetic data for shared demos: source archives include dashboard files,
and a prototype built with supplied snapshots contains that data.

## Sessions page

`/sessions` uses `src/next/routes/sessions-page.tsx` and the canonical session overview.
Rows lead with an authored title, then labels, the first user message, or the ID when
those are absent. Completed summaries are separate from the compact state badge; only an explicit
`attention` tone uses the warning colour. Cost retains its currency and marks partial
pricing; deployments without a price book show tokens.

The list excludes evaluation trials and forks by default, with independent switches
to include them. Merged state, raw status, debug state, specialist, provider, model,
environment, parent, causal budget, exact labels, selectors, and sort are server filters.
Existing Usage and Environment links keep their query parameters. Cursor paging may
scan empty candidate windows; the page reports loaded rows, never a filtered total.
After 25 empty windows it offers Continue searching. Changing filters cancels the old
read chain. Members can read the list but cannot start sessions.

For customised dashboards, the old `src/routes/sessions.tsx` has been removed. Shared
filter parsing remains in `src/lib/session-index-search.ts`; the prototype uses the same
page against its fake API. Tailwind scans `next/` for the components used in production.

## Develop and build

Install Node.js 22.18.0 or newer, then run:

```bash
npm ci
npm run dev
```

The Vite development server proxies `/api` to `http://localhost:8000`. Run a compatible Cayu
server there, or update `vite.config.ts` for your local server.

The stock dashboard follows the server's capability contract. A completely configured
authenticated Evals surface adds corpus import/browsing, promoted-session save, durable
launch/status/cancellation, complete result inspection, compatible comparison, and corpus plus
JSON/HTML downloads. The browser never receives executable target wiring or provider
credentials; it uses the attached server target and generated API contract.

Create production assets with:

```bash
npm run lint
npm run test
npm run typecheck
npm run check:api:repo
npm run build
```

Normal development builds support Node.js 22.18.0 or newer. In the Cayu repository,
regenerate the assets shipped in Python packages from the repository root with:

```bash
uv run python scripts/build_dashboard.py
uv run python scripts/build_dashboard.py --check
```

These commands require a Git checkout and Docker with Linux/amd64 support.
The canonical frontend build uses the immutable Node 22.18.0 Debian Bookworm image declared in
`cayu._dashboard_artifacts`, including npm 10.9.3 and the committed lockfile.
It fixes locale and timezone and normalizes CRLF in UTF-8 inputs declared as text
by Git attributes. Binary assets retain their bytes, including files marked
`-text` or `binary` in `.gitattributes`. Each build installs dependencies in a
disposable Docker environment; it does not use or modify the checkout's `node_modules`.

The command reads the dashboard source tree and root `LICENSE` and `NOTICE`,
excluding local build products declared in `cayu._dashboard_artifacts`. It owns
`src/cayu/server/dashboard/` and `src/cayu/data/dashboard_source/`. After the
frontend build and license finalization, the existing Python bundle writer creates
the source ZIP with fixed metadata and uncompressed entries. Release validation
checks the bundle and compiled digest before any checkout output is written.
`--check` lists added, removed and changed output files and leaves the checkout
untouched. Regenerate and commit both output trees when this check reports drift.

API generation remains a separate explicit step: use `npm run generate:api:repo`
after changing Python contracts or release versions, then run the canonical build.
When dependencies change, review the finalized license inventory and update
`dashboard/THIRD_PARTY_LICENSES.md` before regenerating the bundle.
Ejected dashboards continue to use the local `npm run build` command above;
Docker is required only for authoritative Cayu package regeneration.

`npm run check:api:repo` compares the generated client and `server-openapi.json` with the
repository-root `.venv` and fails if that environment is absent. `CAYU_PYTHON` remains the
explicit highest-priority override. Ejected dashboard source can use `npm run check:api`, which
selects an ambient `python` unless `CAYU_PYTHON` is set; that environment must contain the exact
Cayu version recorded in `src/lib/release-metadata.ts` with the `server` extra.

The same commands generate and check `src/lib/generated/eval-text-limits.ts`
from the evaluation corpus domain. Suite/case name and description limits count
Unicode scalar values. Update the Python definitions and regenerate the client
when changing these limits; the dashboard retains its existing text validators.

`src/lib/generated/label-limits.ts` is generated by the same command. Label key/value
bounds and their Unicode scalar unit come from `cayu._validation`; aggregate
filter, selector and selector-value counts come from `cayu.sessions.queries`.
Session editing and Usage filtering import these declarations. Keep the Python
values authoritative and regenerate after changing them.

`server-openapi.json` is generated from the Python server's OpenAPI schema by
`scripts/generate-api-types.py`; the TypeScript API types are generated from that
JSON. `src/lib/release-metadata.ts` comes from the selected Python distribution's
package version and `cayu._server_contract_version.SERVER_CONTRACT_VERSION`.
Regenerate these files with `npm run generate:api:repo` in the repository, or
`npm run generate:api` in an ejected dashboard using its intended `CAYU_PYTHON`.

The installed Cayu module `cayu._dashboard_artifacts` owns generated-source paths,
required source files, package output locations, and release-metadata rendering.
API generation and source-bundle validation consume that declaration. When adding
a generated output, update the declaration and its generator together, then run
the existing API and bundle checks. Files under `src/lib/generated/` must belong
to the declared output inventory; place custom dashboard code outside that tree.
The bundle manifest continues to record all source files and the compiled-assets
digest. Its version-1 `generated_api_digest` covers OpenAPI and server API types;
the full source digest also covers release metadata, evaluation text limits and label limits.

Serve `dist/` with `DashboardConfig(directory=...)`, `mount_cayu(..., dashboard_dir=...)`, or
`mount_dashboard(..., dashboard_dir=...)`. Cayu injects the configured base path and API URL,
including for non-root mounts and React-router deep links.

The normal build retains `LICENSE`, `NOTICE`, `REDISTRIBUTION.md`, and
`THIRD_PARTY_LICENSES.md` in `dist/` for redistribution.

## Compatibility and ownership

The dashboard refuses to load its data when the server reports a different contract
version. After upgrading Cayu, run `npm run check:api:repo` to detect contract or generated-client
drift. Run `npm run generate:api:repo` only when you intentionally want to update this
application-owned copy for the newly installed server.

You may also replace this project completely with any UI that consumes Cayu's versioned,
authenticated control-plane API.

See `REDISTRIBUTION.md`, `LICENSE`, and `NOTICE` before distributing a modified build.
