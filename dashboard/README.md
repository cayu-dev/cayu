# Cayu editable dashboard

This is the version-matched source for Cayu's open-source developer/operator control plane.
It was extracted from an installed Cayu release and is now owned by your application. Cayu
will not rewrite this directory during upgrades.

## Develop and build

Install Node.js 22.18.0 or newer, then run:

```bash
npm ci
npm run dev
```

The Vite development server proxies `/api` to `http://localhost:8000`. Run a compatible Cayu
server there, or update `vite.config.ts` for your local server.

Create production assets with:

```bash
npm run lint
npm run test
npm run typecheck
npm run check:api
npm run build
```

Normal development builds support Node.js 22.18.0 or newer. Byte-for-byte release asset
reproduction is verified with exactly Node.js 22.18.0 and the committed `package-lock.json`.

`npm run check:api` compares the generated client and `server-openapi.json` with an installed
Cayu Python package that includes the `server` extra (`cayu[server]`). Set `CAYU_PYTHON` when
that package is installed in a Python environment whose interpreter is not available as
`python`.

Serve `dist/` with `DashboardConfig(directory=...)`, `mount_cayu(..., dashboard_dir=...)`, or
`mount_dashboard(..., dashboard_dir=...)`. Cayu injects the configured base path and API URL,
including for non-root mounts and React-router deep links.

The normal build retains `LICENSE`, `NOTICE`, `REDISTRIBUTION.md`, and
`THIRD_PARTY_LICENSES.md` in `dist/` for redistribution.

## Compatibility and ownership

The dashboard refuses to load control-plane data when the server reports a different contract
version. After upgrading Cayu, run `npm run check:api` to detect contract or generated-client
drift. Run `npm run generate:api` only when you intentionally want to update this
application-owned copy for the newly installed server.

You may also replace this project completely with any UI that consumes Cayu's versioned,
authenticated control-plane API.

See `REDISTRIBUTION.md`, `LICENSE`, and `NOTICE` before distributing a modified build.
