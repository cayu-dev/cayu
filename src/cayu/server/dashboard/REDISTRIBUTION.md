# Redistribution and modifications

This dashboard source is provided under the Apache License 2.0 in `LICENSE`.

When redistributing Cayu-derived source or compiled dashboard assets:

- retain the Cayu `LICENSE` and `NOTICE` files;
- retain the generated `THIRD_PARTY_LICENSES.md` produced by the normal build;
- preserve applicable copyright, patent, trademark, and attribution notices; and
- clearly mark Cayu-derived files that you modify.

This guidance summarizes the normal dashboard workflow and does not replace the license terms.

## Dashboard source migration

The production shell is now `src/app/layout.tsx`, mounted through
`src/app/router.tsx` from `src/main.tsx`. Port customisations from the removed
`src/Layout.tsx` and the old Agents, Home, Evals, Knowledge, Pending, Run,
Session detail, Sessions and Tasks route modules to `src/app/`,
`src/next/routes/` and the shared editors in `src/components/dashboard/`.
Usage, Environments, Artifacts and System retain their `src/routes/`
implementations. Do not delete that directory wholesale.

Session Conversation and Trace are active production views. Trace owns exact
server span identities, bounded event history, incremental observation and the
embedded workflow inspector. The former Workflow URL maps to Session Trace;
it does not require the old standalone page implementation.

The prototype entry, mock API/seeds, legacy-page adapters and `CAYU_PROTOTYPE`
build are deliberately retained for development and demos. They are isolated
from the normal production entry and do not establish runtime permissions or
durability. The editable source archive includes them; the packaged production
dashboard does not use their mock entry point. Keep private snapshots outside
the source tree and out of redistributable builds.

Ejected source belongs to the application; Cayu upgrades do not rewrite it.
Match the generated API and server contract when rebuilding. See the editable
source's `README.md` and `src/next/PAGE-MAP.md` for the current route map and build
instructions.
