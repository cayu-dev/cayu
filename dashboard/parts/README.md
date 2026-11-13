# @cayu-tech/dashboard-parts

The Cayu Developer dashboard's pages as parts an Agent's own app can show, starting
with the Inbox. The package's version is the Cayu version it shipped with, and its
types and calls follow that version's server contract, so install the copy your Cayu
wrote:

```sh
cayu dashboard parts            # writes web/vendor/cayu-dashboard-parts-<version>.tgz
cd web && npm install           # after the first run: npm install ./vendor/cayu-dashboard-parts-<version>.tgz
```

`cayu check` warns when `web/package-lock.json` resolves this package to another
version than the installed Cayu.

## What the app provides

- `@kit/ui`, `@kit/lib` and `@kit/icons`: the app's component kit, mapped in Vite's
  `resolve.alias` and in tsconfig `paths`. Parts use `Button`, `Card`, `CardContent`
  and `Input` from `@kit/ui`, `cn` from `@kit/lib`, and `Icon`, `KitIcon` and the
  HugeIcons names they show from `@kit/icons`.
- `optimizeDeps: { exclude: ["@cayu-tech/dashboard-parts"] }` in the Vite config, so
  Vite doesn't pre-bundle a second copy of the app's kit into the package.
- `@import "@cayu-tech/dashboard-parts/styles.css";` in the kit stylesheet, so
  Tailwind generates the parts' classes.
- React 19, React DOM 19 and TanStack Query 5, as peer dependencies.
- A host: `<PartsHostProvider value={host}>` around the parts, with `now`, a `Link`,
  an optional `onSettled`, and an `inbox` section (its `api` and what the person may
  do). `inboxApiFromCayuClient(client)` builds the `api` from client.js.

## Entries

- `@cayu-tech/dashboard-parts`: `PartsHostProvider`, `usePartsHost`, the host and
  record types, `partKeys`, `EmptyState`, `ErrorState`, `relativeTime`,
  `CAYU_VERSION` and `SERVER_CONTRACT_VERSION`; and `ProtectedReviewPart` with its
  `ReviewApi` and `reviewApiFromCayuClient`, for deciding under protected human review.
- `@cayu-tech/dashboard-parts/inbox`: `InboxPart` (the preset), and `InboxFilters`,
  `InboxList` and `InboxCard` on their own; `useInbox`, `useInboxCounts`,
  `inboxKeys` and `inboxApiFromCayuClient`.
- `@cayu-tech/dashboard-parts/styles.css`: the Tailwind `@source` line.
