# A B2B application: several businesses on one agent

`cayu new NAME --preset b2b` generates an agent that several businesses share on one
deployment, one database and one agent. Each business is an organization with departments
and members. People sign in to the application, see their own conversations, their
department's and their organization's, and nothing of another organization. The
organization's admins manage its members from the application's web interface.

## Who decides what

The application decides who belongs where. Cayu enforces what that allows.

| Owner | What it decides | Where |
| --- | --- | --- |
| The application | Organizations, departments, memberships and roles, in its own tables | `database/`, `migrations/` |
| The application | Who a request is (sign-in) | `integrations/sign_in.py` |
| The application | Which labels a new conversation gets (personal, department or organization sharing) | `domain/sharing.py` |
| The application | Turning a person's current memberships into Cayu grants | `policies/access.py` |
| Cayu | Checking those grants at every protected step: listing, reading, continuing, background work | `ResourceAccessPolicy`, `app.access(subject)` |

Cayu asks the policy again at every protected step, so removing a member takes effect at
their next step, including a conversation that is running or a task that is queued.

## Run it locally

```bash
cayu new acme-desk --preset b2b
cd acme-desk
uv sync
export CAYU_PROVIDER=anthropic ANTHROPIC_API_KEY=...  # or openai with OPENAI_API_KEY
export CAYU_DATABASE_URL=postgresql://localhost/acme_desk
uv run --no-sync cayu storage migrate
uv run --no-sync alembic upgrade head
uv run --no-sync python -m operations.demo_seed
uv run --no-sync cayu serve --dev
```

Open `http://127.0.0.1:8000/` and choose a demonstration person:

| Person | Organization and department | Role | What they see |
| --- | --- | --- | --- |
| alice | acme, acme/finance | admin | Her own, and acme/finance's and acme's shared conversations; the Members page |
| carol | acme, acme/finance | member | Her own, and acme/finance's and acme's shared conversations; continues only her own |
| bob | acme, acme/legal | member | Nothing alice shared with acme/finance; what she shared with acme |
| dave | globex, globex/support | admin | Nothing of acme's |

As alice, start a conversation shared with "My department", then pick carol under
"Demonstration person" in the profile menu: she reads it but can't continue it. As bob or dave, it isn't listed and
its address answers that it doesn't exist. As alice, remove carol on the Members page; her
next step is refused.

The demonstration login exists only under `cayu serve --dev`, for the people
`operations/demo_seed.py` adds.

## The web interface

`web/` holds the interface's source (React, TypeScript, Tailwind CSS), and `web/static` the
built files `cayu serve` serves at `/` (`[tool.cayu.serve] ui = "web/static"`).

It is the app shell every Agent's app is generated with, and b2b's own pages as areas in it:

- **The shell** (`web/src/app/`, `web/src/main.tsx`): a sidebar with the Agent's areas, a
  Runtime group linking to Cayu's Inbox and Developer dashboard, a profile menu with the
  person's role and appearance, and the not-found, error and sign-in-again pages.
- **Overview and Settings** (`web/src/generated/`): the Agent's metrics and what waits in
  Cayu's inbox for it, then the Agent's own sections; its managed settings, where each value
  comes from, and the `cayu cloud settings` commands that change them.
- **The Agent's own areas** (`web/src/agent/`): Conversations and Members, the Overview's
  sections, and the demonstration sign-in. Add an area by writing it there and listing it
  in `web/src/agent/index.tsx`.
- **TaskRuns** (`web/src/blocks/task-runs.tsx`): the Agent's Cayu tasks, one row per run, for
  an area of an Agent that queues tasks.
- **The component kit** (`web/src/kit/shadcn/`): stock shadcn on Base UI. Everything imports
  it through the `@kit` alias, so another kit with the same entries replaces it without
  changing a page.

Each file's first doc comment says `GENERATED.` (the generator owns it) or `THE AGENT'S OWN.`
(yours). The built files are committed, so
running and deploying the application needs no Node. After changing `web/src`:

```bash
npm --prefix web ci
npm --prefix web run build
```

`npm --prefix web run check:build` builds into a temporary folder and fails when `web/static`
isn't the build of `web/src`; run it in CI, since a deployment ships the committed build. A
build reads only `web/src`, never the previous build, so the same source always builds the
same files.

`npm --prefix web run dev` serves the interface with live reload and sends `/app`, `/api` and
`/cayu` to `cayu serve --dev` on port 8000.

The built files load from `/`, and every `GET` that no route answers gets the page, so the
interface can route in the browser and a reload or a shared link opens the same page. The
API under `/app`, Cayu's `/api` and the Developer dashboard at `/cayu` answer first, and an
unknown path under them, or a missing file under `/assets/`, is still a 404.

The interface calls only the application's API under `/app`, never Cayu's `/api`:

| Route | What it does |
| --- | --- |
| `GET /app/me` | The signed-in person, their memberships and the agent's name |
| `POST /app/conversation-labels` | The labels for a new conversation, built from the person's membership |
| `/app/resources/...` | Cayu's scoped routes: list and read conversations, start (`runs`) and continue (`resume`) one with a streamed reply |
| `/app/organizations/{organization}/members` | Admins list, add, move and remove members |
| `GET /app/whoami` | The signed-in person and their role: `admin` if they administer any of their organizations, else `member` |
| `GET /app/settings` | The Agent's managed settings, where each value comes from and the `cayu cloud settings` commands that change it, and the settings it reads at start-up |
| `GET /app/overview` | The Agent's metrics for the person (`integrations/http/overview_metrics.py`) and what waits in Cayu's inbox |
| `GET /app/task-runs` | The person's Cayu tasks, one row per run of the Agent's work |
| `/app/inbox` | Cayu's inbox for this Agent, listed and settled as the person; Cayu decides what they may do |

The last five are generated (`integrations/http/generated/`, each file marked `GENERATED.`);
the Agent supplies its people, metrics and start-up settings to `build_generated_router` in
`integrations/http/__init__.py`. Each module's `router(...)` also works on its own.

It doesn't use `/cayu/client.js` (`cayu guide app-ui`): that client calls Cayu's operator
API, which `OPERATOR_ONLY_AUTH` keeps from the application's members. The interface reads a
conversation's transcript when it opens it, and follows a reply through the stream its own
`runs` or `resume` request returns. Pages that show something that changes on its own (the
Overview, the TaskRuns block's `taskRunsQuery`) follow their `/app` route with the kit's
`watchInterval` (`@kit/data`) instead of a timer: the cadence `client.js` keeps, a read 15
seconds after the last, doubling while the answer doesn't change up to two minutes, and
nothing while the tab is hidden. Give your own such query `refetchInterval: watchInterval()`.

A reply runs inside the request that streams it. Moving around the interface keeps it running,
but closing the tab or losing the connection stops the agent's reply there; the conversation
keeps what it had. Work that must outlive the page belongs in a task (`cayu guide
references#background-work`).

The organization chosen in the header is only a filter and the default for new
conversations; the server decides what anyone can see.

## Sign-in outside development

Without `--dev`, `APP_SIGN_IN` selects how people sign in:

- `cloud` (the default): Cayu Cloud's sign-in. Memberships are keyed on the ID it gives
  each person.
- `oidc`: an OIDC bearer token, with `CAYU_OIDC_ISSUER` and `CAYU_OIDC_AUDIENCE`.

Until one is configured, the application's API answers 503. The web interface signs people
in with a cookie, so it works with Cayu Cloud's sign-in and the demonstration login; with
`oidc`, the API is for clients that send their own bearer token.

Add the first admin of each organization with `add_member(...)` in
`database/memberships.py`, from a one-off operation you write; from then on, the
organization's admins manage its members on the Members page.

## Deploy

On Cayu Cloud, `cayu cloud init` and `cayu cloud deploy .` deploy it like any agent. Cloud
runs `cayu storage migrate` and the application's own migration (`[database] migrate` in
`cayu-cloud.toml`) before the new Release starts. The Developer dashboard stays for
operators only: `[tool.cayu.serve].auth` is `OPERATOR_ONLY_AUTH`, which refuses the people
Cloud signs in as members. They reach their data only through `/app`.

Don't run `operations/demo_seed.py` against a deployment: it refuses a database that
already holds real organizations.
