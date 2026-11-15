# Organizations on one deployment: the b2b preset

`cayu new NAME --preset b2b` generates an agent that people from separate organizations
share on one deployment, one database and one agent. Each organization (a client firm, an
agency's client, a household) has departments and several members. People sign in to the
application's web app, see their own conversations, their department's and their
organization's, and nothing of another organization. The organization's admins manage its
members in the web app.

Choose it only for that, not for your own customers or users, who are each one person: use
service or agent. Every preset but service already comes with a web app for your own
team (`cayu guide app-ui`), so an Agent that only you and your colleagues use doesn't need
b2b to get one. b2b is that web app plus the `organizations` capability, with the
application database it needs:

- The `organizations` capability makes the web app's people the organizations' members
  instead of your team: member sign-in, Members, an organization switcher, and
  conversations scoped to each organization. Operators keep the Developer dashboard at
  `/cayu/`.
- `--without organizations` and `--without web-app` are refused on b2b. For an Agent
  without organizations, choose another preset.
- The agent preset takes the same organizations with `cayu new NAME --with organizations`,
  and keeps its knowledge and memory, scoped to each organization (see "Knowledge and
  memory" below). The coding, long-running and service presets don't take organizations.

## Who decides what

The application decides who belongs where. Cayu enforces what that allows.

| Owner | What it decides | Where |
| --- | --- | --- |
| The application | Organizations, departments, memberships and roles, in its own tables | `database/`, `migrations/` |
| The application | Who a request is (sign-in) | `integrations/sign_in.py` |
| The application | Which policy turns a person's current memberships into Cayu grants: `cayu.memberships`' `MembershipAccessPolicy`, or its own | `policies/access.py` |
| Cayu (`cayu.memberships`) | Reading and changing memberships in those tables, the access revision, the sharing labels a new conversation gets (`conversation_labels`) and the grants they allow | `database/memberships.py` wires it to the application's database |
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

## The web app

`web/` holds the app's source (React, TypeScript, Tailwind CSS), and `web/static` the
built files `cayu serve` serves at `/` (`[tool.cayu.serve] ui = "web/static"`).

It is the app every preset but service generates (`cayu guide app-ui`), with the
organizations' pages as areas in it and their member sign-in in place of operator sign-in:

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

`npm --prefix web run dev` serves the app with live reload and sends `/app`, `/api` and
`/cayu` to `cayu serve --dev` on port 8000.

The built files load from `/`, and every `GET` that no route answers gets the page, so the
app can route in the browser and a reload or a shared link opens the same page. The
API under `/app`, Cayu's `/api` and the Developer dashboard at `/cayu` answer first, and an
unknown path under them, or a missing file under `/assets/`, is still a 404.

The app calls only the application's API under `/app`, never Cayu's `/api`:

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
API, which `OPERATOR_ONLY_AUTH` keeps from the application's members. The app reads a
conversation's transcript when it opens it, and follows a reply through the stream its own
`runs` or `resume` request returns. Pages that show something that changes on its own (the
Overview, the TaskRuns block's `taskRunsQuery`) follow their `/app` route with the kit's
`watchInterval` (`@kit/data`) instead of a timer: the cadence `client.js` keeps, a read 15
seconds after the last, doubling while the answer doesn't change up to two minutes, and
nothing while the tab is hidden. Give your own such query `refetchInterval: watchInterval()`.

`GET /app/settings`, `/app/overview` and `/app/task-runs` answer with an `ETag`
(`from cayu.server.etag import tagged_json`, `cayu guide app-ui`), so a read of an unchanged
answer costs a `304` with no body. The browser revalidates the page's reads itself and hands
the kit the answer it kept, so `@kit/data` needs no code for it. The Overview's tag leaves
out `as_of`, which changes on every read, and the page says when it last read the overview
rather than the kept answer's `as_of`. Answer a read route of your own the same way.

A reply runs inside the request that streams it. Moving around the app keeps it running,
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

Until one is configured, the application's API answers 503. The web app signs people
in with a cookie, so it works with Cayu Cloud's sign-in and the demonstration login; with
`oidc`, the API is for clients that send their own bearer token.

Add the first admin of each organization with `cayu memberships add SUBJECT ORGANIZATION
--role admin`, with `CAYU_DATABASE_URL` set to the application's database; from then on, the
organization's admins manage its members on the Members page. `cayu memberships list
--organization ORGANIZATION` lists its members and `cayu memberships remove SUBJECT
ORGANIZATION` removes one; neither the command nor the Members page removes or demotes an
organization's last admin.

## Knowledge and memory

The b2b preset has no knowledge or memory. `cayu new NAME --with organizations` on the agent
preset keeps both, and each organization's knowledge is its own. The knowledge store
enforces it, so no tool has to remember to filter:

- `policies/access.py` builds `MembershipAccessPolicy(memberships(), knowledge=True)`: a
  member reads, creates and revises knowledge labelled with one of their organizations, and
  nothing of any other organization's.
- `knowledge/retrieval.py` sets `session_label_keys=("organization",)` on the knowledge scope:
  inside a conversation, every knowledge read and write is pinned to that conversation's
  verified `organization` label. A person in two organizations recalls only the current
  conversation's organization's knowledge, so it never lands in a conversation the other
  organization's members read.
- `remember_knowledge` copies `organization` from the conversation
  (`RememberKnowledgePolicy(copy_session_labels=("organization",))`), never from the model.
  The store refuses a note without it, or with another organization's.
- Recall, `search_knowledge`, `read_knowledge`, `list_knowledge` and the pending notes recall
  offers all read through the same filter.
- Curation is per organization: `uv run --no-sync cayu knowledge list --status pending
  --label organization=acme`, then `cayu knowledge keep` or `drop`. A curator or a
  maintenance run takes `labels={"organization": ...}` for one organization. Operator and
  background work outside a member's conversation isn't narrowed.
- Knowledge you add yourself (seeding, operator scripts, curation) needs an `organization`
  label, since no member could read it without one: the store refuses an active or pending
  entry without it and says what to add. Entries stored before this rule can still be
  revised to add the label, or archived.

Work queued before you turn on `knowledge=True` keeps the grant it was admitted with, which
has no knowledge, so its knowledge reads are refused.

## Move to `cayu.memberships`

Projects generated before Cayu had `cayu.memberships` carry their own copies of the policy
(`policies/access.py`), the membership SQL (`database/memberships.py`) and the sharing labels
(`domain/sharing.py`). Moving them to the library takes one commit:

1. Upgrade the project's Cayu to a release with `cayu.memberships` and run `uv run --no-sync
   cayu check`. It notes each file that is still exactly the copy Cayu generated
   (`MEMBERSHIPS_GENERATED_COPY`). A file you edited is yours: compare it with what `cayu new`
   generates now and carry your change over, or keep it.
2. Generate a new project with the same capabilities next to it, for example `cayu new
   acme-desk --preset b2b --dir /tmp/fresh`, and copy these files from it:
   `policies/access.py`, `database/memberships.py`, `domain/organizations.py` (now only
   `app_role`), `operations/demo_seed.py`, `integrations/http/__init__.py`,
   `integrations/http/overview_metrics.py`, `tests/test_organizations.py`,
   `tests/test_http_api.py`, `tests/test_generated_routes.py` and
   `tests/test_background_revocation.py`. The "Organizations and sharing" section of
   `README.md` and `AGENTS.md` changed too.
3. Delete `domain/sharing.py`. Import `Membership`, `conversation_labels`, `Sharing`,
   `InMemoryMembershipDirectory` and `MembershipAccessPolicy` from `cayu.memberships`
   wherever your own code and tests took them from `domain/` or `policies/access.py`.
   `DatabaseMembershipDirectory()` becomes `memberships()`, `seed(db)` becomes `seed()`,
   `add_member` and `remove_member` become `memberships().add(...)` and
   `memberships().remove(...)` (they are async), and the member routes call
   `memberships().change_as_admin(...)`.
4. Run the tests with `CAYU_TEST_POSTGRES_DSN` set; `tests/test_organizations.py` checks the
   tables with `check_membership_tables`.

There is no database migration: the library reads and writes the same `memberships` and
`access_revision` tables. Queued and running work stays admitted, because the policy keeps
the authority `b2b-membership-v1` and grants exactly what the generated one did. The one
difference: it refuses a membership whose department isn't its organization's, which the
generated tables already prevent. Members can now be managed with `uv run --no-sync cayu
memberships list|add|remove`.

The library itself is described in Cayu's `docs/memberships.md`.

## Deploy

On Cayu Cloud, `cayu cloud init` and `cayu cloud deploy .` deploy it like any agent. Cloud
runs `cayu storage migrate` and the application's own migration (`[database] migrate` in
`cayu-cloud.toml`) before the new Release starts. The Developer dashboard stays for
operators only: `[tool.cayu.serve].auth` is `OPERATOR_ONLY_AUTH`, which refuses the people
Cloud signs in as members. They reach their data only through `/app`.

Don't run `operations/demo_seed.py` against a deployment: it refuses a database that
already holds real organizations.
