# Memberships and the membership access policy

`cayu.memberships` is the part that decides, from an application's own memberships, who
sees which conversations, tasks and artifacts. The application decides who belongs where;
Cayu enforces what that allows, at every read, listing and background step. `cayu new
--preset b2b` and `cayu new --with organizations` assemble it into a project, and every
piece also works on its own.

It has organizations, members and roles only. Whether an application groups people inside
an organization at all, and how (teams, projects, client matters), is the application's
choice: [Groups of your own](#groups-of-your-own) shows how to add a grouping with a policy of
its own over the same directory.

| Piece | What it does |
| --- | --- |
| `Membership`, `AccessSnapshot`, `MembershipDirectory` | A membership of an organization; a person's memberships with the revision they were read at; the seam a policy reads them through (`current(subject)`) |
| `InMemoryMembershipDirectory` | Memberships in memory, for tests and trying the policy without a database |
| `PostgresMemberships` | The directory over the application's own PostgreSQL tables, and the only writes they need: `add`, `remove`, `change_as_admin` |
| `MembershipAccessPolicy` | A `ResourceAccessPolicy` that turns current memberships into Cayu grants |
| `conversation_labels`, `Sharing`, `SHARING_LABELS` | The labels a new conversation takes: personal or organization sharing |
| `check_membership_tables` | Checks the tables are the shape `PostgresMemberships` relies on |
| `cayu memberships list\|add\|remove` | The same store from the command line |

## The policy

```python
from cayu import CayuApp
from cayu.memberships import (
    InMemoryMembershipDirectory,
    Membership,
    MembershipAccessPolicy,
    conversation_labels,
)

directory = InMemoryMembershipDirectory({"alice": [Membership("acme")]})
app = CayuApp(resource_access_policy=MembershipAccessPolicy(directory))

access = await app.access("alice")
labels = conversation_labels(subject="alice", membership=Membership("acme"), sharing="organization")
```

Each person sees their own conversations and those shared with an organization they belong
to, and nothing of another organization. A person creates only their own conversations,
shared as `personal` or `organization`, inside their organizations. Only the owner continues,
changes or deletes a conversation. A conversation of theirs with another `sharing` (one a
policy of the application's own allowed, or an older project shared with a department) stays
visible to its owner alone. The sharing labels (`organization`, `owner`, `sharing`) can't be
changed on an existing conversation. Tasks and artifacts follow the same rules, except that
the owner creates them in any conversation of their own, whatever its sharing: work inside
the conversation takes its labels, and nobody else sees it unless it is shared with their
organization.

The policy refuses a person with more than `max_memberships` memberships (`ValueError`),
which one stored grant can't hold.

`MembershipAccessPolicy(directory, *, authority="b2b-membership-v1",
decision_lifetime=timedelta(seconds=60), max_memberships=16, knowledge=False)`:

- `authority` is stored with every piece of work Cayu admits, and Cayu refuses work whose
  authority differs from the policy's. Keep the default unless you mean to strand queued
  work; it is the string generated projects have always used.
- `decision_lifetime` is the longest Cayu uses one decision. It asks again at every
  protected step anyway, and every membership change raises the revision, so a decision
  read before a change is refused.
- `max_memberships` bounds one person's memberships. Sixteen keeps a grant with the longest
  IDs well inside the 64 KiB Cayu stores with each admitted piece of work.
- `knowledge=True` grants knowledge as well: a member reads, creates and revises knowledge
  entries labelled with one of their organizations, `organization` can't be changed on an
  entry, and nobody deletes through the grant. Off, the grant has no knowledge, as before,
  so a project without knowledge gets exactly the grant it always had.

### Knowledge scoped by organization

Turning on `knowledge=True` is half of it. The other half is the knowledge scope the store is
built with:

```python
from cayu.storage import KnowledgeAccessScope, KnowledgeStatus
from cayu.tools import RememberKnowledgePolicy, RememberKnowledgeTool

scope = KnowledgeAccessScope(
    allowed_namespaces=["project:acme-desk:agent:assistant"],
    allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
    session_label_keys=("organization",),
)
remember = RememberKnowledgeTool(
    policy=RememberKnowledgePolicy(
        default_namespace="project:acme-desk:agent:assistant",
        copy_session_labels=("organization",),
    )
)
```

`session_label_keys` pins every knowledge read and write inside a conversation to that
conversation's own `organization` label, the one Cayu verified when it admitted the
conversation. The store applies it with the grant, on every get, list, search and write, so
recall, the knowledge tools and the pending entries recall offers are all filtered, and a
tool that forgets to filter still can't reach another organization. A person who belongs to
two organizations recalls only the current conversation's organization's knowledge.
`copy_session_labels` labels each note from the conversation; the store refuses a note with
no `organization`, or another one's, whatever the model asked for. Outside a conversation
(`app.access(subject).knowledge(...)` in the web app) the grant alone decides, so a member sees
all their organizations' knowledge. Operator and background work outside any member's
binding isn't narrowed: curate one organization at a time with `cayu knowledge list --status
pending --label organization=ID`. What it writes is checked, though: an active or pending
entry written through a scope with `session_label_keys` must carry each of those labels,
since no conversation could read it otherwise, so the store refuses one without
`organization` (`KnowledgeAccessDenied`, naming the label to add). An entry stored without it
can still be revised to add it, or archived. For knowledge you mean to keep out of every
conversation, write it through a store or scope without `session_label_keys`.

Never take sharing labels from a request: build them with `conversation_labels` from the
signed-in person and a membership you read yourself. Cayu refuses any the person may not
create.

## Memberships in PostgreSQL

`PostgresMemberships(connect, *, schema="app")` reads and writes the application's own
tables. `connect` returns a DB-API connection to PostgreSQL with autocommit off and no
transaction open (psycopg, or a pool checkout such as SQLAlchemy's `engine.raw_connection`);
the store commits or rolls back each operation and closes the connection, returning it to
its pool. Calls run in a worker thread.

```python
from functools import partial

import psycopg

from cayu.memberships import MembershipAccessPolicy, PostgresMemberships, RemoveMember

memberships = PostgresMemberships(partial(psycopg.connect, database_url))
policy = MembershipAccessPolicy(memberships)

await memberships.add("alice", "acme", role="admin")
await memberships.change_as_admin("alice", "acme", RemoveMember("carol"))
```

- `current(subject)` reads the person's memberships and the access revision in one
  REPEATABLE READ snapshot, so they always agree. It makes the store a `MembershipDirectory`.
- `of_subject(subject)` and `of_organization(organization)` list `Member` rows (subject,
  organization, role).
- `add(subject, organization, *, role="member")` adds a member, or gives one another role.
  `remove(subject, organization)` removes one and says whether they were a member.
- `change_as_admin(caller, organization, change)` makes a `SetMember` or `RemoveMember`
  change for a signed-in person. In one transaction it locks the organization's admins'
  rows, checks the caller is one of them (`NotAnAdmin`), makes the change, keeps at least
  one admin (`LastAdmin`) and raises the revision.
- Every change raises the access revision in the same transaction. No change leaves an
  organization that has an admin without one (`LastAdmin`); an organization with no admin
  yet still takes members. An organization that doesn't exist is `UnknownOrganization`. All
  three are `MembershipChangeRefused`, and nothing was changed.

### The tables

The tables are the application's: it creates and migrates them, and they can carry columns
of its own. The store relies on this shape, which is exactly what generated migrations 0002
and 0003 create, so a generated project needs no migration to use it:

```sql
CREATE TABLE app.organizations (
    id varchar(64) PRIMARY KEY,
    name varchar(200) NOT NULL,
    CHECK (strpos(id, '/') = 0)
);
CREATE TABLE app.memberships (
    id serial PRIMARY KEY,
    subject varchar(200) NOT NULL,
    organization_id varchar(64) NOT NULL REFERENCES app.organizations (id) ON DELETE CASCADE,
    role varchar(16) NOT NULL DEFAULT 'member',
    UNIQUE (subject, organization_id),
    CONSTRAINT memberships_role_check CHECK (role in ('member', 'admin'))
);
CREATE TABLE app.access_revision (
    id serial PRIMARY KEY,
    revision bigint NOT NULL
);
```

The organizations table is the application's own beyond its `id`: the store only relies on
the foreign key from `memberships`. Generated projects keep organization IDs free of `/` so
each is one segment of their routes' paths. Other columns and tables, such as a grouping of
the application's own, are left alone, and projects generated by Cayu 0.16.1 to 0.20.0 keep
their `departments` table and `department_id` column, which the store neither reads nor
writes.

`check_membership_tables(connection, *, schema="app")` checks what the store relies on: the
columns it reads and writes, the unique (subject, organization) and revision keys, the
foreign key to the organizations and the role check. It raises `MembershipTablesInvalid`, whose
`problems` name each difference. Run it in a test against your migrated database.

## From the command line

`cayu memberships` runs the same store against the application's database: `--postgres`,
else `CAYU_DATABASE_URL`, else `[tool.cayu.session_store]`. It checks the tables first.

```console
cayu memberships add alice acme --role admin
cayu memberships list --organization acme
cayu memberships list --subject alice --json
cayu memberships remove carol acme
```

It refuses removing or demoting an organization's last admin, and an organization that
doesn't exist. `--schema` names another schema; `--json` prints JSON.

## In generated projects

`cayu new --preset b2b` and `cayu new --with organizations` wire these pieces into the
project: `database/memberships.py` builds `PostgresMemberships` over the application's own
connection pool, `policies/access.py` builds `MembershipAccessPolicy` over it, and the member
routes call `change_as_admin`. The tables, sign-in, roles (`app_role`), routes and pages stay
the project's. With knowledge (`cayu new --with organizations` on the agent preset), the
policy is `MembershipAccessPolicy(memberships(), knowledge=True)`, `knowledge/retrieval.py`
pins the knowledge scope to the conversation's `organization`, and `remember_knowledge` copies
it from the conversation, as above.

Projects generated before `cayu.memberships` carry their own copies. `cayu check` notes each
`policies/access.py` or `database/memberships.py` that is still exactly the copy a release
generated (`MEMBERSHIPS_GENERATED_COPY`, info) and leaves edited ones alone; `cayu guide
b2b#move-to-cayumemberships` walks through the move. It needs no database migration, and
queued work stays admitted for the person who started it because the authority is
unchanged. Those projects shared conversations with departments too, which this library
doesn't: such a conversation becomes visible only to its owner, so access narrows and never
widens.

## Your own pieces

- Memberships somewhere else (an identity provider's groups, another service): write a
  class with `async def current(self, subject) -> AccessSnapshot` and keep the policy. Raise
  the revision whenever memberships change.
- Different sharing rules: write your own `ResourceAccessPolicy`
  ([resource access](resource-access.md)) and keep the directory and the store.

### Groups of your own

Teams, projects or client matters inside an organization are the application's: it keeps
them in its own tables and adds them to the grant with a policy of its own, over the same
directory. This one starts from `MembershipAccessPolicy`'s decision and adds conversations
shared with a team the person is in:

```python
from dataclasses import replace

from cayu.memberships import MembershipAccessPolicy
from cayu.resource_access import ResourceAccessDecision, ResourceAccessPolicy
from cayu.sessions import SessionAccessRule, SessionAccessSelector


def rule(**labels: list[str]) -> SessionAccessRule:
    return SessionAccessRule(
        selectors=tuple(
            SessionAccessSelector(key=key, values=tuple(values)) for key, values in labels.items()
        )
    )


class Read:
    """A directory that answers with one snapshot already read."""

    def __init__(self, snapshot):
        self._snapshot = snapshot

    async def current(self, subject):
        return self._snapshot


class TeamAccessPolicy(ResourceAccessPolicy):
    def __init__(self, directory, teams) -> None:
        self._directory = directory
        # The application's own: async teams(subject) -> [(organization, team), ...]
        self._teams = teams

    @property
    def authority(self) -> str:
        return "acme-teams-v1"

    async def resolve(self, subject: str) -> ResourceAccessDecision:
        # One read of the memberships, for the library's grant and the teams alike.
        snapshot = await self._directory.current(subject)
        decision = await MembershipAccessPolicy(Read(snapshot), authority=self.authority).resolve(
            subject
        )
        organizations = {item.organization for item in snapshot.memberships}
        # Only teams of organizations the person still belongs to, each rule naming its
        # organization, so a team of the same name elsewhere never matches.
        teams = [(org, team) for org, team in await self._teams(subject) if org in organizations]
        if not teams:
            return decision
        read = tuple(rule(organization=[o], team=[t], sharing=["team"]) for o, t in teams)
        create = tuple(
            rule(organization=[o], owner=[subject], team=[t], sharing=["team"]) for o, t in teams
        )
        grant = decision.grant
        added = {
            family: replace(scope, read=(*scope.read, *read), create=(*scope.create, *create))
            for family, scope in (
                ("sessions", grant.sessions),
                ("tasks", grant.tasks),
                ("artifacts", grant.artifacts),
            )
            if scope is not None
        }
        return replace(decision, grant=replace(grant, **added))
```

A conversation labelled `organization=acme`, `sharing=team` and `team=finance` is then read by
the members of acme's finance team, and still continued only by its owner. Each rule names its
organization: a team ID is the application's and may repeat across organizations, so a rule on
the team alone would let globex's finance team read acme's. Teams of an organization the person
no longer belongs to grant nothing, and Cayu protects the `team` label like the others, since a
rule names it. Each team is one more rule, so keep a person's teams bounded, as
`max_memberships` bounds their organizations.

Raise the access revision (the `access_revision` row) in the same transaction as any change to
a team, as a membership change does, so Cayu refuses a decision read before it. The example
names an `authority` of its own; an application that already runs `MembershipAccessPolicy` with
queued work keeps `b2b-membership-v1`, since Cayu refuses work admitted under another
authority. `tests/memberships/test_memberships_policy.py` runs this example.
