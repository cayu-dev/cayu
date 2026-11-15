# Memberships and the membership access policy

`cayu.memberships` is the part that decides, from an application's own memberships, who
sees which conversations, tasks and artifacts. The application decides who belongs where;
Cayu enforces what that allows, at every read, listing and background step. `cayu new
--preset b2b` and `cayu new --with organizations` assemble it into a project, and every
piece also works on its own.

| Piece | What it does |
| --- | --- |
| `Membership`, `AccessSnapshot`, `MembershipDirectory` | A membership of an organization and at most one of its departments; a person's memberships with the revision they were read at; the seam a policy reads them through (`current(subject)`) |
| `InMemoryMembershipDirectory` | Memberships in memory, for tests and trying the policy without a database |
| `PostgresMemberships` | The directory over the application's own PostgreSQL tables, and the only writes they need: `add`, `remove`, `change_as_admin` |
| `MembershipAccessPolicy` | A `ResourceAccessPolicy` that turns current memberships into Cayu grants |
| `conversation_labels`, `Sharing`, `SHARING_LABELS` | The labels a new conversation takes: personal, department or organization sharing |
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

directory = InMemoryMembershipDirectory({"alice": [Membership("acme", "acme/finance")]})
app = CayuApp(resource_access_policy=MembershipAccessPolicy(directory))

access = await app.access("alice")
labels = conversation_labels(
    subject="alice", membership=Membership("acme", "acme/finance"), sharing="department"
)
```

Each person sees their own personal conversations, those shared with a department they
belong to and those shared with an organization they belong to, and nothing of another
organization. Only the owner continues, changes or deletes a conversation. A department
conversation is decided by its department alone. The sharing labels (`organization`,
`department`, `owner`, `sharing`) can't be changed on an existing conversation. The same
grant covers sessions, tasks and artifacts.

A department ID starts with its organization's ID and `/`, as in `acme/finance`. The policy
refuses a membership whose department doesn't (`ValueError`), as it refuses a person with more
than `max_memberships` memberships, which one stored grant can't hold.

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

await memberships.add("alice", "acme", department="acme/finance", role="admin")
await memberships.change_as_admin("alice", "acme", RemoveMember("carol"))
```

- `current(subject)` reads the person's memberships and the access revision in one
  REPEATABLE READ snapshot, so they always agree. It makes the store a `MembershipDirectory`.
- `of_subject(subject)` and `of_organization(organization)` list `Member` rows (subject,
  organization, department, role).
- `add(subject, organization, *, department=None, role="member")` adds a member, or moves
  one to another department or role. `remove(subject, organization)` removes one and says
  whether they were a member.
- `change_as_admin(caller, organization, change)` makes a `SetMember` or `RemoveMember`
  change for a signed-in person. In one transaction it locks the organization's admins'
  rows, checks the caller is one of them (`NotAnAdmin`), makes the change, keeps at least
  one admin (`LastAdmin`) and raises the revision.
- Every change raises the access revision in the same transaction. No change leaves an
  organization that has an admin without one (`LastAdmin`); an organization with no admin
  yet still takes members. A department of another organization, or a department or
  organization that doesn't exist, is `UnknownDepartment`. All three are
  `MembershipChangeRefused`, and nothing was changed.

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
CREATE TABLE app.departments (
    id varchar(64) PRIMARY KEY,
    organization_id varchar(64) NOT NULL REFERENCES app.organizations (id) ON DELETE CASCADE,
    name varchar(200) NOT NULL,
    UNIQUE (organization_id, id),
    CHECK (starts_with(id, organization_id || '/'))
);
CREATE TABLE app.memberships (
    id serial PRIMARY KEY,
    subject varchar(200) NOT NULL,
    organization_id varchar(64) NOT NULL REFERENCES app.organizations (id) ON DELETE CASCADE,
    department_id varchar(64),
    role varchar(16) NOT NULL DEFAULT 'member',
    UNIQUE (subject, organization_id),
    FOREIGN KEY (organization_id, department_id) REFERENCES app.departments (organization_id, id),
    CONSTRAINT memberships_role_check CHECK (role in ('member', 'admin'))
);
CREATE TABLE app.access_revision (
    id serial PRIMARY KEY,
    revision bigint NOT NULL
);
```

`check_membership_tables(connection, *, schema="app")` checks what the store relies on: the
columns it reads and writes, the unique (subject, organization) and revision keys, the
department foreign key and the role check. It raises `MembershipTablesInvalid`, whose
`problems` name each difference. Run it in a test against your migrated database.

## From the command line

`cayu memberships` runs the same store against the application's database: `--postgres`,
else `CAYU_DATABASE_URL`, else `[tool.cayu.session_store]`. It checks the tables first.

```console
cayu memberships add alice acme --department acme/finance --role admin
cayu memberships list --organization acme
cayu memberships list --subject alice --json
cayu memberships remove carol acme
```

It refuses removing or demoting an organization's last admin, and a department of another
organization. `--schema` names another schema; `--json` prints JSON.

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
queued work stays admitted because the authority and grant are unchanged.

## Your own pieces

- Memberships somewhere else (an identity provider's groups, another service): write a
  class with `async def current(self, subject) -> AccessSnapshot` and keep the policy. Raise
  the revision whenever memberships change.
- Different sharing rules: write your own `ResourceAccessPolicy`
  ([resource access](resource-access.md)) and keep the directory and the store.
