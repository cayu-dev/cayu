# Stability policy

This is Cayu's adopted policy for which APIs are stable within 0.x, how they are
deprecated, when a release is a patch or a minor, and how release notes say
whether a release breaks anything. The maintainers adopted it on 2026-10-15.
The PyPI `Development Status` classifier is `4 - Beta`.

Cayu exports a lot: 19 names from the root `cayu` package and about 3,800 names
across the 29 public packages in
[public-api-packages.json](public-api-packages.json). This policy does not
promise all of that. It promises a small core, gives durable and wire formats
version rules, and labels the rest as provisional or experimental.

## Stability tiers

| Tier | What users can rely on |
| --- | --- |
| **Stable** | No breaking change in a patch release. A breaking change in a minor release only after the [deprecation window](#deprecation-and-breaking-changes). New optional parameters, fields and enum members may be added in any release unless the tables below say otherwise. |
| **Versioned contract** | A durable or wire format with its own version number. It changes only in a minor release, with a version bump, a forward migration or upgrade path, and an entry under `Breaking changes`. Additive changes that keep the version may ship in patches. |
| **Provisional** | Public and supported, with tests and docs. It may change in any minor release, with a release-note entry, but without a guaranteed deprecation window. |
| **Experimental** | Opt-in. It may change or disappear in any release, including a patch. |
| **Internal** | Anything whose module or name starts with `_`, anything not declared in a public package's `_exports.py`, test fixtures and qualification fixtures. No guarantees. |

"Breaking" means: a stable name is removed or renamed, a parameter or field is
removed, renamed, retyped or made required, a default changes in a way that
changes behavior or cost, an enum value or event type string changes meaning,
or code that worked against the previous release raises or behaves differently
without the application opting in.

### Stable

These are the parts every application and generated project touches.

| Surface | Where it lives | What is promised |
| --- | --- | --- |
| Root exports | `cayu`: `Agent`, `AgentSpec`, `CayuApp`, `CayuConfig`, `Message`, `Event`, `EventType`, `Tool`, `ToolSpec`, `ToolContext`, `ToolEffect`, `ToolResult`, `RunRequest`, `ResumeRequest`, `RunOutcome`, `StructuredOutputResult`, `SessionStatus`, `run_to_completion`, `__version__` | Every name stays importable from `cayu` with the same meaning. The list is explicit in `_exports.py` and the generated `__init__.pyi`. |
| Agent definition | `AgentSpec` (`name`, `model`, `provider_name`, `system_prompt`, `workflow_tool_names`, `metadata`, `provider_options`, `thinking`), `Agent` | Existing fields keep their names, types and defaults. `authoring_state` is provisional. |
| Run and resume requests | `RunRequest`, `ResumeRequest` in `cayu.sessions` | Fields that applications set (`agent_name`, `messages`, `session_id`, `target`, `limits`, `budget_limits`, `max_steps`, `metadata`, `labels`, `structured_output`, `thinking`, `retry_policy`, `execution_deadline`) keep their meaning. Task-worker handoff fields (`task_worker_id`, `task_lease_expires_at`, `task_handoff_id`) are runtime plumbing and are provisional. |
| Run outcomes | `RunOutcome`, `StructuredOutputResult`, `SessionStatus`, `run_to_completion` | Status values and outcome fields keep their meaning. New statuses are added only in a minor release. |
| `CayuApp` core | Construction with `CayuConfig`, `register_agent`, `register_provider`, `register_environment`, `get_agent`, `list_agents`, `run`, `resume`, `interrupt_session`, `fork_session`, `compact_session`, `resolve_tool_approval`, `resolve_user_input`, `recover_incomplete_sessions`, `recover_abandoned_sessions`, `get_session_usage`, `get_session_cost`, `create_task`, `dispatch`, `aclose` | These methods keep their signatures and behavior. The other public `CayuApp` methods (collaboration, producers, clarifications, peer content, context views, execution snapshots and so on) are provisional until they are promoted by name. |
| Tool contract | `Tool`, `ToolSpec`, `ToolContext`, `ToolResult`, `ToolEffect` | `Tool.run` and `ToolSpec` keep their shape. `ToolEffect` keeps exactly the meanings of `none`, `idempotent` and `external`, because recovery after a crash depends on them. `ToolContext` fields that tools read (`session_id`, `agent_name`, `idempotency_key`, `workspace`, `artifact_store`, `metadata`) keep their meaning. |
| Tool policy contract | `cayu.tools`: `ToolPolicy`, `ToolPolicyRequest`, `ToolPolicyResult`, `ToolPolicyDecision`, `AllowAllToolPolicy`, `StaticToolPolicy`, `AlwaysRequireApprovalToolPolicy`, `GuardedToolPolicy` | Decisions (allow, deny, require approval) and the request and result fields keep their meaning. |
| Spend and run limits | `cayu.budgets`: `RunLimits`, `BudgetLimit`, `BudgetPolicy` | Limits keep their meaning and enforcement point. Price data is not part of the promise; it is refreshed in patches. |
| Shipped stores | `cayu.sessions.InMemorySessionStore`, `cayu.tasks.InMemoryTaskStore`, `cayu.storage`: `SQLiteSessionStore`, `PostgresSessionStore`, `SQLiteTaskStore`, `PostgresTaskStore`, `open_application_stores`, `configured_database_url` | Constructors and the documented store operations used by applications stay stable. Implementing your own store is provisional (see below). |
| Messages and events | `Message` and its content parts; `Event`, `EventType` | Event type strings never change meaning. New event types may be added in any release, and consumers must ignore types they do not know. The event envelope (`type`, ids, timestamps) is stable; payload fields are additive only. |
| Built-in providers | `cayu.providers`: `ModelProvider`, `OpenAIProvider`, `AnthropicProvider`, `BedrockProvider`, `VertexProvider`, `ChatCompletionsProvider` | Class names and required constructor arguments stay. Provider-specific options, retry tuning and bundled model metadata are provisional. |

### Versioned contracts

| Contract | Version source | Rule |
| --- | --- | --- |
| Storage schema | `cayu.storage.migrations`: `LATEST_REVISION`, `MIN_SUPPORTED_REVISION`, the `cayu_schema_migrations` table | Migrations are forward-only and run only through `cayu storage migrate`; startup never runs DDL ([ADR 0001](adr/0001-database-storage-and-migrations.md)). Additive revisions may ship in patches, because an existing database still validates against the new binary. A breaking revision, one that raises the compatibility floor (`MIN_SUPPORTED_REVISION`), ships only in a minor release, needs `--acknowledge-breaking REVISION`, and is listed under `Breaking changes` with its rollback consequence. |
| Execution profiles | `cayu.runtime` execution-profile identity | Existing sessions continue across a patch upgrade without an `ExecutionProfilePolicy`. Any profile change that needs application adoption ships only in a minor release and is listed under `Breaking changes`. |
| CLI JSON output | `schema_version` in the JSON of `cayu session`, `cayu task`, `cayu storage`, `cayu check`, `cayu inspect`, `cayu doctor`, `cayu diagnostics`, `cayu execution-profile`, `cayu recovery`, `cayu eval` and `cayu guide` | Within one `schema_version`, fields are only added. Removing, renaming or retyping a field bumps `schema_version` and ships only in a minor release. Table and text output is for people and is not covered. JSON without a `schema_version` (for example `cayu agent`) is provisional until it gets one. |
| Server HTTP contract | `SERVER_CONTRACT_VERSION` in `src/cayu/_server_contract_version.py` | Bumps only in minor releases and is listed under `Breaking changes` when older clients must not use the new server. The bundled dashboard always matches. |
| App manifest and generator schema | `APP_MANIFEST_SCHEMA_VERSION` in `cayu.runtime.manifest` | Same rule as the server contract. |
| Cayu Cloud agent contract | `CLOUD_AGENT_CONTRACT_VERSIONS` in `cayu.cloud_reporting`, served at `{api}/cloud/contract` | Within a version, fields and values are only added and readers ignore those they don't know. A new version ships only in a minor release, under its own `{api}/cloud/v{N}` routes, and stopping an older one is listed under `Breaking changes`. See [the contract](runtime-contracts.md#cayu-cloud-agent-contract). |
| Trusted proxy identity | `PROXY_IDENTITY_TOKEN_TYPE` in `cayu.server.proxy_identity`, the token's JWS `typ` | Within v1, claims are only added and servers ignore those they don't use. A change a v1 server would misread uses a new `typ`. See [the contract](runtime-contracts.md#trusted-proxy-identity). |

### Provisional

Public, supported, documented, and still moving. These may change in any minor
release with a release-note entry.

| Surface | Reason |
| --- | --- |
| Everything else in the public packages, including `cayu.collaboration`, `cayu.memory`, `cayu.evals`, `cayu.snapshots`, `cayu.environments`, `cayu.runners`, `cayu.workspaces`, `cayu.delivery`, `cayu.egress`, `cayu.proxies`, `cayu.vaults`, `cayu.verification`, `cayu.workflows`, `cayu.mcp`, `cayu.observability`, `cayu.artifacts`, `cayu.context` and most of `cayu.runtime` | Large and still changing. Promoting a capability to stable is a deliberate, named decision recorded on this page, not a side effect of being exported. |
| `CayuApp` methods outside the core list | Same reason. |
| Implementing your own `SessionStore` or `TaskStore` | The contracts have many abstract methods, new storage revisions often add obligations, and the conformance suites live in the repository tests rather than in `cayu.testing`. |
| Generated project layout from `cayu new` | Generated code belongs to the application. `cayu check` reports outdated imports (`SCAFFOLD_IMPORT_REMOVED`), so changes have a migration path, but the layout is still evolving. |
| `cayu.testing` helpers | Still being shaped. |
| Provider options, bundled model catalog and price book | Provider APIs change outside Cayu's control; catalog and price refreshes ship in patches. |
| Cayu Cloud CLI commands (`cayu cloud ...`) | Tied to a hosted service that is still changing. |

### Experimental

| Surface | Where it is labelled |
| --- | --- |
| `cayu.experimental` (TypeSafe / Jev) | Package docstring; [typesafe.md](typesafe.md) |
| `cayu.extensions.runners`, `cayu.extensions.egress` | Package docstring; [build-a-runner.md](build-a-runner.md#extension-seams-experimental) |
| `OpenAISubscriptionProvider` | [openai-subscription.md](openai-subscription.md) |
| Eval trial pacing control | [evals.md](evals.md) |

This page is the authority for the tiers. A surface not named here follows the
tier of its package: provisional for the public packages, experimental for
`cayu.experimental` and `cayu.extensions`, internal for everything else.

## Deprecation and breaking changes

Rules for stable surfaces within 0.x:

1. **Deprecate before removing.** A stable name, parameter or behavior is first
   deprecated in a minor release. The old form keeps working and emits a
   `DeprecationWarning` that names the replacement. The release notes list it
   under `Deprecations`.
2. **Window.** The old form keeps working for at least one full minor release
   after the one that deprecated it, and for at least 30 days. Both conditions
   must hold before it is removed.
3. **After removal.** Removed imports keep failing with an `ImportError` that
   names the replacement, using the existing replacement inventory
   ([public-api-imports.json](public-api-imports.json) and
   `cayu._import_migration`).
4. **Exceptions.** A security fix, or a fix for data loss or duplicated side
   effects, may break a stable surface without the window, in any release. The
   `Breaking changes` entry says so and explains why.
5. **Provisional surfaces** need no window, but every breaking change still
   gets a `Breaking changes` entry.

How breaking changes are announced:

- Every new release section in [release-notes.md](release-notes.md) has a
  `### Breaking changes` heading, with "None." when there are none (see
  [release-notes format](#release-notes-format)).
- A deprecation that is due for removal in the next minor release is repeated
  in the notes of the release before it, so nobody learns about a removal from
  the removal itself.
- Breaking storage revisions keep the existing operator protections:
  `cayu storage status`, a restorable backup, and `--acknowledge-breaking
  REVISION`.

## Release cadence

There is no fixed release train. Minor releases ship when their changes are
ready.

A release that carries only fixes or purely additive changes is a **patch
release** (`0.N.x`), never a minor bump. Additive changes include new optional
parameters, new exports, new event types, additive storage revisions, model
catalog and price updates, and dependency security floors.

**A patch release never**:

- breaks a stable surface or a versioned contract;
- raises the storage compatibility floor (`MIN_SUPPORTED_REVISION`);
- bumps the server contract, the app manifest schema or a CLI JSON
  `schema_version`;
- requires execution-profile adoption for existing sessions;
- changes a default in a way that changes behavior or cost for existing
  applications.

For example, `0.10.1` turned on prompt caching by default for Anthropic and
Vertex. That changes cost for existing applications, so under this policy it
would have shipped in a minor release, listed under `Breaking changes`.

Only **minor releases** (`0.N.0`) carry breaking changes, deprecations and
removals, new versions of versioned contracts, and default changes.

### v0.16.1 exception

The maintainer approved one explicit exception for v0.16.1: restoring long-running
handler and custom retry-backoff behavior after the implicit time-budget
regressions. New implicit task event budgets keep their attempt cap without an
elapsed-time cap, and custom retry policies whose requested delays exceed the
implicit 300-second budget keep their requested backoff. Explicit limits and
budgets already recorded for admitted events remain enforced.

These changes are listed under `Breaking changes` in the v0.16.1 release notes,
with instructions for applications that need fixed time limits. This exception
does not change the patch-release rules for subsequent releases.

## Release-notes format

Edit [release-notes.md](release-notes.md) only during release preparation; ordinary
PRs leave it unchanged and describe changes and upgrade steps in their PR descriptions.
Every new release section (including a temporary `## Unreleased` draft during
release preparation) has this shape:

```markdown
## vX.Y.Z

Upgrade: one or two sentences. Who needs to do anything, and what. Then a
compatibility line: storage revision, compatibility floor, server contract,
manifest schema, and whether each changed.

### Breaking changes

None.

### Deprecations

(Only when there are any.)

### Changes

- ...

### Upgrade

(Detailed steps, only when the summary is not enough.)
```

The upgrade summary and the `### Breaking changes` heading are required. The
other headings are recommended.

`scripts/verify_release_state.py` enforces the required parts. `## Unreleased`
and every release section newer than `v0.11.4`, the last release published
before this policy, must:

- start with a paragraph that begins with `Upgrade:` followed by text;
- contain exactly one `### Breaking changes` heading;
- have text under that heading, "None." when nothing breaks.

Sections for `v0.11.4` and older releases keep the format they were published
with. The check depends only on the version in each heading, not on which tags a
checkout has, so it applies the same way before tagging, when the tagged release
is published, and in a shallow clone. The release runbook runs the verifier on
the release commit before tagging ([releasing](releasing.md)). GitHub release
bodies use the extracted section verbatim, so they show the same headings.

### Example

The published `v0.11.4` entry predates this format and stays as it is. Written
in this format, it would read:

```markdown
## v0.11.4

Upgrade: patch release; no code, storage or configuration changes are needed
when upgrading from 0.11.x. Storage revision 118 and compatibility floor 115,
server contract 49 and manifest schema 17 are all unchanged.

### Breaking changes

None.

### Changes

- Accept an already-owned `RunLimits` scaffold import while preserving the
  qualification customizer's drift checks.
- Restore release-state verification for historical notes and prevent reuse of
  an already-published package version.
```

A release that breaks something lists each break. `v0.9.0`, abridged from its
`Upgrade from v0.8.1` section, would have started like this:

```markdown
## v0.9.0

Upgrade: stop 0.8.x writers, back up the store and run `cayu storage migrate
--acknowledge-breaking 115` before starting 0.9.0 writers. Regenerate HTTP
clients and upgrade the dashboard with the server. Storage revision 115
(was 114), compatibility floor 115 (was 111), server contract 48 (was 47),
manifest schema 17 (unchanged).

### Breaking changes

- Storage revision 115 is a breaking writer boundary. 0.8.x writers must not
  run against an upgraded store; rolling back needs a database restore.
- Server contract 48. Clients built for contract 47 must not treat a 0.9.0
  server as schema-compatible.
```

## Toward 1.0

A 1.0 release is not planned or scheduled. The following is what would need to
be true before a 1.0 of the stable core is considered. Provisional and
experimental surfaces would keep their labels after 1.0.

1. The policy has held for at least three consecutive minor releases: every
   breaking change to a stable surface went through the deprecation window or
   was a documented exception, and every release had a `Breaking changes`
   heading.
2. The storage compatibility floor has not moved for the last two minor
   releases.
3. CI migrates SQLite and Postgres stores written by earlier published releases
   to the current one, covering every store format, and resumes a session that
   was paused on an approval before the upgrade.
4. A test fails when a stable name or signature changes without a deprecation
   entry.
5. This policy is published on cayu.dev and linked from the README, the PyPI
   project page and `llms.txt`.

After a 1.0, breaking changes to stable surfaces would wait for the next major
version, and deprecations would still follow the rules above.
