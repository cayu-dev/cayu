# Cayu generated applications

`cayu new` compiles a normalized application plan into ordinary explicit
Python. The generated tree is executable architecture for people and coding
agents; it is not a runtime plugin system, service locator, or authority grant.

`cayu new` refuses a parent directory that other users can write: one that is
world-writable without the sticky bit, or group-writable by a shared group. A
group-writable parent (the default home directory under umask 002) is accepted
when you own it and its group is your private group: named after you, your
primary group, with no other members and no other account using it as a
primary group. Cayu reads this from the local user and group databases; with a
directory service that does not enumerate its users (some LDAP or SSSD setups)
it cannot see every account, so prefer a parent that is not group-writable there.

`cayu new` warns when PyPI has a newer final Cayu release than the installed one
and prints the `uvx --from 'cayu==X.Y.Z' cayu new ...` command that scaffolds with
it. Attached to a terminal, it asks `Continue with <installed> anyway? [y/N]` and
stops by default. Without a terminal it warns on stderr and continues; `--json`
output then includes `newer_release` with `installed`, `latest` and `command`.
The check runs at most once a day, gives up after about two seconds, and is
silent when PyPI is unreachable. `CAYU_NO_VERSION_CHECK=1` turns it off.

## Convention

Normal agent, service, coding, and long-running presets share these ownership boundaries:

| Concern | Canonical home |
| --- | --- |
| Agent identity and registration | `agents/` |
| Prompt material | `prompts/` |
| Native model-callable capabilities | `tools/` |
| Exposure, authorization, execution, egress, budgets, and retries | `policies/` |
| Workspaces, runners, artifacts, knowledge bindings, and lifecycle | `environments/` |
| Deterministic orchestration | `workflows/` |
| Tasks, workers, approvals, completion, and recovery | `operations/` |
| Reviewed retrieval and curation | `knowledge/` |
| Context, recall, compaction, and memory attribution | `memory/` |
| Business rules | `domain/` |
| External protocols and MCP adapters | `integrations/` |
| Behavioral evidence | `evals/` |
| Event sinks and tracing | `observability/` |
| Final construction and registration only | `app.py` |

Implement in the owning module first, then connect it through
`agents/registration.py` and the composition root. Explicit imports and
`register_*` calls remain authoritative. Do not infer permission from placement,
prompts, scaffold metadata, or capability selection.

`[tool.cayu.scaffold]` records the convention version, preset, adapters,
capabilities, and minimal choice. It is source-controlled creation intent used
by compatible generators and read-only diagnostics. It never selects runtime
objects. Projects without this declaration remain freeform.

For declared convention projects, `cayu check --fail-on warning --json` reports
missing ownership seams, implementation collapsed into `app.py`, top-level
composition work, and agent registration that no longer originates from the
explicit registration module. Removing the contract is an intentional custom
layout migration, not a supported way to silence a finding.

`cayu check` never imports application code that failed the import-safety check
(or whose scaffold contract is too invalid for that check to run). It still lists
every source finding and adds `SCAFFOLD_APPLICATION_NOT_IMPORTED` naming the
checks that did not run. Any other source error, such as plan drift, no longer
stops the check: the application is loaded and checked as usual, and if its
factory fails, `SCAFFOLD_APPLICATION_LOAD_FAILED` is reported next to the source
findings that likely caused it.

The import check accepts inert application class hierarchies, including exception
classes and subclasses of a shared `Tool` base. Define bases before their uses
and use explicit named imports, such as `from tools.base import ProductTool`;
relative imports, import aliases, and explicit package re-exports are supported.
The checker reads local source without executing it to establish inheritance
safety, including explicit local import dependencies and their package
initializers. These dependencies must also be declarative: a helper import can
install subclass hooks after a class is defined. The proof is bounded to 64 local
source modules and fails closed beyond that limit. Rebindings, import cycles,
opaque descriptors, custom metaclasses,
class decorators on a base other than `@dataclass` with literal options, and
application `__init_subclass__` hooks remain
unproven and fail closed. Dynamic base expressions and class namespaces also
require a stronger proof than this static check supports. Keep lifecycle and
external work in methods or explicit builders. External roots of a local class
hierarchy must also have reviewed subclass-creation behavior: builtin exceptions,
`object`, the builtin value types (`tuple`, `list`, `dict`, `set`, `frozenset`,
`str`, `int`, `float`, `bytes`), `abc.ABC`, and Cayu's `Tool` are supported.
Enum-derived application
bases remain unproven because enum metaclasses can execute inherited member
initializers; mixing a local base with an enum has the same restriction.

Module and class bodies may also use these reviewed declarative forms, which
the check accepts without executing them:

- A class-level `spec = ToolSpec(...)` whose values are literals, `ToolEffect`
  members, and a nested literal `ExecutionProfileBehaviorIdentity(...)` or a
  module-level constant holding one. This is the shape `cayu generate tool`
  emits.
- `logging.getLogger(__name__)` or with a literal name.
- Dataclass fields `field(default=<literal>)`,
  `field(default_factory=list|dict|set|tuple|frozenset)` or
  `field(default_factory=lambda: ...)` with literal flags, and empty
  `frozenset()` or `tuple()` class defaults. A lambda body runs when an instance
  is created, not at import; a lambda with default arguments stays unproven.
- Fixed instants `datetime(...)` and `date(...)` with integer literal fields and
  `tzinfo=UTC` or `tzinfo=timezone.utc`, and `timedelta(...)` with literal fields.
- `Decimal("3")` or `Decimal(3)` with one string or integer literal, as a module
  constant, dataclass default or model default. Model fields may be annotated
  `Decimal` or `ZoneInfo`.
- `Path(...)` and `Path(...).resolve()`, for example `Path(__file__).resolve()`.
- Type aliases such as `Callable[..., AbstractContextManager[T]]` built from
  `typing`, `collections.abc` and the `contextlib` context-manager ABCs.
- `Enum`/`StrEnum`/`IntEnum`, `TypedDict` and `NamedTuple` declarations with
  annotations and literal values.
- `@functools.cache`, `@functools.lru_cache` and `@lru_cache(maxsize=...)` with
  literal options.

Environment reads at import (`os.environ[...]`, `os.environ.get(...)`,
`os.getenv(...)`) are deliberately unsupported, with `reason:
import_time_configuration`: read settings at call time in a
`configuration/settings.py` function, as the generated `configured_*` helpers
do, so tests and deployments can change them.

Anything else that runs at import is reported as `SCAFFOLD_IMPORT_SIDE_EFFECT`.
The finding names the construct and says why: `reason: import_time_effect` for a
call that clearly performs I/O, network, process or file work (for example
`subprocess.run(...)` or `Path(...).write_text(...)`), and
`reason: unsupported_expression` for a shape the check cannot prove inert (for
example an application function call such as `load(...)`, including one nested
inside a `ToolSpec`). Move either kind into a builder, lifecycle hook or tool
`run`. A module-level call to your own helper, such as
`RANKER = _helper_agent("ranker", ...)` returning an `AgentSpec`, is an application
function call too: write the `AgentSpec(...)` out at module level, or build it in a
function the factory calls.

Build `ZoneInfo(...)` values lazily, inside a function or the factory, not at
import: a zone missing from the system database is loaded from the `tzdata`
package, and a project-local `tzdata` module would then run while `cayu check`
imports the application. A module-level `ZoneInfo("...")` is reported as
`SCAFFOLD_IMPORT_SIDE_EFFECT`.

Every module in the application packages is checked, whether or not the factory
imports it at module level. The factory, a service target or a deployment check can
import modules lazily or by `module:attribute` name while `cayu check` runs, so a
finding in any of them stops the import.

Each cause is reported once, at the expression the check cannot prove. A class,
annotation or import that only depends on a failing definition, such as every
subclass of a model whose `Field` default cannot be proven or every module that
imports one with an import-time call, is not reported again; it still blocks
importing the application until the cause is fixed. A class that is inert itself
but cannot be inherited from, for example because it defines
`__init_subclass__`, is reported at that hook with
`reason: unproven_base_definition` and the number of classes that inherit from
it. `reason: unproven_class_base` remains for a base that is not a proven class
at all, such as a constant or a function.

Pydantic `BaseModel` declarations (including named import aliases) may register `@field_validator` and
`@model_validator` methods. Import these helpers explicitly from `pydantic`
(named aliases and `import pydantic as pd` are supported). Field names must be
literal strings; `mode` must be a supported literal string and `check_fields`
a literal boolean or `None`. Model validators require an explicit `mode` of
`before`, `after`, or `wrap`; field validators also accept `plain` and default
to `after`. Validator bodies run during validation, not registration. A wrap
model validator's `handler: ModelWrapValidatorHandler[Self]` parameter is
accepted. Models may also register `@model_serializer` (bare, or with literal
`mode` of `plain` or `wrap` and `when_used`) and `@field_serializer` with literal
field names and literal `mode`, `when_used` and `check_fields`. Pydantic builds a
schema from a serializer's return annotation while it declares the model, so that
annotation must be a proven model annotation; `return_type` stays unproven.

Model assignments may also use explicitly imported `ConfigDict`, `Field` and
`PrivateAttr` with literal data arguments. Numeric options and defaults may use
bounded arithmetic (`+`, `-`, `*`, and `**` with a small exponent) on numeric
literals and module constants, such as `8 * 1024**3`. `default_factory` and
`exclude_if` may be lambdas: Pydantic calls them during validation or
serialization, not while declaring the model, so only lambda default arguments
are evaluated and those stay unproven. `default_factory` may also name a builtin
collection, a proven local model or function, or a class or function imported
from Cayu. Other callbacks (for example `field_title_generator`) and computed
defaults remain unproven. Annotations may use `X | None`, subscripted generics
and quoted forward references directly in model fields or return annotations,
which are proven as the expression they contain. Exported type aliases containing
forward references are not proven: Pydantic resolves those references in the
importing model namespace, which can bind different types and schema hooks.
Unknown options, computed arguments, unpacking, shadowed/rebound imports,
project-local substitutes for Pydantic, opaque model field annotations,
explicit `__annotations__`, `__annotate__`, or `__annotate_func__` bindings,
and model construction/schema hooks
remain unproven. Inherited models retain these checks through explicit named
imports, aliases and package re-exports; literal `ConfigDict` configuration does
not prevent subclassing. Mixing models with ordinary application mixins remains
unproven because the model metaclass could activate inherited hooks. Attribute
and subscription base expressions such as `BaseModel.__mro__[0]` remain unproven.

Standard-library `TypeVar`, `ParamSpec`, and `TypeVarTuple` declarations accept a
single literal string name. `TypeVar` and `ParamSpec` also accept literal boolean
variance options. Bounds, constraints, computed arguments and unpacking remain
unproven. `ContextVar` construction accepts a literal string name and an optional
literal-data `default`. These process-wide identities can remain at module scope.
Standard `contextlib.contextmanager` and `asynccontextmanager` decorators on
functions and methods are declarations; their bodies are not executed by the
scanner. Named import aliases and module aliases are supported for these standard
library forms, while shadowed imports and project-local substitutes fail closed.

## Planning

Discover the package-shipped catalog before choosing a plan:

```console
uv run --no-sync cayu new --list-presets --json
uv run --no-sync cayu new --list-capabilities --json
uv run --no-sync cayu new --explain knowledge --json
```

Resolve the full plan without writing:

```console
uv run --no-sync cayu new my_agent --preset agent --provider neutral --dry-run --json
```

Dry-run and apply use the same normalized plan. Presets select a coherent
application shape; provider and execution flags select maintained adapters; `--with` and `--without` change only package-shipped selectable
capabilities. Extension-only concerns retain their canonical homes but cannot be
claimed active through a flag. Invalid combinations fail before target creation.

Service profiles require `tasks`; Docker coding profiles require `artifacts` for
their durable product results. Excluding either produces `CAPABILITY_REQUIRED`
before writing files. Coding includes artifacts in its declared defaults; local
coding can exclude artifacts, tasks, delegation, and human input independently.
An excluded store cannot be restored through a generated factory's injection seam.

For a normalized declared profile, `inspect` and `check` report
`SCAFFOLD_CAPABILITY_DRIFT` when constructed stores or built-in tool families
disagree with the capability selection. For agent starters with
knowledge, this also compares the starter proposal tool's effective coverage
(`approval_required` with `approvals`, `denied` without it), including maintained
policy compositions; independent extension approval policies do not imply that the
starter capability is enabled. `cayu inspect --json` includes both `manifest`
and `diagnostics` when drift is found and still exits non-zero. These are structural diagnostics, not
proof of provider access or arbitrary custom policy behavior. The default agent's
`search_knowledge` uses its configured project/agent namespace when the caller
omits the namespace; normal knowledge access and active-entry filters still apply.

`NAME` is always the project directory's basename and `--dir` is always its
existing parent. To inspect a maintained variant while changing an existing
project, create a disposable reference instead of passing a path as `NAME` or
running `cayu new` over the current repository:

```console
reference_parent="$(mktemp -d)"
uv run --no-sync cayu new my_agent_reference --agent-name my_agent --preset agent \
  --provider neutral --execution none \
  --dir "$reference_parent" --json
```

The reference is comparison material, not an automatic migration. Review the
owning-file diff and update the existing source plus `[tool.cayu.scaffold]`
explicitly. `cayu new`, `cayu check`, and `cayu inspect` never migrate a project.

All presets use the same application convention. Use `--preset` and `--execution`
to select the application and execution environment. `--preset long-running` is
the agent layout plus a worked example of approval, a waited-for reply,
reminder and deadline timers, a spend cap, and restart tests; see
`cayu guide long-running`.

## Durable data

Every preset selects its database at runtime, not at generation time. The storage
module (`configuration/storage.py`, or `configuration/coding_storage.py` for
coding) calls `open_application_stores(configured_database_url(), sqlite_path=...)`:
`CAYU_DATABASE_URL` selects PostgreSQL, and without it the stores use local SQLite
at an absolute path under the project (`data/cayu.db`, or `.cayu/runtime/cayu.db`
for coding). `[tool.cayu.session_store]` names that same local file for Cayu CLI
tooling, and `CAYU_DATABASE_URL` overrides it for the app and the CLI alike. Every
generated project depends on `cayu[postgres,server]`. `cayu new --database` is deprecated
and ignored. The service preset keeps its product operation records (tenant
authorization, claims, and settlement) in the same database with
`product_operations=True`, so several service processes can share PostgreSQL.

- Keep application-owned durable records, such as orders, cases, ledgers, and
  sync cursors, in the configured database. Application tables may share it with
  their own table prefix; Cayu reserves `cayu_` for its tables.
- `data/` locally and `/data` in a deployment hold files: artifacts, uploads, and
  fixtures. Do not open SQLite files there for durable application state. Cayu Cloud
  blocks such code at deploy time; acknowledge a rebuildable cache or index under
  `/data` with `[storage] local_files = [{ path = "...", reason = "..." }]` in
  `cayu-cloud.toml`.
- Deployments set `CAYU_DATABASE_URL` to a migrated PostgreSQL database, run
  `cayu storage migrate` as a deploy step, and set `CAYU_REQUIRE_POSTGRES=1` so
  every Cayu SQLite store refuses to open.
- `CAYU_DATABASE_POOL_MAX` (default 5) bounds the shared connection pool. Behind a
  transaction-pooling proxy such as PgBouncer, `CAYU_DATABASE_DIRECT_URL` gives the
  task-admission `LISTEN` connection a direct server address.
- Deployments of applications with automatic memory must set
  `CAYU_MEMORY_EVIDENCE_KEY`; the local `data/memory-evidence.key` is git-ignored
  and never reaches a deployment.

`cayu check` reports `SCAFFOLD_PLAN_DRIFT` with field `storage` when the storage
module constructs a fixed SQLite or Postgres store, or when
`[tool.cayu.session_store]` names anything other than the local SQLite file.
Projects generated before this convention may keep `database` in
`[tool.cayu.scaffold]`; it is ignored. Migrate them by replacing the storage module
with the one from a disposable `cayu new` reference.

## Explicit service extensions

Services can declare application-owned `artifacts`, `knowledge`, and `delegation`
extensions without changing preset or maintained authentication. The catalog's
`extension_presets` lists where this declaration is supported; `supported_presets`
continues to describe generator support. `cayu new --with artifacts --preset service`
therefore points to this manual extension path rather than generating unreviewed
service wiring. `cayu new --explain artifacts --json` exposes both contracts.

Keep the normalized generated `capabilities` and add a separate sorted, unique
`extensions` list in the existing `[tool.cayu.scaffold]` table. For example, an
otherwise default service with an explicit artifact extension declares:

```toml
[tool.cayu.scaffold]
convention = 1
preset = "service"
provider = "neutral"
execution = "none"
capabilities = ["approvals", "evals", "observability", "tasks"]
extensions = ["artifacts"]
```

Existing projects may omit `extensions` (equivalent to `[]`). If an earlier
attempt added an extension to `capabilities`, move just that entry to `extensions`.
Retain the project's actual provider, adapters, and generated capability choices.
To declare all three, use `extensions = ["artifacts", "delegation", "knowledge"]`.
Unknown, duplicate, unsorted, and unsupported declarations are errors.

Implement and review the public constructors in their canonical homes:

- **Artifacts:** register `ListArtifactsTool()` in `tools/registration.py` and
  construct the `LocalArtifactStore` and `Environment` in `environments/local.py`.
  Update the environment's explicit behavior identity with the implementation.
- **Knowledge:** construct the scoped store in `configuration/storage.py`, define
  access and retrieval scope in `knowledge/retrieval.py`, and wire the environment
  in `environments/local.py`. Register `ListKnowledgeTool`, `SearchKnowledgeTool`,
  `ReadKnowledgeTool`, and `RememberKnowledgeTool` together for one agent in
  `tools/registration.py`; authorize proposal effects in `policies/tools.py` and
  give the changed tool and policy behavior explicit versioned identities for recovery.
- **Delegation:** define and explicitly register child agents in `agents/`,
  construct `SubagentTool` and `SubagentResultTool` with the owning app and stores
  through `agents/registration.py`, and keep lifecycle work in `operations/`.
  Declare child targets, limits, result access, exposure, and effect policy explicitly.

Generated code contains only the capabilities that were selected; a disabled
concern leaves no guard or placeholder to flip. Add its wiring to the relevant
owning builders with their collaborators and identities, using the public
constructors above, or compare with a disposable `cayu new --with <capability>`
reference. Do not read scaffold metadata
at runtime to enable tools or stores. Retain import safety and composition-only
`app.py`; keep service routes, product authentication, tenant lookup, and operator
policy intact. Artifact visibility, knowledge namespaces, and child-result access
must follow authenticated application scopes. A project-wide knowledge namespace
is suitable only for intentionally shared knowledge, not tenant-private records.
An extension declaration does not establish tenant isolation for custom behavior.

Run `cayu check --deploy --fail-on warning --json` with the service's authentication
configuration and run its security tests plus tests for the new scope boundaries.
The checker compares the union of generated capabilities and declared extensions
against the constructed stores and complete tool families. An undeclared store or
tool remains drift; declaration alone also fails when constructors are absent or
a tool family is incomplete. Source ownership, import safety, and maintained
service/authentication checks still run. Metadata neither changes runtime objects
nor grants model exposure, execution authority, or provider access.

For `cayu cloud deploy`, run `uv lock` in the selected project root and include both
`pyproject.toml` and `uv.lock` in the uploaded source. Preflight checks the actual bundle,
including Git ignore rules; a local wheel build alone does not establish Cloud build
readiness. The deploy command does not generate locks or modify the source for you.

For a public service with a `[web]` process, `cayu cloud deploy` also runs the same
deploy check in-process before uploading and refuses (`deploy_check_failed`) when it
reports a `PUBLIC_SERVICE_*` finding, because `cayu serve` would refuse to start the
release. The check sees only the local environment, so set the service's authentication
configuration for the deploy as for `cayu check`, or pass `--skip-deploy-check` on purpose.

## Cloud build diagnostics

An identical `cayu cloud deploy .` may replay an earlier immutable attempt. Failed or
destroyed attempts whose recorded failure is automatically retryable receive one fresh
retry by default; the JSON `retry` receipt identifies both attempts. In-progress and
successful replays remain idempotent. Use `--no-retry-failed` to report the retained
attempt, or `cayu cloud deployment retry ID --application SLUG --idempotency-key KEY`
to submit a deliberate retry. Reuse KEY after an uncertain HTTP outcome. Source errors
still require corrected source, and expired source bundles require a new upload.

When `cayu cloud deploy` fails, inspect the structured `error.failure`: its code,
phase, repair hint, retry classification, and diagnostic evidence describe the
failure. Supported versioned diagnostics include a bounded redacted build excerpt,
attempt, exit code, and evidence availability. For example, missing build inputs
require fixing the source bundle and submitting a new revision; a temporary
infrastructure failure may permit retrying the same source.

Retrieve deployment-scoped evidence without platform credentials:

```console
cayu cloud deployment timeline DEPLOYMENT_ID --application AGENT_SLUG
cayu cloud deployment logs DEPLOYMENT_ID --application AGENT_SLUG
```

For servers supporting diagnostic pagination, use `--diagnostic-offset` with the
returned `next_diagnostic_offset` and optionally `--diagnostic-limit 20`. If evidence
is unavailable or unsupported, the deploy error retains the failed outcome and
provides a logs command when possible. A CLI upgrade cannot recover evidence the
server never recorded. Verify a later deployment succeeds before claiming recovery.

## Generator compatibility

`cayu generate tool` and `cayu generate slice` inspect the declared scaffold
contract. Convention projects update only delimited regions in
`agents/registration.py` and the narrow agent contract. Legacy generated
projects retain their `app.py` seams. A custom or drifted source shape produces
a reviewable conflict or manual action instead of an arbitrary rewrite.

Use this authoring loop:

```text
understand -> inspect -> plan -> change -> test -> eval -> exercise -> report evidence
```

After a change, run the exact commands emitted by `cayu new`; at minimum:

```console
uv run --no-sync cayu inspect --json
uv run --no-sync cayu check --fail-on warning --json
uv run --no-sync pytest
uv run --no-sync cayu eval run
```

Run application-constructing proof commands sequentially when they share the
generated local SQLite store. A first construction may initialize or migrate
that schema.

Runtime sessions, events, checkpoints, tasks, approvals, receipts, knowledge
entries, usage, artifacts, eval results, and snapshots belong in configured
stores or artifact backends. They do not belong in generated source packages.

For runtime timing around tool work, use
`await app.inspect_recent_tool_round_timing(session_id)` and
`await app.inspect_recent_model_step_preparation_timing(session_id)`.
`CayuApp(runtime_timing=RuntimeTimingConfig(...), timing_sinks=[...])` controls
the bounded recent view and optional best-effort export. Outside the Cayu
server, `await app.aclose()` (or `async with app:`) stops timing delivery at
shutdown along with the app's other work. These observations
are process-local and add no durable writes. Phase definitions, native store
costs and slow-storage interpretation are in `cayu guide durable-service-tools`.
