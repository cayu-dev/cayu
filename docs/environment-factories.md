# Environment Factories and Workspace Bindings

This is a how-to guide for provisioning an execution **environment** per session and
attaching a **workspace binding** that moves files between durable storage and the runner.
Use it when a single static environment isn't enough — when each session needs its own
sandbox, its own checked-out repo, a restored snapshot, or files copied into a remote
runner and copied back out afterwards.

Three nouns underpin everything below:

- **Workspace** — a session's durable file storage (what the agent's file tools read/write).
- **Runner** — where tool commands actually execute (a local process, a container, a cloud sandbox).
- **Binding** — connects the two: it makes workspace files available to the runner before the
  run and persists them after. When the runner already shares the workspace's filesystem the
  binding is a no-op; when it doesn't, the binding copies, restores, or mounts.

Cayu deliberately does **not** encode every vendor/filesystem combination (E2B + S3, ECS +
EFS, Modal + object store, Kubernetes + PVC, …) in core. Instead it gives you two small
contracts — `EnvironmentFactory` and `WorkspaceBinding` — plus a handful of built-in
bindings, and you compose the rest as example-level recipes. This guide covers both
contracts, the built-in bindings, and where to find worked examples.

Two runnable, API-key-free examples accompany this guide:

- [`examples/environments/local_native.py`](../examples/environments/local_native.py) — the
  simplest factory: a fresh local workspace + local runner per session, joined by
  `NativeBinding`.
- [`examples/environments/snapshot_restore.py`](../examples/environments/snapshot_restore.py) —
  a custom binding that restores a workspace from a snapshot before the run and saves a new
  one after, for reproducibility / fork workflows.

## The lifecycle

For each session the runtime walks a fixed order (owned by
`src/cayu/runtime/_environment_lifecycle.py`):

```
factory admission        ->  verifies side-effect-free pre-create evidence
factory.create(request)  ->  produces a concrete Environment (workspace + runner + binding)
runner admission         ->  verifies a runner returned directly by the factory
binding.bind(...)        ->  makes the workspace available to the runner (copy-in / restore / checkout)
final runner admission   ->  verifies the exact runner exposed by the binding
   ... the agent runs, tools read/write/exec against the bound workspace ...
binding.finalize(...)     ->  persist / sync-back / snapshot, using the session outcome
   ... cleanup runs regardless of outcome (completed / failed / interrupted) ...
```

The direct-runner admission happens before reconnect identity is checkpointed. The final
admission always runs after binding because a binding may supply or replace the runner.
`bind` runs before the first tool call; `finalize` runs when the session ends (completed,
failed, or interrupted) and receives the `outcome` so it can decide what to persist.

### Bounded lifecycle progress

Set `EnvironmentSpec.lifecycle_policy` when operators need a content-free view of slow
factory, binding, and finalization work:

```python
from cayu import EnvironmentLifecyclePhase, EnvironmentLifecyclePolicy, EnvironmentSpec

spec = EnvironmentSpec(
    name="hosted",
    lifecycle_policy=EnvironmentLifecyclePolicy(
        lifecycle_timeout_seconds=1800.0,
        phase_timeout_seconds={EnvironmentLifecyclePhase.TRANSFER: 300.0},
        progress_min_interval_seconds=1.0,
        max_progress_events=128,
    ),
)
```

A partial `phase_timeout_seconds` mapping overrides the named phases and retains finite
defaults for every other phase. The complete policy is part of the execution-environment
profile and application manifest, so a resume fails before work when the configured bounds
drift.

Configured environments emit bounded `environment.lifecycle.progress` events for factory,
binding, finalization, release, and retained-cleanup operations. The public
`EnvironmentLifecycleProgress` projection reports only the operation and phase, status,
aggregate item/byte or active/queued counts, elapsed time, deadline, event index, and opaque
operation/binding-generation authority. It never accepts paths, file names, contents,
commands, credentials, or adapter messages. Aggregate counters are limited to exact portable
JSON integers, operation identities are deterministic for the same session, environment,
invocation, operation, and binding generation, and the configured event cap always reserves
room for one terminal observation.

Incomplete-session recovery also inspects the latest progress record for the fenced
interaction. If process loss left a nonterminal operation with no live owner, recovery
appends one deterministic terminal `retained` record with
`recovery_disposition="orphaned_stale"`; acknowledgement-loss retries reconcile that exact
record. Recovery never replays the ambiguous filesystem operation. A successor recovery run
uses its new run epoch in the opaque operation identity, so its progress cannot be confused
with the retained predecessor.

The shared phases cover ownership admission, factory provisioning/reconnect, source and
target observations, staging admission, archive preparation, transfer/materialization,
execution-ready publication, copy-back conflict preflight/publication, release, and retained
cleanup. `SyncBinding` publishes aggregate progress for its built-in phases. Custom factories
and bindings can obtain the runtime-owned reporter with
`current_environment_lifecycle_progress_reporter()` while their lifecycle callback is active.
Release callbacks run under the release reporter as well. When delegated cleanup remains
owned after a release bound, the release operation terminates as `retained` and Cayu publishes
an explicit terminal retained-cleanup owner rather than claiming a clean release.

The current deadline mode is reported explicitly as `cooperative_progress_boundary`.
Cayu records whether the lifecycle or phase bound expired and rejects the overrun when the
runtime or adapter crosses a reporting boundary; it does not claim that a
cancellation-resistant opaque provider call has been stopped. Existing mutation fencing and
retained cleanup remain authoritative for deciding whether resources are safe to release. Do
not treat a deadline event by itself as proof that an external allocation or filesystem
mutation is quiescent.

## Writing an `EnvironmentFactory`

`register_environment` attaches **one** pre-built environment shared by every session;
`register_environment_factory` registers a **builder** the runtime calls once per session via
`create(request)`. Prefer a factory whenever sessions need isolation — their own sandbox, their
own per-session directory or git checkout, or a restored snapshot.

A factory subclasses `cayu.EnvironmentFactory` and implements one async method, `create`,
returning an `EnvironmentFactoryResult` that wraps a concrete `Environment`:

```python
from cayu import (
    Environment, EnvironmentFactory, EnvironmentFactoryRequest,
    EnvironmentFactoryResult, EnvironmentSpec, LocalRunner, LocalWorkspace, NativeBinding,
)


class LocalNativeFactory(EnvironmentFactory):
    def __init__(self, base_root):
        self._base_root = base_root

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        root = self._base_root / request.session_id     # one directory per session
        root.mkdir(parents=True, exist_ok=True)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(root),
                runner=LocalRunner(root),
                binding=NativeBinding(),
            )
        )
```

Register it as the default (or under a name) instead of a static `register_environment`:

```python
app.register_environment_factory(EnvironmentSpec(name="local"), LocalNativeFactory(base), default=True)
```

When a factory attaches an already-configured durable artifact store to every
concrete environment, register that same store with the factory registration:

```python
app.register_environment_factory(
    EnvironmentSpec(name="hosted"),
    hosted_factory,
    artifact_store=durable_artifact_store,
    default=True,
)
```

This does not create a session environment or transfer artifact ownership to
Cayu. It gives control-plane endpoints such as `/api/artifacts` a stable handle
for inventory and reads even after the per-session runner has been finalized.
`CayuApp.attach_file(...)` also uses the handle for prompt-attachment writes
without materializing the factory. Stores that exist only after
`create(request)` cannot be listed outside that session or receive prompt
attachments before the session environment exists unless the application
exposes a stable store this way.

`default=True` is an explicit application-wide choice: a `RunRequest` without
`environment_name` selects this factory. Omit it (or pass `default=False`) when
the factory should be available only by name, then set
`RunRequest.environment_name="local"` for runs that should provision it. The
first registered environment is not made the default automatically.

Notes:

- `EnvironmentFactoryRequest` carries the durable session context: `session_id`,
  `agent_name`, `environment_name`, `operation`, `parent_session_id`,
  `causal_budget_id`, `labels`, `metadata`, and `reconnect_metadata`. `operation`
  is `CREATE` for new sessions and a fork's first child allocation, and
  `RECONNECT` for resume/recovery. A reconnect operation must fail closed when
  its durable metadata is missing; it must never silently allocate a replacement.
  The request also carries the agent's copied `execution_requirements`; those
  requirements come from application-controlled agent registration, not from a
  per-run request. Factories should key per-session resources off `session_id`.
- `EnvironmentFactoryResult.reconnect_metadata` is checkpointed automatically.
  It must be versioned, JSON-safe, non-secret identity only. A fresh factory must
  either attach the exact resource named by that metadata or return a typed
  unsupported/invalid result; silently creating a replacement is not reconnect.
  A reconnectable fresh-path virtual-egress adapter must also preserve Cayu's
  backend allocation fingerprint in that metadata; a reconnect without that
  positive identity proof fails closed before adapter preparation.
- A result that owns live resources represented by `reconnect_metadata` should
  provide `EnvironmentFactoryResult.release`. The runtime calls it with
  `DISCARD` when a new allocation fails before the durable checkpoint and
  `PRESERVE` for a reconnect or when a new allocation was checkpointed but its
  completion event or later setup failed before successful workspace binding.
  `PRESERVE` must detach/release
  host-side handles without deleting
  the reconnectable resource; the callback should be idempotent. Cayu's virtual
  egress factory supplies this callback automatically. Release callbacks are
  cancellation-safe and bounded to 15 seconds by default; set
  `release_timeout_s` on the result when the provider needs a different bound.
  Code that calls a factory directly, outside `CayuApp`, assumes the same
  obligation and must release a result whose binding never succeeds.
- `EnvironmentFactoryResult.environment` must be **exactly** an `Environment` (not a
  subclass) — build one and set `workspace`, `runner`, and `binding` on it.
- The `binding` is attached to the `Environment`. If you omit it, the binding step is
  skipped entirely — no `bind`/`finalize` runs and no binding events are emitted (the
  runtime does **not** substitute a default binding). Pass one (e.g. `NativeBinding()`)
  whenever you want the bind/finalize/snapshot lifecycle.
- `VirtualEgressEnvironmentFactory` adds a `workspace_factory` convenience for
  provider-native workspaces. It receives the public lifecycle-managed runner;
  when supplied without `inner_binding`, the virtual-egress factory attaches
  the workspace with `NativeBinding`. For example,
  `workspace_factory=MicrosandboxWorkspace` produces a first-party workspace
  in the enforced microVM without exposing the raw `MicrosandboxRunner`.

### Process-external allocation

A factory whose `CREATE` path mutates a remote provider must not use ordinary
`create(...)` unless an application-owned transaction already makes that
mutation crash-safe. Override `allocation_scope(...)` with a stable provider
and adapter-generation identity, then implement `create_recoverable(...)`
against the supplied `EnvironmentAllocationContext`. Prepare deterministic,
non-secret exact-resource metadata before dispatch; mark the intent dispatched
before the provider call; and acknowledge the exact reconnect identity before
returning the result.

Recovery receives the same allocation intent and its durable state. It may
look up or adopt only the resource owned by that intent, replay a true provider
idempotency key, or perform bounded race-safe cleanup. Cleanup must first call
`mark_reaping()`: `True` means the durable cleanup fence was acquired, while
`False` means another worker already published the allocation and the resource
must be preserved. After acquiring the fence, cleanup is idempotently retried
from `REAPING` and calls `mark_reaped()` only after exact deletion is positively
complete. It must not create a new incarnation under a reusable name and
describe that as recovery. Cayu acquires the same fence before invoking a
result's `DISCARD` release path, so a losing validation worker cannot delete an
allocation concurrently published by another worker.

Once Cayu publishes an allocation receipt, reconnect must return the same
non-secret reconnect identity. A provider whose attach identity changes needs
an explicit durable transition of its own; ordinary reconnect cannot overwrite
the immutable allocation receipt. A fork that inherits such a receipt cannot
downgrade to an ordinary factory create; the factory must retain its recoverable
scope and atomically replace only the fork's copied ownership state.
The immutable fork relationship records the actual owner of copied state whenever
it differs from immediate lineage; exact forks always bind it as part of their
source evidence. The owner may be an earlier ancestor when several forks are
created before any intermediate child runs. Each descendant still receives its
immediate source as `parent_session_id`, while allocation replacement remains
bound to the recorded owner after either source row is deleted.

Every custom `SandboxEgressAdapter` must explicitly set
`process_external_allocation` to `True` or `False`. Leaving it undeclared fails
`CREATE` before adapter preparation. This is intentional: Cayu cannot safely
infer whether third-party runner creation mutates a process-external provider.

The bundled Microsandbox, E2B, and Lambda MicroVM virtual-egress adapters do
not currently satisfy every exact-recovery and conditional-cleanup requirement,
so new creates through them fail before adapter setup or provider mutation.
Docker remains available because its allocation is process-local. See
[Runtime contracts](runtime-contracts.md) for the complete state and atomic
publication contract.

## Execution admission for custom integrations

Applications attach provider-neutral workload requirements to the agent, not
to a factory or `RunRequest`:

```python
from cayu import AgentSpec, ExecutionRequirements

app.register_agent(
    AgentSpec(name="coding-agent", model="provider/model"),
    execution_requirements=ExecutionRequirements.untrusted(),
)
```

Omitting `execution_requirements` preserves the permissive
`ExecutionRequirements.trusted()` default. Existing integrations that make no
admission claim therefore continue to work for trusted agents, but fail closed
when an agent requires capabilities for which they provide no evidence.

A custom integration supplies evidence at two boundaries. The factory hook
describes the candidate before allocation; the runner hook describes the exact
live candidate that will execute commands:

```python
from cayu import EnvironmentFactory, ExecutionAdmissionCandidate, Runner


class HostedFactory(EnvironmentFactory):
    def execution_admission_candidate(self, request):
        return ExecutionAdmissionCandidate(
            candidate="acme-sandbox",
            evidence=self._declared_evidence,
        )

    async def create(self, request):
        ...


class HostedRunner(Runner):
    def execution_admission_candidate(self):
        return ExecutionAdmissionCandidate(
            candidate="acme-sandbox",
            evidence=self._runtime_evidence,
        )
```

Both evidence objects are `ExecutionCapabilityEvidence` whose `subject` is the
same candidate string. The pre-create hook must be side-effect free: it may
publish integration declarations, but it must not allocate a sandbox or claim
that a live resource was observed. The final runner hook should return quickly
and may publish integration-validated availability or bounded live observations.
Cayu can call it again after asynchronous setup or binding, so evidence with a
validity window must still be fresh at that point.

If a binding replaces the factory's runner, the replacement must report the
same admitted candidate. Missing evidence, a changed candidate, insufficient
claims, and expired live observations fail closed before the runner is exposed
to the agent. Cayu evaluates only the selected candidate; admission never
chooses a provider or silently falls back to another environment.

See [Execution admission](runtime-contracts.md#execution-admission) for the
complete capability vocabulary, evidence states, live-proof bounds, structured
refusals, and trusted versus untrusted presets.

## Workspace bindings

A binding subclasses `cayu.WorkspaceBinding` and implements `bind` + `finalize`. `bind`
returns a `BoundWorkspace` (the workspace + runner the tools will use, plus optional
`metadata`/`snapshot`); `finalize` optionally returns a `WorkspaceSnapshot`:

```python
async def bind(self, workspace, runner, *, session_id,
               agent_name=None, environment_name=None, metadata=None) -> BoundWorkspace: ...
async def finalize(self, bound, *, outcome=None, metadata=None) -> WorkspaceSnapshot | None: ...
```

On the returned `BoundWorkspace`, `workspace` is what the tools actually bind to and
`source_workspace` is the original workspace the factory built. They **differ** when a binding
swaps in a copy or a remote view (e.g. `SyncBinding` binds tools to a target while the source
stays durable) and **coincide** for native/local bindings that pass the same workspace through.

Factory-result ownership transfers only when `bind` returns successfully. On a
failed or cancelled bind, the binding must roll back only state it created
during that attempt; Cayu calls the still-unadopted factory result's `release`
callback for its runner and allocation. After success, `finalize` is the sole
lifecycle owner. A binding that replaces the factory runner must retain enough
state to finalize both the source and replacement resources and must not rely on
the factory release callback as a second cleanup path.

The built-in bindings live in `src/cayu/environments/bindings.py`:

| Binding | What it does | Pairs well with |
|---|---|---|
| `NativeBinding` | Pass-through — workspace and runner already share a filesystem; revision observation is explicitly unsupported. | `LocalRunner` + `LocalWorkspace`; a Docker bind-mount. |
| `DeterministicWorkspaceBinding` | Native pass-through plus bounded backend-neutral workspace revision observation. | Tests, conformance adapters, and simple workspaces whose public list/read APIs are authoritative. |
| `NoWorkspaceBinding` | Exposes no workspace to the runner (compute-only runs). | Any runner when the agent needs no files. |
| `SyncBinding` | Copy-in on bind, conditional copy-out on finalize. | An ephemeral/remote runner + a `RunnerWorkspace` (Docker, E2B, microVM). |
| `GitRepositoryBinding` | Ensures the workspace has a checked-out repo at a ref and provides bounded read-only HEAD/branch/index/worktree observation (never commits or pushes). | Any runner for code-on-a-branch workflows. |

`SyncBinding` is policy-driven: `sync_back` (`always`/`on_success`/`never`) controls when changed files
are copied back, while `clean_target`, `delete_missing`, `pattern`, `max_files`, `max_file_bytes`,
`max_total_bytes`, `max_archive_bytes`, and `preserve_git_modes` control each copy. The per-transfer
defaults are 64 MiB of logical file data and 128 MiB for the complete raw tar, including framing
and path metadata.
Those per-binding limits are independent of the process-wide staging governor. By default, all
`SyncBinding` instances share a governor with four transfer slots and 512 MiB of byte-weighted
working-set capacity. Pass the same `SyncBindingStagingCapacity` instance to a group of bindings to
define another capacity domain:

```python
from cayu import SyncBinding, SyncBindingStagingCapacity

staging = SyncBindingStagingCapacity(
    max_concurrency=8,
    max_staged_bytes=1024 * 1024 * 1024,
)
binding = SyncBinding(target_workspace=target, staging_capacity=staging)
```

Admission reserves the peak archive-plus-transient payload before creating its private spool.
Runner-backed workspaces stream raw tar bytes without whole-archive JSON/base64 copies. Exact
revision-aware fan-out can share one sealed archive while consumers are active; weak identities,
mutable content or executable-mode conflicts, different path/exclusion/mode policies, or different
copy limits never reuse it.
`binding.staging_snapshot()` reports bounded content-free queue, byte, peak, reuse, cleanup, and
wait-duration facts. Durable lifecycle-phase projection remains the responsibility of the
environment progress contract. See the `*_sync_binding_live.py` examples below.

For runtime-managed completed outcomes that require sync-back, finalization is a commit boundary:
the linked task and public session-completed event are published only after the durable source is
updated. A failure leaves a private bounded recovery marker and the retained target generation,
fails the task/session with `workspace_output_committed=false`, and can be retried with
`CayuApp.recover_incomplete_session(...)` after reconnecting the same target. Recovery validates the
binding policy and, when `source_conflict_policy="require_revision"`, the original revision baseline;
it does not repeat model or tool execution.

### Shared immutable Docker inputs

Use an immutable input projection for large runtime or support trees that many
Docker environments must read but must never change. Unlike `SyncBinding`, this
path materializes an exact tree once in a private manager store and attaches the
same host object to each container as a read-only bind mount. The mutable
`/workspace` copy-in/copy-back lifecycle remains separate.

```python
from hashlib import sha256

from cayu import (
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    ImmutableInputStore,
    LocalWorkspace,
    inspect_local_immutable_input,
)


def identity(label: str) -> str:
    return "sha256:" + sha256(label.encode()).hexdigest()


image = DockerImageIdentity(
    reference="registry.example/coding@sha256:" + ("a" * 64)
)
runtime_compatibility = image.fingerprint
runtime_input = inspect_local_immutable_input(
    "/srv/cayu/runtime-input",
    target_path="/opt/cayu/inputs/runtime",
    policy_fingerprint=identity("runtime-input-policy-v1"),
    runtime_compatibility_fingerprint=runtime_compatibility,
    authorization_scope_fingerprint=identity("tenant-and-purpose-scope"),
)
store = ImmutableInputStore("/var/lib/cayu/immutable-inputs")
factory = DockerCodingEnvironmentFactory(
    source_workspace=LocalWorkspace("/srv/cayu/workspace"),
    image_identity=image,
    immutable_inputs=(runtime_input,),
    immutable_input_store=store,
    immutable_input_runtime_compatibility_fingerprint=runtime_compatibility,
)
```

The projection identity binds the exact file manifest and executable modes,
format version, normalized target, copy limits, policy, Runtime compatibility,
and authorization scope. Sources may contain only regular files and real
directories; symlinks and special entries fail inspection. Targets must be
dedicated paths outside `/workspace`, `/tmp`, `/proc`, `/sys`, and `/dev`, and
must not overlap another immutable or tmpfs target. Keep the source, mutable
workspace, and mode-`0700` store in separate directory trees.

`DockerRunner` accepts only opaque mounts issued by `ImmutableInputStore`. It
verifies the daemon's exact bind-mount inspection and proves that even a root
process cannot create a probe file before it exposes the runner. A successful
workspace finalizer hashes and publishes only `/workspace`, closes the
container, and then releases its durable immutable-input references. Exact
attachment replay, release, orphaned-publication adoption, inspection, and
garbage collection work from a fresh `ImmutableInputStore` instance and do not
depend on process-local counters. `store.inspect()` returns only identities,
logical and physical bytes, counts, reuse, wait reason, and cleanup state; it
never returns file names or contents.

Adapters report `shared_read_only`, `mutable_sync_binding`,
`workspace_materialization`, or `unsupported` through
`input_capability()`. Use `require_immutable_input_projection(...)` to reject
weaker adapters. A bounded mutable copy is accepted only when the application
sets `allow_bounded_copy_fallback=True`; no adapter silently downgrades the
guarantee. Docker reconnect verifies the exact retained container and its
immutable mounts; deterministic attachment replay retains one durable reference
for that same physical container instead of allocating a duplicate.

A `SyncBinding` target plan factory is an identity-resolution boundary, not an
allocation owner: `target_workspace_plan_factory` returns a
`SyncTargetWorkspacePlan` containing an already lifecycle-owned, quiescent
`Workspace`.
Allocation that establishes the target's stable identity belongs in the
surrounding `EnvironmentFactory`. Once that identity exists, attachment or setup
that must mutate the workspace is represented by
`SyncTargetWorkspacePlan(workspace=target, provision=setup)`. Cayu
reserves the source and target pair before invoking `setup`; failed-bind resource
cleanup remains the surrounding `EnvironmentFactoryResult.release` owner's
responsibility.

The legacy `target_workspace_factory` entrance is rejected before its callback
is invoked. Its earlier create-or-attach contract could mutate a target before
Cayu knew the target identity and therefore could not satisfy resource
exclusion.

Two patterns the issue that motivated this guide called out are **not** separate classes:

- **Snapshots** are not a `SnapshotBinding`; they are the `WorkspaceSnapshot` a binding
  returns from `bind`/`finalize`. To restore-before / save-after, write a small custom
  binding — see [`examples/environments/snapshot_restore.py`](../examples/environments/snapshot_restore.py).
  There is no `snapshot()`/`restore()` on `Workspace`; the binding owns that policy so core
  doesn't have to know your snapshot backend.
- **Artifacts** are not a binding; they are a separate concern — attach an `ArtifactStore`
  to the `Environment` and persist selected outputs as artifacts.

### Export locations

`Environment`, `EnvironmentSpec`, `EnvironmentFactory`, `WorkspaceBinding`, `BoundWorkspace`,
`WorkspaceSnapshot`, and the concrete bindings (`NativeBinding`, `NoWorkspaceBinding`,
`SyncBinding`, `GitRepositoryBinding`, `DeterministicWorkspaceBinding`) plus `SyncTargetWorkspacePlan` are re-exported from the
top-level `cayu`. The base
`Workspace` and `Runner` types are **not** — import those from their modules:

```python
from cayu import WorkspaceBinding, BoundWorkspace, WorkspaceSnapshot
from cayu.workspaces import Workspace
from cayu.runners import Runner
```

## Vendor integrations are recipes, not core

Vendor/filesystem pairs live as example-level recipes that compose the primitives above. The
repository already ships worked, live ones:

- Copy-in/copy-out with `SyncBinding`:
  [`examples/sync_binding_local.py`](../examples/sync_binding_local.py) (local, no runner),
  [`examples/docker_sync_binding_live.py`](../examples/docker_sync_binding_live.py),
  [`examples/e2b_sync_binding_live.py`](../examples/e2b_sync_binding_live.py),
  [`examples/microsandbox_sync_binding_live.py`](../examples/microsandbox_sync_binding_live.py).
- A custom runner backend (Modal Sandboxes):
  [`examples/modal_runner.py`](../examples/modal_runner.py) and the companion
  [Build a Runner](./build-a-runner.md) guide.

For a new vendor, pick the runner (or build one per *Build a Runner*), pick the binding that
matches how its filesystem relates to the runner (`NativeBinding` if they share files,
`SyncBinding` if you must copy across a boundary, or a custom binding for snapshot/restore),
and wire them in a factory. That combination — not a core-owned matrix — is the extension
point.
