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

For each session the runtime walks a fixed order (see `src/cayu/runtime/app.py`):

```
factory.create(request)   ->  produces a concrete Environment (workspace + runner + binding)
binding.bind(...)         ->  makes the workspace available to the runner (copy-in / restore / checkout)
   ... the agent runs, tools read/write/exec against the bound workspace ...
binding.finalize(...)     ->  persist / sync-back / snapshot, using the session outcome
   ... cleanup runs regardless of outcome (completed / failed / interrupted) ...
```

`bind` runs before the first tool call; `finalize` runs when the session ends (completed,
failed, or interrupted) and receives the `outcome` so it can decide what to persist.

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

`default=True` is an explicit application-wide choice: a `RunRequest` without
`environment_name` selects this factory. Omit it (or pass `default=False`) when
the factory should be available only by name, then set
`RunRequest.environment_name="local"` for runs that should provision it. The
first registered environment is not made the default automatically.

Notes:

- `EnvironmentFactoryRequest` carries the durable session context: `session_id`,
  `agent_name`, `environment_name`, `parent_session_id`, `causal_budget_id`, `labels`,
  `metadata`, `reconnect_metadata`. Key your per-session resources off `session_id`.
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

The built-in bindings live in `src/cayu/environments/bindings.py`:

| Binding | What it does | Pairs well with |
|---|---|---|
| `NativeBinding` | Pass-through — workspace and runner already share a filesystem. | `LocalRunner` + `LocalWorkspace`; a Docker bind-mount. |
| `NoWorkspaceBinding` | Exposes no workspace to the runner (compute-only runs). | Any runner when the agent needs no files. |
| `SyncBinding` | Copy-in on bind, conditional copy-out on finalize. | An ephemeral/remote runner + a `RunnerWorkspace` (Docker, E2B, microVM). |
| `GitRepositoryBinding` | Ensures the workspace has a checked-out repo at a ref; records commit/branch/dirty state (never commits or pushes). | Any runner for code-on-a-branch workflows. |

`SyncBinding` is policy-driven: `sync_back` (`always`/`on_success`/`never`) controls when changed files
are copied back, while `clean_target`, `delete_missing`, `pattern`, `max_files`, `max_file_bytes`,
`max_total_bytes`, and `max_archive_bytes` control the copy. The aggregate cap defaults to 64 MiB of logical file data per
copy-in or copy-back transfer, while `max_archive_bytes` defaults to 128 MiB for the complete raw
tar including framing and path metadata. Bulk tar transfers are buffered and runner protocols add
encoding overhead, so size these controls below the process or sandbox memory ceiling. See the
`*_sync_binding_live.py` examples below.

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
`SyncBinding`, `GitRepositoryBinding`) are re-exported from the top-level `cayu`. The base
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
