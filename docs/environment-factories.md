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
from cayu.environments import EnvironmentLifecyclePhase, EnvironmentLifecyclePolicy, EnvironmentSpec

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

A factory subclasses `cayu.environments.EnvironmentFactory` and implements one async method, `create`,
returning an `EnvironmentFactoryResult` that wraps a concrete `Environment`:

```python
from cayu.environments import Environment, EnvironmentFactory, EnvironmentFactoryRequest, EnvironmentFactoryResult, EnvironmentSpec, NativeBinding
from cayu.runners import LocalRunner
from cayu.workspaces import LocalWorkspace


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

### Per-request local workspaces

`LocalWorkspaceEnvironmentFactory` attaches a prepared directory whose ownership
the application has already established. It resolves symlinks, rejects missing
directories and roots outside `allowed_roots`, and forwards runner admission
proof. A parent-directory allowlist does not authorize access to another
session's or tenant's directory.

```python
from pathlib import Path
from cayu.environments import LocalWorkspaceEnvironmentFactory

# Populate this trusted mapping when the application admits and prepares a job.
prepared_roots = {"review-42": Path("/srv/reviews/review-42").resolve()}
factory = LocalWorkspaceEnvironmentFactory(
    allowed_roots=["/srv/reviews"],
    root_from_request=lambda request: prepared_roots[request.session_id],
)
app.register_environment_factory(EnvironmentSpec(name="checkout"), factory, default=True)
request = RunRequest(
    session_id="review-42", agent_name="reviewer",
    messages=[Message.text("user", "Review this checkout")],
)
```

Metadata selection is also supported, but requires an explicit
`authorize_root(request, resolved_root)` callback returning exactly `True` for
an owned session/path pair. Metadata can be authored by users or subagents:
never treat the path itself as authorization. For the same trusted mapping:

```python
factory = LocalWorkspaceEnvironmentFactory(
    allowed_roots=["/srv/reviews"],
    metadata_key="checkout",
    authorize_root=lambda request, root: prepared_roots.get(request.session_id) == root,
)
```

`workspace_root_from_metadata` only parses a path and
`require_allowed_workspace_root` only checks containment; custom integrations
must separately enforce ownership. All callbacks are synchronous and
side-effect free. Prepare directories before `app.run`.

For Docker, retain the actual `DockerRunner` returned by preparation. A bare
`DockerRunner(name)` lacks the exact container identity needed by final
executable admission. Prepare the container before admission, using an image
with the tools your agent requires:

```python
root = prepared_roots["review-42"]
prepared_runners = {"review-42": await DockerRunner.create(
    "review-42", image="your-review-image:tag", mount_path=str(root),
    close_action="none",
)}
factory = LocalWorkspaceEnvironmentFactory(
    allowed_roots=["/srv/reviews"],
    root_from_request=lambda request: prepared_roots[request.session_id],
    runner_factory=lambda root, request: prepared_runners[request.session_id],
    binding_factory=lambda root, request: NativeBinding(default_path=str(root)),
)
```

`mount_path` mounts the host root at the same absolute guest path. The
application owns allocation, mapping, and disposal, and the callback must return
the same prepared runner at admission and creation. Use
`DockerCodingEnvironmentFactory` when Cayu should allocate and own the container.
A `binding_factory` can also select `GitRepositoryBinding` for a queued commit.

Give each concurrent run a different root. Local execution and path checks do
not sandbox hostile code or prevent a hostile filesystem owner from replacing
paths. Supply `execution_profile_identity` when declaring stable callback
behavior for recovery.

#### Dropping finished sessions from a shared app

One `CayuApp` with this factory can serve every request. Sessions stay in the
session store until something deletes them, so a long-running app should delete
each session once it has the answer. `CayuApp.delete_session` does that:

```python
async def handle(session_id: str, root: Path, prompt: str) -> list[Event]:
    prepared_roots[session_id] = root
    events = [event async for event in app.run(RunRequest(
        session_id=session_id, agent_name="reviewer",
        messages=[Message.text("user", prompt)],
    ))]
    if events[-1].type in {EventType.SESSION_COMPLETED, EventType.SESSION_FAILED}:
        await app.delete_session(session_id)
        prepared_roots.pop(session_id, None)
    return events
```

An interrupted session, for example one waiting for an approval, can still
resume, so keep it and its root until it finishes.

`delete_session` runs the same closure as `DELETE /api/sessions/{id}`. It
removes the session's events, transcript, checkpoint and grants, its settled
tasks and its session-scoped artifacts, and keeps budget records. It does not
touch the workspace directory, which the application owns. It raises
`ValueError` and keeps the session when the session is still pending, running
or interrupted, has active tasks or child sessions, or has anything the closure
cannot fully list or erase. `finished_only=False` also deletes pending and
interrupted sessions. It returns `None` for a session that does not exist.

With `InMemorySessionStore`, deletion keeps memory bounded. The store keeps a
closure receipt of a few kilobytes per deleted session, so the id stays retired
and a repeated erase returns the same report, but only for the newest
`closure_receipt_limit` deletions (10,000 by default). Once a receipt is
evicted, the standalone session store frees its identity reservation and its
erase cannot be replayed. This does not free dependent-store deletion fences:
application run and fork admission refuse IDs still retired by the task store.
Use fresh session IDs when combining stores. SQLite and Postgres keep every receipt. To drop old sessions on a schedule
instead, use storage retention with a durable store
([Storage retention](storage-retention.md)).

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
  `agent_name`, `environment_name`, `interaction_id`, `operation`,
  `parent_session_id`, `causal_budget_id`, `labels`, `metadata`, and
  `reconnect_metadata`. `operation` is `CREATE` for new sessions and a fork's
  first child allocation, and `RECONNECT` for resume/recovery. A reconnect
  operation must fail closed when its durable metadata is missing; it must never
  silently allocate a replacement.
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
- `EnvironmentFactoryResult.allocation_fingerprint` is the content-free identity
  of the backend allocation, a lowercase SHA-256 digest. Cayu checkpoints it
  next to `reconnect_metadata` and uses it to match a disposed allocation to the
  bindings that used it, so a session whose container was already disposed can
  still be settled and resumed. Cayu's Docker coding and virtual-egress
  factories set it. A factory that wraps another factory passes the inner
  result's `allocation_fingerprint` through, even when it nests the inner
  `reconnect_metadata` under its own keys. A reconnect must publish the
  identity its allocation was checkpointed with. A top-level `reconnect_metadata["allocation_fingerprint"]`
  is still read when the field is unset, and must equal the field when both
  are set.
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
  If binding fails for a newly created recoverable allocation, Cayu then
  fences its exact published receipt into `REAPING` and calls the factory's
  `reap_allocation` hook. Cleanup completes only after the provider records
  `REAPED`; errors and timeouts remain owned by `drain_environment_cleanups`.
  Reconnected allocations remain preserved. Wrappers around recoverable
  factories must forward `reap_allocation` as well as `create_recoverable`.
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

### Coding in a cloud sandbox

`SandboxCodingEnvironmentFactory` (in `cayu.egress`) is the cloud counterpart of
`DockerCodingEnvironmentFactory`: each session gets its own sandbox, a durable
`source_workspace` (a `LocalWorkspace` outside the sandbox) is copied in before
the first tool call, file, search, check and command tools run inside the
sandbox, and changes are copied back according to `sync_back` (`on_success` by
default). The sandbox filesystem is reconstructible execution state; the source
stays canonical.

- `.git`, `.cayu` and `.runtime` never cross in either direction. The sandbox
  gets the working tree, not the host's repository, so it can't write git
  config or hooks into the host checkout. Git commands inside the sandbox see
  no repository unless the workload creates one.
- The sandbox copy also excludes whatever the source workspace excludes (for
  example `excluded_directory_names=("__pycache__",)`) and
  `sandbox_excluded_directory_names`.
- Copy-back is revision-aware: if a source file changes outside the sandbox
  after copy-in, the session's finalization fails with a
  `SyncBindingSourceConflictError` instead of overwriting that change.

The provider is chosen only by the adapter, so callers don't change when a
different cloud's sandbox is used. AWS Lambda MicroVM is the first
implementation:

```python
from cayu.egress import (
    ApprovedEgressDestination,
    HttpEgressPolicy,
    SandboxCodingEnvironmentFactory,
    SandboxWorkspaceTransferLimits,
)
from cayu.egress.aws_lambda_microvm_adapter import LambdaMicroVMEgressAdapter
from cayu.egress.proxy_exposure import VpcTaskProxyExposure
from cayu.workspaces import LocalWorkspace

factory = SandboxCodingEnvironmentFactory(
    adapter=LambdaMicroVMEgressAdapter(
        region_name="us-east-1",
        egress_network_connector_arn=egress_connector_arn,
        exposure=VpcTaskProxyExposure(task_private_ip),
        runner_options={"default_cwd": "/workspace/project"},
    ),
    image=microvm_image_arn,
    source_workspace=LocalWorkspace(checkout_path, workspace_id="project"),
    policies={
        "pypi": HttpEgressPolicy(
            name="pypi",
            allowed_hosts=["pypi.org"],
            allowed_endpoints=[("GET", "/simple/")],
        )
    },
    approved_destinations=[
        ApprovedEgressDestination(destination="pypi.org", policy_name="pypi")
    ],
    transfer_limits=SandboxWorkspaceTransferLimits(max_total_bytes=256 * 1024 * 1024),
)
```

The factory is a preset over `VirtualEgressEnvironmentFactory`, `SyncBinding`
and `RunnerWorkspace`, and each stays usable on its own. Tools are admitted
through the adapter's live executable evidence, so a tool whose executables are
missing from the image is refused before the model sees it. Commands and checks
run in the runner's default working directory, which is where the source is
copied. One source workspace belongs to one active session at a time.

#### Network access

All sandbox traffic goes through the adapter's egress proxy, which terminates
TLS and, with `event_emitter` wired, records an `egress.request.authorized` or
`egress.request.denied` event for every request. Pick one explicit mode; there
is no default and no network-free mode:

- **Allowlist**, shown above: `approved_destinations` (credentialless) and
  `credentials` (virtual credentials) name the hosts the work may reach, each
  under its policy. Everything else is denied.
- **Open internet**: name an `OpenInternetEgressPolicy` with
  `open_internet_policy`. Any public HTTPS host is reachable, with every method
  and the sandbox's own headers and body forwarded unchanged, so package
  installs from public registries and `git clone`/`fetch` over HTTPS work
  without listing hosts.
- **Public research** (`public_web_policy`): bodyless GET/HEAD to any public
  host with fixed headers; see
  [Virtual egress](virtual-egress.md#credentialless-public-web-research).

```python
from cayu.egress import (
    VIRTUAL_EGRESS_EVENT_TYPES,
    OpenInternetEgressPolicy,
    SandboxCodingEnvironmentFactory,
)

factory = SandboxCodingEnvironmentFactory(
    adapter=adapter,  # the same LambdaMicroVMEgressAdapter as above
    image=microvm_image_arn,
    source_workspace=LocalWorkspace(checkout_path, workspace_id="project"),
    policies={"internet": OpenInternetEgressPolicy(name="internet")},
    open_internet_policy="internet",
    event_emitter=app.scoped_event_emitter(event_types=VIRTUAL_EGRESS_EVENT_TYPES),
)
```

Open internet is for sandboxes that hold nothing worth stealing. The factory
refuses it together with credentials, a secret resolver, approved destinations
or any other policy, so the sandbox never receives a credential and the broker
never resolves one. It also requires `event_emitter`, so every request is on
the session's event stream. The proxy still refuses IP literals, ports other than 443,
plain HTTP, and any hostname that resolves to a private, loopback, link-local or
metadata address (each connection is pinned to the checked public address).
Each response is buffered under the broker's response-size limit (256 MiB by
default). Tools must use the proxy and CA the adapter sets (`HTTPS_PROXY`,
`SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`); git over SSH and
other non-HTTPS protocols are not available. See
[Virtual egress](virtual-egress.md#credentialless-open-internet-for-isolated-sandboxes).

Not yet available on this factory: toolchain profiles (it has no
network-free mode, which every profile requires; the brokered factory below
takes one), immutable inputs, a guest git baseline (Docker's `git_baseline`),
and a network-free sandbox.

### Coding in a sandbox brokered by Cayu Cloud

On Cayu Cloud an Agent holds no cloud-provider permission. It doesn't call
Lambda MicroVM itself. Cloud allocates the sandbox, runs its egress proxy,
leases it, audits it and destroys it (Cloud's `cayu.cloud.sandbox` v1
contract). `BrokeredSandboxCodingEnvironmentFactory` (in `cayu.sandbox_broker`)
is the sibling of `SandboxCodingEnvironmentFactory` for that setup. It takes
the same workspace options: `source_workspace`, `transfer_limits`, `sync_back`,
`sandbox_excluded_directory_names` and `staging_capacity`. It also keeps the
same protected directories and revision-aware copy-back. In place of Runtime
egress policies it takes one of Cloud's egress modes:

```python
from cayu.sandbox_broker import BrokeredSandboxCodingEnvironmentFactory, SandboxBrokerClient
from cayu.workspaces import LocalWorkspace

factory = BrokeredSandboxCodingEnvironmentFactory(
    broker=SandboxBrokerClient.from_env(),
    source_workspace=LocalWorkspace(checkout_path, workspace_id="project"),
    egress="public_web",  # or "none"
)
# Or configure everything from the environment:
factory = BrokeredSandboxCodingEnvironmentFactory.from_env(
    source_workspace=LocalWorkspace(checkout_path, workspace_id="project"),
)
```

| Variable | Meaning |
|---|---|
| `CAYU_CLOUD_API_URL` | Cloud API base URL. Must be HTTPS; plain HTTP works only for a loopback host. Cloud sets it. |
| `CAYU_CLOUD_APPLICATION_ID` | The Agent whose `/sandboxes` routes to use. Cloud sets it. |
| `CAYU_CLOUD_SANDBOX_TOKEN` | The Agent's sandbox credential (Cloud sets it), or an Organization API key with the `sandboxes` scope. |
| `CAYU_CLOUD_SANDBOX_EGRESS` | `none` (no outbound network) or `public_web` (the open internet through Cloud's logging proxy). Required by `from_env`. |
| `CAYU_CLOUD_SANDBOX_IMAGE` | Catalog image, default `coding-python`. `SandboxBrokerClient.list_images()` lists what the install offers. |
| `CAYU_CLOUD_SANDBOX_LEASE_SECONDS` | Lease length, 60 to 3600, default 900. |
| `CAYU_CLOUD_SANDBOX_REQUEST_TIMEOUT_SECONDS` | Timeout for one broker request, default 30. |

Every install's catalog has `coding-python` (Python with `uv`, `git` and
ripgrep); an install may also offer `coding-agents`, which adds the Claude Code
and Codex CLIs. Catalog images don't contain your project's dependencies.

`setup_commands` run in the fresh sandbox before the workspace is copied in,
so they can't see the project. `workspace_setup_commands` run in `/workspace`
after copy-in and the Git baseline, for example `uv sync --frozen` to install
the project's locked dependencies. That needs `egress="public_web"` to reach
the package index. Anything they write under `/workspace` is copied back like
any other change, so put environments elsewhere
(`UV_PROJECT_ENVIRONMENT=/tmp/venv`).

Each session creates one sandbox. Its lease starts when the sandbox is ready,
and the factory renews it at a third of the lease length. The factory waits up
to 330 seconds for the sandbox, since Cloud fails one still provisioning after
five minutes. Copy-in and copy-back use the broker's archive routes. Copy-in
uploads the tree in archives of at most 8 MiB. Copy-back reads the changed
files by name, in batches, splitting any batch the broker reports as too
large. So a tree of any size moves, but without staged copies one file must
fit in one archive (just under 8 MiB).

Where the install offers staged copies (the sandbox record's
`staged_copies`), a tree larger than one archive goes in as one staged copy:
the factory checks every member, uploads the tar straight to a signed URL for
one object in the install's storage, and the broker extracts it after
checking its size and SHA-256. A changed file too large for one archive comes
back the same way. The storage URL gets only the archive and the headers the
broker signed, never the sandbox credential, and nothing in the sandbox sees
it. `transfer_limits.max_file_bytes` may then exceed 8 MiB; a sandbox that
can't carry a file that large through staged copies is refused at setup and
destroyed. A file the sandbox's staged copies can't carry back fails copy-back
naming those limits, not `max_file_bytes`. The tree goes up straight from the
binding's staged archive, which `staging_capacity` already accounts for, so no
second host copy is made; only a caller of
`BrokeredSandboxWorkspace.write_tar_stream` whose source can't seek gets its
archive spooled to a host temporary file first. While a staged copy is handed
in and runs, a broker outage is retried until the copy's own deadline (ten
minutes plus a second per MiB), as commands are; only a copy still unconfirmed
then closes the runner for good. File tools and commands run as broker
commands.

The parts work on their own too. An Agent that clones a private repository
itself can hand the sandbox the checkout, `.git` included, without the
repository credential: `BrokeredSandboxRunner.copy_in(source, size=...,
sha256=..., path="repo")` (or `SandboxBrokerClient.copy_in`) moves a tar of
any size the install allows, and `copy_out(sink, max_bytes=..., paths=[...])`
reads one back and checks its digest. A failed copy raises
`SandboxCopyFailedError` with the broker's error code. The sandbox is destroyed when the session's
binding finalizes, or when setup fails. If copy-back fails, the factory keeps
the sandbox for a retry but stops renewing it. If the Agent process dies,
nothing renews the lease, and Cloud destroys the sandbox once the lease lapses.
Each turn of a session gets a new sandbox, copied in from the source
workspace. Completion recovery after a crash is refused, because it would run
against an empty replacement and copy that emptiness back as deletions. The
source keeps its last published state.

Like Docker's coding factory, the sandbox commits each copy-in to an ephemeral
local Git repository (`git_baseline=True`, the default), with Docker's inert
configuration: no hooks, filters, pager, credentials, signing or transport.
So `git_changes` and the guest's own `git` see what the turn changed since its
sandbox was filled. `.git` is protected, so the repository never reaches the
source; `BrokeredSandboxCodingBinding` used on its own refuses, before copying
anything, a source or target that doesn't exclude `.git` or a target on another
runner, and makes the repository in its target's directory. The baseline reads
back no output that grows with the workspace, so the broker's 2 MiB cap on a
command's output doesn't limit the tree. `git add` and `git commit` get a
minute plus a quarter second per MiB and ten seconds per thousand files copied
in, at most 30 minutes each, so a large staged tree has time to commit. The
image needs `git` (`coding-python` has it); a sandbox whose baseline fails isn't
bound. `git_baseline=False` drops
the baseline, for an image without `git`.

Each command starts under a fresh command id and is polled until it finishes:

- Its timeout is bounded only by the sandbox's remaining lifetime (eight hours
  from ready). A command with no timeout gets the rest of that lifetime.
- If the connection drops while a command starts or runs, the runner replays
  the same id, and the broker never runs it twice.
- If the caller cancels a command, the runner cancels it in the sandbox and
  waits for the broker to confirm it stopped. Only when that confirmation
  never comes does the runner close its exec path for good.
- A busy answer (4 commands already running) is retried, since the command
  never started.
- Output is captured up to 2 MiB per stream. The broker measures that limit
  on the JSON it returns, where a control character takes six bytes, so the
  runner asks for up to six times the caller's limit and cuts the result back
  to the raw bytes the caller asked for.
- stdin over 1 MiB, any argv argument over 131,071 bytes (Linux's per-argument
  limit), argv over 256 KiB in all, and env over 64 KiB are refused before
  anything is sent.

A broker answer without a contract code (from authentication, a proxy or the
ingress) is retried only for 408, 429 and 5xx. Any other 4xx, such as a 401
for a revoked credential, is final.

Final admission uses the sandbox record's versioned `guarantees`. A capability
such as `untrusted_code_isolation`, `real_credential_non_possession`,
`guest_privilege_containment` or `deny_by_default_network` is available only
when the guarantee item behind it is present. `deny_by_default_network` also
needs `egress="none"`: `public_web` reaches the open internet through a proxy
that logs but denies nothing. So a workload built with
`ExecutionRequirements.untrusted()` is admitted only on a `none` sandbox that
carries every item.
Required executables are probed live in the sandbox.

Each part also works alone:

- `SandboxBrokerClient` is the wire client;
- `SandboxLeaseKeeper` renews one lease;
- `BrokeredSandboxRunner` is a `Runner` over one sandbox and destroys it when
  closed. Where the broker runs interactive commands (the sandbox record's
  `interactive_commands`), its `interactive_process_capability()` opens them
  through the broker, each stopped with its whole process tree, so
  `runner_agent_launch` can run Claude Code or Codex in the sandbox. That needs
  a Cayu Cloud whose broker supports interactive commands; against one that
  doesn't, the capability is `None` and the launch is refused before anything
  starts. Port forwarding isn't available in brokered sandboxes
  (`port_forward_capability()` is `None`), so `runner_port_forward` can't reach
  a gateway on the Agent's host from one;
- `BrokeredSandboxWorkspace` is a `RunnerWorkspace` whose bulk copies use the
  broker's archive routes.

None of them imports a provider SDK or reads a cloud credential.

#### Toolchain profiles in a brokered sandbox

`toolchain_profile=` binds every sandbox to one neutral
`CodingToolchainProfile` (see
[Docker coding toolchains](docker-coding-toolchains.md#one-profile-on-docker-and-cayu-cloud)),
so `run_check` and `run_command` built from that same profile work in the
sandbox as they do on Docker:

```python
factory = BrokeredSandboxCodingEnvironmentFactory(
    broker=SandboxBrokerClient.from_env(),
    source_workspace=LocalWorkspace(checkout_path, workspace_id="project"),
    egress="none",
    image="release",
    toolchain_profile=profile,  # a CodingToolchainProfile
)
```

A profile prohibits the network and installing packages at run time, so the
factory requires `egress="none"` and refuses `setup_commands` and
`workspace_setup_commands`. It refuses a `DockerCodingToolchainProfile`; pass
`profile.coding_toolchain()`. Before allocating, it checks the profile's
dependency inputs in the source workspace. After allocating, and before
anything is copied in, it:

- requires the sandbox image to report the profile's
  `trusted_build_context_sha256`, when the profile sets one (the sandbox
  record's `image_identity`; older brokers report none, and such a profile is
  then refused with `toolchain_image_unverified`);
- checks that the read-only support paths exist (it can't prove they are
  read-only, so it doesn't claim it);
- runs the profile's admission probes and probes its executables live.

The runner's evidence then carries the profile's fingerprint, which the
command tools require. Binding by build inputs rather than image digest lets
one profile describe the image built locally and the one Cloud builds from the
same `Dockerfile.coding`. Until the install builds that image for the Agent,
a catalog image without the project's dependencies fails
admission with a code naming what is missing.

### Choosing a coding backend

`cayu.environments.coding_backends` picks where one toolchain profile runs,
so the same app uses Docker on a laptop and Cayu Cloud's broker when deployed,
with the same agents, tools and prompts:

```python
from cayu.environments import (
    DockerCodingBackendOptions,
    register_coding_environment,
    select_coding_backend,
)

backend = select_coding_backend(
    toolchain_profile=profile,
    options={
        "docker": DockerCodingBackendOptions(image_identity=image, platform_architecture="arm64"),
    },
)
register_coding_environment(
    app, name="coding", backend=backend, source_workspace=workspace, default=True
)
agent_requirements = backend.execution_requirements(real_secret_visibility="non_possession")
```

Selection, in order:

1. `required=` or `CAYU_CODING_BACKEND` names the backend a deployment
   requires. It must be registered and usable, so a missing Cloud credential
   fails at startup instead of falling back to Docker.
2. Otherwise `cayu-cloud` when `CAYU_CLOUD_API_URL`,
   `CAYU_CLOUD_APPLICATION_ID` and `CAYU_CLOUD_SANDBOX_TOKEN` are all set,
   `docker` otherwise. A sandbox token without the other two is an error
   naming what's missing; the API URL and application id alone are not, since
   Cloud's agent registry and the `cayu cloud` CLI use them too.
3. Any other backend is selected by name only.

Each backend supplies the execution requirements its agents need: Docker's
evidence is `live_verified`, the broker's capability claims are `available`.
So agents should take their `ExecutionRequirements` from
`backend.execution_requirements(...)` rather than hard-coding Docker's bar.
`options` are keyed by backend name: `DockerCodingBackendOptions` (image
identity, architecture, restrictions, transfer limits, `cancel_timeout_s`) and
`BrokeredCodingBackendOptions` (image, lease, transfer limits; built from the
`CAYU_CLOUD_*` variables when omitted). Brokered sandboxes always run with
`egress="none"`.

`register_coding_environment` can be called once per workspace, for example
for each clone a parallel worker edits; `RunRequest.environment_name` then
picks the workspace. `named_checks_from_toolchain(profile)` and
`ExactNamedCheckCommandPolicy(checks)` build `RunCheckTool`'s checks and its
defense-in-depth policy from the profile.

`CodingBackendRegistry` holds the builders. `default_coding_backends()` returns
a new registry with `docker` and `cayu-cloud`; another provider registers its
own builder on that registry and passes `registry=` to `select_coding_backend`.
No registry is global. The individual factories stay public, so an app that
always runs on one backend can construct it directly.

A session started on one backend isn't resumed on another: the factories have
different execution-profile identities, so the runtime refuses the resume.

### Per-agent default environment

An agent can name its own default with `register_agent(..., environment_name=...)`.
A new run picks its environment in this order: the request's
`environment_name`, then the agent's registered default, then the application
default. Pass `NO_ENVIRONMENT` to run the agent without any environment even
when the application registers a default, for example a tool-only agent in a
coding app whose default is a Docker sandbox:

```python
from cayu.environments import NO_ENVIRONMENT

app.register_agent(ISSUE_FILER_AGENT, tools=[FileIssueTool()], environment_name=NO_ENVIRONMENT)
app.register_agent(REVIEWER_AGENT, environment_name="review-checkout")
```

A stored session keeps the environment it started with, so resume, recovery
and queued dispatches never re-resolve it, and a fork keeps its source's
environment unless the fork request names one. A subagent child runs in its
parent's environment unless `SubagentSpec.environment_name` names another one;
the child agent's registered default does not override that inheritance, and a
parent that ran without an environment has children that run without one.
`cayu inspect --json` reports an agent's own default as `default_environment`
or `runs_without_environment`, and `cayu check` reports
`AGENT_ENVIRONMENT_NOT_FOUND` when the named environment is not registered.

### Per-child workspaces for subagents

A `SubagentTool` child runs in its parent's environment by default. Set
`SubagentSpec.environment_name` to a registered factory to give every child of
that alias its own allocation, for example one checkout per builder:

```python
app.register_environment_factory(EnvironmentSpec(name="builder-checkout"), LocalNativeFactory(base))

delegate = SubagentTool(
    app,
    agents={
        "builder": SubagentSpec(
            agent_name="builder",
            description="Edits its own copy of the repository.",
            environment_name="builder-checkout",
        )
    },
)
```

The factory's `create(request)` receives the child's `session_id` and the
parent's id in `parent_session_id`, so it can key the copy off the child. The
child environment is part of the spawn fingerprint: recovery reconnects the same
allocation, and a spec whose `environment_name` changed since the spawn fails
with an identity conflict instead of moving the child.

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

## Deferred materialization

A factory can defer creating its runner until a session first uses it. Session
start then binds nothing, and a run whose tools never touch the runner never
creates it. Build the result from one `DeferredMaterialization`:

```python
from cayu.environments import DeferredMaterialization, Environment, EnvironmentFactoryResult


async def create(self, request):
    async def materialize(mode):  # "use" creates fresh; "recover" must adopt
        runner = await self._create_runner(request)
        return runner, self._binding_for(runner)

    materialization = DeferredMaterialization(
        materialize,
        default_cwd="/workspace",
        configured_candidate=self._configured_candidate,  # admission without a runner
        environment_authority=self.execution_environment_authority(),
        dispose_unmaterialized=self._remove_leftover,  # crash leftovers
    )
    return EnvironmentFactoryResult(
        environment=Environment(
            spec,
            workspace=self.source,
            runner=materialization.runner,
            binding=materialization.binding,
        ),
        reconnect_metadata=self._reserved_identity(request),
    )
```

Declare `deferred_materialization = True` on the factory. Manifests, `cayu
inspect` and crash-leftover release read that declaration, because they run
without a factory result to inspect, so the runtime rejects a deferred
environment from a factory that does not declare it.

The contract:

- **Fresh resources only.** A clean run end, completed or paused, finalizes the
  binding, which disposes the resource. So `materialize("use")` must never adopt
  a resource it finds under the session's reserved identity: that is a crash
  leftover (or a failed disposal) whose temporary files and processes would
  carry over. Remove it and create a fresh one. Only `materialize("recover")`
  adopts.
- **First use materializes once.** The first command or workspace operation on
  `materialization.runner` calls `materialize("use")` under a single-flight
  lock. Parallel tool calls share one materialization. The runtime admits the new
  runner, the real binding then runs against it (copy-in), and only then does
  the original call proceed. A live admission check only creates the runner:
  it is bound on the first use after that admission passed, never before. Use the factory's ordinary creation path, so admission
  and isolation evidence match an eager start.
- **Admission timing.** Session start keeps the side-effect-free checks: the
  factory's pre-create candidate and the deferred runner's `configured_candidate`.
  Requirements that need live evidence, such as declared executables, materialize
  at session start exactly like an eager environment: `pre_exposure` admission
  requires `live_verified` executable evidence, which configuration cannot
  provide. Built-in tools such as `git_changes`, search, named checks, structured
  commands and the browser declare executables, so exposing them gives up the
  deferral (a warm spare still shortens that start). In that case the runner is created at
  session start only to collect live evidence; the session-start admission check
  evaluates it once, and nothing is copied in until the first use after that
  admission passed. A refused runner is removed without ever receiving the
  workspace. When a runner materializes,
  the runtime runs its bind-time live admission against it before copy-in and
  before the triggering call: final evidence collection, the environment authority
  check and `pre_exposure` evaluation, recorded as `environment.lifecycle.transition`
  `final_evidence` and `admission` events. A refusal, like any creation failure,
  raises `EnvironmentMaterializationError` (a `RunnerUnavailableError`, reason
  `admission_refused`). That fails the triggering tool call; the session stays
  usable, and a later use retries.
- **Finalization.** Finalize, abandonment and completion recovery delegate to the
  real binding once materialized. Without materialization, finalization only calls
  `dispose_unmaterialized` so a resource created by a crashed process cannot leak.
- **Recovery.** `_recover_completion_finalization` calls `materialize("recover")`,
  which must reach the exact resource holding unpublished output and fail closed
  if it is gone. Give that resource a name derived from the allocation identity
  your factory records at session start (for example the allocation ID); then
  recovery reaches the container a crashed process left, the next use removes it,
  and `is_allocation_disposed` can prove absence by that name.
- **Crash leftovers.** `recover_incomplete_session` calls
  `EnvironmentFactory.release_deferred_materialization(request)` with the
  session's allocation reconnect metadata, unless completion finalization is
  pending (that resource still holds output to publish). The factory removes a
  resource a crashed run created for that allocation, and the recovery result
  reports `reaped_allocation`. The session keeps its allocation and materializes a
  fresh resource on its next use. Validate only the metadata's shape and the
  reserved name here, not the current configuration, so an image re-pin since the
  crash cannot strand the leftover. The cleanup is best effort: if it raises (for
  example while the container daemon is down), recovery logs it and completes.
  The leftover is still never adopted: the next `materialize("use")` removes it
  before creating a fresh resource, and finalization of a run that never
  materializes removes it through `dispose_unmaterialized`.
- **Events.** The runtime records `environment.deferred` after binding, and
  `environment.materialization.started` / `.completed` / `.failed` with the
  triggering `trigger_tool_call_id` and `trigger_tool_name`. They are persisted
  and yielded in the live run stream before the triggering tool call's terminal
  event. Each runner has exactly one `started` followed by one `completed` or
  `failed`. When live evidence creates the runner at session start, `started`
  is recorded then, and the same record completes when the first use after
  admission binds it; if it is never bound (for example admission refused it),
  finalization records `failed` with reason `not_bound` and removes the runner.
- **Binding events.** A deferred environment records no
  `environment.binding.started` / `.completed` at session start, because nothing
  binds then. When the runner materializes, the runtime records the same pair an
  eager bind records, with the same binding generation and the real bound
  workspace (path, workspace ID, metadata), between
  `environment.materialization.started` and `.completed`. A run that never uses
  the runner still records the binding finalization events at teardown; they
  report that nothing was published.

`DockerCodingEnvironmentFactory(deferred=True)` implements this contract: the
session's allocation reserves the container name `cayu-coding-<allocation id>`
(derived from the `ealloc_` allocation ID, not the session; without an
allocation ID the name derives from the session, environment and configuration
fingerprint), and the container is created on first use. Every path that acts
on that name (reconnect, use, disposal, discard and crash release) requires
exactly the reserved name. `warm_spares=N` (requires
`deferred=True`, at most 8) keeps N containers pre-started through the exact
cold-start path. Each idle spare holds its memory reservation (512 MiB by
default) and a container slot. `deferred=True` cannot be combined with
`immutable_inputs` yet, and the factory rejects that at construction: immutable
inputs are attached and mounted when an allocation is created, before a deferred
container exists.

### Warm spare pools

`warm_spares=N` composes a `cayu.environments.WarmSparePool` with a Docker
backend; `factory.warm_spare_pool` returns it. The pool is usable on its own: give
it any `WarmSpareBackend` (create a spare through your cold-start path, check that
an idle spare is alive, claim one for a consumer, discard one, and list or remove
spares by name) and call `take(name, requirements)` from your `materialize`
callable, falling back to your cold start when it returns `None`.

- A spare is claimed by exactly one consumer and never returned. Handout re-checks
  liveness. The Docker backend creates spares with the factory's own executable
  requirements. When a session's tools need more (for example `git` or a search
  executable with a probe), the claim verifies them live on that exact container
  through the strict reconnect completion recovery uses, instead of skipping the
  spare. If that verification fails, the pool removes that one spare and sends
  the session to a cold start. When the check found an executable unavailable
  (the image cannot serve those requirements), later sessions with the same
  requirements go straight to a cold start for the pool's reap interval
  (`reap_interval_s`) instead of spending more spares; a transient Docker error
  is not remembered.
  A spare whose liveness check raises is kept for later; one that is no longer
  running is removed and the next spare is tried. `take` returns `None` instead
  of raising for any backend failure, so the caller cold-starts. A cancelled
  `take` returns an unchecked spare to the pool (or removes it if the pool was
  closed, trimmed or refilled meanwhile) and removes one it was claiming. The
  cancellation propagates at once while that removal continues detached; if
  teardown cancels the removal too, stale-spare reaping or the session's
  reserved-name cleanup removes the container later.
- Refill runs in the background, is single-flight, and backs off on backend
  errors.
- Spare names are `<prefix><pid>-<nonce>-<id>`, with a host-specific prefix.
  Refill re-scans once per reap interval (`reap_interval_s`, five minutes by
  default) and removes spares of dead processes or
  closed pools on the same host. A reused PID makes a dead owner look alive until
  that PID exits, so such spares are removed later, never handed out.
- `release_idle()` removes idle spares and lets the next `take` refill.
  `close()` removes them and creates no more. `CayuApp.drain_environment_cleanups`,
  which an idle application may call at any time, calls
  `EnvironmentFactory.release_idle_resources()`, which trims the Docker factory's
  pool. `CayuApp.close_idle_environment_resources()` calls
  `EnvironmentFactory.close_idle_resources()`, which closes it. Application
  shutdown, including server shutdown, calls it after the environment drain
  within the same deadline. Sessions that materialize after that start cold.
- An SDK application that never drains or closes still loses no containers on a
  normal interpreter exit: the pool registers an exit hook that removes its idle
  spares synchronously when the backend provides `remove_spares_at_exit(spares)`
  (the Docker backend runs one bounded `docker rm -f`). A forked child drops the
  spares it inherited and never hands out or removes them; its exit hook only
  acts on spares the child created itself. A process that is killed
  or crashes leaves its idle spares running until another pool on the same host
  reaps them, which happens on that pool's next refill.
- A backend's `claim_spare` raises `WarmSpareRequirementsUnsatisfied` (exported
  from `cayu.environments`) when a spare cannot serve the consumer's
  requirements at all, as opposed to a transient failure.

## Execution admission for custom integrations

Applications attach provider-neutral workload requirements to the agent, not
to a factory or `RunRequest`:

```python
from cayu import AgentSpec
from cayu.environments import ExecutionRequirements

app.register_agent(
    AgentSpec(name="coding-agent", model="provider/model"),
    execution_requirements=ExecutionRequirements.untrusted(),
)
```

Omitting `execution_requirements` preserves the permissive
`ExecutionRequirements.trusted()` default. Existing integrations that make no
admission claim therefore continue to work for trusted agents whose admitted
tools have no external execution requirements. Agent and tool requirements both
fail closed when the integration provides no corresponding evidence.

For a trusted agent whose tools need local execution evidence, use a prepared
checkout with a known path. Unlike the allocating example above, this factory
requires the directory to exist before `app.run`; it does not depend on a
caller knowing runtime-generated session or child IDs:

```python
class PreparedLocalFactory(EnvironmentFactory):
    def __init__(self, root):
        self._root = root.resolve(strict=True)

    def execution_admission_candidate(self, request):
        return LocalRunner(self._root).execution_admission_candidate_for(
            request.execution_requirements
        )

    async def create(self, request):
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(self._root),
                runner=LocalRunner(self._root),
                binding=NativeBinding(),
            )
        )
```

The hook forwards evidence for the same existing root as the final runner.
Pre-create admission must not create directories or start workloads. A refusal
names the tool, its requirement, and each refused capability or executable;
absent evidence points to `EnvironmentFactory.execution_admission_candidate` for
factories or `Runner.execution_admission_candidate_for` for static environments.
A shared prepared checkout does not provide per-session filesystem isolation.

A custom integration supplies evidence at two boundaries. The factory hook
describes the candidate before allocation; the runner hook describes the exact
live candidate that will execute commands:

```python
from cayu.environments import EnvironmentFactory, ExecutionAdmissionCandidate
from cayu.runners import Runner


class HostedFactory(EnvironmentFactory):
    def execution_admission_candidate(self, request):
        return ExecutionAdmissionCandidate(
            candidate="acme-sandbox",
            evidence=self._declare_candidate_evidence(request.execution_requirements),
        )

    async def create(self, request):
        ...


class HostedRunner(Runner):
    def execution_admission_candidate_for(self, requirements):
        return ExecutionAdmissionCandidate(
            candidate="acme-sandbox",
            evidence=self._snapshot_exact_runtime_evidence(requirements),
        )

    async def collect_execution_admission_candidate_for(self, requirements):
        return ExecutionAdmissionCandidate(
            candidate="acme-sandbox",
            evidence=await self._observe_exact_runtime_evidence(requirements),
        )
```

Both evidence objects are `ExecutionCapabilityEvidence` whose `subject` is the
same candidate string. The example's declaration, snapshot and observation helpers are
integration-owned: receiving a requirement is not evidence that it is met.
The snapshot helper must read matching evidence without dispatching work, and
must reflect the exact runner identity observed by the collection helper. An
async collector alone is insufficient: dispatch re-reads the synchronous
snapshot even before evidence expires. If observation requires retained state,
use the request-local observer described below rather than a shared mutable
current plan.
The request includes the exact admitted tools' immutable requirements composed
with the caller's requirements; an integration must not weaken that policy.
The pre-create hook must be side-effect free: it may
publish integration declarations, but it must not allocate a sandbox or claim
that a live resource was observed. The final runner hook may perform bounded
asynchronous inspection of the exact runner, but it must preserve caller
cancellation and either settle every dispatched inspection or transfer an
authenticated settlement owner before any return or exception. Cayu sequences
unexposed release behind a transferred owner. Runners whose evidence is already
immutable can instead implement
the synchronous `execution_admission_candidate()` hook; the default async
collector delegates to it. Cayu calls the async collector after
allocation/reconnect, binding, and setup, so evidence with a validity window
must still be fresh then and at the later dispatch boundary.

The lifecycle creates one `RunnerExecutionAdmissionObserver` after binding
selects the final runner. Its default implementation delegates to the
request-bound collection and snapshot hooks and the runner's refresh hook.
An integration that retains observations should override
`execution_admission_observer(requirements)` with request-local state for the
exact runner and requirements. Do not store a shared mutable current probe
plan: concurrent sessions may admit different tool sets. The observer is not
serializable; recovery creates a new observer for the newly bound runner.

For executable requirements, a non-toolchain `DockerRunner` uses a request-local
observer and the same supervised guest-probe machinery as strict Docker
admission. It requires the exact container ID supplied by creation or reconnect,
checks the running container and immutable image before and after collection,
and timestamps evidence before the first inspection. A name-only runner cannot
provide this proof. The Docker virtual-egress adapter supplies only planned
executable declarations before creation; final executable proof comes from the
bound runner, without strengthening the adapter's security claims.

This non-toolchain Docker collector uses the configured guest working directory
and environment overlay, including credential-bearing overlays. Overlay values
travel through a private environment file, not host-visible argv or the Docker
CLI environment, and participate in output redaction. The owned probe task keeps
the file until the host CLI is terminal, including when caller cancellation
transfers guest settlement to a retained owner. For configured `secret_env`, the
collector uses the same credential-mode validation, read-only secret resolution,
and overlay precedence as Docker execution. Resolution precedes guest dispatch;
lookup failure or cancellation does not produce executable proof. Secret values
and their digests are not published as admission identity. Changing declared
references or the configured resolver invalidates the request-local snapshot.
Executable evidence is a time-bounded observation, not proof of credential
validity or a subscription to changes in an external vault.

A runner may implement `refresh_execution_admission()` to renew only expired
live observations for that exact admitted runner. The common runtime invokes
this hook under its private exposure authority immediately before an actual
provider, provider-backed token-count, or tool dispatch, then re-reads the
side-effect-free admission candidate. The renewed candidate must retain the
same schema, environment fingerprint, immutable-image fingerprint, toolchain
profile fingerprint, candidate, environment authority, and binding generation.
It cannot allocate or select a replacement or mint authorization itself.
Unsupported renewal, failed inspection, identity drift, and evidence that is
still stale fail closed. A runner whose `lifecycle_state` is `closed` or
`poisoned`, for example after timeout or cancellation cleanup removed its
sandbox, is refused before tool-dependent or expired-evidence dispatch without
a renewal attempt, even while its retained evidence is unexpired. The refusal
is `missing_final_evidence` with reason code `runner_closed` or
`runner_poisoned`. A runner whose execution resource is gone for good must
report one of these, not `fenced` (which a runner may reopen): a brokered
sandbox runner whose lease was lost, or whose sandbox stopped being ready, is
`poisoned`. A renewal implementation must settle every dispatched
probe or transfer an authenticated settlement owner on every return and raise
path; the runtime fences later dispatch and final binding cleanup behind that
owner.

### Live tool-requirement qualification

`tests/runners/test_tool_admission_live.py` qualifies `SearchTextTool` through
`CayuApp.run`, using a scripted provider rather than a paid model. Each backend
has a present-`rg` case that must return an actual guest file match, and a
missing-`rg` case that must refuse before any provider request or successful
session. The test driver creates a small uniquely named fixture in the guest.
Local driver tests and evidence fixtures are not live backend qualification.

Prepare targets explicitly before opting in. Both cases need `python3`, `sh`,
and a writable `/workspace`; only the positive case may provide `rg`. Do not
put credentials or valuable data in these disposable targets. The harness does
not install packages, build images, or request image pulls.

- Docker: set `CAYU_860_DOCKER_PRESENT_IMAGE_ID` and
  `CAYU_860_DOCKER_MISSING_IMAGE_ID` to already-loaded immutable `sha256:` image
  IDs. The test checks each exact local image before creating a uniquely named,
  restricted, network-disabled container and removes its container afterward.
  Its test-only transport guard adds `--pull=never` to real Docker creation, so
  a concurrent image prune cannot trigger an implicit download. Guest dispatch
  and admission evidence still use the production Docker implementation.
- Microsandbox: set `CAYU_860_MICROSANDBOX_PRESENT_SANDBOX` and
  `CAYU_860_MICROSANDBOX_MISSING_SANDBOX` to pre-created disposable running
  sandboxes named `cayu-860-live-*`. The test attaches to the exact provider
  incarnation and **removes the supplied sandbox afterward**, including on test
  failure. Re-create these targets before rerunning the qualification.

After exporting the relevant target variables, run one backend at a time:

```sh
CAYU_RUN_TOOL_ADMISSION_LIVE=1 PYTHONPATH=src pytest -q tests/runners/test_tool_admission_live.py -k docker
CAYU_RUN_TOOL_ADMISSION_LIVE=1 PYTHONPATH=src pytest -q tests/runners/test_tool_admission_live.py -k microsandbox
```

Without opt-in, these tests skip. Once opted in, missing configuration or
infrastructure fails qualification rather than being reported as passing or
silently skipped. A skipped or unavailable live test does not satisfy the live
acceptance requirement.

If a binding replaces the factory's runner, the replacement must report the
same admitted candidate. Missing evidence, a changed candidate, insufficient
claims, and expired live observations fail closed before the runner is exposed
to the agent. Cayu evaluates only the selected candidate; admission never
chooses a provider or silently falls back to another environment.

Factories and runners do not authorize themselves. Only the runtime-owned
`EnvironmentLifecycle` can mint the private exposure marker accepted by
runtime model and tool dispatch before downstream provider, runner, workspace,
or tool effects. Runner- or tool-specific admission checks remain defensive
validation and cannot create that marker. Every new or reconstructed
environment used by a run, resume, recovery, or fork traverses the same
selected → preflight → final-evidence → admitted → exposed gate. A same-process
continuation may reuse only its exact process-local exposure authority and
still repeats the dispatch-time structural and freshness check. Factory-backed
environments additionally record allocation or reconnection, and environments
with setup binding record the bound phase.
The durable `environment.lifecycle.transition` events describe those bounded
milestones and unexposed-result release outcomes without publishing commands,
reconnect metadata, credentials, or executable names.

See [Execution admission](runtime-contracts.md#execution-admission) for the
complete capability vocabulary, evidence states, live-proof bounds, structured
refusals, and trusted versus untrusted presets.

## Workspace bindings

A binding subclasses `cayu.environments.WorkspaceBinding` and implements `bind` + `finalize`. `bind`
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
from cayu.environments import SyncBinding, SyncBindingStagingCapacity

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

Completion-critical bindings retain the same durable retry ownership when an
invocation fails or is interrupted. Recovery preserves that original outcome
and retries finalization only; it does not report a successful application result.
Inspect the recovery plan's `environment_recovery.finalization_outcome` alongside
`completion_finalization_pending`. A failed publication or disposal remains pending
until its exact owner settles, including after process restart.

### Shared immutable Docker inputs

Use an immutable input projection for large runtime or support trees that many
Docker environments must read but must never change. Unlike `SyncBinding`, this
path materializes an exact tree once in a private manager store and attaches the
same host object to each container as a read-only bind mount. The mutable
`/workspace` copy-in/copy-back lifecycle remains separate.

```python
from hashlib import sha256

from cayu.environments import DockerCodingEnvironmentFactory, DockerCodingToolchainProfile, ImmutableInputStore, inspect_local_immutable_input
from cayu.runners import DockerImageIdentity
from cayu.workspaces import LocalWorkspace


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
    toolchain_profile=DockerCodingToolchainProfile(
        profile_id="coding", revision="1", image_identity=image,
        platform_architecture="amd64",
    ),
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

Attachment ownership follows the allocation lifetime, rather than the session.
Recoverable creation uses Runtime's durable allocation ID; direct `create()`
reserves an incarnation in the shared input store before admitting attachments.
Retries and reconstructed workers retain that identity. After confirmed disposal,
a new allocation for the same session receives distinct attachment ownership and
can reuse the retained bytes. Released attachment tombstones remain terminal even
after collection. Admission cancellation leaves references with the durable
allocation so another worker's replay cannot release them.

For interrupted Docker binding finalization, Runtime checkpoints exact disposal
authority before removing the container. Successful disposal retires the live
reconnect receipt. Recovery finishes any pending exact disposal before requesting
a new allocation; it never substitutes a new container for an exact reconnect.
Use the same managed immutable-input store across cooperating Docker factories.

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
`SyncBinding`, `GitRepositoryBinding`, `DeterministicWorkspaceBinding`) plus `SyncTargetWorkspacePlan` are available from
`cayu.environments`. Import `Workspace` and `Runner` from their owning packages:

```python
from cayu.environments import WorkspaceBinding, BoundWorkspace, WorkspaceSnapshot
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
