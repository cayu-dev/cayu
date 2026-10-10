# Execution snapshots

Cayu can capture and restore an explicitly managed process and its owned workspace
on Docker and AWS Lambda MicroVM. This is an opt-in capability. Existing runners
and environments continue to report `unsupported` unless an adapter is attached.

`DmtcpExecutionSnapshotAdapter` declares `selected_processes`: one Linux process,
its threads, memory, mapped-file support, owned regular workspace files and
directories, and its local listener. The process must start under `adapter.launch`.
It does not capture arbitrary already-running commands, fork trees, PTYs,
browsers, external TCP connections, devices, mounted volumes or a complete VM.
The workspace belongs exclusively to that workload; unmanaged writers are not
supported. This is a cooperative application contract, not attestation of a
hostile guest.

## Prepare an image and environment

Bake DMTCP 4.2.0 and Cayu's compiled barrier plugin into an immutable workload
image at `/opt/cayu-dmtcp`. The [Docker recipe](../examples/execution_snapshot_dmtcp/Dockerfile)
pins the source archive and installs the plugin. For AWS, the [image extension recipe](../examples/execution_snapshot_dmtcp/Dockerfile.aws)
bakes the same tools and plugin into an existing Cayu sidecar/studio OCI image.
Supply its immutable reference with `CAYU_MICROVM_BASE_IMAGE`, then use the normal
MicroVM packaging pipeline and pin the resulting image version. This preserves
the sidecar entrypoint; only the explicitly managed workload starts under DMTCP. The adapter
checks engine/plugin hashes, image identity, architecture, kernel, libc, UID/GID,
capabilities, seccomp and no-new-privileges settings. Capture and restore reject
changed compatibility evidence. Workloads and snapshot transfers use `Runner.exec`
on the unprivileged agent lane; the adapter never installs packages or elevates
privileges.

Provide sufficient scratch space for the process image and transfer archives.
The live fixture uses 256 MiB each for `/tmp` and `/workspace` on Docker; its
snapshot is about 55 MiB. DMTCP memory images can be much larger for a real
application. Byte, inventory, time and record limits are explicit in
`ExecutionSnapshotPolicy`.

Use the same absolute workspace path, workload ID, coordinator port and immutable
image on source and replacement allocations. Allocate a private directory inside
the image's workspace if the image already contains studio/control files. The
restore destination must be empty. Cayu never clears an existing workspace to
make restoration fit.

```python
from cayu import (
    DmtcpExecutionSnapshotAdapter, Environment, EnvironmentSpec,
    ExecCommand, LocalArtifactStore, RunnerWorkspace,
)

# runner is an owned DockerRunner or LambdaMicroVMRunner. Provision this
# directory in the image or through the runner before constructing the adapter.
adapter = await DmtcpExecutionSnapshotAdapter.create(
    runner,
    workload_id="my-application",
    workspace_path="/workspace/my-application",
)
environment = Environment(
    EnvironmentSpec(name="sandbox"),
    runner=runner,
    workspace=RunnerWorkspace(runner, cwd="my-application"),
    execution_snapshot_adapter=adapter,
)
app.register_environment(environment, default=True)
await adapter.launch(ExecCommand.process("python3", "application.py"))

# Keep this store separate from all model-visible environment artifact stores.
# Apply application access controls/encryption appropriate for memory contents.
snapshot_store = LocalArtifactStore(private_snapshot_directory)
```

Capture, restoration and retirement require a pending, interrupted, completed or failed
session with no active run. Capture at an application-controlled idle boundary,
after tools have settled. An outbound model request does not establish this
boundary. Cayu's DMTCP barrier holds all application threads after process capture
while copying files, and keeps a restored process inactive until its binding is
durably verified.

## Capture and restore

```python
session = await app.session_store.load(session_id)
source = app.get_environment("sandbox")
record = await app.capture_execution_snapshot(
    session_id, "sandbox",
    snapshot_store=snapshot_store,
    expected_run_epoch=session.run_epoch,
    expected_generation=source.binding_generation_id,
    idempotency_key="capture-idle-boundary-1",
)
```

The record binds the session incarnation, environment, source allocation hash,
registration generation, run epoch, format/adapter version, compatibility,
controller checkpoint digest and transcript cursor. Both artifacts are pinned in
the private store, read back and verified before publication. Failed or partially
published components never become an authoritative snapshot. The source may be
terminated after successful capture without deleting the retained snapshot.

In a fresh Cayu process, reopen the same session and private snapshot stores,
provision a fresh allocation, and register an equivalent adapter under the same
environment name. Allocation creation and disposal remain application/factory
responsibilities. Then call:

```python
await app.restore_execution_snapshot(
    session_id, "sandbox", record.id,
    snapshot_store=snapshot_store,
    expected_run_epoch=session.run_epoch,
    expected_generation=record.binding_generation,
    idempotency_key="restore-allocation-1",
)
```

For a second restoration, use the *current* durable binding generation, not the
original record's generation. Inspection below reports it. Use a new idempotency
key for a different fresh allocation. Repeating the exact same request returns
its recorded terminal result without issuing another checkpoint or restart.

Restoration verifies all artifacts before target mutation. Its process remains
behind the barrier until the new binding is committed. Runtime environment
exposure, model requests and tool dispatch reject a mismatched binding or an
unsettled snapshot operation.

## Continuation and interrupted operations

Restoration retains Cayu's controller state exactly. It does not rewind the
transcript, recorded tool results, pending actions, observations, approvals,
usage, costs, tasks or external effects. The checkpoint digest and transcript
cursor must still match the capture point. An advanced controller position is
rejected; this release does not reconcile arbitrary older restore points with a
newer execution history. Recorded completed effects stay completed and unknown
external outcomes stay unknown. Snapshot restoration itself executes no tools.

Durable operations distinguish `intent`, `submitted`, `unknown`, `verified` and
`succeeded`. A failure, timeout or cancellation before submission removes the
`intent`, because nothing reached the substrate; the same request can be retried.
Adapters reject a capture that is certain to fail in `preflight_capture`, before
submission. The DMTCP adapter checks the process group, external connections and
workspace eligibility (links, special or unreadable files, file and byte limits)
there, so an ineligible workspace never freezes the workload. A timeout,
cancellation or lost acknowledgement after submission preserves ambiguity and
blocks continuation. Cayu retains cancellation-opaque
adapter tasks until settlement and revokes their publication authority instead
of waiting forever. It never blindly repeats an ambiguous checkpoint or restart.

To recover, attach the exact operation allocation and call
`app.reconcile_execution_snapshot` with its operation ID, original idempotency
key, current run epoch and current binding generation. Reconciliation advances
the run epoch, fences the old owner and proves an existing held capture or
inactive restored process before completing publication/activation. A verified
release can be retried idempotently. If the exact allocation has disappeared or
the adapter cannot prove its state, the operation remains unresolved; inspection
reports the gap. There is no automatic retargeting or abandonment of uncertain
ownership. Once a restore has been submitted, settlement uses its recorded
execution position and the exact target allocation, even if the source snapshot
has since expired or its artifacts are unavailable. This does not admit a new
restore from expired or missing material, or repeat a checkpoint restart.
Resume admission rejects an unsettled operation before changing the
controller checkpoint or appending new messages. Recovery remains available for
failed sessions using their current run epoch. Retry interrupted deletion through
the deletion API with the original idempotency key and current run epoch; already
removed components are treated as deleted.

## Release a lost binding

Capture binds the session to its source allocation, and restoration binds it to
the target. If that allocation is lost (for example the Cayu process restarts with
a fresh allocation) and restoration is not wanted or not possible because the
session advanced past every snapshot, runtime exposure stays blocked. Accept the
loss explicitly:

```python
await app.release_execution_snapshot_binding(
    session_id, "sandbox",
    snapshot_store=snapshot_store,
    expected_run_epoch=session.run_epoch,
    expected_generation=current_binding_generation,
)
```

The current registration becomes the binding, and later captures use it. The old
workload's process state is abandoned. Retained snapshots stay restorable while
their execution position still matches. If the registered environment has no
snapshot adapter, the binding is cleared instead, which requires deleting
retained snapshots first.

## Inspect and retire

`cayu session snapshots SESSION_ID --environment sandbox` uses the normal
[session-store target](session-store-targets.md) configuration and bounded JSON
output. `--limit` defaults to 25 and cannot exceed 100. The server's
`GET /api/sessions/{id}/state` exposes the same safe typed projection, and the
packaged dashboard shows execution snapshots in session details and capability
in environment details. Inspection includes operation, binding, fidelity,
retention, expiry, size and ambiguity without memory, file contents, artifact
locations, pin owners or provider-private handles.

`app.delete_execution_snapshot` takes the snapshot ID, expected current run epoch
and binding generation, and a stable idempotency key. It first retires restore
authority, then releases pins and deletes the two artifacts. Expiry prevents
restoration but does not automatically remove artifacts. Deletion is explicit
and does not prove physical erasure by a storage provider. Partial unpublished
capture artifacts can remain pinned under their deterministic operation identity;
operators must account for these when an unresolved operation cannot be recovered.
Cayu keeps the newest `max_records` succeeded operations as tombstones for
idempotent replay and compacts older ones, so state stays bounded and deletion is
always admissible. Capture is rejected while `max_records` snapshots are retained;
delete some to capture again. Deleting an already-deleted snapshot is a no-op.

## Qualification

The [fresh-process live example](../examples/execution_snapshot_dmtcp_live.py)
uses the public CayuApp APIs with SQLite and a private LocalArtifactStore. It
captures on UID 1000, removes the source, then restores the same snapshot twice
in fresh controller processes and allocations. It checks a memory-only value,
matching writer counter, local listener, once-only fixture effect and preserved
controller evidence. Docker and staging AWS both pass. AWS qualification can
bootstrap tools in *new disposable allocations* using a private configuration;
production images must bake those tools instead.

Run Docker from the repository root:

```sh
docker build -t cayu-snapshot-dmtcp:local -f examples/execution_snapshot_dmtcp/Dockerfile .
python -m examples.execution_snapshot_dmtcp_live --state-dir /private/tmp/cayu-snapshot-feature
```

See [sanitized feature evidence](evidence/execution-snapshot-feature-2026-10-07.json)
and the [substrate ADR](adr/0002-execution-snapshot-substrate.md). Workspace-only
checkpoints, remote allocation reconnect/replacement, and model
provider continuation remain separate capabilities.
