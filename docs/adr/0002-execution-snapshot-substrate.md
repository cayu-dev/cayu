# ADR 0002: Execution snapshots on Docker and AWS Lambda MicroVM

- Status: selected-process feasibility proven; opt-in Docker/AWS DMTCP lifecycle implemented and qualified on UID 1000.
- Date: 2026-10-07.
- Contract: [Execution snapshots](../execution-snapshots.md).

## Scope and decision

Keep Docker for local execution and AWS Lambda MicroVM for production.
Ship **selected-process checkpoints with matching owned files** through DMTCP
on both maintained runners. The [execution snapshot guide](../execution-snapshots.md)
describes the opt-in adapter and public lifecycle APIs. The earlier CRIU and
root-process probes below remain substrate evidence, not product qualification.

The qualified fidelity on both substrates is one cooperating Linux Python
process, including its memory, background writer and local listener, plus three
bounded workspace files and engine support files at a quiescent boundary. The fixture runs as a trusted root process with detached
stdio. Arbitrary tool processes, the actual unprivileged agent profile, PTYs,
browser state, live external connections, devices, mounted volumes, complete
container/VM state and cross-host/provider portability remain unqualified.
Privileged checkpoint control belongs to the trusted controller; agent tools
must not gain its privileges.

[Docker's checkpoint API](https://docs.docker.com/reference/cli/docker/checkpoint/)
is experimental and requires CRIU on the daemon. The available Docker Desktop
daemon has it disabled. The probe instead checkpoints a selected guest process
inside isolated privileged containers. Restoration uses fresh containers with
the same exact image and Linux host kernel. This proves process restoration,
rather than Docker's container-checkpoint API, restart, image/commit or file copy.

[AWS Lambda MicroVM's public operations](https://docs.aws.amazon.com/lambda/latest/microvm-api/API_Operations.html)
have no operation to capture an arbitrary running allocation into an immutable
snapshot for a fresh allocation. Its [documented lifecycle](https://docs.aws.amazon.com/lambda/latest/dg/microvms-how-it-works.html)
preserves memory and disk on same-allocation suspend/resume and builds initialized
image snapshots. Neither exposes the required runtime capture after source
termination. Cayu's suspend hook also cancels supervised commands.

The guest CRIU path remains blocked. A real capture on the staging
workspace image failed because the required kcmp syscall returns **ENOSYS**.
A separate direct syscall probe reproduced it with full trusted capabilities and
Seccomp mode 0. This is a guest-kernel capability gap, rather than a missing
credential or an agent-profile denial. The kernel configuration itself was not
exposed; no particular build option is inferred.

[DMTCP](https://github.com/dmtcp/dmtcp) provides a guest user-space alternative.
Its pinned 4.2.0 release captured the synthetic process, then restored its
memory-only value and matching files twice in fresh staging MicroVMs after
source termination. This path does not require the unavailable CRIU kcmp call.
Checkpointable processes must start under DMTCP's launcher; it does not capture
an arbitrary already-running sandbox or the whole VM.

The opt-in DMTCP adapter now captures under UID 1000 and restores twice through
fresh Cayu processes on Docker and staging AWS, after the source is removed.
The lifecycle pins private artifacts, verifies integrity and exact controller
position, fences ownership, and holds restoration inactive until binding commit.
Memory, SQLite and Postgres conformance tests cover ambiguous acknowledgements
and interrupted-operation reconciliation. See the [feature evidence](../evidence/execution-snapshot-feature-2026-10-07.json).
The AWS kernel request remains prepared for manual submission; CRIU still needs
that kernel capability. Whole-VM and arbitrary-tool continuation remain outside
the adapter fidelity. Existing environments default to unsupported.

## Real probes

The [Docker controller](../../examples/execution_snapshot_probe.py),
[synthetic guest](../../examples/_execution_snapshot_guest.py),
[AWS CRIU capability probe](../../examples/aws/lambda_microvm_execution_snapshot_probe.py),
and [AWS DMTCP proof](../../examples/aws/lambda_microvm_dmtcp_snapshot_probe.py)
use private journals and synthetic data. The
[sanitized evidence](../evidence/execution-snapshot-probe-2026-10-07.json)
contains no snapshot contents, memory-only fixture values, allocation IDs,
image ARNs or credentials. Private directories are mode 0700; controller journals use mode 0600,
atomic replacement and fsync. Copied fixture files retain their guest modes
inside the private directories.

The guest increments an in-memory counter and fsyncs its disk counterpart under
one lock. Another random value exists only in memory. Quiescing waits for the
current write to complete and leaves the writer paused while checkpoint images and
files are captured. Observing a model request is not this boundary.

The Docker probe exercises:

1. Background writes before capture and explicit quiescence.
2. Real controller exits before submission and after artifact acceptance/copy;
   a fresh controller discovers unpublished artifacts and discards them.
3. A published process checkpoint and files-only baseline at the same boundary,
   with process images and matching files fsynced before publication.
4. Source memory/files changed after capture, then source allocation removed.
5. Fresh-controller restoration into new containers. Interrupted restores stay
   inactive; accepted containers are discovered through exact run/operation
   labels and removed without resubmitting the uncertain restore.
6. Memory-only value, memory/disk counter, paused writer and file digests verified
   before the fixture environment is marked active.
7. Repeated restoration of the same artifact into another fresh container.
   The completed fixture action reuses its recorded result rather than repeating
   its filesystem mutation.
8. A host SQLite mutation committed before its acknowledgement is lost. Its
   count stays one and the unresolved outcome remains a reconciliation barrier.
9. Files-only restoration with matching digests and no running fixture process.
   Corrupt and deleted process artifacts rejected before allocation.
10. Every probe-labelled container removed and the scoped inventory empty.

The AWS CRIU probe uses the staging workspace image's exact version through supported
allocation APIs and LambdaMicroVMRunner's trusted command lane. It checks the
guest, builds/uploads a pinned CRIU executable if requested, starts and quiesces
the synthetic process, and attempts real capture. Failed/partial captures remain
unpublished and inactive. It does not contain an unverified AWS restore path.
The exact owned allocation is terminated and its terminal state confirmed.

The DMTCP proof launches its fixture under the checkpoint engine, waits for the
coordinator to return to RUNNING with the single expected peer, and requires one
complete memory image with no temporary image. Command acknowledgement alone
was observed before capture completion and cannot publish the artifact. It
copies the image **and its support files**, plus matching workspace files and
the exact tool archive. Omitting a required mapped gconv cache caused a real
restore failure; including the support set passed. Transfers are chunked,
bounded and SHA-256 verified. Artifact publication follows file/directory fsync.

The source advances after capture and is terminated with confirmation before a
fresh host controller starts restoration. Two new allocations each receive the
same verified artifact and matching runtime/tool bytes, start a fresh DMTCP
coordinator, and verify memory, files and quiescence before activation. The
completed fixture receipt is reused; the host SQLite effect remains committed
once with an unresolved outcome. All three allocations are terminated and
confirmed. Invalid artifact tests reject corruption, deletion and missing
support files before any AWS allocation submission. DMTCP crash/network-fault
injection has not been independently exercised.

The [AWS support request](../evidence/aws-lambda-microvm-kcmp-support-request.md)
contains the kernel reproduction and questions about a supported CRIU or native
runtime snapshot path. It is prepared for manual submission; the staging
Support API returned SubscriptionRequiredException. No case was submitted.

AWS allocation submission has durable intent and an owned stable client token.
An unknown receipt can be reconciled with the exact token/parameters within the
bounded window; expired unknown submissions require explicit reconciliation.
This is allocation idempotency, not snapshot-capture idempotency.

These are substrate/controller experiments, not Cayu durable-tool, run-fencing or
external-effect integration. Controller exits at deterministic boundaries are
not provider-storage crashes or in-flight network partitions. Reusing a fixture
receipt does not establish that production tool execution is safely resumed.

## Compatibility and findings

The Docker guest used Linux 7.0.14-linuxkit, aarch64, CRIU 4.2, 2 vCPUs and
512 MiB RAM. Exact image identity is retained in the evidence. CRIU 3.17.1 failed
on the initialized fixture's socket options; [CRIU 4.2](https://github.com/checkpoint-restore/criu/releases/tag/v4.2)
contains the corresponding Linux 6.16+ fix and passed the complete proof.
Capturing an incompletely initialized process is insufficient: memory-only and
consistent-state verification must pass after source removal.

Staging AWS used Linux 6.1.166-24.303.amzn2023.aarch64, ARM64, a 2 GiB baseline,
trusted elevated capabilities and NoNewPrivs=1. CRIU is absent from the existing
image. The distro CRIU 3.17.1 check failed on socket diagnostics, TUN and
nftables concatenation. Pinned CRIU 4.2 built without nftables got past that
initialization and failed actual capture at kcmp, before any publishable image.
The independently observed syscall failure and Seccomp mode identify the
current kernel gap. Other missing features may still matter after kcmp is fixed.

DMTCP 4.2.0 was compiled inside an isolated staging guest from source archive
SHA-256 043410566fd7c09f21e0ec485cf72e9481722629ea877650142b7505e0d7f2d2.
The exact resulting tool archive was retained privately and reused by SHA-256
3c1dd2f462be790b1586bd0dfa0e93530297c47f62bf9041257d0c7855ebacaa.
The proof reports built_in_probe=false because this final run reused that build.
It verifies the DMTCP version, exact image/version, kernel, architecture, glibc
and libatomic/libstdc++ package versions before restart. The tool archive hash
is part of the immutable artifact integrity set. This is a same-image, same
kernel-version proof, not arbitrary portability. Source and restored fixtures
run in the trusted root lane; UID 1000, agent network namespaces, restricted
syscalls and all production workload profiles still need their own proof.

The [Docker build](../../examples/execution_snapshot_criu/Dockerfile) and
[Amazon Linux build](../../examples/execution_snapshot_criu/Dockerfile.aws-build)
pin CRIU's source version and archive SHA-256. Runs pin the resulting image or
executable bytes. Build dependencies are isolated from the repository lockfile.
The AWS variant uses CRIU's supported build configuration without nftables;
this is not a patched implementation that pretends an unavailable syscall exists.

The evidence reports end-to-end capture/copy and fresh-controller restoration
waits, bytes and files-only results for the same Docker fixture. These are
single-run measurements, not performance guarantees. CRIU process-image bytes
exclude the separately copied fixture files. Provider/internal writes and exact
billing are unmeasured. AWS CRIU has no successful capture/restore measurement;
DMTCP timings below use separate measurement boundaries.

| Single Docker run | Process checkpoint + matching files | Files-only baseline |
| --- | --- | --- |
| Capture/private copy | 0.926 s | 0.339 s |
| Fresh allocation restoration and verification | 2.123 s | 5.449 s |
| Recorded bytes | 13,723,771 process-image bytes plus 4,194,356 file bytes | 4,194,356 file bytes |
| Memory-only state preserved | Yes | No |

The fixture stays paused across fault injection and capture. Its total
agent-visible quiescence interval was not measured separately; capture/private
copy time alone is not that delay. The files-only timing includes container
creation and is not evidence of a consistent speed advantage for CRIU.
The AWS failed capture took 1.200 s; the entire bounded negative probe and
confirmed termination took 50.301 s.

The AWS DMTCP run captured/copied 49,100,413 bytes of process images/support
files and 4,194,356 workspace bytes in 26.367 s. Its quiescence-to-resume interval
was 28.173 s, including private transfer through the trusted command channel.
Process restart and verification took 1.698 s **after** allocation setup and
artifact upload. The complete three-allocation proof and cleanup took 186.577 s.
These timing boundaries differ from the Docker measurement. DMTCP disables
native gzip on ARM64 in this release; transport compression or a durable object
store may reduce copy delay, but neither is qualified by this run. Defer
selective checkpoints and model-wait overlap until the actual agent workload
and complete artifact-store contract are proven.

## Contract facts for implementation

| Concern | Required contract |
| --- | --- |
| Identity and integrity | Publish an immutable pinned manifest after all declared state is durable. Verify the complete artifact set before allocation or activation. Keep private handles out of ordinary events. |
| Compatibility | Bind exact image/version, kernel, architecture, checkpoint-engine bytes/configuration and relevant packages. Reject mismatches. Same-host Docker evidence does not prove another kernel; Each backend and workload profile needs its own qualification. |
| Consistency | Quiesce every included writer. Partial captures cannot become published restore points; cached model responses do not suppress tool execution. |
| Ownership | Restore workloads under fresh control authority. Never revive captured sidecar owner claims, lifecycle leases, proxy credentials or run fences as current authority. |
| Continuation | Bind session/run/tool identity, completed results and compatibility outside the sandbox. Reuse compatible completed effects and reconcile unknown external effects. Snapshot restore does not rewind approvals, transcript, usage, costs or irreversible effects. |
| Retention/deletion | Treat RAM and files as secret-bearing data. Pin artifacts outside allocations, enforce retention and delete them explicitly. Source termination alone does not delete snapshots; API/file deletion does not prove physical erasure. |
| Timeout/cancellation | Local deadlines do not prove guest/provider completion. Retain the exact pending operation and ownership until reconciliation or confirmed cleanup. |
| Idempotency | Guest checkpoint capture has no provider idempotency key. Use unique staging artifacts and avoid automatic adoption of incomplete captures. AWS allocation client tokens resolve an owned request, not capture correctness. |
| Placement and security | Keep privileged control outside agent tools. Preserve principal, ToolPolicy and egress enforcement; rebuild control channels before activation. The trusted-process fixture is not an isolation audit. |
| Region/OS scope | Docker evidence is local Linux aarch64; AWS evidence is staging us-east-1. Other architectures, regions, Windows and full-VM portability are unqualified. |
| Quotas/availability | Docker requires local disk, daemon and compatible kernel resources. AWS requires account/Region capacity and guest features. A bounded test is not SLA or capacity qualification. |
| Cost | Docker artifacts are local; AWS allocations are billable and capped at 600 seconds each. Measure compute, transfers and storage for the eventual workload; artifact size does not establish provider-internal cost. |

## Reproduce

The Docker containers are privileged trusted controllers, with no host mounts,
Docker socket, host PID/network namespace, outbound network or exposed ports.
Docker Desktop experimental settings are unchanged.

~~~sh
docker build -t cayu-execution-snapshot-criu:probe examples/execution_snapshot_criu
python examples/execution_snapshot_probe.py \
  --state-dir /private/tmp/cayu-docker-snapshot-proof
~~~

Install Cayu's existing AWS extra and select a dedicated staging identity plus
the exact Cayu sidecar image/version. This creates one isolated billable
allocation, with a 600-second maximum, and terminates it through scoped cleanup:

~~~sh
python -m examples.aws.lambda_microvm_execution_snapshot_probe \
  --profile cayu-staging-probe --region us-east-1 \
  --image "$CAYU_LAMBDA_MICROVM_IMAGE" --image-version "$CAYU_LAMBDA_MICROVM_IMAGE_VERSION" \
  --install-criu --build-criu --state-dir /private/tmp/cayu-aws-snapshot-proof
~~~

To reproduce the DMTCP proof on the same sidecar image/version:

~~~sh
python -m examples.aws.lambda_microvm_dmtcp_snapshot_probe \
  --profile cayu-staging-probe --region us-east-1 \
  --image "$CAYU_LAMBDA_MICROVM_IMAGE" --image-version "$CAYU_LAMBDA_MICROVM_IMAGE_VERSION" \
  --state-dir /private/tmp/cayu-aws-dmtcp-proof
~~~

This builds the pinned DMTCP release in the disposable source guest, creates
two additional allocations for repeated restore, and caps each at 600 seconds.
A previously built private tool archive may be supplied with --tools-archive;
its exact bytes are hashed and pinned with the checkpoint. That option does not
independently authenticate the archive's source provenance. All journals and
memory images remain private. Runtime dependency installation needs internet
egress on these disposable guests.

The expected current AWS result is blocked_kcmp_ENOSYS with qualified=false,
rather than a successful snapshot. Package/source installation uses internet
egress only on the disposable guest. A compatible pinned Amazon Linux binary
can instead be supplied with --criu-binary, avoiding a repeated guest build.

After an unexpected controller exit, rerun the appropriate module with the
private state directory and --cleanup. Never publish private journals or checkpoint
images. AWS unknown submissions beyond the token window need reconciliation.

## Next qualification gates

Repeat the AWS DMTCP and Docker CRIU proofs under the actual unprivileged
agent profile and network namespace. Preserve fresh control/ownership authority, and prove artifact-store conformance,
retention, resource bounds and image admission before integrating downstream snapshot consumers.

Selective capture, model-wait overlap and co-located scheduling from the
[Crab paper](https://arxiv.org/html/2604.28138v1) remain optional experiments.
Establish fidelity and comparable costs on both maintained backends first.
