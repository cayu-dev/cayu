# Cayu Lambda MicroVM sidecar image

This is the deployable guest half of `LambdaMicroVMRunner`. Cayu distributions ship this exact
build context as a versioned, self-verifying artifact. It exposes the runner's command protocol,
keeps each command in its own process group, bounds output while still draining pipes, and
confirms timeout/cancellation cleanup before reporting a terminal result. Commands receive only
the explicit environment supplied by Cayu; the image environment is not inherited. When that
environment has no `PATH`, the sidecar uses the guest shell's default
(`/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`) so process commands, shell
commands, and executable admission probes resolve programs identically. Agent commands run as
`cayu-agent` (UID 1000) inside the agent network namespace. The root-owned `ip` and `setpriv`
launchers that put them there start with only that default `PATH`; the command's environment is
applied by `env -i` after setpriv has dropped the identity, groups, and capabilities, so values
such as `LD_PRELOAD` never reach a process running as root. Because `env` reads `NAME=VALUE`
arguments, an agent command's program and environment names must not contain `=`. The image
ships `git` and a pinned, digest-verified `ripgrep` for coding tools.

## Export the installed artifact

Exporting is a local operation. It does not load AWS credentials, contact AWS, create an image,
or require the `cayu[aws]` optional dependency:

```bash
python -m pip install cayu
cayu lambda-microvm sidecar export ./cayu-lambda-microvm-sidecar
```

The exported `cayu-lambda-microvm-sidecar-manifest.json` records the Cayu version, sidecar
protocol version, artifact format version, exact file inventory, and SHA-256 content digest.
The exporter verifies that inventory before writing anything. A non-empty destination is refused
unless `--replace` is supplied; that flag deletes and replaces every existing destination
content. Publication is staged next to the destination and renamed into place. If publication
fails after the old directory has been renamed away, the CLI leaves that directory in a reported
`.cayu-sidecar-backup-*` path for operator recovery rather than risking a second destructive
rename. Filesystem roots, the current working directory and its ancestors, and the user's home
directory and its ancestors cannot be export destinations.

The digest proves which Cayu build context was exported. Runtime compatibility is still decided
by the runner's authenticated `/health` protocol handshake.

## Build the AWS MicroVM image

AWS Lambda MicroVM image creation consumes a zip build context from S3. Package the exported
directory, upload it, and create the image with operator-owned names and roles:

```bash
cd cayu-lambda-microvm-sidecar
zip -r ../cayu-lambda-microvm-sidecar.zip .
cd ..
aws s3 cp cayu-lambda-microvm-sidecar.zip s3://YOUR_BUCKET/YOUR_KEY
aws lambda-microvms create-microvm-image \
  --name YOUR_IMAGE_NAME \
  --code-artifact uri=s3://YOUR_BUCKET/YOUR_KEY \
  --base-image-arn arn:aws:lambda:YOUR_REGION:aws:microvm-image:al2023-1 \
  --build-role-arn arn:aws:iam::YOUR_ACCOUNT:role/YOUR_MICROVM_BUILD_ROLE
```

Wait for the image build to reach `CREATED`, then pass its ARN to
`LambdaMicroVMRunner.create(...)` or set `CAYU_LAMBDA_MICROVM_IMAGE` for the live contract.

For browser tools, export the browser variant instead with
`cayu lambda-microvm sidecar export --browser DESTINATION`. It adds Playwright 1.62.0,
`chromium-headless-shell`, NSS tools, and this Cayu release's browser worker; see
[Lambda MicroVM browser image and admission](https://github.com/cayu-dev/cayu/blob/main/docs/browser-session.md#lambda-microvm-browser-image-and-admission).

Keep three identities separate:

- the operator creating the image may call the image API and pass the build role;
- the build role may read only the selected S3 object and perform required image-build work;
- the Cayu runtime role may run and manage approved MicroVM images but does not need image-build
  or artifact-upload authority.

Never place AWS keys, profile names, account-specific ARNs, endpoint tokens, or application
secrets in the exported directory or image. See
[AWS credentials for Cayu](https://github.com/cayu-dev/cayu/blob/main/docs/aws-credentials.md)
for the complete trust-boundary guidance.

## Networking and cleanup

The control plane needs permission for `lambda:RunMicrovm`, `lambda:GetMicrovm`,
`lambda:GetMicrovmImage` (to pin the exact image version for recoverable allocation),
`lambda:CreateMicrovmAuthToken`, `lambda:SuspendMicrovm`, `lambda:ResumeMicrovm`, and
`lambda:TerminateMicrovm`. Configure the managed ingress connector so Cayu can reach port 8080.
Add only the egress connectors the workload requires; the sidecar itself does not require
unrestricted internet access.

A production AWS deployment can keep the Cayu web/worker control plane on ECS/Fargate with its
session/task stores and AWS role, while each agent session receives a separate Lambda MicroVM
sandbox. The control plane generates short-lived endpoint tokens and sends explicit command
environment values; do not copy the control-plane role credentials or broad application secrets
into the guest image. Persist required patches/artifacts before terminal binding finalization,
then terminate the MicroVM. Interrupted approval/user-input sessions can suspend and later
reattach from the non-secret reconnect metadata emitted by the example environment factory.
Delete obsolete images and uploaded build objects according to the application's retention
policy. Image ownership, AWS charges, and cleanup remain operator responsibilities.

The Dockerfile pins Python 3.11 because the managed AL2023 image's generic `python3` package
currently resolves to Python 3.9. A Bash PID-1 wrapper forwards shutdown signals to Uvicorn and
reaps orphaned command descendants.

This directory is the sole source for the wheel resource, source distribution, and integrated
AWS example image. After changing any file here, regenerate and verify its manifest:

```bash
uv run python scripts/generate_sidecar_manifest.py
uv run python scripts/generate_sidecar_manifest.py --check
```

## Protocol

The sidecar implements Cayu Lambda MicroVM command protocol version `8`. `GET /health` returns
`{"status":"ok","protocol_version":"8","capabilities":["owner_epoch","carry_commands"]}` so the host can reject an incompatible image before
sending a command. The host requires the exact version, so an image built from an earlier
release must be rebuilt. Version 2 added the boolean `omit_truncated_output` command field: redacted
executions use it to suppress a channel whose unavailable suffix could complete a workload
secret, while ordinary and trusted executions retain their bounded output.

Version 3 adds a single-owner fence. A host claims the MicroVM with `POST /v1/owner` and a
random 32 to 128 character `claim_id` that it keeps in memory; the sidecar stores only its
SHA-256 digest. Every command start carries `owner_claim`, and a start from any other claim, or
before any claim, is rejected with HTTP 412. A claim that supersedes another cancels every
command of earlier owners, reporting them with `"cancel_reason": "owner_superseded"`, and resets
the agent proxy relay so the new owner's proxy can be installed. Re-presenting the current claim is idempotent:
it stops nothing new, and it answers only once earlier owners' commands have stopped, so a host
that retries a takeover whose answer it lost never sees it complete early. `POST /v1/owner/check` reports whether
a claim is still current. The fence lives in sidecar memory, so it survives suspend and resume;
a restarted sidecar forgets it and every earlier host becomes stale until a new claim.

Version 8 accepts an optional nonnegative integer `owner_epoch` in `POST /v1/owner`.
The sidecar retains the highest accepted epoch under the owner lock. A lower epoch
returns HTTP 409 before touching ownership, relays, lifecycle leases, or commands;
the host raises `LambdaMicroVMStaleOwnerEpoch` (a `LambdaMicroVMOwnershipSuperseded`
subclass). Equal epochs retain the normal claim rules. Omitted or null epochs keep
legacy last-claim-wins behavior without lowering the retained maximum. The epoch
survives suspend/resume with the in-memory fence, but not a sidecar restart.

Pass `owner_epoch=` to `LambdaMicroVMRunner.create`, `from_existing`, or its direct
constructor. Before relying on this feature, check protocol 8 and the `owner_epoch`
health capability; the runner checks both before sending a numbered claim. This
caller-supplied epoch is separate from the sidecar-generated `generation` returned
by a successful claim.

A protocol 8 sidecar that advertises the `carry_commands` capability lets a new owner take
over the earlier owner's commands instead of cancelling them, so a host handing a MicroVM to
another (a rolling restart, say) doesn't end the commands running in it. The new owner lists
them in its claim: `POST /v1/owner` with `{"claim_id", "owner_epoch", "carry_commands": [ids]}`
(at most 256 IDs; `owner_epoch` is required with it). The sidecar moves the listed commands to
the new owner only when the claim supersedes an owner and its epoch is strictly above every
epoch accepted before, and the owner it supersedes claimed with an epoch too (never an equal,
older or unnumbered one), and does so while it holds the owner fence, so no other claim can
interleave. A moved command keeps running, or keeps a result nobody read yet; one already being
cancelled or released isn't moved, and neither is an interactive one (its stdio moves by offsets
only its first owner knows). Every other command of an
earlier owner is cancelled as without the field, and the response's `carried_commands` lists the
ones the new owner now holds (empty when the claim wasn't granted a carry-over; a repeated claim
reports the same ones). A claim without the field cancels as it always has.

A carried command belongs to its new owner alone: `DELETE /v1/commands/{id}` and
`POST /v1/commands/{id}/release` for it need that owner's claim in the body
(`{"owner_claim"}`), and answer HTTP 412 to anyone else, so the earlier owner can no longer stop
it or drop its result. An earlier owner's takeover still finishing its cancellations leaves it
alone. Commands that were never carried keep the open cancel and release they always had.

On the host, `LambdaMicroVMRunner.exec(..., command_id=...)` names a command so its successor
can list it (not `.` or `..`), `detach_commands()` stops a runner for good without stopping
its commands,
`from_existing(..., owner_epoch=..., carry_commands=[...])` claims with the list, and
`carried_commands` and `resume_exec(command_id)` give the new owner each carried command's
result. The runner sends the list only to a sidecar that advertises the capability
(`command_carry_supported`); any other cancels the commands as before.

Takeover is atomic with command admission. The owner check, request validation (which may
install the agent proxy relay), and registration of a start all run while the fence is held,
and a claim swaps the owner, resets the relay under the same lock, then cancels every command
admitted by an earlier generation before it responds. A start is therefore either refused or
registered in time to be cancelled; a completed claim leaves no earlier owner's command able to
run. Cancellation is generation-specific, so it never stops a newer owner's commands, and a
command cancelled before it spawned never starts.

Before a control-plane suspend or terminate, the host takes a lifecycle lease with
`POST /v1/owner/lifecycle` and `{"claim_id", "action": "suspend" | "terminate"}`. The lease
confirms the caller is the current owner (HTTP 412 otherwise) and, until it ends, every other
claim and every command start is refused with HTTP 423. The host sends exactly one provider
request per lease: SDK retries are refused before they are sent, and a host retry takes a new
lease, which only the current owner can do.

The lease never expires by time, because no elapsed interval proves that an accepted request
has finished. It ends only on authoritative evidence:

- the guest's terminate hook (for any lease), or its suspend or resume hook (for a suspend
  lease), which show the provider acted;
- `POST /v1/owner/lifecycle/release` from the owner when its single request was never sent or
  AWS definitively rejected it. After a timeout or other ambiguous outcome the host keeps the
  lease.

The cost is liveness: if a host dies while holding a lease, or its request never takes effect,
no other host can claim this MicroVM until it ends. The runtime's allocation reap can terminate
it, and at the latest AWS ends it at its maximum duration. A MicroVM idle policy suspends on its
own and also fires the suspend hook, so with an idle policy configured a suspend lease can be
settled by the platform's suspension rather than the owner's request.

Version 4 sizes transfers for the browser worker and stops retaining request and result data.
Stdin may be up to 24 MiB and output up to 32 MiB per command; the host refuses larger stdin
before dispatch. The sidecar keeps only a SHA-256 digest of each start payload for idempotent
replays, and `POST /v1/commands/{command_id}/release` drops a terminal result once the host has
read it, leaving a digest-only tombstone. A start payload may carry
`CAYU_EGRESS_PROXY_TRANSPORT_TOKEN` in its environment: the sidecar removes it from agent
commands and its root relay presents it to the Cayu proxy in an authenticated transport tunnel
before relaying any agent bytes, so a proxy listener reachable from other MicroVMs on the shared
connector is usable only through this MicroVM.

Version 5 adds the browser control relay. `POST /v1/control-relay` with the current
`owner_claim`, a private RFC 1918 IPv4 `host`, and a `port` relays `cayu-control:18443` in the
agent namespace to that target: the entrypoint accepts only TCP 18443 (besides the proxy relay's
18080) from the agent namespace and maps `cayu-control` to the namespace gateway in
`/etc/netns/<namespace>/hosts`. The relay forwards TCP only, so TLS to the Cayu control server
stays end to end. The target is fixed for the current owner (a different target returns HTTP
409), and a superseding owner claim or the suspend hook clears it. Browser operator view,
takeover, and recording use this path; they never use the egress proxy.

Version 6 adds interactive commands, whose stdio streams while they run. A start with
`"interactive": true` (agent profile only, no `stdin_base64`) returns
`"process_tree": "pid_namespace"`: `unshare --pid --fork --kill-child` gives the command its own
PID namespace, and inside it, after setpriv has dropped the identity, a small init runs the
command and reaps orphans. When the command exits the init exits with its status and the kernel
kills everything else in the namespace, so a descendant that left the process group cannot
survive it. Cancellation kills the init rather than `unshare`, and `unshare` exits only after
reaping it, so a terminal result reports `"process_tree_stopped": true` only once the whole tree
is gone. A record whose tree was not confirmed stopped is never pruned, so a later cancel cannot
be answered by a tombstone.

Interactive stdio is plain request/response over the same authenticated endpoint, and only the
current owner may move it:

- `POST /v1/commands/{command_id}/stdin` with `owner_claim`, a byte `offset`, `data_base64` (up
  to 4 MiB) and `eof`. Bytes before the sidecar's current offset are skipped, so a retried write
  is idempotent; an offset past it is refused. The response reports the new offset and whether
  stdin is closed.
- `POST /v1/commands/{command_id}/stdout` with `owner_claim`, an `offset`, `max_bytes` (up to
  4 MiB) and `wait_s` (up to 20 seconds). It long-polls for bytes at that offset. Reading at an
  offset acknowledges every earlier byte, which is then dropped; reading the same offset again
  recovers a lost response. Unread output is held up to 8 MiB, after which the sidecar stops
  draining the pipe, so a slow host slows the command instead of growing sidecar memory. The
  response reports `eof` once all output was read, with the command state.

Stderr of an interactive command is drained and discarded, and its terminal result carries no
output. Stdio requests run on their own thread pool so long polls cannot starve other endpoints.

Version 7 adds port forwards, which let agent commands reach a host service, such as Cayu's
external-agent model gateway, with no route from the guest to the host. `POST /v1/forwards` with
the current `owner_claim`, a host-generated `forward_id` and an optional `timeout_s` opens a
listener on the agent veth gateway, `192.0.2.1`, at the lowest free port from 18500 to 18563, and
returns `{"forward_id", "host", "port"}`. The entrypoint accepts only that range (besides 18080 and
18443) from the agent namespace, so nothing outside the MicroVM can connect. The relay carries
plain TCP: the host opens its own connection to the service for each guest connection and moves
the bytes over this authenticated endpoint, so the hop inside the guest can be plain HTTP and no
CA has to be installed in the image. Only the current owner may use a forward:

- `POST /v1/forwards/{forward_id}/accept` with `after` and `wait_s` (up to 20 seconds) long-polls
  for connections numbered above `after` and reports whether the forward has closed.
- `POST /v1/forwards/{forward_id}/connections/{connection}/read` with `offset`, `max_bytes` (up to
  4 MiB) and `wait_s` long-polls the guest's bytes. As with interactive stdout, reading at an
  offset acknowledges every earlier byte, and reading the same offset again recovers a lost
  response. Unread bytes are held up to 4 MiB per connection, after which the sidecar stops
  reading the socket, so a slow host applies TCP backpressure to the guest. The response reports
  `eof` once the guest closed its side and every byte was read.
- `POST /v1/forwards/{forward_id}/connections/{connection}/write` with `offset`, `data_base64` (up
  to 4 MiB) and `eof` sends the service's bytes to the guest. Bytes before the current offset are
  skipped, so a retried write is idempotent; `eof` closes the guest's read side.
- `POST /v1/forwards/{forward_id}/connections/{connection}/close` ends one connection.
- `POST /v1/forwards/{forward_id}/close` closes the listener and every connection. Closing an
  unknown forward leaves a short-lived tombstone, so an open that arrives after its own cleanup is
  refused with HTTP 409 instead of leaving a listener behind. A superseded owner's close changes
  nothing and is answered from the registry with `{"forward_id", "closed", "owner_superseded":
  true}`: `closed` is true only once the forward is neither admitted nor retained for teardown,
  so the owner never counts a forward closed while a failed takeover teardown still holds it.

A forward accepts at most 32 connections at a time. A superseding owner claim, the suspend and
terminate hooks, and the forward's `timeout_s` all close it. The hooks close forwards even when
command cancellation fails, and then report the first failure. Forward requests run on their own
thread pools, as stdio requests do: accept and read long polls on one pool with room for every
poll the 64 ports and 32 connections per forward allow, and writes and connection closes on
another, so host bytes never wait behind a long poll.

Concurrent closes share one retained teardown. A duplicate close or lifecycle hook waits for
the listener and accepted connections to close; removing admission or reaching a join timeout
does not confirm closure. Failed teardown remains fenced and retryable. A rejected-client setup
or close failure immediately attempts teardown; it does not abandon the listener or healthy
connections. Aggregate teardown attempts every independently reachable connection and forward
before reporting its first failure. Persistent failures still require a successful explicit close.
If this interrupts an
owner takeover, both owners have no authority until the original cleanup callback succeeds.
Retrying the same claim settles that takeover without another generation; a third claim must
settle it before advancing ownership. Omitting or replacing the retry's callback cannot bypass
the retained cleanup. Rebuild the sidecar image to apply these fixes and the protocol 8
owner-epoch fence.

- `GET /health`
- `POST /v1/owner`
- `POST /v1/owner/check`
- `POST /v1/owner/lifecycle`
- `POST /v1/owner/lifecycle/release`
- `POST /v1/commands`
- `GET /v1/commands/{command_id}`
- `POST /v1/commands/{command_id}/release`
- `POST /v1/commands/{command_id}/stdin`
- `POST /v1/commands/{command_id}/stdout`
- `POST /v1/control-relay`
- `POST /v1/forwards`
- `POST /v1/forwards/{forward_id}/accept`
- `POST /v1/forwards/{forward_id}/connections/{connection}/read`
- `POST /v1/forwards/{forward_id}/connections/{connection}/write`
- `POST /v1/forwards/{forward_id}/connections/{connection}/close`
- `POST /v1/forwards/{forward_id}/close`
- `DELETE /v1/commands/{command_id}`
- AWS lifecycle hooks under `/aws/lambda-microvms/runtime/v1/`

Command IDs are generated by the host. Cancelling an ID before its start request arrives records
a bounded, short-lived cancellation tombstone, so the delayed start remains cancelled instead of
creating an orphan process.

All externally routed requests are protected by Lambda MicroVM's required JWE endpoint token.
The token is generated and refreshed by the host-side runner and is never stored in the image.

## Kubernetes mode

The same image can supervise a workspace inside a Kubernetes sandbox pod, such as a gVisor
Agent Sandbox. Set `CAYU_SIDECAR_MODE=kubernetes` on the supervisor container. The default is
`lambda`; any other value stops the sidecar at startup. The protocol version stays `8`, and Lambda
mode is unchanged.

On the host, `cayu.runners.KubernetesSandboxRunner` drives a sidecar in this mode: it sends the
bearer token, refuses a sidecar whose health lacks `bearer_auth` and `boot_id`, and exposes the
boot ID it last saw. See the `KubernetesSandboxRunner` section of
[Runtime Contracts](https://github.com/cayu-dev/cayu/blob/main/docs/runtime-contracts.md).

### Two containers

A sandbox policy that refuses root containers and added capabilities rules out the Lambda shape,
where a root sidecar drops each command to the agent UID. The pod instead runs this image twice,
as two non-root containers that share no volume:

- The **supervisor container** runs `entrypoint.sh` as its own UID (for example 10001), which
  starts only Uvicorn and refuses to start as root. It holds everything the host talks to and
  every secret: TLS, the bearer token, the owner fence, command records, output limits and
  buffering, and port forwards.
- The **agent container** runs the command executor as PID 1, as the agent's UID (for example
  1000, `cayu-agent`), and mounts the workspace at `/workspace`:
  `command: ["python3.11", "-m", "lambda_microvm_sidecar.executor"]` with the image's working
  directory, `/opt/cayu`. The executor spawns, feeds, drains, signals and confirms commands on
  the supervisor's behalf and holds nothing else. It uses only the standard library (Python 3.11
  or later), so the agent container may also run another image, such as an application's own,
  with the executor copied into a read-only volume (for example `/opt/cayu-exec`, filled by an
  init container from this image) and started by that image's base interpreter in isolated mode:
  `["/usr/local/bin/python3", "-I", "-S", "-c", "import sys; sys.path.insert(0, '/opt/cayu-exec');
  import runpy; runpy.run_module('lambda_microvm_sidecar.executor', run_name='__main__',
  alter_sys=True)"]`. Name the base interpreter rather than `python3` from `PATH`, which may be an
  application virtual environment whose `.pth` files would run inside the executor. Every
  command's launcher also runs that interpreter and its standard library outside the agent
  namespace, as the agent's UID, so neither may be writable by the agent user: keep them
  root-owned, or set `readOnlyRootFilesystem: true`. Such an image may lack `/bin/bash`, so send
  it process commands rather than shell commands.

Set `runAsNonRoot: true`, `allowPrivilegeEscalation: false` and `capabilities: {drop: ["ALL"]}` on
both containers, and request at least 1 GiB of memory per requested vCPU, which GKE's sandbox
policy requires. Both containers need `CAYU_MICROVM_WORKSPACE_ROOT` when the workspace is not at
`/workspace`; the agent container needs `CAYU_EXECUTOR_SUPERVISOR_UID`, the supervisor
container's UID. Agent commands run as the agent container's own user: the supervisor uses no
`setpriv` and has no agent UID to drop to, so `CAYU_MICROVM_AGENT_UID` and `CAYU_MICROVM_AGENT_GID`
are ignored.

The executor listens on an abstract Unix socket, `@cayu-sidecar-executor` unless
`CAYU_EXECUTOR_SOCKET` names another (on both containers); a value without `@` is a filesystem
path, which would need a shared volume. Containers in a pod share its network namespace, which
scopes abstract socket names, so the two containers meet without a volume. The executor binds the
name before anything else and serves only callers whose peer credentials (`SO_PEERCRED`) carry
`CAYU_EXECUTOR_SUPERVISOR_UID`; agent commands share the network namespace but never that UID,
and their connections are refused before any request is read. The peer PID is not used: across
containers under gVisor it reads 0. The executor refuses to start as root or when the supervisor
UID is its own, and marks itself not dumpable, so agent commands, which run as its UID, cannot
read its memory or environment.

The executor runs every agent command in an agent namespace: user, mount and PID namespaces it
creates at start, whose user namespace maps only its own UID and GID and whose init mounts a
`/proc` of its own and reaps orphans. Commands enter it through a launcher that the executor's
own interpreter runs in isolated mode with only the default `PATH`: it enters the namespaces with
`setns`, changes to the working directory inside, and executes the command with exactly its
environment, so values such as `LD_PRELOAD` never run where they could reach the executor. An
agent command's environment names must not contain `=`. The executor needs only its interpreter,
no `nsenter`, `env` or shell from the image. The executor stays outside, so no agent command can
signal it or see it in `/proc` (a sandbox's PID 1 is not protected from `kill -9` by its own
processes under gVisor). If an agent kills the namespace's init, every agent process dies with
it; the executor creates a new agent namespace and draws a new boot ID. Files are untouched. A
background server a command starts with `setsid` outlives that command, inside the namespace,
until the namespace is replaced.
Mounting that `/proc` needs a runtime that does not mask `/proc` paths; gVisor does not, and a
runtime that does stops the executor at startup with an error.

Interactive commands get a PID namespace of their own inside the agent namespace when the
sandbox allows nested namespaces, which the executor probes at start: they report
`"process_tree": "pid_namespace"`, and closing one confirms every descendant exited, as in Lambda
mode. Otherwise they report `"process_group"`: cancellation signals the process group, and a
descendant that left it keeps running until the agent namespace ends.

### Executor protocol

Each request is one connection: the supervisor connects, sends one frame and reads one frame
back. A frame is a 4-byte big-endian length followed by at most 2 MiB of UTF-8 JSON, and bytes
travel base64-encoded. Every request names an `op` and the supervisor's `session` (its boot ID);
every answer carries the executor's `boot_id`, and a refusal carries `error` with a `kind`
(`invalid_request`, `conflict`, `unknown_command`, or `os_error` with the spawn's `errno`).

| `op` | Fields | Answer |
| --- | --- | --- |
| `hello` | | `pid_namespace`, `protocol_version` (`1`) |
| `resolve_cwd` | `cwd` | `cwd`, resolved inside the workspace |
| `spawn` | `command_id`, `argv`, `env`, `cwd`, `process_tree` | `pid`, `process_tree` |
| `read` | `command_id`, `stream` (`stdout` or `stderr`), `offset`, `max_bytes` (at most 1 MiB), `wait_s` | `data_base64`, `eof` |
| `write` | `command_id`, `offset`, `data_base64` (at most 1 MiB), `eof` | `offset`, `closed` |
| `wait` | `command_id`, `wait_s` | `returncode`, null while it runs |
| `stop_group`, `stop_tree` | `command_id` | |
| `tree_stopped` | `command_id`, `timeout_s` | `stopped` |
| `release` | `command_id` | |

Waits are at most 20 seconds. Every request is idempotent, so a connection reset, for example by
a snapshot and restore, only causes a retry; the supervisor retries for up to 30 seconds. A
repeated `spawn` returns the process the first one started, and a different command under the
same ID is a conflict. Reading at an offset acknowledges every earlier byte and a lost answer is
recovered by reading the same offset again; writing names the offset it starts at, so a replayed
chunk is skipped. The executor holds up to 4 MiB of unacknowledged output per stream and then
stops draining the pipe, so the supervisor's output limits and interactive backpressure still
reach the command. A `write` returns once the command has taken the chunk, however long that
takes, so a slow reader holds the writer back as a local pipe does instead of failing it. A
request in a new `session` first stops every command of the earlier one and replaces the agent
namespace, which ends any background process they left, so a restarted supervisor leaves nothing
running that it cannot reach; requests wait until the new namespace is in place.

### Boot ID and health

`GET /health` returns
`{"status":"ok","protocol_version":"8","capabilities":["owner_epoch","carry_commands","bearer_auth","boot_id","egress_relay"],"boot_id":"<uuid>"}`.
The `boot_id` is one UUID derived from the supervisor's boot ID, drawn when it starts, and the
executor's, drawn when it starts and again whenever it replaces the agent namespace, so it changes when
either container restarts or the agent's processes are lost. When the supervisor sees the
executor's boot ID change, it drops the current owner and cancels every command and closes every
forward, as its own restart would: the host's next fenced call is refused and a new claim starts
afresh. While the executor is unreachable, `GET /health` answers HTTP 503, so the supervisor
container's readiness probe fails, and a command start answers HTTP 503 too.

### Host authentication and TLS

Without Lambda MicroVM's endpoint token in front of it, the sidecar authenticates its host
itself. Every route, `GET /health` included, requires `Authorization: Bearer <token>` and
answers HTTP 401 otherwise; the sidecar compares SHA-256 digests in constant time. The token is
read at startup from the file named by `CAYU_SIDECAR_TOKEN_FILE`, never from the environment,
and the sidecar refuses to start when that file:

- is missing or not a regular file;
- holds fewer than 32 bytes (surrounding whitespace is ignored);
- is owned by neither UID 0 nor the sidecar's own UID;
- is writable by its group (`0o020`) or accessible to others (any bit of `0o007`).

Its group may read it, so mount it from a Secret volume in the supervisor container only, with
`defaultMode: 0440` and a pod `fsGroup`. Kubernetes makes Secret files owned by root, so
`defaultMode: 0400` without an `fsGroup` leaves them unreadable to the non-root sidecar, which then
does not start. The agent container mounts neither the token nor the key. Before every check the
sidecar looks at the file's modification time and reads the file again when it has changed, so a
token rotated on disk replaces the earlier one on the next request; an unsafe replacement is
ignored and the earlier token stays.

The host reaches the sidecar over the pod network, so in Kubernetes mode it serves only HTTPS and
the token never crosses that network in clear. Set `CAYU_SIDECAR_TLS_CERT_FILE` to its PEM
certificate and `CAYU_SIDECAR_TLS_KEY_FILE` to the PEM private key; the entrypoint hands both to
Uvicorn. The sidecar refuses to start when either is unset, when the certificate and key don't
load as a pair, or when the key file fails the token file's checks above, and it answers HTTP 401
to any request that did not arrive over TLS. Mount the key from the same kind of Secret volume as
the token. A self-signed certificate issued per sandbox for the host name the host connects to
(for example its Service's `<name>.<namespace>.svc.cluster.local`) lets the host pin exactly that
certificate, as `KubernetesSandboxRunner`'s `verify` does with
`ssl.create_default_context(cadata=...)`. Uvicorn loads the certificate once at startup, so a pod
restored from a snapshot keeps serving the one it started with; replace the sandbox, rather than
its Secret, to change it.

### Routes and network

Kubernetes mode does not serve the Lambda MicroVM hooks under
`/aws/lambda-microvms/runtime/v1/`, the lifecycle lease routes `POST /v1/owner/lifecycle` and
`POST /v1/owner/lifecycle/release`, or the API schema. Every other route behaves as in Lambda mode.

The entrypoint creates no agent network namespace, veth pair or firewall rules, and starts no
EFS mount watchdog; `CAYU_MICROVM_AGENT_NETNS`, which the image sets, is ignored. Agent commands
share the pod's network, so the pod's NetworkPolicy is the network boundary: admit only the host
to port 8080 and restrict egress to what the workload needs. Two ports are reserved in the pod,
because agent commands share its network: 8080, which the sidecar serves, and `127.0.0.1:18080`,
which it holds for the egress relay from the moment it starts (see below). A workload must not
listen on either. Port forwards listen on the pod's loopback, `127.0.0.1`, at ports 18500 to
18563, which agent commands reach from their container.
`POST /v1/control-relay` is refused, because there is no namespace to relay from. The image creates
`/workspace` owned by the agent UID and GID at build time; a volume mounted there must be owned the
same way.

### Egress relay

A sandbox reaches the network through the Cayu egress proxy as in Lambda mode. The host names the
proxy in a command's `HTTPS_PROXY` or `HTTP_PROXY` (either spelling), as an unauthenticated
`http://` origin at a non-public IPv4 address (see below), and passes the sandbox's transport
token in `CAYU_EGRESS_PROXY_TRANSPORT_TOKEN` (64 lowercase hex characters). The supervisor removes
the token from every command, in both execution profiles, before the command reaches the executor,
so the agent container never receives it. It rewrites the command's proxy variables to the relay,
`http://127.0.0.1:18080`; `NO_PROXY` and every other variable reach the command unchanged. For
each connection the relay connects to the proxy, sends `CONNECT cayu-transport.invalid:443` with
the token as `Proxy-Authorization: Basic` credentials for user `cayu`, and relays the command's
bytes only once the proxy answers 200, so the command's own CONNECT reaches the proxy only over
an authenticated connection. A connection the proxy does not admit is closed.

- The supervisor binds `127.0.0.1:18080` when it starts, before any agent command can run, and
  holds it until it exits, so no agent process can take the port over. If something already
  listens there, the sidecar refuses to start and names the port.
- Until a command brings proxy settings, the relay has no target and closes every connection as
  soon as it accepts it. Without proxy settings commands get no proxy variables (a token alone is
  still removed), as for a sandbox without egress.
- The first proxied command of an owner points the relay at that command's proxy and token. Until
  a superseding owner claim, a command naming another proxy or token is refused with HTTP 400.
  After such a claim, or an executor restart, the relay keeps its port and its previous target,
  and the new owner's first proxied command retargets it to that owner's proxy and token in one
  step. Tunnels already open are left alone, and carried commands keep the same proxy variables,
  so they keep working across the handover.
- A command that names a proxy without a token, with a malformed token, at a public or otherwise
  refused address, with any scheme but `http`, or at two different proxies is refused with HTTP
  400, and the relay's target is left as it was.
- The relay holds at most 64 connections, as many tunnels as the proxy allows one sandbox, and
  closes any further connection as soon as it accepts it. Connecting to the proxy and the proxy's
  answer to the token are each bounded by 5 seconds; an open tunnel has no idle limit.

The proxy's address must not be globally routable, because Kubernetes pod networks are not always
RFC 1918: GKE, for example, can assign pods the shared address space `100.64.0.0/10` or the
reserved `240.0.0.0/4`. The relay accepts any IPv4 address that Python's `ipaddress` does not
consider global (`is_global` is false), except addresses that never name another pod: `0.0.0.0/8`,
loopback `127.0.0.0/8` (the pod itself), link-local `169.254.0.0/16` (instance metadata servers),
multicast `224.0.0.0/4` and `255.255.255.255`. Public addresses, including public ranges a cluster
uses privately, are refused. Lambda mode keeps its RFC 1918 rule.

`GET /health` advertises `egress_relay` in Kubernetes mode, so a host can check for the relay
before it gives a sandbox egress. Agent commands can also reach the proxy's address directly, so
the pod's NetworkPolicy should allow egress only to that address and port; the proxy closes a
connection that does not present the token without answering it.
