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

The sidecar implements Cayu Lambda MicroVM command protocol version `7`. `GET /health` returns
`{"status":"ok","protocol_version":"7"}` so the host can reject an incompatible image before
sending a command. The host requires the exact version, so an image built from an earlier
release must be rebuilt. Version 2 added the boolean `omit_truncated_output` command field: redacted
executions use it to suppress a channel whose unavailable suffix could complete a workload
secret, while ordinary and trusted executions retain their bounded output.

Version 3 adds a single-owner fence. A host claims the MicroVM with `POST /v1/owner` and a
random 32 to 128 character `claim_id` that it keeps in memory; the sidecar stores only its
SHA-256 digest. Every command start carries `owner_claim`, and a start from any other claim, or
before any claim, is rejected with HTTP 412. A claim that supersedes another cancels every
command of earlier owners, reporting them with `"cancel_reason": "owner_superseded"`, and resets
the agent proxy relay so the new owner's proxy can be installed. Re-presenting the current claim is idempotent. `POST /v1/owner/check` reports whether
a claim is still current. The fence lives in sidecar memory, so it survives suspend and resume;
a restarted sidecar forgets it and every earlier host becomes stale until a new claim.

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
  refused with HTTP 409 instead of leaving a listener behind.

A forward accepts at most 32 connections at a time. A superseding owner claim, the suspend and
terminate hooks, and the forward's `timeout_s` all close it. Forward requests run on their own
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
the retained cleanup. Rebuild the sidecar image to apply these fixes; the protocol stays 7.

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
