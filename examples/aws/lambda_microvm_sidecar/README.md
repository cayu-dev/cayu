# Cayu Lambda MicroVM sidecar image

This is the deployable guest half of `LambdaMicroVMRunner`. Cayu distributions ship this exact
build context as a versioned, self-verifying artifact. It exposes the runner's command protocol,
keeps each command in its own process group, bounds output while still draining pipes, and
confirms timeout/cancellation cleanup before reporting a terminal result. Commands receive only
the explicit environment supplied by Cayu; the image environment is not inherited. When that
environment has no `PATH`, the sidecar uses the guest shell's default
(`/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`) so process commands, shell
commands, and executable admission probes resolve programs identically. The image ships `git`
and a pinned, digest-verified `ripgrep` for coding tools.

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

The sidecar implements Cayu Lambda MicroVM command protocol version `3`. `GET /health` returns
`{"status":"ok","protocol_version":"3"}` so the host can reject an incompatible image before
sending a command. Version 2 added the boolean `omit_truncated_output` command field: redacted
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

- `GET /health`
- `POST /v1/owner`
- `POST /v1/owner/check`
- `POST /v1/owner/lifecycle`
- `POST /v1/owner/lifecycle/release`
- `POST /v1/commands`
- `GET /v1/commands/{command_id}`
- `DELETE /v1/commands/{command_id}`
- AWS lifecycle hooks under `/aws/lambda-microvms/runtime/v1/`

Command IDs are generated by the host. Cancelling an ID before its start request arrives records
a bounded, short-lived cancellation tombstone, so the delayed start remains cancelled instead of
creating an orphan process.

All externally routed requests are protected by Lambda MicroVM's required JWE endpoint token.
The token is generated and refreshed by the host-side runner and is never stored in the image.
