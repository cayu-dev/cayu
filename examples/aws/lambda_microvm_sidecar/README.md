# Cayu Lambda MicroVM sidecar image

This is the deployable guest half of `LambdaMicroVMRunner`. Cayu distributions ship this exact
build context as a versioned, self-verifying artifact. It exposes the runner's command protocol,
keeps each command in its own process group, bounds output while still draining pipes, and
confirms timeout/cancellation cleanup before reporting a terminal result. Commands receive only
the explicit environment supplied by Cayu; the image environment is not inherited.

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

The sidecar implements Cayu Lambda MicroVM command protocol version `1`. `GET /health` returns
`{"status":"ok","protocol_version":"1"}` so the host can reject an incompatible image before
sending a command.

- `GET /health`
- `POST /v1/commands`
- `GET /v1/commands/{command_id}`
- `DELETE /v1/commands/{command_id}`
- AWS lifecycle hooks under `/aws/lambda-microvms/runtime/v1/`

Command IDs are generated by the host. Cancelling an ID before its start request arrives records
a bounded, short-lived cancellation tombstone, so the delayed start remains cancelled instead of
creating an orphan process.

All externally routed requests are protected by Lambda MicroVM's required JWE endpoint token.
The token is generated and refreshed by the host-side runner and is never stored in the image.
