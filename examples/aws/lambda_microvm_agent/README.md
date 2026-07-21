# AWS Lambda MicroVM agent stack

This is the integrated AWS deployment example. It runs the trusted Cayu control
plane on ECS/Fargate, gives each agent session a Lambda MicroVM sandbox, mounts a
durable access-point-scoped project workspace, stores selected outputs in S3,
resolves real credentials only inside the Fargate broker, and calls a private
receiver service through a policy-scoped virtual credential.

## Service mapping

| Cayu boundary | AWS service | Why |
|---|---|---|
| Workspace (default) | EFS access point | Durable, low-latency POSIX tree for coding agents |
| Workspace (opt-in) | S3 Files access point | POSIX working set synchronized to an S3 prefix |
| Artifact store | S3 objects | Immutable selected outputs, independent of workspace lifecycle |
| Vault | Secrets Manager | Allowlisted logical secret names; values stay in the trusted task |
| Proxy/broker | In-process Cayu broker in ECS/Fargate | Per-session grants, policy, rewrite, audit, revocation |
| Sandbox | Lambda MicroVM | Session isolation with suspend/resume and durable reconnect identity |
| Model | Bedrock | Uses the Fargate task role; no static AWS keys |
| Runtime state | PostgreSQL | Durable sessions/tasks across Fargate replacement |

The control task is the only component that can read the internal-service
secret. The MicroVM receives a short-lived virtual token and a per-session CA.
Its VPC egress connector security group permits only the control task's dynamic
proxy ports (`1024-65535`) and the workspace mount target on NFS port 2049. The
control API listens on port 800, outside that connector range, and remains
reachable only through the HTTPS load balancer. The MicroVM has no unrestricted
connector. At startup Cayu proves that the private proxy works and that direct
`1.1.1.1:443` access fails. The root sidecar retains AWS managed ingress, while
ordinary commands run in a dedicated network namespace with no default route.
That namespace can reach only a root-owned relay to the enforced private Cayu
proxy; an interface-scoped rule denies its access to sidecar port 8080. Agent
commands also run as UID/GID 1000 with empty capability sets and `no_new_privs`.
This example therefore configures `metadata_isolation="required"` and proves
the denial from the agent command profile without blocking sidecar reply
traffic. The narrowly scoped MicroVM execution role remains defense in depth,
not a network-isolation claim.

The adapter itself defaults to `metadata_isolation="required"`. In required
mode a reachable metadata path produces a typed
`UnsupportedEgressCapabilityError` before setup commands or agent execution;
the error's `remediation` field names the enforceable-topology and explicit
fallback choices. The `unverified` mode remains available for custom or legacy
images that do not install this agent network boundary.

For the internal call, the guest sends:

```text
POST https://receiver.internal/v1/actions
Authorization: Bearer <virtual grant>
```

The proxy enforces that exact host/method/path, resolves the real value from
Secrets Manager, and maps the logical destination to the private Cloud Map
origin `http://receiver.service.local:8080`. The receiver security group accepts
traffic only from the control task, never from the MicroVM connector.

## Prerequisites

- An existing VPC with one private subnet for Fargate, Lambda connector ENIs,
  and mount targets, plus two public subnets for the ALB.
- An ACM certificate for the public HTTPS control-plane listener.
- NAT or equivalent VPC endpoints from the control subnet to Lambda, Bedrock,
  Secrets Manager, S3, ECR, CloudWatch Logs, and STS.
- A PostgreSQL database in the VPC. Put its Cayu connection URL in a Secrets
  Manager text secret and pass the database security group to the stack.
- One unused private IPv4 address for the default EFS mount target. The S3 Files
  option needs a second unused address.
- Docker, AWS CLI v2, and a role allowed to deploy IAM resources.

## Build and deploy

Build and push the Fargate image:

```bash
docker build -f examples/aws/lambda_microvm_agent/Dockerfile -t "$CONTROL_ECR" .
docker push "$CONTROL_ECR"
```

Package the Lambda MicroVM image context and upload it:

```bash
uv run python examples/aws/lambda_microvm_agent/package_microvm.py /tmp/cayu-microvm.zip
aws s3 cp /tmp/cayu-microvm.zip "s3://$BUILD_BUCKET/cayu-microvm.zip"
```

The packager validates and consumes the same versioned sidecar resource produced by
`cayu lambda-microvm sidecar export`. The canonical Dockerfile and entrypoint include the
integrated example's EFS/S3 Files mount helpers, watchdog lifecycle, and agent network boundary;
this example does not maintain a second image implementation or file inventory.

Before deploying, you can ask AWS to build and boot that exact package, verify
the EFS and S3 Files mount helpers, and clean up every temporary resource:

```bash
CAYU_AWS_MICROVM_IMAGE_BUILD_LIVE=1 AWS_REGION=us-east-1 \
  uv run --extra aws python -m examples.aws.lambda_microvm_agent.image_build_live
```

Deploy the stack (the receiver uses the same Fargate image with a different
command):

```bash
aws cloudformation deploy \
  --stack-name cayu-aws-agent \
  --template-file examples/aws/lambda_microvm_agent/infra.yaml \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    VpcId="$VPC_ID" \
    PrivateSubnetId="$PRIVATE_SUBNET_ID" \
    LoadBalancerSubnets="$PUBLIC_SUBNET_A,$PUBLIC_SUBNET_B" \
    CertificateArn="$CERTIFICATE_ARN" \
    ControlImageUri="$CONTROL_ECR" \
    ReceiverImageUri="$CONTROL_ECR" \
    MicrovmCodeBucket="$BUILD_BUCKET" \
    MicrovmCodeKey=cayu-microvm.zip \
    DatabaseUrlSecretArn="$DATABASE_URL_SECRET_ARN" \
    DatabaseSecurityGroupId="$DATABASE_SECURITY_GROUP_ID" \
    BedrockModel="$BEDROCK_MODEL" \
    WorkspaceBackend=efs \
    EfsMountTargetIp="$EFS_MOUNT_IP"
```

Set `WorkspaceBackend=s3files` and pass
`S3FilesMountTargetIp="$S3FILES_MOUNT_IP"` to opt into the S3 Files binding and
its S3-backed resources. The application runs Cayu's idempotent PostgreSQL
migration at startup.

Read the output URL and generated password, then start a run through the normal
Cayu server API:

```bash
URL=$(aws cloudformation describe-stacks --stack-name cayu-aws-agent \
  --query 'Stacks[0].Outputs[?OutputKey==`ControlPlaneUrl`].OutputValue' --output text)
PASSWORD_SECRET=$(aws cloudformation describe-stack-resource \
  --stack-name cayu-aws-agent --logical-resource-id ServerPassword \
  --query 'StackResourceDetail.PhysicalResourceId' --output text)
PASSWORD=$(aws secretsmanager get-secret-value --secret-id "$PASSWORD_SECRET" \
  --query SecretString --output text)
curl -u "admin:$PASSWORD" -X POST "$URL/api/run" \
  -H 'content-type: application/json' \
  -d '{"agent":"aws-agent","prompt":"Request the reindex internal action"}'
```

Run the real metadata-boundary check from outside the VPC. It launches a
one-off task from the deployed control task definition and verifies required
mode's scoped proxy request, public-egress and metadata denial,
the deployed MicroVM execution role, UID/GID 1000, empty capabilities,
`no_new_privs`, the route-less agent network namespace, denial of the trusted
sidecar API, process/filesystem/standard-credential inspection, vault-canary
and AWS-credential non-possession, revocation, a credential-free synced
workspace release, and MicroVM cleanup. The guest returns bounded keyed HMAC
fingerprints of observed candidate values; expected vault/server/database
values remain in the trusted control task and are never passed to guest argv.
For the init network namespace, the verifier accepts either a namespace link
that differs from the agent namespace or an explicit permission denial from
the unprivileged guest; missing paths and other read failures remain errors.
The launcher, trusted control-task probe, and unrestricted guest audit live in
separate modules so each execution boundary remains explicit:

```bash
CAYU_AWS_METADATA_ISOLATION_LIVE=1 \
CAYU_AWS_METADATA_ISOLATION_STACK=cayu-aws-agent \
AWS_REGION=us-east-1 \
  uv run --extra aws --extra egress python -m \
  examples.aws.lambda_microvm_agent.metadata_isolation_live
```

The launcher accepts exactly one
`cayu.aws_lambda_microvm_metadata_isolation.v1` evidence record. Its exact
schema requires every proxy, metadata, public-egress, execution-role,
agent-process, namespace/route, sidecar, vault/credential, revocation,
workspace-release, and cleanup result; missing, extra, or non-verifying fields
fail the check.

An interrupted session revokes its grants, finalizes the mounted workspace,
suspends the MicroVM, and persists non-secret reconnect metadata. Resuming
reattaches the same MicroVM with a new broker/grant/CA and remounts the same
access point. Completed and failed sessions terminate the MicroVM.

This example intentionally treats the access-point root as one shared project
workspace. For tenant-isolated production use, provision a distinct access
point (and binding instance) per tenant or project; do not share this root
across unrelated trust domains.

Delete a throwaway stack with the example's explicit teardown command:

```bash
uv run --extra aws python -m examples.aws.lambda_microvm_agent.teardown \
  --stack-name cayu-aws-agent \
  --region us-east-1 \
  --purge-data
```

`--purge-data` is required because this operation is irreversible. The command
discovers the stack-managed artifact bucket and optional S3 Files workspace
bucket, deletes the stack, and waits for every stack-managed writer to stop.
Both buckets use CloudFormation retention policies, so the command then deletes
their object versions and delete markers in bounded passes until a fresh listing
is empty before deleting the retained buckets themselves. Ordinary stack deletion
therefore preserves durable bucket data. The stack deliberately leaves the
externally managed PostgreSQL database and its connection secret untouched.
If bucket cleanup fails after the stack is deleted, rerun the same command
promptly; it resolves the most recent deleted stack record by name and resumes
cleanup, treating only a confirmed missing bucket as already complete.
