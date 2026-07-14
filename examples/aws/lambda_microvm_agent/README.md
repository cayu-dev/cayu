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
`1.1.1.1:443` access fails. This example disables the link-local metadata probe
because Lambda's managed sidecar ingress uses the same endpoint; the MicroVM
execution role is therefore limited to the selected workspace mount.

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

An interrupted session finalizes the mounted workspace, revokes its grants,
suspends the MicroVM, and persists non-secret reconnect metadata. Resuming
reattaches the same MicroVM with a new broker/grant/CA and remounts the same
access point. Completed and failed sessions terminate the MicroVM.

This example intentionally treats the access-point root as one shared project
workspace. For tenant-isolated production use, provision a distinct access
point (and binding instance) per tenant or project; do not share this root
across unrelated trust domains.

Before deleting a throwaway stack, empty its artifact bucket and, when S3 Files
was selected, its workspace bucket; S3 will otherwise reject their deletion.
The stack deliberately leaves the externally managed PostgreSQL database and
its connection secret untouched.
