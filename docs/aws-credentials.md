# AWS credentials for Cayu

Cayu's Amazon Bedrock provider and AWS Lambda MicroVM runner use Boto3's standard
AWS credential chain. Cayu does not require a Bedrock API key, an Anthropic API
key, or a Lambda MicroVM-specific long-lived secret.

An operator must provide three separate things:

1. an AWS identity that Boto3 can resolve;
2. an AWS Region; and
3. IAM permissions for the Bedrock or Lambda operations being used.

The recommended identity is temporary: an IAM Identity Center session, an
`aws login` session for local development, or an IAM role attached to the
workload. Do not create root access keys, and do not give a coding agent root or
administrator credentials.

## Which credential values are needed?

| Where Cayu runs | Recommended credential source | Values configured in Cayu |
| --- | --- | --- |
| Enterprise developer workstation | IAM Identity Center named profile | `AWS_PROFILE` and `AWS_DEFAULT_REGION`, or `profile_name=` and `region_name=` |
| Personal or small-account workstation | Named `aws login` profile using a non-root identity | `AWS_PROFILE` and `AWS_DEFAULT_REGION`, or `profile_name=` and `region_name=` |
| ECS/Fargate, EC2, or EKS | Task role, instance profile, Pod Identity, or web identity | `AWS_DEFAULT_REGION` or `region_name=`; Boto3 discovers temporary role credentials |
| External CI or hosted coding agent | OIDC federation into a dedicated IAM role | `AWS_DEFAULT_REGION` or `region_name=` after the CI platform obtains temporary role credentials |
| Legacy environment that cannot assume a role | Dedicated least-privilege IAM user | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` when temporary, plus a Region |

An AWS access key consists of an access-key ID and secret access key. Temporary
credentials add a session token. A Region and profile name are ordinary,
non-secret configuration. Never commit or disclose access-key IDs, secret keys,
session tokens, or cached sessions; do not paste them into an agent prompt,
store them in Cayu session metadata, or copy them into a Lambda MicroVM guest.

## Recommended local setup

In a Cayu source checkout, install the AWS support with:

```bash
uv sync --extra aws
```

In an application that depends on Cayu, install the optional package extra:

```bash
uv add "cayu[aws]"
# or
python -m pip install "cayu[aws]"
```

Install a current AWS CLI separately and verify it:

```bash
aws --version
```

### Enterprise: IAM Identity Center

Ask the platform team for a permission set scoped to the development account and
the Bedrock models or Lambda MicroVM resources you need. Then configure and sign
in to a named profile:

```bash
aws configure sso --profile cayu-dev
aws sso login --profile cayu-dev
export AWS_PROFILE=cayu-dev
export AWS_DEFAULT_REGION=us-west-2
aws sts get-caller-identity
```

`aws configure sso` records the account, role, and Identity Center session. It
does not create a long-lived IAM access key. Re-run `aws sso login` when the
workforce session expires.

### Local AWS console login

AWS CLI 2.32.0 or newer can exchange an existing console login for temporary
credentials. Cayu's `aws` extra includes the CRT dependency required for Boto3
to consume this credential provider:

```bash
aws login --profile cayu-dev
export AWS_PROFILE=cayu-dev
export AWS_DEFAULT_REGION=us-west-2
aws sts get-caller-identity
```

Use a dedicated non-root IAM identity or federated role. The identity needs
AWS's `SignInLocalDevelopmentAccess` permission in addition to the service
permissions below. Do not select the account root identity. Check the resolved
ARN before starting a coding agent or live test:

```bash
aws sts get-caller-identity --query Arn --output text
```

If the result ends in `:root`, stop and replace the profile with a non-root
identity. Root is an account-recovery identity, not a development credential.

### Last resort: access keys

Only use an IAM-user access key when the environment cannot use federation or an
IAM role. Create the key for a dedicated least-privilege user, then configure a
named profile without putting the secret in shell history:

```bash
aws configure --profile cayu-dev
export AWS_PROFILE=cayu-dev
export AWS_DEFAULT_REGION=us-west-2
aws sts get-caller-identity
```

For temporary credentials supplied directly by a trusted CI system, all three
credential variables are required:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
```

Long-lived IAM-user credentials omit `AWS_SESSION_TOKEN`, but they should be the
exception. Rotate and revoke them under the organization's credential policy.

## Coding-agent rules

A coding agent that can execute shell commands can use every permission granted
to its AWS principal. Treat the principal as the security boundary:

1. Use a dedicated development account and a dedicated least-privilege role or
   profile for the agent. Do not reuse a personal administrator profile.
2. Give the agent only the profile name, Region, model ID, and MicroVM image ARN.
   Do not paste access-key values or cached session files into the prompt.
3. Start the agent from a shell that selects the profile rather than exporting
   raw keys:

   ```bash
   AWS_PROFILE=cayu-agent-dev AWS_DEFAULT_REGION=us-west-2 your-coding-agent
   ```

4. Before any live action, have the agent run `aws sts get-caller-identity` and
   verify the expected account and non-root role ARN. This proves identity, not
   authorization; the service call still needs the policies below.
5. Require human approval for image creation, model invocation with material
   spend, and MicroVM lifecycle operations. Configure account budgets, quotas,
   CloudTrail, and resource tags independently of Cayu.
6. Never print `~/.aws`, `env`, credential-process output, session caches, or
   authentication tokens into logs. `aws configure list` is appropriate for
   troubleshooting because it masks secrets and reports the credential source.
7. Never copy the host control-plane credentials into the guest.
   `LambdaMicroVMRunner` exchanges host credentials for a short-lived,
   port-scoped JWE endpoint token. By default, the guest has no AWS role. When
   `execution_role_arn` is supplied, Lambda separately assumes that narrowly
   scoped role for the guest workload; it does not expose the host role.

For CI, prefer OIDC federation into an IAM role. Do not store AWS access keys as
repository secrets when the CI platform can exchange its identity token for
temporary AWS credentials.

## Bedrock runtime permissions

`BedrockProvider` calls `ConverseStream` and `CountTokens`. The IAM principal
needs:

- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`
- `bedrock:CountTokens`

This policy is suitable as a functional development starting point, not as a
production least-privilege policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CayuBedrockRuntime",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:CountTokens"
      ],
      "Resource": "*"
    }
  ]
}
```

For production, replace `*` with the approved foundation-model and inference-
profile ARNs. Cross-Region inference profiles also require invocation access to
the underlying foundation model in every destination Region and compatible
organization SCPs. A profile that works in one source Region may therefore fail
when an SCP blocks one of its destination Regions.

`CountTokens` is model-specific. Some Claude models exposed only through cross-
Region inference require the separate Bedrock Mantle endpoint; Cayu does not
currently use that endpoint. This is an API-availability limitation, not a
credential failure.

## Lambda MicroVM permissions

The `LambdaMicroVMRunner` control-plane principal calls:

- `lambda:RunMicrovm`
- `lambda:GetMicrovm`
- `lambda:CreateMicrovmAuthToken`
- `lambda:SuspendMicrovm`
- `lambda:ResumeMicrovm`
- `lambda:TerminateMicrovm`

A functional development policy is:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CayuLambdaMicrovmRuntime",
      "Effect": "Allow",
      "Action": [
        "lambda:RunMicrovm",
        "lambda:GetMicrovm",
        "lambda:CreateMicrovmAuthToken",
        "lambda:SuspendMicrovm",
        "lambda:ResumeMicrovm",
        "lambda:TerminateMicrovm"
      ],
      "Resource": "*"
    }
  ]
}
```

Restrict this policy to the approved image and MicroVM resources using the
resource formats and tag conditions supported by the target AWS account. If
`RunMicrovm` passes an execution role or customer-managed network connector, the
principal also needs narrowly scoped `iam:PassRole` or
`lambda:PassNetworkConnector`; do not grant broad `iam:*` permissions.

The optional MicroVM execution role is the guest workload's AWS authority. Its
trust policy allows `lambda.amazonaws.com` to call `sts:AssumeRole` and
`sts:TagSession`; its permissions policy contains only the AWS actions the guest
application needs. Omit `execution_role_arn` when the guest does not need AWS
service access or CloudWatch runtime logs. This role is separate from both the
Cayu control-plane role and the image build role.

Image building is a separate authority from running Cayu sessions. The image
builder needs access to upload the source archive, call
`lambda:CreateMicrovmImage`/`lambda:GetMicrovmImage`, and pass the designated
build role. The build role trusts `lambda.amazonaws.com` for `sts:AssumeRole`
and `sts:TagSession`, reads the source object with `s3:GetObject`, and may write
CloudWatch build logs. Keep image-building permissions off the ordinary Cayu
runtime role.

## Verify before a live test

Check the selected identity and credential source without printing secrets:

```bash
aws sts get-caller-identity
aws configure list
```

Then run the credential-gated contracts deliberately:

```bash
CAYU_BEDROCK_LIVE=1 \
CAYU_BEDROCK_MODEL=us.anthropic.claude-sonnet-4-6 \
uv run --extra aws python scripts/nightly_verification.py --check bedrock-provider-live

CAYU_LAMBDA_MICROVM_LIVE=1 \
CAYU_LAMBDA_MICROVM_IMAGE=arn:aws:lambda:us-west-2:123456789012:microvm-image:cayu-runner \
uv run --extra aws python scripts/nightly_verification.py --check lambda-microvm-live
```

The live flags must equal `1`; any other value is treated as no opt-in. Missing
Region, model, or image prerequisites are reported as skipped by the nightly
verifier. After explicit opt-in, credential discovery is delegated to Boto3, so
missing, expired, or unauthorized credentials fail the live check.

## Troubleshooting

- `NoRegionError`: set `AWS_DEFAULT_REGION`, configure the profile's Region, or
  pass `region_name=`. Cayu's live-example wrappers also accept `AWS_REGION`,
  but ordinary Boto3 session construction should use `AWS_DEFAULT_REGION`.
- `NoCredentialsError`: sign in again, select `AWS_PROFILE`, or attach the
  expected workload role.
- `ExpiredToken`: refresh with `aws sso login --profile ...` or
  `aws login --profile ...`.
- The wrong profile is used: raw environment credentials take precedence when a
  profile is selected through `AWS_PROFILE`. Unset stale `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`, then run
  `aws configure list --profile ...`. Passing Cayu's explicit `profile_name=`
  creates a Boto3 session pinned to that named profile instead.
- `AccessDeniedException`: compare `aws sts get-caller-identity` with the
  intended account and role, then inspect that identity's IAM policy, permission
  boundary, session policy, and organization SCP. Successful STS identity
  lookup does not prove Bedrock or Lambda authorization.

## AWS references

- [Boto3 credential providers](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html)
- [AWS CLI local console login](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html)
- [AWS CLI IAM Identity Center setup](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
- [IAM security best practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Bedrock inference prerequisites](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-prereq.html)
- [Bedrock inference-profile prerequisites](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-prereq.html)
- [Lambda MicroVM security and permissions](https://docs.aws.amazon.com/lambda/latest/dg/microvms-security.html)
- [Create a Lambda MicroVM image](https://docs.aws.amazon.com/lambda/latest/dg/microvms-getting-started.html)
