# AWS Lambda MicroVM managed ingress and metadata isolation

Research and validation date: 2026-07-14

## Verdict

AWS does not expose a first-class Lambda MicroVM metadata switch, but **Cayu can
preserve managed ingress while denying ordinary agent commands access to the
link-local metadata path**. The working enforcement seam is inside Cayu's
first-party guest image, not the AWS connector API.

The following is established:

- AWS documents no first-class Lambda MicroVM setting for disabling or filtering metadata access. `RunMicrovm` exposes ingress connectors, egress connectors, and an optional execution role, but no metadata option ([RunMicrovm API](https://docs.aws.amazon.com/lambda/latest/microvm-api/API_RunMicrovm.html)).
- Customer-managed network connectors configure VPC egress through subnets, security groups, and an IPv4/DualStack selection; they expose no link-local route or metadata policy ([MicroVM networking](https://docs.aws.amazon.com/lambda/latest/dg/microvms-networking.html), [CreateNetworkConnector API](https://docs.aws.amazon.com/lambda/latest/lambda-core/API_CreateNetworkConnector.html), [connector configuration type](https://docs.aws.amazon.com/lambda/latest/lambda-core/API_NetworkConnectorConfiguration.html)).
- Cayu's earlier real-AWS runs established that a broad guest block of `169.254.169.254` disrupted the managed ingress path, while the working configuration allowed a guest TCP connection to `169.254.169.254:80` ([issue #336](https://github.com/vertexkg/cayu/issues/336), [PR #352 real-AWS evidence](https://github.com/vertexkg/cayu/pull/352)).
- AWS explicitly documents that Lambda MicroVM images granted `additionalOsCapabilities=["ALL"]` can create network namespaces and run eBPF programs ([MicroVM images: operating system capabilities](https://docs.aws.amazon.com/lambda/latest/dg/microvms-images.html)). Cayu now uses that documented network-namespace capability.
- In the final real-AWS run, the trusted root sidecar retained managed ingress while ordinary commands ran as UID/GID 1000 with no capabilities in a network namespace with no default route. The namespace could reach only a fixed-port relay to the private Cayu proxy. Metadata and direct public egress were denied, while the scoped request, revocation, workspace release, and cleanup all passed ([PR #352](https://github.com/vertexkg/cayu/pull/352)).

The accurate product statement is:

> AWS supplies no first-class Lambda MicroVM metadata control. Cayu's
> first-party image enforces the boundary by placing agent commands in a
> capability-free network namespace with no default route and only a narrow
> relay to the private Cayu proxy. Required metadata isolation is verified only
> when the guest probe confirms the link-local path is denied.

Custom or legacy images without that boundary may opt into the explicitly
`unverified` mode, but cannot produce a verified claim.

## What AWS explicitly documents

### Ingress and egress are independent service-level associations

AWS says ingress and egress configurations are independent. Ingress is an AWS-managed connector: clients call a service-managed HTTPS endpoint and Lambda forwards the request to an allowed port in the MicroVM. Egress is separately configured, either with public internet access or a customer-managed VPC connector ([Lambda MicroVM networking](https://docs.aws.amazon.com/lambda/latest/dg/microvms-networking.html)).

The public API reflects that model with separate `ingressNetworkConnectors` and `egressNetworkConnectors` arrays ([RunMicrovm API](https://docs.aws.amazon.com/lambda/latest/microvm-api/API_RunMicrovm.html)). AWS does not document the internal address, route, process, or transport used to forward managed ingress inside the guest.

### The customer-managed connector is a VPC egress control, not a metadata control

AWS documents the writable connector shape as `VpcEgressConfiguration`: VPC subnets, security groups, network protocol, and the associated `MicroVm` resource type. No route table, link-local exception, metadata endpoint, or per-process selector is part of this API ([CreateNetworkConnector API](https://docs.aws.amazon.com/lambda/latest/lambda-core/API_CreateNetworkConnector.html), [NetworkConnectorConfiguration](https://docs.aws.amazon.com/lambda/latest/lambda-core/API_NetworkConnectorConfiguration.html)). Connectors also cannot be changed while a MicroVM is running ([MicroVM networking](https://docs.aws.amazon.com/lambda/latest/dg/microvms-networking.html)).

General VPC controls do not provide a documented metadata boundary: AWS says security groups do not filter EC2 instance metadata or ECS task-metadata traffic, and network ACLs cannot block IMDS ([VPC security groups](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-groups.html), [VPC network ACLs](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-network-acls.html)). Those statements are not Lambda-MicroVM-specific, but they explain why the VPC connector's security group/NACL is not evidence of link-local denial.

### Execution-role omission limits AWS permissions, not network reachability

The MicroVM execution role is optional. AWS says that without one, Lambda does not emit runtime logs to CloudWatch and the MicroVM cannot access other AWS services ([MicroVM security](https://docs.aws.amazon.com/lambda/latest/dg/microvms-security.html)). `RunMicrovm` describes the field only as the IAM role assumed during execution ([RunMicrovm API](https://docs.aws.amazon.com/lambda/latest/microvm-api/API_RunMicrovm.html)).

AWS does not document how Lambda MicroVM execution-role credentials are delivered, nor does it associate that role with EC2 IMDS or a specific container-credential endpoint. Consequently, omitting or narrowing the execution role is useful defense in depth, but it is not proof that the link-local network path is denied.

For comparison only, AWS documents `169.254.169.254` as the EC2 IMDS IPv4 address and the `latest/meta-data/iam/security-credentials/` path as an EC2 role-credential source ([using EC2 IMDS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/configuring-instance-metadata-service.html), [EC2 role credentials](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-metadata-security-credentials.html)). These EC2 documents do not say that Lambda MicroVMs implement EC2 IMDS.

### Guest-OS selective controls are available and validated

AWS's EC2 guidance shows UID-scoped `iptables` rules that deny `169.254.169.254` for one user or allow it only for a trusted user ([limit access to IMDS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-metadata-limiting-access.html)). That guidance is for EC2, not Lambda MicroVMs.

More directly relevant, Lambda documents that `additionalOsCapabilities=["ALL"]` enables operations including network namespaces and eBPF inside the MicroVM isolation boundary ([MicroVM images](https://docs.aws.amazon.com/lambda/latest/dg/microvms-images.html)). AWS does not provide a Lambda-specific recipe for using those mechanisms to separate the managed ingress component from agent subprocesses. Cayu validated such a design with a dedicated child network namespace while leaving the trusted sidecar in the original namespace.

Lambda Managed Instances are a separate product with capacity-provider and VPC networking. Their networking documentation does not define the MicroVM connector or guest link-local boundary and should not be used to fill this documentation gap ([Lambda Managed Instances networking](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-networking.html)).

## What the Cayu experiment established

The pre-#336 AWS run reported that a broad guest block of `169.254.169.254` prevented the managed sidecar from serving commands; disabling that probe restored operation ([issue #336](https://github.com/vertexkg/cayu/issues/336), [PR #329](https://github.com/vertexkg/cayu/pull/329)). This is strong evidence of a dependency or conflict in that image and base-image version, but AWS does not document the internal mechanism. Calling it a definitely shared AWS endpoint is an inference.

The #336 run `metadata-isolation-2e0cb47627d6` established the following at the real boundary ([PR #352](https://github.com/vertexkg/cayu/pull/352)):

- managed sidecar and private scoped proxy request worked;
- direct public-IP egress was denied;
- a guest TCP connection to `169.254.169.254:80` succeeded while EC2-shaped token and role-credential paths were attempted;
- no AWS credentials or vault canary were recovered in the no-execution-role configuration;
- the virtual credential, revocation, workspace release, and MicroVM cleanup worked.

The check treats a successful TCP connection as network reachability even if the HTTP exchange resets or returns no parseable response. It therefore proves a reachable link-local socket, not that the guest received a functioning EC2-compatible IMDS response. It separately found no metadata credentials.

The first selective implementation attempted UID-scoped `iptables` owner
rules. The real Lambda MicroVM kernel rejected that image at boot because the
`owner` extension was unavailable. That result shows that the analogous EC2
recipe is not portable to this Lambda MicroVM kernel.

The replacement design uses a dedicated `cayu-agent` network namespace:

- the trusted sidecar and lifecycle commands remain in the original namespace;
- ordinary agent commands enter the child namespace and drop to UID/GID 1000,
  empty capability sets, and `no_new_privs`;
- the child has a point-to-point veth and no default route;
- the root-side veth gateway accepts only fixed relay port 18080 and rejects all
  other input from the child;
- the root-owned relay accepts exactly one RFC 1918 proxy target taken from the
  enforced Cayu proxy environment; and
- the Cayu broker still enforces the virtual token and exact host, method, and
  path at the application layer.

The final run `metadata-isolation-72c8f35f51a6` established on real AWS that
managed sidecar ingress and proxy reachability remained verified, metadata and
direct public egress were denied, AWS credential material and the vault canary
were absent, and required metadata isolation, scoped request, revocation,
workspace release, and cleanup were all verified.

## What remains an inference or undocumented

- AWS does not state that managed ingress and EC2-style metadata are the same service or endpoint inside a Lambda MicroVM.
- AWS does not document an IMDS enable/disable, token-mode, hop-limit, route, firewall, or execution-role credential-delivery setting for Lambda MicroVMs.
- A guest-local control is not a security boundary if untrusted commands retain root, `CAP_NET_ADMIN`, access to the privileged namespace, or the ability to alter the filter. The command process must be placed in a less-privileged namespace/cgroup by a trusted supervisor.
- The first-party boundary depends on the documented `ALL` OS capabilities during trusted image startup. A custom image that omits the namespace setup must use the unverified adapter mode or fail required isolation.
- AWS could change its preview MicroVM kernel, base image, or managed ingress behavior. The real-AWS nightly contract is therefore the continuing compatibility proof.

## Continuing verification

Keep the real-AWS contract as an explicit opt-in/nightly check. It should remain
the release evidence for managed sidecar ingress, namespace identity and
capabilities, fixed relay reachability, link-local and direct-public denial,
credential and vault non-possession, revocation, workspace release, and
MicroVM cleanup. A successful image build alone is not sufficient evidence.
