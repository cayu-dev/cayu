# Execution capabilities: Docker and Lambda MicroVM

Cayu runs agent work in one of two maintained execution backends: Docker locally
and AWS Lambda MicroVM in production. This page lists what each backend supports
for coding and browser work, and what it refuses. Every "no" names the check that
refuses it, so an unsupported capability fails before any model or tool call; it
is never inferred from a backend name. Every "yes" names the deterministic tests
or the opt-in live check (see [nightly verification](nightly-verification.md))
that proves it.

"Docker" below covers both the Docker coding environment
(`DockerCodingEnvironmentFactory`) and Docker virtual egress
(`DockerEgressAdapter`). "Lambda" means `LambdaMicroVMEgressAdapter` behind
`VirtualEgressEnvironmentFactory`, or a `LambdaMicroVMRunner` composed directly,
with the first-party sidecar image.

## Isolation and admission

| Capability | Docker | Lambda MicroVM |
| --- | --- | --- |
| Untrusted-code isolation | No. A container is not a secure sandbox boundary; admission of `ExecutionRequirements.untrusted()` refuses it (`docker_untrusted_isolation_unsupported`). | Yes. Firecracker MicroVM per session. |
| Credential non-possession (virtual egress) | Yes with Docker virtual egress, through the per-session broker. The Docker coding environment has no network at all (`docker_network_disabled`). | Yes, through the private proxy in the control-plane task. |
| Deny-by-default network with metadata denial | Yes (egress adapter preflight). | Yes in `metadata_isolation="required"` mode, verified by preflight before agent code runs (`aws-lambda-microvm-metadata-isolation-live`). |
| Unprivileged agent commands | Container user as configured. | Yes: UID/GID 1000, no capabilities, `no_new_privs`, no AWS credentials, verified by a guest probe on every create and reconnect. |
| Executable admission evidence | Yes: live container probes (`docker-live-toolchain-profiles`). | Yes: live probes bound to the MicroVM, image version, sidecar protocol, and guest boot id (`aws-lambda-microvm-tool-admission-live`). |

## Coding

| Capability | Docker | Lambda MicroVM |
| --- | --- | --- |
| Commands (`ExecCommandTool`), cancellation, timeouts, bounded output | Yes. | Yes (runner conformance suite). |
| File tools (read, write, edit, delete, list, `apply_patch`) | Yes. | Yes, through `RunnerWorkspace` inside the MicroVM. |
| `search_text` | Yes when the image has `rg`. | Yes: the image ships a pinned `ripgrep`. The native `workspace_text_search_v1` capability is not claimed. |
| `git_changes` | Yes when the image has `git`. | Yes: the image ships `git`. |
| Named checks (`RunCheckTool`) without a toolchain profile | Yes. | Yes: edit, failing check, repair, and passing check verified live in one admitted workspace (`aws-lambda-microvm-coding-live`). Checks run in the runner root. |
| Toolchain profiles (`DockerCodingToolchainProfile`, profile-bound `RunCheckTool` and `RunCommandTool`) | Yes. | No. The profile is Docker-typed and admission refuses other backends with `docker_admission_mismatch`. |
| Workspace branches (isolate, publish, recover) | Yes for a `DockerRunner` with an exact container id. | Yes for the MicroVM's lifetime, including suspend and resume; the workspace root must be a subdirectory of `/workspace`. Verified live, including recovery from a fresh process. |

## Browser

| Capability | Docker | Lambda MicroVM |
| --- | --- | --- |
| Browser sessions and web fetch | Yes, with the pinned browser workload image. | Yes, with the browser variant of the sidecar image and `LambdaMicroVMEgressAdapter(browser_workload=True)`. The workload is reported only after a trusted probe matches the installed worker to this Cayu release and one sandboxed Chromium launch succeeds as UID 1000 (`aws-lambda-microvm-browser-live`). |
| Chromium sandbox | Yes: seccomp profile plus the setuid sandbox helper. | Yes: unprivileged user namespaces under `no_new_privs`, renderers seccomp-filtered. There is no `--no-sandbox` fallback. |
| Authenticated browser profiles | Yes. | Yes, within the 24 MiB stdin and 32 MiB output bounds per worker call; larger uploads are refused before dispatch. |
| Operator view and takeover | Yes, through the `cayu-control` network alias. | Yes, through the sidecar's owner-fenced control relay, verified by a TLS handshake as the agent user before any control credential is sent (`aws-lambda-microvm-browser-control-live`). |
| Recording | Yes (`docker_sampled_active_page`). | Yes, over the same relay; the adapter finalizes recordings in the guest before suspend or terminate. |
| Browser separated from agent commands | No when shell tools run in the browser container: both run as `pwuser`. | No: the browser shares UID 1000 with agent commands, so a shell tool in the same session can reach the proxy and the control relay. The control server still requires short-lived credentials. |

Browser state does not survive suspend, reconnect, or replacement on Lambda;
the control relay is configured again at the next admission.

## Recovery

| Capability | Docker | Lambda MicroVM |
| --- | --- | --- |
| Crash-safe creation (lost acknowledgement, concurrent recovery) | Yes: allocation identity labels on the container. | Yes: idempotent `RunMicrovm` client token inside a pinned replay window (`aws-lambda-microvm-recoverable-allocation-live`). |
| Same-allocation reconnect with fresh egress authority | Opt-in (`DockerEgressAdapter(reconnect_state_dir=...)`). | Yes: suspended MicroVMs resume; ended MicroVMs are never silently replaced. |
| One current owner across processes | Local ownership directory. | Yes: sidecar owner claim; stale owners cannot run, suspend, or terminate (verified live across processes). |
| Continue after the allocation is gone | Fresh allocation for a new invocation after completion, on disposal proof. | Yes with `WorkspaceCheckpointPolicy(allocation_replacement="restore")`: disposal proof, replacement pinned to the same image version, last durable checkpoint restored before any model or tool call (`aws-lambda-microvm-replacement-live`). |
| What survives same-allocation reconnect | Files and, with reconnect state, browser pages. | Files and the MicroVM's disk; the sidecar's suspend hook cancels running commands. |
| What survives replacement | Checkpointed files only. | Checkpointed regular files only. Process memory, background processes, open workspace branches, and browser state are lost and never reported as continued. |

Durable checkpoints need a pin-capable artifact store outside the sandbox:
`LocalArtifactStore`, or `S3ArtifactStore` with durable pins.
