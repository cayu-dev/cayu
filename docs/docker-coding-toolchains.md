# Admitted Docker coding toolchains

Use `DockerCodingToolchainProfile` when a trusted repository needs an explicit
prebuilt compiler, runtime, or dependency set. The profile is application-owned;
the model may select only selectors the profile exposes. It never supplies an
image, executable, Dockerfile, mount, environment secret, package-install command,
or probe.

```python
from hashlib import sha256

from cayu import (
    DockerCodingCommandAuthority,
    DockerCodingDependencyInput,
    DockerCodingEnvironmentFactory,
    DockerCodingToolchainProfile,
    DockerImageIdentity,
    LocalWorkspace,
    RunCommandTool,
    StructuredCommandToolPolicy,
)

lock = workspace_root.joinpath("Cargo.lock").read_bytes()
profile = DockerCodingToolchainProfile(
    profile_id="rust-stable",
    revision="2026-08-29",
    image_identity=DockerImageIdentity(
        reference="registry.example/rust@sha256:" + "a" * 64,
    ),
    platform_architecture="amd64",
    command_authorities=(
        DockerCodingCommandAuthority(
            selector="focused-test",
            revision="1",
            description="Run one admitted Rust integration test target.",
            exposure="structured_command",
            executable="/usr/local/cargo/bin/cargo",
            fixed_arguments=("test", "--test"),
            allow_positional_arguments=True,
            allowed_literals=("api", "storage"),
            min_arguments=1,
            max_arguments=1,
            timeout_seconds=120,
            max_output_bytes=100_000,
            allowed_exit_codes=(0, 101),
        ),
    ),
    dependency_inputs=(
        DockerCodingDependencyInput(
            path="Cargo.lock",
            content_sha256="sha256:" + sha256(lock).hexdigest(),
        ),
    ),
)

factory = DockerCodingEnvironmentFactory(
    source_workspace=LocalWorkspace(workspace_root),
    toolchain_profile=profile,
)
run_command = RunCommandTool(toolchain_profile=profile)
tool_policy = StructuredCommandToolPolicy(toolchain_profile=profile)
```

Register `run_command` on the agent and use `tool_policy` as that agent's ordinary
runtime policy, or pass an existing policy as `base_policy`. Authorities marked
`approval="required"` create durable runtime approval checkpoints. Full arguments
remain quarantined; policy evidence publishes digests and bounded selector/profile
metadata.

The factory verifies source dependency hashes before allocating Docker. It then
admits the exact final image, no-network restrictions, executable evidence,
platform, and bounded profile probes. Checks and commands repeat image,
executable, evidence-expiry, and dependency admission before dispatch. If live
evidence expires or a task changes a declared manifest or lockfile, affected checks
and commands return a stale-toolchain/unavailable result before execution.
Prepare and admit a new immutable image/profile revision; Cayu never runs an
installer or falls back to the host.

The generated built-in Python path is selected explicitly with:

```console
cayu new NAME --preset coding --execution docker --coding-toolchain python
```

Its trusted image-build script is a separate operator lifecycle. The runtime
container stays network-disabled and credential-free. Custom applications can
replace the built-in profile in `environments/coding.py`; profile selection must
remain application configuration, not repository auto-detection or prompt
instructions.

Structured commands also capture bounded content-and-Git-mode manifests immediately
before dispatch and after complete process settlement. Read-only selectors must
leave the manifest unchanged; mutating selectors may change only their declared
path prefixes. The receipt publishes counts and digest identities rather than
repository paths, including for working directories and stale dependency inputs.
An out-of-scope mutation is a failed result, while an incomplete post-command
observation or deferred cleanup is explicitly partial/ambiguous.

The opt-in live contract test exercises both the built-in-language shape and a
non-Python Node profile against exact final containers:

```console
CAYU_RUN_DOCKER_TOOLCHAIN_LIVE=1 \
  uv run pytest -q tests/environments/test_docker_toolchain_live.py
```

That test performs an explicit trusted fixture-image build first, resolves each
result to its content digest, and then starts separate network-disabled runtime
containers. It proves hostile image entrypoints cannot replace Cayu's command
path, direct structured commands and independent named checks both settle, a
detached-session descendant cannot outlive a successful receipt, timeouts quiesce
the runner, ordinary workspace mutations copy back, and exact profile/image/dependency
identities remain in receipts. The test removes its ephemeral containers and image
tags on settlement.
