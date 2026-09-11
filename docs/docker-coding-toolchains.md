# Admitted Docker coding toolchains

Use `DockerCodingToolchainProfile` when a trusted repository needs an explicit
prebuilt compiler, runtime, or dependency set. The profile is application-owned;
the model may select only selectors the profile exposes. It never supplies an
image, executable, Dockerfile, mount, environment secret, package-install command,
or probe.

Native runner workspace revision observations hash regular files inside one
bounded guest operation instead of transferring each file to the host. The
canonical revision includes the same paths, content hashes, byte counts, and
executable modes as local observation. Protected/excluded paths and symlinks
remain outside this regular-file view. Size limits, mutation checks, and the
runtime's finalization deadline still apply; incomplete evidence never becomes
a successful revision or permits unsafe publication. Custom workspace subclasses
retain their own list/read behavior.

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

The factory verifies source dependency hashes before allocating Docker, builds
the restricted container, and reports exact image, no-network, executable,
platform, and bounded profile evidence. The common lifecycle admits that
evidence only after binding and repeats admission at each actual provider or
tool dispatch. If live evidence expires, it asks the
strict Docker runner to re-inspect the same container ID and repeat live
restriction, immutable-input, and executable probes before it evaluates the
fresh candidate. Named checks and structured commands repeat that validation
defensively; those downstream checks do not authorize an unexposed runner.
Concurrent renewals share one probe pass; fresh evidence needs no additional
Docker calls. Renewal preserves image, toolchain profile, and environment
identity, and never extends an old observation's lifetime. Invocation-scoped
renewal detaches private probe failures and preserves genuine caller
cancellation even when a runner suppresses or fabricates cancellation.
Missing renewal support, failed probes, configuration drift, and evidence that
expires during renewal still fail closed. A task that changes a declared manifest
or lockfile still receives a stale-toolchain result before execution: prepare and
admit a new immutable image/profile revision. Cayu never runs an installer or
falls back to the host.

The generated built-in Python path is selected explicitly with:

```console
cayu new NAME --preset coding --execution docker --coding-toolchain python
```

Its trusted image-build script is a separate operator lifecycle. The runtime
container stays network-disabled and credential-free. Custom applications can
replace the built-in profile in `environments/coding.py`; profile selection must
remain application configuration, not repository auto-detection or prompt
instructions.

The model-facing `run_command` schema bounds working directories and timeouts
to the selected profile. Its description includes each selector's argument, path,
flag, literal, working-directory, and timeout contract. `args` contains only
additional arguments; a no-argument selector uses `args=[]`. Omit optional fields
to use that selector's defaults. The catalogue does not disclose fixed environment
values or executable paths, and policy still validates every call independently.

Built-in validation denials provide a fixed correction hint and a
`command_denial_code`: `unknown_selector`, `argument_shape`, `argument_count`,
`disallowed_path`, `working_directory`, `timeout_ceiling`, or `output_mode`.
These codes survive durable tool-round recovery. Hints refer to the published
selector contract and never interpolate rejected arguments or private profile
values. Arbitrary policy reasons and metadata remain subject to the existing
publication restrictions; a code permits only its runtime-owned constant hint.


Structured commands also capture bounded content-and-Git-mode manifests immediately
before dispatch and after complete process settlement. Read-only selectors must
leave the manifest unchanged; mutating selectors may change only their declared
path prefixes. The receipt publishes counts and digest identities rather than
repository paths, including for working directories and stale dependency inputs.
An out-of-scope mutation is a failed result, while an incomplete post-command
observation or deferred cleanup is explicitly partial/ambiguous.

`RunnerWorkspace` captures each manifest in one bounded guest operation, hashing
complete file contents under descriptor containment and rechecking observed
entries before returning identities. It preserves exclusions, symlinks, executable
bits, and path/per-file/aggregate limits. Other workspace implementations retain
the complete per-file fallback. An incomplete bulk observation fails closed.

Receipt `started_at`, `finished_at`, and `duration_ms` describe the runner execution
boundary; duration is the nonnegative whole-millisecond timestamp difference.
`pre_capture` and `post_capture` each contain their own timestamps and duration.
Capture time is excluded from process duration. Terminal recovery retains recorded
process/pre-capture timing when available and measures its new post-capture scan;
it does not rerun the command.


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

If timeout or cancellation removes the owned container before workspace
publication, finalization fails explicitly and the session cannot claim committed
output. After that failure is durable, a disposal-only retry can release immutable
input references and the run fence without reading the destroyed workspace or
inventing a publication snapshot. Call `drain_environment_cleanups()` before closing
the session store to settle retained cleanup. A stopped, detached, or merely fenced
runner is not evidence of container removal; its recoverable target remains owned.


## Writable home and tool caches

Runtime-created Docker coding containers receive a private, disposable home at
`/tmp/cayu-home`. Runtime supplies `HOME`, `XDG_CACHE_HOME` (`$HOME/.cache`),
`XDG_CONFIG_HOME` (`$HOME/.config`), `XDG_DATA_HOME` (`$HOME/.local/share`),
`XDG_STATE_HOME` (`$HOME/.local/state`), and `UV_CACHE_DIR` (`$HOME/.cache/uv`).
These are literal absolute paths in the container environment; no shell expansion
is required. Runtime creates the directories as the configured non-root uid/gid
with a restrictive umask before admitting the container.

The home shares the existing bounded `/tmp` tmpfs allocation (64 MiB by default),
including its memory, execution, and lifecycle restrictions. It is separate from
`/workspace`, so caches and home state are not copied back or included in workspace
checkpoints. No host home, credentials, or cache is mounted or copied. Concurrent
candidate containers have separate allocations. Reconnecting to the same running
container retains its home; reconstructing a container starts with an empty home.
Tools must tolerate cache loss. A stopped/restarted container whose tmpfs state was
lost is not admitted as a valid reconnect.

`DockerWorkloadRestrictions.home_directory` can select another normalized path
below `/tmp`. Restrictions require a bounded `/tmp` mount without nested mounts.
The restrictions and toolchain fingerprints include this configuration, and
`toolchain_home_environment` records the non-secret effective paths in profile
evidence. Strict creation and reconnect verify the configured environment and the
writable, non-symlink directory paths before returning live evidence. Older
containers without this contract must be reconstructed.

Applications may add tool-specific defaults through command authorities'
`fixed_environment`, for example a `PIP_CACHE_DIR` value of
`/tmp/cayu-home/.cache/pip`, or a `UV_CACHE_DIR` override beneath the home.
Create tool-specific subdirectories when the tool requires them. Use absolute
paths matching the configured home; fixed values are not shell-expanded. The
command authority and its overrides remain part of the toolchain identity.
`HOME` and the base XDG variables are reserved and cannot be replaced through
`fixed_environment`. Increase the explicitly configured `/tmp` size if the
workload needs larger caches; this does not make them durable.
