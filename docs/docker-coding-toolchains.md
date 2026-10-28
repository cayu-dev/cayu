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

from cayu.environments import DockerCodingCommandAuthority, DockerCodingDependencyInput, DockerCodingEnvironmentFactory, DockerCodingToolchainProfile
from cayu.runners import DockerImageIdentity
from cayu.workspaces import LocalWorkspace
from cayu.tools import RunCommandTool, StructuredCommandToolPolicy

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

Pass `artifact_store=store` to `DockerCodingEnvironmentFactory` when tools need
durable artifacts, including complete named-check output. The same host-side store
is exposed on created and reconnected environments; it is not mounted into the
guest, closed, or deleted by factory cleanup. Its stable ID participates in the
factory configuration identity. Register that same store with
`app.register_environment_factory(..., artifact_store=store)` for inventory and
prompt attachments; registration alone does not attach it to factory results.
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

A dependency input's `content_sha256` covers the file's exact bytes by default.
Build it with `CodingDependencyInput.from_content(path, content)` rather than
hashing by hand. An input whose toolchain reads a `pyproject.toml` only to
install the environment can pass `digest_scope="pyproject_environment"`: its
digest then covers the `[project]` name, `requires-python`, dependencies,
optional dependencies and `dynamic`, and the `[build-system]`,
`[dependency-groups]` and `[tool.uv]` tables, so editing a tool setting such as
`[tool.pytest.ini_options]` doesn't make the toolchain stale. The stale-input
error names up to five changed inputs (longer or unprintable paths are only
counted) and, when the profile sets `rebuild_command`, ends with that command.
`rebuild_command` is operator guidance and isn't part of the profile
fingerprint. The structured fields of tool results and failure events still
carry only the error code, path count and path-set fingerprint; only the message
names paths, and only these declared input paths.

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
A `base_policy` denial from a built-in argument rule, such as a
`RequiredAllowlistRule` on `selector`, reaches the model as runtime-owned text
naming the declared argument and the violation category (for example "argument
'selector' is not one of the values this policy allows"), never the rejected
value or the allowlist.


Structured commands also capture bounded content-and-Git-mode manifests immediately
before dispatch and after complete process settlement. Read-only selectors must
leave the manifest unchanged; mutating selectors may change only their declared
path prefixes. The receipt publishes counts and digest identities rather than
repository paths, including for working directories and stale dependency inputs.
An out-of-scope mutation is a failed result, while an incomplete post-command
observation or deferred cleanup is explicitly partial/ambiguous. The model-facing
text says why in its first line, so a read-only test selector whose test writes a
file reads "failed. The process exited with code 0, but this selector is read-only
and the command changed 1 workspace path: ..." and names up to five such paths.

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
detached-session descendant cannot outlive a successful receipt, a timeout of a
command run without a receipt stops the command's whole process tree while the
container stays usable, ordinary workspace mutations copy back, and exact
profile/image/dependency identities remain in receipts. The test removes its ephemeral containers and image
tags on settlement.

A command timeout or cancellation removes the container by default
(`command_cleanup="sandbox"`): the result carries a completed `kill_sandbox`
artifact, says the sandbox was removed, and the environment cannot run more
commands.

`command_cleanup="command_or_sandbox"` does not change that for `run_command` and
`run_check` calls made through an app. Inside a `CayuApp` tool call, every
`run_command` and `run_check` gets a durable command receipt so its result can be
recovered after a restart, and a timeout or cancellation of a receipt-backed
command always removes the container.

The option only affects commands that run without a receipt: these tools called
with a bare `ToolContext` outside an app, `ExecCommandTool`, and direct
`DockerRunner.exec` calls. For those, `command_or_sandbox` stops only the command:
the runner kills the command's process tree inside the container and keeps the
container, its workspace, and the runner once Cayu verifies, from `/proc` and
as a user other than the command's, that every descendant exited. The tool
result then reports a completed `kill_command` cleanup and says the command
was stopped, and the next command runs in the same workspace. The container is
still removed whenever that stop cannot be verified: a cleanup helper fails or
times out, the Docker CLI or a stream fails, a descendant survives, an
orphaned background process started while the command ran is still running,
the image's `python3` for the check is not a regular file in `/usr/bin` or
`/usr/local/bin`, or a writable mount overlaps the image's system directories.
The result then also carries a completed `kill_sandbox` artifact and says the
sandbox was removed. The option is part of the factory's configuration
fingerprint, so a preserved session resumes only under the rules it was
created with. The `confirmed_cancellation` evidence reports
`command_tree_cleanup` for commands without a receipt and
`receipt_command_tree_cleanup` (always false) for receipt-backed commands,
alongside `container_cleanup`; see the Docker cleanup notes in
[runtime contracts](runtime-contracts.md) for the mechanism.

If timeout or cancellation removes the owned container before workspace
publication, finalization fails explicitly and the session cannot claim committed
output. After that failure is durable, a disposal-only retry can release immutable
input references and the run fence without reading the destroyed workspace or
inventing a publication snapshot. Call `drain_environment_cleanups()` before closing
the session store to settle retained cleanup. A stopped, detached, or merely fenced
runner is not evidence of container removal; its recoverable target remains owned.

After worker loss during tool-policy planning, registered recovery can retain a
non-executable approval gate. Explicit operator interruption closes that gate;
it does not itself prove the retained Docker workspace was released. A subsequent
registered recovery transfers the saved binding to completion-finalization recovery
before reconnecting. It preserves the interrupted outcome and original source
baseline, finalizes and releases the exact allocation, and does not replay model
or tool effects. Missing or conflicting binding evidence remains fenced. The
application's task/source reservation still requires its own supported settlement
using the resulting cleanup evidence; session interruption alone is insufficient.


## One profile on Docker and Cayu Cloud

Only the image, runtime user and container restrictions of a
`DockerCodingToolchainProfile` are about Docker. The rest is
`CodingToolchainProfile` (in `cayu.environments`), which
`DockerCodingToolchainProfile` extends: executables, command authorities,
named checks, fixed environment, dependency inputs, admission probes and the
prohibited capabilities. `CodingCommandAuthority`, `CodingDependencyInput`,
`CodingAdmissionProbe` and `CodingFixedEnvironmentVariable` are the same
classes as their `DockerCoding*` names, with the same schemas, so selector
fingerprints in existing receipts don't change.

`RunCommandTool`, `StructuredCommandToolPolicy` and `RunCheckTool` take either
profile type. They accept exactly those two classes, never another subclass,
and validate their own copy, so a subclass can't change what the model may run.
With a Docker-typed profile they behave as before. With a neutral profile they
admit any runner whose evidence names that profile and meets its backend's
evidence policy:

- `docker`: everything `live_verified`, as for a Docker-typed profile;
- `cayu-cloud`: executables probed live, capabilities `available`, and the
  network denied (`egress="none"`).

They only ever evaluate `ExecutionRequirements.trusted(...)`, so this doesn't
change what an agent's own admission requires: Docker still doesn't satisfy
`ExecutionRequirements.untrusted()`.

`DockerCodingEnvironmentFactory` takes a neutral profile together with
`image_identity` (and `platform_architecture` unless the profile names one):

```python
factory = DockerCodingEnvironmentFactory(
    source_workspace=LocalWorkspace(workspace_root),
    toolchain_profile=profile,  # a CodingToolchainProfile
    image_identity=DockerImageIdentity(reference=tag, content_digest=image_id),
    platform_architecture="arm64",
)
```

It builds the Docker profile with `DockerCodingToolchainProfile.from_toolchain`
and verifies it exactly as a Docker-typed one. `extra_admission_probes` adds
Docker-only checks, such as one that the image reports this architecture,
without tying the neutral profile to a machine. Its runners report the
neutral profile's fingerprint, so tools holding that same profile admit them.
`DockerCodingToolchainProfile.coding_toolchain()` goes the other way.

The same neutral profile runs in a sandbox brokered by Cayu Cloud
([toolchain profiles in a brokered sandbox](environment-factories.md#toolchain-profiles-in-a-brokered-sandbox)),
and `select_coding_backend` picks between the two
([choosing a coding backend](environment-factories.md#choosing-a-coding-backend)).
`cayu new --preset coding` generates that call.

Toolchain failures carry a stable `code`, and, where paths are involved, only
`path_count` and `paths_fingerprint`; validation errors name the selector or
probe that is wrong.

## Target project dependencies in the generated coding image

A `cayu new --preset coding --execution docker` project builds its image from
an unmodified `Dockerfile.coding`; Cayu Cloud accepts only that exact file and
reproduces its steps. A target project inside the repository that has its own
`pyproject.toml` and `uv.lock` is declared in `docker-coding-build.json`
(schema version 2) instead of being copied by an edited Dockerfile:

```json
"extra_uv_projects": [{"path": "targets/my-service", "extras": ["dev"]}]
```

Each entry is an object with exactly the keys `path` and `extras`:

- `path`: a project-relative directory matching `[A-Za-z0-9._/-]{1,256}`, already
  normalized (no empty, `.` or `..` components, no leading or trailing `/`),
  whose first component isn't `.cayu`, `.git` or `.runtime`. Its last component
  is the target's name; it must match `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` and be
  unique across entries.
- `extras`: a list of at most 16 unique names matching
  `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`, passed to `uv sync --extra` in order.

At most 8 entries. A build file without the key has none, so earlier v2 files
stay valid. `build_coding_image.py` and the app's `read_coding_image_receipt`
apply the same rules.

`<path>/pyproject.toml` (at most 1 MiB) and `<path>/uv.lock` (at most 64 MiB)
become build inputs. They must be regular files without symlinks in their
path, and neither may also be listed in `extra_build_inputs`. The builder copies
them into its private build context with the other inputs, records each in
`docker-coding-image.json`'s `dependency_inputs` (checked like the project's own
files: any edit to `uv.lock`, and an edit to what `pyproject.toml` installs,
requires a rebuild; once `Dockerfile.coding` is edited, any edit to either
does), and includes them in
`trusted_build_context_sha256`: the sha256 of the compact, key-sorted JSON list
of `{"path", "content_sha256"}` for every build input sorted by path, where
`content_sha256` is `sha256:` plus the hex digest of the file's bytes.

Every dependency install in `Dockerfile.coding` runs as an unprivileged build
user; root installs only the pinned system packages and `uv` itself (`python -m
pip install uv==<uv_version>`). After `COPY pyproject.toml uv.lock ./` into
`/opt/cayu-project`, one fixed `RUN` step, with the reviewed wheel context
mounted read-only at `/opt/cayu-wheel` and the build context at
`/opt/cayu-build-context`, does this as root:

1. Create group `cayu-build` (gid 10001) and system user `cayu-build` (uid 10001,
   no home directory, home `/nonexistent`, shell `/usr/sbin/nologin`, no
   password).
2. Create `/tmp/cayu-build-home` (mode 0700) and `/opt/targets` (mode 0755) owned
   by 10001:10001, and give 10001:10001 ownership of `/opt/cayu-project` (the
   directory only; the copied `pyproject.toml` and `uv.lock` stay root-owned).
3. Run each of the following as the build user, with `setpriv --reuid=10001
   --regid=10001 --clear-groups --no-new-privs` and the environment
   `HOME=/tmp/cayu-build-home UV_CACHE_DIR=/tmp/cayu-build-home/uv-cache`:
   - in `/opt/cayu-project`: `uv sync --frozen --extra dev --no-install-project`
     (the test tools come from the project's `dev` dependency group, which uv
     installs by default; the `dev` extra is empty in newly generated projects);
   - if the wheel context holds a `*.whl` file: `uv pip install --python
     /opt/cayu-project/.venv/bin/python --no-deps <wheel>`;
   - for each `extra_uv_projects` entry, in order: create `/opt/targets/<name>`
     (an error if it exists), copy `<path>/pyproject.toml` and `<path>/uv.lock`
     from the build context into it, and in that directory, with
     `UV_PROJECT_ENVIRONMENT=/opt/targets/<name>/.venv`, run `uv sync --frozen
     --no-install-project [--extra EXTRA ...] --python /usr/local/bin/python`.
4. As the build user, send `SIGKILL` to every process it can signal
   (`kill(-1, SIGKILL)`), so nothing it started survives.
5. As root: remove `/tmp/cayu-build-home` and anything in `/tmp` or `/var/tmp`
   owned by uid or gid 10001; `chown -h -R root:root /opt/cayu-project
   /opt/targets`; clear setuid and setgid bits on everything under them except
   symlinks; delete the `cayu-build` user and group.
6. Fail the build if uid or gid 10001 still has a passwd or group entry, or if
   any file outside `/proc` and the two mounts is still owned by uid or gid 10001.

The result is root-owned and world-readable, so the runtime user can run but
not change it. The target's own package isn't installed, and path
dependencies or uv workspace members aren't supported. To run the target's
tools, add their absolute paths under `/opt/targets/<name>/.venv/bin` to
`extra_probe_executables`, add `/opt/targets/<name>` to the profile's
`read_only_support_paths`, and give the profile a command authority with that
executable. The generated project's README shows a complete `target-test`
selector.

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
