# Running untrusted code

Use this guide when an application runs code it did not write: a repository's
tests, a model-authored script, a dependency's install hooks. Do not build your
own `docker run` wrapper or allowlisting proxy; Cayu has both parts.

## Choose the path

- **Local development:** a hardened Docker container with virtual egress. The
  container runs as a non-root user with a read-only root, no capabilities,
  no-new-privileges and pid, memory and CPU limits, and it can reach only the
  hosts you allowlist. It still shares the host kernel (the Docker Desktop VM on
  a laptop), so a kernel or Docker escape reaches that host. Use it for code
  whose risk you accept on that machine.
- **Production, or code you treat as hostile:** `SandboxCodingEnvironmentFactory`
  with `LambdaMicroVMEgressAdapter` (its default `metadata_isolation="required"`).
  Each session gets its own Lambda MicroVM, and it is the composition that
  satisfies `ExecutionRequirements.untrusted()`. See
  [Coding in a cloud sandbox](https://github.com/cayu-dev/cayu/blob/main/docs/environment-factories.md#coding-in-a-cloud-sandbox).

Docker never satisfies `ExecutionRequirements.untrusted()`, hardened or not. Do
not drop that requirement to make admission pass for hostile workloads.

## Hardened Docker with allowlisted egress

Install `cayu[egress]`. This runs a repository's tests in a hardened container
that can reach only the Python package index:

```python
import asyncio

from cayu.egress import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.environments import EnvironmentFactoryRequest
from cayu.runners import DockerWorkloadRestrictions
from cayu.runners.base import ExecCommand

HOSTS = ("pypi.org", "files.pythonhosted.org")

factory = VirtualEgressEnvironmentFactory(
    policies={"pypi": BrowserEgressPolicy(name="pypi", allowed_hosts=list(HOSTS))},
    approved_destinations=[
        ApprovedEgressDestination(destination=host, policy_name="pypi") for host in HOSTS
    ],
    adapter=DockerEgressAdapter(workload_restrictions=DockerWorkloadRestrictions()),
    image="python:3.12-slim",
    browser_max_response_bytes=256 * 1024 * 1024,
)


async def run_tests(repo_tar_base64: str) -> str:
    request = EnvironmentFactoryRequest(
        session_id="tests-1", agent_name="tester", environment_name="sandbox"
    )
    environment = (await factory.create(request)).environment
    runner, binding = environment.runner, environment.binding
    bound = await binding.bind(
        None,
        runner,
        session_id=request.session_id,
        agent_name=request.agent_name,
        environment_name=request.environment_name,
    )
    outcome = "failed"
    try:
        await runner.exec(
            ExecCommand.process(
                "sh", "-c", "mkdir -p /workspace/repo && base64 -d | tar -xf - -C /workspace/repo"
            ),
            stdin=repo_tar_base64,
            timeout_s=120,
        )
        result = await runner.exec(
            ExecCommand.bash(
                "cd /workspace/repo && python -m venv /workspace/venv && "
                "/workspace/venv/bin/pip install -q --no-cache-dir --cert /etc/cayu/ca.pem "
                "-r requirements.txt pytest && "
                "/workspace/venv/bin/python -m pytest -q -p no:cacheprovider"
            ),
            timeout_s=900,
        )
        outcome = "completed"
        return result.stdout
    finally:
        await binding.finalize(bound, outcome=outcome)
```

What the container can and cannot do:

- It runs as UID/GID 1000. `/workspace` (256 MiB) and `/tmp` (64 MiB, with
  `HOME` at `/tmp/cayu-home`) are the only writable places, so install packages
  into a virtual environment under `/workspace`. Change the sizes, limits and
  mounts with `DockerWorkloadRestrictions` fields.
- Its only network is an internal Docker network whose one exit is Cayu's
  broker. Requests to allowlisted hosts go through the `HTTPS_PROXY` Cayu sets;
  everything else, including direct IP connections and cloud metadata, fails.
  Trust the per-session CA at `/etc/cayu/ca.pem` (`SSL_CERT_FILE`,
  `REQUESTS_CA_BUNDLE` and `NODE_EXTRA_CA_CERTS` already point to it).
- Host files are not mounted unless you pass `host_workspace_path`.
- `setup_commands` run as the same restricted user, not root. Bake system
  packages into the image.

Before the runner is returned, Cayu inspects the live container and fails
closed, removing it, if any declared control is missing. Admission evidence for
the environment then reports `guest_privilege_containment` and
`unprivileged_guest` as available, `deny_by_default_network` and
`brokered_egress` from the egress adapter, and `untrusted_code_isolation` as
unsupported because of the shared kernel.

## Use the parts on their own

The hardening and the egress are separate options:

- Egress without hardening: `VirtualEgressEnvironmentFactory(runner_kind="docker", ...)`
  or `DockerEgressAdapter()` with no `workload_restrictions`.
- Hardening without egress: `DockerRunner.create(name, image=...,
  workload_restrictions=DockerWorkloadRestrictions(), network="none")`.
- Hardening with your own network: pass any `network` to `DockerRunner.create`.
  The privilege controls are verified the same way; the network is yours to
  police.
- Strict, evidence-bearing Docker (image identity, executable probes and
  reconnect evidence, as used by `DockerCodingEnvironmentFactory`) still
  requires `network="none"`.

Docker egress reconnect (`reconnect_state_dir`) does not yet support
`workload_restrictions`; the adapter refuses that combination.
