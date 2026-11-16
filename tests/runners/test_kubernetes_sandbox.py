"""KubernetesSandboxRunner against the sidecar in Kubernetes mode.

End-to-end tests run the real sidecar app in-process through ``httpx.ASGITransport``,
and over TLS through Uvicorn on the loopback. The sidecar runs commands through
a real executor served from the test process, as the test user; everything else
(token check, owner fence, process-group teardown) is the sidecar's own.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import ssl
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
import uvicorn
from tests.runners.lambda_microvm_harness import (
    DEFAULT_SIDECAR_BOOT_ID,
    KUBERNETES_SANDBOX_TOKEN,
    InProcessExecutor,
    KubernetesSupervisorTransport,
    write_sidecar_tls,
)

from cayu.environments.admission import (
    ExecutionRequirements,
    ExecutionToolRequirement,
)
from cayu.runners import (
    HttpxKubernetesSandboxEndpointTransport,
    KubernetesSandboxRunner,
    SidecarEndpointUnauthorized,
    SidecarOwnershipSuperseded,
    SidecarProtocolError,
    SidecarRunnerError,
)
from cayu.runners.base import ExecCommand
from cayu.tools.base import ToolExecutableRequirement, ToolExecutionRequirement

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX sidecar semantics")

_APP = "examples.aws.lambda_microvm_sidecar.app"
_ENDPOINT = "https://sidecar.sandbox.invalid:8080"
_SANDBOX = ("agents", "sandbox-1")


class _AppClient:
    """The transport's HTTP client, routed to whichever sidecar app is current."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def get(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app))

    async def aclose(self) -> None:
        return None


@pytest.fixture
def sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator:
    """Load the sidecar app in a mode; its token file belongs to the test user, as the sidecar's own."""
    token_file = tmp_path / "token"
    token_file.write_text(f"{KUBERNETES_SANDBOX_TOKEN}\n", encoding="ascii")
    token_file.chmod(0o600)
    monkeypatch.setenv("CAYU_SIDECAR_TOKEN_FILE", str(token_file))
    certificate, key = write_sidecar_tls(tmp_path / "tls")
    monkeypatch.setenv("CAYU_SIDECAR_TLS_CERT_FILE", str(certificate))
    monkeypatch.setenv("CAYU_SIDECAR_TLS_KEY_FILE", str(key))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(workspace))
    executor = InProcessExecutor(workspace)
    monkeypatch.setenv("CAYU_EXECUTOR_SOCKET", executor.address)
    loaded: list[ModuleType] = []

    def load(mode: str = "kubernetes") -> ModuleType:
        monkeypatch.setenv("CAYU_SIDECAR_MODE", mode)
        sys.modules.pop(_APP, None)
        module = importlib.import_module(_APP)
        if mode == "lambda":
            # The test host cannot drop to the agent user.
            monkeypatch.setattr(module.EXECUTION_BOUNDARY, "agent_uid", None)
            monkeypatch.setattr(module.EXECUTION_BOUNDARY, "agent_gid", None)
        module.TEST_EXECUTOR = executor
        loaded.append(module)
        return module

    try:
        yield load
    finally:
        for module in loaded:
            module.SUPERVISOR.cancel_all()
        executor.close()
        sys.modules.pop(_APP, None)


def _boot_id(module: ModuleType) -> str:
    """The boot ID health reports: the supervisor's combined with the executor's."""
    return str(uuid.uuid5(uuid.UUID(module.BOOT_ID), module.TEST_EXECUTOR.executor.boot_id))


def _runner(module: ModuleType, *, token: Any = KUBERNETES_SANDBOX_TOKEN, **options: Any):
    transport = HttpxKubernetesSandboxEndpointTransport()
    client = _AppClient(module.app)
    transport._client = client
    runner = KubernetesSandboxRunner(
        endpoint=_ENDPOINT,
        token=token,
        sandbox=_SANDBOX,
        endpoint_transport=transport,
        default_cwd=module.ROOT,
        poll_interval_s=0,
        **options,
    )
    return runner, client


@pytest.mark.anyio
async def test_runner_executes_through_an_authenticated_kubernetes_sidecar(sidecar) -> None:
    module = sidecar()
    runner, _client = _runner(module, env_overlay={"CAYU_LANE": "agent"})
    try:
        result = await runner.exec(ExecCommand.bash('printf "%s:%s" "$CAYU_LANE" "$(pwd)"'))
        assert result.exit_code == 0
        assert result.stdout == f"agent:{module.ROOT}"
        system = await runner.exec_system(ExecCommand.process("printf", "shared"))
        assert system.stdout == "shared"
        assert runner.boot_id == _boot_id(module)
        assert runner.owner_fence_generation == "agents/sandbox-1:1"
        assert runner.resource_key == ("kubernetes-sandbox", "agents", "sandbox-1")
    finally:
        await runner.close()
    assert runner.lifecycle_state == "closed"


@pytest.mark.anyio
async def test_runner_is_refused_without_the_sidecar_token(sidecar) -> None:
    module = sidecar()
    runner, _client = _runner(module, token="w" * 40)
    with pytest.raises(SidecarEndpointUnauthorized, match="Kubernetes sandbox endpoint token"):
        await runner.exec(ExecCommand.process("true"))
    await runner.close()


@pytest.mark.anyio
async def test_runner_asks_a_token_callable_again_after_a_rejection(sidecar) -> None:
    module = sidecar()
    issued = ["w" * 40, KUBERNETES_SANDBOX_TOKEN]
    calls = 0

    async def token() -> str:
        nonlocal calls
        calls += 1
        return issued[min(calls, len(issued)) - 1]

    runner, _client = _runner(module, token=token)
    try:
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
    finally:
        await runner.close()
    assert calls == 2


@pytest.mark.anyio
async def test_a_token_rotated_while_a_command_runs_does_not_fail_it(sidecar) -> None:
    module = sidecar()
    token_file = Path(os.environ["CAYU_SIDECAR_TOKEN_FILE"])
    calls = 0

    def token() -> str:
        nonlocal calls
        calls += 1
        return token_file.read_text(encoding="ascii").strip()

    runner, _client = _runner(module, token=token)
    try:
        running = asyncio.create_task(
            runner.exec(
                ExecCommand.process(sys.executable, "-c", "import time; time.sleep(1)"),
                command_id="rotation",
            )
        )
        await _until_running(module, running, "rotation")
        modified = token_file.stat().st_mtime_ns + 1_000_000_000
        token_file.write_text(f"{'r' * 40}\n", encoding="ascii")
        os.utime(token_file, ns=(modified, modified))
        # The next command-state poll is refused, asks for the rotated token and goes on.
        assert (await running).exit_code == 0
    finally:
        await runner.close()
    assert calls == 2


async def _detached(module: ModuleType, code: str):
    """A first owner that started ``job`` and let go of it for a successor."""
    first, _client = _runner(module, owner_epoch=1)
    waiting = asyncio.create_task(
        first.exec(ExecCommand.process(sys.executable, "-c", code), command_id="job")
    )
    await _until_running(module, waiting, "job")
    first.detach_commands()
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    return first


@pytest.mark.anyio
async def test_a_new_successor_resumes_a_command_it_carries(sidecar) -> None:
    module = sidecar()
    code = "import time; time.sleep(1); print('carried'); raise SystemExit(3)"
    first = await _detached(module, code)

    # Constructed directly, the successor claims, and so carries, before it resumes.
    successor, _client = _runner(module, owner_epoch=2, carry_commands=("job",))
    try:
        result = await successor.resume_exec("job")
        assert (result.exit_code, result.stdout) == (3, "carried\n")
        assert successor.carried_commands == ("job",)
        with pytest.raises(ValueError, match="carried over"):
            await successor.resume_exec("other")
    finally:
        await first.close()
        await successor.close()


@pytest.mark.anyio
async def test_cancelling_a_resume_during_its_claim_stops_the_carried_command(sidecar) -> None:
    module = sidecar()
    first = await _detached(module, "import time; time.sleep(60)")
    successor, _client = _runner(module, owner_epoch=2, carry_commands=("job",))
    transport = successor._endpoint_transport
    claim = transport.claim_owner
    claimed, answer = asyncio.Event(), asyncio.Event()

    async def claim_answered_late(**kwargs: Any) -> Any:
        # The sidecar has carried the command; its answer has not arrived yet.
        response = await claim(**kwargs)
        claimed.set()
        await answer.wait()
        return response

    transport.claim_owner = claim_answered_late
    try:
        resumed = asyncio.create_task(successor.resume_exec("job"))
        await claimed.wait()
        resumed.cancel()
        await asyncio.sleep(0.05)
        answer.set()
        with pytest.raises(asyncio.CancelledError):
            await resumed
        assert module.SUPERVISOR.get("job")["state"] == "cancelled"
    finally:
        answer.set()
        await first.close()
        await successor.close()


async def _until_running(module: ModuleType, task: asyncio.Task, command_id: str) -> None:
    """Wait until the sidecar runs ``command_id``, which ``task`` started."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while True:
        assert not task.done(), task
        if module.SUPERVISOR.get(command_id)["state"] == "running":
            return
        assert loop.time() < deadline
        await asyncio.sleep(0.01)


@pytest.mark.anyio
async def test_runner_refuses_a_lambda_mode_sidecar(sidecar) -> None:
    module = sidecar("lambda")
    runner, _client = _runner(module)
    marker = Path(module.ROOT) / "ran"
    with pytest.raises(SidecarProtocolError, match="bearer_auth and boot_id"):
        await runner.exec(ExecCommand.process("touch", str(marker)))
    assert not marker.exists()
    assert runner.owner_fence_generation is None
    await runner.close()


@pytest.mark.anyio
async def test_a_restarted_sidecar_reports_a_new_boot_id_and_fences_the_runner(sidecar) -> None:
    first = sidecar()
    runner, client = _runner(first)
    try:
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        assert await runner.observe_boot_id() == _boot_id(first)

        restarted = sidecar()
        client.app = restarted.app
        assert await runner.observe_boot_id() == _boot_id(restarted) != _boot_id(first)
        assert runner.boot_id == _boot_id(restarted)
        # The restart lost the owner claim: the next command is fenced, not run.
        with pytest.raises(SidecarOwnershipSuperseded):
            await runner.exec(ExecCommand.process("true"))
        assert runner.owner_superseded
    finally:
        await runner.close()


@pytest.mark.anyio
async def test_a_restarted_executor_reports_a_new_boot_id_and_fences_the_runner(sidecar) -> None:
    module = sidecar()
    runner, _client = _runner(module)
    try:
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        before = await runner.observe_boot_id()

        # The agent container restarted (or its agent namespace was renewed).
        module.TEST_EXECUTOR.executor.boot_id = str(uuid.uuid4())
        assert await runner.observe_boot_id() == _boot_id(module) != before
        # Its processes are gone, and so is the owner claim, as after a sidecar restart.
        deadline = asyncio.get_running_loop().time() + 5
        while module.OWNER.current_generation(runner._owner_claim) is not None:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)
        with pytest.raises(SidecarOwnershipSuperseded):
            await runner.exec(ExecCommand.process("true"))
    finally:
        await runner.close()


@pytest.mark.anyio
async def test_interactive_process_is_torn_down_by_its_process_group(sidecar) -> None:
    module = sidecar()
    runner, _client = _runner(module)
    capability = runner.interactive_process_capability()
    assert capability is not None
    try:
        process = await capability.open_interactive_process(ExecCommand.process("cat"))
        await process.write(b"ping\n")
        assert await process.read() == b"ping\n"
        await process.close()
        await process.close()
        # The runner stays usable after the process group is torn down.
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
    finally:
        await runner.close()


@pytest.mark.anyio
async def test_admission_is_bound_to_the_boot_id_and_sandbox_identity(sidecar) -> None:
    module = sidecar()
    pod = {"pod_uid": "uid-1", "image_digest": "sha256:" + "a" * 64}

    async def identity() -> dict[str, str]:
        return dict(pod)

    runner, client = _runner(module, identity_observer=identity)
    requirements = ExecutionRequirements(
        tool_requirements=(
            ExecutionToolRequirement(
                tool_name="shell",
                requirement=ToolExecutionRequirement(
                    name="program", alternatives=(ToolExecutableRequirement(executable="sh"),)
                ),
            ),
        )
    )
    try:
        observer = runner.execution_admission_observer(requirements)
        first = await observer.collect()
        assert first.candidate == "kubernetes-sandbox"
        (claim,) = first.evidence.tool_requirements.executables
        assert claim.state == "live_verified"
        assert observer.snapshot() == first

        pod["pod_uid"] = "uid-2"
        renewed = await observer.collect()
        assert renewed.evidence.environment_fingerprint != first.evidence.environment_fingerprint

        client.app = sidecar().app
        with pytest.raises(SidecarOwnershipSuperseded):
            await observer.collect()
    finally:
        await runner.close()


@pytest.mark.anyio
async def test_unit_runner_sends_the_token_on_every_call(tmp_path: Path) -> None:
    transport = KubernetesSupervisorTransport(tmp_path)
    runner = KubernetesSandboxRunner(
        endpoint=_ENDPOINT,
        token=KUBERNETES_SANDBOX_TOKEN,
        sandbox=_SANDBOX,
        endpoint_transport=transport,
        default_cwd=str(tmp_path),
        poll_interval_s=0,
    )
    try:
        assert runner.boot_id is None
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        assert runner.boot_id == DEFAULT_SIDECAR_BOOT_ID
    finally:
        await runner.close()
    assert transport.tokens_seen
    assert set(transport.tokens_seen) == {KUBERNETES_SANDBOX_TOKEN}


def test_execution_profile_identity_follows_the_command_environment() -> None:
    def identity(**options: Any):
        return KubernetesSandboxRunner(
            endpoint=_ENDPOINT, token=KUBERNETES_SANDBOX_TOKEN, sandbox=_SANDBOX, **options
        ).execution_profile_identity

    base = identity(env_overlay={"HTTPS_PROXY": "http://10.0.0.1:3128"})
    assert base.name == "cayu.kubernetes_sandbox_runner"
    assert base.behavior_version == "1"
    # Values may be secrets and never shape the identity; the keys and the root do.
    assert identity(env_overlay={"HTTPS_PROXY": "http://10.0.0.2:3128"}) == base
    assert identity(env_overlay={"HTTPS_PROXY": "x", "NO_PROXY": "y"}) != base
    assert identity(env_overlay={"HTTPS_PROXY": "x"}, default_cwd="/srv") != base
    # The sandbox it runs in is not part of the profile.
    assert (
        KubernetesSandboxRunner(
            endpoint="https://other.invalid",
            token=KUBERNETES_SANDBOX_TOKEN,
            sandbox=("other", "pod"),
            env_overlay={"HTTPS_PROXY": "z"},
        ).execution_profile_identity
        == base
    )


@pytest.mark.parametrize(
    ("endpoint", "message"),
    [
        ("http://sidecar.invalid:8080", "https"),
        ("sidecar.invalid:8080", "https"),
        ("ftp://sidecar.invalid", "https"),
        ("https://user:secret@sidecar.invalid", "without credentials"),
        ("https://sidecar.invalid/?x=1", "without credentials"),
        ("https://:8080", "name a host"),
    ],
)
def test_endpoint_must_be_an_explicit_https_url(endpoint: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        KubernetesSandboxRunner(endpoint=endpoint, token=KUBERNETES_SANDBOX_TOKEN, sandbox=_SANDBOX)
    assert (
        KubernetesSandboxRunner(
            endpoint="https://sidecar.example.test:8080/",
            token=KUBERNETES_SANDBOX_TOKEN,
            sandbox=_SANDBOX,
        ).endpoint
        == "https://sidecar.example.test:8080"
    )


@pytest.mark.anyio
async def test_runner_pins_the_sandboxes_own_certificate_over_tls(sidecar, tmp_path: Path) -> None:
    """The real sidecar behind Uvicorn's TLS, as entrypoint.sh serves it in a pod."""

    module = sidecar()
    server = uvicorn.Server(
        uvicorn.Config(
            module.app,
            host="127.0.0.1",
            port=0,
            ssl_certfile=os.environ["CAYU_SIDECAR_TLS_CERT_FILE"],
            ssl_keyfile=os.environ["CAYU_SIDECAR_TLS_KEY_FILE"],
            log_level="warning",
            lifespan="off",
            ws="none",
        )
    )
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            assert not serving.done()
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]

        def runner(certificate: Path) -> KubernetesSandboxRunner:
            return KubernetesSandboxRunner(
                # The certificate names the host the runner connects to, as a Service's does.
                endpoint=f"https://localhost:{port}",
                token=KUBERNETES_SANDBOX_TOKEN,
                sandbox=_SANDBOX,
                verify=ssl.create_default_context(cadata=certificate.read_text(encoding="ascii")),
                default_cwd=module.ROOT,
                poll_interval_s=0,
                request_timeout_s=2,
            )

        pinned = runner(Path(os.environ["CAYU_SIDECAR_TLS_CERT_FILE"]))
        try:
            result = await pinned.exec(ExecCommand.process("printf", "over tls"))
            assert (result.exit_code, result.stdout) == (0, "over tls")
            assert pinned.boot_id == _boot_id(module)
        finally:
            await pinned.close()

        # Another sandbox's certificate, valid for the same name, is not this one.
        other, _key = write_sidecar_tls(tmp_path / "other")
        impostor = runner(other)
        try:
            with pytest.raises(SidecarRunnerError) as refused:
                await impostor.observe_boot_id()
        finally:
            await impostor.close()
        causes = []
        error: BaseException | None = refused.value
        while error is not None:
            causes.append(error)
            error = error.__cause__ or error.__context__
        assert any(isinstance(cause, ssl.SSLCertVerificationError) for cause in causes)
    finally:
        server.should_exit = True
        await serving


def test_constructor_rejects_invalid_configuration(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="namespace, name"):
        KubernetesSandboxRunner(endpoint=_ENDPOINT, token="t", sandbox=["agents", "pod"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        KubernetesSandboxRunner(endpoint=_ENDPOINT, token=" ", sandbox=_SANDBOX)
    with pytest.raises(TypeError, match="token"):
        KubernetesSandboxRunner(endpoint=_ENDPOINT, token=b"t", sandbox=_SANDBOX)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="verify"):
        KubernetesSandboxRunner(
            endpoint=_ENDPOINT,
            token="t",
            sandbox=_SANDBOX,
            endpoint_transport=KubernetesSupervisorTransport(tmp_path),
            verify=ssl.create_default_context(),
        )
    with pytest.raises(ValueError, match="absolute"):
        KubernetesSandboxRunner(
            endpoint=_ENDPOINT, token="t", sandbox=_SANDBOX, default_cwd="workspace"
        )
    with pytest.raises(ValueError, match="carry_commands requires an owner_epoch"):
        KubernetesSandboxRunner(
            endpoint=_ENDPOINT, token="t", sandbox=_SANDBOX, carry_commands=("cmd-1",)
        )
