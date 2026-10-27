"""Brokered sandboxes: the Runtime client of Cloud's sandbox broker, against a fake broker."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.core.test_workspace_mutation_receipts import _SingleToolProvider
from tests.sandbox_broker.fake_broker import (
    API_URL,
    APPLICATION,
    EGRESS_DENY_BY_DEFAULT,
    GUARANTEES_V1,
    TOKEN,
    FakeBroker,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.egress.sandbox_coding import SandboxWorkspaceTransferLimits
from cayu.environments import (
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentSpec,
)
from cayu.environments._ephemeral_git import initialize_ephemeral_git_baseline
from cayu.environments.admission import ExecutionRequirements, evaluate_execution_admission
from cayu.external_agents import runner_agent_launch
from cayu.external_agents.environment import scrub_external_environment
from cayu.runners import ExecCommand
from cayu.runners.base import runner_workspace_mutation_settlement
from cayu.runtime._environment_exposure import _terminal_runner_state
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sandbox_broker import (
    BrokeredSandboxCodingBinding,
    BrokeredSandboxCodingEnvironmentFactory,
    BrokeredSandboxRunner,
    BrokeredSandboxWorkspace,
    SandboxAllocationFailedError,
    SandboxArchiveTooLargeError,
    SandboxBrokerClient,
    SandboxBrokerRefusedError,
    SandboxBrokerSettings,
    SandboxBrokerUnavailableError,
    SandboxCopyFailedError,
    SandboxGuarantees,
    SandboxLeaseKeeper,
    SandboxLeaseLostError,
    SandboxNotFoundError,
    SandboxQuotaExceededError,
    sandbox_capability_claims,
    validate_egress_hosts,
)
from cayu.sandbox_broker import client as client_module
from cayu.sandbox_broker import runner as runner_module
from cayu.sandbox_broker.client import MAX_COMMAND_OUTPUT_BYTES
from cayu.storage import InMemoryKnowledgeStore, KnowledgeAccessScope, SQLiteSessionStore
from cayu.tools import GitChangesTool
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workspaces import LocalWorkspace

_ENV = {
    "CAYU_CLOUD_API_URL": API_URL,
    "CAYU_CLOUD_SANDBOX_TOKEN": TOKEN,
    "CAYU_CLOUD_APPLICATION_ID": APPLICATION,
}


def _request(session_id: str = "coding", **options) -> EnvironmentFactoryRequest:
    return EnvironmentFactoryRequest(
        session_id=session_id, agent_name="coder", environment_name="sandbox", **options
    )


def _project(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
    (source / "run.sh").write_bytes(b"#!/bin/sh\necho ok\n")
    (source / "run.sh").chmod(0o755)
    return source


def _factory(
    broker: FakeBroker, source: Path, **overrides
) -> BrokeredSandboxCodingEnvironmentFactory:
    options = {
        "broker": broker.client(),
        "source_workspace": LocalWorkspace(source, workspace_id="project"),
        "egress": "public_web",
    }
    options.update(overrides)
    return BrokeredSandboxCodingEnvironmentFactory(**options)


async def _ready_runner(
    broker: FakeBroker,
    *,
    egress: str = "none",
    egress_hosts: list[str] | None = None,
    client_settings: dict | None = None,
    **keeper_options,
) -> BrokeredSandboxRunner:
    client = broker.client(**(client_settings or {}))
    created = await client.create(
        idempotency_key=f"k-{len(broker.sandboxes)}",
        image="coding-python",
        egress_mode=egress,
        egress_hosts=egress_hosts,
        lease_seconds=60,
    )
    # The lease starts at ready, never during provisioning.
    assert created.status == "provisioning" and created.lease_expires_at is None
    sandbox = await client.wait_until_ready(created.id, timeout_s=5, poll_interval_s=0)
    assert sandbox.lease_expires_at is not None
    keeper = None
    if keeper_options:
        keeper = SandboxLeaseKeeper(client, sandbox, lease_seconds=60, **keeper_options).start()
    return BrokeredSandboxRunner(client, sandbox, lease_keeper=keeper, poll_wait_seconds=1)


# Settings ------------------------------------------------------------------


def test_settings_come_from_the_variables_cloud_injects() -> None:
    settings = SandboxBrokerSettings.from_env(_ENV)
    assert (settings.api_url, settings.application_id) == (API_URL, APPLICATION)
    assert TOKEN not in repr(settings)
    assert TOKEN not in repr(SandboxBrokerClient(settings))
    for name in _ENV:
        with pytest.raises(ValueError, match=name):
            SandboxBrokerSettings.from_env({**_ENV, name: ""})
    with pytest.raises(ValueError, match="HTTPS"):
        SandboxBrokerSettings.from_env({**_ENV, "CAYU_CLOUD_API_URL": "http://cloud.example"})
    # Plain HTTP only for a local broker.
    assert SandboxBrokerSettings.from_env({**_ENV, "CAYU_CLOUD_API_URL": "http://127.0.0.1:8080"})


def _sleep_recorder(monkeypatch) -> list[float]:
    """Record the client's retry pauses without waiting them out."""

    pauses: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay: float, *args, **kwargs):
        pauses.append(delay)
        await real_sleep(0)

    patched = type(asyncio)("asyncio")
    patched.__dict__.update(vars(asyncio))
    patched.sleep = sleep
    monkeypatch.setattr(client_module, "asyncio", patched)
    return pauses


def test_the_request_timeout_outlasts_clouds_request_chain() -> None:
    # The portal waits 35 s for API Gateway, which ends a request after 30 s.
    assert client_module.DEFAULT_REQUEST_TIMEOUT_SECONDS == 45.0
    assert SandboxBrokerSettings.from_env(_ENV).request_timeout_s == 45.0
    overridden = {**_ENV, "CAYU_CLOUD_SANDBOX_REQUEST_TIMEOUT_SECONDS": " 60 "}
    assert SandboxBrokerSettings.from_env(overridden).request_timeout_s == 60.0
    for value in ("soon", "0", "-1", "inf", "nan"):
        with pytest.raises(ValueError):
            SandboxBrokerSettings.from_env(
                {**_ENV, "CAYU_CLOUD_SANDBOX_REQUEST_TIMEOUT_SECONDS": value}
            )


@pytest.mark.parametrize("override", [None, 60.0])
def test_each_request_carries_the_request_timeout(override: float | None, monkeypatch) -> None:
    _sleep_recorder(monkeypatch)
    seen: dict[str, float] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path.rsplit("/", 1)[-1]] = request.extensions["timeout"]["read"]
        if request.url.path.endswith("/renew"):
            return httpx.Response(503, text="")
        return httpx.Response(200, json={"command_id": "c1"})

    settings = {} if override is None else {"request_timeout_s": override}
    client = SandboxBrokerClient(
        SandboxBrokerSettings(api_url=API_URL, application_id=APPLICATION, token=TOKEN, **settings),
        transport=httpx.MockTransport(handler),
    )
    expected = 45.0 if override is None else override

    async def scenario():
        with pytest.raises(client_module.SandboxBrokerError):
            await client.get_command("sbx_1", "c1", wait_seconds=20)
        with pytest.raises(SandboxBrokerUnavailableError):
            await client.renew("sbx_1", lease_seconds=60)
        await client.aclose()

    asyncio.run(scenario())
    # A waiting read gets its wait on top of the request timeout.
    assert seen == {"c1": expected + 20, "renew": expected}


def test_archive_transfers_back_off_longer_than_other_retries(monkeypatch) -> None:
    pauses = _sleep_recorder(monkeypatch)
    sent: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(f"{request.method} {request.url.path.rsplit('/', 1)[-1]}")
        # The broker's transfer wait ran out; the contract makes the retry safe.
        return httpx.Response(503, json={"error": {"code": "operation_timed_out"}})

    client = SandboxBrokerClient(
        SandboxBrokerSettings(api_url=API_URL, application_id=APPLICATION, token=TOKEN),
        transport=httpx.MockTransport(handler),
    )

    async def failing(call) -> list[float]:
        pauses.clear()
        with pytest.raises(SandboxBrokerUnavailableError):
            await call
        return list(pauses)

    async def scenario():
        transfer = [2.0, 4.0, 8.0]
        assert await failing(client.put_archive("sbx_1", _tar({"a": b"1"}))) == transfer
        assert await failing(client.get_archive("sbx_1", paths=["a"])) == transfer
        # Other calls, interactive stdio included, keep their quick retries.
        quick = [0.5, 1.0, 2.0]
        assert await failing(client.renew("sbx_1", lease_seconds=60)) == quick
        assert await failing(client.write_command_stdin("sbx_1", "c1", offset=0)) == quick
        assert await failing(client.read_command_stdout("sbx_1", "c1", offset=0)) == quick
        # A call that must not be replayed is still sent once.
        sent.clear()
        assert await failing(client.start_command("sbx_1", "c1", ["true"], retry=False)) == []
        assert sent == ["POST commands"]
        await client.aclose()

    asyncio.run(scenario())


def test_a_lost_transfer_answer_is_retried_with_the_transfer_backoff(monkeypatch) -> None:
    pauses = _sleep_recorder(monkeypatch)
    answers = iter([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")])

    async def handler(request: httpx.Request) -> httpx.Response:
        answer = next(answers, None)
        if answer is not None:
            raise answer
        return httpx.Response(200, json={"files": 1, "bytes": 1})

    client = SandboxBrokerClient(
        SandboxBrokerSettings(api_url=API_URL, application_id=APPLICATION, token=TOKEN),
        transport=httpx.MockTransport(handler),
    )

    async def scenario():
        assert await client.put_archive("sbx_1", _tar({"a": b"1"})) == {"files": 1, "bytes": 1}
        await client.aclose()

    asyncio.run(scenario())
    assert pauses == [2.0, 4.0]


def test_factory_from_env_requires_an_explicit_egress_mode(tmp_path: Path) -> None:
    source = LocalWorkspace(tmp_path)
    with pytest.raises(ValueError, match="CAYU_CLOUD_SANDBOX_EGRESS"):
        BrokeredSandboxCodingEnvironmentFactory.from_env(source_workspace=source, environ=_ENV)
    factory = BrokeredSandboxCodingEnvironmentFactory.from_env(
        source_workspace=source,
        environ={
            **_ENV,
            "CAYU_CLOUD_SANDBOX_EGRESS": "none",
            "CAYU_CLOUD_SANDBOX_LEASE_SECONDS": "300",
        },
    )
    assert (factory.egress, factory.image, factory.lease_seconds) == ("none", "coding-python", 300)
    with pytest.raises(ValueError, match="CAYU_CLOUD_SANDBOX_EGRESS"):
        BrokeredSandboxCodingEnvironmentFactory.from_env(
            source_workspace=source, environ={**_ENV, "CAYU_CLOUD_SANDBOX_EGRESS": "open"}
        )


def test_factory_from_env_reads_an_allowlist_and_its_hosts(tmp_path: Path) -> None:
    source = LocalWorkspace(tmp_path)
    factory = BrokeredSandboxCodingEnvironmentFactory.from_env(
        source_workspace=source,
        environ={
            **_ENV,
            "CAYU_CLOUD_SANDBOX_EGRESS": "allowlist",
            "CAYU_CLOUD_SANDBOX_EGRESS_HOSTS": " PyPI.org., files.pythonhosted.org\n"
            "*.githubusercontent.com,pypi.org ",
        },
    )
    assert factory.egress == "allowlist"
    assert factory.egress_hosts == (
        "*.githubusercontent.com",
        "files.pythonhosted.org",
        "pypi.org",
    )
    with pytest.raises(ValueError, match="CAYU_CLOUD_SANDBOX_EGRESS_HOSTS"):
        BrokeredSandboxCodingEnvironmentFactory.from_env(
            source_workspace=source, environ={**_ENV, "CAYU_CLOUD_SANDBOX_EGRESS": "allowlist"}
        )
    for mode in ("none", "public_web"):
        with pytest.raises(ValueError, match="only for CAYU_CLOUD_SANDBOX_EGRESS=allowlist"):
            BrokeredSandboxCodingEnvironmentFactory.from_env(
                source_workspace=source,
                environ={
                    **_ENV,
                    "CAYU_CLOUD_SANDBOX_EGRESS": mode,
                    "CAYU_CLOUD_SANDBOX_EGRESS_HOSTS": "pypi.org",
                },
            )
    with pytest.raises(ValueError, match="isn't a sandbox egress host"):
        BrokeredSandboxCodingEnvironmentFactory.from_env(
            source_workspace=source,
            environ={
                **_ENV,
                "CAYU_CLOUD_SANDBOX_EGRESS": "allowlist",
                "CAYU_CLOUD_SANDBOX_EGRESS_HOSTS": "pypi.org,10.0.0.1",
            },
        )


def test_allowlist_hosts_follow_clouds_rules() -> None:
    assert validate_egress_hosts(
        ["B.example.com", "a.example.com.", "*.Example.com", "b.example.com"]
    ) == (
        "*.example.com",
        "a.example.com",
        "b.example.com",
    )
    assert len(validate_egress_hosts([f"h{index}.example.com" for index in range(50)])) == 50
    refused = (
        "10.0.0.1",
        "8.8.8.8",
        "::1",
        "2001:db8::1",
        "::ffff:1.2.3.4",
        "*.1.2.3.4",
        "pypi.org:443",
        "https://pypi.org",
        "pypi.org/simple",
        "localhost",
        "pypi",
        "*.com",
        "*.*.example.com",
        "metadata.internal",
        "printer.local",
        "nothing.invalid",
        "example.123",
        "-bad.example.com",
        "a..example.com",
        "",
        "*.",
        "x" * 64 + ".example.com",
    )
    for host in refused:
        with pytest.raises(ValueError, match="isn't a sandbox egress host"):
            validate_egress_hosts([host])
    for hosts in ([], [f"h{index}.example.com" for index in range(51)]):
        with pytest.raises(ValueError, match="1 to 50 hosts"):
            validate_egress_hosts(hosts)
    with pytest.raises(TypeError, match="not one string"):
        validate_egress_hosts("pypi.org")


def test_the_client_sends_an_allowlist_in_the_contracts_shape(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        client = broker.client()
        record = await client.create(
            idempotency_key="allowlist",
            image="coding-python",
            egress_mode="allowlist",
            egress_hosts=["PyPI.org", "*.githubusercontent.com", "pypi.org."],
            lease_seconds=60,
        )
        assert record.egress.mode == "allowlist"
        assert record.egress.hosts == ("*.githubusercontent.com", "pypi.org")
        plain = await client.create(
            idempotency_key="none", image="coding-python", egress_mode="none", lease_seconds=60
        )
        assert plain.egress.hosts is None
        # Refused before anything is sent.
        with pytest.raises(ValueError, match="needs the hosts"):
            await client.create(
                idempotency_key="x",
                image="coding-python",
                egress_mode="allowlist",
                lease_seconds=60,
            )
        with pytest.raises(ValueError, match="only for egress 'allowlist'"):
            await client.create(
                idempotency_key="x",
                image="coding-python",
                egress_mode="public_web",
                egress_hosts=["pypi.org"],
                lease_seconds=60,
            )
        with pytest.raises(ValueError, match="'none', 'public_web' or 'allowlist'"):
            await client.create(
                idempotency_key="x", image="coding-python", egress_mode="open", lease_seconds=60
            )
        # An install that doesn't offer allowlist refuses it with the contract code.
        broker.egress_modes = {"none", "public_web"}
        with pytest.raises(SandboxBrokerRefusedError, match="egress_mode_unsupported"):
            await client.create(
                idempotency_key="unsupported",
                image="coding-python",
                egress_mode="allowlist",
                egress_hosts=["pypi.org"],
                lease_seconds=60,
            )
        await client.aclose()

    asyncio.run(scenario())
    posted = [
        json.loads(request.content)["egress"]
        for request in broker.requests
        if request.method == "POST" and request.url.path.endswith("/sandboxes")
    ]
    assert posted == [
        {"mode": "allowlist", "hosts": ["*.githubusercontent.com", "pypi.org"]},
        {"mode": "none"},
        {"mode": "allowlist", "hosts": ["pypi.org"]},
    ]


def test_a_factory_allocates_its_allowlist_and_fingerprints_its_hosts(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    factory = _factory(
        broker, source, egress="allowlist", egress_hosts=["files.pythonhosted.org", "pypi.org"]
    )
    reordered = _factory(
        broker, source, egress="allowlist", egress_hosts=["pypi.org", "files.pythonhosted.org"]
    )
    other = _factory(broker, source, egress="allowlist", egress_hosts=["pypi.org"])
    assert factory.execution_profile_identity == reordered.execution_profile_identity
    assert factory.execution_profile_identity != other.execution_profile_identity
    with pytest.raises(ValueError, match="needs the hosts"):
        _factory(broker, source, egress="allowlist")
    with pytest.raises(ValueError, match="only for egress 'allowlist'"):
        _factory(broker, source, egress="none", egress_hosts=["pypi.org"])

    async def scenario():
        result = await factory.create(_request())
        assert result.metadata["sandbox_egress"] == "allowlist"
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(scenario())
    (sandbox,) = broker.sandboxes.values()
    assert sandbox.body["egress"] == {
        "mode": "allowlist",
        "hosts": ["files.pythonhosted.org", "pypi.org"],
    }
    assert sandbox.status == "destroyed"


# Allocation and use ----------------------------------------------------------


def test_a_session_copies_in_runs_tools_through_the_broker_and_copies_back(
    tmp_path: Path,
) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    class EditTool(Tool):
        spec = ToolSpec(name="edit_project", parallel_safe=False, workspace_mutation=True)

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.workspace is not None
            content = (await ctx.workspace.read_bytes("calc.py")).content
            await ctx.workspace.create_bytes("calc_fixed.py", content.replace(b"a - b", b"a + b"))
            return ToolResult(content="edited")

    async def scenario():
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        app.register_provider(
            _SingleToolProvider(tool_name="edit_project", arguments={}), default=True
        )
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
            _factory(broker, source),
            default=True,
        )
        app.register_agent(AgentSpec(name="coder", model="scripted-model"), tools=[EditTool()])
        try:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="coding",
                        agent_name="coder",
                        messages=[Message.text("user", "fix it")],
                    )
                )
            ]
        finally:
            await store.close()

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    (sandbox,) = broker.sandboxes.values()
    # The tool edited the sandbox copy, its edit reached the source, and the
    # sandbox was destroyed when the session ended.
    assert (sandbox.root / "calc.py").exists()
    assert (source / "calc_fixed.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert (source / "calc.py").read_bytes() == b"def add(a, b):\n    return a - b\n"
    assert sandbox.status == "destroyed" and sandbox.destroy_reason == "requested"
    assert sandbox.body["egress"] == {"mode": "public_web"}
    # Copy-in used one archive upload with the executable bit, and copy-back
    # read the changed file with the selective archive read.
    assert [call for call in broker.calls if call[0] == "PUT"] == [
        ("PUT", f"/{sandbox.id}/archive")
    ]
    assert os.access(sandbox.root / "run.sh", os.X_OK)
    assert broker.archive_gets == 1


def test_factory_release_and_binding_destroy_the_sandbox(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source, renew_every_s=0.05)
        result = await factory.create(_request())
        runner = result.environment.runner
        assert isinstance(runner, BrokeredSandboxRunner)
        assert result.metadata["sandbox_id"] == runner.sandbox_id
        await asyncio.sleep(0.2)
        assert runner.lease_keeper is not None and runner.lease_keeper.renewals >= 2
        bound = await result.environment.binding.bind(
            result.environment.workspace,
            runner,
            session_id="coding",
            agent_name="coder",
            environment_name="sandbox",
        )
        assert (await bound.workspace.read_bytes("calc.py")).content.startswith(b"def add")
        await result.environment.binding.finalize(bound, outcome="interrupted")
        assert runner.is_closed and not runner.lease_keeper.running
        renewals = broker.renewals
        await asyncio.sleep(0.15)
        assert broker.renewals == renewals

    asyncio.run(scenario())
    assert broker.live() == []


def test_the_factory_gives_each_session_its_knowledge_store_and_scope(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    store = InMemoryKnowledgeStore()
    scope = KnowledgeAccessScope.for_namespace("project:app")

    async def scenario():
        factory = _factory(broker, source, knowledge_store=store, knowledge_access_scope=scope)
        assert factory.configured_knowledge_store is store
        assert factory.configured_knowledge_access_scope == scope
        factory.configured_knowledge_access_scope.allowed_namespaces.append("project:other")
        assert factory.configured_knowledge_access_scope == scope
        result = await factory.create(_request())
        assert result.environment.knowledge_store is store
        assert result.environment.knowledge_access_scope == scope
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

        plain = _factory(broker, source)
        assert plain.configured_knowledge_store is None
        # Knowledge isn't part of the sandbox, so it doesn't change what sessions resume on.
        assert plain.execution_profile_identity == factory.execution_profile_identity
        result = await plain.create(_request("plain"))
        assert result.environment.knowledge_store is None
        assert result.environment.knowledge_access_scope is None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(scenario())
    assert broker.live() == []
    with pytest.raises(ValueError, match="configured together"):
        _factory(broker, source, knowledge_access_scope=scope)


def test_completion_recovery_never_copies_back_from_a_replacement(tmp_path: Path) -> None:
    from cayu.environments import EnvironmentFactoryOperation

    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    factory = _factory(broker, source)
    reconnect = {"operation": EnvironmentFactoryOperation.RECONNECT}

    async def scenario():
        first = await factory.create(_request())
        bound = await first.environment.binding.bind(
            first.environment.workspace,
            first.environment.runner,
            session_id="coding",
            agent_name="coder",
            environment_name="sandbox",
        )
        state = first.environment.binding._completion_finalization_recovery_state(bound)
        # The process that bound it is gone: its in-process ownership goes too.
        assert first.environment.binding.abandon(bound)
        await first.release(None)
        # After a crash the runtime asks again with RECONNECT and no metadata,
        # then recovers the binding against whatever it got.
        second = await factory.create(_request(**reconnect))
        with pytest.raises(RuntimeError, match="can't be reattached"):
            await second.environment.binding._recover_completion_finalization(
                second.environment.workspace,
                second.environment.runner,
                session_id="coding",
                agent_name="coder",
                environment_name="sandbox",
                recovery_state=state,
            )
        await second.release(None)
        # Metadata this factory never wrote is refused before allocating.
        with pytest.raises(RuntimeError, match="do not reconnect"):
            await factory.create(_request(**reconnect, reconnect_metadata={"sandbox_id": "x"}))

    asyncio.run(scenario())
    assert sorted(path.name for path in source.iterdir()) == ["calc.py", "run.sh"]
    assert len(broker.sandboxes) == 2 and broker.live() == []


async def _bind(result):
    return await result.environment.binding.bind(
        result.environment.workspace,
        result.environment.runner,
        session_id="coding",
        agent_name="coder",
        environment_name="sandbox",
    )


def test_git_changes_sees_what_the_session_changed_since_copy_in(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        result = await _factory(broker, source).create(_request())
        bound = await _bind(result)
        context = ToolContext(session_id="coding", runner=bound.runner)
        before = await GitChangesTool().run(context, {"mode": "status"})
        await bound.workspace.create_bytes("calc_fixed.py", b"def add(a, b):\n    return a + b\n")
        after = await GitChangesTool().run(context, {"mode": "status"})
        await result.environment.binding.finalize(bound, outcome="completed")
        return before, after

    before, after = asyncio.run(scenario())
    # The copy-in is the baseline, so only the session's own change shows.
    assert not before.is_error and before.structured["changes"] == []
    assert not after.is_error
    assert [change["path"] for change in after.structured["changes"]] == ["calc_fixed.py"]
    # The change reached the source; the sandbox's repository didn't.
    assert (source / "calc_fixed.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert not (source / ".git").exists()
    assert broker.live() == []


def test_the_baseline_fits_an_index_larger_than_one_command_output(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    # `git ls-files --stage -z` lists each of these files in 57 bytes, so the
    # index outgrows what one broker command returns.
    files = MAX_COMMAND_OUTPUT_BYTES // 57 + 1

    async def scenario():
        runner = await _ready_runner(broker)
        (sandbox,) = broker.sandboxes.values()
        for index in range(files):
            (sandbox.root / f"f{index:05}").write_bytes(b"")
        await initialize_ephemeral_git_baseline(runner)
        listed = await runner.exec(
            ExecCommand.process("git", "ls-files", "--stage", "-z"),
            output_limit_bytes=MAX_COMMAND_OUTPUT_BYTES,
        )
        (sandbox.root / "f00000").write_bytes(b"changed\n")
        changed = await GitChangesTool().run(
            ToolContext(session_id="coding", runner=runner), {"mode": "status"}
        )
        await runner.close()
        return listed, changed

    listed, changed = asyncio.run(scenario())
    assert listed.exit_code == 0 and listed.stdout_truncated
    assert not changed.is_error
    assert [change["path"] for change in changed.structured["changes"]] == ["f00000"]


def test_a_git_baseline_requires_source_and_target_to_exclude_git(tmp_path: Path) -> None:
    source = _project(tmp_path)
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_bytes(b"ref: refs/heads/main\n")
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        other = await _ready_runner(broker)
        protected = (".git",)
        views = (
            (
                LocalWorkspace(source),
                BrokeredSandboxWorkspace(runner, excluded_directory_names=protected),
                "source",
            ),
            (
                LocalWorkspace(source, excluded_directory_names=protected),
                BrokeredSandboxWorkspace(runner),
                "target",
            ),
            (
                LocalWorkspace(source, excluded_directory_names=protected),
                BrokeredSandboxWorkspace(other, excluded_directory_names=protected),
                "target",
            ),
        )
        for source_view, target, role in views:
            binding = BrokeredSandboxCodingBinding(
                target_workspace=target, max_file_bytes=1024 * 1024
            )
            with pytest.raises(ValueError, match=f"requires a {role} .* that excludes .git"):
                await binding.bind(source_view, runner, session_id="coding")
        await runner.close()
        await other.close()

    asyncio.run(scenario())
    # Refused before copy-in: the source's repository never entered a sandbox.
    assert all(list(sandbox.root.iterdir()) == [] for sandbox in broker.sandboxes.values())
    assert (source / ".git" / "HEAD").read_bytes() == b"ref: refs/heads/main\n"


def test_a_subdirectory_target_gets_its_own_baseline(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        (sandbox,) = broker.sandboxes.values()
        (sandbox.root / "project").mkdir()
        protected = (".git",)
        binding = BrokeredSandboxCodingBinding(
            target_workspace=BrokeredSandboxWorkspace(
                runner, cwd="project", excluded_directory_names=protected
            ),
        )
        bound = await binding.bind(
            LocalWorkspace(source, excluded_directory_names=protected),
            runner,
            session_id="coding",
        )
        await bound.workspace.create_bytes("calc_fixed.py", b"def add(a, b):\n    return a + b\n")
        changed = await GitChangesTool().run(
            ToolContext(session_id="coding", runner=runner, workspace=bound.workspace),
            {"mode": "status"},
        )
        await binding.finalize(bound, outcome="completed")
        return changed

    changed = asyncio.run(scenario())
    assert not changed.is_error
    assert [change["path"] for change in changed.structured["changes"]] == ["calc_fixed.py"]
    # The repository is the target's own, not the sandbox root's.
    (sandbox,) = broker.sandboxes.values()
    assert (sandbox.root / "project" / ".git").is_dir()
    assert not (sandbox.root / ".git").exists()
    assert (source / "calc_fixed.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert not (source / ".git").exists()
    assert broker.live() == []


def test_without_a_git_baseline_the_sandbox_runs_no_git(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        result = await _factory(broker, source, git_baseline=False).create(_request())
        bound = await _bind(result)
        await result.environment.binding.finalize(bound, outcome="completed")

    asyncio.run(scenario())
    (sandbox,) = broker.sandboxes.values()
    assert not (sandbox.root / ".git").exists()
    assert not any(argv[0] == "git" for argv in broker.started)
    assert broker.live() == []


def test_a_baseline_that_fails_fails_the_bind(tmp_path: Path, monkeypatch) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    start = broker._start

    async def without_git(sandbox, body):
        # An image without git: every git command exits nonzero.
        if body["argv"][0] == "git":
            body = {**body, "argv": ["false"]}
        return await start(sandbox, body)

    monkeypatch.setattr(broker, "_start", without_git)

    async def scenario():
        result = await _factory(broker, source).create(_request())
        with pytest.raises(RuntimeError, match="could not establish its ephemeral Git baseline"):
            await _bind(result)
        # Binding never adopted the sandbox, so the factory's release destroys it.
        await result.release(None)

    asyncio.run(scenario())
    assert broker.live() == []


def test_git_baseline_must_be_a_bool(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="git_baseline must be a bool"):
        _factory(FakeBroker(tmp_path / "cloud"), _project(tmp_path), git_baseline="yes")


def test_a_resumed_session_gets_a_fresh_sandbox_each_turn(tmp_path: Path) -> None:
    from cayu import ResumeRequest

    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        app.register_provider(_SingleToolProvider(tool_name="noop", arguments={}), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
            _factory(broker, source),
            default=True,
        )

        class Noop(Tool):
            spec = ToolSpec(name="noop")

            async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
                return ToolResult(content="ok")

        app.register_agent(AgentSpec(name="coder", model="scripted-model"), tools=[Noop()])
        try:
            first = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="coding",
                        agent_name="coder",
                        messages=[Message.text("user", "one")],
                    )
                )
            ]
            second = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id="coding", messages=[Message.text("user", "two")])
                )
            ]
            return first, second
        finally:
            await store.close()

    first, second = asyncio.run(scenario())
    assert first[-1].type == EventType.SESSION_COMPLETED, first[-1].payload
    assert second[-1].type == EventType.SESSION_COMPLETED, second[-1].payload
    assert len(broker.sandboxes) == 2 and broker.live() == []


def test_large_trees_move_in_several_archives(tmp_path: Path, monkeypatch) -> None:
    limit = 64 * 1024
    monkeypatch.setattr(runner_module, "MAX_ARCHIVE_BYTES", limit)
    broker = FakeBroker(tmp_path / "cloud")
    broker.archive_limit = limit
    source = tmp_path / "source"
    source.mkdir()
    names = [f"data/part-{index}.bin" for index in range(10)]
    (source / "data").mkdir()
    for index, name in enumerate(names):
        (source / name).write_bytes(bytes([index]) * 20_000)

    async def scenario():
        runner = await _ready_runner(broker)
        workspace = BrokeredSandboxWorkspace(runner)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:") as archive:
            for name in names:
                archive.add(source / name, arcname=name)
        data = buffer.getvalue()
        await workspace.write_tar_stream(io.BytesIO(data), archive_bytes=len(data))
        puts = [call for call in broker.calls if call[0] == "PUT"]
        assert len(puts) >= 4
        # Every upload, padding to the tar record included, fit the bound.
        sizes = [len(request.content) for request in broker.requests if request.method == "PUT"]
        assert max(sizes) <= limit
        for name in names:
            assert (broker.sandboxes[runner.sandbox_id].root / name).read_bytes() == (
                source / name
            ).read_bytes()
        # Reading them back splits the batch the broker reports as too large.
        sink = io.BytesIO()
        result = await workspace.read_tar_stream(names, sink, max_file_bytes=limit)
        assert result.archive_bytes == len(sink.getvalue())
        assert broker.archive_gets > 1
        with tarfile.open(fileobj=io.BytesIO(sink.getvalue()), mode="r:") as archive:
            members = {member.name: member for member in archive}
            assert sorted(members) == sorted(names)
            assert members[names[3]].uname == "cayu.git-mode.v1"
            assert archive.extractfile(names[3]).read() == bytes([3]) * 20_000
        with pytest.raises(FileNotFoundError, match="data/gone.txt"):
            await workspace.read_tar_stream(["data/gone.txt"], io.BytesIO())
        await runner.close()

    asyncio.run(scenario())


def _env_of_exactly(items: int, total: int) -> dict[str, str]:
    """``items`` variables whose ``NAME=value`` strings add up to ``total`` UTF-8 bytes."""

    names = [f"V{index:03d}" for index in range(items)]
    env = dict.fromkeys(names, "")
    remaining = total - sum(len(name) + 1 for name in names)
    for name in names:
        size = min(remaining, 131_071 - len(name) - 1)
        env[name] = "x" * size
        remaining -= size
    assert remaining == 0
    return env


@pytest.mark.parametrize(
    ("env", "refusal"),
    [
        # Cayu Cloud's broker accepts 1,000 variables and 256 KiB, counting each
        # variable as its NAME=value UTF-8 bytes with no terminator.
        ({f"V{index}": "" for index in range(1_000)}, None),
        ({f"V{index}": "" for index in range(1_001)}, "at most 1000 variables"),
        (_env_of_exactly(1_000, 256 * 1024), None),
        (_env_of_exactly(3, 256 * 1024), None),
        (_env_of_exactly(3, 256 * 1024 + 1), "env exceeds 262144 bytes"),
        # One NAME=value is held to Linux's per-string limit, in UTF-8 bytes.
        ({"A": "é" * 65_534 + "x"}, None),
        ({"A": "é" * 65_535}, "131071 bytes each"),
    ],
)
def test_command_env_matches_the_brokers_limits(env: dict[str, str], refusal: str | None) -> None:
    if refusal is None:
        client_module.validate_command_shape(["true"], env, None)
    else:
        with pytest.raises(ValueError, match=refusal):
            client_module.validate_command_shape(["true"], env, None)


def test_copy_back_enforces_the_limits_before_buffering(tmp_path: Path, monkeypatch) -> None:
    limit = 64 * 1024
    monkeypatch.setattr(runner_module, "MAX_ARCHIVE_BYTES", limit)
    broker = FakeBroker(tmp_path / "cloud")
    names = [f"grown-{index}.bin" for index in range(10)]

    async def scenario():
        runner = await _ready_runner(broker)
        root = broker.sandboxes[runner.sandbox_id].root
        for name in names:
            # Files the sandbox grew after the manifest was taken.
            (root / name).write_bytes(b"x" * 20_000)
        workspace = BrokeredSandboxWorkspace(runner)
        # Split batches are streamed one archive at a time, and the read stops
        # at the first member past the total budget.
        broker.archive_limit = limit
        with pytest.raises(RuntimeError, match="max_total_bytes=30000"):
            await workspace.read_tar_stream(names, io.BytesIO(), max_total_bytes=30_000)
        assert broker.archive_gets < len(names)
        # Tar framing never counts as content: files that fit the limits
        # exactly are read even when their combined archive needs a split.
        exact = names[:3]
        broker.archive_gets = 0
        sink = io.BytesIO()
        await workspace.read_tar_stream(
            exact, sink, max_file_bytes=20_000, max_total_bytes=3 * 20_000
        )
        assert broker.archive_gets > 1
        with tarfile.open(fileobj=io.BytesIO(sink.getvalue()), mode="r:") as archive:
            assert sorted(member.name for member in archive) == sorted(exact)
        # Within one archive, a member over the budget is refused before it is read.
        broker.archive_limit = 8 * 1024 * 1024
        sink = io.BytesIO()
        with pytest.raises(RuntimeError, match="max_total_bytes=30000"):
            await workspace.read_tar_stream(names[:3], sink, max_total_bytes=30_000)
        assert len(sink.getvalue()) < 2 * 20_000
        with pytest.raises(RuntimeError, match="max_file_bytes=1000"):
            await workspace.read_tar_stream(names[:1], io.BytesIO(), max_file_bytes=1000)
        await runner.close()

    asyncio.run(scenario())


def _tar(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_large_trees_and_files_move_as_staged_copies(tmp_path: Path, monkeypatch) -> None:
    limit = 64 * 1024
    monkeypatch.setattr(runner_module, "MAX_ARCHIVE_BYTES", limit)
    broker = FakeBroker(tmp_path / "cloud")
    broker.archive_limit = limit
    broker.staged_copies = {"max_bytes": 8 * 1024 * 1024, "sandbox_max_bytes": 32 * 1024 * 1024}
    big = os.urandom(3 * limit)
    files = {"repo/README.md": b"# repo\n", "repo/objects/pack.bin": big}

    async def scenario():
        runner = await _ready_runner(broker)
        assert runner.staged_copies is not None
        workspace = BrokeredSandboxWorkspace(runner)
        data = _tar(files)
        await workspace.write_tar_stream(io.BytesIO(data), archive_bytes=len(data))
        root = broker.sandboxes[runner.sandbox_id].root
        assert (root / "repo/objects/pack.bin").read_bytes() == big
        # One staged copy, no archive uploads, and the storage never saw the broker credential.
        assert [call for call in broker.calls if call[0] == "PUT"] == []
        assert [call[1].split("/")[2:] for call in broker.calls if call[0] == "POST"][-2:] == [
            ["copies"],
            ["copies", next(iter(broker.sandboxes[runner.sandbox_id].copies)), "commit"],
        ]
        (upload,) = [request for request in broker.storage_requests if request.method == "PUT"]
        assert "authorization" not in upload.headers
        assert upload.content == data
        # A file too large for one archive comes back as one staged copy.
        sink = io.BytesIO()
        await workspace.read_tar_stream(["repo/objects/pack.bin"], sink, max_file_bytes=len(big))
        with tarfile.open(fileobj=io.BytesIO(sink.getvalue()), mode="r:") as archive:
            assert archive.extractfile("repo/objects/pack.bin").read() == big
        (download,) = [request for request in broker.storage_requests if request.method == "GET"]
        assert "authorization" not in download.headers
        with pytest.raises(RuntimeError, match=f"max_file_bytes={limit}"):
            await workspace.read_tar_stream(
                ["repo/objects/pack.bin"], io.BytesIO(), max_file_bytes=limit
            )
        # Members are checked before anything is sent.
        escaping = _tar({"../outside": big})
        with pytest.raises(ValueError, match="relative files"):
            await workspace.write_tar_stream(io.BytesIO(escaping), archive_bytes=len(escaping))
        await runner.close()

    asyncio.run(scenario())


def test_staged_copies_mode_chooses_how_the_workspace_moves(tmp_path: Path, monkeypatch) -> None:
    limit = 64 * 1024
    monkeypatch.setattr(runner_module, "MAX_ARCHIVE_BYTES", limit)
    broker = FakeBroker(tmp_path / "cloud")
    broker.archive_limit = limit
    broker.staged_copies = {"max_bytes": 8 * 1024 * 1024, "sandbox_max_bytes": 32 * 1024 * 1024}
    # Larger than one archive in all, but every file fits in one.
    files = {f"repo/part-{index}.bin": os.urandom(limit // 2) for index in range(4)}
    data = _tar(files)

    def archive_puts() -> int:
        return len([call for call in broker.calls if call[0] == "PUT"])

    def staged_uploads() -> int:
        return len([request for request in broker.storage_requests if request.method == "PUT"])

    async def scenario():
        runner = await _ready_runner(broker)
        root = broker.sandboxes[runner.sandbox_id].root
        # auto: a staged copy-in that fails part-way is sent again as archives.
        broker.fail_copy_ins = 1
        await BrokeredSandboxWorkspace(runner).write_tar_stream(
            io.BytesIO(data), archive_bytes=len(data)
        )
        assert staged_uploads() == 1 and archive_puts() > 1
        assert {name: (root / name).read_bytes() for name in files} == files
        # never: archives only, even where staged copies are offered.
        uploads, puts = staged_uploads(), archive_puts()
        never = BrokeredSandboxWorkspace(runner, staged_copies="never")
        await never.write_tar_stream(io.BytesIO(data), archive_bytes=len(data))
        assert staged_uploads() == uploads and archive_puts() > puts
        (root / "big.bin").write_bytes(os.urandom(2 * limit))
        with pytest.raises(RuntimeError, match="larger than one broker archive"):
            await never.read_tar_stream(["big.bin"], io.BytesIO(), max_file_bytes=4 * limit)
        # always: even a small tree moves as one staged copy, and a failed one
        # is not sent again.
        uploads, puts = staged_uploads(), archive_puts()
        always = BrokeredSandboxWorkspace(runner, staged_copies="always")
        small = _tar({"small.txt": b"small"})
        await always.write_tar_stream(io.BytesIO(small), archive_bytes=len(small))
        assert staged_uploads() == uploads + 1 and archive_puts() == puts
        assert (root / "small.txt").read_bytes() == b"small"
        broker.fail_copy_ins = 1
        with pytest.raises(SandboxCopyFailedError):
            await always.write_tar_stream(io.BytesIO(data), archive_bytes=len(data))
        await runner.close()
        broker.staged_copies = None
        plain = await _ready_runner(broker)
        with pytest.raises(RuntimeError, match="offers no staged copies"):
            await BrokeredSandboxWorkspace(plain, staged_copies="always").write_tar_stream(
                io.BytesIO(small), archive_bytes=len(small)
            )
        await plain.close()
        with pytest.raises(ValueError, match="staged_copies"):
            BrokeredSandboxWorkspace(plain, staged_copies="sometimes")  # type: ignore[arg-type]

    asyncio.run(scenario())


def test_the_client_moves_staged_copies_and_reports_failures(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {"max_bytes": 1024 * 1024, "sandbox_max_bytes": 4 * 1024 * 1024}

    async def scenario():
        client = broker.client()
        runner = await _ready_runner(broker)
        data = _tar({"a.txt": b"a", ".git/HEAD": b"ref: refs/heads/main\n"})
        digest = hashlib.sha256(data).hexdigest()
        done = await client.copy_in(
            runner.sandbox_id, io.BytesIO(data), size=len(data), sha256=digest, path="repo"
        )
        assert (done.status, done.files) == ("done", 2)
        root = broker.sandboxes[runner.sandbox_id].root
        assert (root / "repo/.git/HEAD").read_bytes() == b"ref: refs/heads/main\n"
        sink = io.BytesIO()
        out = await client.copy_out(
            runner.sandbox_id, sink, max_bytes=1024 * 1024, paths=["repo", "gone"]
        )
        assert out.missing == ("gone",) and out.size == len(sink.getvalue())
        with tarfile.open(fileobj=io.BytesIO(sink.getvalue()), mode="r:") as archive:
            assert archive.extractfile("repo/a.txt").read() == b"a"
        with pytest.raises(SandboxArchiveTooLargeError) as too_large:
            await client.copy_out(runner.sandbox_id, io.BytesIO(), max_bytes=100, paths=["repo"])
        assert too_large.value.bytes == out.size
        with pytest.raises(SandboxCopyFailedError) as mismatch:
            await client.copy_in(
                runner.sandbox_id, io.BytesIO(data), size=len(data), sha256="0" * 64
            )
        assert mismatch.value.code == "copy_digest_mismatch"
        with pytest.raises(SandboxArchiveTooLargeError):
            await client.start_copy_in(
                runner.sandbox_id, "huge", size=2 * 1024 * 1024, sha256=digest
            )
        await runner.close()
        broker.staged_copies = None
        plain = await _ready_runner(broker)
        assert plain.staged_copies is None
        with pytest.raises(SandboxBrokerRefusedError) as unavailable:
            await client.start_copy_out(plain.sandbox_id, "c")
        assert unavailable.value.code == "copies_unavailable"
        await plain.close()
        await client.aclose()

    asyncio.run(scenario())


def test_staged_urls_must_be_https_without_credentials() -> None:
    assert client_module._staged_url("https://bucket.s3.amazonaws.com/k?X-Amz-Signature=x")
    assert client_module._staged_url("http://127.0.0.1:9000/bucket/k")
    for url in (
        "http://storage.example.test/k",
        "https://user:pass@storage.example.test/k",
        "file:///etc/passwd",
    ):
        with pytest.raises(client_module.SandboxBrokerError):
            client_module._staged_url(url)


def test_files_larger_than_one_archive_need_staged_copies(tmp_path: Path) -> None:
    limits = SandboxWorkspaceTransferLimits(max_file_bytes=16 * 1024 * 1024)
    broker = FakeBroker(tmp_path / "cloud")
    factory = _factory(broker, _project(tmp_path), transfer_limits=limits)
    with pytest.raises(RuntimeError, match="needs staged copies"):
        asyncio.run(factory.create(_request()))
    (sandbox,) = broker.sandboxes.values()
    assert sandbox.status == "destroyed"
    broker.staged_copies = {"max_bytes": 32 * 1024 * 1024, "sandbox_max_bytes": 64 * 1024 * 1024}

    async def scenario():
        result = await factory.create(_request("offered"))
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(scenario())


def test_the_default_file_limit_follows_the_sandboxs_staged_copies(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    project = _project(tmp_path)

    async def file_limit(factory, name: str) -> int:
        result = await factory.create(_request(name))
        try:
            return result.environment.binding.max_file_bytes
        finally:
            await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    async def scenario():
        default = _factory(broker, project)
        # Without staged copies a file must fit in one archive.
        assert await file_limit(default, "plain") == runner_module.MAX_TRANSFER_FILE_BYTES
        broker.staged_copies = {
            "max_bytes": 32 * 1024 * 1024,
            "sandbox_max_bytes": 64 * 1024 * 1024,
        }
        assert await file_limit(default, "staged") == (
            32 * 1024 * 1024 - runner_module.STAGED_FILE_OVERHEAD_BYTES
        )
        # Explicit limits and staged_copies="never" keep their own bound.
        explicit = _factory(
            broker,
            project,
            transfer_limits=SandboxWorkspaceTransferLimits(max_file_bytes=1024 * 1024),
        )
        assert await file_limit(explicit, "explicit") == 1024 * 1024
        never = _factory(broker, project, staged_copies="never")
        assert await file_limit(never, "never") == runner_module.MAX_TRANSFER_FILE_BYTES

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="staged_copies='never'"):
        _factory(
            broker,
            project,
            staged_copies="never",
            transfer_limits=SandboxWorkspaceTransferLimits(max_file_bytes=16 * 1024 * 1024),
        )
    broker.staged_copies = None
    always = _factory(broker, project, staged_copies="always")
    with pytest.raises(RuntimeError, match="offers no staged copies"):
        asyncio.run(always.create(_request("always")))
    assert broker.sandboxes[next(reversed(broker.sandboxes))].status == "destroyed"
    assert always.execution_profile_identity != _factory(broker, project).execution_profile_identity


def test_a_tiny_workspace_finalizes_beside_an_8_gib_staged_copy_limit(tmp_path: Path) -> None:
    # The derived 4 GiB file limit is as large as the default staging domain;
    # the source revision checks must reserve the measured files, not it.
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {
        "max_bytes": 8 * 1024 * 1024 * 1024,
        "sandbox_max_bytes": 16 * 1024 * 1024 * 1024,
    }
    source = _project(tmp_path)

    async def scenario() -> int:
        result = await _factory(broker, source).create(_request())
        binding = result.environment.binding
        bound = await _bind(result)
        await bound.workspace.write_bytes("calc.py", b"def add(a, b):\n    return a + b\n")
        await binding.finalize(bound, outcome="completed")
        return binding.max_file_bytes

    max_file_bytes = asyncio.run(scenario())

    assert max_file_bytes == 4 * 1024 * 1024 * 1024
    assert (source / "calc.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert broker.live() == []


def test_setup_commands_take_a_timeout_and_report_their_output(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    project = _project(tmp_path)
    failing = _factory(
        broker, project, setup_commands=["echo installing; echo 'no such package' >&2; exit 3"]
    )
    with pytest.raises(RuntimeError) as failed:
        asyncio.run(failing.create(_request("failing")))
    message = str(failed.value)
    assert "setup command failed (exit code 3)" in message
    assert "stderr: no such package" in message and "stdout: installing" in message
    slow = _factory(broker, project, setup_commands=["sleep 5"], setup_command_timeout_s=1)
    with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
        asyncio.run(slow.create(_request("slow")))
    assert all(sandbox.status == "destroyed" for sandbox in broker.sandboxes.values())
    with pytest.raises(ValueError, match="setup_command_timeout_s"):
        _factory(broker, project, setup_command_timeout_s=0)


def test_paths_keep_their_whitespace(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        root = broker.sandboxes[runner.sandbox_id].root
        (root / "report.txt").write_bytes(b"plain")
        (root / " report.txt ").write_bytes(b"padded")
        (root / " sub ").mkdir()
        (root / "sub").mkdir()
        sink = io.BytesIO()
        await BrokeredSandboxWorkspace(runner).read_tar_stream([" report.txt "], sink)
        with tarfile.open(fileobj=io.BytesIO(sink.getvalue()), mode="r:") as archive:
            assert archive.extractfile(" report.txt ").read() == b"padded"
        result = await runner.exec(ExecCommand.process("pwd"), cwd=" sub ")
        assert result.stdout.rstrip("\n").endswith("/ sub ")
        await runner.close()

    asyncio.run(scenario())


def test_commands_run_asynchronously_in_the_workspace(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        (broker.sandboxes[runner.sandbox_id].root / "sub").mkdir()
        result = await runner.exec(
            ExecCommand.bash("pwd; echo $GREETING"), cwd="sub", env={"GREETING": "hi"}
        )
        assert result.exit_code == 0
        assert result.stdout.splitlines()[-1] == "hi"
        assert result.stdout.splitlines()[0].endswith("/sub")
        # A command outlasting one poll is polled until it finishes; long
        # timeouts are accepted, up to the sandbox's lifetime.
        slow = await runner.exec(ExecCommand.bash("sleep 1.5; echo done"), timeout_s=1800)
        assert slow.stdout.strip() == "done"
        timed_out = await runner.exec(ExecCommand.process("sleep", "5"), timeout_s=1)
        assert timed_out.timed_out and timed_out.exit_code == -1
        # The broker killed the process group, so workspace fencing can settle.
        assert runner_workspace_mutation_settlement(result=timed_out, error=None) == "complete"
        long = await runner.exec(ExecCommand.process("printf", "x" * 3000), output_limit_bytes=10)
        assert long.stdout == "x" * 10 and long.stdout_truncated
        with pytest.raises(ValueError, match="remaining lifetime"):
            await runner.exec(ExecCommand.process("true"), timeout_s=9 * 3600)
        with pytest.raises(ValueError, match="stdin"):
            await runner.exec(ExecCommand.process("cat"), stdin="x" * (1024 * 1024 + 1))
        # Control characters take six bytes each in the response JSON, so the
        # runner asks for enough encoded output to return the raw limit.
        control = await runner.exec(
            ExecCommand.bash("head -c 3000 /dev/zero | tr '\\0' '\\1'"), output_limit_bytes=1000
        )
        assert control.stdout == "\x01" * 1000 and control.stdout_truncated
        started = len(broker.started)
        with pytest.raises(ValueError, match="131071 bytes each"):
            await runner.exec(ExecCommand.process("echo", "x" * 131_072))
        with pytest.raises(ValueError, match="argv exceeds"):
            await runner.exec(ExecCommand.process("echo", *(["x" * 100_000] * 3)))
        with pytest.raises(ValueError, match="env exceeds"):
            await runner.exec(
                ExecCommand.process("true"),
                env={"A": "x" * 131_069, "B": "x" * 131_069, "C": "x"},
            )
        assert len(broker.started) == started
        with pytest.raises(ValueError, match="outside the runner root"):
            await runner.exec(ExecCommand.process("true"), cwd="/etc")
        # A sandbox at its command limit is retried; the command had not started.
        broker.busy_responses = 2
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        await runner.close()

    asyncio.run(scenario())
    assert broker.started[0] == ["/bin/sh", "-c", "pwd; echo $GREETING"]


def _started_timeouts(broker: FakeBroker, sandbox_id: str) -> list[int]:
    return [
        command.body["timeout_seconds"]
        for command in broker.sandboxes[sandbox_id].commands.values()
    ]


def test_a_command_without_a_timeout_leaves_a_clock_margin(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    # The broker checks the timeout against its own clock, which here runs
    # half a minute ahead of the Agent's.
    broker.clock_ahead_seconds = 30

    async def scenario():
        runner = await _ready_runner(broker)
        remaining = runner._remaining_lifetime_seconds()
        assert remaining is not None
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        await runner.close()
        return remaining

    remaining = asyncio.run(scenario())
    assert broker.lifetime_refusals == 0
    (timeout,) = _started_timeouts(broker, next(iter(broker.sandboxes)))
    assert remaining - 61 <= timeout <= remaining - 60


def test_a_refused_lifetime_timeout_is_retried_once_with_the_brokers_number(
    tmp_path: Path,
) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    # Beyond the margin: the broker refuses, naming the lifetime it sees.
    broker.clock_ahead_seconds = 600

    async def scenario():
        runner = await _ready_runner(broker)
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        # An explicit timeout within the Agent's remaining lifetime is asked
        # again too, so clock skew alone never fails it.
        explicit = runner._remaining_lifetime_seconds() - 300
        assert (await runner.exec(ExecCommand.process("true"), timeout_s=explicit)).exit_code == 0
        # Past the Agent's own remaining lifetime it is refused before sending.
        with pytest.raises(ValueError, match="remaining lifetime"):
            await runner.exec(ExecCommand.process("true"), timeout_s=9 * 3600)
        sandbox = broker.sandboxes[runner.sandbox_id]
        broker_remaining = int((sandbox.max_expires_at - datetime.now(UTC)).total_seconds())
        await runner.close()
        return broker_remaining - 600

    broker_remaining = asyncio.run(scenario())
    assert broker.lifetime_refusals == 2
    timeouts = _started_timeouts(broker, next(iter(broker.sandboxes)))
    assert len(timeouts) == 2
    assert all(broker_remaining - 5 <= timeout <= broker_remaining + 1 for timeout in timeouts)


def test_a_second_lifetime_refusal_is_final(tmp_path: Path, monkeypatch) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        # A broker whose clock keeps moving past every number it names.
        original = broker._start

        async def refusing(sandbox, body):
            broker.clock_ahead_seconds += 3600
            return await original(sandbox, body)

        monkeypatch.setattr(broker, "_start", refusing)
        with pytest.raises(SandboxBrokerRefusedError, match="remaining lifetime"):
            await runner.exec(ExecCommand.process("true"))
        # The refusals started nothing, so the runner stays usable.
        monkeypatch.setattr(broker, "_start", original)
        broker.clock_ahead_seconds = 0
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        await runner.close()

    asyncio.run(scenario())
    assert broker.lifetime_refusals == 2


def test_a_dropped_start_is_recovered_by_replaying_its_id(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        marker = broker.sandboxes[runner.sandbox_id].root / "runs"
        broker.drop_start_responses = 1
        broker.unknown_errors = 1
        result = await runner.exec(ExecCommand.bash("echo run >> runs; echo ok"))
        assert result.stdout.strip() == "ok"
        # The replay returned the started command; it never ran twice.
        assert marker.read_text() == "run\n"
        assert runner.lifecycle_state == "reusable"
        await runner.close()

    asyncio.run(scenario())


@pytest.fixture
def quick_cancel_settlement(monkeypatch):
    monkeypatch.setattr(runner_module, "_CANCEL_SETTLE_SECONDS", 1.0)


async def _cancel(task: asyncio.Task) -> asyncio.CancelledError:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError as cancellation:
        return cancellation
    raise AssertionError("the command was not cancelled")


def _receipts(error: BaseException) -> list[str]:
    return [artifact["status"] for artifact in getattr(error, "artifacts", [])]


def test_a_cancelled_command_is_cancelled_in_the_sandbox(
    tmp_path: Path, quick_cancel_settlement
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.3)
        cancellation = await _cancel(task)
        # The broker confirmed the stop: the runner keeps working, and the
        # cancellation carries the receipt workspace fencing settles on.
        assert broker.running() == []
        assert runner.lifecycle_state == "reusable"
        assert _receipts(cancellation) == ["completed"]
        assert runner_workspace_mutation_settlement(result=None, error=cancellation) == "complete"
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        # Without a confirmed stop, no new work may race the old command.
        broker.cancel_confirms = False
        abandoned = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.3)
        cancellation = await _cancel(abandoned)
        assert runner.lifecycle_state == "poisoned"
        # An unconfirmed stop poisons the runner, but the sandbox is not lost.
        assert runner.loss_reason is None
        assert _receipts(cancellation) == ["failed"]
        await runner.close()

    asyncio.run(scenario())


def test_a_cancel_overtaking_its_start_waits_for_the_start(
    tmp_path: Path, quick_cancel_settlement
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        broker.start_delay = 0.5
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.1)
        # The start is still in flight, so a cancel now would answer 404.
        cancellation = await _cancel(task)
        broker.start_delay = 0.0
        # The runner waited for the start, then cancelled what it started.
        assert len(broker.started) == 1 and broker.running() == []
        assert runner.lifecycle_state == "reusable"
        assert _receipts(cancellation) == ["completed"]
        await runner.close()

    asyncio.run(scenario())


def test_a_start_slower_than_the_settle_window_is_still_cancelled(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner_module, "_CANCEL_SETTLE_SECONDS", 0.2)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        broker.start_delay = 0.5
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.05)
        cancellation = await _cancel(task)
        broker.start_delay = 0.0
        # The window to confirm the stop starts once the start has answered.
        assert len(broker.started) == 1 and broker.running() == []
        assert runner.lifecycle_state == "reusable"
        assert _receipts(cancellation) == ["completed"]
        await runner.close()

    asyncio.run(scenario())


def test_a_cancel_answered_after_the_settle_window_is_still_confirmed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner_module, "_CANCEL_SETTLE_SECONDS", 0.2)
    broker = FakeBroker(tmp_path / "cloud")
    transport = broker.transport

    async def slow_cancel(request: httpx.Request) -> httpx.Response:
        if not (request.method == "POST" and request.url.path.endswith("/cancel")):
            return await transport(request)
        # Slower than the window, though well inside the request timeout, and
        # answered while the process group is still stopping.
        await asyncio.sleep(0.4)
        answer = await transport(request)
        stopping = {**answer.json(), "status": "running", "exit_code": None, "finished_at": None}
        return httpx.Response(202, json=stopping)

    monkeypatch.setattr(broker, "transport", slow_cancel)

    async def scenario():
        runner = await _ready_runner(broker)
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.3)
        cancellation = await _cancel(task)
        assert broker.running() == []
        assert runner.lifecycle_state == "reusable"
        assert _receipts(cancellation) == ["completed"]
        await runner.close()

    asyncio.run(scenario())


def _lost_then_refused_start(broker: FakeBroker, *, garbled: bool = False) -> SandboxBrokerClient:
    """A transport where the first start applies but its answer is lost, then a 401.

    With ``garbled`` the answer arrives as a 2xx body that isn't JSON, as a
    proxy or ingress might send.
    """

    starts = {"count": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/commands"):
            starts["count"] += 1
            if starts["count"] == 1:
                await broker.handle(request)
                if garbled:
                    return httpx.Response(202, text="<html>proxy</html>")
                raise httpx.ReadError("connection reset", request=request)
            if starts["count"] == 2:
                return httpx.Response(401, text="")
        return await broker.transport(request)

    return SandboxBrokerClient(
        SandboxBrokerSettings(api_url=API_URL, application_id=APPLICATION, token=TOKEN),
        transport=httpx.MockTransport(handler),
    )


def test_a_refused_replay_never_certifies_a_started_command(
    tmp_path: Path, quick_cancel_settlement
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        ready = await _ready_runner(broker)
        # The replay is refused, but the first start already runs the command.
        runner = BrokeredSandboxRunner(_lost_then_refused_start(broker), ready.sandbox)
        with pytest.raises(SandboxBrokerRefusedError):
            await runner.exec(ExecCommand.process("sleep", "30"))
        assert len(broker.started) == 1 and broker.running() == []
        assert runner.lifecycle_state == "reusable"
        # A caller cancelling between the lost start and its replay gets the
        # same treatment: the command is stopped before a receipt is issued.
        cancelled = BrokeredSandboxRunner(_lost_then_refused_start(broker), ready.sandbox)
        task = asyncio.create_task(cancelled.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.1)
        cancellation = await _cancel(task)
        assert len(broker.started) == 2 and broker.running() == []
        assert _receipts(cancellation) == ["completed"]
        # An unparseable 2xx answer means the start was applied, too.
        garbled = BrokeredSandboxRunner(
            _lost_then_refused_start(broker, garbled=True), ready.sandbox
        )
        with pytest.raises(SandboxBrokerRefusedError):
            await garbled.exec(ExecCommand.process("sleep", "30"))
        assert len(broker.started) == 3 and broker.running() == []
        assert garbled.lifecycle_state == "reusable"
        await ready.close()

    asyncio.run(scenario())


def test_waiting_for_a_busy_slot_never_dispatches_after_poisoning(
    tmp_path: Path, quick_cancel_settlement
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        first = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.3)
        broker.busy_responses = 1_000
        broker.cancel_confirms = False
        waiting = asyncio.create_task(runner.exec(ExecCommand.process("echo", "after-poison")))
        await asyncio.sleep(0.1)
        await _cancel(first)
        assert runner.lifecycle_state == "poisoned"
        broker.busy_responses = 0
        with pytest.raises(RuntimeError, match="closed"):
            await waiting
        assert ["echo", "after-poison"] not in broker.started
        await runner.close()

    asyncio.run(scenario())


# Interactive processes ----------------------------------------------------

_UPPER = (
    "import sys\n"
    "print('ready', flush=True)\n"
    "for line in sys.stdin:\n"
    "    sys.stdout.write(line.upper())\n"
    "    sys.stdout.flush()\n"
    "sys.exit(3)\n"
)


async def _read_all(process) -> bytes:
    output = bytearray()
    while chunk := await process.read():
        output.extend(chunk)
    return bytes(output)


def test_an_interactive_process_streams_its_stdio_through_the_broker(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        (broker.sandboxes[runner.sandbox_id].root / "sub").mkdir()
        capability = runner.interactive_process_capability()
        assert capability is not None and capability.secret_values == ()
        process = await capability.open_interactive_process(
            ExecCommand.process(sys.executable, "-c", _UPPER),
            cwd="sub",
            env={"MODE": "review"},
            timeout_s=60,
        )
        assert process.secret_values == ()
        assert await process.read() == b"ready\n"
        await process.write(b"one\n")
        assert await process.read() == b"ONE\n"
        # Lost answers are recovered by repeating the same offset.
        broker.drop_stdio_responses = 2
        await process.write(b"two\n")
        await process.close_stdin()
        assert await _read_all(process) == b"TWO\n"
        assert await process.wait() == 3
        with pytest.raises(BrokenPipeError):
            await process.write(b"late\n")
        await process.close()
        await process.close()
        [body] = [command.body for command in broker.running() + _commands(broker)]
        assert body["interactive"] is True
        assert body["cwd"] == "sub"
        assert body["env"] == {"MODE": "review"}
        assert body["timeout_seconds"] == 60
        assert "stdin" not in body and "output_limit_bytes" not in body
        await runner.close()

    asyncio.run(scenario())


def _commands(broker: FakeBroker):
    return [
        command
        for sandbox in broker.sandboxes.values()
        for command in sandbox.commands.values()
        if command.status != "running"
    ]


def test_closing_an_interactive_process_stops_its_whole_tree(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        capability = runner.interactive_process_capability()
        assert capability is not None
        process = await capability.open_interactive_process(
            ExecCommand.bash("sleep 60 & echo $!; wait"), timeout_s=60
        )
        child = int((await process.read()).strip())
        await process.close()
        assert broker.running() == []
        [command] = _commands(broker)
        assert command.status == "cancelled"
        await asyncio.sleep(0.1)
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        # The broker drops output nobody read once the command is stopped.
        assert await process.read() == b""
        await runner.close()

    asyncio.run(scenario())


def test_an_interactive_process_past_its_timeout_ends(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        capability = runner.interactive_process_capability()
        assert capability is not None
        process = await capability.open_interactive_process(
            ExecCommand.process("sleep", "30"), timeout_s=1
        )
        assert await _read_all(process) == b""
        assert await process.wait() == -1
        await process.close()
        await runner.close()

    asyncio.run(scenario())


def test_an_interactive_process_in_a_destroyed_sandbox_reports_the_lost_lease(
    tmp_path: Path,
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        capability = runner.interactive_process_capability()
        assert capability is not None
        process = await capability.open_interactive_process(
            ExecCommand.process("sleep", "30"), timeout_s=60
        )
        await broker.client().destroy(runner.sandbox_id)
        assert await process.read() == b""
        with pytest.raises(SandboxLeaseLostError):
            await process.wait()
        with pytest.raises(BrokenPipeError):
            await process.write(b"x")
        await process.close()
        await runner.close()

    asyncio.run(scenario())


def test_runner_agent_launch_runs_the_cli_in_a_brokered_sandbox(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    home = tmp_path / "home"
    script = "import json, os\nprint(json.dumps({'home': os.environ['HOME']}), flush=True)\n" + (
        _UPPER.removeprefix("import sys\nprint('ready', flush=True)\n")
    )

    async def scenario():
        runner = await _ready_runner(broker)
        async with runner_agent_launch(
            runner=runner,
            cli=sys.executable,
            environment={"HOME": str(home), "MODE": "review"},
            timeout_s=60,
        ) as launch:
            launch.protect_secrets(lambda secret: None)
            shim = await asyncio.create_subprocess_exec(
                launch.executable,
                "-c",
                "import sys\n" + script,
                env=scrub_external_environment(launch.environment),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(shim.communicate(b"a\nb\n"), 30)
            assert shim.returncode == 3
            first, rest = stdout.split(b"\n", 1)
            assert json.loads(first) == {"home": str(home)}
            assert rest == b"A\nB\n"
        # The broker owns HOME in a command's environment; the launch sets it in the guest.
        [command] = _commands(broker)
        assert command.body["env"] == {"MODE": "review"}
        assert (command.status, command.exit_code) == ("exited", 3)
        await runner.close()

    asyncio.run(scenario())


# Admission -----------------------------------------------------------------


def test_untrusted_work_needs_every_guarantee_item_and_no_network(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    untrusted = ExecutionRequirements.untrusted()

    def decision(candidate, *, stage="pre_exposure"):
        return evaluate_execution_admission(
            candidate=candidate.candidate,
            requirements=untrusted,
            evidence=candidate.evidence,
            stage=stage,
        )

    async def scenario():
        request = _request(execution_requirements=untrusted)
        closed = _factory(broker, _project(tmp_path), egress="none")
        pre_create = decision(closed.execution_admission_candidate(request), stage="pre_create")
        assert pre_create.status == "admitted", pre_create.refusals
        runner = await _ready_runner(broker, egress="none")
        admitted = decision(runner.execution_admission_candidate())
        assert admitted.status == "admitted", admitted.refusals
        await runner.close()
        # The open internet is never deny-by-default, before or after allocation.
        open_web = _factory(broker, tmp_path / "source", egress="public_web")
        refused = decision(open_web.execution_admission_candidate(request), stage="pre_create")
        assert {refusal.capability for refusal in refused.refusals} == {"deny_by_default_network"}
        runner = await _ready_runner(broker, egress="public_web")
        refused = decision(runner.execution_admission_candidate())
        assert {refusal.capability for refusal in refused.refusals} == {"deny_by_default_network"}
        await runner.close()
        broker.guarantees = ("untrusted_code_isolation", "egress_only_via_broker")
        weaker = await _ready_runner(broker)
        refused = decision(weaker.execution_admission_candidate())
        assert refused.status == "refused"
        assert {refusal.capability for refusal in refused.refusals} == {
            "real_credential_non_possession",
            "guest_privilege_containment",
        }
        await weaker.close()

    asyncio.run(scenario())


def test_deny_by_default_follows_the_egress_guarantee_of_each_mode(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.guarantees_version = 2
    untrusted = ExecutionRequirements.untrusted()
    hosts = ["pypi.org", "*.githubusercontent.com"]

    def decision(candidate, requirements=untrusted, *, stage="pre_exposure"):
        return evaluate_execution_admission(
            candidate=candidate.candidate,
            requirements=requirements,
            evidence=candidate.evidence,
            stage=stage,
        )

    def network_claim(runner):
        return runner.execution_admission_candidate().evidence.claim_for("deny_by_default_network")

    async def scenario():
        request = _request(execution_requirements=untrusted)
        allowlist = _factory(broker, _project(tmp_path), egress="allowlist", egress_hosts=hosts)
        pre_create = decision(allowlist.execution_admission_candidate(request), stage="pre_create")
        assert pre_create.status == "admitted", pre_create.refusals
        # A version 2 record backs deny-by-default with its egress item, on none and allowlist.
        for egress, egress_hosts in (("allowlist", hosts), ("none", None)):
            runner = await _ready_runner(broker, egress=egress, egress_hosts=egress_hosts)
            assert runner.sandbox.guarantees.version == 2
            assert EGRESS_DENY_BY_DEFAULT in runner.sandbox.guarantees.items
            admitted = decision(runner.execution_admission_candidate())
            assert admitted.status == "admitted", admitted.refusals
            assert network_claim(runner).state == "available"
            # No mode is Runtime's own brokered egress.
            brokered = decision(
                runner.execution_admission_candidate(),
                ExecutionRequirements.untrusted(network_access="brokered_egress"),
            )
            assert {refusal.capability for refusal in brokered.refusals} == {"brokered_egress"}
            await runner.close()
        # public_web never carries the item, and is never deny-by-default even if a
        # record claimed it.
        runner = await _ready_runner(broker, egress="public_web")
        assert EGRESS_DENY_BY_DEFAULT not in runner.sandbox.guarantees.items
        refused = decision(runner.execution_admission_candidate())
        assert {refusal.capability for refusal in refused.refusals} == {"deny_by_default_network"}
        await runner.close()
        broker.guarantees = (*GUARANTEES_V1, EGRESS_DENY_BY_DEFAULT)
        runner = await _ready_runner(broker, egress="public_web")
        assert network_claim(runner).reason_code == "sandbox_egress_public_web"
        await runner.close()
        # A version 2 record without the egress item fails closed, even for none.
        for egress in ("allowlist", "none"):
            claims = sandbox_capability_claims(
                SandboxGuarantees(version=2, items=GUARANTEES_V1),
                egress_mode=egress,
                declared=False,
            )
            (claim,) = [item for item in claims if item.capability == "deny_by_default_network"]
            assert (claim.state, claim.reason_code) == ("unsupported", "sandbox_guarantee_missing")
        # A version 1 record keeps today's rule: none is deny-by-default, and an
        # allowlist (an older Cloud API replica) fails closed.
        broker.guarantees_version = 1
        runner = await _ready_runner(broker, egress="none")
        assert decision(runner.execution_admission_candidate()).status == "admitted"
        await runner.close()
        runner = await _ready_runner(broker, egress="allowlist", egress_hosts=hosts)
        refused = decision(runner.execution_admission_candidate())
        assert {refusal.capability for refusal in refused.refusals} == {"deny_by_default_network"}
        assert network_claim(runner).reason_code == "sandbox_guarantee_missing"
        await runner.close()

    asyncio.run(scenario())


def test_executable_admission_probes_the_exact_sandbox(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        requirements = ExecutionRequirements(required_executables=("cayu-missing-tool", "sh"))
        observer = runner.execution_admission_observer(requirements)
        declared = observer.snapshot()
        assert {item.state for item in declared.evidence.tool_requirements.executables} == {
            "declared"
        }
        candidate = await observer.collect()
        states = {
            item.executable: item.state for item in candidate.evidence.tool_requirements.executables
        }
        assert states == {"sh": "live_verified", "cayu-missing-tool": "unavailable"}
        assert observer.snapshot() == candidate
        await runner.close()

    asyncio.run(scenario())


# Refusals ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code", "error"),
    [
        (429, "sandbox_quota_exceeded", SandboxQuotaExceededError),
        (403, "scope_required", SandboxBrokerRefusedError),
        (422, "egress_mode_unsupported", SandboxBrokerRefusedError),
        (503, "sandboxes_unavailable", SandboxBrokerRefusedError),
    ],
)
def test_broker_refusals_reach_the_caller_and_leave_nothing(
    tmp_path: Path, status: int, code: str, error: type[Exception]
) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.refuse_create = (status, code)
    with pytest.raises(error) as raised:
        asyncio.run(_factory(broker, _project(tmp_path)).create(_request()))
    assert raised.value.code == code
    if isinstance(raised.value, SandboxQuotaExceededError):
        assert (raised.value.scope, raised.value.limit) == ("agent", 8)
    # A refusal with a contract code is not retried.
    assert [call for call in broker.calls if call[0] == "POST"] == [("POST", "/")]
    assert broker.live() == []


def test_unknown_image_and_failed_allocation_are_refused(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    with pytest.raises(SandboxBrokerRefusedError, match="image_unknown"):
        asyncio.run(_factory(broker, _project(tmp_path), image="gpu").create(_request()))
    broker.fail_allocation = True
    with pytest.raises(SandboxAllocationFailedError, match="allocation_failed"):
        asyncio.run(_factory(broker, tmp_path / "source").create(_request()))


def test_errors_without_a_contract_code_are_judged_by_status(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    failures = {"left": 2}

    async def flaky(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and failures["left"]:
            failures["left"] -= 1
            # The broker applied the create, but the answer was lost.
            await broker.handle(request)
            raise httpx.ReadTimeout("lost", request=request)
        return await broker.transport(request)

    async def scenario():
        client = SandboxBrokerClient(
            SandboxBrokerSettings(api_url=API_URL, application_id=APPLICATION, token=TOKEN),
            transport=httpx.MockTransport(flaky),
        )
        record = await client.create(
            idempotency_key="same", image="coding-python", egress_mode="none", lease_seconds=60
        )
        assert record.id == "sbx_0001"
        broker.unknown_errors = 2
        assert (await client.get(record.id)).id == record.id
        broker.unknown_errors = 10
        with pytest.raises(SandboxBrokerUnavailableError):
            await client.get(record.id)
        broker.unknown_errors = 0
        # Without a contract code, 408, 429 and 5xx are retried ...
        broker.bare_statuses = [429, 408, 503]
        assert (await client.get(record.id)).id == record.id
        # ... and every other 4xx is terminal.
        for status, error in ((401, SandboxBrokerRefusedError), (404, SandboxNotFoundError)):
            broker.bare_statuses = [status]
            before = len(broker.requests)
            with pytest.raises(error) as raised:
                await client.get(record.id)
            assert raised.value.status == status and raised.value.code is None
            assert len(broker.requests) == before + 1

    asyncio.run(scenario())
    assert len(broker.sandboxes) == 1


# Leases ----------------------------------------------------------------------


def test_lease_expiry_stops_the_runner(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    lost: list[SandboxLeaseLostError] = []

    async def scenario():
        runner = await _ready_runner(broker, renew_every_s=0.05, on_lost=lost.append)
        await asyncio.sleep(0.12)
        assert broker.renewals >= 1
        broker.expire(runner.sandbox_id)
        await asyncio.sleep(0.15)
        assert lost and runner.lease_keeper.lost is not None
        with pytest.raises(SandboxLeaseLostError):
            await runner.exec(ExecCommand.process("true"))
        await runner.close()

    asyncio.run(scenario())
    assert len(lost) == 1


def test_the_first_renewal_follows_the_remaining_lease(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        # Only 0.3 s of a 60 s lease is left when the keeper starts.
        sandbox = runner.sandbox.model_copy(
            update={"lease_expires_at": datetime.now(UTC) + timedelta(seconds=0.3)}
        )
        keeper = SandboxLeaseKeeper(broker.client(), sandbox, lease_seconds=60).start()
        await asyncio.sleep(0.25)
        assert keeper.renewals == 1 and keeper.lost is None
        await keeper.stop()
        await runner.close()

    asyncio.run(scenario())


def test_a_renewal_unanswered_when_the_lease_runs_out_loses_the_lease(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    broker = FakeBroker(tmp_path / "cloud")
    lost: list[SandboxLeaseLostError] = []

    async def hanging_renewal(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/renew"):
            await asyncio.sleep(30)
        return await broker.transport(request)

    async def scenario():
        runner = await _ready_runner(broker)
        sandbox = runner.sandbox.model_copy(
            update={"lease_expires_at": datetime.now(UTC) + timedelta(seconds=0.3)}
        )
        client = SandboxBrokerClient(
            SandboxBrokerSettings(api_url=API_URL, application_id=APPLICATION, token=TOKEN),
            transport=httpx.MockTransport(hanging_renewal),
        )
        keeper = SandboxLeaseKeeper(client, sandbox, lease_seconds=60, on_lost=lost.append).start()
        # The renewal is still in flight, well inside the request timeout, when
        # the lease runs out: the keeper doesn't wait for it.
        await asyncio.sleep(0.6)
        assert keeper.lost is not None and not keeper.running
        assert "before its renewal was answered" in str(keeper.lost)
        with pytest.raises(SandboxLeaseLostError):
            keeper.check()
        await keeper.stop()
        await client.aclose()
        await runner.close()

    asyncio.run(scenario())
    assert len(lost) == 1


def test_a_lease_extended_while_a_renewal_hangs_is_not_lost(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    broker = FakeBroker(tmp_path / "cloud")
    renewals = {"sent": 0}

    async def first_renewal_hangs(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/renew"):
            renewals["sent"] += 1
            if renewals["sent"] == 1:
                await asyncio.sleep(30)
        return await broker.transport(request)

    async def scenario():
        runner = await _ready_runner(broker)
        sandbox = runner.sandbox.model_copy(
            update={"lease_expires_at": datetime.now(UTC) + timedelta(seconds=0.3)}
        )
        client = SandboxBrokerClient(
            SandboxBrokerSettings(api_url=API_URL, application_id=APPLICATION, token=TOKEN),
            transport=httpx.MockTransport(first_renewal_hangs),
        )
        keeper = SandboxLeaseKeeper(client, sandbox, lease_seconds=60).start()
        await asyncio.sleep(0.15)
        # The scheduled renewal hangs; an explicit one extends the lease.
        await keeper.renew_now()
        await asyncio.sleep(0.4)
        # Past the old expiry the keeper renews against the new one instead.
        assert keeper.lost is None and keeper.running
        assert renewals["sent"] == 3 and keeper.renewals == 2
        await keeper.stop()
        await client.aclose()
        await runner.close()

    asyncio.run(scenario())


def test_the_factory_reserves_its_own_labels(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    eight = {f"k{index}": "v" for index in range(8)}

    async def create_and_release():
        result = await _factory(broker, _project(tmp_path), labels=eight).create(_request())
        await result.release(None)

    asyncio.run(create_and_release())
    (sandbox,) = broker.sandboxes.values()
    assert len(sandbox.body["labels"]) == 10
    with pytest.raises(ValueError, match="at most 8"):
        _factory(broker, tmp_path / "source", labels={**eight, "k8": "v"})
    with pytest.raises(ValueError, match="set by the factory"):
        _factory(broker, tmp_path / "source", labels={"cayu.session": "mine"})


def test_a_lapsed_lease_found_by_a_command_closes_the_runner(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        broker.expire(runner.sandbox_id)
        with pytest.raises(SandboxLeaseLostError):
            await runner.exec(ExecCommand.process("true"))
        # The sandbox never becomes ready again, so the runner can't be reopened,
        # and retained admission evidence must not admit it.
        assert runner.lifecycle_state == "poisoned"
        assert _terminal_runner_state(runner) == "poisoned"
        assert runner.loss_reason == "lease_lost"
        with pytest.raises(RuntimeError, match="poisoned"):
            runner.reopen_exec()
        await runner.close()

    asyncio.run(scenario())


def test_a_lease_the_keeper_lost_poisons_the_runner(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker, renew_every_s=0.05)
        broker.expire(runner.sandbox_id)
        while runner.lease_keeper.lost is None:
            await asyncio.sleep(0.01)
        with pytest.raises(SandboxLeaseLostError):
            await runner.exec(ExecCommand.process("true"))
        assert _terminal_runner_state(runner) == "poisoned"
        assert runner.loss_reason == "lease_lost"
        await runner.close()

    asyncio.run(scenario())


# Cleanup ---------------------------------------------------------------------


def test_close_destroys_once_and_a_failed_destroy_leaves_the_lease_to_lapse(
    tmp_path: Path,
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker, renew_every_s=0.05)
        await runner.close()
        await runner.close()
        assert [call for call in broker.calls if call[0] == "DELETE"] == [
            ("DELETE", f"/{runner.sandbox_id}")
        ]
        second = await _ready_runner(broker, renew_every_s=0.05)
        broker.fail_destroy = True
        with pytest.raises(SandboxBrokerUnavailableError):
            await second.close()
        # Renewal stopped, so the broker destroys it when the lease lapses.
        assert not second.lease_keeper.running and not second.is_closed
        broker.fail_destroy = False
        await second.close()
        assert second.is_closed

    asyncio.run(scenario())
    assert broker.live() == []


def test_failed_setup_destroys_the_sandbox(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    factory = _factory(broker, _project(tmp_path), setup_commands=["exit 3"])
    with pytest.raises(RuntimeError, match="exit code 3"):
        asyncio.run(factory.create(_request()))
    (sandbox,) = broker.sandboxes.values()
    assert sandbox.status == "destroyed"


def test_the_client_covers_the_read_and_archive_routes(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        client = broker.client()
        (image,) = await client.list_images()
        assert image.name == "coding-python" and image.guarantees.has("unprivileged_commands")
        runner = await _ready_runner(broker)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:") as archive:
            info = tarfile.TarInfo("notes/a.txt")
            info.size = 2
            archive.addfile(info, io.BytesIO(b"hi"))
        await runner.put_archive(buffer.getvalue())
        result = await client.get_archive(runner.sandbox_id, paths=["notes", "missing.txt"])
        assert result.missing_paths == ("missing.txt",)
        with tarfile.open(fileobj=io.BytesIO(result.data), mode="r:") as archive:
            assert archive.extractfile("notes/a.txt").read() == b"hi"
        sandboxes, cursor = await client.list_sandboxes(status="ready")
        assert [record.id for record in sandboxes] == [runner.sandbox_id] and cursor is None
        assert await client.events(runner.sandbox_id) == []
        with pytest.raises(ValueError, match="inside /workspace"):
            await client.get_archive(runner.sandbox_id, paths=["../etc"])
        await runner.close()
        await client.aclose()

    asyncio.run(scenario())


# No cloud credentials -------------------------------------------------------------


def test_the_agent_needs_no_cloud_provider_credential(tmp_path: Path, monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith(("AWS_", "GOOGLE_", "AZURE_", "E2B_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing"))
    monkeypatch.setitem(sys.modules, "boto3", None)
    monkeypatch.setitem(sys.modules, "botocore", None)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        result = await _factory(broker, _project(tmp_path)).create(_request())
        await result.release(None)

    asyncio.run(scenario())
    # Only the broker credential ever leaves the Agent.
    assert {request.headers["authorization"] for request in broker.requests} == {f"Bearer {TOKEN}"}
    assert all(request.url.host == "cloud.example.test" for request in broker.requests)
    # Importing the package loads no provider SDK or provider runner.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, cayu.sandbox_broker; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in {'boto3', 'botocore', 'e2b'}"
            " or m == 'cayu.runners.aws_lambda_microvm'))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert probe.stdout.strip() == "[]", probe.stdout


@pytest.mark.parametrize("above_staged_cap", [False, True])
def test_staged_selection_preserves_archive_boundaries(tmp_path, above_staged_cap):
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {"max_bytes": 16 * 1024 * 1024, "sandbox_max_bytes": 64 * 1024 * 1024}
    files = (
        {f"file{i}": b"x" * (1024 * 1024) for i in range(20)}
        if above_staged_cap
        else {"near-limit": b"x" * (8 * 1024 * 1024 - 4096)}
    )

    async def scenario():
        runner = await _ready_runner(broker)
        archive = _tar(files)
        await BrokeredSandboxWorkspace(runner).write_tar_stream(
            io.BytesIO(archive), archive_bytes=len(archive)
        )
        root = broker.sandboxes[runner.sandbox_id].root
        for name, content in files.items():
            assert (root / name).read_bytes() == content
        uploads = [request for request in broker.storage_requests if request.method == "PUT"]
        assert len(uploads) == (0 if above_staged_cap else 1)
        await runner.close()

    asyncio.run(scenario())


def test_failed_staged_copy_size_uses_archive_too_large_error(tmp_path):
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {"max_bytes": 64 * 1024, "sandbox_max_bytes": 1024 * 1024}

    async def scenario():
        runner = await _ready_runner(broker)
        (broker.sandboxes[runner.sandbox_id].root / "large").write_bytes(b"x" * (200 * 1024))
        with pytest.raises(client_module.SandboxArchiveTooLargeError) as caught:
            await runner.copy_out(io.BytesIO(), paths=["large"], max_bytes=64 * 1024)
        assert caught.value.code == "copy_too_large"
        await runner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("seekable", [True, False])
@pytest.mark.parametrize("separator", [b"\0" * 512, b"corrupt" + b"\0" * 505])
def test_staged_archive_rejects_unparsed_trailing_members(tmp_path, separator, seekable):
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {"max_bytes": 1024 * 1024, "sandbox_max_bytes": 2 * 1024 * 1024}
    # Retain just the first file header and padded payload, then hide a member
    # behind a terminator or a malformed header. GNU tar can recover it.
    first = _tar({"ok.txt": b"ok"})[:1024]
    link = tarfile.TarInfo("link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc"
    raw = first + separator + link.tobuf() + b"\0" * 1024

    async def scenario():
        runner = await _ready_runner(broker)
        workspace = BrokeredSandboxWorkspace(runner)
        source = io.BytesIO(raw) if seekable else io.BufferedReader(_Unseekable(raw))
        with pytest.raises(ValueError, match="unvalidated trailing"):
            await workspace._write_staged(source, len(raw), runner.staged_copies)
        assert not broker.storage_requests
        assert not (broker.sandboxes[runner.sandbox_id].root / "link").exists()
        await runner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["cancel", "poll", "timeout"])
def test_abandoned_staged_extraction_fences_runner(tmp_path, monkeypatch, failure):
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {"max_bytes": 1024 * 1024, "sandbox_max_bytes": 2 * 1024 * 1024}
    monkeypatch.setattr(broker, "_copy_in", lambda sandbox, copy: copy.update(status="running"))
    data = _tar({"result": b"new"})

    async def scenario():
        runner = await _ready_runner(broker)
        polling = asyncio.Event()

        async def get_copy(*args, **kwargs):
            polling.set()
            if failure == "poll":
                raise SandboxBrokerUnavailableError("poll unavailable")
            await asyncio.Event().wait()

        monkeypatch.setattr(runner.broker, "get_copy", get_copy)
        if failure == "timeout":
            monkeypatch.setattr(client_module, "_COPY_TIMEOUT_FLOOR_SECONDS", -1000)
        if failure == "poll":
            # An outage is polled through until the copy's own deadline, then poisons.
            monkeypatch.setattr(client_module, "_COPY_TIMEOUT_FLOOR_SECONDS", 0.3)
            monkeypatch.setattr(client_module, "_RETRY_DELAYS_SECONDS", (0.01,))
            monkeypatch.setattr(client_module, "_OUTAGE_RETRY_SECONDS", 0.05)
        copying = asyncio.create_task(
            runner.copy_in(
                io.BytesIO(data), size=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
        )
        if failure == "cancel":
            await polling.wait()
            copying.cancel()
        with pytest.raises((asyncio.CancelledError, SandboxBrokerUnavailableError)):
            await copying
        assert runner.lifecycle_state == "poisoned"
        with pytest.raises(RuntimeError):
            await runner.exec(ExecCommand.process("echo", "unsafe"))
        with pytest.raises(RuntimeError):
            await runner.copy_in(
                io.BytesIO(data), size=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
        assert not broker.started
        await runner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["upload", "terminal"])
def test_settled_or_undispatched_copy_failure_keeps_runner_reusable(tmp_path, monkeypatch, phase):
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {"max_bytes": 1024 * 1024, "sandbox_max_bytes": 2 * 1024 * 1024}
    data = _tar({"result": b"new"})
    if phase == "terminal":
        monkeypatch.setattr(
            broker,
            "_copy_in",
            lambda sandbox, copy: copy.update(
                status="failed", error={"code": "copy_digest_mismatch", "message": ""}
            ),
        )

    async def scenario():
        runner = await _ready_runner(broker)
        if phase == "upload":

            async def failed_upload(*args, **kwargs):
                raise SandboxBrokerUnavailableError("upload unavailable")

            monkeypatch.setattr(runner.broker, "upload_staged", failed_upload)
        with pytest.raises(client_module.SandboxBrokerError):
            await runner.copy_in(
                io.BytesIO(data), size=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
        result = await runner.exec(ExecCommand.process("echo", "safe"))
        assert result.stdout == "safe\n"
        await runner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("lost_first_response", [False, True])
def test_refused_copy_commit_only_fences_previously_uncertain_mutation(
    tmp_path, monkeypatch, lost_first_response
):
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {"max_bytes": 1024 * 1024, "sandbox_max_bytes": 2 * 1024 * 1024}
    data = _tar({"result": b"new"})

    async def scenario():
        runner = await _ready_runner(broker)
        attempts = 0

        async def refused_commit(*args, **kwargs):
            nonlocal attempts
            assert kwargs["retry"] is False
            attempts += 1
            if lost_first_response and attempts == 1:
                raise SandboxBrokerUnavailableError("response lost", outcome_unknown=True)
            raise SandboxBrokerRefusedError("copies unavailable", code="copies_unavailable")

        monkeypatch.setattr(runner.broker, "commit_copy", refused_commit)
        monkeypatch.setattr(client_module, "_RETRY_DELAYS_SECONDS", (0.0,))
        with pytest.raises(SandboxBrokerRefusedError):
            await runner.copy_in(
                io.BytesIO(data), size=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
        if lost_first_response:
            assert runner.lifecycle_state == "poisoned"
            with pytest.raises(RuntimeError):
                await runner.exec(ExecCommand.process("echo", "unsafe"))
        else:
            assert (await runner.exec(ExecCommand.process("echo", "safe"))).stdout == "safe\n"
        await runner.close()

    asyncio.run(scenario())


def test_cancelled_stdin_delivery_fences_later_payloads(tmp_path: Path):
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        process = await runner.interactive_process_capability().open_interactive_process(
            ExecCommand.process("cat"), timeout_s=60
        )
        broker.drop_stdio_responses = 10
        writing = asyncio.create_task(process.write(b"one\n"))
        while broker.drop_stdio_responses == 10:
            await asyncio.sleep(0.01)
        writing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writing
        broker.drop_stdio_responses = 0
        with pytest.raises(runner_module.SandboxBrokerError, match="uncertain"):
            await process.write(b"two\n")
        with pytest.raises(runner_module.SandboxBrokerError, match="uncertain"):
            await process.close_stdin()
        await process.close()
        assert not broker.running()
        await runner.close()

    asyncio.run(scenario())


def test_failed_start_lease_loss_retains_cleanup_handle(tmp_path: Path, monkeypatch):
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        capability = runner.interactive_process_capability()

        async def unconfirmed_start(*args, **kwargs):
            runner._exec_poisoned = True
            args[2].cleanup_required = True
            raise SandboxNotFoundError("lost")

        monkeypatch.setattr(runner, "_start", unconfirmed_start)
        with pytest.raises(SandboxLeaseLostError) as caught:
            await capability.open_interactive_process(ExecCommand.process("cat"), timeout_s=60)
        assert capability.failed_start_process(caught.value) is not None
        await runner.close()

    asyncio.run(scenario())


def test_busy_interactive_start_does_not_claim_another_commands_cleanup(tmp_path: Path):
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        capability = runner.interactive_process_capability()
        broker.busy_responses = 1000
        opening = asyncio.create_task(
            capability.open_interactive_process(ExecCommand.process("cat"), timeout_s=60)
        )
        while broker.busy_responses == 1000:
            await asyncio.sleep(0.01)
        runner._poison_exec("another command did not settle")
        with pytest.raises(RuntimeError) as caught:
            await opening
        assert capability.failed_start_process(caught.value) is None
        assert not broker.running()
        await runner.close()

    asyncio.run(scenario())


def test_interactive_recovery_deadline_starts_after_dispatch(tmp_path: Path, monkeypatch):
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        original = runner._start
        returned_at = []

        async def delayed(*args, **kwargs):
            await asyncio.sleep(0.05)
            result = await original(*args, **kwargs)
            returned_at.append(asyncio.get_running_loop().time())
            return result

        monkeypatch.setattr(runner, "_start", delayed)
        process = await runner.interactive_process_capability().open_interactive_process(
            ExecCommand.process("cat"), timeout_s=60
        )
        assert process._deadline >= returned_at[0] + 60 + runner_module._POLL_RECOVERY_GRACE_SECONDS
        await process.close()
        await runner.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("earlier_uncertainty", [False, True])
def test_failed_start_cleanup_observes_late_refusal(tmp_path, monkeypatch, earlier_uncertainty):
    broker = FakeBroker(tmp_path / "cloud")
    monkeypatch.setattr(runner_module, "_CANCEL_SETTLE_SECONDS", 0)

    async def scenario():
        # The start never answers here, so its wait is the request timeout.
        runner = await _ready_runner(broker, client_settings={"request_timeout_s": 0.05})
        pending = asyncio.Event()
        release = asyncio.Event()
        attempts = 0

        async def late_refusal(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if earlier_uncertainty and attempts == 1:
                raise SandboxBrokerUnavailableError("response lost", outcome_unknown=True)
            pending.set()
            await release.wait()
            raise client_module.SandboxBusyError("busy")

        monkeypatch.setattr(runner.broker, "start_command", late_refusal)
        capability = runner.interactive_process_capability()
        opening = asyncio.create_task(
            capability.open_interactive_process(ExecCommand.process("cat"), timeout_s=60)
        )
        await pending.wait()
        opening.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await opening
        process = capability.failed_start_process(caught.value)
        assert process is not None
        # An unresolved attempt's 404 cannot prove cleanup.
        with pytest.raises(client_module.SandboxBrokerError):
            await process.close()
        release.set()
        await asyncio.sleep(0)
        if earlier_uncertainty:
            # A refused replay does not settle an earlier lost response.
            with pytest.raises(client_module.SandboxBrokerError):
                await process.close()
        else:
            await process.close()
            await process.close()
        await runner.close()

    asyncio.run(scenario())


# Robustness ----------------------------------------------------------------


def _outage_broker(tmp_path: Path, monkeypatch, *, failing: str, outages: int) -> FakeBroker:
    """A broker whose copy polls or commits answer 503 ``outages`` times, then recover."""

    monkeypatch.setattr(client_module, "_RETRY_DELAYS_SECONDS", (0.01, 0.01, 0.01))
    monkeypatch.setattr(client_module, "_OUTAGE_RETRY_SECONDS", 0.01)
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {"max_bytes": 1024 * 1024, "sandbox_max_bytes": 2 * 1024 * 1024}
    extract = broker._copy_in
    committed: list[tuple] = []
    if failing == "poll":
        # A committed copy stays pending in Cloud until a poll gets through.
        def pending(sandbox, copy):
            committed.append((sandbox, copy))
            copy.update(status="pending")

        monkeypatch.setattr(broker, "_copy_in", pending)
    transport = broker.transport
    remaining = outages

    async def flaky(request: httpx.Request) -> httpx.Response:
        nonlocal remaining
        parts = request.url.path.split("/")
        poll = request.method == "GET" and "copies" in parts[:-1]
        commit = request.method == "POST" and parts[-1] == "commit"
        if (poll if failing == "poll" else commit) and remaining:
            remaining -= 1
            return httpx.Response(503, text="bad gateway")
        if poll:
            while committed:
                extract(*committed.pop())
        return await transport(request)

    monkeypatch.setattr(broker, "transport", flaky)
    return broker


@pytest.mark.parametrize("failing", ["poll", "commit"])
def test_a_short_broker_outage_during_a_staged_copy_keeps_the_runner(
    tmp_path, monkeypatch, failing
):
    # More 503s than one request's quick retries take, then the broker answers again.
    broker = _outage_broker(tmp_path, monkeypatch, failing=failing, outages=6)
    data = _tar({"result": b"new"})

    async def scenario():
        runner = await _ready_runner(broker)
        copy = await runner.copy_in(
            io.BytesIO(data), size=len(data), sha256=hashlib.sha256(data).hexdigest()
        )
        assert copy.status == "done"
        assert (broker.sandboxes[runner.sandbox_id].root / "result").read_bytes() == b"new"
        assert runner.lifecycle_state == "reusable"
        assert (await runner.exec(ExecCommand.process("echo", "safe"))).stdout == "safe\n"
        await runner.close()

    asyncio.run(scenario())


def test_an_outage_past_the_copy_deadline_still_poisons(tmp_path, monkeypatch):
    broker = _outage_broker(tmp_path, monkeypatch, failing="poll", outages=10_000)
    monkeypatch.setattr(client_module, "_COPY_TIMEOUT_FLOOR_SECONDS", 0.2)
    data = _tar({"result": b"new"})

    async def scenario():
        runner = await _ready_runner(broker)
        with pytest.raises(SandboxBrokerUnavailableError):
            await runner.copy_in(
                io.BytesIO(data), size=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
        assert runner.lifecycle_state == "poisoned"
        await runner.close()

    asyncio.run(scenario())


def test_interactive_processes_need_a_broker_that_offers_them(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    # A Cloud broker from before interactive commands doesn't say it runs them,
    # and refuses the field.
    broker.interactive_commands = False

    async def scenario():
        runner = await _ready_runner(broker)
        assert runner.sandbox.interactive_commands is False
        assert runner.interactive_process_capability() is None
        with pytest.raises(ValueError, match="cannot open interactive processes") as refused:
            async with runner_agent_launch(runner=runner, cli=sys.executable, environment={}):
                pass
        # The fix is on the Cloud side, not a different runner.
        assert "advertise `interactive_commands`" in str(refused.value)
        assert "LambdaMicroVMRunner" not in str(refused.value)
        with pytest.raises(SandboxBrokerRefusedError) as refused:
            await runner.broker.start_command(runner.sandbox_id, "c-1", ["cat"], interactive=True)
        assert refused.value.code == "sandbox_invalid"
        await runner.close()
        broker.interactive_commands = True
        offered = await _ready_runner(broker)
        assert offered.interactive_process_capability() is not None
        await offered.close()

    asyncio.run(scenario())


def test_the_interactive_offer_tolerates_its_possible_shapes() -> None:
    base = {"id": "s", "image": "coding-python", "egress": {"mode": "none"}, "status": "ready"}
    record = client_module.SandboxRecord.model_validate
    assert record(base).interactive_commands is False
    assert record({**base, "interactive_commands": True}).interactive_commands is True
    assert record({**base, "interactive_commands": {"max_chunk_bytes": 1}}).interactive_commands
    for value in (False, None, "yes", 1):
        assert record({**base, "interactive_commands": value}).interactive_commands is False


def test_a_staged_copy_over_the_sandbox_budget_says_so(tmp_path, monkeypatch) -> None:
    limit = 64 * 1024
    monkeypatch.setattr(runner_module, "MAX_ARCHIVE_BYTES", limit)
    broker = FakeBroker(tmp_path / "cloud")
    broker.archive_limit = limit
    broker.staged_copies = {"max_bytes": 256 * 1024, "sandbox_max_bytes": 1024 * 1024}

    async def scenario():
        runner = await _ready_runner(broker)
        workspace = BrokeredSandboxWorkspace(runner)
        (broker.sandboxes[runner.sandbox_id].root / "big.bin").write_bytes(b"x" * (128 * 1024))
        # The sandbox refuses the copy although the file is within max_file_bytes.
        monkeypatch.setattr(
            broker,
            "_copy_out",
            lambda sandbox, copy: copy.update(
                status="failed", error={"code": "copy_too_large", "message": ""}
            ),
        )
        with pytest.raises(RuntimeError, match="staged copies allow") as caught:
            await workspace.read_tar_stream(["big.bin"], io.BytesIO(), max_file_bytes=200 * 1024)
        assert "max_file_bytes" not in str(caught.value)
        await runner.close()

    asyncio.run(scenario())


class _Unseekable(io.RawIOBase):
    def __init__(self, data: bytes) -> None:
        self._source = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        chunk = self._source.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)


@pytest.mark.parametrize("seekable", [True, False])
def test_a_staged_copy_in_spools_only_a_source_that_cannot_seek(tmp_path, monkeypatch, seekable):
    broker = FakeBroker(tmp_path / "cloud")
    broker.staged_copies = {"max_bytes": 1024 * 1024, "sandbox_max_bytes": 2 * 1024 * 1024}
    data = _tar({"a.txt": b"a" * 1000, "b.txt": b"b"})
    spools = []
    temporary_file = runner_module.tempfile.TemporaryFile

    def counted_temporary_file(*args, **kwargs):
        spools.append(1)
        return temporary_file(*args, **kwargs)

    monkeypatch.setattr(runner_module.tempfile, "TemporaryFile", counted_temporary_file)

    async def scenario():
        runner = await _ready_runner(broker)
        source = io.BytesIO(data) if seekable else io.BufferedReader(_Unseekable(data))
        await BrokeredSandboxWorkspace(runner)._write_staged(
            source, len(data), runner.staged_copies
        )
        assert (broker.sandboxes[runner.sandbox_id].root / "a.txt").read_bytes() == b"a" * 1000
        (upload,) = [request for request in broker.storage_requests if request.method == "PUT"]
        assert upload.content == data
        assert len(spools) == (0 if seekable else 1)
        await runner.close()

    asyncio.run(scenario())


class _RecordingRunner:
    """Answers every baseline step, recording each one's timeout."""

    def __init__(self) -> None:
        self.timeouts: dict[str, int] = {}

    async def exec_system(self, command, *, timeout_s, **_options):
        argv = command.argv
        step = "commit" if "commit" in argv else argv[1]
        self.timeouts.setdefault(step, timeout_s)
        # The filter check finds nothing configured.
        exit_code = 1 if "--get-regexp" in argv else 0
        return runner_module.ExecResult(stdout="", stderr="", exit_code=exit_code)


def test_the_git_baseline_gives_a_large_tree_time_to_commit() -> None:
    from cayu.environments import _ephemeral_git as ephemeral_git

    def timeouts(**options) -> dict[str, int]:
        runner = _RecordingRunner()
        asyncio.run(initialize_ephemeral_git_baseline(runner, **options))
        return runner.timeouts

    small = timeouts()
    assert small["add"] == small["commit"] == small["init"] == 60
    # A 1 GiB tree of 10,000 files gets minutes, not one.
    copied = {"sync_binding": {"copied_bytes": 1024**3, "copied_files": 10_000}}
    tree_bytes, tree_files = ephemeral_git.sync_bound_tree_size(copied)
    large = timeouts(tree_bytes=tree_bytes, tree_files=tree_files)
    assert large["add"] == large["commit"] == 60 + 256 + 100
    assert large["init"] == large["config"] == 60
    huge = timeouts(tree_bytes=64 * 1024**3, tree_files=10_000_000)
    assert huge["add"] == ephemeral_git.MAX_BASELINE_STEP_TIMEOUT_SECONDS
    assert timeouts(tree_bytes=1024**3, timeout_s=90)["commit"] == 90
    with pytest.raises(ValueError):
        timeouts(timeout_s=0)
    assert ephemeral_git.sync_bound_tree_size({}) == (None, None)
