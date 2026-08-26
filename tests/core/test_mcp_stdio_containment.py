from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

import cayu.mcp._stdio_process as stdio_process
import cayu.mcp.stdio as stdio_module
from cayu import (
    McpProtocolError,
    McpServerSpec,
    McpToolset,
    McpTransportLimits,
    ResolvedSecret,
    SecretRef,
    StdioMcpClient,
    StdioMcpProcessLifetime,
)
from cayu.mcp import _stdio_containment as containment
from cayu.mcp._stdio_process import ContainedStdioMcpProcess
from cayu.mcp.stdio import StdioMcpSession

_FAKE_SERVER = Path(__file__).parents[1] / "fixtures" / "fake_mcp_server.py"


class _ForbiddenSecretResolver:
    def __init__(self) -> None:
        self.calls = 0

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> Any:
        del ref, scope
        self.calls += 1
        raise AssertionError("secret resolution must not run")


class _RecordingSecretResolver:
    def __init__(self) -> None:
        self.calls = 0

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        del scope
        self.calls += 1
        return ResolvedSecret(name=ref.name, value=SecretStr("resolved-token"))


def _server_spec(*, secret: bool = False) -> McpServerSpec:
    return McpServerSpec(
        name="stdio-containment",
        command=[sys.executable, str(_FAKE_SERVER)],
        secret_env={"TOKEN": SecretRef(name="token")} if secret else {},
    )


def _prepared_rendezvous(identity: str) -> stdio_process._PreparedContainmentRendezvous:
    return stdio_process._PreparedContainmentRendezvous(
        identity=identity,
        owner_pid=os.getpid(),
        authority=stdio_process._CONTAINMENT_RENDEZVOUS_AUTHORITY,
    )


def test_containment_rendezvous_identity_binds_connection_and_exact_command() -> None:
    base = _server_spec().model_copy(update={"connection_id": "logical-connection"})
    same = base.model_copy(deep=True)
    changed_connection = base.model_copy(update={"connection_id": "other-connection"})
    assert base.command is not None
    changed_command = base.model_copy(update={"command": [*base.command, "--changed"]})

    identity = stdio_module._stdio_containment_rendezvous_identity(base)

    assert stdio_module._stdio_containment_rendezvous_identity(same) == identity
    assert stdio_module._stdio_containment_rendezvous_identity(changed_connection) != identity
    assert stdio_module._stdio_containment_rendezvous_identity(changed_command) != identity


def test_containment_rendezvous_address_is_scoped_to_effective_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = "a" * 64
    monkeypatch.setattr(containment.os, "geteuid", lambda: 1001)
    first = containment._rendezvous_address(identity)
    monkeypatch.setattr(containment.os, "geteuid", lambda: 1002)
    second = containment._rendezvous_address(identity)

    assert first != second
    assert first.endswith(identity.encode("ascii"))
    assert second.endswith(identity.encode("ascii"))


def test_contained_process_rejects_mismatched_prepared_rendezvous_authority() -> None:
    expected_identity = stdio_module._stdio_containment_rendezvous_identity(
        _server_spec().model_copy(update={"connection_id": "expected"})
    )
    other_identity = stdio_module._stdio_containment_rendezvous_identity(
        _server_spec().model_copy(update={"connection_id": "other"})
    )
    prepared = _prepared_rendezvous(other_identity)
    proof = stdio_process._ContainmentPreflightProof(
        owner_pid=os.getpid(),
        authority=stdio_process._PARENT_DEATH_CONTAINMENT_PREFLIGHT_AUTHORITY,
    )

    with pytest.raises(RuntimeError, match="did not match the command"):
        asyncio.run(
            stdio_process.create_contained_stdio_mcp_process(
                "server",
                env={},
                limit=1024,
                startup_timeout_s=0.1,
                term_timeout_s=0.1,
                kill_timeout_s=0.1,
                _preflight_proof=proof,
                _rendezvous_identity=expected_identity,
                _prepared_rendezvous=prepared,
            )
        )

    with pytest.raises(RuntimeError, match="invalid or stale"):
        prepared.consume()


def test_stdio_client_defaults_to_parent_death_containment_with_exact_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stdio_process,
        "_PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED",
        None,
    )
    monkeypatch.setattr(
        stdio_process,
        "_PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID",
        None,
    )
    client = StdioMcpClient()

    assert client.process_lifetime is StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
    evidence = client.process_capability_evidence
    configured_state = (
        "declared"
        if stdio_process.stdio_mcp_parent_death_containment_platform_candidate()
        else "unsupported"
    )
    assert evidence.state_for("graceful_cleanup") == configured_state
    assert evidence.state_for("parent_death_containment") == configured_state
    assert evidence.state_for("persistent_detached") == "unsupported"


@pytest.mark.parametrize(
    ("preflight_result", "expected_state"),
    [(None, "declared"), (True, "available"), (False, "unsupported")],
)
def test_containment_evidence_requires_complete_preflight(
    monkeypatch: pytest.MonkeyPatch,
    preflight_result: bool | None,
    expected_state: str,
) -> None:
    monkeypatch.setattr(
        stdio_process,
        "stdio_mcp_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setattr(
        stdio_process,
        "_PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED",
        preflight_result,
    )
    monkeypatch.setattr(
        stdio_process,
        "_PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID",
        os.getpid() if preflight_result is not None else None,
    )

    evidence = stdio_process.stdio_mcp_process_capability_evidence(
        StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
    )

    for capability in ("graceful_cleanup", "parent_death_containment"):
        assert evidence.state_for(capability) == expected_state
        claim = next(claim for claim in evidence.claims if claim.capability == capability)
        if expected_state == "available":
            assert claim.proof_source == "process_preflight"


def test_containment_evidence_rejects_complete_tree_claims_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stdio_process,
        "stdio_mcp_parent_death_containment_platform_candidate",
        lambda: False,
    )

    evidence = stdio_process.stdio_mcp_process_capability_evidence(
        StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
    )

    assert evidence.state_for("graceful_cleanup") == "unsupported"
    assert evidence.state_for("parent_death_containment") == "unsupported"


@pytest.mark.parametrize(
    ("lifetime", "graceful_state", "parent_state", "detached_state"),
    [
        (
            StdioMcpProcessLifetime.GRACEFUL_CLEANUP,
            "unsupported",
            "unsupported",
            "unsupported",
        ),
        (
            StdioMcpProcessLifetime.PERSISTENT_DETACHED,
            "unsupported",
            "unsupported",
            "available" if os.name == "posix" else "unsupported",
        ),
    ],
)
def test_stdio_client_reports_explicit_weaker_lifecycle(
    lifetime: StdioMcpProcessLifetime,
    graceful_state: str,
    parent_state: str,
    detached_state: str,
) -> None:
    evidence = StdioMcpClient(process_lifetime=lifetime).process_capability_evidence

    assert evidence.state_for("graceful_cleanup") == graceful_state
    assert evidence.state_for("parent_death_containment") == parent_state
    assert evidence.state_for("persistent_detached") == detached_state


def test_unsupported_containment_is_rejected_before_secret_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _ForbiddenSecretResolver()

    def unsupported(_lifetime: StdioMcpProcessLifetime) -> None:
        raise RuntimeError("containment unavailable")

    monkeypatch.setattr(stdio_module, "validate_containment_platform", unsupported)
    client = StdioMcpClient(secret_resolver=resolver)

    with pytest.raises(RuntimeError, match="containment unavailable"):
        asyncio.run(client.connect(_server_spec(secret=True)))

    assert resolver.calls == 0


def test_failed_exact_preflight_is_rejected_before_secret_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _ForbiddenSecretResolver()
    process_calls = 0

    async def unavailable(_timeout_s: float) -> None:
        raise RuntimeError("containment prerequisites unavailable")

    async def forbidden_process(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal process_calls
        process_calls += 1
        raise AssertionError("server process creation must not run")

    monkeypatch.setattr(stdio_module, "validate_containment_platform", lambda _value: None)
    monkeypatch.setattr(
        stdio_module,
        "preflight_stdio_mcp_parent_death_containment",
        unavailable,
    )
    monkeypatch.setattr(
        stdio_module,
        "create_contained_stdio_mcp_process",
        forbidden_process,
    )

    with pytest.raises(RuntimeError, match="prerequisites unavailable"):
        asyncio.run(StdioMcpClient(secret_resolver=resolver).connect(_server_spec(secret=True)))

    assert resolver.calls == 0
    assert process_calls == 0


def test_failed_rendezvous_preflight_is_rejected_before_secret_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _ForbiddenSecretResolver()
    process_calls = 0

    async def available(
        _timeout_s: float,
    ) -> stdio_process._ContainmentPreflightProof:
        return stdio_process._ContainmentPreflightProof(
            owner_pid=os.getpid(),
            authority=stdio_process._PARENT_DEATH_CONTAINMENT_PREFLIGHT_AUTHORITY,
        )

    def unavailable(_identity: str) -> None:
        raise RuntimeError("containment rendezvous unavailable")

    async def forbidden_process(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal process_calls
        process_calls += 1
        raise AssertionError("server process creation must not run")

    monkeypatch.setattr(stdio_module, "validate_containment_platform", lambda _value: None)
    monkeypatch.setattr(
        stdio_module,
        "preflight_stdio_mcp_parent_death_containment",
        available,
    )
    monkeypatch.setattr(
        stdio_module,
        "_prepare_stdio_mcp_containment_rendezvous",
        unavailable,
    )
    monkeypatch.setattr(
        stdio_module,
        "create_contained_stdio_mcp_process",
        forbidden_process,
    )

    with pytest.raises(RuntimeError, match="rendezvous unavailable"):
        asyncio.run(StdioMcpClient(secret_resolver=resolver).connect(_server_spec(secret=True)))

    assert resolver.calls == 0
    assert process_calls == 0


def test_ignored_child_exit_signal_is_rejected_before_secret_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _ForbiddenSecretResolver()
    process_calls = 0

    async def forbidden_process(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal process_calls
        process_calls += 1
        raise AssertionError("containment helper creation must not run")

    monkeypatch.setattr(
        stdio_process,
        "stdio_mcp_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setattr(
        stdio_process.signal,
        "getsignal",
        lambda _signal_number: stdio_process.signal.SIG_IGN,
    )
    monkeypatch.setattr(stdio_process.asyncio, "create_subprocess_exec", forbidden_process)

    with pytest.raises(RuntimeError, match="waitable child-process exits"):
        asyncio.run(StdioMcpClient(secret_resolver=resolver).connect(_server_spec(secret=True)))

    assert resolver.calls == 0
    assert process_calls == 0


def test_connect_revalidates_mutated_lifetime_before_secret_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = _ForbiddenSecretResolver()
    observed: list[StdioMcpProcessLifetime] = []

    def reject_after_snapshot(lifetime: StdioMcpProcessLifetime) -> None:
        observed.append(lifetime)
        raise RuntimeError("captured lifecycle")

    monkeypatch.setattr(stdio_module, "validate_containment_platform", reject_after_snapshot)
    client = StdioMcpClient(
        process_lifetime=StdioMcpProcessLifetime.GRACEFUL_CLEANUP,
        secret_resolver=resolver,
    )
    client.process_lifetime = "parent_death_containment"  # ty: ignore[invalid-assignment]

    with pytest.raises(RuntimeError, match="captured lifecycle"):
        asyncio.run(client.connect(_server_spec(secret=True)))

    assert observed == [StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT]
    assert resolver.calls == 0


def test_connect_rejects_invalid_mutated_containment_timeout_before_secret_lookup() -> None:
    resolver = _ForbiddenSecretResolver()
    client = StdioMcpClient(
        process_lifetime=StdioMcpProcessLifetime.GRACEFUL_CLEANUP,
        secret_resolver=resolver,
    )
    client.containment_term_timeout_s = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        asyncio.run(client.connect(_server_spec(secret=True)))

    assert resolver.calls == 0


@pytest.mark.parametrize(
    "field_name",
    (
        "request_timeout_s",
        "write_timeout_s",
        "graceful_shutdown_timeout_s",
        "cancellation_notification_timeout_s",
    ),
)
@pytest.mark.parametrize("invalid_timeout", (float("nan"), float("inf")))
def test_connect_rejects_nonfinite_client_timeout_before_any_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    invalid_timeout: float,
) -> None:
    resolver = _RecordingSecretResolver()
    preflight_calls = 0
    process_calls = 0

    async def forbidden_preflight(
        _timeout_s: float,
    ) -> stdio_process._ContainmentPreflightProof:
        nonlocal preflight_calls
        preflight_calls += 1
        raise AssertionError("containment preflight must not run")

    async def forbidden_process(*_argv: str, **_kwargs: Any) -> Any:
        nonlocal process_calls
        process_calls += 1
        raise AssertionError("process creation must not run")

    monkeypatch.setattr(
        stdio_module,
        "preflight_stdio_mcp_parent_death_containment",
        forbidden_preflight,
    )
    monkeypatch.setattr(
        stdio_module,
        "create_contained_stdio_mcp_process",
        forbidden_process,
    )
    monkeypatch.setattr(
        stdio_module,
        "create_direct_stdio_mcp_process",
        forbidden_process,
    )
    client = StdioMcpClient(secret_resolver=resolver)
    setattr(client, field_name, invalid_timeout)

    with pytest.raises(ValueError, match=rf"{field_name} must be finite"):
        asyncio.run(client.connect(_server_spec(secret=True)))

    assert preflight_calls == 0
    assert resolver.calls == 0
    assert process_calls == 0


def test_connect_owns_complete_configuration_before_containment_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        preflight_started = asyncio.Event()
        release_preflight = asyncio.Event()
        original_resolver = _RecordingSecretResolver()
        replacement_resolver = _ForbiddenSecretResolver()
        observed_env: dict[str, str] = {}
        observed_limit = 0

        async def suspended_preflight(
            _timeout_s: float,
        ) -> stdio_process._ContainmentPreflightProof:
            preflight_started.set()
            await release_preflight.wait()
            return stdio_process._ContainmentPreflightProof(
                owner_pid=os.getpid(),
                authority=stdio_process._PARENT_DEATH_CONTAINMENT_PREFLIGHT_AUTHORITY,
            )

        async def observe_launch(*_argv: str, **kwargs: Any) -> Any:
            nonlocal observed_limit
            observed_env.update(kwargs["env"])
            observed_limit = kwargs["limit"]
            raise RuntimeError("observed owned launch")

        monkeypatch.setenv("CAYU_TEST_HOST_SECRET", "host-secret-canary")
        monkeypatch.setattr(stdio_module, "validate_containment_platform", lambda _value: None)
        monkeypatch.setattr(
            stdio_module,
            "preflight_stdio_mcp_parent_death_containment",
            suspended_preflight,
        )
        monkeypatch.setattr(
            stdio_module,
            "_prepare_stdio_mcp_containment_rendezvous",
            _prepared_rendezvous,
        )
        monkeypatch.setattr(stdio_module, "create_contained_stdio_mcp_process", observe_launch)
        client = StdioMcpClient(
            inherit_env=False,
            secret_resolver=original_resolver,
            transport_limits=McpTransportLimits(
                max_message_bytes=1024,
                max_response_bytes=2048,
            ),
        )
        connect_task = asyncio.create_task(client.connect(_server_spec(secret=True)))
        await preflight_started.wait()

        client.inherit_env = True
        client.secret_resolver = replacement_resolver
        client.transport_limits = McpTransportLimits(
            max_message_bytes=4096,
            max_response_bytes=8192,
        )
        release_preflight.set()

        with pytest.raises(RuntimeError, match="observed owned launch"):
            await connect_task
        assert original_resolver.calls == 1
        assert replacement_resolver.calls == 0
        assert "CAYU_TEST_HOST_SECRET" not in observed_env
        assert observed_env["TOKEN"] == "resolved-token"
        assert observed_limit == 1026

    asyncio.run(run())


def test_platform_validator_fails_closed_for_parent_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stdio_process.os, "name", "nt")

    with pytest.raises(RuntimeError, match="unavailable"):
        stdio_process.validate_containment_platform(
            StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
        )


def test_platform_validator_requires_usable_pidfd_signaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stdio_process.os, "name", "posix")
    monkeypatch.setattr(stdio_process.sys, "platform", "linux")
    monkeypatch.setattr(stdio_process.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(stdio_process, "_linux_pidfd_signaling_supported", lambda: False)

    with pytest.raises(RuntimeError, match="usable pidfd signaling"):
        stdio_process.validate_containment_platform(
            StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
        )


def test_exact_preflight_exercises_every_containment_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        containment,
        "_establish_linux_child_reaping_semantics",
        lambda: calls.append("child_reaping"),
    )
    monkeypatch.setattr(
        containment,
        "_verify_linux_environment_transfer",
        lambda: calls.append("environment_transfer"),
    )
    monkeypatch.setattr(
        containment,
        "_verify_linux_waitid_without_reaping",
        lambda: calls.append("waitid"),
    )
    monkeypatch.setattr(
        containment,
        "_verify_linux_abstract_rendezvous",
        lambda: calls.append("abstract_rendezvous"),
    )
    monkeypatch.setattr(
        containment,
        "_set_linux_process_nondumpable",
        lambda: calls.append("nondumpable"),
    )
    monkeypatch.setattr(
        containment,
        "_set_linux_child_subreaper",
        lambda: calls.append("subreaper"),
    )
    monkeypatch.setattr(containment.os, "getpid", lambda: 31)
    monkeypatch.setattr(containment.os, "getppid", lambda: 30)
    monkeypatch.setattr(containment.os, "getpgrp", lambda: 31)
    monkeypatch.setattr(containment.os, "getpgid", lambda _pid: 30)

    def install_filter(**kwargs: Any) -> None:
        assert kwargs == {
            "protected_process_groups": (31, 30),
            "protected_process_pids": (31, 30),
        }
        calls.append("process_filter")

    monkeypatch.setattr(containment, "_install_linux_process_tree_filter", install_filter)
    monkeypatch.setattr(
        containment,
        "_linux_process_group_stats",
        lambda _pgid: {31: object()},
    )

    containment._verify_linux_containment_primitives()

    assert calls == [
        "child_reaping",
        "environment_transfer",
        "nondumpable",
        "subreaper",
        "waitid",
        "abstract_rendezvous",
        "process_filter",
    ]


def test_failed_helper_preflight_becomes_truthful_negative_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_environment: dict[str, str] | None = None

    class FailedPreflight:
        returncode: int | None = None

        async def wait(self) -> int:
            self.returncode = 72
            return 72

        def kill(self) -> None:
            self.returncode = -9

    async def fake_spawn(*_args: Any, **kwargs: Any) -> FailedPreflight:
        nonlocal observed_environment
        observed_environment = kwargs["env"]
        return FailedPreflight()

    monkeypatch.setattr(
        stdio_process,
        "stdio_mcp_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setattr(
        stdio_process,
        "_PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED",
        None,
    )
    monkeypatch.setattr(
        stdio_process,
        "_PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID",
        None,
    )
    monkeypatch.setattr(stdio_process.asyncio, "create_subprocess_exec", fake_spawn)

    with pytest.raises(RuntimeError, match="prerequisites are unavailable"):
        asyncio.run(stdio_process.preflight_stdio_mcp_parent_death_containment())

    assert observed_environment == {}
    assert not stdio_process.stdio_mcp_parent_death_containment_supported()
    evidence = stdio_process.stdio_mcp_process_capability_evidence(
        StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
    )
    assert evidence.state_for("parent_death_containment") == "unsupported"


def test_timed_out_preflight_does_not_poison_a_later_exact_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingPreflight:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.released = asyncio.Event()

        async def wait(self) -> int:
            await self.released.wait()
            assert self.returncode is not None
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9
            self.released.set()

    class SuccessfulPreflight:
        returncode: int | None = None

        async def wait(self) -> int:
            self.returncode = 0
            return 0

        def kill(self) -> None:
            raise AssertionError("successful preflight must not be killed")

    processes = [HangingPreflight(), SuccessfulPreflight()]

    async def fake_spawn(*_args: Any, **_kwargs: Any) -> Any:
        return processes.pop(0)

    monkeypatch.setattr(
        stdio_process,
        "stdio_mcp_parent_death_containment_platform_candidate",
        lambda: True,
    )
    monkeypatch.setattr(
        stdio_process,
        "_PARENT_DEATH_CONTAINMENT_PREFLIGHT_SUCCEEDED",
        None,
    )
    monkeypatch.setattr(
        stdio_process,
        "_PARENT_DEATH_CONTAINMENT_PREFLIGHT_PROCESS_ID",
        None,
    )
    monkeypatch.setattr(stdio_process.asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setattr(
        stdio_process,
        "_prepare_stdio_mcp_containment_rendezvous",
        _prepared_rendezvous,
    )

    with pytest.raises(TimeoutError, match="preflight timed out"):
        asyncio.run(stdio_process.preflight_stdio_mcp_parent_death_containment(0.001))

    assert stdio_process._current_parent_death_containment_preflight_result() is None
    proof = asyncio.run(stdio_process.preflight_stdio_mcp_parent_death_containment())
    stdio_process._validate_containment_preflight_proof(proof)
    assert stdio_process.stdio_mcp_parent_death_containment_supported()
    assert not processes


def test_group_member_signal_revalidates_membership_after_pidfd_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[int, int]] = []
    closed: list[int] = []

    monkeypatch.setattr(
        containment,
        "_linux_process_group_pids",
        lambda _pgid: {41: (42, 1233, 9001)},
    )
    monkeypatch.setattr(containment, "_linux_pidfd_open", lambda _pid: 73)
    monkeypatch.setattr(containment, "_linux_process_identity", lambda _pid: (99, 1234, 9002))
    monkeypatch.setattr(
        containment,
        "_linux_pidfd_send_signal",
        lambda pidfd, signal_number: sent.append((pidfd, signal_number)),
    )
    monkeypatch.setattr(containment.os, "close", closed.append)

    containment._signal_owned_group_members(42, 9)

    assert sent == []
    assert closed == [73]


def test_anchor_ready_rejects_unrelated_live_server_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = {
        41: containment._LinuxProcessStat("S", 40, 41, 100, 9001),
        42: containment._LinuxProcessStat("S", 99, 99, 101, 9002),
    }
    monkeypatch.setattr(containment, "_linux_process_stat", stats.get)

    assert (
        containment._validated_anchor_ready(
            {
                "anchor_pid": 41,
                "nonce": "nonce",
                "pgid": 41,
                "server_pid": 42,
                "type": "ready",
            },
            anchor_pid=41,
        )
        is None
    )


@pytest.mark.parametrize("invalid_timeout", [float("inf"), float("nan")])
def test_containment_timeouts_reject_nonfinite_values(invalid_timeout: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        StdioMcpClient(containment_term_timeout_s=invalid_timeout)


def test_rendezvous_completion_cannot_authorize_launch_after_startup_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        process = object.__new__(ContainedStdioMcpProcess)
        loop = asyncio.get_running_loop()
        process._rendezvous_ready = loop.create_future()
        process._ready = loop.create_future()
        launch_authorizations = 0
        kills = 0

        def authorize_launch() -> None:
            nonlocal launch_authorizations
            launch_authorizations += 1

        def kill() -> None:
            nonlocal kills
            kills += 1

        monkeypatch.setattr(process, "_authorize_launch", authorize_launch)
        monkeypatch.setattr(process, "kill", kill)

        ready_task = asyncio.create_task(process.await_ready(0.01))
        await asyncio.sleep(0)
        loop.call_soon(process._rendezvous_ready.set_result, None)
        loop.call_soon(time.sleep, 0.03)

        with pytest.raises(
            McpProtocolError,
            match="containment supervisor did not become ready",
        ):
            await ready_task

        assert launch_authorizations == 0
        assert kills == 1

    asyncio.run(run())


def test_final_settlement_timeout_rejects_an_unverified_supervisor_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        process = object.__new__(ContainedStdioMcpProcess)
        process._process = SimpleNamespace(returncode=0)  # type: ignore[assignment]
        process._capability_evidence = stdio_process.stdio_mcp_process_capability_evidence(
            StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
        )
        process.settlement_timeout_s = 0.01
        process.stdin = None
        process.stdout = asyncio.StreamReader()
        process.stderr = asyncio.StreamReader()

        async def stalled_settlement() -> int:
            await asyncio.Event().wait()
            return 0

        monkeypatch.setattr(process, "wait_for_settlement", stalled_settlement)
        session = StdioMcpSession(
            server=_server_spec(),
            process=process,
            request_timeout_s=1.0,
            write_timeout_s=1.0,
            graceful_shutdown_timeout_s=0.01,
            cancellation_notification_timeout_s=0.01,
            client_name="cayu",
            client_version="test",
        )

        with pytest.raises(McpProtocolError, match="authenticated settlement"):
            await session.close()

    asyncio.run(run())


def test_failed_start_retains_cleanup_without_unbounded_connect_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        release_cleanup = asyncio.Event()
        cleanup_started = asyncio.Event()
        retained_before = set(stdio_process._RETAINED_CONTAINMENT_STARTUP_CLEANUPS)

        class FakeOwner:
            settlement_timeout_s = 0.01

            def __init__(self, **kwargs: Any) -> None:
                self.control = kwargs["control"]
                self.owner_write_fd = kwargs["owner_write_fd"]
                self.killed = False

            async def await_ready(self, _timeout_s: float) -> None:
                raise McpProtocolError("startup failed")

            def kill(self) -> None:
                self.killed = True

            async def wait_for_settlement(self) -> int:
                cleanup_started.set()
                try:
                    await release_cleanup.wait()
                    return 0
                finally:
                    self.control.close()
                    os.close(self.owner_write_fd)

        async def fake_spawn(*_argv: str, **_kwargs: Any) -> object:
            return object()

        async def successful_preflight(
            _timeout_s: float,
        ) -> stdio_process._ContainmentPreflightProof:
            return stdio_process._ContainmentPreflightProof(
                owner_pid=os.getpid(),
                authority=stdio_process._PARENT_DEATH_CONTAINMENT_PREFLIGHT_AUTHORITY,
            )

        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        prepared_rendezvous = _prepared_rendezvous(
            stdio_process._command_containment_rendezvous_identity(("server",))
        )

        monkeypatch.setattr(
            stdio_process, "stdio_mcp_parent_death_containment_supported", lambda: True
        )
        monkeypatch.setattr(
            stdio_process,
            "preflight_stdio_mcp_parent_death_containment",
            successful_preflight,
        )
        monkeypatch.setattr(
            stdio_process,
            "_create_sealed_server_environment_fd",
            lambda _environment: read_fd,
        )
        monkeypatch.setattr(stdio_process, "ContainedStdioMcpProcess", FakeOwner)
        monkeypatch.setattr(stdio_process.asyncio, "create_subprocess_exec", fake_spawn)

        with pytest.raises(McpProtocolError, match="startup failed"):
            await asyncio.wait_for(
                stdio_process.create_contained_stdio_mcp_process(
                    "server",
                    env={},
                    limit=1024,
                    startup_timeout_s=0.01,
                    term_timeout_s=0.01,
                    kill_timeout_s=0.01,
                    _prepared_rendezvous=prepared_rendezvous,
                ),
                timeout=0.2,
            )

        await cleanup_started.wait()
        retained = stdio_process._RETAINED_CONTAINMENT_STARTUP_CLEANUPS - retained_before
        assert len(retained) == 1
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(*retained, return_exceptions=True), timeout=1)
        await asyncio.sleep(0)
        assert not (stdio_process._RETAINED_CONTAINMENT_STARTUP_CLEANUPS - retained_before)

    asyncio.run(run())


def test_post_spawn_owner_handoff_failure_closes_liveness_and_retains_reaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        release_cleanup = asyncio.Event()
        cleanup_started = asyncio.Event()
        retained_before = set(stdio_process._RETAINED_CONTAINMENT_STARTUP_CLEANUPS)

        class FakeProcess:
            async def wait(self) -> int:
                cleanup_started.set()
                await release_cleanup.wait()
                return 0

        async def fake_spawn(*_argv: str, **_kwargs: Any) -> FakeProcess:
            return FakeProcess()

        async def successful_preflight(
            _timeout_s: float,
        ) -> stdio_process._ContainmentPreflightProof:
            return stdio_process._ContainmentPreflightProof(
                owner_pid=os.getpid(),
                authority=stdio_process._PARENT_DEATH_CONTAINMENT_PREFLIGHT_AUTHORITY,
            )

        def fail_owner_construction(**_kwargs: Any) -> Any:
            raise RuntimeError("owner handoff failed")

        original_pipe = os.pipe
        server_env_fd, server_env_writer = original_pipe()
        os.close(server_env_writer)
        owner_read_fd, owner_write_fd = original_pipe()
        owner_read_observer = os.dup(owner_read_fd)
        prepared_rendezvous = _prepared_rendezvous(
            stdio_process._command_containment_rendezvous_identity(("server",))
        )
        try:
            monkeypatch.setattr(
                stdio_process,
                "preflight_stdio_mcp_parent_death_containment",
                successful_preflight,
            )
            monkeypatch.setattr(
                stdio_process,
                "_create_sealed_server_environment_fd",
                lambda _environment: server_env_fd,
            )
            monkeypatch.setattr(
                stdio_process.os,
                "pipe",
                lambda: (owner_read_fd, owner_write_fd),
            )
            monkeypatch.setattr(
                stdio_process,
                "ContainedStdioMcpProcess",
                fail_owner_construction,
            )
            monkeypatch.setattr(stdio_process.asyncio, "create_subprocess_exec", fake_spawn)

            with pytest.raises(RuntimeError, match="owner handoff failed"):
                await stdio_process.create_contained_stdio_mcp_process(
                    "server",
                    env={},
                    limit=1024,
                    startup_timeout_s=0.01,
                    term_timeout_s=0.01,
                    kill_timeout_s=0.01,
                    _prepared_rendezvous=prepared_rendezvous,
                )

            await cleanup_started.wait()
            assert os.read(owner_read_observer, 1) == b""
            retained = stdio_process._RETAINED_CONTAINMENT_STARTUP_CLEANUPS - retained_before
            assert len(retained) == 1
            release_cleanup.set()
            await asyncio.wait_for(asyncio.gather(*retained, return_exceptions=True), timeout=1)
            await asyncio.sleep(0)
            assert not (stdio_process._RETAINED_CONTAINMENT_STARTUP_CLEANUPS - retained_before)
        finally:
            os.close(owner_read_observer)

    asyncio.run(run())


@pytest.mark.skipif(
    not stdio_process.stdio_mcp_parent_death_containment_platform_candidate(),
    reason="contained stdio launch requires supported Linux process-tree enforcement",
)
def test_toolset_exposes_the_session_process_capability_evidence() -> None:
    async def run() -> tuple[str, str]:
        toolset = await McpToolset.connect(_server_spec())
        try:
            toolset_state = toolset.process_capability_evidence.state_for(
                "parent_death_containment"
            )
            assert isinstance(toolset.session, StdioMcpSession)
            session_state = toolset.session.process_capability_evidence.state_for(
                "parent_death_containment"
            )
            return toolset_state, session_state
        finally:
            await toolset.close()

    assert asyncio.run(run()) == ("available", "available")


def test_explicit_graceful_cleanup_preserves_stdio_behavior() -> None:
    async def run() -> int | None:
        session = await StdioMcpClient(
            process_lifetime=StdioMcpProcessLifetime.GRACEFUL_CLEANUP
        ).connect(_server_spec())
        try:
            await session.list_tools()
        finally:
            await session.close()
        return session.process.returncode

    assert asyncio.run(run()) == 0
