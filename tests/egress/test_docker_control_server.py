"""Allocation-owned network attachment for a colocated protected server."""

import asyncio
import json
from pathlib import Path

import pytest
from tests.egress.test_docker_adapter import _credentialless_broker

from cayu.egress.docker_adapter import DockerEgressAdapter
from cayu.egress.errors import UnsupportedEgressError

CONTROL_ID = "a" * 64


class Docker:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = []
        self.attached = set()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.script = ""

    async def read(self, argv):
        if argv[:2] == ["network", "inspect"]:
            return 0, json.dumps({CONTROL_ID: {}} if argv[-1] in self.attached else {})
        return 0, CONTROL_ID

    async def execute(self, argv):
        self.calls.append(list(argv))
        if argv[:3] == ["network", "connect", "--alias"]:
            if self.failure == "before_attach":
                return 1, "attachment unavailable"
            self.attached.add(argv[-2])
            if self.failure == "after_attach":
                return 1, "attachment acknowledgement lost"
            if self.failure == "cancel":
                self.entered.set()
                await self.release.wait()
        if argv[:2] == ["network", "disconnect"]:
            if self.failure == "detach":
                return 1, "detach unavailable"
            self.attached.discard(argv[-2])
        if argv[:2] == ["network", "rm"] and argv[-1] in self.attached:
            return 1, "active endpoints"
        if argv[0] == "run":
            mount = next(value for value in argv if "dst=/run/cayu/connect-broker," in value)
            source = next(part[4:] for part in mount.split(",") if part.startswith("src="))
            self.script = Path(source).read_text()
        return 0, ""

    def adapter(self):
        return DockerEgressAdapter(
            docker_exec=self.execute,
            docker_run=self.read,
            control_server_container_id=CONTROL_ID,
        )


@pytest.mark.parametrize("bad", ["server", "a" * 12, "A" * 64, True, "a" * 63 + ";"])
def test_control_container_requires_exact_non_echoing_identity(bad):
    with pytest.raises(ValueError, match="exact full Docker container ID") as raised:
        DockerEgressAdapter(control_server_container_id=bad)
    assert str(raised.value) == "Control server requires an exact full Docker container ID."


def test_control_server_attachment_and_broker_share_allocation_lifetime():
    async def scenario():
        docker = Docker()
        adapter = docker.adapter()
        binding = await adapter.prepare(
            session_id="session", grants=(), broker=_credentialless_broker()
        )
        assert docker.attached == {binding.network}
        assert "PROXY:cayu-control:cayu-transport.invalid:443" in docker.script
        assert binding.metadata["proxy_bind_host"] == "0.0.0.0"
        assert "CAYU_BROKER" not in repr(binding.env)
        # Distinct bindings attach independently; retiring one must not cut off
        # another session using the same application-owned server.
        other = await adapter.prepare(
            session_id="other", grants=(), broker=_credentialless_broker()
        )
        await binding.close()
        assert docker.attached == {other.network}
        await other.close()
        assert not docker.attached
        assert ["rm", "-f", CONTROL_ID] not in docker.calls

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["before_attach", "after_attach", "cancel"])
def test_failed_prepare_detaches_exact_control_endpoint(failure):
    async def scenario():
        docker = Docker(failure)
        task = asyncio.create_task(
            docker.adapter().prepare(
                session_id="session", grants=(), broker=_credentialless_broker()
            )
        )
        if failure == "cancel":
            await asyncio.wait_for(docker.entered.wait(), 2)
            task.cancel()
            assert task.cancelling() == 1
            await asyncio.sleep(0)
            assert not task.done()
            assert docker.attached
            assert not any(call[:2] == ["network", "disconnect"] for call in docker.calls)
            docker.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Existing rollback accounts for the delivered request before its
            # shielded cleanup, while preserving ordinary cancellation handling.
            assert task.cancelled() and task.cancelling() == 0
        else:
            with pytest.raises(UnsupportedEgressError):
                await task
        assert not docker.attached
        assert not any(call[0] == "run" for call in docker.calls)
        assert any(call[:2] == ["network", "disconnect"] for call in docker.calls)

    asyncio.run(scenario())


def test_unconfirmed_detach_retains_binding_for_retry():
    async def scenario():
        docker = Docker()
        binding = await docker.adapter().prepare(
            session_id="session", grants=(), broker=_credentialless_broker()
        )
        docker.failure = "detach"
        with pytest.raises(RuntimeError, match="teardown incomplete"):
            await binding.close()
        assert docker.attached == {binding.network}
        docker.failure = None
        await binding.close()
        assert not docker.attached

    asyncio.run(scenario())


def test_stale_control_identity_rejects_before_resource_creation():
    async def scenario():
        docker = Docker()

        async def stale(argv):
            return 0, "b" * 64

        adapter = DockerEgressAdapter(
            docker_exec=docker.execute,
            docker_run=stale,
            control_server_container_id=CONTROL_ID,
        )
        with pytest.raises(UnsupportedEgressError):
            await adapter.prepare(session_id="session", grants=(), broker=_credentialless_broker())
        assert docker.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("retry", ["drain", "prepare"])
def test_failed_prepare_retains_rollback_owner_without_touching_other_binding(retry):
    async def scenario():
        docker = Docker()
        adapter = docker.adapter()
        other = await adapter.prepare(
            session_id="other", grants=(), broker=_credentialless_broker()
        )

        async def fail_start_and_detach(argv):
            if argv[0] == "run":
                return 1, "sidecar unavailable"
            if argv[:2] == ["network", "disconnect"]:
                return 1, "detach unavailable"
            return await docker.execute(argv)

        adapter._docker_exec = fail_start_and_detach
        with pytest.raises(UnsupportedEgressError):
            await adapter.prepare(session_id="failed", grants=(), broker=_credentialless_broker())
        failed_networks = docker.attached - {other.network}
        assert len(failed_networks) == 1
        assert set(adapter._preparation_cleanups) == failed_networks
        with pytest.raises(RuntimeError, match="teardown incomplete"):
            await adapter.drain_preparation_cleanup()
        assert docker.attached == failed_networks | {other.network}

        adapter._docker_exec = docker.execute
        if retry == "drain":
            await adapter.drain_preparation_cleanup()
        else:
            replacement = await adapter.prepare(
                session_id="replacement", grants=(), broker=_credentialless_broker()
            )
            await replacement.close()
        assert not adapter._preparation_cleanups
        assert docker.attached == {other.network}
        await other.close()
        assert not docker.attached

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_drain", [False, True])
def test_prepare_rollback_timeout_retains_one_task_until_late_completion(monkeypatch, cancel_drain):
    async def scenario():
        docker = Docker()
        adapter = docker.adapter()
        entered = asyncio.Event()
        release = asyncio.Event()
        detach_calls = 0

        async def delayed_cleanup(argv):
            nonlocal detach_calls
            if argv[0] == "run":
                return 1, "sidecar unavailable"
            if argv[:2] == ["network", "disconnect"]:
                detach_calls += 1
                entered.set()
                await release.wait()
            return await docker.execute(argv)

        adapter._docker_exec = delayed_cleanup
        monkeypatch.setattr(
            "cayu.egress.docker_adapter.DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS", 0.05
        )
        with pytest.raises(UnsupportedEgressError):
            await adapter.prepare(session_id="failed", grants=(), broker=_credentialless_broker())
        assert entered.is_set() and docker.attached
        cleanup = next(iter(adapter._preparation_cleanups.values()))
        owned_task = cleanup.task
        with pytest.raises(TimeoutError):
            await adapter.drain_preparation_cleanup()
        assert cleanup.task is owned_task and detach_calls == 1
        drain = None
        if cancel_drain:
            monkeypatch.setattr(
                "cayu.egress.docker_adapter.DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS", 2
            )
            drain = asyncio.create_task(adapter.drain_preparation_cleanup())
            await asyncio.sleep(0)
            drain.cancel()
            assert drain.cancelling() == 1
            await asyncio.sleep(0)
            assert not drain.done() and docker.attached
        release.set()
        assert owned_task is not None
        await asyncio.wait_for(asyncio.shield(owned_task), 2)
        if drain is not None:
            with pytest.raises(asyncio.CancelledError):
                await drain
            assert drain.cancelled() and drain.cancelling() == 0
        assert not adapter._preparation_cleanups
        assert not docker.attached

    asyncio.run(scenario())


@pytest.mark.parametrize("readback", ["confirmed", "malformed", "unavailable"])
def test_detach_acknowledgement_loss_requires_positive_readback(readback):
    async def scenario():
        docker = Docker()
        adapter = docker.adapter()
        binding = await adapter.prepare(
            session_id="session", grants=(), broker=_credentialless_broker()
        )

        async def lose_ack(argv):
            result = await docker.execute(argv)
            if argv[:2] == ["network", "disconnect"]:
                return 1, "detachment acknowledgement lost"
            return result

        async def read(argv):
            if readback == "malformed":
                return 0, "null"
            if readback == "unavailable":
                return 1, ""
            return await docker.read(argv)

        adapter._docker_exec = lose_ack
        adapter._docker_run = read
        if readback == "confirmed":
            await binding.close()
        else:
            with pytest.raises(RuntimeError, match="teardown incomplete"):
                await binding.close()
            adapter._docker_exec = docker.execute
            adapter._docker_run = docker.read
            await binding.close()
        assert not docker.attached

    asyncio.run(scenario())
