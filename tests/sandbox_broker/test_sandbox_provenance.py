"""A brokered sandbox's provenance: the Release, session and task it was created for, which
Cayu Cloud records so a sandbox left behind can be traced."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from tests.sandbox_broker.fake_broker import API_URL, APPLICATION, TOKEN, FakeBroker

from cayu.environments import EnvironmentFactoryReleaseAction, EnvironmentFactoryRequest
from cayu.runtime._task_lease_authority import (
    TaskLeaseAuthority,
    bind_task_lease_authority,
    current_task_id,
)
from cayu.sandbox_broker import (
    BrokeredSandboxCodingEnvironmentFactory,
    SandboxBrokerRefusedError,
    SandboxBrokerSettings,
    SandboxProvenance,
)
from cayu.sandbox_broker.client import MAX_PROVENANCE_ID_CHARS
from cayu.workspaces import LocalWorkspace

_ENV = {
    "CAYU_CLOUD_API_URL": API_URL,
    "CAYU_CLOUD_SANDBOX_TOKEN": TOKEN,
    "CAYU_CLOUD_APPLICATION_ID": APPLICATION,
}


def _creates(broker: FakeBroker) -> list[dict | None]:
    return [
        json.loads(request.content).get("provenance")
        for request in broker.requests
        if request.method == "POST" and request.url.path.endswith("/sandboxes")
    ]


async def _create(client, key: str, **options):
    return await client.create(
        idempotency_key=key, image="coding-python", egress_mode="none", lease_seconds=60, **options
    )


def test_the_release_comes_from_the_variable_cloud_states() -> None:
    assert SandboxBrokerSettings.from_env(_ENV).release_id is None
    settings = SandboxBrokerSettings.from_env({**_ENV, "CAYU_RELEASE_ID": " dep_7 "})
    assert settings.release_id == "dep_7"
    # A value Cloud wouldn't take is left out rather than failing the process.
    for unusable in ("", "d" * (MAX_PROVENANCE_ID_CHARS + 1), "dep\n7"):
        assert (
            SandboxBrokerSettings.from_env({**_ENV, "CAYU_RELEASE_ID": unusable}).release_id is None
        )
    with pytest.raises(ValueError, match="release_id"):
        SandboxBrokerSettings(
            api_url=API_URL, application_id=APPLICATION, token=TOKEN, release_id=""
        )


def test_a_create_names_its_session_task_and_the_processs_release(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        client = broker.client(release_id="dep_7")
        record = await _create(
            client, "traced", provenance=SandboxProvenance(session_id="ses_1", task_id="task_2")
        )
        assert record.provenance == SandboxProvenance(
            release_id="dep_7", session_id="ses_1", task_id="task_2"
        )
        # Empty provenance names only the Release.
        released = await _create(client, "released", provenance=SandboxProvenance())
        assert released.provenance == SandboxProvenance(release_id="dep_7")
        # A create's own Release wins.
        own = await _create(client, "own", provenance=SandboxProvenance(release_id="dep_9"))
        assert own.provenance is not None and own.provenance.release_id == "dep_9"
        # Without provenance, even with a Release configured, the body is the one older
        # Runtimes sent; and nothing to name sends none either.
        unnamed = await _create(client, "unnamed")
        assert unnamed.provenance is None
        bare = await _create(broker.client(), "bare", provenance=SandboxProvenance())
        assert bare.provenance is None
        with pytest.raises(TypeError):
            await _create(client, "wrong", provenance={"session_id": "ses_1"})
        await client.aclose()

    asyncio.run(scenario())
    assert _creates(broker) == [
        {"release_id": "dep_7", "session_id": "ses_1", "task_id": "task_2"},
        {"release_id": "dep_7"},
        {"release_id": "dep_9"},
        None,
        None,
    ]


def test_a_create_without_provenance_replays_under_its_key_after_a_redeploy(
    tmp_path: Path,
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        before, after = broker.client(release_id="dep_old"), broker.client(release_id="dep_new")
        first = await _create(before, "task-17")
        # A retried task on the new Release gets back the sandbox it already created.
        again = await _create(after, "task-17")
        assert again.id == first.id
        for client in (before, after):
            await client.aclose()

    asyncio.run(scenario())
    assert _creates(broker) == [None, None]


def test_a_record_keeps_a_typed_provenance() -> None:
    from cayu.sandbox_broker import SandboxRecord

    record = SandboxRecord.model_validate(
        {
            "id": "sbx_1",
            "status": "ready",
            "image": "coding-python",
            "egress": {"mode": "none"},
            "provenance": SandboxProvenance(session_id="ses_1"),
        }
    )
    assert record.provenance == SandboxProvenance(session_id="ses_1")
    unreadable = SandboxRecord.model_validate({**record.model_dump(), "provenance": "ses_1"})
    assert unreadable.provenance is None


def test_an_id_cloud_wouldnt_take_is_left_out() -> None:
    provenance = SandboxProvenance(
        session_id="s" * (MAX_PROVENANCE_ID_CHARS + 1), task_id="task\n2", release_id=""
    )
    assert provenance.wire() is None
    assert SandboxProvenance(session_id="ses_1", task_id=7).wire() == {"session_id": "ses_1"}


def test_a_cloud_before_provenance_gets_the_create_without_it(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.provenance = False

    async def scenario():
        client = broker.client(release_id="dep_7")
        first = await _create(client, "a", provenance=SandboxProvenance(session_id="ses_1"))
        assert first.status == "provisioning" and first.provenance is None
        # Once refused, the client sends it no more.
        second = await _create(client, "b", provenance=SandboxProvenance(session_id="ses_1"))
        assert second.provenance is None
        await client.aclose()

    asyncio.run(scenario())
    assert _creates(broker) == [{"release_id": "dep_7", "session_id": "ses_1"}, None, None]
    assert len(broker.sandboxes) == 2


def test_other_refusals_of_a_create_with_provenance_are_raised(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.catalog = set()

    async def scenario():
        client = broker.client(release_id="dep_7")
        with pytest.raises(SandboxBrokerRefusedError, match="image_unknown"):
            await _create(client, "a", provenance=SandboxProvenance())
        await client.aclose()

    asyncio.run(scenario())
    assert _creates(broker) == [{"release_id": "dep_7"}]


def test_the_current_task_is_the_handler_task_this_code_runs_in() -> None:
    from datetime import UTC, datetime

    assert current_task_id() is None
    authority = TaskLeaseAuthority(
        datetime.now(UTC), claim_generation=1, task_id="task_2", worker_id="w-1"
    )

    async def handler() -> str | None:
        await asyncio.sleep(0)
        return current_task_id()

    async def scenario():
        with bind_task_lease_authority(authority):
            running = asyncio.ensure_future(handler())
        # The handler keeps its task after the worker's binding ends.
        assert current_task_id() is None
        return await running

    assert asyncio.run(scenario()) == "task_2"


def test_the_coding_factory_names_the_session_and_task_of_each_sandbox(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_bytes(b"x = 1\n")
    broker = FakeBroker(tmp_path / "cloud")
    factory = BrokeredSandboxCodingEnvironmentFactory(
        broker=broker.client(release_id="dep_7"),
        source_workspace=LocalWorkspace(source, workspace_id="project"),
        egress="none",
    )
    authority = TaskLeaseAuthority(
        datetime.now(UTC), claim_generation=1, task_id="task_2", worker_id="w-1"
    )

    async def create(session_id: str):
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id=session_id, agent_name="coder", environment_name="sandbox"
            )
        )
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    async def scenario():
        with bind_task_lease_authority(authority):
            in_task = asyncio.ensure_future(create("ses_in_task"))
        await in_task
        await create("ses_alone")

    asyncio.run(scenario())
    assert _creates(broker) == [
        {"release_id": "dep_7", "session_id": "ses_in_task", "task_id": "task_2"},
        {"release_id": "dep_7", "session_id": "ses_alone"},
    ]
