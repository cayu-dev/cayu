from __future__ import annotations

import traceback
import warnings

import pytest
from tests.core.test_completion_decision_application import _assert_secret_absent_from_cayu_error

from cayu.applications import CayuApp
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore, TaskStore
from cayu.tasks.graphs import TaskGraphEvent
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryTaskStore()
        return
    value = (
        SQLiteTaskStore(tmp_path / "graph-sdk-errors.sqlite")
        if request.param == "sqlite"
        else PostgresTaskStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    )
    try:
        yield value
    finally:
        await value.close()


async def test_public_sdk_missing_graph_events_preserves_key_error(store: TaskStore) -> None:
    app = CayuApp(task_store=store, enable_logging=False)
    assert await app.load_task_graph("missing-graph") is None
    with pytest.raises(KeyError, match="not found") as captured:
        await app.list_task_graph_events("missing-graph")
    assert type(captured.value) is KeyError
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert await app.load_task_graph("missing-graph") is None


@pytest.mark.parametrize("diagnostic", ["payload", "rendering"])
async def test_public_sdk_graph_key_error_diagnostics_are_credential_safe(
    diagnostic: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    message = "graph-store-credential-canary" if diagnostic == "payload" else "missing graph"
    secret = message if diagnostic == "payload" else repr(KeyError(str(KeyError(message))))

    class CredentialErrorStore(InMemoryTaskStore):
        async def list_task_graph_events(
            self, graph_id: str, *, after_sequence: int = 0, limit: int = 100
        ) -> list[TaskGraphEvent]:
            raise KeyError(message)

    app = CayuApp(
        task_store=CredentialErrorStore(),
        secret_redactor=SecretRedactor(secret),
        enable_logging=False,
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(KeyError) as captured:
            await app.list_task_graph_events("missing-graph")

    error = captured.value
    assert type(error) is KeyError
    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_secret_absent_from_cayu_error(error, secret)
    assert secret not in "".join(traceback.format_exception(error))
    output = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in output.out
    assert secret not in output.err
    assert all(secret not in str(item.message) for item in caught_warnings)
