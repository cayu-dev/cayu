"""Explicit schema CLI ownership; no database service or application construction."""

import asyncio
import importlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from cayu.cli.project import project_context
from tests.qualification.test_repository_maintenance_application import project as project


@pytest.fixture
def schema(project):
    with project_context(project):
        yield importlib.import_module("operations.maintenance_schema")


@pytest.mark.parametrize("action", ["initialize", "check"])
@pytest.mark.parametrize("failure", [None, "constructor", "operation"])
def test_schema_command_delegates_and_sanitizes(
    schema, monkeypatch, capsys, caplog, action, failure
):
    canary = "postgresql://private-schema-canary@database/app"
    calls = []
    monkeypatch.setenv("CAYU_DATABASE_URL", canary)

    class Store:
        def __init__(self, dsn):
            assert dsn == canary
            if failure == "constructor":
                raise ValueError(canary)

        async def initialize(self):
            calls.append("initialize")
            if failure == "operation":
                raise RuntimeError(canary)

        async def check_ready(self):
            calls.append("check")
            if failure == "operation":
                raise RuntimeError(canary)

    monkeypatch.setattr(schema, "PostgresMaintenanceRunStore", Store)
    with warnings.catch_warnings(record=True) as captured:
        assert schema.main([action]) == (1 if failure else 0)
    assert calls == ([] if failure == "constructor" else [action])
    output = capsys.readouterr()
    if failure:
        assert output.out == ""
        assert output.err == "Maintenance reservation schema unavailable.\n"
    else:
        assert output.err == ""
        assert output.out == "Maintenance reservation schema " + (
            "initialized.\n" if action == "initialize" else "ready.\n"
        )
    assert not captured and canary not in output.out + output.err + caplog.text


@pytest.mark.parametrize("dsn", [None, "", "   "])
def test_missing_dsn_precedes_store_construction(schema, monkeypatch, capsys, dsn):
    if dsn is None:
        monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("CAYU_DATABASE_URL", dsn)

    def forbidden(*args):
        pytest.fail("Invalid DSN reached store construction")

    monkeypatch.setattr(schema, "PostgresMaintenanceRunStore", forbidden)
    assert schema.main(["initialize"]) == 1
    assert capsys.readouterr().err == "Maintenance reservation schema unavailable.\n"


def test_schema_operation_cancellation_is_not_success(schema, monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        finished = []

        class Store:
            def __init__(self, dsn):
                pass

            async def initialize(self):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.append(True)

            async def check_ready(self):
                pytest.fail("Cancellation selected another schema operation")

        monkeypatch.setattr(schema, "PostgresMaintenanceRunStore", Store)
        owner = asyncio.create_task(schema._run_schema("initialize", "fixture"))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not owner.done()
            owner.cancel("stop-schema")
            with pytest.raises(asyncio.CancelledError, match="stop-schema"):
                await owner
            assert owner.cancelled() and owner.cancelling() == 1
            assert finished == [True]
        finally:
            if not owner.done():
                owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)

    asyncio.run(scenario())


def test_emitted_cli_imports_without_provider_docker_or_database_configuration(project):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("CAYU_") or key in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}:
            env.pop(key)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "operations.maintenance_schema", "check"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == "Maintenance reservation schema unavailable.\n"
