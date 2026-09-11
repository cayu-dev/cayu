"""Application-owned durable data must survive process-harness teardown."""

import asyncio
import sys
from types import SimpleNamespace

import pytest
from tests.recovery import worker_harness


@pytest.mark.parametrize("owns_backend", [True, False])
def test_sqlite_backend_ownership_is_explicit(tmp_path, owns_backend):
    backend = worker_harness.BackendConfig.sqlite(tmp_path)
    retained = [tmp_path / name for name in ("sessions.sqlite", "tasks.sqlite", "tasks.sqlite-wal")]
    for path in retained:
        path.write_bytes(b"application-owned-test-bytes")
    control = tmp_path / "phase-test.json"
    control.write_text("{}")
    with worker_harness.RecoveryHarness(tmp_path, backend, owns_backend=owns_backend):
        assert all(path.is_file() for path in retained)
    assert not control.exists()
    assert all(path.exists() is not owns_backend for path in retained)
    if not owns_backend:
        assert all(path.read_bytes() == b"application-owned-test-bytes" for path in retained)


def test_borrowed_postgres_never_resets_or_deletes_backend(tmp_path, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("Borrowed backend was mutated by the process harness")

    monkeypatch.setattr(worker_harness, "_reset_postgres_public_authority_registry", forbidden)
    monkeypatch.setattr(worker_harness.RecoveryHarness, "_cleanup_postgres_rows", forbidden)
    control = tmp_path / "result-test.json"
    control.write_text("{}")
    with worker_harness.RecoveryHarness(
        tmp_path, worker_harness.BackendConfig.postgres("not-a-live-dsn"), owns_backend=False
    ):
        pass
    assert not control.exists()


def test_maintenance_scenario_imports_only_on_dispatch(monkeypatch):
    received = []

    async def operation(config):
        received.append(config)
        return {"observed": True}

    monkeypatch.setitem(
        sys.modules,
        "maintenance_approval_scenario",
        SimpleNamespace(run_maintenance_approval=operation),
    )
    config = {"scenario": "maintenance_approval", "action": "recover"}
    assert asyncio.run(worker_harness._run_maintenance_approval(config)) == {"observed": True}
    assert received == [config]
