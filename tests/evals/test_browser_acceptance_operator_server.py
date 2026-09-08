"""Protected fixture startup failures still settle the canonical app owners."""

import asyncio

import pytest
from tests.evals.test_browser_acceptance_operator_trust import certificate_and_key

from cayu import SQLiteSessionStore
from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.internal.browser_acceptance import build
from cayu.evals.internal.browser_acceptance_operator_server import operator_fixture_setup


@pytest.mark.parametrize("phase", ["profile", "construction", "early_exit"])
def test_startup_failure_closes_every_quiescent_canonical_store(tmp_path, monkeypatch, phase):
    import uvicorn

    certificate, key = certificate_and_key()
    cert_path, key_path = tmp_path / "control.crt", tmp_path / "server.key"
    cert_path.write_bytes(certificate)
    key_path.write_bytes(key)
    failure = RuntimeError("fixture startup failed")
    closed = []
    original_close = SQLiteSessionStore.close

    async def close(store):
        closed.append(id(store))
        await original_close(store)

    monkeypatch.setattr(SQLiteSessionStore, "close", close)

    class Server:
        started = False
        should_exit = False

        def __init__(self, config):
            if phase == "construction":
                raise failure

        async def serve(self):
            assert phase == "early_exit"

    monkeypatch.setattr(uvicorn, "Server", Server)

    async def scenario(fixture):
        async with operator_fixture_setup(
            server_container_id="a" * 64,
            ca_certificate=cert_path,
            server_private_key=key_path,
        ) as setup:
            plan = await build(fixture, operator_fixture=setup.binding)
            app = plan.eval_plan.app
            assert app is not None
            applications = {id(app): app, **{id(value): value for _, value in plan.case_apps}}
            stores = {id(value.session_store) for value in applications.values()}
            if phase == "profile":

                async def inspect(request):
                    raise failure

                monkeypatch.setattr(app, "inspect_run_execution_profile", inspect)
            with pytest.raises(RuntimeError) as caught:
                async with setup.serve(plan):
                    pytest.fail("Failed server startup yielded a running fixture")
            if phase == "early_exit":
                assert "before readiness" in str(caught.value)
            else:
                assert caught.value is failure
            assert set(closed) == stores
            assert len(closed) == len(stores)

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))
