"""CLI setup owns contexts but cannot replace the checked-in corpus builder."""

import argparse
import asyncio
import ssl
from contextlib import asynccontextmanager

import httpx
import pytest
import scripts.run_browser_acceptance as command
from pydantic import SecretStr
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.evals.test_browser_acceptance_execution import _plan
from tests.evals.test_browser_acceptance_operator_trust import certificate_and_key

from cayu.evals.internal import browser_acceptance as canonical
from cayu.evals.internal.browser_acceptance_operator import (
    OperatorFixtureBinding,
    OperatorFixtureSetup,
)
from cayu.runtime.browser_control_config import BrowserControlConfig


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_command_keeps_exact_plan_inside_operator_contexts(tmp_path, monkeypatch, outcome):
    certificate, _ = certificate_and_key()
    cert = tmp_path / "control.crt"
    cert.write_bytes(certificate)
    calls = []
    entered = asyncio.Event()
    expected = {}
    primary = RuntimeError("fixture execution failed")

    @asynccontextmanager
    async def factory():
        calls.append("setup")
        async with httpx.AsyncClient(base_url="https://127.0.0.1:8443") as client:
            binding = OperatorFixtureBinding(
                control=BrowserControlConfig(
                    policy=Policy(True),
                    purpose=operator_purpose(),
                    guest_endpoint="wss://cayu-control:8443/api/browser-control/guest",
                ),
                server_container_id="a" * 64,
                ca_certificate=cert,
                client=client,
                tls=ssl.create_default_context(),
                operator_origin="https://operator.test",
                private_text=SecretStr("private-command-fixture"),
            )
            expected["binding"] = binding

            @asynccontextmanager
            async def serve(plan):
                assert plan is expected["plan"]
                calls.append("serve")
                try:
                    yield
                finally:
                    calls.append("settle")

            try:
                yield OperatorFixtureSetup(binding=binding, serve=serve)
            finally:
                calls.append("client-close")

    async def build(fixture, *, operator_fixture):
        assert operator_fixture is expected["binding"]
        calls.append("canonical-build")
        plan = _plan(tmp_path, fixture)
        expected["plan"] = plan
        return plan

    original_run = command._run_plan

    async def execute(args, *, plan, deterministic_fixture):
        calls.append("run")
        entered.set()
        if outcome == "failure":
            raise primary
        if outcome == "cancel":
            await asyncio.Event().wait()
        return await original_run(args, plan=plan, deterministic_fixture=deterministic_fixture)

    def load(target, **kwargs):
        assert target == "application:operator_setup"
        return factory

    monkeypatch.setattr(command, "load_target", load)
    monkeypatch.setattr(canonical, "build", build)
    monkeypatch.setattr(command, "_run_plan", execute)
    monkeypatch.setattr(
        command, "deterministic_browser_acceptance_manifest", lambda: expected["plan"].manifest
    )
    args = argparse.Namespace(
        target=None,
        mode="deterministic",
        operator_setup="application:operator_setup",
        output_directory=tmp_path / "reports",
    )

    async def scenario():
        task = asyncio.create_task(command._run(args))
        if outcome == "cancel":
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelling() == 1
            assert task.cancelled()
        elif outcome == "failure":
            with pytest.raises(RuntimeError) as caught:
                await task
            assert caught.value is primary
        else:
            assert await task == 0
            assert len(list(args.output_directory.glob("*.json"))) == 1
            assert len(list(args.output_directory.glob("*.html"))) == 1

    asyncio.run(scenario())
    assert calls == ["setup", "canonical-build", "serve", "run", "settle", "client-close"]


@pytest.mark.parametrize("mode", ["live_public", "live_authenticated"])
def test_operator_setup_cannot_authorize_live_mode(monkeypatch, mode):
    def forbidden_load(*args, **kwargs):
        pytest.fail("Wrong-mode setup was loaded")

    monkeypatch.setattr(command, "load_target", forbidden_load)
    with pytest.raises(ValueError, match="deterministic"):
        asyncio.run(
            command._run(
                argparse.Namespace(mode=mode, operator_setup="application:setup", target=None)
            )
        )


def test_wrong_setup_is_rejected_before_canonical_build(monkeypatch):
    exited = []

    @asynccontextmanager
    async def factory():
        try:
            yield object()
        finally:
            exited.append(True)

    async def forbidden_build(*args, **kwargs):
        pytest.fail("Invalid setup reached construction")

    monkeypatch.setattr(command, "load_target", lambda *args, **kwargs: factory)
    monkeypatch.setattr(canonical, "build", forbidden_build)
    with pytest.raises(TypeError, match="exact OperatorFixtureSetup"):
        asyncio.run(
            command._run(
                argparse.Namespace(
                    mode="deterministic", operator_setup="application:setup", target=None
                )
            )
        )
    assert exited == [True]
