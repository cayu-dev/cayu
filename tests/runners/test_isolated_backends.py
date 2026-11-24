"""open_isolated_runner picks Docker or Cayu Cloud's broker and always releases the runner."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest
from tests.sandbox_broker.fake_broker import FakeBroker

from cayu.runners import (
    CloudIsolatedOptions,
    ExecCommand,
    IsolatedBackendRegistry,
    IsolatedRunnerContext,
    LocalRunner,
    default_isolated_backends,
    isolated_run_key,
    open_isolated_runner,
    run_isolated,
)

pytestmark = pytest.mark.anyio

CLOUD_ENV = {
    "CAYU_CLOUD_API_URL": "https://cloud.example.test",
    "CAYU_CLOUD_APPLICATION_ID": "invoices",
    "CAYU_CLOUD_SANDBOX_TOKEN": "token",
}
REVERSE = ExecCommand.process(
    "python3",
    "-I",
    "-c",
    "import pathlib; p = pathlib.Path('in.bin'); pathlib.Path('out.bin').write_bytes(p.read_bytes()[::-1])",
)


def test_selection_follows_the_coding_backend_rule() -> None:
    registry = default_isolated_backends()
    assert registry.selected_name({}) == "docker"
    assert registry.selected_name(CLOUD_ENV) == "cayu-cloud"
    assert registry.selected_name({**CLOUD_ENV, "CAYU_ISOLATED_BACKEND": "docker"}) == "docker"
    with pytest.raises(ValueError, match="isolated backend, but"):
        registry.selected_name({}, required="cayu-cloud")
    with pytest.raises(ValueError, match="incomplete"):
        registry.selected_name({"CAYU_CLOUD_SANDBOX_TOKEN": "token"})
    with pytest.raises(ValueError, match="isn't registered"):
        registry.selected_name({"CAYU_ISOLATED_BACKEND": "elsewhere"})


async def test_another_provider_plugs_in_by_name(tmp_path: Path) -> None:
    opened: list[IsolatedRunnerContext] = []
    closed: list[bool] = []

    @contextlib.asynccontextmanager
    async def local(context: IsolatedRunnerContext):
        opened.append(context)
        try:
            yield LocalRunner(root=tmp_path)
        finally:
            closed.append(True)

    registry = IsolatedBackendRegistry()
    registry.register("local-test", local)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("local-test", local)
    key = isolated_run_key(REVERSE, {"in.bin": b"abc"})
    async with open_isolated_runner(
        key,
        environ={"CAYU_ISOLATED_BACKEND": "local-test"},
        registry=registry,
        options={"local-test": {"flag": 1}},
    ) as runner:
        assert isinstance(runner, LocalRunner)
    assert opened[0].key == key and opened[0].options == {"flag": 1}
    assert closed == [True]
    with pytest.raises(ValueError, match="key"):
        async with open_isolated_runner("not a key", registry=registry, required="local-test"):
            pass


async def test_cloud_backend_runs_without_egress_and_destroys_the_sandbox(
    tmp_path: Path,
) -> None:
    broker = FakeBroker(tmp_path)
    broker.staged_copies = {"max_bytes": 8 * 1024 * 1024, "sandbox_max_bytes": 32 * 1024 * 1024}
    client = broker.client()
    options = {
        "cayu-cloud": CloudIsolatedOptions(
            broker=client, poll_interval_s=0, lease_seconds=60, renew_every_s=0.05
        )
    }
    data = bytes(range(256)) * 64
    key = isolated_run_key(REVERSE, {"in.bin": data})
    try:
        async with open_isolated_runner(key, options=options, environ=CLOUD_ENV) as runner:
            (sandbox,) = broker.live()
            assert sandbox.record()["egress"]["mode"] == "none"
            assert sandbox.record()["labels"]["cayu-run-key"] == key[:32]
            run = await run_isolated(runner, REVERSE, inputs={"in.bin": data}, outputs=["out.bin"])
            # The lease keeps being renewed while the block runs.
            for _ in range(100):
                if broker.renewals:
                    break
                await asyncio.sleep(0.02)
            assert broker.renewals > 0
        assert run.succeeded and run.outputs["out.bin"] == data[::-1]
        assert broker.objects  # the files went through staged copies
        assert broker.live() == []
    finally:
        await client.aclose()


async def test_cloud_backend_destroys_the_sandbox_on_every_failure(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path)
    client = broker.client()
    options = {"cayu-cloud": CloudIsolatedOptions(broker=client, poll_interval_s=0)}
    try:
        with pytest.raises(RuntimeError, match="raised inside"):
            async with open_isolated_runner(options=options, environ=CLOUD_ENV):
                raise RuntimeError("raised inside")
        assert broker.live() == []

        ready = client.wait_until_ready

        async def public_web(sandbox_id, **kwargs):
            record = await ready(sandbox_id, **kwargs)
            egress = record.egress.model_copy(update={"mode": "public_web"})
            return record.model_copy(update={"egress": egress})

        client.wait_until_ready = public_web  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="'public_web' egress"):
            async with open_isolated_runner(options=options, environ=CLOUD_ENV):
                raise AssertionError("a sandbox with egress must not be used")
        assert broker.live() == []
    finally:
        await client.aclose()
