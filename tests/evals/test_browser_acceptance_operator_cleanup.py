"""Acceptance harness cleanup keeps independent owners and original failures."""

import asyncio
from types import SimpleNamespace

import pytest

from cayu.evals.internal.browser_acceptance_operator_server import settle_operator_fixture


def application(name, calls, *, drain=True, close_error=None):
    async def drain_environment_cleanups(*, timeout_s):
        assert timeout_s == 30
        calls.append((name, "drain"))
        if isinstance(drain, BaseException):
            raise drain
        return drain

    async def close():
        calls.append((name, "close"))
        if close_error is not None:
            raise close_error

    return SimpleNamespace(
        drain_environment_cleanups=drain_environment_cleanups,
        session_store=SimpleNamespace(close=close),
    )


def test_failed_drain_does_not_skip_other_owners_or_mask_primary_failure():
    async def scenario():
        calls = []
        primary = ValueError("primary")
        drain_error = RuntimeError("drain")
        server_error = RuntimeError("server")
        close_error = RuntimeError("close")
        apps = [
            application("first", calls, drain=drain_error),
            application("second", calls, close_error=close_error),
        ]
        server = SimpleNamespace(should_exit=False)

        async def stop():
            assert server.should_exit
            calls.append(("server", "stop"))
            raise server_error

        with pytest.raises(ExceptionGroup) as caught:
            await settle_operator_fixture(apps, server, stop(), primary)
        assert caught.value.exceptions == (primary, drain_error, server_error, close_error)
        assert calls == [
            ("first", "drain"),
            ("second", "drain"),
            ("server", "stop"),
            ("second", "close"),
        ]

    asyncio.run(scenario())


def test_example_retirement_preserves_cancellation_from_shared_drain():
    from examples.browser_acceptance.local_authenticated import _retire_fixture

    async def scenario():
        entered = asyncio.Event()
        calls = []
        app = application("app", calls)
        primary = ValueError("trial failed")
        secondary = RuntimeError("client close failed")

        async def drain(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        async def close():
            calls.append("client")
            raise secondary

        app.drain_environment_cleanups = drain
        task = asyncio.create_task(
            _retire_fixture(app, None, None, SimpleNamespace(aclose=close), None, None, primary)
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled() and task.cancelling() == 1
        assert caught.value.__cause__.exceptions == (primary, secondary)
        assert calls == ["client"]

    asyncio.run(scenario())


def test_unsettled_drain_does_not_close_its_store():
    async def scenario():
        calls = []
        server = SimpleNamespace(should_exit=False)

        async def stop():
            assert server.should_exit

        with pytest.raises(RuntimeError, match="unsettled"):
            await settle_operator_fixture([application("app", calls, drain=False)], server, stop())
        assert calls == [("app", "drain")]

    asyncio.run(scenario())


@pytest.mark.parametrize("fatal", [False, True])
def test_delivered_cancellation_survives_cleanup_failure(fatal):
    async def scenario():
        class FatalCleanup(BaseException):
            pass

        started = asyncio.Event()
        delivered = []
        calls = []
        close_error = FatalCleanup("close") if fatal else RuntimeError("close")
        server = SimpleNamespace(should_exit=False)

        async def stop():
            assert server.should_exit

        async def owner():
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError as failure:
                delivered.append(failure)
                await settle_operator_fixture(
                    [application("app", calls, close_error=close_error)],
                    server,
                    stop(),
                    failure,
                )
                raise

        task = asyncio.create_task(owner())
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError as failure:
            assert not fatal
            assert failure is delivered[0]
            assert isinstance(failure.__cause__, ExceptionGroup)
            assert failure.__cause__.exceptions == (close_error,)
        except BaseExceptionGroup as failure:
            assert fatal
            assert failure.exceptions == (delivered[0], close_error)
        else:
            pytest.fail("Owner cancellation was lost")
        assert task.cancelling() == 1
        assert task.cancelled() is (not fatal)
        assert calls == [("app", "drain"), ("app", "close")]

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["drain", "server", "store"])
@pytest.mark.parametrize("primary_failed", [False, True])
@pytest.mark.parametrize("fatal", [False, True])
def test_cancellation_delivered_inside_cleanup_preserves_owner(phase, primary_failed, fatal):
    async def scenario():
        class FatalCleanup(BaseException):
            pass

        entered = asyncio.Event()
        delivered = []
        primary = ValueError("trial failure") if primary_failed else None
        secondary = FatalCleanup("fatal cleanup") if fatal else RuntimeError("cleanup failure")
        calls = []

        async def pause():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                delivered.append(error)
                raise

        first = application("first", calls)
        if phase == "drain":

            async def drain(**kwargs):
                await pause()

            first.drain_environment_cleanups = drain
        if phase == "store":
            first.session_store.close = pause

        async def server():
            if phase == "server":
                await pause()

        second = application("second", calls, close_error=secondary)
        task = asyncio.create_task(
            settle_operator_fixture([first, second], None, server(), primary)
        )
        await entered.wait()
        task.cancel()
        if fatal:
            with pytest.raises(BaseExceptionGroup) as error:
                await task
            expected = ([primary] if primary is not None else []) + [delivered[0], secondary]
            assert list(error.value.exceptions) == expected
            assert not task.cancelled()
        else:
            try:
                await task
            except asyncio.CancelledError as error:
                assert error is delivered[0]
                assert list(error.__cause__.exceptions) == ([primary] if primary else []) + [
                    secondary
                ]
            else:
                pytest.fail("Owner cancellation was converted to an ordinary failure")
            assert task.cancelled()
        assert task.cancelling() == 1
        assert ("second", "close") in calls
        if phase == "drain":
            assert ("first", "close") not in calls

    asyncio.run(scenario())
