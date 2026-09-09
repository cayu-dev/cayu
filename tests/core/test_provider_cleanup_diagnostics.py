from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from cayu.core import AgentSpec, EventType, Message
from cayu.providers._credential_boundary import (
    aclosing_provider_stream,
    copy_provider_cancellation_failures,
    provider_cancellation_failures,
)
from cayu.providers.base import ModelProvider, ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    EventQuery,
    IncompleteSessionRecoveryRequest,
    RunRequest,
)
from cayu.storage.sqlite import SQLiteSessionStore

CANARY = "secret-url-header-prompt-credential"


class HostileError(Exception):
    def __str__(self):
        raise AssertionError("format hook invoked")

    def __repr__(self):
        raise AssertionError("repr hook invoked")

    def __getattribute__(self, name):
        if name in {"provider", "status_code", "error_code", "request_id", "__dict__"}:
            raise AssertionError("attribute hook invoked")
        return super().__getattribute__(name)


class InjectedStream:
    def __init__(self, failure: BaseException | None) -> None:
        self.failure = failure
        self.started = asyncio.Event()
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


@pytest.mark.parametrize(
    ("failure", "reason", "exception_type"),
    [
        (RuntimeError(CANARY), "close_exception", "RuntimeError"),
        (TimeoutError(CANARY), "cleanup_timeout", "TimeoutError"),
        (httpx.CloseError(CANARY), "close_exception", "CloseError"),
        (httpx.ReadTimeout(CANARY), "cleanup_timeout", "ReadTimeout"),
        (HostileError(CANARY), "unknown_exception", "unknown"),
        (None, None, None),
    ],
    ids=["known-close", "timeout", "httpx-close", "httpx-timeout", "hostile", "success"],
)
def test_cleanup_diagnostics_preserve_cancellation(failure, reason, exception_type):
    async def scenario():
        stream = InjectedStream(failure)

        async def consume():
            async with aclosing_provider_stream(stream):
                await anext(stream)

        task = asyncio.create_task(consume())
        await stream.started.wait()
        task.cancel("original caller cancellation")
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, 1)
        assert task.cancelling() == 1
        assert stream.close_calls == 1
        failures = provider_cancellation_failures(raised.value)
        if reason is None:
            assert failures == ()
            return
        assert failures[0]["cleanup_exception_message"] == "redacted"
        assert failures[0]["cleanup_cause_type"] == "CancelledError"
        assert failures[0]["cleanup_cause_message"] == "redacted"
        stack = json.loads(failures[0]["cleanup_local_stack"])
        assert 1 <= len(stack) <= 8
        assert all(name == "_credential_boundary.py" and line > 0 for name, line in stack)
        extended = {
            "cleanup_exception_message",
            "cleanup_cause_type",
            "cleanup_cause_message",
            "cleanup_local_stack",
        }
        assert tuple(
            {k: v for k, v in failure.items() if k not in extended} for failure in failures
        ) == (
            {
                "phase": "provider_stream_cleanup",
                "error": "Provider stream cleanup did not complete normally.",
                "error_type": "ProviderStreamCleanupError",
                "cleanup_diagnostic_version": 1,
                "cleanup_action": "stream_close",
                "cleanup_reason": reason,
                "cleanup_exception_type": exception_type,
                "cancellation_requested": True,
                "stream_close_state": "not_confirmed",
                "remote_cancellation_state": "unknown",
                "remote_settlement_state": "unknown",
            },
        )
        assert CANARY not in json.dumps(failures)
        assert copy_provider_cancellation_failures(json.loads(json.dumps(failures))) == failures

    asyncio.run(scenario())


def test_second_cancellation_during_cleanup_is_classified():
    async def scenario():
        closing = asyncio.Event()

        class Stream(InjectedStream):
            async def aclose(self):
                closing.set()
                await asyncio.Event().wait()

        stream = Stream(None)

        async def consume():
            async with aclosing_provider_stream(stream):
                await anext(stream)

        task = asyncio.create_task(consume())
        await stream.started.wait()
        task.cancel("first cancellation")
        await closing.wait()
        task.cancel("second cancellation")
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, 1)
        assert task.cancelling() == 2
        (failure,) = provider_cancellation_failures(raised.value)
        assert failure["cleanup_reason"] == "cleanup_cancelled"
        assert failure["cleanup_exception_type"] == "CancelledError"
        assert failure["stream_close_state"] == "not_confirmed"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "provider,code", [("openai", "rate_limit_exceeded"), ("anthropic", "overloaded_error")]
)
def test_provider_specific_cleanup_fields_are_allowlisted(provider, code):
    from cayu.providers._cleanup_diagnostics import cleanup_diagnostics

    error = ModelProviderError(
        CANARY,
        provider=provider,
        status_code=503,
        error_code=code,
        request_id=CANARY,
        response_body=CANARY,
        error_type=CANARY,
    )
    fields = cleanup_diagnostics(error, unsettled=False, action="stream_close")
    assert fields["cleanup_provider"] == provider
    assert fields["cleanup_status_code"] == 503
    assert fields["cleanup_error_code"] == code
    diagnostic = {
        "phase": "provider_stream_cleanup",
        "error": "Provider stream cleanup did not complete normally.",
        "error_type": "ProviderStreamCleanupError",
        **fields,
        "model_step_id": "mstep_" + "1" * 32,
        "model_attempt_id": "matt_" + "2" * 32,
        "cleanup_cause_type": "RuntimeError",
        "cleanup_cause_message": "redacted",
        "cleanup_local_stack": '[["_credential_boundary.py",1]]',
    }
    assert copy_provider_cancellation_failures(json.loads(json.dumps([diagnostic]))) == (
        diagnostic,
    )
    assert CANARY not in json.dumps(fields)
    error.error_code = CANARY
    assert "cleanup_error_code" not in cleanup_diagnostics(
        error, unsettled=False, action="stream_close"
    )


@pytest.mark.parametrize("repair", [False, True], ids=["publication", "restart-repair"])
@pytest.mark.parametrize("deadline", [False, True], ids=["caller-cancel", "caller-deadline"])
def test_injected_cleanup_failure_survives_sqlite_recovery_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repair: bool, deadline: bool
):
    async def scenario():
        database = tmp_path / "cleanup.db"
        store = SQLiteSessionStore(database)
        stream = InjectedStream(TimeoutError(CANARY) if deadline else httpx.CloseError(CANARY))

        class Provider(ModelProvider):
            name = "injected"
            calls = 0

            def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
                self.calls += 1
                return stream

        provider = Provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="injected"))
        if repair:
            original = app._recovery_coordinator._emit_terminal_event_with_hooks

            async def fail_publication(request):
                if request.event.type is EventType.SESSION_INTERRUPTED:
                    raise RuntimeError("injected terminal publication failure")
                async for event in original(request):
                    yield event

            monkeypatch.setattr(
                app._recovery_coordinator, "_emit_terminal_event_with_hooks", fail_publication
            )

        timeout_slot = []

        async def consume():
            async with asyncio.timeout(None) as timeout:
                timeout_slot.append(timeout)
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="cleanup",
                        messages=[Message.text("user", "hello")],
                    )
                ):
                    pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(stream.started.wait(), 10)
        if deadline:
            timeout_slot[0].reschedule(asyncio.get_running_loop().time())
        else:
            task.cancel("original cancellation")
        with pytest.raises(TimeoutError if deadline else asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert provider.calls == 1
        assert stream.close_calls == 1
        checkpoint = await store.load_checkpoint("cleanup")
        marker = checkpoint.get("pending_session_interrupt") if checkpoint else None
        assert bool(marker) is repair
        await store.close()
        reopened = SQLiteSessionStore(database)
        try:
            recovered_app = CayuApp(session_store=reopened, enable_logging=False)
            if repair:
                # No registered provider is needed or allowed for evidence repair.
                await recovered_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="cleanup")
                )
            records = await reopened.query_events(EventQuery(session_id="cleanup", limit=100))
            events = [
                recovered_app.project_event_record_for_exposure(record).event for record in records
            ]
            terminals = [
                event
                for event in events
                if event.type
                in {
                    EventType.SESSION_INTERRUPTED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_COMPLETED,
                }
            ]
            assert len(terminals) == 1
            interrupted = terminals[0]
            assert interrupted.type is EventType.SESSION_INTERRUPTED
            (failure,) = interrupted.payload["provider_cancellation_failures"]
            assert failure["cleanup_diagnostic_version"] == 1
            assert failure["cleanup_reason"] == (
                "cleanup_timeout" if deadline else "close_exception"
            )
            assert failure["cleanup_exception_type"] == (
                "TimeoutError" if deadline else "CloseError"
            )
            assert failure["stream_close_state"] == "not_confirmed"
            assert failure["remote_settlement_state"] == "unknown"
            raw_interrupted = next(
                record.event
                for record in records
                if record.event.type is EventType.SESSION_INTERRUPTED
            )
            (raw_failure,) = raw_interrupted.payload["provider_cancellation_failures"]
            started = next(
                record.event for record in records if record.event.type is EventType.MODEL_STARTED
            )
            for key in ("model_step_id", "model_attempt_id"):
                assert raw_failure[key] == started.payload[key]
                assert key not in failure
            # The durable attempt joins to its existing interaction-bearing model event.
            assert started.interaction_id is not None
            assert interrupted.payload["interruption_request_id"]
            assert "provider_operation_id" not in failure
            assert CANARY not in json.dumps([event.model_dump(mode="json") for event in events])
            if repair:
                assert (
                    raw_interrupted.payload["provider_cancellation_failures"]
                    == marker["provider_cancellation_failures"]
                )
                assert (
                    raw_interrupted.payload["interruption_request_id"]
                    == marker["interruption_request_id"]
                )
                await recovered_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="cleanup")
                )
                assert (
                    len(
                        await reopened.query_events(
                            EventQuery(
                                session_id="cleanup", event_type=EventType.SESSION_INTERRUPTED
                            )
                        )
                    )
                    == 1
                )
            assert provider.calls == 1
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_cleanup_lookup_failure_and_nested_diagnostics_are_bounded():
    async def scenario():
        class LookupFailure(InjectedStream):
            @property
            def aclose(self):
                raise ValueError(CANARY)

        outer = LookupFailure(None)
        inner = InjectedStream(RuntimeError(CANARY))

        async def consume():
            async with aclosing_provider_stream(outer), aclosing_provider_stream(inner):
                await anext(inner)

        task = asyncio.create_task(consume())
        await inner.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, 1)
        failures = provider_cancellation_failures(raised.value)
        # The first observed failure for a phase remains authoritative; nested
        # closes cannot grow an unbounded list or replace its known reason.
        assert len(failures) == 1
        assert failures[0]["cleanup_exception_type"] == "RuntimeError"
        assert len(json.dumps(failures)) < 1024

        async def lookup():
            async with aclosing_provider_stream(outer):
                await anext(outer)

        task = asyncio.create_task(lookup())
        await outer.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, 1)
        (failure,) = provider_cancellation_failures(raised.value)
        assert failure["cleanup_action"] == "stream_close_lookup"
        assert failure["cleanup_exception_type"] == "ValueError"
        assert failure["stream_close_state"] == "not_confirmed"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("cleanup_diagnostic_version", True),
        ("cleanup_diagnostic_version", 2),
        ("cleanup_reason", CANARY),
        ("cleanup_exception_type", CANARY),
        ("cleanup_status_code", 999999999999999999),
        ("cleanup_status_code", True),
        ("cleanup_error_code", CANARY),
        ("cleanup_provider", CANARY),
        ("remote_settlement_state", "confirmed"),
        ("stream_close_state", "completed"),
        ("cancellation_requested", False),
        ("response_body", CANARY),
        ("model_attempt_id", CANARY),
    ],
)
def test_reconstructed_cleanup_fields_fail_closed(field, value):
    from cayu.providers._cleanup_diagnostics import cleanup_diagnostics

    failure = {
        "phase": "provider_stream_cleanup",
        "error": "Provider stream cleanup did not complete normally.",
        "error_type": "ProviderStreamCleanupError",
        **cleanup_diagnostics(RuntimeError(CANARY), unsettled=False, action="stream_close"),
        field: value,
    }
    with pytest.raises(ValueError) as raised:
        copy_provider_cancellation_failures([failure])
    assert CANARY not in str(raised.value)
    with pytest.raises(ValueError, match="at most two"):
        copy_provider_cancellation_failures([failure] * 1000)


def test_reconstructed_cleanup_does_not_invoke_hostile_key_hooks():
    class Key:
        armed = False

        def __hash__(self):
            if self.armed:
                raise AssertionError("hash hook invoked")
            return 1

        def __eq__(self, other):
            raise AssertionError("equality hook invoked")

    key = Key()
    failure = {key: CANARY}
    key.armed = True
    with pytest.raises(ValueError, match="fields are invalid"):
        copy_provider_cancellation_failures([failure])


def test_owned_close_pending_is_not_reported_as_timeout_or_remote_settlement():
    from cayu.providers._credential_boundary import reserve_provider_stream_cleanup

    async def scenario():
        release = asyncio.Event()
        finished = asyncio.Event()

        class Stream(InjectedStream):
            async def aclose(self):
                await release.wait()
                finished.set()

        stream = Stream(None)

        async def consume():
            async with aclosing_provider_stream(
                stream, cleanup_ownership=reserve_provider_stream_cleanup()
            ):
                await anext(stream)

        task = asyncio.create_task(consume())
        await stream.started.wait()
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError) as raised:
                await asyncio.wait_for(task, 1)
            (failure,) = provider_cancellation_failures(raised.value)
            assert failure["cleanup_reason"] == "cleanup_pending"
            assert failure["cleanup_exception_type"] == "unknown"
            assert failure["stream_close_state"] == "pending"
            assert failure["remote_cancellation_state"] == "unknown"
            assert failure["remote_settlement_state"] == "unknown"
            assert not finished.is_set()
        finally:
            release.set()
            await asyncio.wait_for(finished.wait(), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "message",
    [
        "aclose(): asynchronous generator is already running",
        "anext(): asynchronous generator is already running",
        "Event loop is closed",
        CANARY,
    ],
)
def test_close_exception_message_and_cause_are_sanitized(message):
    from cayu.providers._cleanup_diagnostics import cleanup_diagnostics, copy_cleanup_diagnostics

    error = RuntimeError(message)
    error.__cause__ = httpx.CloseError(CANARY)
    fields = cleanup_diagnostics(error, unsettled=False, action="stream_close")
    assert fields["cleanup_exception_message"] == ("redacted" if message == CANARY else message)
    assert fields["cleanup_cause_type"] == "CloseError"
    assert fields["cleanup_cause_message"] == "redacted"
    assert fields["remote_settlement_state"] == "unknown"
    assert CANARY not in json.dumps(fields)
    assert copy_cleanup_diagnostics(fields) == fields


@pytest.mark.parametrize(
    "field,value",
    [
        ("cleanup_exception_message", CANARY),
        ("cleanup_cause_message", CANARY),
        ("cleanup_cause_type", CANARY),
        ("cleanup_local_stack", json.dumps([[CANARY, 1]])),
        ("cleanup_local_stack", json.dumps([["_http.py", True]])),
        ("cleanup_local_stack", json.dumps([["_http.py", 1]] * 9)),
    ],
)
def test_close_exception_evidence_rejects_untrusted_projection(field, value):
    from cayu.providers._cleanup_diagnostics import cleanup_diagnostics, copy_cleanup_diagnostics

    fields = cleanup_diagnostics(RuntimeError(), unsettled=False, action="stream_close")
    fields[field] = value
    with pytest.raises(ValueError):
        copy_cleanup_diagnostics(fields)


def test_actual_generator_reentrancy_has_distinct_sanitized_message():
    from cayu.providers._cleanup_diagnostics import cleanup_diagnostics

    async def scenario():
        reading = asyncio.Event()

        async def busy_stream():
            reading.set()
            await asyncio.Event().wait()
            yield None

        stream = busy_stream()
        pending = asyncio.create_task(anext(stream))
        try:
            await reading.wait()
            with pytest.raises(RuntimeError) as caught:
                await stream.aclose()
            fields = cleanup_diagnostics(caught.value, unsettled=False, action="stream_close")
            assert (
                fields["cleanup_exception_message"]
                == "aclose(): asynchronous generator is already running"
            )
            assert fields["cleanup_reason"] == "close_exception"
            assert fields["stream_close_state"] == "not_confirmed"
            assert fields["remote_settlement_state"] == "unknown"
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("path_style", ["posix", "windows"])
@pytest.mark.parametrize("location", ["provider", "foreign_directory", "foreign_filename"])
def test_cleanup_stack_matches_native_provider_paths(monkeypatch, path_style, location):
    from pathlib import PurePosixPath, PureWindowsPath

    import cayu.providers._cleanup_diagnostics as diagnostics

    path_type = PureWindowsPath if path_style == "windows" else PurePosixPath
    root = path_type(
        "C:/site-packages/cayu/providers"
        if path_style == "windows"
        else "/site-packages/cayu/providers"
    )
    filename = root / "_http.py"
    if location == "foreign_directory":
        filename = root.parent / "extension" / "_http.py"
    elif location == "foreign_filename":
        filename = root / "extension.py"
    monkeypatch.setattr(diagnostics, "_PROVIDER_ROOT", str(root))
    monkeypatch.setattr(diagnostics, "Path", path_type)
    try:
        exec(compile('raise RuntimeError("Event loop is closed")', str(filename), "exec"))
    except RuntimeError as error:
        fields = diagnostics.cleanup_diagnostics(error, unsettled=False, action="stream_close")
    if location == "provider":
        assert json.loads(fields["cleanup_local_stack"]) == [["_http.py", 1]]
    else:
        assert "cleanup_local_stack" not in fields
    assert diagnostics.copy_cleanup_diagnostics(fields) == fields
