"""Shared public diagnostic assertions for native no-mutation handoffs."""

import logging
import traceback


def private_preparation_failure(coordinator, monkeypatch):
    secret = "private-preparation-credential-canary"
    monkeypatch.setattr(coordinator, "_redactor", coordinator._redactor.with_secret(secret))
    first = ConnectionError("read failed: " + secret)
    first.add_note("private note: " + secret)
    second = OSError("cleanup failed: " + secret)
    first.__cause__ = ValueError("cause: " + secret)
    first.__context__ = RuntimeError("context: " + secret)
    error = ExceptionGroup(secret, [first, ExceptionGroup(secret, [second])])
    return error, secret


def assert_safe_preparation_failure(error, original, secret, captured_warnings, capsys, caplog):
    from cayu.collaboration.participants import CollaborationUnavailable

    assert isinstance(error, CollaborationUnavailable)
    evidence = error.__cause__
    assert isinstance(evidence, ExceptionGroup) and evidence is not original
    assert len(evidence.exceptions) == 2
    first, nested = evidence.exceptions
    assert isinstance(first, ConnectionError)
    assert isinstance(first.__cause__, ValueError)
    assert isinstance(first.__context__, RuntimeError)
    assert isinstance(nested, ExceptionGroup) and len(nested.exceptions) == 1
    assert isinstance(nested.exceptions[0], OSError)
    assert error.__context__ is None
    graph = []
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        graph.extend((str(current), repr(current), repr(getattr(current, "__notes__", ()))))
        pending.extend(
            linked for linked in (current.__cause__, current.__context__) if linked is not None
        )
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    assert len(seen) == 7  # public wrapper, two groups and four ordered causal leaves
    rendered = "".join(traceback.format_exception(error))
    logging.getLogger(__name__).error("Public host failure: %s", rendered)
    streams = capsys.readouterr()
    assert secret not in (
        rendered
        + str(error)
        + repr(error)
        + caplog.text
        + streams.out
        + streams.err
        + "".join(graph)
        + "".join(str(item.message) for item in captured_warnings)
    )
    assert secret in str(original.exceptions[0])  # original owner evidence was not rewritten
