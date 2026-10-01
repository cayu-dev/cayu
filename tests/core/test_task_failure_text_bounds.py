from types import SimpleNamespace

from cayu.tasks import worker


def _app():
    return SimpleNamespace(redact_json=lambda value: value)


def test_short_task_failure_text_is_unchanged() -> None:
    text = "validation failed: " + "detail " * 400
    assert len(text.encode()) < worker._MAX_TASK_FAILURE_MESSAGE_BYTES
    assert worker._redact_and_bound_task_failure_text(_app(), text) == text


def test_long_task_failure_text_is_bounded_and_marked() -> None:
    text = "é" * 10_000

    bounded = worker._redact_and_bound_task_failure_text(_app(), text)

    assert bounded.endswith("…[truncated]")
    assert len(bounded.encode()) <= worker._MAX_TASK_FAILURE_MESSAGE_BYTES
    assert bounded.startswith("é" * 1_000)
