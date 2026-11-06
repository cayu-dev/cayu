"""The instrumented stores must take the same Runtime paths as the plain ones."""

from __future__ import annotations

import inspect

import pytest
from tests.fault_injection.harness import Injector, ManualClock, session_store, task_store

from cayu.runtime._task_store_operation_boundary import (
    _task_store_method_has_stable_concrete_implementation,
    task_store_mutation_is_cancellation_quiescent,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.store import (
    _session_continuation_methods_owned,
    _session_export_methods_owned,
)
from cayu.tasks.memory import InMemoryTaskStore

pytestmark = pytest.mark.fault_injection

# The wrapper declares this base-class coroutine on its own class, so the
# structural proofs accept it; no Runtime path probes it today.
_KNOWN_TASK_PROOF_DIFFERENCES = frozenset({"_task_admission_wakeup"})


def _capability_probes(store: object) -> dict[str, object]:
    return {
        name: getattr(store, name)()
        for name in dir(type(store))
        if (name.startswith("_supports_") and name.endswith("_protocol"))
        or name == "supports_external_waits"
    }


def _capability_declarations(store: object) -> dict[str, object]:
    return {
        name: getattr(store, name)
        for name in dir(type(store))
        if name.endswith(("_version", "_quiescent")) and not callable(getattr(store, name))
    }


def test_instrumented_session_store_keeps_the_plain_store_capabilities() -> None:
    plain = InMemorySessionStore(ownership_clock=ManualClock())
    instrumented = session_store(Injector(), ManualClock())

    assert _capability_probes(instrumented) == _capability_probes(plain)
    assert _capability_declarations(instrumented) == _capability_declarations(plain)
    assert _session_export_methods_owned(instrumented) == _session_export_methods_owned(plain)
    assert _session_continuation_methods_owned(instrumented) == (
        _session_continuation_methods_owned(plain)
    )


def test_instrumented_task_store_keeps_the_plain_store_capabilities() -> None:
    clock = ManualClock()
    plain = InMemoryTaskStore(clock=clock, ownership_clock=clock)
    instrumented = task_store(Injector(), ManualClock())

    assert _capability_probes(instrumented) == _capability_probes(plain)
    assert _capability_declarations(instrumented) == _capability_declarations(plain)
    differences = {
        name
        for name in dir(InMemoryTaskStore)
        if inspect.iscoroutinefunction(inspect.getattr_static(InMemoryTaskStore, name))
        and (
            task_store_mutation_is_cancellation_quiescent(instrumented, name),
            _task_store_method_has_stable_concrete_implementation(instrumented, name),
        )
        != (
            task_store_mutation_is_cancellation_quiescent(plain, name),
            _task_store_method_has_stable_concrete_implementation(plain, name),
        )
    }
    assert differences == _KNOWN_TASK_PROOF_DIFFERENCES
