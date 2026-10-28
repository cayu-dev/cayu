from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import RLock

_DIAGNOSTIC_STATE_LOCK = RLock()


class DiagnosticStoreInspectionChanged(RuntimeError):
    """Raised when a supposedly static diagnostic snapshot changed in flight."""


class DiagnosticStoreInspection:
    """Own post-collection checks for diagnostic-only store inspection."""

    def __init__(self) -> None:
        self._verifiers: list[Callable[[], None]] = []
        self._verified = False

    def add_verifier(self, verifier: Callable[[], None]) -> None:
        with _DIAGNOSTIC_STATE_LOCK:
            if self._verified:
                raise RuntimeError("Diagnostic store inspection was already verified.")
            self._verifiers.append(verifier)

    def verify(self) -> None:
        with _DIAGNOSTIC_STATE_LOCK:
            if self._verified:
                raise RuntimeError("Diagnostic store inspection may only be verified once.")
            self._verified = True
            verifiers = tuple(self._verifiers)
        failures: list[Exception] = []
        for verifier in verifiers:
            try:
                verifier()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup(
                "Diagnostic store inspection changed during collection.",
                failures,
            )


_CURRENT_DIAGNOSTIC_STORE_INSPECTION: DiagnosticStoreInspection | None = None


def alias_keyring_optional(*, requested_read_only: bool) -> bool:
    """Whether a session store may open a keyed alias registry without its keyring.

    Every store backend decides this here, from the ``read_only`` its caller
    passed, before diagnostic inspection can force the store read-only. A store
    its caller opened read-only never writes, so it can't leave rows without
    aliases, and its alias operations fail closed without a codec. Diagnostic
    inspection (``cayu doctor``) forces the app's stores read-only to stand in
    for the writable stores the app really opens, so those keep the writable
    check and doctor still reports a missing keyring.
    """

    if type(requested_read_only) is not bool:
        raise TypeError("requested_read_only must be a bool.")
    return requested_read_only


def current_diagnostic_store_inspection() -> DiagnosticStoreInspection | None:
    with _DIAGNOSTIC_STATE_LOCK:
        return _CURRENT_DIAGNOSTIC_STORE_INSPECTION


@contextmanager
def diagnostic_store_inspection() -> Iterator[DiagnosticStoreInspection]:
    global _CURRENT_DIAGNOSTIC_STORE_INSPECTION

    inspection = DiagnosticStoreInspection()
    with _DIAGNOSTIC_STATE_LOCK:
        if _CURRENT_DIAGNOSTIC_STORE_INSPECTION is not None:
            raise RuntimeError("Diagnostic store inspection contexts may not be nested.")
        _CURRENT_DIAGNOSTIC_STORE_INSPECTION = inspection
    try:
        yield inspection
    finally:
        with _DIAGNOSTIC_STATE_LOCK:
            if _CURRENT_DIAGNOSTIC_STORE_INSPECTION is not inspection:
                raise RuntimeError("Diagnostic store inspection ownership changed.")
            _CURRENT_DIAGNOSTIC_STORE_INSPECTION = None


__all__ = [
    "DiagnosticStoreInspection",
    "DiagnosticStoreInspectionChanged",
    "current_diagnostic_store_inspection",
    "diagnostic_store_inspection",
]
