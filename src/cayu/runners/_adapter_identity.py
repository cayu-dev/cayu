"""Process-wide runner adapter identities trusted in durable diagnostics.

Runner diagnostics, cleanup receipts, and unavailable-runner evidence report an
adapter name and an exception class name. Both are copied into durable
artifacts, so only explicitly registered values are published; anything else
fails closed to ``"unknown"`` (adapter) or ``"Exception"`` (error type).

Built-in adapters are seeded here without importing their optional SDKs.
External adapters opt in through ``register_runner_adapter_identity``; there is
no automatic discovery.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

UNKNOWN_RUNNER_ADAPTER = "unknown"
RUNNER_ADAPTER_NAME_MAX_LENGTH = 63
RUNNER_ERROR_TYPE_NAME_MAX_LENGTH = 128
RUNNER_ADAPTER_TRUSTED_ERROR_TYPES_MAX = 64

_ADAPTER_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_ERROR_TYPE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True, slots=True)
class RunnerAdapterIdentity:
    """One registered runner adapter identity.

    ``name`` is the value published as ``adapter`` in runner diagnostics and
    cleanup receipts. ``trusted_error_types`` are additional exception class
    names that the adapter asked Cayu to publish verbatim instead of the
    generic ``"Exception"`` classification. Built-in identities carry no extra
    names because their SDK exception names are part of the runtime baseline.
    """

    name: str
    trusted_error_types: frozenset[str] = frozenset()


BUILTIN_RUNNER_ADAPTER_NAMES = frozenset(
    {"docker", "e2b", "lambda-microvm", "local", "microsandbox"}
)

_registration_lock = threading.Lock()
# Copy-on-write snapshots: readers never take the lock and always observe one
# complete registration state.
_identities: Mapping[str, RunnerAdapterIdentity] = MappingProxyType(
    {name: RunnerAdapterIdentity(name=name) for name in sorted(BUILTIN_RUNNER_ADAPTER_NAMES)}
)
_adapter_names: frozenset[str] = BUILTIN_RUNNER_ADAPTER_NAMES
_extension_error_types: frozenset[str] = frozenset()


def register_runner_adapter_identity(
    name: str,
    *,
    trusted_error_types: Iterable[str] = (),
) -> RunnerAdapterIdentity:
    """Trust one runner adapter name, and optional error class names, process-wide.

    ``name`` must be a lowercase ASCII slug (letters, digits, single hyphens,
    starting with a letter, at most 63 characters) and must not be
    ``"unknown"``. Each trusted error type must be a Python identifier of at
    most 128 characters; at most 64 may be registered per adapter.

    Registering the same name again with the same error-type set returns the
    existing identity. Registering it with a different set raises
    ``ValueError``; built-in identities cannot be widened. Registration cannot
    be undone for the life of the process.
    """

    global _identities, _adapter_names, _extension_error_types

    adapter = _validate_adapter_name(name)
    error_types = _validate_error_type_names(trusted_error_types)
    identity = RunnerAdapterIdentity(name=adapter, trusted_error_types=error_types)
    with _registration_lock:
        existing = _identities.get(adapter)
        if existing is not None:
            if existing != identity:
                raise ValueError(
                    f"Runner adapter identity {adapter!r} is already registered "
                    "with different trusted error types."
                )
            return existing
        identities = dict(_identities)
        identities[adapter] = identity
        _identities = MappingProxyType(identities)
        _adapter_names = frozenset(identities)
        _extension_error_types = _extension_error_types | error_types
    return identity


def registered_runner_adapter_identities() -> Mapping[str, RunnerAdapterIdentity]:
    """Return a read-only snapshot of every registered identity, including built-ins."""

    return _identities


def trusted_runner_adapter_name(value: object) -> str:
    """Return a registered adapter name exactly, or ``"unknown"``."""

    if type(value) is str and value in _adapter_names:
        return value
    return UNKNOWN_RUNNER_ADAPTER


def is_registered_runner_error_type_name(value: object) -> bool:
    """Whether an adapter registered this exact error class name."""

    return type(value) is str and value in _extension_error_types


def _validate_adapter_name(name: object) -> str:
    if type(name) is not str:
        raise TypeError("Runner adapter name must be a string.")
    if (
        not 0 < len(name) <= RUNNER_ADAPTER_NAME_MAX_LENGTH
        or _ADAPTER_NAME_PATTERN.match(name) is None
    ):
        raise ValueError(
            "Runner adapter name must be a lowercase ASCII slug of at most "
            f"{RUNNER_ADAPTER_NAME_MAX_LENGTH} characters."
        )
    if name == UNKNOWN_RUNNER_ADAPTER:
        raise ValueError("Runner adapter name 'unknown' is reserved.")
    return name


def _validate_error_type_names(values: Iterable[str]) -> frozenset[str]:
    if type(values) is str or isinstance(values, (bytes, bytearray, Mapping)):
        raise TypeError("trusted_error_types must be an iterable of class names.")
    try:
        candidates = list(values)
    except TypeError:
        raise TypeError("trusted_error_types must be an iterable of class names.") from None
    if len(candidates) > RUNNER_ADAPTER_TRUSTED_ERROR_TYPES_MAX:
        raise ValueError(
            "trusted_error_types may contain at most "
            f"{RUNNER_ADAPTER_TRUSTED_ERROR_TYPES_MAX} names."
        )
    names: set[str] = set()
    for candidate in candidates:
        if type(candidate) is not str:
            raise TypeError("Trusted runner error type names must be strings.")
        if (
            not 0 < len(candidate) <= RUNNER_ERROR_TYPE_NAME_MAX_LENGTH
            or _ERROR_TYPE_NAME_PATTERN.match(candidate) is None
        ):
            raise ValueError(
                "Trusted runner error type names must be ASCII Python identifiers of at most "
                f"{RUNNER_ERROR_TYPE_NAME_MAX_LENGTH} characters."
            )
        names.add(candidate)
    return frozenset(names)
