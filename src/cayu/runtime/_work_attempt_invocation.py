"""In-process work-attempt provenance carried by the invocation owner.

This is not a public authentication mechanism. The runtime constructs it only
after resolving and fencing the durable admission. A detached admission from a
caller is not made authoritative merely because its fields happen to match.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import TYPE_CHECKING, Never, SupportsIndex

from cayu.runtime.work_attempt_admission import (
    WorkAttemptAdmission,
    WorkAttemptAdmissionState,
    WorkAttemptExecutionClaimRequest,
    require_work_attempt_admission_result,
)

_WORK_ATTEMPT_INVOCATION_TOKEN = object()
_RECOVERY_OWNERSHIP_TOKEN = object()

if TYPE_CHECKING:
    from cayu.runtime.tasks import TaskStore


@dataclass(frozen=True, slots=True, init=False, repr=False, eq=False)
class WorkAttemptRecoveryOwnership:
    """Private acknowledged claim handoff, not reconstructable caller proof."""

    store: TaskStore
    _admission_json: str
    _request_json: str

    def __init__(
        self,
        store: TaskStore,
        admission: WorkAttemptAdmission,
        request: WorkAttemptExecutionClaimRequest,
        *,
        _token: object = None,
    ) -> None:
        if _token is not _RECOVERY_OWNERSHIP_TOKEN:
            raise TypeError("Recovery ownership requires the runtime claim owner.")
        object.__setattr__(self, "store", store)
        object.__setattr__(self, "_admission_json", admission.model_dump_json(warnings=False))
        object.__setattr__(self, "_request_json", request.model_dump_json(warnings=False))

    @property
    def admission(self) -> WorkAttemptAdmission:
        return WorkAttemptAdmission.model_validate_json(self._admission_json)

    @property
    def request(self) -> WorkAttemptExecutionClaimRequest:
        return WorkAttemptExecutionClaimRequest.model_validate_json(self._request_json)

    def __copy__(self) -> WorkAttemptRecoveryOwnership:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> WorkAttemptRecoveryOwnership:
        return self

    def __reduce_ex__(self, _protocol: SupportsIndex, /) -> Never:
        raise TypeError("Recovery ownership has no serialization form.")

    def __repr__(self) -> str:
        return "WorkAttemptRecoveryOwnership(<authenticated>)"


def _acknowledged_work_attempt_recovery(
    store: TaskStore, admission: WorkAttemptAdmission, request: WorkAttemptExecutionClaimRequest
) -> WorkAttemptRecoveryOwnership:
    return WorkAttemptRecoveryOwnership(store, admission, request, _token=_RECOVERY_OWNERSHIP_TOKEN)


class WorkAttemptInvocationAuthority:
    """Immutable captured admission; live lease checks remain with the store.

    Canonical bytes keep nested request dictionaries from becoming mutable
    authority. Every exposed admission is newly reconstructed. This object is
    intentionally neither a durable record nor transferable authentication.
    """

    __slots__ = ("_admission_json",)
    _admission_json: str

    def __init__(self, admission: WorkAttemptAdmission, *, _token: object = None) -> None:
        if _token is not _WORK_ATTEMPT_INVOCATION_TOKEN:
            raise TypeError("Work-attempt invocation authority requires the runtime owner.")
        copied = require_work_attempt_admission_result(
            admission, operation_name="Work-attempt invocation authority"
        )
        if (
            copied.state
            not in (
                WorkAttemptAdmissionState.ACTIVE,
                WorkAttemptAdmissionState.RECOVERING,
            )
            or copied.run_semantics is None
        ):
            raise ValueError("Work-attempt invocation requires admitted executable authority.")
        object.__setattr__(self, "_admission_json", copied.model_dump_json(warnings=False))

    @property
    def admission(self) -> WorkAttemptAdmission:
        return WorkAttemptAdmission.model_validate_json(self._admission_json)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("Work-attempt invocation authority is immutable.")

    def __copy__(self) -> WorkAttemptInvocationAuthority:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> WorkAttemptInvocationAuthority:
        return self

    def __reduce_ex__(self, _protocol: SupportsIndex, /) -> Never:
        raise TypeError("Work-attempt invocation authority has no serialization form.")

    def __repr__(self) -> str:
        return "WorkAttemptInvocationAuthority(<authenticated>)"


def _authenticated_work_attempt_invocation(
    admission: WorkAttemptAdmission,
) -> WorkAttemptInvocationAuthority:
    return WorkAttemptInvocationAuthority(admission, _token=_WORK_ATTEMPT_INVOCATION_TOKEN)
