"""Private deterministic validation signal and source-owned durable rejection."""

from cayu.collaboration._session_export_store import (
    ExportPreparation,
    ExportValidationFailure,
    operation_key,
)
from cayu.collaboration.exports import (
    SessionExportCapacityExceeded,
    SessionExportDenied,
    SessionExportUnavailable,
)


class _ProjectionRejected(SessionExportDenied):
    def __init__(self, reason):
        super().__init__()
        self.reason = reason


class _ProjectionTooLarge(SessionExportCapacityExceeded):
    reason = "projection_too_large"


class _SourceTooLarge(SessionExportCapacityExceeded):
    reason = "source_too_large"


def validation_error(reason):
    if reason == "source_too_large":
        return _SourceTooLarge()
    return (
        _ProjectionTooLarge() if reason == "projection_too_large" else _ProjectionRejected(reason)
    )


async def retain_validation_failure(exports, session, preparation, reason, authorization, source):
    if preparation is None:
        return
    failure = ExportValidationFailure(
        request=preparation.admission.request,
        source_commitment=preparation.admission.source_commitment,
        reason=reason,
    )
    updated = exports.prepare(
        ExportPreparation, preparation.model_copy(update={"validation_failure": failure})
    )
    root = await exports.root(session)
    if root is None:
        raise SessionExportUnavailable()
    exports.validate_namespace(root, failure.request)
    await exports.publish(
        session,
        root,
        root,
        operation_key(failure.request.ref.operation),
        updated.model_dump(mode="json"),
        authorization,
        [],
        source,
        expected_old=preparation.model_dump(mode="json"),
    )
