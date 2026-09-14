"""Public closure projection, separate from private deletion/replay authority."""

from collections.abc import Callable

from cayu.runtime._session_closure_records import (
    SESSION_CLOSURE_NATIVE_CLASSES,
    ClosureRecordsBuilder,
)
from cayu.runtime.session_closure import (
    SessionClosureExport,
    SessionClosureManifest,
    SessionClosureRecord,
    SessionClosureReport,
)
from cayu.vaults.redaction import SecretRedactor


def project_closure_manifest(
    manifest: SessionClosureManifest,
    *,
    redactor: SecretRedactor,
    project_session_id: Callable[[str], str],
) -> SessionClosureManifest:
    """Preserve typed controls; redact extension text and opaque metadata.

    The source manifest remains private, so public aliases never become deletion
    identities or replace the authority saved in an exact-operation receipt.
    """
    records = tuple(
        SessionClosureRecord(
            store_id=(
                "session-store"
                if record.store_id == "session-store"
                else redactor.redact_text(record.store_id)
            ),
            record_class=(
                record.record_class
                if record.store_id == "session-store"
                and record.record_class in (*SESSION_CLOSURE_NATIVE_CLASSES, "child_sessions")
                else redactor.redact_text(record.record_class)
            ),
            disposition=record.disposition,
            count=record.count,
            bytes=record.bytes,
            detail=None if record.detail is None else redactor.redact_text(record.detail),
            capability=(
                None if record.capability is None else redactor.redact_text(record.capability)
            ),
            erasure_blocked=record.erasure_blocked,
        )
        for record in manifest.records
    )
    return SessionClosureManifest(
        schema_version=manifest.schema_version,
        session_id=project_session_id(manifest.session_id),
        operation=manifest.operation,
        plan_id=manifest.plan_id,
        generated_at=manifest.generated_at,
        complete=manifest.complete,
        records=records,
        metadata=redactor.redact_json(manifest.metadata),
    )


def project_closure_report(
    report: SessionClosureReport,
    *,
    redactor: SecretRedactor,
    project_session_id: Callable[[str], str],
) -> SessionClosureReport:
    return SessionClosureReport(
        schema_version=report.schema_version,
        session_id=project_session_id(report.session_id),
        plan_id=report.plan_id,
        operation=report.operation,
        complete=report.complete,
        already_absent=report.already_absent,
        manifest=project_closure_manifest(
            report.manifest, redactor=redactor, project_session_id=project_session_id
        ),
        error=None if report.error is None else redactor.redact_text(report.error),
    )


def project_closure_export(
    export: SessionClosureExport,
    *,
    redactor: SecretRedactor,
    project_session_id: Callable[[str], str],
) -> SessionClosureExport:
    # Records have already passed bounded defensive copying. Never serialize an
    # extension-owned object in order to discover whether it is safe to expose.
    projected_records = {}
    public_native = None
    manifest = export.manifest
    native = export.session_records.get("session-store/session")
    if isinstance(native, dict) and set(native) == {
        "schema_version",
        "records",
        "counts",
        "record_bytes",
    }:
        # This envelope was authenticated by the coordinator's native snapshot
        # boundary. Preserve its fixed structure while projecting each payload;
        # recompute byte evidence after redaction, which may expand or shrink it.
        builder = ClosureRecordsBuilder(max_records=100_000, max_bytes=export._max_bytes)
        for name in SESSION_CLOSURE_NATIVE_CLASSES:
            builder.add_class(name, (redactor.redact_json(row) for row in native["records"][name]))
        public_native = builder.finish()
        manifest = manifest.model_copy(
            update={
                "records": tuple(
                    record.model_copy(
                        update={
                            "count": public_native["counts"][record.record_class],
                            "bytes": public_native["record_bytes"][record.record_class],
                        }
                    )
                    if record.store_id == "session-store"
                    and record.record_class in SESSION_CLOSURE_NATIVE_CLASSES
                    else record
                    for record in manifest.records
                )
            }
        )
    for key, value in export.session_records.items():
        public_key = key if key == "session-store/session" else redactor.redact_text(key)
        if public_key in projected_records:
            raise ValueError("Closure export store identities collide after projection.")
        projected_records[public_key] = (
            public_native
            if key == "session-store/session" and public_native is not None
            else redactor.redact_json(value)
        )
    projected = SessionClosureExport(
        manifest=project_closure_manifest(
            manifest, redactor=redactor, project_session_id=project_session_id
        ),
        content_redacted=True,
        session_records=projected_records,
    )
    projected._max_bytes = export._max_bytes
    projected.to_bytes()
    return projected
