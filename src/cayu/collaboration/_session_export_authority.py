"""Compose source policy with registered mandate authority outside transactions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial

from cayu.collaboration._contracts import ObjectRef, OwnerRef
from cayu.collaboration._mandate_validation import (
    MandateInput,
    MandateUse,
    validate_mandate_resolution,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.exports import (
    SessionExportAccessContext,
    SessionExportAction,
    SessionExportAuthorization,
    SessionExportDenied,
    SessionExportRegistration,
    SessionExportRequest,
    SessionExportRuntimeOrigin,
)
from cayu.collaboration.mandates import (
    MandateDenied,
    MandateResolution,
    ResourceSelector,
    ResourceSelectorOwner,
)
from cayu.vaults.redaction import SecretRedactor


def selected_resources(
    owner: OwnerRef, request: SessionExportRequest
) -> tuple[ResourceSelector, ...]:
    return tuple(
        ResourceSelector(
            resource=ObjectRef(
                owner=owner,
                kind="session_transcript_row",
                object_id=request.ref.session_id,
                incarnation=request.ref.session_instance_id,
                revision=index + 1,
            )
        )
        for index in request.source_indices
    )


@asynccontextmanager
async def acquire_export_authority(
    registration: SessionExportRegistration,
    context: SessionExportAccessContext,
    *,
    session_id: str,
    session_instance_id: str,
    actions: tuple[SessionExportAction, ...],
    audience: OwnerRef | None,
    request: SessionExportRequest | None,
    redactor: SecretRedactor,
    clock: Callable[[SessionExportAuthorization], Awaitable[datetime]],
    resolver_ref: ObjectRef | None,
    resource_owners: Mapping[OwnerRef, ResourceSelectorOwner],
    runtime_origin: SessionExportRuntimeOrigin | None = None,
    participant_access: Callable[[MandateResolution], Awaitable[None]] | None = None,
) -> AsyncIterator[SessionExportAuthorization]:
    resolver = registration.mandates
    if (resolver is None) != (context.mandate is None):
        raise SessionExportDenied()
    acquire = (
        registration.policy.acquire
        if runtime_origin is None
        else partial(registration.policy.acquire_runtime, origin=runtime_origin)
    )
    async with acquire(
        context,
        session_id=session_id,
        session_instance_id=session_instance_id,
        actions=actions,
        audience=audience,
    ) as raw:
        authorization = prepare_contract(SessionExportAuthorization, raw, redactor=redactor)
        # Source policy values cannot smuggle a delegated authority past the
        # separately registered resolver boundary.
        if authorization.mandate is not None or authorization.runtime is not None:
            raise SessionExportDenied()
        if resolver is None:
            yield prepare_contract(
                SessionExportAuthorization,
                authorization.model_copy(update={"runtime": runtime_origin}),
                redactor=redactor,
            )
            return
        if resolver_ref is None:
            raise SessionExportDenied()
        assert context.mandate is not None
        async with resolver.acquire(context.mandate) as raw_resolution:
            resolution = prepare_contract(MandateResolution, raw_resolution, redactor=redactor)
            now = await clock(authorization)
            mapped_actions = tuple(
                "administer"
                if action == "initialize"
                else "publish"
                if action == "export"
                else action
                for action in actions
            )
            resources = (
                selected_resources(registration.owner, request)
                if request is not None and ("source" in actions or "expose" in actions)
                else ()
            )
            inputs = tuple(
                MandateInput(source=selector.resource, channel="source") for selector in resources
            )
            if (
                request is not None
                and request.release is not None
                and ("source" in actions or "expose" in actions)
            ):
                # This is proposed provenance until the release owner authenticates
                # its exact manifest. It can restrict admission here, never grant
                # disclosure independently of that held review-owner guard.
                exposures = request.release.exposure
                resources = tuple(
                    dict.fromkeys(
                        (
                            *resources,
                            *(ResourceSelector(resource=item.source) for item in exposures),
                        )
                    )
                )
                inputs = tuple(
                    dict.fromkeys(
                        (
                            *inputs,
                            *(
                                MandateInput(source=item.source, channel=item.channel)
                                for item in exposures
                            ),
                        )
                    )
                )
            use = prepare_contract(
                MandateUse,
                {
                    "audience": audience or registration.owner,
                    "scope": registration.owner.application_scope,
                    "actions": mapped_actions,
                    "resources": resources,
                    "inputs": inputs,
                },
                redactor=redactor,
            )
            failed = False
            try:
                resolution = await asyncio.to_thread(
                    partial(
                        validate_mandate_resolution,
                        resolution,
                        context=context.mandate,
                        resolver=resolver_ref,
                        use=use,
                        now_ms=int(now.timestamp() * 1000),
                        resource_owners=resource_owners,
                        redactor=redactor,
                    )
                )
            except MandateDenied:
                failed = True
            if failed:
                raise SessionExportDenied()
            if context.mandate.participant is not None:
                if participant_access is None:
                    raise SessionExportDenied()
                await participant_access(resolution)
            yield prepare_contract(
                SessionExportAuthorization,
                authorization.model_copy(update={"mandate": resolution, "runtime": runtime_origin}),
                redactor=redactor,
            )
