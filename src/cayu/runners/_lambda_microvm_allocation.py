"""Recoverable Lambda MicroVM allocation through ``RunMicrovm`` client-token replay.

These are the provider-level parts of crash-safe allocation that every Cayu
factory over Lambda MicroVMs shares: the virtual-egress adapter
(``cayu.egress.aws_lambda_microvm_adapter``) and the plain environment factory
(``cayu.environments.aws_lambda_microvm``). A durable allocation intent pins,
before any provider call, the exact image version, a lifetime bound and a
replay window. The client token is derived from the intent's allocation id and
never stored, so the intent carries no submission authority and every recovery
worker reconstructs the same token. Within the replay window an identical
``RunMicrovm`` returns the original MicroVM instead of creating another, which
is how a create whose acknowledgement was lost is adopted. Cleanup terminates
the token-owned MicroVM with positive readback, or, once the window has
closed, waits for AWS's own clock to pass the pinned lifetime bound.

Callers pass the current time explicitly, so each keeps its own clock.
"""

from __future__ import annotations

import asyncio
import email.utils
from collections.abc import Mapping
from hashlib import sha256
from typing import Any, TypedDict

from cayu.runners.aws_lambda_microvm import (
    LAMBDA_SIGV4_REQUEST_VALIDITY_SECONDS,
    LambdaMicroVMError,
    LambdaMicroVMProtocolError,
    run_microvm_with_client_token,
    terminate_microvm_confirmed,
)

#: Longest time after intent preparation that Cayu submits or replays one
#: allocation's RunMicrovm client token. It must stay below AWS's token
#: retention, which is undocumented; a replay after retention could create a
#: second MicroVM. Live replays still returned the original MicroVM after more
#: than one hour, the configurable maximum.
DEFAULT_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS = 900
MAXIMUM_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS = 3_600
#: AWS's maximum MicroVM lifetime; recoverable allocations always pin a bound.
LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS = 28_800
#: Allowance for AWS's maximum-duration enforcement to finish terminating.
LIFETIME_EXPIRY_MARGIN_SECONDS = 300
REAP_TERMINATION_TIMEOUT_SECONDS = 8.0
ALLOCATION_METADATA_VERSION = 1
ALLOCATION_METADATA_KEYS = frozenset(
    {
        "version",
        "image_arn",
        "image_version",
        "maximum_duration_s",
        "prepared_at_s",
        "replay_window_s",
    }
)


class LambdaMicroVMAllocationRecoveryError(LambdaMicroVMError):
    """A recoverable allocation cannot be adopted or disposed without guessing.

    The durable intent remains owned and retryable. Cayu never answers this
    error by allocating a replacement MicroVM.
    """


class AllocationMetadata(TypedDict):
    version: int
    image_arn: str
    image_version: str
    maximum_duration_s: int
    prepared_at_s: int
    replay_window_s: int


def require_replay_window(value: object) -> int:
    if type(value) is not int or not 0 < value <= MAXIMUM_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS:
        raise ValueError(
            "client_token_replay_window_s must be an integer between 1 and "
            f"{MAXIMUM_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS} seconds."
        )
    return value


def require_maximum_duration(value: object) -> int:
    if type(value) is not int or not 0 < value <= LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS:
        raise ValueError(
            "Recoverable Lambda MicroVM allocation requires maximum_duration_in_seconds "
            f"between 1 and {LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS}."
        )
    return value


def client_token(adapter_generation: str, allocation_id: str) -> str:
    """Derive the RunMicrovm client token of one durable allocation intent."""

    # Derived rather than stored: the durable intent never carries submission
    # authority, and every recovery worker reconstructs the identical token.
    digest = sha256(
        f"cayu.lambda-microvm.run\0{adapter_generation}\0{allocation_id}".encode()
    ).hexdigest()
    return f"cayu-{digest}"


def pin_allocation_metadata(
    *,
    image_arn: str,
    image_version: str,
    maximum_duration_s: int,
    replay_window_s: int,
    now: float,
) -> AllocationMetadata:
    """Return the provider inputs an intent pins before its first dispatch."""

    return validated_allocation_metadata(
        {
            "version": ALLOCATION_METADATA_VERSION,
            "image_arn": image_arn,
            "image_version": image_version,
            "maximum_duration_s": require_maximum_duration(maximum_duration_s),
            # Floor so every elapsed-time decision errs toward "later".
            "prepared_at_s": int(now),
            "replay_window_s": require_replay_window(replay_window_s),
        }
    )


def validated_allocation_metadata(metadata: Mapping[str, Any]) -> AllocationMetadata:
    if not isinstance(metadata, Mapping) or set(metadata) != ALLOCATION_METADATA_KEYS:
        raise LambdaMicroVMAllocationRecoveryError(
            "Lambda MicroVM allocation intent is missing its pinned provider inputs."
        )
    if metadata["version"] != ALLOCATION_METADATA_VERSION:
        raise LambdaMicroVMAllocationRecoveryError(
            "Lambda MicroVM allocation intent has an unsupported metadata version."
        )
    for key in ("image_arn", "image_version"):
        value = metadata[key]
        if type(value) is not str or not value.strip():
            raise LambdaMicroVMAllocationRecoveryError(
                f"Lambda MicroVM allocation intent has an invalid {key}."
            )
    if not metadata["image_arn"].startswith("arn:"):
        raise LambdaMicroVMAllocationRecoveryError(
            "Lambda MicroVM allocation intent must pin an image ARN."
        )
    for key, upper in (
        ("maximum_duration_s", LAMBDA_MICROVM_MAXIMUM_DURATION_SECONDS),
        ("replay_window_s", MAXIMUM_CLIENT_TOKEN_REPLAY_WINDOW_SECONDS),
        ("prepared_at_s", None),
    ):
        value = metadata[key]
        if type(value) is not int or value <= 0 or (upper is not None and value > upper):
            raise LambdaMicroVMAllocationRecoveryError(
                f"Lambda MicroVM allocation intent has an invalid {key}."
            )
    return {
        "version": metadata["version"],
        "image_arn": metadata["image_arn"],
        "image_version": metadata["image_version"],
        "maximum_duration_s": metadata["maximum_duration_s"],
        "prepared_at_s": metadata["prepared_at_s"],
        "replay_window_s": metadata["replay_window_s"],
    }


def replay_deadline(metadata: AllocationMetadata) -> int:
    return metadata["prepared_at_s"] + metadata["replay_window_s"]


def lifetime_deadline(metadata: AllocationMetadata) -> int:
    # No request for the token is signed after the replay deadline, and AWS
    # accepts a SigV4 request for at most 15 minutes after signing, so any
    # token-owned MicroVM started by then; maximum-duration enforcement has
    # terminated it after this instant in AWS's time.
    return (
        replay_deadline(metadata)
        + LAMBDA_SIGV4_REQUEST_VALIDITY_SECONDS
        + metadata["maximum_duration_s"]
        + LIFETIME_EXPIRY_MARGIN_SECONDS
    )


def require_replay_window_open(metadata: AllocationMetadata, *, action: str, now: float) -> None:
    if now > replay_deadline(metadata):
        raise LambdaMicroVMAllocationRecoveryError(
            f"Cannot {action} Lambda MicroVM allocation: its client-token replay window "
            "has elapsed, so another submission could create a duplicate MicroVM."
        )


async def resolve_exact_image(
    control_client: Any,
    image: str,
    *,
    configured_version: object,
    unavailable_error: type[Exception] = LambdaMicroVMAllocationRecoveryError,
) -> tuple[str, str]:
    """Return the image ARN and the exact version an intent pins.

    ``unavailable_error`` is raised when the image has no active version.
    """

    if configured_version is not None and (
        type(configured_version) is not str or not configured_version.strip()
    ):
        raise ValueError("Lambda MicroVM image_version must be a nonblank string.")
    if image.startswith("arn:") and configured_version is not None:
        return image, configured_version
    response = await asyncio.to_thread(control_client.get_microvm_image, imageIdentifier=image)
    if not isinstance(response, Mapping):
        raise LambdaMicroVMProtocolError("get_microvm_image response must be an object.")
    image_arn = response.get("imageArn")
    version = configured_version or response.get("latestActiveImageVersion")
    if type(image_arn) is not str or not image_arn.startswith("arn:"):
        raise LambdaMicroVMProtocolError("get_microvm_image omitted a valid imageArn.")
    if type(version) is not str or not version.strip():
        raise unavailable_error(
            "Lambda MicroVM image has no active version to pin for recoverable allocation."
        )
    return image_arn, version


async def aws_server_time(client: Any, image_arn: str) -> float | None:
    """Return AWS's clock from a read response's ``Date`` header, or ``None``.

    An error response from AWS carries the same authoritative header.
    """

    try:
        response: Any = await asyncio.to_thread(client.get_microvm_image, imageIdentifier=image_arn)
    except Exception as exc:
        response = getattr(exc, "response", None)
    metadata = response.get("ResponseMetadata") if isinstance(response, Mapping) else None
    headers = metadata.get("HTTPHeaders") if isinstance(metadata, Mapping) else None
    value = headers.get("date") if isinstance(headers, Mapping) else None
    if type(value) is not str:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


async def reap_token_owned_microvm(
    control_client: Any,
    *,
    run_options: Mapping[str, Any],
    token: str,
    metadata: AllocationMetadata,
    acknowledged_microvm_id: object,
    now: float,
) -> None:
    """Terminate the token-owned MicroVM, or prove the platform already did.

    Within the pinned replay window the client token resolves the exact
    allocation (creating it only if the original submission never arrived)
    and the MicroVM is terminated with positive readback. No submission for
    the token is sent after the replay deadline, and AWS rejects a SigV4
    request more than 15 minutes after its signing time, so once AWS's own
    clock passes the lifetime bound its maximum-duration enforcement is the
    termination proof. Between those bounds, or when AWS's time cannot be
    read, ownership stays pending and this raises
    ``LambdaMicroVMAllocationRecoveryError``.
    """

    microvm_id = acknowledged_microvm_id
    if microvm_id is None:
        if now <= replay_deadline(metadata):
            response = await run_microvm_with_client_token(
                control_client,
                run_options,
                client_token=token,
                submit_not_after=replay_deadline(metadata),
            )
            microvm_id = response.get("microvmId")
            if type(microvm_id) is not str or not microvm_id.strip():
                raise LambdaMicroVMProtocolError("run_microvm response omitted microvmId.")
        else:
            # The local clock only decides whether asking is worthwhile;
            # AWS's clock is the proof.
            server_now = (
                await aws_server_time(control_client, metadata["image_arn"])
                if now >= lifetime_deadline(metadata)
                else None
            )
            if server_now is not None and server_now >= lifetime_deadline(metadata):
                return
            raise LambdaMicroVMAllocationRecoveryError(
                "Lambda MicroVM client-token replay window has elapsed and AWS has not "
                "confirmed that the allocation's maximum lifetime has; cleanup remains "
                f"pending until {lifetime_deadline(metadata)} (epoch seconds, AWS time)."
            )
    if type(microvm_id) is not str:
        raise LambdaMicroVMProtocolError("Acknowledged Lambda MicroVM id is invalid.")
    await terminate_microvm_confirmed(
        control_client,
        microvm_id,
        timeout_s=REAP_TERMINATION_TIMEOUT_SECONDS,
    )


async def close_control_client(control_client: Any, owns_client: bool) -> None:
    """Close a control-plane client this caller created; leave an injected one alone."""

    if owns_client:
        close = getattr(control_client, "close", None)
        if callable(close):
            await asyncio.to_thread(close)
