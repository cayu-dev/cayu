"""Application-owned acceptance client for the existing protected operator API.

This client neither grants authority nor drives the browser itself. The server
authenticates and reauthorizes every operation; the existing guest owner performs
input and handback. No credential or private input is returned as model evidence.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import ssl
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit

from pydantic import SecretStr

from cayu.evals.corpus import _content_revision
from cayu.runtime.browser_control import BrowserControlIdentity, BrowserControlPage
from cayu.runtime.browser_control_config import BrowserControlConfig
from cayu.server._browser_input_routes import OPERATOR_INPUT_SUBPROTOCOL
from cayu.tools._browser_control_transport import _private_transport_logger

if TYPE_CHECKING:
    import httpx

    from cayu.evals.browser_acceptance import BrowserAcceptancePlanV1


@dataclass(frozen=True, slots=True)
class OperatorFixtureBinding:
    """Caller-owned colocated server resources for the canonical fixture journey.

    The caller starts the protected server for the built app and retains its
    lifetime through environment cleanup. Only the public CA is mounted into
    the browser workspace; server keys and operator credentials stay outside it.
    """

    control: BrowserControlConfig = field(repr=False)
    server_container_id: str
    ca_certificate: Path
    client: httpx.AsyncClient = field(repr=False)
    tls: ssl.SSLContext = field(repr=False)
    operator_origin: str
    private_text: SecretStr = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.control) is not BrowserControlConfig:
            raise TypeError("Operator fixture requires application-owned browser control.")
        if (
            type(self.server_container_id) is not str
            or len(self.server_container_id) != 64
            or any(char not in "0123456789abcdef" for char in self.server_container_id)
        ):
            raise ValueError("Operator fixture requires the exact application container ID.")
        if not self.ca_certificate.is_absolute() or not self.ca_certificate.is_file():
            raise ValueError("Operator fixture requires an existing absolute public CA path.")

    async def handoff(self, *, session_id: str, browser_session_id: str) -> None:
        endpoint = str(self.client.base_url).rstrip("/") + "/api/browser-control/input"
        await perform_fixture_handoff(
            client=self.client,
            input_endpoint=endpoint.replace("https://", "wss://", 1),
            tls=self.tls,
            operator_origin=self.operator_origin,
            session_id=session_id,
            browser_session_id=browser_session_id,
            private_text=self.private_text,
        )

    @property
    def authority_revision(self) -> str:
        certificate = self.public_ca_pem()
        return _content_revision(
            {
                "policy": self.control.policy.identity,
                "purpose": self.control.purpose.model_dump(mode="json"),
                "server_container_id": self.server_container_id,
                "guest_endpoint": self.control.guest_endpoint,
                "client_endpoint": str(self.client.base_url),
                "operator_origin": self.operator_origin,
                "public_ca_sha256": sha256(certificate).hexdigest(),
            },
            "browser acceptance operator binding",
        )

    def public_ca_pem(self) -> bytes:
        """Return certificates only; never copy neighbouring files or key blocks."""
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import Encoding

        with self.ca_certificate.open("rb") as handle:
            certificate = handle.read(65537)
        if not certificate or len(certificate) > 65536:
            raise ValueError("Operator fixture public CA exceeds its bound.")
        try:
            certificates = x509.load_pem_x509_certificates(certificate)
        except ValueError:
            raise ValueError("Operator fixture public CA is not a certificate bundle.") from None
        if not certificates:
            raise ValueError("Operator fixture public CA is not a certificate bundle.")
        return b"".join(item.public_bytes(Encoding.PEM) for item in certificates)


@dataclass(frozen=True, slots=True)
class OperatorFixtureSetup:
    """Trusted CLI setup; never substitutes the canonical plan or browser owner.

    The outer async context owns ``binding`` resources. ``serve(plan)`` must serve
    the exact canonical application throughout environment cleanup and close only
    positively drained stores. Failed cleanup must propagate: retiring the fixture
    server is not evidence that remote work stopped or authority became reusable.
    """

    binding: OperatorFixtureBinding = field(repr=False)
    serve: Callable[[BrowserAcceptancePlanV1], AbstractAsyncContextManager[None]] = field(
        repr=False
    )

    def __post_init__(self) -> None:
        if type(self.binding) is not OperatorFixtureBinding or not callable(self.serve):
            raise TypeError("Operator setup requires an exact binding and server context.")


class _OperatorRequestRejected(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__("Browser acceptance operator request was not admitted.")
        self.status = status


async def _request(
    client: httpx.AsyncClient, method: str, path: str, *, expected_status: int = 200, **kwargs: Any
) -> dict[str, Any]:
    # Never include a response body or authenticated request in an error message.
    async with client.stream(method, path, **kwargs) as response:
        if response.status_code != expected_status:
            raise _OperatorRequestRejected(response.status_code)
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > 64 * 1024:
                raise RuntimeError("Browser acceptance operator response exceeds its bound.")
            body.extend(chunk)
        try:
            value = json.loads(body)
        except (ValueError, UnicodeError):
            raise RuntimeError("Browser acceptance operator response is malformed.") from None
        if type(value) is not dict:
            raise RuntimeError("Browser acceptance operator response is malformed.")
        return value


async def perform_fixture_handoff(
    *,
    client: httpx.AsyncClient,
    input_endpoint: str,
    tls: ssl.SSLContext,
    operator_origin: str,
    session_id: str,
    browser_session_id: str,
    private_text: SecretStr,
) -> None:
    """Perform one bounded, non-replayed tab/text handoff on a designated fixture.

    ``client`` is application-owned and already configured for HTTPS and server
    authentication. It must not be shared concurrently with another operator
    journey. The caller owns its lifetime and the protected server's lifetime.
    A timeout/cancellation is not proof that private input stopped; ordinary
    browser-control settlement and environment cleanup retain that ownership.
    """
    from websockets.asyncio.client import connect
    from websockets.typing import Origin, Subprotocol

    from cayu.tools._browser_control_transport import validate_control_endpoint

    validate_control_endpoint(input_endpoint)
    if not isinstance(tls, ssl.SSLContext) or tls.verify_mode != ssl.CERT_REQUIRED:
        raise ValueError("Browser acceptance operator transport requires verified TLS.")
    if not tls.check_hostname or client.base_url.scheme != "https":
        raise ValueError("Browser acceptance operator transport requires verified HTTPS.")
    endpoint = urlsplit(input_endpoint)
    if (
        endpoint.hostname != client.base_url.host
        or (endpoint.port or 443) != (client.base_url.port or 443)
        or endpoint.path != "/api/browser-control/input"
    ):
        raise ValueError("Browser acceptance input transport conflicts with its server authority.")
    origin = urlsplit(operator_origin)
    if (
        origin.scheme != "https"
        or not origin.hostname
        or origin.username is not None
        or origin.password is not None
        or origin.path
        or origin.query
        or origin.fragment
    ):
        raise ValueError("Browser acceptance requires an exact operator HTTPS origin.")
    if type(private_text) is not SecretStr or not private_text.get_secret_value():
        raise ValueError("Browser acceptance operator requires designated private fixture input.")
    try:
        encoded_size = len(private_text.get_secret_value().encode("utf-8"))
    except UnicodeError:
        raise ValueError("Browser acceptance private fixture input is not portable text.") from None
    if encoded_size > 4096 or "\x00" in private_text.get_secret_value():
        raise ValueError("Browser acceptance private fixture input exceeds its bound.")
    root = "/api/browser-control"
    async with asyncio.timeout(60):
        token = await _request(client, "POST", root + "/operator-session")
        continuity = token.get("operator_session_token")
        if type(continuity) is not str or not continuity:
            raise RuntimeError("Browser acceptance operator continuity is unavailable.")
        headers = {"X-Cayu-Browser-Operator": continuity}
        discovered_identity: BrowserControlIdentity | None = None

        async def discover() -> dict[str, Any]:
            nonlocal discovered_identity
            response = await _request(
                client, "GET", root + "/sessions/" + quote(session_id, safe=""), headers=headers
            )
            records = response.get("browsers")
            if type(records) is not list:
                raise RuntimeError("Browser acceptance operator discovery is unavailable.")
            matches = []
            for record in records:
                if type(record) is not dict:
                    raise RuntimeError("Browser acceptance operator discovery is malformed.")
                identity = BrowserControlIdentity.model_validate(record.get("identity"))
                if identity.session_id != session_id:
                    raise RuntimeError("Browser acceptance operator discovery conflicts.")
                if identity.browser_session_id == browser_session_id:
                    if discovered_identity is not None and identity != discovered_identity:
                        raise RuntimeError("Browser acceptance operator allocation changed.")
                    discovered_identity = identity
                    matches.append(record)
            if len(matches) != 1:
                raise RuntimeError("Browser acceptance operator allocation is unavailable.")
            return matches[0]

        async def wait_for(state: str, *, sensitive: bool = False) -> dict[str, Any]:
            while True:
                try:
                    record = await discover()
                except _OperatorRequestRejected as failure:
                    if failure.status != 403:
                        raise
                    # Exact discovery can reject a read crossing an already
                    # accepted transition. Retry only this read, under the outer
                    # deadline; no rejected response supplies mutation authority.
                    await asyncio.sleep(0.025)
                    continue
                if record.get("state") == state and (
                    not sensitive
                    or (
                        record.get("sensitive_entry") is True
                        and record.get("sensitive_entry_pending") is False
                    )
                ):
                    return record
                if record.get("state") in {"closed", "allocation_lost", "control_uncertain"}:
                    raise RuntimeError("Browser acceptance operator allocation did not settle.")
                await asyncio.sleep(0.025)

        async def pages(record: dict[str, Any]) -> list[dict[str, Any]]:
            response = await _request(
                client,
                "POST",
                root + "/pages",
                headers=headers,
                json={
                    "identity": record["identity"],
                    "expected_record_revision": record["revision"],
                },
            )
            values = response.get("pages")
            if type(values) is not list or len(values) != 1:
                raise RuntimeError(
                    "Browser acceptance operator requires one admitted fixture page."
                )
            return [BrowserControlPage.model_validate(values[0]).model_dump(mode="json")]

        original = await discover()
        requested_pages = await pages(original)
        now = time.time_ns() // 1_000_000
        request_id = "bt_" + secrets.token_hex(16)
        await _request(
            client,
            "POST",
            root + "/takeover",
            headers=headers,
            json={
                "request_id": request_id,
                "identity": original["identity"],
                "expected_record_revision": original["revision"],
                "expected_control_epoch": original["control_epoch"],
                "pages": requested_pages,
                "purpose_code": original["identity"]["operator_purpose"]["code"],
                "requested_at_ms": now,
                "expires_at_ms": now + 30_000,
                "maximum_until_ms": now + 60_000,
                "checkpoint_consent": "deny",
            },
        )
        acquired = await wait_for("operator_controlled")

        def intent(record: dict[str, Any]) -> dict[str, Any]:
            owned = record.get("owned_request")
            if type(owned) is not dict or owned.get("request_id") != request_id:
                raise RuntimeError("Browser acceptance operator request ownership conflicts.")
            return {
                "identity": record["identity"],
                "request_id": request_id,
                "expected_record_revision": record["revision"],
                "expected_control_epoch": record["control_epoch"],
            }

        await _request(
            client,
            "POST",
            root + "/sensitive-entry",
            headers=headers,
            json=intent(acquired),
            expected_status=202,
        )
        acquired = await wait_for("operator_controlled", sensitive=True)
        for sequence, kind in ((1, "tab"), (2, "text")):
            page = (await pages(acquired))[0]
            ticket = await _request(
                client,
                "POST",
                root + "/input-ticket",
                headers=headers,
                json={
                    **intent(acquired),
                    "page": page,
                    "input_sequence": sequence,
                    "input_kind": kind,
                },
            )
            if type(ticket.get("ticket")) is not str:
                raise RuntimeError("Browser acceptance private input ticket is unavailable.")
            async with connect(
                input_endpoint,
                ssl=tls,
                origin=Origin(operator_origin),
                subprotocols=[Subprotocol(OPERATOR_INPUT_SUBPROTOCOL)],
                proxy=None,
                compression=None,
                max_size=65536,
                logger=_private_transport_logger(),
            ) as channel:
                await channel.send(ticket["ticket"])
                if await channel.recv() != "ready":
                    raise RuntimeError("Browser acceptance private input was not ready.")
                await channel.send(
                    b"tab" if kind == "tab" else private_text.get_secret_value().encode("utf-8")
                )
                receipt = json.loads(await channel.recv())
                if (
                    type(receipt) is not dict
                    or receipt.get("state") != "settled"
                    or type(receipt.get("settled_input_sequence")) is not int
                    or receipt["settled_input_sequence"] != sequence
                ):
                    raise RuntimeError("Browser acceptance private input did not settle.")
            acquired = await wait_for("operator_controlled", sensitive=True)
        await _request(client, "POST", root + "/handback", headers=headers, json=intent(acquired))
        returned = await wait_for("agent_controlled")
        if returned.get("fresh_observation_required") is not True:
            raise RuntimeError("Browser acceptance handback lacks its fresh-observation fence.")
