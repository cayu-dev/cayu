"""Authenticated client for the versioned Cayu Cloud customer API."""

from __future__ import annotations

import ipaddress
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

_SAFE_OBJECT_STORE_ERROR_CODES = frozenset(
    {
        "AccessDenied",
        "AuthenticationFailed",
        "AuthorizationQueryParametersError",
        "ExpiredToken",
        "InvalidArgument",
        "InvalidRequest",
        "RequestExpired",
        "SignatureDoesNotMatch",
    }
)


class CloudApiError(RuntimeError):
    """Stable customer-facing API failure."""

    def __init__(self, category: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code


@dataclass(frozen=True)
class CloudApiClient:
    api_url: str
    api_key: str
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        parsed = urlsplit(self.api_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("A canonical HTTP(S) Cayu Cloud API URL is required.")
        if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
            raise ValueError("Cayu Cloud API URL must use HTTPS outside loopback.")
        if (
            not self.api_key
            or self.api_key != self.api_key.strip()
            or any(character.isspace() for character in self.api_key)
        ):
            raise ValueError("Cayu Cloud API key must be a non-empty canonical value.")
        if self.timeout_seconds <= 0:
            raise ValueError("API timeout must be positive.")

    @classmethod
    def from_key_file(
        cls,
        *,
        api_url: str,
        api_key_file: Path,
        timeout_seconds: float = 30.0,
    ) -> CloudApiClient:
        try:
            api_key = api_key_file.read_text().strip()
        except OSError as exc:
            raise CloudApiError(
                "api_key_unavailable",
                "Could not read the Cayu Cloud API key file.",
            ) from exc
        return cls(
            api_url=api_url.rstrip("/"),
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=self.timeout_seconds,
            ) as client:
                response = client.request(
                    method,
                    self.api_url.rstrip("/") + path,
                    content=body,
                    headers=headers,
                )
        except httpx.RequestError:
            raise CloudApiError(
                "api_unavailable",
                "Cayu Cloud API is unavailable.",
            ) from None
        if not 200 <= response.status_code < 300:
            raise CloudApiError(
                "api_request_rejected",
                f"Cayu Cloud API returned HTTP {response.status_code}.",
                status_code=response.status_code,
            ) from None
        if response.status_code == 204:
            return {}
        try:
            result = response.json()
        except (TypeError, ValueError, json.JSONDecodeError):
            raise CloudApiError(
                "api_response_invalid",
                "Cayu Cloud API returned invalid JSON.",
            ) from None
        if not isinstance(result, dict):
            raise CloudApiError(
                "api_response_invalid",
                "Cayu Cloud API returned an unsupported response.",
            )
        return result

    def upload_bytes(
        self,
        upload_url: str,
        content: bytes,
        *,
        content_type: str = "application/gzip",
    ) -> None:
        parsed = urlsplit(upload_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise CloudApiError("source_upload_invalid", "Source upload URL is invalid.")
        if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
            raise CloudApiError(
                "source_upload_invalid",
                "Source upload URL must use HTTPS outside loopback.",
            )
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=max(self.timeout_seconds, 120.0),
            ) as client:
                response = client.put(
                    upload_url,
                    content=content,
                    headers={"Content-Type": content_type},
                )
        except httpx.RequestError:
            raise CloudApiError(
                "source_upload_failed",
                "Local source bundle upload failed.",
            ) from None
        if not 200 <= response.status_code < 300:
            detail = _object_store_error(response.content)
            suffix = f": {detail.rstrip('.')}" if detail else ""
            raise CloudApiError(
                "source_upload_rejected",
                f"Local source bundle upload returned HTTP {response.status_code}{suffix}.",
                status_code=response.status_code,
            )


def _object_store_error(content: bytes) -> str | None:
    """Return only an allowlisted provider code, never provider-controlled detail."""

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return None
    code = root.findtext("Code")
    if code is None:
        return None
    normalized = code.strip()
    return normalized if normalized in _SAFE_OBJECT_STORE_ERROR_CODES else None


def _is_loopback(hostname: str | None) -> bool:
    if hostname is None:
        return False
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
