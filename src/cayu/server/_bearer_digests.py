"""Bearer credentials that a host checks against SHA-256 digests of its tokens.

The fork-controller entrance and Cayu Cloud's reader both accept exactly the
same ``Authorization`` headers and digest the same way, from these rules.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request

# RFC 6750 b64token: the only credential syntax a bearer header may carry.
BEARER_CREDENTIAL = re.compile(r"[A-Za-z0-9\-._~+/]+=*")
# How a host configures a token it holds only as a digest.
TOKEN_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def token_digest(token: str) -> bytes:
    """The SHA-256 digest a host compares a presented bearer token by."""

    return hashlib.sha256(token.encode("ascii")).digest()


def presented_bearer_digest(request: Request) -> bytes | None:
    """The digest of the request's bearer credential, or ``None`` without exactly one.

    A missing, repeated or malformed ``Authorization`` header, or another
    scheme, presents no credential.
    """

    headers = request.headers.getlist("authorization")
    if len(headers) != 1:
        return None
    scheme, separator, credential = headers[0].partition(" ")
    if (
        separator != " "
        or scheme.lower() != "bearer"
        or BEARER_CREDENTIAL.fullmatch(credential) is None
    ):
        return None
    return token_digest(credential)
