"""A dedicated reporting credential never grants operator or native execution authority."""

from __future__ import annotations

import contextlib
import hmac
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, cast

from pydantic import Field, StrictStr, field_validator
from starlette.requests import (
    Request,  # noqa: TC002 -- Pydantic resolves the callable annotation at runtime.
)

from cayu.evals.reported import _ReportModel
from cayu.server._bearer_digests import TOKEN_SHA256_HEX, presented_bearer_digest
from cayu.server.auth import AuthConfigurationError

EVAL_REPORTERS_VARIABLE = "CAYU_EVAL_REPORTERS"
EVAL_REPORT_MAX_CREDENTIAL_BYTES = 4096
EVAL_REPORT_MAX_PUBLISHERS = 64


class EvalReportPrincipal(_ReportModel):
    publisher_id: StrictStr = Field(min_length=1, max_length=256)
    target_keys: tuple[Annotated[StrictStr, Field(min_length=1, max_length=128)], ...] = Field(
        min_length=1, max_length=128
    )

    @field_validator("publisher_id")
    @classmethod
    def validate_publisher(cls, value):
        if value.strip() != value or not value:
            raise ValueError("Invalid reporting publisher.")
        return value

    @field_validator("target_keys")
    @classmethod
    def validate_targets(cls, value):
        if len(set(value)) != len(value) or any(key.strip() != key or not key for key in value):
            raise ValueError("Invalid reporting targets.")
        return value


ReportAuthenticate = Callable[["Request"], Awaitable[EvalReportPrincipal | None]]


@dataclass(frozen=True, repr=False)
class EvalReportingAccess:
    """Trusted auth extension; a verifier returns the same scoped report principal."""

    authenticate: ReportAuthenticate

    def __post_init__(self):
        if not callable(self.authenticate):
            raise TypeError("Reporting authentication must be callable.")

    def __repr__(self):
        return "EvalReportingAccess(configured=True)"


class EvalReporterBearerAuth:
    """Configured digest-to-publisher grants, with overlap for credential rotation."""

    __slots__ = ("_entries",)

    def __init__(self, entries: object):
        prepared = []
        seen = set()
        valid = True
        try:
            if type(entries) is not list or not 1 <= len(entries) <= EVAL_REPORT_MAX_PUBLISHERS:
                raise ValueError("Invalid grant count.")
            for entry in entries:
                if type(entry) is not dict or set(entry) != {
                    "publisher_id",
                    "token_sha256",
                    "target_keys",
                }:
                    raise ValueError("Invalid grant fields.")
                entry = cast("dict[str, object]", entry)
                digest = entry["token_sha256"]
                if type(digest) is not str or TOKEN_SHA256_HEX.fullmatch(digest) is None:
                    raise ValueError("Invalid digest.")
                if digest in seen:
                    raise ValueError("Duplicate digest.")
                seen.add(digest)
                principal = EvalReportPrincipal.model_validate(
                    {"publisher_id": entry["publisher_id"], "target_keys": entry["target_keys"]}
                )
                prepared.append((bytes.fromhex(digest), principal))
        except (TypeError, ValueError):
            valid = False
        # No exception chain may retain the secret-bearing configuration input.
        if not valid:
            raise AuthConfigurationError("Invalid Evals reporting credential configuration.")
        self._entries = tuple(prepared)

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> EvalReporterBearerAuth | None:
        raw = (os.environ if environ is None else environ).get(EVAL_REPORTERS_VARIABLE)
        if raw is None:
            return None
        parsed = None
        with contextlib.suppress(ValueError):
            if type(raw) is str and len(raw.encode("utf-8")) <= 64 << 10:
                parsed = json.loads(raw)
        return cls(parsed)

    async def __call__(self, request: Request) -> EvalReportPrincipal | None:
        headers = request.headers.getlist("authorization")
        if len(headers) != 1 or len(headers[0].encode("utf-8")) > EVAL_REPORT_MAX_CREDENTIAL_BYTES:
            return None
        digest = presented_bearer_digest(request)
        if digest is None:
            return None
        principal = None
        for expected, candidate in self._entries:
            if hmac.compare_digest(expected, digest):
                principal = candidate
        return principal

    def __repr__(self):
        return f"EvalReporterBearerAuth(grants={len(self._entries)})"
