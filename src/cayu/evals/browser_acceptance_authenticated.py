"""Application-selected, credential-free authority for an authenticated trial.

This describes permission and expected evidence, not credentials or a login
mechanism. The application owns the protected operator server and site observer.
"""

from __future__ import annotations

import re
from decimal import Decimal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu.evals.corpus import _content_revision


class BrowserAcceptanceAuthenticatedConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    authorized: StrictBool
    account_scope_revision: StrictStr
    site_observer_revision: StrictStr
    profile_authority_fingerprint: StrictStr
    operator_policy_fingerprint: StrictStr
    origin: StrictStr = Field(max_length=512)
    login_path: StrictStr = Field(max_length=256)
    protected_path: StrictStr = Field(max_length=256)
    allowed_endpoints: tuple[tuple[StrictStr, StrictStr], ...] = Field(min_length=2, max_length=16)
    operator_inputs: StrictInt = Field(default=6, ge=1, le=32)
    max_estimated_cost: StrictStr = Field(default="1.00 USD", max_length=32)

    @field_validator("authorized")
    @classmethod
    def require_authorization(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError(
                "Authenticated acceptance requires explicit application authorization."
            )
        return value

    @field_validator("account_scope_revision", "site_observer_revision")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ValueError("Account scope must be an opaque content revision.")
        return value

    @field_validator("profile_authority_fingerprint", "operator_policy_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Authenticated authority requires an exact fingerprint.")
        return value

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            valid = parsed.hostname and value == "https://" + parsed.hostname.lower()
        except ValueError:
            valid = False
        if not valid or not value.isascii():
            raise ValueError("Authenticated acceptance requires one canonical HTTPS origin.")
        return value

    @field_validator("login_path", "protected_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if re.fullmatch(r"/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+", value) is None:
            raise ValueError("Authenticated acceptance requires exact credential-free route paths.")
        return value

    @field_validator("allowed_endpoints")
    @classmethod
    def validate_endpoints(cls, value: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("Authenticated endpoints must be unique and sorted.")
        for method, path in value:
            if method not in {"GET", "POST"}:
                raise ValueError("Authenticated acceptance permits only explicit GET/POST routes.")
            cls.validate_path(path)
        return value

    @field_validator("max_estimated_cost")
    @classmethod
    def validate_cost(cls, value: str) -> str:
        if re.fullmatch(r"[0-9]{1,6}(?:\.[0-9]{1,8})? [A-Z]{3}", value) is None:
            raise ValueError("Authenticated acceptance requires a finite cost ceiling.")
        if Decimal(value.split()[0]) <= 0:
            raise ValueError("Authenticated acceptance requires a positive cost ceiling.")
        return value

    @model_validator(mode="after")
    def validate_routes(self) -> BrowserAcceptanceAuthenticatedConfigV1:
        if self.login_path == self.protected_path or not {
            ("GET", self.login_path),
            ("GET", self.protected_path),
        }.issubset(self.allowed_endpoints):
            raise ValueError(
                "Authenticated acceptance must admit distinct login and protected routes."
            )
        return self

    @property
    def revision(self) -> str:
        return _content_revision(
            self.model_dump(mode="json"), "authenticated browser acceptance authority"
        )
