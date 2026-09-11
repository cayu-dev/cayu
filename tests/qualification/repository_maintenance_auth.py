"""Explicit product/operator credentials for the application-owned host."""

import hmac
import re

from fastapi import HTTPException, Request

from cayu.server import AuthContext, ProductPrincipal

_TOKEN = re.compile(r"[-A-Za-z0-9._~+/]+=*", flags=re.ASCII)


def _token(value):
    if type(value) is not str or not 1 <= len(value) <= 4096 or _TOKEN.fullmatch(value) is None:
        raise ValueError("Invalid maintenance authentication configuration.")
    return value


def _presented(request):
    headers = request.headers.getlist("authorization")
    if len(headers) != 1:
        return None
    scheme, separator, token = headers[0].partition(" ")
    if separator != " " or scheme.lower() != "bearer":
        return None
    try:
        return _token(token)
    except ValueError:
        return None


class MaintenanceAccess:
    """Host-only construction; no development header or open-access fallback."""

    def __init__(self, *, product_tokens: dict[str, ProductPrincipal], operator_token: str):
        if type(product_tokens) is not dict or not 1 <= len(product_tokens) <= 64:
            raise ValueError("Invalid maintenance authentication configuration.")
        try:
            operator = _token(operator_token)
            principals = []
            for credential, principal in product_tokens.items():
                credential = _token(credential)
                if credential == operator or type(principal) is not ProductPrincipal:
                    raise ValueError
                copied = ProductPrincipal(
                    tenant_id=principal.tenant_id, subject_id=principal.subject_id
                )
                principals.append((credential, copied))
        except (ValueError, TypeError):
            raise ValueError("Invalid maintenance authentication configuration.") from None
        self._principals = tuple(principals)
        self._operator = operator

    async def product(self, request: Request) -> ProductPrincipal:
        token = _presented(request)
        for credential, principal in self._principals:
            if token is not None and hmac.compare_digest(token, credential):
                return ProductPrincipal(
                    tenant_id=principal.tenant_id, subject_id=principal.subject_id
                )
        raise HTTPException(status_code=401, detail="Product authentication required.")

    async def operator(self, request: Request) -> AuthContext:
        token = _presented(request)
        if token is None or not hmac.compare_digest(token, self._operator):
            raise HTTPException(status_code=401, detail="Operator authentication required.")
        return AuthContext(subject="maintenance-operator")
