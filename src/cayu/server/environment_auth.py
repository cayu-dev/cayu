"""Ready-made ``[tool.cayu.serve].auth`` target backed by environment variables.

Reference it from a project's ``pyproject.toml``::

    [tool.cayu.serve]
    auth = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"

Importing this module builds the dependency once, with ``environment_auth()``:

- ``BasicAuth.from_environment()``, the Developer dashboard login, from
  ``CAYU_OPERATOR_USERNAME`` and ``CAYU_OPERATOR_PASSWORD``. When either is
  unset or empty the import raises ``AuthConfigurationError``, so ``cayu serve``
  exits at startup instead of serving an open Developer dashboard.
- When ``CAYU_PROXY_IDENTITY_ISSUER``, ``CAYU_PROXY_IDENTITY_AUDIENCE`` and
  ``CAYU_PROXY_IDENTITY_JWKS_URL`` are set as well (and optionally
  ``CAYU_PROXY_IDENTITY_HEADER``), a ``ProxyIdentityAuth`` that trusts the
  sign-in proxy's identity header and falls back to the Developer dashboard login, as
  ``admin``, for requests without it. Setting only some of them raises
  ``AuthConfigurationError`` naming the rest.

``OPERATOR_ONLY_AUTH`` is the same, except that people the proxy signs in with the
``member`` role may not use the Developer dashboard: for an application whose own users sign
in through the same proxy, such as the b2b preset's, so they reach its data only through
its own scoped routes. Admins and developers keep full access.

``cayu.server`` does not import this module; it is read only when a serve
configuration names it. Call ``BasicAuth.from_environment(...)`` or
``ProxyIdentityAuth.from_environment(...)`` directly to read other variables or
to build the same dependency for an embedded server.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from cayu.server.auth import BasicAuth
from cayu.server.proxy_identity import (
    ControlPlaneRoles,
    ProxyIdentityAuth,
    proxy_identity_configured,
)

__all__ = ["OPERATOR_BASIC_AUTH", "OPERATOR_ONLY_AUTH", "environment_auth"]


def environment_auth(
    environ: Mapping[str, str] | None = None, *, members_use_dashboard: bool = True
) -> BasicAuth | ProxyIdentityAuth:
    """The Developer dashboard's auth dependency, from the environment.

    The Developer dashboard login alone, or proxy identity with the Developer dashboard login as its
    fallback when the ``CAYU_PROXY_IDENTITY_*`` variables are set. ``environ``
    replaces ``os.environ`` as the source. ``members_use_dashboard=False`` refuses the
    ``member`` role everywhere on the Developer dashboard and its API.
    """

    source: Mapping[str, str] = os.environ if environ is None else environ
    # Checked first, so a half-configured proxy is named even without operator credentials.
    proxied = proxy_identity_configured(source)
    operator = BasicAuth.from_environment(environ=source)
    if not proxied:
        return operator
    return ProxyIdentityAuth.from_environment(
        environ=source,
        fallback=operator,
        control_plane_roles=(
            None if members_use_dashboard else ControlPlaneRoles(read_and_decide=())
        ),
    )


OPERATOR_BASIC_AUTH = environment_auth()
OPERATOR_ONLY_AUTH = environment_auth(members_use_dashboard=False)
