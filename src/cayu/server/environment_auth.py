"""Ready-made ``[tool.cayu.serve].auth`` target backed by environment variables.

Reference it from a project's ``pyproject.toml``::

    [tool.cayu.serve]
    auth = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"

Importing this module builds ``BasicAuth.from_environment()`` once, from
``CAYU_OPERATOR_USERNAME`` and ``CAYU_OPERATOR_PASSWORD``. When either is unset
or empty the import raises ``AuthConfigurationError``, so ``cayu serve`` exits
at startup instead of serving an open control plane. ``cayu.server`` does not
import this module; it is read only when a serve configuration names it.

Call ``BasicAuth.from_environment(...)`` directly to read other variables or to
build the same dependency for an embedded server.
"""

from __future__ import annotations

from cayu.server.auth import BasicAuth

__all__ = ["OPERATOR_BASIC_AUTH"]

OPERATOR_BASIC_AUTH = BasicAuth.from_environment()
