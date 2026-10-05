"""Dependency-light names for the environment-backed operator auth target.

``cayu.server`` requires the server extra, but scaffolding, ``cayu cloud init``,
and deploy checks must name the ready-made target without importing it.
"""

OPERATOR_USERNAME_VARIABLE = "CAYU_OPERATOR_USERNAME"
OPERATOR_PASSWORD_VARIABLE = "CAYU_OPERATOR_PASSWORD"
ENVIRONMENT_OPERATOR_AUTH_TARGET = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"

# Cayu 0.8.1 is the last release without ``cayu.server.environment_auth``; every
# later release ships it. A project whose cayu requirement still allows 0.8.1 or
# older cannot name the ready-made target and gets a project module instead.
LAST_RELEASE_WITHOUT_ENVIRONMENT_AUTH = "0.8.1"
