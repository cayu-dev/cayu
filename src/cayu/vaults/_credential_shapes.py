"""Credential shapes masked in text that may echo a request or a command line.

``SecretRedactor`` removes the secret values an application registered. These
patterns catch credentials it was never told about, by their shape: an
``Authorization`` value (its scheme and credential both), a bearer token, a Git
``x-access-token`` credential in plain or base64 form, and token formats with a
known prefix (GitHub, OpenAI-style ``sk-``). Every pattern is linear: each match
starts at a fixed anchor and its repeated parts are possessive, so they are safe
on large untrusted text.
"""

from __future__ import annotations

import re

# One credential word: it may hold quotes, but it doesn't end in one or take the
# separator or bracket after it, so a value inside a Python repr or JSON string
# stops at its closing quote.
_WORD = r"[^\s'\"]++(?:['\"]++[^\s'\",;)\]}]++)*+"

AUTHORIZATION_HEADER = re.compile(
    # ``Authorization: Basic x``, ``authorization=Bearer x`` (git's
    # ``-c http.extraheader=...``), and ``"Authorization": "Bearer x"``: the
    # scheme and the credential after it, or a bare credential.
    # Names that end in it count too: HTTP_AUTHORIZATION, proxy_authorization.
    rf"(?i)(?P<keep>authorization[\"']?[ \t]*+[:=])(?P<space>\s*+)(?P<quote>[\"']?)"
    rf"{_WORD}(?:[^\S\n]++[\"']?{_WORD})?"
)
BEARER_TOKEN = re.compile(rf"(?i)\bbearer\s++[\"']?{_WORD}")
# ``x-access-token:<token>`` as Git and GitHub App installations send it, and its
# base64 encoding in a Basic credential ("x-access-token:" encodes to the prefix).
GIT_ACCESS_TOKEN = re.compile(
    r"\bx-access-token:[^\s'\"@,;]++|\beC1hY2Nlc3MtdG9rZW46[A-Za-z0-9+/]*+={0,2}"
)
# GitHub App installation tokens may carry a JWT after the prefix, dots included.
PREFIXED_TOKEN = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_.-]{8,}+|github_pat_[A-Za-z0-9_]{8,}+|sk[-_][A-Za-z0-9_-]{8,}+)"
)


def mask_credential_shapes(text: str, marker: str) -> str:
    """Replace every credential shape in *text* with *marker*.

    An ``Authorization`` value keeps its name, spacing and opening quote, so
    ``Authorization: Basic x`` becomes ``Authorization: <marker>``.
    """

    masked = AUTHORIZATION_HEADER.sub(
        lambda match: f"{match['keep']}{match['space']}{match['quote']}{marker}", text
    )
    masked = BEARER_TOKEN.sub(f"Bearer {marker}", masked)
    masked = GIT_ACCESS_TOKEN.sub(marker, masked)
    return PREFIXED_TOKEN.sub(marker, masked)
