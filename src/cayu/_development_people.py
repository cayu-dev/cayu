"""The people a development identity proxy signs in, read without the server extra.

``cayu.server.development_identity`` mints their identities; ``cayu serve`` reads them
from ``[tool.cayu.serve.identity_proxy]`` and ``cayu check`` checks that table with
the same rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final
from urllib.parse import parse_qsl, quote, unquote, urlencode

#: The roles a proxy identity may carry, as ``cayu.server.proxy_identity.PROXY_IDENTITY_ROLES``.
DEVELOPMENT_ROLES: Final = ("admin", "developer", "member")
DEFAULT_DEVELOPMENT_ORG: Final = "local"
DEFAULT_DEVELOPMENT_ROLE: Final = "member"
_MAX_TEXT_CHARS: Final = 320
_PERSON_KEYS: Final = frozenset({"subject", "org", "role", "name", "email"})
_TABLE_KEYS: Final = frozenset({"default", "people"})
# Short keys for a person written into a header or cookie.
_WIRE_KEYS: Final = {
    "sub": "subject",
    "org": "org",
    "role": "role",
    "name": "name",
    "email": "email",
}


@dataclass(frozen=True)
class DevelopmentPerson:
    """One person a development identity proxy can sign in as.

    ``subject`` becomes the token's ``sub``, ``org`` its ``org`` and ``role`` its
    ``role`` (one of ``admin``, ``developer`` and ``member``); ``name`` and
    ``email`` are sent only when set. Invalid values raise ``ValueError``.
    """

    subject: str
    org: str = DEFAULT_DEVELOPMENT_ORG
    role: str = DEFAULT_DEVELOPMENT_ROLE
    name: str | None = None
    email: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("subject", "org"):
            _require_text(getattr(self, field_name), field_name, optional=False)
        for field_name in ("name", "email"):
            _require_text(getattr(self, field_name), field_name, optional=True)
        if self.role not in DEVELOPMENT_ROLES:
            raise ValueError(f"`role` must be one of {', '.join(DEVELOPMENT_ROLES)}.")

    def encode(self) -> str:
        """This person as ``sub=..&org=..&role=..``, the form a header or cookie takes."""

        fields = {"sub": self.subject, "org": self.org, "role": self.role}
        if self.name is not None:
            fields["name"] = self.name
        if self.email is not None:
            fields["email"] = self.email
        return urlencode(fields, quote_via=quote)


def parse_development_person(
    value: str, people: Mapping[str, DevelopmentPerson] | None = None
) -> DevelopmentPerson:
    """Read a person from a header or cookie value.

    A bare value is a subject: the configured person with that subject, or a
    ``member`` of the ``local`` organization. ``sub=..&org=..&role=..&name=..&email=..``
    names the fields; when ``sub`` is a configured person, the given fields
    replace theirs. Raises ``ValueError`` for anything else.
    """

    known = people or {}
    if type(value) is not str or not value.strip():
        raise ValueError("A person needs a subject.")
    text = value.strip()
    if "=" not in text:
        subject = unquote(text)
        return known.get(subject) or DevelopmentPerson(subject=subject)
    fields: dict[str, str] = {}
    for key, item in parse_qsl(text, keep_blank_values=True, strict_parsing=True):
        if key not in _WIRE_KEYS or _WIRE_KEYS[key] in fields:
            raise ValueError(f"Unknown or repeated person field {key!r}.")
        fields[_WIRE_KEYS[key]] = item
    if "subject" not in fields:
        raise ValueError("A person needs `sub`.")
    base = known.get(fields["subject"])
    if base is None:
        return DevelopmentPerson(**fields)
    return replace(base, **fields)


def parse_identity_proxy_table(
    value: Any,
) -> tuple[tuple[DevelopmentPerson, ...], str | None]:
    """The people and default subject of ``[tool.cayu.serve.identity_proxy]``.

    Raises ``ValueError`` naming the first problem: an unknown key, a person
    with a blank subject, an unknown role or an unknown key, a repeated
    subject, or a ``default`` that names nobody configured.
    """

    if not isinstance(value, Mapping):
        raise ValueError("[tool.cayu.serve.identity_proxy] must be a table.")
    unknown = set(value) - _TABLE_KEYS
    if unknown:
        raise ValueError(
            f"[tool.cayu.serve.identity_proxy] has unsupported keys: {', '.join(sorted(unknown))}."
        )
    entries = value.get("people", [])
    if not isinstance(entries, Sequence) or isinstance(entries, str):
        raise ValueError("[tool.cayu.serve.identity_proxy].people must be an array of tables.")
    people: list[DevelopmentPerson] = []
    for index, entry in enumerate(entries):
        where = f"[tool.cayu.serve.identity_proxy].people[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} must be a table.")
        extra = set(entry) - _PERSON_KEYS
        if extra:
            raise ValueError(f"{where} has unsupported keys: {', '.join(sorted(extra))}.")
        try:
            person = DevelopmentPerson(**dict(entry))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: {exc}") from None
        if any(known.subject == person.subject for known in people):
            raise ValueError(f"{where}: subject {person.subject!r} is listed twice.")
        people.append(person)
    default = value.get("default")
    if default is not None:
        if type(default) is not str or not default.strip():
            raise ValueError("[tool.cayu.serve.identity_proxy].default must be a subject.")
        if not any(person.subject == default for person in people):
            raise ValueError(
                f"[tool.cayu.serve.identity_proxy].default {default!r} is not one of its people."
            )
    return tuple(people), default


def _require_text(value: Any, field_name: str, *, optional: bool) -> None:
    if value is None and optional:
        return
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > _MAX_TEXT_CHARS
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError(f"`{field_name}` must be a non-blank string without control characters.")
