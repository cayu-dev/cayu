"""Connection slots an application declares: the outside systems it reaches, by name.

A connector toolkit's tools carry the slot they call through (``Tool.connection_slots``), and
registering them with an agent declares it, with the toolkit's connectors and the scopes those
tools need. A slot called without a toolkit is declared with ``CayuApp.require_connection(slot,
connectors=..., scopes=..., description=...)``, which also overrides a toolkit's connectors or
description for its slot. Cayu Cloud receives the declarations when the
Release registers and offers them when an admin binds the Agent's slots, so the admin picks the
slot, sees which connectors it expects and the scopes it needs, and doesn't type its name. A
declaration is a hint for people: the broker still serves whatever the slot is bound to, with the
binding's scopes, and an undeclared slot can still be bound.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator

from cayu.connections.client import _SLOT

#: A slot name, and a connector id: 1-63 lowercase letters, digits or interior hyphens.
CONNECTION_SLOT_PATTERN = _SLOT
#: A scope a slot needs: 1-128 printable characters without spaces.
CONNECTION_SCOPE_PATTERN = re.compile(r"^[!-~]{1,128}$")
MAX_CONNECTION_SLOTS = 64
MAX_SLOT_CONNECTORS = 16
MAX_SLOT_SCOPES = 64
MAX_CONNECTION_SLOT_DESCRIPTION_LENGTH = 200


def _distinct(values: tuple[str, ...], what: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError(f"A connection slot names each {what} once.")
    return tuple(sorted(values))


class ConnectionSlot(BaseModel):
    """One declared connection slot: its name, the connectors it expects (none: any), the scopes
    its calls need, and a line for people."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: StrictStr
    connectors: tuple[StrictStr, ...] = Field(default=(), max_length=MAX_SLOT_CONNECTORS)
    scopes: tuple[StrictStr, ...] = Field(default=(), max_length=MAX_SLOT_SCOPES)
    description: StrictStr | None = Field(
        default=None, max_length=MAX_CONNECTION_SLOT_DESCRIPTION_LENGTH
    )

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not CONNECTION_SLOT_PATTERN.fullmatch(value):
            raise ValueError(
                "A connection slot is 1-63 lowercase letters, digits or interior hyphens."
            )
        return value

    @field_validator("connectors")
    @classmethod
    def _connectors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not all(CONNECTION_SLOT_PATTERN.fullmatch(connector) for connector in value):
            raise ValueError(
                "A connector id is 1-63 lowercase letters, digits or interior hyphens."
            )
        return _distinct(value, "connector")

    @field_validator("scopes")
    @classmethod
    def _scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not all(CONNECTION_SCOPE_PATTERN.fullmatch(scope) for scope in value):
            raise ValueError("A scope is 1-128 printable characters without spaces.")
        return _distinct(value, "scope")

    @field_validator("description")
    @classmethod
    def _description(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or not value.isprintable()):
            raise ValueError("A connection slot's description is one non-blank line.")
        return value

    def declaration(self) -> dict[str, Any]:
        """The slot as the agent registry publishes it: ``{"name", "connectors", "scopes",
        "description"?}``."""

        declared: dict[str, Any] = {
            "name": self.name,
            "connectors": list(self.connectors),
            "scopes": list(self.scopes),
        }
        if self.description is not None:
            declared["description"] = self.description
        return declared


def _combined(
    name: str, connectors: tuple[str, ...], scopes: set[str], description: str | None
) -> ConnectionSlot:
    try:
        return ConnectionSlot(
            name=name, connectors=connectors, scopes=tuple(scopes), description=description
        )
    except ValidationError as error:
        raise ValueError(
            f"The connection slot {name} needs more than {MAX_SLOT_SCOPES} scopes across its "
            "declarations."
        ) from error


def merged_slot(first: ConnectionSlot, second: ConnectionSlot) -> ConnectionSlot:
    """One slot as two tools declare it: their scopes together, the connectors both accept
    (no connectors accepts any), and the first description given (``first`` is the slot as
    the tools registered earlier declare it, so the earliest registered tool's description wins).

    Raises ``ValueError`` when both name connectors and share none.
    """

    if first.connectors and second.connectors:
        connectors = tuple(item for item in first.connectors if item in second.connectors)
        if not connectors:
            raise ValueError(
                f"The connection slot {first.name} is declared for the connectors "
                f"{', '.join(first.connectors)} and for {', '.join(second.connectors)}, which "
                "share none; give the toolkits different slots."
            )
    else:
        connectors = first.connectors or second.connectors
    return _combined(
        first.name,
        connectors,
        {*first.scopes, *second.scopes},
        first.description or second.description,
    )


def overridden_slot(toolkit: ConnectionSlot, explicit: ConnectionSlot) -> ConnectionSlot:
    """A toolkit's slot with the application's own declaration of it: the declaration's
    connectors and description where it gives them, and both's scopes."""

    return _combined(
        toolkit.name,
        explicit.connectors or toolkit.connectors,
        {*toolkit.scopes, *explicit.scopes},
        explicit.description or toolkit.description,
    )


__all__ = [
    "CONNECTION_SCOPE_PATTERN",
    "CONNECTION_SLOT_PATTERN",
    "MAX_CONNECTION_SLOTS",
    "MAX_CONNECTION_SLOT_DESCRIPTION_LENGTH",
    "MAX_SLOT_CONNECTORS",
    "MAX_SLOT_SCOPES",
    "ConnectionSlot",
]
