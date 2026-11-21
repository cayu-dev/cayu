"""CayuApp's connection slots: the outside systems it declares it reaches, by name.

``CayuApp`` inherits these; they live here so the application module stays within its size
limit. See ``cayu.connections.slots`` for the slot type. A slot comes from the tools an agent is
registered with (a connector toolkit's ``Tool.connection_slots``) or from
``require_connection``, which wins for the slot's connectors and description where it gives them.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType
from typing import TYPE_CHECKING

from cayu.connections.slots import (
    MAX_CONNECTION_SLOTS,
    ConnectionSlot,
    merged_slot,
    overridden_slot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cayu.applications import CayuApp
    from cayu.connections import ModelSlot
    from cayu.tools.base import Tool


class ConnectionSlotEntrances:
    """Connection-slot declarations for ``CayuApp``."""

    # The application's own declarations (require_connection), and those its agents' tools make.
    _connection_slot_declarations: Mapping[str, ConnectionSlot] = MappingProxyType({})
    _toolkit_connection_slots: Mapping[str, ConnectionSlot] = MappingProxyType({})

    def require_connection(
        self: CayuApp,
        slot: str,
        *,
        connectors: Iterable[str] = (),
        scopes: Iterable[str] = (),
        description: str | None = None,
    ) -> ConnectionSlot:
        """Declare a connection slot: an outside system the application calls through, by name.

        A connector toolkit's tools declare their slot when an agent is registered with them, so
        this is for a slot the application calls through a ``ConnectionClient`` itself, or to
        give a toolkit's slot other connectors or a description. ``connectors`` are the ids of
        the connectors a connection bound to it may be for (SAP's ``("sap", "sap-sandbox",
        "sap-certificate")``); none accepts any. ``scopes`` are the scopes its calls need, and
        ``description`` one line on what it is for. For a slot tools declare too, the connectors
        and description given here win and the scopes are both's. Cayu Cloud shows the
        declaration when an admin binds the Agent's slots; the binding, not the declaration,
        decides what calls through the slot may do.
        """

        if isinstance(connectors, str) or isinstance(scopes, str):
            raise TypeError("connectors and scopes are collections of strings, not one string.")
        declared = ConnectionSlot(
            name=slot,
            connectors=tuple(connectors),
            scopes=tuple(scopes),
            description=description,
        )
        if declared.name in self._connection_slot_declarations:
            raise ValueError(f"The connection slot {declared.name} is declared more than once.")
        explicit = {**self._connection_slot_declarations, declared.name: declared}
        self._checked_connection_slots(explicit, self._toolkit_connection_slots)
        self._connection_slot_declarations = MappingProxyType(explicit)
        return declared

    def connection_slots(self: CayuApp) -> tuple[ConnectionSlot, ...]:
        """Every declared connection slot, sorted by name."""

        return self._combined_connection_slots(
            self._connection_slot_declarations, self._toolkit_connection_slots
        )

    def _connection_slot_names(self: CayuApp) -> set[str]:
        return {*self._connection_slot_declarations, *self._toolkit_connection_slots}

    @staticmethod
    def _combined_connection_slots(
        explicit: Mapping[str, ConnectionSlot], toolkit: Mapping[str, ConnectionSlot]
    ) -> tuple[ConnectionSlot, ...]:
        return tuple(
            overridden_slot(toolkit[name], explicit[name])
            if name in explicit and name in toolkit
            else explicit.get(name) or toolkit[name]
            for name in sorted({*explicit, *toolkit})
        )

    def _checked_connection_slots(
        self: CayuApp,
        explicit: Mapping[str, ConnectionSlot],
        toolkit: Mapping[str, ConnectionSlot],
    ) -> None:
        """Refuse declarations that would clash with a model slot, exceed the limit or hold a
        secret."""

        clash = sorted(set(self._model_slot_declarations).intersection({*explicit, *toolkit}))
        if clash:
            raise ValueError(f"The slot {clash[0]} is already declared as a model slot.")
        combined = self._combined_connection_slots(explicit, toolkit)
        if len(combined) > MAX_CONNECTION_SLOTS:
            raise ValueError(
                f"An application declares at most {MAX_CONNECTION_SLOTS} connection slots."
            )
        for slot in combined:
            descriptor = slot.declaration()
            if self.redact_json(descriptor) != descriptor:
                raise ValueError(f"The connection slot {slot.name} contains a secret value.")

    def _toolkit_slots(
        self: CayuApp,
        tools: Iterable[Tool] | None,
        proposal_tools: Iterable[Tool] | None,
    ) -> tuple[Iterable[Tool] | None, Iterable[Tool] | None, Mapping[str, ConnectionSlot]]:
        """The tools an agent is about to be registered with, read once, and the application's
        toolkit slots with theirs, checked; they are kept only once the agent registers.

        Tools that can't be read as a collection are passed on as given, for the registry to
        refuse in its own words."""

        tools, listed = _read_once(tools)
        proposal_tools, proposals = _read_once(proposal_tools)
        toolkit = dict(self._toolkit_connection_slots)
        for tool in [*listed, *proposals]:
            for slot in getattr(tool, "connection_slots", ()):
                if type(slot) is not ConnectionSlot:
                    raise TypeError(
                        f"Tool {tool.name!r} declares a connection slot that isn't one."
                    )
                kept = toolkit.get(slot.name)
                toolkit[slot.name] = slot if kept is None else merged_slot(kept, slot)
        self._checked_connection_slots(self._connection_slot_declarations, toolkit)
        return tools, proposal_tools, MappingProxyType(toolkit)

    def _add_agent_slots(
        self: CayuApp,
        agent_name: str,
        model_slot: ModelSlot | None,
        toolkit: Mapping[str, ConnectionSlot],
    ) -> None:
        """Keep a registered agent's slots: the model slot its model_setting() declares, if any,
        and the connection slots its tools declare."""

        self._toolkit_connection_slots = toolkit
        if model_slot is not None:
            self._add_setting_model_slot(agent_name, model_slot)


def _read_once(
    tools: Iterable[Tool] | None,
) -> tuple[Iterable[Tool] | None, tuple[Tool, ...]]:
    """``tools`` to pass on (a list once read) and the tools read; none when they can't be read
    as a collection of tools, which the registry refuses."""

    if tools is None or isinstance(tools, str | bytes):
        return tools, ()
    try:
        listed = list(tools)
    except TypeError:
        return tools, ()
    return listed, tuple(listed)
