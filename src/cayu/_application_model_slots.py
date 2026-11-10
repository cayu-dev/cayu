"""CayuApp's model slots: the models it declares by name and how new sessions resolve them.

``CayuApp`` inherits these; they live here so the application module stays within its size
limit. See ``cayu.connections.models`` for the slot types and the reader of Cloud's bindings.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType
from typing import TYPE_CHECKING

from cayu.connections.models import (
    MAX_MODEL_SLOTS,
    MODEL_ID_PATTERN,
    MODEL_SLOT_PATTERN,
    ModelSlot,
    ModelSlotResolution,
    ModelSlots,
    resolve_model_slot,
)
from cayu.runtime.application_lifecycle import _admitted_entrance

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.managed_settings import ManagedSetting


def model_setting_slot(agent_name: str) -> str | None:
    """The slot an agent that chose its model with ``model_setting()`` uses: its name with
    ``_`` as ``-``, as Cayu Cloud moves its stored model there; None when that is no slot name."""

    slot = agent_name.replace("_", "-")
    return slot if MODEL_SLOT_PATTERN.fullmatch(slot) else None


class ModelSlotEntrances:
    """Model-slot declarations for ``CayuApp`` and their resolution when a session starts."""

    _model_slot_declarations: Mapping[str, ModelSlot] = MappingProxyType({})
    # Slots declared for agents that still choose their model with model_setting(): slot -> agent.
    _setting_model_slots: Mapping[str, str] = MappingProxyType({})
    _model_slots_reader: ModelSlots | None = None

    def require_model(
        self: CayuApp, slot: str, *, default: str | None, label: str | None = None
    ) -> ModelSlot:
        """Declare a model slot: a model the application needs, by name.

        The Organization binds the slot to one of its approved models in Cayu Cloud, without a
        redeploy; until then it uses ``default`` (None: the slot needs a binding). Name it on
        agents with ``register_agent(..., model_slot=slot)``. A new session reads the slot when
        it starts and keeps that model; resumed sessions keep theirs.
        """

        declared = ModelSlot(name=slot, default=default, label=label)
        legacy_agent = self._setting_model_slots.get(declared.name)
        if legacy_agent is not None:
            raise ValueError(
                f"The model slot {declared.name} is already the slot of agent {legacy_agent!r}, "
                "which declares the deprecated model_setting(); register that agent with "
                f"model_slot={declared.name!r} instead of model_setting()."
            )
        self._declare_model_slot(declared)
        return declared

    def model_slots(self: CayuApp) -> tuple[ModelSlot, ...]:
        """Every declared model slot, sorted by name.

        Raises ``ValueError`` when an agent names a slot the application doesn't declare.
        """

        undeclared = sorted(
            {
                registration.model_slot
                for registration in self._agent_registry.registrations.values()
                if registration.model_slot is not None
                and registration.model_slot not in self._model_slot_declarations
            }
        )
        if undeclared:
            raise ValueError(
                "Agents use model slots the application doesn't declare: "
                + ", ".join(undeclared)
                + ". Declare each with CayuApp.require_model()."
            )
        return tuple(
            self._model_slot_declarations[name] for name in sorted(self._model_slot_declarations)
        )

    def _declare_model_slot(self: CayuApp, declared: ModelSlot) -> None:
        if declared.name in self._model_slot_declarations:
            raise ValueError(f"The model slot {declared.name} is declared more than once.")
        if len(self._model_slot_declarations) >= MAX_MODEL_SLOTS:
            raise ValueError(f"An application declares at most {MAX_MODEL_SLOTS} model slots.")
        descriptor = declared.declaration()
        if self.redact_json(descriptor) != descriptor:
            raise ValueError(f"The model slot {declared.name} contains a secret value.")
        self._model_slot_declarations = MappingProxyType(
            {**self._model_slot_declarations, declared.name: declared}
        )

    def _agent_model_slot(
        self: CayuApp,
        spec: AgentSpec,
        model_slot: str | None,
        settings: Iterable[ManagedSetting],
    ) -> tuple[str | None, ModelSlot | None]:
        """The slot an agent about to register uses, and the slot to declare for it once it has.

        ``settings`` are the agent's managed settings, with their full keys.

        An agent that chooses its model with the deprecated ``model_setting()`` uses a slot
        named after it, declared with the agent's own model as its default, when that name is
        free.
        """

        uses_setting = any(item.key == f"agent.{spec.name}.model" for item in settings)
        if model_slot is not None:
            if type(model_slot) is not str or not MODEL_SLOT_PATTERN.fullmatch(model_slot):
                raise ValueError(
                    "model_slot is 1-63 lowercase letters, digits or interior hyphens."
                )
            if uses_setting:
                raise ValueError(
                    f"Agent {spec.name!r} names a model_slot and declares model_setting(); "
                    "drop the deprecated model_setting()."
                )
            return model_slot, None
        slot = model_setting_slot(spec.name) if uses_setting else None
        if (
            slot is not None
            and slot in self._model_slot_declarations
            and slot not in self._setting_model_slots
        ):
            raise ValueError(
                f"Agent {spec.name!r} declares the deprecated model_setting(), whose model slot "
                f"{slot} is already declared with require_model(); register the agent with "
                f"model_slot={slot!r} instead of model_setting()."
            )
        if (
            slot is None
            or slot in self._model_slot_declarations
            or len(self._model_slot_declarations) >= MAX_MODEL_SLOTS
            or not MODEL_ID_PATTERN.fullmatch(spec.model)
        ):
            return None, None
        return slot, ModelSlot(name=slot, default=spec.model)

    def _add_setting_model_slot(self: CayuApp, agent_name: str, declared: ModelSlot) -> None:
        self._declare_model_slot(declared)
        self._setting_model_slots = MappingProxyType(
            {**self._setting_model_slots, declared.name: agent_name}
        )

    def _model_slot_name(self: CayuApp, agent_name: str) -> str | None:
        registration = self._agent_registry.registrations.get(agent_name)
        return None if registration is None else registration.model_slot

    def _session_model_slot(self: CayuApp, agent_name: str) -> ModelSlotResolution | None:
        """The model a new session of ``agent_name`` uses through its slot, if it has one.

        Raises ``ModelSlotUnavailable`` when the slot has no model or its model isn't approved,
        so the session is refused rather than run on another model. An agent that still
        declares ``model_setting()`` keeps its Agent setting's value until its slot is bound.
        """

        slot = self._model_slot_name(agent_name)
        if slot is None:
            return None
        declared = self._model_slot_declarations.get(slot)
        if declared is None:
            raise ValueError(
                f"Agent {agent_name!r} uses the model slot {slot}, which the application "
                "doesn't declare; declare it with CayuApp.require_model()."
            )
        reader = self._model_slots_reader
        snapshot = None if reader is None else reader.snapshot
        if self._setting_model_slots.get(slot) == agent_name:
            binding = None if snapshot is None else snapshot.bindings.get(slot)
            bound = binding is not None and binding.source == "bound"
            if not bound and self._managed_agent_model(agent_name)[1] is not None:
                return None
        return resolve_model_slot(declared, snapshot, read=reader is not None)

    def _bind_model_slots(self: CayuApp, reader: ModelSlots | None) -> None:
        """Follow Cloud's bindings through ``reader``, or the broker Cloud bound, if any, for
        this process's Release."""

        if reader is not None and type(reader) is not ModelSlots:
            raise TypeError("model_slots must be ModelSlots.")
        if reader is None:
            release = self.release_identity
            reader = ModelSlots.from_environment(release_id=None if release is None else release.id)
        self._model_slots_reader = reader

    @_admitted_entrance
    async def start_model_slots(self: CayuApp) -> None:
        """Read this Agent's model slots from Cayu Cloud, then keep reading them in the
        background; a no-op outside Cloud or without declared slots.

        ``create_server``, ``mount_cayu`` and ``cayu worker`` call it; ``aclose`` stops it.
        """

        # Refuses to start with an agent on a slot the application doesn't declare.
        if self.model_slots() and self._model_slots_reader is not None:
            await self._model_slots_reader.start()

    async def stop_model_slots(self: CayuApp) -> None:
        if self._model_slots_reader is not None:
            await self._model_slots_reader.close()
