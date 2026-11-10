"""Model slots: the models an application declares by name, which Cayu Cloud binds per Agent.

An application declares each slot with ``CayuApp.require_model(slot, default=...)`` and names the
slot an agent uses with ``register_agent(..., model_slot=...)``. The Organization binds a slot to
one of its approved models in Cayu Cloud, without a redeploy. The Agent reads its bindings from
the connection broker in the background (``ModelSlots``) and resolves a slot when a session
starts, so a binding changes the model of new sessions only.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from cayu.connections.client import _SLOT, CloudConnectionBroker

#: A slot name, as for connection slots: 1-63 lowercase letters, digits or interior hyphens.
MODEL_SLOT_PATTERN = _SLOT
#: A model id Cayu Cloud binds: 1-128 letters, digits or ``_.:/-``.
MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
MAX_MODEL_SLOTS = 64
MAX_MODEL_SLOT_LABEL_LENGTH = 80
#: The session metadata key holding the slot a session's model came from and that model,
#: ``{"slot", "source", "model"}``.
MODEL_SLOT_METADATA_KEY = "cayu:model_slot"
DEFAULT_MODEL_SLOTS_POLL_SECONDS = 25.0
DEFAULT_MODEL_SLOTS_TIMEOUT_SECONDS = 10.0
DEFAULT_MODEL_SLOTS_STARTUP_WAIT_SECONDS = 2.0

ModelSlotSource = Literal["bound", "default", "unbound"]
#: Where a session's model came from: Cloud's binding, the slot's default while Cloud's
#: bindings were read, or the slot's default because they were never read.
ModelSlotResolutionSource = Literal["bound", "default", "unread"]

logger = logging.getLogger(__name__)


class ModelSlot(BaseModel):
    """One declared model slot: its name, the model it uses unless bound, and a label."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: StrictStr
    #: The model the slot uses until the Organization binds it; None when it needs a binding.
    default: StrictStr | None
    label: StrictStr | None = Field(default=None, max_length=MAX_MODEL_SLOT_LABEL_LENGTH)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not MODEL_SLOT_PATTERN.fullmatch(value):
            raise ValueError("A model slot is 1-63 lowercase letters, digits or interior hyphens.")
        return value

    @field_validator("default")
    @classmethod
    def _default(cls, value: str | None) -> str | None:
        if value is not None and not MODEL_ID_PATTERN.fullmatch(value):
            raise ValueError("A model id is 1-128 letters, digits or '_.:/-'.")
        return value

    @field_validator("label")
    @classmethod
    def _label(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or "\n" in value or "\r" in value):
            raise ValueError("A model slot's label is one non-blank line.")
        return value

    @staticmethod
    def is_model_id(model: str) -> bool:
        """Whether ``model`` can be a slot's default or binding: 1-128 letters, digits or
        ``_.:/-``."""

        return MODEL_ID_PATTERN.fullmatch(model) is not None

    def declaration(self) -> dict[str, Any]:
        """The slot as the agent registry publishes it: ``{"name", "default", "label"?}``."""

        declared: dict[str, Any] = {"name": self.name, "default": self.default}
        if self.label is not None:
            declared["label"] = self.label
        return declared


@dataclass(frozen=True, slots=True)
class ModelSlotBinding:
    """One slot as Cayu Cloud's broker lists it for this Agent."""

    slot: str
    model: str | None
    source: ModelSlotSource
    #: Whether the Gateway's policies admit ``model``; None when Cloud couldn't say.
    approved: bool | None


@dataclass(frozen=True, slots=True)
class ModelSlotsSnapshot:
    """The Agent's model slots and approved models, as last read from the broker."""

    bindings: Mapping[str, ModelSlotBinding]
    #: None when no model is approved through the Gateway here, or Cloud couldn't say.
    approved_models: frozenset[str] | None


@dataclass(frozen=True, slots=True)
class ModelSlotResolution:
    """The model a new session uses for its agent's slot, and where it came from."""

    slot: str
    model: str
    source: ModelSlotResolutionSource

    def evidence(self) -> dict[str, str]:
        """What ``session.started`` records as its ``model_slot``."""

        return {"slot": self.slot, "source": self.source}

    def record(self) -> dict[str, str]:
        """What the session records when it is created: its evidence and the slot's model."""

        return {**self.evidence(), "model": self.model}


def session_model_slot(session: Any) -> dict[str, str] | None:
    """The slot a session's model came from and that model, as recorded when it was created;
    None when its model came from anywhere else, or the record predates the model in it."""

    return model_slot_record(session.metadata.get(MODEL_SLOT_METADATA_KEY))


def model_slot_record(value: object) -> dict[str, str] | None:
    """``value`` as a session's slot record, ``{"slot", "source", "model"}``; None when it is
    not one."""

    if not isinstance(value, dict) or set(value) != {"slot", "source", "model"}:
        return None
    record = cast("dict[str, object]", value)
    slot, source, model = record["slot"], record["source"], record["model"]
    if (
        not isinstance(slot, str)
        or not MODEL_SLOT_PATTERN.fullmatch(slot)
        or source not in {"bound", "default", "unread"}
        or not isinstance(model, str)
        or not MODEL_ID_PATTERN.fullmatch(model)
    ):
        return None
    return {"slot": slot, "source": str(source), "model": model}


class ModelSlotUnavailable(ValueError):
    """A new session can't start: its slot has no model, or its model isn't approved.

    ``code`` is ``model_not_approved`` or ``model_slot_unbound``, and the message starts with
    it, so the server's run routes answer it as a 409 conflict that names it. Nothing falls
    back to another model: the Organization binds the slot to an approved model, or approves
    its model.
    """

    def __init__(self, message: str, *, code: str, slot: str, model: str | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.slot = slot
        self.model = model


def parse_model_slots(answer: Mapping[str, Any]) -> ModelSlotsSnapshot:
    """Read ``models`` and ``approved_models`` from the broker's listing.

    Raises ``ValueError`` when either is malformed, so the caller keeps what it read before.
    A listing from a Cloud without model slots has neither, and binds nothing.
    """

    items = answer.get("models", [])
    approved = answer.get("approved_models")
    if not isinstance(items, list):
        raise ValueError("models must be a list.")
    if approved is not None and (
        not isinstance(approved, list)
        or any(
            not isinstance(model, str) or not MODEL_ID_PATTERN.fullmatch(model)
            for model in approved
        )
    ):
        raise ValueError("approved_models must be a list of model ids, or null.")
    bindings: dict[str, ModelSlotBinding] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each model slot must be an object.")
        slot, model = item.get("slot"), item.get("model")
        source, is_approved = item.get("source"), item.get("approved")
        if (
            not isinstance(slot, str)
            or not MODEL_SLOT_PATTERN.fullmatch(slot)
            or slot in bindings
            or (
                model is not None
                and (not isinstance(model, str) or not MODEL_ID_PATTERN.fullmatch(model))
            )
            or not isinstance(source, str)
            or (source == "bound" and model is None)
            or (is_approved is not None and type(is_approved) is not bool)
        ):
            raise ValueError("A model slot in the broker's listing is malformed.")
        if source not in {"bound", "default", "unbound"}:
            # A source this Runtime doesn't know binds nothing: the slot uses its default.
            continue
        bindings[slot] = ModelSlotBinding(
            slot=slot, model=model, source=source, approved=is_approved
        )
    return ModelSlotsSnapshot(
        bindings=bindings,
        approved_models=None if approved is None else frozenset(str(model) for model in approved),
    )


def resolve_model_slot(
    declared: ModelSlot, snapshot: ModelSlotsSnapshot | None, *, read: bool
) -> ModelSlotResolution:
    """The model a new session uses for ``declared``.

    A bound slot uses its binding; any other uses its declared default. ``read`` says whether
    this process reads bindings at all: when it does and never got them, the default is used
    and recorded as ``unread``. A model Cloud says isn't approved is refused, never replaced.
    """

    binding = None if snapshot is None else snapshot.bindings.get(declared.name)
    if binding is not None and binding.source == "bound" and binding.model is not None:
        model, approved, source = binding.model, binding.approved, "bound"
    else:
        model = declared.default
        if model is None:
            raise ModelSlotUnavailable(
                f"The model slot {declared.name} has no model: bind it to an approved model "
                f"(cayu cloud connections bind {declared.name} --model MODEL).",
                code="model_slot_unbound",
                slot=declared.name,
            )
        approved = (
            None
            if snapshot is None or snapshot.approved_models is None
            else model in snapshot.approved_models
        )
        source = "default" if snapshot is not None or not read else "unread"
    if approved is False:
        raise ModelSlotUnavailable(
            f"The model {model} of slot {declared.name} isn't approved for this Agent; new "
            "sessions are refused until the slot is bound to an approved model or the model "
            "is approved.",
            code="model_not_approved",
            slot=declared.name,
            model=model,
        )
    return ModelSlotResolution(slot=declared.name, model=model, source=source)


class ModelSlotsSource(Protocol):
    """What ``ModelSlots`` reads: the broker's listing of this Agent's connections."""

    async def listing(self, *, release: str | None = None) -> dict[str, Any]: ...


class ModelSlots:
    """Reads the Agent's model slots from Cayu Cloud's broker in the background.

    It reads in the background from when it is started, then every ``poll_interval`` seconds,
    each read bounded by ``timeout_seconds``, naming this process's Release (``release_id``)
    when it is known; ``start`` waits at most ``startup_wait_seconds``
    for the first read. It keeps the last good snapshot: an unreachable broker or a malformed
    answer changes nothing. Sessions read the snapshot when they start; until a first read,
    they use their slots' defaults.
    """

    def __init__(
        self,
        source: ModelSlotsSource,
        *,
        poll_interval: float = DEFAULT_MODEL_SLOTS_POLL_SECONDS,
        timeout_seconds: float = DEFAULT_MODEL_SLOTS_TIMEOUT_SECONDS,
        startup_wait_seconds: float = DEFAULT_MODEL_SLOTS_STARTUP_WAIT_SECONDS,
        release_id: str | None = None,
    ) -> None:
        if type(poll_interval) not in {int, float} or not 0 < poll_interval <= 300:
            raise ValueError("poll_interval must be more than 0 and at most 300 seconds.")
        if type(timeout_seconds) not in {int, float} or not 0 < timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be more than 0 and at most 60 seconds.")
        if type(startup_wait_seconds) not in {int, float} or not 0 <= startup_wait_seconds <= 10:
            raise ValueError("startup_wait_seconds must be between 0 and 10 seconds.")
        self._source = source
        self.release_id = release_id or None
        self.startup_wait_seconds = float(startup_wait_seconds)
        self._first_read = asyncio.Event()
        self.poll_interval = float(poll_interval)
        self.timeout_seconds = float(timeout_seconds)
        self._snapshot: ModelSlotsSnapshot | None = None
        self._task: asyncio.Task[None] | None = None
        self.status: Literal["stopped", "ready", "unavailable"] = "stopped"

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None, *, release_id: str | None = None
    ) -> ModelSlots | None:
        """The reader for the broker Cayu Cloud bound this process to; None outside Cloud."""

        try:
            broker = CloudConnectionBroker.from_environment(environ)
        except ValueError:
            logger.warning(
                "Cayu Cloud model slots are off: the connection broker is configured without "
                "its token, so slots use their declared defaults."
            )
            return None
        if broker is None:
            return None
        return cls(broker, release_id=release_id)

    @property
    def snapshot(self) -> ModelSlotsSnapshot | None:
        """The last good snapshot; None until one was read."""

        return self._snapshot

    async def poll_once(self) -> None:
        """Read the broker once, keeping the last good snapshot when it fails."""

        try:
            async with asyncio.timeout(self.timeout_seconds):
                # Cloud refusing the Release (422 release_unknown) is a failed read too.
                answer = await self._source.listing(release=self.release_id)
            snapshot = parse_model_slots(answer)
        except Exception as exc:
            if self.status != "unavailable":
                # Once per outage, and never credentials, response bodies or URLs.
                logger.warning(
                    "Cayu Cloud model slots could not be read (%s); new sessions keep using "
                    "the last bindings read, or the slots' defaults.",
                    type(exc).__name__,
                )
            self.status = "unavailable"
            return
        self._snapshot = snapshot
        self.status = "ready"

    async def start(self) -> None:
        """Read in the background now, then every ``poll_interval`` seconds until ``close``;
        wait at most ``startup_wait_seconds`` for the first read."""

        if self._task is not None:
            return
        self._first_read.clear()
        task = self._task = asyncio.create_task(self._run(), name="cayu-model-slots")
        first_read = asyncio.ensure_future(self._first_read.wait())
        try:
            # The first read ends the wait whether it succeeded or failed.
            await asyncio.wait(
                {first_read, task},
                timeout=self.startup_wait_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            first_read.cancel()

    async def _run(self) -> None:
        while True:
            await self.poll_once()
            self._first_read.set()
            await asyncio.sleep(self.poll_interval)

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self.status = "stopped"


__all__ = [
    "DEFAULT_MODEL_SLOTS_POLL_SECONDS",
    "DEFAULT_MODEL_SLOTS_TIMEOUT_SECONDS",
    "MAX_MODEL_SLOTS",
    "MAX_MODEL_SLOT_LABEL_LENGTH",
    "MODEL_ID_PATTERN",
    "MODEL_SLOT_PATTERN",
    "ModelSlot",
    "ModelSlotBinding",
    "ModelSlotResolution",
    "ModelSlotResolutionSource",
    "ModelSlotSource",
    "ModelSlotUnavailable",
    "ModelSlots",
    "ModelSlotsSnapshot",
    "ModelSlotsSource",
    "parse_model_slots",
    "resolve_model_slot",
]
