"""The human-attention handoff a ``CayuApp`` owns: from code, or from the environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.human_attention_handoff import HumanAttentionHandoff, HumanAttentionHandoffSink
from cayu.observability.events import EventSink

if TYPE_CHECKING:
    from cayu.applications import CayuApp

__all__ = ["HumanAttentionHandoff", "attach_handoff", "bind_handoff"]


def attach_handoff(
    app: CayuApp, configured: HumanAttentionHandoff | None, sinks: list[EventSink]
) -> HumanAttentionHandoff | None:
    """The handoff ``app`` owns, with its sink added to ``sinks``; ``None`` without one.

    A deployment that sets ``CAYU_ATTENTION_HANDOFF_URL`` and ``..._TOKEN`` (Cayu Cloud
    injects both) gets the handoff without application code. An application that already
    composes the sink in ``event_sinks`` keeps owning it, and passing both is refused.
    """

    composed = any(isinstance(sink, HumanAttentionHandoffSink) for sink in sinks)
    if configured is not None:
        if not isinstance(configured, HumanAttentionHandoff):
            raise TypeError("human_attention_handoff must be a HumanAttentionHandoff.")
        if composed:
            raise ValueError(
                "Configure the human-attention handoff once: pass human_attention_handoff "
                "or put its sink in event_sinks, not both."
            )
        handoff = configured
    elif composed:
        return None
    else:
        handoff = HumanAttentionHandoff.from_environment()
    if not handoff.enabled:
        return None
    handoff.require_bindable(app)
    sinks.extend(handoff.event_sinks())
    return handoff


def bind_handoff(app: CayuApp, handoff: HumanAttentionHandoff | None) -> None:
    """Bind the handoff ``app`` owns, as the last step of constructing ``app``.

    An application whose construction failed never holds the handoff, so the same handoff
    can be passed to the next ``CayuApp``.
    """

    if handoff is not None:
        handoff.bind(app)
