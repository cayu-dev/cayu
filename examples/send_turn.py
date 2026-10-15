"""Create and continue a conversation without hand-parsing runtime events.

Run: python examples/send_turn.py
Uses a scripted provider; no network access or credentials are required.
"""

from __future__ import annotations

import asyncio

from cayu import AgentSpec, CayuApp, Message, TurnRequest, send_turn
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent


async def main() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta(text),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
                for text in ("Which order?", "Order 42 is shipped.")
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="support", model="scripted-model"))
    first = await send_turn(
        app,
        TurnRequest(agent_name="support", messages=[Message.text("user", "Track my order")]),
    )
    if not first.ok:
        raise RuntimeError(first.error or "The first turn was interrupted.")
    print(first.final_text)
    second = await send_turn(
        app,
        TurnRequest(
            agent_name="support",
            session_id=first.session_id,
            messages=[Message.text("user", "Order 42")],
        ),
    )
    print(second.final_text if second.ok else second.summary())


if __name__ == "__main__":
    asyncio.run(main())
